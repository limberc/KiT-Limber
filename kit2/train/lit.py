"""PyTorch Lightning wiring for KiT-2.

Data flow: a DataLoader yields small sampling plans (CPU); `on_after_batch_transfer` (runs on the GPU)
gathers the bar windows from the GPU-resident store, assembles model inputs and runs the look-ahead audit;
`training_step` computes the rectified-flow loss. Synthetic data uses the same assembly path.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import lightning as L
import torch
from torch.utils.data import DataLoader, IterableDataset

from kit2.model import KiT2, flow_loss
from kit2.model.config import preset
from .batch import assemble, audit, model_inputs, synthetic_meta, synthetic_windows
from .buckets import SCALE_ID, SCALE_SECONDS, Bucket

SPLIT_ID = {"train": 0, "val": 1, "test": 2}


class KiT2Lit(L.LightningModule):
    def __init__(self, model: Dict[str, Any], optim: Dict[str, Any], compile: bool = True,
                 bucket_names: Optional[List[str]] = None, p_cond: float = 0.0) -> None:
        """p_cond: share of training (sample, anchor) pairs that condition on a clean subset of instruments
        (kit2.model.conditioning_subset); validation always uses the joint objective, comparable across runs."""
        super().__init__()
        self.save_hyperparameters()
        self.cfg = preset(model.get("preset", "dev"), **model.get("overrides", {}))
        self.model = KiT2(self.cfg)
        self.bucket_names = bucket_names or []

    def configure_model(self) -> None:
        if self.hparams.compile and not getattr(self, "_compiled", False):
            torch._dynamo.config.recompile_limit = 64  # one graph per bucket shape
            self.model.compile(dynamic=False)  # in place: state_dict keys stay unchanged
            self._compiled = True

    def model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def setup(self, stage: str) -> None:
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and getattr(dm, "bucket_names", None):
            self.bucket_names = dm.bucket_names

    def _loss(self, b: Dict[str, torch.Tensor], generator: Optional[torch.Generator] = None,
              p_cond: float = 0.0) -> torch.Tensor:
        return flow_loss(self.model, model_inputs(b), b["x0"], b["valid"], generator=generator, p_cond=p_cond)

    def on_before_optimizer_step(self, optimizer) -> None:
        # clipping here, not via Trainer(gradient_clip_val): Lightning refuses that with fused AdamW
        clip = self.hparams.optim.get("grad_clip", 1.0)
        if clip:
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.log("train/grad_norm", norm, on_step=True, batch_size=1)

    def training_step(self, b: Dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        loss = self._loss(b, p_cond=self.hparams.get("p_cond", 0.0))
        self.log("train/loss", loss, prog_bar=True, on_step=True, batch_size=1)
        return loss

    def validation_step(self, b: Dict[str, torch.Tensor], idx: int) -> None:
        # fixed flow times and noise per validation batch, so val loss is comparable across checkpoints
        loss = self._loss(b, torch.Generator(device=self.device).manual_seed(1000 + idx))
        name = self.bucket_names[int(b["bucket"])] if self.bucket_names else "all"
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=1)
        self.log(f"val/loss_{name}", loss, on_epoch=True, batch_size=1)

    def configure_optimizers(self):
        o = self.hparams.optim
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            (decay if p.dim() >= 2 and "emb" not in n else no_decay).append(p)
        groups = [{"params": decay, "weight_decay": o.get("weight_decay", 0.1)},
                  {"params": no_decay, "weight_decay": 0.0}]
        if self.model_dtype() == torch.bfloat16:
            # precision="bf16-true" (opt-in): weights and AdamW state in bf16; stochastic rounding keeps small
            # updates from rounding away (round-to-nearest bf16 training stalls)
            from torchao.optim import _AdamW
            opt = _AdamW(groups, lr=o.get("lr", 3e-4), betas=(0.9, 0.95), eps=1e-8, bf16_stochastic_round=True)
        else:
            # precision="bf16-mixed" (default): bf16 compute, fp32 master weights + optimizer state
            opt = torch.optim.AdamW(groups, lr=o.get("lr", 3e-4), betas=(0.9, 0.95), eps=1e-8,
                                    fused=torch.cuda.is_available())
        warmup, total, floor = (o.get("warmup_steps", 1000),
                                o.get("total_steps", 100_000),
                                o.get("min_lr_ratio", 0.1))

        def lr_at(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            p = min(1.0, (step - warmup) / max(1, total - warmup))
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def batch_size_for(bucket: Bucket, G: int, batch_tokens: int) -> int:
    return max(1, round(batch_tokens / bucket.tokens(G)))


def random_anchors(bucket: Bucket, B: int, A: int, device, generator=None) -> torch.Tensor:
    """A distinct anchor patches per sample, uniform over the bucket's anchor range."""
    lo, hi = bucket.anchor_range
    perm = torch.rand(B, hi - lo + 1, device=device, generator=generator).argsort(dim=1)
    return perm[:, :A] + lo


class _Plans(IterableDataset):
    """Endless stream of {bucket} picks, weighted."""

    def __init__(self, weights: List[float], seed: int) -> None:
        self.w, self.seed = torch.tensor(weights, dtype=torch.float), seed

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        while True:
            yield {"bucket": torch.multinomial(self.w, 1, generator=g)[0], "split": torch.tensor(0)}


class SyntheticData(L.LightningDataModule):
    """Random bars with real bucket shapes, assembled on the GPU like real data (benchmarks, smoke tests).
    One batch per bucket is generated once and reused, so timings exclude data generation."""

    def __init__(self, buckets: List[Bucket], model_cfg, G: int = 32, A: int = 8, batch_tokens: int = 65536,
                 weights: Optional[List[float]] = None, seed: int = 0) -> None:
        super().__init__()
        self.buckets, self.cfg, self.G, self.A, self.batch_tokens = buckets, model_cfg, G, A, batch_tokens
        self.weights = weights or [1.0] * len(buckets)
        self.seed = seed
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(_Plans(self.weights, self.seed), batch_size=None,
                          num_workers=0)

    def make(self, k: int, device) -> Dict[str, torch.Tensor]:
        bk = self.buckets[k]
        B = batch_size_for(bk, self.G, self.batch_tokens)
        wins = synthetic_windows(self.cfg, B, self.G, bk.n_patches, [SCALE_SECONDS[s] for s in bk.scales],
                                 device=device)
        for w, s in zip(wins, bk.scales):
            w.scale_id = SCALE_ID[s]
        meta = synthetic_meta(B, self.G, device=device)
        meta["target_scale"] = torch.full((B,), SCALE_ID[bk.scale], dtype=torch.long, device=device)
        b = assemble(wins, random_anchors(bk, B, self.A, device), bk.horizon, meta, self.cfg)
        b["bucket"] = torch.tensor(k, device=device)
        return b

    def on_after_batch_transfer(self, batch: Dict[str, torch.Tensor], dataloader_idx: int) -> Dict[str, torch.Tensor]:
        k = int(batch["bucket"])  # batch = a sampling plan; assembled here, on the GPU
        if k not in self._cache:
            self._cache[k] = self.make(k, batch["bucket"].device)
        b = self._cache[k]
        audit(b)
        return b
