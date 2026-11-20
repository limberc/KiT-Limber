"""Adaptive bucket mixture: how often training draws each (market, scale) bucket.

Recomputed after every validation pass, from two parts:
- prior (UniMax, Chung et al. 2023): an even share of the remaining steps per bucket, except that no bucket's
  training bars are reused as targets more than `max_reuse` times over the run; budget a bucket cannot take goes
  to the others. It follows from bucket sizes and the step budget, so new markets or a longer run need no hand
  weights.
- tilt by learning progress (in the spirit of ADO, Jiang et al. 2024): the decrease of a bucket's validation loss
  per batch drawn from it, smoothed over validation passes, scales its prior share by up to `tilt` either way.
  Progress, not loss level: per-bucket loss levels are mostly irreducible noise (US 1d fat tails, degenerate
  crypto wick features), so chasing high loss would oversample the smallest stores. A bucket whose validation
  loss rises (overfitting) drops towards prior / tilt.

Reuse unit: target uses per training bar = batches x B x G x A x horizon x P / bars. Run 2 (uniform mix, 48k
steps) reached the per-bucket validation minimum at ~20 (US 1h), ~65 (US 2h), ~170 (US 1d) and ~530 (crypto 1d)
uses, so no single cap fits every bucket: the cap is a backstop, the progress tilt does the per-bucket work.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import lightning as L
import numpy as np
import torch


def unimax(capacity: np.ndarray, budget: float) -> np.ndarray:
    """Probabilities over buckets for `budget` steps: equal shares, but bucket k takes at most capacity[k]
    steps and what it cannot take is split evenly among the rest. If all capacity together is below the
    budget, every bucket is spent in proportion to its capacity."""
    cap = np.asarray(capacity, dtype=np.float64)
    total = cap.sum()
    if total <= budget:
        return cap / total if total > 0 else np.full(len(cap), 1.0 / len(cap))
    alloc, left = np.zeros_like(cap), float(budget)
    for i, k in enumerate(np.argsort(cap)):
        alloc[k] = min(float(cap[k]), left / (len(cap) - i))
        left -= alloc[k]
    return alloc / budget


def capped(p: np.ndarray, capacity: np.ndarray, budget: float) -> np.ndarray:
    """Normalise p so that no bucket's expected draws p_k x budget exceed its capacity; the excess goes to
    the uncapped buckets in proportion to their p."""
    p = np.asarray(p, dtype=np.float64) / np.sum(p)
    limit = np.minimum(np.asarray(capacity, dtype=np.float64) / budget, 1.0)
    fixed = np.zeros(len(p), dtype=bool)
    while True:
        over = ~fixed & (p > limit)
        if not over.any():
            break
        p[over] = limit[over]
        fixed |= over
        free = ~fixed
        if not free.any() or p[free].sum() <= 0:
            break
        p[free] *= (1.0 - p[fixed].sum()) / p[free].sum()
    return p / p.sum()


class Mixture:
    """Bucket sampling probabilities from bucket sizes, batches drawn and per-bucket validation losses."""

    def __init__(self, names: Sequence[str], bars: Sequence[float], uses_per_batch: Sequence[float],
                 max_reuse: Optional[float] = 128.0, tilt: float = 4.0, smooth: float = 0.5,
                 warmup_evals: int = 2) -> None:
        self.names = list(names)
        self.bars = np.maximum(np.asarray(bars, dtype=np.float64), 1.0)
        self.upb = np.asarray(uses_per_batch, dtype=np.float64)
        self.max_reuse, self.tilt, self.smooth, self.warmup_evals = max_reuse, tilt, smooth, warmup_evals
        K = len(self.names)
        self.drawn = np.zeros(K)                 # batches drawn per bucket so far
        self.progress = np.zeros(K)              # smoothed validation-loss decrease per even share of batches
        self.last_loss = np.full(K, np.nan)
        self.last_drawn = np.zeros(K)
        self.evals = 0                           # progress measurements so far
        self.p = np.full(K, 1.0 / K)

    def capacity(self) -> np.ndarray:
        """Batches each bucket may still draw before its bars are reused `max_reuse` times."""
        if self.max_reuse is None:
            return np.full(len(self.names), np.inf)
        return np.maximum(self.max_reuse * self.bars / self.upb - self.drawn, 0.0)

    def reuse(self) -> np.ndarray:
        return self.drawn * self.upb / self.bars

    def start(self, budget: float) -> np.ndarray:
        self.p = unimax(self.capacity(), max(float(budget), 1.0))
        return self.p

    def update(self, loss: np.ndarray, drawn: np.ndarray, remaining: float) -> np.ndarray:
        """loss: per-bucket validation loss (NaN = not measured); drawn: cumulative batches per bucket;
        remaining: training steps left. Returns the new probabilities."""
        loss = np.asarray(loss, dtype=np.float64)
        self.drawn = np.asarray(drawn, dtype=np.float64)
        if np.isfinite(self.last_loss).any():
            got = self.drawn - self.last_drawn
            share = got / max(got.sum(), 1.0) * len(got)                    # 1 = an even share of the interval
            gain = (self.last_loss - loss) / np.maximum(share, 0.1)
            ok = np.isfinite(gain)
            self.progress[ok] = (1 - self.smooth) * self.progress[ok] + self.smooth * gain[ok]
            self.evals += 1
        self.last_loss = np.where(np.isfinite(loss), loss, self.last_loss)
        self.last_drawn = self.drawn.copy()
        budget = max(float(remaining), 1.0)
        cap = self.capacity()
        p = unimax(cap, budget)
        mag = np.abs(self.progress)
        scale = np.median(mag) if np.median(mag) > 0 else mag.mean()    # typical bucket's progress
        if self.evals >= self.warmup_evals and scale > 0:
            p = p * self.tilt ** np.tanh(self.progress / scale)
        self.p = capped(p, cap, budget)
        return self.p

    def state_dict(self) -> Dict[str, Any]:
        return {"names": self.names, "drawn": self.drawn.tolist(), "progress": self.progress.tolist(),
                "last_loss": self.last_loss.tolist(), "last_drawn": self.last_drawn.tolist(),
                "evals": self.evals, "p": self.p.tolist()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restores by bucket name; buckets new to this run start fresh."""
        pos = {n: i for i, n in enumerate(state["names"])}
        for key in ("drawn", "progress", "last_loss", "last_drawn"):
            old = np.asarray(state[key], dtype=np.float64)
            cur = getattr(self, key)
            for i, n in enumerate(self.names):
                if n in pos:
                    cur[i] = old[pos[n]]
        self.evals = int(state["evals"])
        if list(state["names"]) == self.names:
            self.p = np.asarray(state["p"], dtype=np.float64)


class AdaptiveMixture(L.Callback):
    """Lightning wiring for `Mixture` with `StoreData`: counts training batches per bucket (on the device, no
    sync), updates the probabilities the training workers draw from after every validation pass during fit,
    and logs them as mix/p_<bucket> and mix/reuse_<bucket>. Overrides StoreData's static `weights`."""

    def __init__(self, max_reuse: Optional[float] = 128.0, tilt: float = 4.0, smooth: float = 0.5,
                 warmup_evals: int = 2) -> None:
        self.kw = {"max_reuse": max_reuse, "tilt": tilt, "smooth": smooth, "warmup_evals": warmup_evals}
        self.mix: Optional[Mixture] = None
        self.dm = None
        self._restored: Optional[Dict[str, Any]] = None
        self._counts: Optional[torch.Tensor] = None

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        if stage != "fit" or self.mix is not None:
            return
        dm = trainer.datamodule
        if not hasattr(dm, "bucket_sizes"):
            raise TypeError("AdaptiveMixture needs a datamodule with bucket_sizes() and set_weights() (StoreData)")
        if trainer.max_steps <= 0:
            raise ValueError("AdaptiveMixture needs trainer.max_steps (the step budget)")
        bars, uses = dm.bucket_sizes()
        self.dm, self.mix = dm, Mixture(dm.bucket_names, bars, uses, **self.kw)
        self.mix.start(trainer.max_steps)
        if self._restored is not None:
            self.mix.load_state_dict(self._restored)
        dm.set_weights(self.mix.p)
        print("mix start: " + self._describe(), flush=True)

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._counts = torch.tensor(self.mix.drawn, dtype=torch.float64, device=pl_module.device)
        self._one = torch.ones(1, dtype=torch.float64, device=pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs,
                           batch, batch_idx) -> None:
        self._counts.index_add_(0, batch["bucket"].reshape(1).long(), self._one)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking or trainer.state.fn != "fit" or self._counts is None:
            return
        m = trainer.callback_metrics
        loss = np.array([float(m[f"val/loss_{n}"]) if f"val/loss_{n}" in m else np.nan for n in self.mix.names])
        self.mix.update(loss, self._counts.cpu().numpy(), trainer.max_steps - trainer.global_step)
        self.dm.set_weights(self.mix.p)
        if trainer.logger is not None and trainer.is_global_zero:
            trainer.logger.log_metrics({**{f"mix/p_{n}": float(v) for n, v in zip(self.mix.names, self.mix.p)},
                                        **{f"mix/reuse_{n}": float(v) for n, v in zip(self.mix.names, self.mix.reuse())}},
                                       step=trainer.global_step)
        print(f"mix @{trainer.global_step}: " + self._describe(), flush=True)

    def _describe(self) -> str:
        return " ".join(f"{n} {p:.3f}/{r:.0f}" for n, p, r in zip(self.mix.names, self.mix.p, self.mix.reuse()))

    def state_dict(self) -> Dict[str, Any]:
        if self.mix is None:
            return {}
        if self._counts is not None:
            self.mix.drawn = self._counts.cpu().numpy().astype(np.float64)
        return self.mix.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not state_dict:
            return
        self._restored = state_dict
        if self.mix is not None:          # setup already ran (Lightning restores callbacks after setup)
            self.mix.load_state_dict(state_dict)
            self.dm.set_weights(self.mix.p)
