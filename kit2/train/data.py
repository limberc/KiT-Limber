"""Real-data pipeline for Lightning: point-in-time group windows from the memory-mapped store.

CPU workers (DataLoader) choose a bucket, sample B moments in the split's time range, pick G instruments that
were in that month's sampling set (decided before the month) and fully cover the window at every scale, and
copy the bar windows out of the memory-mapped store. `on_after_batch_transfer` (GPU) assembles model inputs
with the grid/calendar tables kept on the GPU and runs the look-ahead audit.

Splits (purged by construction): a training sample's target bars all close <= TRAIN_END; a validation
sample's target bars all close in (TRAIN_END, VAL_END]; test is after VAL_END. History may reach back into
earlier periods, which is the past for every split.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from kit2.data import MarketStore
from kit2.data.store import CHANNELS, TRAIN_END, VAL_END
from kit2.train.batch import ScaleWindow, assemble, audit
from kit2.train.buckets import MARKET_ID, SCALE_ID, Bucket

ALWAYS = {"us": ("SPY", "QQQ")}  # market anchors put in every US group when covered
SPLIT_RANGE = {"train": (None, TRAIN_END), "val": (TRAIN_END, VAL_END), "test": (VAL_END, None)}


class GroupSampler:
    """Builds one bucket batch (numpy) from the store. Shared by train / val / test datasets."""

    def __init__(self, root: Path, buckets: List[Bucket], G: int, A: int, batch_tokens: int, split: str) -> None:
        self.buckets, self.G, self.A, self.split = buckets, G, A, split
        scales = sorted({s for b in buckets for s in b.scales})
        self.stores = {m: MarketStore(root, m, scales) for m in sorted({b.market for b in buckets})}
        self.B = [max(1, round(batch_tokens / b.tokens(G))) for b in buckets]
        self.always = {m: self._anchor_idx(m) for m in self.stores}
        lo, hi = SPLIT_RANGE[split]
        self.lo_ns = None if lo is None else lo.value
        self.hi_ns = None if hi is None else hi.value
        self._ends = [self._valid_ends(b) for b in buckets]

    def _anchor_idx(self, market: str) -> np.ndarray:
        """Market anchors put first in every group (store column `anchor`; US fallback: SPY, QQQ)."""
        inst = self.stores[market].instruments
        mask = inst["anchor"].to_numpy(bool) if "anchor" in inst else inst["key"].isin(
            ALWAYS.get(market, ())).to_numpy()
        return inst.loc[mask, "idx"].to_numpy(np.int64)

    def _valid_ends(self, bk: Bucket) -> np.ndarray:
        """Target-scale grid indices e allowed as window ends: every scale has its history, the window ends
        inside the split (grid[e] <= split end) and the latest anchor's targets start after the split start."""
        st = self.stores[bk.market]
        if bk.scale not in st.scales:
            return np.zeros(0, dtype=np.int64)
        grid = st.scales[bk.scale].grid
        P, L = 8, bk.n_patches[0] * 8
        e = np.arange(L - 1, len(grid))
        ok = np.ones(len(e), dtype=bool)
        if self.lo_ns is not None:
            ok &= grid[e - bk.horizon * P + 1] > self.lo_ns
        if self.hi_ns is not None:
            ok &= grid[e] <= self.hi_ns
        for s, n in zip(bk.scales[1:], bk.n_patches[1:]):  # coarse history must exist on its grid
            if s not in st.scales:
                return np.zeros(0, dtype=np.int64)
            gc = st.scales[s].grid
            ok &= np.searchsorted(gc, grid[e], side="right") - 1 >= n * P - 1
        return e[ok]

    def _anchors(self, bk: Bucket, e: int, rng: np.random.Generator) -> np.ndarray:
        """A anchor patches whose targets all lie inside the split (with replacement if fewer exist)."""
        grid, P, h = self.stores[bk.market].scales[bk.scale].grid, 8, bk.horizon
        lo, hi = bk.anchor_range
        p = np.arange(lo, hi + 1)
        w0 = e - bk.n_patches[0] * P + 1
        ok = np.ones(len(p), dtype=bool)
        if self.lo_ns is not None:
            ok &= grid[w0 + (p + 1) * P] > self.lo_ns
        if self.hi_ns is not None:
            ok &= grid[w0 + (p + 1 + h) * P - 1] <= self.hi_ns
        p = p[ok]
        return np.sort(rng.choice(p, size=self.A, replace=len(p) < self.A))

    def window_offsets(self, k: int, e: int, cand: np.ndarray):
        """Window start per scale for window end e, and store row offsets [n_scales, len(cand)] (-1: the
        instrument does not cover that window contiguously)."""
        bk, P = self.buckets[k], 8
        st = self.stores[bk.market]
        Ls = [n * P for n in bk.n_patches]
        t_end = st.scales[bk.scale].grid[e]
        gs = [e - Ls[0] + 1] + [int(np.searchsorted(st.scales[s].grid, t_end, side="right")) - Lb
                                for s, Lb in zip(bk.scales[1:], Ls[1:])]
        cand = np.asarray(cand, dtype=np.int64)
        offs = np.stack([st.scales[s].locate(cand, np.full(len(cand), g), Lb) for s, g, Lb in zip(bk.scales, gs, Ls)])
        return gs, offs

    def build(self, k: int, ends: np.ndarray, cands: List[np.ndarray], anchors: np.ndarray) -> Dict[str, np.ndarray]:
        """Batch for window ends [B] (target grid index), candidate instruments per sample (in order; the first G
        covered at every scale are used) and anchor patches [B, A]."""
        bk, G, P = self.buckets[k], self.G, 8
        st = self.stores[bk.market]
        B, Ls = len(ends), [n * P for n in bk.n_patches]
        xs = [np.zeros((B, G, Lb, len(CHANNELS)), dtype=np.float16) for Lb in Ls]
        starts = [np.zeros(B, dtype=np.int64) for _ in Ls]
        inst = np.zeros((B, G), dtype=np.int64)
        valid = np.zeros((B, G), dtype=bool)
        for i, (e, cand) in enumerate(zip(ends, cands)):
            gs, offs = self.window_offsets(k, int(e), cand)
            for j, g in enumerate(gs):
                starts[j][i] = g
            if len(cand) == 0:
                continue
            take = np.flatnonzero(np.all(offs >= 0, axis=0))[:G]
            for slot, c in enumerate(take):
                for j, s in enumerate(bk.scales):
                    o = int(offs[j, c])
                    xs[j][i, slot] = st.scales[s].X[o: o + Ls[j]]
                inst[i, slot] = cand[c]
                valid[i, slot] = True
        meta = st.instruments.set_index("idx").loc[inst.reshape(-1)]
        return {"bucket": np.int64(k), **{f"x{j}": x for j, x in enumerate(xs)},
                **{f"start{j}": s for j, s in enumerate(starts)}, "anchors": anchors.astype(np.int64),
                "inst": meta["vocab"].to_numpy(np.int64).reshape(B, G) * valid,
                "market": np.full((B, G), MARKET_ID[bk.market], dtype=np.int64),
                "chain": meta["chain"].to_numpy(np.int64).reshape(B, G) * valid, "valid": valid,
                "is_anchor": valid & np.isin(inst, self.always[bk.market]),   # market anchors (conditioning sets)
                "inst_idx": np.where(valid, inst, -1)}

    def sample(self, k: int, rng: np.random.Generator) -> Optional[Dict[str, np.ndarray]]:
        bk, B, G = self.buckets[k], self.B[k], self.G
        ends_ok = self._ends[k]
        if len(ends_ok) == 0:
            return None
        st = self.stores[bk.market]
        grid = st.scales[bk.scale].grid
        ends, cands = [], []
        for _ in range(B):
            for _ in range(8):  # retry moments with no instrument in the sampling set
                e = int(ends_ok[rng.integers(len(ends_ok))])
                act = st.active_at(int(grid[e]))
                if act is not None and len(act):
                    break
            ends.append(e)
            a = self.always[bk.market]
            cands.append(np.concatenate([a, rng.permutation(np.setdiff1d(act, a))])[: 4 * G]
                         if act is not None else np.zeros(0, dtype=np.int64))
        anchors = np.stack([self._anchors(bk, e, rng) for e in ends])
        return self.build(k, np.asarray(ends), cands, anchors)


def train_bars(root: Path, bk: Bucket) -> int:
    """Bars of the bucket's target scale (all instruments) that close on or before TRAIN_END."""
    d = root / "store" / bk.market / bk.scale
    if not (d / "grid.npy").is_file():
        return 0
    grid = np.load(d / "grid.npy")
    r = pd.read_parquet(d / "runs.parquet", columns=["grid_start", "length"])
    g_end = np.searchsorted(grid, TRAIN_END.value, side="right")
    g0 = r["grid_start"].to_numpy(np.int64)
    return int(np.clip(np.minimum(g0 + r["length"].to_numpy(np.int64), g_end) - g0, 0, None).sum())


class _Stream(IterableDataset):
    """Training (n_batches None): endless, bucket drawn from `weights` at every batch - a shared-memory tensor
    the main process may update while persistent workers run. Validation: n_batches, buckets in turn."""

    def __init__(self, sampler_args: dict, weights: torch.Tensor, seed: int, n_batches: Optional[int]) -> None:
        self.args, self.w, self.seed, self.n = sampler_args, weights, seed, n_batches
        self.sampler: Optional[GroupSampler] = None

    def __iter__(self):
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        if self.sampler is None:
            self.sampler = GroupSampler(**self.args)  # memmaps opened inside the worker
        rng = np.random.default_rng([self.seed, wid])
        i = 0
        while self.n is None or i < self.n:
            if self.n is None:
                p = self.w.numpy().astype(np.float64)
                k = int(rng.choice(len(p), p=p / p.sum()))
            else:
                k = i % len(self.w)
            if self.n is None or i % nw == wid:
                b = self.sampler.sample(k, rng)
                if b is not None:
                    yield {key: torch.from_numpy(np.asarray(v)) for key, v in b.items()}
            i += 1


class StoreData(L.LightningDataModule):
    def __init__(self, root: str, buckets: List[str], G: int = 32, A: int = 8, batch_tokens: int = 32768,
                 weights: Optional[List[float]] = None, num_workers: int = 8, val_batches_per_bucket: int = 16,
                 seed: int = 0) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.buckets = [Bucket(*b.split(":")) for b in buckets]
        self.bucket_names = [b.name for b in self.buckets]
        self._tables: Dict[tuple, tuple] = {}
        w = torch.tensor(weights or [1.0] * len(self.buckets), dtype=torch.float64)
        self.train_weights = (w / w.sum()).share_memory_()   # read by the training workers at every draw

    def set_weights(self, p: Sequence[float]) -> None:
        """New bucket probabilities for training; persistent workers use them from their next draw (batches
        already prefetched, up to num_workers x prefetch_factor, keep the old mix)."""
        self.train_weights.copy_(torch.as_tensor(np.asarray(p, dtype=np.float64)))

    def bucket_sizes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per bucket: training bars, and target-bar uses per batch (B x G x A x horizon x P)."""
        h = self.hparams
        bars = np.array([train_bars(Path(h.root), b) for b in self.buckets], dtype=np.float64)
        uses = np.array([max(1, round(h.batch_tokens / b.tokens(h.G))) * h.G * h.A * b.horizon * 8
                         for b in self.buckets], dtype=np.float64)
        return bars, uses

    def _args(self, split: str) -> dict:
        h = self.hparams
        return {"root": Path(h.root), "buckets": self.buckets, "G": h.G, "A": h.A, "batch_tokens": h.batch_tokens,
                "split": split}

    def train_dataloader(self) -> DataLoader:
        h = self.hparams
        ds = _Stream(self._args("train"), self.train_weights, h.seed, None)
        return DataLoader(ds, batch_size=None, num_workers=h.num_workers, pin_memory=True,
                          persistent_workers=h.num_workers > 0, prefetch_factor=4 if h.num_workers else None)

    def val_dataloader(self) -> DataLoader:
        h = self.hparams
        ds = _Stream(self._args("val"), torch.ones(len(self.buckets)), 12345,
                     h.val_batches_per_bucket * len(self.buckets))
        return DataLoader(ds, batch_size=None, num_workers=min(h.num_workers, 4), pin_memory=True)

    def _gpu_tables(self, market: str, scale: str, device) -> tuple:
        key = (market, scale, str(device))
        if key not in self._tables:
            st = MarketStore(Path(self.hparams.root), market, [scale]).scales[scale]
            self._tables[key] = (torch.from_numpy(st.grid).to(device), torch.from_numpy(st.cal).to(device))
        return self._tables[key]

    def on_after_batch_transfer(self, batch: Dict[str, torch.Tensor], dataloader_idx: int) -> Dict[str, torch.Tensor]:
        bk = self.buckets[int(batch["bucket"])]
        cfg = self.trainer.lightning_module.cfg
        b = to_model_batch(batch, bk, lambda m, s, d: self._gpu_tables(m, s, d), cfg)
        audit(b, split_end_ns=TRAIN_END.value if self.trainer.training else None)
        return b


def to_model_batch(batch: Dict[str, torch.Tensor], bk: Bucket, tables, cfg) -> Dict[str, torch.Tensor]:
    """Sampler output (on the GPU) -> model batch. tables(market, scale, device) -> (grid, cal) tensors."""
    dev = batch["anchors"].device
    wins = []
    for j, s in enumerate(bk.scales):
        grid, cal = tables(bk.market, s, dev)
        wins.append(ScaleWindow(batch[f"x{j}"], batch[f"start{j}"], grid, cal, SCALE_ID[s]))
    B = batch["anchors"].shape[0]
    meta = {k: batch[k] for k in ("inst", "market", "chain", "valid", "is_anchor") if k in batch}
    meta["target_scale"] = torch.full((B,), SCALE_ID[bk.scale], dtype=torch.long, device=dev)
    b = assemble(wins, batch["anchors"], bk.horizon, meta, cfg)
    b["bucket"] = batch["bucket"]
    return b
