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
from typing import Dict, List, Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from kit2.data.store import CHANNELS, TRAIN_END, VAL_END, MarketStore
from kit2.train.batch import ScaleWindow, assemble, audit
from kit2.train.buckets import MARKET_ID, SCALE_ID, Bucket

ALWAYS = {"us": ("SPY", "QQQ")}          # market anchors put in every US group when covered
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
        inst = self.stores[market].instruments
        return inst.loc[inst["key"].isin(ALWAYS.get(market, ())), "idx"].to_numpy(np.int64)

    def _valid_ends(self, bk: Bucket) -> np.ndarray:
        """Target-scale grid indices e allowed as window ends: all scales have enough history, and every bar
        that can be a target lies inside the split's time range."""
        st = self.stores[bk.market]
        if bk.scale not in st.scales:
            return np.zeros(0, dtype=np.int64)
        grid = st.scales[bk.scale].grid
        P, L = 8, bk.n_patches[0] * 8
        lo_a, _ = bk.anchor_range
        e = np.arange(L - 1, len(grid))
        first_target = e - L + 1 + (lo_a + 1) * P
        ok = np.ones(len(e), dtype=bool)
        if self.lo_ns is not None:
            ok &= grid[first_target] > self.lo_ns
        if self.hi_ns is not None:
            ok &= grid[e] <= self.hi_ns
        for s, n in zip(bk.scales[1:], bk.n_patches[1:]):            # coarse history must exist on its grid
            if s not in st.scales:
                return np.zeros(0, dtype=np.int64)
            gc = st.scales[s].grid
            ok &= np.searchsorted(gc, grid[e], side="right") - 1 >= n * P - 1
        return e[ok]

    def sample(self, k: int, rng: np.random.Generator) -> Optional[Dict[str, np.ndarray]]:
        bk, B, G, P = self.buckets[k], self.B[k], self.G, 8
        ends = self._ends[k]
        if len(ends) == 0:
            return None
        st = self.stores[bk.market]
        tgt = st.scales[bk.scale]
        Ls = [n * P for n in bk.n_patches]
        xs = [np.zeros((B, G, Lb, len(CHANNELS)), dtype=np.float16) for Lb in Ls]
        starts = [np.zeros(B, dtype=np.int64) for _ in Ls]
        inst = np.zeros((B, G), dtype=np.int64)
        valid = np.zeros((B, G), dtype=bool)
        for i in range(B):
            for _ in range(8):                                      # retry moments with no covered instrument
                e = int(ends[rng.integers(len(ends))])
                act = st.active_at(int(tgt.grid[e]))
                if act is not None and len(act):
                    break
            else:
                continue
            gs = [e - Ls[0] + 1]
            for s, Lb in zip(bk.scales[1:], Ls[1:]):
                gc = st.scales[s].grid
                gs.append(int(np.searchsorted(gc, tgt.grid[e], side="right")) - Lb)
            for j, g in enumerate(gs):
                starts[j][i] = g
            anchors = self.always[bk.market]
            cand = np.concatenate([anchors, rng.permutation(np.setdiff1d(act, anchors))])[: 4 * G]
            offs = []
            for s, g, Lb in zip(bk.scales, gs, Ls):
                offs.append(st.scales[s].locate(cand, np.full(len(cand), g), Lb))
            ok = np.all(np.stack(offs) >= 0, axis=0)
            take = np.flatnonzero(ok)[:G]
            for slot, c in enumerate(take):
                for j, s in enumerate(bk.scales):
                    o = int(offs[j][c])
                    xs[j][i, slot] = st.scales[s].X[o: o + Ls[j]]
                inst[i, slot] = cand[c]
                valid[i, slot] = True
        lo, hi = bk.anchor_range
        anchors = np.sort(np.stack([rng.choice(np.arange(lo, hi + 1), size=self.A, replace=False) for _ in range(B)]), 1)
        meta = st.instruments.set_index("idx").loc[inst.reshape(-1)]
        return {"bucket": np.int64(k), **{f"x{j}": x for j, x in enumerate(xs)},
                **{f"start{j}": s for j, s in enumerate(starts)}, "anchors": anchors,
                "inst": meta["vocab"].to_numpy(np.int64).reshape(B, G) * valid,
                "market": np.full((B, G), MARKET_ID[bk.market], dtype=np.int64),
                "chain": meta["chain"].to_numpy(np.int64).reshape(B, G) * valid, "valid": valid}


class _Stream(IterableDataset):
    def __init__(self, sampler_args: dict, weights: List[float], seed: int, n_batches: Optional[int]) -> None:
        self.args, self.w, self.seed, self.n = sampler_args, np.asarray(weights, float), seed, n_batches
        self.sampler: Optional[GroupSampler] = None

    def __iter__(self):
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        if self.sampler is None:
            self.sampler = GroupSampler(**self.args)                # memmaps opened inside the worker
        rng = np.random.default_rng([self.seed, wid])
        p = self.w / self.w.sum()
        i = 0
        while self.n is None or i < self.n:
            k = int(rng.choice(len(p), p=p)) if self.n is None else i % len(p)
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

    def _args(self, split: str) -> dict:
        h = self.hparams
        return {"root": Path(h.root), "buckets": self.buckets, "G": h.G, "A": h.A, "batch_tokens": h.batch_tokens,
                "split": split}

    def train_dataloader(self) -> DataLoader:
        h = self.hparams
        ds = _Stream(self._args("train"), h.weights or [1.0] * len(self.buckets), h.seed, None)
        return DataLoader(ds, batch_size=None, num_workers=h.num_workers, pin_memory=True,
                          persistent_workers=h.num_workers > 0, prefetch_factor=4 if h.num_workers else None)

    def val_dataloader(self) -> DataLoader:
        h = self.hparams
        ds = _Stream(self._args("val"), [1.0] * len(self.buckets), 12345, h.val_batches_per_bucket * len(self.buckets))
        return DataLoader(ds, batch_size=None, num_workers=min(h.num_workers, 4), pin_memory=True)

    def _gpu_tables(self, market: str, scale: str, device) -> tuple:
        key = (market, scale, str(device))
        if key not in self._tables:
            st = MarketStore(Path(self.hparams.root), market, [scale]).scales[scale]
            self._tables[key] = (torch.from_numpy(st.grid).to(device), torch.from_numpy(st.cal).to(device))
        return self._tables[key]

    def on_after_batch_transfer(self, batch: Dict[str, torch.Tensor], dataloader_idx: int) -> Dict[str, torch.Tensor]:
        bk = self.buckets[int(batch["bucket"])]
        dev = batch["anchors"].device
        wins = []
        for j, s in enumerate(bk.scales):
            grid, cal = self._gpu_tables(bk.market, s, dev)
            wins.append(ScaleWindow(batch[f"x{j}"], batch[f"start{j}"], grid, cal, SCALE_ID[s]))
        B = batch["anchors"].shape[0]
        meta = {k: batch[k] for k in ("inst", "market", "chain", "valid")}
        meta["target_scale"] = torch.full((B,), SCALE_ID[bk.scale], dtype=torch.long, device=dev)
        cfg = self.trainer.lightning_module.cfg if self.trainer is not None else None
        b = assemble(wins, batch["anchors"], bk.horizon, meta, cfg)
        split_end = TRAIN_END.value if (self.trainer is not None and self.trainer.training) else None
        audit(b, split_end_ns=split_end)
        b["bucket"] = batch["bucket"]
        return b
