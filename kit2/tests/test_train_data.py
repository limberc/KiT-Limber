"""Real-data batches: split purging, window contents, coarse-scale alignment (synthetic on-disk store)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from kit2.data.store import CHANNELS, TRAIN_END, VAL_END, MarketStore, normalise, write_scale
from kit2.data.us.prepare import us_grid
from kit2.train.batch import ScaleWindow, assemble, audit
from kit2.train.buckets import SCALE_ID, Bucket
from kit2.train.data import GroupSampler
from kit2.model.config import preset

START, END = "2026-03-02", "2026-10-30"
SCALES = ("1h", "2h", "1d")
N_INST = 12


def encode(inst, pos):
    """Clip-safe tags: instrument id in 0.25 steps, grid position as two base-16 digits in 0.25 steps."""
    pos = np.asarray(pos)
    return np.column_stack([np.full(len(pos), inst * 0.25), (pos % 16) * 0.25 + (pos // 16 % 16) * 0.015625])


def expected(inst, pos, stats):
    raw = np.zeros((len(pos), len(CHANNELS)), dtype=np.float32)
    raw[:, :2] = encode(inst, pos)
    return normalise(raw, stats, np.zeros(len(pos), bool))[:, :2]


STATS = {"center": [0.0] * len(CHANNELS), "scale": [1.0] * len(CHANNELS), "gap_first_scale": 1.0}


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("kit2data")
    d = root / "store" / "us"
    d.mkdir(parents=True)
    pd.DataFrame({"idx": np.arange(N_INST), "key": [f"S{i}" for i in range(N_INST)], "vocab": np.arange(N_INST) + 1,
                  "market": 1, "chain": 0}).to_parquet(d / "instruments.parquet")
    months = pd.date_range("2025-01-01", END, freq="MS")
    pd.DataFrame([(m, i) for m in months for i in range(N_INST)], columns=["month", "idx"]).to_parquet(d / "active.parquet")
    stats = STATS
    for s in SCALES:
        start = "2025-01-02" if s == "1d" else START
        grid, cal = us_grid(s, start, END)
        rng = np.random.default_rng(SCALES.index(s))
        parts = []
        for i in range(N_INST):
            lo = int(rng.integers(0, len(grid) // 10))          # instruments start at different times
            n = len(grid) - lo
            raw = np.zeros((n, len(CHANNELS)), dtype=np.float32)
            raw[:, :2] = encode(i, np.arange(lo, len(grid)))    # channels 0/1 = instrument id, grid position
            parts.append((i, raw, np.arange(lo, len(grid)), np.zeros(n, dtype=np.int64)))
        write_scale(d / s, grid, cal, parts, stats)
    (d / "stats.json").write_text(json.dumps({"scales": {s: stats for s in SCALES}}))
    return root


def batch_for(root, split, seed=0):
    bk = Bucket("us", "1h")
    gs = GroupSampler(root, [bk], G=8, A=4, batch_tokens=8 * sum(bk.n_patches) * 6, split=split)
    raw = gs.sample(0, np.random.default_rng(seed))
    st = MarketStore(root, "us", list(bk.scales))
    wins = [ScaleWindow(torch.from_numpy(raw[f"x{j}"]).float(), torch.from_numpy(raw[f"start{j}"]),
                        torch.from_numpy(st.scales[s].grid), torch.from_numpy(st.scales[s].cal).float(), SCALE_ID[s])
            for j, s in enumerate(bk.scales)]
    meta = {k: torch.from_numpy(raw[k]) for k in ("inst", "market", "chain", "valid")}
    meta["target_scale"] = torch.full((len(raw["anchors"]),), SCALE_ID["1h"])
    return raw, assemble(wins, torch.from_numpy(raw["anchors"]), bk.horizon, meta, preset("tiny")), st, bk


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_train_targets_before_train_end(root, seed):
    raw, b, _, _ = batch_for(root, "train", seed)
    audit(b, split_end_ns=TRAIN_END.value)
    assert raw["valid"].any()


@pytest.mark.parametrize("seed", [0, 1])
def test_val_targets_inside_val_period(root, seed):
    raw, b, _, _ = batch_for(root, "val", seed)
    audit(b)
    assert bool((b["fut_ts"] > TRAIN_END.value).all()) and bool((b["fut_ts"] <= VAL_END.value).all())


def test_windows_are_exact_store_rows(root):
    raw, b, st, bk = batch_for(root, "train", 3)
    for j, s in enumerate(bk.scales):
        x, start = raw[f"x{j}"].astype(np.float32), raw[f"start{j}"]
        L = x.shape[2]
        for i in range(x.shape[0]):
            for g in np.flatnonzero(raw["valid"][i]):
                inst = int(raw["inst"][i, g]) - 1
                want = expected(inst, np.arange(start[i], start[i] + L), STATS).astype(np.float32)
                assert np.array_equal(x[i, g, :, :2], want)          # right instrument, right grid positions
    # coarse windows end at the last coarse bar closing at or before the target window's end
    tgt_end = st.scales["1h"].grid[raw["start0"] + raw["x0"].shape[2] - 1]
    for j, s in enumerate(bk.scales[1:], 1):
        g = st.scales[s].grid
        end = raw[f"start{j}"] + raw[f"x{j}"].shape[2] - 1
        assert np.all(g[end] <= tgt_end) and np.all(g[np.minimum(end + 1, len(g) - 1)] > tgt_end)
