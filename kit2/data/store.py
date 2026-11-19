"""Training store: normalised bar channels per (market, scale), memory-mapped, plus the tables a sampler needs.

Layout under <data_root>/store/<market>/:
  instruments.parquet       idx, key, vocab, market, chain   (one row per instrument, shared by all scales)
  active.parquet            month, idx   (instruments a sample may draw at that month; decided before it)
  stats.json                per (scale, channel): center, scale; fit on bars with ts <= train_end only
  <scale>/grid.npy          int64 bar close times (ns, naive UTC) of the market's bar grid
  <scale>/cal.npy           float16 [n_grid, CAL_DIM] calendar features (published in advance)
  <scale>/X.npy             float16 [n_bars, CHANNELS] normalised channels, runs back to back
  <scale>/runs.parquet      idx, grid_start, length, offset   (one contiguous stretch of bars on the grid)

Channels: r_gap, r_body, r_up, r_dn, v (generated), has_obs, aux0..aux3 (history-only inputs).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .features import FEATURES

CHANNELS = (*FEATURES, "has_obs", "aux0", "aux1", "aux2", "aux3")
N_CANDLE = len(FEATURES)
CAL_DIM = 12
CAL_FIRST = 10                     # calendar column: first bar of the session (known in advance)
SOFT_CLIP = 6.0
TRAIN_END = pd.Timestamp("2026-06-30 23:59:59")
VAL_END = pd.Timestamp("2026-08-31 23:59:59")


# ---- calendar ------------------------------------------------------------------------------------------
def calendar_features(ts: np.ndarray, local_minute: np.ndarray, sess_pos: np.ndarray, sess_len: np.ndarray) -> np.ndarray:
    """[n, 12]: time-of-day (2 harmonics), day of week, day of year, session position / remaining,
    first / last bar of session. ts naive UTC ns; local_minute = exchange-local minute of the bar close."""
    t = pd.DatetimeIndex(ts)
    tod = 2 * np.pi * local_minute / 1440.0
    dow = 2 * np.pi * t.dayofweek.to_numpy() / 7.0
    doy = 2 * np.pi * (t.dayofyear.to_numpy() - 1) / 365.25
    denom = np.maximum(sess_len - 1, 1)
    out = np.stack([np.sin(tod), np.cos(tod), np.sin(2 * tod), np.cos(2 * tod), np.sin(dow), np.cos(dow),
                    np.sin(doy), np.cos(doy), sess_pos / denom, (sess_len - 1 - sess_pos) / denom,
                    (sess_pos == 0).astype(float), (sess_pos == sess_len - 1).astype(float)], axis=1)
    return out.astype(np.float16)


# ---- normalisation -------------------------------------------------------------------------------------
def _robust_scale(x: np.ndarray) -> float:
    """max(1.4826 MAD, q90(|x - median|) / 1.645): equal for Gaussian data; the quantile term keeps the scale
    from collapsing on zero-heavy channels (intraday gaps, wicks), where MAD alone would clip the tail."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    dev = np.abs(x - np.median(x))
    s = max(1.4826 * float(np.median(dev)), float(np.quantile(dev, 0.9)) / 1.645)
    return s if s > 1e-12 else max(float(np.std(x)), 1e-6)


def fit_stats(samples: np.ndarray, first: np.ndarray) -> Dict[str, List[float]]:
    """samples [n, CHANNELS] raw values from training-period bars; first [n] = first bar of its session.
    Candle features keep KiT's convention (no centering, scale = 1.4826 MAD); aux channels are centered on
    the median; has_obs is left as is. The open gap of a session's first bar (overnight / weekend) gets its own
    scale: intraday gaps are tiny, and one shared scale would clip every overnight gap."""
    center, scale = [], []
    for j, name in enumerate(CHANNELS):
        x = samples[:, j]
        if name == "has_obs":
            center.append(0.0), scale.append(1.0)
            continue
        if name == "r_gap":
            x = x[~first]
        xf = x[np.isfinite(x)]
        center.append(0.0 if j < N_CANDLE or xf.size == 0 else float(np.median(xf)))
        scale.append(_robust_scale(x))
    gap_first = samples[first, CHANNELS.index("r_gap")]
    return {"center": center, "scale": scale,
            "gap_first_scale": _robust_scale(gap_first) if gap_first.size > 100 else scale[0]}


def channel_scales(stats: Dict[str, List[float]], first: np.ndarray) -> np.ndarray:
    """[n, CHANNELS] per-row scales (r_gap uses the first-bar scale on session-first bars)."""
    s = np.broadcast_to(np.asarray(stats["scale"], dtype=np.float64), (len(first), len(CHANNELS))).copy()
    s[first, CHANNELS.index("r_gap")] = stats["gap_first_scale"]
    return s


def normalise(raw: np.ndarray, stats: Dict[str, List[float]], first: np.ndarray) -> np.ndarray:
    """raw [n, CHANNELS] -> soft-clipped normalised float16; NaN (aux missing on that bar) -> 0."""
    y = (raw - np.asarray(stats["center"])) / channel_scales(stats, first)
    y = SOFT_CLIP * np.tanh(y / SOFT_CLIP)
    y[:, CHANNELS.index("has_obs")] = raw[:, CHANNELS.index("has_obs")]
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)


# ---- writer --------------------------------------------------------------------------------------------
def runs_from_segments(grid_idx: np.ndarray, seg: np.ndarray) -> List[tuple]:
    """(row_start, row_end, grid_start) for every stretch of consecutive grid positions in one segment."""
    brk = np.r_[True, (np.diff(grid_idx) != 1) | (np.diff(seg) != 0)]
    starts = np.flatnonzero(brk)
    ends = np.r_[starts[1:], len(grid_idx)]
    return [(int(a), int(b), int(grid_idx[a])) for a, b in zip(starts, ends)]


def write_scale(out_dir: Path, grid: np.ndarray, cal: np.ndarray, parts: Iterable[tuple],
                stats: Dict[str, List[float]]) -> int:
    """parts: (idx, raw [n, CHANNELS] float32, grid_idx [n], seg [n]) per instrument. Writes X.npy + runs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "grid.npy", grid.astype(np.int64))
    np.save(out_dir / "cal.npy", cal.astype(np.float16))
    parts = list(parts)
    n_total = sum(len(p[1]) for p in parts)
    X = np.lib.format.open_memmap(out_dir / "X.npy.tmp", mode="w+", dtype=np.float16, shape=(n_total, len(CHANNELS)))
    runs, off = [], 0
    first_of = cal[:, CAL_FIRST] > 0.5
    for idx, raw, gidx, seg in parts:
        X[off: off + len(raw)] = normalise(raw, stats, first_of[gidx])
        for a, b, g0 in runs_from_segments(gidx, seg):
            runs.append((idx, g0, b - a, off + a))
        off += len(raw)
    X.flush()
    del X
    (out_dir / "X.npy.tmp").replace(out_dir / "X.npy")
    pd.DataFrame(runs, columns=["idx", "grid_start", "length", "offset"]).to_parquet(out_dir / "runs.parquet")
    return n_total


# ---- reader --------------------------------------------------------------------------------------------
class ScaleStore:
    """Read side of one (market, scale): memory-mapped X, grid, calendar, run lookup."""

    def __init__(self, d: Path) -> None:
        self.X = np.load(d / "X.npy", mmap_mode="r")
        self.grid = np.load(d / "grid.npy")
        self.cal = np.load(d / "cal.npy")
        r = pd.read_parquet(d / "runs.parquet").sort_values(["idx", "grid_start"])
        self.r_idx = r["idx"].to_numpy(np.int64)
        self.r_g0 = r["grid_start"].to_numpy(np.int64)
        self.r_len = r["length"].to_numpy(np.int64)
        self.r_off = r["offset"].to_numpy(np.int64)
        self._key = self.r_idx * (1 << 40) + self.r_g0

    def locate(self, idx: np.ndarray, g_start: np.ndarray, length: int) -> np.ndarray:
        """Row offset in X of grid positions [g_start, g_start + length) for each instrument, or -1."""
        k = np.searchsorted(self._key, idx * (1 << 40) + g_start, side="right") - 1
        k = np.clip(k, 0, len(self._key) - 1)
        ok = (self.r_idx[k] == idx) & (g_start >= self.r_g0[k]) & (g_start + length <= self.r_g0[k] + self.r_len[k])
        ok &= g_start >= 0
        return np.where(ok, self.r_off[k] + g_start - self.r_g0[k], -1)


class MarketStore:
    def __init__(self, root: Path, market: str, scales: Sequence[str]) -> None:
        d = root / "store" / market
        self.market = market
        self.instruments = pd.read_parquet(d / "instruments.parquet").sort_values("idx").reset_index(drop=True)
        self.stats = json.loads((d / "stats.json").read_text())
        act = pd.read_parquet(d / "active.parquet")
        self.active: Dict[pd.Timestamp, np.ndarray] = {m: g["idx"].to_numpy(np.int64) for m, g in act.groupby("month")}
        self.scales = {s: ScaleStore(d / s) for s in scales if (d / s / "X.npy").is_file()}

    def active_at(self, ts_ns: int) -> Optional[np.ndarray]:
        return self.active.get(pd.Timestamp(ts_ns).normalize().replace(day=1))
