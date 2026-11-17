"""Candle features from point-in-time bars (KiT's 5-dim state, KiT-2 conventions).

    r_gap  = ln(O_t / C_{t-1}) + gap_adj_t      gap_adj: corporate actions effective at t (0 otherwise)
    r_body = ln(C_t / O_t)
    r_up   = ln(H_t / max(O_t, C_t))  >= 0
    r_dn   = ln(min(O_t, C_t) / L_t)  >= 0
    v_t    = ln((DV_t + 1) / (EMA_t + 1)),  EMA_t from DV_0..DV_{t-1} only

DV is dollar volume (split-invariant). Every feature at bar t uses bars <= t only, and the first bar of a
segment (data gap) has r_gap = 0 and restarts the EMA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ("r_gap", "r_body", "r_up", "r_dn", "v")


def ema_alpha(half_life_bars: float) -> float:
    return float(1.0 - 2.0 ** (-1.0 / half_life_bars))


def causal_ema(x: np.ndarray, half_life_bars: float) -> np.ndarray:
    """EMA[t] from x[0..t-1]; EMA[0] = x[0]."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x.copy()
    y = pd.Series(x).ewm(alpha=ema_alpha(half_life_bars), adjust=False).mean().to_numpy()
    out = np.empty_like(x)
    out[0] = x[0]
    out[1:] = y[:-1]
    return out


def candle_features(bars: pd.DataFrame, half_life_bars: float) -> pd.DataFrame:
    """bars: ts, open, high, low, close, dvol, gap_adj, seg (ordered by ts). Returns the 5 features by row."""
    o, h, l, c = (bars[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    seg = bars["seg"].to_numpy()
    first = np.r_[True, seg[1:] != seg[:-1]]
    prev_c = np.r_[np.nan, c[:-1]]
    r_gap = np.where(first, 0.0, np.log(o / prev_c) + bars["gap_adj"].to_numpy(np.float64))
    out = pd.DataFrame({
        "r_gap": r_gap,
        "r_body": np.log(c / o),
        "r_up": np.log(h / np.maximum(o, c)),
        "r_dn": np.log(np.minimum(o, c) / l),
    }, index=bars.index)
    dv = bars["dvol"].to_numpy(np.float64)
    v = np.empty(len(dv))
    starts = np.flatnonzero(first)
    for s, e in zip(starts, np.r_[starts[1:], len(dv)]):
        ema = causal_ema(dv[s:e], half_life_bars)
        v[s:e] = np.log((dv[s:e] + 1.0) / (ema + 1.0))
    out["v"] = v
    return out
