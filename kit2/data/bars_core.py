"""Bar schema and helpers shared by the market builders (us/bars.py, crypto/bars.py).

Every builder returns the core columns, plus its own market columns:
  ts       bar close time (naive UTC) = the time the bar becomes known
  open/high/low/close, volume, dvol (dollar volume), n_min (observations: 1-min rows or swaps)
  gap_adj  log corporate-action correction for the bar's opening gap (0 for crypto)
  seg      data segment; a new segment starts after a coverage gap
kit2.data.features reads only these core columns, so it works for every market.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCALE_SECONDS: Dict[str, int] = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200,
                                 "1d": 86400}
CORE_COLS: Tuple[str, ...] = ("ts", "open", "high", "low", "close", "volume", "dvol", "n_min", "gap_adj", "seg")
DATETIME_COLS = ("ts", "session")


def typed(bars: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Fixed column order and dtypes, so the output never depends on which rows happened to be missing."""
    out = bars.reindex(columns=list(cols)).reset_index(drop=True)
    for c in cols:
        if c in DATETIME_COLS:
            out[c] = out[c].astype("datetime64[ns]")
        elif c == "seg":
            out[c] = out[c].astype(np.int64)
        else:
            out[c] = out[c].astype(np.float64)
    return out


def empty(cols: Sequence[str]) -> pd.DataFrame:
    return typed(pd.DataFrame(columns=list(cols)), cols)


def ohlc(o: str = "open", h: str = "high", l: str = "low", c: str = "close") -> Dict[str, Tuple[str, str]]:
    """Named aggregations for groupby(...).agg(**ohlc()); pass one price column four times for tick data."""
    return {"open": (o, "first"), "high": (h, "max"), "low": (l, "min"), "close": (c, "last")}


def fill_flat(bars: pd.DataFrame, zero_cols: Iterable[str], seg: Optional[np.ndarray] = None) -> Optional[pd.DataFrame]:
    """Drop grid bars before the first priced bar of their segment; a missing bar becomes flat at the
    previous close of the same segment with zero flow (a price never carries across a coverage gap).
    Uses only earlier bars. Sets `seg`; returns None if no bar has a price. The index is kept."""
    seg = np.zeros(len(bars), dtype=np.int64) if seg is None else np.asarray(seg, dtype=np.int64)
    priced = pd.Series(bars["close"].notna().to_numpy(np.int8))
    started = priced.groupby(seg).cummax().to_numpy().astype(bool)
    if not started.any():
        return None
    bars = bars.assign(seg=seg)[started].copy()
    close = bars.groupby("seg")["close"].ffill()
    for k in ("open", "high", "low"):
        bars[k] = bars[k].fillna(close)
    bars["close"] = close
    zero_cols = list(zero_cols)
    bars[zero_cols] = bars[zero_cols].fillna(0.0)
    return bars


def segments(period_idx: np.ndarray) -> np.ndarray:
    """Segment id per bar from calendar period numbers (trading-session index, month ordinal): a new
    segment wherever consecutive bars skip a period."""
    return np.cumsum(np.r_[False, np.diff(period_idx) > 1])
