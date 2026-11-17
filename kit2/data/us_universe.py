"""Point-in-time US stock universe, chosen from daily data that includes delisted names.

Membership for month M uses only daily bars dated strictly before the first day of M:
rank by the median dollar volume (raw close x raw volume) of the last `lookback` sessions.
The delisting date only bounds the exchange-listed window; it never decides membership.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable

import numpy as np
import pandas as pd

EXCHANGES = {"NYSE", "NASDAQ", "NYSE MKT", "AMEX", "BATS", "NYSE ARCA"}
_SYMBOL = re.compile(r"^[A-Z]{1,5}(?:-[A-Z])?$")  # common + class shares (BRK-B); no warrants/units
_NOT_COMMON = re.compile(r"warrant|\bunits?\b|\brights?\b|preferred|depositary|\bnotes?\b|%|"
                         r"acquisition corp|when issued", re.I)

# Market / sector / rates / commodity / credit / volatility context, kept in every month.
CONTEXT_ETFS = ("SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
                "XLRE", "XLC", "SMH", "KRE", "TLT", "IEF", "SHY", "HYG", "LQD", "GLD", "SLV", "USO", "UNG",
                "UUP", "EEM", "EFA", "VIXY")


def candidates(active: pd.DataFrame, delisted: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Common stocks listed at some point in [start, end]; one row per listing window."""
    df = pd.concat([active, delisted], ignore_index=True)
    df["ipoDate"] = pd.to_datetime(df["ipoDate"], errors="coerce")
    df["delistingDate"] = pd.to_datetime(df["delistingDate"].replace("null", None), errors="coerce")
    keep = ((df["assetType"] == "Stock") & df["exchange"].isin(EXCHANGES)
            & df["symbol"].astype(str).str.match(_SYMBOL)
            & ~df["name"].astype(str).str.contains(_NOT_COMMON)
            & ~(df["delistingDate"] < pd.Timestamp(start))
            & ~(df["ipoDate"] > pd.Timestamp(end)))
    return df[keep].reset_index(drop=True)


def listing_windows(cands: pd.DataFrame) -> Dict[str, list]:
    """symbol -> [(ipo, delist or None), ...]; a reused symbol has several windows."""
    out: Dict[str, list] = {}
    for r in cands.itertuples():
        out.setdefault(r.symbol, []).append((r.ipoDate, r.delistingDate))
    return out


def _in_windows(dates: pd.Series, windows: list) -> np.ndarray:
    ok = np.zeros(len(dates), dtype=bool)
    for ipo, delist in windows:
        lo = dates >= ipo if pd.notna(ipo) else np.ones(len(dates), dtype=bool)
        hi = dates <= delist if pd.notna(delist) else np.ones(len(dates), dtype=bool)
        ok |= np.asarray(lo & hi)
    return ok


def monthly_universe(daily: Dict[str, pd.DataFrame], windows: Dict[str, list], months: Iterable[pd.Timestamp],
                     top_n: int = 1000, lookback: int = 63, min_price: float = 5.0,
                     max_stale_days: int = 10) -> pd.DataFrame:
    """Top `top_n` stocks per month by trailing median dollar volume, from bars before the month only.

    Rows: month, symbol, rank (1 = most liquid), adv (USD). A stock needs `lookback` sessions of history
    (seasoning), a last close >= min_price, and a bar within `max_stale_days` before the month start.
    """
    months = pd.DatetimeIndex(sorted(months))
    stats = []  # per symbol: (month idx array, adv, last close, last date)
    for sym, df in daily.items():
        df = df[_in_windows(df["date"], windows.get(sym, [(pd.NaT, pd.NaT)]))]
        if len(df) < lookback:
            continue
        dates = df["date"].to_numpy()
        dv = (df["close"] * df["volume"]).rolling(lookback, min_periods=lookback).median().to_numpy()
        idx = np.searchsorted(dates, months.to_numpy()) - 1  # last bar strictly before M
        ok = idx >= 0
        for mi in np.flatnonzero(ok):
            i = idx[mi]
            if not np.isfinite(dv[i]):
                continue
            stale = (months[mi] - pd.Timestamp(dates[i])).days
            if stale <= max_stale_days and df["close"].iloc[i] >= min_price:
                stats.append((months[mi], sym, float(dv[i])))
    uni = pd.DataFrame(stats, columns=["month", "symbol", "adv"])
    uni["rank"] = uni.groupby("month")["adv"].rank(ascending=False, method="first").astype(int)
    return uni[uni["rank"] <= top_n].sort_values(["month", "rank"]).reset_index(drop=True)
