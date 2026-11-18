"""US bars: raw Alpha Vantage data -> point-in-time bars on KiT's session grid, plus history-only aux signals.

Point-in-time contract (tested in kit2/tests/test_no_lookahead.py):
- Alpha Vantage labels a 1-min bar by its START; a minute is available at start + 1 min.
- A bar of scale S is stamped with its grid close `ts` and uses only minutes ending <= ts.
- Coverage (which months the stock is expected to trade) is an explicit input decided before the month
  (universe membership + warm-up), never inferred from data that arrives later. Missing minutes inside a
  covered session give flat bars at the previous close with zero volume.
- `gap_adj` holds corporate actions effective on the bar's session (first bar only): ln(split factor)
  and -ln(1 - dividend / previous close). Raw prices are never back-adjusted.
- Aux columns on the first bar of a session describe extended-hours trading that ended before the open.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from kit2.data.bars_core import CORE_COLS, SCALE_SECONDS, empty, fill_flat, ohlc, segments, typed

ET = "America/New_York"
INTRADAY_MIN = {k: s // 60 for k, s in SCALE_SECONDS.items() if k != "1d"}
AUX_COLS = ("aux_pre_ret", "aux_post_ret", "aux_pre_ldv", "aux_post_ldv")
BAR_COLS = ["ts", "session", *CORE_COLS[1:], *AUX_COLS]


@lru_cache(maxsize=4)
def create_cal(start: str = "2015-06-01", end: str = "2026-12-31") -> pd.DataFrame:
    """XNYS schedule (published in advance): session date, open/close as naive UTC, length in minutes."""
    import exchange_calendars as xc
    s = xc.get_calendar("XNYS", start=start, end=end).schedule
    out = pd.DataFrame({"session": s.index.tz_localize(None) if s.index.tz is not None else s.index,
                        "open": s["open"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(),
                        "close": s["close"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()})
    for k in ("session", "open", "close"):
        out[k] = out[k].astype("datetime64[ns]")
    out["minutes"] = ((out["close"] - out["open"]) / pd.Timedelta(minutes=1)).astype(int)
    return out.reset_index(drop=True)


def _corp_adjust(sess_first: pd.DataFrame, prev_close: np.ndarray, splits: Optional[pd.DataFrame],
                 divs: Optional[pd.DataFrame]) -> np.ndarray:
    """gap_adj per session-first bar: ln(split factor) - ln(1 - dividend / previous close)."""
    adj = np.zeros(len(sess_first))
    day = sess_first["session"].to_numpy()
    if splits is not None and len(splits):
        sf = splits.groupby("effective_date")["split_factor"].prod()
        m = pd.Series(day).map(sf).to_numpy(np.float64)
        adj += np.where(np.isfinite(m) & (m > 0), np.log(np.where(np.isfinite(m) & (m > 0), m, 1.0)), 0.0)
    if divs is not None and len(divs):
        d = divs[divs["amount"] > 0].groupby("ex_dividend_date")["amount"].sum()
        a = pd.Series(day).map(d).to_numpy(np.float64)
        frac = np.where(np.isfinite(a) & np.isfinite(prev_close) & (prev_close > 0), a / prev_close, 0.0)
        adj -= np.log1p(-np.clip(frac, 0.0, 0.5))
    return adj


def minutes_utc(raw: pd.DataFrame) -> pd.DataFrame:
    """Alpha Vantage 1-min rows (ts_start, US/Eastern naive) -> start/end as naive UTC."""
    start = (pd.DatetimeIndex(raw["ts_start"]).tz_localize(ET, ambiguous="NaT", nonexistent="NaT")
             .tz_convert("UTC").tz_localize(None))
    out = raw.assign(start=start.to_numpy().astype("datetime64[ns]")).dropna(subset=["start"])
    return out.assign(end=out["start"] + pd.Timedelta(minutes=1)).sort_values("start").reset_index(drop=True)


def build_intraday(raw: pd.DataFrame, scale: str, months: Iterable[pd.Period],
                   splits: Optional[pd.DataFrame] = None, divs: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """raw: concatenated Alpha Vantage 1-min rows (ts_start, open, high, low, close, volume), extended hours.
    months: covered months (Period 'M') - the download plan for this symbol, known before each month."""
    step = INTRADAY_MIN[scale]
    m = minutes_utc(raw)
    if m.empty:
        return empty(BAR_COLS)
    cal = create_cal()
    # session of each minute: the last session whose open <= start (extended minutes belong to that day's date)
    et_day = pd.DatetimeIndex(m["start"]).tz_localize("UTC").tz_convert(ET).tz_localize(None).normalize()
    cal_by_day = cal.set_index("session")
    known = et_day.isin(cal_by_day.index)
    m, et_day = m[known].reset_index(drop=True), et_day[known]
    if m.empty:
        return empty(BAR_COLS)
    m["session"] = et_day.astype("datetime64[ns]")
    m["s_open"] = cal_by_day.loc[et_day, "open"].to_numpy()
    m["s_close"] = cal_by_day.loc[et_day, "close"].to_numpy()
    m["dvol"] = m["close"] * m["volume"]
    reg = m[(m["start"] >= m["s_open"]) & (m["end"] <= m["s_close"])].copy()
    pre = m[m["end"] <= m["s_open"]]
    post = m[m["start"] >= m["s_close"]]

    # full grid of bars over the covered months' sessions
    covered = cal[cal["session"].dt.to_period("M").isin(set(pd.PeriodIndex(list(months), freq="M")))]
    n_bars = -(-covered["minutes"].to_numpy() // step)
    g_sess = np.repeat(covered["session"].to_numpy(), n_bars)
    g_j = np.concatenate([np.arange(n) for n in n_bars])
    g_open = np.repeat(covered["open"].to_numpy(), n_bars)
    g_len = np.repeat(covered["minutes"].to_numpy(), n_bars)
    g_ts = g_open + pd.to_timedelta(np.minimum((g_j + 1) * step, g_len), unit="min").to_numpy()
    grid = pd.DataFrame({"session": g_sess, "j": g_j, "ts": g_ts})
    grid["sess_idx"] = np.repeat(cal.index[cal["session"].isin(covered["session"])].to_numpy(), n_bars)

    if reg.empty or covered.empty:
        return empty(BAR_COLS)
    reg["j"] = ((reg["start"] - reg["s_open"]) // pd.Timedelta(minutes=1)).astype(int) // step
    agg = reg.groupby(["session", "j"]).agg(**ohlc(), volume=("volume", "sum"), dvol=("dvol", "sum"),
                                            n_min=("open", "size")).reset_index()
    bars = grid.merge(agg, on=["session", "j"], how="left")
    bars = fill_flat(bars, ("volume", "dvol", "n_min"), segments(bars["sess_idx"].to_numpy()))
    if bars is None:
        return empty(BAR_COLS)
    bars = bars.reset_index(drop=True)

    # session-level: previous regular close, corporate actions, extended-hours aux (all known before the open)
    first = bars.groupby("session").head(1).index
    prev_close = bars["close"].shift(1).to_numpy()[first]
    sess_first = bars.loc[first, ["session"]]
    bars["gap_adj"] = 0.0
    bars.loc[first, "gap_adj"] = _corp_adjust(sess_first, prev_close, splits, divs)
    pre_s = pre.groupby("session").agg(pre_last=("close", "last"), pre_dv=("dvol", "sum"))
    post_s = post.groupby("session").agg(post_last=("close", "last"), post_dv=("dvol", "sum"))
    prev_of = pd.Series(cal["session"].shift(1).to_numpy(), index=cal["session"])
    prev_sess = sess_first["session"].map(prev_of)
    pre_last = sess_first["session"].map(pre_s["pre_last"]).to_numpy(np.float64)
    post_last = prev_sess.map(post_s["post_last"]).to_numpy(np.float64)
    split_now = np.exp(bars.loc[first, "gap_adj"].to_numpy())          # rescale post-market of the day before
    for col in AUX_COLS:
        bars[col] = np.nan
    bars.loc[first, "aux_post_ret"] = np.log(post_last / prev_close)
    bars.loc[first, "aux_pre_ret"] = np.log(pre_last * split_now / prev_close)
    bars.loc[first, "aux_pre_ldv"] = np.log1p(sess_first["session"].map(pre_s["pre_dv"]).fillna(0.0).to_numpy())
    bars.loc[first, "aux_post_ldv"] = np.log1p(prev_sess.map(post_s["post_dv"]).fillna(0.0).to_numpy())
    return typed(bars, BAR_COLS)


def build_daily(daily: pd.DataFrame, splits: Optional[pd.DataFrame] = None,
                divs: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Raw Alpha Vantage daily rows (date, o/h/l/c/v) -> 1d bars stamped at the session close."""
    cal = create_cal()
    daily = daily.assign(date=pd.to_datetime(daily["date"]).astype("datetime64[ns]"))
    d = daily.merge(cal, left_on="date", right_on="session", how="inner").sort_values("session")
    d = d[(d["close_x"] > 0) & (d["open_x"] > 0)].reset_index(drop=True)
    bars = pd.DataFrame({"ts": d["close_y"], "session": d["session"], "open": d["open_x"], "high": d["high"],
                         "low": d["low"], "close": d["close_x"], "volume": d["volume"].astype(np.float64)})
    bars["dvol"] = bars["close"] * bars["volume"]
    bars["n_min"] = np.nan
    sess_idx = cal.index[cal["session"].isin(d["session"])].to_numpy()
    bars["seg"] = segments(sess_idx)
    bars["gap_adj"] = _corp_adjust(bars[["session"]], bars["close"].shift(1).to_numpy(), splits, divs)
    for col in AUX_COLS:
        bars[col] = np.nan
    return typed(bars, BAR_COLS)


def bars_per_day(scale: str) -> int:
    return 1 if scale == "1d" else -(-390 // INTRADAY_MIN[scale])


SCALES: Dict[str, int] = {k: s // 60 for k, s in SCALE_SECONDS.items()}   # minutes
