"""No-look-ahead enforcement for the KiT-2 data pipeline (spec: 2026-11-17-kit2-data-pipeline-design.md).

For random cut times t, everything stamped <= t must be identical whether built from all raw data,
from raw data truncated at t, or from raw data whose future (> t) was replaced by random values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kit2.data.us.bars import ET, build_daily, build_intraday, create_cal
from kit2.data.features import candle_features
from kit2.data.us.universe import monthly_universe

SPLIT_DAY = pd.Timestamp("2024-06-10")
DIV_DAY = pd.Timestamp("2024-05-29")


def synth_minutes(seed: int = 0, start="2024-05-13", end="2024-06-21") -> pd.DataFrame:
    """Alpha Vantage-like 1-min rows (ts_start ET, extended hours) with a 10:1 split, missing minutes, a missing day."""
    rng = np.random.default_rng(seed)
    cal = create_cal()
    days = cal[(cal["session"] >= start) & (cal["session"] <= end)]
    rows, price = [], 100.0
    for d, o, c in zip(days["session"], days["open"], days["close"]):
        if d == pd.Timestamp("2024-05-22"):
            continue                                            # a day with no data at all
        if d == SPLIT_DAY:
            price /= 10.0
        o_et = pd.Timestamp(o).tz_localize("UTC").tz_convert(ET).tz_localize(None)
        c_et = pd.Timestamp(c).tz_localize("UTC").tz_convert(ET).tz_localize(None)
        for ts in pd.date_range(d + pd.Timedelta(hours=4), d + pd.Timedelta(hours=20), freq="1min", inclusive="left"):
            regular = o_et <= ts < c_et
            if rng.random() < (0.15 if regular else 0.8):
                continue                                        # illiquid minutes
            r = rng.normal(0, 0.001)
            op, cl = price, price * np.exp(r)
            hi, lo = max(op, cl) * np.exp(abs(rng.normal(0, 3e-4))), min(op, cl) * np.exp(-abs(rng.normal(0, 3e-4)))
            rows.append((ts, op, hi, lo, cl, float(rng.integers(100, 10_000))))
            price = cl
    return pd.DataFrame(rows, columns=["ts_start", "open", "high", "low", "close", "volume"])


MONTHS = list(pd.period_range("2024-05", "2024-06", freq="M"))   # download plan: known before each month
SPLITS = pd.DataFrame({"effective_date": [SPLIT_DAY], "split_factor": [10.0]})
DIVS = pd.DataFrame({"ex_dividend_date": [DIV_DAY], "amount": [0.4]})


def _end_utc(raw: pd.DataFrame) -> pd.Series:
    start = pd.DatetimeIndex(raw["ts_start"]).tz_localize(ET).tz_convert("UTC").tz_localize(None)
    return pd.Series(start + pd.Timedelta(minutes=1), index=raw.index)


def _cuts(raw: pd.DataFrame, n: int, seed: int):
    end = _end_utc(raw)
    rng = np.random.default_rng(seed)
    lo, hi = end.min().value, end.max().value
    return [pd.Timestamp(int(x)) for x in rng.integers(lo, hi, size=n)]


def _upto(df: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    return df[df["ts"] <= t].reset_index(drop=True)


def _features(bars: pd.DataFrame, scale: str) -> pd.DataFrame:
    return candle_features(bars, half_life_bars=5.0 * (390 if scale == "1m" else 7)).assign(ts=bars["ts"].to_numpy())


@pytest.fixture(scope="module")
def raw():
    return synth_minutes()


@pytest.mark.parametrize("scale", ["1m", "5m", "30m", "1h", "2h"])
def test_intraday_prefix_invariance(raw, scale):
    full = build_intraday(raw, scale, MONTHS, SPLITS, DIVS)
    f_full = _features(full, scale)
    end = _end_utc(raw)
    assert (full["gap_adj"] != 0).sum() == 2                    # split + dividend, on their sessions only
    for t in _cuts(raw, 12, seed=hash(scale) % 1000):
        trunc = build_intraday(raw[end <= t], scale, MONTHS, SPLITS, DIVS)
        pd.testing.assert_frame_equal(_upto(full, t), _upto(trunc, t), check_exact=True)
        pd.testing.assert_frame_equal(_upto(f_full, t), _upto(_features(trunc, scale), t), check_exact=True)


@pytest.mark.parametrize("scale", ["1m", "15m", "2h"])
def test_intraday_future_perturbation(raw, scale):
    full = build_intraday(raw, scale, MONTHS, SPLITS, DIVS)
    end = _end_utc(raw)
    rng = np.random.default_rng(1)
    for t in _cuts(raw, 6, seed=7):
        fake = raw.copy()
        fut = (end > t).to_numpy()
        noise = np.exp(rng.normal(0, 0.05, size=fut.sum()))
        for k in ("open", "high", "low", "close"):
            fake.loc[fut, k] = fake.loc[fut, k] * noise
        fake.loc[fut, "volume"] = rng.integers(1, 10**6, size=fut.sum()).astype(float)
        pert = build_intraday(fake, scale, MONTHS, SPLITS, DIVS)
        pd.testing.assert_frame_equal(_upto(full, t), _upto(pert, t), check_exact=True)
        pd.testing.assert_frame_equal(_upto(_features(full, scale), t), _upto(_features(pert, scale), t),
                                      check_exact=True)


def test_bar_stamped_at_close_uses_only_earlier_minutes(raw):
    """Every regular minute lands in the bar whose close is >= the minute's end, never an earlier bar."""
    bars = build_intraday(raw, "30m", MONTHS, SPLITS, DIVS)
    end = _end_utc(raw).sort_values().to_numpy()
    ts = bars["ts"].to_numpy()
    # the bar at ts covers (ts - 30min, ts]; count minutes ending in that window must equal n_min
    lo = np.searchsorted(end, ts - np.timedelta64(30, "m"), side="right")
    hi = np.searchsorted(end, ts, side="right")
    assert np.all(hi - lo >= bars["n_min"].to_numpy())


def test_daily_prefix_invariance():
    cal = create_cal()
    days = cal[(cal["session"] >= "2023-01-01") & (cal["session"] <= "2024-12-31")]["session"]
    rng = np.random.default_rng(3)
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, len(days))))
    o = c * np.exp(rng.normal(0, 0.005, len(days)))
    daily = pd.DataFrame({"date": days.to_numpy(), "open": o, "high": np.maximum(o, c) * 1.01,
                          "low": np.minimum(o, c) * 0.99, "close": c, "volume": rng.integers(1e5, 1e6, len(days))})
    full = build_daily(daily, SPLITS, DIVS)
    for t in [pd.Timestamp(x) for x in rng.integers(full["ts"].min().value, full["ts"].max().value, 15)]:
        avail = daily["date"].map(cal.set_index("session")["close"])     # a daily bar is known at the close
        trunc = build_daily(daily[(avail <= t).to_numpy()], SPLITS, DIVS)
        pd.testing.assert_frame_equal(_upto(full, t), _upto(trunc, t), check_exact=True)
        f1, f2 = (candle_features(b, 5.0).assign(ts=b["ts"].to_numpy()) for b in (full, trunc))
        pd.testing.assert_frame_equal(_upto(f1, t), _upto(f2, t), check_exact=True)


def test_universe_uses_only_data_before_month():
    rng = np.random.default_rng(5)
    days = pd.bdate_range("2022-01-03", "2023-12-29")
    daily = {f"S{i}": pd.DataFrame({"date": days, "close": 10 + rng.random(len(days)) * 50,
                                     "volume": rng.integers(1e4, 1e7, len(days)).astype(float)})
             for i in range(30)}
    windows = {s: [(pd.NaT, pd.NaT)] for s in daily}
    months = pd.date_range("2022-06-01", "2023-12-01", freq="MS")
    full = monthly_universe(daily, windows, months, top_n=10)
    for m in months[::3]:
        trunc = {s: d[d["date"] < m] for s, d in daily.items()}
        got = monthly_universe(trunc, windows, [m], top_n=10)
        want = full[full["month"] == m].reset_index(drop=True)
        pd.testing.assert_frame_equal(want, got.reset_index(drop=True), check_exact=True)


# ---- crypto ------------------------------------------------------------------------------------------
from kit2.data.crypto.bars import build_token_bars, reference_series  # noqa: E402

CRYPTO_MONTHS = list(pd.period_range("2025-01", "2025-02", freq="M"))


def synth_swaps(seed: int = 0):
    """Token X with a USDC pool (X=token0) and a WETH pool (WETH=token0), plus a USDC/WETH reference pool.
    Blocks every 12 s, a few swaps per block sharing the block timestamp."""
    rng = np.random.default_rng(seed)
    t0 = int(pd.Timestamp("2025-01-01").timestamp())
    n_blocks = 59 * 24 * 300                                   # ~59 days of 12 s blocks
    blocks = np.sort(rng.choice(n_blocks, size=n_blocks // 6, replace=False))
    rows, eth_lp, x_lp = [], np.log(3000.0), np.log(0.5)
    for b in blocks:
        ts = t0 + 12 * int(b)
        eth_lp += rng.normal(0, 2e-4)
        x_lp += rng.normal(0, 6e-4)
        for k in range(int(rng.integers(1, 4))):
            pool = rng.choice(["usdc_x", "weth_x", "ref"], p=[0.4, 0.3, 0.3])
            amt = float(rng.lognormal(10, 1)) * rng.choice([-1, 1])
            jitter = rng.normal(0, 1e-4)                         # intra-block noise (sandwich-like)
            if pool == "usdc_x":    # token0 = X (18 dec), token1 = USDC (6 dec): ln(USDC raw / X raw)
                lp = x_lp + jitter + np.log(1e6) - np.log(1e18)
                a0, a1 = -amt * 1e12, amt
            elif pool == "weth_x":  # token0 = WETH (18), token1 = X (18): ln(X raw / WETH raw)
                lp = eth_lp - x_lp + jitter
                a0, a1 = amt * 1e12, -amt * 1e12
            else:                   # token0 = USDC (6), token1 = WETH (18): ln(WETH raw / USDC raw)
                lp = -(eth_lp + jitter) + np.log(1e18) - np.log(1e6)
                a0, a1 = amt, -amt * 1e12
            rows.append((int(b), k, ts, pool, a0, a1, lp, 1e18, 0, f"0x{rng.integers(0, 50):040x}", 1.5))
    sw = pd.DataFrame(rows, columns=["block", "log_index", "ts", "pool", "amount0", "amount1", "log_price",
                                     "liquidity", "tick", "tx_from", "base_fee_gwei"])
    pools = pd.DataFrame({"pool": ["usdc_x", "weth_x"], "quote_is_token0": [False, True],
                          "quote_kind": ["usd", "eth"], "quote_dec": [6, 18]})
    return sw, pools


@pytest.fixture(scope="module")
def crypto():
    sw, pools = synth_swaps()
    return sw[sw["pool"] != "ref"], sw[sw["pool"] == "ref"], pools


def _crypto_bars(tok, ref, pools, scale):
    refs = {"eth": reference_series(ref, quote_is_token0=True, base_dec=18, quote_dec=6)}
    primary = {CRYPTO_MONTHS[0]: "usdc_x", CRYPTO_MONTHS[1]: "weth_x"}      # switches quote type mid-sample
    return build_token_bars(tok, pools, primary, scale, CRYPTO_MONTHS, refs)


@pytest.mark.parametrize("scale", ["1m", "15m", "1h", "1d"])
def test_crypto_prefix_invariance(crypto, scale):
    tok, ref, pools = crypto
    full = _crypto_bars(tok, ref, pools, scale)
    assert full["close"].notna().all() and (full["close"] > 0).all()
    rng = np.random.default_rng(11)
    for t in rng.integers(tok["ts"].min(), tok["ts"].max(), size=8):
        tt = pd.Timestamp(int(t), unit="s")
        trunc = _crypto_bars(tok[tok["ts"] <= t], ref[ref["ts"] <= t], pools, scale)
        pd.testing.assert_frame_equal(_upto(full, tt), _upto(trunc, tt), check_exact=True)


def test_crypto_future_perturbation(crypto):
    tok, ref, pools = crypto
    full = _crypto_bars(tok, ref, pools, "5m")
    rng = np.random.default_rng(12)
    for t in rng.integers(tok["ts"].min(), tok["ts"].max(), size=5):
        f_tok, f_ref = tok.copy(), ref.copy()
        for f in (f_tok, f_ref):
            fut = (f["ts"] > t).to_numpy()
            f.loc[fut, "log_price"] += rng.normal(0, 0.1, fut.sum())
            f.loc[fut, ["amount0", "amount1"]] *= rng.lognormal(0, 1, (fut.sum(), 1))
        pert = _crypto_bars(f_tok, f_ref, pools, "5m")
        tt = pd.Timestamp(int(t), unit="s")
        pd.testing.assert_frame_equal(_upto(full, tt), _upto(pert, tt), check_exact=True)


def test_crypto_usd_price_is_consistent_across_quote_pools(crypto):
    """Switching the primary pool from USDC- to WETH-quoted must not create a level jump (decimals/ref right)."""
    tok, ref, pools = crypto
    bars = _crypto_bars(tok, ref, pools, "1h")
    switch = bars["ts"] >= pd.Timestamp("2025-02-01")
    i = int(np.argmax(switch.to_numpy()))
    gap = np.log(bars["open"].iloc[i] / bars["close"].iloc[i - 1])
    assert abs(gap) < 0.01, gap
    assert 0.3 < bars["close"].iloc[i] / 1.0 < 1.0                # X starts near 0.5 USD


# ---- shared bar helpers ------------------------------------------------------------------------------
from kit2.data.bars_core import fill_flat, segments  # noqa: E402


def test_fill_flat_never_carries_a_price_across_a_gap():
    bars = pd.DataFrame({"open": [1.0, np.nan, 2.0, np.nan, np.nan, 3.0],
                         "high": [1.0, np.nan, 2.0, np.nan, np.nan, 3.0],
                         "low": [1.0, np.nan, 2.0, np.nan, np.nan, 3.0],
                         "close": [1.0, np.nan, 2.0, np.nan, np.nan, 3.0],
                         "dvol": [5.0, np.nan, 5.0, np.nan, np.nan, 5.0]})
    seg = segments(np.array([10, 10, 11, 13, 13, 14]))              # period 12 skipped -> new segment at row 3
    out = fill_flat(bars, ["dvol"], seg)
    assert list(out.index) == [0, 1, 2, 5]                          # rows 3-4: unpriced start of a new segment
    assert out["close"].tolist() == [1.0, 1.0, 2.0, 3.0] and out.loc[1, "dvol"] == 0.0
    assert out["seg"].tolist() == [0, 0, 0, 1]


def test_crypto_coverage_gap_starts_a_new_segment(crypto):
    tok, ref, pools = crypto
    jan_mar = [pd.Period("2025-01", "M"), pd.Period("2025-03", "M")]   # February not covered; no March swaps
    bars = build_token_bars(tok, pools, {jan_mar[0]: "usdc_x"}, "1h", jan_mar,
                            {"eth": reference_series(ref, quote_is_token0=True, base_dec=18, quote_dec=6)})
    assert bars["ts"].max() <= pd.Timestamp("2025-02-01")           # no stale flat bars invented for March
