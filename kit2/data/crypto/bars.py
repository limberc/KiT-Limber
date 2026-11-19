"""Crypto bars: decoded Uniswap v3 swaps -> point-in-time USD bars per (chain, base token), plus aux signals.

Point-in-time contract (tested in kit2/tests/test_no_lookahead.py):
- A swap is known at its block timestamp. A bar closing at T holds swaps with T - step < ts <= T.
- Prices are block-end mid prices (last swap of a block per pool), which drops intra-block sandwich spikes.
  All swaps of a block share one timestamp, so a time cut never splits a block.
- ETH/BTC-quoted pools are converted to USD with the backward as-of reference price (same block allowed:
  both are known at that block's time). Stable quotes count as 1 USD.
- The primary pool of a token for month M is the token's most active downloaded pool in the activity sample
  for M (days before M). Coverage months are an explicit input, as for US bars.
Price columns are USD per 1e18 raw units of the token (= per token for 18-decimal tokens). Base-token
decimals are a constant factor per token, so every candle feature (a log ratio) is unaffected.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from kit2.data.bars_core import CORE_COLS, SCALE_SECONDS, empty, fill_flat, ohlc, segments, typed

LN10 = float(np.log(10.0))
AUX_COLS = ("aux_flow_imb", "aux_ln_traders", "aux_ln_gas", "aux_ln_liq")
BAR_COLS = [*CORE_COLS, *AUX_COLS]


def block_end(sw: pd.DataFrame) -> pd.DataFrame:
    """Last swap of each (pool, block): block, ts, pool, log_price, liquidity."""
    last = sw.sort_values(["block", "log_index"]).groupby(["pool", "block"], sort=False).tail(1)
    return last[["pool", "block", "ts", "log_price", "liquidity"]].sort_values(["block", "pool"])


def base_quote_lp(log_price: np.ndarray, quote_is_token0: np.ndarray) -> np.ndarray:
    """ln(quote raw per base raw) from the pool's ln(token1 raw / token0 raw)."""
    return np.where(quote_is_token0, -log_price, log_price)


def reference_series(ref_swaps: pd.DataFrame, quote_is_token0: bool, base_dec: int, quote_dec: int) -> pd.DataFrame:
    """USD price of a quote asset (ETH/BTC) per human unit, at each block-end of its USD reference pool."""
    be = block_end(ref_swaps)
    lp = base_quote_lp(be["log_price"].to_numpy(), np.full(len(be), quote_is_token0))
    return pd.DataFrame({"block": be["block"].to_numpy(np.int64),
                         "lp_ref": lp + LN10 * (base_dec - quote_dec)})


def _asof_ref(df: pd.DataFrame, ref: Optional[pd.DataFrame]) -> np.ndarray:
    """Backward as-of reference log price for rows of df (by block, same block allowed); 0 if no reference."""
    if ref is None:
        return np.zeros(len(df))
    m = pd.merge_asof(df[["block"]].reset_index().sort_values("block"), ref.sort_values("block"),
                      on="block", direction="backward", allow_exact_matches=True)
    return m.set_index("index").reindex(df.index)["lp_ref"].to_numpy()


def build_token_bars(swaps: pd.DataFrame, pools: pd.DataFrame, primary: Dict[pd.Period, str], scale: str,
                     months: Iterable[pd.Period], refs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """swaps: decoded swaps of the token's pools (any order); pools: oriented rows (pool, quote_is_token0,
    quote_kind, quote_dec); primary: month -> primary pool; refs: quote_kind ('eth'/'btc') -> reference series."""
    step = SCALE_SECONDS[scale]
    months = sorted(set(pd.PeriodIndex(list(months), freq="M")))
    if swaps.empty or not months:
        return empty(BAR_COLS)
    sw = swaps.merge(pools[["pool", "quote_is_token0", "quote_kind", "quote_dec"]], on="pool", how="inner")
    sw = sw.sort_values(["block", "log_index"]).reset_index(drop=True)
    sw["month"] = pd.to_datetime(sw["ts"], unit="s").dt.to_period("M")
    sw = sw[sw["month"].isin(months)]
    if sw.empty:
        return empty(BAR_COLS)

    # USD conversion: ln(USD per quote human unit) as of each swap's block
    lp_ref = np.zeros(len(sw))
    for kind, ref in refs.items():
        sel = (sw["quote_kind"] == kind).to_numpy()
        if sel.any():
            lp_ref[sel] = _asof_ref(sw[sel], ref)
    sw["lp_ref"] = lp_ref
    q_amt = np.where(sw["quote_is_token0"], sw["amount0"], sw["amount1"])
    sw["usd"] = np.abs(q_amt) / 10.0 ** sw["quote_dec"].to_numpy() * np.exp(sw["lp_ref"])
    sw["buy_usd"] = np.where(q_amt > 0, sw["usd"], 0.0)  # quote paid into the pool = base bought
    sw = sw[np.isfinite(sw["usd"])]
    sw["bar"] = ((sw["ts"].to_numpy(np.int64) - 1) // step + 1) * step

    # price events: block-end mid of the month's primary pool, in USD per 1e18 raw base units
    is_primary = sw["pool"].to_numpy() == sw["month"].map({m: p for m, p in primary.items()}).to_numpy()
    prim = sw[is_primary].groupby("block", sort=True).tail(1)
    lp = (base_quote_lp(prim["log_price"].to_numpy(), prim["quote_is_token0"].to_numpy())
          - LN10 * prim["quote_dec"].to_numpy() + prim["lp_ref"].to_numpy() + 18 * LN10)
    ev = pd.DataFrame({"bar": prim["bar"].to_numpy(), "px": np.exp(lp), "liq": prim["liquidity"].to_numpy()})
    px = ev.groupby("bar").agg(**ohlc("px", "px", "px", "px"), liq=("liq", "last"))
    flow = sw.groupby("bar").agg(dvol=("usd", "sum"), buy=("buy_usd", "sum"), n_min=("usd", "size"),
                                 traders=("tx_from", "nunique"), gas=("base_fee_gwei", "median"))

    # full UTC grid over covered months, from the first priced bar
    lo = pd.Timestamp(months[0].start_time).value // 10 ** 9
    hi = pd.Timestamp((months[-1] + 1).start_time).value // 10 ** 9
    grid = np.arange(lo + step, hi + 1, step, dtype=np.int64)
    gm = pd.to_datetime(grid - 1, unit="s").to_period("M")
    grid = grid[gm.isin(months)]
    month_ord = pd.to_datetime(grid - 1, unit="s").to_period("M").asi8  # calendar month number per bar
    bars = pd.DataFrame(index=pd.Index(grid, name="bar")).join(px).join(flow)
    bars = fill_flat(bars, ("dvol", "buy", "n_min"), segments(month_ord))
    if bars is None:
        return empty(BAR_COLS)
    bars["volume"] = bars["dvol"]
    bars["gap_adj"] = 0.0
    bars["aux_flow_imb"] = np.where(bars["dvol"] > 0, (2 * bars["buy"] - bars["dvol"]) / bars["dvol"], 0.0)
    bars["aux_ln_traders"] = np.log1p(bars["traders"].fillna(0.0))
    bars["aux_ln_gas"] = np.log(bars["gas"]).where(bars["gas"] > 0)
    bars["aux_ln_liq"] = np.log(bars["liq"].ffill()).where(bars["liq"].ffill() > 0)
    bars["ts"] = pd.to_datetime(bars.index.to_numpy(), unit="s")
    return typed(bars, BAR_COLS)


def primary_pools(activity: Dict[pd.Period, pd.DataFrame], token_pools: Iterable[str],
                  downloaded: Dict[pd.Period, set]) -> Dict[pd.Period, str]:
    """Point-in-time primary pool per month: most active downloaded pool of the token in that month's sample."""
    token_pools = set(token_pools)
    out = {}
    for m, act in activity.items():
        a = act[act["pool"].isin(token_pools & downloaded.get(m, set()))]
        if len(a):
            out[m] = a.sort_values(["n_swaps", "pool"], ascending=[False, True])["pool"].iloc[0]
    return out
