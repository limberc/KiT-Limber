"""Resumable on-chain DEX downloader (Envio HyperSync, Uniswap v3 pools). One chain per process.

Usage: python -m kit2.data.download_crypto --chain eth [--phases pools,activity,candidates,swaps]

Layout under <data_root>/raw/crypto/<chain>/:
  pools.parquet                      every Uniswap v3 pool: pool, token0, token1, fee, tick_spacing, block
  activity/blk_<from>_<to>_{act,days}.parquet  per-chunk swap counts per (pool, UTC day) + day blocks
  activity_daily.parquet             merged counts + day_blocks.parquet (first block of each UTC day)
  candidates_monthly.parquet         point-in-time candidate pools per month (selection uses days < month)
  swaps/<YYYY-MM>.parquet            decoded swaps of the month's download set, ordered by (block, log_index)

Point-in-time rule: candidates for month M are ranked on swap counts from complete UTC days before M.
The download set may look ahead (a pool that becomes a candidate later is fetched early, so its history
exists); which bytes we fetch never feeds a feature or a selection.
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from kit2.data.io import atomic_write_parquet
from kit2.data.sources.hypersync_dex import CHAINS, Dex
from kit2.env import data_root

# blocks per activity chunk (~1 month of blocks at each chain's typical block time)
CHUNK_BLOCKS = {"eth": 216_000, "base": 1_296_000, "arbitrum": 10_000_000}
TRAIL_DAYS = 30  # candidate ranking window (complete days before the month)
SEASON_DAYS = 30  # a pool must have been active this long before it can be a candidate
N_TOKENS = 150  # candidate base tokens per chain per month
POOLS_PER_TOKEN = 3  # most active pools kept per base token
LOOKAHEAD_MONTHS = 9  # download history this far before a pool's first candidate month (1d scale: 250 days)


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def orient(pools: pd.DataFrame, chain: str) -> pd.DataFrame:
    """Pick base/quote per pool: quote = the preferred quote token (stable < ETH < BTC); drop pairs
    with no quote token or two stables."""
    ch = CHAINS[chain]
    r0 = pools["token0"].map(ch.quote_rank)
    r1 = pools["token1"].map(ch.quote_rank)
    keep = (np.minimum(r0, r1) < 99) & ~((r0 == 0) & (r1 == 0))
    p = pools[keep].copy()
    q_is_0 = (r0[keep] <= r1[keep]).to_numpy()
    p["quote"] = np.where(q_is_0, p["token0"], p["token1"])
    p["base"] = np.where(q_is_0, p["token1"], p["token0"])
    p["quote_is_token0"] = q_is_0
    p["quote_symbol"] = p["quote"].map(lambda a: ch.quotes[a].symbol)
    p["quote_kind"] = p["quote"].map(ch.quote_rank).map({0: "usd", 1: "eth", 2: "btc"})
    return p.reset_index(drop=True)


class CryptoDownloader:
    def __init__(self, root: Path, chain: str) -> None:
        self.chain = chain
        self.root = root / "raw" / "crypto" / chain
        self.dex = Dex(chain)

    # ---- pools ---------------------------------------------------------------------------------
    def pools(self) -> None:
        out = self.root / "pools.parquet"
        df = asyncio.run(self.dex.pools())  # always refreshed: new pools appear every day
        atomic_write_parquet(df, out)
        log(f"[pools] {len(df)} pools")

    # ---- activity ------------------------------------------------------------------------------
    def activity(self) -> None:
        ch, step = CHAINS[self.chain], CHUNK_BLOCKS[self.chain]
        height = asyncio.run(self.dex.height())
        d = self.root / "activity"
        starts = list(range(ch.factory_block, height, step))
        log(f"[activity] {len(starts)} chunks of {step} blocks up to {height}")
        for i, lo in enumerate(starts):
            hi = min(lo + step, height)
            stem = d / f"blk_{lo:012d}_{lo + step:012d}"
            act_f, day_f = stem.with_name(stem.name + "_act.parquet"), stem.with_name(stem.name + "_days.parquet")
            if act_f.is_file() and day_f.is_file() and hi == lo + step:
                continue  # the open-ended last chunk is always re-scanned
            t = time.time()
            act, days = asyncio.run(self.dex.activity(lo, hi))
            atomic_write_parquet(days, day_f)
            atomic_write_parquet(act, act_f)  # written last: its presence marks the chunk complete
            log(f"[activity] chunk {i + 1}/{len(starts)} blocks {lo}-{hi}: {len(act)} pool-days, "
                f"{int(act['n_swaps'].sum())} swaps, {time.time() - t:.1f}s")
        self._merge_activity()

    def _merge_activity(self) -> None:
        d = self.root / "activity"
        act = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("blk_*_act.parquet"))], ignore_index=True)
        act = act.groupby(["pool", "day"], as_index=False)["n_swaps"].sum()
        days = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("blk_*_days.parquet"))], ignore_index=True)
        days = days.groupby("day", as_index=False)["first_block"].min().sort_values("day")
        atomic_write_parquet(act, self.root / "activity_daily.parquet")
        atomic_write_parquet(days, self.root / "day_blocks.parquet")
        log(f"[activity] merged: {len(act)} pool-days, {act['pool'].nunique()} pools, "
            f"days {days['day'].min().date()}..{days['day'].max().date()}")

    # ---- candidates ----------------------------------------------------------------------------
    def candidates(self) -> None:
        pools = orient(pd.read_parquet(self.root / "pools.parquet"), self.chain)
        act = pd.read_parquet(self.root / "activity_daily.parquet")
        act = act.merge(pools[["pool", "base", "quote_kind"]], on="pool")
        first_day = act.groupby("pool")["day"].min()
        months = pd.date_range(act["day"].min().normalize().replace(day=1) + pd.DateOffset(months=1),
                               pd.Timestamp.now().normalize().replace(day=1), freq="MS")
        rows: List[pd.DataFrame] = []
        for m in months:
            win = act[(act["day"] >= m - pd.Timedelta(days=TRAIL_DAYS)) & (act["day"] < m)]
            seasoned = first_day[first_day <= m - pd.Timedelta(days=SEASON_DAYS)].index
            win = win[win["pool"].isin(seasoned)]
            per_pool = win.groupby(["pool", "base", "quote_kind"], as_index=False)["n_swaps"].sum()
            per_pool = per_pool.sort_values("n_swaps", ascending=False)
            top_pools = per_pool.groupby("base").head(POOLS_PER_TOKEN)
            tok = top_pools.groupby("base")["n_swaps"].sum().sort_values(ascending=False)
            keep_tok = set(tok.index[:N_TOKENS])
            # USD reference pools for ETH and BTC are always kept, ranked the same point-in-time way
            refs = per_pool[per_pool["quote_kind"] == "usd"]
            refs = refs[refs["base"].isin({a for a, t in CHAINS[self.chain].quotes.items() if not t.stable})]
            sel = pd.concat([top_pools[top_pools["base"].isin(keep_tok)], refs.groupby("base").head(2)])
            sel = sel.drop_duplicates("pool").assign(month=m)
            sel["token_rank"] = sel["base"].map({b: i + 1 for i, b in enumerate(tok.index)})
            rows.append(sel)
        cand = pd.concat(rows, ignore_index=True)
        atomic_write_parquet(cand, self.root / "candidates_monthly.parquet")
        log(f"[candidates] {cand['pool'].nunique()} pools / {cand['base'].nunique()} tokens over "
            f"{cand['month'].nunique()} months")

    # ---- swaps ---------------------------------------------------------------------------------
    def swaps(self) -> None:
        cand = pd.read_parquet(self.root / "candidates_monthly.parquet")
        days = pd.read_parquet(self.root / "day_blocks.parquet").set_index("day")["first_block"]
        height = asyncio.run(self.dex.height())
        month_first_block = days.groupby(days.index.to_period("M")).min()
        months = sorted(set(month_first_block.index.to_timestamp()))
        member: Dict[pd.Timestamp, set] = {m: set(g["pool"]) for m, g in cand.groupby("month")}
        cur = pd.Timestamp.now().normalize().replace(day=1)
        log(f"[swaps] {len(months)} months")
        for i, m in enumerate(months[::-1]):  # newest first: recent data is the most valuable
            ahead = [member.get(m + pd.DateOffset(months=k), set()) for k in range(LOOKAHEAD_MONTHS + 1)]
            pool_set = sorted(set().union(*ahead))
            if not pool_set:
                continue
            out = self.root / "swaps" / f"{m:%Y-%m}.parquet"
            if out.is_file() and not (m == cur and time.time() - out.stat().st_mtime > 6 * 3600):
                continue
            lo = int(month_first_block[m.to_period("M")])
            nxt = m + pd.DateOffset(months=1)
            hi = int(month_first_block[nxt.to_period("M")]) if nxt.to_period("M") in month_first_block.index \
                else height
            t = time.time()
            df = asyncio.run(self.dex.swaps(pool_set, lo, hi))
            atomic_write_parquet(df, out)
            log(f"[swaps] {m:%Y-%m}: {len(pool_set)} pools, {len(df)} swaps, blocks {lo}-{hi}, "
                f"{time.time() - t:.0f}s ({i + 1}/{len(months)})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Uniswap v3 swaps via HyperSync (resumable)")
    ap.add_argument("--chain", required=True, choices=sorted(CHAINS))
    ap.add_argument("--phases", default="pools,activity,candidates,swaps")
    ap.add_argument("--root", type=Path, default=data_root())
    args = ap.parse_args()
    dl = CryptoDownloader(args.root, args.chain)
    for phase in args.phases.split(","):
        log(f"=== {args.chain} phase {phase}")
        getattr(dl, phase)()
    log("done")


if __name__ == "__main__":
    main()
