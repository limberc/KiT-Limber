"""Resumable on-chain DEX downloader (Envio HyperSync, Uniswap v3 pools). One chain per process.

Usage: python -m kit2.data.crypto.download --chain eth [--phases pools,anchors,activity,candidates,swaps]

Layout under <data_root>/raw/crypto/<chain>/:
  pools.parquet                every Uniswap v3 pool: pool, token0, token1, fee, tick_spacing, block (created)
  month_blocks.parquet         first block at/after each month start (UTC), found by timestamp search
  activity/<YYYY-MM>.parquet   swap counts per pool over one-day sample windows before that month
  candidates_monthly.parquet   point-in-time candidate pools per month
  swaps/<YYYY-MM>.parquet      decoded swaps of the month's download set, ordered by (block, log_index)

Point-in-time rule: candidates for month M are ranked on activity sampled from days before M and on pool
creation blocks before M. The download set may look ahead (a pool that becomes a candidate later is fetched
early so its history exists); which bytes we fetch never feeds a feature or a selection.

Budget: the HyperSync token allows ~100 requests/min and a response holds ~6.5k logs (~11k logs/s), so the
ranking samples SAMPLE_DAYS one-day windows per month instead of scanning every day.
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
from kit2.env import data_root
from .hypersync_dex import Dex, CHAINS

START = {"eth": "2023-01-01", "base": "2023-08-01", "arbitrum": "2023-01-01"}
SAMPLE_DAYS = (1, 15)  # one-day windows starting this many days before the month start
SEASON_DAYS = 30  # a pool must have been created this long before the month
N_TOKENS = 100  # candidate base tokens per chain per month
POOLS_PER_TOKEN = 2  # most active pools kept per base token
LOOKAHEAD_MONTHS = 9  # history fetched before a pool's first candidate month (1d scale: 250 days)


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
        self.months = pd.date_range(START[chain], pd.Timestamp.now().normalize(), freq="MS")

    # ---- pools ---------------------------------------------------------------------------------
    def pools(self) -> None:
        """Pool registry; each run appends pools created after the newest one already saved."""
        out = self.root / "pools.parquet"
        have = pd.read_parquet(out) if out.is_file() else None
        start = int(have["block"].max()) if have is not None and len(have) else None   # re-scan that block
        new = asyncio.run(self.dex.pools(from_block=start))
        df = new if have is None else pd.concat([have, new], ignore_index=True).drop_duplicates("pool")
        atomic_write_parquet(df.sort_values(["block", "pool"]).reset_index(drop=True), out)
        log(f"[pools] {len(df)} pools ({len(df) - (0 if have is None else len(have))} new)")

    # ---- month anchors -------------------------------------------------------------------------
    def anchors(self) -> None:
        """Exact first block of every month start (one month before START too, for interpolation)."""
        out = self.root / "month_blocks.parquet"
        have = pd.read_parquet(out) if out.is_file() else pd.DataFrame({"ts": [], "block": []})
        done = set(pd.to_datetime(have["ts"]))
        want = [m for m in [self.months[0] - pd.DateOffset(months=1), *self.months] if m not in done]

        async def run():
            height = await self.dex.height()
            known = sorted(zip(pd.to_datetime(have["ts"]), have["block"].astype(int)))
            rows = []
            for t in want:
                lo = max([b for tt, b in known if tt <= t], default=CHAINS[self.chain].factory_block)
                hi = min([b for tt, b in known if tt >= t], default=height)
                b = await self.dex.block_at(int(t.timestamp()), lo, hi)
                rows.append((t, b))
                known = sorted(known + [(t, b)])
            return rows

        t0 = time.time()
        rows = asyncio.run(run())
        allb = pd.concat([have, pd.DataFrame(rows, columns=["ts", "block"])], ignore_index=True)
        allb["ts"] = pd.to_datetime(allb["ts"])
        allb["block"] = allb["block"].astype(np.int64)
        atomic_write_parquet(allb.sort_values("ts").reset_index(drop=True), out)
        log(f"[anchors] {len(rows)} month anchors in {time.time() - t0:.0f}s")

    def _blocks(self) -> pd.Series:
        b = pd.read_parquet(self.root / "month_blocks.parquet")
        return pd.Series(b["block"].to_numpy(np.int64), index=pd.to_datetime(b["ts"])).sort_index()

    def _approx_block(self, blk: pd.Series, t: pd.Timestamp) -> int:
        """Block at time t, linearly interpolated between exact month anchors (for sampling windows)."""
        x = blk.index.as_unit("ns").asi8.astype(np.float64)  # pandas 3 may store datetime64[us]
        return int(np.interp(float(pd.Timestamp(t).as_unit("ns").value), x, blk.to_numpy(np.float64)))

    # ---- activity ------------------------------------------------------------------------------
    def activity(self) -> None:
        blk = self._blocks()
        for m in self.months[::-1]:
            out = self.root / "activity" / f"{m:%Y-%m}.parquet"
            if out.is_file():
                continue
            t, parts = time.time(), []
            for d in SAMPLE_DAYS:
                lo = self._approx_block(blk, m - pd.Timedelta(days=d))
                hi = self._approx_block(blk, m - pd.Timedelta(days=d - 1))
                act, _ = asyncio.run(self.dex.activity(lo, hi))
                parts.append(act)
            act = pd.concat(parts).groupby("pool", as_index=False)["n_swaps"].sum()   # may be empty: then no
            # candidates that month, and the empty file stops it being re-scanned
            atomic_write_parquet(act.assign(month=m, sample_days=len(SAMPLE_DAYS)), out)
            log(f"[activity] {m:%Y-%m}: {len(act)} pools, {int(act['n_swaps'].sum())} sampled swaps, "
                f"{time.time() - t:.0f}s")

    # ---- candidates ----------------------------------------------------------------------------
    def candidates(self) -> None:
        pools = orient(pd.read_parquet(self.root / "pools.parquet"), self.chain)
        blk = self._blocks()
        nonstable_quotes = {a for a, t in CHAINS[self.chain].quotes.items() if not t.stable}
        rows: List[pd.DataFrame] = []
        for m in self.months:
            f = self.root / "activity" / f"{m:%Y-%m}.parquet"
            if not f.is_file():
                continue
            act = pd.read_parquet(f).merge(pools[["pool", "base", "quote_kind", "block"]], on="pool")
            if act.empty:
                continue
            season_block = self._approx_block(blk, m - pd.Timedelta(days=SEASON_DAYS))
            act = act[act["block"] <= season_block].sort_values("n_swaps", ascending=False)
            top_pools = act.groupby("base").head(POOLS_PER_TOKEN)
            tok = top_pools.groupby("base")["n_swaps"].sum().sort_values(ascending=False)
            keep_tok = set(tok.index[:N_TOKENS])
            # USD reference pools for ETH and BTC are always kept, ranked the same point-in-time way
            refs = act[(act["quote_kind"] == "usd") & act["base"].isin(nonstable_quotes)].groupby("base").head(2)
            sel = pd.concat([top_pools[top_pools["base"].isin(keep_tok)], refs]).drop_duplicates("pool")
            sel = sel.assign(month=m, token_rank=sel["base"].map({b: i + 1 for i, b in enumerate(tok.index)}))
            rows.append(sel.drop(columns=["block"]))
        cand = pd.concat(rows, ignore_index=True)
        atomic_write_parquet(cand, self.root / "candidates_monthly.parquet")
        log(f"[candidates] {cand['pool'].nunique()} pools / {cand['base'].nunique()} tokens over "
            f"{cand['month'].nunique()} months")

    # ---- swaps ---------------------------------------------------------------------------------
    def swaps(self) -> None:
        cand = pd.read_parquet(self.root / "candidates_monthly.parquet")
        blk = self._blocks()
        height = asyncio.run(self.dex.height())
        member: Dict[pd.Timestamp, set] = {m: set(g["pool"]) for m, g in cand.groupby("month")}
        cur = pd.Timestamp.now().normalize().replace(day=1)
        for i, m in enumerate(self.months[::-1]):  # newest first: recent data is the most valuable
            ahead = [member.get(m + pd.DateOffset(months=k), set()) for k in range(LOOKAHEAD_MONTHS + 1)]
            pool_set = sorted(set().union(*ahead))
            if not pool_set:
                continue
            out = self.root / "swaps" / f"{m:%Y-%m}.parquet"
            if out.is_file() and not (m == cur and time.time() - out.stat().st_mtime > 6 * 3600):
                continue
            nxt = m + pd.DateOffset(months=1)
            lo, hi = int(blk[m]), (int(blk[nxt]) if nxt in blk.index else height)
            t = time.time()
            df = asyncio.run(self.dex.swaps(pool_set, lo, hi))
            atomic_write_parquet(df, out)
            log(f"[swaps] {m:%Y-%m}: {len(pool_set)} pools, {len(df)} swaps, blocks {lo}-{hi}, "
                f"{time.time() - t:.0f}s ({i + 1}/{len(self.months)})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Uniswap v3 swaps via HyperSync (resumable)")
    ap.add_argument("--chain", required=True, choices=sorted(CHAINS))
    ap.add_argument("--phases", default="pools,anchors,activity,candidates,swaps")
    ap.add_argument("--root", type=Path, default=data_root())
    args = ap.parse_args()
    dl = CryptoDownloader(args.root, args.chain)
    for phase in args.phases.split(","):
        log(f"=== {args.chain} phase {phase}")
        getattr(dl, phase)()
    log("done")


if __name__ == "__main__":
    main()
