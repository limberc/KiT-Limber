"""Build the crypto training store from downloaded Uniswap v3 swaps (see kit2/data/store.py for the format).

Usage: python -m kit2.data.crypto.prepare [--chains eth,base,arbitrum] [--workers 10]

Instrument = (chain, base token), priced in USD (stable quotes, or ETH / BTC quotes through the chain's own
on-chain USD reference). Point-in-time choices:
- sampling set of month M = the month's candidate tokens (ranked on activity sampled before M);
- primary pool of a token for month M = its most active downloaded pool in that pre-month activity sample;
- reference pool for ETH / BTC in month M = their most active downloaded USD pool in the same sample;
- coverage = months whose swaps were downloaded for one of the token's pools (the download plan).
Steps: split each month's swaps by token into tmp files; build bars per token (parallel); shared finalize.
"""
from __future__ import annotations

import argparse
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from kit2.data.crypto.bars import AUX_COLS, build_token_bars, primary_pools, reference_series
from kit2.data.crypto.download import LOOKAHEAD_MONTHS, orient
from kit2.data.crypto.hypersync_dex import CHAINS
from kit2.data.features import candle_features
from kit2.data.store import calendar_features, finalize_store
from kit2.env import data_root

SCALES = ("1m", "5m", "15m", "30m", "1h", "2h", "1d")
STEP_S = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200, "1d": 86400}
CHAIN_ID = {"eth": 1, "base": 2, "arbitrum": 3}
VOCAB_OFFSET = 4000                 # crypto instrument ids follow the US ones in the shared embedding table
HALF_LIFE_DAYS = 5.0
KEEP = ["block", "log_index", "ts", "pool", "amount0", "amount1", "log_price", "liquidity", "tx_from", "base_fee_gwei"]


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def crypto_grid(scale: str, start: pd.Timestamp, end: pd.Timestamp):
    """Regular UTC grid of bar close times (ns) and calendar features; a UTC day is the 'session'."""
    step = STEP_S[scale]
    t0, t1 = start.value // 10**9, end.value // 10**9
    close = np.arange(t0 + step, t1 + 1, step, dtype=np.int64)
    per_day = 86400 // step
    pos = ((close - 1) % 86400) // step
    minute = ((close % 86400) // 60).astype(float)
    ts = close * 10**9
    return ts, calendar_features(ts, minute, pos.astype(float), np.full(len(ts), float(per_day)))


# ---- per chain tables ----------------------------------------------------------------------------------
class ChainTables:
    def __init__(self, raw: Path, chain: str) -> None:
        self.chain, self.raw = chain, raw
        q = CHAINS[chain].quotes
        p = orient(pd.read_parquet(raw / "pools.parquet"), chain)
        p["quote_dec"] = p["quote"].map(lambda a: q[a].decimals)
        self.pools = p.set_index("pool")
        cand = pd.read_parquet(raw / "candidates_monthly.parquet")
        cand["m"] = pd.to_datetime(cand["month"]).dt.to_period("M")
        self.cand = cand
        self.swap_months = sorted(pd.Period(f.stem, "M") for f in (raw / "swaps").glob("*.parquet"))
        member = {m: set(g["pool"]) for m, g in cand.groupby("m")}
        self.downloaded = {m: set().union(*[member.get(m + k, set()) for k in range(LOOKAHEAD_MONTHS + 1)])
                           for m in self.swap_months}
        self.activity = {m: pd.read_parquet(raw / "activity" / f"{m}.parquet")[["pool", "n_swaps"]]
                         for m in self.swap_months if (raw / "activity" / f"{m}.parquet").is_file()}
        self.nonstable = {a: t for a, t in q.items() if not t.stable}

    def tokens(self) -> List[str]:
        return sorted(set(self.cand["base"]))

    def token_pools(self, base: str) -> List[str]:
        return list(self.pools.index[self.pools["base"] == base])


def split_chain(raw: Path, tmp: Path, chain: str) -> int:
    """Each month's swaps -> tmp/<chain>/<base>/<YYYY-MM>.parquet (compact columns, hashed senders)."""
    ct = ChainTables(raw, chain)
    base_of = ct.pools["base"]
    n = 0
    for m in ct.swap_months:
        df = pd.read_parquet(raw / "swaps" / f"{m}.parquet", columns=KEEP)
        df["base"] = df["pool"].map(base_of)
        df = df[df["base"].notna()]
        df["tx_from"] = pd.util.hash_array(df["tx_from"].fillna("").to_numpy(object)).view(np.int64)
        df["log_index"] = df["log_index"].astype(np.int32)
        df["base_fee_gwei"] = df["base_fee_gwei"].astype(np.float32)
        for b, g in df.groupby("base", sort=False):
            d = tmp / chain / b
            d.mkdir(parents=True, exist_ok=True)
            g.drop(columns="base").to_parquet(d / f"{m}.parquet", index=False)
        n += len(df)
        log(f"[split] {chain} {m}: {len(df)} swaps")
    return n


def build_refs(raw: Path, tmp: Path, chain: str) -> None:
    """USD reference per quote kind ('eth', 'btc'): block-end prices of the month's primary USD pool."""
    ct = ChainTables(raw, chain)
    out = {"eth": [], "btc": []}
    for addr, tok in ct.nonstable.items():
        kind = "eth" if "ETH" in tok.symbol else "btc"
        usd = ct.pools[(ct.pools["base"] == addr) & (ct.pools["quote_kind"] == "usd")]
        for m in ct.swap_months:
            f = tmp / chain / addr / f"{m}.parquet"
            prim = primary_pools({m: ct.activity[m]}, usd.index, ct.downloaded).get(m) if m in ct.activity else None
            if prim is None or not f.is_file():
                continue
            sw = pd.read_parquet(f)
            sw = sw[sw["pool"] == prim]
            if sw.empty:
                continue
            row = usd.loc[prim]
            out[kind].append(reference_series(sw, bool(row["quote_is_token0"]), tok.decimals, int(row["quote_dec"])))
    for kind, parts in out.items():
        if parts:
            r = pd.concat(parts).sort_values("block").drop_duplicates("block", keep="last")
            r.to_parquet(tmp / chain / f"_ref_{kind}.parquet", index=False)
            log(f"[refs] {chain} {kind}: {len(r)} blocks")


@lru_cache(maxsize=3)
def _chain(raw_str: str, tmp_str: str, chain: str):
    raw, tmp = Path(raw_str), Path(tmp_str)
    ct = ChainTables(raw, chain)
    refs = {k: pd.read_parquet(tmp / chain / f"_ref_{k}.parquet") for k in ("eth", "btc")
            if (tmp / chain / f"_ref_{k}.parquet").is_file()}
    return ct, refs


def channels(bars: pd.DataFrame, scale: str) -> np.ndarray:
    f = candle_features(bars, HALF_LIFE_DAYS * 86400 / STEP_S[scale])
    has = (bars["n_min"] > 0).astype(float).to_numpy()
    return np.column_stack([f.to_numpy(np.float64), has, bars[list(AUX_COLS)].to_numpy(np.float64)]).astype(np.float32)


def token_bars(raw: Path, tmp: Path, chain: str, base: str, scales) -> Dict[str, pd.DataFrame]:
    """Point-in-time bars of one token at every scale (the store and the evaluation both use this)."""
    ct, refs = _chain(str(raw), str(tmp), chain)
    d = tmp / chain / base
    files = {pd.Period(f.stem, "M"): f for f in d.glob("*.parquet")} if d.is_dir() else {}
    tok_pools = ct.token_pools(base)
    primary = primary_pools({m: a for m, a in ct.activity.items() if m in files}, tok_pools, ct.downloaded)
    cover = sorted(m for m in files if m in primary)
    if not cover:
        return {}
    sw = pd.concat([pd.read_parquet(files[m]) for m in cover], ignore_index=True)
    pools = ct.pools.loc[tok_pools, ["quote_is_token0", "quote_kind", "quote_dec"]].reset_index()
    asof = pd.Timestamp(max((raw / "swaps" / f"{m}.parquet").stat().st_mtime for m in cover), unit="s")
    out = {}
    for s in scales:
        bars = build_token_bars(sw, pools, primary, s, cover, refs, asof=asof)
        if len(bars) >= 2:
            out[s] = bars
    return out


def build_token(job) -> Dict[str, int]:
    raw, tmp, chain, base, idx, store_tmp, scales = job
    out = {}
    for s, bars in token_bars(raw, tmp, chain, base, scales).items():
        np.savez(store_tmp / s / f"{idx}.npz", raw=channels(bars, s),
                 ts=bars["ts"].to_numpy("datetime64[ns]").astype(np.int64), seg=bars["seg"].to_numpy(np.int64))
        out[s] = len(bars)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", default="eth,base,arbitrum")
    ap.add_argument("--scales", default=",".join(SCALES))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--root", type=Path, default=data_root())
    ap.add_argument("--stats-sample", type=int, default=300_000)
    ap.add_argument("--skip-split", action="store_true", help="reuse tmp/ split files from a previous run")
    args = ap.parse_args()
    chains, scales = args.chains.split(","), args.scales.split(",")
    raw_root = args.root / "raw" / "crypto"
    split_dir = args.root / "tmp" / "crypto_split"
    store = args.root / "store" / "crypto"
    store_tmp = store / "_tmp"
    if not args.skip_split:
        shutil.rmtree(split_dir, ignore_errors=True)
        with ProcessPoolExecutor(len(chains)) as ex:
            n = list(ex.map(split_chain, [raw_root / c for c in chains], [split_dir] * len(chains), chains))
        log(f"split {dict(zip(chains, n))} swaps")
        for c in chains:
            build_refs(raw_root / c, split_dir, c)

    rows, act = [], []
    for c in chains:
        ct = ChainTables(raw_root / c, c)
        for base in ct.tokens():
            rows.append((c, base, base in ct.nonstable))
        a = ct.cand[["month", "base"]].drop_duplicates()
        act.append(a.assign(key=c + ":" + a["base"]))
    inst = pd.DataFrame(rows, columns=["chain_name", "base", "anchor"])
    inst["key"] = inst["chain_name"] + ":" + inst["base"]
    inst["idx"] = np.arange(len(inst))
    inst["vocab"] = VOCAB_OFFSET + inst["idx"] + 1
    inst["market"] = 2
    inst["chain"] = inst["chain_name"].map(CHAIN_ID)
    shutil.rmtree(store_tmp, ignore_errors=True)
    for s in scales:
        (store_tmp / s).mkdir(parents=True)
    store.mkdir(parents=True, exist_ok=True)
    jobs = [(raw_root / r.chain_name, split_dir, r.chain_name, r.base, r.idx, store_tmp, scales)
            for r in inst.itertuples()]
    jobs.sort(key=lambda j: j[2])                       # a worker's chain tables are cached per chain
    t0, counts = time.time(), {s: 0 for s in scales}
    with ProcessPoolExecutor(args.workers) as ex:
        for i, res in enumerate(ex.map(build_token, jobs), 1):
            for s, n in res.items():
                counts[s] += n
            if i % 50 == 0 or i == len(jobs):
                log(f"built {i}/{len(jobs)} tokens, {time.time() - t0:.0f}s, bars {counts}")

    built = {int(f.stem) for f in (store_tmp / scales[0]).glob("*.npz")}
    inst[["idx", "key", "vocab", "market", "chain", "anchor"]].to_parquet(store / "instruments.parquet")
    act = pd.concat(act)
    act = act.assign(idx=act["key"].map(dict(zip(inst["key"], inst["idx"]))))
    act = act[act["idx"].isin(built)]
    pd.DataFrame({"month": pd.to_datetime(act["month"]).to_numpy(), "idx": act["idx"].to_numpy(np.int64)}).to_parquet(
        store / "active.parquet")
    starts = [pd.Timestamp(m.start_time) for c in chains for m in ChainTables(raw_root / c, c).swap_months[:1]]
    start, end = min(starts), pd.Timestamp.now(tz="UTC").tz_localize(None).floor("min")
    finalize_store(store, store_tmp, scales, lambda s: crypto_grid(s, start, end), args.stats_sample, log)
    log("done")


if __name__ == "__main__":
    main()
