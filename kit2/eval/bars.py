"""Rebuild the panel's point-in-time bars with the store's own builders (unnormalised features for targets,
KiT inputs and the bootstrap baseline).

Usage: python -m kit2.eval.bars --market us [--workers 16]
Writes bars/<market>/<idx>.npz with, per scale s: s_ts (ns), s_feat [n, 5] raw candle features, s_dvol, s_has.
Instruments that already have a file are skipped (files cover all dates, so plans for different periods can
share them); delete a file to rebuild it.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from kit2.data.features import candle_features
from kit2.eval import EVAL_SCALES, OUT
from kit2.env import data_root


def _us(job):
    idx, key, root = job
    from kit2.data.us.prepare import HALF_LIFE_DAYS, bars_per_day, symbol_bars
    raw = root / "raw" / "us"
    plan = pd.read_parquet(raw / "universe" / "intraday_plan_top1000.parquet")
    months = sorted(set(plan.loc[(plan["symbol"] == key) & (plan["month"] >= pd.Timestamp("2023-10")), "month"]))
    bars = symbol_bars(key, months, raw, "2023-10", EVAL_SCALES)
    return idx, {s: (b, HALF_LIFE_DAYS * bars_per_day(s)) for s, b in bars.items()}


def _crypto(job):
    idx, key, root = job
    from kit2.data.crypto.prepare import HALF_LIFE_DAYS, STEP_S, token_bars
    chain, base = key.split(":")
    bars = token_bars(root / "raw" / "crypto" / chain, root / "tmp" / "crypto_split", chain, base, EVAL_SCALES)
    return idx, {s: (b, HALF_LIFE_DAYS * 86400 / STEP_S[s]) for s, b in bars.items()}


def save(idx: int, bars: dict, out: Path) -> None:
    arrs = {}
    for s, (b, hl) in bars.items():
        arrs[f"{s}_ts"] = b["ts"].to_numpy("datetime64[ns]").astype(np.int64)
        arrs[f"{s}_feat"] = candle_features(b, hl).to_numpy(np.float64)
        arrs[f"{s}_dvol"] = b["dvol"].to_numpy(np.float64)
        arrs[f"{s}_has"] = (b["n_min"].fillna(1.0) > 0).to_numpy()
    np.savez(out / f"{idx}.npz", **arrs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--root", type=Path, default=data_root())
    args = ap.parse_args()
    plan = json.loads((OUT / f"plan_{args.market}.json").read_text())
    out = OUT / "bars" / args.market
    out.mkdir(parents=True, exist_ok=True)
    fn = _us if args.market == "us" else _crypto
    jobs = [(i, k, args.root) for i, k in zip(plan["panel"], plan["keys"]) if not (out / f"{i}.npz").is_file()]
    print(f"{len(plan['panel']) - len(jobs)} of {len(plan['panel'])} instruments already built", flush=True)
    if args.market == "crypto" and jobs:
        from kit2.data.crypto.prepare import ensure_split
        ensure_split(args.root / "raw" / "crypto", args.root / "tmp" / "crypto_split", [k for _, k, _ in jobs])
    empty = []
    with ProcessPoolExecutor(args.workers) as ex:
        for n, (idx, bars) in enumerate(ex.map(fn, jobs), 1):
            if bars:
                save(idx, bars, out)
            else:
                empty.append(idx)
            if n % 16 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} instruments", flush=True)
    if empty:
        # crypto: token_bars reads the per-token split files under tmp/crypto_split (rebuilt above when missing);
        # a token without downloaded swaps has none
        raise SystemExit(f"no bars for {len(empty)} instruments (e.g. {empty[:5]}): source files missing; "
                         "nothing written for them")


if __name__ == "__main__":
    main()
