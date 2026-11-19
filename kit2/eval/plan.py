"""Evaluation plan: test moments per (market, scale) and one fixed instrument panel per market.

A moment is a target-scale bar close T after VAL_END such that the KiT-2 window ending h patches after T fits
in the store (so the forecast is made at the latest anchor of a training-shaped window). The panel is drawn
from instruments in the sampling set of every test month and covered at every moment; both models forecast
exactly these (instrument, moment) pairs.

Usage: python -m kit2.eval.plan --market us [--n-inst 64 --moments 6]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kit2.data.store import VAL_END
from kit2.eval import EVAL_SCALES, KIT_LH, OUT
from kit2.env import data_root
from kit2.train.buckets import Bucket
from kit2.train.data import GroupSampler


def make_plan(root: Path, market: str, n_inst: int, n_moments: int, seed: int) -> dict:
    buckets = [Bucket(market, s) for s in EVAL_SCALES]
    gs = GroupSampler(root, buckets, G=32, A=1, batch_tokens=1, split="test")
    st = gs.stores[market]
    months = sorted(m for m in st.active if m > VAL_END.normalize().replace(day=1))
    pool = set(st.active[months[0]])
    for m in months[1:]:
        pool &= set(st.active[m])
    pool = np.array(sorted(pool), dtype=np.int64)
    rng = np.random.default_rng(seed)
    moments = []
    for k, bk in enumerate(buckets):
        ends = gs._ends[k]
        grid = st.scales[bk.scale].grid
        # coverage of the pool at every candidate window end; keep ends where most of the pool is covered
        cand = ends[np.linspace(0, len(ends) - 1, min(len(ends), 40)).round().astype(int)]
        cov = np.stack([np.all(gs.window_offsets(k, int(e), pool)[1] >= 0, axis=0) for e in cand])
        good = cand[cov.mean(1) >= 0.5 * cov.mean(1).max()]
        pick = good[np.linspace(0, len(good) - 1, min(n_moments, len(good))).round().astype(int)]
        for e in pick:
            e_t = int(e) - bk.horizon * 8
            moments.append({"scale": bk.scale, "k": k, "end": int(e), "t_idx": e_t, "t_ns": int(grid[e_t]),
                            "t": str(pd.Timestamp(int(grid[e_t]))), "lh": KIT_LH[bk.scale]})
    covered = np.ones(len(pool), dtype=bool)
    for mo in moments:
        covered &= np.all(gs.window_offsets(mo["k"], mo["end"], pool)[1] >= 0, axis=0)
    full = pool[covered]
    panel = np.sort(rng.choice(full, size=min(n_inst, len(full)), replace=False))
    keys = st.instruments.set_index("idx").loc[panel, "key"].tolist()
    return {"market": market, "panel": panel.tolist(), "keys": keys, "moments": moments,
            "pool_size": int(len(pool)), "fully_covered": int(len(full)), "test_start": str(VAL_END)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True)
    ap.add_argument("--n-inst", type=int, default=64)
    ap.add_argument("--moments", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", type=Path, default=data_root())
    args = ap.parse_args()
    plan = make_plan(args.root, args.market, args.n_inst, args.moments, args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"plan_{args.market}.json").write_text(json.dumps(plan, indent=1))
    by = pd.DataFrame(plan["moments"]).groupby("scale")["t"].agg(["count", "min", "max"])
    print(f"{args.market}: panel {len(plan['panel'])} of {plan['fully_covered']} fully covered "
          f"(pool {plan['pool_size']})\n{by}")


if __name__ == "__main__":
    main()
