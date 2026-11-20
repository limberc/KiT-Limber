"""KiT-2 forecasts for the evaluation plan from a Lightning checkpoint (EMA weights).

Each moment: the panel is split into groups; each group = market anchors (context, as in training) + panel
instruments, G = 32. The forecast is made at the close of the moment's bar (the window's latest anchor patch),
K paths, Heun sampler. Paths are denormalised with the store's training-period stats.

Usage: python -m kit2.eval.kit2pred --market us --ckpt <path> [--paths 16 --steps 8]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

from kit2.data.store import CAL_FIRST, denormalise
from kit2.env import data_root
from kit2.eval import EVAL_SCALES, OUT
from kit2.train import Bucket, GroupSampler
from kit2.train.data import to_model_batch
from kit2.train.lit import KiT2Lit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--paths", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", type=Path, default=data_root())
    ap.add_argument("--tag", default="kit2")
    args = ap.parse_args()
    plan = json.loads((OUT / f"plan_{args.market}.json").read_text())
    lit = KiT2Lit.load_from_checkpoint(args.ckpt, map_location="cuda", compile=False)
    model = lit.model.eval()
    G = 32
    buckets = [Bucket(args.market, s) for s in EVAL_SCALES]
    gs = GroupSampler(args.root, buckets, G=G, A=1, batch_tokens=1, split="test")
    st = gs.stores[args.market]
    anchors = gs.always[args.market]
    panel = np.array(plan["panel"], dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(panel)
    room = G - len(anchors)
    groups = [np.concatenate([anchors, order[i: i + room]]) for i in range(0, len(order), room)]
    tables = {}

    def gpu_tables(m, s, dev):
        if (m, s) not in tables:
            sc = st.scales[s]
            tables[(m, s)] = (torch.from_numpy(sc.grid).to(dev), torch.from_numpy(sc.cal).to(dev))
        return tables[(m, s)]

    res, t0 = {}, time.time()
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    for mi, mo in enumerate(plan["moments"]):
        k = EVAL_SCALES.index(mo["scale"])
        bk = buckets[k]
        p_anchor = bk.n_patches[0] - bk.horizon - 1  # the window's latest anchor = the moment
        raw = gs.build(k, np.full(len(groups), mo["end"]), groups, np.full((len(groups), 1), p_anchor))
        batch = {key: torch.as_tensor(v).cuda() for key, v in raw.items()}
        b = to_model_batch(batch, bk, gpu_tables, lit.cfg)
        assert int(b["anchor_close"][0, 0]) == mo["t_ns"], "anchor is not the planned moment"
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.sample({k_: v for k_, v in b.items() if k_ not in ("x0", "fut_ts")}, n_paths=args.paths,
                             steps=args.steps, solver="heun", generator=gen)
        z = rearrange(z[:, 0].float(), "b k g h (p c) -> b g k (h p) c", p=8).cpu().numpy()  # normalised
        grid_idx = mo["t_idx"] + 1 + np.arange(z.shape[3])
        first = st.scales[bk.scale].cal[grid_idx, CAL_FIRST] > 0.5
        feat = denormalise(z, st.stats["scales"][bk.scale], first)  # [B, G, K, n, 5]
        ret = feat[..., 0] + feat[..., 1]
        inst = raw["inst_idx"]
        keep = np.isin(inst, panel)
        res[f"{mi}_inst"] = inst[keep]
        res[f"{mi}_ret"] = ret[keep][..., : mo["lh"]].astype(np.float32)  # [N, K, lh]
        el = time.time() - t0
        print(f"[{mi + 1}/{len(plan['moments'])}] {mo['scale']} {mo['t']} n={int(keep.sum())} {el:.0f}s", flush=True)
    out = OUT / "pred"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{args.tag}_{args.market}.npz", **res)


if __name__ == "__main__":
    main()
