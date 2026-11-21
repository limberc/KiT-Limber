"""KiT-2 forecasts for the evaluation plan from a Lightning checkpoint (EMA weights).

Each moment: the panel is split into groups; each group = market anchors (context, as in training) + panel
instruments, filled up to G = 32 with context instruments from the moment's sampling set, so every group is full
as in training (only panel instruments are scored). The forecast is made at the close of the moment's bar (the
window's latest anchor patch), K paths, Heun sampler. Paths are denormalised with the store's training-period stats.
Besides the panel paths, each moment stores the group of every panel name and the market anchors' paths per group
(path k of one group is one joint draw), which kit2.eval.joint scores.

Usage: python -m kit2.eval.kit2pred --market us --ckpt <path> [--paths 16 --steps 8]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from einops import rearrange

from kit2.data.store import CAL_FIRST, denormalise
from kit2.env import data_root
from kit2.eval import EVAL_SCALES, OUT
from kit2.train import Bucket, GroupSampler, KiT2Lit
from kit2.train.data import to_model_batch


def panel_chunks(panel: np.ndarray, anchors: np.ndarray, G: int, rng: np.random.Generator) -> List[np.ndarray]:
    """Panel names in random groups of G - len(anchors). Panel names that are market anchors are left out: every group
    already holds them (in the crypto panels such a name sat twice in a group, which training never produces)."""
    order = rng.permutation(np.setdiff1d(panel, anchors))
    room = G - len(anchors)
    return [order[i: i + room] for i in range(0, len(order), room)]


def fill_groups(chunks: List[np.ndarray], anchors: np.ndarray, panel: np.ndarray, active: np.ndarray, G: int,
                rng: np.random.Generator) -> List[np.ndarray]:
    """Candidate list per group: market anchors, the group's panel names, then context instruments from the sampling
    set. GroupSampler.build keeps the first G covered candidates, so groups are full. Without the context fill the
    64-name panel was split 30 + 30 + 4, and KiT-2 forecast ~4.5x the volatility for the names in the small group."""
    ctx = rng.permutation(np.setdiff1d(active, np.concatenate([anchors, panel])))[: 4 * G]
    return [np.concatenate([anchors, c, ctx]) for c in chunks]


def group_layout(inst: np.ndarray, panel: np.ndarray, anchors: np.ndarray):
    """inst [n_groups, G] instrument ids of the groups sent to the model (-1 = padding) -> keep, grp, anc, anc_pos:
    keep     flat positions of the panel names to score (a name in several groups - a market anchor - once, first group)
    grp      group row of each kept name: path k of the names in one row is one joint draw of that group's future
    anc      market anchors present in every group; anc_pos [n_groups, len(anc)] their position in each row."""
    flat = inst.reshape(-1)
    _, first_at = np.unique(flat, return_index=True)
    keep = np.sort(first_at[np.isin(flat[first_at], panel)])
    anc = np.array([a for a in anchors if (inst == a).any(1).all()], dtype=np.int64)
    anc_pos = np.array([[int(np.flatnonzero(row == a)[0]) for a in anc] for row in inst], dtype=np.int64)
    return keep, keep // inst.shape[1], anc, anc_pos.reshape(len(inst), len(anc))


def moment_batches(plan: dict, lit, root: Path, seed: int, G: int = 32):
    """Per moment of the plan: the full groups sent to the model (fill_groups) and their model batch on the GPU.
    Yields dicts with mi, mo, bk, st (store), raw (sampler output), b (model batch; b["x0"] = the realised normalised
    futures), keep / grp / anc / anc_pos (group_layout) and anc_mask [n_groups, G] (the market anchors of each group)."""
    buckets = [Bucket(plan["market"], s) for s in EVAL_SCALES]
    gs = GroupSampler(root, buckets, G=G, A=1, batch_tokens=1, split=plan.get("split", "test"))
    st = gs.stores[plan["market"]]
    anchors = gs.always[plan["market"]]
    panel = np.array(plan["panel"], dtype=np.int64)
    rng = np.random.default_rng(seed)
    chunks = panel_chunks(panel, anchors, G, rng)
    tables = {}

    def gpu_tables(m, s, dev):
        if (m, s) not in tables:
            sc = st.scales[s]
            tables[(m, s)] = (torch.from_numpy(sc.grid).to(dev), torch.from_numpy(sc.cal).to(dev))
        return tables[(m, s)]

    for mi, mo in enumerate(plan["moments"]):
        k = EVAL_SCALES.index(mo["scale"])
        bk = buckets[k]
        p_anchor = bk.n_patches[0] - bk.horizon - 1  # the window's latest anchor = the moment
        act = st.active_at(mo["t_ns"])
        groups = fill_groups(chunks, anchors, panel, act if act is not None else np.zeros(0, np.int64), G, rng)
        raw = gs.build(k, np.full(len(groups), mo["end"]), groups, np.full((len(groups), 1), p_anchor))
        assert np.isin(panel, raw["inst_idx"]).all(), "a panel instrument is not covered at this moment"
        batch = {key: torch.as_tensor(v).cuda() for key, v in raw.items()}
        b = to_model_batch(batch, bk, gpu_tables, lit.cfg)
        assert int(b["anchor_close"][0, 0]) == mo["t_ns"], "anchor is not the planned moment"
        keep, grp, anc, anc_pos = group_layout(raw["inst_idx"], panel, anchors)
        assert len(keep) == len(panel)
        mask = torch.zeros(b["valid"].shape, dtype=torch.bool, device="cuda")
        mask[torch.arange(len(anc_pos), device="cuda")[:, None], torch.as_tensor(anc_pos, device="cuda")] = True
        yield {"mi": mi, "mo": mo, "bk": bk, "st": st, "raw": raw, "b": b, "keep": keep, "grp": grp, "anc": anc,
               "anc_pos": anc_pos, "anc_mask": mask}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--paths", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", type=Path, default=data_root())
    ap.add_argument("--tag", default="kit2")
    ap.add_argument("--condition", choices=("none", "oracle", "couple"), default="none",
                    help="oracle: market anchors take their realised futures (scenario evaluation - uses the future, "
                         "not a forecast); couple: every group takes group 0's sampled anchor paths, so path k is one "
                         "joint draw of the whole panel")
    ap.add_argument("--cond-mode", choices=("clean", "replace"), default="replace",
                    help="clean needs a checkpoint trained with p_cond > 0; replace (inpainting) works with any")
    args = ap.parse_args()
    plan = json.loads((OUT / f"plan_{args.market}.json").read_text())
    lit = KiT2Lit.load_from_checkpoint(args.ckpt, map_location="cuda", compile=False)
    model = lit.model.eval()
    res, t0 = {}, time.time()
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    for m in moment_batches(plan, lit, args.root, args.seed):
        mi, mo, bk, st, raw, b = m["mi"], m["mo"], m["bk"], m["st"], m["raw"], m["b"]
        keep, grp, anc, anc_pos, mask = m["keep"], m["grp"], m["anc"], m["anc_pos"], m["anc_mask"]
        inp = {k_: v for k_, v in b.items() if k_ not in ("x0", "fut_ts")}
        rows = torch.arange(len(anc_pos), device="cuda")[:, None]
        pos = torch.as_tensor(anc_pos, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            kw = dict(n_paths=args.paths, steps=args.steps, solver="heun", generator=gen)
            if args.condition == "none" or len(anc) == 0:
                z = model.sample(inp, **kw)
            elif args.condition == "oracle":
                z = model.sample(inp, given=b["x0"], given_mask=mask, mode=args.cond_mode, **kw)
            else:
                z0 = model.sample(inp, **kw)  # [B, A, K, G, h, C]
                given = torch.zeros_like(z0)
                given[rows, :, :, pos] = z0[0, :, :, pos[0]].unsqueeze(0).movedim(3, 1)  # group 0's anchor paths
                z = model.sample(inp, given=given, given_mask=mask, mode=args.cond_mode, **kw)
        z = rearrange(z[:, 0].float(), "b k g h (p c) -> b g k (h p) c", p=8).cpu().numpy()  # normalised
        grid_idx = mo["t_idx"] + 1 + np.arange(z.shape[3])
        first = st.scales[bk.scale].cal[grid_idx, CAL_FIRST] > 0.5
        feat = denormalise(z, st.stats["scales"][bk.scale], first)  # [B, G, K, n, 5]
        ret = feat[..., 0] + feat[..., 1]  # [n_groups, G, K, n]
        inst = raw["inst_idx"].reshape(-1)
        res[f"{mi}_inst"] = inst[keep]
        res[f"{mi}_ret"] = ret.reshape(-1, *ret.shape[2:])[keep][..., : mo["lh"]].astype(np.float32)  # [N, K, lh]
        res[f"{mi}_grp"] = grp  # joint draws: same path k within one group
        res[f"{mi}_anc_inst"] = anc
        res[f"{mi}_anc_ret"] = ret[np.arange(len(ret))[:, None], anc_pos][..., : mo["lh"]].astype(np.float32)
        el = time.time() - t0
        print(f"[{mi + 1}/{len(plan['moments'])}] {mo['scale']} {mo['t']} n={len(keep)} "
              f"group sizes {raw['valid'].sum(1).min()}-{raw['valid'].sum(1).max()} {el:.0f}s", flush=True)
    out = OUT / "pred"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{args.tag}_{args.market}.npz", **res)


if __name__ == "__main__":
    main()
