"""Surprise scores: the log density of the realised future window under KiT-2, and how well it separates real windows
from structurally corrupted ones.

Per moment, every group's realised futures (normalised, all 5 candle channels) are scored with the probability-flow
ODE (KiT2.log_likelihood: each instrument's share of the joint density), together with four corrupted copies of the
panel names' futures, each breaking one kind of structure while keeping the per-bar values:
  shuffle    bars permuted in time within the window (volatility clustering, intraday shape)
  decouple   the volume channel permuted in time on its own (price-volume coupling)
  swap       the future of another panel name of the same group (fit between an instrument's history and future)
  spike      one bar's body set to +-5 (normalised units, ~7 robust-scale units raw): a jump
Densities are taken in raw feature space (the soft clip's log-Jacobian added), surprise = -log density, and the
paired AUROC per corruption is P(surprise(corrupted) > surprise(real)) over (name, moment).
Baseline trail_gauss: each channel i.i.d. Gaussian with the mean / std of the instrument's last 4 * lh target-scale
history bars - blind to shuffle and decouple by construction; those two are what only a joint path model can see.
--given anchors (checkpoints trained with p_cond > 0): densities conditional on the market anchors' realised futures.

Usage: python -m kit2.eval.surprise --market us --ckpt <path> [--steps 32 --probes 4 --tag kit2_mix]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange, repeat

from kit2.data.store import CAL_FIRST, SOFT_CLIP
from kit2.env import data_root
from kit2.eval import OUT
from kit2.eval.kit2pred import moment_batches
from kit2.train import KiT2Lit
from kit2.train.buckets import SCALE_ID

CORRUPTIONS = ("shuffle", "decouple", "swap", "spike")
BODY, VOL, NC = 1, 4, 5     # candle channels: r_gap, r_body, r_up, r_dn, v


def corrupt(x: torch.Tensor, panel: torch.Tensor, kind: str, g: torch.Generator) -> torch.Tensor:
    """x [B, G, n, 5] realised normalised futures -> a copy with the panel names' (panel [B, G]) futures corrupted."""
    B, G, n, C = x.shape
    dev = x.device
    if kind == "shuffle":
        perm = torch.rand((B, G, n), generator=g, device=dev).argsort(-1)
        y = torch.gather(x, 2, repeat(perm, "b g n -> b g n c", c=C))
    elif kind == "decouple":
        perm = torch.rand((B, G, n), generator=g, device=dev).argsort(-1)
        y = x.clone()
        y[..., VOL] = torch.gather(x[..., VOL], 2, perm)
    elif kind == "swap":
        y = x.clone()
        for bi in range(B):
            idx = torch.nonzero(panel[bi]).flatten()
            if len(idx) > 1:
                y[bi, idx] = x[bi, idx.roll(1)]
    elif kind == "spike":
        y = x.clone()
        j = torch.randint(0, n, (B, G), generator=g, device=dev)
        sign = (torch.randint(0, 2, (B, G), generator=g, device=dev) * 2 - 1).to(x.dtype)
        y[torch.arange(B, device=dev)[:, None], torch.arange(G, device=dev)[None], j, BODY] = 5.0 * sign
    else:
        raise ValueError(kind)
    return torch.where(rearrange(panel, "b g -> b g 1 1"), y, x)


def log_jacobian(y: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    """log |d normalised / d raw| summed over bars and channels: y [..., n, 5], log_scale [n, 5] -> [...]."""
    return (torch.log1p(-(y / SOFT_CLIP) ** 2).clamp_min(-30.0) - log_scale).sum((-1, -2))


def trail_gauss(hist: torch.Tensor, fut: torch.Tensor) -> torch.Tensor:
    """Per-channel i.i.d. Gaussian log density (normalised space): hist [B, G, m, 5], fut [..., B, G, n, 5]."""
    mu, sd = hist.mean(-2), hist.std(-2).clamp_min(1e-3)
    zz = (fut - mu[..., None, :]) / sd[..., None, :]
    return (-0.5 * zz ** 2 - torch.log(sd[..., None, :]) - 0.5 * math.log(2 * math.pi)).sum((-1, -2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--probes", type=int, default=4)
    ap.add_argument("--given", choices=("none", "anchors"), default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", type=Path, default=data_root())
    ap.add_argument("--tag", default="kit2")
    args = ap.parse_args()
    plan = json.loads((OUT / f"plan_{args.market}.json").read_text())
    lit = KiT2Lit.load_from_checkpoint(args.ckpt, map_location="cuda", compile=False)
    model = lit.model.eval()
    P = lit.cfg.patch
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    rows, t0 = [], time.time()
    for m in moment_batches(plan, lit, args.root, args.seed):
        mi, mo, bk, st, raw, b = m["mi"], m["mo"], m["bk"], m["st"], m["raw"], m["b"]
        inp = {k: v for k, v in b.items() if k not in ("x0", "fut_ts")}
        x0 = rearrange(b["x0"][:, 0].float(), "b g h (p c) -> b g (h p) c", p=P)     # [B, G, n, 5]
        B, G, n, _ = x0.shape
        panel = torch.as_tensor(np.isin(raw["inst_idx"], plan["panel"]), device="cuda")
        variants = torch.stack([x0] + [corrupt(x0, panel, c, gen) for c in CORRUPTIONS], 1)  # [B, V, G, n, 5]
        nv = variants.shape[1]
        rep = {k: repeat(b[k][:, :1], "b 1 ... -> b v ...", v=nv) for k in ("anchor_close", "anchor_pos", "fut_cal")}
        given = m["anc_mask"] if args.given == "anchors" else None
        ll = model.log_likelihood({**inp, **rep}, rearrange(variants, "b v g (h p) c -> b v g h (p c)", p=P),
                                  steps=args.steps, n_probe=args.probes, given_mask=given, generator=gen)
        # raw-space densities: + log-Jacobian of the soft clip and the per-bar channel scales
        stats = st.stats["scales"][bk.scale]
        grid_idx = mo["t_idx"] + 1 + np.arange(n)
        first = st.scales[bk.scale].cal[grid_idx, CAL_FIRST] > 0.5
        sc = np.broadcast_to(np.asarray(stats["scale"][:NC], dtype=np.float64), (n, NC)).copy()
        sc[first, 0] = stats["gap_first_scale"]
        lj = log_jacobian(variants, torch.as_tensor(np.log(sc), device="cuda", dtype=torch.float32))   # [B, V, G]
        # target-scale history tokens up to the moment (the window also holds the future patches as tokens)
        sel = (b["tok_scale"] == SCALE_ID[bk.scale]) & (b["tok_close"][0] <= b["anchor_close"][0, 0])
        hist = rearrange(b["x_hist"][:, :, sel].float(), "b g s (p c) -> b g (s p) c", p=P)[..., -4 * mo["lh"]:, :NC]
        tg = trail_gauss(hist, rearrange(variants, "b v g n c -> v b g n c"))           # [V, B, G]
        scores = {"kit2": -(ll + lj), "trail_gauss": -(rearrange(tg, "v b g -> b v g") + lj)}
        inst = raw["inst_idx"]
        for gi in range(B):
            for slot in range(G):
                if not bool(panel[gi, slot]) or (given is not None and bool(given[gi, slot])):
                    continue
                for model_name, s in scores.items():
                    r = {"market": args.market, "scale": mo["scale"], "moment": mo["t"], "inst": int(inst[gi, slot]),
                         "model": model_name, "real": float(s[gi, 0, slot])}
                    r |= {c: float(s[gi, 1 + j, slot]) for j, c in enumerate(CORRUPTIONS)}
                    rows.append(r)
        print(f"[{mi + 1}/{len(plan['moments'])}] {mo['scale']} {mo['t']} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows).drop_duplicates(["moment", "scale", "inst", "model"])   # anchors in the panel: once
    df.to_csv(OUT / f"surprise_{args.tag}_{args.market}.csv", index=False)
    auc = df.assign(**{c: df[c] > df["real"] for c in CORRUPTIONS}).groupby(["scale", "model"])[list(CORRUPTIONS)]
    pd.set_option("display.width", 200)
    print("paired AUROC: P(surprise of corrupted > real)\n" + auc.mean().round(3).to_string())


if __name__ == "__main__":
    main()
