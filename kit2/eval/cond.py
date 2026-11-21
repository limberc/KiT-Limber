"""Scenario baseline for conditional forecasts: LightGBM + Gaussian copula, conditioned on the market anchors'
realised per-bar returns over the horizon (the same information as `kit2pred --condition oracle`).

Per moment and panel name i: per-bar returns of (anchors, i) are Gaussian with means mu / lh and vols sigma from the
LightGBM forecasts (pred/<lgbm>_<market>.npz, which include the anchors) and the Ledoit-Wolf correlation of their
trailing per-bar returns; i's bars are drawn from the conditional Gaussian given the anchors' realised bars. This is
the "linear beta + Gaussian residual" answer to "what does the panel do given this market path"; KiT-2's conditional
paths have to beat it on the same names and moments (kit2.eval.score).

Usage: python -m kit2.eval.cond --market us [--lgbm lgbm --paths 128 --tag lgbm_orc]
"""
from __future__ import annotations

import argparse
import json
from functools import reduce

import numpy as np

from kit2.eval import OUT
from kit2.eval.joint import ledoit_wolf_corr, load_bars, window


def conditional_paths(mu: np.ndarray, sigma: np.ndarray, C: np.ndarray, y_anc: np.ndarray, K: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Members = anchors (first n_a) + one name (last). mu, sigma [n_a + 1] horizon mean / per-bar vol, C correlation,
    y_anc [n_a, lh] realised anchor bars -> [K, lh] draws of the name's bars given the anchors."""
    n_a, lh = y_anc.shape
    S = C * np.outer(sigma, sigma)
    w = np.linalg.solve(S[:n_a, :n_a] + 1e-12 * np.eye(n_a), S[:n_a, n_a])      # regression of the name on anchors
    m = mu[n_a] / lh + w @ (y_anc - (mu[:n_a] / lh)[:, None])                   # [lh] conditional mean per bar
    sd = np.sqrt(max(S[n_a, n_a] - S[:n_a, n_a] @ w, 1e-24))
    return m[None] + sd * rng.standard_normal((K, lh))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True)
    ap.add_argument("--lgbm", default="lgbm")
    ap.add_argument("--paths", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="lgbm_orc")
    args = ap.parse_args()
    plan = json.loads((OUT / f"plan_{args.market}.json").read_text())
    anchors = [int(a) for a in plan.get("anchors", [])]
    bars = load_bars(args.market, set(plan["panel"]) | set(anchors))
    lg = np.load(OUT / "pred" / f"{args.lgbm}_{args.market}.npz")
    rng = np.random.default_rng(args.seed)
    res, skipped = {}, 0
    for mi, mo in enumerate(plan["moments"]):
        s, lh = mo["scale"], mo["lh"]
        lpos = {int(i): j for j, i in enumerate(lg[f"{mi}_inst"])}
        L = lg[f"{mi}_ret"].astype(np.float64)
        mu_all, sig_all = L.sum(2).mean(1), L.std(2, ddof=1).mean(1)
        wins = {i: w for i in bars if (w := window(bars[i], s, mo["t_ns"], lh)) is not None}
        anc = [a for a in anchors if a in wins and a in lpos]
        if not anc:
            continue
        inst, out = [], []
        for i in plan["panel"]:
            if i in anc or i not in wins or i not in lpos or not np.array_equal(wins[i][2], wins[anc[0]][2]):
                skipped += 1
                continue
            ids = anc + [i]
            common = reduce(np.intersect1d, [wins[j][0] for j in ids])
            if len(common) < lh + 100:
                skipped += 1
                continue
            Rh = np.stack([wins[j][1][np.searchsorted(wins[j][0], common)] for j in ids])
            if (Rh.std(1) == 0).any():
                skipped += 1
                continue
            mu, sig = mu_all[[lpos[j] for j in ids]], sig_all[[lpos[j] for j in ids]]
            y_anc = np.stack([wins[a][3] for a in anc])
            inst.append(i)
            out.append(conditional_paths(mu, sig, ledoit_wolf_corr(Rh.T), y_anc, args.paths, rng))
        if inst:
            res[f"{mi}_inst"] = np.array(inst, dtype=np.int64)
            res[f"{mi}_ret"] = np.stack(out).astype(np.float32)
    print(f"{args.market}: {len(res) // 2} moments, {skipped} (name, moment) pairs skipped "
          f"(an anchor, no bars / aligned horizon / trailing history)", flush=True)
    (OUT / "pred").mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "pred" / f"{args.tag}_{args.market}.npz", **res)


if __name__ == "__main__":
    main()
