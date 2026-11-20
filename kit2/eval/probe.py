"""Linear probe: where does KiT-2 lose the volatility signal - in the encoder or in the generator?

Per eval bucket (market x EVAL_SCALES), validation windows are drawn as in training (split "val", Jul-Aug 2026:
unseen by the model, test untouched). For every (window, anchor, instrument):
  y       log realised vol: std of per-bar returns (r_gap + r_body, raw) over the first KIT_LH bars after the anchor
  enc     encoder states: the anchor's target-scale token and the mean of all memory tokens visible at the anchor
  trail   log trailing vol over lh/4, lh/2, lh, 2 lh, 4 lh bars up to the anchor (hand-made history features)
  model   the model's own forecast: mean over K sampled paths of each path's per-bar return std (as kit2.eval.score)
Ridge probes (on within-cross-section ranks of y and of the trailing vols, demeaned encoder states; penalty by
window-grouped CV) are fit on the earlier half of the windows and scored on the later half. Score = Spearman across
instruments per (window, anchor) - the vol RankIC definition - averaged per window, then over windows (s.e. over
windows).

Reading: probe(enc) >= trailing vol while the model's sampled vol is far below -> the encoder has the signal and
the generator loses it; probe(enc) well below trailing vol -> the encoder discards it.

Usage: python -m kit2.eval.probe --ckpt <path> [--tag 20k --windows 128 --paths 16 --steps 8]
"""
from __future__ import annotations

import os

# aa3's cgroup allows ~10 CPUs but shows 48: OpenBLAS's default 48 threads made one 1024 x 1024 SVD take 76 s (0.5 s
# with 8). Must be set before numpy loads OpenBLAS.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable, Dict, List, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from einops import rearrange  # noqa: E402

from kit2.data.store import CAL_FIRST, denormalise  # noqa: E402
from kit2.env import data_root  # noqa: E402
from kit2.eval import EVAL_SCALES, KIT_LH, OUT  # noqa: E402
from kit2.train import KiT2Lit, GroupSampler, Bucket  # noqa: E402
from kit2.train.batch import model_inputs  # noqa: E402
from kit2.train.data import to_model_batch  # noqa: E402
from .score import spearman  # noqa: E402

P = 8
G = 32
TRAIL = (0.25, 0.5, 1.0, 2.0, 4.0)  # trailing windows in multiples of lh


def log_vol(y: np.ndarray, stats: dict, first: np.ndarray) -> np.ndarray:
    """Normalised candle bars [..., n, >=5] -> log std (ddof 1) of raw per-bar returns over the n bars."""
    f = denormalise(y, stats, first)
    return np.log(np.maximum((f[..., 0] + f[..., 1]).std(-1, ddof=1), 1e-12))


@torch.no_grad()
def collect(model, cfg, root: Path, bk: Bucket, n_windows: int, chunk: int, paths: int, steps: int,
            seed: int) -> List[Dict[str, np.ndarray]]:
    """One record per (window, anchor): arrays over that cross-section's valid instruments."""
    gs = GroupSampler(root, [bk], G=G, A=4, batch_tokens=chunk * bk.tokens(G), split="val")
    st = gs.stores[bk.market]
    stats, cal = st.stats["scales"][bk.scale], st.scales[bk.scale].cal
    lh, h = KIT_LH[bk.scale], bk.horizon
    tables = {}

    def gpu_tables(m, s, dev):
        if (m, s) not in tables:
            sc = st.scales[s]
            tables[(m, s)] = (torch.from_numpy(sc.grid).to(dev), torch.from_numpy(sc.cal).to(dev))
        return tables[(m, s)]

    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    recs: List[Dict[str, np.ndarray]] = []
    for c in range(n_windows // chunk):
        raw = gs.sample(0, rng)
        if raw is None:
            break
        batch = {k: torch.as_tensor(v).cuda() for k, v in raw.items()}
        b = to_model_batch(batch, bk, gpu_tables, cfg)
        inp = model_inputs(b)
        anchors = raw["anchors"]  # [B, A]
        B, A = anchors.shape
        bi = torch.arange(B, device="cuda")[:, None]
        at = torch.as_tensor(anchors, device="cuda")
        assert bool((b["tok_close"][bi, at] == b["anchor_close"]).all()), "anchor token is not the target patch"
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hmem = model.encode(inp)["h"].float()  # [B, G, S, d]
            z = model.sample(inp, n_paths=paths, steps=steps, solver="heun", generator=gen)
        tok = hmem[bi, :, at]  # [B, A, G, d] anchor token
        vis = (b["tok_close"][:, None, :] <= b["anchor_close"][:, :, None]).float()  # [B, A, S]
        vis = vis / vis.sum(-1, keepdim=True)
        mean = torch.einsum("bas,bgsd->bagd", vis, hmem)
        enc = torch.cat([tok, mean], -1).cpu().numpy()  # [B, A, G, 2d]
        x0 = rearrange(b["x0"].float(), "b a g h (p c) -> b a g (h p) c", p=P).cpu().numpy()
        zz = rearrange(z.float(), "b a k g h (p c) -> b a k g (h p) c", p=P).cpu().numpy()
        t_anchor = b["anchor_close"].cpu().numpy()
        hist, start = raw["x0"], raw["start0"]  # target-scale window
        for i in range(B):
            ok = raw["valid"][i]
            for a in range(A):
                fut = start[i] + (anchors[i, a] + 1) * P + np.arange(h * P)
                first_f = cal[fut[:lh], CAL_FIRST] > 0.5
                end = (anchors[i, a] + 1) * P  # bars [.., end) are history
                w_max = int(TRAIL[-1] * lh)
                first_h = cal[start[i] + np.arange(end - w_max, end), CAL_FIRST] > 0.5
                f = denormalise(hist[i, :, end - w_max: end].astype(np.float64), stats, first_h)
                r = f[..., 0] + f[..., 1]  # [G, w_max]
                trail = np.stack([np.log(np.maximum(r[:, -max(2, int(m * lh)):].std(-1, ddof=1), 1e-12))
                                  for m in TRAIL], -1)
                recs.append({
                    "win": np.int64(c * B + i), "t": np.int64(t_anchor[i, a]),
                    "y": log_vol(x0[i, a, :, :lh], stats, first_f)[ok],
                    "model": np.log(np.exp(log_vol(zz[i, a, :, :, :lh], stats, first_f)).mean(0))[ok],
                    "trail": trail[ok], "enc": enc[i, a][ok],
                })
    return recs


def ridge(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
          alphas=np.logspace(-1, 6, 15)) -> Tuple[Callable[[np.ndarray], np.ndarray], float]:
    """Ridge on standardised X, penalty chosen by 5-fold CV over groups (windows)."""
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd

    def path(Zf, yf, als):
        U, S, Vt = np.linalg.svd(Zf, full_matrices=False)
        Uy = U.T @ yf
        return np.stack([Vt.T @ (S / (S ** 2 + a) * Uy) for a in als])

    err = np.zeros(len(alphas))
    for fold in np.array_split(np.unique(groups), 5):
        te = np.isin(groups, fold)
        W = path(Z[~te], y[~te], alphas)
        err += ((Z[te] @ W.T - y[te, None]) ** 2).sum(0)
    best = float(alphas[int(err.argmin())])
    w = path(Z, y, [best])[0]
    return (lambda Xn: ((Xn - mu) / sd) @ w), best


def score(recs: List[dict], pred: Callable[[dict], np.ndarray]) -> Tuple[float, float]:
    """Spearman per cross-section, mean per window, then mean and s.e. over windows."""
    by_win: Dict[int, List[float]] = {}
    for r in recs:
        by_win.setdefault(int(r["win"]), []).append(spearman(pred(r), r["y"]))
    per = np.array([np.nanmean(v) for v in by_win.values()])
    per = per[np.isfinite(per)]
    return float(per.mean()), float(per.std(ddof=1) / np.sqrt(len(per)))


def cs_rank(x: np.ndarray) -> np.ndarray:
    """Ranks within one cross-section (columns separately, ties averaged), scaled to [-1, 1]. Flat-priced
    instruments have zero realised vol (log vol -27), which would dominate a squared-error fit."""
    r = pd.DataFrame(x).rank().to_numpy()
    return (2 * (r - 1) / max(len(x) - 1, 1) - 1).reshape(x.shape)


def feats(r: dict, keys) -> np.ndarray:
    """Probe inputs of one cross-section: trailing vols as ranks, encoder states demeaned."""
    return np.concatenate([cs_rank(r[k]) if k == "trail" else r[k] - r[k].mean(0, keepdims=True) for k in keys], 1)


def evaluate(recs: List[dict]) -> List[dict]:
    t_win = pd.Series({int(r["win"]): int(r["t"]) for r in recs})
    split = t_win.median()
    fit = [r for r in recs if t_win[int(r["win"])] <= split]
    test = [r for r in recs if t_win[int(r["win"])] > split]
    groups = np.concatenate([np.full(len(r["y"]), int(r["win"])) for r in fit])
    y = np.concatenate([cs_rank(r["y"]) for r in fit])
    out = []
    for name, keys in (("probe_trail", ("trail",)), ("probe_enc", ("enc",)), ("probe_enc+trail", ("enc", "trail"))):
        f, alpha = ridge(np.concatenate([feats(r, keys) for r in fit]), y, groups)
        m, se = score(test, lambda r, f=f, keys=keys: f(feats(r, keys)))
        out.append({"method": name, "rankic": m, "se": se, "alpha": alpha})
    for name, fn in (("trail_lh", lambda r: r["trail"][:, TRAIL.index(1.0)]), ("model", lambda r: r["model"])):
        m, se = score(test, fn)
        out.append({"method": name, "rankic": m, "se": se, "alpha": np.nan})
    agree = np.nanmean([spearman(r["model"], r["trail"][:, TRAIL.index(1.0)]) for r in test])
    out.append({"method": "model~trail_lh", "rankic": float(agree), "se": np.nan, "alpha": np.nan})
    n_days = len({pd.Timestamp(int(r["t"])).normalize() for r in test})
    for o in out:
        o.update({"n_fit_rows": len(y), "n_test_windows": len({int(r["win"]) for r in test}), "test_days": n_days})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tag", default="kit2")
    ap.add_argument("--markets", default="us,crypto")
    ap.add_argument("--scales", default=",".join(EVAL_SCALES))
    ap.add_argument("--windows", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--paths", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", type=Path, default=data_root())
    args = ap.parse_args()
    lit = KiT2Lit.load_from_checkpoint(args.ckpt, map_location="cuda", compile=False)
    model = lit.model.eval()
    rows, t0 = [], time.time()
    for market in args.markets.split(","):
        for scale in args.scales.split(","):
            bk = Bucket(market, scale)
            recs = collect(model, lit.cfg, args.root, bk, args.windows, args.chunk, args.paths, args.steps, args.seed)
            for o in evaluate(recs):
                rows.append({"market": market, "scale": scale, **o})
            print(f"{market} {scale}: {len(recs)} cross-sections, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"probe_{args.tag}.csv", index=False)
    wide = df.pivot_table(index=["market", "scale"], columns="method", values="rankic", sort=False)
    cols = ["trail_lh", "probe_trail", "probe_enc", "probe_enc+trail", "model", "model~trail_lh"]
    print(wide[cols].round(3).to_string())
    se = df.pivot_table(index=["market", "scale"], columns="method", values="se", sort=False)
    print("\ns.e. (over test windows)\n" + se[cols[:-1]].round(3).to_string())


if __name__ == "__main__":
    main()
