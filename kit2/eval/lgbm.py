"""LightGBM baseline on the evaluation plan: gradient-boosted trees on hand-made history features.

Per (market, scale): two LGBMRegressors - horizon return (sum of the next lh per-bar log returns) and log realised vol
(log std of those returns) - fit on random (instrument, time) samples from the training store whose targets close on
or before TRAIN_END; samples with targets in the last two training months are held out for early stopping. Features
use only the H = 4 lh target-scale bars up to the forecast time: trailing log vol and return over lh/4 .. 4 lh, wick,
volume and observed-bar means, the last bar's return, the share of variance in session gaps, and the calendar
features of the forecast bar (time of day / week, session position; no day of year).

Forecasts are written as K paths in the kit2pred format, so kit2.eval.score scores them like the other models:
path k = (mu + c_k) / lh + sigma e_k, with e_k mean-zero, unit-std noise (each path's per-bar std is exactly sigma) and
c_k = sigma sqrt(lh) z_k at normal quantiles z_k. Return and vol RankIC therefore use mu and sigma directly; CRPS and
coverage assume a Gaussian horizon return with i.i.d. bars.

Usage: python -m kit2.eval.lgbm --market us [--samples 150000]
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, Tuple  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from kit2.data import MarketStore  # noqa: E402
from kit2.data.store import CAL_FIRST, SOFT_CLIP, TRAIN_END  # noqa: E402
from kit2.env import data_root  # noqa: E402
from kit2.eval import EVAL_SCALES, KIT_LH, OUT  # noqa: E402

ES_START = pd.Timestamp("2026-04-30 23:59:59")   # targets after this (and <= TRAIN_END): early-stopping set
CAL_COLS = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11]      # calendar features without day of year
FLOOR = 1e-6                                     # per-bar vol floor (flat-priced windows)
PARAMS = dict(n_estimators=3000, learning_rate=0.03, num_leaves=63, min_child_samples=500, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=8, verbose=-1)


def raw_returns(X: np.ndarray, gidx: np.ndarray, first_of: np.ndarray, stats: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Normalised windows X [N, n, C] on grid positions gidx [N, n] -> raw r_gap and r_gap + r_body [N, n]."""
    def inv(y, s):
        return np.arctanh(np.clip(y.astype(np.float64) / SOFT_CLIP, -(1 - 1e-6), 1 - 1e-6)) * SOFT_CLIP * s
    s_gap = np.where(first_of[gidx], stats["gap_first_scale"], stats["scale"][0])
    gap = inv(X[..., 0], s_gap)
    return gap, gap + inv(X[..., 1], stats["scale"][1])


def features(X: np.ndarray, gidx: np.ndarray, first_of: np.ndarray, cal: np.ndarray, stats: dict,
             lh: int) -> np.ndarray:
    """History windows X [N, H, C] (last bar = forecast time) -> features [N, F]."""
    gap, r = raw_returns(X, gidx, first_of, stats)
    H = X.shape[1]
    cols = []
    for w in sorted({max(2, lh // 4), max(2, lh // 2), lh, min(H, 2 * lh), H}):
        cols += [np.log(np.maximum(r[:, -w:].std(1, ddof=1), FLOOR)), r[:, -w:].sum(1)]
    up, dn, v, has = (X[:, -lh:, c].astype(np.float64) for c in (2, 3, 4, 5))
    cols += [up.mean(1), dn.mean(1), v.mean(1), X[:, -1, 4].astype(np.float64), has.mean(1),
             v.mean(1) - X[:, :, 4].astype(np.float64).mean(1), r[:, -1], np.abs(r[:, -1]),
             (gap ** 2).sum(1) / np.maximum((r ** 2).sum(1), FLOOR ** 2)]
    return np.column_stack(cols + [cal[gidx[:, -1]][:, CAL_COLS].astype(np.float64)])


def targets(X: np.ndarray, gidx: np.ndarray, first_of: np.ndarray, stats: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Future windows [N, lh, C] -> horizon return and log realised vol."""
    _, r = raw_returns(X, gidx, first_of, stats)
    return r.sum(1), np.log(np.maximum(r.std(1, ddof=1), FLOOR))


def sample(sc, H: int, lh: int, lo_ns: int, hi_ns: int, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """n (store row, grid index) of forecast bars with H history bars and lh target bars in one run, the first target
    bar closing after lo_ns and the last on or before hi_ns."""
    room = np.maximum(sc.r_len - H - lh + 1, 0)
    out_off, out_g = [], []
    while sum(len(o) for o in out_off) < n:
        k = rng.choice(len(room), size=2 * n, p=room / room.sum())
        j = (rng.random(2 * n) * room[k]).astype(np.int64)
        g = sc.r_g0[k] + H - 1 + j
        ok = (sc.grid[g + 1] > lo_ns) & (sc.grid[g + lh] <= hi_ns)
        out_off.append(sc.r_off[k][ok] + H - 1 + j[ok]), out_g.append(g[ok])
    return np.concatenate(out_off)[:n], np.concatenate(out_g)[:n]


def build(sc, stats: dict, off: np.ndarray, g: np.ndarray, H: int, lh: int, chunk: int = 20000):
    """Features and targets for forecast bars at store rows off / grid positions g (chunked: windows are large)."""
    first_of = sc.cal[:, CAL_FIRST] > 0.5
    F, R, V = [], [], []
    hist, fut = np.arange(-H + 1, 1), np.arange(1, lh + 1)
    for a in range(0, len(off), chunk):
        o, gg = off[a: a + chunk, None], g[a: a + chunk, None]
        F.append(features(sc.X[o + hist], gg + hist, first_of, sc.cal, stats, lh))
        ret, vol = targets(sc.X[o + fut], gg + fut, first_of, stats)
        R.append(ret), V.append(vol)
    return np.concatenate(F), np.concatenate(R), np.concatenate(V)


def fit(F: np.ndarray, y: np.ndarray, es: np.ndarray, clip: bool) -> lgb.LGBMRegressor:
    if clip:   # heavy-tailed horizon returns: winsorise at the fit set's 1st / 99th percentiles
        lo, hi = np.quantile(y[~es], [0.01, 0.99])
        y = np.clip(y, lo, hi)
    m = lgb.LGBMRegressor(**PARAMS, random_state=0)
    m.fit(F[~es], y[~es], eval_set=[(F[es], y[es])], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def gaussian_paths(mu: np.ndarray, sigma: np.ndarray, lh: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """[N, K, lh] per-bar paths: horizon sums mu + sigma sqrt(lh) z_k, per-path std (ddof 1) exactly sigma."""
    e = rng.standard_normal((len(mu), K, lh))
    e -= e.mean(2, keepdims=True)
    e /= e.std(2, ddof=1, keepdims=True)
    c = sigma[:, None] * np.sqrt(lh) * norm.ppf((np.arange(K) + 0.5) / K)[None, :]
    return (mu[:, None, None] + c[:, :, None]) / lh + sigma[:, None, None] * e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True)
    ap.add_argument("--samples", type=int, default=150_000)
    ap.add_argument("--paths", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", type=Path, default=data_root())
    ap.add_argument("--tag", default="lgbm")
    args = ap.parse_args()
    plan = json.loads((OUT / f"plan_{args.market}.json").read_text())
    panel = np.array(plan["panel"], dtype=np.int64)
    st = MarketStore(args.root, args.market, list(EVAL_SCALES))
    rng = np.random.default_rng(args.seed)
    models: Dict[str, tuple] = {}
    for s in EVAL_SCALES:
        t0 = time.time()
        sc, stats, lh = st.scales[s], st.stats["scales"][s], KIT_LH[s]
        H = 4 * lh
        off, g = sample(sc, H, lh, np.iinfo(np.int64).min, TRAIN_END.value, args.samples, rng)
        F, R, V = build(sc, stats, off, g, H, lh)
        es = sc.grid[g + 1] > ES_START.value
        keep = es | (sc.grid[g + lh] <= ES_START.value)       # drop samples whose targets straddle the split
        F, R, V, es = F[keep], R[keep], V[keep], es[keep]
        m_ret, m_vol = fit(F, R, es, clip=True), fit(F, V, es, clip=False)
        models[s] = (m_ret, m_vol)
        print(f"{args.market} {s}: fit {int((~es).sum())} / early-stop {int(es.sum())} samples, trees ret "
              f"{m_ret.best_iteration_} vol {m_vol.best_iteration_}, {time.time() - t0:.0f}s", flush=True)
    res, t0 = {}, time.time()
    for mi, mo in enumerate(plan["moments"]):
        s, lh = mo["scale"], mo["lh"]
        sc, stats, H = st.scales[s], st.stats["scales"][s], 4 * mo["lh"]
        off = sc.locate(panel, np.full(len(panel), mo["t_idx"] - H + 1), H)
        ok = off >= 0
        g = np.full(int(ok.sum()), mo["t_idx"])
        F = features(sc.X[(off[ok] + H - 1)[:, None] + np.arange(-H + 1, 1)], g[:, None] + np.arange(-H + 1, 1),
                     sc.cal[:, CAL_FIRST] > 0.5, sc.cal, stats, lh)
        m_ret, m_vol = models[s]
        mu, sigma = m_ret.predict(F), np.exp(m_vol.predict(F))
        res[f"{mi}_inst"] = panel[ok]
        res[f"{mi}_ret"] = gaussian_paths(mu, sigma, lh, args.paths, rng).astype(np.float32)
    el = time.time() - t0
    print(f"inference: {len(plan['moments'])} moments x {len(panel)} instruments in {el:.2f}s (CPU)", flush=True)
    out = OUT / "pred"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{args.tag}_{args.market}.npz", **res)


if __name__ == "__main__":
    main()
