"""Score KiT-2, KiT and a historical-bootstrap baseline on the same (instrument, moment) pairs.

Per moment (cross-section of instruments), forecasts are read off K sampled paths of per-bar log returns
over KiT's horizon (lh bars after the moment); actuals are the realised per-bar log returns:
  return RankIC     Spearman(mean_k sum(path), realised horizon return)
  vol RankIC        Spearman(mean_k std(path), realised std of per-bar returns)
  CRPS skill        1 - CRPS(model) / CRPS(bootstrap) of the horizon return (ensemble CRPS, per instrument)
  coverage 80%      share of realised horizon returns inside the paths' 10-90% range
Bootstrap baseline: K random contiguous lh-bar blocks of the instrument's own trailing history (last 2000 bars,
ending at the moment) - a no-model forecast with realistic tails and the instrument's own volatility.

Usage: python -m kit2.eval.score [--markets us,crypto] [--models kit2,kit]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from kit2.eval import OUT


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5:
        return float("nan")
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def crps(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ensemble CRPS per row: samples [N, K], y [N]."""
    t1 = np.abs(samples - y[:, None]).mean(1)
    t2 = np.abs(samples[:, :, None] - samples[:, None, :]).mean((1, 2))
    return t1 - 0.5 * t2


def actual(z, scale: str, t_ns: int, lh: int):
    ts = z[f"{scale}_ts"]
    i = int(np.searchsorted(ts, t_ns, side="right"))
    if i == 0 or ts[i - 1] != t_ns or i + lh > len(ts):
        return None, None
    f = z[f"{scale}_feat"]
    fut = f[i: i + lh, 0] + f[i: i + lh, 1]
    hist = f[max(0, i - 2000): i, 0] + f[max(0, i - 2000): i, 1]
    return fut, hist


def bootstrap(hist: np.ndarray, lh: int, K: int, rng) -> np.ndarray:
    starts = rng.integers(0, len(hist) - lh, size=K)
    return np.stack([hist[s: s + lh] for s in starts])  # [K, lh]


def score_market(market: str, models, K: int = 16, seed: int = 0) -> pd.DataFrame:
    plan = json.loads((OUT / f"plan_{market}.json").read_text())
    bars = {i: np.load(OUT / "bars" / market / f"{i}.npz") for i in plan["panel"]}
    preds = {m: np.load(OUT / "pred" / f"{m}_{market}.npz") for m in models}
    rng = np.random.default_rng(seed)
    rows = []
    for mi, mo in enumerate(plan["moments"]):
        s, lh = mo["scale"], mo["lh"]
        if not all(f"{mi}_inst" in preds[m].files for m in models):
            continue  # a model has not forecast this moment yet
        common = None
        for m in models:
            ids = set(preds[m][f"{mi}_inst"].tolist())
            common = ids if common is None else common & ids
        acts, hists, keep = [], [], []
        for i in sorted(common):
            fut, hist = actual(bars[i], s, mo["t_ns"], lh)
            if fut is not None and len(hist) > lh + 10:
                keep.append(i), acts.append(fut), hists.append(hist)
        if len(keep) < 10:
            continue
        A = np.stack(acts)  # [N, lh]
        y_ret, y_vol = A.sum(1), A.std(1, ddof=1)
        paths = {"bootstrap": np.stack([bootstrap(h, lh, K, rng) for h in hists])}
        for m in models:
            p = preds[m]
            pos = {int(v): j for j, v in enumerate(p[f"{mi}_inst"])}
            paths[m] = p[f"{mi}_ret"][[pos[i] for i in keep], :K, :lh].astype(np.float64)
        base_crps = crps(paths["bootstrap"].sum(2), y_ret)
        for m, P in paths.items():
            r = P.sum(2)  # [N, K] horizon returns
            q10, q90 = np.quantile(r, 0.1, axis=1), np.quantile(r, 0.9, axis=1)
            c = crps(r, y_ret)
            rows.append(
                {"market": market, "scale": s, "moment": mo["t"], "model": m, "n": len(keep),
                 "ret_rankic": spearman(r.mean(1), y_ret),
                 "vol_rankic": spearman(P.std(2, ddof=1).mean(1), y_vol),
                 "crps_skill": float(1 - c.mean() / base_crps.mean()),
                 "cov80": float(((y_ret >= q10) & (y_ret <= q90)).mean())}
            )
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["market", "scale", "model"])
    out = g[["ret_rankic", "vol_rankic", "crps_skill", "cov80"]].mean()
    out["ret_rankic_se"] = g["ret_rankic"].std() / np.sqrt(g.size())
    out["moments"] = g.size()
    out["n_avg"] = g["n"].mean().round(0)
    return out.round(3)


def paired(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Model a minus model b on the same moments: mean difference, standard error, wins."""
    rows = []
    keys = ["market", "scale", "moment"]
    w = df.pivot_table(index=keys, columns="model", values=["ret_rankic", "vol_rankic", "crps_skill"])
    for (market, scale), g in w.groupby(level=["market", "scale"]):
        r = {"market": market, "scale": scale, "moments": len(g)}
        for met in ("ret_rankic", "vol_rankic", "crps_skill"):
            d = (g[(met, a)] - g[(met, b)]).dropna()
            r[f"d_{met}"] = d.mean()
            r[f"se_{met}"] = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
            r[f"wins_{met}"] = f"{int((d > 0).sum())}/{len(d)}"
        rows.append(r)
    return pd.DataFrame(rows).round(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="us,crypto")
    ap.add_argument("--models", default="kit2,kit")
    args = ap.parse_args()
    models = args.models.split(",")
    dfs = []
    for m in args.markets.split(","):
        if (OUT / f"plan_{m}.json").is_file() and all((OUT / "pred" / f"{x}_{m}.npz").is_file() for x in models):
            dfs.append(score_market(m, models))
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(OUT / "scores_by_moment.csv", index=False)
    s = summarise(df)
    s.to_csv(OUT / "scores_summary.csv")
    pd.set_option("display.width", 200)
    print(s.to_string())
    overall = df.groupby(["market", "model"])[["ret_rankic", "vol_rankic", "crps_skill", "cov80"]].mean().round(3)
    print("\nall scales\n" + overall.to_string())
    if len(models) >= 2:
        pr = paired(df, models[0], models[1])
        pr.to_csv(OUT / f"paired_{models[0]}_vs_{models[1]}.csv", index=False)
        print(f"\n{models[0]} minus {models[1]} (same moments)\n" + pr.to_string(index=False))


if __name__ == "__main__":
    main()
