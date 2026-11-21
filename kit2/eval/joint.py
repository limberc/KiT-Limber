"""Joint scenario scoring: do the K sampled paths get the dependence between instruments right?

KiT-2 samples each group (market anchors + panel names + context fill) jointly: path k of every member is one draw of
the group's future. Per moment and group, the members' per-bar log returns over KiT's horizon are scored as one
multivariate forecast:
  corr RankIC   Spearman over member pairs of the forecast correlation (mean over paths of the per-path sample
                correlation of per-bar returns) against the realised correlation over the horizon
  corr bias     mean forecast minus mean realised correlation over the pairs
  ES / VS skill energy score / variogram score (p = 0.5) of the members' horizon returns, each member divided by
                its trailing vol * sqrt(lh) (fixed in advance, the same for every model)
  portfolios    N_PF random portfolios per group, long-short (unit gross) and long-only, each divided by its
                trailing vol: CRPS skill of the portfolio horizon return, and the share of realised returns below the
                forecast 5% quantile (nominal 0.05)
  hedge gain    each non-anchor member hedged with the anchor it is most correlated with on trailing bars (the same
                hedge for every model), beta read off the forecast; HV = realised mean square of the hedged per-bar
                returns over the horizon; gain = 1 - geometric mean over members of HV(beta) / HV(trailing OLS beta)
  GMV gain      global minimum-variance portfolio (shorts allowed) on the forecast's per-bar covariance; realised
                mean square of its per-bar returns against the same portfolio on the trailing Ledoit-Wolf covariance,
                gain = 1 - geometric mean over moments of the ratio (> 0: the forecast's portfolio is less risky)
Skills are 1 - mean score / mean score of indep over all groups and moments (a per-group ratio is unstable: the
portfolios of one group share one market outcome).
Baselines on the same members, with the same K:
  indep         K random trailing lh-bar blocks per member, drawn independently (score.py's bootstrap): no dependence
  sync          K random trailing blocks at the same (common) timestamps for every member: historical dependence
  lgbm_cop      LightGBM horizon mean and per-bar vol (pred/<lgbm>_<market>.npz) with i.i.d. Gaussian bars whose
                correlation is the Ledoit-Wolf shrunk correlation of the members' trailing per-bar returns - the
                "gradient boosting + simulation glue" null
Members missing bars, an aligned horizon or a LightGBM forecast are dropped for every model.

Usage: python -m kit2.eval.joint --market us --models kit2_mix [--lgbm lgbm]
"""
from __future__ import annotations

import argparse
import json
from functools import reduce
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from kit2.eval import OUT
from kit2.eval.score import crps, spearman

H_TRAIL = 2000    # trailing bars: bootstrap blocks, copula correlation, trailing beta, standardisation
N_PF = 64         # random portfolios per (group, kind)
METRICS = ["corr_rankic", "corr_bias", "es", "vs", "pf_ls_crps", "pf_ls_exc05", "pf_lo_crps", "pf_lo_exc05",
           "hedge_gain", "gmv_lr"]
SCORES = ["es", "vs", "pf_ls_crps", "pf_lo_crps"]   # lower is better; reported as skill against indep


# ---- scores -------------------------------------------------------------------------------------------------------
def path_corr(X: np.ndarray) -> np.ndarray:
    """X [M, K, n] -> mean over paths of the per-path sample correlation of per-bar returns [M, M] (nan if flat)."""
    Z = X - X.mean(2, keepdims=True)
    sd = np.sqrt((Z ** 2).sum(2))
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.einsum("ikt,jkt->ijk", Z, Z) / (sd[:, None] * sd[None, :])
    C[~np.isfinite(C)] = np.nan
    with np.errstate(invalid="ignore"):
        return np.nanmean(C, axis=2)


def energy_score(X: np.ndarray, y: np.ndarray) -> float:
    """X [K, M] samples, y [M] outcome."""
    return float(np.linalg.norm(X - y, axis=1).mean() - 0.5 * np.linalg.norm(X[:, None] - X[None], axis=2).mean())


def variogram_score(X: np.ndarray, y: np.ndarray, p: float = 0.5) -> float:
    """Scheuerer-Hamill variogram score of order p, unit weights. X [K, M], y [M]."""
    iu = np.triu_indices(len(y), 1)
    dy = np.abs(y[:, None] - y[None])[iu] ** p
    dx = (np.abs(X[:, :, None] - X[:, None, :]) ** p).mean(0)[iu]
    return float(((dy - dx) ** 2).sum())


def path_cov(X: np.ndarray) -> np.ndarray:
    """X [M, K, n] -> covariance of per-bar returns pooled over paths (each path demeaned) [M, M]."""
    Z = X - X.mean(2, keepdims=True)
    return np.einsum("ikt,jkt->ij", Z, Z) / (Z.shape[1] * (Z.shape[2] - 1))


def gmv_weights(S: np.ndarray) -> np.ndarray:
    """Global minimum-variance weights (shorts allowed, sum 1) of covariance S, with a small ridge."""
    S = S + 1e-6 * np.trace(S) / len(S) * np.eye(len(S))
    w = np.linalg.solve(S, np.ones(len(S)))
    return w / w.sum()


def path_beta(xi: np.ndarray, xh: np.ndarray) -> float:
    """Hedge ratio of member paths xi on hedge paths xh [K, n]: per-path demeaned covariance / variance, pooled."""
    a, b = xi - xi.mean(1, keepdims=True), xh - xh.mean(1, keepdims=True)
    return float((a * b).sum() / max(float((b * b).sum()), 1e-30))


def ledoit_wolf_corr(X: np.ndarray) -> np.ndarray:
    """Correlation matrix of X [T, M] with Ledoit-Wolf (2004) shrinkage of the standardised covariance toward a
    scaled identity."""
    X = X - X.mean(0)
    X = X / X.std(0)
    n, p = X.shape
    S = X.T @ X / n
    m = np.trace(S) / p
    d2 = ((S - m * np.eye(p)) ** 2).sum() / p
    b2 = min((((X ** 2).sum(1) ** 2).sum() - n * (S ** 2).sum()) / (n ** 2 * p), d2)
    delta = b2 / d2 if d2 > 0 else 1.0
    C = delta * m * np.eye(p) + (1 - delta) * S
    s = np.sqrt(np.diag(C))
    return C / s[:, None] / s[None]


# ---- baselines ----------------------------------------------------------------------------------------------------
def indep_paths(hists: List[np.ndarray], lh: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """[M, K, lh]: K blocks of each member's own trailing returns, independent across members."""
    out = []
    for h in hists:
        s = rng.integers(0, len(h) - lh, size=K)
        out.append(h[s[:, None] + np.arange(lh)])
    return np.stack(out)


def sync_paths(Rh: np.ndarray, lh: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """Rh [M, T] trailing returns on common timestamps -> [M, K, lh]: the same K blocks for every member."""
    s = rng.integers(0, Rh.shape[1] - lh, size=K)
    return Rh[:, s[:, None] + np.arange(lh)]


def copula_paths(mu: np.ndarray, sigma: np.ndarray, C: np.ndarray, lh: int, K: int,
                 rng: np.random.Generator) -> np.ndarray:
    """[M, K, lh] i.i.d. Gaussian bars with per-bar mean mu / lh, vol sigma and correlation C across members."""
    L = np.linalg.cholesky(C + 1e-9 * np.eye(len(C)))
    e = np.einsum("ij,jkt->ikt", L, rng.standard_normal((len(mu), K, lh)))
    return mu[:, None, None] / lh + sigma[:, None, None] * e


# ---- data ---------------------------------------------------------------------------------------------------------
def load_bars(market: str, ids) -> Dict[int, Dict[str, np.ndarray]]:
    """Bar files of the given instruments (those that exist): per scale s, s_ts and per-bar log returns s_ret."""
    bars = {}
    for i in ids:
        f = OUT / "bars" / market / f"{i}.npz"
        if f.is_file():
            with np.load(f) as z:
                bars[int(i)] = {k: z[k] for k in z.files if k.endswith("_ts")}
                bars[int(i)] |= {k[:-5] + "_ret": z[k][:, 0] + z[k][:, 1] for k in z.files if k.endswith("_feat")}
    return bars


def window(z: Dict[str, np.ndarray], scale: str, t_ns: int, lh: int):
    """Trailing (ts, returns) up to and including the bar closing at t_ns, and the next lh (ts, returns).
    z: one bars file with per-bar log returns `<scale>_ret` added."""
    ts = z[f"{scale}_ts"]
    i = int(np.searchsorted(ts, t_ns, side="right"))
    if i == 0 or ts[i - 1] != t_ns or i + lh > len(ts):
        return None
    r = z[f"{scale}_ret"]
    lo = max(0, i - H_TRAIL)
    return ts[lo:i], r[lo:i], ts[i: i + lh], r[i: i + lh]


def kit2_member_paths(p, mi: int, g: Optional[int], ids: List[int]) -> np.ndarray:
    """[M, K, lh] of group g's members from a kit2pred file (anchors from the group's own anchor paths). g None: the
    whole panel, anchors from group 0 - one joint draw per path only if the groups were coupled (--condition couple)."""
    anc = {int(a): j for j, a in enumerate(p[f"{mi}_anc_inst"])}
    inst, grp = p[f"{mi}_inst"], p[f"{mi}_grp"]
    pos = {int(i): j for j, i in enumerate(inst) if g is None or grp[j] == g}
    ga = 0 if g is None else g
    return np.stack([p[f"{mi}_anc_ret"][ga, anc[i]] if i in anc else p[f"{mi}_ret"][pos[i]] for i in ids])


def group_members(p, mi: int, g: Optional[int]) -> List[int]:
    """Market anchors first, then the panel names forecast in group g (g None: in any group)."""
    anc = [int(a) for a in p[f"{mi}_anc_inst"]]
    names = [int(i) for i, gg in zip(p[f"{mi}_inst"], p[f"{mi}_grp"]) if (g is None or gg == g) and int(i) not in anc]
    return anc + names


# ---- scoring ------------------------------------------------------------------------------------------------------
def score_group(paths: Dict[str, np.ndarray], Y: np.ndarray, Rh: np.ndarray, n_anc: int,
                rng: np.random.Generator) -> Dict[str, Dict[str, float]]:
    """paths: model -> [M, K, lh]; Y [M, lh] realised; Rh [M, T] trailing returns on common timestamps; the first
    n_anc members are market anchors. Returns model -> raw metrics (scores are standardised by trailing vol; skills
    are formed in `summarise` as ratios of means - the portfolios of one group share one market outcome, so a
    per-group ratio is dominated by groups whose outcome happens to be near zero)."""
    M, lh = Y.shape
    sd = Rh.std(1) * np.sqrt(lh)                          # standardisation of horizon returns (trailing, fixed)
    y = Y.sum(1)
    iu = np.triu_indices(M, 1)
    real_c = np.corrcoef(Y)[iu]
    W = {"ls": rng.standard_normal((N_PF, M)), "lo": rng.dirichlet(np.ones(M), N_PF)}
    W["ls"] /= np.abs(W["ls"]).sum(1, keepdims=True)
    cov_t = np.cov(Rh)
    sd_pf = {k: np.sqrt(np.einsum("pi,ij,pj->p", w, cov_t, w) * lh) for k, w in W.items()}
    # hedges: most correlated anchor on trailing bars; trailing OLS beta is the reference hedge
    Ct = np.corrcoef(Rh)
    hedge = {i: int(np.argmax(Ct[i, :n_anc])) for i in range(n_anc, M)} if n_anc else {}
    beta_t = {i: float(cov_t[i, h] / cov_t[h, h]) for i, h in hedge.items()}

    def hv(i: int, beta: float) -> float:
        return float(((Y[i] - beta * Y[hedge[i]]) ** 2).mean())

    # minimum-variance portfolio: reference = GMV on the trailing Ledoit-Wolf covariance
    sd_t = np.sqrt(np.diag(cov_t))
    rv_lw = float(((gmv_weights(ledoit_wolf_corr(Rh.T) * np.outer(sd_t, sd_t)) @ Y) ** 2).mean())

    out = {}
    for m, X in paths.items():
        R = X.sum(2)                                      # [M, K] horizon returns
        Xs = (R / sd[:, None]).T                          # [K, M] standardised
        fc = path_corr(X)[iu]
        ok = np.isfinite(fc) & np.isfinite(real_c)
        r = {"corr_rankic": spearman(fc[ok], real_c[ok]) if ok.sum() >= 5 else np.nan,
             "corr_bias": float(np.mean(fc[ok] - real_c[ok])) if ok.any() else np.nan,
             "es": energy_score(Xs, y / sd), "vs": variogram_score(Xs, y / sd)}
        for kind, w in W.items():
            xp, yp = (w @ R) / sd_pf[kind][:, None], (w @ y) / sd_pf[kind]   # [N_PF, K], [N_PF]
            r[f"pf_{kind}_crps"] = float(crps(xp, yp).mean())
            r[f"pf_{kind}_exc05"] = float((yp < np.quantile(xp, 0.05, axis=1)).mean())
        if hedge:
            lr = [np.log(max(hv(i, path_beta(X[i], X[h])), 1e-30) / max(hv(i, beta_t[i]), 1e-30))
                  for i, h in hedge.items()]
            r["hedge_gain"] = float(1 - np.exp(np.mean(lr)))
        else:
            r["hedge_gain"] = np.nan
        r["gmv_lr"] = float(np.log(max(((gmv_weights(path_cov(X)) @ Y) ** 2).mean(), 1e-30) / max(rv_lw, 1e-30)))
        out[m] = r
    return out


def score_market(market: str, models: List[str], lgbm: Optional[str], seed: int = 0,
                 universe: bool = False) -> pd.DataFrame:
    plan = json.loads((OUT / f"plan_{market}.json").read_text())
    bars = load_bars(market, set(plan["panel"]) | set(plan.get("anchors", [])))
    preds = {m: np.load(OUT / "pred" / f"{m}_{market}.npz") for m in models}
    lg = np.load(OUT / "pred" / f"{lgbm}_{market}.npz") if lgbm else None
    rng = np.random.default_rng(seed)
    rows = []
    for mi, mo in enumerate(plan["moments"]):
        s, lh = mo["scale"], mo["lh"]
        if not all(f"{mi}_grp" in preds[m].files for m in models):
            continue
        p0 = preds[models[0]]
        for m in models[1:]:   # one group layout (same plan and seed) so every model forecasts the same draws
            assert np.array_equal(preds[m][f"{mi}_inst"], p0[f"{mi}_inst"]) and np.array_equal(
                preds[m][f"{mi}_grp"], p0[f"{mi}_grp"]), f"{m}: different groups at moment {mi}"
        K = min(preds[m][f"{mi}_ret"].shape[1] for m in models)
        if lg is not None and f"{mi}_inst" in lg.files:
            lpos = {int(i): j for j, i in enumerate(lg[f"{mi}_inst"])}
            lret = lg[f"{mi}_ret"].astype(np.float64)
        else:
            lpos, lret = {}, None
        for g in ([None] if universe else [int(x) for x in np.unique(p0[f"{mi}_grp"])]):
            ids = [i for i in group_members(p0, mi, g) if i in bars and (lret is None or i in lpos)]
            wins = {i: w for i in ids if (w := window(bars[i], s, mo["t_ns"], lh)) is not None}
            ids = list(wins)
            if len(ids) < 4:
                continue
            ref_ts = wins[ids[0]][2]
            ids = [i for i in ids if np.array_equal(wins[i][2], ref_ts)]
            common = reduce(np.intersect1d, [wins[i][0] for i in ids])
            if len(common) < lh + 100:
                continue
            Rh = np.stack([wins[i][1][np.searchsorted(wins[i][0], common)] for i in ids])
            Y = np.stack([wins[i][3] for i in ids])
            live = (Rh.std(1) > 0) & (Y.std(1) > 0)        # flat-priced members carry no dependence
            ids = [i for i, ok in zip(ids, live) if ok]
            Rh, Y = Rh[live], Y[live]
            anc_set = {int(a) for a in p0[f"{mi}_anc_inst"]}
            n_anc = sum(i in anc_set for i in ids)
            if len(ids) - n_anc < 3:
                continue
            paths = {m: kit2_member_paths(preds[m], mi, g, ids)[:, :K].astype(np.float64) for m in models}
            paths["indep"] = indep_paths([wins[i][1] for i in ids], lh, K, rng)
            paths["sync"] = sync_paths(Rh, lh, K, rng)
            if lret is not None:
                L = lret[[lpos[i] for i in ids]]
                mu, sigma = L.sum(2).mean(1), L.std(2, ddof=1).mean(1)
                paths["lgbm_cop"] = copula_paths(mu, sigma, ledoit_wolf_corr(Rh.T), lh, K, rng)
            res = score_group(paths, Y, Rh, n_anc, rng)
            for m, r in res.items():
                rows.append({"market": market, "scale": s, "moment": mo["t"], "group": -1 if g is None else g,
                             "model": m, "members": len(ids), **{k: r[k] for k in METRICS}})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over moments (groups averaged within a moment first); scores turned into skills against indep as
    1 - mean score / mean indep score."""
    per = df.groupby(["market", "scale", "model", "moment"])[METRICS].mean()
    out = per.groupby(level=["market", "scale", "model"]).mean()
    for sc in SCORES:
        ref = out[sc].xs("indep", level="model")
        out[f"{sc}_skill"] = 1 - out[sc] / ref.reindex(out.index.droplevel("model")).to_numpy()
    out["gmv_gain"] = 1 - np.exp(out["gmv_lr"])
    out["moments"] = per.groupby(level=["market", "scale", "model"]).size()
    cols = ["corr_rankic", "corr_bias"] + [f"{sc}_skill" for sc in SCORES] + ["pf_ls_exc05", "pf_lo_exc05",
                                                                             "hedge_gain", "gmv_gain", "moments"]
    return out[cols].round(3)


def paired(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Model a minus model b per moment, mean / standard error over moments / wins. Scores enter as
    (score_b - score_a) / mean indep score, whose mean is skill_a - skill_b."""
    per = df.groupby(["market", "scale", "model", "moment"])[METRICS].mean()
    rows = []
    for (market, scale), g in per.groupby(level=["market", "scale"]):
        w = g.droplevel(["market", "scale"]).unstack("model")
        r = {"market": market, "scale": scale, "moments": len(w)}
        for met in ["corr_rankic", *SCORES, "hedge_gain", "gmv_lr"]:
            if met in SCORES:
                d, name = (w[(met, b)] - w[(met, a)]) / w[(met, "indep")].mean(), f"{met}_skill"
            elif met == "gmv_lr":                                 # log realised-variance ratio: lower is better
                d, name = w[(met, b)] - w[(met, a)], "gmv_dlog"
            else:
                d, name = w[(met, a)] - w[(met, b)], met
            d = d.dropna()
            r[f"d_{name}"] = d.mean()
            r[f"se_{name}"] = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
            r[f"wins_{name}"] = f"{int((d > 0).sum())}/{len(d)}"
        rows.append(r)
    return pd.DataFrame(rows).round(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="us,crypto")
    ap.add_argument("--models", default="kit2")
    ap.add_argument("--lgbm", default="lgbm", help="LightGBM forecast tag for the copula baseline ('' to skip)")
    ap.add_argument("--universe", action="store_true",
                    help="score each moment's whole panel as one forecast (needs coupled groups to be joint)")
    ap.add_argument("--out", default="joint", help="output file prefix")
    args = ap.parse_args()
    models = args.models.split(",")
    dfs = [score_market(m, models, args.lgbm or None, universe=args.universe) for m in args.markets.split(",")
           if (OUT / f"plan_{m}.json").is_file()]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(OUT / f"{args.out}_by_group.csv", index=False)
    s = summarise(df)
    s.to_csv(OUT / f"{args.out}_summary.csv")
    pd.set_option("display.width", 250)
    print(s.to_string())
    overall = summarise(df.assign(moment=df["scale"] + " " + df["moment"], scale="all"))
    print("\nall scales\n" + overall.to_string())
    for b in ("lgbm_cop", "sync", *models[1:]):
        if b in set(df["model"]):
            pr = paired(df, models[0], b)
            pr.to_csv(OUT / f"{args.out}_paired_{models[0]}_vs_{b}.csv", index=False)
            print(f"\n{models[0]} minus {b} (same moments)\n" + pr.to_string(index=False))


if __name__ == "__main__":
    main()
