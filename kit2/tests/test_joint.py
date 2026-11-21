"""Joint scenario scoring (kit2.eval.joint) and the group layout kit2pred stores for it."""
from __future__ import annotations

import numpy as np

from kit2.eval.joint import (copula_paths, energy_score, indep_paths, ledoit_wolf_corr, path_beta, path_corr,
                             score_group, sync_paths, variogram_score)
from kit2.eval.kit2pred import group_layout


def correlated(rng, M: int, n: int, rho: float) -> np.ndarray:
    """[M, n] standard normal series with equal pairwise correlation rho."""
    return np.sqrt(rho) * rng.standard_normal(n) + np.sqrt(1 - rho) * rng.standard_normal((M, n))


def test_group_layout_scores_each_panel_name_once_and_finds_anchors():
    anchors = np.array([100, 101])
    inst = np.array([[100, 101, 1, 2, 50, 51],             # group 0: anchors, panel 1 2, context 50 51
                     [101, 100, 3, 4, 52, -1]])            # group 1: anchors in another order, padding
    keep, grp, anc, anc_pos = group_layout(inst, np.array([1, 2, 3, 4, 100]), anchors)
    flat = inst.reshape(-1)
    assert sorted(flat[keep].tolist()) == [1, 2, 3, 4, 100]  # the anchor in the panel once, from group 0
    assert grp.tolist() == [0, 0, 0, 1, 1]
    assert anc.tolist() == [100, 101]
    assert anc_pos.tolist() == [[0, 1], [1, 0]]


def test_ledoit_wolf_recovers_equicorrelation():
    rng = np.random.default_rng(0)
    C = ledoit_wolf_corr(correlated(rng, 6, 5000, 0.6).T)
    off = C[np.triu_indices(6, 1)]
    assert np.allclose(np.diag(C), 1.0) and abs(off.mean() - 0.6) < 0.03


def test_path_corr_and_beta_recover_the_generating_dependence():
    rng = np.random.default_rng(1)
    xh = rng.standard_normal((64, 50))
    xi = 0.7 * xh + 0.3 * rng.standard_normal((64, 50))
    assert abs(path_beta(xi, xh) - 0.7) < 0.03
    X = np.stack([xi, xh])
    rho = 0.7 / np.sqrt(0.49 + 0.09)
    assert abs(path_corr(X)[0, 1] - rho) < 0.03
    assert np.isnan(path_corr(np.stack([xi, np.zeros_like(xh)]))[0, 1])   # a flat member has no correlation


def test_sync_bootstrap_keeps_history_dependence_and_indep_drops_it():
    rng = np.random.default_rng(2)
    Rh = correlated(rng, 5, 2000, 0.8)
    s = path_corr(sync_paths(Rh, 20, 128, rng))[np.triu_indices(5, 1)]
    i = path_corr(indep_paths(list(Rh), 20, 128, rng))[np.triu_indices(5, 1)]
    assert abs(s.mean() - 0.8) < 0.05 and abs(i.mean()) < 0.05


def test_copula_paths_have_the_requested_vol_and_correlation():
    rng = np.random.default_rng(3)
    C = np.array([[1.0, 0.5], [0.5, 1.0]])
    mu, sigma, lh, K = np.array([0.02, -0.01]), np.array([0.01, 0.03]), 40, 512
    X = copula_paths(mu, sigma, C, lh, K, rng)
    assert np.allclose(X.std(2, ddof=1).mean(1), sigma, rtol=0.03)
    assert (np.abs(X.sum(2).mean(1) - mu) < 4 * sigma * np.sqrt(lh / K)).all()   # within 4 s.e.
    assert abs(path_corr(X)[0, 1] - 0.5) < 0.03


def test_dependence_aware_scores_prefer_the_true_joint_distribution():
    """Same marginals, with or without the true correlation: VS must prefer the true joint on average (ES too)."""
    rng = np.random.default_rng(4)
    vs_true, vs_ind, es_true, es_ind = [], [], [], []
    for _ in range(300):
        y = correlated(rng, 8, 1, 0.8)[:, 0]
        X_true = correlated(rng, 8, 128, 0.8).T
        X_ind = rng.standard_normal((128, 8))
        vs_true.append(variogram_score(X_true, y)), vs_ind.append(variogram_score(X_ind, y))
        es_true.append(energy_score(X_true, y)), es_ind.append(energy_score(X_ind, y))
    assert np.mean(vs_true) < 0.8 * np.mean(vs_ind)
    assert np.mean(es_true) < np.mean(es_ind)


def test_score_group_rewards_correct_dependence_and_hedges():
    """A forecaster with the true correlation beats indep on VS, ES, portfolios and hedging, judged as in
    `summarise`: mean score over replicates (the portfolios of one replicate share one outcome)."""
    rng = np.random.default_rng(5)
    M, lh, K, rho = 10, 40, 128, 0.7
    rows = []
    for _ in range(80):
        Rh = 0.01 * correlated(rng, M, 1000, rho)
        Y = 0.01 * correlated(rng, M, lh, rho)
        true = 0.01 * np.stack([correlated(rng, M, lh, rho) for _ in range(K)], axis=1)   # [M, K, lh]
        rows.append(score_group({"true": true, "indep": indep_paths(list(Rh), lh, K, rng)}, Y, Rh, 1, rng))
    mean = {m: {k: np.mean([r[m][k] for r in rows]) for k in rows[0][m]} for m in ("true", "indep")}
    for sc in ("es", "vs", "pf_ls_crps", "pf_lo_crps"):
        assert mean["true"][sc] < mean["indep"][sc], (sc, mean)
    assert mean["true"]["hedge_gain"] > mean["indep"]["hedge_gain"] + 0.3
    assert abs(mean["true"]["corr_bias"]) < 0.05 and mean["indep"]["corr_bias"] < -0.5
    assert abs(mean["true"]["pf_lo_exc05"] - 0.05) < abs(mean["indep"]["pf_lo_exc05"] - 0.05)


def test_conditional_copula_follows_the_given_anchor_path():
    from kit2.eval.cond import conditional_paths
    rng = np.random.default_rng(6)
    rho, s_a, s_i, lh, K = 0.6, 0.01, 0.02, 30, 4096
    C = np.array([[1.0, rho], [rho, 1.0]])
    y_anc = rng.normal(0, s_a, (1, lh))
    X = conditional_paths(np.array([0.0, 0.0]), np.array([s_a, s_i]), C, y_anc, K, rng)
    np.testing.assert_allclose(X.mean(0), rho * s_i / s_a * y_anc[0], atol=4 * s_i / np.sqrt(K))
    np.testing.assert_allclose(X.std(0).mean(), s_i * np.sqrt(1 - rho ** 2), rtol=0.03)
