"""Adaptive bucket mixture: UniMax prior, progress tilt, reuse cap, state restore, and that weight updates reach
the DataLoader workers; one short CPU fit on a synthetic store exercises the Lightning wiring."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from kit2.train.data import StoreData, _Stream
from kit2.train.mixture import AdaptiveMixture, Mixture, capped, unimax
from kit2.tests.test_train_data import root  # noqa: F401  (synthetic on-disk store fixture)

INF = np.inf


def test_unimax_even_when_no_cap_binds():
    np.testing.assert_allclose(unimax(np.full(4, INF), 100), 0.25)


def test_unimax_gives_what_a_capped_bucket_cannot_take_to_the_rest():
    np.testing.assert_allclose(unimax(np.array([10.0, 1000.0, 1000.0]), 300), np.array([10, 145, 145]) / 300)


def test_unimax_spends_short_capacity_in_proportion():
    np.testing.assert_allclose(unimax(np.array([10.0, 20.0]), 100), [1 / 3, 2 / 3])


def test_capped_moves_excess_to_uncapped_in_proportion():
    np.testing.assert_allclose(capped(np.array([0.7, 0.2, 0.1]), np.array([30.0, INF, INF]), 100),
                               [0.3, 0.7 * 2 / 3, 0.7 / 3])


def run_evals(mix, losses, per_eval=1000):
    """Feed validation losses [n_evals, K]; batches are drawn by the current probabilities."""
    drawn = np.zeros(len(mix.names))
    for loss in losses:
        drawn += mix.p * per_eval
        mix.update(np.asarray(loss), drawn, remaining=10 * per_eval)
    return mix.p


def test_progress_tilts_shares_and_overfitting_bucket_drops():
    mix = Mixture(["rising", "fast", "slow"], [1e6] * 3, [1.0] * 3, max_reuse=None, tilt=4.0)
    mix.start(10_000)
    t = np.arange(8)[:, None]
    losses = np.hstack([1.0 + 0.01 * t, 1.0 - 0.02 * t, 1.0 - 0.005 * t])
    p = run_evals(mix, losses)
    assert p[1] > p[2] > p[0]
    assert p.max() / p.min() <= 4.0 ** 2 + 1e-9                  # each share within prior x tilt^(+-1)


def test_loss_level_does_not_set_the_share():
    mix = Mixture(["fat_tails", "quiet"], [1e6] * 2, [1.0] * 2, max_reuse=None)
    mix.start(10_000)
    t = np.arange(6)[:, None]
    p = run_evals(mix, np.hstack([1.35 - 0.01 * t, 0.40 - 0.01 * t]))
    np.testing.assert_allclose(p, [0.5, 0.5], atol=1e-6)


def test_reuse_cap_is_a_backstop():
    # bucket 0 is tiny: 1000 bars, 100 uses per batch -> at most 128 x 1000 / 100 = 1280 batches over the run
    mix = Mixture(["tiny", "big"], [1e3, 1e9], [100.0, 100.0], max_reuse=128.0)
    mix.start(20_000)
    drawn = np.zeros(2)
    for step in range(0, 20_000, 1000):
        drawn += mix.p * 1000
        loss = np.array([1.0 - 0.05 * step / 1000, 1.0 - 0.001 * step / 1000])   # tiny keeps "improving"
        mix.update(loss, drawn, remaining=20_000 - step - 1000)
    assert mix.reuse()[0] <= 128.0 + 1e-6


def test_state_restores_by_bucket_name():
    a = Mixture(["x", "y", "z"], [1e6] * 3, [1.0] * 3)
    a.start(5000)
    run_evals(a, [[1.0, 1.0, 1.0], [0.9, 1.0, 1.1], [0.8, 1.0, 1.2]])
    b = Mixture(["z", "x", "new"], [1e6] * 3, [1.0] * 3)
    b.load_state_dict(a.state_dict())
    assert b.drawn[0] == a.drawn[2] and b.progress[1] == a.progress[0] and b.drawn[2] == 0
    assert b.evals == a.evals


class _Fake:
    def sample(self, k, rng):
        return {"bucket": np.int64(k)}


def _stream(w):
    ds = _Stream({}, w, seed=0, n_batches=None)
    ds.sampler = _Fake()
    return ds


def test_stream_reads_updated_weights_in_process():
    w = torch.tensor([1.0, 0.0]).share_memory_()
    it = iter(_stream(w))
    assert {int(next(it)["bucket"]) for _ in range(50)} == {0}
    w.copy_(torch.tensor([0.0, 1.0]))
    assert {int(next(it)["bucket"]) for _ in range(50)} == {1}


def test_weight_updates_reach_persistent_workers():
    w = torch.tensor([1.0, 0.0], dtype=torch.float64).share_memory_()
    workers, prefetch = 2, 2
    dl = DataLoader(_stream(w), batch_size=None, num_workers=workers, persistent_workers=True,
                    prefetch_factor=prefetch)
    it = iter(dl)
    assert {int(next(it)["bucket"]) for _ in range(40)} == {0}
    w.copy_(torch.tensor([0.0, 1.0], dtype=torch.float64))
    for _ in range(workers * prefetch):                       # drawn before the update
        next(it)
    assert {int(next(it)["bucket"]) for _ in range(40)} == {1}


def test_fit_updates_the_mixture(root):  # noqa: F811
    import lightning as L
    from kit2.train.lit import KiT2Lit
    dm = StoreData(str(root), ["us:1h", "us:1d"], G=8, A=4, batch_tokens=8 * 112 * 2, num_workers=0,
                   val_batches_per_bucket=1)
    lit = KiT2Lit({"preset": "tiny"}, {"lr": 1e-3, "warmup_steps": 1, "total_steps": 8}, compile=False)
    cb = AdaptiveMixture(max_reuse=None, warmup_evals=1)
    tr = L.Trainer(accelerator="cpu", max_steps=8, val_check_interval=2, check_val_every_n_epoch=None,
                   logger=False, enable_checkpointing=False, enable_progress_bar=False, num_sanity_val_steps=0,
                   callbacks=[cb])
    tr.fit(lit, dm)
    assert float(cb._counts.sum()) == tr.global_step == 8 and cb.mix.evals >= 2
    np.testing.assert_allclose(dm.train_weights.numpy(), cb.mix.p)
    state = cb.state_dict()
    assert sum(state["drawn"]) == 8
    restored = AdaptiveMixture(max_reuse=None)
    restored.load_state_dict(state)
    assert restored._restored["drawn"] == state["drawn"]
