"""Conditional training and sampling: per-instrument flow time, conditioning_subset, flow_loss(p_cond), sample(given)."""
from __future__ import annotations

import torch
from einops import repeat

from kit2.model import conditioning_subset, flow_loss
from kit2.tests.test_model_causality import make, model  # noqa: F401  (synthetic batch, randomised tiny model)
from kit2.train.batch import model_inputs


def test_shared_flow_time_equals_its_per_instrument_copy(model):  # noqa: F811
    b, _ = make(0, n_valid=3)
    g = torch.Generator().manual_seed(0)
    z = torch.randn(b["x0"].shape, generator=g)
    t = torch.rand(b["x0"].shape[:2], generator=g)
    with torch.no_grad():
        v1 = model(model_inputs(b), z, t)
        v2 = model(model_inputs(b), z, repeat(t, "b a -> b a g", g=z.shape[2]))
    torch.testing.assert_close(v1, v2, rtol=0, atol=1e-6)


def test_conditioning_subset_rules():
    g = torch.Generator().manual_seed(0)
    B, A, G = 64, 8, 6
    valid = torch.ones(B, G, dtype=torch.bool)
    valid[:, 5] = False                                           # padding
    is_anchor = torch.zeros(B, G, dtype=torch.bool)
    is_anchor[: B // 2, :2] = True                                # half the groups have market anchors
    given = conditioning_subset(valid, is_anchor, A, 1.0, g)
    assert not (given & ~valid[:, None]).any()                    # never padding
    assert ((valid[:, None] & ~given).any(-1)).all()              # something is always left to generate
    anc_sets = (given == is_anchor[:, None]).all(-1)[: B // 2]
    assert 0.3 < anc_sets.float().mean() < 0.7                    # ~half of the anchored groups condition on them
    assert given.any(-1).float().mean() > 0.6
    assert not conditioning_subset(valid, is_anchor, A, 0.0, g).any()


def test_flow_loss_feeds_given_futures_clean_and_drops_their_loss():
    g = torch.Generator().manual_seed(1)
    B, A, G, h, C = 4, 8, 6, 2, 40
    x0 = torch.randn(B, A, G, h, C, generator=g)
    valid = torch.ones(B, G, dtype=torch.bool)
    seen = {}

    class Stub:
        dtype = torch.float32

        def __call__(self, b, z, t):
            given = t == 0
            seen["given"] = int(given.sum())
            assert torch.equal(z[given], x0[given])               # given instruments enter as their true future
            return torch.where(given[..., None, None], 1e6, 0.0) * torch.ones_like(z)

    loss = flow_loss(Stub(), {}, x0, valid, generator=g, p_cond=1.0)
    assert seen["given"] > 0 and float(loss) < 10.0               # 1e6 velocities of given instruments are not scored


def test_sample_ends_exactly_on_the_given_futures(model):  # noqa: F811
    b, _ = make(1, B=2, G=4)
    B, A, G = b["x0"].shape[:3]
    mask = torch.zeros(B, G, dtype=torch.bool)
    mask[:, 0] = True
    g = torch.Generator().manual_seed(2)
    for mode in ("clean", "replace"):
        out = model.sample(model_inputs(b), n_paths=3, steps=3, given=b["x0"], given_mask=mask, mode=mode,
                           generator=g)
        assert out.shape == (B, A, 3, G) + b["x0"].shape[3:]
        torch.testing.assert_close(out[:, :, :, 0], repeat(b["x0"][:, :, 0], "b a h c -> b a k h c", k=3))
        assert torch.isfinite(out).all() and (out[:, :, 0, 1:] != out[:, :, 1, 1:]).any()
    per_path = torch.randn((B, A, 3) + b["x0"].shape[2:], generator=g)   # one given future per path
    out = model.sample(model_inputs(b), n_paths=3, steps=3, given=per_path, given_mask=mask, generator=g)
    torch.testing.assert_close(out[:, :, :, 0], per_path[:, :, :, 0])


def test_log_likelihood_recovers_a_known_gaussian_density(model, monkeypatch):  # noqa: F811
    """Data x0 ~ N(0, s^2 I) has the exact rectified-flow velocity v(z, t) = (t - (1 - t) s^2) / s_t^2 z with
    s_t^2 = (1 - t)^2 s^2 + t^2. With it in place of the denoiser, the ODE likelihood must equal log N(x0; 0, s^2 I)
    per instrument, and a given instrument must get 0 without changing the others."""
    import math
    s = 0.5
    b, _ = make(2, B=2, G=4)

    def denoise(z, t, bb, cond, kv):
        st2 = (1 - t) ** 2 * s ** 2 + t ** 2
        return ((t - (1 - t) * s ** 2) / st2)[..., None, None] * z

    monkeypatch.setattr(model, "denoise", denoise)
    g = torch.Generator().manual_seed(3)
    x0 = s * torch.randn(b["x0"].shape, generator=g)
    D = x0.shape[3] * x0.shape[4]
    exact = (-0.5 * (x0 / s) ** 2).sum((3, 4)) - D * math.log(s) - 0.5 * D * math.log(2 * math.pi)
    ll = model.log_likelihood(model_inputs(b), x0, steps=32, n_probe=2, generator=g)
    torch.testing.assert_close(ll, exact, rtol=2e-3, atol=0.05)
    mask = torch.zeros(x0.shape[0], x0.shape[2], dtype=torch.bool)
    mask[:, 0] = True
    ll_c = model.log_likelihood(model_inputs(b), x0, steps=32, n_probe=2, given_mask=mask, generator=g)
    assert (ll_c[:, :, 0] == 0).all()
    torch.testing.assert_close(ll_c[:, :, 1:], exact[:, :, 1:], rtol=2e-3, atol=0.05)
