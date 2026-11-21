"""Surprise evaluation helpers (kit2.eval.surprise): corruptions keep per-bar values, the baseline is blind to order."""
from __future__ import annotations

import torch

from kit2.eval.surprise import BODY, VOL, corrupt, log_jacobian, trail_gauss


def setup(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 5, 24, 5, generator=g)
    panel = torch.tensor([[True, True, True, False, False], [False, True, True, True, True]])
    return x, panel, g


def test_corruptions_touch_only_panel_names_and_keep_bar_values():
    x, panel, g = setup()
    for kind in ("shuffle", "decouple", "swap", "spike"):
        y = corrupt(x, panel, kind, g)
        assert torch.equal(y[~panel], x[~panel]), kind                       # context / anchors untouched
        assert not torch.equal(y[panel], x[panel]), kind
    y = corrupt(x, panel, "shuffle", g)                                       # same bars, other order
    torch.testing.assert_close(y[panel].sort(dim=1).values, x[panel].sort(dim=1).values)
    y = corrupt(x, panel, "decouple", g)                                      # only volume moves
    assert torch.equal(y[..., :VOL], x[..., :VOL])
    torch.testing.assert_close(y[panel][..., VOL].sort(-1).values, x[panel][..., VOL].sort(-1).values)
    y = corrupt(x, panel, "spike", g)                                         # exactly one body per name changed
    assert ((y != x).sum((-1, -2))[panel] == 1).all() and ((y - x)[..., BODY].abs().amax(-1)[panel] > 0).all()


def test_baseline_is_blind_to_shuffle_and_decouple_but_not_to_spikes():
    x, panel, g = setup(1)
    hist = torch.randn(2, 5, 80, 5, generator=g)
    real = trail_gauss(hist, x)
    for kind in ("shuffle", "decouple"):
        torch.testing.assert_close(trail_gauss(hist, corrupt(x, panel, kind, g)), real)
    assert (trail_gauss(hist, corrupt(x, panel, "spike", g))[panel] < real[panel]).all()


def test_log_jacobian_penalises_values_near_the_soft_clip():
    y = torch.zeros(1, 3, 5)
    near = y.clone()
    near[0, 0, 0] = 5.9
    zero = torch.zeros(3, 5)
    assert float(log_jacobian(y, zero)) == 0.0 and float(log_jacobian(near, zero)) < -3.0
