"""Model-level no-look-ahead tests (spec: 2026-11-19-kit2-model-design.md).

1. Truncation equivalence: the velocity for anchor p computed from the full training window equals the one
   computed from a window that only contains history tokens closing at or before the anchor.
2. Zero gradient: the forecast for anchor p has exactly zero gradient w.r.t. every history input whose token
   closes after the anchor, and non-zero gradient w.r.t. earlier ones.
3. Batch assembly: targets are the bars after the anchor, positions and anchor times are consistent.
"""
from __future__ import annotations

import pytest
import torch

from kit2.model import KiT2
from kit2.model.config import preset
from kit2.train.batch import assemble, audit, model_inputs, synthetic_meta, synthetic_windows

CFG = preset("tiny")
N_PATCHES, STEPS = [16, 6, 6], [60, 300, 900]      # target 1m + 5m + 15m history, US-like sessions
H = 3


def make(seed=0, B=2, G=4, n_valid=None):
    g = torch.Generator().manual_seed(seed)
    wins = synthetic_windows(CFG, B, G, N_PATCHES, STEPS, generator=g)
    anchors = torch.tensor([[7, 10, 12]] * B)
    b = assemble(wins, anchors, H, synthetic_meta(B, G, n_valid=n_valid), CFG)
    return b, wins


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = KiT2(CFG).eval()
    for p in m.parameters():                         # zero-initialised heads would make every test trivial
        if p.dim() >= 1 and float(p.detach().abs().sum()) == 0:
            torch.nn.init.normal_(p, std=0.05)
    return m


def _truncate(b, bi, a):
    """Single-sample batch for anchor a of sample bi, keeping only history tokens closing <= the anchor."""
    keep = b["tok_close"][bi] <= b["anchor_close"][bi, a]
    one = {k: v[bi:bi + 1] for k, v in b.items() if k != "tok_scale"}
    one["tok_scale"] = b["tok_scale"][keep]
    for k in ("x_hist",):
        one[k] = one[k][:, :, keep]
    for k in ("tok_close", "tok_pos", "tok_cal"):
        one[k] = one[k][:, keep]
    for k in ("anchor_close", "anchor_pos", "fut_cal", "x0", "fut_ts"):
        one[k] = one[k][:, a:a + 1]
    return one, keep


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_truncation_equivalence(model, seed):
    b, _ = make(seed, n_valid=3)
    B, A, G = b["x0"].shape[:3]
    g = torch.Generator().manual_seed(100 + seed)
    z = torch.randn(b["x0"].shape, generator=g)
    t = torch.rand(B, A, generator=g)
    with torch.no_grad():
        full = model(model_inputs(b), z, t)
        for bi in range(B):
            for a in range(A):
                one, keep = _truncate(b, bi, a)
                assert keep.sum() < keep.numel()     # something was actually dropped
                part = model(model_inputs(one), z[bi:bi + 1, a:a + 1], t[bi:bi + 1, a:a + 1])
                torch.testing.assert_close(part[0, 0], full[bi, a], rtol=1e-4, atol=1e-5)


def test_zero_gradient_after_anchor(model):
    b, _ = make(3)
    x = b["x_hist"].clone().requires_grad_(True)
    inp = {**model_inputs(b), "x_hist": x}
    z = torch.randn(b["x0"].shape)
    t = torch.full(b["anchor_close"].shape, 0.5)
    v = model(inp, z, t)
    for bi in range(v.shape[0]):
        for a in range(v.shape[1]):
            (grad,) = torch.autograd.grad(v[bi, a].sum(), x, retain_graph=True)
            g_tok = grad[bi].abs().sum(dim=(0, 2))                               # [S]
            late = b["tok_close"][bi] > b["anchor_close"][bi, a]
            assert late.any() and (~late).any()
            assert torch.all(g_tok[late] == 0), "forecast depends on a token that closes after the anchor"
            assert torch.all(g_tok[~late] > 0)
            assert torch.all(grad[:bi].abs().sum() == 0) and torch.all(grad[bi + 1:].abs().sum() == 0)


def test_group_padding_is_ignored(model):
    """Padding instruments (valid = False) must not influence the valid ones."""
    b, _ = make(4, n_valid=2)
    z = torch.randn(b["x0"].shape)
    t = torch.full(b["anchor_close"].shape, 0.3)
    with torch.no_grad():
        v1 = model(model_inputs(b), z, t)
        b2 = dict(b)
        b2["x_hist"] = b["x_hist"].clone()
        b2["x_hist"][:, 2:] = torch.randn_like(b2["x_hist"][:, 2:]) * 10
        z2 = z.clone()
        z2[:, :, 2:] = torch.randn_like(z2[:, :, 2:]) * 10
        v2 = model(model_inputs(b2), z2, t)
    torch.testing.assert_close(v1[:, :, :2], v2[:, :, :2], rtol=1e-4, atol=1e-5)


def test_assembly_targets_and_positions():
    b, wins = make(5)
    audit(b)
    P, n_t = CFG.patch, N_PATCHES[0]
    tgt = wins[0]
    # target tokens sit at positions 1..n_t; every token position is consistent with its close time
    assert torch.equal(b["tok_pos"][:, :n_t], torch.arange(1, n_t + 1).float().expand(2, -1))
    assert torch.all(torch.diff(b["tok_pos"][:, n_t:n_t + N_PATCHES[1]]) >= 0)
    # x0 = the next H target patches after the anchor, first 5 channels
    for bi in range(2):
        for a, p in enumerate([7, 10, 12]):
            want = tgt.x[bi, :, (p + 1) * P:(p + 1 + H) * P, :5].reshape(-1, H, P * 5)
            assert torch.equal(b["x0"][bi, a], want)
    # coarse tokens never close after the target window end
    assert torch.all(b["tok_close"].max(1).values <= b["tok_close"][:, n_t - 1])


def test_audit_rejects_lookahead():
    b, _ = make(6)
    bad = dict(b)
    bad["anchor_close"] = b["fut_ts"][:, :, -1]      # pretend the forecast was made after the target closed
    with pytest.raises(AssertionError):
        audit(bad)
    with pytest.raises(AssertionError):
        audit(b, split_end_ns=int(b["fut_ts"].min()) - 1)


def test_sampler_shapes(model):
    b, _ = make(7)
    out = model.sample(model_inputs(b), n_paths=3, steps=2, solver="heun", generator=torch.Generator().manual_seed(0))
    assert out.shape == (2, 3, 3, 4, H, CFG.patch * CFG.out_channels) and torch.isfinite(out).all()


def test_pure_bf16_training_step():
    """bf16-true: weights, activations, loss and gradients are bf16; RoPE frequencies stay exact."""
    from kit2.model.kit2 import flow_loss
    torch.manual_seed(1)
    m = KiT2(CFG).to(torch.bfloat16).train()
    b, _ = make(8)
    loss = flow_loss(m, model_inputs(b), b["x0"], b["valid"])
    loss.backward()
    assert loss.dtype == torch.bfloat16 and torch.isfinite(loss)
    assert all(p.dtype == torch.bfloat16 and p.grad is not None and p.grad.dtype == torch.bfloat16
               for p in m.parameters() if p.requires_grad)
    assert m.rotary.inv_freq.dtype == torch.float32
    pos = torch.tensor([[200.0]])
    cos, _ = m.rotary(pos, torch.float32)
    want = torch.cos(200.0 * m.rotary.inv_freq)
    torch.testing.assert_close(cos[0, 0, : len(want)], want, rtol=1e-5, atol=1e-5)
