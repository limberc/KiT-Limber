"""Turn gathered bar windows into model inputs; audit them for look-ahead.

A sample is G instruments of one market whose windows end at the same moment. For each scale s (target
first, then coarser scales) the caller provides the gathered window x [B, G, n_s * P, C], the grid index of
the window's first bar [B], and that scale's grid (bar close times, int64 ns) and calendar table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch

from kit2.model.config import ModelConfig


@dataclass
class ScaleWindow:
    x: torch.Tensor            # [B, G, n*P, C] normalised bar channels
    start: torch.Tensor        # [B] grid index of the first bar
    grid_ts: torch.Tensor      # [n_grid] int64 bar close times (ns), ascending
    grid_cal: torch.Tensor     # [n_grid, cal_dim]
    scale_id: int


def assemble(windows: Sequence[ScaleWindow], anchors: torch.Tensor, horizon: int, meta: Dict[str, torch.Tensor],
             cfg: ModelConfig) -> Dict[str, torch.Tensor]:
    """windows[0] is the target scale. anchors [B, A]: target patch index p (forecast made at the close of
    patch p, futures are patches p+1 .. p+horizon). meta: inst, market, chain, valid [B, G]; target_scale [B].
    Returns the model batch plus `x0` [B, A, G, h, P*out] (targets) and `fut_ts` [B, A, h*P] (target bar
    close times, for the audit)."""
    P = cfg.patch
    toks, closes, cals, sids = [], [], [], []
    for w in windows:
        B, G, L, C = w.x.shape
        n = L // P
        end = w.start[:, None] + (torch.arange(n, device=w.x.device) + 1) * P - 1          # [B, n]
        toks.append(w.x.reshape(B, G, n, P * C))
        closes.append(w.grid_ts[end])
        cals.append(w.grid_cal[end])
        sids.append(torch.full((n,), w.scale_id, dtype=torch.long, device=w.x.device))
    tgt = windows[0]
    B, G, Lt, C = tgt.x.shape
    n_t = Lt // P
    tok_close = torch.cat(closes, 1)
    # position = target bars closed by the token's close time, counted from the window start, in patches
    tok_pos = (torch.searchsorted(tgt.grid_ts, tok_close, right=True) - tgt.start[:, None]).float() / P

    A = anchors.shape[1]
    fut = anchors[:, :, None] + 1 + torch.arange(horizon, device=anchors.device)            # [B, A, h]
    if int(fut.max()) >= n_t or int(anchors.min()) < 0:
        raise ValueError("anchor + horizon must lie inside the target window")
    bi = torch.arange(B, device=anchors.device)
    gi = torch.arange(G, device=anchors.device)
    tgt_tok = toks[0].view(B, G, n_t, P, C)[..., : cfg.out_channels].reshape(B, G, n_t, P * cfg.out_channels)
    x0 = tgt_tok[bi[:, None, None, None], gi[None, None, :, None], fut[:, :, None, :]]   # [B, A, G, h, P*out]
    bar = tgt.start[:, None, None, None] + fut[..., None] * P + torch.arange(P, device=anchors.device)
    return {
        "x_hist": torch.cat(toks, 2), "tok_scale": torch.cat(sids), "tok_close": tok_close, "tok_pos": tok_pos,
        "tok_cal": torch.cat(cals, 1), "anchor_close": closes[0][bi[:, None], anchors],
        "anchor_pos": (anchors + 1).float(), "fut_cal": cals[0][bi[:, None, None], fut],
        "x0": x0, "fut_ts": tgt.grid_ts[bar.reshape(B, A, horizon * P)], **meta,
    }


def audit(b: Dict[str, torch.Tensor], split_end_ns: int | None = None) -> None:
    """Runtime look-ahead check (cheap): every target bar closes strictly after its anchor, the anchor is a
    history token close, and (training) no target bar is after the split end."""
    a = b["anchor_close"]
    if not bool((b["fut_ts"] > a[:, :, None]).all()):
        raise AssertionError("a target bar closes at or before its anchor")
    if not bool((b["tok_close"][:, None, :] == a[:, :, None]).any(-1).all()):
        raise AssertionError("an anchor is not a history token close")
    if split_end_ns is not None and bool((b["fut_ts"] > split_end_ns).any()):
        raise AssertionError("a training target lies after train_end")


def model_inputs(b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in b.items() if k not in ("x0", "fut_ts")}


# ---- synthetic data (benchmarks and tests) -------------------------------------------------------------
def synthetic_grid(n: int, step_s: int, gap_every: int = 0, gap_s: int = 0, t0_s: int = 1_700_000_000,
                   device="cpu") -> torch.Tensor:
    """Bar close times: regular steps, optionally with a gap (overnight / weekend) every `gap_every` bars."""
    i = torch.arange(n, device=device, dtype=torch.int64)
    extra = (i // gap_every) * gap_s if gap_every else torch.zeros_like(i)
    return (t0_s + (i + 1) * step_s + extra) * 1_000_000_000


def synthetic_windows(cfg: ModelConfig, B: int, G: int, n_patches: List[int], steps_s: List[int],
                      device="cpu", generator: torch.Generator | None = None,
                      gap_every: int = 390, gap_s: int = 63000) -> List[ScaleWindow]:
    """Random bars on US-like grids (sessions with overnight gaps); windows of all scales end together."""
    P = cfg.patch
    span_s = 3 * max(n * P * st for n, st in zip(n_patches, steps_s))       # trading seconds every grid covers
    grids = [synthetic_grid(span_s // st + n * P + 64, st, max(1, gap_every * steps_s[0] // st), gap_s, device=device)
             for n, st in zip(n_patches, steps_s)]
    lo = max(int(g[n * P - 1]) for g, n in zip(grids, n_patches))
    hi = min(int(g[-1]) for g in grids)
    tgt = grids[0]
    ok = torch.nonzero((tgt >= lo) & (tgt <= hi)).flatten()
    end = ok[torch.randint(len(ok), (B,), generator=generator, device=device)]
    out = []
    for k, (g, n) in enumerate(zip(grids, n_patches)):
        start = end - n * P + 1 if k == 0 else torch.searchsorted(g, tgt[end], right=True) - n * P
        cal = torch.randn(len(g), cfg.cal_dim, generator=generator, device=device)
        x = torch.randn(B, G, n * P, cfg.in_channels, generator=generator, device=device)
        out.append(ScaleWindow(x, start, g, cal, scale_id=k + 1))
    return out


def synthetic_meta(B: int, G: int, device="cpu", n_valid: int | None = None) -> Dict[str, torch.Tensor]:
    valid = torch.ones(B, G, dtype=torch.bool, device=device)
    if n_valid is not None:
        valid[:, n_valid:] = False
    return {"inst": torch.randint(1, 4000, (B, G), device=device), "market": torch.ones(B, G, dtype=torch.long,
            device=device), "chain": torch.zeros(B, G, dtype=torch.long, device=device), "valid": valid,
            "target_scale": torch.ones(B, dtype=torch.long, device=device)}
