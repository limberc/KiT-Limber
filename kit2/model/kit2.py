"""KiT-2: patched multi-scale history encoder + flow-matching denoiser (spec: 2026-11-19-kit2-model-design.md).

Shapes: B samples, G instruments per sample (one market, same moment), S history tokens per instrument
(target scale + coarser scales, 8-bar patches), A anchors (forecast points) per sample, h future patches.

No look-ahead by construction:
- encoder time attention: token i sees token j only if close_j <= close_i (same instrument, any scale);
- encoder group attention: only tokens at the same slot, which share one close time across instruments;
- denoiser cross attention: the memory of anchor a is the tokens with close <= anchor_close[a];
- future positions get only calendar inputs (published in advance) and the noisy target itself.
Many anchors share one encoder pass in training; tests check this equals encoding the history truncated
at each anchor (kit2/tests/test_model_causality.py).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from diffusers.models.normalization import RMSNorm as AffinelessRMSNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from kit2.model.config import ModelConfig
from kit2.model.layers import DenoiserLayer, EncoderLayer, RotaryAt, flow_time_embedding, modulate

Batch = Dict[str, torch.Tensor]


class KiT2(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, P = cfg.d_model, cfg.patch
        # encoder inputs
        self.patch_in = nn.Linear(P * cfg.in_channels, d)
        self.cal_in = nn.Linear(cfg.cal_dim, d)
        self.scale_emb = nn.Embedding(cfg.n_scales, d)  # scale of each history token
        # instrument-level conditions (encoder tokens and denoiser AdaLN)
        self.market_emb = nn.Embedding(cfg.n_markets, d)
        self.chain_emb = nn.Embedding(cfg.n_chains, d)
        self.inst_emb = nn.Embedding(cfg.inst_vocab, d)
        self.target_scale_emb = nn.Embedding(cfg.n_scales, d)
        self.rotary = RotaryAt(cfg)
        self.enc = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.enc_layers)])
        self.enc_norm = LlamaRMSNorm(d)
        # denoiser
        self.fut_in = nn.Linear(P * cfg.out_channels, d)
        self.fut_cal = nn.Linear(cfg.cal_dim, d)
        self.slot_emb = nn.Embedding(cfg.max_horizon_patches, d)
        self.t_emb = flow_time_embedding(cfg)
        self.dec = nn.ModuleList([DenoiserLayer(cfg) for _ in range(cfg.dec_layers)])
        self.ada = nn.Linear(d, DenoiserLayer.N_MOD * d)  # adaLN-single: shared by all layers
        self.final_norm = AffinelessRMSNorm(d, eps=1e-6, elementwise_affine=False)
        self.final_table = nn.Parameter(torch.zeros(2, d))
        self.out = nn.Linear(d, P * cfg.out_channels)
        self.apply(self._init)
        for m in (self.ada, self.out):  # start as "identity layers, zero velocity"
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ---- conditions ----------------------------------------------------------------------------
    def inst_cond(self, b: Batch) -> torch.Tensor:
        """[B, G, d] from market, chain, instrument (dropped to 0 at random in training), target scale."""
        inst = b["inst"]
        if self.training and self.cfg.inst_dropout > 0:
            inst = torch.where(torch.rand(inst.shape, device=inst.device) < self.cfg.inst_dropout, 0, inst)
        return (self.market_emb(b["market"]) + self.chain_emb(b["chain"]) + self.inst_emb(inst)
                + rearrange(self.target_scale_emb(b["target_scale"]), "b d -> b 1 d"))

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_in.weight.dtype

    # ---- encoder -------------------------------------------------------------------------------
    def encode(self, b: Batch, cond: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """History -> memory. b: x_hist [B, G, S, P*C], tok_scale [S], tok_close [B, S] (int64),
        tok_pos [B, S] (target-patch units), tok_cal [B, S, cal], valid [B, G], inst/market/chain [B, G],
        target_scale [B]."""
        cond = self.inst_cond(b) if cond is None else cond
        dt = self.dtype                                    # inputs follow the weights (bf16 in training)
        x = (self.patch_in(b["x_hist"].to(dt)) + rearrange(self.cal_in(b["tok_cal"].to(dt)), "b s d -> b 1 s d")
             + self.scale_emb(b["tok_scale"]) + rearrange(cond, "b g d -> b g 1 d"))
        close = b["tok_close"]
        time_mask = rearrange(close, "b k -> b 1 1 k") <= rearrange(close, "b q -> b 1 q 1")   # key closes <= query
        group_mask = rearrange(b["valid"], "b g -> b 1 1 g")
        rot = self.rotary(b["tok_pos"], dt)
        for layer in self.enc:
            x = layer(x, time_mask, group_mask, rot)
        return {"h": self.enc_norm(x), "cond": cond}

    def memory_kv(self, h: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Per denoiser layer, cross-attention k/v of the encoder states: computed once, reused every step."""
        flat = rearrange(h, "b g s d -> (b g) s d")
        return [layer.cross_attn.project_kv(flat) for layer in self.dec]

    # ---- denoiser ------------------------------------------------------------------------------
    def denoise(self, z: torch.Tensor, t: torch.Tensor, b: Batch, cond: torch.Tensor,
                mem_kv: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Velocity for the noisy future z [B, A, G, h, P*5] at flow time t [B, A] (1 = noise), or [B, A, G]: one
        flow time per instrument (0 = a given, clean future the others are conditioned on).
        b: anchor_close [B, A] (int64), anchor_pos [B, A], fut_cal [B, A, h, cal], tok_close/tok_pos, valid."""
        h = z.shape[3]
        k = torch.arange(h, device=z.device)
        dt = self.dtype
        if t.dim() == 2:
            t = repeat(t, "b a -> b a g", g=z.shape[2])
        x = (self.fut_in(z.to(dt)) + rearrange(self.fut_cal(b["fut_cal"].to(dt)), "b a h d -> b a 1 h d")
             + self.slot_emb(k))
        c = F.silu(self.t_emb(t) + rearrange(cond, "b g d -> b 1 g d"))
        mod = rearrange(self.ada(c), "b a g (m d) -> b a g m d",
                        m=DenoiserLayer.N_MOD)
        fut_pos = rearrange(b["anchor_pos"], "b a -> b a 1") + 1.0 + k                      # [B, A, h]
        rot_self = self.rotary(rearrange(fut_pos, "b a h -> (b a) h"), dt)
        rot_q = self.rotary(rearrange(fut_pos, "b a h -> b (a h)"), dt)
        rot_mem = self.rotary(b["tok_pos"], dt)
        mem_mask = rearrange(b["tok_close"], "b s -> b 1 s") <= rearrange(b["anchor_close"], "b a -> b a 1")
        mem_mask = repeat(mem_mask, "b a s -> b 1 (a h) s", h=h)                       # memory closes <= anchor
        group_mask = rearrange(b["valid"], "b g -> b 1 1 g")
        for layer, kv in zip(self.dec, mem_kv):
            x = layer(x, mod, kv, mem_mask, group_mask, rot_self, rot_q, rot_mem)
        shift, scale = rearrange(mod[..., :2, :] + self.final_table, "b a g m d -> b a g 1 m d").unbind(dim=4)
        return self.out(modulate(self.final_norm(x), shift, scale))

    def forward(self, b: Batch, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mem = self.encode(b)
        return self.denoise(z, t, b, mem["cond"], self.memory_kv(mem["h"]))

    # ---- sampling ------------------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, b: Batch, n_paths: int, steps: int = 8, solver: str = "heun",
               generator: Optional[torch.Generator] = None, given: Optional[torch.Tensor] = None,
               given_mask: Optional[torch.Tensor] = None, mode: str = "clean") -> torch.Tensor:
        """K paths per anchor: [B, A, K, G, h, P*5] in normalised feature space. The history is encoded once;
        paths are extra anchors sharing the same memory, and instruments in one path are sampled jointly.

        Conditional paths: the instruments in given_mask [B, G] take the futures `given` - [B, A, G, h, P*5] (the
        same for every path) or [B, A, K, G, h, P*5] (one per path) - and the others are sampled conditionally.
        mode "clean": given instruments enter at flow time 0 (models trained with p_cond > 0); "replace": they
        share the flow time and are reset to (1 - t) given + t eps along their own initial noise before every
        velocity evaluation (inpainting: works with any checkpoint, but the conditioning is approximate)."""
        mem = self.encode(b)
        kv = self.memory_kv(mem["h"])
        B, A = b["anchor_close"].shape
        G, h = b["valid"].shape[1], b["fut_cal"].shape[2]
        rep = {k: repeat(b[k], "b a ... -> b (a k) ...", k=n_paths) for k in ("anchor_close", "anchor_pos", "fut_cal")}
        bb = {**b, **rep}
        shape = (B, A * n_paths, G, h, self.cfg.patch * self.cfg.out_channels)
        z = torch.randn(shape, generator=generator, device=mem["h"].device, dtype=self.dtype)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=z.device)
        if given is not None:
            assert given_mask is not None and mode in ("clean", "replace"), "given needs given_mask [B, G]"
            x = (rearrange(given, "b a k g h c -> b (a k) g h c") if given.dim() == 6
                 else repeat(given, "b a g h c -> b (a k) g h c", k=n_paths)).to(z.dtype)
            gm = rearrange(given_mask, "b g -> b 1 g")
            eps0 = z.clone()

        def at(zz, t):
            """State and flow times at flow time t, with the given futures imposed."""
            if given is None:
                return zz, t.expand(B, A * n_paths)
            if mode == "clean":
                return torch.where(gm[..., None, None], x, zz), torch.where(gm, 0.0, t).expand(B, A * n_paths, G)
            return torch.where(gm[..., None, None], (1 - t) * x + t * eps0, zz), t.expand(B, A * n_paths)

        for i in range(steps):
            t0, t1 = ts[i], ts[i + 1]
            zi, tt = at(z, t0)
            v = self.denoise(zi, tt, bb, mem["cond"], kv)
            if solver == "heun" and i < steps - 1:
                z1, tt1 = at(zi - (t0 - t1) * v, t1)
                v2 = self.denoise(z1, tt1, bb, mem["cond"], kv)
                v = 0.5 * (v + v2)
            z = zi - (t0 - t1) * v
        z = at(z, ts[-1])[0]                                  # given instruments end exactly on their futures
        return rearrange(z, "b (a k) g h c -> b a k g h c", k=n_paths)

    # ---- likelihood ----------------------------------------------------------------------------
    @torch.no_grad()
    def log_likelihood(self, b: Batch, x0: torch.Tensor, steps: int = 32, n_probe: int = 4,
                       given_mask: Optional[torch.Tensor] = None,
                       generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Log density of the futures x0 [B, A, G, h, P*5] (normalised space) under the model: [B, A, G], per
        instrument, summing to the joint log density of the generated instruments (fp32).

        Probability-flow ODE dz/dt = v from data (t = 0) to noise (t = 1), Heun steps:
        log p(x0) = log N(z_1; 0, I) + int_0^1 div v dt, the divergence by Hutchinson with n_probe Rademacher probes
        (fixed along the path; one VJP for all of them, stacked along the anchor axis). An instrument's share is its
        block of the divergence plus its prior terms - an exact split of the joint density, not its conditional
        density. given_mask [B, G]: those instruments stay at their futures with flow time 0 (clean conditioning;
        checkpoints trained with p_cond > 0) and get 0; the rest is then the log density of the others given them."""
        mem = self.encode(b)
        kv = self.memory_kv(mem["h"])
        B, A, G = x0.shape[:3]
        R = n_probe
        bb = {**b, **{k: repeat(b[k], "b a ... -> b (a r) ...", r=R) for k in ("anchor_close", "anchor_pos", "fut_cal")}}
        free = torch.ones(B, G, dtype=torch.bool, device=x0.device) if given_mask is None else ~given_mask
        fm = rearrange(free, "b g -> b 1 g 1 1")
        x = repeat(x0.float(), "b a g h c -> b (a r) g h c", r=R)
        e = (torch.randint(0, 2, x.shape, generator=generator, device=x.device) * 2 - 1).float() * fm

        def vel_div(zz: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            tg = t.expand(B, A * R, G) if given_mask is None else torch.where(
                rearrange(given_mask, "b g -> b 1 g"), 0.0, t).expand(B, A * R, G)
            with torch.enable_grad():
                zz = zz.detach().requires_grad_(True)
                v = self.denoise(torch.where(fm, zz, x), tg, bb, mem["cond"], kv).float() * fm
                (g,) = torch.autograd.grad((v * e).sum(), zz)
            return v.detach(), reduce(g * e, "b a g h c -> b a g", "sum")

        ts = torch.linspace(0.0, 1.0, steps + 1, device=x.device)
        z, div = x.clone(), torch.zeros(B, A * R, G, device=x.device)
        for i in range(steps):
            dt = ts[i + 1] - ts[i]
            v0, d0 = vel_div(z, ts[i])
            v1, d1 = vel_div(z + dt * v0, ts[i + 1])
            z = z + 0.5 * dt * (v0 + v1)
            div = div + 0.5 * dt * (d0 + d1)
        D = x0.shape[3] * x0.shape[4]
        prior = -0.5 * reduce(z ** 2, "b a g h c -> b a g", "sum") - 0.5 * D * math.log(2 * math.pi)
        ll = reduce(prior + div, "b (a r) g -> b a g", "mean", r=R)
        return ll * rearrange(free, "b g -> b 1 g")


def conditioning_subset(valid: torch.Tensor, is_anchor: Optional[torch.Tensor], A: int, p_cond: float,
                        generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Instruments whose future is given (clean, no loss) per (sample, anchor): [B, A, G] bool. A share p_cond of the
    (sample, anchor) pairs condition: on the market anchors (half of them, when the group has anchors) or on a random
    share u ~ U(0, 1) of the valid instruments. A pair that would leave nothing to generate does not condition."""
    B, G = valid.shape
    dev = valid.device
    r = torch.rand((B, A, 3), device=dev, generator=generator)
    u = torch.rand((B, A, G), device=dev, generator=generator)
    rand_set = (u < r[..., 2:3]) & rearrange(valid, "b g -> b 1 g")
    if is_anchor is not None:
        use_anc = (r[..., 1:2] < 0.5) & rearrange(is_anchor.any(1), "b -> b 1 1")
        given = torch.where(use_anc, rearrange(is_anchor & valid, "b g -> b 1 g"), rand_set)
    else:
        given = rand_set
    given = given & (r[..., 0:1] < p_cond)
    left = (rearrange(valid, "b g -> b 1 g") & ~given).any(-1, keepdim=True)
    return given & left


def flow_loss(model: KiT2, b: Batch, x0: torch.Tensor, weight: torch.Tensor,
              generator: Optional[torch.Generator] = None, p_cond: float = 0.0) -> torch.Tensor:
    """Rectified flow: z_t = (1 - t) x0 + t eps, target v = eps - x0, t ~ logit-normal(0, 1) per (sample, anchor).
    x0 [B, A, G, h, P*5]; weight [B, G] (0 for padding instruments). Computed in the model's dtype (bf16 in
    training); reductions accumulate in fp32 inside the kernels.
    p_cond > 0: some (sample, anchor) pairs condition on a clean subset of instruments (`conditioning_subset`, using
    b["is_anchor"] when present): those get flow time 0 and their true future as input, and no loss. One set of
    weights then serves joint sampling, conditional scenarios and conditional likelihoods. p_cond = 0 is the joint
    objective exactly (same random draws)."""
    dt = model.dtype
    x0 = x0.to(dt)
    B, A = x0.shape[:2]
    t = torch.sigmoid(torch.randn((B, A), device=x0.device,
                                  generator=generator))
    eps = torch.randn(x0.shape, device=x0.device,
                      generator=generator, dtype=dt)
    if p_cond <= 0:
        tt = rearrange(t, "b a -> b a 1 1 1").to(dt)
        v = model(b, (1 - tt) * x0 + tt * eps, t)
        err = reduce((v - (eps - x0)).pow(2), "b a g h c -> b a g", "mean")
        w = rearrange(weight, "b g -> b 1 g").to(dt)
        return (err * w).sum() / (w.sum() * A).clamp_min(1.0)
    given = conditioning_subset(weight.bool(), b.get("is_anchor"), A, p_cond, generator)
    tg = torch.where(given, 0.0, rearrange(t, "b a -> b a 1"))
    tt = rearrange(tg, "b a g -> b a g 1 1").to(dt)
    v = model(b, (1 - tt) * x0 + tt * eps, tg)
    err = reduce((v - (eps - x0)).pow(2), "b a g h c -> b a g", "mean")
    w = (rearrange(weight, "b g -> b 1 g") * ~given).to(dt)
    return (err * w).sum() / w.sum().clamp_min(1.0)
