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
        self.enc_norm = LlamaRMSNorm(d, eps=1e-6)
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
        """Velocity for the noisy future z [B, A, G, h, P*5] at flow time t [B, A] (1 = noise).
        b: anchor_close [B, A] (int64), anchor_pos [B, A], fut_cal [B, A, h, cal], tok_close/tok_pos, valid."""
        h = z.shape[3]
        k = torch.arange(h, device=z.device)
        dt = self.dtype
        x = (self.fut_in(z.to(dt)) + rearrange(self.fut_cal(b["fut_cal"].to(dt)), "b a h d -> b a 1 h d")
             + self.slot_emb(k))
        c = F.silu(rearrange(self.t_emb(t), "b a d -> b a 1 d") + rearrange(cond, "b g d -> b 1 g d"))
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
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """K paths per anchor: [B, A, K, G, h, P*5] in normalised feature space. The history is encoded once;
        paths are extra anchors sharing the same memory, and instruments in one path are sampled jointly."""
        mem = self.encode(b)
        kv = self.memory_kv(mem["h"])
        B, A = b["anchor_close"].shape
        G, h = b["valid"].shape[1], b["fut_cal"].shape[2]
        rep = {k: repeat(b[k], "b a ... -> b (a k) ...", k=n_paths) for k in ("anchor_close", "anchor_pos", "fut_cal")}
        bb = {**b, **rep}
        shape = (B, A * n_paths, G, h, self.cfg.patch * self.cfg.out_channels)
        z = torch.randn(shape, generator=generator, device=mem["h"].device, dtype=self.dtype)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=z.device)
        for i in range(steps):
            t0, t1 = ts[i], ts[i + 1]
            v = self.denoise(z, t0.expand(B, A * n_paths), bb, mem["cond"], kv)
            if solver == "heun" and i < steps - 1:
                z1 = z - (t0 - t1) * v
                v2 = self.denoise(z1, t1.expand(B, A * n_paths), bb, mem["cond"], kv)
                v = 0.5 * (v + v2)
            z = z - (t0 - t1) * v
        return rearrange(z, "b (a k) g h c -> b a k g h c", k=n_paths)


def flow_loss(model: KiT2, b: Batch, x0: torch.Tensor, weight: torch.Tensor,
              generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Rectified flow: z_t = (1 - t) x0 + t eps, target v = eps - x0, t ~ logit-normal(0, 1) per (sample, anchor).
    x0 [B, A, G, h, P*5]; weight [B, G] (0 for padding instruments). Computed in the model's dtype (bf16 in
    training); reductions accumulate in fp32 inside the kernels."""
    dt = model.dtype
    x0 = x0.to(dt)
    B, A = x0.shape[:2]
    t = torch.sigmoid(torch.randn((B, A), device=x0.device, generator=generator))
    eps = torch.randn(x0.shape, device=x0.device, generator=generator, dtype=dt)
    tt = rearrange(t, "b a -> b a 1 1 1").to(dt)
    v = model(b, (1 - tt) * x0 + tt * eps, t)
    err = reduce((v - (eps - x0)).pow(2), "b a g h c -> b a g", "mean")
    w = rearrange(weight, "b g -> b 1 g").to(dt)
    return (err * w).sum() / (w.sum() * A).clamp_min(1.0)
