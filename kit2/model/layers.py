"""Building blocks, reusing Hugging Face layers where they exist.

- RMSNorm (pre-norm, with weight): transformers LlamaRMSNorm; AdaLN norms (no weight): diffusers RMSNorm.
- SwiGLU: transformers LlamaMLP. Flow-time embedding: diffusers Timesteps + TimestepEmbedding.
- RoPE: transformers LlamaRotaryEmbedding for cos/sin at continuous positions (time measured in target
  patches), rotate_half for the rotation; queries and keys may sit at different positions (cross-attention).
- Attention: torch SDPA with boolean masks (True = may attend). Batch dimensions that share a mask are
  folded into the head dimension, so a [B, 1, Lq, Lk] mask broadcasts without copies.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.normalization import RMSNorm as AffinelessRMSNorm
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (LlamaMLP, LlamaRMSNorm, LlamaRotaryEmbedding,
                                                      rotate_half)

from kit2.model.config import ModelConfig

Rotary = Tuple[torch.Tensor, torch.Tensor]          # cos, sin: [B, L, head_dim] fp32


def llama_config(cfg: ModelConfig) -> LlamaConfig:
    return LlamaConfig(hidden_size=cfg.d_model, num_attention_heads=cfg.n_heads,
                       intermediate_size=cfg.ffn_hidden, hidden_act="silu", mlp_bias=False,
                       rope_parameters={"rope_type": "default", "rope_theta": float(cfg.rope_theta)})


class RotaryAt(nn.Module):
    """cos/sin at arbitrary float positions [B, L] (Llama RoPE frequencies)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.rope = LlamaRotaryEmbedding(llama_config(cfg))

    def forward(self, pos: torch.Tensor) -> Rotary:
        dummy = torch.empty(0, device=pos.device, dtype=torch.float32)
        return self.rope(dummy, pos.float())


def apply_rope(x: torch.Tensor, rot: Optional[Rotary]) -> torch.Tensor:
    """x [N', H', L, Dh] (after folding); rot cos/sin [N', L, Dh], one row per folded batch row."""
    if rot is None:
        return x
    cos, sin = (r[:, None] for r in rot)
    xf = x.float()
    return (xf * cos + rotate_half(xf) * sin).to(x.dtype)


class Attention(nn.Module):
    """Multi-head attention with QK-norm. Self-attention when `kv` is None, else cross-attention.

    Inputs are [N, L, d]. `fold` > 1 means consecutive groups of `fold` rows of N share one mask/rotary
    row: they are folded into the head dimension so masks [N/fold, 1, Lq, Lk] broadcast with no copy.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d, self.h, self.dh = cfg.d_model, cfg.n_heads, cfg.head_dim
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.q_norm = LlamaRMSNorm(self.dh, eps=1e-6)
        self.k_norm = LlamaRMSNorm(self.dh, eps=1e-6)

    def project_kv(self, kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[N, Lk, d] -> normalised k, v [N, H, Lk, Dh] (cacheable across sampling steps)."""
        N, L, _ = kv.shape
        k, v = self.kv(kv).view(N, L, 2, self.h, self.dh).unbind(2)
        return self.k_norm(k).transpose(1, 2), v.transpose(1, 2)

    def forward(self, x: torch.Tensor, kv: Optional[torch.Tensor] = None,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                mask: Optional[torch.Tensor] = None, rot_q: Optional[Rotary] = None,
                rot_k: Optional[Rotary] = None, fold: int = 1) -> torch.Tensor:
        N, Lq, d = x.shape
        q = self.q_norm(self.q(x).view(N, Lq, self.h, self.dh)).transpose(1, 2)        # [N, H, Lq, Dh]
        k, v = kv_cache if kv_cache is not None else self.project_kv(x if kv is None else kv)
        if fold > 1:                                   # [N, H, L, Dh] -> [N/fold, fold*H, L, Dh]
            q, k, v = (t.reshape(N // fold, fold * self.h, t.shape[2], self.dh) for t in (q, k, v))
        q, k = apply_rope(q, rot_q), apply_rope(k, rot_k if rot_k is not None else rot_q)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        o = o.reshape(N, self.h, Lq, self.dh).transpose(1, 2).reshape(N, Lq, d)
        return self.out(o)


class EncoderLayer(nn.Module):
    """time attention (per instrument, causal by close time) -> group attention (across instruments at the
    same token slot) -> SwiGLU. Pre-norm residual."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n1, self.n2, self.n3 = (LlamaRMSNorm(cfg.d_model, eps=1e-6) for _ in range(3))
        self.time_attn, self.group_attn = Attention(cfg), Attention(cfg)
        self.mlp = LlamaMLP(llama_config(cfg))

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor, group_mask: torch.Tensor,
                rot: Rotary) -> torch.Tensor:
        """x [B, G, S, d]; time_mask [B, 1, S, S]; group_mask [B, 1, 1, G]; rot over [B, S]."""
        B, G, S, d = x.shape
        h = self.n1(x).reshape(B * G, S, d)                                # G rows of a sample share rot/mask
        x = x + self.time_attn(h, mask=time_mask, rot_q=rot, fold=G).view(B, G, S, d)
        h = self.n2(x).transpose(1, 2).reshape(B * S, G, d)
        x = x + self.group_attn(h, mask=group_mask, fold=S).view(B, S, G, d).transpose(1, 2)
        return x + self.mlp(self.n3(x))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class DenoiserLayer(nn.Module):
    """AdaLN-modulated: self attention over the future slots -> group attention across instruments ->
    cross attention to the encoder memory (masked to close <= anchor) -> SwiGLU.

    adaLN-single (PixArt-alpha): the model computes one shared modulation from (flow time, conditions);
    each layer adds its own learnable table, so the per-layer cost is 12 x d parameters, not 12 x d^2."""

    N_MOD = 12

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.norm = AffinelessRMSNorm(d, eps=1e-6, elementwise_affine=False)
        self.self_attn, self.group_attn, self.cross_attn = Attention(cfg), Attention(cfg), Attention(cfg)
        self.mlp = LlamaMLP(llama_config(cfg))
        self.mod_table = nn.Parameter(torch.zeros(self.N_MOD, d))

    def forward(self, x: torch.Tensor, mod: torch.Tensor, mem_kv: Tuple[torch.Tensor, torch.Tensor],
                mem_mask: torch.Tensor, group_mask: torch.Tensor, rot_self: Rotary, rot_q: Rotary,
                rot_mem: Rotary) -> torch.Tensor:
        """x [B, A, G, h, d]; mod [B, A, G, 12, d] shared modulation; mem_kv k, v [B*G, H, S, Dh];
        mem_mask [B, 1, A*h, S]; group_mask [B, 1, 1, G]; rot_self over [B*A, h]; rot_q over [B, A*h];
        rot_mem over [B, S]."""
        B, A, G, h, d = x.shape
        m = (mod + self.mod_table)[:, :, :, None].unbind(dim=4)                   # each [B, A, G, 1, d]
        # self attention over the h future slots of each (b, a, g)
        y = modulate(self.norm(x), m[0], m[1]).reshape(B * A * G, h, d)
        rs = tuple(r.repeat_interleave(G, 0) for r in rot_self)                   # [B*A*G, h, Dh]
        x = x + m[2] * self.self_attn(y, rot_q=rs).view(B, A, G, h, d)
        # group attention across the G instruments at each (b, a, slot)
        y = modulate(self.norm(x), m[3], m[4]).permute(0, 1, 3, 2, 4).reshape(B * A * h, G, d)
        x = x + m[5] * self.group_attn(y, mask=group_mask, fold=A * h).view(B, A, h, G, d).permute(0, 1, 3, 2, 4)
        # cross attention: (b, g) rows see that instrument's memory; queries of all anchors share the row
        y = modulate(self.norm(x), m[6], m[7]).permute(0, 2, 1, 3, 4).reshape(B * G, A * h, d)
        o = self.cross_attn(y, kv_cache=mem_kv, mask=mem_mask, rot_q=rot_q, rot_k=rot_mem, fold=G)
        x = x + m[8] * o.view(B, G, A, h, d).permute(0, 2, 1, 3, 4)
        return x + m[11] * self.mlp(modulate(self.norm(x), m[9], m[10]))


def flow_time_embedding(cfg: ModelConfig) -> nn.Module:
    """diffusers Timesteps(t * 1000) -> TimestepEmbedding, the DiT / PixArt pair."""
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps

    class FlowTime(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = Timesteps(cfg.t_emb_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.emb = TimestepEmbedding(cfg.t_emb_dim, cfg.d_model)

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            return self.emb(self.proj(t.reshape(-1) * 1000.0).to(self.emb.linear_1.weight.dtype)).view(
                *t.shape, -1)

    return FlowTime()
