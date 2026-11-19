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
from einops import pack, rearrange, unpack
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
    cos, sin = (rearrange(r, "n l d -> n 1 l d") for r in rot)
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
        k, v = rearrange(self.kv(kv), "n l (two h d) -> two n h l d", two=2, h=self.h)
        return self.k_norm(k), v

    def forward(self, x: torch.Tensor, kv: Optional[torch.Tensor] = None,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                mask: Optional[torch.Tensor] = None, rot_q: Optional[Rotary] = None,
                rot_k: Optional[Rotary] = None, fold: int = 1) -> torch.Tensor:
        q = self.q_norm(rearrange(self.q(x), "n l (h d) -> n h l d", h=self.h))
        k, v = kv_cache if kv_cache is not None else self.project_kv(x if kv is None else kv)
        q, k, v = (rearrange(t, "(n f) h l d -> n (f h) l d", f=fold) for t in (q, k, v))
        q, k = apply_rope(q, rot_q), apply_rope(k, rot_k if rot_k is not None else rot_q)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(rearrange(o, "n (f h) l d -> (n f) l (h d)", f=fold))


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
        B, G, S, _ = x.shape
        y = rearrange(self.n1(x), "b g s d -> (b g) s d")                  # the G rows of a sample share rot/mask
        x = x + rearrange(self.time_attn(y, mask=time_mask, rot_q=rot, fold=G), "(b g) s d -> b g s d", g=G)
        y = rearrange(self.n2(x), "b g s d -> (b s) g d")
        x = x + rearrange(self.group_attn(y, mask=group_mask, fold=S), "(b s) g d -> b g s d", s=S)
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
        A, G, h = x.shape[1:4]
        # split along the modulation axis in place: moving it to the front (m b a g ...) makes inductor emit
        # a strided reduction for the table's gradient that took 70% of a 1d training step
        m = rearrange(mod + self.mod_table, "b a g m d -> b a g 1 m d").unbind(dim=4)
        # self attention over the h future slots of each (b, a, g); the G rows of a (b, a) share rotary
        y = rearrange(modulate(self.norm(x), m[0], m[1]), "b a g h d -> (b a g) h d")
        x = x + m[2] * rearrange(self.self_attn(y, rot_q=rot_self, fold=G), "(b a g) h d -> b a g h d", a=A, g=G)
        # group attention across the G instruments at each (b, a, slot)
        y = rearrange(modulate(self.norm(x), m[3], m[4]), "b a g h d -> (b a h) g d")
        o = self.group_attn(y, mask=group_mask, fold=A * h)
        x = x + m[5] * rearrange(o, "(b a h) g d -> b a g h d", a=A, h=h)
        # cross attention: (b, g) rows see that instrument's memory; all anchors' queries share the row
        y = rearrange(modulate(self.norm(x), m[6], m[7]), "b a g h d -> (b g) (a h) d")
        o = self.cross_attn(y, kv_cache=mem_kv, mask=mem_mask, rot_q=rot_q, rot_k=rot_mem, fold=G)
        x = x + m[8] * rearrange(o, "(b g) (a h) d -> b a g h d", g=G, a=A)
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
            flat, ps = pack([t], "*")
            e = self.emb(self.proj(flat * 1000.0).to(self.emb.linear_1.weight.dtype))
            return unpack(e, ps, "* d")[0]

    return FlowTime()
