"""DiT block with per-role AdaLN modulation."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from kit_v1.model.layers import MultiheadAttention, RMSNorm, SwiGLUMLP, llama_config


class DiTBlock(nn.Module):
    """One DiT block: attention + SwiGLU, modulated by shared AdaLN plus a per-layer bias."""

    def __init__(self, d: int, n_heads: int, ffn_hidden: int,
                 qk_norm: bool = True, backend: str = "native") -> None:
        super().__init__()
        self.norm1 = RMSNorm(d, eps=1e-6, elementwise_affine=False)
        self.norm2 = RMSNorm(d, eps=1e-6, elementwise_affine=False)
        self.attn = MultiheadAttention(d, n_heads, qk_norm=qk_norm, backend=backend)
        self.mlp = SwiGLUMLP(llama_config(d, n_heads, ffn_hidden, rope_theta=10000.0))
        self.bias_i = nn.Parameter(torch.zeros(6 * d))

    def forward(self, x: torch.Tensor, c_mod_ctx: torch.Tensor,
                c_mod_tgt: torch.Tensor, role: torch.Tensor,
                pad_mask: Optional[torch.Tensor],
                rotary: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """x [B,S,d]; c_mod_* [B,6d]; role [B,S] long (0=ctx, 1=tgt); rotary = (cos, sin)."""
        # pick modulation per token by role: [B,S,6d]
        mod = torch.where(role.unsqueeze(-1) == 1,
                          c_mod_tgt.unsqueeze(1), c_mod_ctx.unsqueeze(1))
        mod = mod + self.bias_i
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)

        h = self.norm1(x) * (1 + scale1) + shift1
        x = x + gate1 * self.attn(h, pad_mask=pad_mask, rotary=rotary)
        h = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.mlp(h)
        return x
