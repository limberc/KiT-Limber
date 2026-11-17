"""Model building blocks reused from Hugging Face transformers and diffusers.

- RMSNorm with weight (QK-Norm): transformers LlamaRMSNorm; kernelize_hub() can swap in a hub kernel (Liger).
- RMSNorm without weight (AdaLN supplies scale/shift): diffusers RMSNorm(elementwise_affine=False).
- SwiGLU MLP: transformers LlamaMLP, registered as the hub "SwiGLUMLP" layer (Liger fused silu*up).
- RoPE: transformers LlamaRotaryEmbedding (cos/sin once per forward) + apply_rotary_pos_emb.
- Timestep embedding: diffusers Timesteps + TimestepEmbedding (the DiT / PixArt pair).
- Attention kernel: diffusers dispatch_attention_fn (native SDPA, cuDNN, flash, sage, flex, hub kernels).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers.models.attention_dispatch import AttentionBackendName, dispatch_attention_fn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.normalization import RMSNorm
from transformers import LlamaConfig
from transformers.integrations.hub_kernels import use_kernel_forward_from_hub
from transformers.models.llama.modeling_llama import (LlamaMLP, LlamaRMSNorm,
                                                      LlamaRotaryEmbedding,
                                                      apply_rotary_pos_emb)

__all__ = ["RMSNorm", "LlamaRMSNorm", "SwiGLUMLP", "RotaryEmbedding", "Timesteps",
           "TimestepEmbedding", "MultiheadAttention", "resolve_backend", "kernelize_hub"]

# old ModelConfig.attn_backend names -> diffusers backends
_LEGACY_BACKENDS = {"auto": "native", "sdpa": "native", "math": "_native_math", "flash2": "flash"}


def llama_config(d: int, n_heads: int, ffn_hidden: int, rope_theta: float) -> LlamaConfig:
    """The few LlamaConfig fields the reused Llama layers read."""
    return LlamaConfig(hidden_size=d, num_attention_heads=n_heads, intermediate_size=ffn_hidden,
                       hidden_act="silu", mlp_bias=False,
                       rope_parameters={"rope_type": "default", "rope_theta": float(rope_theta)})


@use_kernel_forward_from_hub("SwiGLUMLP")
class SwiGLUMLP(LlamaMLP):
    """down_proj(silu(gate_proj(x)) * up_proj(x)), no bias."""


class RotaryEmbedding(nn.Module):
    """fp32 (cos, sin) [1,S,Dh] for positions 0..S-1, computed once per forward and shared by all layers."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.rope = LlamaRotaryEmbedding(config)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(seq_len, device=device)[None]
        return self.rope(torch.empty(0, device=device, dtype=torch.float32), pos)


def resolve_backend(name: str) -> AttentionBackendName:
    """Map a config name (diffusers backend, or a legacy auto/sdpa/math/flash2) to a diffusers backend."""
    name = str(name).lower()
    return AttentionBackendName(_LEGACY_BACKENDS.get(name, name))


def prepare_backend(backend: AttentionBackendName) -> None:
    """Check the backend's requirements and fetch its hub kernel, as ModelMixin.set_attention_backend does."""
    from diffusers.models.attention_dispatch import (_check_attention_backend_requirements,
                                                     _maybe_download_kernel_for_backend)
    _check_attention_backend_requirements(backend)
    _maybe_download_kernel_for_backend(backend)


def _dense_attn_mask(pad_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
    """pad_mask [B,S] bool (True=valid) -> [B,1,S,S] bool; the diagonal stays on so padded rows aren't all -inf."""
    eye = torch.eye(seq_len, dtype=torch.bool, device=pad_mask.device)
    return pad_mask[:, None, None, :] | eye[None, None, :, :]


class MultiheadAttention(nn.Module):
    """Bidirectional attention: fused qkv -> QK-Norm -> RoPE -> dispatched attention kernel -> out_proj."""

    def __init__(self, d: int, n_heads: int, qk_norm: bool = True, backend: str = "native") -> None:
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"d={d} is not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        # QK-Norm: RMSNorm over head_dim, weights shared across heads
        self.q_norm: Optional[LlamaRMSNorm] = LlamaRMSNorm(self.head_dim) if qk_norm else None
        self.k_norm: Optional[LlamaRMSNorm] = LlamaRMSNorm(self.head_dim) if qk_norm else None
        self.backend = resolve_backend(backend)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor],
                rotary: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """x [B,S,d]; pad_mask [B,S] bool (True=valid) or None (all valid); rotary = (cos, sin) [1,S,Dh] fp32."""
        B, S, d = x.shape
        q, k, v = self.qkv(x).view(B, S, 3, self.n_heads, self.head_dim).unbind(dim=2)  # [B,S,H,Dh]
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        cos, sin = rotary
        q, k = apply_rotary_pos_emb(q.float(), k.float(), cos, sin, unsqueeze_dim=2)  # RoPE in fp32
        q, k = q.to(v.dtype), k.to(v.dtype)
        if pad_mask is None:
            out = dispatch_attention_fn(q, k, v, backend=self.backend)
        else:
            # KiT pads after the registers, not at the end, so flash/sage varlen kernels don't apply
            out = dispatch_attention_fn(q, k, v, attn_mask=_dense_attn_mask(pad_mask, S),
                                        backend=AttentionBackendName.NATIVE)
        return self.out_proj(out.reshape(B, S, d))


def kernelize_hub(model: nn.Module, compile: bool = False, device: str = "cuda") -> None:
    """Swap transformers-registered layers (RMSNorm, SwiGLUMLP) for hub kernels; layers without a build keep PyTorch."""
    from kernels import Mode, kernelize
    from transformers.integrations.hub_kernels import register_kernel_mapping_transformers
    register_kernel_mapping_transformers()
    mode = Mode.INFERENCE | Mode.TORCH_COMPILE if compile else Mode.INFERENCE
    kernelize(model, mode=mode, device=device, use_fallback=True)
