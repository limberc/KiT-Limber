"""Model hyperparameters. Presets: `dev` (~40M) for building on aa3, `main` (~250M) for RunPod runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 512
    n_heads: int = 8
    enc_layers: int = 8          # each: time attention + group attention + SwiGLU
    dec_layers: int = 4          # each: self + group + cross attention + SwiGLU, AdaLN-modulated
    ffn_hidden: int = 1408       # SwiGLU hidden size (~8/3 d, multiple of 64)
    patch: int = 8               # bars per token
    in_channels: int = 10        # 5 candle features + has_obs + 4 aux (history only)
    out_channels: int = 5        # generated: the 5 candle features
    cal_dim: int = 12            # calendar features per token (known in advance)
    n_markets: int = 4           # slot 0 = null
    n_chains: int = 8            # 0 = none (US) / null
    n_scales: int = 8            # 0 = null; 1..7 = 1m..1d
    inst_vocab: int = 8192       # 0 = unknown / dropped instrument
    max_horizon_patches: int = 16
    rope_theta: float = 10000.0
    t_emb_dim: int = 256
    inst_dropout: float = 0.1    # training only: instrument id -> 0, so unseen instruments work

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


PRESETS: Dict[str, ModelConfig] = {
    "tiny": ModelConfig(d_model=128, n_heads=4, enc_layers=2, dec_layers=2, ffn_hidden=384),   # tests
    "dev": ModelConfig(),                                                                    # ~60M
    "main": ModelConfig(d_model=1024, n_heads=16, enc_layers=12, dec_layers=6, ffn_hidden=2816),   # ~330M
}


def preset(name: str, **overrides: Any) -> ModelConfig:
    return replace(PRESETS[name], **overrides)
