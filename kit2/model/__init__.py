"""KiT-2 model: patched multi-scale history encoder + flow-matching denoiser."""
from kit2.model.config import ModelConfig
from kit2.model.kit2 import KiT2
from .kit2 import conditioning_subset, flow_loss

__all__ = ["KiT2", "ModelConfig", "conditioning_subset", "flow_loss"]
