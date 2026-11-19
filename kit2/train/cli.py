"""KiT-2 training entry point (Lightning CLI).

Usage: python -m kit2.train.cli fit --config configs/kit2/dev_stage_a.yaml [--trainer.max_steps 2000 ...]
       python -m kit2.train.cli validate --config ... --ckpt_path <ckpt>
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from lightning.pytorch.cli import LightningCLI  # noqa: E402

from kit2.train.data import StoreData  # noqa: E402
from kit2.train.lit import KiT2Lit  # noqa: E402


def main() -> None:
    torch.set_float32_matmul_precision("high")
    LightningCLI(KiT2Lit, StoreData, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
