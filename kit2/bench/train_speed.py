"""Training throughput / MFU benchmark on synthetic data with the real bucket shapes (stage A on aa3, stage B
on rented GPUs). Runs the real Lightning training loop (bf16-mixed, torch.compile, fused AdamW).

Usage: python -m kit2.bench.train_speed --preset dev --buckets us:1m,us:15m,us:1d --batch-tokens 65536
Prints one JSON line per bucket and a summary; --out writes them to a file.
MFU is reported against the measured bf16 matmul peak of this GPU (and against --peak-tflops if given).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before CUDA initialises

import lightning as L
import torch
from torch.utils.flop_counter import FlopCounterMode

from kit2.model import KiT2, flow_loss
from kit2.train.batch import model_inputs
from kit2.train import KiT2Lit, SyntheticData, Bucket
from kit2.train.lit import batch_size_for


def matmul_peak_tflops(n: int = 8192, iters: int = 30) -> float:
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    return 2 * n ** 3 * iters / (time.perf_counter() - t) / 1e12


class StepTimer(L.Callback):
    def __init__(self, warmup: int) -> None:
        self.warmup, self.times, self._t = warmup, [], None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        torch.cuda.synchronize()
        self._t = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        torch.cuda.synchronize()
        if batch_idx >= self.warmup:
            self.times.append(time.perf_counter() - self._t)


def flops_per_step(lit: KiT2Lit, b) -> float:
    """Forward + backward FLOPs of one training step, counted on a fresh eager copy (bf16 autocast, GPU)."""
    model = KiT2(lit.cfg).cuda().to(lit.model_dtype()).train()
    with FlopCounterMode(display=False) as fc, torch.autocast("cuda", dtype=torch.bfloat16):
        flow_loss(model, model_inputs(b), b["x0"], b["valid"]).backward()
    del model
    return float(fc.get_total_flops())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="dev")
    ap.add_argument("--buckets", default="us:1m,us:15m,us:1h,us:1d")
    ap.add_argument("--G", type=int, default=32)
    ap.add_argument("--A", type=int, default=8)
    ap.add_argument("--batch-tokens", type=int, default=65536)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--precision", default="bf16-mixed", help="bf16-mixed (bf16 compute, fp32 master) or bf16-true")
    ap.add_argument("--peak-tflops", type=float, default=0.0, help="vendor dense bf16 peak, for a second MFU figure")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")

    gpu = torch.cuda.get_device_name()
    peak = matmul_peak_tflops()
    head = {"gpu": gpu, "preset": args.preset, "matmul_peak_tflops": round(peak, 1), "torch": torch.__version__,
            "precision": args.precision}
    print(json.dumps(head), flush=True)
    rows = []
    for spec in args.buckets.split(","):
        market, scale = spec.split(":")
        bk = Bucket(market, scale)
        lit = KiT2Lit({"preset": args.preset}, {"lr": 3e-4, "warmup_steps": 10, "total_steps": 1000},
                      compile=not args.no_compile, bucket_names=[bk.name])
        dm = SyntheticData([bk], lit.cfg, G=args.G, A=args.A, batch_tokens=args.batch_tokens)
        timer = StepTimer(args.warmup)
        torch.cuda.reset_peak_memory_stats()
        trainer = L.Trainer(accelerator="gpu", devices=1, precision=args.precision, max_steps=args.warmup + args.steps,
                            logger=False, enable_checkpointing=False, enable_progress_bar=False,
                            enable_model_summary=False, callbacks=[timer], num_sanity_val_steps=0)
        t0 = time.perf_counter()
        trainer.fit(lit, datamodule=dm)
        wall = time.perf_counter() - t0
        peak_mem = torch.cuda.max_memory_allocated() / 2 ** 30  # before the eager FLOP-count pass
        b = dm._cache[0]
        flops = flops_per_step(lit, b)
        step = sum(timer.times) / len(timer.times)
        B = batch_size_for(bk, args.G, args.batch_tokens)
        enc_tok = B * bk.tokens(args.G)
        dec_tok = B * args.A * args.G * bk.horizon
        row = {**head, "bucket": bk.name, "B": B, "G": args.G, "A": args.A, "enc_tokens": enc_tok,
               "dec_tokens": dec_tok, "step_ms": round(step * 1000, 1),
               "enc_tokens_per_s": round(enc_tok / step), "bars_per_s": round(enc_tok * 8 / step),
               "gflop_per_step": round(flops / 1e9, 1), "tflops": round(flops / step / 1e12, 1),
               "mfu_vs_matmul": round(flops / step / 1e12 / peak, 3),
               "mfu_vs_vendor": round(flops / step / 1e12 / args.peak_tflops, 3) if args.peak_tflops else None,
               "peak_mem_gb": round(peak_mem, 1),
               "params_m": round(sum(p.numel() for p in lit.model.parameters()) / 1e6, 1),
               "compile_and_run_s": round(wall, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        # release this bucket's model, optimizer, batch and compiled graphs before the next one
        del lit, dm, trainer, timer, b
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
    if args.out:
        args.out.write_text("\n".join(json.dumps(r) for r in [head] + rows) + "\n")


if __name__ == "__main__":
    main()
