"""KiT-2 vs KiT evaluation on the held-out test period (after VAL_END), same instruments and moments for both.

Steps (python -m kit2.eval.<step>):
  plan    test moments per (market, scale) + a fixed instrument panel per market       -> plan_<market>.json
  bars    point-in-time bars of the panel, rebuilt with the store's own builders       -> bars/<market>/<idx>.npz
  kit     KiT (step_market_284M) forecasts: K=16 paths, 32 Euler steps (as published)  -> pred/kit_<market>.npz
  kit2    KiT-2 forecasts from a checkpoint: K=16 paths, 8 Heun steps                  -> pred/kit2_<market>.npz
  score   RankIC (return, volatility), CRPS and 80% coverage of the horizon return; historical bootstrap baseline
"""
from pathlib import Path

EVAL_SCALES = ("5m", "15m", "1h", "1d")
# KiT's forecast horizon per scale (bars); both models are scored on these bars only
KIT_LH = {"1m": 120, "5m": 96, "15m": 64, "30m": 40, "1h": 20, "2h": 20, "1d": 20}
KIT_LC = {"1m": 1200, "5m": 960, "15m": 800, "30m": 640, "1h": 400, "2h": 360, "1d": 250}
OUT = Path("/root/kit2runs/eval")
