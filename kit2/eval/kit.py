"""KiT baseline forecasts for the evaluation plan (checkpoint step_market_284M: A-shares + US + crypto, trained
through 2025-12-31, so the whole test period is unseen). Published sampler: K=16 paths, 32 Euler steps, bf16.

KiT reads OHLCV and computes its own features. We give it a price series rebuilt from our point-in-time candle
features (split/dividend-adjusted gaps, segment starts with zero gap) and dollar volume, so both models see the
same history. Timestamps follow KiT's conventions: US bars in exchange-local time on the XNYS calendar, crypto
in UTC with every calendar day a trading day.

KiT is not part of this repo: $KIT_ROOT (default /root/kit_hf) is a KiT v1 checkout with the `kit_v1` package (this
repo up to commit 1113d69) and the checkpoint under ckpt/step_market_284M/.

Usage: KIT_ROOT=<kit checkout> python -m kit2.eval.kit --market us
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kit2.eval import KIT_LC, KIT_LH, OUT

KIT_ROOT = Path(os.environ.get("KIT_ROOT", "/root/kit_hf"))
CKPT = KIT_ROOT / "ckpt" / "step_market_284M" / "step_market_284M.pt"
# on-chain tokens KiT knows under a CEX ticker
CRYPTO_ALIAS = {"WETH": "ETHUSDT", "WBTC": "BTCUSDT", "cbBTC": "BTCUSDT"}


def kit_frame(z, scale: str, market: str, t_ns: int, keep: int) -> pd.DataFrame:
    """Synthetic adjusted OHLCV from raw candle features, up to and including the bar closing at t_ns."""
    ts = z[f"{scale}_ts"]
    n = int(np.searchsorted(ts, t_ns, side="right"))
    lo = max(0, n - keep)
    f = z[f"{scale}_feat"][lo:n]
    gap, body, up, dn = f[:, 0], f[:, 1], f[:, 2], f[:, 3]
    logc = np.cumsum(gap + body)
    o = np.exp(logc - body)
    c = np.exp(logc)
    t = pd.DatetimeIndex(ts[lo:n])
    if market == "us":
        t = t.tz_localize("UTC").tz_convert("America/New_York").tz_localize(None)
    return pd.DataFrame({"datetime": t, "open": o, "high": np.maximum(o, c) * np.exp(up),
                         "low": np.minimum(o, c) * np.exp(-dn), "close": c, "volume": z[f"{scale}_dvol"][lo:n]})


def kit_ids(market: str, keys, instruments: dict):
    out = []
    for key in keys:
        if market == "crypto":
            from kit2.data.crypto.hypersync_dex import CHAINS
            chain, base = key.split(":")
            tok = CHAINS[chain].quotes.get(base)
            key = CRYPTO_ALIAS.get(tok.symbol) if tok is not None else None
        meta = instruments.get(key or "", {})
        out.append((int(meta.get("sector", 0)), int(meta.get("vocab_index", 0))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True)
    ap.add_argument("--paths", type=int, default=16)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, str(KIT_ROOT))
    from kit_v1.data.calendar import bars_per_day as kit_bars_per_day
    from kit_v1.data.normalize import load_stats
    from kit_v1.infer import denormalize_paths, flow_sample, load_model_and_config, prepare_inference_window

    plan = json.loads((OUT / f"plan_{args.market}.json").read_text())
    market = args.market
    model, cfg = load_model_and_config(str(CKPT), "tiny", 0)
    model.cuda().eval()
    torch._dynamo.config.recompile_limit = 64
    model = torch.compile(model, dynamic=False)
    stats = load_stats(CKPT.parent / "stats.json")
    instruments = json.loads((CKPT.parent / "instruments.json").read_text())
    if market == "us":
        from kit2.data.us.bars import create_cal
        tdays = create_cal()["session"].to_numpy("datetime64[D]")
    else:
        tdays = np.arange(np.datetime64("2022-01-01"), np.datetime64("2027-12-31"), dtype="datetime64[D]")
    ids = kit_ids(market, plan["keys"], instruments)
    known = sum(v > 0 for _, v in ids)
    print(f"KiT {market}: {len(plan['panel'])} instruments ({known} in KiT's vocabulary), "
          f"{len(plan['moments'])} moments", flush=True)
    bars = {i: np.load(OUT / "bars" / market / f"{i}.npz") for i in plan["panel"]}
    out = OUT / "pred"
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    t_start = time.time()
    for mi, mo in enumerate(plan["moments"]):
        s, t_ns = mo["scale"], mo["t_ns"]
        Lc, Lh = KIT_LC[s], KIT_LH[s]
        # as in KiT's eval/infer_for_eval.py: enough bars before the window for KiT's volume EMA to warm up
        keep = Lc + int(math.ceil(12.0 * cfg.data.vol_ema_half_life_days * kit_bars_per_day(market, s))) + 8
        wins, used = [], []
        for (sector, vocab), i in zip(ids, plan["panel"]):
            z = bars[i]
            if f"{s}_ts" not in z or not np.any(z[f"{s}_ts"] == t_ns):
                continue
            df = kit_frame(z, s, market, t_ns, keep)
            end = df["datetime"].iloc[-1]
            win = prepare_inference_window(df, market, s, Lc=Lc, Lh=Lh, stats=stats, soft_clip_c=cfg.data.soft_clip_c,
                                           half_life_days=cfg.data.vol_ema_half_life_days, end_ts=end,
                                           sector_id=sector, inst_id=vocab, trading_days=tdays)
            wins.append(win)
            used.append(i)
        x = torch.cat([w["x_ctx"] for w in wins])
        cond = {k: torch.cat([w["cond"][k] for w in wins]) for k in wins[0]["cond"]}
        if bool(cond["pad_mask"].all()):
            del cond["pad_mask"]
        g = torch.Generator().manual_seed(args.seed + mi)
        paths = flow_sample(model, x, cond, Lh=Lh, steps=args.steps, solver="euler", n_paths=args.paths,
                            generator=g, device=torch.device("cuda"), chunk_size=args.chunk, precision="bf16")
        feats = denormalize_paths(paths, market, s, stats, c=cfg.data.soft_clip_c)          # [N, K, Lh, 5]
        res[f"{mi}_inst"] = np.array(used)
        res[f"{mi}_ret"] = np.asarray(feats[..., 0] + feats[..., 1], dtype=np.float32)
        el = time.time() - t_start
        print(f"[{mi + 1}/{len(plan['moments'])}] {s} {mo['t']} n={len(used)} "
              f"{el / 60:.1f} min elapsed, ETA {el / (mi + 1) * (len(plan['moments']) - mi - 1) / 60:.0f} min", flush=True)
        np.savez(out / f"kit_{market}.npz", **res)


if __name__ == "__main__":
    main()
