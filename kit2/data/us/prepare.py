"""Build the US training store from raw Alpha Vantage files (see kit2/data/store.py for the format).

Usage: python -m kit2.data.us.prepare [--start 2023-10] [--scales 1m,...,1d] [--workers 32]

Per symbol: intraday bars for every planned month that has been downloaded (coverage = the download plan,
fixed before each month), daily bars from the survivorship-free daily file, candle features + extended-hours
aux. Normalisation stats are fit on bars with ts <= TRAIN_END only. Sampling sets (active.parquet) are the
point-in-time monthly universe plus the context ETFs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from kit2.data.features import candle_features
from kit2.data.store import CAL_DIM, CAL_FIRST, CHANNELS, TRAIN_END, calendar_features, fit_stats, write_scale
from kit2.data.us.bars import ET, INTRADAY_MIN, build_daily, build_intraday, create_cal
from kit2.data.us.universe import CONTEXT_ETFS
from kit2.env import data_root

SCALES = ("1m", "5m", "15m", "30m", "1h", "2h", "1d")
AUX = ("aux_pre_ret", "aux_post_ret", "aux_pre_ldv", "aux_post_ldv")
HALF_LIFE_DAYS = 5.0
DAILY_LOOKBACK_DAYS = 420          # 1d history kept before --start (32 coarse patches x 8 bars + EMA warm-up)


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def us_grid(scale: str, start: str, end: str):
    """Bar close times on the XNYS grid (naive UTC ns) + calendar features."""
    cal = create_cal()
    cal = cal[(cal["session"] >= start) & (cal["close"] <= pd.Timestamp(end))]   # sessions closed by `end`
    if scale == "1d":
        ts = cal["close"].to_numpy("datetime64[ns]")
        pos, n = np.zeros(len(ts)), np.ones(len(ts))
    else:
        step = INTRADAY_MIN[scale]
        nb = -(-cal["minutes"].to_numpy() // step)
        j = np.concatenate([np.arange(k) for k in nb])
        opn = np.repeat(cal["open"].to_numpy("datetime64[ns]"), nb)
        ln = np.repeat(cal["minutes"].to_numpy(), nb)
        ts = opn + (np.minimum((j + 1) * step, ln) * 60_000_000_000).astype("timedelta64[ns]")
        pos, n = j.astype(float), np.repeat(nb, nb).astype(float)
    local = pd.DatetimeIndex(ts).tz_localize("UTC").tz_convert(ET)
    minute = (local.hour * 60 + local.minute).to_numpy().astype(float)
    return ts.astype(np.int64), calendar_features(ts, minute, pos, n)


def bars_per_day(scale: str) -> int:
    return 1 if scale == "1d" else -(-390 // INTRADAY_MIN[scale])


def channels(bars: pd.DataFrame, scale: str) -> np.ndarray:
    f = candle_features(bars, HALF_LIFE_DAYS * bars_per_day(scale))
    has = (bars["n_min"].fillna(1.0) > 0).astype(float).to_numpy()
    aux = bars[list(AUX)].to_numpy(np.float64)
    return np.column_stack([f.to_numpy(np.float64), has, aux]).astype(np.float32)


def _read(p: Path) -> Optional[pd.DataFrame]:
    return pd.read_parquet(p) if p.is_file() else None


def build_symbol(job) -> Dict[str, Optional[tuple]]:
    """All scales for one symbol -> {scale: (raw [n, C], ts ns [n], seg [n])} written to tmp npz files."""
    sym, idx, months, raw_root, tmp, start, scales = job
    out = {}
    splits = _read(raw_root / "corp" / "splits" / f"{sym}.parquet")
    divs = _read(raw_root / "corp" / "dividends" / f"{sym}.parquet")
    files = [raw_root / "intraday_1m" / sym / f"{m:%Y-%m}.parquet" for m in months]
    files = [f for f in files if f.is_file()]
    cover = [pd.Period(f.stem, "M") for f in files]
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else None
    # snapshot time: the newest month file was written then; no bars may be emitted after it
    asof = (pd.Timestamp(max(f.stat().st_mtime for f in files), unit="s") if files else None)
    for scale in scales:
        if scale == "1d":
            f = raw_root / "daily" / f"{sym}.parquet"
            d = _read(f)
            if d is None:
                continue
            d = d[pd.to_datetime(d["date"]) >= pd.Timestamp(start) - pd.Timedelta(days=DAILY_LOOKBACK_DAYS)]
            bars = build_daily(d, splits, divs)
            # a daily file fetched during a session ends with a partial day stamped at the close: drop it
            bars = bars[bars["ts"] <= pd.Timestamp(f.stat().st_mtime, unit="s")].reset_index(drop=True)
        else:
            if raw is None:
                continue
            bars = build_intraday(raw, scale, cover, splits, divs, asof=asof)
        if len(bars) < 2:
            continue
        np.savez(tmp / scale / f"{idx}.npz", raw=channels(bars, scale),
                 ts=bars["ts"].to_numpy("datetime64[ns]").astype(np.int64), seg=bars["seg"].to_numpy(np.int64))
        out[scale] = len(bars)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-10")
    ap.add_argument("--end", default=pd.Timestamp.now(tz="UTC").tz_localize(None).isoformat())
    ap.add_argument("--scales", default=",".join(SCALES))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--root", type=Path, default=data_root())
    ap.add_argument("--stats-sample", type=int, default=300_000, help="training rows per scale used for stats")
    ap.add_argument("--symbols", default="", help="comma list: build only these (smoke tests), into --out-name")
    ap.add_argument("--out-name", default="us", help="store directory name under <root>/store/")
    args = ap.parse_args()
    scales = args.scales.split(",")
    raw_root, store = args.root / "raw" / "us", args.root / "store" / args.out_name
    tmp = store / "_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    for s in scales:
        (tmp / s).mkdir(parents=True)

    plan = pd.read_parquet(raw_root / "universe" / "intraday_plan_top1000.parquet")
    plan = plan[plan["month"] >= pd.Timestamp(args.start)]
    uni = pd.read_parquet(raw_root / "universe" / "monthly_top1000.parquet")
    uni = uni[uni["month"] >= pd.Timestamp(args.start)]
    syms = sorted(set(plan["symbol"]) | set(uni["symbol"]))
    if args.symbols:
        syms = [s for s in syms if s in set(args.symbols.split(","))]
    inst = pd.DataFrame({"idx": np.arange(len(syms)), "key": syms, "vocab": np.arange(len(syms)) + 1,
                         "market": 1, "chain": 0})
    store.mkdir(parents=True, exist_ok=True)
    inst.to_parquet(store / "instruments.parquet")
    idx_of = dict(zip(syms, inst["idx"]))
    act = pd.concat([uni[["month", "symbol"]],
                     pd.DataFrame([(m, s) for m in sorted(uni["month"].unique()) for s in CONTEXT_ETFS],
                                  columns=["month", "symbol"])]).drop_duplicates()
    act = act[act["symbol"].isin(idx_of)]
    pd.DataFrame({"month": act["month"].to_numpy(), "idx": act["symbol"].map(idx_of).to_numpy()}).to_parquet(
        store / "active.parquet")

    months_of = plan.groupby("symbol")["month"].apply(lambda m: sorted(set(m))).to_dict()
    jobs = [(s, idx_of[s], months_of.get(s, []), raw_root, tmp, args.start, scales) for s in syms]
    log(f"{len(jobs)} symbols, scales {scales}, start {args.start}")
    t0, counts = time.time(), {s: 0 for s in scales}
    with ProcessPoolExecutor(args.workers) as ex:
        for i, res in enumerate(ex.map(build_symbol, jobs, chunksize=4), 1):
            for s, n in res.items():
                counts[s] += n
            if i % 200 == 0 or i == len(jobs):
                log(f"built {i}/{len(jobs)} symbols, {time.time() - t0:.0f}s, bars so far {counts}")

    # normalisation stats from training-period bars only, then write each scale
    stats, rng = {}, np.random.default_rng(0)
    train_end_ns = TRAIN_END.value
    for s in scales:
        files = sorted((tmp / s).glob("*.npz"), key=lambda p: int(p.stem))
        grid, cal = us_grid(s, str((pd.Timestamp(args.start) - pd.Timedelta(days=DAILY_LOOKBACK_DAYS if s == "1d" else 0)).date()),
                            args.end)
        first_of = cal[:, CAL_FIRST] > 0.5
        pick = rng.choice(len(files), size=min(len(files), 400), replace=False)
        samp, firsts = [], []
        for k in pick:
            z = np.load(files[k])
            keep = np.flatnonzero(z["ts"] <= train_end_ns)
            keep = rng.choice(keep, size=min(len(keep), args.stats_sample // 400 + 1), replace=False) if len(keep) else keep
            samp.append(z["raw"][keep])
            firsts.append(first_of[np.searchsorted(grid, z["ts"][keep])])
        stats[s] = fit_stats(np.concatenate(samp), np.concatenate(firsts))

        def parts(files=files, grid=grid):
            for f in files:
                z = np.load(f)
                g = np.searchsorted(grid, z["ts"])
                ok = (g < len(grid)) & (grid[np.minimum(g, len(grid) - 1)] == z["ts"])
                if not ok.all():
                    raise AssertionError(f"{f}: {int((~ok).sum())} bars are not on the {s} grid")
                yield int(f.stem), z["raw"], g, z["seg"]

        n = write_scale(store / s, grid, cal, parts(), stats[s])
        log(f"wrote {s}: {n} bars, {len(files)} instruments, grid {len(grid)}")
    (store / "stats.json").write_text(json.dumps({"train_end": str(TRAIN_END), "channels": list(CHANNELS),
                                                  "cal_dim": CAL_DIM, "scales": stats}, indent=1))
    shutil.rmtree(tmp, ignore_errors=True)
    log("done")


if __name__ == "__main__":
    main()
