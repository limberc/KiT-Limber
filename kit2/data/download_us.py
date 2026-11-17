"""Resumable US downloader (Alpha Vantage). Every phase skips work whose output already exists.

Usage: python -m kit2.data.download_us [--phases listings,daily,universe,corp,intraday] [--top 1000]

Layout under <data_root>/raw/us/:
  listing/{active,delisted}.csv          listing snapshot (download date in listing/meta.json)
  daily/<SYM>.parquet                    raw daily bars (all candidates, incl. delisted)
  universe/monthly_top<N>.parquet        point-in-time membership (month, symbol, rank, adv)
  corp/{splits,dividends}/<SYM>.parquet  corporate actions with their dates
  intraday_1m/<SYM>/<YYYY-MM>.parquet    raw 1min bars incl. extended hours, labelled by bar START
  */_markers/                            JSON markers for requests that can never succeed
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Sequence

import pandas as pd

from kit2.data.io import atomic_write_bytes, atomic_write_parquet, write_marker
from kit2.data.sources import AlphaVantage, NotCovered
from kit2.data.us_universe import CONTEXT_ETFS, candidates, listing_windows, monthly_universe
from kit2.env import data_root

START = "2016-01-01"
WARMUP_MONTHS = 3  # 1min history fetched before a stock enters the universe (2h scale needs ~90 days)


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def run_jobs(name: str, jobs: Sequence, fn: Callable, workers: int, av: AlphaVantage) -> None:
    """Run fn(job) on a thread pool; each fn does its own skip check and error handling.

    Progress reports requests/min over the last interval (skipped jobs cost no request)."""
    t0, done = time.time(), 0
    t_last, req_last, thr_last, err_last, busy_last = t0, av.n_requests, av.n_throttled, av.n_net_errors, av.busy_s
    log(f"[{name}] {len(jobs)} jobs")
    with ThreadPoolExecutor(workers) as ex:
        for _ in ex.map(fn, jobs):
            done += 1
            if done % 100 == 0 or done == len(jobs):
                now = time.time()
                rpm = (av.n_requests - req_last) / max(now - t_last, 1e-9) * 60
                jpm = done / max(now - t0, 1e-9) * 60
                thr, err = av.n_throttled - thr_last, av.n_net_errors - err_last
                n_req = max(av.n_requests - req_last, 1)
                log(f"[{name}] {done}/{len(jobs)}  {rpm:.0f} req/min, mean latency "
                    f"{(av.busy_s - busy_last) / n_req:.1f}s, {thr} throttled, {err} net errors  "
                    f"ETA {(len(jobs) - done) / max(rpm, 1e-9) / 60:.1f}h"
                    + (f"  last throttle: {av.last_throttle_msg[:120]!r}" if thr else "")
                    + (f"  last error: {av.last_net_error!r}" if err else ""))
                t_last, req_last, thr_last, err_last, busy_last = (now, av.n_requests, av.n_throttled,
                                                                   av.n_net_errors, av.busy_s)


class USDownloader:
    def __init__(self, root: Path, top_n: int, end: str, workers: int = 8) -> None:
        self.root = root / "raw" / "us"
        self.workers = workers  # requests overlap ~2-6 s latency; the rate limiter caps the total
        self.top_n = top_n
        self.end = pd.Timestamp(end)
        self.av = AlphaVantage()

    # ---- listings ------------------------------------------------------------------------------
    def listings(self) -> None:
        d = self.root / "listing"
        for state in ("active", "delisted"):
            if not (d / f"{state}.csv").is_file():
                df = self.av.listing_status(state)
                atomic_write_bytes(d / f"{state}.csv", df.to_csv(index=False).encode())
                log(f"[listings] {state}: {len(df)} rows")
        if not (d / "meta.json").is_file():
            atomic_write_bytes(d / "meta.json", json.dumps({"downloaded": time.strftime("%Y-%m-%d")}).encode())

    def _candidates(self) -> pd.DataFrame:
        d = self.root / "listing"
        return candidates(pd.read_csv(d / "active.csv"), pd.read_csv(d / "delisted.csv"),
                          START, str(self.end.date()))

    # ---- daily ---------------------------------------------------------------------------------
    def _daily_one(self, sym: str) -> None:
        out, mark = self.root / "daily" / f"{sym}.parquet", self.root / "daily" / "_markers" / f"{sym}.json"
        if out.is_file() or mark.is_file():
            return
        try:
            atomic_write_parquet(self.av.daily_raw(sym), out)
        except NotCovered as e:
            write_marker(mark, {"symbol": sym, "error": str(e)})
        except Exception as e:  # transient after retries: leave for the next run
            log(f"[daily] {sym} failed: {e!r}")

    def daily(self) -> None:
        syms = sorted(set(self._candidates()["symbol"]) | set(CONTEXT_ETFS))
        run_jobs("daily", syms, self._daily_one, self.workers, self.av)

    # ---- universe ------------------------------------------------------------------------------
    def universe_path(self) -> Path:
        return self.root / "universe" / f"monthly_top{self.top_n}.parquet"

    def universe(self) -> None:
        cands = self._candidates()
        win = listing_windows(cands)
        daily = {}
        for sym in win:
            p = self.root / "daily" / f"{sym}.parquet"
            if p.is_file():
                daily[sym] = pd.read_parquet(p, columns=["date", "close", "volume"])
        months = pd.date_range(START, self.end, freq="MS")
        uni = monthly_universe(daily, win, months, top_n=self.top_n)
        atomic_write_parquet(uni, self.universe_path())
        log(f"[universe] {uni['symbol'].nunique()} symbols over {uni['month'].nunique()} months "
            f"(daily coverage {len(daily)}/{len(win)} candidates)")

    def _union_symbols(self) -> List[str]:
        uni = pd.read_parquet(self.universe_path())
        return sorted(set(uni["symbol"]) | set(CONTEXT_ETFS))

    # ---- corporate actions ---------------------------------------------------------------------
    def _corp_one(self, job) -> None:
        kind, sym = job
        out, mark = self.root / "corp" / kind / f"{sym}.parquet", self.root / "corp" / kind / "_markers" / f"{sym}.json"
        if out.is_file() or mark.is_file():
            return
        try:
            df = self.av.splits(sym) if kind == "splits" else self.av.dividends(sym)
            atomic_write_parquet(df, out)
        except NotCovered as e:
            write_marker(mark, {"symbol": sym, "error": str(e)})
        except Exception as e:
            log(f"[corp] {kind} {sym} failed: {e!r}")

    def corp(self) -> None:
        syms = self._union_symbols()
        run_jobs("corp", [(k, s) for k in ("splits", "dividends") for s in syms], self._corp_one, self.workers, self.av)

    # ---- intraday ------------------------------------------------------------------------------
    def intraday_plan(self) -> pd.DataFrame:
        """(symbol, month, priority) ticker-months to fetch: membership months + warm-up, newest first."""
        uni = pd.read_parquet(self.universe_path())
        all_months = pd.date_range(START, self.end, freq="MS")
        rows = []
        for sym, g in uni.groupby("symbol"):
            member = set(g["month"])
            want = set()
            for m in member:
                want.update(pd.date_range(m - pd.DateOffset(months=WARMUP_MONTHS), m, freq="MS"))
            best = int(g["rank"].min())
            rows += [(sym, m, best) for m in want if all_months[0] <= m <= all_months[-1]]
        rows += [(sym, m, 0) for sym in CONTEXT_ETFS for m in all_months]
        plan = pd.DataFrame(rows, columns=["symbol", "month", "rank"]).drop_duplicates(["symbol", "month"])
        return plan.sort_values(["month", "rank"], ascending=[False, True]).reset_index(drop=True)

    def _intraday_one(self, job) -> None:
        sym, month = job
        d = self.root / "intraday_1m" / sym
        tag = month.strftime("%Y-%m")
        out, mark, sym_mark = d / f"{tag}.parquet", d / "_markers" / f"{tag}.json", d / "_not_covered.json"
        current = tag == pd.Timestamp.now().strftime("%Y-%m")
        if sym_mark.is_file() or mark.is_file():
            return
        if out.is_file() and not (current and time.time() - out.stat().st_mtime > 6 * 3600):
            return  # the running month is refreshed when stale
        try:
            atomic_write_parquet(self.av.intraday_month(sym, tag), out)
        except NotCovered as e:
            have_any = d.is_dir() and any(d.glob("*.parquet"))
            # no intraday for this symbol at all (delisted names): mark once, skip its other months
            write_marker(mark if have_any else sym_mark, {"symbol": sym, "month": tag, "error": str(e)})
        except Exception as e:
            log(f"[intraday] {sym} {tag} failed: {e!r}")

    def intraday(self) -> None:
        plan = self.intraday_plan()
        atomic_write_parquet(plan, self.root / "universe" / f"intraday_plan_top{self.top_n}.parquet")
        run_jobs("intraday", list(zip(plan["symbol"], plan["month"])), self._intraday_one, self.workers, self.av)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download US raw data from Alpha Vantage (resumable)")
    ap.add_argument("--phases", default="listings,daily,universe,corp,intraday")
    ap.add_argument("--top", type=int, default=1000, help="universe size per month")
    ap.add_argument("--end", default=pd.Timestamp.now().strftime("%Y-%m-%d"))
    ap.add_argument("--root", type=Path, default=data_root())
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    dl = USDownloader(args.root, args.top, args.end, args.workers)
    for phase in args.phases.split(","):
        log(f"=== phase {phase}")
        getattr(dl, phase)()
    log(f"done; {dl.av.n_requests} requests this run")


if __name__ == "__main__":
    main()
