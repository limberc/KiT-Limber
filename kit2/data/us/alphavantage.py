"""Alpha Vantage client: rate-limited, retrying, returns parsed DataFrames.

Facts this relies on (probed 2026-11-17, see docs/superpowers/specs/2026-11-17-kit2-data-pipeline-design.md):
- TIME_SERIES_INTRADAY labels a bar by its START (regular day = 09:30 .. 15:59 for 1min), US/Eastern.
  We keep that label as `ts_start`; converting to close time happens once, in the bar builder.
- adjusted=false returns raw traded prices; splits/dividends come from SPLITS / DIVIDENDS with dates.
- Intraday data does not exist for delisted symbols ("Invalid API call"); daily data does.
"""
from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Optional

import pandas as pd

from kit2.env import load_env, secret


class NotCovered(Exception):
    """Alpha Vantage has no data for this request (e.g. intraday for a delisted symbol). Permanent."""


class RateLimiter:
    """Spaces request starts evenly so at most `rpm` start per minute, shared across threads."""

    def __init__(self, rpm: float) -> None:
        self.interval = 60.0 / rpm
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        time.sleep(max(0.0, start - now))


class AlphaVantage:
    def __init__(self, rpm: Optional[float] = 75, safety: float = 0.95) -> None:
        self.key = secret("ALPHAVANTAGE_API_KEY")
        rpm = float(rpm or load_env().get("ALPHAVANTAGE_RPM") or 75)
        self.limiter = RateLimiter(rpm * safety)
        self.n_requests = 0
        self.n_throttled = 0
        self.last_throttle_msg = ""
        self.n_net_errors = 0
        self.last_net_error = ""
        self.busy_s = 0.0  # summed request wall time, for mean latency
        self.URL = "https://www.alphavantage.co/query"

    def _get(self, params: Dict[str, str], retries: int = 6) -> str:
        q = urllib.parse.urlencode({**params, "apikey": self.key})
        delay = 5.0
        for attempt in range(retries):
            self.limiter.wait()
            self.n_requests += 1
            t0 = time.monotonic()
            try:
                with urllib.request.urlopen(f"{self.URL}?{q}", timeout=60) as r:
                    body = r.read().decode("utf-8")
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                err = repr(e)
                self.n_net_errors += 1
                self.last_net_error = err[:160]
                self.busy_s += time.monotonic() - t0
            else:
                self.busy_s += time.monotonic() - t0
                if not body.lstrip().startswith("{"):
                    return body  # CSV payload
                msg = json.loads(body)
                if "Error Message" in msg:
                    raise NotCovered(msg["Error Message"][:200])
                if "Information" in msg or "Note" in msg:  # throttled or a plan message
                    err = str(msg.get("Information") or msg.get("Note"))[:200]
                    self.n_throttled += 1
                    self.last_throttle_msg = err
                else:
                    return body  # JSON payload (splits, dividends, ...)
            if attempt == retries - 1:
                raise RuntimeError(f"Alpha Vantage request failed after {retries} tries: {err}")
            time.sleep(delay)
            delay = min(delay * 2, 120.0)
        raise AssertionError("unreachable")

    @staticmethod
    def _csv(body: str) -> pd.DataFrame:
        return pd.read_csv(io.StringIO(body))

    def listing_status(self, state: str) -> pd.DataFrame:
        """state = active | delisted. Columns: symbol, name, exchange, assetType, ipoDate, delistingDate, status."""
        return self._csv(self._get({"function": "LISTING_STATUS", "state": state}))

    def daily_raw(self, symbol: str) -> pd.DataFrame:
        """Unadjusted daily bars, full history. Columns: date, open, high, low, close, volume (ascending)."""
        df = self._csv(self._get({"function": "TIME_SERIES_DAILY", "symbol": symbol,
                                  "outputsize": "full", "datatype": "csv"}))
        df = df.rename(columns={"timestamp": "date"})
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def intraday_month(self, symbol: str, month: str, interval: str = "1min") -> pd.DataFrame:
        """Unadjusted bars incl. extended hours for YYYY-MM. Columns: ts_start (US/Eastern, naive), o/h/l/c/v."""
        df = self._csv(self._get({"function": "TIME_SERIES_INTRADAY", "symbol": symbol, "interval": interval,
                                  "month": month, "outputsize": "full", "adjusted": "false",
                                  "extended_hours": "true", "datatype": "csv"}))
        df = df.rename(columns={"timestamp": "ts_start"})
        df["ts_start"] = pd.to_datetime(df["ts_start"])
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype("float64")
        return df.sort_values("ts_start").reset_index(drop=True)

    def splits(self, symbol: str) -> pd.DataFrame:
        """Columns: effective_date, split_factor (new shares per old share)."""
        data = json.loads(self._get({"function": "SPLITS", "symbol": symbol})).get("data", [])
        df = pd.DataFrame(data, columns=["effective_date", "split_factor"])
        df["effective_date"] = pd.to_datetime(df["effective_date"])
        df["split_factor"] = df["split_factor"].astype("float64")
        return df.sort_values("effective_date").reset_index(drop=True)

    def dividends(self, symbol: str) -> pd.DataFrame:
        """Columns: ex_dividend_date, declaration_date, record_date, payment_date, amount."""
        data = json.loads(self._get({"function": "DIVIDENDS", "symbol": symbol})).get("data", [])
        cols = ["ex_dividend_date", "declaration_date", "record_date", "payment_date", "amount"]
        df = pd.DataFrame(data, columns=cols)
        for c in cols[:-1]:
            df[c] = pd.to_datetime(df[c], errors="coerce")  # 'None' -> NaT
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        return df.sort_values("ex_dividend_date").reset_index(drop=True)
