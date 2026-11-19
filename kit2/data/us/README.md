# US data download (`USDownloader`)

`kit2/data/us/download.py` downloads raw US equity data from Alpha Vantage for KiT-2. It has five phases, and each one
skips work whose output already exists. If a run is interrupted, start the same command again and it picks up where it
stopped.

Bars are stored exactly as Alpha Vantage returns them. Prices are not adjusted, and nothing is converted to bar close
time or UTC; the bar builder does that later.

## Setup

1. Install the dependencies: `pandas` and `pyarrow` (only the standard library is used for HTTP).
2. Provide credentials. Values are read from the environment first, then from the first `.env` file found in this order:
    - the file named by `$KIT_ENV`
    - `~/.config/kit/.env`
    - `<repo>/.env`

| Key                    | Required        | Meaning                         |
|------------------------|-----------------|---------------------------------|
| `ALPHAVANTAGE_API_KEY` | yes             | premium key                     |
| `ALPHAVANTAGE_RPM`     | no (default 75) | your plan's requests per minute |

3. The data root is `$KIT2_DATA`, default `/workspace/kit2/data`. Override it with `--root`.

 ```bash
python -m kit2.data.us.download \
  --phases listings,daily,universe,corp,intraday \
  --top 1000 \
  --end 2026-11-17 \
  --workers 8
```

| Flag        | Default            | Meaning                                                                                     |
|-------------|--------------------|---------------------------------------------------------------------------------------------|
| `--phases`  | all five, in order | comma-separated subset of the phases                                                        |
| `--top`     | 1000               | universe size per month                                                                     |
| `--end`     | today              | last date covered                                                                           |
| `--root`    | `$KIT2_DATA`       | data root                                                                                   |
| `--workers` | 8                  | parallel requests; they overlap network latency, and the shared rate limiter caps the total |

The phases depend on each other, so run them in this order: `listings → daily → universe → corp / intraday`.

## Phases

| Phase      | What it does                                                                                                                | Requests (approx.)  |
|------------|-----------------------------------------------------------------------------------------------------------------------------|---------------------|
| `listings` | Fetches `LISTING_STATUS` for active and delisted symbols                                                                    | 2                   |
| `daily`    | Raw full-history daily bars for every candidate stock listed since 2016-01-01 (delisted ones included) plus 30 context ETFs | ~10k                |
| `universe` | Builds point-in-time monthly membership (see below); no API calls                                                           | 0                   |
| `corp`     | `SPLITS` and `DIVIDENDS` for every universe member and context ETF                                                          | 2 per symbol        |
| `intraday` | Raw 1-minute bars, extended hours included, one request per ticker-month                                                    | ~130k+ for top 1000 |

At 75 requests/min the `daily` phase takes about 2.5 h and `intraday` takes 30+ h. Only a higher Alpha Vantage tier
makes them faster.

### Which stocks are candidates

`assetType == Stock`, listed on NYSE, NASDAQ, NYSE MKT, AMEX, BATS or NYSE ARCA.
   - Symbol matches `^[A-Z]{1,5}(-[A-Z])?$`. Class shares such as `BRK-B` are kept.
   - Names that look like warrants, units, rights, preferred shares, depositary shares, notes, SPAC shells or when-issued listings are dropped.
   - The stock must have been listed at some point between 2016-01-01 and `--end`.

### Universe rules (no look-ahead)

 Membership for month M uses only daily bars dated **strictly before** the first day of M:
      - rank by the median dollar volume (raw close × volume) over the last 63 sessions;
      - require at least 63 sessions of history, a last close of at least $5, and a bar within
        the 10 days before M;
      - use listing and delisting dates only to bound the window a symbol was listed. They never
        decide membership.

Thirty context ETFs are always included: SPY, QQQ, IWM, DIA, the sector SPDRs, SMH, KRE, TLT/IEF/SHY, HYG/LQD, GLD/SLV/USO/UNG, UUP, EEM/EFA and VIXY.

The HyperSync scans are running slow: Ethereum chunks take about 5 minutes, Arbitrum about 11, and Base has sat in its pool-registry phase for 30 minutes. I'm checking whether HyperSync is rate-limiting the token and whether the Base process is still alive.