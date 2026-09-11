#!/usr/bin/env python3
"""
Download REAL historical OHLCV from Binance public data archives
(data.binance.vision) — monthly ZIP + daily ZIP for the current partial month,
with SHA256 checksum verification, dedup, gap detection and OHLC validation.

This is the production-grade historical source requested for the BTC-vs-DOGE
scientific experiment (5+ years). It does NOT use the trading API (no rate limits).

Env vars (all optional; defaults download BTCUSDT full history):
    ADAN_BV_SYMBOL     e.g. "DOGEUSDT"   (default "BTCUSDT")
    ADAN_BV_INTERVAL   e.g. "5m"         (default "5m")
    ADAN_BV_START      first month "YYYY-MM" (default = symbol's first available)
    ADAN_BV_END        last  month "YYYY-MM" (default = current month)
    ADAN_BV_OUTPUT     output base dir   (default "data/raw/<SYMBOL>")
    ADAN_BV_WORKERS    parallel downloads (default 8)

Output:
    <OUTPUT>/<interval>/<SYMBOL>_<interval>_raw.parquet   (concatenated, clean)
    <OUTPUT>/<interval>/<SYMBOL>_<interval>_raw.csv
    data/validation/binance_vision_<SYMBOL>_<interval>_report.json

Binance kline CSV columns (12):
    open_time, open, high, low, close, volume, close_time, quote_volume,
    trades, taker_base, taker_quote, ignore
"""
import os
import sys
import io
import json
import time
import hashlib
import zipfile
import logging
import calendar
from datetime import datetime, timezone, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = "https://data.binance.vision/data/spot"

# ─── Configuration ──────────────────────────────────────────────────────
SYMBOL = os.environ.get("ADAN_BV_SYMBOL", "BTCUSDT")
INTERVAL = os.environ.get("ADAN_BV_INTERVAL", "5m")
OUTPUT_BASE = os.environ.get("ADAN_BV_OUTPUT", f"data/raw/{SYMBOL}")
WORKERS = int(os.environ.get("ADAN_BV_WORKERS", "8"))

# Known first-available months on Binance vision (probed). Fallback = scan.
KNOWN_FIRST = {
    "BTCUSDT": "2017-08",
    "DOGEUSDT": "2019-07",
}

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume",
              "close_time", "quote_volume", "trades", "taker_base",
              "taker_quote", "ignore"]


def http_get(url, timeout=60):
    req = Request(url, headers={"User-Agent": "adan-data/1.0"})
    with urlopen(req, timeout=timeout) as r:
        return r.read()


def url_exists(url):
    try:
        req = Request(url, method="HEAD", headers={"User-Agent": "adan-data/1.0"})
        with urlopen(req, timeout=30) as r:
            return r.status == 200
    except (HTTPError, URLError):
        return False


def month_iter(start_ym, end_ym):
    sy, sm = map(int, start_ym.split("-"))
    ey, em = map(int, end_ym.split("-"))
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m > 12:
            m = 1; y += 1


def find_first_month(symbol, interval):
    """Binary-ish scan for earliest available monthly archive."""
    if symbol in KNOWN_FIRST:
        ym = KNOWN_FIRST[symbol]
        url = f"{BASE}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
        if url_exists(url):
            return ym
    # Scan from 2017-07 forward until first hit
    for ym in month_iter("2017-07", datetime.now(timezone.utc).strftime("%Y-%m")):
        url = f"{BASE}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
        if url_exists(url):
            return ym
    raise RuntimeError(f"No monthly archive found for {symbol} {interval}")


def verify_checksum(zip_bytes, checksum_url):
    """Verify SHA256 against .CHECKSUM sidecar. Returns (ok, expected, actual)."""
    try:
        chk = http_get(checksum_url, timeout=30).decode().strip().split()[0].lower()
    except (HTTPError, URLError):
        return None, None, None  # checksum not available
    actual = hashlib.sha256(zip_bytes).hexdigest().lower()
    return (actual == chk), chk, actual


def parse_zip(zip_bytes):
    """Extract the single CSV inside a Binance kline ZIP into a DataFrame."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            raw = f.read()
    # Some 2025+ archives include a header row; detect it.
    first = raw[:32].decode(errors="ignore").lower()
    header = 0 if first.startswith("open_time") else None
    df = pd.read_csv(io.BytesIO(raw), header=header, names=KLINE_COLS)
    return df


def download_one(kind, symbol, interval, tag):
    """Download+verify one archive (monthly or daily). Returns (tag, df|None, meta)."""
    sub = "monthly" if kind == "monthly" else "daily"
    zip_url = f"{BASE}/{sub}/klines/{symbol}/{interval}/{symbol}-{interval}-{tag}.zip"
    chk_url = zip_url + ".CHECKSUM"
    meta = {"tag": tag, "kind": kind, "status": "OK", "checksum": "n/a", "rows": 0}
    try:
        zb = http_get(zip_url)
    except HTTPError as e:
        meta["status"] = f"HTTP_{e.code}"
        return tag, None, meta
    except URLError as e:
        meta["status"] = f"URLERR"
        return tag, None, meta
    ok, exp, act = verify_checksum(zb, chk_url)
    if ok is False:
        meta["status"] = "CHECKSUM_FAIL"
        meta["checksum"] = f"exp={exp[:12]} act={act[:12]}"
        logger.error(f"  ✗ checksum FAIL {tag}: {meta['checksum']}")
        return tag, None, meta
    meta["checksum"] = "verified" if ok else "unavailable"
    try:
        df = parse_zip(zb)
    except Exception as e:
        meta["status"] = f"PARSE_ERR:{str(e)[:40]}"
        return tag, None, meta
    meta["rows"] = len(df)
    return tag, df, meta


def to_ohlcv(df):
    """Normalize to DatetimeIndex + lowercase OHLCV, matching CCXT pipeline output."""
    # open_time is ms epoch (or sometimes us in newer archives). Detect scale.
    ot = df["open_time"].astype("int64")
    unit = "us" if ot.iloc[0] > 10**14 else "ms"
    ts = pd.to_datetime(ot, unit=unit, utc=True).dt.tz_localize(None)
    out = pd.DataFrame({
        "open": pd.to_numeric(df["open"], errors="coerce"),
        "high": pd.to_numeric(df["high"], errors="coerce"),
        "low": pd.to_numeric(df["low"], errors="coerce"),
        "close": pd.to_numeric(df["close"], errors="coerce"),
        "volume": pd.to_numeric(df["volume"], errors="coerce"),
    })
    out.index = ts
    out.index.name = "timestamp"
    return out


def validate_ohlc(df, interval):
    """Return dict of data-quality diagnostics (no mutation)."""
    minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}[interval]
    step = pd.Timedelta(minutes=minutes)
    idx = df.index
    diffs = idx.to_series().diff().dropna()
    expected = step
    gaps = diffs[diffs > expected]
    n_gaps = int((diffs != expected).sum())
    missing_bars = int(((diffs / expected).round() - 1).clip(lower=0).sum())
    # Aberrant candles: high<low, or high<max(open,close), or low>min(open,close)
    bad_hl = int((df["high"] < df["low"]).sum())
    bad_hi = int((df["high"] < df[["open", "close"]].max(axis=1)).sum())
    bad_lo = int((df["low"] > df[["open", "close"]].min(axis=1)).sum())
    zero_vol = int((df["volume"] <= 0).sum())
    nan_rows = int(df.isna().any(axis=1).sum())
    return {
        "rows": len(df),
        "range": [str(idx[0]), str(idx[-1])],
        "duplicated_timestamps": int(idx.duplicated().sum()),
        "n_interval_gaps": n_gaps,
        "estimated_missing_bars": missing_bars,
        "largest_gap": str(gaps.max()) if len(gaps) else "none",
        "aberrant_high_lt_low": bad_hl,
        "aberrant_high_lt_maxoc": bad_hi,
        "aberrant_low_gt_minoc": bad_lo,
        "zero_or_neg_volume": zero_vol,
        "nan_rows": nan_rows,
    }


def main():
    interval = INTERVAL
    symbol = SYMBOL
    start_ym = os.environ.get("ADAN_BV_START") or find_first_month(symbol, interval)
    end_ym = os.environ.get("ADAN_BV_END") or datetime.now(timezone.utc).strftime("%Y-%m")

    logger.info("=" * 60)
    logger.info(f"BINANCE VISION DOWNLOAD — {symbol} {interval}")
    logger.info(f"  months {start_ym} -> {end_ym}  (workers={WORKERS})")
    logger.info("=" * 60)

    months = list(month_iter(start_ym, end_ym))
    cur_ym = datetime.now(timezone.utc).strftime("%Y-%m")
    # For the current (partial) month, monthly archive may not exist yet -> use daily.
    monthly_tags = [m for m in months if m != cur_ym]
    use_daily_for_current = (end_ym == cur_ym)

    frames = []
    file_reports = []

    # --- Monthly archives in parallel ---
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(download_one, "monthly", symbol, interval, m): m
                for m in monthly_tags}
        for fut in as_completed(futs):
            tag, df, meta = fut.result()
            file_reports.append(meta)
            if df is not None:
                frames.append(to_ohlcv(df))
            else:
                logger.warning(f"  monthly {tag}: {meta['status']}")

    # --- Daily archives for the current partial month ---
    if use_daily_for_current:
        y, m = map(int, cur_ym.split("-"))
        today = datetime.now(timezone.utc).date()
        last_day = min(calendar.monthrange(y, m)[1], today.day)
        daily_tags = [f"{cur_ym}-{d:02d}" for d in range(1, last_day + 1)]
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(download_one, "daily", symbol, interval, d): d
                    for d in daily_tags}
            for fut in as_completed(futs):
                tag, df, meta = fut.result()
                file_reports.append(meta)
                if df is not None:
                    frames.append(to_ohlcv(df))

    if not frames:
        logger.error("FATAL: no data downloaded")
        sys.exit(1)

    # --- Concatenate, dedup, sort ---
    full = pd.concat(frames).sort_index()
    before = len(full)
    full = full[~full.index.duplicated(keep="first")]
    deduped = before - len(full)
    logger.info(f"Concatenated {before} rows -> {len(full)} after dedup ({deduped} dupes)")

    # --- Validate ---
    diag = validate_ohlc(full, interval)
    logger.info(f"  range {diag['range'][0]} -> {diag['range'][1]}")
    logger.info(f"  interval gaps={diag['n_interval_gaps']}  est missing bars={diag['estimated_missing_bars']}")
    logger.info(f"  aberrant(hl/hi/lo)={diag['aberrant_high_lt_low']}/{diag['aberrant_high_lt_maxoc']}/{diag['aberrant_low_gt_minoc']}  zero_vol={diag['zero_or_neg_volume']}  nan={diag['nan_rows']}")

    # --- Save ---
    out_dir = os.path.join(OUTPUT_BASE, interval)
    os.makedirs(out_dir, exist_ok=True)
    pq = os.path.join(out_dir, f"{symbol}_{interval}_raw.parquet")
    full.to_parquet(pq)
    full.to_csv(os.path.join(out_dir, f"{symbol}_{interval}_raw.csv"))
    logger.info(f"  Saved {pq} ({len(full)} rows)")

    # --- Report ---
    n_fail = sum(1 for r in file_reports if r["status"] != "OK")
    report = {
        "timestamp": datetime.now().isoformat(),
        "source": "data.binance.vision",
        "symbol": symbol,
        "interval": interval,
        "months_requested": [start_ym, end_ym],
        "archives_total": len(file_reports),
        "archives_failed": n_fail,
        "deduped_rows": deduped,
        "diagnostics": diag,
        "files": sorted(file_reports, key=lambda r: r["tag"]),
    }
    os.makedirs("data/validation", exist_ok=True)
    rpath = f"data/validation/binance_vision_{symbol}_{interval}_report.json"
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"  Report: {rpath}  (archives_failed={n_fail})")

    if n_fail > 0:
        logger.warning(f"{n_fail} archive(s) failed/missing — see report")
    logger.info("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
