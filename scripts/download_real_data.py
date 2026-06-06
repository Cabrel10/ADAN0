#!/usr/bin/env python3
"""
DL-02: Download real BTC/USDT klines from Binance US public API.
Falls back to data.binance.vision if the API fails.
No API key required.
"""
import os, sys, time, json, logging, io, zipfile, requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Primary: Binance US (not geo-blocked), Fallback: data.binance.vision
API_URL = "https://api.binance.us/api/v3/klines"
ARCHIVE_URL = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT"
SYMBOL = "BTCUSDT"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = _PROJECT_ROOT / "data/raw/BTCUSDT"

CONFIG = {
    "5m":  {"interval": "5m",  "days_back": 90,  "min_rows": 25_000},
    "1h":  {"interval": "1h",  "days_back": 180, "min_rows": 4_000},
    "4h":  {"interval": "4h",  "days_back": 365, "min_rows": 2_000},
}

KLINE_COLS = ["timestamp", "open", "high", "low", "close", "volume",
              "close_time", "quote_volume", "n_trades",
              "taker_buy_base", "taker_buy_quote", "ignore"]


def fetch_klines_api(symbol, interval, start_ms, end_ms):
    """Fetch klines from Binance US REST API."""
    all_data = []
    current_start = start_ms
    batch_num = 0
    while current_start < end_ms:
        params = {"symbol": symbol, "interval": interval,
                  "startTime": current_start, "endTime": end_ms, "limit": 1000}
        batch = []
        for attempt in range(3):
            try:
                resp = requests.get(API_URL, params=params, timeout=30)
                if resp.status_code == 451:
                    return None  # geo-blocked, signal to try fallback
                resp.raise_for_status()
                batch = resp.json()
                break
            except requests.exceptions.HTTPError as e:
                if "451" in str(e):
                    return None
                logger.warning(f"  Retry {attempt+1}/3: {e}")
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.warning(f"  Retry {attempt+1}/3: {e}")
                time.sleep(2 ** attempt)
        if not batch:
            break
        all_data.extend(batch)
        batch_num += 1
        if batch_num % 10 == 0:
            logger.info(f"  ... {len(all_data)} klines fetched")
        if len(batch) < 1000:
            break
        current_start = batch[-1][0] + 1
        time.sleep(0.15)
    return all_data


def fetch_klines_archive(interval, start_date, end_date):
    """Fetch klines from data.binance.vision daily archives."""
    all_dfs = []
    current = start_date
    failed = 0
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        url = f"{ARCHIVE_URL}/{interval}/{SYMBOL}-{interval}-{date_str}.zip"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    for name in z.namelist():
                        if name.endswith('.csv'):
                            with z.open(name) as f:
                                df = pd.read_csv(f, header=None, names=KLINE_COLS)
                                all_dfs.append(df)
                if len(all_dfs) % 30 == 0:
                    logger.info(f"  ... {len(all_dfs)} days downloaded")
            else:
                failed += 1
        except Exception as e:
            failed += 1
            if failed % 10 == 0:
                logger.warning(f"  {failed} days failed so far (latest: {e})")
        current += timedelta(days=1)
        time.sleep(0.05)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        return combined.values.tolist()
    return []


def klines_to_df(raw):
    df = pd.DataFrame(raw, columns=KLINE_COLS)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
    df = df.set_index("timestamp")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def validate_df(df, tf, cfg):
    report = {
        "tf": tf, "rows": len(df),
        "min_rows_required": cfg["min_rows"],
        "nan_count": int(df.isna().sum().sum()),
        "zero_close_count": int((df["close"] == 0).sum()),
        "negative_volume_count": int((df["volume"] < 0).sum()),
        "start_date": str(df.index.min()),
        "end_date": str(df.index.max()),
        "price_range": f"{df['close'].min():.2f} - {df['close'].max():.2f}",
        "passed": False, "errors": []
    }
    if report["rows"] < cfg["min_rows"]:
        report["errors"].append(f"ROWS: {report['rows']} < {cfg['min_rows']}")
    if report["nan_count"] > 0:
        report["errors"].append(f"NaN: {report['nan_count']}")
    if report["zero_close_count"] > 0:
        report["errors"].append(f"Close=0: {report['zero_close_count']}")
    if report["negative_volume_count"] > 0:
        report["errors"].append(f"Neg volume: {report['negative_volume_count']}")
    if df['close'].min() < 5000 or df['close'].max() > 300000:
        report["errors"].append(f"Price range suspicious: {report['price_range']}")
    report["passed"] = len(report["errors"]) == 0
    return report


def main():
    all_reports = []
    now = datetime.now(timezone.utc)

    for tf, cfg in CONFIG.items():
        logger.info(f"=== Downloading {SYMBOL}/{tf} ({cfg['days_back']} days) ===")
        end_ms = int(now.timestamp() * 1000)
        start_ms = int((now - timedelta(days=cfg["days_back"])).timestamp() * 1000)

        # Try API first
        logger.info(f"  Trying Binance US API...")
        raw = fetch_klines_api(SYMBOL, cfg["interval"], start_ms, end_ms)

        if raw is None or len(raw) < cfg["min_rows"] // 2:
            # Fallback to archives
            logger.info(f"  API insufficient ({len(raw) if raw else 0} rows), trying data.binance.vision archives...")
            start_date = (now - timedelta(days=cfg["days_back"])).date()
            end_date = (now - timedelta(days=1)).date()
            raw = fetch_klines_archive(cfg["interval"], start_date, end_date)
            if not raw:
                logger.error(f"  BOTH sources failed for {tf}")
                all_reports.append({"tf": tf, "passed": False, "errors": ["No data from any source"], "rows": 0})
                continue

        logger.info(f"  Raw klines: {len(raw)}")
        df = klines_to_df(raw)

        out_dir = OUTPUT_DIR / tf
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"BTCUSDT_{tf}_raw.csv")
        df.to_parquet(out_dir / f"BTCUSDT_{tf}_raw.parquet")

        report = validate_df(df, tf, cfg)
        all_reports.append(report)
        if report["passed"]:
            logger.info(f"  PASS {tf}: {report['rows']} rows, {report['start_date']} -> {report['end_date']}")
        else:
            for err in report["errors"]:
                logger.error(f"  FAIL {tf}: {err}")

    (_PROJECT_ROOT / "data/validation").mkdir(parents=True, exist_ok=True)
    with open(_PROJECT_ROOT / "data/validation/download_report.json", "w") as f:
        json.dump(all_reports, f, indent=2)

    passed = sum(1 for r in all_reports if r["passed"])
    print(f"\n{'='*60}")
    print(f"DL-02: {passed}/{len(all_reports)} timeframes validated")
    for r in all_reports:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {status} {r['tf']}: {r.get('rows', 0)} rows {r.get('price_range', '')}")
    if passed < len(all_reports):
        print("WARNING: some timeframes failed validation")
        return 1
    print("ALL DATA VALIDATED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
