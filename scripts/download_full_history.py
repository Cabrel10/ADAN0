#!/usr/bin/env python3
"""
Download full BTC/USDT history from 2022-01-01 to present.
Uses data.binance.vision daily archives (no API key required).
Saves only BTCUSDT in 5m, 1h, 4h timeframes.
Columns: open, high, low, close, volume (DatetimeIndex, no timestamp column).
"""
import os, sys, time, json, logging, io, zipfile, requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = _PROJECT_ROOT / "data/raw/BTCUSDT"
ARCHIVE_URL = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT"

# Download from 2022-01-01 to today
START_DATE = datetime(2022, 1, 1, tzinfo=timezone.utc).date()
END_DATE = (datetime.now(timezone.utc) - timedelta(days=1)).date()

TIMEFRAMES = ["5m", "1h", "4h"]
SYMBOL = "BTCUSDT"

KLINE_COLS = ["timestamp", "open", "high", "low", "close", "volume",
              "close_time", "quote_volume", "n_trades",
              "taker_buy_base", "taker_buy_quote", "ignore"]


def download_timeframe(interval):
    """Download full history for a timeframe from archives."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Downloading {SYMBOL} {interval}")
    logger.info(f"Period: {START_DATE} → {END_DATE}")
    logger.info(f"{'='*60}")
    
    all_dfs = []
    current = START_DATE
    success_count = 0
    failed_count = 0
    
    while current <= END_DATE:
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
                success_count += 1
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
            if failed_count % 50 == 0:
                logger.warning(f"  {failed_count} days failed (latest: {str(e)[:60]})")
        
        # Progress every 100 days
        if (success_count + failed_count) % 100 == 0:
            logger.info(f"  Progress: {success_count} days OK, {failed_count} failed")
        
        current += timedelta(days=1)
        time.sleep(0.02)  # Respect rate limits
    
    logger.info(f"  Total: {success_count} days downloaded, {failed_count} failed")
    
    if not all_dfs:
        logger.error(f"  FAILED: No data for {interval}")
        return None
    
    # Combine all days
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Convert to DataFrame with proper columns
    df = combined[["timestamp", "open", "high", "low", "close", "volume"]].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    
    logger.info(f"  Total rows: {len(df):,}")
    logger.info(f"  Date range: {df.index.min()} → {df.index.max()}")
    logger.info(f"  Columns: {df.columns.tolist()}")
    
    return df


def validate_df(df, interval):
    """Validate downloaded data."""
    report = {
        "interval": interval,
        "rows": len(df),
        "date_range": f"{df.index.min()} → {df.index.max()}",
        "columns": df.columns.tolist(),
        "nan_count": int(df.isna().sum().sum()),
        "zero_close_count": int((df["close"] == 0).sum()),
        "negative_volume_count": int((df["volume"] < 0).sum()),
        "price_range": f"{df['close'].min():.2f} - {df['close'].max():.2f}",
        "passed": True,
        "errors": []
    }
    
    if report["nan_count"] > 0:
        report["errors"].append(f"NaN: {report['nan_count']}")
        report["passed"] = False
    
    if report["zero_close_count"] > 0:
        report["errors"].append(f"Close=0: {report['zero_close_count']}")
        report["passed"] = False
    
    if report["negative_volume_count"] > 0:
        report["errors"].append(f"Neg volume: {report['negative_volume_count']}")
        report["passed"] = False
    
    if df['close'].min() < 5000 or df['close'].max() > 300000:
        report["errors"].append(f"Price range suspicious: {report['price_range']}")
        report["passed"] = False
    
    return report


def main():
    logger.info("=" * 60)
    logger.info("FULL HISTORY DOWNLOAD (2022-01-01 to present)")
    logger.info("=" * 60)
    
    all_reports = []
    
    for interval in TIMEFRAMES:
        df = download_timeframe(interval)
        
        if df is None:
            all_reports.append({
                "interval": interval,
                "passed": False,
                "errors": ["No data downloaded"],
                "rows": 0
            })
            continue
        
        # Validate
        report = validate_df(df, interval)
        all_reports.append(report)
        
        # Save
        out_dir = OUTPUT_DIR / interval
        out_dir.mkdir(parents=True, exist_ok=True)
        
        parquet_path = out_dir / f"BTCUSDT_{interval}_raw.parquet"
        csv_path = out_dir / f"BTCUSDT_{interval}_raw.csv"
        
        df.to_parquet(parquet_path)
        df.to_csv(csv_path)
        
        if report["passed"]:
            logger.info(f"  ✓ PASS {interval}: {report['rows']:,} rows")
            logger.info(f"    Saved: {parquet_path}")
        else:
            logger.error(f"  ✗ FAIL {interval}: {report['errors']}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for r in all_reports if r["passed"])
    for r in all_reports:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        rows = r.get("rows", 0)
        print(f"  {status} {r['interval']}: {rows:,} rows")
        if r.get("errors"):
            for err in r["errors"]:
                print(f"       {err}")
    
    # Save report
    (OUTPUT_DIR.parent.parent / "data/validation").mkdir(parents=True, exist_ok=True)
    with open(_PROJECT_ROOT / "data/validation/download_full_history_report.json", "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    
    if passed == len(TIMEFRAMES):
        logger.info("\n✓ ALL DOWNLOADS SUCCESSFUL")
        return 0
    else:
        logger.error(f"\n✗ {len(TIMEFRAMES) - passed} timeframe(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
