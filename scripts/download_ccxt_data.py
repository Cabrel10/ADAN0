#!/usr/bin/env python3
"""
Download real OHLCV candles from public CCXT (Binance) for ADAN testing.
Stores data in data/raw/ as parquet files ready for environment consumption.
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime, timezone

import ccxt
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ASSETS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"]
TIMEFRAMES = ["5m", "1h", "4h"]


def download_ohlcv(exchange, symbol: str, timeframe: str, limit: int = 5000) -> pd.DataFrame:
    """Download OHLCV data from exchange using pagination."""
    logger.info(f"Downloading {symbol} {timeframe} ({limit} candles)...")
    
    all_data = []
    remaining = limit
    since = None  # start from most recent
    
    # For large downloads, work backwards from now
    batch_size = min(1000, remaining)
    
    # Calculate start time based on limit and timeframe
    tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
    minutes = tf_minutes.get(timeframe, 5)
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = end_time - (limit * minutes * 60 * 1000)
    since = start_time
    
    while remaining > 0:
        batch = min(1000, remaining)
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=batch)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            remaining -= len(ohlcv)
            since = ohlcv[-1][0] + 1  # next millisecond after last candle
            logger.info(f"  Got {len(ohlcv)} candles, total={len(all_data)}, remaining={remaining}")
            time.sleep(0.2)  # Rate limit
            if len(ohlcv) < batch:
                break  # No more data
        except Exception as e:
            logger.error(f"  Error downloading {symbol} {timeframe}: {e}")
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(5)
                continue
            break
    
    if not all_data:
        logger.warning(f"  No data for {symbol} {timeframe}")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data, columns=["TIMESTAMP", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])
    df["TIMESTAMP"] = df["TIMESTAMP"].astype(np.int64)
    df = df.drop_duplicates(subset=["TIMESTAMP"]).sort_values("TIMESTAMP").reset_index(drop=True)
    logger.info(f"  Final: {len(df)} candles for {symbol} {timeframe}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Download CCXT OHLCV data")
    parser.add_argument("--limit", type=int, default=5000, help="Number of candles per asset/timeframe")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name (binance or bitget)")
    parser.add_argument("--output-dir", type=str, default="data/raw/ccxt", help="Output directory")
    args = parser.parse_args()

    # Try Binance first, fallback to Bitget
    exchange = None
    for exch_name in [args.exchange, "bitget", "binance"]:
        try:
            exch_class = getattr(ccxt, exch_name)
            exchange = exch_class({
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
            exchange.load_markets()
            logger.info(f"Connected to {exch_name.upper()}")
            break
        except Exception as e:
            logger.warning(f"Cannot connect to {exch_name}: {e}")
    
    if not exchange:
        logger.error("Failed to connect to any exchange!")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Download data for each asset and timeframe
    summary = {}
    for symbol in ASSETS:
        safe_symbol = symbol.replace("/", "")
        for tf in TIMEFRAMES:
            df = download_ohlcv(exchange, symbol, tf, args.limit)
            if df.empty:
                logger.warning(f"Skipping {symbol} {tf} - no data")
                continue
            
            # Save as parquet
            fname = f"{safe_symbol}_{tf}.parquet"
            fpath = os.path.join(args.output_dir, fname)
            df.to_parquet(fpath, index=False)
            
            # Also save as CSV for debugging
            csv_fname = f"{safe_symbol}_{tf}.csv"
            csv_path = os.path.join(args.output_dir, csv_fname)
            df.to_csv(csv_path, index=False)
            
            summary[f"{symbol}_{tf}"] = len(df)
            logger.info(f"Saved {fpath} ({len(df)} rows)")

    # Print summary
    logger.info("=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    for key, count in summary.items():
        logger.info(f"  {key}: {count} candles")
    logger.info(f"Total files: {len(summary)}")
    logger.info(f"Output dir: {args.output_dir}")
    
    # Verify minimum data requirements
    btc_5m = summary.get("BTC/USDT_5m", 0)
    if btc_5m < 1000:
        logger.error(f"INSUFFICIENT DATA: BTC/USDT 5m has only {btc_5m} candles (need >= 1000)")
        sys.exit(1)
    else:
        logger.info(f"DATA OK: BTC/USDT 5m has {btc_5m} candles")


if __name__ == "__main__":
    main()
