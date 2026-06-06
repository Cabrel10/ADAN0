#!/usr/bin/env python3
"""
Download REAL OHLCV data via CCXT — no mocks, no fallbacks, no excuses.

Sources: Bitget (primary), OKX (secondary) — Binance blocked from this sandbox.
Volume validation: rejects data with unrealistic volume (< 50K USD median).
Saves in format compatible with ChunkedDataLoader (lowercase columns, DatetimeIndex).
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone, timedelta

import ccxt
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Configuration ──────────────────────────────────────────────────────
SYMBOL = "BTC/USDT"
TIMEFRAMES = {
    "5m":  {"limit": 25000, "minutes": 5},
    "1h":  {"limit": 4500,  "minutes": 60},
    "4h":  {"limit": 2200,  "minutes": 240},
}
OUTPUT_BASE = "data/raw/BTCUSDT"
MIN_MEDIAN_VOLUME_USD = 50_000  # Reject data with lower median volume

# Exchanges to try (in order)
EXCHANGES = ["bitget", "okx", "kucoin"]


def connect_exchange():
    """Connect to first working exchange."""
    for name in EXCHANGES:
        try:
            ex_cls = getattr(ccxt, name)
            ex = ex_cls({
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
                "timeout": 30000,
            })
            ex.load_markets()
            if SYMBOL not in ex.markets:
                logger.warning(f"{name}: {SYMBOL} not found")
                continue
            # Test fetch
            test = ex.fetch_ohlcv(SYMBOL, "5m", limit=2)
            if test and len(test) >= 1:
                logger.info(f"✓ Connected to {name.upper()}")
                return ex, name
        except Exception as e:
            logger.warning(f"✗ {name}: {str(e)[:80]}")
    return None, None


def download_timeframe(exchange, tf_name, tf_config):
    """Download full history for a timeframe with pagination."""
    limit = tf_config["limit"]
    minutes = tf_config["minutes"]
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Downloading {SYMBOL} {tf_name} — target: {limit} candles")
    
    # Calculate start time
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - (limit * minutes * 60 * 1000)
    since = start_ms
    
    all_data = []
    batch_count = 0
    
    while since < end_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, tf_name, since=since, limit=1000)
            if not ohlcv:
                break
            
            all_data.extend(ohlcv)
            batch_count += 1
            since = ohlcv[-1][0] + 1
            
            if batch_count % 5 == 0:
                logger.info(f"  batch {batch_count}: {len(all_data)} candles so far")
            
            if len(ohlcv) < 100:
                break
            
            time.sleep(0.3)  # Respect rate limits
            
        except ccxt.RateLimitExceeded:
            logger.warning("  Rate limit — waiting 5s")
            time.sleep(5)
        except Exception as e:
            logger.error(f"  Error: {e}")
            time.sleep(2)
            break
    
    if not all_data:
        logger.error(f"  FAILED: No data for {tf_name}")
        return None
    
    # Build DataFrame
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")
    
    # Remove timezone info for compatibility
    df.index = df.index.tz_localize(None)
    
    logger.info(f"  Downloaded: {len(df)} candles")
    logger.info(f"  Range: {df.index[0]} → {df.index[-1]}")
    
    return df


def validate_volume(df, tf_name):
    """Validate volume is realistic — reject if not."""
    vol_usd = df["volume"] * df["close"]
    median_vol = vol_usd.median()
    mean_vol = vol_usd.mean()
    
    logger.info(f"  Volume analysis ({tf_name}):")
    logger.info(f"    Median volume (USD):  ${median_vol:,.0f}")
    logger.info(f"    Mean volume (USD):    ${mean_vol:,.0f}")
    logger.info(f"    Min volume (USD):     ${vol_usd.min():,.0f}")
    logger.info(f"    Max volume (USD):     ${vol_usd.max():,.0f}")
    logger.info(f"    Rows < 1000 USD:      {(vol_usd < 1000).sum()} / {len(df)}")
    
    if median_vol < MIN_MEDIAN_VOLUME_USD:
        logger.error(f"  ✗ REJECTED: median volume ${median_vol:,.0f} < ${MIN_MEDIAN_VOLUME_USD:,.0f}")
        return False
    
    logger.info(f"  ✓ Volume validated: median ${median_vol:,.0f}")
    return True


def main():
    logger.info("=" * 60)
    logger.info("CCXT REAL DATA DOWNLOAD — NO MOCKS")
    logger.info("=" * 60)
    
    exchange, exchange_name = connect_exchange()
    if not exchange:
        logger.error("FATAL: Cannot connect to any exchange")
        sys.exit(1)
    
    results = {}
    
    for tf_name, tf_config in TIMEFRAMES.items():
        df = download_timeframe(exchange, tf_name, tf_config)
        if df is None:
            results[tf_name] = {"status": "FAILED", "rows": 0}
            continue
        
        # Validate volume
        if not validate_volume(df, tf_name):
            results[tf_name] = {"status": "VOLUME_REJECTED", "rows": len(df)}
            continue
        
        # Save
        safe_tf = tf_name.replace(" ", "_")
        out_dir = os.path.join(OUTPUT_BASE, safe_tf)
        os.makedirs(out_dir, exist_ok=True)
        
        out_path = os.path.join(out_dir, f"BTCUSDT_{safe_tf}_raw.parquet")
        df.to_parquet(out_path)
        
        # Also save CSV for human inspection
        csv_path = os.path.join(out_dir, f"BTCUSDT_{safe_tf}_raw.csv")
        df.to_csv(csv_path)
        
        results[tf_name] = {
            "status": "OK",
            "rows": len(df),
            "range": f"{df.index[0]} to {df.index[-1]}",
            "median_volume_usd": float(round((df['volume'] * df['close']).median(), 0)),
            "path": out_path,
        }
        
        logger.info(f"  Saved: {out_path} ({len(df)} rows)")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Exchange: {exchange_name.upper()}")
    
    all_ok = True
    for tf, info in results.items():
        status = info["status"]
        rows = info["rows"]
        if status == "OK":
            logger.info(f"  ✓ {tf}: {rows} candles, median vol ${info['median_volume_usd']:,.0f}")
        else:
            logger.info(f"  ✗ {tf}: {status} ({rows} candles)")
            all_ok = False
    
    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "exchange": exchange_name,
        "symbol": SYMBOL,
        "results": results,
        "all_ok": all_ok,
    }
    os.makedirs("data/validation", exist_ok=True)
    with open("data/validation/download_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    if not all_ok:
        logger.error("SOME DOWNLOADS FAILED")
        sys.exit(1)
    
    logger.info("\nALL DOWNLOADS SUCCESSFUL ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
