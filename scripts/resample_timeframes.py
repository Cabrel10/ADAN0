#!/usr/bin/env python3
"""
Resample 5m data to 1h and 4h timeframes.
This ensures consistency across all timeframes.
"""
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = _PROJECT_ROOT / "data/raw/BTCUSDT"

def resample_ohlcv(df, target_freq):
    """Resample OHLCV data to target frequency."""
    resampled = df.resample(target_freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    # Remove rows with NaN (gaps in data)
    resampled = resampled.dropna()
    return resampled

def main():
    logger.info("=" * 60)
    logger.info("RESAMPLE 5m → 1h, 4h")
    logger.info("=" * 60)
    
    # Load 5m data
    path_5m = OUTPUT_DIR / "5m" / "BTCUSDT_5m_raw.parquet"
    if not path_5m.exists():
        logger.error(f"MISSING: {path_5m}")
        return 1
    
    df_5m = pd.read_parquet(path_5m)
    logger.info(f"Loaded 5m: {len(df_5m):,} rows")
    
    # Resample to 1h
    logger.info("\nResampling to 1h...")
    df_1h = resample_ohlcv(df_5m, '1h')
    out_dir_1h = OUTPUT_DIR / "1h"
    out_dir_1h.mkdir(parents=True, exist_ok=True)
    out_path_1h = out_dir_1h / "BTCUSDT_1h_raw.parquet"
    df_1h.to_parquet(out_path_1h)
    logger.info(f"  Saved: {out_path_1h} ({len(df_1h):,} rows)")
    
    # Resample to 4h
    logger.info("\nResampling to 4h...")
    df_4h = resample_ohlcv(df_5m, '4h')
    out_dir_4h = OUTPUT_DIR / "4h"
    out_dir_4h.mkdir(parents=True, exist_ok=True)
    out_path_4h = out_dir_4h / "BTCUSDT_4h_raw.parquet"
    df_4h.to_parquet(out_path_4h)
    logger.info(f"  Saved: {out_path_4h} ({len(df_4h):,} rows)")
    
    logger.info("\n" + "=" * 60)
    logger.info("RESAMPLE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"5m: {len(df_5m):,} rows")
    logger.info(f"1h: {len(df_1h):,} rows")
    logger.info(f"4h: {len(df_4h):,} rows")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
