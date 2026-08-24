#!/usr/bin/env python3
"""
Create train/test/val splits from featured data.

Loads featured parquet files and splits them into train/test/val directories.
Structure: data/processed/indicators/{train,test,val}/BTCUSDT/{5m,1h,4h}.parquet
"""
import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Configuration. ASSET is parameterized via env var; default preserves BTC behavior.
ASSET = os.environ.get("ADAN_DL_PAIR", "BTCUSDT")
# Source featured dir and output-asset label can be overridden (e.g. "_binance").
PROC_DIRNAME = os.environ.get("ADAN_DL_PROC_DIR", ASSET)
OUT_ASSET = os.environ.get("ADAN_DL_OUT_ASSET", ASSET)
FEATURED_DIR = _PROJECT_ROOT / f"data/processed/{PROC_DIRNAME}"
OUTPUT_BASE = _PROJECT_ROOT / "data/processed/indicators"
TIMEFRAMES = ["5m", "1h", "4h"]

# Split ratios. Defaults preserve legacy 70/20/10 (train/test/val) behavior.
# For the Binance 5y experiment set 70/15/15 via env with chronological
# order TRAIN -> VAL -> TEST (TEST = final, never-seen out-of-sample segment).
TRAIN_RATIO = float(os.environ.get("ADAN_DL_TRAIN_RATIO", "0.70"))
VAL_RATIO = float(os.environ.get("ADAN_DL_VAL_RATIO", "0.10"))
TEST_RATIO = float(os.environ.get("ADAN_DL_TEST_RATIO", "0.20"))
# Chronological order of the two post-train segments: "test_val" (legacy:
# test in middle, val last) or "val_test" (val in middle, test last).
SPLIT_ORDER = os.environ.get("ADAN_DL_SPLIT_ORDER", "test_val")

def create_splits():
    """Create train/test/val splits from featured data."""
    
    logger.info(f"Creating train/test/val splits from {FEATURED_DIR}")
    
    for tf in TIMEFRAMES:
        featured_path = FEATURED_DIR / f"{ASSET}_{tf}_featured.parquet"
        
        if not featured_path.exists():
            logger.error(f"MISSING: {featured_path}")
            continue
        
        logger.info(f"\n=== Processing {tf} ===")
        df = pd.read_parquet(featured_path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Calculate split indices
        n = len(df)
        train_end = int(n * TRAIN_RATIO)
        df_train = df.iloc[:train_end]
        if SPLIT_ORDER == "val_test":
            # TRAIN -> VAL -> TEST  (TEST = final unseen segment)
            val_end = train_end + int(n * VAL_RATIO)
            df_val = df.iloc[train_end:val_end]
            df_test = df.iloc[val_end:]
        else:
            # Legacy: TRAIN -> TEST -> VAL
            test_end = train_end + int(n * TEST_RATIO)
            df_test = df.iloc[train_end:test_end]
            df_val = df.iloc[test_end:]
        
        logger.info(f"Order={SPLIT_ORDER} | Train: {len(df_train)} ({TRAIN_RATIO*100:.0f}%)  "
                    f"Val: {len(df_val)} ({VAL_RATIO*100:.0f}%)  Test: {len(df_test)} ({TEST_RATIO*100:.0f}%)")
        
        # Create output directories
        for split in ["train", "test", "val"]:
            split_dir = OUTPUT_BASE / split / OUT_ASSET
            split_dir.mkdir(parents=True, exist_ok=True)
        
        # Save splits
        splits = {
            "train": (df_train, OUTPUT_BASE / "train" / OUT_ASSET / f"{tf}.parquet"),
            "test": (df_test, OUTPUT_BASE / "test" / OUT_ASSET / f"{tf}.parquet"),
            "val": (df_val, OUTPUT_BASE / "val" / OUT_ASSET / f"{tf}.parquet"),
        }
        
        for split_name, (df_split, out_path) in splits.items():
            df_split.to_parquet(out_path)
            logger.info(f"  ✓ {split_name}: {out_path}")
    
    logger.info("\n✓ All splits created successfully")

if __name__ == "__main__":
    create_splits()
