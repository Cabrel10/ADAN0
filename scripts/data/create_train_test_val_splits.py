#!/usr/bin/env python3
"""
Create train/test/val splits from featured data.

Loads featured parquet files and splits them into train/test/val directories.
Structure: data/processed/indicators/{train,test,val}/BTCUSDT/{5m,1h,4h}.parquet
"""
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Configuration
FEATURED_DIR = _PROJECT_ROOT / "data/processed/BTCUSDT"
OUTPUT_BASE = _PROJECT_ROOT / "data/processed/indicators"
TIMEFRAMES = ["5m", "1h", "4h"]
ASSET = "BTCUSDT"

# Split ratios (train/test/val)
# 47,440 lignes 5m = 717 jours
# Train = 70%
# Test = 20%
# Val = 10%
TRAIN_RATIO = 0.70
TEST_RATIO = 0.20
VAL_RATIO = 0.10

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
        test_end = train_end + int(n * TEST_RATIO)
        
        # Split data (chronological order)
        df_train = df.iloc[:train_end]
        df_test = df.iloc[train_end:test_end]
        df_val = df.iloc[test_end:]
        
        logger.info(f"Train: {len(df_train)} rows ({TRAIN_RATIO*100:.0f}%)")
        logger.info(f"Test:  {len(df_test)} rows ({TEST_RATIO*100:.0f}%)")
        logger.info(f"Val:   {len(df_val)} rows ({VAL_RATIO*100:.0f}%)")
        
        # Create output directories
        for split in ["train", "test", "val"]:
            split_dir = OUTPUT_BASE / split / ASSET
            split_dir.mkdir(parents=True, exist_ok=True)
        
        # Save splits
        splits = {
            "train": (df_train, OUTPUT_BASE / "train" / ASSET / f"{tf}.parquet"),
            "test": (df_test, OUTPUT_BASE / "test" / ASSET / f"{tf}.parquet"),
            "val": (df_val, OUTPUT_BASE / "val" / ASSET / f"{tf}.parquet"),
        }
        
        for split_name, (df_split, out_path) in splits.items():
            df_split.to_parquet(out_path)
            logger.info(f"  ✓ {split_name}: {out_path}")
    
    logger.info("\n✓ All splits created successfully")

if __name__ == "__main__":
    create_splits()
