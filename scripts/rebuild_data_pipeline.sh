#!/bin/bash
# ============================================================================
# REBUILD DATA PIPELINE: Clean old data, download full history, compute features, create splits
# ============================================================================

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================================"
echo "STEP 1: CLEAN OLD DATA"
echo "============================================================================"

echo "Removing old raw data..."
rm -f data/raw/BTCUSDT/5m/*.parquet data/raw/BTCUSDT/5m/*.csv
rm -f data/raw/BTCUSDT/1h/*.parquet data/raw/BTCUSDT/1h/*.csv
rm -f data/raw/BTCUSDT/4h/*.parquet data/raw/BTCUSDT/4h/*.csv

echo "Removing old featured data..."
rm -f data/processed/BTCUSDT/*.parquet

echo "Removing old splits..."
rm -rf data/processed/indicators/train/BTCUSDT/*.parquet
rm -rf data/processed/indicators/test/BTCUSDT/*.parquet
rm -rf data/processed/indicators/val/BTCUSDT/*.parquet

echo "✓ Old data cleaned"

echo ""
echo "============================================================================"
echo "STEP 2: DOWNLOAD FULL HISTORY (2022-01-01 to present)"
echo "============================================================================"
echo "This will take 10-20 minutes..."
echo ""

PYTHONPATH=src python scripts/download_full_history.py

if [ $? -ne 0 ]; then
    echo "✗ Download failed"
    exit 1
fi

echo ""
echo "============================================================================"
echo "STEP 3: COMPUTE INDICATORS ON FULL DATASET"
echo "============================================================================"

PYTHONPATH=src python scripts/compute_features_real.py

if [ $? -ne 0 ]; then
    echo "✗ Feature engineering failed"
    exit 1
fi

echo ""
echo "============================================================================"
echo "STEP 4: CREATE TRAIN/TEST/VAL SPLITS (70/20/10)"
echo "============================================================================"

PYTHONPATH=src python scripts/create_train_test_val_splits.py

if [ $? -ne 0 ]; then
    echo "✗ Split creation failed"
    exit 1
fi

echo ""
echo "============================================================================"
echo "VERIFICATION"
echo "============================================================================"

python3 << 'EOF'
import pandas as pd

print("\nRAW DATA:")
for tf in ['5m', '1h', '4h']:
    df = pd.read_parquet(f'data/raw/BTCUSDT/{tf}/BTCUSDT_{tf}_raw.parquet')
    print(f"  {tf}: {len(df):,} rows, {(df.index.max() - df.index.min()).days} calendar days")

print("\nFEATURED DATA:")
for tf in ['5m', '1h', '4h']:
    df = pd.read_parquet(f'data/processed/BTCUSDT/BTCUSDT_{tf}_featured.parquet')
    print(f"  {tf}: {len(df):,} rows, {len(df.columns)} columns")

print("\nSPLITS (5m):")
for split in ['train', 'test', 'val']:
    df = pd.read_parquet(f'data/processed/indicators/{split}/BTCUSDT/5m.parquet')
    pct = len(df) / 47440 * 100 if split != 'train' else len(df) / 47440 * 100
    print(f"  {split}: {len(df):,} rows ({pct:.1f}%)")

EOF

echo ""
echo "============================================================================"
echo "✓ PIPELINE COMPLETE"
echo "============================================================================"
echo ""
echo "Next steps:"
echo "  1. Launch training: ADAN_TRAINING_SILENT=1 PYTHONPATH=src python scripts/train_parallel_agents.py --mode heavy --steps 500000 --num-cpus 4 --num-samples 2 --no-subproc --checkpoint-dir /mnt/new_data/adan_logs/checkpoints > /mnt/new_data/adan_logs/training/train_v10_full_data.log 2>&1 &"
echo "  2. Monitor: tail -f /mnt/new_data/adan_logs/training/train_v10_full_data.log"
echo ""
