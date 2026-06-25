#!/usr/bin/env python3
"""Extract and save production scalers from training data.

This script simulates what MultiAssetChunkedEnv does on its first reset:
it fits StateBuilder scalers on the training parquet data and saves them
to `prod_scalers/` so that LiveStateBuilder and deterministic_backtest
can load them for consistent inference.

Usage:
    PYTHONPATH=src python scripts/extract_training_scalers.py

Output:
    prod_scalers/scaler_5m.pkl
    prod_scalers/scaler_1h.pkl
    prod_scalers/scaler_4h.pkl
    prod_scalers/scalers_manifest.json
"""
import sys
from pathlib import Path

# Ensure src/ is in path
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = _SCRIPT_DIR.parent

# Import StateBuilder
from adan_trading_bot.data_processing.state_builder import StateBuilder

# Use the same TRAIN_COLUMNS as LiveStateBuilder for consistency
from adan_trading_bot.trading.live_state_builder import TRAIN_COLUMNS, OBS_WINDOW


def main():
    logger.info("=" * 60)
    logger.info("EXTRACTING TRAINING SCALERS")
    logger.info("=" * 60)

    # Load training parquet data (same as what the env sees)
    train_dir = PROJECT_ROOT / "data" / "processed" / "indicators" / "train" / "BTCUSDT"
    
    if not train_dir.exists():
        logger.error(f"Training data not found: {train_dir}")
        sys.exit(1)

    data_dict = {}
    for tf in ["5m", "1h", "4h"]:
        path = train_dir / f"{tf}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            data_dict[tf] = df
            logger.info(f"  {tf}: {len(df)} rows, {len(df.columns)} cols")
        else:
            logger.warning(f"  {tf}: NOT FOUND at {path}")

    if not data_dict:
        logger.error("No training data found!")
        sys.exit(1)

    # Create StateBuilder with the same config as training env
    state_builder = StateBuilder(
        features_config=TRAIN_COLUMNS,
        window_sizes={tf: OBS_WINDOW for tf in ["5m", "1h", "4h"]},
        include_portfolio_state=True,
        normalize=True,
    )

    # Fit scalers on training data (exactly what the env does)
    logger.info("\nFitting scalers on training data...")
    state_builder.fit_scalers({"BTCUSDT": data_dict})

    # Save to prod_scalers/
    output_dir = str(PROJECT_ROOT / "prod_scalers")
    state_builder.save_scalers(output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("DONE — Production scalers saved to: prod_scalers/")
    logger.info("=" * 60)

    # Verify saved files
    out_path = Path(output_dir)
    for f in sorted(out_path.iterdir()):
        logger.info(f"  {f.name} ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
