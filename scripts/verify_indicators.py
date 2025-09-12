import pandas as pd
import yaml
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def verify_parquet_files(parquet_files, expected_features_config):
    """
    Verifies each Parquet file for NaN values and expected columns.
    """
    overall_status = "SUCCESS"
    
    for file_path in parquet_files:
        logging.info(f"\n--- Verifying {file_path} ---")
        try:
            df = pd.read_parquet(file_path)
            
            # Extract asset and timeframe from path
            path_parts = Path(file_path).parts
            asset = path_parts[-2] # e.g., 'BTC'
            timeframe = path_parts[-1].replace('.parquet', '') # e.g., '5m'
            
            logging.info(f"Asset: {asset}, Timeframe: {timeframe}")
            logging.info(f"DataFrame shape: {df.shape}")

            # 1. Check for NaN values
            nan_count = df.isnull().sum().sum()
            if nan_count > 0:
                logging.warning(f"  FAIL: Found {nan_count} NaN values in the DataFrame.")
                overall_status = "WARNING"
                # Log columns with NaNs
                nan_columns = df.columns[df.isnull().any()].tolist()
                logging.warning(f"  Columns with NaNs: {nan_columns}")
            else:
                logging.info("  PASS: No NaN values found.")

            # 2. Verify expected indicator columns
            expected_features = expected_features_config.get(timeframe, [])
            if not expected_features:
                logging.warning(f"  WARNING: No expected features defined for timeframe {timeframe} in config. Skipping column check.")
            else:
                missing_features = [f for f in expected_features if f not in df.columns]
                if missing_features:
                    logging.warning(f"  FAIL: Missing {len(missing_features)} expected features: {missing_features}")
                    overall_status = "WARNING"
                else:
                    logging.info(f"  PASS: All {len(expected_features)} expected features are present.")
                    
                # Optional: Check for unexpected columns (might indicate issues in calculation)
                unexpected_columns = [col for col in df.columns if col not in expected_features and col != 'timestamp']
                if unexpected_columns:
                    logging.info(f"  INFO: Found {len(unexpected_columns)} unexpected columns (not in config): {unexpected_columns}")

        except Exception as e:
            logging.error(f"  ERROR: Could not process {file_path}: {e}")
            overall_status = "ERROR"
            
    logging.info(f"\n=== Overall Verification Status: {overall_status} ===")
    return overall_status

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    config_path = os.path.join(project_root, 'ADAN', 'config', 'config.yaml')
    
    config = load_config(config_path)
    expected_features_config = config['data']['features_per_timeframe']

    # List of parquet files obtained from previous glob command
    parquet_files = [
        "/home/morningstar/Documents/trading/ADAN/data/processed/indicators/ADA/4h.parquet",
        "/home/morningstar/Documents/trading/ADAN/data/processed/indicators/XRP/4h.parquet",
        "/home/morningstar/Documents/trading/ADAN/data/processed/indicators/SOL/4h.parquet",
        "/home/morningstar/Documents/trading/ADAN/data/processed/indicators/ETH/4h.parquet",
        "/home/morningstar/Documents/trading/ADAN/data/processed/indicators/BTC/4h.parquet",
        "/home/morningstar/Documents/trading/ADAN/data/processed/indicators/ADA/1h.parquet",
        "/home/morningstar/Documents/trading/ADAN/data/processed/indicators/XRP/1h.parquet",
        "/home/morningstar/Documents/trading/ADAN/data/processed/indicators/SOL/1h.parquet",
        "/home/morningstar/Documents/trading/ADAN/data/processed/indicators/ETH/1h.parquet",
        "/home/morningstar/Documents/trading/ADAN/data/processed/indicators/BTC/1h.parquet",
        "/home/morningstar/Documents/trading/ADAN/data/processed/indicators/ADA/5m.parquet",
        "/home/morningstar/Documents/trading/ADAN/data/processed/indicators/XRP/5m.parquet",
        "/home/morningstar/Documents/trading/ADAN/data/processed/indicators/SOL/5m.parquet",
        "/home/morningstar/Documents/trading/ADAN/data/processed/indicators/ETH/5m.parquet",
        "/home/morningstar/Documents/trading/ADAN/data/processed/indicators/BTC/5m.parquet"
    ]
    
    verify_parquet_files(parquet_files, expected_features_config)
