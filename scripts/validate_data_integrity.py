import pandas as pd
from pathlib import Path
import yaml
import logging

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed' / 'indicators'
CONFIG_FILE = BASE_DIR / 'config' / 'config.yaml'
DATA_CONFIG_FILE = BASE_DIR / 'config' / 'data.yaml'
LOG_FILE = BASE_DIR / 'logs' / 'data_validation.log'

# --- Setup Logger ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)

def load_config():
    """Loads asset and timeframe configuration."""
    try:
        with open(DATA_CONFIG_FILE, 'r') as f:
            data_config = yaml.safe_load(f)
        
        assets = data_config['data']['file_structure']['assets']
        timeframes = data_config['data']['file_structure']['timeframes']
        return assets, timeframes
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        return [], []

def validate_file(file_path: Path):
    """Validates a single parquet file."""
    try:
        df = pd.read_parquet(file_path)
        logging.info(f"Validating {file_path}...")

        # 1. Check for NaNs
        if df.isna().sum().sum() > 0:
            logging.warning(f"  -> NaN values found in {file_path}")

        # 2. Check for duplicates
        if 'TIMESTAMP' in df.columns:
            if len(df) != len(df.drop_duplicates('TIMESTAMP')):
                logging.warning(f"  -> Duplicate TIMESTAMPS found in {file_path}")
        elif df.index.name == 'timestamp':
             if len(df) != len(df.reset_index().drop_duplicates('timestamp')):
                logging.warning(f"  -> Duplicate timestamps found in index of {file_path}")
        else:
            logging.warning(f"  -> No TIMESTAMP column or index found in {file_path} to check for duplicates.")


        # 3. Check data types
        if 'TIMESTAMP' in df.columns and df['TIMESTAMP'].dtype != 'int64':
             logging.warning(f"  -> TIMESTAMP column in {file_path} is not int64.")
        
        for col in df.columns:
            if col.upper() not in ['TIMESTAMP'] and df[col].dtype != 'float32':
                # This might be too strict, as some columns might be int64.
                # Let's check for float64 and suggest float32.
                if df[col].dtype == 'float64':
                    logging.info(f"  -> Column {col} in {file_path} is float64, consider converting to float32 for memory efficiency.")

        # 4. Check for reasonable length
        # These are just examples, the user can adjust them
        if '5m' in str(file_path) and not (500000 < len(df) < 1000000):
             logging.warning(f"  -> Length of {file_path} ({len(df)}) is outside the expected range for 5m data.")
        if '1h' in str(file_path) and not (40000 < len(df) < 80000):
            logging.warning(f"  -> Length of {file_path} ({len(df)}) is outside the expected range for 1h data.")
        if '4h' in str(file_path) and not (10000 < len(df) < 20000):
            logging.warning(f"  -> Length of {file_path} ({len(df)}) is outside the expected range for 4h data.")

        logging.info(f"  -> Validation for {file_path} complete.")

    except Exception as e:
        logging.error(f"Failed to validate {file_path}: {e}")

def main():
    """Main function to validate all data files."""
    logging.info("=== STARTING DATA INTEGRITY VALIDATION ===")
    assets, timeframes = load_config()
    
    if not assets or not timeframes:
        logging.error("Could not load assets/timeframes. Aborting.")
        return

    for split in ['train', 'val', 'test']:
        logging.info(f"\n--- Validating {split} data ---")
        split_dir = PROCESSED_DATA_DIR / split
        if not split_dir.exists():
            logging.warning(f"Directory not found: {split_dir}")
            continue
            
        for asset in assets:
            for timeframe in timeframes:
                file_path = split_dir / asset / f"{timeframe}.parquet"
                if file_path.exists():
                    validate_file(file_path)
                else:
                    logging.warning(f"File not found: {file_path}")

    logging.info("\n=== DATA INTEGRITY VALIDATION COMPLETE ===")
    logging.info(f"Log file saved to: {LOG_FILE}")

if __name__ == "__main__":
    main()
