
import pandas as pd

files = [
    "/home/morningstar/Documents/trading/bot/data/processed/indicators/train/ADAUSDT/1h.parquet",
    "/home/morningstar/Documents/trading/bot/data/processed/indicators/train/ADAUSDT/4h.parquet",
    "/home/morningstar/Documents/trading/bot/data/processed/indicators/train/ADAUSDT/5m.parquet",
    "/home/morningstar/Documents/trading/bot/data/processed/indicators/train/BTCUSDT/1h.parquet",
    "/home/morningstar/Documents/trading/bot/data/processed/indicators/train/BTCUSDT/4h.parquet",
    "/home/morningstar/Documents/trading/bot/data/processed/indicators/train/BTCUSDT/5m.parquet",
    "/home/morningstar/Documents/trading/bot/data/processed/indicators/train/ETHUSDT/1h.parquet",
    "/home/morningstar/Documents/trading/bot/data/processed/indicators/train/ETHUSDT/4h.parquet",
    "/home/morningstar/Documents/trading/bot/data/processed/indicators/train/ETHUSDT/5m.parquet",
    "/home/morningstar/Documents/trading/bot/data/processed/indicators/train/SOLUSDT/1h.parquet",
    "/home/morningstar/Documents/trading/bot/data/processed/indicators/train/SOLUSDT/4h.parquet",
    "/home/morningstar/Documents/trading/bot/data/processed/indicators/train/SOLUSDT/5m.parquet",
    "/home/morningstar/Documents/trading/bot/data/processed/indicators/train/XRPUSDT/1h.parquet",
    "/home/morningstar/Documents/trading/bot/data/processed/indicators/train/XRPUSDT/4h.parquet",
    "/home/morningstar/Documents/trading/bot/data/processed/indicators/train/XRPUSDT/5m.parquet"
]

for file in files:
    try:
        df = pd.read_parquet(file)
        print(f"--- {file} ---")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"Column Names: {df.columns.tolist()}")
        print("\n")
    except Exception as e:
        print(f"Error reading {file}: {e}")
