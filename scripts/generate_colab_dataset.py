#!/usr/bin/env python3
"""
ADAN Master Clock Dataset Generator
====================================
Generates temporally-aligned multi-timeframe datasets for training.

The 5m timeframe is the MASTER CLOCK. Higher timeframes (1h, 4h) are
reindexed onto the 5m DatetimeIndex using forward-fill (ffill), so at
any row i, ALL timeframes refer to the same wall-clock minute.

This eliminates the "time-travel bug" where the agent could see future
candles from misaligned timeframe indices.

Usage:
    # From Binance API (live or testnet):
    ADAN_CANDLES=2000 python scripts/generate_colab_dataset.py

    # With custom output directory:
    python scripts/generate_colab_dataset.py --output data/processed/indicators/train

Environment variables:
    ADAN_CANDLES     Number of 5m candles to fetch (default: 2000)
    BINANCE_API_KEY  API key (uses testnet if BINANCE_TESTNET=true)
    BINANCE_SECRET_KEY  API secret
    BINANCE_TESTNET  Set to 'true' for testnet (default: true)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("generate_colab_dataset")


# ── Indicator calculation (self-contained, no external dependency) ─────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute essential technical indicators on an OHLCV DataFrame.

    Adds RSI-14, MACD, Bollinger Bands, ATR-14, EMA-20, SMA-50, OBV, etc.
    All indicators are computed in-place; NaN rows from warm-up are dropped.
    """
    c = df["close"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"]

    # RSI-14
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # EMA-12 / EMA-26 / MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20, 2)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20.replace(0, np.nan)

    # ATR-14
    tr = pd.concat(
        [h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1
    ).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # EMAs & SMAs
    df["ema_20"] = c.ewm(span=20, adjust=False).mean()
    df["sma_50"] = c.rolling(50).mean()

    # OBV (on balance volume)
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df["obv"] = obv

    return df


# ── Master Clock alignment ─────────────────────────────────────────────────
def align_to_master_clock(
    df_5m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
) -> dict:
    """Reindex 1h and 4h DataFrames onto the 5m Master Clock using ffill.

    At every 5m timestamp t:
      - 1h data shows the LAST COMPLETED 1h candle as of t.
      - 4h data shows the LAST COMPLETED 4h candle as of t.

    Returns:
        dict: {"5m": df_5m, "1h": df_1h_aligned, "4h": df_4h_aligned}
    """
    master_idx = df_5m.index.copy()

    # Reindex 1h onto 5m index, forward-fill
    df_1h_aligned = df_1h.reindex(master_idx, method="ffill")

    # Reindex 4h onto 5m index, forward-fill
    df_4h_aligned = df_4h.reindex(master_idx, method="ffill")

    # Drop rows where ffill couldn't fill (beginning of dataset)
    valid_mask = df_1h_aligned["close"].notna() & df_4h_aligned["close"].notna()
    master_idx = master_idx[valid_mask]

    return {
        "5m": df_5m.loc[master_idx].copy(),
        "1h": df_1h_aligned.loc[master_idx].copy(),
        "4h": df_4h_aligned.loc[master_idx].copy(),
    }


# ── Data fetching (ccxt) ───────────────────────────────────────────────────
def fetch_binance_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "5m",
    limit: int = 1000,
    testnet: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV data from Binance (or testnet) via ccxt."""
    try:
        import ccxt
    except ImportError:
        logger.error("ccxt not installed. Run: pip install ccxt")
        sys.exit(1)

    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_SECRET_KEY", "")

    exchange = ccxt.binance(
        {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
    )
    if testnet:
        exchange.set_sandbox_mode(True)

    logger.info(f"Fetching {limit} candles of {symbol} {timeframe} from Binance{'(testnet)' if testnet else ''}...")

    # Multi-pass fetch for more data
    all_candles = []
    since = None
    remaining = limit

    while remaining > 0:
        batch_limit = min(remaining, 1000)
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=batch_limit)
        if not candles:
            break
        all_candles.extend(candles)
        remaining -= len(candles)
        if len(candles) < batch_limit:
            break
        # Move since to after the last candle
        since = candles[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)  # Respect rate limit

    df = pd.DataFrame(
        all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    logger.info(f"  -> {len(df)} candles from {df.index.min()} to {df.index.max()}")
    return df


def resample_ohlcv(df_base: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample a base-timeframe OHLCV DataFrame to a higher timeframe."""
    agg_rules = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return df_base.resample(target_tf).agg(agg_rules).dropna()


# ── Main logic ─────────────────────────────────────────────────────────────
def generate_dataset(
    output_dir: str,
    symbol: str = "BTC/USDT",
    n_candles: int = 2000,
    testnet: bool = True,
    split: str = "train",
):
    """Generate Master-Clock-aligned dataset and save as parquet."""
    asset_name = symbol.replace("/", "")

    # 1. Fetch 5m data (master clock)
    df_5m_raw = fetch_binance_ohlcv(symbol, "5m", limit=n_candles, testnet=testnet)

    if len(df_5m_raw) < 100:
        logger.error(f"Insufficient 5m data: {len(df_5m_raw)} < 100. Cannot proceed.")
        sys.exit(1)

    # 2. Resample to 1h and 4h
    df_1h_raw = resample_ohlcv(df_5m_raw, "1h")
    df_4h_raw = resample_ohlcv(df_5m_raw, "4h")

    logger.info(f"Resampled: 1h={len(df_1h_raw)} candles, 4h={len(df_4h_raw)} candles")

    # 3. Compute indicators on each timeframe BEFORE alignment
    df_5m = compute_indicators(df_5m_raw.copy())
    df_1h = compute_indicators(df_1h_raw.copy())
    df_4h = compute_indicators(df_4h_raw.copy())

    # Drop NaN rows from indicator warm-up
    df_5m = df_5m.dropna(subset=["rsi_14", "atr_14"])
    df_1h = df_1h.dropna(subset=["rsi_14", "atr_14"])
    df_4h = df_4h.dropna(subset=["rsi_14", "atr_14"])

    logger.info(
        f"After indicators: 5m={len(df_5m)}, 1h={len(df_1h)}, 4h={len(df_4h)}"
    )

    # 4. MASTER CLOCK ALIGNMENT
    aligned = align_to_master_clock(df_5m, df_1h, df_4h)
    for tf, df in aligned.items():
        logger.info(
            f"  Aligned {tf}: {len(df)} rows, "
            f"{df.index.min()} -> {df.index.max()}"
        )

    # 5. Save to parquet
    out_base = Path(output_dir) / split / asset_name
    out_base.mkdir(parents=True, exist_ok=True)

    for tf, df in aligned.items():
        out_path = out_base / f"{tf}.parquet"
        df.to_parquet(out_path, engine="pyarrow")
        logger.info(f"  Saved {out_path} ({len(df)} rows, {len(df.columns)} cols)")

    # 6. Verification: all DataFrames share the same index
    idx_5m = set(aligned["5m"].index)
    idx_1h = set(aligned["1h"].index)
    idx_4h = set(aligned["4h"].index)
    assert idx_5m == idx_1h == idx_4h, (
        f"Index mismatch after alignment! "
        f"5m={len(idx_5m)}, 1h={len(idx_1h)}, 4h={len(idx_4h)}"
    )
    logger.info(f"MASTER CLOCK VERIFIED: All {len(idx_5m)} timestamps identical.")

    return aligned


def main():
    parser = argparse.ArgumentParser(description="ADAN Master Clock Dataset Generator")
    parser.add_argument(
        "--output",
        default="data/processed/indicators",
        help="Base output directory (split/ASSET/tf.parquet)",
    )
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair")
    parser.add_argument("--split", default="train", help="Data split (train/test/val)")
    parser.add_argument(
        "--candles",
        type=int,
        default=int(os.getenv("ADAN_CANDLES", "2000")),
        help="Number of 5m candles to fetch",
    )
    parser.add_argument(
        "--no-testnet",
        action="store_true",
        help="Use Binance mainnet instead of testnet",
    )
    args = parser.parse_args()

    testnet = not args.no_testnet
    if os.getenv("BINANCE_TESTNET", "true").lower() == "false":
        testnet = False

    generate_dataset(
        output_dir=args.output,
        symbol=args.symbol,
        n_candles=args.candles,
        testnet=testnet,
        split=args.split,
    )
    logger.info("Dataset generation complete.")


if __name__ == "__main__":
    main()
