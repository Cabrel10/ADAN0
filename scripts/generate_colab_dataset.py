#!/usr/bin/env python3
"""Generate the ADAN training dataset on Google Colab (or any machine).

This script uses **ccxt** to download 6 months of OHLCV candles for each
asset / timeframe pair defined in ``config/config.yaml``, computes the
exact same indicators with **pandas_ta**, and writes the Parquet files
into the directory tree expected by ``ChunkedDataLoader``::

    data/processed/indicators/{split}/{ASSET}/{tf}.parquet

Usage (Colab cell)::

    !cd ADAN/bot && python scripts/generate_colab_dataset.py

Environment variables (optional)::

    ADAN_CANDLES   – number of candles to fetch per tf (default: auto ~6 months)
    ADAN_EXCHANGE  – ccxt exchange id (default: binance)
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    print("pandas_ta not installed – run: pip install pandas_ta")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # bot/
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "indicators"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("generate_dataset")

# ---------------------------------------------------------------------------
# Configuration (must match config/config.yaml exactly)
# ---------------------------------------------------------------------------
ASSETS = ["BTCUSDT", "XRPUSDT"]
SPLITS = ["train", "test"]
TRAIN_RATIO = 0.85

TIMEFRAMES = {
    "5m": "5m",
    "1h": "1h",
    "4h": "4h",
}

# Approximate candles for ~6 months per timeframe
CANDLE_COUNTS = {
    "5m":  52_560,   # 6 months ≈ 182 days × 288 candles/day
    "1h":  4_380,    # 6 months ≈ 182 days × 24 candles/day
    "4h":  1_095,    # 6 months ≈ 182 days × 6 candles/day
}

# Allow override via env
_env_candles = os.environ.get("ADAN_CANDLES")
if _env_candles:
    _n = int(_env_candles)
    CANDLE_COUNTS = {tf: _n for tf in CANDLE_COUNTS}


# ---------------------------------------------------------------------------
# Indicator functions (matching config.yaml features_config)
# ---------------------------------------------------------------------------

def compute_5m_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 5m indicators: rsi_14, macd_12_26_9, bb_percent_b_20_2,
    atr_14, atr_20, atr_50, volume_ratio_20, ema_20_ratio, stoch_k_14_3_3.
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    df["rsi_14"] = ta.rsi(c, length=14)

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df["macd_12_26_9"] = macd.iloc[:, 0]  # MACD line
    else:
        df["macd_12_26_9"] = 0.0

    bbands = ta.bbands(c, length=20, std=2)
    if bbands is not None and not bbands.empty:
        df["bb_percent_b_20_2"] = bbands.iloc[:, -1]  # %B
    else:
        df["bb_percent_b_20_2"] = 0.5

    df["atr_14"] = ta.atr(h, l, c, length=14)
    df["atr_20"] = ta.atr(h, l, c, length=20)
    df["atr_50"] = ta.atr(h, l, c, length=50)

    sma_vol = v.rolling(20).mean()
    df["volume_ratio_20"] = v / sma_vol.replace(0, np.nan)

    ema20 = ta.ema(c, length=20)
    df["ema_20_ratio"] = c / ema20.replace(0, np.nan)

    stoch = ta.stoch(h, l, c, k=14, d=3, smooth_k=3)
    if stoch is not None and not stoch.empty:
        df["stoch_k_14_3_3"] = stoch.iloc[:, 0]
    else:
        df["stoch_k_14_3_3"] = 50.0

    return df


def compute_1h_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 1h indicators: rsi_21, macd_21_42_9, bb_width_20_2,
    adx_14, obv_ratio_20, ema_50_ratio, ichimoku_base, fib_ratio,
    price_ema_ratio_50.
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    df["rsi_21"] = ta.rsi(c, length=21)

    macd = ta.macd(c, fast=21, slow=42, signal=9)
    if macd is not None and not macd.empty:
        df["macd_21_42_9"] = macd.iloc[:, 0]
    else:
        df["macd_21_42_9"] = 0.0

    bbands = ta.bbands(c, length=20, std=2)
    if bbands is not None and not bbands.empty:
        upper = bbands.iloc[:, 2]
        lower = bbands.iloc[:, 0]
        mid = bbands.iloc[:, 1]
        df["bb_width_20_2"] = (upper - lower) / mid.replace(0, np.nan)
    else:
        df["bb_width_20_2"] = 0.0

    adx = ta.adx(h, l, c, length=14)
    if adx is not None and not adx.empty:
        df["adx_14"] = adx.iloc[:, 0]
    else:
        df["adx_14"] = 25.0

    # OBV ratio
    obv = ta.obv(c, v)
    if obv is not None:
        obv_sma = obv.rolling(20).mean()
        df["obv_ratio_20"] = obv / obv_sma.replace(0, np.nan)
    else:
        df["obv_ratio_20"] = 1.0

    ema50 = ta.ema(c, length=50)
    df["ema_50_ratio"] = c / ema50.replace(0, np.nan)

    # Ichimoku base line (Kijun-sen = (high_26 + low_26) / 2)
    high_26 = h.rolling(26).max()
    low_26 = l.rolling(26).min()
    kijun = (high_26 + low_26) / 2
    df["ichimoku_base"] = kijun / c.replace(0, np.nan)

    # Fibonacci ratio (price position in recent high-low range)
    high_55 = h.rolling(55).max()
    low_55 = l.rolling(55).min()
    fib_range = high_55 - low_55
    df["fib_ratio"] = (c - low_55) / fib_range.replace(0, np.nan)

    df["price_ema_ratio_50"] = c / ema50.replace(0, np.nan)

    return df


def compute_4h_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 4h indicators: rsi_28, macd_26_52_18, supertrend_10_3,
    volume_sma_20_ratio, ema_100_ratio, pivot_level, donchian_width_20,
    market_structure, volatility_ratio_14_50.
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    df["rsi_28"] = ta.rsi(c, length=28)

    macd = ta.macd(c, fast=26, slow=52, signal=18)
    if macd is not None and not macd.empty:
        df["macd_26_52_18"] = macd.iloc[:, 0]
    else:
        df["macd_26_52_18"] = 0.0

    # Supertrend
    st = ta.supertrend(h, l, c, length=10, multiplier=3)
    if st is not None and not st.empty:
        # Supertrend direction: 1 = bullish, -1 = bearish
        df["supertrend_10_3"] = st.iloc[:, 1]
    else:
        df["supertrend_10_3"] = 1.0

    sma_vol = v.rolling(20).mean()
    df["volume_sma_20_ratio"] = v / sma_vol.replace(0, np.nan)

    ema100 = ta.ema(c, length=100)
    df["ema_100_ratio"] = c / ema100.replace(0, np.nan)

    # Pivot level (classic: (H + L + C) / 3, normalized)
    pivot = (h.shift(1) + l.shift(1) + c.shift(1)) / 3
    df["pivot_level"] = c / pivot.replace(0, np.nan)

    # Donchian width
    high_20 = h.rolling(20).max()
    low_20 = l.rolling(20).min()
    df["donchian_width_20"] = (high_20 - low_20) / c.replace(0, np.nan)

    # Market structure: higher highs & higher lows detection (simplified)
    hh = (h > h.shift(1)).astype(float)
    hl = (l > l.shift(1)).astype(float)
    df["market_structure"] = (hh + hl) / 2.0  # 0=bearish, 1=bullish

    # Volatility ratio
    atr14 = ta.atr(h, l, c, length=14)
    atr50 = ta.atr(h, l, c, length=50)
    df["volatility_ratio_14_50"] = atr14 / atr50.replace(0, np.nan)

    return df


INDICATOR_FN = {
    "5m": compute_5m_indicators,
    "1h": compute_1h_indicators,
    "4h": compute_4h_indicators,
}


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def fetch_ohlcv(exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Download OHLCV from exchange via ccxt, handling pagination."""
    logger.info(f"Fetching {symbol} {timeframe} ({limit} candles)...")

    all_candles = []
    since = None
    batch = min(limit, 1000)

    while len(all_candles) < limit:
        remaining = limit - len(all_candles)
        fetch_count = min(batch, remaining)
        try:
            candles = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since, limit=fetch_count
            )
        except Exception as e:
            logger.warning(f"fetch_ohlcv error: {e}; retrying in 2s...")
            time.sleep(2)
            continue

        if not candles:
            break

        all_candles.extend(candles)
        since = candles[-1][0] + 1  # next ms
        time.sleep(exchange.rateLimit / 1000)  # respect rate limit

    df = pd.DataFrame(
        all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    logger.info(f"  → {len(df)} candles for {symbol} {timeframe}")
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_dataset():
    exchange_id = os.environ.get("ADAN_EXCHANGE", "")

    # Auto-detect a working exchange (binance is geo-blocked in many clouds)
    EXCHANGE_PRIORITY = [exchange_id] if exchange_id else []
    EXCHANGE_PRIORITY += ["binanceus", "binance", "kucoin"]

    exchange = None
    for eid in EXCHANGE_PRIORITY:
        eid = eid.strip()
        if not eid or not hasattr(ccxt, eid):
            continue
        try:
            logger.info(f"Trying exchange: {eid}")
            ex = getattr(ccxt, eid)({"enableRateLimit": True})
            ex.load_markets()
            exchange = ex
            logger.info(f"Using exchange: {eid} ({len(ex.markets)} markets)")
            break
        except Exception as e:
            logger.warning(f"Exchange {eid} unavailable: {e}")
            continue

    if exchange is None:
        logger.error("No exchange available – cannot generate dataset")
        sys.exit(1)

    for asset in ASSETS:
        symbol = asset.replace("USDT", "/USDT")  # ccxt format
        for tf_key, tf_ccxt in TIMEFRAMES.items():
            limit = CANDLE_COUNTS[tf_key]
            df = fetch_ohlcv(exchange, symbol, tf_ccxt, limit)

            if df.empty:
                logger.error(f"No data for {asset}/{tf_key} – skipping")
                continue

            # Compute indicators
            compute_fn = INDICATOR_FN[tf_key]
            df = compute_fn(df)

            # Replace NaN/Inf with forward-fill then zero
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill().fillna(0.0)

            # Set timestamp as DatetimeIndex (required by MultiAssetChunkedEnv)
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            else:
                df.index = pd.date_range("2024-01-01", periods=len(df), freq="1h", tz="UTC")
                df.index.name = "timestamp"

            # Split into train / test
            split_idx = int(len(df) * TRAIN_RATIO)
            splits = {
                "train": df.iloc[:split_idx].copy(),
                "test": df.iloc[split_idx:].copy(),
            }

            for split_name, split_df in splits.items():
                if split_df.empty:
                    continue
                out_dir = DATA_ROOT / split_name / asset
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{tf_key}.parquet"
                split_df.to_parquet(out_path, index=True)
                logger.info(
                    f"  ✅ {out_path} — {len(split_df)} rows, "
                    f"{len(split_df.columns)} cols: {list(split_df.columns)[:6]}..."
                )

    logger.info("=" * 60)
    logger.info("Dataset generation complete!")
    logger.info(f"Data root: {DATA_ROOT}")

    # Print tree
    for split in SPLITS:
        split_dir = DATA_ROOT / split
        if split_dir.exists():
            for asset_dir in sorted(split_dir.iterdir()):
                if asset_dir.is_dir():
                    for pq in sorted(asset_dir.glob("*.parquet")):
                        size_mb = pq.stat().st_size / 1024 / 1024
                        logger.info(f"  {pq.relative_to(DATA_ROOT)} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    build_dataset()
