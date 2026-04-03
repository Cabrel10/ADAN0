#!/usr/bin/env python3
"""
ADAN Master Clock Dataset Generator
====================================
Generates temporally-aligned multi-timeframe datasets for training.

The 5m timeframe is the MASTER CLOCK. Higher timeframes (1h, 4h) are
reindexed onto the 5m DatetimeIndex using forward-fill (ffill).

Two modes:
  1. default (--live): fetch real OHLCV from Binance PUBLIC API (no key needed)
  2. --synthetic: generate synthetic data locally

Examples:
    # Real data from Binance (no API key needed):
    python scripts/generate_colab_dataset.py --candles 5000 --symbols BTCUSDT

    # Synthetic data:
    python scripts/generate_colab_dataset.py --synthetic --candles 5000
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Project root
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("generate_dataset")

# Import the real IndicatorCalculator
from adan_trading_bot.indicators.calculator import IndicatorCalculator

# ---------------------------------------------------------------------------
# Asset profiles for synthetic generation
# ---------------------------------------------------------------------------
ASSET_PROFILES = {
    "BTCUSDT":  {"base": 65000, "daily_vol": 0.025, "trend": 0.0001},
    "ETHUSDT":  {"base": 3400,  "daily_vol": 0.030, "trend": 0.00008},
    "XRPUSDT":  {"base": 0.55,  "daily_vol": 0.035, "trend": 0.00005},
    "SOLUSDT":  {"base": 140,   "daily_vol": 0.040, "trend": 0.00012},
    "BNBUSDT":  {"base": 580,   "daily_vol": 0.022, "trend": 0.00006},
    "DOGEUSDT": {"base": 0.15,  "daily_vol": 0.045, "trend": 0.00003},
}
DEFAULT_PROFILE = {"base": 100, "daily_vol": 0.030, "trend": 0.00005}


# ---------------------------------------------------------------------------
# Synthetic OHLCV generation
# ---------------------------------------------------------------------------
def generate_synthetic_ohlcv(symbol: str, n_candles: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic 5m OHLCV via geometric Brownian motion."""
    rng = np.random.default_rng(seed + hash(symbol) % 2**31)
    profile = ASSET_PROFILES.get(symbol.upper(), DEFAULT_PROFILE)
    base_price = profile["base"]
    bar_vol = profile["daily_vol"] / np.sqrt(288)
    trend = profile["trend"]

    n = n_candles
    regimes = np.ones(n)
    regime = 0
    for i in range(n):
        if regime == 0 and rng.random() < 0.005:
            regime = 1
        elif regime == 1 and rng.random() < 0.02:
            regime = 0
        regimes[i] = 1.0 if regime == 0 else 2.5

    returns = trend + bar_vol * regimes * rng.standard_normal(n)
    log_prices = np.log(base_price) + np.cumsum(returns)
    close = np.exp(log_prices)

    spread = bar_vol * regimes * close
    high = close + rng.uniform(0.1, 1.0, n) * spread
    low = close - rng.uniform(0.1, 1.0, n) * spread
    low = np.maximum(low, close * 0.995)
    opn = np.roll(close, 1)
    opn[0] = base_price
    volume = base_price * 10 * regimes * rng.uniform(0.5, 2.0, n)

    end = pd.Timestamp.now(tz="UTC").floor("5min")
    idx = pd.date_range(end=end, periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "timestamp"
    logger.info(f"  Synthetic {symbol}: {n} candles, price {close[0]:.2f} -> {close[-1]:.2f}")
    return df


# ---------------------------------------------------------------------------
# Live data fetching via ccxt (NO API KEY needed for public endpoints)
# ---------------------------------------------------------------------------
def fetch_public_ohlcv(symbol: str = "BTC/USDT", timeframe: str = "5m", limit: int = 1000) -> pd.DataFrame:
    """Fetch OHLCV from public exchange APIs -- ZERO API KEY REQUIRED.

    Tries exchanges in order: Binance -> Bybit -> Bitget.
    All connections are 100% anonymous (enableRateLimit only).
    """
    try:
        import ccxt
    except ImportError:
        logger.error("ccxt not installed. Run: pip install ccxt")
        sys.exit(1)

    # Priority list of exchanges -- all anonymous, no API key.
    exchanges = [
        ("binance", ccxt.binance({"enableRateLimit": True})),
        ("bybit",   ccxt.bybit({"enableRateLimit": True})),
        ("bitget",  ccxt.bitget({"enableRateLimit": True})),
    ]

    tf_ms = {"1m": 60_000, "5m": 300_000, "15m": 900_000,
             "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
    bar_ms = tf_ms.get(timeframe, 300_000)

    for name, exchange in exchanges:
        logger.info(f"Trying {name} (public, no API key) for {symbol} {timeframe} x{limit}...")
        try:
            now_ms = exchange.milliseconds()
            since = now_ms - limit * bar_ms
            all_candles: list = []
            while len(all_candles) < limit:
                batch = min(1000, limit - len(all_candles))
                candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=batch)
                if not candles:
                    break
                all_candles.extend(candles)
                since = candles[-1][0] + 1
                if candles[-1][0] >= now_ms:
                    break
                time.sleep(exchange.rateLimit / 1000)

            if not all_candles:
                logger.warning(f"  {name}: returned 0 candles, trying next exchange...")
                continue

            df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp").sort_index()
            df = df[~df.index.duplicated(keep="first")]
            logger.info(f"  -> {name}: {len(df)} candles [{df.index.min()} .. {df.index.max()}]")
            return df

        except Exception as e:
            logger.warning(f"  {name} failed: {e}")
            continue

    raise RuntimeError(
        f"All exchanges failed for {symbol}. "
        "If geo-blocked, use --synthetic instead."
    )


def resample_ohlcv(df_base: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df_base.resample(target_tf).agg(agg).dropna()


# ---------------------------------------------------------------------------
# Master Clock alignment
# ---------------------------------------------------------------------------
def align_to_master_clock(df_5m, df_1h, df_4h):
    master_idx = df_5m.index.copy()
    df_1h_aligned = df_1h.reindex(master_idx, method="ffill")
    df_4h_aligned = df_4h.reindex(master_idx, method="ffill")
    valid = df_1h_aligned["close"].notna() & df_4h_aligned["close"].notna()
    master_idx = master_idx[valid]
    return {
        "5m": df_5m.loc[master_idx].copy(),
        "1h": df_1h_aligned.loc[master_idx].copy(),
        "4h": df_4h_aligned.loc[master_idx].copy(),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def generate_dataset(
    output_dir: str,
    symbol: str = "BTCUSDT",
    n_candles: int = 5000,
    synthetic: bool = False,
    split: str = "train",
    seed: int = 42,
):
    asset_name = symbol.replace("/", "").upper()
    # Build the ccxt symbol (e.g. "BTC/USDT") from the asset name.
    # We try known quote currencies in order of specificity.
    if "/" in symbol:
        ccxt_symbol = symbol
    else:
        _quotes = ["USDT", "BUSD", "USDC", "BTC", "ETH", "BNB"]
        ccxt_symbol = symbol  # fallback
        for q in _quotes:
            if asset_name.endswith(q) and len(asset_name) > len(q):
                ccxt_symbol = f"{asset_name[:-len(q)]}/{q}"
                break

    mode = "synthetic" if synthetic else "live (Binance public)"
    logger.info(f"Generating dataset for {asset_name} ({n_candles} candles, mode={mode})")

    # 1. Get raw 5m data
    if synthetic:
        df_5m_raw = generate_synthetic_ohlcv(asset_name, n_candles, seed=seed)
    else:
        df_5m_raw = fetch_public_ohlcv(ccxt_symbol, "5m", limit=n_candles)

    if len(df_5m_raw) < 200:
        logger.error(f"Insufficient data: {len(df_5m_raw)} candles. Need at least 200.")
        return None

    # 2. Resample to higher timeframes
    df_1h_raw = resample_ohlcv(df_5m_raw, "1h")
    df_4h_raw = resample_ohlcv(df_5m_raw, "4h")
    logger.info(f"Resampled: 1h={len(df_1h_raw)}, 4h={len(df_4h_raw)}")

    # 3. Compute indicators using IndicatorCalculator (pandas_ta based)
    logger.info("Computing indicators...")
    df_5m = IndicatorCalculator.calculate_features_df(df_5m_raw.copy(), "5m")
    df_1h = IndicatorCalculator.calculate_features_df(df_1h_raw.copy(), "1h")
    df_4h = IndicatorCalculator.calculate_features_df(df_4h_raw.copy(), "4h")

    # Add extra columns needed by state_builder but not in IndicatorCalculator
    for df, tf in [(df_5m, "5m"), (df_1h, "1h"), (df_4h, "4h")]:
        c = df["close"]
        v = df["volume"]
        # log_return
        df["log_return"] = np.log(c / c.shift(1).replace(0, np.nan))
        # close_ema20_ratio (alias of ema_20_ratio if present, else compute)
        if "ema_20_ratio" in df.columns:
            df["close_ema20_ratio"] = df["ema_20_ratio"]
        else:
            ema20 = c.ewm(span=20, adjust=False).mean()
            df["close_ema20_ratio"] = c / ema20.replace(0, np.nan)
        # spread_bps and liquidity_score (synthetic microstructure)
        if "atr_14" in df.columns:
            df["spread_bps"] = (df["atr_14"] / c.replace(0, np.nan)) * 10000 * 0.1
        else:
            df["spread_bps"] = 0.0
        vol_log = np.log1p(v)
        vol_sma = vol_log.rolling(20).mean().replace(0, np.nan)
        df["liquidity_score"] = vol_log / vol_sma
        # price_action (candle body position)
        h, lo = df["high"], df["low"]
        df["price_action"] = (c - lo) / (h - lo).replace(0, np.nan)
        # fib_ratio (for 1h)
        if tf == "1h":
            high52 = h.rolling(52).max()
            low52 = lo.rolling(52).min()
            df["fib_ratio"] = (c - low52) / (high52 - low52).replace(0, np.nan)
            ema50 = c.ewm(span=50, adjust=False).mean()
            df["price_ema_ratio_50"] = c / ema50.replace(0, np.nan)
        # market_structure (for 4h)
        if tf == "4h":
            ema20 = c.ewm(span=20, adjust=False).mean()
            ema50 = c.ewm(span=50, adjust=False).mean()
            df["market_structure"] = np.where(
                ema20 > ema50, 1.0, np.where(ema20 < ema50, -1.0, 0.0)
            ).astype(np.float64)
            if "atr_14" in df.columns and "atr_50" in df.columns:
                df["volatility_ratio_14_50"] = df["atr_14"] / df["atr_50"].replace(0, np.nan)

    # 4. Drop NaN warmup rows
    df_5m = df_5m.replace([np.inf, -np.inf], np.nan).dropna(subset=["rsi_14", "atr_14"])
    df_1h = df_1h.replace([np.inf, -np.inf], np.nan).dropna(subset=["rsi_21", "atr_14"])
    df_4h = df_4h.replace([np.inf, -np.inf], np.nan).dropna(subset=["rsi_28", "atr_14"])
    logger.info(f"After indicators: 5m={len(df_5m)}, 1h={len(df_1h)}, 4h={len(df_4h)}")

    if len(df_4h) < 10:
        logger.error("Not enough 4h candles after indicator warmup. Use more candles (--candles 5000+).")
        return None

    # 5. Master Clock alignment
    aligned = align_to_master_clock(df_5m, df_1h, df_4h)
    for tf, df in aligned.items():
        logger.info(f"  Aligned {tf}: {len(df)} rows, {len(df.columns)} cols")

    if len(aligned["5m"]) == 0:
        logger.error("Alignment produced 0 rows. Check data coverage.")
        return None

    # 6. Save to parquet
    out_base = Path(output_dir) / split / asset_name
    out_base.mkdir(parents=True, exist_ok=True)
    for tf, df in aligned.items():
        # Final cleanup: replace inf/nan with 0
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out_path = out_base / f"{tf}.parquet"
        df.to_parquet(out_path, engine="pyarrow")
        logger.info(f"  Saved {out_path} ({len(df)} rows, {len(df.columns)} cols)")

    # 7. Verify index consistency
    assert set(aligned["5m"].index) == set(aligned["1h"].index) == set(aligned["4h"].index)
    logger.info(f"MASTER CLOCK VERIFIED: {len(aligned['5m'])} aligned timestamps for {asset_name}.")
    return aligned


def main():
    parser = argparse.ArgumentParser(description="ADAN Dataset Generator")
    parser.add_argument("--output", default="data/processed/indicators")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--candles", type=int, default=int(os.getenv("ADAN_CANDLES", "5000")))
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of real Binance data")
    parser.add_argument("--live", action="store_true",
                        help="(legacy flag, now default behaviour)")
    parser.add_argument("--no-testnet", action="store_true", help="(legacy, ignored)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for symbol in args.symbols:
        generate_dataset(
            output_dir=args.output,
            symbol=symbol,
            n_candles=args.candles,
            synthetic=args.synthetic,
            split=args.split,
            seed=args.seed,
        )
    logger.info(f"Done: {args.symbols}")


if __name__ == "__main__":
    main()
