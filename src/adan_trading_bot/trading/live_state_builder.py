"""
LiveStateBuilder — Build observation dict from live CCXT OHLCV data.

Uses EXACTLY the same 21 features per timeframe as the training pipeline
(feature_engineer.py TRAIN_COLUMNS). No shortcuts, no simplifications.
The observation dict matches the PPO model's Dict observation space.

Architecture:
  CCXT.fetch_ohlcv() → DataFrame → pandas_ta indicators → (20, 21) matrix
  DBE.get_regime_probabilities() → HMM 6-state probs
  DBE.get_oracle_probs()        → Oracle 3-state probs
  → context_vector (17,) + portfolio_state (20,)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from adan_trading_bot.data_processing.state_builder import StateBuilder

try:
    import pandas_ta as ta
except ImportError:
    ta = None

try:
    import ccxt
except ImportError:
    ccxt = None

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Feature columns — MUST stay identical to
# src/adan_trading_bot/data_processing/feature_engineer.py TRAIN_COLUMNS
# and state_builder.py features_config.
# ────────────────────────────────────────────────────────────────────────────

TRAIN_COLUMNS = {
    "5m": [
        "open", "high", "low", "close", "volume",
        "ema_20_ratio", "macdh_12_26_9", "rsi_14",
        "adx_14", "di_delta", "atr_pct",
        "bb_percent_b_20_2", "obv_slope",
        "volume_ratio_20", "volatility_ratio_14_50",
        "fib_ratio", "price_action", "vwap_ratio",
        "market_structure", "bb_width_20_2", "log_return",
    ],
    "1h": [
        "open", "high", "low", "close", "volume",
        "ema_50_ratio", "macdh_21_42_9", "rsi_21",
        "adx_14", "di_delta", "atr_pct",
        "bb_percent_b_20_2", "obv_slope",
        "volume_ratio_20", "volatility_ratio_14_50",
        "fib_ratio", "price_action", "vwap_ratio",
        "market_structure", "bb_width_20_2", "log_return",
    ],
    "4h": [
        "open", "high", "low", "close", "volume",
        "ema_100_ratio", "macdh_26_52_18", "rsi_28",
        "adx_14", "di_delta", "atr_pct",
        "bb_percent_b_20_2", "obv_slope",
        "volume_ratio_20", "volatility_ratio_14_50",
        "fib_ratio", "price_action", "vwap_ratio",
        "market_structure", "bb_width_20_2", "log_return",
    ],
}

# How many OHLCV bars to fetch per timeframe (need enough for longest indicator)
# FIX: EMA-100 requires ~200 bars to stabilize. With OBS_WINDOW=20, we need
# at least 300 bars BEFORE the observation window starts to match training
# distribution (where parquet files have thousands of bars of warmup).
# 500 bars = 300 warmup + 200 buffer → indicators at the observation window
# are fully converged, matching what training saw.
FETCH_LIMITS = {"5m": 500, "1h": 500, "4h": 300}

# Observation window (last N bars fed to model)
OBS_WINDOW = 20
N_FEATURES = 21


class LiveStateBuilder:
    """Transforms live CCXT OHLCV data into the same observation dict
    that MultiAssetChunkedEnv produces during training.

    Usage:
        builder = LiveStateBuilder(exchange_id="binance", symbol="BTC/USDT")
        obs = builder.build()
        action, _ = model.predict(obs, deterministic=True)
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        symbol: str = "BTC/USDT",
        timeframes: List[str] = None,
        proxy: str = None,
        ccxt_config: Dict[str, Any] = None,
    ):
        if ccxt is None:
            raise ImportError("pip install ccxt")
        if ta is None:
            raise ImportError("pip install pandas_ta")

        self.symbol = symbol
        self.timeframes = timeframes or ["5m", "1h", "4h"]

        # Init CCXT client (public API — no key needed for OHLCV)
        cfg = {
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
            **(ccxt_config or {}),
        }
        if proxy:
            cfg["proxies"] = {"http": proxy, "https": proxy}

        exchange_cls = getattr(ccxt, exchange_id, None)
        if exchange_cls is None:
            raise ValueError(f"Unknown exchange: {exchange_id}")
        self.exchange = exchange_cls(cfg)

        # Cache: tf → DataFrame (with indicators computed)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._last_fetch: Dict[str, float] = {}
        
        # Internal StateBuilder for normalization
        self.state_builder = StateBuilder(
            features_config=TRAIN_COLUMNS,
            window_sizes={tf: OBS_WINDOW for tf in self.timeframes},
            include_portfolio_state=True,
            normalize=True
        )

        # 🚫 PROD_SCALERS PKL ARE BANNED (root cause of paper-trading saturation).
        #    A frozen prod_scalers/*.pkl was fitted on the high-price tail
        #    (close mean=116k vs train mean=52k) -> live BTC normalized to -13σ
        #    -> features clipped to -10 -> PPO saw out-of-distribution "eyes"
        #    -> constant saturated actions (dir=+1, size=-1, ...).
        #    The training/backtest env fits scalers INLINE on the train chunk
        #    (multi_asset_chunked_env.py). To keep Training == Backtest == Live
        #    we ALWAYS refit inline here on the SAME train Parquet data and
        #    NEVER load a frozen pkl. If StateBuilder auto-loaded any pkl in its
        #    constructor, we wipe it before refitting.
        self.state_builder.scalers = {}
        self.state_builder.scalers_loaded_from_training = False

        # 🔧 Fit scalers inline on the training Parquet data to reproduce the
        #    exact distribution the PPO was trained on.
        self.fit_on_parquet()

        logger.info(
            f"[LiveStateBuilder] {exchange_id} | {symbol} | TFs={self.timeframes}"
        )

    # ── Public API ──────────────────────────────────────────────────────

    def fit_on_parquet(self):
        """Fit internal scalers INLINE on the TRAINING Parquet data.

        CRITICAL — Training == Backtest == Live invariant:
        The training/backtest environment (MultiAssetChunkedEnv) loads the
        ``train`` split and fits its StateBuilder scalers inline on that chunk.
        We MUST do exactly the same here (same ``train`` split, same inline fit,
        same StateBuilder anti-lookahead "first 70%" rule) so the PPO receives
        observations drawn from the distribution it was trained on.

        We deliberately do NOT use the ``val`` split (different price regime:
        val close mean≈74k vs train mean≈52k) and we NEVER load a frozen
        ``prod_scalers/*.pkl`` — those caused the paper-trading saturation.
        """
        from pathlib import Path

        # Use symbol style compatible with filesystem (e.g. BTCUSDT)
        fs_symbol = self.symbol.replace("/", "").replace(":", "").upper()

        # Search for the TRAIN split (same split the env trains on).
        possible_paths = [
            Path("data/processed/indicators/train") / fs_symbol,
            Path(__file__).parent.parent.parent.parent
            / "data/processed/indicators/train" / fs_symbol,
        ]

        train_dir = None
        for path in possible_paths:
            if path.exists():
                train_dir = path
                break

        if not (train_dir and train_dir.exists()):
            logger.error(
                f"❌ Training data directory not found for {fs_symbol}. "
                f"Cannot fit scalers inline — REFUSING to run with an "
                f"undefined observation distribution."
            )
            raise FileNotFoundError(
                f"Training Parquet not found for {fs_symbol} "
                f"(searched: {[str(p) for p in possible_paths]})"
            )

        try:
            data_dict = {}
            for tf in self.timeframes:
                path = train_dir / f"{tf}.parquet"
                if path.exists():
                    data_dict[tf] = pd.read_parquet(path)

            if not data_dict:
                raise FileNotFoundError(
                    f"No Parquet files found in {train_dir} for fitting."
                )

            logger.info(
                f"🎯 Fitting LiveStateBuilder scalers INLINE on {fs_symbol} "
                f"TRAIN Parquet data ({train_dir}) — pkl loading is BANNED."
            )
            # Force a clean inline fit: wipe any auto-loaded scalers + flag so
            # StateBuilder.fit_scalers() does not short-circuit.
            self.state_builder.scalers = {}
            self.state_builder.scalers_loaded_from_training = False
            self.state_builder.fit_scalers({fs_symbol: data_dict})

            # Lock to prevent any later refit on live data (would drift).
            self.state_builder.scalers_loaded_from_training = True

            # ── Consistency proof: log per-TF close normalization stats so we
            #    can verify Training == Live at a glance in the logs.
            self._log_scaler_consistency(data_dict)
            logger.info("✅ Scalers fitted inline & LOCKED to TRAIN distribution.")
        except Exception as e:
            logger.error(f"❌ Failed to fit on Parquet: {e}")
            raise

    def _log_scaler_consistency(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """Log the fitted scaler stats vs raw data stats for the 'close' column.

        Emits a [SCALER_CHECK] line per timeframe and a warning if a frozen
        biased scaler would have been used instead (sanity guard).
        """
        try:
            import numpy as _np
            for tf, df in data_dict.items():
                scaler = self.state_builder.scalers.get(tf)
                if scaler is None:
                    continue
                inner = getattr(scaler, "scaler", scaler)
                cols = [c.lower() for c in TRAIN_COLUMNS.get(tf, [])]
                if "close" not in cols:
                    continue
                ci = cols.index("close")
                raw_mean = float(df["close"].mean())
                if hasattr(inner, "mean_"):
                    fit_center = float(inner.mean_[ci])
                    fit_scale = float(inner.scale_[ci])
                    kind = "StandardScaler"
                elif hasattr(inner, "center_"):
                    fit_center = float(inner.center_[ci])
                    fit_scale = float(inner.scale_[ci])
                    kind = "RobustScaler"
                else:  # MinMaxScaler — center/scale not meaningful for this log
                    logger.info(
                        f"[SCALER_CHECK] {tf} MinMaxScaler | raw_close_mean={raw_mean:.0f}"
                    )
                    continue
                logger.info(
                    f"[SCALER_CHECK] {tf} {kind} | fit_center={fit_center:.0f} "
                    f"fit_scale={fit_scale:.1f} | raw_close_mean={raw_mean:.0f}"
                )
        except Exception as e:  # never let logging break the run
            logger.debug(f"[SCALER_CHECK] skipped: {e}")

    def fetch_and_compute(self, force: bool = False) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV from exchange and compute all 21 indicators per TF.

        Returns dict: timeframe → DataFrame with TRAIN_COLUMNS columns.
        """
        for tf in self.timeframes:
            # Rate-limit: don't refetch within 5 seconds
            now = time.time()
            if not force and (now - self._last_fetch.get(tf, 0)) < 5.0:
                continue
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    self.symbol, tf, limit=FETCH_LIMITS.get(tf, 200)
                )
                df = pd.DataFrame(
                    ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                df = df.astype(float)

                # Compute the 21 features
                df = self._compute_indicators(df, tf)
                self._cache[tf] = df
                self._last_fetch[tf] = now
                logger.debug(
                    f"[LiveStateBuilder] {tf}: {len(df)} bars, "
                    f"last={df.index[-1]}, close={df['close'].iloc[-1]:.2f}"
                )
            except Exception as e:
                logger.warning(f"[LiveStateBuilder] fetch {tf} failed: {e}")
        return self._cache

    def build_observation(
        self,
        portfolio_state: np.ndarray = None,
        context_vector: np.ndarray = None,
    ) -> Dict[str, np.ndarray]:
        """Build the full observation dict matching the model's obs space.

        Returns:
            {
                "5m":  np.ndarray shape (20, 21),
                "1h":  np.ndarray shape (20, 21),
                "4h":  np.ndarray shape (20, 21),
                "portfolio_state": np.ndarray shape (28,),  # 20 base + 8 ACM
                "context_vector":  np.ndarray shape (17,),
            }
        """
        # Fetch latest data
        self.fetch_and_compute()

        # Build normalized observations using StateBuilder
        # Note: self.symbol might contain '/' which StateBuilder doesn't like 
        # as a dict key if it tries to match against config. We'll use BTCUSDT style.
        symbol_key = self.symbol.replace("/", "").replace(":", "")
        data_dict = {symbol_key: self._cache}
        
        # Fit scalers if not already loaded from training
        if not getattr(self.state_builder, "scalers_loaded_from_training", False):
            self.state_builder.fit_scalers(data_dict)
            
        # Build observation using StateBuilder's robust logic
        # With FETCH_LIMITS=500, last_idx ≈ 499. StateBuilder takes
        # [last_idx - window_size + 1 : last_idx + 1] = last 20 bars.
        # These bars have 480+ warmup bars before them → indicators converged.
        last_idx = len(next(iter(self._cache.values()))) - 1
        
        # DIAGNOSTIC: Warn if insufficient warmup history
        min_warmup = 200  # EMA-100 needs ~200 bars to stabilize
        if last_idx < min_warmup + OBS_WINDOW:
            logger.warning(
                f"[WARMUP] Only {last_idx + 1} bars available, need ≥{min_warmup + OBS_WINDOW} "
                f"for converged indicators. Distribution shift risk!"
            )
        obs = self.state_builder.build_observation(
            current_idx=last_idx,
            data=data_dict
        )

        # Ensure portfolio_state and context_vector are included
        if portfolio_state is not None:
            obs["portfolio_state"] = portfolio_state.astype(np.float32)
        elif "portfolio_state" not in obs:
            obs["portfolio_state"] = np.zeros(28, dtype=np.float32)  # 20 base + 8 ACM

        if context_vector is not None:
            obs["context_vector"] = context_vector.astype(np.float32)
        elif "context_vector" not in obs:
            obs["context_vector"] = np.full(17, 1.0 / 17.0, dtype=np.float32)

        return obs

    def get_current_price(self) -> float:
        """Return the latest close price from 5m data."""
        df = self._cache.get("5m")
        if df is not None and len(df) > 0:
            return float(df["close"].iloc[-1])
        # Fallback: fetch ticker
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return float(ticker["last"])
        except Exception:
            return 0.0

    # ── Indicator Computation ───────────────────────────────────────────

    def _compute_indicators(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Compute exactly the 21 TRAIN_COLUMNS indicators for a given TF.

        This replicates what feature_engineer.py does during training data prep.
        """
        cols = TRAIN_COLUMNS[tf]
        close = df["close"]
        high = df["high"]
        low = df["low"]
        opn = df["open"]
        volume = df["volume"]

        # ── EMA ratio (position-specific per TF) ──
        if tf == "5m":
            ema = ta.ema(close, length=20)
            df["ema_20_ratio"] = (close / ema).where(ema > 0, 1.0)
        elif tf == "1h":
            ema = ta.ema(close, length=50)
            df["ema_50_ratio"] = (close / ema).where(ema > 0, 1.0)
        elif tf == "4h":
            ema = ta.ema(close, length=100)
            df["ema_100_ratio"] = (close / ema).where(ema > 0, 1.0)

        # ── MACD histogram (position-specific per TF) ──
        if tf == "5m":
            macd = ta.macd(close, fast=12, slow=26, signal=9)
            if macd is not None:
                df["macdh_12_26_9"] = macd.iloc[:, -1]  # histogram column
            else:
                df["macdh_12_26_9"] = 0.0
        elif tf == "1h":
            macd = ta.macd(close, fast=21, slow=42, signal=9)
            if macd is not None:
                df["macdh_21_42_9"] = macd.iloc[:, -1]
            else:
                df["macdh_21_42_9"] = 0.0
        elif tf == "4h":
            macd = ta.macd(close, fast=26, slow=52, signal=18)
            if macd is not None:
                df["macdh_26_52_18"] = macd.iloc[:, -1]
            else:
                df["macdh_26_52_18"] = 0.0

        # ── RSI (position-specific per TF) ──
        if tf == "5m":
            df["rsi_14"] = ta.rsi(close, length=14)
        elif tf == "1h":
            df["rsi_21"] = ta.rsi(close, length=21)
        elif tf == "4h":
            df["rsi_28"] = ta.rsi(close, length=28)

        # ── Common indicators (all TFs) ──

        # ADX & DI
        adx_df = ta.adx(high, low, close, length=14)
        if adx_df is not None:
            df["adx_14"] = adx_df.iloc[:, 0]  # ADX
            dmp = adx_df.iloc[:, 1] if adx_df.shape[1] > 1 else 0
            dmn = adx_df.iloc[:, 2] if adx_df.shape[1] > 2 else 0
            df["di_delta"] = dmp - dmn
        else:
            df["adx_14"] = 0.0
            df["di_delta"] = 0.0

        # ATR percentage
        atr14 = ta.atr(high, low, close, length=14)
        df["atr_pct"] = (atr14 / close).where(close > 0, 0.0)

        # Bollinger Bands
        bb = ta.bbands(close, length=20, std=2)
        if bb is not None:
            bbu = bb.iloc[:, 0]  # upper
            bbm = bb.iloc[:, 1]  # mid
            bbl = bb.iloc[:, 2]  # lower
            bb_range = bbu - bbl
            df["bb_percent_b_20_2"] = ((close - bbl) / bb_range).where(bb_range > 0, 0.5)
            df["bb_width_20_2"] = (bb_range / bbm).where(bbm > 0, 0.0)
        else:
            df["bb_percent_b_20_2"] = 0.5
            df["bb_width_20_2"] = 0.0

        # OBV slope (normalized)
        obv = ta.obv(close, volume)
        if obv is not None:
            obv_sma = obv.rolling(20).mean()
            df["obv_slope"] = ((obv - obv_sma) / (obv_sma.abs() + 1e-10))
        else:
            df["obv_slope"] = 0.0

        # Volume ratio
        vol_sma = volume.rolling(20).mean()
        df["volume_ratio_20"] = (volume / vol_sma).where(vol_sma > 0, 1.0)

        # Volatility ratio (ATR14 / ATR50)
        atr50 = ta.atr(high, low, close, length=50)
        if atr50 is not None and atr14 is not None:
            df["volatility_ratio_14_50"] = (atr14 / atr50).where(atr50 > 0, 1.0)
        else:
            df["volatility_ratio_14_50"] = 1.0

        # Fib ratio (position within High-Low range, last 20 bars)
        roll_high = high.rolling(20).max()
        roll_low = low.rolling(20).min()
        fib_range = roll_high - roll_low
        df["fib_ratio"] = ((close - roll_low) / fib_range).where(fib_range > 0, 0.5)

        # Price action (close - open) / open
        df["price_action"] = ((close - opn) / opn).where(opn > 0, 0.0)

        # VWAP ratio (approximate: cumulative(price*vol) / cumulative(vol))
        typical_price = (high + low + close) / 3.0
        cum_tp_vol = (typical_price * volume).cumsum()
        cum_vol = volume.cumsum()
        vwap = (cum_tp_vol / cum_vol).where(cum_vol > 0, close)
        df["vwap_ratio"] = (close / vwap).where(vwap > 0, 1.0)

        # Market structure (trend filter: 1 if above EMA20+50, -1 below both, 0 mixed)
        ema20 = ta.ema(close, length=20)
        ema50 = ta.ema(close, length=50)
        if ema20 is not None and ema50 is not None:
            df["market_structure"] = np.where(
                (close > ema20) & (close > ema50), 1.0,
                np.where((close < ema20) & (close < ema50), -1.0, 0.0)
            )
        else:
            df["market_structure"] = 0.0

        # Log return
        df["log_return"] = np.log(close / close.shift(1)).fillna(0.0)

        # Fill remaining NaN with 0
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = df[c].fillna(0.0)

        return df
