#!/usr/bin/env python3
"""
ADAN Paper Trading Monitor — Production Version (v2)
======================================================
Real-time execution for ADAN paper trading on Binance Testnet.

Architecture alignment (v2):
  - Uses ``StateBuilder.build_observation()`` for the 12-dim context_vector
    (6 market: Volatility, Trend, ADX, Regime, Drawdown, Candle_Progress
     + 6 Time2Vec: sin/cos of hour, weekday, day-of-month)
  - ``ContextualTemporalFusionExtractor`` with FiLM Meta-RL modulation
  - Action is a continuous Box(25,) "Target Weight" vector:
      * action[0] < -0.1 while long → DYNAMIC EXIT (close at market)
      * action[0] > +0.33 → BUY signal
      * action[0] < -0.33 → SELL signal
  - Capital Tier supremacy (Micro/Small/Medium/High/Enterprise)

Data pipeline:
  1. Fetch 5m OHLCV from Binance → resample to 1h, 4h
  2. Master Clock alignment (ffill higher TFs onto 5m index)
  3. Build observation via MultiAssetChunkedEnv's internal StateBuilder
  4. PPO model.predict() → interpret Target-Weight action

Usage:
    python scripts/paper_trading_monitor.py --api-key <KEY> --api-secret <SECRET>
    python scripts/paper_trading_monitor.py  # reads from .env
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

try:
    from adan_trading_bot.agent.feature_extractors import (
        ContextualTemporalFusionExtractor,
        WorldModelPPO,
    )
except ImportError:
    ContextualTemporalFusionExtractor = None
    WorldModelPPO = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(str(PROJECT_ROOT / "paper_trading.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("paper_trading_monitor")


# ── Capital Tier resolution ────────────────────────────────────────────────
def get_capital_tier(balance: float, tiers: list) -> dict:
    """Return the matching capital tier dict for the given balance."""
    for tier in tiers:
        min_cap = tier.get("min_capital", 0)
        max_cap = tier.get("max_capital") or float("inf")
        if min_cap <= balance < max_cap:
            return tier
    return {"name": "Micro Capital", "exposure_range": [70, 90], "risk_per_trade_pct": 4.0}


# ── Action interpreter ─────────────────────────────────────────────────────
def interpret_target_weight_action(action_raw, has_position: bool) -> dict:
    """Interpret the continuous Target-Weight action vector.

    Returns dict with keys: signal, size_raw, confidence.

    ``action_raw`` may arrive as:
      - shape (25,) -- single-env predict
      - shape (1, 25) -- batched predict (VecEnv)
    We flatten to 1-D before interpretation.
    """
    import numpy as _np
    arr = _np.asarray(action_raw).flatten()
    if arr.size > 0:
        signal_raw = float(arr[0])
        size_raw = float(arr[1]) if arr.size > 1 else 0.0
    else:
        signal_raw = 0.0
        size_raw = 0.0

    # Dynamic exit: agent signals negative while already long
    if has_position and signal_raw < -0.1:
        return {"signal": "DYNAMIC_EXIT", "size_raw": size_raw, "confidence": abs(signal_raw)}

    if signal_raw > 0.33:
        return {"signal": "BUY", "size_raw": size_raw, "confidence": min(signal_raw, 1.0)}
    elif signal_raw < -0.33:
        return {"signal": "SELL", "size_raw": size_raw, "confidence": min(abs(signal_raw), 1.0)}

    return {"signal": "HOLD", "size_raw": size_raw, "confidence": 1.0 - abs(signal_raw) * 2}


class PaperTradingMonitor:
    """Real-time paper trading monitor for ADAN on Binance Testnet.

    Supports two modes:
      - Live:    connects to Binance Testnet via ccxt, fetches real OHLCV.
      - Offline:  uses locally generated parquet data (from generate_colab_dataset.py)
                  and replays it step-by-step through the environment.

    The offline mode lets you validate the full inference pipeline
    (VecNormalize, StateBuilder, model.predict) without network access.
    """

    def __init__(self, config_path="config/config.yaml", api_key=None, api_secret=None,
                 offline=False):
        self.config = ConfigLoader.load_config(config_path)
        self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BINANCE_SECRET_KEY", "")
        self.testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
        self.symbol = os.getenv("TRADING_PAIR", "BTC/USDT")
        self.initial_balance = float(os.getenv("INITIAL_BALANCE", "25"))
        self.offline = offline

        self.virtual_balance = self.initial_balance
        self.active_position = None  # {side, entry_price, tp, sl, timestamp}
        self.trades = []
        self.timeframes = ["5m", "1h", "4h"]
        self.analysis_interval = 5  # 5s in offline mode, 300s live
        self.tp_sl_check_interval = 2  # seconds (2s offline, 30s live)

        if not self.offline:
            self.analysis_interval = 300
            self.tp_sl_check_interval = 30

        # Capital tiers from config
        self.capital_tiers = self.config.get("capital_tiers", [])

        # Model
        self.model = None
        self.vec_env = None

        # State
        self.exchange = None
        self.last_analysis_time = 0
        self.last_tp_sl_check = 0
        self.latest_data = None

        # Offline replay state
        self._offline_vec_env = None
        self._offline_obs = None
        self._offline_step = 0

        mode_str = "OFFLINE (local data)" if self.offline else "LIVE (Binance Testnet)"
        logger.info(f"Paper Trading Monitor initialized -- {mode_str}")
        logger.info(f"  Symbol: {self.symbol}")
        logger.info(f"  Balance: ${self.initial_balance}")
        logger.info(f"  Testnet: {self.testnet}")

    def setup_exchange(self):
        """Initialize ccxt exchange connection.

        In offline mode, skip connection entirely and load local data instead.
        If the exchange is unreachable, automatically fall back to offline mode.
        """
        if self.offline:
            logger.info("Offline mode: skipping exchange connection")
            return self._setup_offline_env()

        try:
            import ccxt
            self.exchange = ccxt.binance({
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
            if self.testnet:
                self.exchange.set_sandbox_mode(True)

            # Test connection
            self.exchange.fetch_time()
            logger.info("Exchange connected (Binance Testnet)")
            return True
        except Exception as e:
            logger.warning(f"Exchange setup failed: {e}")
            logger.info("Falling back to offline mode with local data")
            self.offline = True
            return self._setup_offline_env()

    def _setup_offline_env(self):
        """Build vectorised env from local parquet data for offline replay."""
        try:
            wc = copy.deepcopy(self.config.get("workers", {}).get("w1", {}))
            wc["worker_id"] = 0

            loader = ChunkedDataLoader(self.config, worker_config=wc, worker_id=0)
            data = loader.load_chunk(0)

            if not data:
                logger.error("No local data found. Run generate_colab_dataset.py first.")
                return False

            raw_env = MultiAssetChunkedEnv(
                data=data, config=self.config, worker_config=wc,
                worker_id=0, live_mode=False,
            )
            dummy = DummyVecEnv([lambda: raw_env])

            vecnorm_path = PROJECT_ROOT / "models" / "rl_agents" / "vecnormalize.pkl"
            if vecnorm_path.exists():
                self._offline_vec_env = VecNormalize.load(str(vecnorm_path), dummy)
                self._offline_vec_env.training = False
                self._offline_vec_env.norm_reward = False
                logger.info(f"VecNormalize loaded from {vecnorm_path} (training=False)")
            else:
                gamma = self.config.get("agent", {}).get("gamma", 0.99)
                self._offline_vec_env = VecNormalize(
                    dummy, norm_obs=True, norm_reward=False,
                    clip_obs=10.0, gamma=gamma, training=False,
                )
                logger.warning("VecNormalize stats not found -- identity normalisation")

            self._offline_obs = self._offline_vec_env.reset()
            self._offline_step = 0

            # Cache data for get_current_price
            self.latest_data = data

            asset = self.symbol.replace("/", "")
            rows = 0
            for tf, df in data.get(asset, {}).items():
                rows = max(rows, len(df))
            logger.info(f"Offline env ready: {rows} rows for {asset}, 3 timeframes")
            return True

        except Exception as e:
            logger.error(f"Offline env setup failed: {e}", exc_info=True)
            return False

    def load_model(self, model_path=None):
        """Load the PPO model for inference."""
        if model_path is None:
            # Try default paths
            candidates = [
                PROJECT_ROOT / "models" / "rl_agents" / "ppo_adan_simple.zip",
                PROJECT_ROOT / "models" / "w1" / "w1_model_final.zip",
                PROJECT_ROOT / "models" / "w1" / "model.zip",
            ]
            for p in candidates:
                if p.exists():
                    model_path = str(p)
                    break

        if model_path is None or not Path(model_path).exists():
            logger.error("No model found. Train first or specify --model.")
            return False

        PPOClass = WorldModelPPO if WorldModelPPO is not None else PPO
        self.model = PPOClass.load(model_path, device="cpu")
        logger.info(f"Model loaded: {model_path} ({type(self.model).__name__})")
        return True

    def fetch_live_data(self) -> dict:
        """Fetch 5m OHLCV and resample to 1h, 4h with Master Clock alignment.

        Returns dict: {symbol: {tf: DataFrame}} matching ChunkedDataLoader format.
        """
        if not self.exchange:
            return None

        try:
            # Fetch 2000 5m candles (multi-pass)
            all_candles = []
            since = None
            for _ in range(2):  # 2 x 1000
                batch = self.exchange.fetch_ohlcv(
                    self.symbol, "5m", since=since, limit=1000
                )
                if not batch:
                    break
                all_candles.extend(batch)
                since = batch[-1][0] + 1
                time.sleep(0.5)

            if len(all_candles) < 100:
                logger.error(f"Insufficient data: {len(all_candles)} < 100")
                return None

            df_5m = pd.DataFrame(
                all_candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"], unit="ms", utc=True)
            df_5m = df_5m.set_index("timestamp").sort_index()
            df_5m = df_5m[~df_5m.index.duplicated(keep="first")]

            # Resample to higher timeframes
            agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            df_1h = df_5m.resample("1h").agg(agg).dropna()
            df_4h = df_5m.resample("4h").agg(agg).dropna()

            # Master Clock: reindex 1h/4h onto 5m
            df_1h = df_1h.reindex(df_5m.index, method="ffill").dropna(subset=["close"])
            df_4h = df_4h.reindex(df_5m.index, method="ffill").dropna(subset=["close"])

            # Common index
            common = df_5m.index.intersection(df_1h.index).intersection(df_4h.index)
            asset_name = self.symbol.replace("/", "")

            data = {
                asset_name: {
                    "5m": df_5m.loc[common],
                    "1h": df_1h.loc[common],
                    "4h": df_4h.loc[common],
                }
            }

            logger.info(
                f"Live data: {len(common)} aligned rows, "
                f"5m price={df_5m['close'].iloc[-1]:.2f}"
            )
            self.latest_data = data
            return data

        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            return None

    def get_current_price(self) -> float:
        """Get latest price from cached data."""
        if self.latest_data:
            asset = self.symbol.replace("/", "")
            for tf in ["5m", "1h", "4h"]:
                df = self.latest_data.get(asset, {}).get(tf)
                if df is not None and not df.empty:
                    return float(df["close"].iloc[-1])
        return 0.0

    def check_tp_sl(self):
        """Check if TP or SL has been hit."""
        if not self.active_position:
            return

        price = self.get_current_price()
        if price <= 0:
            return

        pos = self.active_position
        hit = None

        if pos["side"] == "BUY":
            if price >= pos["tp"]:
                hit = "TP"
            elif price <= pos["sl"]:
                hit = "SL"
        else:
            if price <= pos["tp"]:
                hit = "TP"
            elif price >= pos["sl"]:
                hit = "SL"

        if hit:
            self._close_position(price, reason=hit)

    def _close_position(self, exit_price: float, reason: str = "Manual"):
        """Close the active position and record the trade."""
        if not self.active_position:
            return

        pos = self.active_position
        entry = pos["entry_price"]
        pnl_pct = ((exit_price - entry) / entry * 100) if pos["side"] == "BUY" else ((entry - exit_price) / entry * 100)
        pnl_abs = pnl_pct / 100 * self.virtual_balance * 0.5  # ~50% exposure

        self.virtual_balance += pnl_abs

        trade = {
            "side": pos["side"],
            "entry": entry,
            "exit": exit_price,
            "pnl_pct": pnl_pct,
            "pnl_abs": pnl_abs,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "balance_after": self.virtual_balance,
        }
        self.trades.append(trade)
        self.active_position = None

        logger.info(
            f"CLOSED {pos['side']} ({reason}): "
            f"PnL={pnl_pct:+.2f}% (${pnl_abs:+.2f}), "
            f"Balance=${self.virtual_balance:.2f}"
        )

    def execute_signal(self, action_info: dict, price: float):
        """Execute a trading signal based on the interpreted action."""
        signal = action_info["signal"]
        tier = get_capital_tier(self.virtual_balance, self.capital_tiers)
        tier_name = tier.get("name", "Micro Capital")

        if signal == "DYNAMIC_EXIT":
            if self.active_position:
                self._close_position(price, reason="DYNAMIC_EXIT")
            return

        if signal == "BUY" and not self.active_position:
            # Apply tier risk limits
            max_risk_pct = tier.get("risk_per_trade_pct", 4.0) / 100
            exposure = tier.get("exposure_range", [70, 90])
            tp_pct = 0.03  # 3% default
            sl_pct = min(0.02, max_risk_pct)  # capped by tier

            self.active_position = {
                "side": "BUY",
                "entry_price": price,
                "tp": price * (1 + tp_pct),
                "sl": price * (1 - sl_pct),
                "timestamp": datetime.now().isoformat(),
                "tier": tier_name,
            }
            logger.info(
                f"OPENED BUY @ {price:.2f} | "
                f"TP={self.active_position['tp']:.2f} SL={self.active_position['sl']:.2f} | "
                f"Tier={tier_name}"
            )

        elif signal == "SELL" and self.active_position:
            self._close_position(price, reason="SELL_SIGNAL")

    def run_analysis_cycle(self):
        """One analysis cycle: fetch data -> build obs -> predict -> execute.

        Inference pipeline:
          1. Wrap data in MultiAssetChunkedEnv -> DummyVecEnv.
          2. Load VecNormalize stats from training (training=False, norm_reward=False).
          3. StateBuilder.build_observation() produces the 12-dim context_vector.
          4. model.predict(obs, deterministic=True) -> interpret target-weight action.

        In offline mode the environment is already built; we simply step through it.
        """
        if self.offline:
            return self._run_offline_cycle()

        data = self.fetch_live_data()
        if not data:
            return

        price = self.get_current_price()
        if price <= 0:
            return

        try:
            # Build observation through the real env pipeline
            asset = self.symbol.replace("/", "")
            wc = copy.deepcopy(self.config.get("workers", {}).get("w1", {}))
            wc["worker_id"] = 0

            raw_env = MultiAssetChunkedEnv(
                data=data, config=self.config, worker_config=wc,
                worker_id=0, live_mode=False,
            )
            dummy_env = DummyVecEnv([lambda: raw_env])

            # -- VecNormalize: load training stats for inference
            vecnorm_path = PROJECT_ROOT / "models" / "rl_agents" / "vecnormalize.pkl"
            if vecnorm_path.exists():
                vec_env = VecNormalize.load(str(vecnorm_path), dummy_env)
                vec_env.training = False
                vec_env.norm_reward = False
                logger.info(f"VecNormalize loaded from {vecnorm_path} (inference mode)")
            else:
                gamma = self.config.get("agent", {}).get("gamma", 0.99)
                vec_env = VecNormalize(
                    dummy_env, norm_obs=True, norm_reward=False,
                    clip_obs=10.0, gamma=gamma, training=False,
                )
                logger.warning("VecNormalize stats not found -- identity normalisation")

            obs = vec_env.reset()
            self._log_context_vector(obs)

            action, _ = self.model.predict(obs, deterministic=True)
            action_info = interpret_target_weight_action(
                action, has_position=self.active_position is not None
            )
            self._log_prediction(action, action_info)

            if action_info["signal"] != "HOLD":
                self.execute_signal(action_info, price)

            vec_env.close()

        except Exception as e:
            logger.error(f"Analysis cycle failed: {e}", exc_info=True)

    # ---- Offline replay helpers ------------------------------------------

    def _run_offline_cycle(self):
        """Step through the pre-built offline env and predict."""
        if self._offline_vec_env is None or self._offline_obs is None:
            logger.error("Offline env not initialised")
            return

        try:
            obs = self._offline_obs
            self._log_context_vector(obs)

            action, _ = self.model.predict(obs, deterministic=True)
            action_info = interpret_target_weight_action(
                action, has_position=self.active_position is not None
            )

            # Derive simulated price from context or latest data
            price = self.get_current_price()
            if price <= 0:
                price = 65000.0  # fallback

            self._log_prediction(action, action_info)

            if action_info["signal"] != "HOLD":
                self.execute_signal(action_info, price)

            # Step the environment forward
            obs_next, reward, done, info = self._offline_vec_env.step(action)
            self._offline_step += 1

            if done[0]:
                logger.info(f"Offline episode ended at step {self._offline_step}, resetting")
                obs_next = self._offline_vec_env.reset()
                self._offline_step = 0

            self._offline_obs = obs_next

            logger.info(
                f"[offline step {self._offline_step}] "
                f"reward={reward[0]:.4f}, balance=${self.virtual_balance:.2f}"
            )

        except Exception as e:
            logger.error(f"Offline analysis cycle failed: {e}", exc_info=True)

    # ---- Logging helpers -------------------------------------------------

    def _log_context_vector(self, obs):
        """Log the 12-D context vector from the observation."""
        if isinstance(obs, dict):
            cv = obs.get("context_vector")
        else:
            cv = None
        if cv is not None:
            cv_flat = cv[0] if cv.ndim > 1 else cv
            logger.info(
                f"Context vector (12D): "
                f"vol={cv_flat[0]:.3f} trend={cv_flat[1]:.3f} adx={cv_flat[2]:.3f} "
                f"regime={cv_flat[3]:.3f} dd={cv_flat[4]:.3f} candle={cv_flat[5]:.3f} "
                f"sinH={cv_flat[6]:.3f} cosH={cv_flat[7]:.3f}"
            )

    def _log_prediction(self, action, action_info):
        """Log model prediction details."""
        raw0 = float(action[0][0]) if action.ndim > 1 else float(action[0])
        logger.info(
            f"PREDICTION: {action_info['signal']} "
            f"(confidence={action_info['confidence']:.3f}, "
            f"raw[0]={raw0:.4f})"
        )

    def run(self, duration_minutes: int = 360):
        """Main event loop.

        In offline mode the loop sleep is shortened so the full duration
        runs through many environment steps in wall-clock time.
        """
        logger.info("=" * 70)
        logger.info("ADAN Paper Trading Monitor v2 (FiLM + context_vector)")
        mode_str = "OFFLINE" if self.offline else "LIVE"
        logger.info(f"  Mode: {mode_str}")
        logger.info(f"  Duration: {duration_minutes} min")
        logger.info(f"  Analysis interval: {self.analysis_interval}s")
        logger.info(f"  Balance: ${self.virtual_balance:.2f}")
        logger.info("=" * 70)

        if not self.setup_exchange():
            return
        if not self.load_model():
            return

        end_time = time.time() + duration_minutes * 60
        loop_sleep = 1 if self.offline else 10

        while time.time() < end_time:
            try:
                now = time.time()

                # TP/SL check
                if now - self.last_tp_sl_check > self.tp_sl_check_interval:
                    self.check_tp_sl()
                    self.last_tp_sl_check = now

                # Analysis cycle
                if now - self.last_analysis_time > self.analysis_interval:
                    self.run_analysis_cycle()
                    self.last_analysis_time = now

                time.sleep(loop_sleep)

            except KeyboardInterrupt:
                logger.info("Stopping paper trading...")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                time.sleep(5)

        # Final report
        logger.info("=" * 70)
        logger.info("PAPER TRADING SESSION COMPLETE")
        logger.info(f"  Trades: {len(self.trades)}")
        logger.info(f"  Final balance: ${self.virtual_balance:.2f}")
        logger.info(f"  Return: {((self.virtual_balance - self.initial_balance) / self.initial_balance * 100):+.2f}%")
        logger.info("=" * 70)

        # Save results
        results = {
            "initial_balance": self.initial_balance,
            "final_balance": self.virtual_balance,
            "return_pct": (self.virtual_balance - self.initial_balance) / self.initial_balance * 100,
            "trades": self.trades,
            "timestamp": datetime.now().isoformat(),
        }
        results_path = PROJECT_ROOT / "results" / "paper_trading_report.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Report saved: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="ADAN Paper Trading Monitor v2")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", default=None, help="Model .zip path")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-secret", default=None)
    parser.add_argument("--duration", type=int, default=360, help="Duration in minutes")
    parser.add_argument(
        "--offline", action="store_true",
        help="Run in offline mode using local parquet data (no exchange needed)",
    )
    args = parser.parse_args()

    monitor = PaperTradingMonitor(
        config_path=args.config,
        api_key=args.api_key,
        api_secret=args.api_secret,
        offline=args.offline,
    )
    if args.model:
        monitor.load_model(args.model)
    monitor.run(duration_minutes=args.duration)


if __name__ == "__main__":
    main()
