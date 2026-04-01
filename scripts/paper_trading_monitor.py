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
    """
    if hasattr(action_raw, "__len__"):
        signal_raw = float(action_raw[0])
        size_raw = float(action_raw[1]) if len(action_raw) > 1 else 0.0
    else:
        signal_raw = float(action_raw)
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
    """Real-time paper trading monitor for ADAN on Binance Testnet."""

    def __init__(self, config_path="config/config.yaml", api_key=None, api_secret=None):
        self.config = ConfigLoader.load_config(config_path)
        self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BINANCE_SECRET_KEY", "")
        self.testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
        self.symbol = os.getenv("TRADING_PAIR", "BTC/USDT")
        self.initial_balance = float(os.getenv("INITIAL_BALANCE", "25"))

        self.virtual_balance = self.initial_balance
        self.active_position = None  # {side, entry_price, tp, sl, timestamp}
        self.trades = []
        self.timeframes = ["5m", "1h", "4h"]
        self.analysis_interval = 300  # 5 minutes (match training)
        self.tp_sl_check_interval = 30  # seconds

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

        logger.info(f"Paper Trading Monitor initialized")
        logger.info(f"  Symbol: {self.symbol}")
        logger.info(f"  Balance: ${self.initial_balance}")
        logger.info(f"  Testnet: {self.testnet}")

    def setup_exchange(self):
        """Initialize ccxt exchange connection."""
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
            logger.error(f"Exchange setup failed: {e}")
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
        """One analysis cycle: fetch data → build obs → predict → execute."""
        data = self.fetch_live_data()
        if not data:
            return

        price = self.get_current_price()
        if price <= 0:
            return

        # Use the model to predict
        # For paper trading, we create a lightweight env observation
        # by wrapping the data through the env's observation builder
        try:
            # Build observation through the real env pipeline
            asset = self.symbol.replace("/", "")
            wc = copy.deepcopy(self.config.get("workers", {}).get("w1", {}))
            wc["worker_id"] = 0

            env = MultiAssetChunkedEnv(
                data=data, config=self.config, worker_config=wc,
                worker_id=0, live_mode=False,
            )
            obs, _ = env.reset()

            # Context vector verification
            cv = obs.get("context_vector")
            if cv is not None:
                logger.info(
                    f"Context vector (12D): "
                    f"vol={cv[0]:.3f} trend={cv[1]:.3f} adx={cv[2]:.3f} "
                    f"regime={cv[3]:.3f} dd={cv[4]:.3f} candle={cv[5]:.3f} "
                    f"sinH={cv[6]:.3f} cosH={cv[7]:.3f}"
                )

            # Predict
            action, _ = self.model.predict(obs, deterministic=True)
            action_info = interpret_target_weight_action(
                action, has_position=self.active_position is not None
            )

            logger.info(
                f"PREDICTION: {action_info['signal']} "
                f"(confidence={action_info['confidence']:.3f}, "
                f"raw[0]={float(action[0]):.4f})"
            )

            # Execute
            if action_info["signal"] != "HOLD":
                self.execute_signal(action_info, price)

            env.close()

        except Exception as e:
            logger.error(f"Analysis cycle failed: {e}", exc_info=True)

    def run(self, duration_minutes: int = 360):
        """Main event loop."""
        logger.info("=" * 70)
        logger.info("ADAN Paper Trading Monitor v2 (FiLM + context_vector)")
        logger.info(f"  Duration: {duration_minutes} min")
        logger.info(f"  Analysis interval: {self.analysis_interval}s")
        logger.info(f"  Balance: ${self.virtual_balance:.2f}")
        logger.info("=" * 70)

        if not self.setup_exchange():
            return
        if not self.load_model():
            return

        end_time = time.time() + duration_minutes * 60

        while time.time() < end_time:
            try:
                now = time.time()

                # TP/SL check (every 30s)
                if now - self.last_tp_sl_check > self.tp_sl_check_interval:
                    self.check_tp_sl()
                    self.last_tp_sl_check = now

                # Analysis cycle (every 5 min)
                if now - self.last_analysis_time > self.analysis_interval:
                    self.run_analysis_cycle()
                    self.last_analysis_time = now

                time.sleep(10)

            except KeyboardInterrupt:
                logger.info("Stopping paper trading...")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                time.sleep(30)

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
    args = parser.parse_args()

    monitor = PaperTradingMonitor(
        config_path=args.config,
        api_key=args.api_key,
        api_secret=args.api_secret,
    )
    if args.model:
        monitor.load_model(args.model)
    monitor.run(duration_minutes=args.duration)


if __name__ == "__main__":
    main()
