#!/usr/bin/env python3
"""
ADAN Paper Trading Monitor — Production Version (v3)
======================================================
100% Auditable Paper Trading Engine with isolated virtual wallet.

Architecture alignment (v3):
  - Uses ``StateBuilder.build_observation()`` for the 12-dim context_vector
    (6 market: Volatility, Trend, ADX, Regime, Drawdown, Candle_Progress
     + 6 Time2Vec: sin/cos of hour, weekday, day-of-month)
  - ``ContextualTemporalFusionExtractor`` with FiLM Meta-RL modulation
  - Action is a continuous Box(25,) "Target Weight" vector:
      * action[0] < -0.1 while long -> DYNAMIC EXIT (close at market)
      * action[0] > +0.33 -> BUY signal
      * action[0] < -0.33 -> SELL signal
  - Capital Tier supremacy (Micro/Small/Medium/High/Enterprise)

Virtual Wallet:
  - initial_balance = $20.50 (completely isolated from Binance Testnet)
  - max_balance = $25.00 (Micro Capital tier ceiling)
  - max_concurrent_positions = 1 (Micro Capital)
  - NO real orders are ever placed; ALL trades are simulated locally
  - The Binance Testnet is used ONLY for candle data (read-only)

Logging:
  - [INTENTION] Worker {profil} | Action raw: {action_raw} | Size raw: {size_raw}
  - [EXECUTION] ... BUY {size} BTC @ {price}

Data pipeline:
  1. Fetch 5m OHLCV from Binance Testnet (or offline parquet)
  2. Resample to 1h, 4h with Master Clock alignment
  3. Build observation via MultiAssetChunkedEnv's internal StateBuilder
  4. PPO model.predict() -> interpret Target-Weight action

Usage:
    python scripts/paper_trading_monitor.py --offline
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

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass

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


# ============================================================================
# ISOLATED VIRTUAL PORTFOLIO MANAGER
# ============================================================================
# This is 100% local. No connection to any exchange for order execution.
# The Binance Testnet API is used ONLY for reading candle data.
# ============================================================================

class VirtualPortfolioManager:
    """Isolated virtual portfolio — NO real orders, NO exchange leaks.

    Parameters:
        initial_balance: Starting cash ($20.50 default)
        max_balance: Maximum allowed balance ($25.00 Micro Capital ceiling)
        max_concurrent_positions: Maximum simultaneous positions (1 for Micro)
        fee_rate: Simulated trading fee (0.001 = 0.1%)

    All PnL calculations are local. The exchange is never contacted for
    order placement, only for market data.
    """

    def __init__(
        self,
        initial_balance: float = 20.50,
        max_balance: float = 25.00,
        max_concurrent_positions: int = 1,
        fee_rate: float = 0.001,
    ):
        self.initial_balance = initial_balance
        self.max_balance = max_balance
        self.max_concurrent_positions = max_concurrent_positions
        self.fee_rate = fee_rate

        # Virtual cash — completely isolated
        self.cash = initial_balance
        self.positions = []  # List of open position dicts
        self.closed_trades = []  # Historical trade log
        self.total_fees_paid = 0.0

        # Statistics
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance

        logger.info(
            f"[VIRTUAL_WALLET] Initialized: "
            f"cash=${initial_balance:.2f}, max=${max_balance:.2f}, "
            f"max_positions={max_concurrent_positions}, fee={fee_rate*100:.2f}%"
        )

    @property
    def equity(self) -> float:
        """Total equity = cash + unrealized PnL of open positions."""
        unrealized = sum(self._unrealized_pnl(p) for p in self.positions)
        return self.cash + unrealized

    def _unrealized_pnl(self, pos: dict) -> float:
        """Calculate unrealized PnL for a position."""
        if pos["side"] == "BUY":
            return (pos["current_price"] - pos["entry_price"]) * pos["size"]
        else:
            return (pos["entry_price"] - pos["current_price"]) * pos["size"]

    def can_open_position(self) -> bool:
        """Check if we can open a new position."""
        return len(self.positions) < self.max_concurrent_positions

    def open_position(
        self, side: str, price: float, size_usd: float,
        stop_loss_pct: float = 0.02, take_profit_pct: float = 0.03,
        asset: str = "BTCUSDT",
    ) -> dict:
        """Open a virtual position — NO exchange order placed.

        Args:
            side: "BUY" or "SELL"
            price: Current market price
            size_usd: Notional value in USD
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            asset: Trading pair

        Returns:
            Position dict if opened, None if rejected.
        """
        if not self.can_open_position():
            logger.warning(
                f"[VIRTUAL_WALLET] REJECTED: max positions "
                f"({self.max_concurrent_positions}) reached"
            )
            return None

        # Calculate fee
        fee = size_usd * self.fee_rate
        total_cost = size_usd + fee

        if total_cost > self.cash:
            logger.warning(
                f"[VIRTUAL_WALLET] REJECTED: insufficient cash "
                f"${self.cash:.2f} < ${total_cost:.2f}"
            )
            return None

        # Size in asset units
        size_asset = size_usd / price

        # SL/TP prices
        if side == "BUY":
            sl_price = price * (1 - stop_loss_pct)
            tp_price = price * (1 + take_profit_pct)
        else:
            sl_price = price * (1 + stop_loss_pct)
            tp_price = price * (1 - take_profit_pct)

        # Deduct from cash
        self.cash -= total_cost
        self.total_fees_paid += fee

        position = {
            "id": len(self.closed_trades) + len(self.positions) + 1,
            "side": side,
            "asset": asset,
            "entry_price": price,
            "current_price": price,
            "size": size_asset,
            "notional_usd": size_usd,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "entry_fee": fee,
            "opened_at": datetime.now().isoformat(),
            "steps_held": 0,
        }
        self.positions.append(position)
        self.total_trades += 1

        logger.info(
            f"[EXECUTION] VIRTUAL {side} {size_asset:.6f} {asset} @ ${price:.2f} "
            f"| Notional=${size_usd:.2f} | Fee=${fee:.4f} "
            f"| SL=${sl_price:.2f} TP=${tp_price:.2f} "
            f"| Cash remaining=${self.cash:.2f}"
        )

        return position

    def close_position(self, pos_idx: int, price: float, reason: str = "Manual") -> dict:
        """Close a virtual position — NO exchange order placed.

        Returns:
            Trade receipt dict.
        """
        if pos_idx >= len(self.positions):
            return None

        pos = self.positions.pop(pos_idx)

        # Calculate PnL
        if pos["side"] == "BUY":
            pnl_usd = (price - pos["entry_price"]) * pos["size"]
        else:
            pnl_usd = (pos["entry_price"] - price) * pos["size"]

        # Exit fee
        exit_notional = price * pos["size"]
        exit_fee = exit_notional * self.fee_rate
        net_pnl = pnl_usd - exit_fee
        self.total_fees_paid += exit_fee

        # Return proceeds to cash
        self.cash += pos["notional_usd"] + net_pnl
        pnl_pct = (net_pnl / pos["notional_usd"]) * 100 if pos["notional_usd"] > 0 else 0

        # Track wins/losses
        if net_pnl > 0:
            self.wins += 1
        else:
            self.losses += 1

        # Update peak and drawdown
        current_equity = self.equity
        if current_equity > self.peak_balance:
            self.peak_balance = current_equity
        dd = (self.peak_balance - current_equity) / self.peak_balance if self.peak_balance > 0 else 0
        if dd > self.max_drawdown:
            self.max_drawdown = dd

        receipt = {
            "id": pos["id"],
            "side": pos["side"],
            "asset": pos["asset"],
            "entry_price": pos["entry_price"],
            "exit_price": price,
            "size": pos["size"],
            "notional_usd": pos["notional_usd"],
            "pnl_usd": net_pnl,
            "pnl_pct": pnl_pct,
            "fees": pos["entry_fee"] + exit_fee,
            "reason": reason,
            "opened_at": pos["opened_at"],
            "closed_at": datetime.now().isoformat(),
            "steps_held": pos["steps_held"],
            "balance_after": self.cash,
        }
        self.closed_trades.append(receipt)

        logger.info(
            f"[EXECUTION] VIRTUAL CLOSE {pos['side']} ({reason}): "
            f"PnL=${net_pnl:+.4f} ({pnl_pct:+.2f}%) | "
            f"Fees=${pos['entry_fee'] + exit_fee:.4f} | "
            f"Balance=${self.cash:.2f}"
        )

        return receipt

    def update_prices(self, price: float, asset: str = "BTCUSDT"):
        """Update current prices and check SL/TP for open positions."""
        to_close = []
        for i, pos in enumerate(self.positions):
            if pos["asset"] != asset:
                continue
            pos["current_price"] = price
            pos["steps_held"] += 1

            # Check SL/TP
            if pos["side"] == "BUY":
                if price <= pos["stop_loss"]:
                    to_close.append((i, "STOP_LOSS"))
                elif price >= pos["take_profit"]:
                    to_close.append((i, "TAKE_PROFIT"))
            else:
                if price >= pos["stop_loss"]:
                    to_close.append((i, "STOP_LOSS"))
                elif price <= pos["take_profit"]:
                    to_close.append((i, "TAKE_PROFIT"))

        # Close in reverse order to preserve indices
        for idx, reason in reversed(to_close):
            self.close_position(idx, price, reason=reason)

    def get_summary(self) -> dict:
        """Generate portfolio summary."""
        total_return = (self.equity - self.initial_balance) / self.initial_balance * 100
        win_rate = self.wins / max(1, self.wins + self.losses) * 100
        return {
            "initial_balance": self.initial_balance,
            "current_cash": self.cash,
            "current_equity": self.equity,
            "total_return_pct": total_return,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate_pct": win_rate,
            "max_drawdown_pct": self.max_drawdown * 100,
            "total_fees_paid": self.total_fees_paid,
            "open_positions": len(self.positions),
        }


# ── Capital Tier resolution ────────────────────────────────────────────────
def get_capital_tier(balance: float, tiers: list) -> dict:
    """Return the matching capital tier dict for the given balance."""
    for tier in tiers:
        min_cap = tier.get("min_capital", 0)
        max_cap = tier.get("max_capital") or float("inf")
        if min_cap <= balance < max_cap:
            return tier
    return {"name": "Micro Capital", "exposure_range": [70, 90],
            "risk_per_trade_pct": 4.0, "max_concurrent_positions": 1}


# ── Action interpreter ─────────────────────────────────────────────────────
def interpret_target_weight_action(action_raw, has_position: bool) -> dict:
    """Interpret the continuous Target-Weight action vector.

    Returns dict with keys: signal, size_raw, confidence.
    """
    arr = np.asarray(action_raw).flatten()
    if arr.size > 0:
        signal_raw = float(arr[0])
        size_raw = float(arr[1]) if arr.size > 1 else 0.0
    else:
        signal_raw = 0.0
        size_raw = 0.0

    # Dynamic exit: agent signals negative while already long
    if has_position and signal_raw < -0.1:
        return {"signal": "DYNAMIC_EXIT", "size_raw": size_raw,
                "confidence": abs(signal_raw)}

    if signal_raw > 0.33:
        return {"signal": "BUY", "size_raw": size_raw,
                "confidence": min(signal_raw, 1.0)}
    elif signal_raw < -0.33:
        return {"signal": "SELL", "size_raw": size_raw,
                "confidence": min(abs(signal_raw), 1.0)}

    return {"signal": "HOLD", "size_raw": size_raw,
            "confidence": 1.0 - abs(signal_raw) * 2}


class PaperTradingMonitor:
    """Real-time paper trading monitor for ADAN.

    Supports two modes:
      - Live:    connects to Binance Testnet via ccxt for CANDLE DATA ONLY
      - Offline: uses locally generated parquet data and replays step-by-step

    IMPORTANT: The virtual wallet ($20.50) is 100% isolated.
    No real orders are ever placed. The Testnet connection is READ-ONLY.
    """

    def __init__(self, config_path="config/config.yaml", api_key=None,
                 api_secret=None, offline=False):
        self.config = ConfigLoader.load_config(config_path)
        self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BINANCE_SECRET_KEY", "")
        self.testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
        self.symbol = os.getenv("TRADING_PAIR", "BTC/USDT")
        self.offline = offline

        # Capital tiers from config
        self.capital_tiers = self.config.get("capital_tiers", [])

        # ── ISOLATED VIRTUAL WALLET ──
        # This wallet has NO connection to Binance Testnet or any exchange.
        # It is purely a local simulation.
        self.portfolio = VirtualPortfolioManager(
            initial_balance=20.50,
            max_balance=25.00,
            max_concurrent_positions=1,
            fee_rate=0.001,
        )

        self.timeframes = ["5m", "1h", "4h"]
        self.analysis_interval = 5 if offline else 300  # seconds
        self.tp_sl_check_interval = 2 if offline else 30  # seconds

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

        mode_str = "OFFLINE (local data)" if self.offline else "LIVE (Binance Testnet READ-ONLY)"
        logger.info(f"Paper Trading Monitor initialized -- {mode_str}")
        logger.info(f"  Symbol: {self.symbol}")
        logger.info(f"  Virtual Balance: ${self.portfolio.cash:.2f} (ISOLATED)")
        logger.info(f"  Max Balance: ${self.portfolio.max_balance:.2f}")
        logger.info(f"  Testnet data: {self.testnet}")
        logger.info(f"  WARNING: NO real orders -- virtual wallet only")

    def setup_exchange(self):
        """Initialize ccxt exchange for READ-ONLY candle data.

        In offline mode, skip connection entirely.
        IMPORTANT: This connection is NEVER used for placing orders.
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

            # Test connection (read-only)
            self.exchange.fetch_time()
            logger.info("Exchange connected (READ-ONLY for candle data)")
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
                logger.error("No local data. Run generate_colab_dataset.py first.")
                return False

            raw_env = MultiAssetChunkedEnv(
                data=data, config=self.config, worker_config=wc,
                worker_id=0, live_mode=False,
            )
            dummy = DummyVecEnv([lambda: raw_env])

            # Try production model first, then simple model
            vecnorm_path = PROJECT_ROOT / "models" / "rl_agents" / "production" / "vecnormalize.pkl"
            if not vecnorm_path.exists():
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
            candidates = [
                PROJECT_ROOT / "models" / "rl_agents" / "production" / "model.zip",
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
        """Fetch 5m OHLCV (READ-ONLY) and resample to 1h, 4h."""
        if not self.exchange:
            return None

        try:
            all_candles = []
            since = None
            for _ in range(2):
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

            agg = {"open": "first", "high": "max", "low": "min",
                   "close": "last", "volume": "sum"}
            df_1h = df_5m.resample("1h").agg(agg).dropna()
            df_4h = df_5m.resample("4h").agg(agg).dropna()

            df_1h = df_1h.reindex(df_5m.index, method="ffill").dropna(subset=["close"])
            df_4h = df_4h.reindex(df_5m.index, method="ffill").dropna(subset=["close"])

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
                f"Live data (READ-ONLY): {len(common)} aligned rows, "
                f"5m price=${df_5m['close'].iloc[-1]:.2f}"
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

    def execute_signal(self, action_info: dict, action_raw, price: float):
        """Execute a trading signal via the VIRTUAL portfolio.

        Logs [INTENTION] before decision and [EXECUTION] after.
        NO real exchange orders are placed.
        """
        signal = action_info["signal"]
        tier = get_capital_tier(self.portfolio.cash, self.capital_tiers)
        tier_name = tier.get("name", "Micro Capital")

        # Extract raw values for logging
        arr = np.asarray(action_raw).flatten()
        raw0 = float(arr[0]) if arr.size > 0 else 0.0
        raw1 = float(arr[1]) if arr.size > 1 else 0.0

        # [INTENTION] log
        logger.info(
            f"[INTENTION] Worker {tier_name} | "
            f"Action raw: {raw0:.4f} | Size raw: {raw1:.4f} | "
            f"Signal: {signal} | Confidence: {action_info['confidence']:.3f} | "
            f"Price: ${price:.2f} | Cash: ${self.portfolio.cash:.2f}"
        )

        if signal == "DYNAMIC_EXIT":
            if self.portfolio.positions:
                self.portfolio.close_position(0, price, reason="DYNAMIC_EXIT")
            return

        if signal == "BUY" and self.portfolio.can_open_position():
            # Compute notional from tier exposure
            exposure_range = tier.get("exposure_range", [70, 90])
            exposure_pct = (exposure_range[0] + exposure_range[1]) / 2 / 100
            risk_pct = tier.get("risk_per_trade_pct", 4.0) / 100
            notional = min(
                self.portfolio.cash * exposure_pct,
                self.portfolio.cash * 0.95,  # Never use more than 95%
            )
            # Minimum order check
            if notional < 11.0:
                logger.info(
                    f"[EXECUTION] REJECTED: notional ${notional:.2f} < $11.00 minimum"
                )
                return

            self.portfolio.open_position(
                side="BUY",
                price=price,
                size_usd=notional,
                stop_loss_pct=min(0.02, risk_pct),
                take_profit_pct=0.03,
                asset=self.symbol.replace("/", ""),
            )

        elif signal == "SELL" and self.portfolio.positions:
            self.portfolio.close_position(0, price, reason="SELL_SIGNAL")

    def run_analysis_cycle(self):
        """One analysis cycle: fetch data -> predict -> execute."""
        if self.offline:
            return self._run_offline_cycle()

        data = self.fetch_live_data()
        if not data:
            return

        price = self.get_current_price()
        if price <= 0:
            return

        try:
            asset = self.symbol.replace("/", "")
            wc = copy.deepcopy(self.config.get("workers", {}).get("w1", {}))
            wc["worker_id"] = 0

            raw_env = MultiAssetChunkedEnv(
                data=data, config=self.config, worker_config=wc,
                worker_id=0, live_mode=False,
            )
            dummy_env = DummyVecEnv([lambda: raw_env])

            vecnorm_path = PROJECT_ROOT / "models" / "rl_agents" / "production" / "vecnormalize.pkl"
            if not vecnorm_path.exists():
                vecnorm_path = PROJECT_ROOT / "models" / "rl_agents" / "vecnormalize.pkl"

            if vecnorm_path.exists():
                vec_env = VecNormalize.load(str(vecnorm_path), dummy_env)
                vec_env.training = False
                vec_env.norm_reward = False
            else:
                gamma = self.config.get("agent", {}).get("gamma", 0.99)
                vec_env = VecNormalize(
                    dummy_env, norm_obs=True, norm_reward=False,
                    clip_obs=10.0, gamma=gamma, training=False,
                )

            obs = vec_env.reset()
            action, _ = self.model.predict(obs, deterministic=True)
            action_info = interpret_target_weight_action(
                action, has_position=len(self.portfolio.positions) > 0
            )

            if action_info["signal"] != "HOLD":
                self.execute_signal(action_info, action, price)

            vec_env.close()

        except Exception as e:
            logger.error(f"Analysis cycle failed: {e}", exc_info=True)

    def _run_offline_cycle(self):
        """Step through the pre-built offline env and predict."""
        if self._offline_vec_env is None or self._offline_obs is None:
            logger.error("Offline env not initialised")
            return

        try:
            obs = self._offline_obs

            action, _ = self.model.predict(obs, deterministic=True)
            action_info = interpret_target_weight_action(
                action, has_position=len(self.portfolio.positions) > 0
            )

            price = self.get_current_price()
            if price <= 0:
                price = 65000.0

            # Update virtual positions with current price
            self.portfolio.update_prices(price, self.symbol.replace("/", ""))

            if action_info["signal"] != "HOLD":
                self.execute_signal(action_info, action, price)

            # Step the environment forward
            obs_next, reward, done, info = self._offline_vec_env.step(action)
            self._offline_step += 1

            if done[0]:
                logger.info(f"Offline episode ended at step {self._offline_step}, resetting")
                obs_next = self._offline_vec_env.reset()
                self._offline_step = 0

            self._offline_obs = obs_next

            if self._offline_step % 50 == 0:
                logger.info(
                    f"[OFFLINE] step={self._offline_step} "
                    f"env_reward={reward[0]:.4f} "
                    f"virtual_equity=${self.portfolio.equity:.2f} "
                    f"positions={len(self.portfolio.positions)}"
                )

        except Exception as e:
            logger.error(f"Offline cycle failed: {e}", exc_info=True)

    def run(self, duration_minutes: int = 360):
        """Main event loop."""
        logger.info("=" * 70)
        logger.info("ADAN Paper Trading Monitor v3 (Isolated Virtual Wallet)")
        mode_str = "OFFLINE" if self.offline else "LIVE (READ-ONLY data)"
        logger.info(f"  Mode: {mode_str}")
        logger.info(f"  Duration: {duration_minutes} min")
        logger.info(f"  Analysis interval: {self.analysis_interval}s")
        logger.info(f"  Virtual Balance: ${self.portfolio.cash:.2f} (ISOLATED)")
        logger.info(f"  Tier: Micro Capital (max_positions=1)")
        logger.info(f"  NO REAL ORDERS will be placed")
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
                    price = self.get_current_price()
                    if price > 0:
                        self.portfolio.update_prices(
                            price, self.symbol.replace("/", "")
                        )
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
        summary = self.portfolio.get_summary()
        logger.info("=" * 70)
        logger.info("PAPER TRADING SESSION COMPLETE (VIRTUAL)")
        logger.info(f"  Trades: {summary['total_trades']}")
        logger.info(f"  Wins/Losses: {summary['wins']}/{summary['losses']}")
        logger.info(f"  Win Rate: {summary['win_rate_pct']:.1f}%")
        logger.info(f"  Final Equity: ${summary['current_equity']:.2f}")
        logger.info(f"  Return: {summary['total_return_pct']:+.2f}%")
        logger.info(f"  Max Drawdown: {summary['max_drawdown_pct']:.2f}%")
        logger.info(f"  Total Fees: ${summary['total_fees_paid']:.4f}")
        logger.info("=" * 70)

        # Save results
        results = {
            **summary,
            "trades": self.portfolio.closed_trades,
            "timestamp": datetime.now().isoformat(),
            "mode": "offline" if self.offline else "live",
            "wallet_type": "VIRTUAL_ISOLATED",
        }
        results_path = PROJECT_ROOT / "results" / "paper_trading_report.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Report saved: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ADAN Paper Trading Monitor v3 (Isolated Virtual Wallet)"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", default=None, help="Model .zip path")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-secret", default=None)
    parser.add_argument("--duration", type=int, default=360,
                        help="Duration in minutes")
    parser.add_argument(
        "--offline", action="store_true",
        help="Run offline using local parquet data (no exchange needed)",
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
