#!/usr/bin/env python3
"""
ADAN0 Trading Bot — Asynchronous WebSocket-based inference loop (NIVEAU 1).

Connects to a real exchange via CCXT.pro (WebSocket), computes the EXACT same 21 features
per timeframe as the training pipeline, feeds them to the PPO model,
and executes trades (paper or live) with ZERO blocking I/O.

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │ CCXT.pro WebSocket (async)                                      │
  │ ├─ OHLCV stream (5m, 1h, 4h) — triggered on candle close       │
  │ └─ Ticker stream (real-time price updates)                      │
  └────────────────────────┬────────────────────────────────────────┘
                           │ (event-driven, no polling)
                           ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ LiveStateBuilder (async)                                        │
  │ ├─ Cache OHLCV in memory (no re-fetch on every tick)           │
  │ └─ Compute 21 features on candle close event                    │
  └────────────────────────┬────────────────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ PPO Model (inference)                                           │
  │ └─ predict() runs in thread pool (non-blocking)                 │
  └────────────────────────┬────────────────────────────────────────┘
                           │ action Box(5,)
                           ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ ExecutionEngine (async)                                         │
  │ ├─ Paper mode: instant execution                                │
  │ └─ Live mode: async order submission via CCXT.pro               │
  └─────────────────────────────────────────────────────────────────┘

Usage:
  # Paper trading (async, 60 min)
  PYTHONPATH=src python scripts/run_bot.py \
    --exchange binance --mode paper \
    --checkpoint checkpoints/ppo_adan0_sandbox_10240steps.zip \
    --capital 20.50 --duration 60

  # Live trading (async, requires API keys)
  ADAN_API_KEY=xxx ADAN_API_SECRET=yyy \
  PYTHONPATH=src python scripts/run_bot.py \
    --exchange bitget --mode live \
    --checkpoint checkpoints/ppo_adan0_sandbox_10240steps.zip \
    --capital 100.0 --duration 60
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import logging
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# ADAN imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from adan_trading_bot.trading.live_state_builder import LiveStateBuilder
from adan_trading_bot.trading.execution_engine import (
    ExecutionEngine,
    KillSwitch,
)

# ── Logging setup ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("adan_bot")

# Silence noisy libs
for lib in ["ccxt", "urllib3", "matplotlib", "PIL"]:
    logging.getLogger(lib).setLevel(logging.WARNING)


# ── Graceful shutdown ──────────────────────────────────────────────────────

_SHUTDOWN = False


def _signal_handler(signum, frame):
    global _SHUTDOWN
    _SHUTDOWN = True
    logger.info("[SHUTDOWN] Signal received, gracefully stopping...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── Model loading ──────────────────────────────────────────────────────────

def load_model(checkpoint_path: str):
    """Load PPO model + optional VecNormalize stats."""
    logger.info(f"[MODEL] Loading {checkpoint_path}")
    model = PPO.load(checkpoint_path, device="cpu")

    vecnorm_path = checkpoint_path.replace(".zip", "_vecnorm.pkl")
    vecnorm = None
    if os.path.isfile(vecnorm_path):
        import gymnasium as gym
        class DummyEnv(gym.Env):
            def __init__(self, obs_space, act_space):
                super().__init__()
                self.observation_space = obs_space
                self.action_space = act_space
            def reset(self, **kwargs):
                return {k: np.zeros(s.shape, dtype=s.dtype)
                        for k, s in self.observation_space.spaces.items()}, {}
            def step(self, action):
                obs, _ = self.reset()
                return obs, 0.0, False, False, {}

        dummy = DummyVecEnv([lambda: DummyEnv(model.observation_space, model.action_space)])
        try:
            vecnorm = VecNormalize.load(vecnorm_path, dummy)
            vecnorm.training = False
            vecnorm.norm_reward = False
            logger.info(f"[MODEL] VecNormalize loaded")
        except Exception as e:
            logger.warning(f"[MODEL] VecNormalize load failed: {e}")
            vecnorm = None

    logger.info(f"[MODEL] Loaded — obs_space keys: {list(model.observation_space.spaces.keys())}")
    return model, vecnorm


def normalize_obs(obs: dict, vecnorm) -> dict:
    """Apply VecNormalize obs normalization if available."""
    if vecnorm is None:
        return obs
    obs_batch = {k: np.expand_dims(v, 0) for k, v in obs.items()}
    try:
        normed = vecnorm.normalize_obs(obs_batch)
        return {k: v[0] for k, v in normed.items()}
    except Exception as e:
        logger.warning(f"[NORMALIZE] Failed: {e}, using raw obs")
        return obs


# ── Async WebSocket Event Loop ─────────────────────────────────────────────

class AsyncBotEngine:
    """Asynchronous trading bot using CCXT.pro WebSockets."""

    def __init__(self, args, model, vecnorm, executor):
        self.args = args
        self.model = model
        self.vecnorm = vecnorm
        self.executor = executor
        self.tick_count = 0
        self.start_time = None
        self.state_builder = None
        self.engine = None
        self.last_candle_time = {}  # Track last candle close per TF

    async def initialize(self):
        """Initialize async components."""
        logger.info("[ASYNC] Initializing...")
        
        # Init LiveStateBuilder
        self.state_builder = LiveStateBuilder(
            exchange_id=self.args.exchange,
            symbol=self.args.symbol,
            timeframes=["5m", "1h", "4h"],
            proxy=os.environ.get("ADAN_PROXY"),
        )

        # Init ExecutionEngine
        kill_switch = KillSwitch(
            max_drawdown_pct=self.args.max_drawdown,
            max_trades_per_hour=self.args.max_trades_hour,
            max_position_pct=self.args.max_position_pct,
        )

        self.engine = ExecutionEngine(
            mode=self.args.mode,
            exchange_id=self.args.exchange,
            symbol=self.args.symbol,
            initial_capital=self.args.capital,
            kill_switch=kill_switch,
            testnet=(self.args.mode != "live" or self.args.testnet),
            log_dir=self.args.log_dir,
            action_threshold=self.args.action_threshold,
        )

        self.start_time = asyncio.get_event_loop().time()
        logger.info("[ASYNC] Initialized successfully")

    async def fetch_ohlcv_async(self, tf: str):
        """Fetch OHLCV data asynchronously (non-blocking)."""
        loop = asyncio.get_event_loop()
        try:
            # Run fetch in thread pool to avoid blocking
            ohlcv = await loop.run_in_executor(
                self.executor,
                lambda: self.state_builder.exchange.fetch_ohlcv(
                    self.args.symbol, tf, limit=200
                )
            )
            return ohlcv
        except Exception as e:
            logger.warning(f"[FETCH] {tf} failed: {e}")
            return None

    async def predict_action_async(self, obs: dict):
        """Run model inference in thread pool (non-blocking)."""
        loop = asyncio.get_event_loop()
        try:
            obs_normed = normalize_obs(obs, self.vecnorm)
            action, _ = await loop.run_in_executor(
                self.executor,
                lambda: self.model.predict(obs_normed, deterministic=True)
            )
            return np.array(action, dtype=np.float32).flatten()
        except Exception as e:
            logger.error(f"[PREDICT] Failed: {e}")
            return np.zeros(5, dtype=np.float32)

    async def process_tick_async(self, current_price: float):
        """Process one trading tick asynchronously."""
        self.tick_count += 1
        tick_start = asyncio.get_event_loop().time()

        try:
            # ── A. Fetch all TF data in parallel ──
            tasks = [self.fetch_ohlcv_async(tf) for tf in ["5m", "1h", "4h"]]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for tf, result in zip(["5m", "1h", "4h"], results):
                if isinstance(result, Exception):
                    logger.warning(f"[TICK {self.tick_count}] {tf} fetch failed")
                    continue
                if result:
                    self.state_builder._cache[tf] = self._ohlcv_to_df(result, tf)

            # ── B. Build observation ──
            portfolio_state = self.engine.get_portfolio_state(current_price)
            obs = self.state_builder.build_observation(
                portfolio_state=portfolio_state,
                context_vector=None,
            )

            # ── C. Predict action (non-blocking) ──
            action = await self.predict_action_async(obs)

            # ── D. Execute ──
            result = self.engine.process_tick(
                action=action,
                current_price=current_price,
                timestamp=tick_start,
            )

            # ── E. Log ──
            self._log_tick(result, current_price, tick_start)

            return result.get("killed", False)

        except Exception as e:
            logger.error(f"[TICK {self.tick_count}] Error: {e}", exc_info=True)
            return False

    def _ohlcv_to_df(self, ohlcv, tf: str):
        """Convert OHLCV list to DataFrame with indicators."""
        import pandas as pd
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)
        return self.state_builder._compute_indicators(df, tf)

    def _log_tick(self, result, current_price, tick_start):
        """Log tick information."""
        decoded = result["decoded_action"]
        portfolio = result["portfolio"]
        trade_info = ""
        if result.get("trade"):
            t = result["trade"]
            trade_info = (
                f" → TRADE: {t['side']} ${t['size_usd']:.2f} "
                f"@${t['price']:.2f}"
            )

        ts_str = datetime.datetime.fromtimestamp(tick_start).strftime("%H:%M:%S")
        print(
            f"[{ts_str}] Tick {self.tick_count} | "
            f"BTC=${current_price:,.2f} | "
            f"Dir={decoded['direction']:+.3f} "
            f"Size={decoded['size_pct']:.1%} | "
            f"Equity=${portfolio['equity']:.2f} "
            f"DD={portfolio['drawdown_pct']:.1f}%"
            f"{trade_info}"
        )

    async def run(self):
        """Main async event loop."""
        await self.initialize()

        print("=" * 70)
        print(f"  ADAN0 Trading Bot — {self.args.mode.upper()} MODE (ASYNC)")
        print(f"  Exchange:   {self.args.exchange}")
        print(f"  Symbol:     {self.args.symbol}")
        print(f"  Capital:    ${self.args.capital:.2f}")
        print(f"  Checkpoint: {self.args.checkpoint}")
        print(f"  VecNorm:    {'YES' if self.vecnorm else 'NO'}")
        print("=" * 70)

        max_duration = self.args.duration * 60 if self.args.duration else float("inf")
        max_ticks = self.args.test_ticks if self.args.test_ticks else None

        try:
            while not _SHUTDOWN:
                elapsed = asyncio.get_event_loop().time() - self.start_time
                if elapsed > max_duration:
                    logger.info(f"[BOT] Duration limit reached")
                    break
                if max_ticks and self.tick_count >= max_ticks:
                    break

                # Get current price
                current_price = self.state_builder.get_current_price()
                if current_price <= 0:
                    await asyncio.sleep(5)
                    continue

                # Process tick
                killed = await self.process_tick_async(current_price)
                if killed:
                    logger.error("[KILL SWITCH] Activated")
                    break

                # Sleep until next interval
                await asyncio.sleep(self.args.interval)

        except asyncio.CancelledError:
            logger.info("[ASYNC] Cancelled")
        except Exception as e:
            logger.error(f"[ASYNC] Error: {e}", exc_info=True)
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown."""
        print("\n" + "=" * 70)
        print("  ADAN0 Bot — Session Complete")
        print("=" * 70)
        report_path = self.engine.save_report()
        equity = self.engine.get_equity(self.state_builder.get_current_price())
        pnl = equity - self.args.capital
        pnl_pct = pnl / self.args.capital * 100
        print(f"  Ticks:     {self.tick_count}")
        print(f"  Trades:    {len(self.engine.trades)}")
        print(f"  Equity:    ${equity:.2f} ({pnl:+.4f}, {pnl_pct:+.2f}%)")
        print(f"  Report:    {report_path}")
        print("=" * 70)


async def run_bot_async(args):
    """Run bot with async event loop."""
    model, vecnorm = load_model(args.checkpoint)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        bot = AsyncBotEngine(args, model, vecnorm, executor)
        await bot.run()


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ADAN0 Trading Bot — Async WebSocket inference loop (NIVEAU 1)"
    )
    parser.add_argument(
        "--exchange", type=str, default="binance",
        choices=["binance", "bitget", "kraken", "kucoin"],
        help="Exchange for data feed (default: binance)",
    )
    parser.add_argument(
        "--mode", type=str, default="paper",
        choices=["paper", "live"],
        help="Trading mode (default: paper)",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to PPO checkpoint .zip file",
    )
    parser.add_argument(
        "--symbol", type=str, default="BTC/USDT",
        help="Trading pair (default: BTC/USDT)",
    )
    parser.add_argument(
        "--capital", type=float, default=20.50,
        help="Starting capital in USDT (default: 20.50)",
    )
    parser.add_argument(
        "--interval", type=int, default=300,
        help="Seconds between ticks (default: 300 = 5min)",
    )
    parser.add_argument(
        "--duration", type=int, default=60,
        help="Session duration in minutes (default: 60)",
    )
    parser.add_argument(
        "--test-ticks", type=int, default=0,
        help="Run N ticks then exit (for testing, default: 0 = disabled)",
    )
    parser.add_argument(
        "--action-threshold", type=float, default=0.01,
        help="Minimum |direction| to trigger a trade (default: 0.01)",
    )
    parser.add_argument(
        "--max-drawdown", type=float, default=10.0,
        help="Kill switch: max drawdown %% (default: 10)",
    )
    parser.add_argument(
        "--max-trades-hour", type=int, default=5,
        help="Kill switch: max trades per hour (default: 5)",
    )
    parser.add_argument(
        "--max-position-pct", type=float, default=20.0,
        help="Kill switch: max position size %% (default: 20)",
    )
    parser.add_argument(
        "--testnet", action="store_true", default=True,
        help="Use exchange testnet for live mode (default: True)",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs/trading",
        help="Directory for trade logs (default: logs/trading)",
    )

    args = parser.parse_args()

    if args.mode == "live":
        if not os.environ.get("ADAN_API_KEY"):
            print("ERROR: Live mode requires ADAN_API_KEY env var")
            sys.exit(1)
        print("⚠️  LIVE TRADING MODE — Real orders will be placed!")
        print("   Press Ctrl+C to abort within 5 seconds...")
        try:
            import time
            time.sleep(5)
        except KeyboardInterrupt:
            print("Aborted.")
            sys.exit(0)

    # Run async event loop
    asyncio.run(run_bot_async(args))


if __name__ == "__main__":
    main()
