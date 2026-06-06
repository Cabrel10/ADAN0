"""
ExecutionEngine — Paper trading (virtual portfolio) and Live trading (CCXT orders).

Decodes the PPO Box(5,) action:
  action[0] = direction   ∈ [-1, 1]  → -1=SHORT, 0=HOLD, +1=LONG
  action[1] = size_pct    ∈ [-1, 1]  → mapped to [0, max_position_pct]
  action[2] = tf_pref     ∈ [-1, 1]  → timeframe preference (informational)
  action[3] = sl_pct      ∈ [-1, 1]  → mapped to [0.5%, 5%] stop-loss
  action[4] = tp_pct      ∈ [-1, 1]  → mapped to [0.5%, 10%] take-profit

Architecture:
  Paper mode:  virtual portfolio dict, no real orders, full logging
  Live mode:   ccxt.create_market_order() with kill switches
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import ccxt
except ImportError:
    ccxt = None

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────────────────────

SLIPPAGE_BPS = 2.0  # 2 basis points per side (same as training env)


@dataclass
class TradeRecord:
    timestamp: float
    side: str       # "BUY" or "SELL"
    symbol: str
    price: float
    size_usd: float
    size_asset: float
    sl_pct: float
    tp_pct: float
    fee_usd: float
    source: str     # "paper" or "live"
    order_id: str = ""
    pnl_usd: float = 0.0
    reason: str = ""


@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    size_usd: float
    size_asset: float
    sl_price: float
    tp_price: float
    open_time: float
    unrealized_pnl: float = 0.0

    def update_pnl(self, current_price: float):
        if self.side == "BUY":
            self.unrealized_pnl = (current_price - self.entry_price) * self.size_asset
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.size_asset


@dataclass
class KillSwitch:
    """Safety limits — breaching any of these shuts down the bot."""
    max_drawdown_pct: float = 10.0
    max_trades_per_hour: int = 5
    max_loss_per_trade_pct: float = 3.0
    max_position_pct: float = 20.0  # max % of capital per trade
    min_trade_interval_sec: float = 30.0


class ExecutionEngine:
    """Unified execution engine for paper and live trading.

    Handles:
    - Action decoding (Box(5,) → direction, size, SL, TP)
    - Slippage simulation (2 bps directional, same as training)
    - Paper portfolio tracking
    - Live CCXT order placement
    - Kill switch enforcement
    - Trade logging to JSON
    """

    def __init__(
        self,
        mode: str = "paper",          # "paper" or "live"
        exchange_id: str = "binance",
        symbol: str = "BTC/USDT",
        initial_capital: float = 20.50,
        kill_switch: KillSwitch = None,
        api_key: str = None,
        api_secret: str = None,
        testnet: bool = True,
        log_dir: str = "logs/trading",
        action_threshold: float = 0.01,
    ):
        self.mode = mode
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.kill_switch = kill_switch or KillSwitch()
        self.action_threshold = action_threshold
        self.testnet = testnet

        # Portfolio state
        self.cash = initial_capital
        self.position: Optional[Position] = None
        self.equity_high = initial_capital
        self.trades: List[TradeRecord] = []
        self._last_trade_time = 0.0
        self._killed = False
        self._kill_reason = ""

        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Live CCXT client
        self.live_client = None
        if mode == "live" and ccxt is not None:
            self._init_live_client(api_key, api_secret)

        logger.info(
            f"[ExecutionEngine] mode={mode} exchange={exchange_id} "
            f"symbol={symbol} capital=${initial_capital:.2f} "
            f"testnet={testnet}"
        )

    def _init_live_client(self, api_key: str, api_secret: str):
        """Initialize CCXT client for live trading."""
        # NEVER hardcode keys — must come from env vars or explicit args
        key = api_key or os.environ.get("ADAN_API_KEY", "")
        secret = api_secret or os.environ.get("ADAN_API_SECRET", "")
        passphrase = os.environ.get("ADAN_API_PASSPHRASE", "")

        if not key or not secret:
            logger.error("[ExecutionEngine] LIVE mode requires API keys. "
                         "Set ADAN_API_KEY and ADAN_API_SECRET env vars.")
            self._killed = True
            self._kill_reason = "Missing API keys"
            return

        exchange_cls = getattr(ccxt, self.exchange_id, None)
        if exchange_cls is None:
            raise ValueError(f"Unknown exchange: {self.exchange_id}")

        cfg = {
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
        if passphrase:
            cfg["password"] = passphrase
        if self.testnet:
            cfg["sandbox"] = True

        self.live_client = exchange_cls(cfg)
        if self.testnet:
            self.live_client.set_sandbox_mode(True)
        logger.info(f"[ExecutionEngine] LIVE client initialized: {self.exchange_id} "
                     f"testnet={self.testnet}")

    # ── Action decoding ────────────────────────────────────────────────

    def decode_action(self, action: np.ndarray) -> Dict[str, float]:
        """Decode Box(5,) action to human-readable parameters.

        action[0] = direction   ∈ [-1, 1]
        action[1] = size_pct    ∈ [-1, 1] → [0, kill_switch.max_position_pct]%
        action[2] = tf_pref     ∈ [-1, 1] (informational)
        action[3] = sl_pct      ∈ [-1, 1] → [0.5, 5]%
        action[4] = tp_pct      ∈ [-1, 1] → [0.5, 10]%
        """
        direction = float(action[0])
        raw_size = float(action[1])
        tf_pref = float(action[2])
        raw_sl = float(action[3])
        raw_tp = float(action[4])

        # Map to real ranges
        size_pct = abs(raw_size) * self.kill_switch.max_position_pct / 100.0
        sl_pct = 0.005 + abs(raw_sl) * 0.045   # [0.5%, 5%]
        tp_pct = 0.005 + abs(raw_tp) * 0.095    # [0.5%, 10%]

        return {
            "direction": direction,
            "size_pct": size_pct,
            "tf_pref": tf_pref,
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
            "raw_action": action.tolist(),
        }

    # ── Kill switch checks ─────────────────────────────────────────────

    def _check_kill_switch(self) -> bool:
        """Return True if trading should stop."""
        if self._killed:
            return True

        # Max drawdown
        equity = self.get_equity(self.get_current_price_cached())
        if equity < self.equity_high:
            dd_pct = (self.equity_high - equity) / self.equity_high * 100
            if dd_pct > self.kill_switch.max_drawdown_pct:
                self._killed = True
                self._kill_reason = f"MAX_DRAWDOWN: {dd_pct:.2f}% > {self.kill_switch.max_drawdown_pct}%"
                logger.error(f"[KILL SWITCH] {self._kill_reason}")
                return True

        # Trades per hour
        now = time.time()
        recent_trades = [t for t in self.trades if now - t.timestamp < 3600]
        if len(recent_trades) >= self.kill_switch.max_trades_per_hour:
            logger.warning(f"[KILL SWITCH] Rate limit: {len(recent_trades)} trades/hour")
            return True

        # Min interval between trades
        if now - self._last_trade_time < self.kill_switch.min_trade_interval_sec:
            return True

        return False

    _price_cache: float = 0.0
    _price_cache_ts: float = 0.0

    def get_current_price_cached(self) -> float:
        """Return cached price (updated by process_tick)."""
        return self._price_cache or self.initial_capital

    # ── Core execution ─────────────────────────────────────────────────

    def process_tick(
        self,
        action: np.ndarray,
        current_price: float,
        timestamp: float = None,
    ) -> Dict[str, Any]:
        """Process one inference tick: decode action, check SL/TP, execute.

        Returns a dict with tick results for logging.
        """
        ts = timestamp or time.time()
        self._price_cache = current_price
        self._price_cache_ts = ts

        decoded = self.decode_action(action)
        direction = decoded["direction"]
        result = {
            "timestamp": ts,
            "price": current_price,
            "decoded_action": decoded,
            "position": None,
            "trade": None,
            "portfolio": self._portfolio_snapshot(current_price),
            "killed": self._killed,
            "kill_reason": self._kill_reason,
        }

        # Update equity high water mark
        equity = self.get_equity(current_price)
        if equity > self.equity_high:
            self.equity_high = equity

        # Update existing position PnL
        if self.position is not None:
            self.position.update_pnl(current_price)
            result["position"] = asdict(self.position)

            # Check SL/TP
            sl_tp_result = self._check_sl_tp(current_price, ts)
            if sl_tp_result:
                result["trade"] = asdict(sl_tp_result)
                return result

        # Kill switch
        if self._check_kill_switch():
            result["killed"] = self._killed
            result["kill_reason"] = self._kill_reason
            return result

        # Decide action
        if abs(direction) < self.action_threshold:
            # HOLD — action too weak
            return result

        if direction > self.action_threshold and self.position is None:
            # BUY signal, no existing position
            trade = self._execute_open(
                side="BUY",
                price=current_price,
                size_pct=decoded["size_pct"],
                sl_pct=decoded["sl_pct"],
                tp_pct=decoded["tp_pct"],
                timestamp=ts,
            )
            if trade:
                result["trade"] = asdict(trade)

        elif direction < -self.action_threshold and self.position is not None:
            # SELL signal, close existing position
            trade = self._execute_close(
                price=current_price,
                reason="AGENT_CLOSE",
                timestamp=ts,
            )
            if trade:
                result["trade"] = asdict(trade)

        result["portfolio"] = self._portfolio_snapshot(current_price)
        return result

    # ── SL/TP check ────────────────────────────────────────────────────

    def _check_sl_tp(self, price: float, ts: float) -> Optional[TradeRecord]:
        """Check stop-loss and take-profit on the current position."""
        if self.position is None:
            return None

        pos = self.position
        if pos.side == "BUY":
            if price <= pos.sl_price:
                return self._execute_close(price, "STOP_LOSS", ts)
            if price >= pos.tp_price:
                return self._execute_close(price, "TAKE_PROFIT", ts)
        return None

    # ── Open / Close ───────────────────────────────────────────────────

    def _execute_open(
        self, side: str, price: float, size_pct: float,
        sl_pct: float, tp_pct: float, timestamp: float,
    ) -> Optional[TradeRecord]:
        """Open a new position (paper or live)."""
        if self.position is not None:
            return None  # Only one position at a time

        # Slippage: BUY at higher price
        fill_price = price * (1.0 + SLIPPAGE_BPS / 10000.0) if side == "BUY" else \
                     price * (1.0 - SLIPPAGE_BPS / 10000.0)

        # Position size
        size_usd = self.cash * size_pct
        if size_usd < 1.0:  # minimum trade size
            return None
        size_usd = min(size_usd, self.cash * 0.95)  # Never use 100% of cash

        size_asset = size_usd / fill_price
        fee_usd = size_usd * 0.001  # 0.1% maker/taker fee

        # SL/TP prices
        if side == "BUY":
            sl_price = fill_price * (1.0 - sl_pct)
            tp_price = fill_price * (1.0 + tp_pct)
        else:
            sl_price = fill_price * (1.0 + sl_pct)
            tp_price = fill_price * (1.0 - tp_pct)

        # Execute
        if self.mode == "live" and self.live_client:
            order = self._place_live_order(side, size_asset)
            if order is None:
                return None
            fill_price = order.get("average", fill_price)
            fee_usd = order.get("fee", {}).get("cost", fee_usd)

        # Update portfolio
        self.cash -= (size_usd + fee_usd)
        self.position = Position(
            symbol=self.symbol,
            side=side,
            entry_price=fill_price,
            size_usd=size_usd,
            size_asset=size_asset,
            sl_price=sl_price,
            tp_price=tp_price,
            open_time=timestamp,
        )

        trade = TradeRecord(
            timestamp=timestamp,
            side=side,
            symbol=self.symbol,
            price=fill_price,
            size_usd=size_usd,
            size_asset=size_asset,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            fee_usd=fee_usd,
            source=self.mode,
            reason="OPEN",
        )
        self.trades.append(trade)
        self._last_trade_time = timestamp

        logger.info(
            f"[{self.mode.upper()}_TRADE] {side} {self.symbol} "
            f"size=${size_usd:.2f} price=${fill_price:.2f} "
            f"SL=${sl_price:.2f} TP=${tp_price:.2f} fee=${fee_usd:.4f}"
        )
        return trade

    def _execute_close(
        self, price: float, reason: str, timestamp: float,
    ) -> Optional[TradeRecord]:
        """Close the existing position."""
        if self.position is None:
            return None

        pos = self.position

        # Slippage: SELL at lower price
        fill_price = price * (1.0 - SLIPPAGE_BPS / 10000.0)

        # PnL
        if pos.side == "BUY":
            pnl = (fill_price - pos.entry_price) * pos.size_asset
        else:
            pnl = (pos.entry_price - fill_price) * pos.size_asset

        fee_usd = pos.size_usd * 0.001  # closing fee

        # Execute live order
        if self.mode == "live" and self.live_client:
            order = self._place_live_order("sell", pos.size_asset)
            if order:
                fill_price = order.get("average", fill_price)

        # Update portfolio
        self.cash += pos.size_usd + pnl - fee_usd
        self.position = None

        trade = TradeRecord(
            timestamp=timestamp,
            side="SELL",
            symbol=self.symbol,
            price=fill_price,
            size_usd=pos.size_usd,
            size_asset=pos.size_asset,
            sl_pct=0,
            tp_pct=0,
            fee_usd=fee_usd,
            source=self.mode,
            pnl_usd=pnl,
            reason=reason,
        )
        self.trades.append(trade)
        self._last_trade_time = timestamp

        logger.info(
            f"[{self.mode.upper()}_TRADE] CLOSE({reason}) {self.symbol} "
            f"price=${fill_price:.2f} PnL=${pnl:+.4f} fee=${fee_usd:.4f} "
            f"cash=${self.cash:.2f}"
        )
        return trade

    # ── Live order placement ───────────────────────────────────────────

    def _place_live_order(self, side: str, amount: float) -> Optional[Dict]:
        """Place a real order via CCXT. Returns order dict or None."""
        if not self.live_client:
            return None
        try:
            order = self.live_client.create_market_order(
                symbol=self.symbol,
                side=side.lower(),
                amount=amount,
            )
            logger.info(f"[LIVE ORDER] {side} {amount} {self.symbol} → {order['id']}")
            return order
        except Exception as e:
            logger.error(f"[LIVE ORDER FAILED] {e}")
            return None

    # ── Portfolio helpers ──────────────────────────────────────────────

    def get_equity(self, current_price: float) -> float:
        """Total equity = cash + unrealized position value."""
        eq = self.cash
        if self.position is not None:
            self.position.update_pnl(current_price)
            eq += self.position.size_usd + self.position.unrealized_pnl
        return eq

    def get_portfolio_state(self, current_price: float) -> np.ndarray:
        """Build portfolio_state vector (20,) matching training env."""
        state = np.zeros(20, dtype=np.float32)
        equity = self.get_equity(current_price)
        state[0] = equity / self.initial_capital  # equity ratio
        state[1] = self.cash / self.initial_capital  # cash ratio
        state[2] = 1.0 if self.position is not None else 0.0  # has_position
        if self.position is not None:
            state[3] = self.position.unrealized_pnl / self.initial_capital
            state[4] = (current_price - self.position.entry_price) / self.position.entry_price
            state[5] = self.position.size_usd / self.initial_capital
            state[6] = 1.0 if self.position.side == "BUY" else -1.0
        state[7] = len(self.trades)  # trade count
        state[8] = (self.equity_high - equity) / self.equity_high if self.equity_high > 0 else 0
        return state

    def _portfolio_snapshot(self, price: float) -> Dict[str, Any]:
        return {
            "cash": round(self.cash, 4),
            "equity": round(self.get_equity(price), 4),
            "drawdown_pct": round(
                (self.equity_high - self.get_equity(price)) / self.equity_high * 100
                if self.equity_high > 0 else 0, 2
            ),
            "n_trades": len(self.trades),
            "has_position": self.position is not None,
        }

    # ── Reporting ─────────────────────────────────────────────────────

    def save_report(self, filename: str = None) -> str:
        """Save trading session report to JSON."""
        if filename is None:
            filename = f"trading_{self.mode}_{self.exchange_id}_{int(time.time())}.json"

        price = self._price_cache or 0.0
        report = {
            "mode": self.mode,
            "exchange": self.exchange_id,
            "symbol": self.symbol,
            "initial_capital": self.initial_capital,
            "final_equity": round(self.get_equity(price), 4),
            "return_pct": round(
                (self.get_equity(price) - self.initial_capital) / self.initial_capital * 100, 4
            ),
            "n_trades": len(self.trades),
            "trades": [asdict(t) for t in self.trades],
            "killed": self._killed,
            "kill_reason": self._kill_reason,
            "max_drawdown_pct": round(
                (self.equity_high - min(
                    self.get_equity(price), self.equity_high
                )) / self.equity_high * 100 if self.equity_high > 0 else 0, 2
            ),
        }

        path = self.log_dir / filename
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"[ExecutionEngine] Report saved: {path}")
        return str(path)
