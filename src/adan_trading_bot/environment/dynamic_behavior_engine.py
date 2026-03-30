"""
Dynamic Behavior Engine (DBE) — SENSOR-ONLY version.

OMEGA-3 Refactor: All manual position-sizing, SL/TP overrides, and frequency
forcing have been REMOVED.  The PPO agent is the sole decision maker.

The DBE now provides ONLY:
  • Market regime detection  (bull / bear / sideways / volatile)
  • Volatility sensor        (current ATR-based volatility)
  • Trend strength sensor    (ADX-based)
  • Drawdown tracker         (current vs max drawdown)
  • Capital-tier lookup      (returns the matching tier dict)

These values are injected into the observation space so the agent can
*learn* to modulate its own actions instead of having rules imposed.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DynamicBehaviorEngine:
    """Sensor-only DBE: provides market context to the RL agent without
    overriding its decisions."""

    def __init__(self, config: Dict[str, Any] = None, worker_id: int = 0, **kwargs):
        self.config = config or {}
        self.worker_id = worker_id
        self.env = None
        self.current_regime = "sideways"
        self.smart_logger = kwargs.get("smart_logger")
        self.finance_manager = kwargs.get("finance_manager")

        # Sensor state
        self.state: Dict[str, Any] = {
            "current_step": 0,
            "market_regime": "sideways",
            "regime_confidence": 0.5,
            "volatility": 0.0,
            "trend_strength": 0.0,
            "drawdown": 0.0,
            "max_drawdown": 0.0,
            "current_risk_level": 1.0,
            "win_rate": 0.0,
            "winrate": 0.0,
            "last_trade_pnl": 0.0,
            "consecutive_losses": 0,
            "position_duration": 0,
            "last_modulation": {},
            "performance_metrics": {},
        }

        # Lightweight history for regime smoothing
        self._volatility_window: List[float] = []
        self._MAX_VOL_WINDOW = 50

        # Worker tracking (lightweight)
        self.worker_states: Dict[str, Any] = {}
        self.decision_history: List[Dict] = []
        self.trade_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------
    def log_info(self, message, step=None):
        if self.smart_logger:
            try:
                self.smart_logger.smart_info(logger, message, step)
            except Exception:
                logger.info(message)
        else:
            logger.info(message)

    # ------------------------------------------------------------------
    # 1. Market regime detection  (pure sensor)
    # ------------------------------------------------------------------
    def detect_market_regime(self, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """Detect the market regime from indicator values.

        Returns (regime_str, confidence).
        """
        adx = float(market_data.get("adx", market_data.get("adx_14", 20)))
        rsi = float(market_data.get("rsi", market_data.get("rsi_14", 50)))
        ema_fast = float(market_data.get("ema_fast", 0))
        ema_slow = float(market_data.get("ema_slow", 0))
        volatility = float(market_data.get("volatility", market_data.get("atr_pct", 0)))

        adx_thr = self.config.get("market_regime_detection", {}).get("adx_threshold", 25)

        if adx > adx_thr:
            if ema_fast > ema_slow:
                regime, confidence = "bull", 0.7 + 0.3 * min(adx / 100, 1.0)
            else:
                regime, confidence = "bear", 0.7 + 0.3 * min(adx / 100, 1.0)
        elif volatility > 0.02 or rsi > 70 or rsi < 30:
            regime, confidence = "volatile", 0.8
        else:
            regime, confidence = "sideways", 0.9

        self.current_regime = regime
        self.state["market_regime"] = regime
        self.state["regime_confidence"] = confidence
        return regime, confidence

    # ------------------------------------------------------------------
    # 2. Capital tier lookup  (pure sensor)
    # ------------------------------------------------------------------
    def get_capital_tier(self, portfolio_value: float) -> Optional[Dict[str, Any]]:
        """Return the matching capital-tier dict from config."""
        tiers = self.config.get("capital_tiers", [])
        if not tiers:
            return None
        for tier in tiers:
            lo = tier.get("min_capital", 0)
            hi = tier.get("max_capital") or float("inf")
            if lo <= portfolio_value < hi:
                return tier
        return tiers[-1] if tiers else None

    # Alias kept for backward compat
    _get_capital_tier = get_capital_tier

    # ------------------------------------------------------------------
    # 3. Sensor snapshot  (injected into obs)
    # ------------------------------------------------------------------
    def get_sensor_snapshot(self, market_data: Dict[str, Any] = None,
                            portfolio_value: float = 0.0) -> Dict[str, float]:
        """Return a flat dict of sensor readings for the observation builder."""
        if market_data:
            regime, conf = self.detect_market_regime(market_data)
        else:
            regime = self.current_regime
            conf = self.state.get("regime_confidence", 0.5)

        vol = float(market_data.get("volatility", 0.0)) if market_data else 0.0
        self._volatility_window.append(vol)
        if len(self._volatility_window) > self._MAX_VOL_WINDOW:
            self._volatility_window.pop(0)

        avg_vol = float(np.mean(self._volatility_window)) if self._volatility_window else 0.0
        trend = float(market_data.get("trend_strength", market_data.get("adx_14", 0.0))) if market_data else 0.0

        tier = self.get_capital_tier(portfolio_value)
        tier_idx = 0
        if tier:
            tier_names = ["Micro Capital", "Small Capital", "Medium Capital", "High Capital", "Enterprise"]
            tier_idx = tier_names.index(tier.get("name", "Micro Capital")) if tier.get("name") in tier_names else 0

        return {
            "regime_bull": 1.0 if regime == "bull" else 0.0,
            "regime_bear": 1.0 if regime == "bear" else 0.0,
            "regime_sideways": 1.0 if regime == "sideways" else 0.0,
            "regime_volatile": 1.0 if regime == "volatile" else 0.0,
            "regime_confidence": conf,
            "volatility_current": vol,
            "volatility_avg": avg_vol,
            "trend_strength": min(trend / 100.0, 1.0),
            "drawdown": self.state.get("drawdown", 0.0),
            "tier_index_norm": tier_idx / 4.0,  # 0..1
        }

    # ------------------------------------------------------------------
    # 4. State updates  (called by env each step)
    # ------------------------------------------------------------------
    def update_state(self, live_metrics: Dict[str, Any]) -> None:
        """Update internal sensor state from environment metrics."""
        self.state["current_step"] = self.state.get("current_step", 0) + 1
        for k in ("volatility", "drawdown", "max_drawdown", "win_rate",
                   "sharpe_ratio", "sortino_ratio", "current_drawdown"):
            if k in live_metrics:
                self.state[k] = live_metrics[k]
        if "win_rate" in live_metrics:
            self.state["winrate"] = live_metrics["win_rate"]

    def set_env_reference(self, env):
        self.env = env

    # ------------------------------------------------------------------
    # 5. Minimal backward-compat API
    # ------------------------------------------------------------------
    def compute_dynamic_modulation(self, env=None, risk_horizon: float = 0.0) -> Dict[str, Any]:
        """Backward-compat stub.  Returns neutral modulation (no override)."""
        return {
            "sl_pct": 0.0,   # 0 means "let the agent decide"
            "tp_pct": 0.0,
            "position_size_pct": 0.0,
            "regime": self.current_regime,
            "regime_confidence": self.state.get("regime_confidence", 0.5),
            "reward_boost": 1.0,
            "penalty_inaction": 0.0,
        }

    def update_risk_parameters(self, market_data: Dict[str, Any],
                                portfolio_value: float) -> Dict[str, float]:
        """Backward-compat stub: returns regime info only."""
        regime, conf = self.detect_market_regime(market_data)
        return {
            "regime": regime,
            "regime_confidence": conf,
            "stop_loss_pct": 0.0,
            "take_profit_pct": 0.0,
            "position_size_pct": 0.0,
        }

    def calculate_trade_parameters(self, **kwargs) -> Dict[str, Any]:
        """Backward-compat stub: always returns feasible with zeros (agent decides)."""
        return {"feasible": True, "position_size_pct": 0.0, "sl_pct": 0.0, "tp_pct": 0.0}

    # ------------------------------------------------------------------
    # 6. Reset / lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.state = {
            "current_step": 0,
            "market_regime": "sideways",
            "regime_confidence": 0.5,
            "volatility": 0.0,
            "trend_strength": 0.0,
            "drawdown": 0.0,
            "max_drawdown": 0.0,
            "current_risk_level": 1.0,
            "win_rate": 0.0,
            "winrate": 0.0,
            "last_trade_pnl": 0.0,
            "consecutive_losses": 0,
            "position_duration": 0,
            "last_modulation": {},
            "performance_metrics": {},
        }
        self._volatility_window.clear()
        self.trade_history.clear()
        self.decision_history.clear()
        self.log_info("DBE reset (sensor-only)")

    def reset_for_new_chunk(self, continuity=True):
        if not continuity:
            self._volatility_window.clear()
            self.current_regime = "sideways"

    # Hunt stubs (backward compat)
    def start_hunt(self, *a, **kw):
        pass

    def end_hunt(self, *a, **kw):
        pass

    def is_hunting(self, worker_id: int = 0) -> bool:
        return False

    def get_hunt_info(self, worker_id: int = 0):
        return None

    # Status / properties
    @property
    def market_regime(self) -> str:
        return self.state.get("market_regime", "sideways")

    @property
    def current_step(self) -> int:
        return self.state.get("current_step", 0)

    @property
    def risk_level(self) -> float:
        return self.state.get("current_risk_level", 1.0)

    def get_status(self) -> Dict[str, Any]:
        return {
            "step": self.state.get("current_step", 0),
            "market_regime": self.current_regime,
            "volatility": self.state.get("volatility", 0.0),
            "drawdown": self.state.get("drawdown", 0.0),
        }

    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()

    def on_trade_closed(self, trade_result: Dict[str, Any]) -> None:
        self.trade_history.append(trade_result)

    def get_performance_metrics(self) -> Dict[str, Any]:
        return self.state.get("performance_metrics", {})

    def get_decision_history(self, limit=100):
        return self.decision_history[-limit:]

    def get_trade_history(self, limit=100):
        return self.trade_history[-limit:]

    def deep_update(self, d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self.deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def save_state(self, filepath) -> bool:
        """No-op: sensor-only DBE has nothing expensive to persist."""
        return True

    @classmethod
    def load_state(cls, filepath, finance_manager=None):
        return cls(finance_manager=finance_manager)

    def check_reset_conditions(self, worker_id) -> Tuple[bool, str]:
        return False, ""

    def perform_full_reset(self, worker_id, restore_capital=None):
        pass

    def reset_flow(self, worker_id) -> bool:
        return False
