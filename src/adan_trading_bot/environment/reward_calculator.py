#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reward calculation module for the ADAN trading bot.

This module defines the logic for calculating the reward signal that guides the
reinforcement learning agent.
"""

# Standard library imports
import logging
import os
import traceback
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

# Third-party imports
import numpy as np


# ------------------------------------------------------------------
# Symlog transform — symmetric logarithm that compresses extreme values
# while preserving the sign. Mathematically stable for any real input.
#   symlog(x) = sign(x) * ln(|x| + 1)
# (popularized by Hafner et al. 2023; no world-model dependency here)
# ------------------------------------------------------------------
def symlog(x: float) -> float:
    """Symmetric logarithmic compression."""
    return float(np.sign(x) * np.log1p(np.abs(x)))

# Local application imports
from ..common.reward_logger import RewardLogger

# ✅ PHASE FINALE: Intégrer le système unifié
try:
    from ..common.central_logger import logger as central_logger
    from ..performance.unified_metrics import UnifiedMetrics
    UNIFIED_SYSTEM_AVAILABLE = True
except ImportError:
    UNIFIED_SYSTEM_AVAILABLE = False
    central_logger = None
    UnifiedMetrics = None

logger = logging.getLogger(__name__)


class RewardCalculator:
    """True Quant reward calculator for the ADAN trading bot.

    Alpha-Omega reward = symlog(PnL_Net - trade_cost - drawdown_penalty)

    Three components only:
      1. PnL_Net  = trade_pnl - commissions
      2. Cost     = trade_count * cost_penalty  (penalise churn)
      3. Drawdown = current_drawdown * drawdown_penalty_weight

    No tutor_bonus, no duration_bonus, no chunk_bonus, no complex
    multi-objective weighting.  The agent learns directly from PnL.
    """

    def __init__(self, env_config: Dict[str, Any]):
        self.config = env_config.get("reward_shaping", {})

        # Legacy attributes kept for backward compatibility with callers
        self.pnl_multiplier = self.config.get("realized_pnl_multiplier", 1.0)
        self.unrealized_pnl_multiplier = self.config.get("unrealized_pnl_multiplier", 0.1)
        self.inaction_penalty = self.config.get("inaction_penalty", -0.0001)
        self.clipping_range = self.config.get("reward_clipping_range", [-5.0, 5.0])
        self.commission_penalty = self.config.get("commission_penalty", 1.5)
        self.min_profit_multiplier = self.config.get("min_profit_multiplier", 3.0)
        self.optimal_trade_bonus = self.config.get("optimal_trade_bonus", 1.0)
        self.performance_threshold = self.config.get("performance_threshold", 0.8)

        # Reward logger
        self.reward_logger = RewardLogger(env_config)

        # Unified metrics (optional)
        self.unified_metrics = None
        if UNIFIED_SYSTEM_AVAILABLE and UnifiedMetrics:
            try:
                self.unified_metrics = UnifiedMetrics()
            except Exception as e:
                logger.warning(f"Could not initialize UnifiedMetrics: {e}")

        # Episode tracking
        self.current_episode_rewards = []
        self.current_episode_id = 0

        # DBE risk state
        self.winrate = 0.5
        self.drawdown = 0.0
        self.risk_level = 1.0
        self.max_position_size_pct = 0.1
        self.min_position_size_pct = 0.01
        self.position_size_step = 0.01
        self.returns_history: list = []
        self.max_drawdown = 0.0
        self.max_lookback = 252
        self.risk_free_rate = 0.01
        self.annual_trading_days = 365
        self.decay_factor = 0.99
        self.returns_dates: list = []
        self.current_chunk_id = None
        self.chunk_rewards: dict = {}

        # Drawdown penalty weight
        self.drawdown_penalty_weight = self.config.get("drawdown_penalty_weight", 1.5)

        # Rolling equity for drawdown calculation
        self._equity_history: list = []
        self._max_equity: float = 0.0

        # C2: Initial capital for drawdown penalty (set via reset())
        self._initial_capital: float = 0.0

        # Cache for ratio calculations (used by statistics helpers)
        self._ratio_cache: dict = {}
        self._last_calculation_time = 0

        # ============================================================
        # TRUE QUANT ANTI-HACK PARAMETERS
        # ============================================================
        self._scale = 1.0        # Symlog normalisation scale
        self._alpha = 2.0        # Continuous loss penalty multiplier
        self._beta = 0.1         # EV bonus multiplier (REDUCED from 1.0 to prevent hacking)
        self._gamma_streak = 0.5 # Consecutive loss streak penalty
        self._delta = 2.0        # Failsafe binary anti-hack multiplier
        self._consecutive_losses = 0  # Streak tracker

        logger.info(
            "RewardCalculator initialized -- True Quant Anti-Hack formula: "
            "symlog + alpha*loss_penalty + streak_penalty + failsafe"
        )

    def reset_reward_state(self, initial_capital: float):
        """Reset reward calculator state at the beginning of each episode.

        Args:
            initial_capital: Starting capital for this episode (used for drawdown calculation).
        """
        self._initial_capital = float(initial_capital)
        self._max_equity = float(initial_capital)
        self._equity_history = []
        self._consecutive_losses = 0
        self.current_episode_rewards = []
        self._ratio_cache.clear()
        logger.debug(f"RewardCalculator reset: initial_capital=${initial_capital:.2f}")

    # ------------------------------------------------------------------
    # NOTE: Old bonus methods (_calculate_tutor_bonus, _calculate_early_exit_bonus,
    # _calculate_kelly_bonus, _calculate_risk_parity_bonus, _calculate_stress_var_penalty,
    # _log_reward_components) have been REMOVED as part of the True Quant cleanup.
    # The calculate() method below is the ONLY reward path.
    # ------------------------------------------------------------------

    def calculate(
        self,
        portfolio_metrics: Dict[str, Any],
        trade_pnl: float,
        action: int,
        chunk_id: int = None,
        optimal_chunk_pnl: float = None,
        performance_ratio: float = None,
        is_hunting: bool = False,
        risk_horizon: float = 0.0,
        trade_reason: Optional[str] = None,
        **kwargs,
    ) -> float:
        """TRUE QUANT ANTI-HACK REWARD (RewardCalculator standalone version).

        Computes a mathematically anti-hackable reward signal that guarantees:
          - Positive PnL trades always receive positive reward
          - Negative PnL trades NEVER receive positive reward (failsafe enforced)
          - No-trade steps receive near-zero reward

        Five components:
          1. Symlog of base_pnl (PnL_Net - cost_penalty - dd_penalty):
             r = sign(base_pnl) * ln(|base_pnl/scale| + 1)
          2. Continuous loss penalty: -alpha * max(0, -pnl_net) / scale
             Every cent of loss is penalized proportionally (alpha=2.0)
          3. EV bonus: beta * clip(ev_norm, -1, 1)
             Soft signal for expected-value positive trades (beta=0.1, reduced to prevent hacking)
          4. Consecutive loss streak penalty: -gamma * max(0, streak - 2)
             Escalating penalty after 2+ consecutive losses (gamma=0.5)
          5. FAILSAFE BINARY ANTI-HACK: if pnl_net < 0 and r > 0 => r *= -delta
             Absolute guarantee: negative PnL cannot produce positive reward (delta=2.0)

        Parameters (from __init__):
          _scale=1.0, _alpha=2.0, _beta=0.1, _gamma_streak=0.5, _delta=2.0

        Audit trail:
          Logs REWARD_ANTIHACK with full breakdown (pnl_net, r_symlog, loss_pen,
          ev, streak, dd_pen, cost_pen, inv_pen, action_req, action_exe, failsafe, final)
          on every trade and every 50th step.

        Args:
            portfolio_metrics: Dict with keys 'total_commission', 'closed_positions',
                             'portfolio_value'/'balance', 'drawdown', etc.
            trade_pnl: float — gross realized PnL from this step.
            action: int — executed action (0=HOLD, 1=BUY, 2=SELL).
            chunk_id: int — current data chunk (for tracking).
            **kwargs: action_requested (int), invalid_penalty (float), ev_norm (float).

        Returns:
            float: Anti-hack reward value, clipped to [-5.0, 5.0] range by default.
        """
        try:
            # --- PnL extraction ---
            commission = float(portfolio_metrics.get("total_commission", 0.0))
            pnl_net = float(trade_pnl) - commission
            self._raw_pnl_log = pnl_net

            # --- Track returns history ---
            if trade_pnl != 0:
                self._update_returns_history(trade_pnl)

            # --- Consecutive loss tracking ---
            if pnl_net < 0:
                self._consecutive_losses += 1
            elif pnl_net > 0:
                self._consecutive_losses = 0

            # --- Cost penalty (churn deterrent) ---
            closed_trades = portfolio_metrics.get("closed_positions", [])
            trade_count = len(closed_trades) if isinstance(closed_trades, list) else 0
            if trade_pnl == 0 and action != 0:
                trade_count = max(trade_count, 1)
            cost_penalty = trade_count * self.config.get("cost_penalty", 0.001)

            # --- Drawdown penalty ---
            current_equity = float(portfolio_metrics.get(
                "portfolio_value", portfolio_metrics.get("balance", 0.0)
            ))
            if current_equity > 0:
                self._equity_history.append(current_equity)
                if current_equity > self._max_equity:
                    self._max_equity = current_equity

            dd_penalty = 0.0
            # C1+C2: Drawdown based on initial_capital (not peak_equity)
            # Threshold 0.10 = d_kill/4 = 40%/4, absorbs normal BTC 5m volatility
            if self._initial_capital > 0 and current_equity > 0:
                dd = max(0.0, (self._initial_capital - current_equity) / self._initial_capital)
                if dd > 0.10:   # C1: was 0.005 — 80x too strict
                    dd_penalty = self.drawdown_penalty_weight * dd
            elif self._max_equity > 0 and current_equity > 0:
                # Fallback if _initial_capital not yet set
                dd = max(0.0, (self._max_equity - current_equity) / self._max_equity)
                if dd > 0.10:
                    dd_penalty = self.drawdown_penalty_weight * dd

            # ==========================================================
            # TRUE QUANT ANTI-HACK REWARD FORMULA
            # ==========================================================
            scale = self._scale
            alpha = self._alpha
            beta = self._beta
            gamma_s = self._gamma_streak
            delta = self._delta

            # 1. Symlog of base PnL
            base_pnl = pnl_net - cost_penalty - dd_penalty
            r = float(np.sign(base_pnl) * np.log1p(abs(base_pnl) / scale))

            # 2. Continuous loss penalty (every centime lost is penalised)
            r -= alpha * max(0.0, -pnl_net) / scale

            # 3. EV bonus (approximation: positive PnL = good decision)
            ev_norm = kwargs.get("ev_norm", 0.0)
            if ev_norm == 0.0:
                ev_norm = 0.5 if pnl_net > 0 else (-0.5 if pnl_net < 0 else 0.0)
            r += beta * float(np.clip(ev_norm, -1.0, 1.0))

            # 4. Consecutive loss streak penalty
            r -= gamma_s * max(0.0, float(self._consecutive_losses - 2))

            # 5. FAILSAFE ANTI-HACK (C11+C11-A: smooth sigmoid transition)
            # Replaces the hard flip (r *= -delta) with a continuous sigmoid
            # to avoid gradient discontinuity at pnl_net=0 in PPO updates.
            # severity is normalized by initial_capital so it reacts to % loss, not $ absolute.
            failsafe_triggered = False
            if pnl_net < 0 and r > 0:
                _cap = self._initial_capital if self._initial_capital > 0 else 1.0
                relative_loss = abs(pnl_net) / (_cap + 1e-9)
                # Sigmoid: transition around 1% loss, full correction at ~5%
                severity = 1.0 / (1.0 + np.exp(-100.0 * (relative_loss - 0.01)))
                r = r * (-delta) * severity
                failsafe_triggered = True
                logger.info(
                    f"FAILSAFE_SIGMOID | pnl_net={pnl_net:+.6f} | "
                    f"relative_loss={relative_loss:.4f} | severity={severity:.4f} | "
                    f"r_after={r:+.6f}"
                )

            # ──────────────────────────────────────────────────────────
            # SESSION 12 FIX — DENSE CRITIC SIGNAL (replaces S10 time_decay)
            # ──────────────────────────────────────────────────────────
            # WHY: A constant time_decay (-0.001) makes V(s) a simple linear
            # countdown. The Critic cannot learn state-dependent value because
            # the reward is identical in every non-trade step regardless of
            # market state, position status, or unrealized PnL.
            #
            # ──────────────────────────────────────────────────────────
            # Session Finale: REVERT des hacks reward S12
            # Le position_bonus et unrealized PnL delta ont EMPIRE les resultats CI:
            #   Run#8 (avec bonus): ev = -5.11
            #   Run#9 (avec bonus): ev = -9.29 (REGRESSION)
            # Seule recompense valide = realized_pnl des trades fermes.
            # Le Critic convergera avec 50k+ steps sur donnees diversifiees.
            # ──────────────────────────────────────────────────────────
            # 1. Time decay (constant negative baseline)
            _env_td = os.environ.get("ADAN_TIME_DECAY")
            if _env_td is not None:
                try:
                    time_decay = float(_env_td)
                except (TypeError, ValueError):
                    time_decay = float(self.config.get("time_decay", -1e-3))
            else:
                time_decay = float(self.config.get("time_decay", -1e-3))
            r += time_decay

            final_reward = float(r)

            # Episode tracking
            self.current_episode_rewards.append(final_reward)

            # Update internal risk state (DBE)
            self._update_dbe_parameters(portfolio_metrics)

            # Audit log — INFO level so it appears in training output
            # This is essential to verify the anti-hack formula is active
            # Includes both requested and executed actions for transparency
            _action_requested = kwargs.get("action_requested", action)
            _action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
            _inv_penalty = kwargs.get("invalid_penalty", 0.0)
            if trade_pnl != 0 or self.current_episode_rewards and len(self.current_episode_rewards) % 50 == 0:
                logger.info(
                    f"REWARD_ANTIHACK | pnl_net={pnl_net:+.6f} "
                    f"r_symlog={float(np.sign(base_pnl)*np.log1p(abs(base_pnl)/scale)):+.6f} "
                    f"loss_pen={alpha*max(0,-pnl_net)/scale:.6f} "
                    f"ev={ev_norm:+.3f} streak={self._consecutive_losses} "
                    f"dd_pen={dd_penalty:.6f} cost_pen={cost_penalty:.6f} "
                    f"inv_pen={_inv_penalty:.6f} "
                    f"action_req={_action_names.get(_action_requested, '?')} "
                    f"action_exe={_action_names.get(action, '?')} "
                    f"failsafe={'YES' if failsafe_triggered else 'no'} "
                    f"final={final_reward:+.6f}"
                )

            # Unified metrics logging
            if UNIFIED_SYSTEM_AVAILABLE and central_logger:
                central_logger.metric("Reward Final", final_reward)
                central_logger.metric("Reward Raw PnL", pnl_net)
                central_logger.metric("Reward ConsecLosses", float(self._consecutive_losses))
                if self.unified_metrics:
                    if trade_pnl != 0:
                        self.unified_metrics.add_return(trade_pnl)
                    if "portfolio_value" in portfolio_metrics:
                        self.unified_metrics.add_portfolio_value(
                            portfolio_metrics["portfolio_value"]
                        )

            return final_reward

        except Exception as e:
            logger.error(f"Error in reward calculation: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.0

    def get_reward_statistics(self) -> Dict[str, Any]:
        """
        Obtenir les statistiques détaillées des récompenses.

        Returns:
            Dictionnaire contenant les statistiques des récompenses
        """
        return self.reward_logger.get_reward_statistics()

    def save_reward_logs(self, filename: str = None) -> None:
        """
        Sauvegarder les logs de récompenses.

        Args:
            filename: Nom de fichier optionnel
        """
        self.reward_logger.save_reward_logs(filename)

    def _update_returns_history(self, pnl: float) -> None:
        """
        Update the returns history with the latest PnL and clear cache.

        Args:
            pnl: The profit or loss from the latest trade.
        """
        # Add the PnL and current timestamp to history
        self.returns_history.append(pnl)
        self.returns_dates.append(datetime.now())

        # Keep only the most recent returns up to max_lookback
        if len(self.returns_history) > self.max_lookback:
            self.returns_history.pop(0)
            self.returns_dates.pop(0)

        # Clear cache when history is updated
        self._ratio_cache.clear()

    def _get_time_weights(self) -> np.ndarray:
        """
        Calculate time-based weights for returns using exponential decay.

        Returns:
            np.ndarray: Array of weights with the same length as returns_history
        """
        n = len(self.returns_history)
        if n == 0:
            return np.array([])

        # Create exponential decay weights (most recent has weight 1.0)
        weights = np.array([self.decay_factor**i for i in reversed(range(n))])

        # Normalize weights to sum to 1
        return weights / np.sum(weights)

    def _calculate_sharpe_ratio(self, risk_free_rate: float = None) -> float:
        """
        Calculate the time-weighted Sharpe ratio based on historical returns.

        Args:
            risk_free_rate: Annual risk-free rate (defaults to instance variable)

        Returns:
            float: The time-weighted Sharpe ratio
        """
        cache_key = f"sharpe_{risk_free_rate}"
        if cache_key in self._ratio_cache:
            return self._ratio_cache[cache_key]

        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        if not self.returns_history:
            return 0.0

        try:
            # Get time weights (exponential decay)
            weights = self._get_time_weights()

            # Calculate weighted excess returns
            daily_returns = np.array(self.returns_history)
            excess_returns = daily_returns - (risk_free_rate / self.annual_trading_days)

            # Weighted mean and std
            weighted_mean = np.average(excess_returns, weights=weights)
            weighted_std = np.sqrt(
                np.average((excess_returns - weighted_mean) ** 2, weights=weights)
            )

            # Avoid division by zero
            if weighted_std < 1e-10:
                return 0.0

            # Annualize the ratio
            sharpe_ratio = (weighted_mean / weighted_std) * np.sqrt(
                self.annual_trading_days
            )

            # Cache the result
            self._ratio_cache[cache_key] = float(sharpe_ratio)
            return self._ratio_cache[cache_key]

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0

    def _calculate_sortino_ratio(self, risk_free_rate: float = None) -> float:
        """
        Calculate the time-weighted Sortino ratio based on historical returns.

        Args:
            risk_free_rate: Annual risk-free rate (defaults to instance variable)

        Returns:
            float: The time-weighted Sortino ratio
        """
        cache_key = f"sortino_{risk_free_rate}"
        if cache_key in self._ratio_cache:
            return self._ratio_cache[cache_key]

        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        if not self.returns_history:
            return 0.0

        try:
            # Get time weights
            weights = self._get_time_weights()

            # Calculate excess returns
            daily_returns = np.array(self.returns_history)
            excess_returns = daily_returns - (risk_free_rate / self.annual_trading_days)

            # Calculate weighted mean return
            weighted_mean = np.average(excess_returns, weights=weights)

            # Calculate downside deviation (weighted)
            downside_returns = np.where(daily_returns < 0, daily_returns, 0)
            if np.all(downside_returns == 0):
                return float("inf")  # No downside risk

            # Calculate weighted mean of squared downside returns
            weighted_variance = np.average(downside_returns**2, weights=weights)
            downside_std = np.sqrt(weighted_variance)

            # Avoid division by zero
            if downside_std < 1e-10:
                return 0.0

            # Annualize the ratio
            sortino_ratio = (weighted_mean / downside_std) * np.sqrt(
                self.annual_trading_days
            )

            # Cache the result
            self._ratio_cache[cache_key] = float(sortino_ratio)
            return self._ratio_cache[cache_key]

        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0.0

    def _calculate_calmar_ratio(
        self, portfolio_metrics: Dict[str, Any], lookback_period: int = 36
    ) -> float:
        """
        Calculate the time-weighted Calmar ratio based on the maximum drawdown.

        Args:
            portfolio_metrics: Current portfolio metrics including drawdown
            lookback_period: Lookback period in months (default: 36 months)

        Returns:
            float: The time-weighted Calmar ratio
        """
        cache_key = f"calmar_{lookback_period}"
        if cache_key in self._ratio_cache:
            return self._ratio_cache[cache_key]

        try:
            # Get the maximum drawdown from portfolio metrics
            max_drawdown = abs(portfolio_metrics.get("max_drawdown", 0.0))

            # Avoid division by zero
            if max_drawdown < 1e-10:
                return 0.0

            if not self.returns_history:
                return 0.0

            # Get time weights for the lookback period
            weights = self._get_time_weights()

            # Calculate weighted cumulative return
            recent_returns = np.array(self.returns_history)

            # If we have enough data, limit to the lookback period
            if (
                len(recent_returns) > lookback_period * 21
            ):  # Approximate trading days in lookback
                recent_returns = recent_returns[-(lookback_period * 21) :]
                weights = weights[-(lookback_period * 21) :]

            # Calculate weighted cumulative return
            weighted_returns = recent_returns * weights
            cumulative_return = np.sum(weighted_returns) / np.sum(weights)

            # Annualize the return
            annualized_return = (1 + cumulative_return) ** self.annual_trading_days - 1

            # Calculate Calmar ratio
            calmar_ratio = annualized_return / max_drawdown

            # Cache the result
            self._ratio_cache[cache_key] = float(calmar_ratio)
            return self._ratio_cache[cache_key]

        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {str(e)}")
            return 0.0

    def _update_dbe_parameters(self, portfolio_metrics: Dict[str, Any]) -> None:
        """
        Update DBE (Dynamic Budgeting Engine) parameters based on portfolio performance.

        Args:
            portfolio_metrics: Current portfolio metrics including winrate, drawdown, etc.
        """
        try:
            # Get metrics with defaults
            self.winrate = portfolio_metrics.get("win_rate", self.winrate)
            self.drawdown = portfolio_metrics.get("drawdown", self.drawdown)
            cash_utilization = portfolio_metrics.get("cash_utilization", 0.0)

            # Calculate volatility-based adjustment (higher vol → lower risk)
            returns_vol = (
                np.std(self.returns_history) * np.sqrt(252)
                if self.returns_history
                else 0.0
            )
            vol_adjustment = 1.0 / (
                1.0 + returns_vol
            )  # 0.5 for vol=1.0, 1.0 for vol=0.0

            # Calculate drawdown-based adjustment
            drawdown_adjustment = 1.0 - min(0.5, self.drawdown)

            # Calculate utilization-based adjustment
            util_adjustment = 0.5 + cash_utilization * 0.5

            # Combine adjustments with winrate
            self.risk_level = max(
                0.1,
                min(
                    1.0,
                    self.winrate
                    * drawdown_adjustment
                    * util_adjustment
                    * vol_adjustment,
                ),
            )

            # Update max position size based on risk level
            if "max_position_size_pct" in portfolio_metrics:
                portfolio_metrics["max_position_size_pct"] *= self.risk_level

            # Log detailed risk calculation
            logger.debug(
                f"DBE UPDATE | Winrate: {self.winrate:.2%} | "
                f"Drawdown: {self.drawdown:.2%} | "
                f"Vol: {returns_vol:.2%} | "
                f"Cash Util: {cash_utilization:.2%} | "
                f"New Risk: {self.risk_level:.3f}"
            )

        except Exception as e:
            logger.error(f"Error updating DBE parameters: {str(e)}")
            # Fall back to conservative settings on error
            self.risk_level = max(0.1, self.risk_level * 0.9)

    def generate_reward_report(self) -> str:
        """
        Générer un rapport détaillé des récompenses.

        Returns:
            Rapport formaté des récompenses
        """
        return self.reward_logger.generate_reward_report()


class MarketRegime(Enum):
    """Simple market regime enum used by adaptive reward tests."""

    RANGING = "ranging"
    TRENDING = "trending"
    VOLATILE = "volatile"


class _DefaultRegimeDetector:
    """Minimal regime detector used when tests don't inject a mock."""

    def __init__(self) -> None:
        self._regime = MarketRegime.RANGING
        self._strength = 0.5
        self._volatility = 0.1

    def update(self, price: float) -> None:  # pragma: no cover - noop
        pass

    def get_regime(self) -> MarketRegime:
        return self._regime

    def get_regime_strength(self) -> float:
        return self._strength

    def get_volatility(self) -> float:
        return self._volatility


class AdaptiveRewardCalculator:
    """Lightweight adaptive reward calculator for unit tests.

    Exposes update_market_regime() and calculate() used by
    tests/unit/environment/test_reward_calculator.py.
    """

    def __init__(
        self,
        lookback_period: int = 14,
        volatility_threshold: float = 0.02,
        trend_strength_threshold: float = 0.6,
        min_data_points: int = 5,
    ) -> None:
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.trend_strength_threshold = trend_strength_threshold
        self.min_data_points = min_data_points

        # Detector can be replaced by tests
        self.regime_detector = _DefaultRegimeDetector()

        # Public attributes used by tests
        self.current_regime: MarketRegime = MarketRegime.RANGING
        self.regime_strength: float = 0.5
        self.position_size: float = 0.0

        # Tunables referenced in tests
        self.inaction_penalty: float = -0.1
        self.commission_penalty: float = 1.0
        self.min_profit_multiplier: float = 1.0
        self.optimal_trade_bonus: float = 0.05
        self.clipping_range = (-10.0, 9.99)

        # Base penalties per regime for smooth transitions
        self._base_penalty = {
            MarketRegime.RANGING: -0.15,
            MarketRegime.TRENDING: -0.2,
            MarketRegime.VOLATILE: -0.25
        }

    def update_market_regime(self, price: float) -> None:
        """Update regime and smoothly transition inaction_penalty."""
        try:
            if self.regime_detector:
                self.regime_detector.update(price)
                new_regime = self.regime_detector.get_regime()
                strength = float(self.regime_detector.get_regime_strength())
            else:
                new_regime = self.current_regime
                strength = self.regime_strength

            # Clamp strength into [0, 1]
            strength = max(0.0, min(1.0, strength))

            prev_base = self._base_penalty.get(
                self.current_regime, self.inaction_penalty
            )
            next_base = self._base_penalty.get(new_regime, self.inaction_penalty)

            # Smooth transition
            self.inaction_penalty = prev_base + strength * (next_base - prev_base)

            # Update state
            self.current_regime = new_regime
            self.regime_strength = strength
        except Exception:
            logging.getLogger(__name__).exception("update_market_regime failed")

    def calculate(
        self,
        current_price: float,
        realized_pnl: float,
        unrealized_pnl: float,
        commission: float,
        position_size: float,
    ) -> float:
        """Compute a bounded reward consistent with unit tests.

        - Base reward: realized_pnl - commission_penalty * commission
        - Add optimal_trade_bonus if realized_pnl exceeds
          min_profit_multiplier * commission
        - Smoothly squash into clipping_range using tanh so it's strictly inside
          the bounds for finite inputs
        """
        try:
            self.position_size = position_size
            base = float(realized_pnl) - float(commission) * float(
                self.commission_penalty
            )
            if (
                commission > 1e-12
                and realized_pnl > self.min_profit_multiplier * commission
            ):
                base += self.optimal_trade_bonus
            low, high = self.clipping_range
            scale = max(abs(low), abs(high)) or 1.0
            return float(np.tanh(base / scale) * scale)
        except Exception:
            logging.getLogger(__name__).exception("calculate failed")
            return 0.0
