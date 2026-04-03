#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Realistic Trading Environment for ADAN 2.0.
Inherits from MultiAssetChunkedEnv and integrates:
1. TradeFrequencyController (Cooldown, Daily Limits)
2. StableRewardCalculator (Normalized rewards)
3. Circuit Breaker (Risk management)
4. Min Notional (Binance limits)
5. Candle Sync (Live mode alignment)
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple

from .multi_asset_chunked_env import MultiAssetChunkedEnv
from .trade_frequency_controller import TradeFrequencyController, FrequencyConfig
from .stable_reward_calculator import StableRewardCalculator
from .market_friction import (
    AdaptiveSlippage,
    LatencySimulator,
    LiquidityModel,
    BinanceFeeModel,
    MarketConditions,
    StaleDataSimulator
)

# ✅ JOUR 2: Importer le système unifié
try:
    from adan_trading_bot.common.central_logger import logger as central_logger
    from adan_trading_bot.performance.unified_metrics import UnifiedMetrics
    UNIFIED_SYSTEM_AVAILABLE = True
except ImportError:
    UNIFIED_SYSTEM_AVAILABLE = False
    central_logger = None
    UnifiedMetrics = None

# ✅ PHASE FINALE: Importer le RiskManager robuste
try:
    from ..risk_management.risk_manager import RiskManager
    RISK_MANAGER_AVAILABLE = True
except ImportError:
    RISK_MANAGER_AVAILABLE = False
    RiskManager = None


class RealisticTradingEnv(MultiAssetChunkedEnv):
    """
    Enhanced environment with modular constraints for realistic trading.
    Per ADAN 2.0 Spec Phase 2.
    """
    
    def __init__(
        self,
        live_mode: bool = False,
        min_hold_steps: int = 6,  # 30 mins (6 * 5m)
        cooldown_steps: int = 3,  # 15 mins (3 * 5m)
        min_notional: float = 10.0,  # $10 USDT
        circuit_breaker_pct: float = 0.15,  # 15% max drawdown
        daily_trade_limit: int = 10,
        use_stable_reward: bool = True,  # Toggle for new reward calculator
        enable_market_friction: bool = True,  # Toggle for friction models
        reward_config: Optional[Dict[str, float]] = None,
        friction_config: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.live_mode = live_mode
        self.min_hold_steps = min_hold_steps
        self.min_notional = min_notional
        self.circuit_breaker_pct = circuit_breaker_pct
        self.use_stable_reward = use_stable_reward
        
        # Initialize TradeFrequencyController
        freq_config = FrequencyConfig(
            min_interval_steps=1,  # Allow 1 step between trades globally
            daily_trade_limit=daily_trade_limit,
            asset_cooldown_steps=cooldown_steps,
            force_trade_steps_by_tf=self.config.get("trading_rules", {}).get("force_trade_steps_by_timeframe", {
                "5m": 15,
                "1h": 20,
                "4h": 50
            })
        )
        self.freq_controller = TradeFrequencyController(freq_config)
        
        # Initialize StableRewardCalculator
        if self.use_stable_reward:
            r_cfg = reward_config or {}
            self.reward_calculator = StableRewardCalculator(
                pnl_normalization_factor=r_cfg.get("pnl_normalization", 100.0),
                sharpe_weight=r_cfg.get("sharpe_weight", 0.2),
                drawdown_penalty_weight=r_cfg.get("drawdown_weight", 0.3),
                frequency_penalty_weight=r_cfg.get("frequency_weight", 0.1),
                consistency_bonus_weight=r_cfg.get("consistency_weight", 0.1)
            )
        
        # Circuit breaker state
        self.circuit_breaker_triggered = False
        
        # Hold minimum tracking (per-asset entry steps)
        self.asset_entry_steps: Dict[str, int] = {}
        
        # Optuna Overrides (Priority over DBE)
        self.optuna_sl_override = kwargs.get("stop_loss_pct")
        self.optuna_tp_override = kwargs.get("take_profit_pct")
        
        if self.optuna_sl_override:
            self.logger.info(f"🔒 OPTUNA OVERRIDE: Stop Loss fixed at {self.optuna_sl_override:.2%}")
        if self.optuna_tp_override:
            self.logger.info(f"🔒 OPTUNA OVERRIDE: Take Profit fixed at {self.optuna_tp_override:.2%}")

        # Initialize Market Friction Models
        self.enable_market_friction = enable_market_friction
        if self.enable_market_friction:
            f_cfg = friction_config or {}
            self.slippage_model = AdaptiveSlippage(
                base_slippage_bps=f_cfg.get("slippage_bps", 2.0) / 100.0,
                size_impact_factor=0.1,
                volatility_impact_factor=0.5
            )
            self.latency_model = LatencySimulator(
                min_latency_ms=50.0,
                max_latency_ms=200.0,
                price_drift_per_ms=0.00001
            )
            self.liquidity_model = LiquidityModel(
                depth_factor=0.001,
                impact_exponent=1.5
            )
            # Stale Data Simulation (Robustness)
            # Only active if explicitly configured or in training mode (not live)
            stale_prob = f_cfg.get("stale_data_prob", 0.0)
            if not self.live_mode and stale_prob == 0.0:
                 stale_prob = 0.05 # Default 5% stale data in training for robustness
            
            self.stale_data_simulator = StaleDataSimulator(
                prob_stale=stale_prob,
                max_lag_steps=3
            )
            self.fee_model = BinanceFeeModel(
                maker_fee=f_cfg.get("fee_bps", 0.04) / 100.0,
                taker_fee=f_cfg.get("fee_bps", 0.04) / 100.0,
                tier="VIP0",
                use_bnb_discount=False
            )
        
        # ✅ PHASE FINALE: Initialize RiskManager
        self._initialize_risk_manager(circuit_breaker_pct)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🛡️ RealisticTradingEnv initialized (Live={live_mode}, StableReward={use_stable_reward}, Friction={enable_market_friction})")
        self.logger.info(f"   Hold={min_hold_steps} steps, Cooldown={cooldown_steps} steps, DailyLimit={daily_trade_limit}")
        self.logger.info(f"   Min Notional=${min_notional}, Circuit Breaker={circuit_breaker_pct:.1%}")
        self.logger.info(f"   Risk Manager: {'RiskManager' if self.risk_manager else 'Circuit Breaker'}")

    def reset(self, **kwargs):
        """Reset environment and all controllers."""
        obs = super().reset(**kwargs)
        
        # Reset controllers
        self.freq_controller.reset()
        if self.use_stable_reward:
            self.reward_calculator.reset()
            # Dynamically adjust PnL normalization based on initial capital
            # Target: 1% gain ~= 1.0 normalized reward (before tanh)
            initial_cap = self.portfolio_manager.initial_capital
            if initial_cap > 0:
                # e.g. $20 -> factor 0.2; $1000 -> factor 10.0
                dynamic_factor = max(initial_cap * 0.01, 1.0)
                self.reward_calculator.update_normalization_factor(dynamic_factor)
        
        # Reset state tracking
        self.circuit_breaker_triggered = False
        self.asset_entry_steps.clear()
        
        return obs

    def _initialize_risk_manager(self, circuit_breaker_pct: float):
        """Initialize risk management system."""
        self.risk_manager = None
        if RISK_MANAGER_AVAILABLE and RiskManager:
            try:
                # Configuration pour le RiskManager
                risk_config = {
                    'max_daily_drawdown': circuit_breaker_pct,  # Utiliser le circuit breaker comme max drawdown
                    'max_position_risk': 0.02,  # 2% par position
                    'max_portfolio_risk': 0.10,  # 10% du portefeuille
                    'initial_capital': self.config.get('portfolio', {}).get('initial_balance', 10000)
                }
                self.risk_manager = RiskManager(risk_config)
                self.logger.info("✅ RiskManager robuste initialisé")
                
                # Logger avec le système unifié
                if UNIFIED_SYSTEM_AVAILABLE and central_logger:
                    central_logger.sync(
                        component="RiskManager",
                        status="initialized",
                        details=risk_config
                    )
            except Exception as e:
                self.logger.warning(f"Could not initialize RiskManager: {e}")
                self.risk_manager = None
        else:
            self.logger.warning("RiskManager not available, using fallback circuit breaker")

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Override step to enforce global constraints and use StableRewardCalculator.
        """
        # 0. Sanity Checks (Phase 2)
        # Validate market data before processing anything
        if not self._validate_market_data():
            self.logger.warning("⚠️ SANITY CHECK FAILED: Market data anomaly detected. Skipping step.")
            return self._get_observation(), 0.0, False, False, {"sanity_check_failed": True}

        # 1. Circuit Breaker Check (Phase 2)
        if self.circuit_breaker_triggered:
            self.logger.warning("🚨 CIRCUIT BREAKER ACTIVE - Trading halted.")
            return self._get_observation(), -1.0, True, False, {"circuit_breaker": True}
            
        # Check dynamic circuit breakers (Drawdown, etc.)
        # Check dynamic circuit breakers (Drawdown, etc.)
        if self._check_circuit_breakers():
            self.circuit_breaker_triggered = True
            self.logger.critical("🚨 CIRCUIT BREAKER TRIGGERED: Max Drawdown or Risk Limit exceeded! Closing all positions.")
            self.portfolio_manager.close_all_positions(self.current_step, reason="CIRCUIT_BREAKER")
            return self._get_observation(), -10.0, True, False, {"circuit_breaker_triggered": True}

        # Proceed with normal step execution
        obs, reward, terminated, truncated, info = super().step(action)
        
        # ------------------------------------------------------------------
        # PARETO RISK DETECTOR UPDATE (Phase 2: Security & Robustness)
        # ------------------------------------------------------------------
        # Update Pareto detector with portfolio return for regime detection
        if hasattr(self, 'order_manager') and self.order_manager.pareto_detector is not None:
            # Calculate step return: (current_value - previous_value) / previous_value
            current_value = self.portfolio_manager.get_total_value()
            if not hasattr(self, '_previous_portfolio_value'):
                self._previous_portfolio_value = current_value
            
            if self._previous_portfolio_value > 0:
                portfolio_return = (current_value - self._previous_portfolio_value) / self._previous_portfolio_value
                self.order_manager.pareto_detector.update(portfolio_return)
            
            self._previous_portfolio_value = current_value

        # CRITICAL: Enforce Optuna Overrides on Risk Parameters
        # This must happen AFTER DBE updates but BEFORE any trading logic uses them
        if hasattr(self, 'portfolio_manager'):
            if self.optuna_sl_override is not None:
                self.portfolio_manager.sl_pct = self.optuna_sl_override
            if self.optuna_tp_override is not None:
                self.portfolio_manager.tp_pct = self.optuna_tp_override
        
        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, action: np.ndarray, realized_pnl: float) -> float:
        """
        Override to use StableRewardCalculator.
        """
        if not self.use_stable_reward:
            return super()._calculate_reward(action, realized_pnl)
            
        # Gather metrics
        current_value = self.portfolio_manager.get_total_value()
        initial_value = self.portfolio_manager.initial_capital
        
        # Get trade count for this step
        trade_count = 0
        if hasattr(self, '_step_info') and isinstance(self._step_info, dict):
            trade_count = self._step_info.get('trades_executed', 0)
            
        # Get invalid sell attempts
        invalid_sells = getattr(self, 'invalid_sell_attempts', 0)
        
        # Calculate reward
        reward_dict = self.reward_calculator.calculate_reward(
            pnl=realized_pnl,
            portfolio_value=current_value,
            initial_value=initial_value,
            trade_count=trade_count,
            invalid_sell_attempts=invalid_sells
        )
        
        # Store breakdown in info for debugging
        if hasattr(self, '_step_info') and isinstance(self._step_info, dict):
            self._step_info['reward_breakdown'] = reward_dict
            
        return reward_dict['total_reward']

    def _validate_market_data(self, current_prices: Dict[str, float] = None) -> bool:
        """
        Phase 2: Sanity Checks / Anti-Adversarial Filters
        Returns True if data is valid, False if anomaly detected.
        """
        if current_prices is None:
            current_prices = self._get_current_prices()
        
        if not current_prices:
            # Only log if we expected prices but got none (avoid spamming if env is resetting)
            if hasattr(self, "current_step") and self.current_step > 0:
                self.logger.warning("⚠️ SANITY CHECK FAILED: No prices returned from _get_current_prices")
            return False
            
        for asset, price in current_prices.items():
            if price is None:
                self.logger.warning(f"⚠️ SANITY CHECK FAILED: Price for {asset} is None")
                return False
            if price <= 0:
                self.logger.warning(f"⚠️ SANITY CHECK FAILED: Price for {asset} is invalid ({price})")
                return False
            
            # Check for extreme price jumps (> 10% in 1 step) if we have history
            # (Simplified check, ideally would compare to previous step's price)
            # This requires tracking previous prices, which we can add later if needed.
            
        return True

    def _check_circuit_breakers(self) -> bool:
        """
        Phase 2: Validation robuste avec RiskManager ou fallback
        Returns True if trading should stop immediately.
        """
        initial_capital = self.portfolio_manager.initial_capital
        total_value = self.portfolio_manager.get_total_value()
        
        # ✅ PHASE FINALE: Utiliser RiskManager robuste si disponible
        if self.risk_manager:
            try:
                # Mettre à jour les peaks
                self.risk_manager.update_peak(total_value)
                
                # Vérifier le drawdown avec le RiskManager
                current_drawdown = (self.risk_manager.portfolio_peak - total_value) / self.risk_manager.portfolio_peak
                
                if current_drawdown > self.risk_manager.max_daily_drawdown:
                    # Logger avec le système unifié
                    if UNIFIED_SYSTEM_AVAILABLE and central_logger:
                        central_logger.validation(
                            "Risk Management",
                            False,
                            f"Drawdown {current_drawdown:.2%} > Max {self.risk_manager.max_daily_drawdown:.2%}"
                        )
                    
                    self.logger.critical(
                        f"🛡️  RISK MANAGER: Drawdown {current_drawdown:.2%} exceeds limit {self.risk_manager.max_daily_drawdown:.2%}"
                    )
                    return True
                
                return False
                
            except Exception as e:
                self.logger.error(f"RiskManager error, falling back to circuit breaker: {e}")
                # Fallback au circuit breaker simple
        
        # Fallback: Circuit breaker simple
        if total_value < initial_capital * (1 - self.circuit_breaker_pct):
            # Logger avec le système unifié
            if UNIFIED_SYSTEM_AVAILABLE and central_logger:
                central_logger.validation(
                    "Circuit Breaker",
                    False,
                    f"Portfolio {total_value:.2f} < {(1-self.circuit_breaker_pct)*100:.0f}% of initial {initial_capital:.2f}"
                )
            
            self.logger.critical(
                f"💀 CIRCUIT BREAKER: Total Value ({total_value:.2f}) < {(1-self.circuit_breaker_pct)*100:.0f}% of initial ({initial_capital:.2f})"
            )
            return True
        
        return False

    def _check_new_day(self):
        """
        Override to ensure freq_controller is also reset on new day.
        """
        # Call parent to reset positions_count and handle logging
        super()._check_new_day()
        
        # Reset freq_controller daily counts
        if hasattr(self, 'freq_controller'):
            self.freq_controller.reset_daily()
            self.logger.info(f"[NEW_DAY] Reset freq_controller daily counts for Worker {self.worker_id}")

    def _sync_to_candle(self) -> None:
        """Synchronize to nearest candle boundary in live mode."""
        if not self.live_mode:
            return
        # Implementation would sync with real-time candle data
        pass

    def _execute_trades(
        self,
        action: np.ndarray,
        dbe_modulation: dict,
        action_threshold: float,
        force_trade: bool = False,
    ) -> tuple[float, int]:
        """
        Override _execute_trades to integrate TradeFrequencyController and enforce constraints.
        """
        if not hasattr(self, "portfolio_manager"):
            return 0.0, 0, 0

        current_prices = self._get_current_prices()
        if not current_prices:
            return 0.0, 0, 0

        # Update positions first (mark to market)
        pnl_from_update, sl_tp_receipts = self.portfolio_manager.update_market_price(
            current_prices, self.current_step
        )
        if sl_tp_receipts:
            self._step_closed_receipts.extend(sl_tp_receipts)
        
        realized_pnl = pnl_from_update
        first_discrete_action = 0
        trades_executed_this_step = 0

        # Iterate assets and apply frequency/hold constraints
        for i, asset in enumerate(self.assets):
            if i * 3 + 2 >= len(action) or asset not in current_prices:
                continue

            base_idx = i * 3
            main_decision = action[base_idx + 0]
            
            # 🔍 DIAGNOSTIC LOGGING - See what model produces
            if i == 0 and self.current_step % 100 == 0:  # Log only for first asset, every 100 steps
                self.logger.info(
                    f"[ACTION_DIAG] Step {self.current_step} | "
                    f"Sample action: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}] | "
                    f"Stats: min={action.min():.3f}, max={action.max():.3f}, mean={action.mean():.3f}"
                )
            
            # --- FREQUENCY CONTROLLER CHECKS ---
            
            # 1. Check if we can open a trade for this asset
            can_trade, reason = self.freq_controller.can_open_trade(
                asset=asset,
                current_step=self.current_step,
                check_global=True,
                check_asset=True,
                check_daily=True
            )
            
            if not can_trade and main_decision > action_threshold:
                # Block OPENING trade if frequency constraints violated
                if self.current_step % 500 == 0:  # Log occasionally for diagnosis
                    self.logger.debug(f"❄️ Trade blocked for {asset}: {reason}")
                action[base_idx + 0] = 0.0  # Neutralize action
                continue
            
            # 2. Check hold minimum (prevent flickering)
            position = self.portfolio_manager.positions.get(asset)
            if position and position.is_open:
                entry_step = self.asset_entry_steps.get(asset, -9999)
                steps_held = self.current_step - entry_step
                
                if steps_held < self.min_hold_steps:
                    # Block CLOSING if hold minimum not met
                    if main_decision < -action_threshold:  # Trying to sell/close
                        if self.current_step % 500 == 0:
                            self.logger.debug(f"⏳ Hold Min for {asset}: {steps_held}/{self.min_hold_steps}")
                        action[base_idx + 0] = 0.0
                        continue
            
            # 3. Record trade if action is significant (for tracking)
            if abs(action[base_idx + 0]) > action_threshold:
                # Record opening a position
                if not (position and position.is_open):
                    self.asset_entry_steps[asset] = self.current_step
                    self.freq_controller.record_trade(
                        asset=asset,
                        current_step=self.current_step,
                        timeframe=getattr(self, 'current_timeframe_for_trade', '5m'),
                        is_forced=False  # Natural trade
                    )
                    # CRITICAL: Increment counters per timeframe AND daily total
                    if hasattr(self, 'positions_count'):
                        tf = getattr(self, 'current_timeframe_for_trade', '5m')
                        # Increment timeframe-specific counter
                        if tf in self.positions_count:
                            self.positions_count[tf] = self.positions_count.get(tf, 0) + 1
                        # Increment daily total
                        self.positions_count['daily_total'] = self.positions_count.get('daily_total', 0) + 1
                        self.logger.debug(f"[NATURAL_TRADE] TF={tf} Count={self.positions_count.get(tf, 0)}, Daily={self.positions_count['daily_total']}")
                    
                    # ✅ JOUR 2: Logger le trade dans le système unifié
                    if UNIFIED_SYSTEM_AVAILABLE and central_logger:
                        action_type = "BUY" if action[base_idx + 0] > 0 else "SELL"
                        current_price = current_prices.get(asset, 0)
                        central_logger.trade(
                            action=action_type,
                            symbol=asset,
                            quantity=abs(action[base_idx + 0]),
                            price=current_price,
                            pnl=0.0  # PnL sera calculé à la fermeture
                        )
                    
                    trades_executed_this_step += 1

        # Call parent implementation with filtered actions
        # Parent will call OrderManager which handles Min Notional checks
        result = super()._execute_trades(action, dbe_modulation, action_threshold, force_trade)
        
        # Store trades count for reward calculation
        if hasattr(self, '_step_info'):
            self._step_info['trades_executed'] = trades_executed_this_step
        
        return result

    def apply_market_friction(
        self,
        target_price: float,
        order_size_usd: float,
        side: str = "buy",
        asset: str = "BTCUSDT"
    ) -> Dict[str, float]:
        """
        Apply all market friction models to determine final execution price.
        
        Args:
            target_price: Intended execution price
            order_size_usd: Order size in USDT
            side: "buy" or "sell"
            asset: Asset symbol
            
        Returns:
            Dictionary with breakdown of execution costs
        """
        if not self.enable_market_friction:
            # No friction - return target price with basic fee
            basic_fee = order_size_usd * 0.001  # 0.1%
            return {
                'target_price': target_price,
                'slippage': 0.0,
                'latency_impact': 0.0,
                'liquidity_impact': 0.0,
                'execution_price': target_price,
                'fee': basic_fee,
                'total_cost': order_size_usd + (basic_fee if side == "buy" else -basic_fee)
            }
        
        # Estimate market conditions (simplified)
        market_conditions = MarketConditions(
            volatility=0.01,  # Default 1%
            spread_bps=5.0,
            volume_24h=1e9
        )
        
        # Apply friction models sequentially
        price_after_slippage = self.slippage_model.apply_slippage(
            target_price, order_size_usd, market_conditions, side
        )
        slippage = price_after_slippage - target_price
        
        price_after_latency = self.latency_model.apply_latency(
            price_after_slippage, market_conditions
        )
        latency_impact = price_after_latency - price_after_slippage
        
        final_price = self.liquidity_model.apply_impact(
            price_after_latency, order_size_usd, market_conditions, side
        )
        liquidity_impact = final_price - price_after_latency
        
        # Calculate fee
        fee = self.fee_model.calculate_fee(order_size_usd, is_maker=False)
        
        return {
            'target_price': float(target_price),
            'slippage': float(slippage),
            'latency_impact': float(latency_impact),
            'liquidity_impact': float(liquidity_impact),
            'execution_price': float(final_price),
            'fee': float(fee),
            'total_cost': float((order_size_usd / target_price) * final_price + (fee if side == "buy" else -fee))
        }
