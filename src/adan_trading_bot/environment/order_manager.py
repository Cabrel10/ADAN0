#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Order Manager for ADAN Trading Bot.
Handles order validation, execution, and risk management.
"""

import logging
from typing import Optional

# Import Pareto Risk Detector for tail event protection
try:
    from adan_trading_bot.risk.pareto_risk_detector import ParetoRiskDetector
    PARETO_AVAILABLE = True
except ImportError:
    PARETO_AVAILABLE = False
    
from ..portfolio.portfolio_manager import PortfolioManager


logger = logging.getLogger(__name__)


class OrderManager:
    """
    Manages trade orders with risk controls and position sizing.
    Integrates Pareto Risk Detector for dynamic risk adjustment during extreme market conditions.
    """

    def __init__(self, trading_rules: dict, penalties: dict, enable_pareto: bool = True):
        """
        Initialize OrderManager.

        Args:
            trading_rules: Trading rules configuration
            penalties: Penalty configuration for invalid trades
            enable_pareto: Whether to enable Pareto Risk Detector
        """
        self.trading_rules = trading_rules
        self.penalties = penalties
        
        # Initialize Pareto Risk Detector if available and enabled
        self.pareto_detector = None
        if PARETO_AVAILABLE and enable_pareto:
            self.pareto_detector = ParetoRiskDetector(
                window_size=100,
                update_frequency=20,
                high_vol_threshold=2.0,
                extreme_threshold=5.0
            )
            logger.info("✅ Pareto Risk Detector enabled")
        elif enable_pareto and not PARETO_AVAILABLE:
            logger.warning("⚠️ Pareto Risk Detector requested but not available")
        
        logger.info("OrderManager initialized.")

    def open_position(
        self,
        portfolio: PortfolioManager,
        asset: str,
        price: float,
        size: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        confidence: float = 1.0,
    ) -> bool:
        """Open a new position via the portfolio manager.

        Args:
            portfolio: The portfolio instance.
            asset: The asset to open a position for.
            price: The current price of the asset.
            size: The size of the position to open. If None, calculated
                based on risk.
            stop_loss: Optional stop loss price.
            take_profit: Optional take profit price.
            confidence: The confidence level of the action.

        Returns:
            bool: True if the position was opened successfully,
                False otherwise.
        """
        if portfolio.positions[asset].is_open:
            logger.warning(
                "Cannot open a new position for %s, one is already open.", asset
            )
            return False

        # If size is not provided, calculate it based on risk
        # Uses Capital Tiers from config.yaml (risk_per_trade_pct varies by capital level)
        if size is None:
            stop_loss_pct = self.trading_rules.get("stop_loss", 0.0)
            risk_per_trade = self.trading_rules.get("risk_per_trade", 0.01)
            
            # ------------------------------------------------------------------
            # PARETO RISK ADJUSTMENT (Phase 2: Security & Robustness)
            # ------------------------------------------------------------------
            # Adjust risk based on market regime (fat tails detection)
            if self.pareto_detector is not None:
                original_risk = risk_per_trade
                risk_per_trade = self.pareto_detector.get_risk_multiplier(risk_per_trade)
                
                # Log adjustment if regime is not NORMAL
                regime_info = self.pareto_detector.get_regime_info()
                if regime_info['regime'] != "NORMAL":
                    logger.info(
                        f"📊 Pareto Adjustment: {regime_info['regime']} regime detected | "
                        f"Risk: {original_risk*100:.1f}% → {risk_per_trade*100:.1f}% "
                        f"(×{regime_info['multiplier']:.2f})"
                    )
            
            available_capital = portfolio.get_available_capital()

            # Simple position sizing based on risk per trade
            if stop_loss_pct > 0:
                risk_amount = available_capital * risk_per_trade
                size = risk_amount / (stop_loss_pct * price)
            else:
                # Default to 10% of available capital if no stop loss
                size = (available_capital * 0.1) / price
        # Else: size was provided by agent (from SL/TP autonomy), use it as-is
        # The Capital Tiers system in the environment will validate it

        # Round to appropriate decimal places for the asset
        size = round(size, 8)  # 8 decimal places for crypto

        if size <= 0:
            logger.warning(
                "Invalid position size %s for %s at price %s",
                size,
                asset,
                price,
            )
            return False

        # Open the position through the portfolio manager
        # Updated to pass SL/TP for full autonomy
        return portfolio.open_position(
            asset, 
            price, 
            size, 
            stop_loss=stop_loss, 
            take_profit=take_profit
        )

    def close_position(
        self,
        portfolio: PortfolioManager,
        asset: str,
        price: float,
    ) -> float:
        """Close the current position for a given asset.

        Uses the portfolio manager to close the position.

        Args:
            portfolio: The portfolio instance.
            asset: The asset to close the position for.
            price: The current price of the asset.

        Returns:
            float: The realized PnL from closing the position.
        """
        if asset not in portfolio.positions or not portfolio.positions[asset].is_open:
            logger.warning(
                "Cannot close a position for %s, none is open.",
                asset,
            )
            return 0.0

        # The portfolio manager handles the logic of closing
        return portfolio.close_position(asset, price)

    def validate_order(
        self,
        order: dict,
        portfolio_manager: PortfolioManager,
    ) -> tuple[bool, float]:
        """Validate a generic trade order.

        Note: This seems to be a legacy method. The primary logic is now in
        open_position and close_position which use the portfolio's own
        validation.

        Args:
            order: Dictionary containing order details.
            portfolio_manager: The portfolio manager instance.

        Returns:
            tuple[bool, float]: A tuple containing a boolean
            indicating if the order is valid and a penalty value
            (0.0 if valid, penalty value if invalid).
        """
        # This method's logic is largely incompatible with the
        # current PortfolioManager structure. It relies on dictionary
        # access to positions and a 'capital' attribute. The core
        # validation is now handled within PortfolioManager's
        # validate_position. We can perform a basic check here.
        size = order.get("units", 0)
        price = order.get("price", 0)
        asset = order.get("asset", "BTC")

        if size == 0 or price <= 0:
            return False, self.penalties.get("invalid_action", 1.0)

        is_valid = portfolio_manager.validate_position(
            asset,
            abs(size),
            price,
        )
        penalty = 0.0 if is_valid else self.penalties.get("invalid_action", 1.0)

        return is_valid, penalty

    def reset(self) -> None:
        """Reset the order manager's internal state.

        This method is called at the beginning of each episode to reset any
        internal state that needs resetting. Currently, OrderManager doesn't
        maintain internal state that needs resetting, but this method is
        provided for API consistency.
        """
        logger.debug("OrderManager reset called.")
        # Reset internal state
        self.penalties = {
            "invalid_action": 1.0,  # Penalty for invalid actions
            "slippage": 0.01,  # Small penalty for slippage
            "overnight": 0.05,  # Penalty for holding overnight
            "overtrading": 0.1,  # Penalty for excessive trading
            "risk_limit": 0.5,  # Penalty for exceeding risk limits
            "timeout": 0.2,  # Penalty for taking too long to act
        }
        logger.info(
            "OrderManager reset complete. Penalties: %s",
            self.penalties,
        )
