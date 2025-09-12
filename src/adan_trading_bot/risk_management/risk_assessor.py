#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Risk assessment module for the ADAN trading bot.

This module provides tools for evaluating various risk metrics, assessing
the overall risk profile, and managing risk controls including stop-loss.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classification."""

    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    EXTREME = auto()


@dataclass
class PositionRisk:
    """Risk metrics for a single position."""

    asset: str
    current_value: float
    entry_price: float
    current_price: float
    size: float
    pnl: float
    pnl_pct: float
    risk_score: float = 0.0
    var: Optional[float] = None
    cvar: Optional[float] = None
    stop_loss: Optional[float] = None
    trailing_stop: Optional[float] = None
    max_drawdown: float = 0.0
    volatility: float = 0.0


class RiskAssessor:
    """
    Comprehensive risk assessment and management for the trading system.

    Handles risk metrics calculation, position monitoring, and implements
    risk controls including stop-loss and position sizing.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RiskAssessor with configuration.

        Args:
            config: Configuration dictionary with risk parameters
        """
        self.config = config.get("risk_management", {})
        self.positions: Dict[str, PositionRisk] = {}
        self.historical_returns: Dict[str, List[float]] = {}
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        self.var_horizon = self.config.get("var_horizon", 1)  # In days
        self.var_confidence = self.config.get("var_confidence", 0.95)
        self.max_position_size = self.config.get(
            "max_position_size", 0.1
        )  # 10% of capital
        self.max_portfolio_risk = self.config.get(
            "max_portfolio_risk", 0.02
        )  # 2% risk per trade

        logger.info("RiskAssessor initialized with config: %s", self.config)

    def update_market_data(self, market_data: Dict[str, Any]) -> None:
        """
        Update market data for risk calculations.

        Args:
            market_data: Dictionary containing current market data
        """
        self.current_prices = market_data.get("prices", {})
        self.volumes = market_data.get("volumes", {})
        self.spreads = market_data.get("spreads", {})

        # Update historical returns for volatility calculation
        for asset, price in self.current_prices.items():
            if asset not in self.historical_returns:
                self.historical_returns[asset] = []
            if len(self.historical_returns[asset]) > 0:
                prev_price = self.historical_returns[asset][-1]
                if prev_price > 0:
                    ret = (price - prev_price) / prev_price
                    self.historical_returns[asset].append(ret)
                    # Keep only last N returns for calculation
                    if len(self.historical_returns[asset]) > 100:
                        self.historical_returns[asset].pop(0)
            else:
                self.historical_returns[asset].append(0) # Add a 0 return for the first price

    def calculate_var(
        self, returns: List[float], confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR).

        Args:
            returns: List of historical returns
            confidence: Confidence level for VaR (0-1)

        Returns:
            Tuple of (VaR, CVaR)
        """
        if not returns:
            return 0.0, 0.0

        returns = np.array(returns)
        var = -np.percentile(returns, (1 - confidence) * 100)
        cvar = (
            -returns[returns <= -var].mean()
            if (returns <= -var).any()
            else var
        )
        return var, cvar

    def update_position_risk(self, position: Dict[str, Any]) -> PositionRisk:
        """
        Calculate risk metrics for a single position.

        Args:
            position: Dictionary containing position data

        Returns:
            PositionRisk object with calculated metrics
        """
        asset = position["asset"]
        current_price = self.current_prices.get(
            asset, position.get("entry_price", 0)
        )
        size = position["size"]
        entry_price = position["entry_price"]
        current_value = size * current_price
        pnl = (current_price - entry_price) * size
        pnl_pct = (
            (current_price / entry_price - 1) * 100 if entry_price > 0 else 0
        )

        # Calculate volatility
        returns = self.historical_returns.get(asset, [])
        volatility = (
            np.std(returns) * np.sqrt(252) if returns else 0
        )  # Annualized

        # Calculate VaR and CVaR
        var, cvar = self.calculate_var(returns, self.var_confidence)

        # Update trailing stop if enabled
        trailing_stop = None
        if position.get("trailing_stop_pct"):
            if "highest_price" not in position:
                position["highest_price"] = current_price
            else:
                position["highest_price"] = max(position["highest_price"], current_price)
            trailing_stop = position["highest_price"] * (1 - position["trailing_stop_pct"])

        # Create or update position risk
        if asset in self.positions:
            pos_risk = self.positions[asset]
            pos_risk.current_price = current_price
            pos_risk.current_value = current_value
            pos_risk.pnl = pnl
            pos_risk.pnl_pct = pnl_pct
            pos_risk.volatility = volatility
            pos_risk.var = var
            pos_risk.cvar = cvar
            if trailing_stop:
                pos_risk.trailing_stop = trailing_stop
        else:
            pos_risk = PositionRisk(
                asset=asset,
                current_value=current_value,
                entry_price=entry_price,
                current_price=current_price,
                size=size,
                pnl=pnl,
                pnl_pct=pnl_pct,
                volatility=volatility,
                var=var,
                cvar=cvar,
                stop_loss=position.get("stop_loss"),
                trailing_stop=trailing_stop,
            )
            self.positions[asset] = pos_risk

        return pos_risk

    def calculate_position_size(
        self,
        asset: str,
        entry_price: float,
        stop_loss: float,
        risk_percent: float,
    ) -> float:
        """
        Calculate optimal position size based on risk parameters.

        Args:
            asset: Asset symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_percent: Maximum risk as percentage of capital

        Returns:
            Position size in units
        """
        if entry_price <= 0 or stop_loss >= entry_price:
            return 0.0

        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            return 0.0

        # Get account equity or use default
        account_equity = self.positions.get(
            "total_equity", 10000
        )  # Default 10k if not available

        # Calculate position size
        risk_amount = account_equity * (risk_percent / 100)
        position_size = risk_amount / risk_per_share

        # Apply position size limits
        max_position_value = account_equity * self.max_position_size
        max_position_size = (
            max_position_value / entry_price if entry_price > 0 else 0
        )

        return min(position_size, max_position_size)

    def check_stop_loss(self, asset: str, current_price: float) -> bool:
        """
        Check if stop loss or trailing stop is triggered.

        Args:
            asset: Asset symbol
            current_price: Current market price

        Returns:
            True if stop loss is triggered, False otherwise
        """
        if asset not in self.positions or current_price is None:
            return False

        position = self.positions[asset]

        # Check regular stop loss
        if position.stop_loss and current_price <= position.stop_loss:
            logger.info(f"Stop loss triggered for {asset} at {current_price}")
            return True

        # Check trailing stop
        if position.trailing_stop and current_price <= position.trailing_stop:
            logger.info(
                f"Trailing stop triggered for {asset} at {current_price}"
            )
            return True

        return False

    def assess_portfolio_risk(
        self, portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess overall portfolio risk.

        Args:
            portfolio: Dictionary containing portfolio data

        Returns:
            Dictionary with portfolio risk metrics
        """
        if not portfolio:
            return {}
        total_value = portfolio.get("total_value", 0)
        equity = portfolio.get("equity", 0)
        used_margin = portfolio.get("used_margin", 0)

        # Calculate drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity

        drawdown = (
            (self.peak_equity - equity) / self.peak_equity
            if self.peak_equity > 0
            else 0
        )
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # Calculate portfolio VaR and CVaR
        portfolio_returns = []
        for asset, returns in self.historical_returns.items():
            if asset in portfolio.get("positions", {}):
                weight = (
                    portfolio["positions"][asset].get("value", 0) / total_value
                    if total_value > 0
                    else 0
                )
                portfolio_returns.extend([r * weight for r in returns])

        var, cvar = self.calculate_var(portfolio_returns, self.var_confidence)

        # Calculate risk metrics
        risk_metrics = {
            "total_value": total_value,
            "equity": equity,
            "used_margin": used_margin,
            "margin_ratio": used_margin / equity if equity > 0 else 0,
            "drawdown": drawdown,
            "max_drawdown": self.max_drawdown,
            "var": var,
            "cvar": cvar,
            "leverage": total_value / equity if equity > 0 else 0,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Check risk limits
        risk_metrics["risk_level"] = self._determine_risk_level(risk_metrics)

        return risk_metrics

    def _determine_risk_level(self, metrics: Dict[str, Any]) -> RiskLevel:
        """Determine overall portfolio risk level."""
        if metrics["drawdown"] > 0.2 or metrics["leverage"] > 10:
            return RiskLevel.EXTREME
        elif metrics["drawdown"] > 0.1 or metrics["leverage"] > 5:
            return RiskLevel.HIGH
        elif metrics["drawdown"] > 0.05 or metrics["leverage"] > 2:
            return RiskLevel.MODERATE
        return RiskLevel.LOW

    def get_risk_limits(self) -> Dict[str, Any]:
        """Get current risk limits and settings."""
        return {
            "max_position_size": self.max_position_size,
            "max_portfolio_risk": self.max_portfolio_risk,
            "var_confidence": self.var_confidence,
            "var_horizon": self.var_horizon,
        }

    def update_risk_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update risk parameters dynamically.

        Args:
            params: Dictionary of parameters to update

        Raises:
            ValueError: If any parameter value is invalid
        """
        # Validate parameters before updating
        for key, value in params.items():
            if not hasattr(self, key):
                continue

            # Specific validations for each parameter
            if key == "max_position_size" and value <= 0:
                raise ValueError(f"{key} must be greater than 0, got {value}")
            elif key == "var_confidence" and not (0 < value < 1):
                raise ValueError(f"{key} must be between 0 and 1, got {value}")
            elif key == "max_portfolio_risk" and value <= 0:
                raise ValueError(f"{key} must be greater than 0, got {value}")
            elif key == "var_horizon" and value <= 0:
                raise ValueError(f"{key} must be greater than 0, got {value}")

        # Update parameters if all validations pass
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated risk parameter {key} to {value}")

    def get_position_risk(self, asset: str) -> Optional[PositionRisk]:
        """Get risk metrics for a specific position."""
        return self.positions.get(asset)

    def get_all_positions_risk(self) -> Dict[str, PositionRisk]:
        """Get risk metrics for all positions."""
        return self.positions

    def clear_positions(self) -> None:
        """Clear all position data."""
        self.positions.clear()
        logger.info("Cleared all position data")
