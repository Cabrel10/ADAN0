#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Position management module for the ADAN trading bot.

This module is responsible for managing open trading positions, including
opening, closing, and adjusting them based on risk and market conditions.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class PositionManager:
    """
    Manages trading positions within the portfolio.

    This class interacts with the PortfolioManager to execute trades and
    adjust positions based on signals from the RiskAssessor and agent actions.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the PositionManager.

        Args:
            config: Configuration dictionary for position management.
        """
        self.config = config.get('position_management', {})
        self.trailing_stop_loss_pct = self.config.get('trailing_stop_loss_pct', 0.01) # Default 1%
        logger.info("PositionManager initialized.")

    def update_trailing_stop_loss(self, asset: str, current_price: float, portfolio_manager: Any) -> None:
        """
        Updates the trailing stop-loss level for a given asset.

        Args:
            asset: The asset to update the stop-loss for.
            current_price: The current market price of the asset.
            portfolio_manager: The PortfolioManager instance.
        """
        position = portfolio_manager.get_position(asset)
        if position and position.is_open:
            new_stop_loss = current_price * (1 - self.trailing_stop_loss_pct)
            if new_stop_loss > position.stop_loss:
                portfolio_manager.update_position_stop_loss(asset, new_stop_loss)
                logger.info(f"Updated trailing stop-loss for {asset} to {new_stop_loss}")

    def open_position(self, asset: str, size: float, price: float, portfolio_manager: Any) -> bool:
        """
        Opens a new position for a given asset.

        Args:
            asset: The asset to trade.
            size: The size of the position (in units).
            price: The entry price.
            portfolio_manager: The PortfolioManager instance.

        Returns:
            True if the position was opened successfully, False otherwise.
        """
        logger.info(f"Opening position for {asset} with size {size} at price {price}")
        return portfolio_manager.open_position(asset, size, price)

    def close_position(self, asset: str, price: float, portfolio_manager: Any) -> bool:
        """
        Closes an existing position for a given asset.

        Args:
            asset: The asset to close the position for.
            price: The exit price.
            portfolio_manager: The PortfolioManager instance.

        Returns:
            True if the position was closed successfully, False otherwise.
        """
        logger.info(f"Closing position for {asset} at price {price}")
        return portfolio_manager.close_position(asset, price)

    def adjust_position(self, asset: str, new_size: float, price: float, portfolio_manager: Any) -> bool:
        """
        Adjusts the size of an existing position.

        Args:
            asset: The asset to adjust.
            new_size: The new size of the position (in units).
            price: The current market price.
            portfolio_manager: The PortfolioManager instance.

        Returns:
            True if the position was adjusted successfully, False otherwise.
        """
        logger.info(f"Adjusting position for {asset} to new size {new_size} at price {price}")
        return portfolio_manager.adjust_position(asset, new_size, price)
