#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Order management module for the ADAN trading bot.

In a backtesting environment, this module translates agent actions into simple
portfolio operations without interacting with a live exchange.
"""

import logging
from enum import Enum
from typing import Dict
from ..portfolio.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    NEW = "NEW"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"

class Order:
    def __init__(self, id: str, symbol: str, type: OrderType, side: OrderSide, price: float, quantity: float, status: OrderStatus = OrderStatus.NEW):
        self.id = id
        self.symbol = symbol
        self.type = type
        self.side = side
        self.price = price
        self.quantity = quantity
        self.status = status

class OrderManager:
    """
    Manages order execution within the backtesting environment.

    This class acts as a simplified broker, translating the agent's discrete
    actions (Hold, Buy, Sell) into state changes in the PortfolioManager.
    """
    def __init__(self, portfolio_manager: PortfolioManager):
        """
        Initializes the OrderManager.

        Args:
            portfolio_manager: An instance of the PortfolioManager to interact with.
        """
        self.portfolio = portfolio_manager
        self.open_orders: Dict[str, Order] = {}
        logger.info("OrderManager initialized for backtesting.")

    def place_order(self, order: Order) -> bool:
        """
        Places a new order.
        """
        if order.id in self.open_orders:
            logger.warning(f"Order with ID {order.id} already exists.")
            return False
        self.open_orders[order.id] = order
        logger.info(f"Order placed: ID={order.id}, Symbol={order.symbol}, Type={order.type.value}, Side={order.side.value}, Price={order.price}, Quantity={order.quantity}")
        return True

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancels an open order.
        """
        order = self.open_orders.get(order_id)
        if not order:
            logger.warning(f"Order with ID {order_id} not found.")
            return False
        if order.status == OrderStatus.FILLED or order.status == OrderStatus.PARTIALLY_FILLED:
            logger.warning(f"Cannot cancel order {order_id}: already filled or partially filled.")
            return False
        order.status = OrderStatus.CANCELED
        del self.open_orders[order_id]
        logger.info(f"Order {order_id} canceled.")
        return True

    def process_order(self, order_id: str, fill_quantity: float, fill_price: float):
        """
        Processes a fill for an open order, handling partial fills.
        """
        order = self.open_orders.get(order_id)
        if not order:
            logger.warning(f"Order with ID {order_id} not found for processing.")
            return

        if order.status == OrderStatus.CANCELED or order.status == OrderStatus.FILLED:
            logger.warning(f"Order {order_id} is already {order.status.value}. Cannot process fill.")
            return

        remaining_quantity = order.quantity - fill_quantity

        if remaining_quantity <= 0:
            order.status = OrderStatus.FILLED
            logger.info(f"Order {order_id} fully filled. Quantity: {fill_quantity}, Price: {fill_price}")
            del self.open_orders[order_id]
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
            order.quantity = remaining_quantity # Update remaining quantity
            logger.info(f"Order {order_id} partially filled. Filled: {fill_quantity}, Remaining: {remaining_quantity}, Price: {fill_price}")

        # Here you would typically interact with the PortfolioManager to update positions
        # For simplicity, this is a placeholder.
        # if order.side == OrderSide.BUY:
        #     self.portfolio.open_position(order.symbol, fill_price, fill_quantity)
        # else:
        #     self.portfolio.close_position(order.symbol, fill_price, fill_quantity)

    def execute_action(self, action: int, current_price: float, asset: str) -> float:
        """
        Executes a trading action based on the agent's decision.

        Args:
            action: The discrete action from the agent.
                    - 0: Hold (do nothing)
                    - 1: Buy (open a long position)
                    - 2: Sell (close the long position)
            current_price: The current market price for the asset.
            asset: The asset symbol.

        Returns:
            The realized PnL from the action, if any. Returns 0.0 for
            hold or buy actions, or if a sell action is invalid.
        """
        realized_pnl = 0.0

        if action == 1:  # Buy Action
            # Check if a position for the asset exists and if it's closed
            if asset not in self.portfolio.positions or not self.portfolio.positions[asset].is_open:
                # Calculate size based on portfolio manager's logic
                size = self.portfolio.calculate_position_size(
                    action_type="buy", asset=asset, current_price=current_price, confidence=1.0
                )
                if self.portfolio.validate_position(asset, size, current_price):
                    self.portfolio.open_position(asset, current_price, size)
                    logger.debug(f"Action: BUY {size:.4f} {asset} at {current_price:.2f}")
                else:
                    logger.warning(f"Action: BUY for {asset} invalid. Holding.")
            else:
                # If a position is already open, buying again is treated as holding
                logger.debug("Action: Attempted BUY, but position already open. Holding.")

        elif action == 2:  # Sell Action
            # Check if a position for the asset exists and is open
            if asset in self.portfolio.positions and self.portfolio.positions[asset].is_open:
                realized_pnl = self.portfolio.close_position(asset, current_price)
                logger.debug(f"Action: SELL {asset} at {current_price:.2f}, PnL: {realized_pnl:.2f}")
            else:
                # If no position is open, selling is treated as holding
                logger.debug("Action: Attempted SELL, but no position open. Holding.")
        
        # For action == 0 (Hold), we do nothing.
        
        return realized_pnl