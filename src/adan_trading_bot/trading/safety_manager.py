#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Safety Manager for handling stop-loss, take-profit, and other risk management orders.
"""
import time
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from .order_manager import Order, OrderManager, OrderSide, OrderStatus, OrderType

logger = logging.getLogger(__name__)

class SafetyManager:
    """Manages safety orders like stop-loss and take-profit."""
    
    def __init__(self, order_manager: OrderManager, config: Dict[str, Any]):
        """
        Initialize the SafetyManager.
        
        Args:
            order_manager: OrderManager instance
            config: Configuration dictionary
        """
        self.order_manager = order_manager
        self.config = config
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        
        # Default safety parameters
        self.default_stop_loss = config.get('default_stop_loss', 0.02)  # 2%
        self.default_take_profit = config.get('default_take_profit', 0.04)  # 4%
        self.trailing_stop = config.get('trailing_stop', False)
        self.trailing_distance = config.get('trailing_distance', 0.01)  # 1%
        
        # DBE modulated parameters
        self.dbe_sl_pct = None
        self.dbe_tp_pct = None
        self.dbe_trailing_stop = None
        self.dbe_trailing_distance = None

        logger.info("SafetyManager initialized")
    
    def update_dbe_params(self, dbe_modulation: Dict[str, Any]) -> None:
        """
        Updates safety parameters based on DBE modulation.
        """
        self.dbe_sl_pct = dbe_modulation.get('sl_pct', self.dbe_sl_pct)
        self.dbe_tp_pct = dbe_modulation.get('tp_pct', self.dbe_tp_pct)
        self.dbe_trailing_stop = dbe_modulation.get('trailing_stop', self.dbe_trailing_stop)
        self.dbe_trailing_distance = dbe_modulation.get('trailing_distance', self.dbe_trailing_distance)
        logger.debug(f"SafetyManager updated with DBE params: SL={self.dbe_sl_pct}, TP={self.dbe_tp_pct}, Trailing={self.dbe_trailing_stop}")

    def add_position(
        self, 
        symbol: str, 
        entry_price: float, 
        amount: float, 
        side: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: int = 1,
        trailing_stop: Optional[bool] = None,
        trailing_distance: Optional[float] = None
    ) -> bool:
        """
        Add a new position with safety orders.
        
        Args:
            symbol: Trading pair symbol
            entry_price: Entry price
            amount: Position size in base currency
            side: 'long' or 'short'
            stop_loss: Stop-loss distance (e.g., 0.02 for 2%)
            take_profit: Take-profit distance (e.g., 0.04 for 4%)
            leverage: Leverage for the position
            trailing_stop: Whether to use trailing stop
            trailing_distance: Distance for trailing stop
            
        Returns:
            bool: True if safety orders were placed successfully
        """
        # Use DBE parameters if available, otherwise use provided or defaults
        stop_loss_to_use = self.dbe_sl_pct if self.dbe_sl_pct is not None else (stop_loss if stop_loss is not None else self.default_stop_loss)
        take_profit_to_use = self.dbe_tp_pct if self.dbe_tp_pct is not None else (take_profit if take_profit is not None else self.default_take_profit)
        trailing_stop_to_use = self.dbe_trailing_stop if self.dbe_trailing_stop is not None else (trailing_stop if trailing_stop is not None else self.trailing_stop)
        trailing_distance_to_use = self.dbe_trailing_distance if self.dbe_trailing_distance is not None else (trailing_distance if trailing_distance is not None else self.trailing_distance)
        
        # Calculate stop and take profit prices
        if side.lower() == 'long':
            stop_price = entry_price * (1 - stop_loss_to_use)
            take_profit_price = entry_price * (1 + take_profit_to_use)
            stop_side = OrderSide.SELL
            take_profit_side = OrderSide.SELL
        elif side.lower() == 'short':
            stop_price = entry_price * (1 + stop_loss_to_use)
            take_profit_price = entry_price * (1 - take_profit_to_use)
            stop_side = OrderSide.BUY
            take_profit_side = OrderSide.BUY
        else:
            logger.error(f"Invalid position side: {side}")
            return False
        
        # Place stop-loss order
        stop_order = self.order_manager.create_stop_loss_order(
            symbol=symbol,
            side=stop_side,
            amount=amount,
            stop_price=stop_price,
            price=stop_price * 0.999 if stop_side == OrderSide.SELL else stop_price * 1.001,  # Slightly better price
            leverage=leverage,
            params={
                'trailingStop': trailing_stop_to_use,
                'trailingDistance': trailing_distance_to_use if trailing_stop_to_use else None,
                'closePosition': True
            }
        )
        
        if not stop_order:
            logger.error(f"Failed to place stop-loss order for {symbol}")
            return False
        
        # Place take-profit order
        take_profit_order = self.order_manager.create_take_profit_order(
            symbol=symbol,
            side=take_profit_side,
            amount=amount,
            stop_price=take_profit_price,
            price=take_profit_price,
            leverage=leverage,
            params={'closePosition': True}
        )
        
        if not take_profit_order:
            # Cancel the stop-loss order if take-profit fails
            self.order_manager.cancel_order(stop_order.id, symbol)
            logger.error(f"Failed to place take-profit order for {symbol}")
            return False
        
        # Store the active orders
        self.active_orders[symbol] = {
            'entry_price': entry_price,
            'amount': amount,
            'side': side,
            'stop_loss': stop_order,
            'take_profit': take_profit_order,
            'leverage': leverage,
            'trailing_stop': trailing_stop_to_use,
            'trailing_distance': trailing_distance_to_use,
            'timestamp': time.time()
        }
        
        logger.info(f"Added safety orders for {symbol} {side} position: "
                   f"SL={stop_price:.2f}, TP={take_profit_price:.2f}")
        return True
    
    def update_trailing_stop(self, symbol: str, current_price: float) -> bool:
        """
        Update trailing stop price based on current price.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            
        Returns:
            bool: True if stop was updated, False otherwise
        """
        if symbol not in self.active_orders:
            return False
        
        position = self.active_orders[symbol]
        
        # Only update for long positions with trailing stop enabled
        if not position['trailing_stop'] or position['side'].lower() != 'long':
            return False
        
        # Get current stop order
        stop_order = position['stop_loss']
        if not stop_order or stop_order.status != OrderStatus.NEW:
            return False
        
        # Calculate new stop price
        new_stop_price = current_price * (1 - position['trailing_distance'])
        current_stop_price = stop_order.stop_price
        
        # Only move stop up, not down
        if new_stop_price <= current_stop_price:
            return False
        
        try:
            # Cancel existing stop order
            self.order_manager.cancel_order(stop_order.id, symbol)
            
            # Create new stop order with updated price
            new_stop_order = self.order_manager.create_stop_loss_order(
                symbol=symbol,
                side=OrderSide.SELL,
                amount=position['amount'],
                stop_price=new_stop_price,
                price=new_stop_price * 0.999,  # Slightly better price
                leverage=position['leverage'],
                params={
                    'trailingStop': True,
                    'trailingDistance': position['trailing_distance'],
                    'closePosition': True
                }
            )
            
            if new_stop_order:
                # Update the active orders
                self.active_orders[symbol]['stop_loss'] = new_stop_order
                logger.info(f"Updated trailing stop for {symbol}: {current_stop_price:.2f} -> {new_stop_price:.2f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update trailing stop for {symbol}: {e}")
            return False
    
    def remove_position(self, symbol: str) -> bool:
        """
        Remove a position and cancel its safety orders.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            bool: True if successful, False otherwise
        """
        if symbol not in self.active_orders:
            return False
        
        position = self.active_orders.pop(symbol)
        success = True
        
        # Cancel stop-loss order if it exists and is still active
        stop_order = position.get('stop_loss')
        if stop_order and stop_order.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED):
            if not self.order_manager.cancel_order(stop_order.id, symbol):
                success = False
        
        # Cancel take-profit order if it exists and is still active
        take_profit_order = position.get('take_profit')
        if take_profit_order and take_profit_order.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED):
            if not self.order_manager.cancel_order(take_profit_order.id, symbol):
                success = False
        
        if success:
            logger.info(f"Removed safety orders for {symbol}")
        else:
            logger.warning(f"Failed to remove some safety orders for {symbol}")
        
        return success
    
    def check_and_update_orders(self, symbol: str, current_price: float) -> bool:
        """
        Check and update safety orders for a symbol.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            
        Returns:
            bool: True if any updates were made, False otherwise
        """
        if symbol not in self.active_orders:
            return False
        
        updated = False
        
        # Update trailing stop if enabled
        if self.active_orders[symbol]['trailing_stop']:
            if self.update_trailing_stop(symbol, current_price):
                updated = True
        
        # Check if any orders have been filled
        position = self.active_orders[symbol]
        
        # Check stop-loss order
        stop_order = position['stop_loss']
        if stop_order and stop_order.status == OrderStatus.FILLED:
            logger.info(f"Stop-loss order filled for {symbol} at {stop_order.price}")
            # Remove the position since stop was hit
            self.remove_position(symbol)
            return True
        
        # Check take-profit order
        take_profit_order = position['take_profit']
        if take_profit_order and take_profit_order.status == OrderStatus.FILLED:
            logger.info(f"Take-profit order filled for {symbol} at {take_profit_order.price}")
            # Remove the position since take profit was hit
            self.remove_position(symbol)
            return True
        
        return updated
    
    def get_active_positions(self) -> List[Dict[str, Any]]:
        """
        Get all active positions with their safety orders.
        
        Returns:
            List of active positions with their safety orders
        """
        positions = []
        
        for symbol, position in self.active_orders.items():
            pos_data = {
                'symbol': symbol,
                'side': position['side'],
                'amount': position['amount'],
                'entry_price': position['entry_price'],
                'leverage': position['leverage'],
                'stop_loss': position['stop_loss'].to_dict() if position['stop_loss'] else None,
                'take_profit': position['take_profit'].to_dict() if position['take_profit'] else None,
                'trailing_stop': position['trailing_stop'],
                'trailing_distance': position['trailing_distance'],
                'timestamp': position['timestamp']
            }
            positions.append(pos_data)
        
        return positions
    
    def cleanup_expired_orders(self, max_age_seconds: int = 86400) -> int:
        """
        Clean up expired orders from active positions.
        
        Args:
            max_age_seconds: Maximum age of orders in seconds
            
        Returns:
            int: Number of orders cleaned up
        """
        current_time = time.time()
        removed_count = 0
        
        for symbol in list(self.active_orders.keys()):
            position = self.active_orders[symbol]
            
            # Check if position is too old
            if current_time - position['timestamp'] > max_age_seconds:
                if self.remove_position(symbol):
                    removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired positions")
        
        return removed_count
