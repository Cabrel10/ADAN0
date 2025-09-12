"""
Action translation module for the ADAN Trading Bot.

This module provides translation between RL agent actions and actual trading
actions, including validation, position sizing, and risk management
integration.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
from enum import Enum

from .fee_manager import FeeManager, FeeType
from .position_sizer import PositionSizer, PositionSizingMethod

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Types of trading actions."""
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE_LONG = 3
    CLOSE_SHORT = 4


class PositionSizeMethod(Enum):
    """Methods for calculating position sizes."""
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY = "kelly"


@dataclass
class TradingAction:
    """Represents a translated trading action."""
    action_type: ActionType
    asset: str
    size: float
    price: Optional[float] = None
    fees: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ActionValidationResult:
    """Result of action validation."""
    is_valid: bool
    message: str
    adjusted_action: Optional[TradingAction] = None
    warnings: List[str] = None


class ActionTranslator:
    """
    Translates RL agent actions into executable trading actions.

    This class handles the conversion from raw agent outputs to structured
    trading actions with proper validation and risk management.
    """

    def __init__(self,
                 action_space_type: str = "discrete",
                 position_size_method: PositionSizeMethod = (
                     PositionSizeMethod.PERCENTAGE),
                 default_position_size: float = 0.1,
                 max_position_size: float = 0.5,
                 min_position_size: float = 0.01,
                 enable_stop_loss: bool = True,
                 enable_take_profit: bool = True,
                 default_stop_loss_pct: float = 0.02,
                 default_take_profit_pct: float = 0.04,
                 risk_free_rate: float = 0.02,
                 fee_manager: Optional[FeeManager] = None,
                 exchange: str = "default",
                 sizing_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ActionTranslator.

        Args:
            action_space_type: Type of action space ("discrete" or
                "continuous")
            position_size_method: Method for calculating position sizes
            default_position_size: Default position size (as percentage of
                capital)
            max_position_size: Maximum allowed position size
            min_position_size: Minimum allowed position size
            enable_stop_loss: Whether to automatically set stop losses
            enable_take_profit: Whether to automatically set take profits
            default_stop_loss_pct: Default stop loss percentage
            default_take_profit_pct: Default take profit percentage
            risk_free_rate: Risk-free rate for Kelly criterion
            sizing_config: Configuration for position sizing, including capital tiers.
        """
        self.action_space_type = action_space_type
        self.position_size_method = position_size_method
        self.default_position_size = default_position_size
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.enable_stop_loss = enable_stop_loss
        self.enable_take_profit = enable_take_profit
        self.default_stop_loss_pct = default_stop_loss_pct
        self.default_take_profit_pct = default_take_profit_pct
        self.risk_free_rate = risk_free_rate
        self.fee_manager = fee_manager or FeeManager()
        self.exchange = exchange

        # Initialize position sizer with compatible method mapping
        position_sizing_method = self._map_position_size_method(position_size_method)
        self.position_sizer = PositionSizer(
            default_method=position_sizing_method,
            min_position_size=min_position_size,
            max_position_size=max_position_size,
            sizing_config=sizing_config
        )

        # Action mapping for discrete action space
        self.discrete_action_map = {
            0: ActionType.HOLD,
            1: ActionType.BUY,
            2: ActionType.SELL,
            3: ActionType.CLOSE_LONG,
            4: ActionType.CLOSE_SHORT
        }

        # Statistics tracking
        self.translation_stats = {
            'total_translations': 0,
            'successful_translations': 0,
            'failed_translations': 0,
            'action_type_counts': {action_type: 0 for action_type in ActionType},
            'validation_failures': 0,
            'size_adjustments': 0
        }

        logger.info(f"ActionTranslator initialized with "
                    f"{action_space_type} action space")

    def _map_position_size_method(self, method: PositionSizeMethod) -> PositionSizingMethod:
        """Map old PositionSizeMethod to new PositionSizingMethod."""
        mapping = {
            PositionSizeMethod.FIXED: PositionSizingMethod.FIXED,
            PositionSizeMethod.PERCENTAGE: PositionSizingMethod.PERCENTAGE,
            PositionSizeMethod.VOLATILITY_ADJUSTED: PositionSizingMethod.VOLATILITY_ADJUSTED,
            PositionSizeMethod.KELLY: PositionSizingMethod.KELLY_CRITERION
        }
        return mapping.get(method, PositionSizingMethod.PERCENTAGE)

    def translate_action(self,
                        agent_action: Union[int, np.ndarray, List[float]],
                        asset: str,
                        current_price: float,
                        portfolio_state: Dict[str, Any],
                        market_data: Optional[Dict[str, Any]] = None
                        ) -> TradingAction:
        """
        Translate agent action to trading action.

        Args:
            agent_action: Raw action from RL agent
            asset: Asset symbol
            current_price: Current market price
            portfolio_state: Current portfolio state
            market_data: Additional market data for context

        Returns:
            TradingAction object
        """
        self.translation_stats['total_translations'] += 1

        try:
            # Parse the agent action
            if self.action_space_type == "discrete":
                action_type, confidence = self._parse_discrete_action(agent_action)
            else:
                action_type, confidence = self._parse_continuous_action(agent_action)

            # Calculate position size
            position_size = self._calculate_position_size(
                action_type, asset, current_price, portfolio_state, market_data, confidence
            )

            # Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_stop_take_levels(
                action_type, current_price, market_data
            )

            # Calculate fees
            fees = self._calculate_fees(
                action_type, position_size, current_price, asset
            )

            # Create trading action
            trading_action = TradingAction(
                action_type=action_type,
                asset=asset,
                size=position_size,
                price=current_price,
                fees=fees,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                metadata={
                    'raw_action': agent_action,
                    'translation_method': self.action_space_type,
                    'position_size_method': self.position_size_method.value,
                    'market_data': market_data
                }
            )

            # Update statistics
            self.translation_stats['successful_translations'] += 1
            self.translation_stats['action_type_counts'][action_type] += 1

            logger.debug(f"Translated action for {asset}: {trading_action}")

            return trading_action

        except Exception as e:
            self.translation_stats['failed_translations'] += 1
            logger.error(f"Failed to translate action for {asset}: {e}")

            # Return safe default action (HOLD)
            return TradingAction(
                action_type=ActionType.HOLD,
                asset=asset,
                size=0.0,
                price=current_price,
                confidence=0.0,
                metadata={'error': str(e), 'raw_action': agent_action}
            )

    def _parse_discrete_action(self, action: Union[int, np.ndarray]
                              ) -> Tuple[ActionType, float]:
        """Parse discrete action from agent."""
        if isinstance(action, np.ndarray):
            if action.size == 1:
                action_idx = int(action.item())
                confidence = 1.0
            else:
                # Softmax output - get action with highest probability
                action_idx = int(np.argmax(action))
                confidence = float(np.max(action))
        else:
            action_idx = int(action)
            confidence = 1.0

        if action_idx not in self.discrete_action_map:
            logger.warning(f"Invalid discrete action index: {action_idx}, "
                          f"defaulting to HOLD")
            return ActionType.HOLD, 0.0

        return self.discrete_action_map[action_idx], confidence

    def _parse_continuous_action(self, action: Union[np.ndarray, List[float]]
                                ) -> Tuple[ActionType, float]:
        """Parse continuous action from agent."""
        if isinstance(action, list):
            action = np.array(action)

        if action.size < 2:
            logger.warning("Continuous action must have at least 2 dimensions")
            return ActionType.HOLD, 0.0

        # First dimension: action type (mapped to discrete)
        # Second dimension: confidence/strength
        action_value = float(action[0])
        confidence = float(np.clip(action[1], 0.0, 1.0))

        # Map continuous action to discrete action type
        if action_value < -0.5:
            action_type = ActionType.SELL
        elif action_value > 0.5:
            action_type = ActionType.BUY
        elif action_value < -0.25:
            action_type = ActionType.CLOSE_LONG
        elif action_value > 0.25:
            action_type = ActionType.CLOSE_SHORT
        else:
            action_type = ActionType.HOLD

        return action_type, confidence

    def _calculate_position_size(self,
                               action_type: ActionType,
                               asset: str,
                               current_price: float,
                               portfolio_state: Dict[str, Any],
                               market_data: Optional[Dict[str, Any]] = None,
                               confidence: float = 1.0) -> float:
        """Calculate position size using the advanced PositionSizer."""

        if action_type == ActionType.HOLD:
            return 0.0

        # Get portfolio information
        available_capital = portfolio_state.get('cash', 0.0)
        total_value = portfolio_state.get('total_value', available_capital)
        current_positions = portfolio_state.get('positions', {})

        # Handle closing positions first
        if action_type in [ActionType.CLOSE_LONG, ActionType.CLOSE_SHORT]:
            current_position = current_positions.get(asset, 0.0)
            return abs(current_position)  # Close entire position

        try:
            # Use the advanced PositionSizer
            sizing_result = self.position_sizer.calculate_position_size(
                asset=asset,
                current_price=current_price,
                portfolio_value=total_value,
                available_capital=available_capital,
                market_data=market_data,
                confidence=confidence
            )

            # Convert position value to position size (number of shares)
            position_size = sizing_result.size / current_price

            # Log any warnings from position sizing
            if sizing_result.warnings:
                for warning in sizing_result.warnings:
                    logger.debug(f"Position sizing warning: {warning}")
                self.translation_stats['size_adjustments'] += 1

            return position_size

        except Exception as e:
            logger.warning(f"Advanced position sizing failed, using fallback: {e}")
            # Fallback to simple percentage-based sizing
            fallback_size = (total_value * self.default_position_size * confidence) / current_price
            return min(fallback_size, available_capital / current_price)

    def _calculate_volatility_adjusted_size(self, market_data: Optional[Dict[str, Any]]) -> float:
        """Calculate position size adjusted for volatility."""
        if not market_data or 'volatility' not in market_data:
            return self.default_position_size

        volatility = market_data['volatility']
        target_volatility = 0.02  # 2% target volatility

        # Inverse relationship: higher volatility = smaller position
        if volatility > 0:
            volatility_adjustment = target_volatility / volatility
            adjusted_size = self.default_position_size * volatility_adjustment
        else:
            adjusted_size = self.default_position_size

        return np.clip(adjusted_size, self.min_position_size, self.max_position_size)

    def _calculate_kelly_size(self, market_data: Optional[Dict[str, Any]]) -> float:
        """Calculate position size using Kelly criterion."""
        if not market_data:
            return self.default_position_size

        # Get expected return and volatility
        expected_return = market_data.get('expected_return', 0.0)
        volatility = market_data.get('volatility', 0.1)

        if volatility <= 0:
            return self.default_position_size

        # Kelly formula: f = (bp - q) / b
        # where b = odds, p = probability of win, q = probability of loss
        # Simplified version using expected return and volatility
        kelly_fraction = (expected_return - self.risk_free_rate) / (volatility ** 2)

        # Apply conservative scaling (typically use 25-50% of Kelly)
        conservative_kelly = kelly_fraction * 0.25

        return np.clip(conservative_kelly, self.min_position_size, self.max_position_size)

    def _calculate_stop_take_levels(self,
                                  action_type: ActionType,
                                  current_price: float,
                                  market_data: Optional[Dict[str, Any]] = None) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels."""

        if action_type == ActionType.HOLD:
            return None, None

        stop_loss = None
        take_profit = None

        # Get volatility-adjusted levels if available
        if market_data and 'volatility' in market_data:
            volatility = market_data['volatility']
            # Adjust stop/take levels based on volatility
            stop_loss_pct = max(self.default_stop_loss_pct, volatility * 1.5)
            take_profit_pct = max(self.default_take_profit_pct, volatility * 2.0)
        else:
            stop_loss_pct = self.default_stop_loss_pct
            take_profit_pct = self.default_take_profit_pct

        # Calculate levels based on action type
        if action_type == ActionType.BUY:
            if self.enable_stop_loss:
                stop_loss = current_price * (1 - stop_loss_pct)
            if self.enable_take_profit:
                take_profit = current_price * (1 + take_profit_pct)

        elif action_type == ActionType.SELL:
            if self.enable_stop_loss:
                stop_loss = current_price * (1 + stop_loss_pct)
            if self.enable_take_profit:
                take_profit = current_price * (1 - take_profit_pct)

        return stop_loss, take_profit

    def _calculate_fees(self,
                        action_type: ActionType,
                        position_size: float,
                        current_price: float,
                        asset: str = "BTC") -> float:
        """Calculate trading fees using the FeeManager."""

        if action_type == ActionType.HOLD or position_size == 0:
            return 0.0

        trade_value = position_size * current_price

        # Determine if this is a maker or taker order
        # For simplicity, assume market orders (taker) for now
        is_maker = False

        try:
            fee_result = self.fee_manager.calculate_trading_fee(
                trade_value=trade_value,
                asset=asset,
                is_maker=is_maker,
                exchange=self.exchange
            )
            return fee_result.total_fee
        except Exception as e:
            logger.warning(f"Fee calculation failed, using fallback: {e}")
            # Fallback to simple percentage fee
            return trade_value * 0.001  # 0.1% default fee

    def validate_action(self,
                       trading_action: TradingAction,
                       portfolio_state: Dict[str, Any],
                       market_constraints: Optional[Dict[str, Any]] = None) -> ActionValidationResult:
        """
        Validate a trading action against portfolio and market constraints.

        Args:
            trading_action: Trading action to validate
            portfolio_state: Current portfolio state
            market_constraints: Market-specific constraints

        Returns:
            ActionValidationResult
        """
        warnings = []
        adjusted_action = None

        try:
            # Check basic constraints
            if trading_action.size < 0:
                return ActionValidationResult(
                    is_valid=False,
                    message="Position size cannot be negative"
                )

            # Check capital constraints
            available_capital = portfolio_state.get('cash', 0.0)
            required_capital = trading_action.size * (trading_action.price or 0) + (trading_action.fees or 0)

            if trading_action.action_type == ActionType.BUY and required_capital > available_capital:
                # Adjust position size to available capital
                max_affordable_size = available_capital / (trading_action.price or 1)

                if max_affordable_size >= self.min_position_size:
                    adjusted_action = TradingAction(
                        action_type=trading_action.action_type,
                        asset=trading_action.asset,
                        size=max_affordable_size,
                        price=trading_action.price,
                        fees=trading_action.fees, # This should be recalculated or adjusted
                        stop_loss=trading_action.stop_loss,
                        take_profit=trading_action.take_profit,
                        confidence=trading_action.confidence,
                        metadata=trading_action.metadata
                    )
                    warnings.append(f"Position size adjusted from {trading_action.size:.4f} to {max_affordable_size:.4f} due to capital constraints")
                else:
                    return ActionValidationResult(
                        is_valid=False,
                        message=f"Insufficient capital: need {required_capital:.2f}, have {available_capital:.2f}"
                    )

            # Check position constraints
            current_positions = portfolio_state.get('positions', {})
            current_position = current_positions.get(trading_action.asset, 0.0)

            # Validate closing actions
            if trading_action.action_type == ActionType.CLOSE_LONG and current_position <= 0:
                return ActionValidationResult(
                    is_valid=False,
                    message="Cannot close long position: no long position exists"
                )

            if trading_action.action_type == ActionType.CLOSE_SHORT and current_position >= 0:
                return ActionValidationResult(
                    is_valid=False,
                    message="Cannot close short position: no short position exists"
                )

            # Check market constraints
            if market_constraints:
                min_order_size = market_constraints.get('min_order_size', 0)
                max_order_size = market_constraints.get('max_order_size', float('inf'))

                if trading_action.size < min_order_size:
                    return ActionValidationResult(
                        is_valid=False,
                        message=f"Order size {trading_action.size:.4f} below minimum {min_order_size:.4f}"
                    )

                if trading_action.size > max_order_size:
                    adjusted_action = TradingAction(
                        action_type=trading_action.action_type,
                        asset=trading_action.asset,
                        size=max_order_size,
                        price=trading_action.price,
                        stop_loss=trading_action.stop_loss,
                        take_profit=trading_action.take_profit,
                        confidence=trading_action.confidence,
                        metadata=trading_action.metadata
                    )
                    warnings.append(f"Position size capped at maximum allowed: {max_order_size:.4f}")

            # Check stop loss and take profit levels
            if trading_action.stop_loss and trading_action.price:
                if trading_action.action_type == ActionType.BUY and trading_action.stop_loss >= trading_action.price:
                    warnings.append("Stop loss level is above entry price for buy order")
                elif trading_action.action_type == ActionType.SELL and trading_action.stop_loss <= trading_action.price:
                    warnings.append("Stop loss level is below entry price for sell order")

            return ActionValidationResult(
                is_valid=True,
                message="Action validation passed",
                adjusted_action=adjusted_action,
                warnings=warnings
            )

        except Exception as e:
            self.translation_stats['validation_failures'] += 1
            logger.error(f"Action validation failed: {e}")
            return ActionValidationResult(
                is_valid=False,
                message=f"Validation error: {str(e)}"
            )

    def get_action_space_info(self) -> Dict[str, Any]:
        """Get information about the action space."""
        if self.action_space_type == "discrete":
            return {
                'type': 'discrete',
                'n_actions': len(self.discrete_action_map),
                'action_meanings': {idx: action.name for idx, action in self.discrete_action_map.items()}
            }
        else:
            return {
                'type': 'continuous',
                'dimensions': 2,
                'dimension_meanings': ['action_type', 'confidence'],
                'ranges': [[-1.0, 1.0], [0.0, 1.0]]
            }

    def get_translation_stats(self) -> Dict[str, Any]:
        """Get translation statistics."""
        stats = self.translation_stats.copy()

        # Add success rate
        total = stats['total_translations']
        if total > 0:
            stats['success_rate'] = stats['successful_translations'] / total * 100
            stats['failure_rate'] = stats['failed_translations'] / total * 100
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0

        # Convert action type counts to readable format
        stats['action_type_counts'] = {
            action_type.name: count
            for action_type, count in stats['action_type_counts'].items()
        }

        return stats

    def reset_stats(self) -> None:
        """Reset translation statistics."""
        self.translation_stats = {
            'total_translations': 0,
            'successful_translations': 0,
            'failed_translations': 0,
            'action_type_counts': {action_type: 0 for action_type in ActionType},
            'validation_failures': 0,
            'size_adjustments': 0
        }
        logger.info("Translation statistics reset")

    def update_config(self, **kwargs) -> None:
        """Update translator configuration."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated {key} to {value}")
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
