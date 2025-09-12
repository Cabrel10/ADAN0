"""
Position sizing module for the ADAN Trading Bot.

This module provides advanced position sizing strategies for optimal
risk management and capital allocation.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class PositionSizingMethod(Enum):
    """Position sizing methods."""
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY_CRITERION = "kelly_criterion"
    RISK_PARITY = "risk_parity"
    OPTIMAL_F = "optimal_f"
    ATR_BASED = "atr_based"
    SHARPE_OPTIMIZED = "sharpe_optimized"


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""
    size: float
    method_used: PositionSizingMethod
    risk_amount: float
    confidence_level: float
    warnings: List[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RiskParameters:
    """Risk parameters for position sizing."""
    max_risk_per_trade: float = 0.02  # 2% max risk per trade
    max_portfolio_risk: float = 0.20  # 20% max total portfolio risk
    risk_free_rate: float = 0.02  # 2% risk-free rate
    confidence_level: float = 0.95  # 95% confidence level
    lookback_period: int = 252  # 1 year of trading days


class PositionSizer:
    """
    Advanced position sizing system.

    This class implements various position sizing strategies to optimize
    risk-adjusted returns and manage portfolio risk effectively.
    """

    def __init__(self,
                 default_method: PositionSizingMethod = PositionSizingMethod.PERCENTAGE,
                 risk_params: Optional[RiskParameters] = None,
                 enable_dynamic_sizing: bool = True,
                 min_position_size: float = 0.001,
                 max_position_size: float = 1.0,
                 sizing_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the PositionSizer.

        Args:
            default_method: Default position sizing method
            risk_params: Risk parameters for calculations
            enable_dynamic_sizing: Whether to use dynamic sizing
            min_position_size: Minimum position size
            max_position_size: Maximum position size
            sizing_config: Configuration for position sizing, including capital tiers.
        """
        self.default_method = default_method
        self.risk_params = risk_params or RiskParameters()
        self.enable_dynamic_sizing = enable_dynamic_sizing
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
        self.sizing_config = sizing_config
        self.capital_tiers = sizing_config.get('capital_tiers') if sizing_config else None

        # Historical data for calculations
        self.price_history = {}
        self.return_history = {}
        self.volatility_cache = {}

        # Position sizing statistics
        self.sizing_stats = {
            'total_calculations': 0,
            'method_usage': {method: 0 for method in PositionSizingMethod},
            'average_position_size': 0.0,
            'risk_adjusted_positions': 0
        }

        logger.info(f"PositionSizer initialized with {default_method.value} method")

    def calculate_position_size(self,
                              asset: str,
                              current_price: float,
                              portfolio_value: float,
                              available_capital: float,
                              market_data: Optional[Dict[str, Any]] = None,
                              method: Optional[PositionSizingMethod] = None,
                              confidence: float = 1.0) -> PositionSizeResult:
        """
        Calculate optimal position size.

        Args:
            asset: Asset symbol
            current_price: Current asset price
            portfolio_value: Total portfolio value
            available_capital: Available capital for trading
            market_data: Market data including volatility, returns, etc.
            method: Specific method to use (overrides default)
            confidence: Confidence in the trade (0-1)

        Returns:
            PositionSizeResult
        """
        self.sizing_stats['total_calculations'] += 1
        method = method or self.default_method
        self.sizing_stats['method_usage'][method] += 1

        warnings = []
        metadata = {}

        try:
            if method == PositionSizingMethod.FIXED:
                result = self._calculate_fixed_size(
                    portfolio_value, available_capital, confidence
                )
            elif method == PositionSizingMethod.PERCENTAGE:
                result = self._calculate_percentage_size(
                    portfolio_value, available_capital, confidence
                )
            elif method == PositionSizingMethod.VOLATILITY_ADJUSTED:
                result = self._calculate_volatility_adjusted_size(
                    asset, current_price, portfolio_value, available_capital,
                    market_data, confidence
                )
            elif method == PositionSizingMethod.KELLY_CRITERION:
                result = self._calculate_kelly_size(
                    asset, current_price, portfolio_value, available_capital,
                    market_data, confidence
                )
            elif method == PositionSizingMethod.RISK_PARITY:
                result = self._calculate_risk_parity_size(
                    asset, current_price, portfolio_value, available_capital,
                    market_data, confidence
                )
            elif method == PositionSizingMethod.ATR_BASED:
                result = self._calculate_atr_based_size(
                    asset, current_price, portfolio_value, available_capital,
                    market_data, confidence
                )
            else:
                # Fallback to percentage method
                result = self._calculate_percentage_size(
                    portfolio_value, available_capital, confidence
                )
                warnings.append(f"Unknown method {method}, using percentage")

            # Apply constraints
            constrained_size = self._apply_constraints(
                result['size'], current_price, available_capital
            )

            if constrained_size != result['size']:
                # Standardized sizer diagnostic for easy grep during integration runs
                try:
                    logger.warning(
                        "[SIZER] Adjusted position value: raw=%.8f -> adjusted=%.8f (price=%.8f, available_capital=%.2f)",
                        float(result['size']), float(constrained_size), float(current_price), float(available_capital)
                    )
                except Exception:
                    # Don't let logging issues affect sizing
                    pass
                warnings.append(f"Position size adjusted from {result['size']:.4f} to {constrained_size:.4f}")
                result['size'] = constrained_size

            # Calculate risk amount
            risk_amount = self._calculate_risk_amount(
                result['size'], current_price, market_data
            )

            # Update statistics
            self._update_stats(result['size'])

            return PositionSizeResult(
                size=result['size'],
                method_used=method,
                risk_amount=risk_amount,
                confidence_level=confidence,
                warnings=warnings,
                metadata={**metadata, **result.get('metadata', {})}
            )

        except Exception as e:
            logger.error(f"Position sizing calculation failed: {e}")
            # Return safe fallback
            fallback_size = min(
                available_capital * 0.1 / current_price,
                self.max_position_size
            )
            return PositionSizeResult(
                size=fallback_size,
                method_used=PositionSizingMethod.PERCENTAGE,
                risk_amount=fallback_size * current_price * 0.02,
                confidence_level=confidence,
                warnings=[f"Calculation failed, using fallback: {str(e)}"]
            )

    def _calculate_fixed_size(self,
                            portfolio_value: float,
                            available_capital: float,
                            confidence: float) -> Dict[str, Any]:
        """Calculate fixed position size."""
        base_size = 1000.0  # Fixed $1000 position
        adjusted_size = base_size * confidence

        return {
            'size': adjusted_size,
            'metadata': {
                'base_size': base_size,
                'confidence_adjustment': confidence
            }
        }

    def _get_tiered_percentage(self, portfolio_value: float) -> tuple[float, str]:
        """
        Determine position size percentage based on capital tiers.

        Args:
            portfolio_value: The total value of the portfolio.

        Returns:
            A tuple containing the percentage of the portfolio to allocate and the tier name.
        """
        if not self.capital_tiers:
            return 0.1, "Default"  # Fallback to default

        # Sort tiers to handle them in order
        sorted_tiers = sorted(self.capital_tiers, key=lambda x: x.get('min_capital', 0))

        for tier in sorted_tiers:
            min_capital = tier.get('min_capital')
            max_capital = tier.get('max_capital')
            tier_name = tier.get('name', 'Unnamed Tier')

            if min_capital is None:
                continue

            is_in_tier = False
            if max_capital is None:  # For the highest tier
                if portfolio_value >= min_capital:
                    is_in_tier = True
            elif min_capital <= portfolio_value < max_capital:
                is_in_tier = True

            if is_in_tier:
                return (tier.get('max_position_size_pct', 10) / 100.0), tier_name

        return 0.1, "Fallback" # Default if no tier is matched

    def _calculate_percentage_size(self,
                                 portfolio_value: float,
                                 available_capital: float,
                                 confidence: float) -> Dict[str, Any]:
        """
        Calculate percentage-based position size.
        Uses capital tiers if available, otherwise a fixed percentage.
        """
        base_percentage, tier_name = self._get_tiered_percentage(portfolio_value)

        metadata = {
            'tier_name': tier_name,
            'base_percentage': base_percentage,
            'portfolio_value': portfolio_value
        }

        adjusted_percentage = base_percentage * confidence
        position_value = portfolio_value * adjusted_percentage

        metadata['adjusted_percentage'] = adjusted_percentage

        return {
            'size': position_value,
            'metadata': metadata
        }

    def _calculate_volatility_adjusted_size(self,
                                          asset: str,
                                          current_price: float,
                                          portfolio_value: float,
                                          available_capital: float,
                                          market_data: Optional[Dict[str, Any]],
                                          confidence: float) -> Dict[str, Any]:
        """Calculate volatility-adjusted position size."""
        if not market_data or 'volatility' not in market_data:
            # Fallback to percentage method
            return self._calculate_percentage_size(
                portfolio_value, available_capital, confidence
            )

        volatility = market_data['volatility']
        target_volatility = 0.15  # 15% target portfolio volatility

        # Inverse volatility scaling
        if volatility > 0:
            vol_adjustment = target_volatility / volatility
            base_percentage = 0.1  # 10% base allocation
            adjusted_percentage = base_percentage * vol_adjustment * confidence

            # Cap the adjustment to reasonable bounds
            adjusted_percentage = max(0.01, min(0.5, adjusted_percentage))
            position_value = portfolio_value * adjusted_percentage
            vol_adjustment_final = vol_adjustment
        else:
            position_value = portfolio_value * 0.1 * confidence
            vol_adjustment_final = 1.0

        return {
            'size': position_value,
            'metadata': {
                'volatility': volatility,
                'target_volatility': target_volatility,
                'vol_adjustment': vol_adjustment_final,
                'adjusted_percentage': adjusted_percentage if volatility > 0 else 0.1
            }
        }

    def _calculate_kelly_size(self,
                            asset: str,
                            current_price: float,
                            portfolio_value: float,
                            available_capital: float,
                            market_data: Optional[Dict[str, Any]],
                            confidence: float) -> Dict[str, Any]:
        """Calculate Kelly criterion position size."""
        if not market_data:
            return self._calculate_percentage_size(
                portfolio_value, available_capital, confidence
            )

        expected_return = market_data.get('expected_return', 0.0)
        volatility = market_data.get('volatility', 0.1)
        win_rate = market_data.get('win_rate', 0.5)
        avg_win = market_data.get('avg_win', 0.02)
        avg_loss = market_data.get('avg_loss', 0.02)

        # Kelly formula: f = (bp - q) / b
        # where b = odds, p = win probability, q = loss probability
        if avg_loss > 0 and win_rate > 0:
            b = avg_win / avg_loss  # Odds
            p = win_rate  # Win probability
            q = 1 - win_rate  # Loss probability

            kelly_fraction = (b * p - q) / b
        else:
            # Alternative Kelly using expected return and volatility
            if volatility > 0:
                kelly_fraction = (expected_return - self.risk_params.risk_free_rate) / (volatility ** 2)
            else:
                kelly_fraction = 0.1

        # Apply conservative scaling (typically 25-50% of full Kelly)
        conservative_kelly = kelly_fraction * 0.25 * confidence

        # Ensure reasonable bounds
        conservative_kelly = max(0.01, min(0.3, conservative_kelly))
        position_value = portfolio_value * conservative_kelly

        return {
            'size': position_value,
            'metadata': {
                'kelly_fraction': kelly_fraction,
                'conservative_kelly': conservative_kelly,
                'expected_return': expected_return,
                'volatility': volatility,
                'win_rate': win_rate
            }
        }

    def _calculate_risk_parity_size(self,
                                  asset: str,
                                  current_price: float,
                                  portfolio_value: float,
                                  available_capital: float,
                                  market_data: Optional[Dict[str, Any]],
                                  confidence: float) -> Dict[str, Any]:
        """Calculate risk parity position size."""
        if not market_data or 'volatility' not in market_data:
            return self._calculate_percentage_size(
                portfolio_value, available_capital, confidence
            )

        volatility = market_data['volatility']
        target_risk = self.risk_params.max_risk_per_trade

        # Risk parity: position size inversely proportional to volatility
        if volatility > 0:
            # Calculate position size to achieve target risk
            risk_adjusted_size = (target_risk * portfolio_value) / (volatility * current_price)
            position_value = risk_adjusted_size * current_price * confidence
        else:
            position_value = portfolio_value * 0.1 * confidence

        return {
            'size': position_value,
            'metadata': {
                'volatility': volatility,
                'target_risk': target_risk,
                'risk_adjusted_size': risk_adjusted_size if volatility > 0 else 0
            }
        }

    def _calculate_atr_based_size(self,
                                asset: str,
                                current_price: float,
                                portfolio_value: float,
                                available_capital: float,
                                market_data: Optional[Dict[str, Any]],
                                confidence: float) -> Dict[str, Any]:
        """Calculate ATR-based position size."""
        if not market_data or 'atr' not in market_data:
            return self._calculate_percentage_size(
                portfolio_value, available_capital, confidence
            )

        atr = market_data['atr']
        risk_per_trade = portfolio_value * self.risk_params.max_risk_per_trade

        # Position size based on ATR stop loss
        if atr > 0:
            # Assume stop loss at 2 * ATR
            stop_distance = 2 * atr
            position_size_shares = risk_per_trade / stop_distance
            position_value = position_size_shares * current_price * confidence
        else:
            position_value = portfolio_value * 0.1 * confidence

        return {
            'size': position_value,
            'metadata': {
                'atr': atr,
                'stop_distance': 2 * atr if atr > 0 else 0,
                'risk_per_trade': risk_per_trade
            }
        }

    def _apply_constraints(self,
                         position_size: float,
                         current_price: float,
                         available_capital: float) -> float:
        """Apply position size constraints."""
        # Position size is already in value (dollars)
        position_value = position_size

        # Apply minimum and maximum constraints
        min_value = self.min_position_size * available_capital
        max_value = self.max_position_size * available_capital

        constrained_value = max(min_value, min(max_value, position_value))

        # Ensure we don't exceed available capital
        constrained_value = min(constrained_value, available_capital * 0.95)

        return constrained_value

    def _calculate_risk_amount(self,
                             position_size: float,
                             current_price: float,
                             market_data: Optional[Dict[str, Any]]) -> float:
        """Calculate the risk amount for the position."""
        if not market_data:
            # Default 2% risk assumption
            return position_size * 0.02

        volatility = market_data.get('volatility', 0.02)
        atr = market_data.get('atr', current_price * 0.02)

        # Use ATR if available, otherwise use volatility
        if atr and atr > 0:
            # Assume 2 ATR stop loss
            risk_amount = (position_size / current_price) * (2 * atr)
        else:
            # Use volatility-based risk
            risk_amount = position_size * volatility

        return risk_amount

    def _update_stats(self, position_size: float):
        """Update position sizing statistics."""
        total_calcs = self.sizing_stats['total_calculations']
        current_avg = self.sizing_stats['average_position_size']

        # Update running average
        self.sizing_stats['average_position_size'] = (
            (current_avg * (total_calcs - 1) + position_size) / total_calcs
        )

    def optimize_position_size(self,
                             asset: str,
                             current_price: float,
                             portfolio_value: float,
                             available_capital: float,
                             market_data: Optional[Dict[str, Any]] = None,
                             confidence: float = 1.0) -> PositionSizeResult:
        """
        Optimize position size using multiple methods and select the best.

        Args:
            asset: Asset symbol
            current_price: Current asset price
            portfolio_value: Total portfolio value
            available_capital: Available capital
            market_data: Market data
            confidence: Trade confidence

        Returns:
            PositionSizeResult with optimized size
        """
        if not self.enable_dynamic_sizing:
            return self.calculate_position_size(
                asset, current_price, portfolio_value, available_capital,
                market_data, self.default_method, confidence
            )

        # Calculate using multiple methods
        methods_to_try = [
            PositionSizingMethod.PERCENTAGE,
            PositionSizingMethod.VOLATILITY_ADJUSTED,
            PositionSizingMethod.KELLY_CRITERION,
            PositionSizingMethod.RISK_PARITY
        ]

        results = []
        for method in methods_to_try:
            try:
                result = self.calculate_position_size(
                    asset, current_price, portfolio_value, available_capital,
                    market_data, method, confidence
                )
                results.append((method, result))
            except Exception as e:
                logger.warning(f"Method {method} failed: {e}")

        if not results:
            # Fallback to default method
            return self.calculate_position_size(
                asset, current_price, portfolio_value, available_capital,
                market_data, self.default_method, confidence
            )

        # Select the method with the best risk-adjusted size
        best_method, best_result = self._select_best_method(results, market_data)

        best_result.metadata = best_result.metadata or {}
        best_result.metadata['optimization_used'] = True
        best_result.metadata['methods_evaluated'] = len(results)

        return best_result

    def _select_best_method(self,
                          results: List[tuple],
                          market_data: Optional[Dict[str, Any]]) -> tuple:
        """Select the best position sizing method from results."""
        if len(results) == 1:
            return results[0]

        # Score each method based on risk-adjusted criteria
        scored_results = []
        for method, result in results:
            score = self._score_position_size(result, market_data)
            scored_results.append((score, method, result))

        # Return the highest scoring method
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return scored_results[0][1], scored_results[0][2]

    def _score_position_size(self,
                           result: PositionSizeResult,
                           market_data: Optional[Dict[str, Any]]) -> float:
        """Score a position size result."""
        score = 0.0

        # Prefer moderate position sizes (not too small, not too large)
        size_score = 1.0 - abs(result.size - 0.1) / 0.1
        score += size_score * 0.3

        # Prefer lower risk amounts
        if result.risk_amount > 0:
            risk_score = max(0, 1.0 - result.risk_amount / (result.size * 0.05))
            score += risk_score * 0.4

        # Prefer higher confidence
        score += result.confidence_level * 0.2

        # Penalty for warnings
        if result.warnings:
            score -= len(result.warnings) * 0.1

        return max(0.0, score)

    def get_sizing_stats(self) -> Dict[str, Any]:
        """Get position sizing statistics."""
        return {
            'total_calculations': self.sizing_stats['total_calculations'],
            'method_usage': {
                method.value: count
                for method, count in self.sizing_stats['method_usage'].items()
            },
            'average_position_size': self.sizing_stats['average_position_size'],
            'risk_adjusted_positions': self.sizing_stats['risk_adjusted_positions']
        }

    def update_market_data(self, asset: str, price_data: List[float]):
        """Update historical market data for calculations."""
        self.price_history[asset] = price_data[-self.risk_params.lookback_period:]

        # Calculate returns
        if len(price_data) > 1:
            returns = []
            for i in range(1, len(price_data)):
                ret = (price_data[i] - price_data[i-1]) / price_data[i-1]
                returns.append(ret)
            self.return_history[asset] = returns

            # Calculate volatility
            if len(returns) > 1:
                mean_return = sum(returns) / len(returns)
                variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
                self.volatility_cache[asset] = math.sqrt(variance * 252)  # Annualized

        logger.debug(f"Updated market data for {asset}")
