#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reward calculation module for the ADAN trading bot.

This module defines the logic for calculating the reward signal that guides the
reinforcement learning agent.
"""

import numpy as np
from typing import Dict, Any
import logging
from ..common.reward_logger import RewardLogger
from enum import Enum

logger = logging.getLogger(__name__)

class RewardCalculator:
    """
    Calculates the reward for a given step in the trading environment.

    The reward function is designed to guide the agent towards profitable and
    consistent trading behavior.
    """
    def __init__(self, env_config: Dict[str, Any]):
        """
        Initializes the RewardCalculator.

        Args:
            env_config: The environment configuration dictionary, containing the
                        `reward_shaping` section.
        """
        self.config = env_config.get('reward_shaping', {})
        self.pnl_multiplier = self.config.get('realized_pnl_multiplier', 1.0)
        self.unrealized_pnl_multiplier = self.config.get('unrealized_pnl_multiplier', 0.1)
        self.inaction_penalty = self.config.get('inaction_penalty', -0.0001)
        self.clipping_range = self.config.get('reward_clipping_range', [-5.0, 5.0])

        # Commission and profit threshold parameters
        self.commission_penalty = self.config.get('commission_penalty', 1.5)  # Multiplier for commission penalty
        self.min_profit_multiplier = self.config.get('min_profit_multiplier', 3.0)  # Minimum profit multiple of commission

        # Chunk-based reward parameters
        self.optimal_trade_bonus = self.config.get('optimal_trade_bonus', 1.0)
        self.performance_threshold = self.config.get('performance_threshold', 0.8)  # 80% of optimal

        # Track chunk information
        self.current_chunk_id = 0
        self.chunk_rewards = {}

        # Initialize reward logger
        self.reward_logger = RewardLogger(env_config)

        # Episode tracking for detailed logging
        self.current_episode_rewards = []
        self.current_episode_id = 0

        # DBE (Dynamic Budgeting Engine) parameters
        self.winrate = 0.5  # Will be updated based on performance
        self.drawdown = 0.0  # Will be updated based on portfolio metrics
        self.risk_level = 0.3  # Initial risk level (0.1 to 1.0)

        # Multi-objective optimization parameters
        self.returns_history = []  # Store historical returns for ratio calculations
        self.returns_dates = []    # Store timestamps for time-weighted calculations
        self.max_lookback = 252    # Maximum number of returns to store (1 year of daily data)
        self.risk_free_rate = 0.01  # Annual risk-free rate (1%)
        self.annual_trading_days = 365  # Number of trading days in a year (crypto 24/7)
        self.decay_factor = 0.99   # Decay factor for time-weighted calculations

        # Weights for composite reward calculation
        self.weights = {
            'pnl': 0.4,           # Base PnL component
            'sharpe': 0.25,       # Risk-adjusted return (volatility)
            'sortino': 0.25,      # Downside risk-adjusted return
            'calmar': 0.1         # Drawdown-adjusted return
        }

        # Risk management formula bonus parameters
        self.kelly_bonus_weight = 0.1  # Bonus weight for respecting Kelly criterion
        self.risk_parity_bonus_weight = 0.05  # Bonus weight for respecting risk parity
        self.stress_var_penalty_weight = 0.15  # Penalty weight for exceeding stress VaR

        # Cache for expensive calculations
        self._ratio_cache = {}
        self._last_calculation_time = 0

        logger.info("RewardCalculator initialized with multi-objective optimization and detailed logging.")

    def _calculate_kelly_bonus(self, position_metadata: Dict[str, Any]) -> float:
        """
        Calculate bonus for respecting Kelly criterion.

        Args:
            position_metadata: Metadata from position sizing including Kelly information

        Returns:
            Kelly bonus (positive if Kelly criterion is respected)
        """
        kelly_respected = position_metadata.get('kelly_respected', False)
        if kelly_respected:
            return self.kelly_bonus_weight
        return 0.0

    def _calculate_risk_parity_bonus(self, position_metadata: Dict[str, Any]) -> float:
        """
        Calculate bonus for respecting risk parity.

        Args:
            position_metadata: Metadata from position sizing including risk parity information

        Returns:
            Risk parity bonus (positive if risk parity is respected)
        """
        risk_respected = position_metadata.get('risk_respected', False)
        if risk_respected:
            return self.risk_parity_bonus_weight
        return 0.0

    def _calculate_stress_var_penalty(self, portfolio_metrics: Dict[str, Any]) -> float:
        """
        Calculate penalty for exceeding stress VaR thresholds.

        Args:
            portfolio_metrics: Portfolio risk metrics including stress VaR

        Returns:
            Stress VaR penalty (negative if exceeding thresholds)
        """
        stress_var_99 = portfolio_metrics.get('stress_var_0.99', 0.0)
        var_threshold = 0.1  # 10% stress VaR threshold

        if stress_var_99 > var_threshold:
            excess_var = stress_var_99 - var_threshold
            penalty = -self.stress_var_penalty_weight * excess_var
            return penalty
        return 0.0

    def _log_reward_components(self, components: Dict[str, Any]) -> None:
        """
        Log detailed information about reward components.

        Args:
            components: Dictionary containing all reward components
        """
        try:
            # Log to debug
            logger.debug(
                f"REWARD COMPONENTS | "
                f"Base: {components['base_reward']:.4f} | "
                f"Commission: -{components['commission_penalty']:.4f} | "
                f"Chunk: +{components['chunk_bonus']:.4f} | "
                f"Sharpe: {components['sharpe_ratio']:.2f} | "
                f"Sortino: {components['sortino_ratio']:.2f} | "
                f"Kelly Bonus: {components.get('kelly_bonus', 0):.4f} | "
                f"Risk Parity Bonus: {components.get('risk_parity_bonus', 0):.4f} | "
                f"Stress VaR Penalty: {components.get('stress_var_penalty', 0):.4f} | "
                f"Calmar: {components['calmar_ratio']:.2f} | "
                f"Drawdown: {components['drawdown']:.2%} | "
                f"Action: {components['action']} | "
                f"Final: {components['final_reward']:.4f}"
            )

            # Log to reward logger for analysis
            self.reward_logger.log_reward_calculation({
                'total_reward': components['final_reward'],
                'components': {
                    'realized_pnl': components['base_reward'],
                    'commission_penalty': -components['commission_penalty'],
                    'chunk_bonus': components['chunk_bonus'],
                    'sharpe_ratio': components['sharpe_ratio'],
                    'sortino_ratio': components['sortino_ratio'],
                    'calmar_ratio': components['calmar_ratio'],
                    'drawdown_penalty': components.get('drawdown_penalty', 0.0),
                    'inaction_penalty': components.get('inaction_penalty', 0.0),
                    'action': components['action'],
                    'trade_pnl': components['trade_pnl']
                }
            })

        except Exception as e:
            logger.error(f"Error logging reward components: {str(e)}")

    def calculate(self, portfolio_metrics: Dict[str, Any], trade_pnl: float, action: int,
                 chunk_id: int = None, optimal_chunk_pnl: float = None, performance_ratio: float = None, is_hunting: bool = False) -> float:
        """
        Calculate the total reward for the current timestep using multi-objective optimization.

        The reward is a weighted combination of:
        - Base PnL (profit and loss)
        - Risk-adjusted returns (Sharpe ratio)
        - Downside risk-adjusted returns (Sortino ratio)
        - Drawdown-adjusted returns (Calmar ratio)
        - Chunk-based performance bonuses

        Args:
            portfolio_metrics: A dictionary of performance metrics from the PortfolioManager.
            trade_pnl: The realized profit or loss from a trade executed in the current step.
            action: The action taken by the agent (0: Hold, 1: Buy, 2: Sell).
            chunk_id: The current chunk ID for chunk-based rewards.
            optimal_chunk_pnl: The optimal possible PnL for the current chunk.
            performance_ratio: The performance ratio for the current chunk (actual_pnl / optimal_pnl).

        Returns:
            float: The total reward for the current timestep, clipped to the configured range.
        """
        try:
            # Update returns history if this is a new trade
            if trade_pnl != 0:
                self._update_returns_history(trade_pnl)

            # 1. Calculate base reward components
            commission = portfolio_metrics.get('total_commission', 0.0)
            commission_penalty = commission * self.commission_penalty
            base_reward = (trade_pnl - commission_penalty) * self.pnl_multiplier

            # 2. Check minimum profit threshold
            drawdown_penalty = 0.0
            min_profit = self.min_profit_multiplier * commission
            if trade_pnl > 0 and commission > 0 and trade_pnl < min_profit:
                # Penalize trades that don't meet minimum profit threshold
                drawdown_penalty = (min_profit - trade_pnl) * 2
                base_reward -= drawdown_penalty
                logger.debug(f"Trade PnL ({trade_pnl}) below minimum threshold ({self.min_profit_multiplier}x commission = {min_profit}), penalty: {drawdown_penalty:.4f}")

            # 3. Calculate chunk-based performance bonus if applicable
            chunk_bonus = 0.0
            if (chunk_id is not None and optimal_chunk_pnl is not None and optimal_chunk_pnl > 0 and
                chunk_id != self.current_chunk_id):
                self.current_chunk_id = chunk_id

                if performance_ratio is not None and performance_ratio >= self.performance_threshold:
                    chunk_bonus = self.optimal_trade_bonus * (performance_ratio - self.performance_threshold)

                    # Store chunk rewards for analysis
                    self.chunk_rewards[chunk_id] = {
                        'optimal_pnl': optimal_chunk_pnl,
                        'performance_ratio': performance_ratio,
                        'bonus': chunk_bonus
                    }

                    logger.debug(
                        f"Chunk {chunk_id} performance bonus: {chunk_bonus:.4f} "
                        f"(Ratio: {performance_ratio:.2f}, Optimal PnL: {optimal_chunk_pnl:.2f}%)"
                    )

            # 4. Calculate risk metrics
            drawdown = portfolio_metrics.get('drawdown', 0.0)

            # 5. Calculate advanced performance metrics if we have enough data
            if len(self.returns_history) >= 5:  # Minimum 5 data points for meaningful calculations
                # Calculate all performance ratios
                sharpe_ratio = self._calculate_sharpe_ratio()
                sortino_ratio = self._calculate_sortino_ratio()
                calmar_ratio = self._calculate_calmar_ratio(portfolio_metrics)

                # 6. Calculate risk management bonuses
                position_metadata = portfolio_metrics.get('position_metadata', {})
                kelly_bonus = self._calculate_kelly_bonus(position_metadata)
                risk_parity_bonus = self._calculate_risk_parity_bonus(position_metadata)
                stress_var_penalty = self._calculate_stress_var_penalty(portfolio_metrics)

                # Calculate composite score using weighted sum of components
                composite_score = (
                    self.weights['pnl'] * base_reward +
                    self.weights['sharpe'] * sharpe_ratio +
                    self.weights['sortino'] * sortino_ratio +
                    self.weights['calmar'] * calmar_ratio +
                    chunk_bonus +  # Add chunk bonus as an additional component
                    kelly_bonus +  # Bonus for respecting Kelly criterion
                    risk_parity_bonus +  # Bonus for respecting risk parity
                    stress_var_penalty  # Penalty for exceeding stress VaR
                )

                # Apply drawdown penalty (outside the composite score to maintain scale)
                if drawdown < -0.05:  # If drawdown is worse than -5%
                    drawdown_penalty = abs(drawdown) * 10
                    composite_score -= drawdown_penalty
                    logger.debug(f"Applied drawdown penalty: {drawdown_penalty:.4f}")

                # Apply inaction penalty for hold actions
                inaction_penalty = 0.0
                if action == 0 and not is_hunting:  # Hold action
                    inaction_penalty = self.inaction_penalty
                    composite_score += inaction_penalty
                    logger.debug(f"Applied inaction penalty (not hunting): {inaction_penalty:.4f}")

                # Log detailed metrics
                self._log_reward_components({
                    'base_reward': base_reward,
                    'commission_penalty': commission_penalty,
                    'chunk_bonus': chunk_bonus,
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': sortino_ratio,
                    'calmar_ratio': calmar_ratio,
                    'drawdown': drawdown,
                    'drawdown_penalty': drawdown_penalty,
                    'inaction_penalty': inaction_penalty,
                    'kelly_bonus': kelly_bonus,
                    'risk_parity_bonus': risk_parity_bonus,
                    'stress_var_penalty': stress_var_penalty,
                    'action': action,
                    'trade_pnl': trade_pnl,
                    'final_reward': composite_score
                })

                # Final reward is the composite score
                final_reward = composite_score

            else:
                # Not enough data for advanced metrics, use simple reward
                final_reward = base_reward + chunk_bonus

                # Apply inaction penalty for hold actions
                inaction_penalty = 0.0
                if action == 0 and not is_hunting:  # Hold action
                    inaction_penalty = self.inaction_penalty
                    final_reward += inaction_penalty

            # Update DBE parameters based on performance
            self._update_dbe_parameters(portfolio_metrics)

            # Track episode rewards for logging
            self.current_episode_rewards.append(final_reward)

            # Clip the final reward to prevent extreme values
            final_reward = np.clip(final_reward, *self.clipping_range)

            return float(final_reward)

        except Exception as e:
            logger.error(f"Error in reward calculation: {str(e)}")
            # Return a neutral reward in case of errors
            return 0.0

        # Update DBE parameters based on performance
        self._update_dbe_parameters(portfolio_metrics)

        # Log DBE metrics
        logger.info(
            f"DBE ADAPT | Winrate: {self.winrate:.2%} | "
            f"Drawdown: {self.drawdown:.2%} | "
            f"Risk Level: {self.risk_level:.2f}"
        )

        # Log detailed reward calculation
        self.reward_logger.log_reward_calculation({
            'total_reward': reward,
            'components': reward_components,
            'dbe_metrics': {
                'winrate': self.winrate,
                'drawdown': self.drawdown,
                'risk_level': self.risk_level,
                'commission_penalty': commission_penalty
            },
            'metadata': {
                'action': action,
                'trade_pnl': trade_pnl,
                'drawdown': drawdown,
                'sharpe_ratio': sharpe_ratio,
                'chunk_id': chunk_id,
                'performance_ratio': performance_ratio,
                'optimal_chunk_pnl': optimal_chunk_pnl,
                'clipped': bool(reward != np.sum(list(reward_components.values())))
            }
        })

        # Check if trade meets minimum profit threshold
        if trade_pnl > 0 and commission > 0:
            min_profit = self.min_profit_multiplier * commission
            if trade_pnl < min_profit:
                # Penalize trades that don't meet minimum profit threshold
                penalty = (min_profit - trade_pnl) * 2
                reward_before_penalty = reward
                reward -= penalty
                logger.debug(
                    f"Trade PnL ({trade_pnl}) below minimum threshold "
                    f"({self.min_profit_multiplier}x commission = {min_profit})"
                )
                logger.debug(
                    f"Applying penalty: {penalty} (reward before: {reward_before_penalty}, "
                    f"after: {reward})"
                )

        # Small penalty for inaction to encourage trading when opportunities exist
        if action == 0:  # Hold action
            reward += self.inaction_penalty

        # Add chunk-based performance bonus if chunk information is provided
        if chunk_id is not None and optimal_chunk_pnl is not None and optimal_chunk_pnl > 0:
            # Only add the bonus once per chunk
            if chunk_id != self.current_chunk_id:
                self.current_chunk_id = chunk_id

                # Add bonus if performance exceeds threshold
                if performance_ratio is not None and performance_ratio >= self.performance_threshold:
                    bonus = self.optimal_trade_bonus * (performance_ratio - self.performance_threshold)
                    reward += bonus

                    # Store chunk rewards for analysis
                    self.chunk_rewards[chunk_id] = {
                        'optimal_pnl': optimal_chunk_pnl,
                        'performance_ratio': performance_ratio,
                        'bonus': bonus
                    }

                    logger.info(
                        f"Chunk {chunk_id} performance bonus: {bonus:.4f} "
                        f"(Ratio: {performance_ratio:.2f}, Optimal PnL: {optimal_chunk_pnl:.2f}%)"
                    )

        # Incorporate risk metrics into reward
        drawdown = portfolio_metrics.get('drawdown', 0.0)
        sharpe_ratio = portfolio_metrics.get('sharpe_ratio', 0.0)

        # Track episode rewards
        self.current_episode_rewards.append(reward)

        # Return the final reward
        return reward

        episode_data = {
            'total_reward': np.sum(episode_rewards),
            'average_reward': np.mean(episode_rewards),
            'reward_std': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'episode_length': len(episode_rewards),
            'performance_bonuses': sum(1 for r in self.chunk_rewards.values() if r.get('bonus', 0) > 0),
            'risk_penalties': sum(1 for r in episode_rewards if r < -0.1)
        }

        # Logger l'épisode
        self.reward_logger.log_episode_reward(self.current_episode_id, episode_data)

        # Réinitialiser pour le prochain épisode
        self.current_episode_rewards = []
        self.current_episode_id += 1

        logger.info(f"Episode {self.current_episode_id - 1} finalized: "
                   f"Total reward: {episode_data['total_reward']:.4f}, "
                   f"Length: {episode_data['episode_length']}")

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
        from datetime import datetime

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
        weights = np.array([self.decay_factor ** i for i in reversed(range(n))])

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
        cache_key = f'sharpe_{risk_free_rate}'
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
            weighted_std = np.sqrt(np.average(
                (excess_returns - weighted_mean) ** 2,
                weights=weights
            ))

            # Avoid division by zero
            if weighted_std < 1e-10:
                return 0.0

            # Annualize the ratio
            sharpe_ratio = (weighted_mean / weighted_std) * np.sqrt(self.annual_trading_days)

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
        cache_key = f'sortino_{risk_free_rate}'
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
                return float('inf')  # No downside risk

            # Calculate weighted mean of squared downside returns
            weighted_variance = np.average(
                downside_returns**2,
                weights=weights
            )
            downside_std = np.sqrt(weighted_variance)

            # Avoid division by zero
            if downside_std < 1e-10:
                return 0.0

            # Annualize the ratio
            sortino_ratio = (weighted_mean / downside_std) * np.sqrt(self.annual_trading_days)

            # Cache the result
            self._ratio_cache[cache_key] = float(sortino_ratio)
            return self._ratio_cache[cache_key]

        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0.0

    def _calculate_calmar_ratio(self, portfolio_metrics: Dict[str, Any],
                              lookback_period: int = 36) -> float:
        """
        Calculate the time-weighted Calmar ratio based on the maximum drawdown.

        Args:
            portfolio_metrics: Current portfolio metrics including drawdown
            lookback_period: Lookback period in months (default: 36 months)

        Returns:
            float: The time-weighted Calmar ratio
        """
        cache_key = f'calmar_{lookback_period}'
        if cache_key in self._ratio_cache:
            return self._ratio_cache[cache_key]

        try:
            # Get the maximum drawdown from portfolio metrics
            max_drawdown = abs(portfolio_metrics.get('max_drawdown', 0.0))

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
            if len(recent_returns) > lookback_period * 21:  # Approximate trading days in lookback
                recent_returns = recent_returns[-(lookback_period * 21):]
                weights = weights[-(lookback_period * 21):]

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
            self.winrate = portfolio_metrics.get('win_rate', self.winrate)
            self.drawdown = portfolio_metrics.get('drawdown', self.drawdown)
            cash_utilization = portfolio_metrics.get('cash_utilization', 0.0)

            # Calculate volatility-based adjustment (higher vol → lower risk)
            returns_vol = np.std(self.returns_history) * np.sqrt(252) if self.returns_history else 0.0
            vol_adjustment = 1.0 / (1.0 + returns_vol)  # 0.5 for vol=1.0, 1.0 for vol=0.0

            # Calculate drawdown-based adjustment
            drawdown_adjustment = 1.0 - min(0.5, self.drawdown)

            # Calculate utilization-based adjustment
            util_adjustment = 0.5 + cash_utilization * 0.5

            # Combine adjustments with winrate
            self.risk_level = max(0.1, min(1.0,
                self.winrate * drawdown_adjustment * util_adjustment * vol_adjustment
            ))

            # Update max position size based on risk level
            if 'max_position_size_pct' in portfolio_metrics:
                portfolio_metrics['max_position_size_pct'] *= self.risk_level

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
            MarketRegime.TRENDING: -0.20,
            MarketRegime.VOLATILE: -0.25,
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
             next_base = self._base_penalty.get(
                 new_regime, self.inaction_penalty
             )

             # Smooth transition
             self.inaction_penalty = prev_base + strength * (next_base - prev_base)

             # Update state
             self.current_regime = new_regime
             self.regime_strength = strength
         except Exception:
             logging.getLogger(__name__).exception(
                 "update_market_regime failed"
             )

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
