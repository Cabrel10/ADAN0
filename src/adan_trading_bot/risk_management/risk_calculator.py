#!/usr/bin/env python3
"""
Risk calculation module for the ADAN trading bot.

This module provides comprehensive risk assessment and calculation functionality,
including various risk metrics, portfolio risk analysis, and risk-adjusted returns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RiskCalculator:
    """
    Comprehensive risk calculation and assessment system.

    Provides various risk metrics including:
    - Value at Risk (VaR) and Conditional VaR (CVaR)
    - Maximum Drawdown and recovery metrics
    - Volatility and correlation analysis
    - Risk-adjusted return metrics (Sharpe, Sortino, Calmar)
    - Portfolio risk decomposition
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the risk calculator.

        Args:
            config: Configuration dictionary containing risk parameters
        """
        self.config = config.get('risk_calculation', {})
        self.confidence_levels = self.config.get('confidence_levels', [0.95, 0.99])
        self.lookback_periods = self.config.get('lookback_periods', [30, 60, 365])  # 365 for crypto
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)  # 2% annual

        # Risk thresholds
        self.max_drawdown_threshold = self.config.get('max_drawdown_threshold', 0.2)
        self.var_threshold = self.config.get('var_threshold', 0.05)
        self.volatility_threshold = self.config.get('volatility_threshold', 0.3)

        logger.info("RiskCalculator initialized with comprehensive risk metrics")

    def calculate_var(self, returns: List[float], confidence_level: float = 0.95,
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: List of return values
            confidence_level: Confidence level for VaR calculation
            method: Method to use ('historical', 'parametric', 'monte_carlo')

        Returns:
            VaR value (positive number representing potential loss)
        """
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)

        if method == 'historical':
            # Historical simulation method - improved for crypto fat tails
            var_index = int(len(returns_array) * (1 - confidence_level))
            sorted_returns = np.sort(returns_array)
            var = -sorted_returns[var_index] if var_index < len(sorted_returns) else 0.0

            # Stress test for crypto: use 99% percentile for extreme scenarios
            if confidence_level >= 0.95:
                stress_var_index = int(len(returns_array) * 0.01)  # 99% percentile
                stress_var = -sorted_returns[stress_var_index] if stress_var_index < len(sorted_returns) else var
                var = max(var, stress_var)  # Use the more conservative estimate

        elif method == 'parametric':
            # Parametric method (assumes normal distribution)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(mean_return + z_score * std_return)

        elif method == 'monte_carlo':
            # Monte Carlo simulation
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)

            # Generate random scenarios
            n_simulations = 10000
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            var_index = int(n_simulations * (1 - confidence_level))
            var = -np.sort(simulated_returns)[var_index]

        else:
            raise ValueError(f"Unknown VaR method: {method}")

        return max(0.0, var)

    def calculate_stress_var(self, returns: List[float], percentiles: List[float] = [0.99, 0.995, 0.999]) -> Dict[str, float]:
        """
        Calculate stress test VaR for crypto extreme scenarios.

        Args:
            returns: List of return values
            percentiles: List of percentiles for stress testing

        Returns:
            Dictionary with stress test VaR values
        """
        if len(returns) < 10:
            return {f'stress_var_{p}': 0.0 for p in percentiles}

        returns_array = np.array(returns)
        sorted_returns = np.sort(returns_array)
        stress_vars = {}

        for percentile in percentiles:
            stress_index = int(len(returns_array) * (1 - percentile))
            stress_var = -sorted_returns[stress_index] if stress_index < len(sorted_returns) else 0.0
            stress_vars[f'stress_var_{percentile}'] = max(0.0, stress_var)

        return stress_vars

    def calculate_cvar(self, returns: List[float], confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

        Args:
            returns: List of return values
            confidence_level: Confidence level for CVaR calculation

        Returns:
            CVaR value (positive number representing expected loss beyond VaR)
        """
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        var = self.calculate_var(returns, confidence_level)

        # Calculate CVaR as the mean of returns worse than VaR
        tail_returns = returns_array[returns_array <= -var]

        if len(tail_returns) > 0:
            cvar = -np.mean(tail_returns)
        else:
            cvar = var

        return max(0.0, cvar)

    def calculate_maximum_drawdown(self, equity_curve: List[float]) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.

        Args:
            equity_curve: List of portfolio values over time

        Returns:
            Dictionary containing drawdown metrics
        """
        if len(equity_curve) < 2:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'drawdown_duration': 0,
                'recovery_time': 0,
                'current_drawdown': 0.0
            }

        equity_array = np.array(equity_curve)

        # Calculate running maximum (peak)
        running_max = np.maximum.accumulate(equity_array)

        # Calculate drawdown
        drawdown = equity_array - running_max
        drawdown_pct = drawdown / running_max

        # Find maximum drawdown
        max_drawdown_idx = np.argmin(drawdown_pct)
        max_drawdown = abs(drawdown[max_drawdown_idx])
        max_drawdown_pct = abs(drawdown_pct[max_drawdown_idx])

        # Calculate drawdown duration
        peak_idx = np.argmax(running_max[:max_drawdown_idx + 1])
        drawdown_duration = max_drawdown_idx - peak_idx

        # Calculate recovery time
        recovery_time = 0
        if max_drawdown_idx < len(equity_array) - 1:
            peak_value = running_max[max_drawdown_idx]
            for i in range(max_drawdown_idx + 1, len(equity_array)):
                if equity_array[i] >= peak_value:
                    recovery_time = i - max_drawdown_idx
                    break
            else:
                recovery_time = len(equity_array) - max_drawdown_idx - 1

        # Current drawdown
        current_drawdown = abs(drawdown_pct[-1])

        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'drawdown_duration': drawdown_duration,
            'recovery_time': recovery_time,
            'current_drawdown': current_drawdown
        }

    def calculate_sharpe_ratio(self, returns: List[float],
                              risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: List of return values
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        returns_array = np.array(returns)

        # Convert to excess returns
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate

        if np.std(excess_returns) == 0:
            return 0.0

        # Annualized Sharpe ratio - 365 days for crypto (24/7 trading)
        sharpe = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(365)

        return sharpe

    def calculate_sortino_ratio(self, returns: List[float],
                               risk_free_rate: Optional[float] = None,
                               target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio (downside deviation version of Sharpe).

        Args:
            returns: List of return values
            risk_free_rate: Risk-free rate (annualized)
            target_return: Target return threshold

        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0

        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)

        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < target_return]

        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0

        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))

        if downside_deviation == 0:
            return 0.0

        # Annualized Sortino ratio
        sortino = (np.mean(excess_returns) / downside_deviation) * np.sqrt(252)

        return sortino

    def calculate_calmar_ratio(self, returns: List[float],
                              equity_curve: List[float]) -> float:
        """
        Calculate Calmar ratio (annual return / maximum drawdown).

        Args:
            returns: List of return values
            equity_curve: List of portfolio values

        Returns:
            Calmar ratio
        """
        if len(returns) < 2 or len(equity_curve) < 2:
            return 0.0

        # Annualized return
        annual_return = np.mean(returns) * 252

        # Maximum drawdown
        dd_metrics = self.calculate_maximum_drawdown(equity_curve)
        max_drawdown = dd_metrics['max_drawdown_pct']

        if max_drawdown == 0:
            return float('inf') if annual_return > 0 else 0.0

        calmar = annual_return / max_drawdown

        return calmar

    def calculate_volatility_metrics(self, returns: List[float]) -> Dict[str, float]:
        """
        Calculate various volatility metrics.

        Args:
            returns: List of return values

        Returns:
            Dictionary containing volatility metrics
        """
        if len(returns) < 2:
            return {
                'volatility': 0.0,
                'annualized_volatility': 0.0,
                'upside_volatility': 0.0,
                'downside_volatility': 0.0,
                'volatility_skew': 0.0
            }

        returns_array = np.array(returns)

        # Basic volatility
        volatility = np.std(returns_array)
        annualized_volatility = volatility * np.sqrt(252)

        # Upside and downside volatility
        mean_return = np.mean(returns_array)
        upside_returns = returns_array[returns_array > mean_return]
        downside_returns = returns_array[returns_array < mean_return]

        upside_volatility = np.std(upside_returns) if len(upside_returns) > 0 else 0.0
        downside_volatility = np.std(downside_returns) if len(downside_returns) > 0 else 0.0

        # Volatility skew
        volatility_skew = (upside_volatility - downside_volatility) / volatility if volatility > 0 else 0.0

        return {
            'volatility': volatility,
            'annualized_volatility': annualized_volatility,
            'upside_volatility': upside_volatility,
            'downside_volatility': downside_volatility,
            'volatility_skew': volatility_skew
        }

    def calculate_portfolio_risk(self, positions: Dict[str, Dict[str, float]],
                                correlations: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, float]:
        """
        Calculate portfolio-level risk metrics.

        Args:
            positions: Dictionary of positions with weights and individual risks
            correlations: Correlation matrix between assets

        Returns:
            Dictionary containing portfolio risk metrics
        """
        if not positions:
            return {'portfolio_var': 0.0, 'portfolio_volatility': 0.0, 'concentration_risk': 0.0}

        # Extract weights and individual risks
        weights = []
        individual_vars = []
        assets = list(positions.keys())

        for asset in assets:
            pos = positions[asset]
            weights.append(pos.get('weight', 0.0))
            individual_vars.append(pos.get('var', 0.0))

        weights = np.array(weights)
        individual_vars = np.array(individual_vars)

        # Portfolio VaR calculation
        if correlations is None:
            # Assume perfect correlation (conservative estimate)
            portfolio_var = np.sum(weights * individual_vars)
        else:
            # Use correlation matrix
            corr_matrix = np.eye(len(assets))
            for i, asset1 in enumerate(assets):
                for j, asset2 in enumerate(assets):
                    if asset1 in correlations and asset2 in correlations[asset1]:
                        corr_matrix[i, j] = correlations[asset1][asset2]

            # Portfolio variance
            portfolio_variance = np.dot(weights, np.dot(corr_matrix * np.outer(individual_vars, individual_vars), weights))
            portfolio_var = np.sqrt(portfolio_variance)

        # Concentration risk (Herfindahl index)
        concentration_risk = np.sum(weights ** 2)

        # Portfolio volatility (simplified)
        portfolio_volatility = np.sqrt(np.sum((weights * individual_vars) ** 2))

        return {
            'portfolio_var': portfolio_var,
            'portfolio_volatility': portfolio_volatility,
            'concentration_risk': concentration_risk
        }

    def assess_risk_level(self, metrics: Dict[str, float]) -> str:
        """
        Assess overall risk level based on multiple metrics.

        Args:
            metrics: Dictionary containing various risk metrics

        Returns:
            Risk level: 'LOW', 'MEDIUM', 'HIGH', or 'EXTREME'
        """
        risk_score = 0

        # Check maximum drawdown
        max_dd = metrics.get('max_drawdown_pct', 0.0)
        if max_dd > 0.3:
            risk_score += 3
        elif max_dd > 0.2:
            risk_score += 2
        elif max_dd > 0.1:
            risk_score += 1

        # Check VaR
        var_95 = metrics.get('var_95', 0.0)
        if var_95 > 0.1:
            risk_score += 3
        elif var_95 > 0.05:
            risk_score += 2
        elif var_95 > 0.02:
            risk_score += 1

        # Check volatility
        volatility = metrics.get('annualized_volatility', 0.0)
        if volatility > 0.5:
            risk_score += 3
        elif volatility > 0.3:
            risk_score += 2
        elif volatility > 0.2:
            risk_score += 1

        # Check Sharpe ratio (lower is worse)
        sharpe = metrics.get('sharpe_ratio', 0.0)
        if sharpe < -1.0:
            risk_score += 3
        elif sharpe < 0.0:
            risk_score += 2
        elif sharpe < 0.5:
            risk_score += 1

        # Determine risk level
        if risk_score >= 8:
            return 'EXTREME'
        elif risk_score >= 6:
            return 'HIGH'
        elif risk_score >= 3:
            return 'MEDIUM'
        else:
            return 'LOW'

    def calculate_comprehensive_risk_report(self, returns: List[float],
                                          equity_curve: List[float],
                                          positions: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive risk assessment report.

        Args:
            returns: List of return values
            equity_curve: List of portfolio values
            positions: Optional position information

        Returns:
            Comprehensive risk report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_points': len(returns),
            'analysis_period_days': len(returns)
        }

        if len(returns) < 2:
            report['error'] = 'Insufficient data for risk analysis'
            return report

        # Basic risk metrics
        for confidence_level in self.confidence_levels:
            report[f'var_{int(confidence_level*100)}'] = self.calculate_var(returns, confidence_level)
            report[f'cvar_{int(confidence_level*100)}'] = self.calculate_cvar(returns, confidence_level)

        # Drawdown analysis
        dd_metrics = self.calculate_maximum_drawdown(equity_curve)
        report.update(dd_metrics)

        # Risk-adjusted returns
        report['sharpe_ratio'] = self.calculate_sharpe_ratio(returns)
        report['sortino_ratio'] = self.calculate_sortino_ratio(returns)
        report['calmar_ratio'] = self.calculate_calmar_ratio(returns, equity_curve)

        # Volatility analysis
        vol_metrics = self.calculate_volatility_metrics(returns)
        report.update(vol_metrics)

        # Portfolio risk (if positions provided)
        if positions:
            portfolio_risk = self.calculate_portfolio_risk(positions)
            report.update(portfolio_risk)

        # Overall risk assessment
        report['risk_level'] = self.assess_risk_level(report)

        # Risk warnings
        warnings = []
        if report.get('max_drawdown_pct', 0) > self.max_drawdown_threshold:
            warnings.append(f"Maximum drawdown ({report['max_drawdown_pct']:.1%}) exceeds threshold ({self.max_drawdown_threshold:.1%})")

        if report.get('var_95', 0) > self.var_threshold:
            warnings.append(f"95% VaR ({report['var_95']:.1%}) exceeds threshold ({self.var_threshold:.1%})")

        if report.get('annualized_volatility', 0) > self.volatility_threshold:
            warnings.append(f"Annualized volatility ({report['annualized_volatility']:.1%}) exceeds threshold ({self.volatility_threshold:.1%})")

        report['warnings'] = warnings

        return report
