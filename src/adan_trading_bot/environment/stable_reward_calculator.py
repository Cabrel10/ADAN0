#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
StableRewardCalculator - Normalized reward function with explicit penalties.

Per ADAN 2.0 Spec (Requirements 3.1-3.6):
- Normalized PnL component
- Sharpe ratio contribution
- Drawdown penalty
- Trade frequency penalty
- Consistency bonus
- Reward clipping to [-1.0, 1.0]
"""

import logging
import numpy as np
from typing import List, Dict, Optional
from collections import deque

logger = logging.getLogger(__name__)


class StableRewardCalculator:
    """
    Calculates normalized rewards with clear penalty structure.
    
    Components:
    1. Normalized PnL (baseline)
    2. Sharpe ratio contribution (quality)
    3. Drawdown penalty (risk)
    4. Trade frequency penalty (overtrading)
    5. Consistency bonus (stability)
    
    Final reward is clipped to [-1.0, 1.0]
    """
    
    def __init__(
        self,
        pnl_normalization_factor: float = 100.0,  # Normalize PnL to ~[-1, 1]
        sharpe_weight: float = 0.2,
        drawdown_penalty_weight: float = 0.3,
        frequency_penalty_weight: float = 0.1,
        consistency_bonus_weight: float = 0.1,
        invalid_sell_penalty_weight: float = 0.05,  # Penalty for selling without position
        returns_window: int = 100  # Window for Sharpe calculation
    ):
        """
        Initialize the calculator.
        
        Args:
            pnl_normalization_factor: Factor to normalize PnL
            sharpe_weight: Weight for Sharpe component
            drawdown_penalty_weight: Weight for drawdown penalty
            frequency_penalty_weight: Weight for frequency penalty
            consistency_bonus_weight: Weight for consistency bonus
            invalid_sell_penalty_weight: Weight for invalid sell penalty
            returns_window: Window size for returns tracking
        """
        self.pnl_norm_factor = pnl_normalization_factor
        self.sharpe_weight = sharpe_weight
        self.drawdown_penalty_weight = drawdown_penalty_weight
        self.frequency_penalty_weight = frequency_penalty_weight
        self.consistency_bonus_weight = consistency_bonus_weight
        self.invalid_sell_penalty_weight = invalid_sell_penalty_weight
        
        # Track returns for Sharpe calculation
        self.returns_window = returns_window
        self.returns_history: deque = deque(maxlen=returns_window)
        
        # Track metrics
        self.peak_value: float = 0.0
        
        logger.info(
            f"StableRewardCalculator initialized "
            f"(pnl_norm={self.pnl_norm_factor}, sharpe_w={sharpe_weight}, "
            f"dd_w={drawdown_penalty_weight}, freq_w={frequency_penalty_weight})"
            f"dd_w={drawdown_penalty_weight}, freq_w={frequency_penalty_weight})"
        )

    def update_normalization_factor(self, new_factor: float) -> None:
        """
        Update the PnL normalization factor dynamically.
        Useful when capital scales significantly (e.g. Micro vs Enterprise tiers).
        """
        if new_factor <= 0:
            return
        old_factor = self.pnl_norm_factor
        self.pnl_norm_factor = float(new_factor)
        logger.info(f"Updated PnL Normalization Factor: {old_factor:.2f} -> {self.pnl_norm_factor:.2f}")
    
    def calculate_reward(
        self,
        pnl: float,
        portfolio_value: float,
        initial_value: float,
        trade_count: int,
        step_return: Optional[float] = None,
        invalid_sell_attempts: int = 0
    ) -> Dict[str, float]:
        """
        Calculate normalized reward with all components.
        
        Args:
            pnl: Realized PnL for this step
            portfolio_value: Current portfolio value
            initial_value: Initial portfolio value
            trade_count: Number of trades executed this step
            step_return: Optional pre-calculated step return
            invalid_sell_attempts: Number of invalid sell attempts (no position)
            
        Returns:
            Dictionary with reward breakdown:
            {
                'pnl_component': float,
                'sharpe_component': float,
                'drawdown_penalty': float,
                'frequency_penalty': float,
                'invalid_sell_penalty': float,
                'consistency_bonus': float,
                'total_reward': float  # Clipped to [-1, 1]
            }
        """
        # 1. Normalized PnL component
        pnl_component = self._normalize_pnl(pnl)
        
        # 2. Calculate step return and update history
        if step_return is None:
            step_return = pnl / max(portfolio_value, 1.0)
        self.returns_history.append(step_return)
        
        # 3. Sharpe ratio contribution
        sharpe_component = self._calculate_sharpe_contribution()
        
        # 4. Drawdown penalty
        self.peak_value = max(self.peak_value, portfolio_value)
        current_drawdown = (self.peak_value - portfolio_value) / max(self.peak_value, 1.0)
        drawdown_penalty = self._apply_drawdown_penalty(current_drawdown)
        
        # 5. Trade frequency penalty
        frequency_penalty = self._apply_frequency_penalty(trade_count)
        
        # 6. Consistency bonus
        consistency_bonus = self._apply_consistency_bonus(list(self.returns_history))
        
        # 6.5 Invalid Sell Penalty
        invalid_sell_penalty = float(invalid_sell_attempts) * self.invalid_sell_penalty_weight
        
        # 7. Combine components
        total_reward = (
            pnl_component +
            self.sharpe_weight * sharpe_component -
            self.drawdown_penalty_weight * drawdown_penalty -
            self.frequency_penalty_weight * frequency_penalty -
            invalid_sell_penalty +
            self.consistency_bonus_weight * consistency_bonus
        )
        
        # 8. Clip to [-1.0, 1.0]
        total_reward = np.clip(total_reward, -1.0, 1.0)
        
        # 🔧 CRITICAL FIX: Ensure all values are finite (no NaN/Inf)
        total_reward = np.nan_to_num(total_reward, nan=0.0, posinf=1.0, neginf=-1.0)
        pnl_component = np.nan_to_num(pnl_component, nan=0.0, posinf=1.0, neginf=-1.0)
        sharpe_component = np.nan_to_num(sharpe_component, nan=0.0, posinf=1.0, neginf=-1.0)
        drawdown_penalty = np.nan_to_num(drawdown_penalty, nan=0.0, posinf=1.0, neginf=-1.0)
        frequency_penalty = np.nan_to_num(frequency_penalty, nan=0.0, posinf=1.0, neginf=-1.0)
        invalid_sell_penalty = np.nan_to_num(invalid_sell_penalty, nan=0.0, posinf=1.0, neginf=-1.0)
        consistency_bonus = np.nan_to_num(consistency_bonus, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return {
            'pnl_component': float(pnl_component),
            'sharpe_component': float(sharpe_component),
            'drawdown_penalty': float(drawdown_penalty),
            'frequency_penalty': float(frequency_penalty),
            'invalid_sell_penalty': float(invalid_sell_penalty),
            'consistency_bonus': float(consistency_bonus),
            'total_reward': float(total_reward)
        }
    
    def _normalize_pnl(self, pnl: float) -> float:
        """
        Normalize PnL to bounded range.
        
        Args:
            pnl: Raw PnL value
            
        Returns:
            Normalized PnL in approximate range [-1, 1]
        """
        # Hyperbolic tangent provides smooth normalization
        return float(np.tanh(pnl / self.pnl_norm_factor))
    
    def _calculate_sharpe_contribution(self) -> float:
        """
        Calculate Sharpe ratio contribution from historical returns.
        
        Returns:
            Sharpe component (normalized)
        """
        if len(self.returns_history) < 2:
            return 0.0
        
        returns_array = np.array(self.returns_history)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return < 1e-6:
            # No volatility - neutral contribution
            return 0.0
        
        # Sharpe ratio (annualization factor omitted for simplicity)
        sharpe = mean_return / std_return
        
        # Normalize to [-1, 1] range
        return float(np.tanh(sharpe))
    
    def _apply_drawdown_penalty(self, drawdown: float) -> float:
        """
        Apply penalty based on current drawdown.
        
        Penalty increases quadratically with drawdown to discourage risk.
        
        Args:
            drawdown: Current drawdown as fraction (0.0 to 1.0)
            
        Returns:
            Penalty value (0.0 to 1.0)
        """
        # Quadratic penalty: small drawdowns are tolerated, large ones punished heavily
        return float(np.clip(drawdown ** 2, 0.0, 1.0))
    
    def _apply_frequency_penalty(self, trade_count: int) -> float:
        """
        Apply penalty for excessive trading frequency.
        
        Args:
            trade_count: Number of trades this step
            
        Returns:
            Penalty value (0.0 to 1.0)
        """
        # Penalize more than 1 trade per step (flickering)
        if trade_count == 0:
            return 0.0
        elif trade_count == 1:
            return 0.1  # Small penalty even for normal trading
        else:
            # Heavy penalty for multiple trades in one step
            return float(np.clip(0.5 + 0.25 * (trade_count - 1), 0.0, 1.0))
    
    def _apply_consistency_bonus(self, returns: List[float]) -> float:
        """
        Apply bonus for consistent (stable) returns.
        
        Rewards agents that generate steady returns vs volatile ones.
        
        Args:
            returns: List of recent returns
            
        Returns:
            Bonus value (0.0 to 1.0)
        """
        if len(returns) < 5:
            return 0.0
        
        returns_array = np.array(returns)
        
        # Check for consistent positive returns
        positive_ratio = np.sum(returns_array > 0) / len(returns_array)
        
        # Check for low volatility
        std = np.std(returns_array)
        volatility_penalty = min(std * 10, 1.0)  # Penalize high volatility
        
        # Bonus if mostly positive AND low volatility
        if positive_ratio > 0.6:
            bonus = positive_ratio * (1.0 - volatility_penalty)
            return float(np.clip(bonus, 0.0, 1.0))
        
        return 0.0
    
    def reset(self) -> None:
        """Reset calculator state (called at episode start)."""
        self.returns_history.clear()
        self.peak_value = 0.0
        logger.debug("StableRewardCalculator reset")
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        returns_array = np.array(self.returns_history) if self.returns_history else np.array([])
        return {
            "returns_count": len(self.returns_history),
            "mean_return": float(np.mean(returns_array)) if len(returns_array) > 0 else 0.0,
            "std_return": float(np.std(returns_array)) if len(returns_array) > 0 else 0.0,
            "peak_value": self.peak_value
        }
