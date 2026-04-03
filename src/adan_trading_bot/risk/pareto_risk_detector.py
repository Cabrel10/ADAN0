#!/usr/bin/env python3
"""
Pareto Risk Detector - Detects fat-tailed regimes (extreme volatility)  
WITH TIER-AWARE MINIMUM MULTIPLIERS to respect Binance $11 minimum

Based on kurtosis analysis + capital tier constraints
"""

import numpy as np
from scipy import stats
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class ParetoRiskDetector:
    """
    Detects market regimes and adjusts risk multipliers.
    INCLUDES TIER-AWARE FLOORS to prevent trades < $11 Binance minimum.
    """
    
    # Tier-specific minimum safe multipliers (calculated from analyze_pareto_viability.py)
    TIER_MIN_MULTIPLIERS = {
        "Micro": 1.0,    # DISABLE Pareto (capital too small)
        "Small": 0.92,   # Can't go below 0.92x
        "Medium": 0.5,   # Full Pareto allowed
        "High": 0.5,
        "Enterprise": 0.5
    }
    
    def __init__(
        self,
        window_size: int = 100,
        update_frequency: int = 20,
        high_vol_threshold: float = 2.0,
        extreme_threshold: float = 5.0,
        current_tier: str = "Medium"
    ):
        """
        Args:
            window_size: Number of returns to analyze
            update_frequency: How often to recalculate
            high_vol_threshold: Kurtosis for HIGH_VOL regime
            extreme_threshold: Kurtosis for EXTREME regime
            current_tier: Current capital tier name
        """
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.high_vol_threshold = high_vol_threshold
        self.extreme_threshold = extreme_threshold
        self.current_tier = current_tier
        
        # Get minimum safe multiplier for this tier
        self.min_safe_multiplier = self.TIER_MIN_MULTIPLIERS.get(current_tier, 0.5)
        
        # State
        self.returns_history: List[float] = []
        self.current_regime = "NORMAL"
        self.current_kurtosis = 0.0
        self.current_multiplier = 1.0
        self.step_counter = 0
        
        # Base regime multipliers (will be floored by tier minimum)
        self.base_regime_multipliers = {
            "NORMAL": 1.0,
            "HIGH_VOL": 0.75,
            "EXTREME": 0.5
        }
        
        logger.info(
            f"ParetoRiskDetector initialized: tier={current_tier}, "
            f"min_multiplier={self.min_safe_multiplier:.2f}"
        )
    
    def update(self, portfolio_return: float) -> Dict[str, float]:
        """Update detector with new return and recalculate regime."""
        self.returns_history.append(portfolio_return)
        
        if len(self.returns_history) > self.window_size:
            self.returns_history = self.returns_history[-self.window_size:]
        
        self.step_counter += 1
        
        if self.step_counter % self.update_frequency == 0:
            self._detect_regime()
        
        return self.get_regime_info()
    
    def _detect_regime(self):
        """Detect regime and apply tier-aware floor."""
        if len(self.returns_history) < 30:
            self.current_regime = "NORMAL"
            self.current_multiplier = 1.0
            self.current_kurtosis = 0.0
            return
        
        returns = np.array(self.returns_history)
        
        try:
            self.current_kurtosis = stats.kurtosis(returns, fisher=True)
        except Exception as e:
            logger.warning(f"Failed to calculate kurtosis: {e}")
            self.current_kurtosis = 0.0
            return
        
        # Determine regime
        if self.current_kurtosis >= self.extreme_threshold:
            self.current_regime = "EXTREME"
        elif self.current_kurtosis >= self.high_vol_threshold:
            self.current_regime = "HIGH_VOL"
        else:
            self.current_regime = "NORMAL"
        
        # Get base multiplier for regime
        base_mult = self.base_regime_multipliers[self.current_regime]
        
        # Apply tier-aware floor to prevent trades < $11
        self.current_multiplier = max(base_mult, self.min_safe_multiplier)
        
        # Log if floored
        if self.current_multiplier > base_mult:
            logger.info(
                f"📊 Pareto FLOORED: {self.current_regime} wanted {base_mult:.2f}x, "
                f"but tier {self.current_tier} minimum is {self.min_safe_multiplier:.2f}x"
            )
        
        # Log regime changes
        if self.step_counter % (self.update_frequency * 5) == 0:
            logger.info(
                f"📊 Pareto: {self.current_regime} | "
                f"Kurtosis: {self.current_kurtosis:.2f} | "
                f"Multiplier: {self.current_multiplier:.2f}x"
            )
    
    def update_tier(self, new_tier: str):
        """Update current tier (when capital crosses tier boundary)."""
        if new_tier != self.current_tier:
            old_min = self.min_safe_multiplier
            self.current_tier = new_tier
            self.min_safe_multiplier = self.TIER_MIN_MULTIPLIERS.get(new_tier, 0.5)
            
            logger.info(
                f"Pareto tier updated: {new_tier} | "
                f"Min multiplier: {old_min:.2f} → {self.min_safe_multiplier:.2f}"
            )
            
            # Recheck current multiplier against new floor
            base_mult = self.base_regime_multipliers[self.current_regime]
            self.current_multiplier = max(base_mult, self.min_safe_multiplier)
    
    def get_risk_multiplier(self, base_risk: float) -> float:
        """Get adjusted risk based on current regime + tier floor."""
        return base_risk * self.current_multiplier
    
    def get_regime_info(self) -> Dict[str, float]:
        """Get current regime information for logging/monitoring."""
        return {
            "regime": self.current_regime,
            "kurtosis": self.current_kurtosis,
            "multiplier": self.current_multiplier,
            "tier": self.current_tier,
            "min_safe_mult": self.min_safe_multiplier,
            "n_samples": len(self.returns_history)
        }
    
    def reset(self):
        """Reset detector state (for new episode/day)."""
        self.returns_history = []
        self.current_regime = "NORMAL"
        self.current_kurtosis = 0.0
        self.current_multiplier = 1.0
        self.step_counter = 0
        logger.debug("ParetoRiskDetector reset")
