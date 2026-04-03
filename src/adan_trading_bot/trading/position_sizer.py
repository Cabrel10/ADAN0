"""
Position sizing module for the ADAN Trading Bot.

This module provides advanced position sizing strategies for optimal
risk management and capital allocation.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)

MIN_ATR_VALUE = 1e-5  # Minimum ATR value to prevent division by zero

@dataclass
class PositionSizingResult:
    """Result of position size calculation."""
    size_in_asset_units: float
    size_in_usd: float
    warnings: List[str] = None
    metadata: Optional[Dict[str, Any]] = None

class PositionSizer:
    """
    Advanced position sizing system based on worker-specific risk formulas.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PositionSizer.

        Args:
            config: The main configuration dictionary.
        """
        self.config = config
        self.worker_configs = {
            'scalper': self.config.get('configs', {}).get('scalper_config', {}),
            'intraday': self.config.get('configs', {}).get('swing_config', {}), # Assuming intraday uses swing for now
            'swing': self.config.get('configs', {}).get('swing_config', {}),
            'position': self.config.get('configs', {}).get('position_config', {})
        }
        logger.info("PositionSizer initialized with worker-specific configurations.")

    def calculate_position_size(
        self,
        worker_type: str,
        capital: float,
        entry_price: float,
        market_data: Dict[str, Any]
    ) -> PositionSizingResult:
        """
        Calculates position size based on the worker type and its specific risk formula.

        Args:
            worker_type: The type of worker ('scalper', 'intraday', 'swing', 'position').
            capital: Current portfolio value.
            entry_price: The current price of the asset.
            market_data: A dictionary containing necessary indicators like ATR, trend_strength, etc.

        Returns:
            A PositionSizingResult object.
        """
        warnings = []
        metadata = {'worker_type': worker_type}

        # --- Dispatch to the correct sizing function based on worker_type ---
        if worker_type == 'scalper':
            size_in_asset_units = self._scalper_position_size(capital, entry_price, market_data, warnings)
        elif worker_type == 'intraday':
            size_in_asset_units = self._intraday_position_size(capital, entry_price, market_data, warnings)
        elif worker_type == 'swing':
            size_in_asset_units = self._swing_position_size(capital, entry_price, market_data, warnings)
        elif worker_type == 'position':
            size_in_asset_units = self._position_position_size(capital, entry_price, market_data, warnings)
        else:
            logger.warning(f"Unknown worker_type '{worker_type}'. Defaulting to 'scalper'.")
            warnings.append(f"Unknown worker_type '{worker_type}'. Defaulting to 'scalper'")
            size_in_asset_units = self._scalper_position_size(capital, entry_price, market_data, warnings)

        size_in_usd = size_in_asset_units * entry_price

        min_order_value = 11.0
    
        if size_in_usd < min_order_value:
            warnings.append(f"Taille calculée ({size_in_usd:.2f}) < minimum ({min_order_value}).")
            
            # Vérifier si le capital total permet d'atteindre le minimum
            if capital >= min_order_value:
                size_in_usd = min_order_value
                size_in_asset_units = size_in_usd / entry_price
                warnings.append(f"Taille ajustée au minimum de {min_order_value:.2f} USDT.")
            else:
                warnings.append("Capital insuffisant pour atteindre la taille minimale.")
                return PositionSizingResult(0.0, 0.0, warnings, metadata)

        return PositionSizingResult(
            size_in_asset_units=size_in_asset_units,
            size_in_usd=size_in_usd,
            warnings=warnings,
            metadata=metadata
        )

    def _get_risk_percentage(self, worker_type: str) -> float:
        """Get risk percentage from config based on worker type."""
        # Default risk percentages (in decimal)
        default_risks = {
            'scalper': 0.01,   # 1%
            'intraday': 0.015, # 1.5%
            'swing': 0.02,     # 2%
            'position': 0.025  # 2.5%
        }
        
        # Try to get from worker config
        worker_config = self.worker_configs.get(worker_type, {})
        risk_pct = worker_config.get('risk_per_trade_pct')
        
        # If not found in worker config, try to get from global config
        if risk_pct is None:
            risk_pct = self.config.get('trading', {}).get('risk_management', {}).get('risk_per_trade_pct')
        
        # Convert to decimal if it's a percentage (e.g., 2.0 for 2%)
        if risk_pct is not None:
            if risk_pct > 1.0:  # If it's in percentage
                risk_pct = risk_pct / 100.0
            return max(0.001, min(risk_pct, 0.1))  # Clamp between 0.1% and 10%
        
        # Fallback to default risk for the worker type
        return default_risks.get(worker_type, 0.01)  # Default to 1% if worker type not found

    def _scalper_position_size(self, capital, entry_price, market_data, warnings):
        atr_14 = market_data.get('atr_14')
        if atr_14 is None or atr_14 <= 0:
            warnings.append("atr_14 not available for scalper sizing, using fallback.")
            return (capital * 0.05) / entry_price  # Fallback: 5% of capital

        # Get risk percentage from config
        risk_pct = self._get_risk_percentage('scalper')
        risk_amount = capital * risk_pct
        
        # Calculate position size based on ATR
        atr_14 = max(atr_14, MIN_ATR_VALUE)
        atr_adjusted_risk = atr_14 * 0.5  # Stop-loss at 0.5 ATR
        
        base_size = risk_amount / atr_adjusted_risk if atr_adjusted_risk > 0 else 0
        max_size = (capital * 0.10) / entry_price  # Max 10% of capital
        
        # Log the calculation for debugging
        if hasattr(self, 'logger'):
            self.logger.debug(f"[PositionSizer] Scalper position - Capital: {capital:.2f}, Risk %: {risk_pct*100:.2f}%, "
                           f"ATR: {atr_14:.6f}, Size: {min(base_size, max_size) * entry_price:.2f} USDT")
        
        return min(base_size, max_size)

    def _intraday_position_size(self, capital, entry_price, market_data, warnings):
        atr_14 = market_data.get('atr_14')
        volatility_ratio = market_data.get('volatility_ratio')

        if atr_14 is None or atr_14 <= 0:
            warnings.append("atr_14 not available for intraday sizing, using fallback.")
            return (capital * 0.10) / entry_price  # Fallback: 10% of capital

        # Get risk percentage from config
        risk_pct = self._get_risk_percentage('intraday')
        risk_amount = capital * risk_pct
        
        # Calculate position size based on ATR
        atr_14 = max(atr_14, MIN_ATR_VALUE)
        atr_adjusted_risk = atr_14 * 1.0  # Stop-loss at 1.0 ATR
        
        base_size = risk_amount / atr_adjusted_risk if atr_adjusted_risk > 0 else 0
        max_size = (capital * 0.20) / entry_price  # Max 20% of capital
        
        # Volatility adjustment
        volatility_adjustment = 1.0
        if volatility_ratio and volatility_ratio > 0:
            volatility_adjustment = 1.0 / (1.0 + volatility_ratio * 0.1)
        else:
            warnings.append("volatility_ratio not available for intraday adjustment.")
        
        # Log the calculation for debugging
        if hasattr(self, 'logger'):
            self.logger.debug(f"[PositionSizer] Intraday position - Capital: {capital:.2f}, Risk %: {risk_pct*100:.2f}%, "
                           f"ATR: {atr_14:.6f}, VolAdj: {volatility_adjustment:.2f}, "
                           f"Size: {min(base_size * volatility_adjustment, max_size) * entry_price:.2f} USDT")

        return min(base_size * volatility_adjustment, max_size)

    def _swing_position_size(self, capital, entry_price, market_data, warnings):
        atr_20 = market_data.get('atr_20')
        trend_strength = market_data.get('trend_strength')

        if atr_20 is None or atr_20 <= 0:
            warnings.append("atr_20 not available for swing sizing, using fallback.")
            return (capital * 0.15) / entry_price  # Fallback: 15% of capital

        # Get risk percentage from config
        risk_pct = self._get_risk_percentage('swing')
        risk_amount = capital * risk_pct
        
        # Calculate position size based on ATR
        atr_20 = max(atr_20, MIN_ATR_VALUE)
        atr_adjusted_risk = atr_20 * 2.0  # Stop-loss at 2.0 ATR

        base_size = risk_amount / atr_adjusted_risk if atr_adjusted_risk > 0 else 0
        max_size = (capital * 0.30) / entry_price  # Max 30% of capital

        # Trend strength adjustment
        trend_adjustment = 1.0
        if trend_strength is not None:
            trend_adjustment = 1.0 + (trend_strength * 0.2)  # Up to 20% increase in size with strong trend
        else:
            warnings.append("trend_strength not available for swing adjustment.")
        
        # Log the calculation for debugging
        if hasattr(self, 'logger'):
            self.logger.debug(f"[PositionSizer] Swing position - Capital: {capital:.2f}, Risk %: {risk_pct*100:.2f}%, "
                           f"ATR20: {atr_20:.6f}, TrendAdj: {trend_adjustment:.2f}, "
                           f"Size: {min(base_size * trend_adjustment, max_size) * entry_price:.2f} USDT")

        return min(base_size * trend_adjustment, max_size)

    def _position_position_size(self, capital, entry_price, market_data, warnings):
        atr_50 = market_data.get('atr_50')
        fundamental_score = market_data.get('fundamental_score')

        if atr_50 is None or atr_50 <= 0:
            warnings.append("atr_50 not available for position sizing, using fallback.")
            return (capital * 0.20) / entry_price  # Fallback: 20% of capital

        # Get risk percentage from config
        risk_pct = self._get_risk_percentage('position')
        risk_amount = capital * risk_pct
        
        # Calculate position size based on ATR
        atr_50 = max(atr_50, MIN_ATR_VALUE)
        atr_adjusted_risk = atr_50 * 3.0  # Stop-loss at 3.0 ATR

        base_size = risk_amount / atr_adjusted_risk if atr_adjusted_risk > 0 else 0
        max_size = (capital * 0.50) / entry_price  # Max 50% of capital

        # Fundamental score adjustment
        fundamental_adjustment = 1.0
        if fundamental_score is not None:
            fundamental_adjustment = 1.0 + (fundamental_score * 0.3)  # Up to 30% increase with strong fundamentals
        else:
            warnings.append("fundamental_score not available for position adjustment.")
        
        # Log the calculation for debugging
        if hasattr(self, 'logger'):
            self.logger.debug(f"[PositionSizer] Position trading - Capital: {capital:.2f}, Risk %: {risk_pct*100:.2f}%, "
                           f"ATR50: {atr_50:.6f}, FundAdj: {fundamental_adjustment:.2f}, "
                           f"Size: {min(base_size * fundamental_adjustment, max_size) * entry_price:.2f} USDT")

        return min(base_size * fundamental_adjustment, max_size)
