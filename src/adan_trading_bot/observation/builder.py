"""
Observation Builder - Constructs feature vectors from validated indicators.

This module builds observation vectors from validated market indicators,
ensuring they reflect current market conditions using live statistics
rather than stale training data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from ..indicators.calculator import IndicatorCalculator
from ..validation.data_validator import DataValidator

logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    """Market regime classification."""
    regime: str  # "bullish", "bearish", "ranging"
    strength: float  # 0-1, confidence in the regime
    timestamp: datetime


@dataclass
class Observation:
    """Market observation with validated indicators."""
    features: np.ndarray  # Feature vector for model input
    rsi: float
    adx: float
    atr: float
    atr_percent: float
    regime: MarketRegime
    timestamp: datetime
    metadata: Dict


class ObservationBuilder:
    """Build observation vectors from validated indicators."""
    
    # Market regime thresholds
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    ADX_STRONG_TREND = 25
    ADX_WEAK_TREND = 20
    
    def __init__(self):
        """Initialize observation builder."""
        self.calculator = IndicatorCalculator()
        self.validator = DataValidator()
        logger.info("ObservationBuilder initialized")
    
    def build_observation(self, df: pd.DataFrame, 
                         current_stats: Optional[Dict[str, float]] = None) -> Observation:
        """
        Build observation from market data.
        
        Args:
            df: DataFrame with OHLCV data
            current_stats: Current market statistics for normalization
                          (if None, calculated from df)
            
        Returns:
            Observation with feature vector and metadata
            
        Raises:
            ValueError: If data is insufficient or invalid
        """
        if len(df) < 30:
            raise ValueError(f"Insufficient data: {len(df)} candles, need at least 30")
        
        # Calculate indicators
        indicators = self.calculator.calculate_all(df)
        
        # Validate indicators
        validation_result = self.validator.validate_full_pipeline(
            calculated_indicators=indicators
        )
        
        if validation_result.status == "halt":
            logger.error(f"Validation halt: {validation_result.message}")
            raise ValueError(f"Data validation failed: {validation_result.message}")
        
        if validation_result.status == "warning":
            logger.warning(f"Validation warning: {validation_result.message}")
        
        # Get current market statistics
        if current_stats is None:
            current_stats = self._calculate_current_stats(df)
        
        # Classify market regime
        regime = self._classify_regime(indicators)
        
        # Build feature vector
        features = self._build_feature_vector(indicators, current_stats)
        
        # Create observation
        observation = Observation(
            features=features,
            rsi=indicators['rsi'],
            adx=indicators['adx'],
            atr=indicators['atr'],
            atr_percent=indicators['atr_percent'],
            regime=regime,
            timestamp=datetime.utcnow(),
            metadata={
                'data_timestamp': df['timestamp'].iloc[-1] if 'timestamp' in df.columns else None,
                'validation_status': validation_result.status,
                'validation_message': validation_result.message,
                'current_stats': current_stats,
                'indicators': {
                    'rsi': indicators['rsi'],
                    'adx': indicators['adx'],
                    'atr': indicators['atr'],
                    'atr_percent': indicators['atr_percent']
                }
            }
        )
        
        logger.info(f"Observation built: regime={regime.regime}, RSI={indicators['rsi']:.1f}, "
                   f"ADX={indicators['adx']:.1f}, ATR%={indicators['atr_percent']:.2f}%")
        
        return observation
    
    def _calculate_current_stats(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate current market statistics from data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with current market statistics
        """
        close_prices = df['close'].values
        
        stats = {
            'close': close_prices[-1],
            'sma_20': close_prices[-20:].mean() if len(close_prices) >= 20 else close_prices.mean(),
            'sma_50': close_prices[-50:].mean() if len(close_prices) >= 50 else close_prices.mean(),
            'volatility': np.std(close_prices[-20:]) if len(close_prices) >= 20 else np.std(close_prices),
            'high_20': close_prices[-20:].max() if len(close_prices) >= 20 else close_prices.max(),
            'low_20': close_prices[-20:].min() if len(close_prices) >= 20 else close_prices.min(),
        }
        
        logger.debug(f"Current stats: close={stats['close']:.2f}, volatility={stats['volatility']:.4f}")
        
        return stats
    
    def _classify_regime(self, indicators: Dict[str, float]) -> MarketRegime:
        """
        Classify market regime based on indicators.
        
        Args:
            indicators: Dictionary with calculated indicators
            
        Returns:
            MarketRegime classification
        """
        rsi = indicators['rsi']
        adx = indicators['adx']
        
        # Determine trend direction
        if rsi > self.RSI_OVERBOUGHT:
            direction = "bullish"
        elif rsi < self.RSI_OVERSOLD:
            direction = "bearish"
        else:
            direction = "ranging"
        
        # Determine trend strength
        if adx > self.ADX_STRONG_TREND:
            strength = min(1.0, adx / 50.0)  # Normalize to 0-1
        elif adx > self.ADX_WEAK_TREND:
            strength = 0.5
        else:
            strength = 0.2
        
        regime = MarketRegime(
            regime=direction,
            strength=strength,
            timestamp=datetime.utcnow()
        )
        
        logger.debug(f"Market regime: {regime.regime} (strength={regime.strength:.2f})")
        
        return regime
    
    def _build_feature_vector(self, indicators: Dict[str, float], 
                             current_stats: Dict[str, float]) -> np.ndarray:
        """
        Build feature vector for model input.
        
        Args:
            indicators: Dictionary with calculated indicators
            current_stats: Current market statistics
            
        Returns:
            Feature vector as numpy array
        """
        # Normalize indicators to 0-1 range
        rsi_norm = indicators['rsi'] / 100.0  # RSI is 0-100
        adx_norm = min(1.0, indicators['adx'] / 50.0)  # ADX normalized to 50
        atr_percent_norm = min(1.0, indicators['atr_percent'] / 5.0)  # ATR% normalized to 5%
        
        # Price position relative to moving averages
        close = current_stats['close']
        sma_20 = current_stats['sma_20']
        sma_50 = current_stats['sma_50']
        
        price_vs_sma20 = (close - sma_20) / sma_20 if sma_20 > 0 else 0
        price_vs_sma50 = (close - sma_50) / sma_50 if sma_50 > 0 else 0
        
        # Volatility normalized
        volatility_norm = min(1.0, current_stats['volatility'] / current_stats['close'] * 100)
        
        # Build feature vector
        features = np.array([
            rsi_norm,           # RSI normalized
            adx_norm,           # ADX normalized
            atr_percent_norm,   # ATR% normalized
            price_vs_sma20,     # Price vs SMA20
            price_vs_sma50,     # Price vs SMA50
            volatility_norm,    # Volatility normalized
        ], dtype=np.float32)
        
        logger.debug(f"Feature vector: {features}")
        
        return features
    
    def build_batch_observations(self, df: pd.DataFrame, 
                                window_size: int = 30,
                                step_size: int = 1) -> list:
        """
        Build multiple observations from sliding window.
        
        Args:
            df: DataFrame with OHLCV data
            window_size: Size of observation window
            step_size: Step size for sliding window
            
        Returns:
            List of Observation objects
        """
        observations = []
        
        for i in range(0, len(df) - window_size + 1, step_size):
            window_df = df.iloc[i:i + window_size].copy()
            
            try:
                obs = self.build_observation(window_df)
                observations.append(obs)
            except Exception as e:
                logger.warning(f"Failed to build observation at index {i}: {e}")
                continue
        
        logger.info(f"Built {len(observations)} observations from {len(df)} candles")
        
        return observations
