"""
Unit tests for observation builder.

Tests verify that the observation builder correctly constructs feature vectors
from validated indicators and classifies market regimes.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from src.adan_trading_bot.observation.builder import (
    ObservationBuilder, Observation, MarketRegime
)


class TestObservationBuilder:
    """Test ObservationBuilder functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = ObservationBuilder()
        
        # Create sample OHLCV data
        self.sample_df = self._create_sample_data(100)
    
    def _create_sample_data(self, num_candles: int) -> pd.DataFrame:
        """Create sample OHLCV data."""
        np.random.seed(42)
        
        close_prices = 47000 + np.cumsum(np.random.randn(num_candles) * 100)
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=num_candles, freq='5min'),
            'open': close_prices + np.random.randn(num_candles) * 50,
            'high': close_prices + np.abs(np.random.randn(num_candles) * 100),
            'low': close_prices - np.abs(np.random.randn(num_candles) * 100),
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, num_candles)
        })
        
        return df
    
    def test_builder_initialization(self):
        """Test builder initializes correctly."""
        builder = ObservationBuilder()
        assert builder is not None
        assert builder.RSI_OVERBOUGHT == 70
        assert builder.RSI_OVERSOLD == 30
        assert builder.ADX_STRONG_TREND == 25
        assert builder.ADX_WEAK_TREND == 20
    
    @patch('src.adan_trading_bot.observation.builder.DataValidator')
    def test_build_observation_success(self, mock_validator_class):
        """Test successful observation building."""
        # Mock validator to avoid API calls
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator
        
        mock_result = Mock()
        mock_result.status = "pass"
        mock_result.message = "Validation passed"
        mock_validator.validate_full_pipeline.return_value = mock_result
        
        builder = ObservationBuilder()
        builder.validator = mock_validator
        
        observation = builder.build_observation(self.sample_df)
        
        assert observation is not None
        assert isinstance(observation.features, np.ndarray)
        assert len(observation.features) == 6  # 6 features
        assert 0 <= observation.rsi <= 100
        assert 0 <= observation.adx <= 100
        assert observation.atr > 0
        assert observation.atr_percent > 0
        assert observation.regime is not None
        assert observation.regime.regime in ["bullish", "bearish", "ranging"]
        assert 0 <= observation.regime.strength <= 1
    
    def test_build_observation_insufficient_data(self):
        """Test observation building with insufficient data."""
        small_df = self.sample_df.iloc[:10]
        
        with pytest.raises(ValueError, match="Insufficient data"):
            self.builder.build_observation(small_df)
    
    def test_calculate_current_stats(self):
        """Test current statistics calculation."""
        stats = self.builder._calculate_current_stats(self.sample_df)
        
        assert 'close' in stats
        assert 'sma_20' in stats
        assert 'sma_50' in stats
        assert 'volatility' in stats
        assert 'high_20' in stats
        assert 'low_20' in stats
        
        # Verify stats are reasonable
        assert stats['close'] > 0
        assert stats['sma_20'] > 0
        assert stats['sma_50'] > 0
        assert stats['volatility'] >= 0
        assert stats['high_20'] >= stats['close']
        assert stats['low_20'] <= stats['close']
    
    def test_classify_regime_bullish(self):
        """Test market regime classification - bullish."""
        indicators = {
            'rsi': 75,  # Overbought
            'adx': 30,  # Strong trend
            'atr': 100,
            'atr_percent': 0.2
        }
        
        regime = self.builder._classify_regime(indicators)
        
        assert regime.regime == "bullish"
        assert regime.strength > 0.5
    
    def test_classify_regime_bearish(self):
        """Test market regime classification - bearish."""
        indicators = {
            'rsi': 25,  # Oversold
            'adx': 30,  # Strong trend
            'atr': 100,
            'atr_percent': 0.2
        }
        
        regime = self.builder._classify_regime(indicators)
        
        assert regime.regime == "bearish"
        assert regime.strength > 0.5
    
    def test_classify_regime_ranging(self):
        """Test market regime classification - ranging."""
        indicators = {
            'rsi': 50,  # Neutral
            'adx': 15,  # Weak trend
            'atr': 100,
            'atr_percent': 0.2
        }
        
        regime = self.builder._classify_regime(indicators)
        
        assert regime.regime == "ranging"
        assert regime.strength < 0.5
    
    def test_build_feature_vector(self):
        """Test feature vector construction."""
        indicators = {
            'rsi': 50,
            'adx': 25,
            'atr': 100,
            'atr_percent': 0.5
        }
        
        current_stats = {
            'close': 47000,
            'sma_20': 46900,
            'sma_50': 46800,
            'volatility': 100,
            'high_20': 47500,
            'low_20': 46500
        }
        
        features = self.builder._build_feature_vector(indicators, current_stats)
        
        assert isinstance(features, np.ndarray)
        assert len(features) == 6
        assert features.dtype == np.float32
        
        # Verify features are normalized
        for feature in features:
            assert -1 <= feature <= 1 or feature == 0
    
    @patch('src.adan_trading_bot.observation.builder.DataValidator')
    def test_observation_metadata(self, mock_validator_class):
        """Test observation metadata is complete."""
        # Mock validator to avoid API calls
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator
        
        mock_result = Mock()
        mock_result.status = "pass"
        mock_result.message = "Validation passed"
        mock_validator.validate_full_pipeline.return_value = mock_result
        
        builder = ObservationBuilder()
        builder.validator = mock_validator
        
        observation = builder.build_observation(self.sample_df)
        
        assert 'data_timestamp' in observation.metadata
        assert 'validation_status' in observation.metadata
        assert 'validation_message' in observation.metadata
        assert 'current_stats' in observation.metadata
        assert 'indicators' in observation.metadata
        
        # Verify indicators in metadata
        indicators = observation.metadata['indicators']
        assert 'rsi' in indicators
        assert 'adx' in indicators
        assert 'atr' in indicators
        assert 'atr_percent' in indicators
    
    @patch('src.adan_trading_bot.observation.builder.DataValidator')
    def test_build_batch_observations(self, mock_validator_class):
        """Test batch observation building."""
        # Mock validator to avoid API calls
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator
        
        mock_result = Mock()
        mock_result.status = "pass"
        mock_result.message = "Validation passed"
        mock_validator.validate_full_pipeline.return_value = mock_result
        
        builder = ObservationBuilder()
        builder.validator = mock_validator
        
        observations = builder.build_batch_observations(
            self.sample_df,
            window_size=30,
            step_size=10
        )
        
        assert len(observations) > 0
        assert all(isinstance(obs, Observation) for obs in observations)
        
        # Verify each observation is valid
        for obs in observations:
            assert obs.features is not None
            assert len(obs.features) == 6
            assert obs.regime is not None
    
    @patch('src.adan_trading_bot.observation.builder.DataValidator')
    def test_build_batch_observations_small_step(self, mock_validator_class):
        """Test batch observation building with small step size."""
        # Mock validator to avoid API calls
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator
        
        mock_result = Mock()
        mock_result.status = "pass"
        mock_result.message = "Validation passed"
        mock_validator.validate_full_pipeline.return_value = mock_result
        
        builder = ObservationBuilder()
        builder.validator = mock_validator
        
        observations = builder.build_batch_observations(
            self.sample_df,
            window_size=30,
            step_size=1
        )
        
        # Should have many observations with step_size=1
        assert len(observations) > 50
    
    @patch('src.adan_trading_bot.observation.builder.DataValidator')
    def test_build_observation_validation_halt(self, mock_validator_class):
        """Test observation building with validation halt."""
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator
        
        # Mock validation to return halt
        mock_result = Mock()
        mock_result.status = "halt"
        mock_result.message = "Test halt"
        mock_validator.validate_full_pipeline.return_value = mock_result
        
        builder = ObservationBuilder()
        builder.validator = mock_validator
        
        with pytest.raises(ValueError, match="Data validation failed"):
            builder.build_observation(self.sample_df)
    
    @patch('src.adan_trading_bot.observation.builder.DataValidator')
    def test_build_observation_validation_warning(self, mock_validator_class):
        """Test observation building with validation warning."""
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator
        
        # Mock validation to return warning
        mock_result = Mock()
        mock_result.status = "warning"
        mock_result.message = "Test warning"
        mock_validator.validate_full_pipeline.return_value = mock_result
        
        builder = ObservationBuilder()
        builder.validator = mock_validator
        
        # Should still build observation with warning
        observation = builder.build_observation(self.sample_df)
        assert observation is not None
        assert observation.metadata['validation_status'] == "warning"


class TestMarketRegime:
    """Test MarketRegime dataclass."""
    
    def test_market_regime_creation(self):
        """Test MarketRegime can be created."""
        regime = MarketRegime(
            regime="bullish",
            strength=0.8,
            timestamp=datetime.utcnow()
        )
        
        assert regime.regime == "bullish"
        assert regime.strength == 0.8
        assert isinstance(regime.timestamp, datetime)


class TestObservation:
    """Test Observation dataclass."""
    
    def test_observation_creation(self):
        """Test Observation can be created."""
        features = np.array([0.5, 0.6, 0.7, 0.1, 0.2, 0.3], dtype=np.float32)
        regime = MarketRegime("bullish", 0.8, datetime.utcnow())
        
        observation = Observation(
            features=features,
            rsi=50,
            adx=25,
            atr=100,
            atr_percent=0.5,
            regime=regime,
            timestamp=datetime.utcnow(),
            metadata={}
        )
        
        assert observation.rsi == 50
        assert observation.adx == 25
        assert observation.atr == 100
        assert observation.atr_percent == 0.5
        assert len(observation.features) == 6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
