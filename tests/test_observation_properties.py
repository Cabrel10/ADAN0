"""
Property-based tests for observation builder.

Tests verify that the observation builder correctly constructs feature vectors
and classifies market regimes across all possible market conditions.
"""

import pytest
from hypothesis import given, settings, HealthCheck, strategies as st
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from src.adan_trading_bot.observation.builder import ObservationBuilder, Observation


class TestObservationAccuracyProperties:
    """Property-based tests for observation accuracy."""
    
    @given(
        rsi=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        adx=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        atr_percent=st.floats(min_value=0, max_value=5, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_observation_features_normalized(self, rsi, adx, atr_percent):
        """
        Property 8: Observation Accuracy
        
        For any set of indicators, the feature vector should contain
        normalized values that are bounded and meaningful.
        
        **Validates: Requirements 4.1, 4.4, 4.5**
        """
        builder = ObservationBuilder()
        
        indicators = {
            'rsi': rsi,
            'adx': adx,
            'atr': 100.0,
            'atr_percent': atr_percent
        }
        
        current_stats = {
            'close': 47000,
            'sma_20': 46900,
            'sma_50': 46800,
            'volatility': 100,
            'high_20': 47500,
            'low_20': 46500
        }
        
        features = builder._build_feature_vector(indicators, current_stats)
        
        # Verify features are normalized
        assert len(features) == 6
        assert features.dtype == np.float32
        
        # Each feature should be bounded
        for i, feature in enumerate(features):
            assert np.isfinite(feature), f"Feature {i} is not finite: {feature}"
            # Most features should be in [-1, 1] range
            if i < 5:  # First 5 features are normalized
                assert -2 <= feature <= 2, f"Feature {i} out of bounds: {feature}"
    
    @given(
        close_price=st.floats(min_value=40000, max_value=50000, allow_nan=False, allow_infinity=False),
        sma_20=st.floats(min_value=40000, max_value=50000, allow_nan=False, allow_infinity=False),
        sma_50=st.floats(min_value=40000, max_value=50000, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_observation_price_relationships(self, close_price, sma_20, sma_50):
        """
        Property: Feature vector correctly captures price relationships.
        
        For any set of prices, the feature vector should correctly represent
        the relationship between current price and moving averages.
        
        **Validates: Requirements 4.1, 4.4**
        """
        builder = ObservationBuilder()
        
        indicators = {
            'rsi': 50,
            'adx': 25,
            'atr': 100.0,
            'atr_percent': 0.5
        }
        
        current_stats = {
            'close': close_price,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'volatility': 100,
            'high_20': max(close_price, sma_20, sma_50) + 100,
            'low_20': min(close_price, sma_20, sma_50) - 100
        }
        
        features = builder._build_feature_vector(indicators, current_stats)
        
        # Feature 3 is price vs SMA20
        price_vs_sma20 = features[3]
        expected = (close_price - sma_20) / sma_20 if sma_20 > 0 else 0
        
        assert abs(price_vs_sma20 - expected) < 0.01, \
            f"Price vs SMA20 mismatch: {price_vs_sma20} vs {expected}"
        
        # Feature 4 is price vs SMA50
        price_vs_sma50 = features[4]
        expected = (close_price - sma_50) / sma_50 if sma_50 > 0 else 0
        
        assert abs(price_vs_sma50 - expected) < 0.01, \
            f"Price vs SMA50 mismatch: {price_vs_sma50} vs {expected}"


class TestNormalizationConsistencyProperties:
    """Property-based tests for normalization consistency."""
    
    @given(
        rsi=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        adx=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_normalization_consistency(self, rsi, adx):
        """
        Property 9: Normalization Consistency
        
        For any set of indicators, normalizing the same indicators multiple
        times should produce identical feature vectors.
        
        **Validates: Requirements 4.2**
        """
        builder = ObservationBuilder()
        
        indicators = {
            'rsi': rsi,
            'adx': adx,
            'atr': 100.0,
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
        
        # Build feature vectors multiple times
        features1 = builder._build_feature_vector(indicators, current_stats)
        features2 = builder._build_feature_vector(indicators, current_stats)
        features3 = builder._build_feature_vector(indicators, current_stats)
        
        # All should be identical
        assert np.allclose(features1, features2), "Features not consistent on second call"
        assert np.allclose(features2, features3), "Features not consistent on third call"
    
    @given(
        scale_factor=st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_normalization_scale_invariance(self, scale_factor):
        """
        Property: Normalization handles different price scales.
        
        For any price scale, the feature vector should be properly normalized
        regardless of the absolute price level.
        
        **Validates: Requirements 4.2**
        """
        builder = ObservationBuilder()
        
        indicators = {
            'rsi': 50,
            'adx': 25,
            'atr': 100.0,
            'atr_percent': 0.5
        }
        
        # Original stats
        current_stats1 = {
            'close': 47000,
            'sma_20': 46900,
            'sma_50': 46800,
            'volatility': 100,
            'high_20': 47500,
            'low_20': 46500
        }
        
        # Scaled stats
        current_stats2 = {
            'close': 47000 * scale_factor,
            'sma_20': 46900 * scale_factor,
            'sma_50': 46800 * scale_factor,
            'volatility': 100 * scale_factor,
            'high_20': 47500 * scale_factor,
            'low_20': 46500 * scale_factor
        }
        
        features1 = builder._build_feature_vector(indicators, current_stats1)
        features2 = builder._build_feature_vector(indicators, current_stats2)
        
        # Price relationships should be the same (features 3 and 4)
        assert abs(features1[3] - features2[3]) < 0.01, \
            f"Price vs SMA20 not scale invariant: {features1[3]} vs {features2[3]}"
        assert abs(features1[4] - features2[4]) < 0.01, \
            f"Price vs SMA50 not scale invariant: {features1[4]} vs {features2[4]}"


class TestMarketRegimeProperties:
    """Property-based tests for market regime classification."""
    
    @given(
        rsi=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        adx=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_regime_classification_consistency(self, rsi, adx):
        """
        Property: Market regime classification is consistent.
        
        For any set of indicators, classifying the regime multiple times
        should produce identical results.
        
        **Validates: Requirements 4.1, 4.4**
        """
        builder = ObservationBuilder()
        
        indicators = {
            'rsi': rsi,
            'adx': adx,
            'atr': 100.0,
            'atr_percent': 0.5
        }
        
        # Classify regime multiple times
        regime1 = builder._classify_regime(indicators)
        regime2 = builder._classify_regime(indicators)
        regime3 = builder._classify_regime(indicators)
        
        # All should have same regime and strength
        assert regime1.regime == regime2.regime == regime3.regime
        assert regime1.strength == regime2.strength == regime3.strength
    
    @given(
        rsi=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        adx=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_regime_strength_bounds(self, rsi, adx):
        """
        Property: Market regime strength is always bounded.
        
        For any set of indicators, the regime strength should always be
        between 0 and 1.
        
        **Validates: Requirements 4.1**
        """
        builder = ObservationBuilder()
        
        indicators = {
            'rsi': rsi,
            'adx': adx,
            'atr': 100.0,
            'atr_percent': 0.5
        }
        
        regime = builder._classify_regime(indicators)
        
        assert 0 <= regime.strength <= 1, \
            f"Regime strength out of bounds: {regime.strength}"
    
    @given(
        rsi=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        adx=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_regime_classification_rules(self, rsi, adx):
        """
        Property: Market regime classification follows defined rules.
        
        For any set of indicators, the regime should follow the classification rules:
        - RSI > 70 => bullish
        - RSI < 30 => bearish
        - Otherwise => ranging
        
        **Validates: Requirements 4.1, 4.4**
        """
        builder = ObservationBuilder()
        
        indicators = {
            'rsi': rsi,
            'adx': adx,
            'atr': 100.0,
            'atr_percent': 0.5
        }
        
        regime = builder._classify_regime(indicators)
        
        # Verify regime follows rules
        if rsi > builder.RSI_OVERBOUGHT:
            assert regime.regime == "bullish", \
                f"RSI {rsi} > {builder.RSI_OVERBOUGHT} should be bullish, got {regime.regime}"
        elif rsi < builder.RSI_OVERSOLD:
            assert regime.regime == "bearish", \
                f"RSI {rsi} < {builder.RSI_OVERSOLD} should be bearish, got {regime.regime}"
        else:
            assert regime.regime == "ranging", \
                f"RSI {rsi} in neutral zone should be ranging, got {regime.regime}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
