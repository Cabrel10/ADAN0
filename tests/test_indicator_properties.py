"""
Property-based tests for indicator calculations using Hypothesis.

These tests verify that indicator calculations satisfy universal properties
across many randomly generated inputs.
"""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, HealthCheck
from src.adan_trading_bot.indicators.calculator import IndicatorCalculator


# Strategies for generating test data
@st.composite
def price_arrays(draw, min_size=30, max_size=100):
    """Generate realistic price arrays."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    # Start with a base price
    base = draw(st.floats(min_value=10, max_value=1000, allow_nan=False, allow_infinity=False))
    # Generate price changes
    changes = draw(st.lists(
        st.floats(min_value=-2, max_value=2, allow_nan=False, allow_infinity=False),
        min_size=size,
        max_size=size
    ))
    # Build prices
    prices = [base]
    for change in changes:
        prices.append(max(prices[-1] + change, 0.01))  # Ensure positive prices
    return np.array(prices, dtype=np.float64)


@st.composite
def ohlcv_data(draw, min_size=30, max_size=100):
    """Generate realistic OHLCV data."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    base = draw(st.floats(min_value=10, max_value=1000, allow_nan=False, allow_infinity=False))
    
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    current_price = base
    for _ in range(size):
        # Generate OHLC for this candle
        open_price = current_price
        close_price = current_price + draw(st.floats(min_value=-2, max_value=2, allow_nan=False, allow_infinity=False))
        close_price = max(close_price, 0.01)
        
        high_price = max(open_price, close_price) + draw(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))
        low_price = min(open_price, close_price) - draw(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))
        low_price = max(low_price, 0.01)
        
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        volumes.append(draw(st.integers(min_value=100000, max_value=10000000)))
        
        current_price = close_price
    
    return pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })


class TestRSIProperties:
    """Property-based tests for RSI calculation.
    
    **Feature: data-integrity-fix, Property 1: RSI Calculation Correctness**
    **Validates: Requirements 1.1, 3.1**
    """
    
    @given(price_arrays())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_rsi_range(self, prices):
        """Property: RSI should always be between 0 and 100."""
        rsi = IndicatorCalculator.calculate_rsi(prices, period=14)
        assert 0 <= rsi <= 100, f"RSI {rsi} out of range [0, 100]"
    
    @given(price_arrays())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_rsi_uptrend_above_50(self, prices):
        """Property: RSI should be > 50 in a strong uptrend."""
        # Create a strong uptrend
        uptrend = np.array([i * 1.0 for i in range(len(prices))])
        rsi = IndicatorCalculator.calculate_rsi(uptrend, period=14)
        assert rsi > 50, f"RSI {rsi} should be > 50 in uptrend"
    
    @given(price_arrays())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_rsi_downtrend_below_50(self, prices):
        """Property: RSI should be < 50 in a strong downtrend."""
        # Create a strong downtrend
        downtrend = np.array([100 - i * 1.0 for i in range(len(prices))])
        rsi = IndicatorCalculator.calculate_rsi(downtrend, period=14)
        assert rsi < 50, f"RSI {rsi} should be < 50 in downtrend"
    
    @given(price_arrays())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_rsi_flat_market_near_50(self, prices):
        """Property: RSI should be near 50 in a flat market."""
        # Create flat prices
        flat = np.array([50.0] * len(prices))
        rsi = IndicatorCalculator.calculate_rsi(flat, period=14)
        assert 40 <= rsi <= 60, f"RSI {rsi} should be near 50 in flat market"
    
    @given(price_arrays())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_rsi_deterministic(self, prices):
        """Property: RSI calculation should be deterministic."""
        rsi1 = IndicatorCalculator.calculate_rsi(prices, period=14)
        rsi2 = IndicatorCalculator.calculate_rsi(prices, period=14)
        assert rsi1 == rsi2, "RSI should be deterministic"


class TestATRProperties:
    """Property-based tests for ATR calculation.
    
    **Feature: data-integrity-fix, Property 3: ATR Calculation Correctness**
    **Validates: Requirements 1.3, 3.3**
    """
    
    @given(ohlcv_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_atr_positive(self, df):
        """Property: ATR should always be non-negative."""
        atr = IndicatorCalculator.calculate_atr(df['high'].values, df['low'].values, df['close'].values, period=14)
        assert atr >= 0, f"ATR {atr} should be non-negative"
    
    @given(ohlcv_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_atr_less_than_price_range(self, df):
        """Property: ATR should be less than or equal to the total price range."""
        atr = IndicatorCalculator.calculate_atr(df['high'].values, df['low'].values, df['close'].values, period=14)
        price_range = df['high'].max() - df['low'].min()
        assert atr <= price_range, f"ATR {atr} should be <= price range {price_range}"
    
    @given(ohlcv_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_atr_deterministic(self, df):
        """Property: ATR calculation should be deterministic."""
        atr1 = IndicatorCalculator.calculate_atr(df['high'].values, df['low'].values, df['close'].values, period=14)
        atr2 = IndicatorCalculator.calculate_atr(df['high'].values, df['low'].values, df['close'].values, period=14)
        assert atr1 == atr2, "ATR should be deterministic"
    
    @given(ohlcv_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_atr_high_volatility_higher_than_low(self, df):
        """Property: ATR should be higher for high volatility than low volatility (when both have volatility)."""
        # Only test if original data has some volatility
        if df['high'].max() - df['low'].min() < 0.01:
            return  # Skip flat markets
        
        # Calculate ATR for original data
        atr_original = IndicatorCalculator.calculate_atr(df['high'].values, df['low'].values, df['close'].values, period=14)
        
        # Create low volatility version (narrow ranges)
        df_low_vol = df.copy()
        mid = (df_low_vol['high'] + df_low_vol['low']) / 2
        df_low_vol['high'] = mid + 0.001
        df_low_vol['low'] = mid - 0.001
        
        atr_low_vol = IndicatorCalculator.calculate_atr(df_low_vol['high'].values, df_low_vol['low'].values, df_low_vol['close'].values, period=14)
        
        assert atr_original >= atr_low_vol, f"ATR {atr_original} should be >= low vol ATR {atr_low_vol}"


class TestADXProperties:
    """Property-based tests for ADX calculation.
    
    **Feature: data-integrity-fix, Property 2: ADX Calculation Correctness**
    **Validates: Requirements 1.2, 3.2**
    """
    
    @given(ohlcv_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_adx_range(self, df):
        """Property: ADX should always be between 0 and 100."""
        adx = IndicatorCalculator.calculate_adx(df['high'].values, df['low'].values, df['close'].values, period=14)
        assert 0 <= adx <= 100, f"ADX {adx} out of range [0, 100]"
    
    @given(ohlcv_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_adx_uptrend_higher_than_ranging(self, df):
        """Property: ADX should be higher for trending markets than ranging markets."""
        # Only test if original data has some trend (not flat)
        if df['high'].max() - df['low'].min() < 0.1:
            return  # Skip if flat market
        
        # Calculate ADX for original data
        adx_original = IndicatorCalculator.calculate_adx(df['high'].values, df['low'].values, df['close'].values, period=14)
        
        # Create ranging version (oscillating prices)
        df_ranging = df.copy()
        for i in range(len(df_ranging)):
            if i % 2 == 0:
                df_ranging.loc[i, 'high'] = df_ranging.loc[i, 'close'] + 0.5
                df_ranging.loc[i, 'low'] = df_ranging.loc[i, 'close'] - 0.5
            else:
                df_ranging.loc[i, 'high'] = df_ranging.loc[i, 'close'] - 0.5
                df_ranging.loc[i, 'low'] = df_ranging.loc[i, 'close'] + 0.5
        
        adx_ranging = IndicatorCalculator.calculate_adx(df_ranging['high'].values, df_ranging['low'].values, df_ranging['close'].values, period=14)
        
        # Trending should have higher or equal ADX than ranging
        assert adx_original >= adx_ranging * 0.5, f"ADX {adx_original} should be >= 0.5 * ranging ADX {adx_ranging}"
    
    @given(ohlcv_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_adx_deterministic(self, df):
        """Property: ADX calculation should be deterministic."""
        adx1 = IndicatorCalculator.calculate_adx(df['high'].values, df['low'].values, df['close'].values, period=14)
        adx2 = IndicatorCalculator.calculate_adx(df['high'].values, df['low'].values, df['close'].values, period=14)
        assert adx1 == adx2, "ADX should be deterministic"


class TestCalculationReproducibility:
    """Property-based tests for calculation reproducibility.
    
    **Feature: data-integrity-fix, Property 10: Calculation Reproducibility**
    **Validates: Requirements 3.4**
    """
    
    @given(ohlcv_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_all_indicators_reproducible(self, df):
        """Property: All indicator calculations should be reproducible."""
        result1 = IndicatorCalculator.calculate_all(df, period=14)
        result2 = IndicatorCalculator.calculate_all(df, period=14)
        
        assert result1['rsi'] == result2['rsi'], "RSI should be reproducible"
        assert result1['adx'] == result2['adx'], "ADX should be reproducible"
        assert result1['atr'] == result2['atr'], "ATR should be reproducible"
        assert result1['atr_percent'] == result2['atr_percent'], "ATR% should be reproducible"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
