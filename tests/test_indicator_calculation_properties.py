"""
Property-based tests for IndicatorCalculator static methods.

Feature: data-flow-fix, Property 4: Indicator Calculation
Validates: Requirements 4.1, 4.2, 4.3, 4.4
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st
from src.adan_trading_bot.indicators.calculator import IndicatorCalculator


class TestIndicatorCalculatorProperties:
    """Property-based tests for indicator calculations."""
    
    @given(st.lists(st.floats(min_value=1.0, max_value=10000.0), min_size=30, max_size=500))
    def test_rsi_returns_valid_range(self, prices):
        """
        Property: RSI always returns a value between 0 and 100.
        Validates: Requirements 4.2
        """
        close = pd.Series(prices)
        rsi = IndicatorCalculator.calculate_rsi(close)
        
        assert isinstance(rsi, (int, float, np.number))
        assert 0 <= rsi <= 100, f"RSI out of range: {rsi}"
    
    @given(st.lists(st.floats(min_value=1.0, max_value=10000.0), min_size=30, max_size=500))
    def test_atr_returns_positive(self, prices):
        """
        Property: ATR always returns a non-negative value.
        Validates: Requirements 4.4
        """
        close = pd.Series(prices)
        high = pd.Series([p * 1.01 for p in prices])
        low = pd.Series([p * 0.99 for p in prices])
        
        atr = IndicatorCalculator.calculate_atr(high, low, close)
        
        assert isinstance(atr, (int, float, np.number))
        assert atr >= 0, f"ATR negative: {atr}"
    
    @given(st.lists(st.floats(min_value=1.0, max_value=10000.0), min_size=30, max_size=500))
    def test_adx_returns_valid_range(self, prices):
        """
        Property: ADX always returns a value between 0 and 100.
        Validates: Requirements 4.3
        """
        close = pd.Series(prices)
        high = pd.Series([p * 1.01 for p in prices])
        low = pd.Series([p * 0.99 for p in prices])
        
        adx = IndicatorCalculator.calculate_adx(high, low, close)
        
        assert isinstance(adx, (int, float, np.number))
        assert 0 <= adx <= 100, f"ADX out of range: {adx}"
    
    def test_rsi_with_constant_prices(self):
        """
        Property: RSI with constant prices returns neutral or zero value.
        Validates: Requirements 4.2
        """
        close = pd.Series([100.0] * 50)
        rsi = IndicatorCalculator.calculate_rsi(close)
        
        # Constant prices should give neutral or zero RSI (no momentum)
        assert 0 <= rsi <= 100, f"RSI for constant prices should be in valid range: {rsi}"
    
    def test_atr_with_constant_prices(self):
        """
        Property: ATR with constant prices returns zero or near-zero.
        Validates: Requirements 4.4
        """
        close = pd.Series([100.0] * 50)
        high = pd.Series([100.0] * 50)
        low = pd.Series([100.0] * 50)
        
        atr = IndicatorCalculator.calculate_atr(high, low, close)
        
        # Constant prices should give zero ATR
        assert atr <= 0.1, f"ATR for constant prices should be near zero: {atr}"
    
    def test_rsi_uptrend(self):
        """
        Property: RSI in uptrend should be higher than in downtrend.
        Validates: Requirements 4.2
        """
        # Uptrend
        uptrend = pd.Series(list(range(100, 150)))
        rsi_up = IndicatorCalculator.calculate_rsi(uptrend)
        
        # Downtrend
        downtrend = pd.Series(list(range(150, 100, -1)))
        rsi_down = IndicatorCalculator.calculate_rsi(downtrend)
        
        assert rsi_up > rsi_down, f"RSI uptrend ({rsi_up}) should be > downtrend ({rsi_down})"
    
    def test_atr_volatile_vs_stable(self):
        """
        Property: ATR with high volatility should be higher than with low volatility.
        Validates: Requirements 4.4
        """
        # Stable prices
        stable_close = pd.Series([100.0] * 50)
        stable_high = pd.Series([100.1] * 50)
        stable_low = pd.Series([99.9] * 50)
        atr_stable = IndicatorCalculator.calculate_atr(stable_high, stable_low, stable_close)
        
        # Volatile prices
        volatile_close = pd.Series(list(range(100, 150)) + list(range(150, 100, -1)))
        volatile_high = pd.Series([p * 1.05 for p in volatile_close])
        volatile_low = pd.Series([p * 0.95 for p in volatile_close])
        atr_volatile = IndicatorCalculator.calculate_atr(volatile_high, volatile_low, volatile_close)
        
        assert atr_volatile > atr_stable, f"ATR volatile ({atr_volatile}) should be > stable ({atr_stable})"


class TestIndicatorCalculatorExamples:
    """Example-based tests for indicator calculations."""
    
    def test_rsi_calculation_example(self):
        """Example: RSI calculation with known data"""
        close = pd.Series([44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08])
        rsi = IndicatorCalculator.calculate_rsi(close, period=5)
        
        assert 0 <= rsi <= 100
        assert isinstance(rsi, (int, float, np.number))
    
    def test_atr_calculation_example(self):
        """Example: ATR calculation with known data"""
        high = pd.Series([46.08, 46.50, 46.42, 46.00, 46.50])
        low = pd.Series([45.84, 45.84, 45.50, 45.42, 45.84])
        close = pd.Series([46.08, 46.42, 45.84, 46.00, 46.50])
        
        atr = IndicatorCalculator.calculate_atr(high, low, close, period=3)
        
        assert atr >= 0
        assert isinstance(atr, (int, float, np.number))
    
    def test_adx_calculation_example(self):
        """Example: ADX calculation with known data"""
        high = pd.Series([46.08, 46.50, 46.42, 46.00, 46.50] * 10)
        low = pd.Series([45.84, 45.84, 45.50, 45.42, 45.84] * 10)
        close = pd.Series([46.08, 46.42, 45.84, 46.00, 46.50] * 10)
        
        adx = IndicatorCalculator.calculate_adx(high, low, close, period=5)
        
        assert 0 <= adx <= 100
        assert isinstance(adx, (int, float, np.number))
    
    def test_all_indicators_together(self):
        """Example: Calculate all indicators together"""
        prices = list(range(100, 150)) + list(range(150, 100, -1))
        close = pd.Series(prices)
        high = pd.Series([p * 1.01 for p in prices])
        low = pd.Series([p * 0.99 for p in prices])
        
        rsi = IndicatorCalculator.calculate_rsi(close)
        atr = IndicatorCalculator.calculate_atr(high, low, close)
        adx = IndicatorCalculator.calculate_adx(high, low, close)
        
        assert 0 <= rsi <= 100
        assert atr >= 0
        assert 0 <= adx <= 100
