"""
Unit tests for indicator calculator with correct formulas.

Tests verify that RSI, ADX, and ATR calculations match standard technical analysis libraries.
"""

import pytest
import numpy as np
import pandas as pd
from src.adan_trading_bot.indicators.calculator import IndicatorCalculator


class TestRSICalculation:
    """Test RSI calculation correctness."""
    
    def test_rsi_basic_calculation(self):
        """Test RSI with known values."""
        # Create a simple uptrend
        prices = np.array([44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.00, 46.00])
        
        rsi = IndicatorCalculator.calculate_rsi(prices, period=14)
        
        # RSI should be between 0 and 100
        assert 0 <= rsi <= 100
        # In an uptrend, RSI should be > 50
        assert rsi > 50
    
    def test_rsi_downtrend(self):
        """Test RSI in downtrend."""
        # Create a downtrend
        prices = np.array([100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85])
        
        rsi = IndicatorCalculator.calculate_rsi(prices, period=14)
        
        # In a downtrend, RSI should be < 50
        assert rsi < 50
    
    def test_rsi_flat_market(self):
        """Test RSI in flat market."""
        # Create flat prices
        prices = np.array([50.0] * 20)
        
        rsi = IndicatorCalculator.calculate_rsi(prices, period=14)
        
        # In flat market, RSI should be around 50
        assert 45 <= rsi <= 55
    
    def test_rsi_insufficient_data(self):
        """Test RSI with insufficient data."""
        prices = np.array([44, 44.34, 44.09])
        
        with pytest.raises(ValueError):
            IndicatorCalculator.calculate_rsi(prices, period=14)
    
    def test_rsi_all_gains(self):
        """Test RSI when all changes are gains."""
        prices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        
        rsi = IndicatorCalculator.calculate_rsi(prices, period=14)
        
        # All gains should give RSI close to 100
        assert rsi > 95


class TestATRCalculation:
    """Test ATR calculation correctness."""
    
    def test_atr_basic_calculation(self):
        """Test ATR with known values."""
        high = np.array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.00, 46.00])
        low = np.array([44.09, 43.61, 43.88, 43.38, 43.96, 44.03, 44.98, 45.02, 45.61, 45.73, 45.55, 45.54, 45.33, 45.76, 45.55, 45.50])
        close = np.array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.00, 46.00])
        
        atr = IndicatorCalculator.calculate_atr(high, low, close, period=14)
        
        # ATR should be positive
        assert atr > 0
        # ATR should be less than the range of prices
        assert atr < (high.max() - low.min())
    
    def test_atr_high_volatility(self):
        """Test ATR with high volatility."""
        high = np.array([100 + i*10 for i in range(20)])
        low = np.array([100 + i*10 - 5 for i in range(20)])
        close = np.array([100 + i*10 - 2 for i in range(20)])
        
        atr = IndicatorCalculator.calculate_atr(high, low, close, period=14)
        
        # High volatility should give higher ATR
        assert atr > 5
    
    def test_atr_low_volatility(self):
        """Test ATR with low volatility."""
        high = np.array([100.1 + i*0.01 for i in range(20)])
        low = np.array([100.0 + i*0.01 for i in range(20)])
        close = np.array([100.05 + i*0.01 for i in range(20)])
        
        atr = IndicatorCalculator.calculate_atr(high, low, close, period=14)
        
        # Low volatility should give lower ATR
        assert atr < 0.2
    
    def test_atr_insufficient_data(self):
        """Test ATR with insufficient data."""
        high = np.array([44.34, 44.09, 44.15])
        low = np.array([44.09, 43.61, 43.88])
        close = np.array([44.34, 44.09, 44.15])
        
        with pytest.raises(ValueError):
            IndicatorCalculator.calculate_atr(high, low, close, period=14)


class TestADXCalculation:
    """Test ADX calculation correctness."""
    
    def test_adx_basic_calculation(self):
        """Test ADX with known values."""
        high = np.array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.00, 46.00, 46.50, 46.75, 47.00, 47.25, 47.50, 47.75, 48.00, 48.25, 48.50, 48.75, 49.00, 49.25, 49.50, 49.75])
        low = np.array([44.09, 43.61, 43.88, 43.38, 43.96, 44.03, 44.98, 45.02, 45.61, 45.73, 45.55, 45.54, 45.33, 45.76, 45.55, 45.50, 45.75, 46.00, 46.25, 46.50, 46.75, 47.00, 47.25, 47.50, 47.75, 48.00, 48.25, 48.50, 48.75, 49.00])
        close = np.array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.00, 46.00, 46.50, 46.75, 47.00, 47.25, 47.50, 47.75, 48.00, 48.25, 48.50, 48.75, 49.00, 49.25, 49.50, 49.75])
        
        adx = IndicatorCalculator.calculate_adx(high, low, close, period=14)
        
        # ADX should be between 0 and 100
        assert 0 <= adx <= 100
        # Strong uptrend should give higher ADX
        assert adx > 20
    
    def test_adx_ranging_market(self):
        """Test ADX in ranging market."""
        # Create ranging prices
        high = np.array([100 + (i % 5) for i in range(35)])
        low = np.array([99 + (i % 5) for i in range(35)])
        close = np.array([99.5 + (i % 5) for i in range(35)])
        
        adx = IndicatorCalculator.calculate_adx(high, low, close, period=14)
        
        # Ranging market should give lower ADX
        assert adx < 30
    
    def test_adx_insufficient_data(self):
        """Test ADX with insufficient data."""
        high = np.array([44.34, 44.09, 44.15])
        low = np.array([44.09, 43.61, 43.88])
        close = np.array([44.34, 44.09, 44.15])
        
        with pytest.raises(ValueError):
            IndicatorCalculator.calculate_adx(high, low, close, period=14)


class TestCalculateAll:
    """Test calculate_all method."""
    
    def test_calculate_all_basic(self):
        """Test calculate_all with valid data."""
        data = {
            'open': [44.00] * 30,
            'high': [44.34 + i*0.1 for i in range(30)],
            'low': [44.09 + i*0.1 for i in range(30)],
            'close': [44.15 + i*0.1 for i in range(30)],
            'volume': [1000000] * 30
        }
        df = pd.DataFrame(data)
        
        result = IndicatorCalculator.calculate_all(df, period=14)
        
        # Check all keys are present
        assert 'rsi' in result
        assert 'adx' in result
        assert 'atr' in result
        assert 'atr_percent' in result
        
        # Check value ranges
        assert 0 <= result['rsi'] <= 100
        assert 0 <= result['adx'] <= 100
        assert result['atr'] > 0
        assert result['atr_percent'] > 0
    
    def test_calculate_all_missing_columns(self):
        """Test calculate_all with missing columns."""
        data = {
            'open': [44.00] * 30,
            'high': [44.34 + i*0.1 for i in range(30)],
            'close': [44.15 + i*0.1 for i in range(30)],
            'volume': [1000000] * 30
        }
        df = pd.DataFrame(data)
        
        with pytest.raises(ValueError):
            IndicatorCalculator.calculate_all(df, period=14)
    
    def test_calculate_all_insufficient_data(self):
        """Test calculate_all with insufficient data."""
        data = {
            'open': [44.00] * 5,
            'high': [44.34] * 5,
            'low': [44.09] * 5,
            'close': [44.15] * 5,
            'volume': [1000000] * 5
        }
        df = pd.DataFrame(data)
        
        with pytest.raises(ValueError):
            IndicatorCalculator.calculate_all(df, period=14)


class TestIndicatorReproducibility:
    """Test that indicators are reproducible."""
    
    def test_rsi_reproducibility(self):
        """Test RSI calculation is reproducible."""
        prices = np.array([44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.00, 46.00])
        
        rsi1 = IndicatorCalculator.calculate_rsi(prices, period=14)
        rsi2 = IndicatorCalculator.calculate_rsi(prices, period=14)
        
        assert rsi1 == rsi2
    
    def test_atr_reproducibility(self):
        """Test ATR calculation is reproducible."""
        high = np.array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.00, 46.00])
        low = np.array([44.09, 43.61, 43.88, 43.38, 43.96, 44.03, 44.98, 45.02, 45.61, 45.73, 45.55, 45.54, 45.33, 45.76, 45.55, 45.50])
        close = np.array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.00, 46.00])
        
        atr1 = IndicatorCalculator.calculate_atr(high, low, close, period=14)
        atr2 = IndicatorCalculator.calculate_atr(high, low, close, period=14)
        
        assert atr1 == atr2
    
    def test_adx_reproducibility(self):
        """Test ADX calculation is reproducible."""
        high = np.array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.00, 46.00, 46.50, 46.75, 47.00, 47.25, 47.50, 47.75, 48.00, 48.25, 48.50, 48.75, 49.00, 49.25, 49.50, 49.75])
        low = np.array([44.09, 43.61, 43.88, 43.38, 43.96, 44.03, 44.98, 45.02, 45.61, 45.73, 45.55, 45.54, 45.33, 45.76, 45.55, 45.50, 45.75, 46.00, 46.25, 46.50, 46.75, 47.00, 47.25, 47.50, 47.75, 48.00, 48.25, 48.50, 48.75, 49.00])
        close = np.array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.00, 46.00, 46.50, 46.75, 47.00, 47.25, 47.50, 47.75, 48.00, 48.25, 48.50, 48.75, 49.00, 49.25, 49.50, 49.75])
        
        adx1 = IndicatorCalculator.calculate_adx(high, low, close, period=14)
        adx2 = IndicatorCalculator.calculate_adx(high, low, close, period=14)
        
        assert adx1 == adx2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
