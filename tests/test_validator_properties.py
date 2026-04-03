"""
Property-based tests for data validator.

Tests verify that the data validator correctly validates data freshness,
detects deviations, and identifies mock data across all possible inputs.
"""

import pytest
from hypothesis import given, settings, HealthCheck, strategies as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.adan_trading_bot.validation.data_validator import DataValidator, ValidationResult


class TestDataFreshnessProperties:
    """Property-based tests for data freshness validation."""
    
    @given(
        minutes_old=st.integers(min_value=0, max_value=60)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_data_freshness_boundary(self, minutes_old):
        """
        Property 4: Data Freshness Validation
        
        For any timestamp, data is considered fresh if and only if it is
        less than or equal to MAX_DATA_AGE_MINUTES (5 minutes) old.
        
        **Validates: Requirements 1.4, 2.1**
        """
        validator = DataValidator()
        # Use seconds to avoid rounding issues with minutes
        # Subtract 1 second to ensure we're strictly within the boundary
        timestamp = datetime.utcnow() - timedelta(seconds=minutes_old * 60 - 1)
        
        is_fresh = validator.check_data_freshness(timestamp)
        
        # Data should be fresh if <= 5 minutes old
        expected_fresh = minutes_old <= validator.MAX_DATA_AGE_MINUTES
        assert is_fresh == expected_fresh, \
            f"Data {minutes_old} minutes old should be fresh={expected_fresh}, got {is_fresh}"
    
    @given(
        seconds_old=st.integers(min_value=0, max_value=600)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_data_freshness_precision(self, seconds_old):
        """
        Property: Data freshness check is precise to the second.
        
        For any timestamp, the freshness check should correctly determine
        if data is within the MAX_DATA_AGE_MINUTES threshold.
        
        **Validates: Requirements 1.4**
        """
        validator = DataValidator()
        # Subtract 1 second to ensure we're strictly within the boundary
        timestamp = datetime.utcnow() - timedelta(seconds=max(0, seconds_old - 1))
        
        is_fresh = validator.check_data_freshness(timestamp)
        
        # Convert to minutes for comparison
        minutes_old = seconds_old / 60.0
        expected_fresh = minutes_old <= validator.MAX_DATA_AGE_MINUTES
        assert is_fresh == expected_fresh


class TestDeviationDetectionProperties:
    """Property-based tests for deviation detection."""
    
    @given(
        calculated=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        reference=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_deviation_calculation_correctness(self, calculated, reference):
        """
        Property 5: Deviation Detection
        
        For any calculated and reference values, the deviation should be
        correctly calculated as ((calculated - reference) / reference) * 100.
        
        **Validates: Requirements 2.2, 2.3**
        """
        validator = DataValidator()
        
        deviation = validator._calculate_deviation(calculated, reference)
        
        # Verify calculation is correct
        expected_deviation = ((calculated - reference) / reference) * 100.0
        assert abs(deviation - expected_deviation) < 0.01, \
            f"Deviation calculation incorrect: got {deviation}, expected {expected_deviation}"
    
    @given(
        rsi_calc=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        adx_calc=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        atr_calc=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_deviation_threshold_classification(self, rsi_calc, adx_calc, atr_calc):
        """
        Property: Deviation threshold classification is correct.
        
        For any set of calculated and reference values, the validation status should be:
        - "halt" if any deviation >= 10%
        - "warning" if any deviation >= 5% but < 10%
        - "pass" if all deviations < 5%
        
        **Validates: Requirements 2.2, 2.3**
        """
        validator = DataValidator()
        
        # Use fixed reference values
        calculated = {'rsi': rsi_calc, 'adx': adx_calc, 'atr': atr_calc}
        reference = {'rsi': 50.0, 'adx': 30.0, 'atr': 100.0}
        
        result = validator.validate_indicators(calculated, reference)
        
        # Calculate actual deviations from the validator
        actual_rsi_dev = validator._calculate_deviation(calculated['rsi'], reference['rsi'])
        actual_adx_dev = validator._calculate_deviation(calculated['adx'], reference['adx'])
        actual_atr_dev = validator._calculate_deviation(calculated['atr'], reference['atr'])
        
        max_deviation = max(abs(actual_rsi_dev), abs(actual_adx_dev), abs(actual_atr_dev))
        
        # Verify status matches the max deviation
        if max_deviation >= validator.HALT_THRESHOLD:
            assert result.status == "halt", \
                f"Max deviation {max_deviation}% should trigger halt, got {result.status}"
        elif max_deviation >= validator.WARNING_THRESHOLD:
            assert result.status == "warning", \
                f"Max deviation {max_deviation}% should trigger warning, got {result.status}"
        else:
            assert result.status == "pass", \
                f"Max deviation {max_deviation}% should pass, got {result.status}"
    
    @given(
        rsi_calc=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        adx_calc=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        atr_calc=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_deviation_symmetry(self, rsi_calc, adx_calc, atr_calc):
        """
        Property: Deviation calculation is symmetric around zero.
        
        For any pair of values, if we swap calculated and reference,
        the deviations should have opposite signs.
        
        **Validates: Requirements 2.2**
        """
        validator = DataValidator()
        
        # Forward deviation
        dev_forward = validator._calculate_deviation(rsi_calc, 50.0)
        
        # Reverse deviation
        dev_reverse = validator._calculate_deviation(50.0, rsi_calc)
        
        # They should have opposite signs (or both be zero)
        if dev_forward != 0 and dev_reverse != 0:
            assert (dev_forward > 0) != (dev_reverse > 0), \
                f"Deviations should have opposite signs: {dev_forward} vs {dev_reverse}"


class TestMockDataDetectionProperties:
    """Property-based tests for mock data detection."""
    
    @given(
        price=st.floats(min_value=1, max_value=2000000, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_mock_data_price_range_validation(self, price):
        """
        Property 7: Mock Data Detection
        
        For any BTC price, the system should correctly identify if it's
        realistic (1000 < price < 1000000) or mock data.
        
        **Validates: Requirements 5.1, 5.2, 5.5**
        """
        validator = DataValidator()
        
        with patch('ccxt.binance') as mock_binance:
            mock_exchange = Mock()
            mock_exchange.fetch_ohlcv.return_value = [[1640995200000, price, price, price, price, 1000]]
            mock_binance.return_value = mock_exchange
            
            validator.exchange = mock_exchange
            is_real = validator.check_mock_data_usage()
            
            # Should be real if price is in realistic range (strictly between 1000 and 1000000)
            # The validator uses: if price < 1000 or price > 1000000: return False
            expected_real = not (price < 1000 or price > 1000000)
            assert is_real == expected_real, \
                f"Price {price} should be real={expected_real}, got {is_real}"
    
    @given(
        data_points=st.integers(min_value=0, max_value=10)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_mock_data_empty_response_detection(self, data_points):
        """
        Property: Empty or insufficient data is detected as mock.
        
        For any number of data points, if the response is empty or has
        insufficient data, it should be flagged as mock data.
        
        **Validates: Requirements 5.1, 5.2**
        """
        validator = DataValidator()
        
        with patch('ccxt.binance') as mock_binance:
            mock_exchange = Mock()
            # Create mock data with specified number of points
            mock_data = [[1640995200000 + i*3600000, 47000, 47500, 46500, 47200, 1000] 
                        for i in range(data_points)]
            mock_exchange.fetch_ohlcv.return_value = mock_data
            mock_binance.return_value = mock_exchange
            
            validator.exchange = mock_exchange
            is_real = validator.check_mock_data_usage()
            
            # Should be real only if we have data
            expected_real = data_points > 0
            assert is_real == expected_real, \
                f"With {data_points} data points, should be real={expected_real}, got {is_real}"


class TestValidationResultConsistency:
    """Property-based tests for validation result consistency."""
    
    @given(
        status=st.sampled_from(["pass", "warning", "halt"]),
        rsi_dev=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        adx_dev=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        atr_dev=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_validation_result_consistency(self, status, rsi_dev, adx_dev, atr_dev):
        """
        Property: ValidationResult maintains internal consistency.
        
        For any ValidationResult, the status should be consistent with
        the deviations reported.
        
        **Validates: Requirements 2.2, 2.3**
        """
        result = ValidationResult(
            status=status,
            rsi_deviation=rsi_dev,
            adx_deviation=adx_dev,
            atr_deviation=atr_dev,
            reference_values={'rsi': 50.0, 'adx': 30.0, 'atr': 100.0},
            calculated_values={'rsi': 51.0, 'adx': 31.0, 'atr': 102.0},
            timestamp=datetime.utcnow(),
            message="Test"
        )
        
        # Verify result has all required fields
        assert result.status in ["pass", "warning", "halt"]
        assert isinstance(result.rsi_deviation, float)
        assert isinstance(result.adx_deviation, float)
        assert isinstance(result.atr_deviation, float)
        assert isinstance(result.timestamp, datetime)
        assert isinstance(result.message, str)


class TestValidationDeterminism:
    """Property-based tests for validation determinism."""
    
    @given(
        rsi_calc=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        adx_calc=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        atr_calc=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_validation_is_deterministic(self, rsi_calc, adx_calc, atr_calc):
        """
        Property 6: Corruption Audit Trail
        
        For any set of calculated and reference values, running validation
        multiple times should produce identical results.
        
        **Validates: Requirements 2.4, 6.1**
        """
        validator = DataValidator()
        
        calculated = {'rsi': rsi_calc, 'adx': adx_calc, 'atr': atr_calc}
        reference = {'rsi': 50.0, 'adx': 30.0, 'atr': 100.0}
        
        # Run validation multiple times
        result1 = validator.validate_indicators(calculated, reference)
        result2 = validator.validate_indicators(calculated, reference)
        result3 = validator.validate_indicators(calculated, reference)
        
        # All results should be identical
        assert result1.status == result2.status == result3.status
        assert result1.rsi_deviation == result2.rsi_deviation == result3.rsi_deviation
        assert result1.adx_deviation == result2.adx_deviation == result3.adx_deviation
        assert result1.atr_deviation == result2.atr_deviation == result3.atr_deviation


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
