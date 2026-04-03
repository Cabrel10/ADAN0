"""
Property-based tests for clean_worker_id function.

Feature: data-flow-fix, Property 2: Worker ID Normalization
Validates: Requirements 2.1, 2.2, 2.3, 2.4
"""

import pytest
from hypothesis import given, strategies as st
from src.adan_trading_bot.environment.multi_asset_chunked_env import clean_worker_id


class TestCleanWorkerIdProperties:
    """Property-based tests for worker ID normalization."""
    
    @given(st.integers(min_value=0, max_value=100))
    def test_integer_input_returns_same_integer(self, worker_id):
        """
        Property: For any integer input, clean_worker_id returns the same integer.
        Validates: Requirements 2.3
        """
        result = clean_worker_id(worker_id)
        assert result == worker_id
        assert isinstance(result, int)
    
    @given(st.just(None))
    def test_none_input_returns_zero(self, worker_id):
        """
        Property: For None input, clean_worker_id returns 0.
        Validates: Requirements 2.2
        """
        result = clean_worker_id(worker_id)
        assert result == 0
        assert isinstance(result, int)
    
    @given(st.text(alphabet='Ww', min_size=1, max_size=1))
    def test_w_prefix_with_digits(self, prefix):
        """
        Property: For string 'W' or 'w' prefix with digits, extract numeric part.
        Validates: Requirements 2.4
        """
        for digit in range(10):
            worker_id = f"{prefix}{digit}"
            result = clean_worker_id(worker_id)
            assert result == digit
            assert isinstance(result, int)
    
    @given(st.integers(min_value=0, max_value=100))
    def test_w_prefix_with_integer_string(self, num):
        """
        Property: For string like 'W0', 'W1', etc., extract numeric part.
        Validates: Requirements 2.4
        """
        worker_id = f"W{num}"
        result = clean_worker_id(worker_id)
        assert result == num
        assert isinstance(result, int)
    
    @given(st.integers(min_value=0, max_value=100))
    def test_lowercase_w_prefix_with_integer_string(self, num):
        """
        Property: For string like 'w0', 'w1', etc., extract numeric part.
        Validates: Requirements 2.4
        """
        worker_id = f"w{num}"
        result = clean_worker_id(worker_id)
        assert result == num
        assert isinstance(result, int)
    
    def test_invalid_string_returns_zero(self):
        """
        Property: For invalid string input, clean_worker_id returns 0.
        Validates: Requirements 2.2
        """
        invalid_inputs = ['invalid', 'abc', 'W', 'w', 'WW', '']
        for invalid_input in invalid_inputs:
            result = clean_worker_id(invalid_input)
            assert result == 0
            assert isinstance(result, int)
    
    @given(st.one_of(
        st.integers(min_value=0, max_value=100),
        st.just(None),
        st.text(alphabet='Ww0123456789', min_size=1, max_size=3)
    ))
    def test_result_is_always_integer(self, worker_id):
        """
        Property: For any valid input, clean_worker_id always returns an integer.
        Validates: Requirements 2.1
        """
        result = clean_worker_id(worker_id)
        assert isinstance(result, int)
    
    @given(st.one_of(
        st.integers(min_value=0, max_value=100),
        st.just(None),
        st.text(alphabet='Ww0123456789', min_size=1, max_size=3)
    ))
    def test_result_is_non_negative(self, worker_id):
        """
        Property: For any valid input, clean_worker_id returns a non-negative integer.
        Validates: Requirements 2.1
        """
        result = clean_worker_id(worker_id)
        assert result >= 0


class TestCleanWorkerIdExamples:
    """Example-based tests for worker ID normalization."""
    
    def test_w0_returns_0(self):
        """Example: 'W0' should return 0"""
        assert clean_worker_id('W0') == 0
    
    def test_w1_returns_1(self):
        """Example: 'W1' should return 1"""
        assert clean_worker_id('W1') == 1
    
    def test_lowercase_w0_returns_0(self):
        """Example: 'w0' should return 0"""
        assert clean_worker_id('w0') == 0
    
    def test_lowercase_w5_returns_5(self):
        """Example: 'w5' should return 5"""
        assert clean_worker_id('w5') == 5
    
    def test_integer_0_returns_0(self):
        """Example: 0 should return 0"""
        assert clean_worker_id(0) == 0
    
    def test_integer_5_returns_5(self):
        """Example: 5 should return 5"""
        assert clean_worker_id(5) == 5
    
    def test_none_returns_0(self):
        """Example: None should return 0"""
        assert clean_worker_id(None) == 0
    
    def test_invalid_string_returns_0(self):
        """Example: 'invalid' should return 0"""
        assert clean_worker_id('invalid') == 0
