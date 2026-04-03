"""
Unit tests for data validator.

Tests verify that the data validator correctly compares calculated indicators
against Binance reference data and detects corruption within specified thresholds.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.adan_trading_bot.validation.data_validator import DataValidator, ValidationResult


class TestDataValidator:
    """Test DataValidator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator()
        
        # Mock OHLCV data
        self.mock_ohlcv = [
            [1640995200000, 47000, 47500, 46500, 47200, 1000],  # timestamp, o, h, l, c, v
            [1640995500000, 47200, 47600, 46800, 47400, 1100],
            [1640995800000, 47400, 47800, 47000, 47300, 1200],
        ] * 20  # Repeat to get enough data
    
    def test_validator_initialization(self):
        """Test validator initializes correctly."""
        validator = DataValidator(api_key="test_key", api_secret="test_secret")
        assert validator is not None
        assert validator.WARNING_THRESHOLD == 5.0
        assert validator.HALT_THRESHOLD == 10.0
        assert validator.MAX_DATA_AGE_MINUTES == 5
    
    @patch('ccxt.binance')
    def test_get_reference_indicators_success(self, mock_binance):
        """Test successful retrieval of reference indicators."""
        # Mock exchange
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = self.mock_ohlcv
        mock_binance.return_value = mock_exchange
        
        validator = DataValidator()
        validator.exchange = mock_exchange
        
        result = validator.get_reference_indicators()
        
        # Verify result structure
        assert 'rsi' in result
        assert 'adx' in result
        assert 'atr' in result
        assert 'atr_percent' in result
        assert 'data_timestamp' in result
        assert 'data_age_seconds' in result
        
        # Verify values are reasonable
        assert 0 <= result['rsi'] <= 100
        assert 0 <= result['adx'] <= 100
        assert result['atr'] > 0
        assert result['atr_percent'] > 0
    
    @patch('ccxt.binance')
    def test_get_reference_indicators_insufficient_data(self, mock_binance):
        """Test handling of insufficient data from Binance."""
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = self.mock_ohlcv[:5]  # Too little data
        mock_binance.return_value = mock_exchange
        
        validator = DataValidator()
        validator.exchange = mock_exchange
        
        with pytest.raises(ValueError, match="Insufficient data"):
            validator.get_reference_indicators()
    
    def test_validate_indicators_pass(self):
        """Test validation passes with small deviations."""
        calculated = {'rsi': 50.0, 'adx': 30.0, 'atr': 100.0}
        reference = {'rsi': 51.0, 'adx': 31.0, 'atr': 102.0}  # ~2% deviation
        
        result = self.validator.validate_indicators(calculated, reference)
        
        assert result.status == "pass"
        assert abs(result.rsi_deviation) < 5.0
        assert abs(result.adx_deviation) < 5.0
        assert abs(result.atr_deviation) < 5.0
        assert "PASS" in result.message
    
    def test_validate_indicators_warning(self):
        """Test validation triggers warning with medium deviations."""
        calculated = {'rsi': 50.0, 'adx': 30.0, 'atr': 100.0}
        reference = {'rsi': 53.5, 'adx': 32.1, 'atr': 107.5}  # ~7% deviation
        
        result = self.validator.validate_indicators(calculated, reference)
        
        assert result.status == "warning"
        assert "WARNING" in result.message
        assert max(abs(result.rsi_deviation), abs(result.adx_deviation), abs(result.atr_deviation)) >= 5.0
    
    def test_validate_indicators_halt(self):
        """Test validation triggers halt with large deviations."""
        calculated = {'rsi': 50.0, 'adx': 30.0, 'atr': 100.0}
        reference = {'rsi': 56.0, 'adx': 34.0, 'atr': 120.0}  # ~12-20% deviation
        
        result = self.validator.validate_indicators(calculated, reference)
        
        assert result.status == "halt"
        assert "CRITICAL" in result.message
        assert max(abs(result.rsi_deviation), abs(result.adx_deviation), abs(result.atr_deviation)) >= 10.0
    
    def test_check_data_freshness_fresh(self):
        """Test data freshness check with fresh data."""
        fresh_timestamp = datetime.utcnow() - timedelta(minutes=2)
        
        is_fresh = self.validator.check_data_freshness(fresh_timestamp)
        
        assert is_fresh is True
    
    def test_check_data_freshness_stale(self):
        """Test data freshness check with stale data."""
        stale_timestamp = datetime.utcnow() - timedelta(minutes=10)
        
        is_fresh = self.validator.check_data_freshness(stale_timestamp)
        
        assert is_fresh is False
    
    @patch('ccxt.binance')
    def test_check_mock_data_usage_real(self, mock_binance):
        """Test mock data detection with real data."""
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = [[1640995200000, 47000, 47500, 46500, 47200, 1000]]
        mock_binance.return_value = mock_exchange
        
        validator = DataValidator()
        validator.exchange = mock_exchange
        
        is_real = validator.check_mock_data_usage()
        
        assert is_real is True
    
    @patch('ccxt.binance')
    def test_check_mock_data_usage_mock(self, mock_binance):
        """Test mock data detection with unrealistic data."""
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = [[1640995200000, 1, 2, 0.5, 1.5, 1000]]  # Unrealistic BTC price
        mock_binance.return_value = mock_exchange
        
        validator = DataValidator()
        validator.exchange = mock_exchange
        
        is_real = validator.check_mock_data_usage()
        
        assert is_real is False
    
    @patch('ccxt.binance')
    def test_check_mock_data_usage_no_data(self, mock_binance):
        """Test mock data detection with no data."""
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = []
        mock_binance.return_value = mock_exchange
        
        validator = DataValidator()
        validator.exchange = mock_exchange
        
        is_real = validator.check_mock_data_usage()
        
        assert is_real is False
    
    def test_calculate_deviation(self):
        """Test deviation calculation."""
        # Test normal case
        deviation = self.validator._calculate_deviation(110, 100)
        assert deviation == 10.0
        
        # Test negative deviation
        deviation = self.validator._calculate_deviation(90, 100)
        assert deviation == -10.0
        
        # Test zero reference
        deviation = self.validator._calculate_deviation(50, 0)
        assert deviation == 100.0
        
        # Test both zero
        deviation = self.validator._calculate_deviation(0, 0)
        assert deviation == 0.0
    
    @patch('src.adan_trading_bot.validation.data_validator.DataValidator.get_reference_indicators')
    @patch('src.adan_trading_bot.validation.data_validator.DataValidator.check_mock_data_usage')
    def test_validate_full_pipeline_success(self, mock_check_mock, mock_get_ref):
        """Test successful full validation pipeline."""
        # Mock successful checks
        mock_check_mock.return_value = True
        mock_get_ref.return_value = {
            'rsi': 50.0,
            'adx': 30.0,
            'atr': 100.0,
            'atr_percent': 0.2,
            'data_timestamp': datetime.utcnow()
        }
        
        calculated = {'rsi': 51.0, 'adx': 31.0, 'atr': 102.0}
        
        result = self.validator.validate_full_pipeline(calculated_indicators=calculated)
        
        assert result.status == "pass"
        assert "PASS" in result.message
    
    @patch('src.adan_trading_bot.validation.data_validator.DataValidator.check_mock_data_usage')
    def test_validate_full_pipeline_mock_data(self, mock_check_mock):
        """Test full validation pipeline with mock data detected."""
        mock_check_mock.return_value = False
        
        result = self.validator.validate_full_pipeline()
        
        assert result.status == "halt"
        assert "Mock or test data detected" in result.message
    
    @patch('src.adan_trading_bot.validation.data_validator.DataValidator.get_reference_indicators')
    @patch('src.adan_trading_bot.validation.data_validator.DataValidator.check_mock_data_usage')
    def test_validate_full_pipeline_stale_data(self, mock_check_mock, mock_get_ref):
        """Test full validation pipeline with stale data."""
        mock_check_mock.return_value = True
        mock_get_ref.return_value = {
            'rsi': 50.0,
            'adx': 30.0,
            'atr': 100.0,
            'atr_percent': 0.2,
            'data_timestamp': datetime.utcnow() - timedelta(minutes=10)  # Stale
        }
        
        result = self.validator.validate_full_pipeline()
        
        assert result.status == "warning"
        assert "stale" in result.message.lower()
    
    def test_export_diagnostic_report(self, tmp_path):
        """Test exporting diagnostic report."""
        result = ValidationResult(
            status="warning",
            rsi_deviation=6.0,
            adx_deviation=4.0,
            atr_deviation=8.0,
            reference_values={'rsi': 50.0, 'adx': 30.0, 'atr': 100.0},
            calculated_values={'rsi': 53.0, 'adx': 31.2, 'atr': 108.0},
            timestamp=datetime.utcnow(),
            message="Test warning"
        )
        
        filepath = tmp_path / "test_report.json"
        
        self.validator.export_diagnostic_report(result, str(filepath))
        
        assert filepath.exists()
        
        # Verify content
        import json
        with open(filepath) as f:
            report = json.load(f)
        
        assert report['status'] == "warning"
        assert report['deviations']['rsi'] == 6.0
        assert report['thresholds']['warning'] == 5.0
        assert report['thresholds']['halt'] == 10.0


class TestValidationResult:
    """Test ValidationResult dataclass."""
    
    def test_validation_result_creation(self):
        """Test ValidationResult can be created with all fields."""
        result = ValidationResult(
            status="pass",
            rsi_deviation=2.0,
            adx_deviation=1.5,
            atr_deviation=3.0,
            reference_values={'rsi': 50.0},
            calculated_values={'rsi': 51.0},
            timestamp=datetime.utcnow(),
            message="Test message"
        )
        
        assert result.status == "pass"
        assert result.rsi_deviation == 2.0
        assert result.message == "Test message"
        assert isinstance(result.timestamp, datetime)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
