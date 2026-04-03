"""
Unit tests for ADAN Dashboard formatting utilities

Tests for USD, BTC, percentage, time, and other formatting functions.
"""

import pytest
from datetime import timedelta

from src.adan_trading_bot.dashboard.formatters import (
    format_usd,
    format_btc,
    format_percentage,
    format_time,
    format_time_hms,
    format_confidence,
    format_price,
    format_number,
    format_ratio,
    format_rsi_level,
    format_adx_strength,
    format_outcome_symbol,
    format_status_symbol,
    format_signal_symbol,
)


class TestUSDFormatting:
    """Tests for USD formatting"""
    
    def test_format_usd_positive(self):
        """Test formatting positive USD values"""
        assert format_usd(1234.56) == "$1,234.56"
        assert format_usd(1000000.00) == "$1,000,000.00"
    
    def test_format_usd_negative(self):
        """Test formatting negative USD values"""
        assert format_usd(-1234.56) == "-$1,234.56"
    
    def test_format_usd_zero(self):
        """Test formatting zero"""
        assert format_usd(0.0) == "$0.00"
    
    def test_format_usd_decimals(self):
        """Test formatting with different decimal places"""
        assert format_usd(1234.5, decimals=1) == "$1,234.5"
        assert format_usd(1234.567, decimals=3) == "$1,234.567"


class TestBTCFormatting:
    """Tests for BTC formatting"""
    
    def test_format_btc_default(self):
        """Test formatting BTC with default 4 decimals"""
        assert format_btc(0.0245) == "0.0245"
        assert format_btc(1.0) == "1.0000"
    
    def test_format_btc_custom_decimals(self):
        """Test formatting BTC with custom decimals"""
        assert format_btc(0.0245, decimals=2) == "0.02"
        assert format_btc(0.0245, decimals=6) == "0.024500"


class TestPercentageFormatting:
    """Tests for percentage formatting"""
    
    def test_format_percentage_positive(self):
        """Test formatting positive percentages"""
        assert format_percentage(2.5) == "+2.50%"
        assert format_percentage(100.0) == "+100.00%"
    
    def test_format_percentage_negative(self):
        """Test formatting negative percentages"""
        assert format_percentage(-2.5) == "-2.50%"
    
    def test_format_percentage_zero(self):
        """Test formatting zero percentage"""
        assert format_percentage(0.0) == "0.00%"
    
    def test_format_percentage_no_sign(self):
        """Test formatting without sign"""
        assert format_percentage(2.5, include_sign=False) == "2.50%"
        assert format_percentage(-2.5, include_sign=False) == "-2.50%"


class TestTimeFormatting:
    """Tests for time formatting"""
    
    def test_format_time_hours(self):
        """Test formatting time with hours"""
        duration = timedelta(hours=2, minutes=14)
        assert format_time(duration) == "2h14m"
    
    def test_format_time_minutes(self):
        """Test formatting time with minutes"""
        duration = timedelta(minutes=45, seconds=30)
        assert format_time(duration) == "45m30s"
    
    def test_format_time_seconds(self):
        """Test formatting time with seconds"""
        duration = timedelta(seconds=30)
        assert format_time(duration) == "30s"
    
    def test_format_time_hms(self):
        """Test formatting time as HH:MM:SS"""
        duration = timedelta(hours=2, minutes=14, seconds=30)
        assert format_time_hms(duration) == "02:14:30"
    
    def test_format_time_from_seconds(self):
        """Test formatting time from seconds"""
        assert format_time(3600) == "1h0m"
        assert format_time_hms(3600) == "01:00:00"


class TestConfidenceFormatting:
    """Tests for confidence formatting"""
    
    def test_format_confidence(self):
        """Test formatting confidence scores"""
        assert format_confidence(0.87) == "0.87"
        assert format_confidence(0.5) == "0.50"
        assert format_confidence(1.0) == "1.00"


class TestPriceFormatting:
    """Tests for price formatting"""
    
    def test_format_price(self):
        """Test formatting prices"""
        assert format_price(43217.50) == "43,217.50"
        assert format_price(1000000.00) == "1,000,000.00"


class TestRSILevelFormatting:
    """Tests for RSI level formatting"""
    
    def test_rsi_oversold(self):
        """Test RSI oversold classification"""
        assert format_rsi_level(25) == "Oversold"
    
    def test_rsi_neutral(self):
        """Test RSI neutral classification"""
        assert format_rsi_level(50) == "Neutral"
    
    def test_rsi_overbought(self):
        """Test RSI overbought classification"""
        assert format_rsi_level(75) == "Overbought"


class TestADXStrengthFormatting:
    """Tests for ADX strength formatting"""
    
    def test_adx_weak(self):
        """Test ADX weak classification"""
        assert format_adx_strength(20) == "Weak"
    
    def test_adx_moderate(self):
        """Test ADX moderate classification"""
        assert format_adx_strength(35) == "Moderate"
    
    def test_adx_strong(self):
        """Test ADX strong classification"""
        assert format_adx_strength(60) == "Strong"


class TestOutcomeSymbolFormatting:
    """Tests for outcome symbol formatting"""
    
    def test_outcome_win(self):
        """Test win symbol"""
        assert format_outcome_symbol(True, False) == "✅"
    
    def test_outcome_loss(self):
        """Test loss symbol"""
        assert format_outcome_symbol(False, False) == "❌"
    
    def test_outcome_breakeven(self):
        """Test breakeven symbol"""
        assert format_outcome_symbol(False, True) == "⚠️"


class TestStatusSymbolFormatting:
    """Tests for status symbol formatting"""
    
    def test_status_ok(self):
        """Test OK status symbol"""
        assert format_status_symbol(True) == "✅"
    
    def test_status_error(self):
        """Test error status symbol"""
        assert format_status_symbol(False) == "❌"
    
    def test_status_warning(self):
        """Test warning status symbol"""
        assert format_status_symbol(True, warning=True) == "⚠️"


class TestSignalSymbolFormatting:
    """Tests for signal symbol formatting"""
    
    def test_signal_buy(self):
        """Test BUY signal symbol"""
        assert format_signal_symbol("BUY") == "🟢"
    
    def test_signal_sell(self):
        """Test SELL signal symbol"""
        assert format_signal_symbol("SELL") == "🔴"
    
    def test_signal_hold(self):
        """Test HOLD signal symbol"""
        assert format_signal_symbol("HOLD") == "🟡"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
