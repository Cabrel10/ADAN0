"""
Property-based tests for ADAN Dashboard color coding

Tests that colors are correctly assigned based on P&L, confidence, and risk values.
"""

import pytest
from hypothesis import given, strategies as st

from src.adan_trading_bot.dashboard.colors import (
    get_pnl_color,
    get_confidence_color,
    get_risk_color,
    get_signal_color,
    get_status_color,
    TradingColor,
)


class TestPnLColorCoding:
    """Tests for P&L color coding"""
    
    @given(pnl=st.floats(min_value=-10, max_value=10))
    def test_pnl_color_bounds(self, pnl):
        """
        Property: For any P&L percentage, get_pnl_color returns a valid color.
        
        **Feature: adan-btc-dashboard, Property 8: Color Coding Correctness**
        **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**
        """
        color = get_pnl_color(pnl)
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7  # Hex color format
    
    def test_pnl_color_large_profit(self):
        """Test color for large profit (>2%)"""
        color = get_pnl_color(3.0)
        assert color == TradingColor.PROFIT_LARGE.value
    
    def test_pnl_color_small_profit(self):
        """Test color for small profit (0-2%)"""
        color = get_pnl_color(1.0)
        assert color == TradingColor.PROFIT_SMALL.value
    
    def test_pnl_color_breakeven(self):
        """Test color for breakeven (-0.5% to +0.5%)"""
        color = get_pnl_color(0.0)
        assert color == TradingColor.BREAKEVEN.value
        
        color = get_pnl_color(0.3)
        assert color == TradingColor.BREAKEVEN.value
        
        color = get_pnl_color(-0.3)
        assert color == TradingColor.BREAKEVEN.value
    
    def test_pnl_color_small_loss(self):
        """Test color for small loss (0-2%)"""
        color = get_pnl_color(-1.0)
        assert color == TradingColor.LOSS_SMALL.value
    
    def test_pnl_color_large_loss(self):
        """Test color for large loss (>2%)"""
        color = get_pnl_color(-3.0)
        assert color == TradingColor.LOSS_LARGE.value
    
    @given(pnl=st.floats(min_value=-10, max_value=-2.1))
    def test_pnl_large_loss_range(self, pnl):
        """Property: All P&L < -2% should get large loss color"""
        color = get_pnl_color(pnl)
        assert color == TradingColor.LOSS_LARGE.value
    
    @given(pnl=st.floats(min_value=2.1, max_value=10))
    def test_pnl_large_profit_range(self, pnl):
        """Property: All P&L > 2% should get large profit color"""
        color = get_pnl_color(pnl)
        assert color == TradingColor.PROFIT_LARGE.value


class TestConfidenceColorCoding:
    """Tests for confidence color coding"""
    
    @given(confidence=st.floats(min_value=0.0, max_value=1.0))
    def test_confidence_color_bounds(self, confidence):
        """
        Property: For any confidence (0.0-1.0), get_confidence_color returns a valid color.
        
        **Feature: adan-btc-dashboard, Property 8: Color Coding Correctness**
        **Validates: Requirements 8.6, 8.7, 8.8, 8.9, 8.10**
        """
        color = get_confidence_color(confidence)
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7
    
    def test_confidence_very_high(self):
        """Test color for very high confidence (0.9-1.0)"""
        color = get_confidence_color(0.95)
        assert color == TradingColor.CONFIDENCE_VERY_HIGH.value
    
    def test_confidence_high(self):
        """Test color for high confidence (0.8-0.9)"""
        color = get_confidence_color(0.85)
        assert color == TradingColor.CONFIDENCE_HIGH.value
    
    def test_confidence_moderate(self):
        """Test color for moderate confidence (0.7-0.8)"""
        color = get_confidence_color(0.75)
        assert color == TradingColor.CONFIDENCE_MODERATE.value
    
    def test_confidence_low(self):
        """Test color for low confidence (0.6-0.7)"""
        color = get_confidence_color(0.65)
        assert color == TradingColor.CONFIDENCE_LOW.value
    
    def test_confidence_very_low(self):
        """Test color for very low confidence (0.0-0.6)"""
        color = get_confidence_color(0.5)
        assert color == TradingColor.CONFIDENCE_VERY_LOW.value
    
    @given(confidence=st.floats(min_value=0.9, max_value=1.0))
    def test_confidence_very_high_range(self, confidence):
        """Property: All confidence >= 0.9 should get very high color"""
        color = get_confidence_color(confidence)
        assert color == TradingColor.CONFIDENCE_VERY_HIGH.value
    
    @given(confidence=st.floats(min_value=0.0, max_value=0.6))
    def test_confidence_very_low_range(self, confidence):
        """Property: All confidence < 0.6 should get very low color"""
        color = get_confidence_color(confidence)
        assert color == TradingColor.CONFIDENCE_VERY_LOW.value


class TestRiskColorCoding:
    """Tests for risk color coding"""
    
    @given(risk=st.floats(min_value=0.0, max_value=10.0))
    def test_risk_color_bounds(self, risk):
        """
        Property: For any risk percentage, get_risk_color returns a valid color.
        
        **Feature: adan-btc-dashboard, Property 8: Color Coding Correctness**
        **Validates: Requirements 8.11, 8.12, 8.13, 8.14**
        """
        color = get_risk_color(risk)
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7
    
    def test_risk_low(self):
        """Test color for low risk (<1%)"""
        color = get_risk_color(0.5)
        assert color == TradingColor.RISK_LOW.value
    
    def test_risk_medium(self):
        """Test color for medium risk (1-2%)"""
        color = get_risk_color(1.5)
        assert color == TradingColor.RISK_MEDIUM.value
    
    def test_risk_high(self):
        """Test color for high risk (2-3%)"""
        color = get_risk_color(2.5)
        assert color == TradingColor.RISK_HIGH.value
    
    def test_risk_critical(self):
        """Test color for critical risk (>3%)"""
        color = get_risk_color(5.0)
        assert color == TradingColor.RISK_CRITICAL.value
    
    @given(risk=st.floats(min_value=0.0, max_value=0.99))
    def test_risk_low_range(self, risk):
        """Property: All risk < 1% should get low color"""
        color = get_risk_color(risk)
        assert color == TradingColor.RISK_LOW.value
    
    @given(risk=st.floats(min_value=3.0, max_value=10.0))
    def test_risk_critical_range(self, risk):
        """Property: All risk >= 3% should get critical color"""
        color = get_risk_color(risk)
        assert color == TradingColor.RISK_CRITICAL.value


class TestSignalColorCoding:
    """Tests for signal color coding"""
    
    def test_signal_buy_color(self):
        """Test color for BUY signal"""
        color = get_signal_color("BUY")
        assert color == TradingColor.SIGNAL_BUY.value
    
    def test_signal_sell_color(self):
        """Test color for SELL signal"""
        color = get_signal_color("SELL")
        assert color == TradingColor.SIGNAL_SELL.value
    
    def test_signal_hold_color(self):
        """Test color for HOLD signal"""
        color = get_signal_color("HOLD")
        assert color == TradingColor.SIGNAL_HOLD.value
    
    @given(signal=st.sampled_from(["BUY", "SELL", "HOLD"]))
    def test_signal_color_valid(self, signal):
        """
        Property: For any valid signal, get_signal_color returns a valid color.
        
        **Feature: adan-btc-dashboard, Property 8: Color Coding Correctness**
        **Validates: Requirements 8.15, 8.16, 8.17**
        """
        color = get_signal_color(signal)
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7


class TestStatusColorCoding:
    """Tests for status color coding"""
    
    def test_status_ok(self):
        """Test color for OK status"""
        color = get_status_color(True)
        assert color == TradingColor.STATUS_OK.value
    
    def test_status_error(self):
        """Test color for error status"""
        color = get_status_color(False)
        assert color == TradingColor.STATUS_ERROR.value
    
    def test_status_warning(self):
        """Test color for warning status"""
        color = get_status_color(True, warning=True)
        assert color == TradingColor.STATUS_WARNING.value
        
        color = get_status_color(False, warning=True)
        assert color == TradingColor.STATUS_WARNING.value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
