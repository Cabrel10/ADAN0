"""
Unit tests for ADAN Dashboard section renderers

Tests for header, decision matrix, positions, trades, performance, and health sections.
"""

import pytest
from datetime import datetime, timedelta
from rich.panel import Panel

from src.adan_trading_bot.dashboard.sections import (
    render_header,
    render_decision_matrix,
    render_positions,
    render_closed_trades,
    render_performance,
    render_system_health,
)
from src.adan_trading_bot.dashboard.models import (
    Position,
    ClosedTrade,
    Signal,
    MarketContext,
    PortfolioState,
)


class TestHeaderRenderer:
    """Tests for header section renderer"""
    
    def test_render_header_returns_panel(self):
        """Test that render_header returns a Panel"""
        portfolio = PortfolioState(
            total_value_usd=1000.0,
            available_capital_usd=500.0,
        )
        result = render_header(portfolio)
        assert isinstance(result, Panel)
    
    def test_render_header_with_positions(self):
        """Test header rendering with open positions"""
        pos = Position(
            pair="BTCUSDT",
            side="LONG",
            size_btc=0.1,
            entry_price=40000.0,
            current_price=41000.0,
            sl_price=39000.0,
            tp_price=42000.0,
            open_time=datetime.now(),
            entry_signal_strength=0.85,
            entry_market_regime="Trending",
            entry_volatility=2.0,
            entry_rsi=45,
        )
        
        portfolio = PortfolioState(
            total_value_usd=1100.0,
            available_capital_usd=500.0,
            open_positions=[pos],
        )
        
        result = render_header(portfolio)
        assert isinstance(result, Panel)


class TestDecisionMatrixRenderer:
    """Tests for decision matrix section renderer"""
    
    def test_render_decision_matrix_returns_panel(self):
        """Test that render_decision_matrix returns a Panel"""
        signal = Signal(
            direction="BUY",
            confidence=0.87,
            horizon="4h",
            worker_votes={"W1": 0.82, "W2": 0.91, "W3": 0.85, "W4": 0.88},
            decision_driver="Trend",
        )
        
        market_context = MarketContext(
            price=43850.25,
            volatility_atr=2.1,
            rsi=42,
            adx=28,
            trend_strength="Moderate",
            market_regime="Trending",
            volume_change=18.0,
        )
        
        result = render_decision_matrix(signal, market_context)
        assert isinstance(result, Panel)
        assert result.title is not None


class TestPositionsRenderer:
    """Tests for positions section renderer"""
    
    def test_render_positions_empty(self):
        """Test rendering with no positions"""
        result = render_positions([])
        assert isinstance(result, Panel)
    
    def test_render_positions_with_data(self):
        """Test rendering with positions"""
        pos = Position(
            pair="BTCUSDT",
            side="LONG",
            size_btc=0.1,
            entry_price=40000.0,
            current_price=41000.0,
            sl_price=39000.0,
            tp_price=42000.0,
            open_time=datetime.now() - timedelta(hours=2),
            entry_signal_strength=0.85,
            entry_market_regime="Trending",
            entry_volatility=2.0,
            entry_rsi=45,
        )
        
        result = render_positions([pos])
        assert isinstance(result, Panel)
        assert result.title is not None


class TestClosedTradesRenderer:
    """Tests for closed trades section renderer"""
    
    def test_render_closed_trades_empty(self):
        """Test rendering with no trades"""
        result = render_closed_trades([])
        assert isinstance(result, Panel)
    
    def test_render_closed_trades_with_data(self):
        """Test rendering with trades"""
        now = datetime.now()
        trade = ClosedTrade(
            pair="BTCUSDT",
            side="LONG",
            size_btc=0.1,
            entry_price=40000.0,
            exit_price=41000.0,
            open_time=now - timedelta(hours=3),
            close_time=now,
            close_reason="TP Hit",
            entry_confidence=0.85,
        )
        
        result = render_closed_trades([trade])
        assert isinstance(result, Panel)
        assert result.title is not None


class TestPerformanceRenderer:
    """Tests for performance analytics section renderer"""
    
    def test_render_performance_empty(self):
        """Test rendering with no trades"""
        result = render_performance([])
        assert isinstance(result, Panel)
    
    def test_render_performance_with_data(self):
        """Test rendering with trades"""
        now = datetime.now()
        trades = [
            ClosedTrade(
                pair="BTCUSDT",
                side="LONG",
                size_btc=1.0,
                entry_price=40000.0,
                exit_price=41000.0,
                open_time=now - timedelta(hours=3),
                close_time=now,
                close_reason="TP Hit",
                entry_confidence=0.85,
            ),
            ClosedTrade(
                pair="BTCUSDT",
                side="LONG",
                size_btc=1.0,
                entry_price=40000.0,
                exit_price=39000.0,
                open_time=now - timedelta(hours=2),
                close_time=now,
                close_reason="SL Hit",
                entry_confidence=0.75,
            ),
        ]
        
        result = render_performance(trades)
        assert isinstance(result, Panel)
        assert result.title is not None


class TestSystemHealthRenderer:
    """Tests for system health section renderer"""
    
    def test_render_system_health_returns_panel(self):
        """Test that render_system_health returns a Panel"""
        health_data = {
            "api_status": True,
            "api_latency_ms": 50,
            "feed_status": True,
            "feed_lag_ms": 100,
            "model_status": True,
            "model_latency_ms": 75,
            "db_status": True,
            "cpu_percent": 35.5,
            "memory_gb": 1.2,
            "memory_total_gb": 4.0,
            "threads": 8,
            "uptime_percent": 99.7,
            "alerts": [],
        }
        
        result = render_system_health(health_data)
        assert isinstance(result, Panel)
        assert result.title is not None
    
    def test_render_system_health_with_alerts(self):
        """Test rendering with alerts"""
        health_data = {
            "api_status": True,
            "api_latency_ms": 50,
            "feed_status": True,
            "feed_lag_ms": 100,
            "model_status": True,
            "model_latency_ms": 75,
            "db_status": True,
            "cpu_percent": 35.5,
            "memory_gb": 1.2,
            "memory_total_gb": 4.0,
            "threads": 8,
            "uptime_percent": 99.7,
            "alerts": [
                {"severity": "WARNING", "message": "High volatility detected"},
                {"severity": "INFO", "message": "New signal generated"},
            ],
        }
        
        result = render_system_health(health_data)
        assert isinstance(result, Panel)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
