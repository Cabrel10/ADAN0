"""
Unit tests for ADAN Dashboard data models

Tests for Position, ClosedTrade, Signal, MarketContext, and PortfolioState models.
"""

import pytest
from datetime import datetime, timedelta
from src.adan_trading_bot.dashboard.models import (
    Position,
    ClosedTrade,
    Signal,
    MarketContext,
    PortfolioState,
    SignalDirection,
    CloseReason,
)


class TestPosition:
    """Tests for Position model"""
    
    def test_position_creation(self):
        """Test creating a position"""
        pos = Position(
            pair="BTCUSDT",
            side="LONG",
            size_btc=0.0245,
            entry_price=43217.50,
            current_price=43850.25,
            sl_price=41890.98,
            tp_price=44579.63,
            open_time=datetime.now(),
            entry_signal_strength=0.91,
            entry_market_regime="Trending",
            entry_volatility=1.8,
            entry_rsi=38,
        )
        assert pos.pair == "BTCUSDT"
        assert pos.side == "LONG"
        assert pos.size_btc == 0.0245
    
    def test_long_position_pnl_calculation(self):
        """Test P&L calculation for LONG position"""
        pos = Position(
            pair="BTCUSDT",
            side="LONG",
            size_btc=1.0,
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
        
        # P&L should be (41000 - 40000) * 1.0 = 1000
        assert pos.unrealized_pnl_usd == 1000.0
        # P&L % should be (41000 - 40000) / 40000 * 100 = 2.5%
        assert abs(pos.unrealized_pnl_pct - 2.5) < 0.01
    
    def test_short_position_pnl_calculation(self):
        """Test P&L calculation for SHORT position"""
        pos = Position(
            pair="BTCUSDT",
            side="SHORT",
            size_btc=1.0,
            entry_price=40000.0,
            current_price=39000.0,
            sl_price=41000.0,
            tp_price=38000.0,
            open_time=datetime.now(),
            entry_signal_strength=0.85,
            entry_market_regime="Trending",
            entry_volatility=2.0,
            entry_rsi=45,
        )
        
        # P&L should be (40000 - 39000) * 1.0 = 1000
        assert pos.unrealized_pnl_usd == 1000.0
        # P&L % should be (40000 - 39000) / 40000 * 100 = 2.5%
        assert abs(pos.unrealized_pnl_pct - 2.5) < 0.01
    
    def test_position_duration(self):
        """Test position duration calculation"""
        now = datetime.now()
        past = now - timedelta(hours=2, minutes=14)
        
        pos = Position(
            pair="BTCUSDT",
            side="LONG",
            size_btc=0.0245,
            entry_price=43217.50,
            current_price=43850.25,
            sl_price=41890.98,
            tp_price=44579.63,
            open_time=past,
            entry_signal_strength=0.91,
            entry_market_regime="Trending",
            entry_volatility=1.8,
            entry_rsi=38,
        )
        
        duration = pos.duration
        assert duration.total_seconds() > 0
        assert duration.total_seconds() < 10000  # Less than 3 hours
    
    def test_sl_distance_calculation(self):
        """Test SL distance calculation"""
        pos = Position(
            pair="BTCUSDT",
            side="LONG",
            size_btc=1.0,
            entry_price=40000.0,
            current_price=41000.0,
            sl_price=38800.0,  # 3% below entry
            tp_price=42000.0,
            open_time=datetime.now(),
            entry_signal_strength=0.85,
            entry_market_regime="Trending",
            entry_volatility=2.0,
            entry_rsi=45,
        )
        
        # SL distance should be 3%
        assert abs(pos.sl_distance_pct - 3.0) < 0.01
    
    def test_tp_distance_calculation(self):
        """Test TP distance calculation"""
        pos = Position(
            pair="BTCUSDT",
            side="LONG",
            size_btc=1.0,
            entry_price=40000.0,
            current_price=41000.0,
            sl_price=38800.0,
            tp_price=41200.0,  # 3% above entry
            open_time=datetime.now(),
            entry_signal_strength=0.85,
            entry_market_regime="Trending",
            entry_volatility=2.0,
            entry_rsi=45,
        )
        
        # TP distance should be 3%
        assert abs(pos.tp_distance_pct - 3.0) < 0.01
    
    def test_position_value_calculation(self):
        """Test position value calculation"""
        pos = Position(
            pair="BTCUSDT",
            side="LONG",
            size_btc=0.5,
            entry_price=40000.0,
            current_price=41000.0,
            sl_price=38800.0,
            tp_price=42000.0,
            open_time=datetime.now(),
            entry_signal_strength=0.85,
            entry_market_regime="Trending",
            entry_volatility=2.0,
            entry_rsi=45,
        )
        
        # Position value should be 0.5 * 41000 = 20500
        assert pos.position_value_usd == 20500.0


class TestClosedTrade:
    """Tests for ClosedTrade model"""
    
    def test_closed_trade_creation(self):
        """Test creating a closed trade"""
        now = datetime.now()
        past = now - timedelta(hours=3, minutes=22)
        
        trade = ClosedTrade(
            pair="BTCUSDT",
            side="LONG",
            size_btc=0.018,
            entry_price=42150.0,
            exit_price=43217.5,
            open_time=past,
            close_time=now,
            close_reason="TP Hit",
            entry_confidence=0.91,
        )
        
        assert trade.pair == "BTCUSDT"
        assert trade.side == "LONG"
        assert trade.is_win is True
    
    def test_long_trade_pnl_calculation(self):
        """Test P&L calculation for LONG closed trade"""
        now = datetime.now()
        past = now - timedelta(hours=3)
        
        trade = ClosedTrade(
            pair="BTCUSDT",
            side="LONG",
            size_btc=1.0,
            entry_price=40000.0,
            exit_price=41000.0,
            open_time=past,
            close_time=now,
            close_reason="TP Hit",
            entry_confidence=0.85,
        )
        
        # P&L should be (41000 - 40000) * 1.0 = 1000
        assert trade.realized_pnl_usd == 1000.0
        # P&L % should be 2.5%
        assert abs(trade.realized_pnl_pct - 2.5) < 0.01
    
    def test_short_trade_pnl_calculation(self):
        """Test P&L calculation for SHORT closed trade"""
        now = datetime.now()
        past = now - timedelta(hours=1, minutes=45)
        
        trade = ClosedTrade(
            pair="BTCUSDT",
            side="SHORT",
            size_btc=1.0,
            entry_price=40000.0,
            exit_price=39000.0,
            open_time=past,
            close_time=now,
            close_reason="TP Hit",
            entry_confidence=0.88,
        )
        
        # P&L should be (40000 - 39000) * 1.0 = 1000
        assert trade.realized_pnl_usd == 1000.0
        # P&L % should be 2.5%
        assert abs(trade.realized_pnl_pct - 2.5) < 0.01
    
    def test_losing_trade(self):
        """Test losing trade detection"""
        now = datetime.now()
        past = now - timedelta(hours=1, minutes=45)
        
        trade = ClosedTrade(
            pair="BTCUSDT",
            side="LONG",
            size_btc=1.0,
            entry_price=40000.0,
            exit_price=39000.0,
            open_time=past,
            close_time=now,
            close_reason="SL Hit",
            entry_confidence=0.75,
        )
        
        assert trade.is_win is False
        assert trade.realized_pnl_usd == -1000.0
    
    def test_breakeven_trade(self):
        """Test breakeven trade detection"""
        now = datetime.now()
        past = now - timedelta(hours=1)
        
        trade = ClosedTrade(
            pair="BTCUSDT",
            side="LONG",
            size_btc=1.0,
            entry_price=40000.0,
            exit_price=40000.005,  # Essentially breakeven
            open_time=past,
            close_time=now,
            close_reason="Manual",
            entry_confidence=0.70,
        )
        
        assert trade.is_breakeven is True
    
    def test_trade_duration(self):
        """Test trade duration calculation"""
        now = datetime.now()
        past = now - timedelta(hours=3, minutes=22)
        
        trade = ClosedTrade(
            pair="BTCUSDT",
            side="LONG",
            size_btc=0.018,
            entry_price=42150.0,
            exit_price=43217.5,
            open_time=past,
            close_time=now,
            close_reason="TP Hit",
            entry_confidence=0.91,
        )
        
        duration = trade.duration
        assert duration.total_seconds() > 0
        assert duration.total_seconds() < 15000  # Less than 5 hours


class TestSignal:
    """Tests for Signal model"""
    
    def test_signal_creation(self):
        """Test creating a signal"""
        signal = Signal(
            direction="BUY",
            confidence=0.87,
            horizon="4h",
            worker_votes={"W1": 0.82, "W2": 0.91, "W3": 0.85, "W4": 0.88},
            decision_driver="Trend",
        )
        
        assert signal.direction == "BUY"
        assert signal.confidence == 0.87
        assert len(signal.worker_votes) == 4
    
    def test_signal_confidence_bounds(self):
        """Test signal confidence bounds validation"""
        # Valid confidence
        signal = Signal(
            direction="BUY",
            confidence=0.5,
            horizon="1h",
            worker_votes={"W1": 0.5},
            decision_driver="Trend",
        )
        assert signal.confidence == 0.5
        
        # Invalid confidence (too high)
        with pytest.raises(ValueError):
            Signal(
                direction="BUY",
                confidence=1.5,
                horizon="1h",
                worker_votes={"W1": 0.5},
                decision_driver="Trend",
            )
        
        # Invalid confidence (negative)
        with pytest.raises(ValueError):
            Signal(
                direction="BUY",
                confidence=-0.1,
                horizon="1h",
                worker_votes={"W1": 0.5},
                decision_driver="Trend",
            )
    
    def test_worker_vote_bounds(self):
        """Test worker vote bounds validation"""
        # Invalid worker vote
        with pytest.raises(ValueError):
            Signal(
                direction="BUY",
                confidence=0.8,
                horizon="1h",
                worker_votes={"W1": 1.5},  # Invalid
                decision_driver="Trend",
            )
    
    def test_average_worker_vote(self):
        """Test average worker vote calculation"""
        signal = Signal(
            direction="BUY",
            confidence=0.87,
            horizon="4h",
            worker_votes={"W1": 0.80, "W2": 0.90, "W3": 0.85, "W4": 0.89},
            decision_driver="Trend",
        )
        
        # Average should be (0.80 + 0.90 + 0.85 + 0.89) / 4 = 0.86
        assert abs(signal.average_worker_vote - 0.86) < 0.01


class TestMarketContext:
    """Tests for MarketContext model"""
    
    def test_market_context_creation(self):
        """Test creating market context"""
        ctx = MarketContext(
            price=43850.25,
            volatility_atr=2.1,
            rsi=42,
            adx=28,
            trend_strength="Moderate",
            market_regime="Trending",
            volume_change=18.0,
        )
        
        assert ctx.price == 43850.25
        assert ctx.rsi == 42
        assert ctx.adx == 28
    
    def test_rsi_bounds_validation(self):
        """Test RSI bounds validation"""
        # Valid RSI
        ctx = MarketContext(
            price=43850.25,
            volatility_atr=2.1,
            rsi=50,
            adx=28,
            trend_strength="Moderate",
            market_regime="Trending",
            volume_change=18.0,
        )
        assert ctx.rsi == 50
        
        # Invalid RSI (too high)
        with pytest.raises(ValueError):
            MarketContext(
                price=43850.25,
                volatility_atr=2.1,
                rsi=101,
                adx=28,
                trend_strength="Moderate",
                market_regime="Trending",
                volume_change=18.0,
            )
    
    def test_adx_bounds_validation(self):
        """Test ADX bounds validation"""
        # Invalid ADX (negative)
        with pytest.raises(ValueError):
            MarketContext(
                price=43850.25,
                volatility_atr=2.1,
                rsi=50,
                adx=-5,
                trend_strength="Moderate",
                market_regime="Trending",
                volume_change=18.0,
            )
    
    def test_rsi_level_classification(self):
        """Test RSI level classification"""
        # Oversold
        ctx_oversold = MarketContext(
            price=43850.25,
            volatility_atr=2.1,
            rsi=25,
            adx=28,
            trend_strength="Moderate",
            market_regime="Trending",
            volume_change=18.0,
        )
        assert ctx_oversold.rsi_level == "Oversold"
        
        # Overbought
        ctx_overbought = MarketContext(
            price=43850.25,
            volatility_atr=2.1,
            rsi=75,
            adx=28,
            trend_strength="Moderate",
            market_regime="Trending",
            volume_change=18.0,
        )
        assert ctx_overbought.rsi_level == "Overbought"
        
        # Neutral
        ctx_neutral = MarketContext(
            price=43850.25,
            volatility_atr=2.1,
            rsi=50,
            adx=28,
            trend_strength="Moderate",
            market_regime="Trending",
            volume_change=18.0,
        )
        assert ctx_neutral.rsi_level == "Neutral"


class TestPortfolioState:
    """Tests for PortfolioState model"""
    
    def test_portfolio_state_creation(self):
        """Test creating portfolio state"""
        portfolio = PortfolioState(
            total_value_usd=1243.75,
            available_capital_usd=456.20,
        )
        
        assert portfolio.total_value_usd == 1243.75
        assert portfolio.available_capital_usd == 456.20
        assert portfolio.position_count == 0
    
    def test_portfolio_with_positions(self):
        """Test portfolio with open positions"""
        pos = Position(
            pair="BTCUSDT",
            side="LONG",
            size_btc=0.0245,
            entry_price=43217.50,
            current_price=43850.25,
            sl_price=41890.98,
            tp_price=44579.63,
            open_time=datetime.now(),
            entry_signal_strength=0.91,
            entry_market_regime="Trending",
            entry_volatility=1.8,
            entry_rsi=38,
        )
        
        portfolio = PortfolioState(
            total_value_usd=1243.75,
            available_capital_usd=456.20,
            open_positions=[pos],
        )
        
        assert portfolio.position_count == 1
        assert portfolio.total_unrealized_pnl_usd > 0
    
    def test_portfolio_with_closed_trades(self):
        """Test portfolio with closed trades"""
        now = datetime.now()
        past = now - timedelta(hours=3, minutes=22)
        
        trade = ClosedTrade(
            pair="BTCUSDT",
            side="LONG",
            size_btc=0.018,
            entry_price=42150.0,
            exit_price=43217.5,
            open_time=past,
            close_time=now,
            close_reason="TP Hit",
            entry_confidence=0.91,
        )
        
        portfolio = PortfolioState(
            total_value_usd=1243.75,
            available_capital_usd=456.20,
            closed_trades=[trade],
        )
        
        assert len(portfolio.closed_trades) == 1
        assert portfolio.total_realized_pnl_usd > 0
    
    def test_portfolio_win_rate(self):
        """Test portfolio win rate calculation"""
        now = datetime.now()
        
        # Create 3 winning trades and 2 losing trades
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
            ClosedTrade(
                pair="BTCUSDT",
                side="LONG",
                size_btc=1.0,
                entry_price=40000.0,
                exit_price=41000.0,
                open_time=now - timedelta(hours=1),
                close_time=now,
                close_reason="TP Hit",
                entry_confidence=0.90,
            ),
            ClosedTrade(
                pair="BTCUSDT",
                side="LONG",
                size_btc=1.0,
                entry_price=40000.0,
                exit_price=39500.0,
                open_time=now - timedelta(minutes=30),
                close_time=now,
                close_reason="SL Hit",
                entry_confidence=0.70,
            ),
            ClosedTrade(
                pair="BTCUSDT",
                side="LONG",
                size_btc=1.0,
                entry_price=40000.0,
                exit_price=41500.0,
                open_time=now - timedelta(minutes=15),
                close_time=now,
                close_reason="TP Hit",
                entry_confidence=0.88,
            ),
        ]
        
        portfolio = PortfolioState(
            total_value_usd=1243.75,
            available_capital_usd=456.20,
            closed_trades=trades,
        )
        
        # Win rate should be 3/5 = 60%
        assert abs(portfolio.win_rate - 60.0) < 0.1
    
    def test_portfolio_profit_factor(self):
        """Test portfolio profit factor calculation"""
        now = datetime.now()
        
        # Create trades with known P&L
        trades = [
            ClosedTrade(
                pair="BTCUSDT",
                side="LONG",
                size_btc=1.0,
                entry_price=40000.0,
                exit_price=41000.0,  # +1000
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
                exit_price=39000.0,  # -1000
                open_time=now - timedelta(hours=2),
                close_time=now,
                close_reason="SL Hit",
                entry_confidence=0.75,
            ),
            ClosedTrade(
                pair="BTCUSDT",
                side="LONG",
                size_btc=1.0,
                entry_price=40000.0,
                exit_price=41500.0,  # +1500
                open_time=now - timedelta(hours=1),
                close_time=now,
                close_reason="TP Hit",
                entry_confidence=0.90,
            ),
        ]
        
        portfolio = PortfolioState(
            total_value_usd=1243.75,
            available_capital_usd=456.20,
            closed_trades=trades,
        )
        
        # Gross profit = 1000 + 1500 = 2500
        # Gross loss = 1000
        # Profit factor = 2500 / 1000 = 2.5
        assert abs(portfolio.profit_factor - 2.5) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
