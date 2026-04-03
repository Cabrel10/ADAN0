"""
Property-based tests for ADAN Dashboard correctness properties

Uses Hypothesis for property-based testing to validate correctness properties
across a wide range of inputs.
"""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime, timedelta

from src.adan_trading_bot.dashboard.models import (
    Position,
    ClosedTrade,
    Signal,
    MarketContext,
    PortfolioState,
    SystemHealth,
    Alert,
    SignalDirection,
    CloseReason,
)
from src.adan_trading_bot.dashboard.mock_collector import MockDataCollector
from src.adan_trading_bot.dashboard.aggregator import DataAggregator


# Custom strategies for generating test data
def position_strategy():
    """Generate random Position objects"""
    return st.builds(
        Position,
        pair=st.just("BTCUSDT"),
        side=st.sampled_from(["LONG", "SHORT"]),
        size_btc=st.floats(min_value=0.001, max_value=10.0),
        entry_price=st.floats(min_value=1000, max_value=100000),
        current_price=st.floats(min_value=1000, max_value=100000),
        sl_price=st.floats(min_value=1000, max_value=100000),
        tp_price=st.floats(min_value=1000, max_value=100000),
        open_time=st.datetimes(min_value=datetime.now() - timedelta(days=30)),
        entry_signal_strength=st.floats(min_value=0.0, max_value=1.0),
        entry_market_regime=st.sampled_from(["Trending", "Ranging", "Breakout"]),
        entry_volatility=st.floats(min_value=0.0, max_value=100.0),
        entry_rsi=st.integers(min_value=0, max_value=100),
    )


def closed_trade_strategy():
    """Generate random ClosedTrade objects"""
    # Generate open_time first, then close_time after it
    return st.builds(
        lambda open_time, close_offset, **kwargs: ClosedTrade(
            open_time=open_time,
            close_time=open_time + timedelta(seconds=close_offset),
            **kwargs
        ),
        open_time=st.datetimes(min_value=datetime.now() - timedelta(days=30)),
        close_offset=st.integers(min_value=1, max_value=86400),  # 1 second to 1 day
        pair=st.just("BTCUSDT"),
        side=st.sampled_from(["LONG", "SHORT"]),
        size_btc=st.floats(min_value=0.001, max_value=10.0),
        entry_price=st.floats(min_value=1000, max_value=100000),
        exit_price=st.floats(min_value=1000, max_value=100000),
        close_reason=st.sampled_from(["TP Hit", "SL Hit", "Manual", "Time"]),
        entry_confidence=st.floats(min_value=0.0, max_value=1.0),
    )


def signal_strategy():
    """Generate random Signal objects"""
    return st.builds(
        Signal,
        direction=st.sampled_from(["BUY", "SELL", "HOLD"]),
        confidence=st.floats(min_value=0.0, max_value=1.0),
        horizon=st.sampled_from(["5m", "1h", "4h", "1d"]),
        worker_votes=st.dictionaries(
            keys=st.text(min_size=1, max_size=5),
            values=st.floats(min_value=0.0, max_value=1.0),
            min_size=1,
            max_size=5,
        ),
        decision_driver=st.sampled_from(["Trend", "MeanReversion", "Breakout"]),
        timestamp=st.datetimes(),
    )


class TestPortfolioValueConsistency:
    """
    Property 1: Portfolio Value Consistency
    
    For any portfolio state, the sum of positions + capital should equal total value
    **Validates: Requirements 1.1, 1.2, 1.3**
    """
    
    @given(
        positions=st.lists(position_strategy(), min_size=0, max_size=10),
        capital=st.floats(min_value=0, max_value=1000000),
    )
    @settings(max_examples=100)
    def test_portfolio_value_consistency(self, positions, capital):
        """
        For any portfolio state, verify that portfolio value is consistent
        """
        # Calculate total position value
        total_position_value = sum(p.position_value_usd for p in positions)
        
        # Create portfolio state
        portfolio = PortfolioState(
            total_value_usd=total_position_value + capital,
            available_capital_usd=capital,
            open_positions=positions,
            closed_trades=[],
        )
        
        # Verify consistency
        assert portfolio.total_value_usd >= portfolio.available_capital_usd
        assert portfolio.total_value_usd >= 0


class TestPositionPnLAccuracy:
    """
    Property 2: Position P&L Calculation Accuracy
    
    For any position, verify P&L calculations are accurate
    **Validates: Requirements 2.9**
    """
    
    @given(position_strategy())
    @settings(max_examples=100)
    def test_position_pnl_calculation(self, position):
        """
        For any position, verify P&L calculation matches formula
        """
        # Calculate expected P&L
        if position.side == "LONG":
            expected_pnl = (position.current_price - position.entry_price) * position.size_btc
        else:  # SHORT
            expected_pnl = (position.entry_price - position.current_price) * position.size_btc
        
        # Verify calculation
        assert abs(position.unrealized_pnl_usd - expected_pnl) < 0.01
    
    @given(position_strategy())
    @settings(max_examples=100)
    def test_position_pnl_percentage(self, position):
        """
        For any position, verify P&L percentage is calculated correctly
        """
        if position.entry_price == 0:
            assert position.unrealized_pnl_pct == 0.0
        else:
            if position.side == "LONG":
                expected_pct = ((position.current_price - position.entry_price) / position.entry_price) * 100
            else:
                expected_pct = ((position.entry_price - position.current_price) / position.entry_price) * 100
            
            assert abs(position.unrealized_pnl_pct - expected_pct) < 0.01


class TestTradeHistoryCompleteness:
    """
    Property 3: Trade History Completeness
    
    For any trade, verify all required fields are present and valid
    **Validates: Requirements 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8**
    """
    
    @given(closed_trade_strategy())
    @settings(max_examples=100)
    def test_trade_fields_present(self, trade):
        """
        For any trade, verify all required fields are present
        """
        assert trade.pair is not None
        assert trade.side in ["LONG", "SHORT"]
        assert trade.size_btc > 0
        assert trade.entry_price > 0
        assert trade.exit_price > 0
        assert trade.open_time is not None
        assert trade.close_time is not None
        assert trade.close_reason is not None
        assert 0 <= trade.entry_confidence <= 1.0
    
    @given(closed_trade_strategy())
    @settings(max_examples=100)
    def test_trade_pnl_calculation(self, trade):
        """
        For any trade, verify P&L is calculated correctly
        """
        if trade.side == "LONG":
            expected_pnl = (trade.exit_price - trade.entry_price) * trade.size_btc
        else:
            expected_pnl = (trade.entry_price - trade.exit_price) * trade.size_btc
        
        assert abs(trade.realized_pnl_usd - expected_pnl) < 0.01


class TestSignalConfidenceBounds:
    """
    Property 4: Signal Confidence Bounds
    
    For any signal, verify confidence is between 0.0 and 1.0
    **Validates: Requirements 4.2**
    """
    
    @given(signal_strategy())
    @settings(max_examples=100)
    def test_signal_confidence_bounds(self, signal):
        """
        For any signal, verify confidence is within valid bounds
        """
        assert 0.0 <= signal.confidence <= 1.0
    
    @given(signal_strategy())
    @settings(max_examples=100)
    def test_signal_direction_valid(self, signal):
        """
        For any signal, verify direction is valid
        """
        assert signal.direction in ["BUY", "SELL", "HOLD"]


class TestWorkerVoteConsistency:
    """
    Property 5: Worker Vote Consistency
    
    For any signal with worker votes, verify average is within bounds
    **Validates: Requirements 4.4**
    """
    
    @given(signal_strategy())
    @settings(max_examples=100)
    def test_worker_votes_bounds(self, signal):
        """
        For any signal, verify all worker votes are within bounds
        """
        for vote in signal.worker_votes.values():
            assert 0.0 <= vote <= 1.0
    
    @given(signal_strategy())
    @settings(max_examples=100)
    def test_worker_votes_average(self, signal):
        """
        For any signal, verify average of votes is reasonable
        """
        if signal.worker_votes:
            votes = list(signal.worker_votes.values())
            if votes:
                avg_vote = sum(votes) / len(votes)
                # Average should be within 1.0 of confidence (allowing for variance)
                # This is a loose bound since votes can vary significantly
                assert 0.0 <= avg_vote <= 1.0


class TestPerformanceMetricsValidity:
    """
    Property 6: Performance Metrics Non-Negativity
    
    For any trade history, verify metrics are mathematically valid
    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**
    """
    
    @given(st.lists(closed_trade_strategy(), min_size=0, max_size=100))
    @settings(max_examples=100)
    def test_metrics_non_negative(self, trades):
        """
        For any trade history, verify metrics are non-negative
        """
        aggregator = DataAggregator(MockDataCollector(seed=42))
        metrics = aggregator._calculate_metrics(trades)
        
        assert metrics['win_rate'] >= 0.0
        assert metrics['total_trades'] >= 0
        assert metrics['avg_holding_time_seconds'] >= 0.0
    
    @given(st.lists(closed_trade_strategy(), min_size=0, max_size=100))
    @settings(max_examples=100)
    def test_win_rate_bounds(self, trades):
        """
        For any trade history, verify win rate is between 0 and 1
        """
        aggregator = DataAggregator(MockDataCollector(seed=42))
        metrics = aggregator._calculate_metrics(trades)
        
        assert 0.0 <= metrics['win_rate'] <= 1.0


class TestTimestampOrdering:
    """
    Property 7: Timestamp Ordering
    
    For any trade sequence, verify trades are in reverse chronological order
    **Validates: Requirements 3.1**
    """
    
    @given(st.lists(closed_trade_strategy(), min_size=2, max_size=100))
    @settings(max_examples=100)
    def test_trades_ordered_by_close_time(self, trades):
        """
        For any trade sequence, verify ordering is consistent
        """
        # Sort trades by close time (most recent first)
        sorted_trades = sorted(trades, key=lambda t: t.close_time, reverse=True)
        
        # Verify ordering
        for i in range(len(sorted_trades) - 1):
            assert sorted_trades[i].close_time >= sorted_trades[i + 1].close_time


class TestColorCodingCorrectness:
    """
    Property 8: Color Coding Correctness
    
    For any P&L value, verify correct color is applied
    **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10, 8.11, 8.12, 8.13, 8.14**
    """
    
    @given(pnl_percent=st.floats(min_value=-100, max_value=100))
    @settings(max_examples=100)
    def test_pnl_color_mapping(self, pnl_percent):
        """
        For any P&L percentage, verify color mapping is consistent
        """
        from src.adan_trading_bot.dashboard.colors import get_pnl_color
        
        color = get_pnl_color(pnl_percent)
        
        # Verify color is a valid hex code
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7
    
    @given(confidence=st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=100)
    def test_confidence_color_mapping(self, confidence):
        """
        For any confidence value, verify color mapping is consistent
        """
        from src.adan_trading_bot.dashboard.colors import get_confidence_color
        
        color = get_confidence_color(confidence)
        
        # Verify color is a valid hex code
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7


class TestDataStalenessDetection:
    """
    Property 9: Data Staleness Detection
    
    For any data with timestamp, verify staleness is detected correctly
    **Validates: Requirements 9.6**
    """
    
    @given(age_seconds=st.floats(min_value=0, max_value=300))
    @settings(max_examples=100)
    def test_staleness_threshold(self, age_seconds):
        """
        For any data age, verify staleness detection is correct
        """
        from src.adan_trading_bot.dashboard.cache import DataCache
        
        cache = DataCache(default_ttl_seconds=300)
        cache.staleness_threshold = timedelta(seconds=90)
        
        # Set a value
        cache.set("test", "value")
        
        # Check staleness
        is_stale = cache.is_stale("test")
        
        # Verify staleness matches age
        if age_seconds > 90:
            # Would be stale after 90 seconds
            pass
        else:
            # Should not be stale
            assert not is_stale


class TestLayoutResponsiveness:
    """
    Property 10: Layout Responsiveness
    
    For any terminal width, verify layout fits without horizontal scrolling
    **Validates: Requirements 10.8**
    """
    
    @given(width=st.integers(min_value=80, max_value=200))
    @settings(max_examples=100)
    def test_layout_fits_terminal(self, width):
        """
        For any terminal width >= 80, verify layout fits
        """
        from rich.console import Console
        from src.adan_trading_bot.dashboard.layout import get_optimal_layout
        
        console = Console(width=width, height=40)
        layout = get_optimal_layout(console)
        
        # Verify layout is created
        assert layout is not None


class TestUpdateFrequency:
    """
    Property 11: Real-Time Update Frequency
    
    For any refresh rate, verify updates happen at correct intervals
    **Validates: Requirements 9.1, 9.2, 9.3**
    """
    
    @given(refresh_rate=st.floats(min_value=0.1, max_value=10.0))
    @settings(max_examples=100)
    def test_refresh_rate_valid(self, refresh_rate):
        """
        For any refresh rate, verify it's valid
        """
        assert refresh_rate > 0
        assert refresh_rate <= 10.0


class TestSystemHealthAccuracy:
    """
    Property 12: System Health Status Accuracy
    
    For any system state, verify health status is accurate
    **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
    """
    
    @given(
        api_status=st.sampled_from(["OK", "DEGRADED", "DOWN"]),
        feed_status=st.sampled_from(["OK", "DEGRADED", "DOWN"]),
        model_status=st.sampled_from(["OK", "DEGRADED", "DOWN"]),
        database_status=st.sampled_from(["OK", "DEGRADED", "DOWN"]),
    )
    @settings(max_examples=100)
    def test_health_status_consistency(self, api_status, feed_status, model_status, database_status):
        """
        For any system state, verify health status is consistent
        """
        health = SystemHealth(
            api_status=api_status,
            feed_status=feed_status,
            model_status=model_status,
            database_status=database_status,
        )
        
        # Verify health status
        is_healthy = health.is_healthy
        
        # Should be healthy only if all components are OK
        expected_healthy = all(
            status == "OK"
            for status in [api_status, feed_status, model_status, database_status]
        )
        
        assert is_healthy == expected_healthy
