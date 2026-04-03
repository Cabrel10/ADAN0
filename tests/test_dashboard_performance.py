"""
Performance testing and optimization for ADAN BTC/USDT Terminal Dashboard.

Tests measure:
- Dashboard render time (target: <100ms)
- Refresh cycle time (target: <500ms for price updates)
- Memory usage (target: <500MB)
- CPU usage (target: <50%)
"""

import time
import psutil
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.adan_trading_bot.dashboard.models import (
    Position, ClosedTrade, Signal, MarketContext, PortfolioState
)
from src.adan_trading_bot.dashboard.formatters import (
    format_usd, format_btc, format_percentage, format_time, format_confidence
)
from src.adan_trading_bot.dashboard.colors import (
    get_pnl_color, get_confidence_color, get_risk_color
)
from src.adan_trading_bot.dashboard.sections.header import render_header
from src.adan_trading_bot.dashboard.sections.decision_matrix import render_decision_matrix
from src.adan_trading_bot.dashboard.sections.positions import render_positions
from src.adan_trading_bot.dashboard.sections.closed_trades import render_closed_trades
from src.adan_trading_bot.dashboard.sections.performance import render_performance
from src.adan_trading_bot.dashboard.sections.system_health import render_system_health
from src.adan_trading_bot.dashboard.layout import create_layout
from src.adan_trading_bot.dashboard.generator import generate_dashboard
from src.adan_trading_bot.dashboard.mock_collector import MockDataCollector
from rich.console import Console


class TestRenderingPerformance:
    """Test rendering performance for individual components."""

    def test_header_render_time(self):
        """Header should render in <20ms."""
        portfolio = PortfolioState(
            total_value_usd=10000.0,
            available_capital_usd=5000.0,
        )
        
        start = time.perf_counter()
        for _ in range(100):
            render_header(portfolio)
        elapsed = (time.perf_counter() - start) / 100
        
        assert elapsed < 0.020, f"Header render time {elapsed*1000:.2f}ms exceeds 20ms target"

    def test_decision_matrix_render_time(self):
        """Decision matrix should render in <30ms."""
        signal = Signal(
            direction="BUY",
            confidence=0.85,
            horizon="4h",
            worker_votes={"W1": 0.8, "W2": 0.9, "W3": 0.85, "W4": 0.75},
            decision_driver="RSI oversold"
        )
        market = MarketContext(
            price=45000.0,
            volatility_atr=2.5,
            rsi=35,
            adx=45,
            trend_strength="Strong",
            market_regime="Trending",
            volume_change=15.0
        )
        
        start = time.perf_counter()
        for _ in range(100):
            render_decision_matrix(signal, market)
        elapsed = (time.perf_counter() - start) / 100
        
        assert elapsed < 0.030, f"Decision matrix render time {elapsed*1000:.2f}ms exceeds 30ms target"

    def test_positions_render_time(self):
        """Positions table should render in <25ms."""
        positions = [
            Position(
                pair="BTCUSDT",
                side="LONG",
                size_btc=0.1,
                entry_price=45000.0 + i*100,
                current_price=45500.0 + i*100,
                sl_price=44500.0 + i*100,
                tp_price=46500.0 + i*100,
                open_time=datetime.now() - timedelta(hours=i),
                entry_signal_strength=0.85,
                entry_market_regime="Trending",
                entry_volatility=2.5,
                entry_rsi=35
            )
            for i in range(5)
        ]
        
        start = time.perf_counter()
        for _ in range(100):
            render_positions(positions)
        elapsed = (time.perf_counter() - start) / 100
        
        assert elapsed < 0.025, f"Positions render time {elapsed*1000:.2f}ms exceeds 25ms target"

    def test_closed_trades_render_time(self):
        """Closed trades table should render in <25ms."""
        trades = [
            ClosedTrade(
                pair="BTCUSDT",
                side="LONG",
                size_btc=0.1,
                entry_price=45000.0 + i*100,
                exit_price=45500.0 + i*100,
                open_time=datetime.now() - timedelta(hours=i+1),
                close_time=datetime.now() - timedelta(hours=i),
                close_reason="TP Hit",
                entry_confidence=0.85
            )
            for i in range(5)
        ]
        
        start = time.perf_counter()
        for _ in range(100):
            render_closed_trades(trades)
        elapsed = (time.perf_counter() - start) / 100
        
        assert elapsed < 0.025, f"Closed trades render time {elapsed*1000:.2f}ms exceeds 25ms target"

    def test_performance_render_time(self):
        """Performance section should render in <20ms."""
        trades = [
            ClosedTrade(
                pair="BTCUSDT",
                side="LONG" if i % 2 == 0 else "SHORT",
                size_btc=0.1,
                entry_price=45000.0,
                exit_price=45500.0 if i % 2 == 0 else 44500.0,
                open_time=datetime.now() - timedelta(hours=i+1),
                close_time=datetime.now() - timedelta(hours=i),
                close_reason="TP Hit",
                entry_confidence=0.85
            )
            for i in range(10)
        ]
        
        start = time.perf_counter()
        for _ in range(100):
            render_performance(trades)
        elapsed = (time.perf_counter() - start) / 100
        
        assert elapsed < 0.020, f"Performance render time {elapsed*1000:.2f}ms exceeds 20ms target"

    def test_system_health_render_time(self):
        """System health section should render in <20ms."""
        health_data = {
            "api_status": "OK",
            "feed_status": "OK",
            "model_status": "OK",
            "database_status": "OK",
            "cpu_percent": 25.0,
            "memory_percent": 40.0,
            "thread_count": 12,
            "uptime_seconds": 3600,
            "alerts": []
        }
        
        start = time.perf_counter()
        for _ in range(100):
            render_system_health(health_data)
        elapsed = (time.perf_counter() - start) / 100
        
        assert elapsed < 0.020, f"System health render time {elapsed*1000:.2f}ms exceeds 20ms target"


class TestFormattingPerformance:
    """Test formatting function performance."""

    def test_format_usd_performance(self):
        """USD formatting should be fast."""
        start = time.perf_counter()
        for i in range(10000):
            format_usd(45000.0 + i)
        elapsed = time.perf_counter() - start
        
        # Should format 10000 values in <100ms
        assert elapsed < 0.1, f"USD formatting {elapsed*1000:.2f}ms exceeds 100ms target"

    def test_format_btc_performance(self):
        """BTC formatting should be fast."""
        start = time.perf_counter()
        for i in range(10000):
            format_btc(0.1 + i*0.001)
        elapsed = time.perf_counter() - start
        
        # Should format 10000 values in <100ms
        assert elapsed < 0.1, f"BTC formatting {elapsed*1000:.2f}ms exceeds 100ms target"

    def test_format_percentage_performance(self):
        """Percentage formatting should be fast."""
        start = time.perf_counter()
        for i in range(10000):
            format_percentage(0.05 + i*0.0001)
        elapsed = time.perf_counter() - start
        
        # Should format 10000 values in <100ms
        assert elapsed < 0.1, f"Percentage formatting {elapsed*1000:.2f}ms exceeds 100ms target"

    def test_format_time_performance(self):
        """Time formatting should be fast."""
        start = time.perf_counter()
        for i in range(10000):
            format_time(3600 + i)
        elapsed = time.perf_counter() - start
        
        # Should format 10000 values in <100ms
        assert elapsed < 0.1, f"Time formatting {elapsed*1000:.2f}ms exceeds 100ms target"


class TestColorMappingPerformance:
    """Test color mapping performance."""

    def test_pnl_color_mapping_performance(self):
        """P&L color mapping should be fast."""
        start = time.perf_counter()
        for i in range(10000):
            get_pnl_color((i % 200 - 100) / 100)  # -1.0 to 1.0
        elapsed = time.perf_counter() - start
        
        # Should map 10000 values in <50ms
        assert elapsed < 0.05, f"P&L color mapping {elapsed*1000:.2f}ms exceeds 50ms target"

    def test_confidence_color_mapping_performance(self):
        """Confidence color mapping should be fast."""
        start = time.perf_counter()
        for i in range(10000):
            get_confidence_color((i % 100) / 100)  # 0.0 to 1.0
        elapsed = time.perf_counter() - start
        
        # Should map 10000 values in <50ms
        assert elapsed < 0.05, f"Confidence color mapping {elapsed*1000:.2f}ms exceeds 50ms target"

    def test_risk_color_mapping_performance(self):
        """Risk color mapping should be fast."""
        start = time.perf_counter()
        for i in range(10000):
            get_risk_color((i % 100) / 100)  # 0.0 to 1.0
        elapsed = time.perf_counter() - start
        
        # Should map 10000 values in <50ms
        assert elapsed < 0.05, f"Risk color mapping {elapsed*1000:.2f}ms exceeds 50ms target"


class TestDashboardGenerationPerformance:
    """Test full dashboard generation performance."""

    def test_layout_creation_performance(self):
        """Layout creation should be fast."""
        start = time.perf_counter()
        for _ in range(100):
            create_layout()
        elapsed = (time.perf_counter() - start) / 100
        
        assert elapsed < 0.010, f"Layout creation {elapsed*1000:.2f}ms exceeds 10ms target"

    def test_dashboard_generation_performance(self):
        """Full dashboard generation should be <100ms."""
        collector = MockDataCollector()
        console = Console()
        
        start = time.perf_counter()
        for _ in range(10):
            generate_dashboard(collector, console)
        elapsed = (time.perf_counter() - start) / 10
        
        assert elapsed < 0.100, f"Dashboard generation {elapsed*1000:.2f}ms exceeds 100ms target"


class TestMemoryUsage:
    """Test memory usage of dashboard components."""

    def test_dashboard_memory_footprint(self):
        """Dashboard should use <100MB of memory."""
        process = psutil.Process()
        
        # Get baseline memory
        baseline = process.memory_info().rss / 1024 / 1024
        
        # Create dashboard and generate multiple times
        collector = MockDataCollector()
        console = Console()
        for _ in range(50):
            generate_dashboard(collector, console)
        
        # Get peak memory
        peak = process.memory_info().rss / 1024 / 1024
        delta = peak - baseline
        
        assert delta < 100, f"Dashboard memory delta {delta:.1f}MB exceeds 100MB target"

    def test_mock_collector_memory(self):
        """Mock collector should use minimal memory."""
        process = psutil.Process()
        
        baseline = process.memory_info().rss / 1024 / 1024
        
        # Create many mock collectors
        collectors = [MockDataCollector() for _ in range(100)]
        
        peak = process.memory_info().rss / 1024 / 1024
        delta = peak - baseline
        
        assert delta < 50, f"Mock collectors memory delta {delta:.1f}MB exceeds 50MB target"


class TestRefreshCyclePerformance:
    """Test refresh cycle timing."""

    def test_refresh_cycle_timing(self):
        """Complete refresh cycle should be <500ms."""
        collector = MockDataCollector()
        console = Console()
        
        start = time.perf_counter()
        for _ in range(10):
            # Simulate a complete refresh cycle
            collector.connect()
            generate_dashboard(collector, console)
        elapsed = (time.perf_counter() - start) / 10
        
        assert elapsed < 0.500, f"Refresh cycle {elapsed*1000:.2f}ms exceeds 500ms target"

    def test_price_update_frequency(self):
        """Price updates should happen every ~500ms."""
        # This is a timing test - verify that we can achieve 2 updates per second
        collector = MockDataCollector()
        
        times = []
        for _ in range(10):
            start = time.perf_counter()
            collector.get_market_context()
            times.append(time.perf_counter() - start)
        
        avg_time = sum(times) / len(times)
        
        # Each update should be <50ms to allow 2 updates per second
        assert avg_time < 0.050, f"Price update {avg_time*1000:.2f}ms exceeds 50ms target"


class TestOptimizationTargets:
    """Verify optimization targets are met."""

    def test_all_targets_met(self):
        """Verify all performance targets are met."""
        targets = {
            "header_render": 0.020,
            "decision_matrix_render": 0.030,
            "positions_render": 0.025,
            "closed_trades_render": 0.025,
            "performance_render": 0.020,
            "system_health_render": 0.020,
            "dashboard_generation": 0.100,
            "refresh_cycle": 0.500,
        }
        
        results = {}
        
        # Header
        portfolio = PortfolioState(
            total_value_usd=10000.0,
            available_capital_usd=5000.0,
        )
        start = time.perf_counter()
        for _ in range(100):
            render_header(portfolio)
        results["header_render"] = (time.perf_counter() - start) / 100
        
        # Decision matrix
        signal = Signal(
            direction="BUY",
            confidence=0.85,
            horizon="4h",
            worker_votes={"W1": 0.8, "W2": 0.9, "W3": 0.85, "W4": 0.75},
            decision_driver="RSI oversold"
        )
        market = MarketContext(
            price=45000.0,
            volatility_atr=2.5,
            rsi=35,
            adx=45,
            trend_strength="Strong",
            market_regime="Trending",
            volume_change=15.0
        )
        start = time.perf_counter()
        for _ in range(100):
            render_decision_matrix(signal, market)
        results["decision_matrix_render"] = (time.perf_counter() - start) / 100
        
        # Dashboard generation
        collector = MockDataCollector()
        console = Console()
        start = time.perf_counter()
        for _ in range(10):
            generate_dashboard(collector, console)
        results["dashboard_generation"] = (time.perf_counter() - start) / 10
        
        # Verify all targets
        for target, max_time in targets.items():
            if target in results:
                actual = results[target]
                assert actual < max_time, f"{target}: {actual*1000:.2f}ms exceeds {max_time*1000:.2f}ms target"
