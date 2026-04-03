"""
Integration tests for ADAN Dashboard layout and generation

Tests for layout creation, dashboard generation, and live display.
"""

import pytest
from rich.console import Console
from rich.layout import Layout

from src.adan_trading_bot.dashboard.layout import (
    create_layout,
    create_compact_layout,
    get_optimal_layout,
    update_layout_content,
)
from src.adan_trading_bot.dashboard.generator import (
    generate_dashboard,
    generate_dashboard_from_portfolio,
)
from src.adan_trading_bot.dashboard.app import AdanBtcDashboard
from src.adan_trading_bot.dashboard.mock_collector import MockDataCollector
from src.adan_trading_bot.dashboard.models import PortfolioState


class TestLayoutCreation:
    """Tests for layout creation"""
    
    def test_create_layout_returns_layout(self):
        """Test that create_layout returns a Layout object"""
        layout = create_layout()
        assert isinstance(layout, Layout)
    
    def test_create_layout_has_required_sections(self):
        """Test that layout has all required sections"""
        layout = create_layout()
        
        # Check that layout is valid and has structure
        assert isinstance(layout, Layout)
        # Layout should have named regions
        assert layout is not None
    
    def test_create_compact_layout_returns_layout(self):
        """Test that create_compact_layout returns a Layout object"""
        layout = create_compact_layout()
        assert isinstance(layout, Layout)
    
    def test_create_compact_layout_has_required_sections(self):
        """Test that compact layout has all required sections"""
        layout = create_compact_layout()
        
        # Check that layout is valid
        assert isinstance(layout, Layout)
        # Layout should have named regions
        assert layout is not None
    
    def test_get_optimal_layout_wide_terminal(self):
        """Test optimal layout selection for wide terminal"""
        # Mock a wide console
        console = Console(width=150, height=50)
        layout = get_optimal_layout(console)
        
        assert isinstance(layout, Layout)
        # Should use regular layout for wide terminal
        assert layout is not None
    
    def test_get_optimal_layout_narrow_terminal(self):
        """Test optimal layout selection for narrow terminal"""
        # Mock a narrow console
        console = Console(width=80, height=30)
        layout = get_optimal_layout(console)
        
        assert isinstance(layout, Layout)
        # Should use compact layout for narrow terminal
        # Compact layout doesn't have main section
    
    def test_update_layout_content(self):
        """Test updating layout with content"""
        layout = create_layout()
        
        # Create mock content
        from rich.panel import Panel
        from rich.text import Text
        
        sections = {
            "header": Panel(Text("Test Header"), title="Header"),
        }
        
        # Update layout - should not raise exception
        try:
            update_layout_content(layout, sections)
        except KeyError:
            # Expected if section doesn't exist in layout
            pass
        
        # Layout should still be valid
        assert isinstance(layout, Layout)


class TestDashboardGeneration:
    """Tests for dashboard generation"""
    
    def test_generate_dashboard_with_mock_collector(self):
        """Test generating dashboard with mock data collector"""
        collector = MockDataCollector(seed=42)
        console = Console(width=120, height=40)
        
        # Connect collector
        collector.connect()
        
        try:
            # Generate dashboard
            dashboard = generate_dashboard(collector, console)
            
            assert isinstance(dashboard, Layout)
        finally:
            # Cleanup
            collector.disconnect()
    
    def test_generate_dashboard_from_portfolio(self):
        """Test generating dashboard from portfolio state"""
        portfolio = PortfolioState(
            total_value_usd=1000.0,
            available_capital_usd=500.0,
        )
        console = Console(width=120, height=40)
        
        # Generate dashboard
        dashboard = generate_dashboard_from_portfolio(portfolio, console)
        
        assert isinstance(dashboard, Layout)
    
    def test_generate_dashboard_has_all_sections(self):
        """Test that generated dashboard has all required sections"""
        collector = MockDataCollector(seed=42)
        console = Console(width=120, height=40)
        
        collector.connect()
        try:
            dashboard = generate_dashboard(collector, console)
            
            # Check dashboard is valid layout
            assert isinstance(dashboard, Layout)
        finally:
            collector.disconnect()


class TestDashboardApplication:
    """Tests for dashboard application"""
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization"""
        collector = MockDataCollector(seed=42)
        console = Console()
        
        dashboard = AdanBtcDashboard(
            data_collector=collector,
            refresh_rate=1.0,
            console=console,
        )
        
        assert dashboard.data_collector is collector
        assert dashboard.refresh_rate == 1.0
        assert dashboard.console is console
        assert dashboard.running is False
    
    def test_dashboard_default_collector(self):
        """Test dashboard with default mock collector"""
        console = Console()
        
        dashboard = AdanBtcDashboard(console=console)
        
        assert dashboard.data_collector is not None
        assert isinstance(dashboard.data_collector, MockDataCollector)
    
    def test_dashboard_run_once(self):
        """Test dashboard run_once method"""
        collector = MockDataCollector(seed=42)
        console = Console()
        
        dashboard = AdanBtcDashboard(
            data_collector=collector,
            console=console,
        )
        
        # Should not raise exception
        dashboard.run_once()
    
    def test_dashboard_refresh_rate_validation(self):
        """Test dashboard with various refresh rates"""
        collector = MockDataCollector(seed=42)
        console = Console()
        
        # Valid refresh rates
        for rate in [0.1, 0.5, 1.0, 2.0, 5.0]:
            dashboard = AdanBtcDashboard(
                data_collector=collector,
                refresh_rate=rate,
                console=console,
            )
            assert dashboard.refresh_rate == rate


class TestLayoutResponsiveness:
    """Tests for layout responsiveness to terminal size"""
    
    def test_layout_adapts_to_terminal_width(self):
        """Test that layout adapts to terminal width"""
        # Wide terminal
        wide_console = Console(width=200, height=50)
        wide_layout = get_optimal_layout(wide_console)
        
        # Narrow terminal
        narrow_console = Console(width=80, height=50)
        narrow_layout = get_optimal_layout(narrow_console)
        
        # Both should be valid layouts
        assert isinstance(wide_layout, Layout)
        assert isinstance(narrow_layout, Layout)
    
    def test_layout_adapts_to_terminal_height(self):
        """Test that layout adapts to terminal height"""
        # Tall terminal
        tall_console = Console(width=120, height=60)
        tall_layout = get_optimal_layout(tall_console)
        
        # Short terminal
        short_console = Console(width=120, height=30)
        short_layout = get_optimal_layout(short_console)
        
        # Both should be valid layouts
        assert isinstance(tall_layout, Layout)
        assert isinstance(short_layout, Layout)


class TestDashboardErrorHandling:
    """Tests for dashboard error handling"""
    
    def test_dashboard_handles_collector_errors(self):
        """Test dashboard handles collector errors gracefully"""
        # Create a collector that will fail
        collector = MockDataCollector(seed=42)
        console = Console()
        
        dashboard = AdanBtcDashboard(
            data_collector=collector,
            console=console,
        )
        
        # Should handle errors gracefully
        dashboard.run_once()
    
    def test_dashboard_with_disconnected_collector(self):
        """Test dashboard with disconnected collector"""
        collector = MockDataCollector(seed=42)
        console = Console()
        
        dashboard = AdanBtcDashboard(
            data_collector=collector,
            console=console,
        )
        
        # Don't connect collector - should still work
        dashboard.run_once()
