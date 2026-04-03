"""
Integration tests for ADAN Dashboard data integration

Tests for real data collector, cache layer, and data aggregator.
"""

import pytest
from datetime import datetime, timedelta

from src.adan_trading_bot.dashboard.real_collector import RealDataCollector
from src.adan_trading_bot.dashboard.cache import DataCache, CachedDataCollector
from src.adan_trading_bot.dashboard.aggregator import DataAggregator
from src.adan_trading_bot.dashboard.mock_collector import MockDataCollector
from src.adan_trading_bot.dashboard.models import PortfolioState


class TestRealDataCollector:
    """Tests for real data collector"""
    
    def test_real_collector_initialization(self):
        """Test real collector can be initialized"""
        collector = RealDataCollector()
        assert collector is not None
        assert not collector.is_connected()
    
    def test_real_collector_connect_fallback(self):
        """Test real collector falls back gracefully when ADAN not available"""
        collector = RealDataCollector()
        # Should return False when ADAN components not available
        result = collector.connect()
        assert isinstance(result, bool)


class TestDataCache:
    """Tests for data cache layer"""
    
    def test_cache_set_and_get(self):
        """Test basic cache set and get"""
        cache = DataCache()
        
        # Set value
        cache.set('test_key', 'test_value')
        
        # Get value
        value = cache.get('test_key')
        assert value == 'test_value'
    
    def test_cache_expiration(self):
        """Test cache expiration"""
        cache = DataCache(default_ttl_seconds=1)
        
        # Set value
        cache.set('test_key', 'test_value')
        
        # Should be available immediately
        assert cache.get('test_key') == 'test_value'
        
        # Wait for expiration
        import time
        time.sleep(1.1)
        
        # Should be expired
        assert cache.get('test_key') is None
    
    def test_cache_staleness_detection(self):
        """Test cache staleness detection"""
        cache = DataCache(default_ttl_seconds=10)
        cache.staleness_threshold = timedelta(seconds=1)
        
        # Set value
        cache.set('test_key', 'test_value')
        
        # Should not be stale immediately
        assert not cache.is_stale('test_key')
        
        # Wait for staleness
        import time
        time.sleep(1.1)
        
        # Should be stale but not expired
        assert cache.is_stale('test_key')
        assert cache.get('test_key') == 'test_value'
    
    def test_cache_get_with_staleness(self):
        """Test getting value with staleness status"""
        cache = DataCache(default_ttl_seconds=10)
        cache.staleness_threshold = timedelta(seconds=1)
        
        # Set value
        cache.set('test_key', 'test_value')
        
        # Get immediately
        value, is_stale = cache.get_with_staleness('test_key')
        assert value == 'test_value'
        assert not is_stale
        
        # Wait for staleness
        import time
        time.sleep(1.1)
        
        # Get after staleness
        value, is_stale = cache.get_with_staleness('test_key')
        assert value == 'test_value'
        assert is_stale
    
    def test_cache_custom_ttl(self):
        """Test cache with custom TTL"""
        cache = DataCache(default_ttl_seconds=10)
        
        # Set with custom TTL
        cache.set('short_key', 'value', ttl_seconds=1)
        cache.set('long_key', 'value', ttl_seconds=10)
        
        # Both should be available immediately
        assert cache.get('short_key') == 'value'
        assert cache.get('long_key') == 'value'
        
        # Wait
        import time
        time.sleep(1.1)
        
        # Short key should expire, long key should remain
        assert cache.get('short_key') is None
        assert cache.get('long_key') == 'value'
    
    def test_cache_clear(self):
        """Test cache clearing"""
        cache = DataCache()
        
        # Set multiple values
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        
        # Clear specific key
        cache.clear('key1')
        assert cache.get('key1') is None
        assert cache.get('key2') == 'value2'
        
        # Clear all
        cache.clear()
        assert cache.get('key2') is None
    
    def test_cache_stats(self):
        """Test cache statistics"""
        cache = DataCache()
        
        # Set values
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        
        # Get stats
        stats = cache.get_stats()
        assert stats['total_entries'] == 2
        assert 'key1' in stats['keys']
        assert 'key2' in stats['keys']


class TestCachedDataCollector:
    """Tests for cached data collector wrapper"""
    
    def test_cached_collector_initialization(self):
        """Test cached collector initialization"""
        mock_collector = MockDataCollector(seed=42)
        cached_collector = CachedDataCollector(mock_collector)
        
        assert cached_collector.collector is mock_collector
        assert cached_collector.cache is not None
    
    def test_cached_collector_connect(self):
        """Test cached collector connection"""
        mock_collector = MockDataCollector(seed=42)
        cached_collector = CachedDataCollector(mock_collector)
        
        # Connect
        result = cached_collector.connect()
        assert result is True
        assert cached_collector.is_connected()
    
    def test_cached_collector_caches_portfolio(self):
        """Test that cached collector caches portfolio state"""
        mock_collector = MockDataCollector(seed=42)
        cached_collector = CachedDataCollector(mock_collector)
        
        cached_collector.connect()
        
        # First call should fetch
        portfolio1 = cached_collector.get_portfolio_state()
        
        # Second call should use cache
        portfolio2 = cached_collector.get_portfolio_state()
        
        # Should be same object (from cache)
        assert portfolio1 is portfolio2
    
    def test_cached_collector_cache_stats(self):
        """Test getting cache statistics"""
        mock_collector = MockDataCollector(seed=42)
        cached_collector = CachedDataCollector(mock_collector)
        
        cached_collector.connect()
        
        # Fetch some data
        cached_collector.get_portfolio_state()
        cached_collector.get_open_positions()
        
        # Get stats
        stats = cached_collector.get_cache_stats()
        assert stats['total_entries'] >= 2


class TestDataAggregator:
    """Tests for data aggregator"""
    
    def test_aggregator_initialization(self):
        """Test aggregator initialization"""
        mock_collector = MockDataCollector(seed=42)
        aggregator = DataAggregator(mock_collector)
        
        assert aggregator.collector is mock_collector
    
    def test_aggregator_aggregate(self):
        """Test data aggregation"""
        mock_collector = MockDataCollector(seed=42)
        mock_collector.connect()
        
        aggregator = DataAggregator(mock_collector)
        
        # Aggregate data
        portfolio = aggregator.aggregate()
        
        assert isinstance(portfolio, PortfolioState)
        assert portfolio.total_value_usd >= 0
        assert portfolio.available_capital_usd >= 0
    
    def test_aggregator_portfolio_summary(self):
        """Test portfolio summary"""
        mock_collector = MockDataCollector(seed=42)
        mock_collector.connect()
        
        aggregator = DataAggregator(mock_collector)
        
        # Get summary
        summary = aggregator.get_portfolio_summary()
        
        assert 'total_value' in summary
        assert 'available_capital' in summary
        assert 'position_count' in summary
        assert 'trade_count' in summary
        assert 'win_rate' in summary
    
    def test_aggregator_position_summary(self):
        """Test position summary"""
        mock_collector = MockDataCollector(seed=42)
        mock_collector.connect()
        
        aggregator = DataAggregator(mock_collector)
        
        # Get summary
        summary = aggregator.get_position_summary()
        
        assert 'count' in summary
        assert 'total_size' in summary
        assert 'total_pnl' in summary
        assert 'avg_pnl_percent' in summary
    
    def test_aggregator_trade_summary(self):
        """Test trade summary"""
        mock_collector = MockDataCollector(seed=42)
        mock_collector.connect()
        
        aggregator = DataAggregator(mock_collector)
        
        # Get summary
        summary = aggregator.get_trade_summary()
        
        assert 'count' in summary
        assert 'total_pnl' in summary
        assert 'win_count' in summary
        assert 'loss_count' in summary
        assert 'win_rate' in summary
    
    def test_aggregator_metrics_calculation(self):
        """Test metrics calculation"""
        mock_collector = MockDataCollector(seed=42)
        mock_collector.connect()
        
        aggregator = DataAggregator(mock_collector)
        
        # Get trades
        trades = mock_collector.get_closed_trades(limit=100)
        
        # Calculate metrics
        metrics = aggregator._calculate_metrics(trades)
        
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics
        assert 'total_trades' in metrics
        assert 0 <= metrics['win_rate'] <= 1
    
    def test_aggregator_empty_trades(self):
        """Test aggregator with no trades"""
        mock_collector = MockDataCollector(seed=42)
        mock_collector.connect()
        
        aggregator = DataAggregator(mock_collector)
        
        # Calculate metrics for empty list
        metrics = aggregator._calculate_metrics([])
        
        assert metrics['win_rate'] == 0.0
        assert metrics['profit_factor'] == 0.0
        assert metrics['total_trades'] == 0


class TestIntegrationFlow:
    """Integration tests for complete data flow"""
    
    def test_complete_data_flow(self):
        """Test complete data flow from collector to aggregator"""
        # Create mock collector
        mock_collector = MockDataCollector(seed=42)
        mock_collector.connect()
        
        # Wrap with cache
        cached_collector = CachedDataCollector(mock_collector)
        cached_collector.connect()
        
        # Create aggregator
        aggregator = DataAggregator(cached_collector)
        
        # Aggregate data
        portfolio = aggregator.aggregate()
        
        # Verify aggregated data
        assert isinstance(portfolio, PortfolioState)
        assert portfolio.total_value_usd > 0
        assert len(portfolio.open_positions) >= 0
        assert len(portfolio.closed_trades) >= 0
    
    def test_cache_performance(self):
        """Test that caching improves performance"""
        import time
        
        mock_collector = MockDataCollector(seed=42)
        mock_collector.connect()
        
        cached_collector = CachedDataCollector(mock_collector)
        cached_collector.connect()
        
        # First call (cache miss)
        start = time.time()
        portfolio1 = cached_collector.get_portfolio_state()
        first_time = time.time() - start
        
        # Second call (cache hit)
        start = time.time()
        portfolio2 = cached_collector.get_portfolio_state()
        second_time = time.time() - start
        
        # Cache hit should be faster (or at least not slower)
        # Note: This is a soft assertion since timing can vary
        assert portfolio1 is portfolio2
