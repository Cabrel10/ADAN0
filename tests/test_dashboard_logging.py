"""
Tests for ADAN Dashboard logging system

Tests for DashboardLogger and LogStorage.
"""

import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from src.adan_trading_bot.dashboard.logger import DashboardLogger
from src.adan_trading_bot.dashboard.log_storage import LogStorage
from src.adan_trading_bot.dashboard.models import (
    Position,
    ClosedTrade,
    Alert,
)


class TestDashboardLogger:
    """Tests for DashboardLogger"""
    
    def test_logger_initialization(self):
        """Test logger initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DashboardLogger(log_dir=tmpdir)
            assert logger is not None
            assert logger.trade_entries == 0
            assert logger.trade_exits == 0
            assert logger.alerts_logged == 0
    
    def test_log_trade_entry(self):
        """Test logging trade entry"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DashboardLogger(log_dir=tmpdir)
            
            position = Position(
                pair="BTCUSDT",
                side="LONG",
                size_btc=1.0,
                entry_price=50000.0,
                current_price=50000.0,
                sl_price=49000.0,
                tp_price=51000.0,
                open_time=datetime.now(),
                entry_signal_strength=0.85,
                entry_market_regime="Trending",
                entry_volatility=2.5,
                entry_rsi=65,
            )
            
            logger.log_trade_entry(position, 0.85, 50000.0)
            
            assert logger.trade_entries == 1
    
    def test_log_trade_exit(self):
        """Test logging trade exit"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DashboardLogger(log_dir=tmpdir)
            
            trade = ClosedTrade(
                pair="BTCUSDT",
                side="LONG",
                size_btc=1.0,
                entry_price=50000.0,
                exit_price=51000.0,
                open_time=datetime.now() - timedelta(hours=1),
                close_time=datetime.now(),
                close_reason="TP Hit",
                entry_confidence=0.85,
            )
            
            logger.log_trade_exit(trade, 51000.0)
            
            assert logger.trade_exits == 1
    
    def test_log_alert(self):
        """Test logging alert"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DashboardLogger(log_dir=tmpdir)
            
            alert = Alert(
                message="Test alert",
                severity="WARNING",
                timestamp=datetime.now(),
            )
            
            logger.log_alert(alert)
            
            assert logger.alerts_logged == 1
    
    def test_log_system_startup(self):
        """Test logging system startup"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DashboardLogger(log_dir=tmpdir)
            
            config = {
                'refresh_rate': 2.0,
                'data_source': 'mock',
            }
            
            logger.log_system_startup(config)
    
    def test_log_system_shutdown(self):
        """Test logging system shutdown"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DashboardLogger(log_dir=tmpdir)
            
            logger.log_system_shutdown(3600.0)
    
    def test_get_stats(self):
        """Test getting logging statistics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DashboardLogger(log_dir=tmpdir)
            
            # Log some events
            position = Position(
                pair="BTCUSDT",
                side="LONG",
                size_btc=1.0,
                entry_price=50000.0,
                current_price=50000.0,
                sl_price=49000.0,
                tp_price=51000.0,
                open_time=datetime.now(),
                entry_signal_strength=0.85,
                entry_market_regime="Trending",
                entry_volatility=2.5,
                entry_rsi=65,
            )
            
            logger.log_trade_entry(position, 0.85, 50000.0)
            
            stats = logger.get_stats()
            
            assert stats['trade_entries'] == 1
            assert stats['trade_exits'] == 0
            assert stats['alerts_logged'] == 0


class TestLogStorage:
    """Tests for LogStorage"""
    
    def test_storage_initialization(self):
        """Test storage initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LogStorage(storage_dir=tmpdir)
            assert storage is not None
            assert storage.storage_dir.exists()
    
    def test_store_trade_entry(self):
        """Test storing trade entry"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LogStorage(storage_dir=tmpdir)
            
            trade_data = {
                'pair': 'BTCUSDT',
                'side': 'LONG',
                'size_btc': 1.0,
                'entry_price': 50000.0,
            }
            
            storage.store_trade_entry(trade_data)
            
            assert storage.trades_file.exists()
    
    def test_store_trade_exit(self):
        """Test storing trade exit"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LogStorage(storage_dir=tmpdir)
            
            trade_data = {
                'pair': 'BTCUSDT',
                'side': 'LONG',
                'size_btc': 1.0,
                'exit_price': 51000.0,
                'realized_pnl_usd': 1000.0,
            }
            
            storage.store_trade_exit(trade_data)
            
            assert storage.trades_file.exists()
    
    def test_store_metrics(self):
        """Test storing metrics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LogStorage(storage_dir=tmpdir)
            
            metrics = {
                'win_rate': 0.65,
                'profit_factor': 2.5,
                'total_trades': 20,
            }
            
            storage.store_metrics(metrics)
            
            assert storage.metrics_file.exists()
    
    def test_store_event(self):
        """Test storing event"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LogStorage(storage_dir=tmpdir)
            
            event_data = {
                'message': 'Test event',
                'severity': 'INFO',
            }
            
            storage.store_event('TEST_EVENT', event_data)
            
            assert storage.events_file.exists()
    
    def test_get_recent_trades(self):
        """Test getting recent trades"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LogStorage(storage_dir=tmpdir)
            
            # Store some trades
            for i in range(5):
                storage.store_trade_entry({
                    'pair': 'BTCUSDT',
                    'side': 'LONG',
                    'size_btc': 1.0,
                    'entry_price': 50000.0 + i * 100,
                })
            
            trades = storage.get_recent_trades(limit=3)
            
            assert len(trades) <= 3
    
    def test_get_trades_by_date(self):
        """Test getting trades by date range"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LogStorage(storage_dir=tmpdir)
            
            # Store trades
            storage.store_trade_entry({
                'pair': 'BTCUSDT',
                'side': 'LONG',
                'size_btc': 1.0,
                'entry_price': 50000.0,
            })
            
            # Get trades from today
            start_date = datetime.now() - timedelta(days=1)
            end_date = datetime.now() + timedelta(days=1)
            
            trades = storage.get_trades_by_date(start_date, end_date)
            
            assert len(trades) >= 1
    
    def test_get_recent_metrics(self):
        """Test getting recent metrics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LogStorage(storage_dir=tmpdir)
            
            # Store metrics
            for i in range(3):
                storage.store_metrics({
                    'win_rate': 0.5 + i * 0.1,
                    'profit_factor': 2.0 + i * 0.5,
                })
            
            metrics = storage.get_recent_metrics(limit=2)
            
            assert len(metrics) <= 2
    
    def test_get_events_by_type(self):
        """Test getting events by type"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LogStorage(storage_dir=tmpdir)
            
            # Store events
            storage.store_event('STARTUP', {'config': {}})
            storage.store_event('SHUTDOWN', {'runtime': 3600})
            storage.store_event('STARTUP', {'config': {}})
            
            startup_events = storage.get_events_by_type('STARTUP')
            
            assert len(startup_events) == 2
    
    def test_get_trade_statistics(self):
        """Test getting trade statistics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LogStorage(storage_dir=tmpdir)
            
            # Store trades
            storage.store_trade_entry({'pair': 'BTCUSDT', 'side': 'LONG'})
            storage.store_trade_exit({
                'pair': 'BTCUSDT',
                'side': 'LONG',
                'realized_pnl_usd': 1000.0,
            })
            
            storage.store_trade_entry({'pair': 'BTCUSDT', 'side': 'SHORT'})
            storage.store_trade_exit({
                'pair': 'BTCUSDT',
                'side': 'SHORT',
                'realized_pnl_usd': -500.0,
            })
            
            stats = storage.get_trade_statistics()
            
            assert stats['total_trades'] == 2
            assert stats['winning_trades'] == 1
            assert stats['losing_trades'] == 1
            assert stats['win_rate'] == 0.5
            assert stats['total_pnl'] == 500.0
    
    def test_clear_old_data(self):
        """Test clearing old data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LogStorage(storage_dir=tmpdir)
            
            # Store data
            storage.store_trade_entry({'pair': 'BTCUSDT'})
            storage.store_metrics({'win_rate': 0.5})
            
            # Clear old data (should keep recent data)
            storage.clear_old_data(days=30)
            
            # Files should still exist
            assert storage.trades_file.exists()
            assert storage.metrics_file.exists()
