
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from datetime import datetime

# Add the project root to the Python path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.adan_trading_bot.portfolio.portfolio_manager import PortfolioManager, Position
from src.adan_trading_bot.performance.metrics import PerformanceMetrics

class TestPortfolioManager(unittest.TestCase):

    def setUp(self):
        """Set up a mock config and a PortfolioManager instance for tests."""
        self.config = {
            "environment": {
                "initial_balance": 100.0,
            },
            "assets": ["BTCUSDT"],
            "risk_management": {
                "global_sl_pct": 20.0,
            },
            "trading_rules": {
                "duration_tracking": {
                    "5m": {"max_duration_steps": 12}
                }
            }
        }
        # Mock PerformanceMetrics to avoid complex dependencies
        self.mock_metrics = MagicMock()
        self.mock_metrics.returns = MagicMock()
        self.mock_metrics.drawdowns = MagicMock()
        self.mock_metrics.equity_curve = MagicMock()
        self.mock_metrics.trades = MagicMock()
        self.mock_metrics.closed_positions = MagicMock()
        self.mock_metrics.frequency_history = MagicMock()
        
        # Patch the logger to suppress output during tests
        self.patcher = patch('src.adan_trading_bot.portfolio.portfolio_manager.logger')
        self.mock_logger = self.patcher.start()

        self.pm = PortfolioManager(config=self.config, worker_id=1, performance_metrics=self.mock_metrics)

    def tearDown(self):
        """Stop the logger patcher."""
        self.patcher.stop()

    def test_initial_state(self):
        """Test that the portfolio manager initializes correctly."""
        self.assertEqual(self.pm.get_cash(), 100.0)
        self.assertEqual(self.pm.get_portfolio_value(), 100.0)
        self.assertEqual(len(self.pm.positions), 1)
        self.assertIn("BTCUSDT", self.pm.positions)
        self.assertFalse(self.pm.positions["BTCUSDT"].is_open)

    def test_open_position_insufficient_cash(self):
        """Test that opening a position fails if cash is insufficient."""
        # Try to open a position worth 110 USDT with only 100 USDT cash
        receipt = self.pm.open_position(
            asset="BTCUSDT",
            price=50000,
            size=110 / 50000,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            timestamp=datetime.now(),
            current_step=1
        )
        self.assertIsNone(receipt, "Position should not be opened due to insufficient cash")
        self.assertEqual(self.pm.get_cash(), 100.0, "Cash should not change if position fails to open")
        self.assertFalse(self.pm.positions["BTCUSDT"].is_open, "Position should be marked as closed")

    def test_open_position_sufficient_cash(self):
        """Test that opening a position succeeds with enough cash."""
        receipt = self.pm.open_position(
            asset="BTCUSDT",
            price=50000,
            size=50 / 50000,  # 50 USDT worth
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            timestamp=datetime.now(),
            current_step=1
        )
        self.assertIsNotNone(receipt)
        self.assertEqual(self.pm.get_cash(), 50.0)
        self.assertTrue(self.pm.positions["BTCUSDT"].is_open)
        self.assertEqual(self.pm.positions["BTCUSDT"].size, 50 / 50000)

    def test_cash_gate_with_open_position(self):
        """Test the cash gate when one position is already open."""
        # Open a first position worth 60 USDT
        self.pm.open_position(
            asset="BTCUSDT",
            price=50000,
            size=60 / 50000,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            timestamp=datetime.now(),
            current_step=1
        )
        self.assertAlmostEqual(self.pm.get_cash(), 40.0)

        # Try to open another position that requires more than the remaining cash (e.g. 50 USDT)
        # Note: This test assumes one position per asset. To test multi-asset, we need to add another asset.
        # Let's add another asset to the config for this test.
        self.pm.positions["ETHUSDT"] = Position()
        
        receipt = self.pm.open_position(
            asset="ETHUSDT",
            price=3000,
            size=50 / 3000, # Requires 50 USDT
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            timestamp=datetime.now(),
            current_step=2
        )
        
        self.assertIsNone(receipt, "Second position should not open due to insufficient remaining cash")
        self.assertAlmostEqual(self.pm.get_cash(), 40.0, msg="Cash should not change")
        self.assertFalse(self.pm.positions["ETHUSDT"].is_open, "ETHUSDT position should not be open")
        self.assertTrue(self.pm.positions["BTCUSDT"].is_open, "BTCUSDT position should remain open")

    def test_close_position(self):
        """Test closing a position and cash update."""
        open_time = datetime.now()
        # Open a position
        self.pm.open_position(
            asset="BTCUSDT",
            price=50000,
            size=50 / 50000,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            timestamp=open_time,
            current_step=1
        )
        self.assertEqual(self.pm.get_cash(), 50.0)

        # Close the position at a different price
        close_receipt = self.pm.close_position(
            asset="BTCUSDT",
            price=51000, # Price increased
            timestamp=datetime.now(),
            reason="TP"
        )
        
        self.assertIsNotNone(close_receipt)
        # Expected cash: 50 (remaining) + 51 (from closing the position) = 101
        self.assertAlmostEqual(self.pm.get_cash(), 101.0)
        self.assertFalse(self.pm.positions["BTCUSDT"].is_open)
        self.assertAlmostEqual(close_receipt['pnl'], 1.0)

    def test_max_duration_position_closure(self):
        """Test that a position is automatically closed if it exceeds max duration."""
        # Config max duration to 10 steps for 5m timeframe
        self.config["trading_rules"]["duration_tracking"]["5m"]["max_duration_steps"] = 10
        self.pm = PortfolioManager(config=self.config, worker_id=1, performance_metrics=self.mock_metrics)

        open_step = 1
        open_price = 50000
        size = 50 / open_price
        
        # Open a position
        receipt = self.pm.open_position(
            asset="BTCUSDT",
            price=open_price,
            size=size,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            timestamp=datetime.now(),
            current_step=open_step,
            timeframe="5m"
        )
        self.assertIsNotNone(receipt)
        self.assertTrue(self.pm.positions["BTCUSDT"].is_open)

        # Register a timestamp before updating market price
        self.pm.register_market_timestamp(datetime.now())

        # Simulate steps within the duration limit
        for step in range(open_step + 1, open_step + 11):
            self.pm.update_market_price(current_prices={"BTCUSDT": 50100}, current_step=step)
            self.assertTrue(self.pm.positions["BTCUSDT"].is_open, f"Position should be open at step {step}")

        # Simulate the step that exceeds the max duration
        final_step = open_step + 11
        closing_price = 50200
        
        # Register a final timestamp
        self.pm.register_market_timestamp(datetime.now())
        
        realized_pnl, closed_receipts = self.pm.update_market_price(
            current_prices={"BTCUSDT": closing_price}, current_step=final_step
        )

        self.assertFalse(self.pm.positions["BTCUSDT"].is_open, "Position should be closed after exceeding max duration")
        self.assertEqual(len(closed_receipts), 1, "There should be one closed receipt")
        self.assertEqual(closed_receipts[0]['reason'], "MaxDuration")
        
        # Check if cash is updated correctly
        # Initial cash: 100. Cost: 50. Remaining cash: 50.
        # Value at close: size * closing_price = (50 / 50000) * 50200 = 50.2
        # Final cash: 50 (remaining) + 50.2 = 100.2
        self.assertAlmostEqual(self.pm.get_cash(), 100.2)
        self.assertAlmostEqual(realized_pnl, 0.2)


if __name__ == '__main__':
    unittest.main()
