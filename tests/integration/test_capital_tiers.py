
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import copy

# Add the project root to the Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.adan_trading_bot.common.enhanced_config_manager import get_config_manager
from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

class TestCapitalTiers(unittest.TestCase):

    def setUp(self):
        """Set up a mock environment for integration tests."""
        # Create mock data
        self.assets = ["BTCUSDT"]
        self.timeframes = ["5m"]
        self.data = {self.assets[0]: {"5m": pd.DataFrame({
            'open': np.random.uniform(40000, 41000, size=100),
            'high': np.random.uniform(41000, 42000, size=100),
            'low': np.random.uniform(39000, 40000, size=100),
            'close': np.random.uniform(40000, 41000, size=100),
            'volume': np.random.uniform(10, 100, size=100),
        })}}

        # Load the entire monolithic config file
        config_manager = get_config_manager(config_dir='config', enable_hot_reload=False, force_new=True)
        self.config = config_manager.load_config('config/config.yaml')

        # Mock the data loader
        self.mock_data_loader = MagicMock()
        self.mock_data_loader.load_chunk.return_value = self.data
        self.mock_data_loader.features_by_timeframe = {'5m': ['open', 'high', 'low', 'close', 'volume']}
        self.mock_data_loader.total_chunks = 1

        # Patch the ChunkedDataLoader
        self.patcher = patch('src.adan_trading_bot.data_processing.data_loader.ChunkedDataLoader', return_value=self.mock_data_loader)
        self.mock_chunked_data_loader = self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_capital_tier_position_sizing(self):
        """Test that position sizing respects capital tier rules."""
        tiers = self.config['capital_tiers']

        for tier in tiers:
            if tier['name'] == 'Enterprise': continue

            initial_balance = (tier['min_capital'] + tier.get('max_capital', tier['min_capital'] * 2)) / 2
            if initial_balance == 0: initial_balance = 15 # for Micro tier

            config_copy = copy.deepcopy(self.config)
            config_copy['environment']['initial_balance'] = initial_balance
            
            env = MultiAssetChunkedEnv(
                data=self.data,
                timeframes=self.timeframes,
                window_size=20,
                features_config=config_copy['data']['features_config']['timeframes'],
                config=config_copy,
                worker_config=config_copy['workers']['w1']
            )
            env.reset()

            with patch.object(env.portfolio_manager, 'open_position', wraps=env.portfolio_manager.open_position) as mock_open_position:
                action = np.array([0.9, 0.1]) # High confidence buy
                env.step(action)

                if mock_open_position.called:
                    args, kwargs = mock_open_position.call_args
                    opened_size_usd = kwargs['size'] * kwargs['price']
                    portfolio_value = env.portfolio_manager.get_portfolio_value()
                    position_size_pct = opened_size_usd / portfolio_value
                    
                    max_pos_size_pct_from_config = tier['max_position_size_pct'] / 100.0

                    print(f"\n--- Testing Tier: {tier['name']} ---")
                    print(f"Initial Balance: {initial_balance:.2f}")
                    print(f"Max Position Size (config): {max_pos_size_pct_from_config:.2%}")
                    print(f"Actual Position Size: {position_size_pct:.2%}")

                    self.assertLessEqual(position_size_pct, max_pos_size_pct_from_config + 0.001)

if __name__ == '__main__':
    unittest.main()
