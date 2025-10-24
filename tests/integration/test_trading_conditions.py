import unittest
import sys
import os
import pandas as pd
import copy
from unittest.mock import patch, MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from src.adan_trading_bot.portfolio.portfolio_manager import PortfolioManager
from src.adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine

class TestTradingConditionsIntegration(unittest.TestCase):
    
    def setUp(self):
        """Setup pour les tests d'intégration"""
        self.test_config = {
            'data': {
                'timeframes': ['5m', '1h', '4h'],
                'features_config': {
                    'timeframes': {
                        '5m': {
                            'price': ['open', 'high', 'low', 'close'],
                            'volume': ['volume'],
                            'indicators': []
                        },
                        '1h': {
                            'price': ['open', 'high', 'low', 'close'],
                            'volume': ['volume'],
                            'indicators': []
                        },
                        '4h': {
                            'price': ['open', 'high', 'low', 'close'],
                            'volume': ['volume'],
                            'indicators': []
                        }
                    }
                }
            },
            'capital_tiers': [
                {'name': 'Micro', 'min_capital': 0, 'max_capital': 20, 'max_position_size_pct': 90, 'stop_loss_pct': 0.1, 'take_profit_pct': 0.2},
                {'name': 'Mini', 'min_capital': 20, 'max_capital': 50, 'max_position_size_pct': 80, 'stop_loss_pct': 0.08, 'take_profit_pct': 0.15},
                {'name': 'Standard', 'min_capital': 50, 'max_capital': 100, 'max_position_size_pct': 70, 'stop_loss_pct': 0.06, 'take_profit_pct': 0.12},
                {'name': 'Professional', 'min_capital': 100, 'max_capital': 500, 'max_position_size_pct': 50, 'stop_loss_pct': 0.05, 'take_profit_pct': 0.1},
                {'name': 'Institutional', 'min_capital': 500, 'max_capital': 10000, 'max_position_size_pct': 30, 'stop_loss_pct': 0.04, 'take_profit_pct': 0.08},
            ],
            'trading_rules': {
                'frequency': {
                    'max_positions': {'5m': 10, '1h': 5, '4h': 2}
                }
            },
            'environment': {
                'initial_balance': 50.0,
                'assets': ['BTCUSDT'],
                'timeframes': ['5m', '1h', '4h'],
                'max_steps': 100
            },
            'dbe': {
                'enabled': True,
                'base_params': {'stop_loss_pct': 0.02, 'take_profit_pct': 0.04, 'position_size_pct': 0.7},
                'modulation_config': {
                    'bull': {'stop_loss_pct': 1.5, 'take_profit_pct': 0.8, 'position_size_pct': 1.2},
                    'bear': {'stop_loss_pct': 0.8, 'take_profit_pct': 1.2, 'position_size_pct': 0.8},
                    'neutral': {'stop_loss_pct': 1.0, 'take_profit_pct': 1.0, 'position_size_pct': 1.0}
                }
            }
        }
    
    def test_capital_tier_transitions(self):
        """Test des transitions entre paliers de capital"""
        print("\n=== TEST TRANSITIONS PALIERS ===")
        
        capitals_to_test = [10, 25, 50, 100, 200]
        expected_tiers = ['Micro', 'Mini', 'Standard', 'Professional', 'Professional']
        
        for capital, expected_tier in zip(capitals_to_test, expected_tiers):
            pm_config = {
                'capital_tiers': self.test_config['capital_tiers'],
                'environment': {'initial_balance': capital}
            }
            pm = PortfolioManager(config=pm_config, worker_id=0)
            
            self.assertEqual(pm.get_current_tier()['name'], expected_tier,
                           f"Capital {capital} devrait être dans le palier {expected_tier}")

    def test_timeframe_alignment(self):
        """Test de l'alignement des timeframes avec les paramètres de trading"""
        print("\n=== TEST ALIGNEMENT TIMEFRAMES ===")
        
        config = copy.deepcopy(self.test_config)
        config['environment']['assets'] = ['BTCUSDT']

        mock_data = {'BTCUSDT': {'5m': pd.DataFrame({'open': [1,2,3], 'high': [1,2,3], 'low': [1,2,3], 'close': [1,2,3], 'volume': [1,2,3]})}}

        with patch('src.adan_trading_bot.data_processing.data_loader.ChunkedDataLoader') as mock_loader:
            mock_instance = mock_loader.return_value
            mock_instance.load_chunk.return_value = mock_data
            mock_instance.total_chunks = 1
            mock_instance.features_by_timeframe = {'5m': ['open', 'high', 'low', 'close', 'volume']}

            env = MultiAssetChunkedEnv(
                data=mock_data,
                timeframes=['5m'],
                window_size=2,
                features_config=config['data']['features_config']['timeframes'],
                worker_config={'assets': ['BTCUSDT'], 'timeframes': ['5m']},
                config=config
            )
            env.reset()
            
            timeframes = ['5m']
            for tf in timeframes:
                action = [0.5, 0.5, 0.5, 0.1]
                obs, reward, done, truncated, info = env.step(action)
                timeframe_limits = self.test_config['trading_rules']['frequency']
                current_counts = env.positions_count[tf]
                
                self.assertLessEqual(current_counts, timeframe_limits['max_positions'][tf],
                                   f"Timeframe {tf} dépasse le nombre max de positions")
                print(f"✅ Timeframe {tf}: {current_counts}/{timeframe_limits['max_positions'][tf]} positions")
    
    def test_dynamic_behavior_engine_integration(self):
        """Test de l'intégration du Dynamic Behavior Engine"""
        print("\n=== TEST DBE INTÉGRATION ===")
        
        dbe = DynamicBehaviorEngine(config=self.test_config['dbe'])
        
        test_conditions = [
            {'regime': 'BULL', 'volatility': 0.1, 'win_rate': 0.7},
            {'regime': 'BEAR', 'volatility': 0.3, 'win_rate': 0.3},
            {'regime': 'NEUTRAL', 'volatility': 0.2, 'win_rate': 0.5}
        ]
        
        for condition in test_conditions:
            dbe.update_state({'win_rate': condition['win_rate'], 'volatility': condition['volatility']})
            dbe.current_regime = condition['regime']
            modulation = dbe.compute_dynamic_modulation()
            
            self.assertIn('sl_pct', modulation)
            self.assertIn('tp_pct', modulation)
            self.assertIn('position_size_pct', modulation)

            print(f"✅ Régime {condition['regime']}: SL={modulation['sl_pct']:.2f}%, "
                  f"TP={modulation['tp_pct']:.2f}%, "
                  f"PosSize={modulation['position_size_pct']:.1f}%")
    
    def test_end_to_end_trading_cycle(self):
        """Test complet du cycle de trading"""
        print("\n=== TEST CYCLE COMPLET ===")
        
        config = copy.deepcopy(self.test_config)
        config['environment']['initial_balance'] = 100.0

        mock_data = {'BTCUSDT': {'5m': pd.DataFrame({'open': [1,2,3,4,5], 'high': [1,2,3,4,5], 'low': [1,2,3,4,5], 'close': [1,2,3,4,5], 'volume': [1,2,3,4,5]})}}

        with patch('src.adan_trading_bot.data_processing.data_loader.ChunkedDataLoader') as mock_loader:
            mock_instance = mock_loader.return_value
            mock_instance.load_chunk.return_value = mock_data
            mock_instance.total_chunks = 1
            mock_instance.features_by_timeframe = {'5m': ['open', 'high', 'low', 'close', 'volume']}

            env = MultiAssetChunkedEnv(
                data=mock_data,
                timeframes=['5m'],
                window_size=2,
                features_config=config['data']['features_config']['timeframes'],
                worker_config={'assets': ['BTCUSDT'], 'timeframes': ['5m']},
                config=config
            )
            env.reset()
            
            test_actions = [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ]
            
            for i, action in enumerate(test_actions):
                obs, reward, done, truncated, info = env.step(action)
                
                print(f"Step {i+1}: PV={info.get('portfolio_value', 'N/A'):.2f}, "
                      f"Reward={reward:.2f}, Trades={info.get('executed_trades_opened', 0)}")
                
                if done:
                    break
            
            final_stats = env.portfolio_manager.get_metrics()
            print(f"\n📊 STATISTIQUES FINALES:")
            print(f"   Valeur portefeuille: {final_stats.get('total_value', 'N/A'):.2f}")
            print(f"   Trades ouverts: {final_stats.get('total_trades', 0)}")
            print(f"   Trades fermés: {final_stats.get('closed_trades_count', 0)}")
            print(f"   Sharpe Ratio: {final_stats.get('sharpe_ratio', 'N/A'):.2f}")

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestTradingConditionsIntegration('test_capital_tier_transitions'))
    suite.addTest(TestTradingConditionsIntegration('test_timeframe_alignment')) 
    suite.addTest(TestTradingConditionsIntegration('test_dynamic_behavior_engine_integration'))
    suite.addTest(TestTradingConditionsIntegration('test_end_to_end_trading_cycle'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
