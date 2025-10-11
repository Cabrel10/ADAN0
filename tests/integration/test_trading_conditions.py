import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from src.adan_trading_bot.portfolio.portfolio_manager import PortfolioManager
from src.adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine

class TestTradingConditionsIntegration(unittest.TestCase):
    
    def setUp(self):
        """Setup pour les tests d'intégration"""
        self.test_config = {
            'trading': {
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
                }
            },
            'environment': {
                'initial_balance': 50.0,
                'assets': ['BTCUSDT'],
                'timeframes': ['5m', '1h', '4h'],
                'max_steps': 100
            }
        }
    
    def test_capital_tier_transitions(self):
        """Test des transitions entre paliers de capital"""
        print("\n=== TEST TRANSITIONS PALIERS ===")
        
        capitals_to_test = [10, 25, 50, 100, 200]
        expected_tiers = ['Micro', 'Mini', 'Standard', 'Professional', 'Professional']
        
        for capital, expected_tier in zip(capitals_to_test, expected_tiers):
            pm = PortfolioManager(config={'trading': self.test_config['trading'], 'environment': {'initial_balance': capital}}, worker_id=0)
            
            self.assertEqual(pm.get_current_tier()['name'], expected_tier,
                           f"Capital {capital} devrait être dans le palier {expected_tier}")
            

    
    def test_timeframe_alignment(self):
        """Test de l'alignement des timeframes avec les paramètres de trading"""
        print("\n=== TEST ALIGNEMENT TIMEFRAMES ===")
        
        # Création d'un environnement de test
        env = MultiAssetChunkedEnv(
            assets=['BTCUSDT'],
            initial_cash=50.0,
            max_steps=50,
            data={'BTCUSDT': {'5m': {'open': [1,2,3], 'high': [1,2,3], 'low': [1,2,3], 'close': [1,2,3], 'volume': [1,2,3]}}},
            timeframes=['5m', '1h', '4h'],
            window_size=2,
            features_config={
                '5m': ['open', 'high', 'low', 'close', 'volume'],
                '1h': ['open', 'high', 'low', 'close', 'volume'],
                '4h': ['open', 'high', 'low', 'close', 'volume'],
            },
            worker_config={'assets': ['BTCUSDT'], 'timeframes': ['5m', '1h', '4h']},
            config=self.test_config
        )
        
        # Test des différents timeframes
        timeframes = ['5m', '1h', '4h']
        for tf in timeframes:
            # Simuler la sélection du timeframe
            action = [0.5, 0.5, 0.5, timeframes.index(tf)/len(timeframes)]  # Dernier élément = choix timeframe
            
            obs, reward, done, info = env.step(action)
            
            # Vérifier que les limites de timeframe sont respectées
            timeframe_limits = env.timeframe_limits[tf]
            current_counts = env.timeframe_trade_counts[tf]
            
            self.assertLessEqual(current_counts, timeframe_limits['max_positions'],
                               f"Timeframe {tf} dépasse le nombre max de positions")
            
            print(f"✅ Timeframe {tf}: {current_counts}/{timeframe_limits['max_positions']} positions")
    
    def test_dynamic_behavior_engine_integration(self):
        """Test de l'intégration du Dynamic Behavior Engine"""
        print("\n=== TEST DBE INTÉGRATION ===")
        
        dbe = DynamicBehaviorEngine()
        pm = PortfolioManager(config=self.test_config, worker_id=0)
        
        # Simuler différentes conditions de marché
        test_conditions = [
            {'regime': 'bull', 'volatility': 0.1, 'win_rate': 0.7},
            {'regime': 'bear', 'volatility': 0.3, 'win_rate': 0.3},
            {'regime': 'neutral', 'volatility': 0.2, 'win_rate': 0.5}
        ]
        
        for condition in test_conditions:
            dbe.update_state({
                'profit_loss': condition['win_rate'] * 100 - (1-condition['win_rate']) * 50,
                'win_rate': condition['win_rate'],
                'drawdown': 0.1,
                'sharpe_ratio': 1.5 if condition['win_rate'] > 0.5 else 0.5
            })
            
            modulation = dbe.compute_dynamic_modulation()
            
            print(f"✅ Régime {condition['regime']}: SL={modulation['stop_loss_pct']:.2f}%, "
                  f"TP={modulation['take_profit_pct']:.2f}%, "
                  f"PosSize={modulation['position_size_pct']:.1f}%")
    
    def test_end_to_end_trading_cycle(self):
        """Test complet du cycle de trading"""
        print("\n=== TEST CYCLE COMPLET ===")
        
        env = MultiAssetChunkedEnv(
            assets=['BTCUSDT'],
            initial_cash=100.0,
            max_steps=20,
            data={'BTCUSDT': {'5m': {'open': [1,2,3], 'high': [1,2,3], 'low': [1,2,3], 'close': [1,2,3], 'volume': [1,2,3]}}},
            timeframes=['5m', '1h', '4h'],
            window_size=2,
            features_config={
                '5m': ['open', 'high', 'low', 'close', 'volume'],
                '1h': ['open', 'high', 'low', 'close', 'volume'],
                '4h': ['open', 'high', 'low', 'close', 'volume'],
            },
            worker_config={'assets': ['BTCUSDT'], 'timeframes': ['5m', '1h', '4h']},
            config=self.test_config
        )
        
        # Simuler plusieurs actions de trading
        test_actions = [
            [1.0, 0.0, 0.0, 0.0],  # Achat timeframe 5m
            [0.0, 0.0, 0.0, 0.0],  # Hold
            [-1.0, 0.0, 0.0, 0.5], # Vente timeframe 1h
            [0.0, 0.0, 0.0, 1.0]   # Hold timeframe 4h
        ]
        
        for i, action in enumerate(test_actions):
            obs, reward, done, info = env.step(action)
            
            print(f"Step {i+1}: PV={info.get('portfolio_value', 'N/A'):.2f}, "
                  f"Reward={reward:.2f}, Trades={info.get('executed_trades_opened', 0)}")
            
            if done:
                break
        
        final_stats = env.get_trading_statistics()
        print(f"\n📊 STATISTIQUES FINALES:")
        print(f"   Valeur portefeuille: {final_stats.get('final_portfolio_value', 'N/A'):.2f}")
        print(f"   Trades ouverts: {final_stats.get('total_trades_opened', 0)}")
        print(f"   Trades fermés: {final_stats.get('total_trades_closed', 0)}")
        print(f"   Sharpe Ratio: {final_stats.get('sharpe_ratio', 'N/A'):.2f}")

if __name__ == '__main__':
    # Création de la suite de tests
    suite = unittest.TestSuite()
    suite.addTest(TestTradingConditionsIntegration('test_capital_tier_transitions'))
    suite.addTest(TestTradingConditionsIntegration('test_timeframe_alignment')) 
    suite.addTest(TestTradingConditionsIntegration('test_dynamic_behavior_engine_integration'))
    suite.addTest(TestTradingConditionsIntegration('test_end_to_end_trading_cycle'))
    
    # Exécution des tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)