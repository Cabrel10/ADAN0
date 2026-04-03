#!/usr/bin/env python3
"""
🔗 TESTS D'INTÉGRATION - SCRIPTS CRITIQUES
Valider l'intégration des 4 scripts critiques avec le système unifié
"""

import unittest
import sys
import os
from pathlib import Path
import time
import json

# Ajouter le chemin src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestRealisticTradingEnv(unittest.TestCase):
    """Tests de l'environnement de trading réaliste"""

    def test_01_env_import(self):
        """Test 1: Import de l'environnement"""
        try:
            from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv
            self.assertIsNotNone(RealisticTradingEnv)
            print("✅ RealisticTradingEnv importé avec succès")
        except Exception as e:
            self.fail(f"Erreur import env: {e}")

    def test_02_env_creation(self):
        """Test 2: Création de l'environnement"""
        try:
            from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv
            
            env = RealisticTradingEnv(
                live_mode=False,
                min_hold_steps=6,
                cooldown_steps=3,
                circuit_breaker_pct=0.15
            )
            self.assertIsNotNone(env)
            print("✅ Environnement créé avec succès")
        except Exception as e:
            self.fail(f"Erreur création env: {e}")

    def test_03_env_reset(self):
        """Test 3: Reset de l'environnement"""
        try:
            from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv
            
            env = RealisticTradingEnv(live_mode=False)
            obs = env.reset()
            
            self.assertIsNotNone(obs)
            print("✅ Environnement reset avec succès")
        except Exception as e:
            self.fail(f"Erreur reset env: {e}")

    def test_04_risk_manager_initialized(self):
        """Test 4: RiskManager initialisé"""
        try:
            from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv
            
            env = RealisticTradingEnv(live_mode=False)
            env.reset()
            
            # Vérifier que le RiskManager est initialisé
            self.assertTrue(hasattr(env, 'risk_manager'))
            print(f"✅ RiskManager initialisé: {env.risk_manager is not None}")
        except Exception as e:
            self.fail(f"Erreur vérification RiskManager: {e}")

    def test_05_circuit_breaker_check(self):
        """Test 5: Vérification du circuit breaker"""
        try:
            from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv
            
            env = RealisticTradingEnv(live_mode=False, circuit_breaker_pct=0.15)
            env.reset()
            
            # Vérifier que la méthode existe
            self.assertTrue(hasattr(env, '_check_circuit_breakers'))
            
            # Appeler la méthode
            should_stop = env._check_circuit_breakers()
            
            # Au démarrage, ne devrait pas arrêter
            self.assertFalse(should_stop)
            print("✅ Circuit breaker fonctionne")
        except Exception as e:
            self.fail(f"Erreur circuit breaker: {e}")


class TestRewardCalculatorIntegration(unittest.TestCase):
    """Tests d'intégration du calculateur de récompense"""

    def test_01_reward_calculator_with_unified_system(self):
        """Test 1: RewardCalculator utilise le système unifié"""
        try:
            from adan_trading_bot.environment.reward_calculator import RewardCalculator
            
            config = {
                "reward_shaping": {
                    "realized_pnl_multiplier": 1.0,
                    "unrealized_pnl_multiplier": 0.1,
                    "inaction_penalty": -0.0001,
                    "reward_clipping_range": [-5.0, 5.0]
                }
            }
            
            calc = RewardCalculator(config)
            
            # Vérifier que UnifiedMetrics est initialisé
            self.assertTrue(hasattr(calc, 'unified_metrics'))
            print("✅ RewardCalculator utilise UnifiedMetrics")
        except Exception as e:
            self.fail(f"Erreur RewardCalculator: {e}")

    def test_02_reward_calculation(self):
        """Test 2: Calcul de récompense"""
        try:
            from adan_trading_bot.environment.reward_calculator import RewardCalculator
            
            config = {
                "reward_shaping": {
                    "realized_pnl_multiplier": 1.0,
                    "unrealized_pnl_multiplier": 0.1,
                    "inaction_penalty": -0.0001,
                    "reward_clipping_range": [-5.0, 5.0]
                }
            }
            
            calc = RewardCalculator(config)
            
            # Préparer les données de test
            portfolio_metrics = {
                "total_commission": 0.1,
                "drawdown": -0.05,
                "closed_positions": [],
                "position_metadata": {},
                "portfolio_value": 10000
            }
            
            # Calculer la récompense
            reward = calc.calculate(
                portfolio_metrics=portfolio_metrics,
                trade_pnl=100,
                action=1,
                chunk_id=None,
                optimal_chunk_pnl=None,
                performance_ratio=None,
                is_hunting=False,
                risk_horizon=0.0,
                trade_reason=None
            )
            
            # Vérifier que c'est un nombre valide
            self.assertIsInstance(reward, float)
            self.assertFalse(float('inf') == reward)
            self.assertFalse(float('-inf') == reward)
            print(f"✅ Récompense calculée: {reward:.4f}")
        except Exception as e:
            self.fail(f"Erreur calcul récompense: {e}")


class TestCentralLoggerIntegration(unittest.TestCase):
    """Tests d'intégration du logger centralisé"""

    def test_01_logger_creates_files(self):
        """Test 1: Logger crée les fichiers"""
        try:
            from adan_trading_bot.common.central_logger import logger as central_logger
            
            # Générer quelques logs
            central_logger.metric("test_metric", 1.5)
            central_logger.trade("BUY", "BTCUSDT", 0.5, 45000, 500)
            central_logger.validation("test", True, "test message")
            
            # Vérifier que les fichiers existent
            logs_dir = Path("logs")
            self.assertTrue(logs_dir.exists())
            
            log_files = list(logs_dir.glob("*.log"))
            self.assertGreater(len(log_files), 0)
            
            print(f"✅ Logger a créé {len(log_files)} fichiers")
        except Exception as e:
            self.fail(f"Erreur logger files: {e}")

    def test_02_logger_json_output(self):
        """Test 2: Logger génère du JSON"""
        try:
            from adan_trading_bot.common.central_logger import logger as central_logger
            
            # Générer un log
            central_logger.metric("test_json", 2.5)
            
            # Vérifier que le fichier JSON existe
            json_files = list(Path("logs").glob("*.json"))
            self.assertGreater(len(json_files), 0)
            
            # Vérifier le contenu JSON
            with open(json_files[-1], 'r') as f:
                content = f.read()
                # Vérifier que c'est du JSON valide
                if content.strip():
                    json.loads(content)
            
            print("✅ Logger génère du JSON valide")
        except Exception as e:
            self.fail(f"Erreur logger JSON: {e}")


class TestUnifiedMetricsIntegration(unittest.TestCase):
    """Tests d'intégration des métriques unifiées"""

    def test_01_metrics_persistence(self):
        """Test 1: Persistance des métriques"""
        try:
            from adan_trading_bot.performance.unified_metrics import UnifiedMetrics
            
            metrics = UnifiedMetrics("test_integration.db")
            
            # Ajouter des données
            for i in range(10):
                metrics.add_return(0.01 * (i % 3 - 1))
                metrics.add_portfolio_value(10000 + i * 100)
            
            # Vérifier que les données sont persistées
            import sqlite3
            conn = sqlite3.connect("test_integration.db")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM metrics")
            count = cursor.fetchone()[0]
            conn.close()
            
            self.assertGreater(count, 0)
            print(f"✅ {count} métriques persistées")
        except Exception as e:
            self.fail(f"Erreur persistance: {e}")

    def test_02_metrics_calculation(self):
        """Test 2: Calcul des métriques"""
        try:
            from adan_trading_bot.performance.unified_metrics import UnifiedMetrics
            
            metrics = UnifiedMetrics("test_integration.db")
            
            # Ajouter des returns
            test_returns = [0.01, -0.005, 0.02, 0.015, -0.01, 0.008, 0.012, 0.005, -0.003, 0.01]
            for ret in test_returns:
                metrics.add_return(ret)
            
            # Calculer les métriques
            sharpe = metrics.calculate_sharpe()
            max_dd = metrics.calculate_max_drawdown()
            total_return = metrics.calculate_total_return()
            
            # Vérifier que ce sont des nombres valides
            self.assertIsInstance(sharpe, (int, float))
            self.assertIsInstance(max_dd, (int, float))
            self.assertIsInstance(total_return, (int, float))
            
            print(f"✅ Métriques calculées: Sharpe={sharpe:.2f}, MaxDD={max_dd:.2f}, Return={total_return:.2f}")
        except Exception as e:
            self.fail(f"Erreur calcul métriques: {e}")


class TestEndToEndIntegration(unittest.TestCase):
    """Tests end-to-end complets"""

    def test_01_full_cycle(self):
        """Test 1: Cycle complet"""
        try:
            from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv
            from adan_trading_bot.common.central_logger import logger as central_logger
            
            # Créer l'environnement
            env = RealisticTradingEnv(live_mode=False)
            obs = env.reset()
            
            # Exécuter quelques steps
            for i in range(5):
                action = 1 if i % 2 == 0 else 0  # Alterner BUY et HOLD
                obs, reward, done, truncated, info = env.step(action)
                
                # Logger les résultats
                central_logger.metric(f"Step {i} Reward", reward)
            
            print("✅ Cycle complet exécuté avec succès")
        except Exception as e:
            self.fail(f"Erreur cycle complet: {e}")

    def test_02_logging_during_trading(self):
        """Test 2: Logging pendant le trading"""
        try:
            from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv
            from adan_trading_bot.common.central_logger import logger as central_logger
            
            # Créer l'environnement
            env = RealisticTradingEnv(live_mode=False)
            obs = env.reset()
            
            # Exécuter des steps
            for i in range(3):
                action = 1
                obs, reward, done, truncated, info = env.step(action)
            
            # Vérifier que les logs ont été créés
            logs_dir = Path("logs")
            log_files = list(logs_dir.glob("*.log"))
            
            self.assertGreater(len(log_files), 0)
            print(f"✅ Logs créés pendant le trading: {len(log_files)} fichiers")
        except Exception as e:
            self.fail(f"Erreur logging trading: {e}")


def run_tests():
    """Exécuter tous les tests d'intégration"""
    print("=" * 70)
    print("🔗 TESTS D'INTÉGRATION - SCRIPTS CRITIQUES")
    print("=" * 70)
    print()
    
    # Créer une suite de tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Ajouter tous les tests
    suite.addTests(loader.loadTestsFromTestCase(TestRealisticTradingEnv))
    suite.addTests(loader.loadTestsFromTestCase(TestRewardCalculatorIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestCentralLoggerIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedMetricsIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndIntegration))
    
    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Résumé
    print()
    print("=" * 70)
    print("📊 RÉSUMÉ DES TESTS D'INTÉGRATION")
    print("=" * 70)
    print(f"Tests exécutés: {result.testsRun}")
    print(f"Succès: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Échecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")
    print()
    
    if result.wasSuccessful():
        print("✅ TOUS LES TESTS D'INTÉGRATION SONT PASSÉS!")
        return 0
    else:
        print("❌ CERTAINS TESTS D'INTÉGRATION ONT ÉCHOUÉ")
        return 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
