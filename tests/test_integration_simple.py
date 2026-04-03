#!/usr/bin/env python3
"""
🔗 TESTS D'INTÉGRATION SIMPLIFIÉS - COMPOSANTS CRITIQUES
Valider l'intégration des composants sans dépendre de la config complète
"""

import unittest
import sys
import os
from pathlib import Path
import sqlite3
import json

# Ajouter le chemin src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


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

    def test_03_reward_weights_correct(self):
        """Test 3: Vérifier les poids de récompense"""
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
            
            # Vérifier les poids
            self.assertEqual(calc.weights["pnl"], 0.25)
            self.assertEqual(calc.weights["sharpe"], 0.30)
            self.assertEqual(calc.weights["sortino"], 0.30)
            self.assertEqual(calc.weights["calmar"], 0.15)
            
            # Vérifier que la somme est 1.0
            total_weight = sum(calc.weights.values())
            self.assertAlmostEqual(total_weight, 1.0, places=2)
            
            print(f"✅ Poids de récompense corrects: {calc.weights}")
        except Exception as e:
            self.fail(f"Erreur poids: {e}")


class TestCentralLoggerIntegration(unittest.TestCase):
    """Tests d'intégration du logger centralisé"""

    def test_01_logger_creates_files(self):
        """Test 1: Logger crée les fichiers"""
        try:
            from adan_trading_bot.common.central_logger import logger as central_logger
            
            # Générer quelques logs
            central_logger.metric("test_metric_integration", 1.5)
            central_logger.trade("BUY", "BTCUSDT", 0.5, 45000, 500)
            central_logger.validation("test_integration", True, "test message")
            
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
            central_logger.metric("test_json_integration", 2.5)
            
            # Vérifier que le fichier JSON existe
            json_files = list(Path("logs").glob("*.json"))
            self.assertGreater(len(json_files), 0)
            
            # Vérifier le contenu JSON
            with open(json_files[-1], 'r') as f:
                content = f.read()
                # Vérifier que c'est du JSON valide
                if content.strip():
                    data = json.loads(content)
                    self.assertIsInstance(data, (dict, list))
            
            print("✅ Logger génère du JSON valide")
        except Exception as e:
            self.fail(f"Erreur logger JSON: {e}")

    def test_03_logger_all_methods(self):
        """Test 3: Toutes les méthodes du logger"""
        try:
            from adan_trading_bot.common.central_logger import logger as central_logger
            
            # Tester toutes les méthodes
            central_logger.trade("SELL", "ETHUSDT", 1.0, 2500, -100)
            central_logger.metric("test_metric_all", 3.5)
            central_logger.validation("test_all", False, "test failed")
            central_logger.sync("test_component", "running", {"data": "test"})
            
            print("✅ Toutes les méthodes du logger fonctionnent")
        except Exception as e:
            self.fail(f"Erreur méthodes logger: {e}")


class TestUnifiedMetricsIntegration(unittest.TestCase):
    """Tests d'intégration des métriques unifiées"""

    def test_01_metrics_persistence(self):
        """Test 1: Persistance des métriques"""
        try:
            from adan_trading_bot.performance.unified_metrics import UnifiedMetrics
            
            metrics = UnifiedMetrics("test_integration_metrics.db")
            
            # Ajouter des données
            for i in range(10):
                metrics.add_return(0.01 * (i % 3 - 1))
                metrics.add_portfolio_value(10000 + i * 100)
            
            # Vérifier que les données sont persistées
            conn = sqlite3.connect("test_integration_metrics.db")
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
            
            metrics = UnifiedMetrics("test_integration_metrics.db")
            
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

    def test_03_metrics_add_trade(self):
        """Test 3: Ajouter un trade aux métriques"""
        try:
            from adan_trading_bot.performance.unified_metrics import UnifiedMetrics
            
            metrics = UnifiedMetrics("test_integration_metrics.db")
            
            # Ajouter un trade
            metrics.add_trade(
                action="BUY",
                symbol="BTCUSDT",
                quantity=0.5,
                price=45000,
                pnl=500
            )
            
            print("✅ Trade ajouté aux métriques")
        except Exception as e:
            self.fail(f"Erreur ajout trade: {e}")


class TestRiskManagerIntegration(unittest.TestCase):
    """Tests d'intégration du RiskManager"""

    def test_01_risk_manager_validation(self):
        """Test 1: Validation de trades par RiskManager"""
        try:
            from adan_trading_bot.risk_management.risk_manager import RiskManager
            
            config = {
                'max_daily_drawdown': 0.15,
                'max_position_risk': 0.02,
                'max_portfolio_risk': 0.10,
                'initial_capital': 10000
            }
            
            risk_manager = RiskManager(config)
            
            # Tester plusieurs trades
            trades = [
                {"portfolio_value": 10000, "position_size": 100, "entry_price": 50000, "stop_loss": 49000},
                {"portfolio_value": 10000, "position_size": 200, "entry_price": 50000, "stop_loss": 48000},
                {"portfolio_value": 10000, "position_size": 500, "entry_price": 50000, "stop_loss": 45000},
            ]
            
            results = []
            for trade in trades:
                is_valid = risk_manager.validate_trade(**trade)
                results.append(is_valid)
            
            # Au moins un devrait être valide
            self.assertTrue(any(results))
            print(f"✅ RiskManager a validé {sum(results)}/{len(results)} trades")
        except Exception as e:
            self.fail(f"Erreur validation RiskManager: {e}")

    def test_02_risk_manager_drawdown_tracking(self):
        """Test 2: Suivi du drawdown par RiskManager"""
        try:
            from adan_trading_bot.risk_management.risk_manager import RiskManager
            
            config = {
                'max_daily_drawdown': 0.15,
                'max_position_risk': 0.02,
                'max_portfolio_risk': 0.10,
                'initial_capital': 10000
            }
            
            risk_manager = RiskManager(config)
            
            # Simuler une évolution du portefeuille
            values = [10000, 10500, 10200, 9800, 9500, 10100]
            
            for value in values:
                risk_manager.update_peak(value)
            
            # Vérifier que le peak a été mis à jour
            self.assertEqual(risk_manager.portfolio_peak, 10500)
            print(f"✅ Peak tracking fonctionne: {risk_manager.portfolio_peak}")
        except Exception as e:
            self.fail(f"Erreur drawdown tracking: {e}")


class TestSystemIntegration(unittest.TestCase):
    """Tests d'intégration du système complet"""

    def test_01_all_components_available(self):
        """Test 1: Tous les composants sont disponibles"""
        try:
            from adan_trading_bot.common.central_logger import logger as central_logger
            from adan_trading_bot.performance.unified_metrics import UnifiedMetrics
            from adan_trading_bot.performance.unified_metrics_db import UnifiedMetricsDB
            from adan_trading_bot.risk_management.risk_manager import RiskManager
            from adan_trading_bot.environment.reward_calculator import RewardCalculator
            
            # Vérifier que tous les composants sont importables
            self.assertIsNotNone(central_logger)
            self.assertIsNotNone(UnifiedMetrics)
            self.assertIsNotNone(UnifiedMetricsDB)
            self.assertIsNotNone(RiskManager)
            self.assertIsNotNone(RewardCalculator)
            
            print("✅ Tous les composants du système unifié sont disponibles")
        except Exception as e:
            self.fail(f"Erreur composants: {e}")

    def test_02_system_unified_logging(self):
        """Test 2: Logging unifié du système"""
        try:
            from adan_trading_bot.common.central_logger import logger as central_logger
            from adan_trading_bot.performance.unified_metrics import UnifiedMetrics
            
            # Créer les composants
            metrics = UnifiedMetrics("test_system_integration.db")
            
            # Simuler un cycle complet
            central_logger.sync("System Integration Test", "started", {})
            
            # Ajouter des données
            metrics.add_return(0.01)
            metrics.add_portfolio_value(10100)
            
            # Logger les résultats
            central_logger.metric("Integration Test Sharpe", metrics.calculate_sharpe())
            central_logger.validation("System Integration", True, "All components working")
            
            central_logger.sync("System Integration Test", "completed", {})
            
            print("✅ Logging unifié du système fonctionne")
        except Exception as e:
            self.fail(f"Erreur logging unifié: {e}")


def run_tests():
    """Exécuter tous les tests d'intégration"""
    print("=" * 70)
    print("🔗 TESTS D'INTÉGRATION - COMPOSANTS CRITIQUES")
    print("=" * 70)
    print()
    
    # Créer une suite de tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Ajouter tous les tests
    suite.addTests(loader.loadTestsFromTestCase(TestRewardCalculatorIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestCentralLoggerIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedMetricsIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskManagerIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestSystemIntegration))
    
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
