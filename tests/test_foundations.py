#!/usr/bin/env python3
"""
🧪 TESTS UNITAIRES - FONDATIONS CRITIQUES
Valider que les composants de base fonctionnent correctement
"""

import unittest
import sqlite3
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Ajouter le chemin src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np


class TestCentralLogger(unittest.TestCase):
    """Tests du logger centralisé"""

    def test_01_logger_import(self):
        """Test 1: Import du logger centralisé"""
        try:
            from adan_trading_bot.common.central_logger import logger as central_logger
            self.assertIsNotNone(central_logger)
            print("✅ Logger centralisé importé avec succès")
        except Exception as e:
            self.fail(f"Erreur import logger: {e}")

    def test_02_logger_trade_method(self):
        """Test 2: Méthode trade() du logger"""
        try:
            from adan_trading_bot.common.central_logger import logger as central_logger
            
            # Tester la méthode trade (avec la bonne signature)
            central_logger.trade(
                action="BUY",
                symbol="BTCUSDT",
                quantity=0.5,
                price=45000,
                pnl=500
            )
            print("✅ Méthode trade() fonctionne")
        except Exception as e:
            self.fail(f"Erreur méthode trade: {e}")

    def test_03_logger_metric_method(self):
        """Test 3: Méthode metric() du logger"""
        try:
            from adan_trading_bot.common.central_logger import logger as central_logger
            
            # Tester la méthode metric
            central_logger.metric("Sharpe Ratio", 1.85)
            central_logger.metric("Max Drawdown", -0.15)
            print("✅ Méthode metric() fonctionne")
        except Exception as e:
            self.fail(f"Erreur méthode metric: {e}")

    def test_04_logger_validation_method(self):
        """Test 4: Méthode validation() du logger"""
        try:
            from adan_trading_bot.common.central_logger import logger as central_logger
            
            # Tester la méthode validation
            central_logger.validation(
                "Risk Check",
                True,
                "Trade validé par RiskManager"
            )
            print("✅ Méthode validation() fonctionne")
        except Exception as e:
            self.fail(f"Erreur méthode validation: {e}")

    def test_05_logger_sync_method(self):
        """Test 5: Méthode sync() du logger"""
        try:
            from adan_trading_bot.common.central_logger import logger as central_logger
            
            # Tester la méthode sync
            central_logger.sync(
                component="Test Component",
                status="initialized",
                details={"test": "data"}
            )
            print("✅ Méthode sync() fonctionne")
        except Exception as e:
            self.fail(f"Erreur méthode sync: {e}")


class TestUnifiedMetricsDB(unittest.TestCase):
    """Tests de la base de données unifiée"""

    def test_01_db_import(self):
        """Test 1: Import de la base de données"""
        try:
            from adan_trading_bot.performance.unified_metrics_db import UnifiedMetricsDB
            self.assertIsNotNone(UnifiedMetricsDB)
            print("✅ UnifiedMetricsDB importé avec succès")
        except Exception as e:
            self.fail(f"Erreur import DB: {e}")

    def test_02_db_creation(self):
        """Test 2: Création de la base de données"""
        try:
            from adan_trading_bot.performance.unified_metrics_db import UnifiedMetricsDB
            db = UnifiedMetricsDB("test_foundations.db")
            self.assertIsNotNone(db)
            print("✅ Base de données créée avec succès")
        except Exception as e:
            self.fail(f"Erreur création DB: {e}")

    def test_03_db_tables_exist(self):
        """Test 3: Vérifier que les tables existent"""
        try:
            from adan_trading_bot.performance.unified_metrics_db import UnifiedMetricsDB
            db = UnifiedMetricsDB("test_foundations.db")
            
            # Vérifier les tables
            conn = sqlite3.connect("test_foundations.db")
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            required_tables = ['metrics', 'trades', 'validations', 'synchronizations']
            for table in required_tables:
                self.assertIn(table, tables, f"Table {table} manquante")
            
            print(f"✅ Toutes les tables existent: {tables}")
        except Exception as e:
            self.fail(f"Erreur vérification tables: {e}")

    def test_04_db_insert_metric(self):
        """Test 4: Insérer une métrique"""
        try:
            from adan_trading_bot.performance.unified_metrics_db import UnifiedMetricsDB
            db = UnifiedMetricsDB("test_foundations.db")
            
            # Insérer une métrique (utiliser la méthode correcte)
            db.add_metric("test_sharpe", 1.85, "test_source")
            
            # Vérifier l'insertion
            conn = sqlite3.connect("test_foundations.db")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM metrics WHERE metric_name = 'test_sharpe'")
            count = cursor.fetchone()[0]
            conn.close()
            
            self.assertGreater(count, 0)
            print("✅ Métrique insérée avec succès")
        except Exception as e:
            self.fail(f"Erreur insertion métrique: {e}")

    def test_05_db_insert_trade(self):
        """Test 5: Insérer un trade"""
        try:
            from adan_trading_bot.performance.unified_metrics_db import UnifiedMetricsDB
            db = UnifiedMetricsDB("test_foundations.db")
            
            # Insérer un trade (utiliser la méthode correcte)
            db.add_trade(
                action="BUY",
                symbol="BTCUSDT",
                quantity=0.5,
                price=45000,
                pnl=500
            )
            
            # Vérifier l'insertion
            conn = sqlite3.connect("test_foundations.db")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trades WHERE symbol = 'BTCUSDT'")
            count = cursor.fetchone()[0]
            conn.close()
            
            self.assertGreater(count, 0)
            print("✅ Trade inséré avec succès")
        except Exception as e:
            self.fail(f"Erreur insertion trade: {e}")


class TestUnifiedMetrics(unittest.TestCase):
    """Tests du calculateur de métriques unifié"""

    def test_01_metrics_import(self):
        """Test 1: Import du calculateur de métriques"""
        try:
            from adan_trading_bot.performance.unified_metrics import UnifiedMetrics
            self.assertIsNotNone(UnifiedMetrics)
            print("✅ UnifiedMetrics importé avec succès")
        except Exception as e:
            self.fail(f"Erreur import metrics: {e}")

    def test_02_metrics_creation(self):
        """Test 2: Création du calculateur"""
        try:
            from adan_trading_bot.performance.unified_metrics import UnifiedMetrics
            metrics = UnifiedMetrics("test_metrics.db")
            self.assertIsNotNone(metrics)
            print("✅ Calculateur de métriques créé avec succès")
        except Exception as e:
            self.fail(f"Erreur création metrics: {e}")

    def test_03_add_return(self):
        """Test 3: Ajouter un return"""
        try:
            from adan_trading_bot.performance.unified_metrics import UnifiedMetrics
            metrics = UnifiedMetrics("test_metrics.db")
            
            # Ajouter des returns
            test_returns = [0.01, -0.005, 0.02, 0.015, -0.01]
            for ret in test_returns:
                metrics.add_return(ret)
            
            print("✅ Returns ajoutés avec succès")
        except Exception as e:
            self.fail(f"Erreur ajout returns: {e}")

    def test_04_add_portfolio_value(self):
        """Test 4: Ajouter une valeur de portefeuille"""
        try:
            from adan_trading_bot.performance.unified_metrics import UnifiedMetrics
            metrics = UnifiedMetrics("test_metrics.db")
            
            # Ajouter des valeurs de portefeuille
            metrics.add_portfolio_value(10000)
            metrics.add_portfolio_value(10100)
            metrics.add_portfolio_value(10050)
            
            print("✅ Valeurs de portefeuille ajoutées avec succès")
        except Exception as e:
            self.fail(f"Erreur ajout portfolio values: {e}")

    def test_05_calculate_sharpe(self):
        """Test 5: Calculer le Sharpe Ratio"""
        try:
            from adan_trading_bot.performance.unified_metrics import UnifiedMetrics
            metrics = UnifiedMetrics("test_metrics.db")
            
            # Ajouter des returns
            test_returns = [0.01, -0.005, 0.02, 0.015, -0.01, 0.008, 0.012]
            for ret in test_returns:
                metrics.add_return(ret)
            
            # Calculer Sharpe
            sharpe = metrics.calculate_sharpe()
            
            # Vérifier que c'est un nombre valide
            self.assertIsInstance(sharpe, (int, float))
            self.assertFalse(np.isnan(sharpe))
            self.assertFalse(np.isinf(sharpe))
            
            print(f"✅ Sharpe Ratio calculé: {sharpe:.2f}")
        except Exception as e:
            self.fail(f"Erreur calcul Sharpe: {e}")


class TestRiskManager(unittest.TestCase):
    """Tests du RiskManager"""

    def test_01_risk_manager_import(self):
        """Test 1: Import du RiskManager"""
        try:
            from adan_trading_bot.risk_management.risk_manager import RiskManager
            self.assertIsNotNone(RiskManager)
            print("✅ RiskManager importé avec succès")
        except Exception as e:
            self.fail(f"Erreur import RiskManager: {e}")

    def test_02_risk_manager_creation(self):
        """Test 2: Création du RiskManager"""
        try:
            from adan_trading_bot.risk_management.risk_manager import RiskManager
            
            config = {
                'max_daily_drawdown': 0.15,
                'max_position_risk': 0.02,
                'max_portfolio_risk': 0.10,
                'initial_capital': 10000
            }
            
            risk_manager = RiskManager(config)
            self.assertIsNotNone(risk_manager)
            print("✅ RiskManager créé avec succès")
        except Exception as e:
            self.fail(f"Erreur création RiskManager: {e}")

    def test_03_validate_trade(self):
        """Test 3: Valider un trade"""
        try:
            from adan_trading_bot.risk_management.risk_manager import RiskManager
            
            config = {
                'max_daily_drawdown': 0.15,
                'max_position_risk': 0.02,
                'max_portfolio_risk': 0.10,
                'initial_capital': 10000
            }
            
            risk_manager = RiskManager(config)
            
            # Valider un trade
            is_valid = risk_manager.validate_trade(
                portfolio_value=10000,
                position_size=100,
                entry_price=50000,
                stop_loss=49000
            )
            
            self.assertIsInstance(is_valid, bool)
            print(f"✅ Trade validé: {is_valid}")
        except Exception as e:
            self.fail(f"Erreur validation trade: {e}")

    def test_04_update_peak(self):
        """Test 4: Mettre à jour le peak"""
        try:
            from adan_trading_bot.risk_management.risk_manager import RiskManager
            
            config = {
                'max_daily_drawdown': 0.15,
                'max_position_risk': 0.02,
                'max_portfolio_risk': 0.10,
                'initial_capital': 10000
            }
            
            risk_manager = RiskManager(config)
            
            # Mettre à jour le peak
            risk_manager.update_peak(11000)
            
            self.assertEqual(risk_manager.portfolio_peak, 11000)
            print(f"✅ Peak mis à jour: {risk_manager.portfolio_peak}")
        except Exception as e:
            self.fail(f"Erreur update peak: {e}")


class TestRewardCalculator(unittest.TestCase):
    """Tests du calculateur de récompense"""

    def test_01_reward_calculator_import(self):
        """Test 1: Import du calculateur de récompense"""
        try:
            from adan_trading_bot.environment.reward_calculator import RewardCalculator
            self.assertIsNotNone(RewardCalculator)
            print("✅ RewardCalculator importé avec succès")
        except Exception as e:
            self.fail(f"Erreur import RewardCalculator: {e}")

    def test_02_reward_weights_balanced(self):
        """Test 2: Vérifier que les poids sont équilibrés"""
        try:
            with open('src/adan_trading_bot/environment/reward_calculator.py', 'r') as f:
                content = f.read()
            
            # Vérifier les poids
            self.assertIn('"pnl": 0.25', content)
            self.assertIn('"sharpe": 0.30', content)
            self.assertIn('"sortino": 0.30', content)
            self.assertIn('"calmar": 0.15', content)
            
            print("✅ Poids de récompense équilibrés")
        except Exception as e:
            self.fail(f"Erreur vérification poids: {e}")

    def test_03_unified_system_integration(self):
        """Test 3: Vérifier l'intégration du système unifié"""
        try:
            with open('src/adan_trading_bot/environment/reward_calculator.py', 'r') as f:
                content = f.read()
            
            # Vérifier les imports
            self.assertIn('from ..common.central_logger import logger as central_logger', content)
            self.assertIn('from ..performance.unified_metrics import UnifiedMetrics', content)
            
            # Vérifier le logging
            self.assertIn('central_logger.metric("Reward Final"', content)
            self.assertIn('self.unified_metrics.add_return', content)
            
            print("✅ Système unifié intégré dans RewardCalculator")
        except Exception as e:
            self.fail(f"Erreur vérification intégration: {e}")


def run_tests():
    """Exécuter tous les tests"""
    print("=" * 70)
    print("🧪 TESTS UNITAIRES - FONDATIONS CRITIQUES")
    print("=" * 70)
    print()
    
    # Créer une suite de tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Ajouter tous les tests
    suite.addTests(loader.loadTestsFromTestCase(TestCentralLogger))
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedMetricsDB))
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskManager))
    suite.addTests(loader.loadTestsFromTestCase(TestRewardCalculator))
    
    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Résumé
    print()
    print("=" * 70)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 70)
    print(f"Tests exécutés: {result.testsRun}")
    print(f"Succès: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Échecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")
    print()
    
    if result.wasSuccessful():
        print("✅ TOUS LES TESTS SONT PASSÉS!")
        return 0
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        return 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
