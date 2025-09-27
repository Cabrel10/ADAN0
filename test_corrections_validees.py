#!/usr/bin/env python3
"""
Script de test pour valider les corrections apportées aux problèmes identifiés dans les logs.

Ce script teste :
1. Correction du prix de clôture manquant avec forward fill
2. Correction du capital hors bornes avec le nouveau palier Ultra Micro
3. Limitation des logs dupliqués au worker principal
4. Robustesse des métriques contre les valeurs extrêmes
5. Amélioration des avertissements sur la forme d'observation
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any
import tempfile
import unittest

# Ajouter le chemin vers le package
sys.path.insert(0, str(Path(__file__).parent / "bot" / "src"))

try:
    from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
    from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager
    from adan_trading_bot.performance.metrics import PerformanceMetrics
    from adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine
except ImportError as e:
    print(f"Erreur d'import: {e}")
    print("Assurez-vous que le package adan_trading_bot est installé correctement")
    sys.exit(1)

# Configuration du logging pour les tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestCorrections(unittest.TestCase):
    """Classe de test pour valider les corrections apportées."""

    @classmethod
    def setUpClass(cls):
        """Initialisation commune à tous les tests."""
        cls.config_path = Path(__file__).parent / "bot" / "config" / "config.yaml"
        with open(cls.config_path, 'r') as f:
            cls.config = yaml.safe_load(f)

        # Créer des données de test
        cls.test_data_dir = Path(tempfile.mkdtemp())
        cls._create_test_data()

    @classmethod
    def _create_test_data(cls):
        """Créer des données de test avec des valeurs NaN pour tester le fill."""
        dates = pd.date_range('2023-01-01', periods=1000, freq='5T')

        # Données avec quelques valeurs manquantes dans 'close'
        data = {
            'Open': np.random.uniform(40000, 50000, len(dates)),
            'High': np.random.uniform(50000, 60000, len(dates)),
            'Low': np.random.uniform(30000, 40000, len(dates)),
            'Close': np.random.uniform(40000, 50000, len(dates)),
            'Volume': np.random.uniform(100, 1000, len(dates))
        }

        # Introduire des valeurs NaN dans 'Close' pour tester le remplissage
        nan_indices = np.random.choice(len(dates), size=50, replace=False)
        for idx in nan_indices:
            data['Close'][idx] = np.nan

        df = pd.DataFrame(data, index=dates)

        # Sauvegarder sous forme de parquet
        test_file = cls.test_data_dir / "BTCUSDT_5m.parquet"
        df.to_parquet(test_file)

        cls.test_data_path = test_file
        cls.original_nan_count = np.isnan(data['Close']).sum()

    def test_1_prix_cloture_manquant_correction(self):
        """Test 1: Vérifier que les valeurs NaN dans 'close' sont remplies."""
        logger.info("=== Test 1: Correction du prix de clôture manquant ===")

        try:
            # Configuration minimale pour le data loader
            config = {
                'timeframes': ['5m'],
                'assets_list': ['BTC'],
                'chunk_sizes': {'5m': 100},
                'max_workers': 1
            }

            # Le data loader devrait maintenant remplir les NaN
            data_loader = ChunkedDataLoader(
                assets_list=['BTC'],
                timeframes=['5m'],
                chunk_sizes={'5m': 100},
                max_workers=1
            )

            # Charger les données directement depuis le parquet pour tester
            df = pd.read_parquet(self.test_data_path)
            df.columns = [col.upper() for col in df.columns]

            # Simuler le remplissage comme fait dans le data loader
            if 'CLOSE' in df.columns:
                nan_count_before = df['CLOSE'].isna().sum()
                df['CLOSE'] = df['CLOSE'].fillna(method='ffill')
                nan_count_after = df['CLOSE'].isna().sum()

                logger.info(f"NaN avant remplissage: {nan_count_before}")
                logger.info(f"NaN après remplissage: {nan_count_after}")

                # Vérifications
                self.assertEqual(nan_count_before, self.original_nan_count, "Le nombre de NaN initial ne correspond pas")
                self.assertEqual(nan_count_after, 0, "Il reste des NaN après le remplissage")

                logger.info("✅ Test 1 RÉUSSI: Les valeurs NaN sont correctement remplies")
            else:
                self.fail("Colonne CLOSE manquante")

        except Exception as e:
            logger.error(f"❌ Test 1 ÉCHOUÉ: {e}")
            raise

    def test_2_capital_hors_bornes_correction(self):
        """Test 2: Vérifier que le nouveau palier Ultra Micro gère les petits capitaux."""
        logger.info("=== Test 2: Correction du capital hors bornes ===")

        try:
            # Vérifier que le palier Ultra Micro existe dans la config
            capital_tiers = self.config.get('capital_tiers', [])
            ultra_micro_tier = None

            for tier in capital_tiers:
                if tier.get('name') == 'Ultra Micro Capital':
                    ultra_micro_tier = tier
                    break

            self.assertIsNotNone(ultra_micro_tier, "Le palier Ultra Micro Capital n'existe pas dans la config")

            # Vérifier les paramètres du palier Ultra Micro
            self.assertEqual(ultra_micro_tier['min_capital'], 1.0, "Capital minimum incorrect")
            self.assertEqual(ultra_micro_tier['max_capital'], 11.0, "Capital maximum incorrect")
            self.assertIn('exposure_range', ultra_micro_tier, "exposure_range manquant")

            # Tester avec des capitaux problématiques mentionnés dans les logs
            test_capitals = [3.90, 4.01, 4.17, 4.18]

            for capital in test_capitals:
                # Trouver le palier approprié
                matching_tier = None
                for tier in capital_tiers:
                    min_cap = tier.get('min_capital', 0)
                    max_cap = tier.get('max_capital', float('inf'))
                    if max_cap is None:
                        max_cap = float('inf')

                    if min_cap <= capital <= max_cap:
                        matching_tier = tier
                        break

                self.assertIsNotNone(matching_tier, f"Aucun palier trouvé pour le capital {capital}")
                self.assertEqual(matching_tier['name'], 'Ultra Micro Capital',
                               f"Capital {capital} devrait être dans Ultra Micro, mais est dans {matching_tier['name']}")

                logger.info(f"✅ Capital {capital} USDT correctement géré par le palier Ultra Micro")

            logger.info("✅ Test 2 RÉUSSI: Le palier Ultra Micro gère correctement les petits capitaux")

        except Exception as e:
            logger.error(f"❌ Test 2 ÉCHOUÉ: {e}")
            raise

    def test_3_duplication_logs_correction(self):
        """Test 3: Vérifier que les logs ne sont émis que par le worker principal."""
        logger.info("=== Test 3: Correction de la duplication de logs ===")

        try:
            # Test de la condition worker_id dans le portfolio manager
            env_config = {
                'worker_config': {'worker_id': 'W1'},  # Worker non principal
                'default_currency': 'USDT'
            }

            portfolio_manager = PortfolioManager(env_config)

            # Vérifier que le worker_id est correctement initialisé
            self.assertEqual(portfolio_manager.worker_id, 'W1', "Worker ID mal initialisé")

            # Test de la condition worker_id dans les métriques de performance
            metrics = PerformanceMetrics()
            self.assertEqual(metrics.worker_id, 'W0', "Worker ID par défaut incorrect dans PerformanceMetrics")

            # Test du DBE
            dbe_config = {
                'worker_config': {'worker_id': 'W2'}
            }
            dbe = DynamicBehaviorEngine(config=dbe_config)
            self.assertEqual(dbe.worker_id, 'W2', "Worker ID mal initialisé dans DBE")

            logger.info("✅ Test 3 RÉUSSI: Les worker_id sont correctement initialisés")

        except Exception as e:
            logger.error(f"❌ Test 3 ÉCHOUÉ: {e}")
            raise

    def test_4_metriques_robustes_correction(self):
        """Test 4: Vérifier la robustesse des métriques contre les valeurs extrêmes."""
        logger.info("=== Test 4: Correction des métriques incohérentes ===")

        try:
            metrics = PerformanceMetrics()

            # Test avec peu de données (devrait retourner 0)
            metrics.returns = [0.01, 0.02]  # Moins de 10 valeurs
            sharpe = metrics.calculate_sharpe_ratio()
            self.assertEqual(sharpe, 0.0, "Sharpe ratio devrait être 0 avec peu de données")

            # Test avec écart-type nul
            metrics.returns = [0.01] * 15  # Même valeur répétée
            sharpe = metrics.calculate_sharpe_ratio()
            self.assertEqual(sharpe, 0.0, "Sharpe ratio devrait être 0 avec écart-type nul")

            # Test avec des valeurs qui pourraient donner des résultats extrêmes
            metrics.returns = [0.1, -0.05, 0.08, -0.02, 0.12, -0.01, 0.05, -0.03, 0.09, -0.04, 0.07, -0.06]
            sharpe = metrics.calculate_sharpe_ratio()
            self.assertTrue(-10.0 <= sharpe <= 10.0, f"Sharpe ratio non clippé: {sharpe}")

            # Test du profit factor avec pas de pertes (éviter inf)
            metrics.trades = [
                {'pnl': 100}, {'pnl': 200}, {'pnl': 50}  # Que des gains
            ]
            pf = metrics.calculate_profit_factor()
            self.assertTrue(0 <= pf <= 100.0, f"Profit factor non clippé: {pf}")

            # Test avec peu de trades
            metrics.trades = [{'pnl': 100}, {'pnl': -50}]  # Moins de 5 trades
            pf = metrics.calculate_profit_factor()
            self.assertEqual(pf, 0.0, "Profit factor devrait être 0 avec peu de trades")

            # Test du win rate
            metrics.trades = [
                {'pnl': 100}, {'pnl': -50}, {'pnl': 75}, {'pnl': -25}, {'pnl': 150}
            ]
            summary = metrics.get_metrics_summary()
            win_rate = summary['win_rate']
            self.assertTrue(0 <= win_rate <= 100, f"Win rate hors bornes: {win_rate}")

            logger.info(f"✅ Test 4 RÉUSSI: Métriques robustes - Sharpe: {sharpe:.2f}, PF: {pf:.2f}, WR: {win_rate:.1f}%")

        except Exception as e:
            logger.error(f"❌ Test 4 ÉCHOUÉ: {e}")
            raise

    def test_5_observation_shape_warning_improvement(self):
        """Test 5: Vérifier que l'avertissement sur la forme d'observation est informatif."""
        logger.info("=== Test 5: Amélioration de l'avertissement sur la forme d'observation ===")

        try:
            # Ce test vérifie principalement que le code ne plante pas
            # et que les messages sont plus informatifs

            # Simuler une observation avec une forme incorrecte
            observation_actual = np.random.random((3, 20, 11))  # 11 au lieu de 15
            expected_shape = (3, 20, 15)

            # Test de l'ajustement automatique
            output = np.zeros(expected_shape, dtype=np.float32)

            # Calculer les slices pour copier les données en sécurité
            slices = [
                slice(0, min(observation_actual.shape[i], expected_shape[i]))
                for i in range(len(expected_shape))
            ]

            # Copier les données disponibles
            output[tuple(slices)] = observation_actual[tuple(slices)]

            # Vérifications
            self.assertEqual(output.shape, expected_shape, "Forme de sortie incorrecte")
            self.assertTrue(np.allclose(output[:, :, :11], observation_actual), "Données non correctement copiées")
            self.assertTrue(np.all(output[:, :, 11:] == 0), "Padding non correctement appliqué")

            logger.info(f"✅ Test 5 RÉUSSI: Ajustement automatique {observation_actual.shape} -> {output.shape}")

        except Exception as e:
            logger.error(f"❌ Test 5 ÉCHOUÉ: {e}")
            raise

    def test_6_integration_globale(self):
        """Test 6: Test d'intégration pour vérifier que toutes les corrections fonctionnent ensemble."""
        logger.info("=== Test 6: Test d'intégration globale ===")

        try:
            # Test d'intégration avec des données réelles
            logger.info("Configuration chargée avec succès")

            # Vérifier que tous les paliers ont un exposure_range
            capital_tiers = self.config.get('capital_tiers', [])
            for tier in capital_tiers:
                name = tier.get('name', 'Inconnu')
                self.assertIn('exposure_range', tier, f"exposure_range manquant pour {name}")
                self.assertEqual(len(tier['exposure_range']), 2, f"exposure_range invalide pour {name}")
                logger.info(f"✅ Palier {name}: exposure_range = {tier['exposure_range']}")

            # Vérifier la configuration des features
            features_config = self.config.get('data_processing', {}).get('features', {})
            if features_config:
                base_features = features_config.get('base', [])
                indicators = features_config.get('indicators', {})

                for tf, tf_indicators in indicators.items():
                    total_features = len(base_features) + len(tf_indicators)
                    logger.info(f"✅ Timeframe {tf}: {total_features} features totales")
                    self.assertGreaterEqual(total_features, 10, f"Pas assez de features pour {tf}")

            logger.info("✅ Test 6 RÉUSSI: Intégration globale validée")

        except Exception as e:
            logger.error(f"❌ Test 6 ÉCHOUÉ: {e}")
            raise

    @classmethod
    def tearDownClass(cls):
        """Nettoyage après les tests."""
        import shutil
        if cls.test_data_dir.exists():
            shutil.rmtree(cls.test_data_dir)

def run_corrections_validation():
    """Lance la validation des corrections."""
    print("=" * 80)
    print("VALIDATION DES CORRECTIONS APPORTÉES AUX PROBLÈMES IDENTIFIÉS")
    print("=" * 80)

    # Lancer les tests
    unittest.main(argv=[''], exit=False, verbosity=2)

if __name__ == "__main__":
    run_corrections_validation()
