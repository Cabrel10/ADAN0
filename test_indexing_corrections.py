#!/usr/bin/env python3
"""
Script de test pour valider les corrections d'indexation critiques.
Ce script vérifie que les erreurs EXCESSIVE_FORWARD_FILL sont corrigées.

Tests inclus:
1. Indexation correcte avec step_in_chunk
2. Gestion des limites d'index
3. Réduction des erreurs de forward-fill
4. Performance mémoire avec deque
5. Cohérence des données lues
"""

import sys
import os
import numpy as np
import pandas as pd
import unittest
from unittest.mock import Mock, patch, MagicMock
from collections import deque
import logging
import tempfile
import time
from pathlib import Path

# Ajouter le chemin du bot au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bot', 'src'))

try:
    from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
    from adan_trading_bot.performance.metrics import PerformanceMetrics
except ImportError as e:
    print(f"Erreur d'import: {e}")
    print("Assurez-vous que le bot est correctement installé")
    sys.exit(1)

# Configuration du logging pour les tests
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestIndexingCorrections(unittest.TestCase):
    """Tests des corrections d'indexation critiques."""

    def setUp(self):
        """Préparation des tests."""
        # Configuration minimale pour les tests
        self.config = {
            'data': {
                'assets': ['BTC'],
                'timeframes': ['5m'],
                'chunk_size': 1000
            },
            'trading': {
                'initial_balance': 10000,
                'performance': {
                    'enable_data_caching': False
                }
            },
            'rewards': {
                'frequency_weight': 0.1
            }
        }

        # Créer des données de test réalistes
        self.test_data = self._create_test_data()

    def _create_test_data(self):
        """Crée des données de test avec des variations de prix."""
        dates = pd.date_range('2024-01-01', periods=1000, freq='5min')

        # Prix avec variations réalistes pour BTC
        base_price = 50000
        prices = []
        current_price = base_price

        for _ in range(1000):
            # Variation aléatoire de -0.5% à +0.5%
            change = np.random.uniform(-0.005, 0.005)
            current_price *= (1 + change)
            prices.append(current_price)

        df = pd.DataFrame({
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 5000, 1000)
        }, index=dates)

        return {
            'BTC': {
                '5m': df
            }
        }

    def test_step_in_chunk_indexing(self):
        """Test que step_in_chunk est utilisé correctement pour l'indexation."""
        logger.info("🔍 Test: Indexation correcte avec step_in_chunk")

        # Mock de l'environnement avec les données de test
        env = Mock(spec=MultiAssetChunkedEnv)
        env.current_data = self.test_data
        env.step_in_chunk = 100  # Index de test dans le chunk
        env.current_step = 500   # Step global différent
        env.worker_id = 0
        env._price_forward_fill_count = 0
        env._last_known_prices = {}
        env._price_read_success_count = 0
        env._forward_fill_threshold = 0.5
        env._last_ff_check_step = 0

        # Appliquer la méthode corrigée
        result = MultiAssetChunkedEnv._get_current_prices(env)

        # Vérifications
        self.assertIn('BTC', result, "BTC devrait être dans les résultats")
        expected_price = self.test_data['BTC']['5m'].iloc[100]['close']
        actual_price = result['BTC']

        self.assertAlmostEqual(actual_price, expected_price, places=4,
                               msg=f"Prix lu ({actual_price}) != prix attendu ({expected_price}) à l'index 100")

        logger.info(f"✅ Prix correct lu: {actual_price:.4f} à l'index step_in_chunk={env.step_in_chunk}")

    def test_index_bounds_handling(self):
        """Test la gestion des limites d'index."""
        logger.info("🔍 Test: Gestion des limites d'index")

        env = Mock(spec=MultiAssetChunkedEnv)
        env.current_data = self.test_data
        env.worker_id = 0
        env.current_step = 100
        env._price_forward_fill_count = 0
        env._last_known_prices = {}

        # Test index hors limites (trop grand)
        env.step_in_chunk = 2000  # > len(data) = 1000

        result = MultiAssetChunkedEnv._get_current_prices(env)

        # Doit utiliser la dernière valeur
        expected_price = self.test_data['BTC']['5m'].iloc[-1]['close']
        actual_price = result['BTC']
        self.assertAlmostEqual(actual_price, expected_price, places=4,
                               msg="Index hors limites devrait utiliser la dernière valeur")

        # Vérifier que le compteur forward-fill a augmenté
        self.assertGreater(env._price_forward_fill_count, 0,
                           "Le compteur forward-fill devrait avoir augmenté")

        # Test index négatif
        env._price_forward_fill_count = 0
        env.current_step = 105
        env.step_in_chunk = -5
        result = MultiAssetChunkedEnv._get_current_prices(env)

        expected_price = self.test_data['BTC']['5m'].iloc[0]['close']
        actual_price = result['BTC']
        self.assertAlmostEqual(actual_price, expected_price, places=4,
                               msg="Index négatif devrait utiliser la première valeur")

        logger.info("✅ Gestion des limites d'index validée")

    def test_forward_fill_rate_calculation(self):
        """Test le calcul du taux de forward-fill."""
        logger.info("🔍 Test: Calcul du taux de forward-fill")

        env = Mock(spec=MultiAssetChunkedEnv)
        env.current_data = self.test_data
        env.worker_id = 0
        env.step_in_chunk = 100
        env.current_step = 100
        env._price_forward_fill_count = 0
        env._price_read_success_count = 0
        env._forward_fill_threshold = 0.5
        env._last_ff_check_step = 0
        env._last_known_prices = {}

        # Simuler plusieurs lectures réussies
        for i in range(20):
            env.step_in_chunk = i
            env.current_step = i + 100
            MultiAssetChunkedEnv._get_current_prices(env)

        # Le taux de forward-fill devrait être faible
        total_reads = env._price_read_success_count + env._price_forward_fill_count
        ff_rate = env._price_forward_fill_count / total_reads if total_reads > 0 else 0

        logger.info(f"Taux forward-fill: {ff_rate*100:.1f}% ({env._price_forward_fill_count}/{total_reads})")
        self.assertLess(ff_rate, 0.1, "Le taux de forward-fill devrait être < 10% avec des index valides")

        logger.info("✅ Calcul du taux de forward-fill validé")

    def test_data_progression_consistency(self):
        """Test la cohérence de la progression des données."""
        logger.info("🔍 Test: Cohérence de la progression des données")

        env = Mock(spec=MultiAssetChunkedEnv)
        env.current_data = self.test_data
        env.worker_id = 0
        env.current_step = 100
        env._price_forward_fill_count = 0
        env._price_read_success_count = 0
        env._last_known_prices = {}
        env._forward_fill_threshold = 0.5
        env._last_ff_check_step = 0

        prices = []
        for step in range(10, 20):  # 10 steps consécutifs
            env.step_in_chunk = step
            env.current_step = step + 100
            result = MultiAssetChunkedEnv._get_current_prices(env)
            prices.append(result['BTC'])

        # Vérifier que les prix correspondent aux données réelles
        expected_prices = [self.test_data['BTC']['5m'].iloc[i]['close'] for i in range(10, 20)]

        for i, (actual, expected) in enumerate(zip(prices, expected_prices)):
            self.assertAlmostEqual(actual, expected, places=4,
                                   msg=f"Prix à l'étape {i+10}: {actual} != {expected}")

        # Vérifier qu'il y a bien une progression (les prix ne sont pas tous identiques)
        unique_prices = set(prices)
        self.assertGreater(len(unique_prices), 5,
                           "Les prix devraient varier sur 10 étapes (données non statiques)")

        logger.info(f"✅ Progression cohérente validée: {len(unique_prices)} prix uniques sur 10 étapes")

class TestMemoryImprovements(unittest.TestCase):
    """Tests des améliorations mémoire avec deque."""

    def test_deque_memory_management(self):
        """Test que les deque limitent correctement la mémoire."""
        logger.info("🔍 Test: Gestion mémoire avec deque")

        # Test avec des métriques
        metrics = PerformanceMetrics(worker_id=0)

        # Vérifier que les structures utilisent deque avec maxlen
        self.assertIsInstance(metrics.returns, deque, "returns devrait être un deque")
        self.assertEqual(metrics.returns.maxlen, 10000, "returns devrait avoir maxlen=10000")

        self.assertIsInstance(metrics.equity_curve, deque, "equity_curve devrait être un deque")
        self.assertEqual(metrics.equity_curve.maxlen, 10000, "equity_curve devrait avoir maxlen=10000")

        self.assertIsInstance(metrics.trades, deque, "trades devrait être un deque")
        self.assertEqual(metrics.trades.maxlen, 5000, "trades devrait avoir maxlen=5000")

        # Test de la limitation automatique
        for i in range(15000):  # Plus que la limite
            metrics.returns.append(i * 0.001)

        self.assertEqual(len(metrics.returns), 10000,
                         "Le deque devrait être limité à 10000 éléments")

        # Vérifier que les dernières valeurs sont préservées
        self.assertEqual(metrics.returns[-1], 14.999,
                         "La dernière valeur devrait être préservée")

        logger.info("✅ Gestion mémoire avec deque validée")

    def test_sharpe_calculation_with_deque(self):
        """Test que le calcul de Sharpe fonctionne avec deque."""
        logger.info("🔍 Test: Calcul Sharpe avec deque")

        metrics = PerformanceMetrics(worker_id=0)

        # Ajouter des rendements de test
        test_returns = np.random.normal(0.001, 0.02, 1000)  # Rendements simulés
        for ret in test_returns:
            metrics.returns.append(ret)

        # Calculer Sharpe
        sharpe = metrics.calculate_sharpe_ratio()

        # Vérifications de base
        self.assertIsInstance(sharpe, float, "Sharpe devrait être un float")
        self.assertFalse(np.isnan(sharpe), "Sharpe ne devrait pas être NaN")
        self.assertFalse(np.isinf(sharpe), "Sharpe ne devrait pas être infini")
        self.assertGreaterEqual(sharpe, -10.0, "Sharpe devrait être >= -10")
        self.assertLessEqual(sharpe, 10.0, "Sharpe devrait être <= 10")

        logger.info(f"✅ Calcul Sharpe validé: {sharpe:.4f}")

def run_integration_test():
    """Test d'intégration rapide pour valider les corrections."""
    logger.info("🚀 Test d'intégration: Simulation rapide d'entraînement")

    # Configuration minimale
    config = {
        'data': {
            'assets': ['BTC'],
            'timeframes': ['5m'],
            'data_dir': 'data',
            'chunk_size': 100
        },
        'trading': {
            'initial_balance': 10000,
            'max_positions': 1
        },
        'rewards': {
            'base_reward_multiplier': 1.0,
            'frequency_weight': 0.1
        },
        'model': {
            'observation_space': {
                'shape': [3, 20, 15]
            }
        }
    }

    try:
        # Créer un environnement de test avec données fictives
        with tempfile.TemporaryDirectory() as temp_dir:
            # Créer des données fictives
            data_dir = Path(temp_dir) / 'data' / 'processed'
            data_dir.mkdir(parents=True, exist_ok=True)

            # Créer un fichier de données minimal
            dates = pd.date_range('2024-01-01', periods=200, freq='5min')
            prices = 50000 + np.cumsum(np.random.normal(0, 100, 200))

            df = pd.DataFrame({
                'open': prices,
                'high': prices * 1.01,
                'low': prices * 0.99,
                'close': prices,
                'volume': np.random.uniform(1000, 5000, 200)
            }, index=dates)

            # Sauvegarder en format parquet
            btc_file = data_dir / 'BTC_5m_train_chunk_0.parquet'
            df.to_parquet(btc_file)

            config['data']['data_dir'] = str(data_dir.parent)

            # Créer et tester l'environnement
            env = MultiAssetChunkedEnv(config, worker_id=0)

            # Test de reset
            obs, info = env.reset()
            logger.info(f"Reset réussi - observation shape: {obs['observation'].shape}")

            # Test de quelques étapes
            forward_fill_errors = 0
            successful_steps = 0

            for step in range(50):
                action = np.array([0.0], dtype=np.float32)  # Action neutre

                try:
                    obs, reward, terminated, truncated, info = env.step(action)
                    successful_steps += 1

                    # Vérifier les erreurs de forward-fill
                    if hasattr(env, '_price_forward_fill_count'):
                        current_ff_count = env._price_forward_fill_count
                        if current_ff_count > forward_fill_errors:
                            forward_fill_errors = current_ff_count

                    if step % 10 == 0:
                        logger.info(f"Step {step}: reward={reward:.4f}, ff_errors={forward_fill_errors}")

                    if terminated or truncated:
                        logger.info(f"Episode terminé à l'étape {step}")
                        break

                except Exception as e:
                    logger.error(f"Erreur à l'étape {step}: {e}")
                    break

            # Évaluation des résultats
            success_rate = successful_steps / 50 * 100
            ff_rate = forward_fill_errors / max(1, successful_steps) * 100

            logger.info(f"📊 Résultats du test d'intégration:")
            logger.info(f"  - Steps réussis: {successful_steps}/50 ({success_rate:.1f}%)")
            logger.info(f"  - Erreurs forward-fill: {forward_fill_errors} ({ff_rate:.1f}% du total)")
            logger.info(f"  - Taux forward-fill: {'✅ ACCEPTABLE' if ff_rate < 10 else '❌ EXCESSIF'}")

            # Critères de réussite
            assert success_rate >= 90, f"Taux de réussite trop faible: {success_rate:.1f}%"
            assert ff_rate < 50, f"Taux forward-fill trop élevé: {ff_rate:.1f}%"

            logger.info("✅ Test d'intégration réussi!")

    except Exception as e:
        logger.error(f"❌ Test d'intégration échoué: {e}")
        raise

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 TESTS DES CORRECTIONS D'INDEXATION CRITIQUES")
    print("=" * 80)

    # Tests unitaires
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestIndexingCorrections))
    suite.addTest(unittest.makeSuite(TestMemoryImprovements))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Test d'intégration
    if result.wasSuccessful():
        try:
            run_integration_test()
            print("\n" + "=" * 80)
            print("🎉 TOUS LES TESTS SONT RÉUSSIS!")
            print("✅ Les corrections d'indexation sont validées")
            print("✅ La gestion mémoire est améliorée")
            print("✅ Le système est prêt pour l'entraînement")
            print("=" * 80)
        except Exception as e:
            print(f"\n❌ Test d'intégration échoué: {e}")
            sys.exit(1)
    else:
        print("\n❌ Certains tests unitaires ont échoué")
        sys.exit(1)
