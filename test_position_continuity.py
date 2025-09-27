#!/usr/bin/env python3
"""
Script de test pour vérifier la préservation des positions.

Ce script teste les corrections apportées pour résoudre le problème où les positions
étaient fermées prématurément à cause de la recréation du PortfolioManager.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Ajouter le chemin du projet
sys.path.insert(0, '/home/morningstar/Documents/trading')
sys.path.insert(0, '/home/morningstar/Documents/trading/bot/src')

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager, Position
# from adan_trading_bot.common.data_loader import DataLoader  # Not needed for tests

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PositionContinuityTester:
    """Testeur pour vérifier la continuité des positions."""

    def __init__(self):
        self.config_path = "/home/morningstar/Documents/trading/bot/config/config.yaml"
        self.config = self.load_config()
        self.results = {
            'portfolio_manager_recreation': False,
            'position_preservation': False,
            'soft_reset_preservation': False,
            'position_ids_stable': False,
            'tests_passed': 0,
            'total_tests': 4
        }

    def load_config(self) -> Dict[str, Any]:
        """Charger la configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la config: {e}")
            return self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        """Configuration par défaut pour les tests."""
        return {
            'environment': {
                'initial_balance': 20.0,
                'min_capital_before_reset': 11.0,
                'max_steps': 1000,
                'observation': {
                    'window_sizes': {'5m': 20, '1h': 10, '4h': 5}
                }
            },
            'portfolio': {
                'initial_balance': 20.0
            },
            'trading_rules': {},
            'dbe': {},
            'min_order_value_usdt': 11.0,
            'trading_fees': 0.001
        }

    def create_sample_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Créer des données de test."""
        timeframes = ['5m', '1h', '4h']
        assets = ['BTCUSDT', 'ETHUSDT']

        # Générer 500 points de données
        dates = pd.date_range(start='2024-01-01', periods=500, freq='5T')

        data = {}
        for asset in assets:
            data[asset] = {}
            base_price = 50000 if 'BTC' in asset else 3000

            for tf in timeframes:
                # Générer des données OHLCV réalistes
                np.random.seed(42)
                prices = []
                current_price = base_price

                for i in range(len(dates)):
                    variation = np.random.normal(0, 0.02)  # 2% de volatilité
                    current_price = max(current_price * (1 + variation), 1000)
                    prices.append(current_price)

                df = pd.DataFrame({
                    'timestamp': dates,
                    'open': prices,
                    'high': [p * 1.01 for p in prices],
                    'low': [p * 0.99 for p in prices],
                    'close': prices,
                    'volume': [100 + np.random.randint(0, 50) for _ in prices]
                })
                df.set_index('timestamp', inplace=True)
                data[asset][tf] = df

        return data

    def test_portfolio_manager_recreation(self) -> bool:
        """Test 1: Vérifier que le PortfolioManager n'est pas recréé."""
        logger.info("=== TEST 1: PortfolioManager Recreation ===")

        try:
            # Créer l'environnement
            data = self.create_sample_data()
            timeframes = ['5m', '1h', '4h']
            features_config = {tf: ['open', 'high', 'low', 'close', 'volume'] for tf in timeframes}

            env = MultiAssetChunkedEnv(
                data=data,
                timeframes=timeframes,
                window_size=20,
                features_config=features_config,
                config=self.config
            )

            # Enregistrer l'ID initial du PortfolioManager
            initial_id = id(env.portfolio_manager)
            logger.info(f"ID initial du PortfolioManager: {initial_id}")

            # Simuler plusieurs steps
            env.reset()
            for i in range(5):
                action = np.array([0.1, -0.1])  # Actions simples
                obs, reward, done, truncated, info = env.step(action)
                current_id = id(env.portfolio_manager)

                if current_id != initial_id:
                    logger.error(f"PortfolioManager recréé au step {i}: {initial_id} -> {current_id}")
                    return False

            # Tester un reset de l'environnement
            env.reset()
            after_reset_id = id(env.portfolio_manager)

            if after_reset_id != initial_id:
                logger.error(f"PortfolioManager recréé après reset: {initial_id} -> {after_reset_id}")
                return False

            logger.info("✅ TEST 1 RÉUSSI: PortfolioManager préservé")
            return True

        except Exception as e:
            logger.error(f"❌ TEST 1 ÉCHOUÉ: {e}")
            return False

    def test_position_preservation(self) -> bool:
        """Test 2: Vérifier que les positions ouvertes restent ouvertes."""
        logger.info("=== TEST 2: Position Preservation ===")

        try:
            # Créer l'environnement avec capital suffisant
            data = self.create_sample_data()
            timeframes = ['5m', '1h', '4h']
            features_config = {tf: ['open', 'high', 'low', 'close', 'volume'] for tf in timeframes}

            # Configuration avec capital élevé pour éviter les resets
            test_config = self.config.copy()
            test_config['environment']['initial_balance'] = 100.0
            test_config['portfolio']['initial_balance'] = 100.0

            env = MultiAssetChunkedEnv(
                data=data,
                timeframes=timeframes,
                window_size=20,
                features_config=features_config,
                config=test_config
            )

            env.reset()

            # Ouvrir une position manuellement
            portfolio = env.portfolio_manager
            btc_position = portfolio.positions.get('BTCUSDT')

            if btc_position:
                # Ouvrir la position
                btc_position.open(entry_price=50000.0, size=0.001)
                initial_position_id = id(btc_position)

                logger.info(f"Position ouverte: id={initial_position_id}, is_open={btc_position.is_open}")

                # Vérifier que la position reste ouverte après plusieurs steps
                for i in range(10):
                    action = np.array([0.0, 0.0])  # Pas d'action pour préserver la position
                    obs, reward, done, truncated, info = env.step(action)

                    current_position = portfolio.positions.get('BTCUSDT')
                    if not current_position or not current_position.is_open:
                        logger.error(f"Position fermée au step {i}")
                        return False

                    if id(current_position) != initial_position_id:
                        logger.error(f"Position recréée au step {i}: {initial_position_id} -> {id(current_position)}")
                        return False

                logger.info("✅ TEST 2 RÉUSSI: Position préservée")
                return True
            else:
                logger.error("❌ TEST 2 ÉCHOUÉ: Position BTCUSDT non trouvée")
                return False

        except Exception as e:
            logger.error(f"❌ TEST 2 ÉCHOUÉ: {e}")
            return False

    def test_soft_reset_preservation(self) -> bool:
        """Test 3: Vérifier que les soft resets préservent les positions."""
        logger.info("=== TEST 3: Soft Reset Preservation ===")

        try:
            # Créer un PortfolioManager pour test direct
            config = {
                'portfolio': {'initial_balance': 50.0},
                'environment': {'min_capital_before_reset': 11.0},
                'trading_fees': 0.001,
                'min_order_value_usdt': 11.0
            }

            portfolio = PortfolioManager(env_config=config, assets=['BTCUSDT', 'ETHUSDT'])

            # Ouvrir une position
            btc_position = portfolio.positions.get('BTCUSDT')
            btc_position.open(entry_price=50000.0, size=0.001)

            initial_position_id = id(btc_position)
            logger.info(f"Position avant soft reset: id={initial_position_id}, is_open={btc_position.is_open}")

            # Effectuer un soft reset
            portfolio._perform_soft_reset(current_value=60.0)

            # Vérifier que la position est préservée
            after_reset_position = portfolio.positions.get('BTCUSDT')

            if not after_reset_position:
                logger.error("❌ Position BTCUSDT disparue après soft reset")
                return False

            if not after_reset_position.is_open:
                logger.error("❌ Position fermée après soft reset")
                return False

            if id(after_reset_position) != initial_position_id:
                logger.error(f"❌ Position recréée: {initial_position_id} -> {id(after_reset_position)}")
                return False

            logger.info("✅ TEST 3 RÉUSSI: Soft reset préserve les positions")
            return True

        except Exception as e:
            logger.error(f"❌ TEST 3 ÉCHOUÉ: {e}")
            return False

    def test_position_ids_stable(self) -> bool:
        """Test 4: Vérifier que les IDs des positions restent stables."""
        logger.info("=== TEST 4: Position IDs Stability ===")

        try:
            config = {
                'portfolio': {'initial_balance': 100.0},
                'environment': {'min_capital_before_reset': 11.0},
                'trading_fees': 0.001,
                'min_order_value_usdt': 11.0
            }

            portfolio = PortfolioManager(env_config=config, assets=['BTCUSDT', 'ETHUSDT'])

            # Enregistrer les IDs initiaux
            initial_ids = {asset: id(pos) for asset, pos in portfolio.positions.items()}
            logger.info(f"IDs initiaux: {initial_ids}")

            # Ouvrir une position
            btc_position = portfolio.positions.get('BTCUSDT')
            btc_position.open(entry_price=50000.0, size=0.001)

            # Simuler plusieurs resets qui ne devraient PAS recréer les positions
            for i in range(3):
                # Reset avec capital suffisant (pas de hard reset)
                portfolio.reset(new_epoch=False, force=False, min_capital_before_reset=11.0)

                current_ids = {asset: id(pos) for asset, pos in portfolio.positions.items()}

                for asset in initial_ids:
                    if current_ids[asset] != initial_ids[asset]:
                        logger.error(f"❌ ID changé pour {asset} après reset {i}: {initial_ids[asset]} -> {current_ids[asset]}")
                        return False

            logger.info("✅ TEST 4 RÉUSSI: IDs des positions stables")
            return True

        except Exception as e:
            logger.error(f"❌ TEST 4 ÉCHOUÉ: {e}")
            return False

    def run_all_tests(self):
        """Exécuter tous les tests."""
        logger.info("🚀 DÉBUT DES TESTS DE CONTINUITÉ DES POSITIONS")
        logger.info("=" * 60)

        tests = [
            ('portfolio_manager_recreation', self.test_portfolio_manager_recreation),
            ('position_preservation', self.test_position_preservation),
            ('soft_reset_preservation', self.test_soft_reset_preservation),
            ('position_ids_stable', self.test_position_ids_stable)
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            try:
                if test_func():
                    self.results[test_name] = True
                    passed += 1
                    logger.info(f"✅ {test_name.upper()} - RÉUSSI")
                else:
                    logger.info(f"❌ {test_name.upper()} - ÉCHOUÉ")
            except Exception as e:
                logger.error(f"💥 {test_name.upper()} - ERREUR: {e}")

            logger.info("-" * 40)

        self.results['tests_passed'] = passed
        self.results['total_tests'] = total

        # Résumé final
        logger.info("=" * 60)
        logger.info("📊 RÉSULTATS FINAUX")
        logger.info(f"Tests réussis: {passed}/{total}")

        if passed == total:
            logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS ! Le problème de continuité des positions est résolu.")
        else:
            logger.warning(f"⚠️  {total - passed} test(s) ont échoué. Des corrections supplémentaires sont nécessaires.")

        return self.results

def main():
    """Fonction principale."""
    try:
        # Activer l'environnement conda
        os.system("source ~/miniconda3/bin/activate trading_env")

        # Exécuter les tests
        tester = PositionContinuityTester()
        results = tester.run_all_tests()

        # Écrire les résultats dans un fichier
        import json
        with open('/home/morningstar/Documents/trading/position_continuity_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        return results['tests_passed'] == results['total_tests']

    except Exception as e:
        logger.error(f"Erreur lors de l'exécution des tests: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
