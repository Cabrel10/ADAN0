#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test pour valider les corrections de fréquence et métriques par worker.

Ce script teste les corrections implémentées selon l'analyse fournie :
1. Clarifier les métriques par worker (worker_id dans les logs)
2. Forcer la fréquence de trading (5-15 trades/jour)
3. Corriger le winrate (inclure trades neutres)
4. Éliminer la duplication des logs [RISK]
5. Corriger le drawdown affiché
"""

import logging
import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import yaml

# Ajouter le chemin vers le module bot
sys.path.insert(0, str(Path(__file__).parent / 'bot' / 'src'))

try:
    from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
    from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager
    from adan_trading_bot.performance.metrics import PerformanceMetrics
    from adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine
except ImportError as e:
    print(f"Erreur d'import : {e}")
    print("Assurez-vous que le chemin vers le module bot est correct.")
    sys.exit(1)

# Configuration du logging pour capturer les messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_worker_frequency.log')
    ]
)

logger = logging.getLogger(__name__)

class TestWorkerFrequencyCorrections:
    """Classe de test pour valider les corrections de fréquence et worker."""

    def __init__(self):
        """Initialise le test."""
        self.config = self._load_test_config()
        self.test_data = self._create_test_data()
        self.workers = []
        self.log_messages = []
        self.test_results = {}

    def _load_test_config(self) -> Dict:
        """Charge la configuration de test."""
        config_path = Path(__file__).parent / 'bot' / 'config' / 'config.yaml'
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("Configuration non trouvée, utilisation de la config par défaut")
            config = {}

        # Configuration de test avec les nouvelles valeurs de fréquence
        test_config = {
            'trading_rules': {
                'frequency': {
                    'min_positions': {
                        '5m': 6,
                        '1h': 3,
                        '4h': 1,
                        'total_daily': 5
                    },
                    'max_positions': {
                        '5m': 15,
                        '1h': 10,
                        '4h': 3,
                        'total_daily': 15
                    },
                    'frequency_bonus_weight': 0.3,
                    'frequency_penalty_weight': 1.0,  # Augmenté de 0.5 à 1.0
                    'action_threshold': 0.3,  # Réduit de 0.5 à 0.3
                    'force_trade_steps': 50,
                    'frequency_check_interval': 288
                }
            },
            'environment': {
                'initial_balance': 10000.0,
                'max_steps': 300,  # Test court
                'commission': 0.001
            },
            'assets': ['BTCUSDT'],
            'timeframes': ['5m', '1h', '4h']
        }

        # Fusionner avec la config existante si disponible
        if config:
            config.update(test_config)
            return config
        return test_config

    def _create_test_data(self) -> Dict:
        """Crée des données de test simulées."""
        # Simuler 1 jour de données (288 points pour 5m)
        n_points = 288
        timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='5T')

        # Prix simulés avec tendance et volatilité
        np.random.seed(42)
        price_base = 50000
        price_changes = np.cumsum(np.random.normal(0, 100, n_points))
        prices = price_base + price_changes

        # Créer les données OHLCV
        data = {
            'TIMESTAMP': timestamps.astype(np.int64) // 10**6,  # Millisecondes
            'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
            'high': prices * (1 + np.abs(np.random.normal(0.001, 0.002, n_points))),
            'low': prices * (1 - np.abs(np.random.normal(0.001, 0.002, n_points))),
            'close': prices,
            'volume': np.random.uniform(100, 1000, n_points),
            # Indicateurs simulés
            'rsi_14': np.random.uniform(20, 80, n_points),
            'macd_hist': np.random.normal(0, 50, n_points),
            'atr_14': np.random.uniform(100, 500, n_points),
            'bb_upper': prices * 1.02,
            'bb_middle': prices,
            'bb_lower': prices * 0.98,
            'ema_12': prices * (1 + np.random.normal(0, 0.001, n_points)),
            'ema_26': prices * (1 + np.random.normal(0, 0.001, n_points)),
            'sma_200': prices * (1 + np.random.normal(0, 0.005, n_points)),
            'adx_14': np.random.uniform(10, 70, n_points)
        }

        return {'BTCUSDT': {tf: pd.DataFrame(data) for tf in ['5m', '1h', '4h']}}

    def test_worker_identification(self, n_workers: int = 2) -> Dict:
        """Test 1: Vérification de l'identification des workers dans les logs."""
        logger.info("=" * 80)
        logger.info("TEST 1: Vérification de l'identification des workers")
        logger.info("=" * 80)

        results = {'passed': True, 'errors': []}
        worker_logs = {i: [] for i in range(n_workers)}

        try:
            # Créer plusieurs workers
            for worker_id in range(n_workers):
                logger.info(f"Création du Worker {worker_id}")

                # Configuration spécifique au worker
                worker_config = self.config.copy()
                worker_config['worker_id'] = worker_id

                # Test des composants individuels
                portfolio_manager = PortfolioManager(config=worker_config, worker_id=worker_id)
                performance_metrics = PerformanceMetrics(config=worker_config, worker_id=worker_id)
                dbe = DynamicBehaviorEngine(config=worker_config, worker_id=worker_id)

                # Vérifier que les worker_id sont correctement assignés
                if portfolio_manager.worker_id != worker_id:
                    results['errors'].append(f"PortfolioManager Worker {worker_id}: worker_id incorrect ({portfolio_manager.worker_id})")
                    results['passed'] = False

                if performance_metrics.worker_id != worker_id:
                    results['errors'].append(f"PerformanceMetrics Worker {worker_id}: worker_id incorrect ({performance_metrics.worker_id})")
                    results['passed'] = False

                if dbe.worker_id != worker_id:
                    results['errors'].append(f"DynamicBehaviorEngine Worker {worker_id}: worker_id incorrect ({dbe.worker_id})")
                    results['passed'] = False

                # Simuler quelques opérations pour déclencher des logs
                logger.info(f"Test des logs pour Worker {worker_id}")
                portfolio_manager.get_drawdown()  # Devrait logger [RISK] uniquement pour worker 0

                # Simuler une métrique
                fake_metrics = performance_metrics.calculate_metrics()

                self.workers.append({
                    'id': worker_id,
                    'portfolio': portfolio_manager,
                    'metrics': performance_metrics,
                    'dbe': dbe
                })

        except Exception as e:
            results['errors'].append(f"Erreur lors de la création des workers: {e}")
            results['passed'] = False

        # Résultats du test
        if results['passed']:
            logger.info("✅ TEST 1 RÉUSSI: Workers correctement identifiés")
        else:
            logger.error("❌ TEST 1 ÉCHOUÉ:")
            for error in results['errors']:
                logger.error(f"  - {error}")

        return results

    def test_frequency_forcing(self, worker_id: int = 0, n_steps: int = 100) -> Dict:
        """Test 2: Vérification du forçage de la fréquence de trading."""
        logger.info("=" * 80)
        logger.info("TEST 2: Vérification du forçage de la fréquence")
        logger.info("=" * 80)

        results = {'passed': True, 'errors': [], 'stats': {}}

        try:
            if not self.workers:
                self.test_worker_identification()

            worker = self.workers[worker_id]
            portfolio_manager = worker['portfolio']

            # Initialiser les compteurs de positions
            positions_count = {'5m': 0, '1h': 0, '4h': 0, 'daily_total': 0}
            last_trade_step = {'5m': 0, '1h': 0, '4h': 0}

            # Configuration de fréquence
            frequency_config = self.config['trading_rules']['frequency']
            action_threshold = frequency_config.get('action_threshold', 0.3)
            force_trade_steps = frequency_config.get('force_trade_steps', 50)

            # Simuler des actions de trading
            forced_trades = 0
            normal_trades = 0

            for step in range(n_steps):
                # Action aléatoire
                action = np.random.uniform(-1, 1)
                current_tf = ['5m', '1h', '4h'][step % 3]  # Alterner entre timeframes

                # Vérifier si un trade doit être forcé
                steps_since_last_trade = step - last_trade_step.get(current_tf, 0)
                min_pos_tf = frequency_config['min_positions'].get(current_tf, 1)
                should_force_trade = (positions_count[current_tf] < min_pos_tf and
                                     steps_since_last_trade >= force_trade_steps)

                # Simuler l'exécution d'un trade
                if abs(action) > action_threshold or should_force_trade:
                    positions_count[current_tf] += 1
                    positions_count['daily_total'] += 1
                    last_trade_step[current_tf] = step

                    if should_force_trade:
                        forced_trades += 1
                        logger.info(f"[FORCED TRADE Worker {worker_id}] Step {step}, TF: {current_tf}, Action: {action:.3f}")
                    else:
                        normal_trades += 1
                        logger.info(f"[NORMAL TRADE Worker {worker_id}] Step {step}, TF: {current_tf}, Action: {action:.3f}")

            # Vérifier les objectifs de fréquence
            results['stats'] = {
                'positions_count': positions_count,
                'forced_trades': forced_trades,
                'normal_trades': normal_trades,
                'total_trades': forced_trades + normal_trades
            }

            # Vérifications
            min_positions = frequency_config['min_positions']
            for tf in ['5m', '1h', '4h']:
                if positions_count[tf] < min_positions[tf]:
                    results['errors'].append(f"Timeframe {tf}: {positions_count[tf]} positions < minimum {min_positions[tf]}")
                    results['passed'] = False

            # Vérifier le total journalier
            daily_total = positions_count['daily_total']
            min_daily = min_positions['total_daily']
            max_daily = frequency_config['max_positions']['total_daily']

            if daily_total < min_daily:
                results['errors'].append(f"Total journalier: {daily_total} < minimum {min_daily}")
                results['passed'] = False
            elif daily_total > max_daily:
                results['errors'].append(f"Total journalier: {daily_total} > maximum {max_daily}")
                results['passed'] = False

            logger.info(f"Statistiques du test de fréquence:")
            logger.info(f"  - Positions par TF: {positions_count}")
            logger.info(f"  - Trades forcés: {forced_trades}")
            logger.info(f"  - Trades normaux: {normal_trades}")
            logger.info(f"  - Total trades: {forced_trades + normal_trades}")

        except Exception as e:
            results['errors'].append(f"Erreur lors du test de fréquence: {e}")
            results['passed'] = False

        # Résultats du test
        if results['passed']:
            logger.info("✅ TEST 2 RÉUSSI: Fréquence de trading correcte")
        else:
            logger.error("❌ TEST 2 ÉCHOUÉ:")
            for error in results['errors']:
                logger.error(f"  - {error}")

        return results

    def test_winrate_calculation(self, worker_id: int = 0) -> Dict:
        """Test 3: Vérification du calcul correct du winrate avec trades neutres."""
        logger.info("=" * 80)
        logger.info("TEST 3: Vérification du calcul du winrate")
        logger.info("=" * 80)

        results = {'passed': True, 'errors': [], 'stats': {}}

        try:
            if not self.workers:
                self.test_worker_identification()

            worker = self.workers[worker_id]
            performance_metrics = worker['metrics']

            # Simuler des positions fermées avec différents résultats
            test_positions = [
                {'asset': 'BTCUSDT', 'entry_price': 50000, 'exit_price': 51000, 'size': 0.1},  # Win
                {'asset': 'BTCUSDT', 'entry_price': 50000, 'exit_price': 49000, 'size': 0.1},  # Loss
                {'asset': 'BTCUSDT', 'entry_price': 50000, 'exit_price': 50000, 'size': 0.1},  # Neutral
                {'asset': 'BTCUSDT', 'entry_price': 50000, 'exit_price': 52000, 'size': 0.1},  # Win
                {'asset': 'BTCUSDT', 'entry_price': 50000, 'exit_price': 48000, 'size': 0.1},  # Loss
            ]

            expected_wins = 2
            expected_losses = 2
            expected_neutrals = 1
            expected_total = 5
            expected_winrate = (expected_wins / expected_total) * 100

            # Ajouter les positions fermées
            for pos in test_positions:
                performance_metrics.close_position(pos, pos['exit_price'])

            # Calculer les métriques
            metrics = performance_metrics.calculate_metrics()

            # Vérifications
            if metrics['wins'] != expected_wins:
                results['errors'].append(f"Wins: attendu {expected_wins}, obtenu {metrics['wins']}")
                results['passed'] = False

            if metrics['losses'] != expected_losses:
                results['errors'].append(f"Losses: attendu {expected_losses}, obtenu {metrics['losses']}")
                results['passed'] = False

            if metrics['neutrals'] != expected_neutrals:
                results['errors'].append(f"Neutrals: attendu {expected_neutrals}, obtenu {metrics['neutrals']}")
                results['passed'] = False

            if metrics['total_trades'] != expected_total:
                results['errors'].append(f"Total trades: attendu {expected_total}, obtenu {metrics['total_trades']}")
                results['passed'] = False

            if abs(metrics['winrate'] - expected_winrate) > 0.1:
                results['errors'].append(f"Winrate: attendu {expected_winrate:.1f}%, obtenu {metrics['winrate']:.1f}%")
                results['passed'] = False

            results['stats'] = {
                'expected': {
                    'wins': expected_wins,
                    'losses': expected_losses,
                    'neutrals': expected_neutrals,
                    'total': expected_total,
                    'winrate': expected_winrate
                },
                'actual': metrics
            }

            logger.info(f"Statistiques du winrate:")
            logger.info(f"  - Wins: {metrics['wins']}/{expected_wins}")
            logger.info(f"  - Losses: {metrics['losses']}/{expected_losses}")
            logger.info(f"  - Neutrals: {metrics['neutrals']}/{expected_neutrals}")
            logger.info(f"  - Total: {metrics['total_trades']}/{expected_total}")
            logger.info(f"  - Winrate: {metrics['winrate']:.1f}%/{expected_winrate:.1f}%")

        except Exception as e:
            results['errors'].append(f"Erreur lors du test de winrate: {e}")
            results['passed'] = False

        # Résultats du test
        if results['passed']:
            logger.info("✅ TEST 3 RÉUSSI: Calcul du winrate correct")
        else:
            logger.error("❌ TEST 3 ÉCHOUÉ:")
            for error in results['errors']:
                logger.error(f"  - {error}")

        return results

    def test_log_duplication(self, n_workers: int = 2) -> Dict:
        """Test 4: Vérification de l'absence de duplication des logs [RISK]."""
        logger.info("=" * 80)
        logger.info("TEST 4: Vérification de l'absence de duplication des logs")
        logger.info("=" * 80)

        results = {'passed': True, 'errors': [], 'stats': {}}

        try:
            if not self.workers or len(self.workers) < n_workers:
                self.test_worker_identification(n_workers)

            # Capturer les logs dans un handler personnalisé
            log_capture = []

            class LogCapture(logging.Handler):
                def emit(self, record):
                    log_capture.append(record.getMessage())

            capture_handler = LogCapture()
            capture_handler.setLevel(logging.INFO)
            logging.getLogger().addHandler(capture_handler)

            # Déclencher des logs [RISK] depuis plusieurs workers
            for i, worker in enumerate(self.workers[:n_workers]):
                portfolio_manager = worker['portfolio']
                logger.info(f"Déclenchement des logs depuis Worker {i}")
                portfolio_manager.get_drawdown()  # Devrait logger [RISK] uniquement depuis worker 0

            # Supprimer le handler
            logging.getLogger().removeHandler(capture_handler)

            # Compter les logs [RISK]
            risk_logs = [log for log in log_capture if '[RISK]' in log]
            risk_log_count = len(risk_logs)

            results['stats'] = {
                'total_logs_captured': len(log_capture),
                'risk_logs_count': risk_log_count,
                'risk_logs': risk_logs
            }

            # Vérification: il ne devrait y avoir qu'un seul log [RISK] (depuis worker 0)
            expected_risk_logs = 1
            if risk_log_count != expected_risk_logs:
                results['errors'].append(f"Logs [RISK]: attendu {expected_risk_logs}, obtenu {risk_log_count}")
                results['passed'] = False

            logger.info(f"Statistiques des logs:")
            logger.info(f"  - Total logs capturés: {len(log_capture)}")
            logger.info(f"  - Logs [RISK]: {risk_log_count}")
            for risk_log in risk_logs:
                logger.info(f"    -> {risk_log}")

        except Exception as e:
            results['errors'].append(f"Erreur lors du test de duplication: {e}")
            results['passed'] = False

        # Résultats du test
        if results['passed']:
            logger.info("✅ TEST 4 RÉUSSI: Pas de duplication des logs [RISK]")
        else:
            logger.error("❌ TEST 4 ÉCHOUÉ:")
            for error in results['errors']:
                logger.error(f"  - {error}")

        return results

    def test_drawdown_calculation(self, worker_id: int = 0) -> Dict:
        """Test 5: Vérification du calcul correct du drawdown."""
        logger.info("=" * 80)
        logger.info("TEST 5: Vérification du calcul du drawdown")
        logger.info("=" * 80)

        results = {'passed': True, 'errors': [], 'stats': {}}

        try:
            if not self.workers:
                self.test_worker_identification()

            worker = self.workers[worker_id]
            portfolio_manager = worker['portfolio']

            # Simuler des changements d'équité
            initial_equity = 10000.0
            portfolio_manager.cash = initial_equity
            portfolio_manager.peak_equity = initial_equity
            portfolio_manager.max_dd = 0.0

            # Simuler une baisse d'équité
            portfolio_manager.cash = 9500.0  # Perte de 500 USDT
            drawdown, max_dd = portfolio_manager.get_drawdown()

            expected_drawdown = 5.0  # 500/10000 * 100 = 5%

            # Vérifications
            if abs(drawdown - expected_drawdown) > 0.1:
                results['errors'].append(f"Drawdown: attendu {expected_drawdown:.1f}%, obtenu {drawdown:.1f}%")
                results['passed'] = False

            if abs(max_dd - expected_drawdown) > 0.1:
                results['errors'].append(f"Max DD: attendu {expected_drawdown:.1f}%, obtenu {max_dd:.1f}%")
                results['passed'] = False

            # Simuler une nouvelle baisse plus importante
            portfolio_manager.cash = 9000.0  # Perte de 1000 USDT total
            drawdown2, max_dd2 = portfolio_manager.get_drawdown()

            expected_drawdown2 = 10.0  # 1000/10000 * 100 = 10%

            if abs(drawdown2 - expected_drawdown2) > 0.1:
                results['errors'].append(f"Drawdown2: attendu {expected_drawdown2:.1f}%, obtenu {drawdown2:.1f}%")
                results['passed'] = False

            if abs(max_dd2 - expected_drawdown2) > 0.1:
                results['errors'].append(f"Max DD2: attendu {expected_drawdown2:.1f}%, obtenu {max_dd2:.1f}%")
                results['passed'] = False

            results['stats'] = {
                'initial_equity': initial_equity,
                'drawdown1': drawdown,
                'max_dd1': max_dd,
                'drawdown2': drawdown2,
                'max_dd2': max_dd2
            }

            logger.info(f"Statistiques du drawdown:")
            logger.info(f"  - Équité initiale: {initial_equity}")
            logger.info(f"  - Drawdown 1: {drawdown:.1f}%")
            logger.info(f"  - Max DD 1: {max_dd:.1f}%")
            logger.info(f"  - Drawdown 2: {drawdown2:.1f}%")
            logger.info(f"  - Max DD 2: {max_dd2:.1f}%")

        except Exception as e:
            results['errors'].append(f"Erreur lors du test de drawdown: {e}")
            results['passed'] = False

        # Résultats du test
        if results['passed']:
            logger.info("✅ TEST 5 RÉUSSI: Calcul du drawdown correct")
        else:
            logger.error("❌ TEST 5 ÉCHOUÉ:")
            for error in results['errors']:
                logger.error(f"  - {error}")

        return results

    def run_all_tests(self) -> Dict:
        """Exécute tous les tests et génère un rapport complet."""
        logger.info("🚀 DÉBUT DES TESTS DE VALIDATION DES CORRECTIONS")
        logger.info("=" * 80)

        start_time = time.time()
        all_results = {}

        # Test 1: Identification des workers
        all_results['worker_identification'] = self.test_worker_identification()

        # Test 2: Forçage de la fréquence
        all_results['frequency_forcing'] = self.test_frequency_forcing()

        # Test 3: Calcul du winrate
        all_results['winrate_calculation'] = self.test_winrate_calculation()

        # Test 4: Duplication des logs
        all_results['log_duplication'] = self.test_log_duplication()

        # Test 5: Calcul du drawdown
        all_results['drawdown_calculation'] = self.test_drawdown_calculation()

        end_time = time.time()
        execution_time = end_time - start_time

        # Rapport final
        logger.info("=" * 80)
        logger.info("📊 RAPPORT FINAL DES TESTS")
        logger.info("=" * 80)

        passed_tests = sum(1 for result in all_results.values() if result['passed'])
        total_tests = len(all_results)
        success_rate = (passed_tests / total_tests) * 100

        logger.info(f"Tests réussis: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        logger.info(f"Temps d'exécution: {execution_time:.2f} secondes")

        for test_name, result in all_results.items():
            status = "✅ RÉUSSI" if result['passed'] else "❌ ÉCHOUÉ"
            logger.info(f"  - {test_name}: {status}")
            if not result['passed']:
                for error in result['errors']:
                    logger.info(f"    -> {error}")

        # Recommandations
        if success_rate == 100:
            logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS !")
            logger.info("Les corrections ont été correctement implémentées.")
            logger.info("Vous pouvez maintenant relancer l'entraînement avec:")
            logger.info("timeout 30s /home/morningstar/miniconda3/envs/trading_env/bin/python bot/scripts/train_parallel_agents.py --config bot/config/config.yaml --checkpoint-dir bot/checkpoints")
        else:
            logger.warning(f"⚠️ {total_tests - passed_tests} test(s) ont échoué.")
            logger.warning("Vérifiez les erreurs ci-dessus avant de continuer.")

        return {
            'results': all_results,
            'summary': {
                'passed_tests': passed_tests,
                'total_tests': total_tests,
                'success_rate': success_rate,
                'execution_time': execution_time
            }
        }


def main():
    """Fonction principale pour exécuter les tests."""
    try:
        # Créer l'instance de test
        tester = TestWorkerFrequencyCorrections()

        # Exécuter tous les tests
        final_results = tester.run_all_tests()

        # Sauvegarder les résultats
        results_file = Path('test_results_worker_frequency.yaml')
        with open(results_file, 'w', encoding='utf-8') as f:
            yaml.dump(final_results, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Résultats sauvegardés dans: {results_file}")

        # Code de sortie basé sur les résultats
        if final_results['summary']['success_rate'] == 100:
            sys.exit(0)  # Succès
        else:
            sys.exit(1)  # Échec

    except Exception as e:
        logger.error(f"Erreur lors de l'exécution des tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
