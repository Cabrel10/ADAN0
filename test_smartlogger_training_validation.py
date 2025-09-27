#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test de Validation Finale - SmartLogger avec Simulation d'Entrainement
=====================================================================

Ce script valide le système SmartLogger dans un environnement simulé
d'entrainement multi-workers pour confirmer la résolution du
Problème #4: "Logs Restreints au Worker 0"

Test de validation finale:
✅ Simulation d'entrainement avec 4 workers parallèles
✅ Validation que tous les workers peuvent loguer
✅ Test des logs critiques (portfolio, positions, trades)
✅ Comparaison avant/après la correction
✅ Validation des métriques de logging

Auteur: Trading Bot Team
Date: 2024
"""

import sys
import time
import logging
import threading
from pathlib import Path
from collections import defaultdict, deque
from unittest.mock import Mock

# Ajouter le chemin du bot pour les imports
sys.path.insert(0, str(Path(__file__).parent / "bot" / "src"))

try:
    from adan_trading_bot.utils.smart_logger import create_smart_logger
except ImportError as e:
    print(f"❌ Erreur d'import SmartLogger: {e}")
    sys.exit(1)


class TrainingSimulationLogger:
    """Logger simulé pour capturer les logs d'entrainement."""

    def __init__(self):
        self.logs = deque(maxlen=10000)
        self.worker_logs = defaultdict(list)
        self.level_counts = defaultdict(int)

    def _log(self, level, message):
        """Enregistre un log avec métadonnées."""
        worker_id = self._extract_worker_id(message)

        log_entry = {
            'level': level,
            'message': message,
            'worker_id': worker_id,
            'timestamp': time.time()
        }

        self.logs.append(log_entry)
        if worker_id is not None:
            self.worker_logs[worker_id].append(log_entry)
        self.level_counts[level] += 1

        # Afficher les logs importants
        if level in ['ERROR', 'WARNING'] or any(keyword in message.lower()
                                               for keyword in ['portfolio', 'position', 'trade', 'profit', 'loss']):
            print(f"[{level}] {message}")

    def _extract_worker_id(self, message):
        """Extrait l'ID du worker depuis le message."""
        if '[Worker ' in message:
            try:
                start = message.find('[Worker ') + 8
                end = message.find(']', start)
                return int(message[start:end])
            except:
                pass
        return None

    def error(self, message, *args, **kwargs):
        self._log('ERROR', message)

    def warning(self, message, *args, **kwargs):
        self._log('WARNING', message)

    def info(self, message, *args, **kwargs):
        self._log('INFO', message)

    def debug(self, message, *args, **kwargs):
        self._log('DEBUG', message)

    def critical(self, message, *args, **kwargs):
        self._log('CRITICAL', message)

    def get_stats(self):
        """Retourne les statistiques des logs."""
        return {
            'total_logs': len(self.logs),
            'unique_workers': len(self.worker_logs),
            'workers_with_logs': list(self.worker_logs.keys()),
            'level_counts': dict(self.level_counts),
            'logs_by_worker': {wid: len(logs) for wid, logs in self.worker_logs.items()}
        }


class MockPortfolioManager:
    """Simulateur de PortfolioManager pour test."""

    def __init__(self, worker_id, smart_logger, sim_logger):
        self.worker_id = worker_id
        self.smart_logger = smart_logger
        self.sim_logger = sim_logger
        self.cash = 10000.0
        self.portfolio_value = 10000.0
        self.positions = {}

    def simulate_trading_activity(self, steps=50):
        """Simule une activité de trading."""
        for step in range(steps):
            # Simulation d'ouverture de position
            if step % 10 == 0:
                self.smart_logger.smart_info(
                    self.sim_logger,
                    f"Portfolio value: {self.portfolio_value:.2f} USDT, Cash: {self.cash:.2f} USDT",
                    step
                )

            # Simulation d'erreurs occasionnelles
            if step % 20 == 0 and step > 0:
                self.smart_logger.smart_error(
                    self.sim_logger,
                    f"Failed to update position for BTCUSDT at step {step}"
                )

            # Simulation de warnings
            if step % 15 == 0:
                self.smart_logger.smart_warning(
                    self.sim_logger,
                    f"High drawdown detected: {(step * 0.1):.1f}%"
                )

            # Simulation de logs de debug
            if step % 5 == 0:
                self.smart_logger.smart_debug(
                    self.sim_logger,
                    f"Debug: Processing market data at step {step}"
                )

            # Simulation de trades
            if step % 25 == 0 and step > 0:
                profit = (step * 0.5) - 10
                self.smart_logger.smart_info(
                    self.sim_logger,
                    f"Trade completed: PnL = {profit:.2f} USDT",
                    step
                )

            time.sleep(0.01)  # Petite pause pour simuler le traitement


class TrainingSimulationValidator:
    """Validateur principal pour la simulation d'entrainement."""

    def __init__(self):
        self.sim_logger = TrainingSimulationLogger()
        self.results = {}

    def print_header(self, title):
        """Affiche un en-tête."""
        print(f"\n{'='*80}")
        print(f"🎯 {title}")
        print(f"{'='*80}")

    def print_result(self, test_name, passed, details=""):
        """Affiche le résultat d'un test."""
        status = "✅ RÉUSSI" if passed else "❌ ÉCHOUÉ"
        self.results[test_name] = passed
        print(f"\n{status} - {test_name}")
        if details:
            print(f"📋 Détails: {details}")

    def test_old_system_simulation(self):
        """Simule l'ancien système (worker_id == 0 seulement)."""
        self.print_header("SIMULATION ANCIEN SYSTÈME (worker_id == 0 uniquement)")

        # Simuler l'ancien système où seul worker 0 peut loguer
        old_sim_logger = TrainingSimulationLogger()

        for worker_id in range(4):
            # Ancien système: seulement worker 0 logue
            if worker_id == 0:
                old_sim_logger.info(f"[Worker {worker_id}] Portfolio value: 10000.00 USDT")
                old_sim_logger.error(f"[Worker {worker_id}] Failed to update position")
                old_sim_logger.warning(f"[Worker {worker_id}] High drawdown detected")
                old_sim_logger.info(f"[Worker {worker_id}] Trade completed: PnL = 15.50 USDT")

        old_stats = old_sim_logger.get_stats()

        print(f"📊 Ancien système - Worker actifs: {old_stats['unique_workers']}/4")
        print(f"📊 Total logs: {old_stats['total_logs']}")
        print(f"📊 Logs par niveau: {old_stats['level_counts']}")

        return old_stats

    def test_new_system_simulation(self):
        """Simule le nouveau système avec SmartLogger."""
        self.print_header("SIMULATION NOUVEAU SYSTÈME (SmartLogger Multi-Workers)")

        # Créer des workers avec SmartLogger
        workers = []
        portfolio_managers = []

        for worker_id in range(4):
            smart_logger = create_smart_logger(worker_id, 4, f"training_worker_{worker_id}")
            portfolio_manager = MockPortfolioManager(worker_id, smart_logger, self.sim_logger)
            workers.append(smart_logger)
            portfolio_managers.append(portfolio_manager)

        # Simuler l'entrainement parallèle
        threads = []
        for pm in portfolio_managers:
            thread = threading.Thread(target=pm.simulate_trading_activity, args=(30,))
            threads.append(thread)
            thread.start()

        # Attendre la fin des simulations
        for thread in threads:
            thread.join()

        new_stats = self.sim_logger.get_stats()

        print(f"📊 Nouveau système - Workers actifs: {new_stats['unique_workers']}/4")
        print(f"📊 Total logs: {new_stats['total_logs']}")
        print(f"📊 Logs par niveau: {new_stats['level_counts']}")
        print(f"📊 Workers avec logs: {sorted(new_stats['workers_with_logs'])}")

        return new_stats

    def validate_multi_worker_logging(self, stats):
        """Valide que tous les workers peuvent loguer."""
        self.print_header("VALIDATION MULTI-WORKERS")

        # Test 1: Tous les workers ont loggé
        all_workers_active = stats['unique_workers'] == 4
        self.print_result(
            "Tous workers actifs",
            all_workers_active,
            f"Workers actifs: {stats['unique_workers']}/4"
        )

        # Test 2: Tous les workers ont des erreurs (critiques)
        workers_with_errors = len([wid for wid in stats['workers_with_logs']
                                  if any(log['level'] == 'ERROR' for log in self.sim_logger.worker_logs[wid])])
        all_workers_errors = workers_with_errors == 4
        self.print_result(
            "Tous workers loggent erreurs",
            all_workers_errors,
            f"Workers avec erreurs: {workers_with_errors}/4"
        )

        # Test 3: Logs INFO de portfolio/positions de plusieurs workers
        workers_with_portfolio_logs = len([
            wid for wid in stats['workers_with_logs']
            if any('portfolio' in log['message'].lower() or 'trade' in log['message'].lower()
                   for log in self.sim_logger.worker_logs[wid] if log['level'] == 'INFO')
        ])
        multi_worker_info = workers_with_portfolio_logs >= 2
        self.print_result(
            "Multiples workers loggent infos critiques",
            multi_worker_info,
            f"Workers avec logs portfolio/trade: {workers_with_portfolio_logs}"
        )

        # Test 4: Volume de logs suffisant
        sufficient_logs = stats['total_logs'] >= 100
        self.print_result(
            "Volume de logs suffisant",
            sufficient_logs,
            f"Total logs: {stats['total_logs']}"
        )

        return all([all_workers_active, all_workers_errors, multi_worker_info, sufficient_logs])

    def compare_systems(self, old_stats, new_stats):
        """Compare l'ancien et le nouveau système."""
        self.print_header("COMPARAISON ANCIEN vs NOUVEAU SYSTÈME")

        # Amélioration du nombre de workers actifs
        worker_improvement = new_stats['unique_workers'] > old_stats['unique_workers']
        self.print_result(
            "Plus de workers actifs",
            worker_improvement,
            f"Ancien: {old_stats['unique_workers']} → Nouveau: {new_stats['unique_workers']}"
        )

        # Amélioration du volume de logs
        log_improvement = new_stats['total_logs'] > old_stats['total_logs']
        self.print_result(
            "Plus de logs informatifs",
            log_improvement,
            f"Ancien: {old_stats['total_logs']} → Nouveau: {new_stats['total_logs']}"
        )

        # Amélioration des erreurs loggées
        old_errors = old_stats['level_counts'].get('ERROR', 0)
        new_errors = new_stats['level_counts'].get('ERROR', 0)
        error_improvement = new_errors > old_errors
        self.print_result(
            "Plus d'erreurs capturées",
            error_improvement,
            f"Ancien: {old_errors} → Nouveau: {new_errors}"
        )

        # Calcul du facteur d'amélioration
        if old_stats['total_logs'] > 0:
            improvement_factor = new_stats['total_logs'] / old_stats['total_logs']
            print(f"\n📈 Facteur d'amélioration: {improvement_factor:.1f}x plus de logs")

        return all([worker_improvement, log_improvement, error_improvement])

    def run_validation(self):
        """Lance la validation complète."""
        print("🚀 VALIDATION FINALE - SMARTLOGGER TRAINING SIMULATION")
        print("=" * 80)

        start_time = time.time()

        # 1. Simuler l'ancien système
        old_stats = self.test_old_system_simulation()

        # 2. Simuler le nouveau système
        new_stats = self.test_new_system_simulation()

        # 3. Valider le multi-workers
        multi_worker_ok = self.validate_multi_worker_logging(new_stats)

        # 4. Comparer les systèmes
        comparison_ok = self.compare_systems(old_stats, new_stats)

        duration = time.time() - start_time

        # Résumé final
        passed = sum(1 for result in self.results.values() if result)
        total = len(self.results)
        success_rate = (passed / total) * 100 if total > 0 else 0

        self.print_header("RÉSUMÉ FINAL")
        print(f"✅ Tests réussis: {passed}/{total} ({success_rate:.1f}%)")
        print(f"⏱️  Durée: {duration:.2f}s")
        print(f"📊 Nouveau système - Workers actifs: {new_stats['unique_workers']}/4")
        print(f"📊 Total logs capturés: {new_stats['total_logs']}")

        overall_success = all([multi_worker_ok, comparison_ok])

        if overall_success and success_rate >= 75:
            print(f"\n🎉 VALIDATION RÉUSSIE!")
            print(f"✅ Problème #4 'Logs Restreints au Worker 0' est RÉSOLU!")
            print(f"✅ Tous les workers peuvent maintenant loguer intelligemment!")
            print(f"✅ Amélioration significative par rapport à l'ancien système!")
        else:
            print(f"\n⚠️  VALIDATION PARTIELLE ({success_rate:.1f}%)")
            print(f"Certains aspects nécessitent encore des ajustements")

        return overall_success and success_rate >= 75


def main():
    """Point d'entrée principal."""
    print("🔧 Validation Finale - SmartLogger Training Simulation")
    print("=" * 60)
    print("Test du Problème #4: Logs Restreints au Worker 0")
    print("=" * 60)

    # Exécuter la validation
    validator = TrainingSimulationValidator()
    success = validator.run_validation()

    if success:
        print(f"\n🚀 PRÊT POUR L'ENTRAINEMENT!")
        print(f"Le système SmartLogger est opérationnel pour l'entrainement multi-workers.")
        print(f"\nCommande d'entrainement recommandée:")
        print(f"timeout 30s /home/morningstar/miniconda3/envs/trading_env/bin/python bot/scripts/train_parallel_agents.py --config bot/config/config.yaml --checkpoint-dir bot/checkpoints")
    else:
        print(f"\n❌ VALIDATION ÉCHOUÉE")
        print(f"Des ajustements sont nécessaires avant l'entrainement.")

    # Code de sortie
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
