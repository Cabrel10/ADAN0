#!/usr/bin/env python3
"""
Test du système de logging intelligent pour vérifier que tous les workers
peuvent loguer leurs informations sans être restreints au worker 0 uniquement.

Ce test vérifie :
1. Que tous les workers peuvent loguer des erreurs et warnings
2. Que le système de rotation fonctionne pour les logs informationnels
3. Que la déduplication évite les doublons
4. Que le sampling fonctionne pour les logs de debug
5. Que les statistiques sont correctement collectées

Usage:
    cd trading/
    python test_worker_logging.py
"""

import sys
import os
import time
import logging
import threading
from io import StringIO
from unittest.mock import Mock, patch
from typing import List, Dict

# Ajouter le chemin du bot
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bot', 'src'))

try:
    from adan_trading_bot.common.logging_utils import (
        SmartLogger, create_smart_logger, configure_smart_logger
    )
except ImportError as e:
    print(f"Erreur d'import: {e}")
    sys.exit(1)


class MockLogHandler(logging.Handler):
    """Handler personnalisé pour capturer les logs durant les tests."""

    def __init__(self):
        super().__init__()
        self.logs = []
        self.lock = threading.Lock()

    def emit(self, record):
        with self.lock:
            self.logs.append({
                'level': record.levelname,
                'message': record.getMessage(),
                'worker_id': self._extract_worker_id(record.getMessage())
            })

    def _extract_worker_id(self, message: str) -> int:
        """Extrait le worker_id du message de log."""
        import re
        match = re.search(r'\[Worker (\d+)\]', message)
        return int(match.group(1)) if match else -1

    def get_logs_by_worker(self, worker_id: int) -> List[Dict]:
        """Retourne les logs d'un worker spécifique."""
        with self.lock:
            return [log for log in self.logs if log['worker_id'] == worker_id]

    def get_logs_by_level(self, level: str) -> List[Dict]:
        """Retourne les logs d'un niveau spécifique."""
        with self.lock:
            return [log for log in self.logs if log['level'] == level]

    def clear(self):
        """Vide les logs capturés."""
        with self.lock:
            self.logs.clear()


class WorkerLoggingTest:
    """Test du système de logging multi-worker."""

    def __init__(self):
        # Configuration du logging de test
        self.mock_handler = MockLogHandler()
        self.test_logger = logging.getLogger('test_logger')
        self.test_logger.setLevel(logging.DEBUG)
        self.test_logger.addHandler(self.mock_handler)

        # Éviter la propagation vers le logger racine
        self.test_logger.propagate = False

        self.results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'details': []
        }

    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log le résultat d'un test."""
        if success:
            self.results['tests_passed'] += 1
            print(f"✅ {test_name}")
        else:
            self.results['tests_failed'] += 1
            print(f"❌ {test_name}")

        if details:
            print(f"   {details}")

        self.results['details'].append({
            'test_name': test_name,
            'success': success,
            'details': details
        })

    def test_basic_smart_logger_creation(self):
        """Test 1: Création de base du SmartLogger."""
        test_name = "Création SmartLogger pour différents workers"

        try:
            # Créer des SmartLoggers pour différents workers
            loggers = []
            for worker_id in range(4):
                smart_logger = create_smart_logger(self.test_logger, worker_id, 4)
                configure_smart_logger(smart_logger, 'testing')
                loggers.append(smart_logger)

            # Vérifier les propriétés
            success = all(logger.worker_id == i for i, logger in enumerate(loggers))
            success = success and all(logger.total_workers == 4 for logger in loggers)

            self.log_test_result(test_name, success,
                               f"Créé {len(loggers)} SmartLoggers avec IDs corrects")

            return loggers

        except Exception as e:
            self.log_test_result(test_name, False, f"Erreur: {e}")
            return []

    def test_all_workers_can_log_errors(self):
        """Test 2: Tous les workers peuvent loguer des erreurs."""
        test_name = "Tous workers loggent les erreurs"

        try:
            self.mock_handler.clear()
            loggers = []

            # Créer des loggers pour 4 workers
            for worker_id in range(4):
                smart_logger = create_smart_logger(self.test_logger, worker_id, 4)
                loggers.append(smart_logger)

            # Chaque worker log une erreur unique
            for i, logger in enumerate(loggers):
                logger.error(f"Erreur critique du worker {i}", dedupe=False)

            time.sleep(0.1)  # Attendre que les logs soient traités

            # Vérifier que chaque worker a bien loggé
            success = True
            details = []

            for worker_id in range(4):
                worker_logs = self.mock_handler.get_logs_by_worker(worker_id)
                worker_errors = [log for log in worker_logs if log['level'] == 'ERROR']

                if len(worker_errors) >= 1:
                    details.append(f"Worker {worker_id}: ✅ {len(worker_errors)} erreur(s)")
                else:
                    details.append(f"Worker {worker_id}: ❌ Aucune erreur loggée")
                    success = False

            self.log_test_result(test_name, success, "; ".join(details))

        except Exception as e:
            self.log_test_result(test_name, False, f"Erreur: {e}")

    def test_all_workers_can_log_warnings(self):
        """Test 3: Tous les workers peuvent loguer des warnings."""
        test_name = "Tous workers loggent les warnings"

        try:
            self.mock_handler.clear()
            loggers = []

            # Créer des loggers pour 4 workers
            for worker_id in range(4):
                smart_logger = create_smart_logger(self.test_logger, worker_id, 4)
                loggers.append(smart_logger)

            # Chaque worker log un warning unique
            for i, logger in enumerate(loggers):
                logger.warning(f"Avertissement du worker {i}", dedupe=False)

            time.sleep(0.1)

            # Vérifier que chaque worker a bien loggé
            success = True
            details = []

            for worker_id in range(4):
                worker_logs = self.mock_handler.get_logs_by_worker(worker_id)
                worker_warnings = [log for log in worker_logs if log['level'] == 'WARNING']

                if len(worker_warnings) >= 1:
                    details.append(f"Worker {worker_id}: ✅ {len(worker_warnings)} warning(s)")
                else:
                    details.append(f"Worker {worker_id}: ❌ Aucun warning loggé")
                    success = False

            self.log_test_result(test_name, success, "; ".join(details))

        except Exception as e:
            self.log_test_result(test_name, False, f"Erreur: {e}")

    def test_info_rotation_system(self):
        """Test 4: Système de rotation pour les logs informationnels."""
        test_name = "Rotation des logs informationnels"

        try:
            self.mock_handler.clear()
            loggers = []

            # Créer des loggers pour 4 workers
            for worker_id in range(4):
                smart_logger = create_smart_logger(self.test_logger, worker_id, 4)
                loggers.append(smart_logger)

            # Chaque worker essaie de loguer la même information avec rotation
            for round_num in range(2):  # Deux tours
                for i, logger in enumerate(loggers):
                    logger.info(f"Information partagée round {round_num}", rotate=True)
                time.sleep(0.1)  # Pause entre les rounds

            # Vérifier que seulement certains workers ont loggé (rotation)
            all_info_logs = self.mock_handler.get_logs_by_level('INFO')
            total_info_logs = len(all_info_logs)

            # Avec rotation, on devrait avoir moins de logs que workers * rounds
            expected_max = 4 * 2  # 4 workers * 2 rounds
            success = 0 < total_info_logs < expected_max

            # Compter combien de workers ont effectivement loggé
            workers_that_logged = set()
            for log in all_info_logs:
                if log['worker_id'] != -1:
                    workers_that_logged.add(log['worker_id'])

            details = f"Logs info: {total_info_logs}/{expected_max}, Workers actifs: {len(workers_that_logged)}/4"
            self.log_test_result(test_name, success, details)

        except Exception as e:
            self.log_test_result(test_name, False, f"Erreur: {e}")

    def test_deduplication_system(self):
        """Test 5: Système de déduplication."""
        test_name = "Déduplication des logs identiques"

        try:
            self.mock_handler.clear()

            # Créer un seul logger
            smart_logger = create_smart_logger(self.test_logger, 0, 4)

            # Loguer le même message plusieurs fois rapidement
            duplicate_message = "Message dupliqué pour test"
            for i in range(5):
                smart_logger.warning(duplicate_message, dedupe=True)
                time.sleep(0.1)  # Pause très courte

            # Vérifier qu'il y a moins de 5 logs (déduplication)
            warning_logs = self.mock_handler.get_logs_by_level('WARNING')
            duplicate_warnings = [log for log in warning_logs if duplicate_message in log['message']]

            success = len(duplicate_warnings) < 5
            details = f"Messages dupliqués: {len(duplicate_warnings)}/5 (déduplication {'✅' if success else '❌'})"

            self.log_test_result(test_name, success, details)

        except Exception as e:
            self.log_test_result(test_name, False, f"Erreur: {e}")

    def test_debug_sampling(self):
        """Test 6: Système de sampling pour les logs de debug."""
        test_name = "Sampling des logs de debug"

        try:
            self.mock_handler.clear()

            # Créer un logger avec sampling bas
            smart_logger = create_smart_logger(self.test_logger, 0, 4)
            smart_logger.default_sample_rate = 0.3  # 30% de sampling

            # Loguer beaucoup de messages de debug
            debug_messages = []
            for i in range(20):
                message = f"Debug message {i}"
                debug_messages.append(message)
                smart_logger.debug(message, sample_rate=0.3)

            time.sleep(0.1)

            # Vérifier que moins de 20 messages ont été loggés (sampling)
            debug_logs = self.mock_handler.get_logs_by_level('DEBUG')
            success = len(debug_logs) < 20

            # Calculer le taux de sampling effectif
            actual_rate = len(debug_logs) / 20 if debug_logs else 0
            details = f"Debug logs: {len(debug_logs)}/20 (taux: {actual_rate:.1%}, attendu: ~30%)"

            self.log_test_result(test_name, success, details)

        except Exception as e:
            self.log_test_result(test_name, False, f"Erreur: {e}")

    def test_logging_statistics(self):
        """Test 7: Collecte des statistiques de logging."""
        test_name = "Statistiques de logging"

        try:
            # Créer un logger
            smart_logger = create_smart_logger(self.test_logger, 0, 4)

            # Effectuer différents types de logs
            smart_logger.error("Test erreur")
            smart_logger.warning("Test warning")
            smart_logger.info("Test info")
            smart_logger.debug("Test debug", sample_rate=1.0)  # 100% pour être sûr

            time.sleep(0.1)

            # Récupérer les statistiques
            stats = smart_logger.get_stats()

            # Vérifier que les statistiques sont cohérentes
            success = (
                stats['worker_id'] == 0 and
                stats['total_logged'] > 0 and
                'by_level' in stats and
                'total_filtered' in stats
            )

            details = f"Worker {stats['worker_id']}: {stats['total_logged']} loggés, {stats['total_filtered']} filtrés"
            self.log_test_result(test_name, success, details)

        except Exception as e:
            self.log_test_result(test_name, False, f"Erreur: {e}")

    def test_concurrent_logging(self):
        """Test 8: Logging concurrent entre workers."""
        test_name = "Logging concurrent multi-worker"

        try:
            self.mock_handler.clear()

            def worker_logging_task(worker_id: int):
                """Tâche de logging pour un worker."""
                smart_logger = create_smart_logger(self.test_logger, worker_id, 4)

                # Chaque worker fait différents types de logs
                for i in range(5):
                    smart_logger.error(f"Worker {worker_id} erreur {i}", dedupe=False)
                    smart_logger.info(f"Worker {worker_id} info {i}")
                    time.sleep(0.05)  # Petite pause

            # Lancer 4 threads simultanément
            threads = []
            for worker_id in range(4):
                thread = threading.Thread(target=worker_logging_task, args=(worker_id,))
                threads.append(thread)
                thread.start()

            # Attendre que tous les threads finissent
            for thread in threads:
                thread.join()

            time.sleep(0.2)  # Attendre le traitement des logs

            # Vérifier que tous les workers ont loggé
            success = True
            details = []

            for worker_id in range(4):
                worker_logs = self.mock_handler.get_logs_by_worker(worker_id)
                if worker_logs:
                    details.append(f"Worker {worker_id}: {len(worker_logs)} logs")
                else:
                    details.append(f"Worker {worker_id}: ❌ Aucun log")
                    success = False

            self.log_test_result(test_name, success, "; ".join(details))

        except Exception as e:
            self.log_test_result(test_name, False, f"Erreur: {e}")

    def test_old_vs_new_system_comparison(self):
        """Test 9: Comparaison ancien vs nouveau système."""
        test_name = "Comparaison ancien vs nouveau système"

        try:
            self.mock_handler.clear()

            # Simuler l'ancien système (seulement worker 0 logue)
            old_system_logs = 0
            for worker_id in range(4):
                if worker_id == 0:  # Ancien système
                    logger = create_smart_logger(self.test_logger, worker_id, 4)
                    logger.error(f"Ancien système - Worker {worker_id}")
                    old_system_logs += 1

            # Nouveau système (tous les workers loggent)
            new_system_logs = 0
            for worker_id in range(4):
                logger = create_smart_logger(self.test_logger, worker_id, 4)
                logger.error(f"Nouveau système - Worker {worker_id}", dedupe=False)
                new_system_logs += 1

            time.sleep(0.1)

            # Vérifier l'amélioration
            total_errors = len(self.mock_handler.get_logs_by_level('ERROR'))
            success = new_system_logs > old_system_logs and total_errors > old_system_logs

            details = f"Ancien: {old_system_logs} workers actifs, Nouveau: {new_system_logs} workers actifs, Total logs: {total_errors}"
            self.log_test_result(test_name, success, details)

        except Exception as e:
            self.log_test_result(test_name, False, f"Erreur: {e}")

    def run_all_tests(self):
        """Lance tous les tests."""
        print("🧪 TEST DU SYSTÈME DE LOGGING INTELLIGENT")
        print("=" * 60)

        # Lancer tous les tests
        self.test_basic_smart_logger_creation()
        self.test_all_workers_can_log_errors()
        self.test_all_workers_can_log_warnings()
        self.test_info_rotation_system()
        self.test_deduplication_system()
        self.test_debug_sampling()
        self.test_logging_statistics()
        self.test_concurrent_logging()
        self.test_old_vs_new_system_comparison()

        # Résumé des résultats
        print("\n📊 RÉSULTATS DES TESTS")
        print("=" * 30)

        total_tests = self.results['tests_passed'] + self.results['tests_failed']
        success_rate = (self.results['tests_passed'] / total_tests * 100) if total_tests > 0 else 0

        print(f"Tests réussis: {self.results['tests_passed']}/{total_tests}")
        print(f"Tests échoués: {self.results['tests_failed']}/{total_tests}")
        print(f"Taux de réussite: {success_rate:.1f}%")

        if self.results['tests_failed'] == 0:
            print("\n🎉 TOUS LES TESTS SONT PASSÉS!")
            print("Le système de logging intelligent fonctionne correctement.")
            print("✅ Problème #4 (Logs restreints au Worker 0) est RÉSOLU!")
        else:
            print(f"\n⚠️  {self.results['tests_failed']} test(s) ont échoué.")
            print("Vérifiez les détails ci-dessus pour diagnostic.")

        return self.results['tests_failed'] == 0


def main():
    """Fonction principale."""
    test_runner = WorkerLoggingTest()
    success = test_runner.run_all_tests()

    if success:
        print("\n🎯 Le système de logging multi-worker fonctionne parfaitement!")
        return 0
    else:
        print("\n❌ Des problèmes ont été détectés dans le système de logging.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
