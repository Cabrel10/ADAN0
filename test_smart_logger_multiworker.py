#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test du système SmartLogger Multi-Workers
=========================================

Ce script valide que le système SmartLogger permet à tous les workers
de loguer intelligemment sans restriction excessive.

Test des problèmes corrigés :
- ✅ Tous les workers peuvent loguer les erreurs/warnings
- ✅ Déduplication intelligente des warnings
- ✅ Rotation des logs INFO entre workers
- ✅ Sampling des logs DEBUG

Auteur: Trading Bot Team
Date: 2024
"""

import logging
import sys
import time
import threading
from collections import defaultdict
from pathlib import Path

# Ajouter le chemin du bot pour les imports
sys.path.insert(0, str(Path(__file__).parent / "bot" / "src"))

try:
    from adan_trading_bot.utils.smart_logger import create_smart_logger
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("Vérifiez que le chemin vers smart_logger.py est correct")
    sys.exit(1)


class LogCapture:
    """Capture les logs pour analyse."""

    def __init__(self):
        self.logs = []
        self.logs_by_level = defaultdict(list)
        self.logs_by_worker = defaultdict(list)

    def add_log(self, level, message, worker_id=None):
        """Ajoute un log capturé."""
        log_entry = {
            'level': level,
            'message': message,
            'worker_id': worker_id,
            'timestamp': time.time()
        }
        self.logs.append(log_entry)
        self.logs_by_level[level].append(log_entry)
        if worker_id is not None:
            self.logs_by_worker[worker_id].append(log_entry)

    def get_stats(self):
        """Retourne des statistiques sur les logs capturés."""
        return {
            'total_logs': len(self.logs),
            'by_level': {level: len(logs) for level, logs in self.logs_by_level.items()},
            'by_worker': {worker_id: len(logs) for worker_id, logs in self.logs_by_worker.items()},
            'unique_workers': len(self.logs_by_worker)
        }


class MockLogger:
    """Logger simulé pour capturer les logs."""

    def __init__(self, log_capture):
        self.log_capture = log_capture

    def _extract_worker_id(self, message):
        """Extrait l'ID du worker du message."""
        if '[Worker ' in message:
            try:
                start = message.find('[Worker ') + 8
                end = message.find(']', start)
                return int(message[start:end])
            except:
                return None
        return None

    def critical(self, message, *args, **kwargs):
        worker_id = self._extract_worker_id(message)
        self.log_capture.add_log('CRITICAL', message, worker_id)

    def error(self, message, *args, **kwargs):
        worker_id = self._extract_worker_id(message)
        self.log_capture.add_log('ERROR', message, worker_id)

    def warning(self, message, *args, **kwargs):
        worker_id = self._extract_worker_id(message)
        self.log_capture.add_log('WARNING', message, worker_id)

    def info(self, message, *args, **kwargs):
        worker_id = self._extract_worker_id(message)
        self.log_capture.add_log('INFO', message, worker_id)

    def debug(self, message, *args, **kwargs):
        worker_id = self._extract_worker_id(message)
        self.log_capture.add_log('DEBUG', message, worker_id)


class SmartLoggerMultiWorkerTest:
    """Test complet du système SmartLogger Multi-Workers."""

    def __init__(self):
        self.log_capture = LogCapture()
        self.mock_logger = MockLogger(self.log_capture)
        self.results = {}

    def print_test_header(self, test_name):
        """Affiche un en-tête de test."""
        print(f"\n{'='*60}")
        print(f"🧪 TEST: {test_name}")
        print(f"{'='*60}")

    def print_test_result(self, test_name, passed, details=""):
        """Affiche le résultat d'un test."""
        status = "✅ RÉUSSI" if passed else "❌ ÉCHOUÉ"
        self.results[test_name] = passed
        print(f"{status} - {test_name}")
        if details:
            print(f"    📋 Détails: {details}")

    def test_all_workers_can_log_errors(self):
        """Test 1: Tous les workers peuvent loguer des erreurs."""
        self.print_test_header("Tous les workers peuvent loguer des ERREURS")

        # Nettoyer les logs précédents
        self.log_capture = LogCapture()
        self.mock_logger = MockLogger(self.log_capture)

        # Créer 4 workers et faire loguer des erreurs
        workers = [create_smart_logger(i, 4, f"test_worker_{i}") for i in range(4)]

        for i, smart_logger in enumerate(workers):
            smart_logger.smart_error(self.mock_logger, f"Erreur critique du worker {i}")

        # Vérifier que tous les workers ont loggé
        stats = self.log_capture.get_stats()

        expected_workers = 4
        expected_errors = 4

        passed = (
            stats['by_level']['ERROR'] == expected_errors and
            stats['unique_workers'] == expected_workers
        )

        details = f"Workers uniques: {stats['unique_workers']}/{expected_workers}, Erreurs: {stats['by_level']['ERROR']}/{expected_errors}"
        self.print_test_result("Tous workers loggent erreurs", passed, details)

        return passed

    def test_warning_deduplication(self):
        """Test 2: Déduplication des warnings."""
        self.print_test_header("Déduplication des WARNINGS")

        # Nettoyer les logs précédents
        self.log_capture = LogCapture()
        self.mock_logger = MockLogger(self.log_capture)

        workers = [create_smart_logger(i, 4, f"test_worker_{i}") for i in range(4)]

        # Tous les workers loggent le même warning
        duplicate_message = "Message warning dupliqué"
        for smart_logger in workers:
            smart_logger.smart_warning(self.mock_logger, duplicate_message)

        # Attendre un peu puis re-loguer (doit passer après fenêtre de dédup)
        time.sleep(1.1)  # Dépasser la fenêtre de dédup de 1s
        workers[0].smart_warning(self.mock_logger, duplicate_message)

        stats = self.log_capture.get_stats()

        # Doit avoir 2 warnings : 1 initial + 1 après fenêtre de dédup
        expected_warnings = 2
        passed = stats['by_level']['WARNING'] == expected_warnings

        details = f"Warnings uniques: {stats['by_level']['WARNING']}/{expected_warnings} (déduplication active)"
        self.print_test_result("Déduplication warnings", passed, details)

        return passed

    def test_info_rotation(self):
        """Test 3: Rotation des logs INFO entre workers."""
        self.print_test_header("Rotation des logs INFO entre workers")

        # Nettoyer les logs précédents
        self.log_capture = LogCapture()
        self.mock_logger = MockLogger(self.log_capture)

        workers = [create_smart_logger(i, 4, f"test_worker_{i}") for i in range(4)]

        # Loguer des messages INFO avec différents steps pour tester la rotation
        total_logs = 20
        for step in range(total_logs):
            for i, smart_logger in enumerate(workers):
                smart_logger.smart_info(self.mock_logger, f"Message INFO général step {step}", step)

        stats = self.log_capture.get_stats()

        # Avec la rotation, chaque worker doit loguer environ 1/4 des messages
        info_count = stats['by_level']['INFO']
        expected_range = (total_logs // 2, total_logs)  # Tolérance large

        passed = expected_range[0] <= info_count <= expected_range[1]

        details = f"Messages INFO: {info_count} (attendu: {expected_range[0]}-{expected_range[1]})"
        self.print_test_result("Rotation INFO", passed, details)

        return passed

    def test_critical_info_messages(self):
        """Test 4: Messages INFO critiques loggés par tous workers."""
        self.print_test_header("Messages INFO critiques (tous workers)")

        # Nettoyer les logs précédents
        self.log_capture = LogCapture()
        self.mock_logger = MockLogger(self.log_capture)

        workers = [create_smart_logger(i, 4, f"test_worker_{i}") for i in range(4)]

        # Messages contenant des mots-clés critiques
        critical_messages = [
            "Portfolio value updated",
            "Position opened successfully",
            "Trade completed with profit",
            "Error in trading logic"
        ]

        for message in critical_messages:
            for smart_logger in workers:
                smart_logger.smart_info(self.mock_logger, message)

        stats = self.log_capture.get_stats()

        # Tous les messages critiques doivent passer
        expected_infos = len(critical_messages) * 4  # 4 workers
        passed = stats['by_level']['INFO'] == expected_infos

        details = f"Messages INFO critiques: {stats['by_level']['INFO']}/{expected_infos}"
        self.print_test_result("Messages INFO critiques", passed, details)

        return passed

    def test_debug_sampling(self):
        """Test 5: Sampling des logs DEBUG."""
        self.print_test_header("Sampling des logs DEBUG")

        # Nettoyer les logs précédents
        self.log_capture = LogCapture()
        self.mock_logger = MockLogger(self.log_capture)

        smart_logger = create_smart_logger(0, 4, "test_worker_debug")

        # Générer beaucoup de messages DEBUG
        total_debug_messages = 100
        for i in range(total_debug_messages):
            smart_logger.smart_debug(self.mock_logger, f"Debug message {i}")

        stats = self.log_capture.get_stats()
        debug_count = stats['by_level']['DEBUG']

        # Avec sampling 10%, on attend environ 10% des messages
        expected_min = total_debug_messages * 0.02  # 2% tolérance basse
        expected_max = total_debug_messages * 0.20  # 20% tolérance haute

        passed = expected_min <= debug_count <= expected_max

        details = f"Messages DEBUG: {debug_count}/{total_debug_messages} (sampling ~{debug_count/total_debug_messages*100:.1f}%)"
        self.print_test_result("Sampling DEBUG", passed, details)

        return passed

    def test_concurrent_logging(self):
        """Test 6: Logging concurrent de plusieurs threads."""
        self.print_test_header("Logging concurrent multi-threads")

        # Nettoyer les logs précédents
        self.log_capture = LogCapture()
        self.mock_logger = MockLogger(self.log_capture)

        def worker_thread(worker_id, num_messages=10):
            """Thread simulant un worker loggant."""
            smart_logger = create_smart_logger(worker_id, 4, f"thread_worker_{worker_id}")

            for i in range(num_messages):
                smart_logger.smart_error(self.mock_logger, f"Concurrent error {i} from worker {worker_id}")
                smart_logger.smart_info(self.mock_logger, f"Concurrent info {i} from worker {worker_id}", i)
                time.sleep(0.01)  # Petite pause pour simuler le travail réel

        # Créer et lancer 4 threads workers
        threads = []
        num_messages_per_worker = 5

        for worker_id in range(4):
            thread = threading.Thread(target=worker_thread, args=(worker_id, num_messages_per_worker))
            threads.append(thread)
            thread.start()

        # Attendre que tous les threads finissent
        for thread in threads:
            thread.join()

        stats = self.log_capture.get_stats()

        # Vérifier qu'on a des logs de tous les workers
        expected_errors = 4 * num_messages_per_worker  # Tous les errors doivent passer
        passed = (
            stats['by_level']['ERROR'] == expected_errors and
            stats['unique_workers'] == 4 and
            stats['by_level']['INFO'] > 0  # Au moins quelques INFO avec rotation
        )

        details = f"Erreurs: {stats['by_level']['ERROR']}/{expected_errors}, Workers: {stats['unique_workers']}/4, INFO: {stats['by_level']['INFO']}"
        self.print_test_result("Logging concurrent", passed, details)

        return passed

    def test_old_vs_new_comparison(self):
        """Test 7: Comparaison ancien système vs nouveau."""
        self.print_test_header("Comparaison ancien vs nouveau système")

        # Simuler l'ancien système (seulement worker 0)
        old_logs = 0
        for worker_id in range(4):
            if worker_id == 0:  # Ancien système: seulement worker 0
                old_logs += 10

        # Nouveau système avec SmartLogger
        self.log_capture = LogCapture()
        self.mock_logger = MockLogger(self.log_capture)

        workers = [create_smart_logger(i, 4, f"comparison_worker_{i}") for i in range(4)]

        for smart_logger in workers:
            for i in range(10):
                smart_logger.smart_error(self.mock_logger, f"Error message {i}")

        stats = self.log_capture.get_stats()
        new_logs = stats['by_level']['ERROR']

        # Le nouveau système doit loguer plus (tous les workers pour les erreurs)
        improvement = new_logs > old_logs

        details = f"Ancien: {old_logs} logs, Nouveau: {new_logs} logs (amélioration: {new_logs/old_logs:.1f}x)"
        self.print_test_result("Amélioration vs ancien système", improvement, details)

        return improvement

    def run_all_tests(self):
        """Lance tous les tests et affiche un résumé."""
        print(f"\n🚀 DÉMARRAGE DES TESTS SMART LOGGER MULTI-WORKERS")
        print(f"{'='*80}")

        # Liste des tests à exécuter
        tests = [
            ("Erreurs tous workers", self.test_all_workers_can_log_errors),
            ("Déduplication warnings", self.test_warning_deduplication),
            ("Rotation INFO", self.test_info_rotation),
            ("Messages INFO critiques", self.test_critical_info_messages),
            ("Sampling DEBUG", self.test_debug_sampling),
            ("Logging concurrent", self.test_concurrent_logging),
            ("Comparaison ancien/nouveau", self.test_old_vs_new_comparison)
        ]

        # Exécuter tous les tests
        start_time = time.time()

        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                print(f"❌ ERREUR dans {test_name}: {e}")
                self.results[test_name] = False

        duration = time.time() - start_time

        # Résumé des résultats
        passed = sum(1 for result in self.results.values() if result)
        total = len(self.results)
        success_rate = (passed / total) * 100 if total > 0 else 0

        print(f"\n{'='*80}")
        print(f"📊 RÉSUMÉ DES TESTS")
        print(f"{'='*80}")
        print(f"✅ Tests réussis: {passed}/{total} ({success_rate:.1f}%)")
        print(f"⏱️  Durée: {duration:.2f}s")

        if success_rate >= 85:
            print(f"\n🎉 SUCCÈS! Le système SmartLogger fonctionne correctement!")
            print(f"✅ Problème #4 'Logs Restreints au Worker 0' est RÉSOLU!")
        else:
            print(f"\n⚠️  ATTENTION: Certains tests ont échoué ({success_rate:.1f}% de réussite)")

        # Détails des échecs
        failed_tests = [name for name, result in self.results.items() if not result]
        if failed_tests:
            print(f"\n❌ Tests échoués:")
            for test in failed_tests:
                print(f"   - {test}")

        return success_rate >= 85


def main():
    """Point d'entrée principal."""
    print("🔧 Test du Système SmartLogger Multi-Workers")
    print("=" * 50)

    # Exécuter les tests
    tester = SmartLoggerMultiWorkerTest()
    success = tester.run_all_tests()

    # Code de sortie
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
