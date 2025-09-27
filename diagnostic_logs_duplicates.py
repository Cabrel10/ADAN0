#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de diagnostic pour identifier la source des doublons de logs dans le bot de trading.

Ce script analyse les logs pour détecter :
1. Les doublons exacts (même timestamp, même contenu)
2. Les patterns de duplication par worker
3. Les appels multiples vs instances multiples
4. Les problèmes de filtrage worker_id
"""

import subprocess
import sys
import time
import re
import logging
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogDuplicationDiagnostic:
    def __init__(self):
        self.log_patterns = {
            'RISK': r'\[RISK\]',
            'METRICS_DEBUG': r'\[METRICS DEBUG\]',
            'POSITION_OPEN': r'\[POSITION_OPEN\]|\[POSITION OUVERTE\]',
            'DATA_LOADER': r'\[DATA_LOADER\]',
            'DBE_DECISION': r'\[DBE_DECISION\]|\[DBE CONTINUITY\]'
        }

        self.duplicates_by_pattern = defaultdict(list)
        self.worker_analysis = defaultdict(lambda: defaultdict(int))
        self.timestamp_analysis = defaultdict(list)

    def extract_log_info(self, line):
        """Extrait les informations importantes d'une ligne de log."""
        # Extraction du timestamp
        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.]?(\d{3})?', line)
        timestamp = timestamp_match.group(1) if timestamp_match else "unknown"

        # Extraction du worker_id s'il existe
        worker_match = re.search(r'Worker (\d+)', line)
        worker_id = worker_match.group(1) if worker_match else "unknown"

        # Extraction du pattern de log
        pattern_found = None
        for pattern_name, pattern_regex in self.log_patterns.items():
            if re.search(pattern_regex, line):
                pattern_found = pattern_name
                break

        # Extraction du contenu principal (sans timestamp et module)
        content_match = re.search(r'- INFO - (.+)$', line)
        content = content_match.group(1) if content_match else line.strip()

        return {
            'timestamp': timestamp,
            'worker_id': worker_id,
            'pattern': pattern_found,
            'content': content,
            'full_line': line.strip()
        }

    def analyze_logs(self, logs_text):
        """Analyse les logs pour détecter les doublons."""
        lines = logs_text.split('\n')

        # Regrouper par timestamp et pattern
        logs_by_time_pattern = defaultdict(list)

        for line in lines:
            if not line.strip():
                continue

            log_info = self.extract_log_info(line)

            if log_info['pattern']:
                key = f"{log_info['timestamp']}_{log_info['pattern']}"
                logs_by_time_pattern[key].append(log_info)

                # Analyse par worker
                self.worker_analysis[log_info['worker_id']][log_info['pattern']] += 1

        # Détecter les doublons
        for key, log_group in logs_by_time_pattern.items():
            if len(log_group) > 1:
                # Vérifier si c'est un vrai doublon (même contenu)
                contents = [log['content'] for log in log_group]
                content_counter = Counter(contents)

                for content, count in content_counter.items():
                    if count > 1:
                        timestamp = log_group[0]['timestamp']
                        pattern = log_group[0]['pattern']
                        workers = [log['worker_id'] for log in log_group if log['content'] == content]

                        self.duplicates_by_pattern[pattern].append({
                            'timestamp': timestamp,
                            'content': content,
                            'count': count,
                            'workers': workers,
                            'full_logs': [log['full_line'] for log in log_group if log['content'] == content]
                        })

    def print_analysis(self):
        """Affiche l'analyse des doublons."""
        logger.info("=" * 80)
        logger.info("🔍 DIAGNOSTIC DES DOUBLONS DE LOGS")
        logger.info("=" * 80)

        # Analyse globale
        total_duplicates = sum(len(dups) for dups in self.duplicates_by_pattern.values())
        if total_duplicates == 0:
            logger.info("✅ Aucun doublon détecté!")
            return True

        logger.info(f"❌ {total_duplicates} groupes de doublons détectés")

        # Analyse par pattern
        for pattern, duplicates in self.duplicates_by_pattern.items():
            if duplicates:
                logger.info(f"\n📊 Pattern [{pattern}]: {len(duplicates)} groupes de doublons")

                # Statistiques des doublons
                duplicate_counts = [dup['count'] for dup in duplicates]
                max_duplicates = max(duplicate_counts)
                avg_duplicates = sum(duplicate_counts) / len(duplicate_counts)

                logger.info(f"   • Max doublons par timestamp: {max_duplicates}")
                logger.info(f"   • Moyenne doublons par timestamp: {avg_duplicates:.1f}")

                # Analyse des workers impliqués
                all_workers = []
                for dup in duplicates:
                    all_workers.extend(dup['workers'])

                worker_counter = Counter(all_workers)
                logger.info(f"   • Workers impliqués: {dict(worker_counter)}")

                # Exemples de doublons
                logger.info("   • Exemples:")
                for i, dup in enumerate(duplicates[:3]):  # Afficher 3 exemples max
                    logger.info(f"     [{i+1}] {dup['timestamp']}: {dup['count']} occurrences")
                    logger.info(f"         Workers: {dup['workers']}")
                    logger.info(f"         Contenu: {dup['content'][:100]}...")

        # Analyse par worker
        logger.info("\n👥 ANALYSE PAR WORKER:")
        for worker_id, patterns in self.worker_analysis.items():
            total_logs = sum(patterns.values())
            logger.info(f"   Worker {worker_id}: {total_logs} logs")
            for pattern, count in patterns.items():
                logger.info(f"     • {pattern}: {count}")

        # Diagnostic des causes probables
        logger.info("\n🔬 DIAGNOSTIC DES CAUSES:")

        # Vérifier si c'est un problème de worker filtering
        worker_0_only = all(
            all(worker in ['0', 'unknown'] for workers in dup['workers'] for worker in workers)
            for duplicates in self.duplicates_by_pattern.values()
            for dup in duplicates
        )

        if not worker_0_only:
            logger.warning("   ⚠️  PROBLÈME: Des logs proviennent de workers autres que 0")
            logger.warning("   🔧 SOLUTION: Vérifier le filtrage worker_id dans le code")
        else:
            logger.info("   ✅ Filtrage worker_id correct: tous les logs viennent du worker 0")
            logger.warning("   ⚠️  PROBLÈME: Appels multiples même avec filtrage correct")
            logger.warning("   🔧 SOLUTIONS POSSIBLES:")
            logger.warning("      • Multiples instances du même worker")
            logger.warning("      • Méthodes appelées plusieurs fois")
            logger.warning("      • Problème de multiprocessing/threading")

        # Vérifier les timestamps pour détecter les appels rapprochés
        rapid_calls = []
        for pattern, duplicates in self.duplicates_by_pattern.items():
            for dup in duplicates:
                if dup['count'] >= 4:  # 4 doublons ou plus = suspect
                    rapid_calls.append((pattern, dup['timestamp'], dup['count']))

        if rapid_calls:
            logger.warning(f"\n   ⚠️  {len(rapid_calls)} cas d'appels multiples rapides détectés:")
            for pattern, timestamp, count in rapid_calls[:5]:
                logger.warning(f"      • {pattern} à {timestamp}: {count} appels")

        return False

    def suggest_fixes(self):
        """Suggère des corrections basées sur l'analyse."""
        logger.info("\n🔧 SUGGESTIONS DE CORRECTIONS:")

        if self.duplicates_by_pattern:
            logger.info("1. Ajouter un verrou (lock) autour des logs critiques:")
            logger.info("   ```python")
            logger.info("   import threading")
            logger.info("   log_lock = threading.Lock()")
            logger.info("   ")
            logger.info("   def log_info(self, message):")
            logger.info("       with log_lock:")
            logger.info("           if self.worker_id == 0:")
            logger.info("               logger.info(f'[Worker {self.worker_id}] {message}')")
            logger.info("   ```")

            logger.info("\n2. Ajouter un cache de logs pour éviter les répétitions:")
            logger.info("   ```python")
            logger.info("   def __init__(self):")
            logger.info("       self._last_logs = {}")
            logger.info("   ")
            logger.info("   def log_info(self, message):")
            logger.info("       if self.worker_id == 0:")
            logger.info("           current_time = time.time()")
            logger.info("           if message not in self._last_logs or current_time - self._last_logs[message] > 1:")
            logger.info("               logger.info(f'[Worker {self.worker_id}] {message}')")
            logger.info("               self._last_logs[message] = current_time")
            logger.info("   ```")

            logger.info("\n3. Utiliser un décorateur pour éviter les doublons:")
            logger.info("   ```python")
            logger.info("   from functools import wraps")
            logger.info("   ")
            logger.info("   def no_duplicate_logs(func):")
            logger.info("       @wraps(func)")
            logger.info("       def wrapper(self, *args, **kwargs):")
            logger.info("           if not hasattr(self, '_calling_log'):")
            logger.info("               self._calling_log = True")
            logger.info("               try:")
            logger.info("                   return func(self, *args, **kwargs)")
            logger.info("               finally:")
            logger.info("                   self._calling_log = False")
            logger.info("       return wrapper")
            logger.info("   ```")

def run_diagnostic_test():
    """Lance un test de diagnostic en temps réel."""
    logger.info("🚀 Lancement du diagnostic des doublons de logs")

    # Lancer l'entraînement avec capture des logs
    cmd = [
        "timeout", "20s",
        "/home/morningstar/miniconda3/envs/trading_env/bin/python",
        "bot/scripts/train_parallel_agents.py",
        "--config", "bot/config/config.yaml",
        "--checkpoint-dir", "bot/checkpoints"
    ]

    try:
        logger.info("📊 Capture des logs en cours...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="/home/morningstar/Documents/trading"
        )

        all_output = result.stdout + result.stderr

        if not all_output:
            logger.error("❌ Aucune sortie capturée")
            return

        # Analyser les logs
        diagnostic = LogDuplicationDiagnostic()
        diagnostic.analyze_logs(all_output)
        success = diagnostic.print_analysis()

        if not success:
            diagnostic.suggest_fixes()

        # Sauvegarder les logs pour analyse manuelle
        log_file = Path("diagnostic_logs_duplicates_output.txt")
        with open(log_file, 'w') as f:
            f.write(all_output)
        logger.info(f"📁 Logs complets sauvegardés dans: {log_file}")

    except Exception as e:
        logger.error(f"Erreur lors du diagnostic: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Fonction principale du script de diagnostic."""
    logger.info("🧪 DIAGNOSTIC DES DOUBLONS DE LOGS")
    logger.info("=" * 60)

    # Vérifier que nous sommes dans le bon répertoire
    if not Path("bot/scripts/train_parallel_agents.py").exists():
        logger.error("❌ Script d'entraînement introuvable. Vérifiez le répertoire de travail.")
        sys.exit(1)

    run_diagnostic_test()
    logger.info("✅ Diagnostic terminé!")

if __name__ == "__main__":
    main()
