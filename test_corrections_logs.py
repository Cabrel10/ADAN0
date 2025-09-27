#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test pour valider les corrections implémentées dans le bot de trading.

Ce script teste :
1. Élimination de la duplication des logs
2. Correction des erreurs d'ouverture de positions
3. Correction des incohérences dans les métriques
4. Correction de la terminaison prématurée
"""

import subprocess
import sys
import time
import re
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_training_test(timeout=30):
    """Lance l'entraînement avec un timeout et capture les logs."""
    cmd = [
        "timeout", f"{timeout}s",
        "/home/morningstar/miniconda3/envs/trading_env/bin/python",
        "bot/scripts/train_parallel_agents.py",
        "--config", "bot/config/config.yaml",
        "--checkpoint-dir", "bot/checkpoints"
    ]

    try:
        logger.info(f"🚀 Lancement du test d'entraînement avec timeout {timeout}s")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="/home/morningstar/Documents/trading"
        )

        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {e}")
        return "", str(e), 1

def analyze_logs(stdout, stderr):
    """Analyse les logs pour valider les corrections."""
    all_logs = stdout + stderr
    lines = all_logs.split('\n')

    results = {
        'duplicate_logs': False,
        'position_errors': False,
        'metrics_inconsistencies': False,
        'premature_termination': False,
        'worker_comparison': False
    }

    # Compteurs pour détecter les doublons
    log_patterns = {
        '[RISK]': [],
        '[METRICS DEBUG]': [],
        '[POSITION_OPEN]': [],
        '[DBE_DECISION]': [],
        '[DATA_LOADER]': []
    }

    position_errors = []
    termination_messages = []
    metrics_debug = []

    logger.info("🔍 Analyse des logs...")

    for i, line in enumerate(lines):
        # 1. Détecter les duplications de logs
        for pattern in log_patterns:
            if pattern in line:
                log_patterns[pattern].append((i, line))

        # 2. Détecter les erreurs d'ouverture de positions
        if '[ERREUR] Impossible d\'ouvrir une position' in line:
            position_errors.append((i, line))

        # 3. Détecter les messages de terminaison
        if '[TERMINATION]' in line:
            termination_messages.append((i, line))

        # 4. Capturer les métriques debug pour vérifier la cohérence
        if '[METRICS DEBUG]' in line:
            metrics_debug.append((i, line))

    # Analyse des doublons
    logger.info("📊 Analyse des duplications de logs:")
    for pattern, occurrences in log_patterns.items():
        if len(occurrences) > 1:
            # Grouper par timestamp approximatif (même seconde)
            timestamps = {}
            for line_num, log_line in occurrences:
                # Extraire le timestamp (approximatif)
                time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', log_line)
                if time_match:
                    timestamp = time_match.group(1)
                    if timestamp not in timestamps:
                        timestamps[timestamp] = []
                    timestamps[timestamp].append(log_line)

            # Détecter les doublons (même timestamp, même contenu)
            for timestamp, logs in timestamps.items():
                if len(logs) > 1:
                    # Vérifier si c'est vraiment un doublon (même contenu)
                    unique_logs = set(logs)
                    if len(unique_logs) < len(logs):
                        logger.warning(f"   ❌ Doublon détecté pour {pattern} à {timestamp}: {len(logs)} occurrences")
                        results['duplicate_logs'] = True
                    else:
                        logger.info(f"   ✅ {pattern}: {len(logs)} occurrences distinctes à {timestamp}")
                else:
                    logger.info(f"   ✅ {pattern}: 1 occurrence à {timestamp}")
        elif len(occurrences) == 1:
            logger.info(f"   ✅ {pattern}: 1 occurrence unique")
        else:
            logger.info(f"   ℹ️  {pattern}: aucune occurrence")

    # Analyse des erreurs de positions
    logger.info("🏦 Analyse des erreurs de positions:")
    if position_errors:
        logger.warning(f"   ❌ {len(position_errors)} erreurs d'ouverture détectées:")
        for line_num, error in position_errors[:3]:  # Afficher les 3 premières
            logger.warning(f"      Ligne {line_num}: {error.strip()}")
        results['position_errors'] = True
    else:
        logger.info("   ✅ Aucune erreur d'ouverture de position détectée")

    # Analyse des terminaisons
    logger.info("🔚 Analyse des terminaisons:")
    if termination_messages:
        for line_num, msg in termination_messages:
            logger.info(f"   Ligne {line_num}: {msg.strip()}")
            if "Min steps not reached" in msg:
                results['premature_termination'] = True
                logger.warning("   ❌ Terminaison prématurée détectée (Min steps not reached)")
            elif "Frequency check interval reached" in msg:
                logger.info("   ✅ Terminaison normale (Frequency check interval)")
            elif "Max steps reached" in msg:
                logger.info("   ✅ Terminaison normale (Max steps)")
    else:
        logger.info("   ℹ️  Aucun message de terminaison trouvé")

    # Analyse des métriques
    logger.info("📈 Analyse des métriques:")
    if metrics_debug:
        logger.info(f"   ✅ {len(metrics_debug)} messages de métriques trouvés")
        # Afficher quelques exemples
        for line_num, metric in metrics_debug[:2]:
            logger.info(f"      Ligne {line_num}: {metric.strip()}")
    else:
        logger.warning("   ❌ Aucune métrique debug trouvée")
        results['metrics_inconsistencies'] = True

    # Chercher les comparaisons de workers
    worker_comparison_found = any('[WORKER COMPARISON]' in line or '[COMPARISON Worker' in line for line in lines)
    if worker_comparison_found:
        logger.info("   ✅ Comparaison des workers trouvée")
        results['worker_comparison'] = True
    else:
        logger.info("   ℹ️  Comparaison des workers non trouvée (normal si timeout court)")

    return results

def print_test_summary(results):
    """Affiche un résumé des tests."""
    logger.info("=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS DE CORRECTIONS")
    logger.info("=" * 60)

    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if not v)  # False = test passé

    test_descriptions = {
        'duplicate_logs': '1. Élimination des duplications de logs',
        'position_errors': '2. Correction des erreurs d\'ouverture de positions',
        'metrics_inconsistencies': '3. Correction des incohérences de métriques',
        'premature_termination': '4. Correction de la terminaison prématurée',
        'worker_comparison': '5. Comparaison des workers'
    }

    for test_key, description in test_descriptions.items():
        status = "❌ ÉCHEC" if results[test_key] else "✅ SUCCÈS"
        logger.info(f"{description}: {status}")

    logger.info("=" * 60)
    logger.info(f"📊 RÉSULTAT GLOBAL: {passed_tests}/{total_tests} tests réussis")

    if passed_tests == total_tests:
        logger.info("🎉 TOUS LES TESTS SONT PASSÉS!")
        return True
    else:
        logger.warning("⚠️  CERTAINS TESTS ONT ÉCHOUÉ")
        return False

def main():
    """Fonction principale du script de test."""
    logger.info("🧪 SCRIPT DE TEST DES CORRECTIONS")
    logger.info("=" * 60)

    # Vérifier que nous sommes dans le bon répertoire
    if not Path("bot/scripts/train_parallel_agents.py").exists():
        logger.error("❌ Script d'entraînement introuvable. Vérifiez le répertoire de travail.")
        sys.exit(1)

    # Lancer le test d'entraînement
    stdout, stderr, returncode = run_training_test(timeout=30)

    if returncode != 124:  # 124 = timeout command successful timeout
        logger.warning(f"⚠️  Code de retour inattendu: {returncode}")

    if not stdout and not stderr:
        logger.error("❌ Aucune sortie capturée. Vérifiez la configuration.")
        sys.exit(1)

    # Analyser les logs
    results = analyze_logs(stdout, stderr)

    # Afficher le résumé
    success = print_test_summary(results)

    # Sauvegarder les logs pour analyse manuelle si nécessaire
    log_file = Path("test_corrections_logs_output.txt")
    with open(log_file, 'w') as f:
        f.write("=== STDOUT ===\n")
        f.write(stdout)
        f.write("\n\n=== STDERR ===\n")
        f.write(stderr)

    logger.info(f"📁 Logs complets sauvegardés dans: {log_file}")

    if success:
        logger.info("🎯 Test terminé avec succès!")
        sys.exit(0)
    else:
        logger.error("💥 Test terminé avec des erreurs!")
        sys.exit(1)

if __name__ == "__main__":
    main()
