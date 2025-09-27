#!/usr/bin/env python3
"""
Test rapide pour vérifier que les corrections d'indexation ont supprimé
les erreurs EXCESSIVE_FORWARD_FILL.

Ce script lance un entraînement court et surveille les logs pour confirmer :
1. Plus d'erreurs EXCESSIVE_FORWARD_FILL
2. Les prix évoluent correctement (pas de données statiques)
3. L'entraînement progresse normalement
4. Les métriques sont calculées sans crash

Usage:
    cd trading/
    python test_final_indexing_fix.py
"""

import sys
import os
import time
import subprocess
import threading
import re
from datetime import datetime

class TrainingMonitor:
    """Monitore l'entraînement pour détecter les problèmes et succès."""

    def __init__(self):
        self.start_time = time.time()
        self.errors = {
            'excessive_forward_fill': 0,
            'missing_price': 0,
            'data_quality_issue': 0,
            'keyboard_interrupt': 0,
            'memory_error': 0
        }
        self.success_indicators = {
            'portfolio_values': [],
            'rewards_calculated': 0,
            'steps_completed': 0,
            'trades_executed': 0,
            'price_changes': set()
        }
        self.last_portfolio_value = None
        self.training_process = None

    def parse_log_line(self, line):
        """Parse une ligne de log pour extraire les informations importantes."""
        line = line.strip()

        # Détecter les erreurs critiques
        if "EXCESSIVE_FORWARD_FILL" in line:
            self.errors['excessive_forward_fill'] += 1

        if "MISSING_PRICE" in line:
            self.errors['missing_price'] += 1

        if "DATA_QUALITY_ISSUE" in line:
            self.errors['data_quality_issue'] += 1

        if "KeyboardInterrupt" in line:
            self.errors['keyboard_interrupt'] += 1

        if "MemoryError" in line or "memory" in line.lower():
            self.errors['memory_error'] += 1

        # Détecter les indicateurs de succès
        if "Portfolio value:" in line:
            match = re.search(r'Portfolio value: ([\d.]+)', line)
            if match:
                value = float(match.group(1))
                self.success_indicators['portfolio_values'].append(value)

        if "REWARD Worker" in line and "Total:" in line:
            self.success_indicators['rewards_calculated'] += 1

        if "STEP" in line and "Executing step" in line:
            self.success_indicators['steps_completed'] += 1

        # Détecter les changements de prix (pour vérifier que les données ne sont pas statiques)
        price_match = re.search(r'price.*?(\d{4,}\.\d{2,})', line.lower())
        if price_match:
            price = price_match.group(1)
            self.success_indicators['price_changes'].add(price[:6])  # Garder 6 premiers chiffres

    def get_status(self):
        """Retourne le statut actuel du monitoring."""
        runtime = time.time() - self.start_time

        # Calculer les taux
        total_steps = max(1, self.success_indicators['steps_completed'])
        ff_rate = (self.errors['excessive_forward_fill'] / total_steps) * 100

        unique_prices = len(self.success_indicators['price_changes'])

        return {
            'runtime': runtime,
            'total_errors': sum(self.errors.values()),
            'critical_errors': self.errors['excessive_forward_fill'] + self.errors['data_quality_issue'],
            'forward_fill_rate': ff_rate,
            'steps_completed': total_steps,
            'unique_prices': unique_prices,
            'portfolio_changes': len(set(self.success_indicators['portfolio_values'])),
            'errors_detail': self.errors,
            'success_detail': self.success_indicators
        }

    def is_healthy(self):
        """Détermine si l'entraînement est en bonne santé."""
        status = self.get_status()

        # Critères de santé
        criteria = {
            'no_excessive_ff': status['errors_detail']['excessive_forward_fill'] == 0,
            'low_missing_prices': status['errors_detail']['missing_price'] < status['steps_completed'] * 0.05,  # < 5%
            'price_variation': status['unique_prices'] > 3,  # Au moins 3 prix différents
            'portfolio_changes': status['portfolio_changes'] > 1,  # Le portefeuille évolue
            'no_crashes': status['errors_detail']['keyboard_interrupt'] == 0 and status['errors_detail']['memory_error'] == 0
        }

        return criteria, all(criteria.values())

def run_training_test(duration_seconds=90):
    """Lance l'entraînement et le surveille pendant la durée spécifiée."""

    print("🚀 VALIDATION FINALE DES CORRECTIONS D'INDEXATION")
    print("=" * 60)
    print(f"⏱️  Durée du test: {duration_seconds}s")
    print(f"🎯 Objectif: Confirmer l'absence d'erreurs EXCESSIVE_FORWARD_FILL")
    print("=" * 60)

    monitor = TrainingMonitor()

    # Commande d'entraînement
    cmd = [
        "/home/morningstar/miniconda3/envs/trading_env/bin/python",
        "bot/scripts/train_parallel_agents.py",
        "--config", "bot/config/config.yaml",
        "--checkpoint-dir", "bot/checkpoints"
    ]

    try:
        # Lancer l'entraînement
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd="/home/morningstar/Documents/trading"
        )

        monitor.training_process = process

        print("📊 Monitoring en cours...")
        start_time = time.time()

        # Surveiller les logs en temps réel
        while time.time() - start_time < duration_seconds:
            try:
                line = process.stdout.readline()
                if line:
                    monitor.parse_log_line(line)

                    # Affichage périodique du statut
                    if monitor.success_indicators['steps_completed'] % 20 == 0 and monitor.success_indicators['steps_completed'] > 0:
                        status = monitor.get_status()
                        print(f"⚡ Steps: {status['steps_completed']} | "
                              f"FF Errors: {status['errors_detail']['excessive_forward_fill']} | "
                              f"Unique Prices: {status['unique_prices']} | "
                              f"Runtime: {status['runtime']:.1f}s")

                # Vérifier si le process est toujours en vie
                if process.poll() is not None:
                    break

            except Exception as e:
                print(f"⚠️  Erreur lors de la lecture des logs: {e}")
                break

        # Terminer le processus proprement
        if process.poll() is None:
            print("⏹️  Arrêt du training...")
            process.terminate()
            time.sleep(5)
            if process.poll() is None:
                process.kill()

    except Exception as e:
        print(f"❌ Erreur lors du lancement: {e}")
        return False

    # Analyse finale
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS DE LA VALIDATION")
    print("=" * 60)

    status = monitor.get_status()
    criteria, is_healthy = monitor.is_healthy()

    # Affichage des métriques
    print(f"⏱️  Durée d'exécution: {status['runtime']:.1f}s")
    print(f"📈 Steps complétés: {status['steps_completed']}")
    print(f"💰 Variations de portefeuille: {status['portfolio_changes']}")
    print(f"💹 Prix uniques détectés: {status['unique_prices']}")
    print()

    # Affichage des erreurs
    print("🔍 ANALYSE DES ERREURS:")
    for error_type, count in status['errors_detail'].items():
        status_icon = "✅" if count == 0 else "❌" if count > 5 else "⚠️"
        print(f"  {status_icon} {error_type.replace('_', ' ').title()}: {count}")
    print()

    # Affichage des critères de santé
    print("🎯 CRITÈRES DE VALIDATION:")
    for criterion, passed in criteria.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon} {criterion.replace('_', ' ').title()}: {'PASSED' if passed else 'FAILED'}")
    print()

    # Conclusion
    if is_healthy:
        print("🎉 VALIDATION RÉUSSIE!")
        print("✅ Les corrections d'indexation sont effectives")
        print("✅ Plus d'erreurs EXCESSIVE_FORWARD_FILL")
        print("✅ Les données évoluent correctement")
        print("✅ L'entraînement fonctionne normalement")
        print("\n🚀 Le système est prêt pour un entraînement complet!")
        return True
    else:
        print("⚠️  VALIDATION PARTIELLE")
        failed_criteria = [k for k, v in criteria.items() if not v]
        print(f"❌ Critères échoués: {', '.join(failed_criteria)}")
        print("\n🔧 Des ajustements supplémentaires peuvent être nécessaires.")
        return False

def main():
    """Fonction principale."""

    # Vérifier l'environnement
    if not os.path.exists("bot/scripts/train_parallel_agents.py"):
        print("❌ Script d'entraînement non trouvé. Exécutez depuis le répertoire trading/")
        sys.exit(1)

    # Lancer le test
    success = run_training_test(duration_seconds=120)  # 2 minutes de test

    # Code de sortie
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
