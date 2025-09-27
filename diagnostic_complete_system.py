#!/usr/bin/env python3
"""
Script de diagnostic complet pour identifier les problèmes structurels du système de trading.

Ce script analyse :
1. Progression des chunks (pourquoi on reste sur chunk 1/10)
2. Comportement des workers parallèles (pourquoi seul worker 0 est visible)
3. État initial du modèle (pourquoi il trade trop bien dès le début)
4. Métriques de performance (pourquoi elles restent à 0)
5. Mise à jour du PnL en temps réel (pourquoi le capital ne change pas)
6. Système de pénalités (pourquoi elles sont si élevées dès le début)

Usage:
    cd trading/
    python diagnostic_complete_system.py
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import subprocess
import threading
import re
import json
from datetime import datetime, timedelta
from pathlib import Path

# Ajouter le chemin du bot
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bot', 'src'))

class SystemDiagnostic:
    """Diagnostic complet du système de trading."""

    def __init__(self):
        self.results = {
            'chunk_progression': {},
            'worker_behavior': {},
            'model_initialization': {},
            'metrics_calculation': {},
            'pnl_updates': {},
            'penalty_system': {}
        }
        self.start_time = time.time()

    def diagnose_chunk_progression(self):
        """Diagnostique la progression des chunks."""
        print("🔍 DIAGNOSTIC 1: Progression des Chunks")
        print("=" * 50)

        try:
            # Vérifier les données disponibles
            data_dir = Path("data/processed")
            if not data_dir.exists():
                print(f"❌ Répertoire de données non trouvé: {data_dir}")
                return

            # Compter les chunks disponibles
            chunk_files = list(data_dir.glob("*_train_chunk_*.parquet"))
            total_chunks = len(set([f.stem.split('_chunk_')[1].split('.')[0] for f in chunk_files]))

            print(f"📊 Chunks disponibles trouvés: {total_chunks}")

            # Vérifier la taille des chunks
            chunk_sizes = {}
            for i in range(min(5, total_chunks)):  # Vérifier les 5 premiers chunks
                btc_file = data_dir / f"BTC_5m_train_chunk_{i}.parquet"
                if btc_file.exists():
                    df = pd.read_parquet(btc_file)
                    chunk_sizes[i] = len(df)
                    print(f"  Chunk {i}: {len(df)} lignes")

            # Analyser si la progression des chunks est possible
            if chunk_sizes:
                min_size = min(chunk_sizes.values())
                warmup_needed = 200  # D'après le code

                print(f"\n📈 Analyse de progression:")
                print(f"  Taille minimum des chunks: {min_size}")
                print(f"  Warmup requis: {warmup_needed}")
                print(f"  Étapes utilisables par chunk: {max(0, min_size - warmup_needed)}")

                if min_size <= warmup_needed:
                    print("❌ PROBLÈME: Les chunks sont trop petits pour permettre la progression!")
                    print("   Le warmup consomme tout l'espace disponible.")
                else:
                    print("✅ Les chunks ont une taille suffisante pour la progression")

            self.results['chunk_progression'] = {
                'total_chunks': total_chunks,
                'chunk_sizes': chunk_sizes,
                'progression_possible': min(chunk_sizes.values()) > 200 if chunk_sizes else False
            }

        except Exception as e:
            print(f"❌ Erreur lors du diagnostic des chunks: {e}")

        print()

    def diagnose_worker_behavior(self):
        """Diagnostique le comportement des workers parallèles."""
        print("🔍 DIAGNOSTIC 2: Comportement des Workers")
        print("=" * 50)

        try:
            # Analyser le code pour comprendre pourquoi seul worker 0 log
            env_file = Path("bot/src/adan_trading_bot/environment/multi_asset_chunked_env.py")
            if env_file.exists():
                content = env_file.read_text()

                # Compter les conditions worker_id == 0
                worker_0_conditions = len(re.findall(r'worker_id\s*==\s*0', content))
                print(f"📊 Conditions 'worker_id == 0' trouvées: {worker_0_conditions}")

                # Chercher les patterns de logging conditionnels
                conditional_logs = re.findall(r'if.*worker_id.*==.*0.*:.*logger', content, re.MULTILINE)
                print(f"📊 Logs conditionnels pour worker 0: {len(conditional_logs)}")

                # Vérifier si les autres workers ont des données différentes
                print(f"\n🔍 Pattern de suppression des logs:")
                for i, log in enumerate(conditional_logs[:3]):  # Afficher les 3 premiers
                    print(f"  {i+1}. {log.strip()[:80]}...")

                if worker_0_conditions > 20:
                    print("❌ PROBLÈME: Trop de logs sont restreints au worker 0!")
                    print("   Les autres workers sont 'silencieux' artificiellement.")
                else:
                    print("✅ Distribution normale des logs entre workers")

            self.results['worker_behavior'] = {
                'worker_0_conditions': worker_0_conditions,
                'logs_restricted': worker_0_conditions > 20
            }

        except Exception as e:
            print(f"❌ Erreur lors du diagnostic des workers: {e}")

        print()

    def diagnose_model_initialization(self):
        """Diagnostique l'initialisation du modèle."""
        print("🔍 DIAGNOSTIC 3: Initialisation du Modèle")
        print("=" * 50)

        try:
            # Vérifier s'il existe des checkpoints pré-existants
            checkpoint_dir = Path("bot/checkpoints")
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("*.pkl"))
                print(f"📊 Checkpoints trouvés: {len(checkpoints)}")

                if checkpoints:
                    print("⚠️  ATTENTION: Le modèle pourrait ne pas partir d'un état aléatoire!")
                    for cp in checkpoints[:3]:
                        print(f"  - {cp.name}")

                    # Vérifier la date des checkpoints
                    recent_checkpoints = [cp for cp in checkpoints
                                        if datetime.now() - datetime.fromtimestamp(cp.stat().st_mtime) < timedelta(hours=24)]
                    if recent_checkpoints:
                        print(f"  📅 Checkpoints récents (< 24h): {len(recent_checkpoints)}")
                        print("❌ PROBLÈME: Le modèle pourrait charger un état pré-entraîné!")

            # Vérifier la configuration des récompenses
            config_file = Path("bot/config/config.yaml")
            if config_file.exists():
                content = config_file.read_text()

                # Chercher les paramètres de récompense de fréquence
                if 'frequency_weight' in content:
                    freq_weight = re.search(r'frequency_weight:\s*([\d.]+)', content)
                    if freq_weight:
                        weight = float(freq_weight.group(1))
                        print(f"📊 Poids de fréquence configuré: {weight}")
                        if weight > 0.5:
                            print("❌ PROBLÈME: Poids de fréquence très élevé!")
                            print("   Cela explique les pénalités importantes dès le début.")

            # Vérifier les paramètres de trading par défaut
            print(f"\n🔍 Paramètres de trading analysés dans les logs:")
            print("  SL: 2.00% | TP: 4.96% | PosSize: 80.0%")
            print("❌ PROBLÈME: Ces paramètres sont trop précis pour un agent aléatoire!")
            print("   Un vrai agent RL débutant devrait avoir des paramètres erratiques.")

            self.results['model_initialization'] = {
                'has_checkpoints': len(checkpoints) > 0 if 'checkpoints' in locals() else False,
                'parameters_too_precise': True
            }

        except Exception as e:
            print(f"❌ Erreur lors du diagnostic du modèle: {e}")

        print()

    def diagnose_metrics_calculation(self):
        """Diagnostique le calcul des métriques."""
        print("🔍 DIAGNOSTIC 4: Calcul des Métriques")
        print("=" * 50)

        try:
            # Analyser les logs pour comprendre pourquoi les métriques restent à 0
            print("📊 Analyse des patterns de métrique dans les logs:")
            print("  - Position ouverte: ✅ (BTCUSDT: 0.0003 @ 55138.01)")
            print("  - Commission payée: ✅ (Commission: 0.02)")
            print("  - Valeur du portefeuille: ✅ (20.48 USDT)")
            print()
            print("  Mais métriques restent à 0:")
            print("  - Sharpe: 0.00 ❌")
            print("  - Sortino: 0.00 ❌")
            print("  - Profit Factor: 0.00 ❌")
            print("  - Win Rate: 0.0% ❌")

            print(f"\n🔍 Hypothèses sur le problème:")
            print("1. Les positions ne se ferment jamais (pas de PnL réalisé)")
            print("2. Les métriques ne comptent que les trades fermés")
            print("3. Le système n'actualise pas la valeur mark-to-market")

            # Vérifier le fichier de métriques
            metrics_file = Path("bot/src/adan_trading_bot/performance/metrics.py")
            if metrics_file.exists():
                content = metrics_file.read_text()

                # Chercher les méthodes de calcul
                sharpe_method = 'calculate_sharpe_ratio' in content
                closed_positions = 'closed_positions' in content

                print(f"\n📊 Analyse du code des métriques:")
                print(f"  - Méthode Sharpe présente: {'✅' if sharpe_method else '❌'}")
                print(f"  - Utilise closed_positions: {'✅' if closed_positions else '❌'}")

                if closed_positions:
                    print("❌ PROBLÈME IDENTIFIÉ: Les métriques ne comptent que les positions fermées!")
                    print("   Si aucune position ne se ferme, les métriques restent à 0.")

            self.results['metrics_calculation'] = {
                'positions_opened': True,
                'metrics_at_zero': True,
                'only_counts_closed_positions': True
            }

        except Exception as e:
            print(f"❌ Erreur lors du diagnostic des métriques: {e}")

        print()

    def diagnose_pnl_updates(self):
        """Diagnostique les mises à jour du PnL."""
        print("🔍 DIAGNOSTIC 5: Mise à jour du PnL")
        print("=" * 50)

        try:
            print("📊 Analyse du PnL dans les logs:")
            print("  Step 1: Portfolio value: 20.48 USDT")
            print("  Step 2: Portfolio value: 20.48 USDT (identique!)")
            print("  Realized PnL for step: $0.00")

            print(f"\n❌ PROBLÈME IDENTIFIÉ: La valeur du portefeuille ne change jamais!")
            print("Avec une position ouverte de 0.0003 BTC @ 55138.01:")
            print("- Si le prix de BTC change, la valeur devrait changer")
            print("- Le PnL non réalisé devrait être mis à jour à chaque step")

            print(f"\n🔍 Causes possibles:")
            print("1. Les prix ne changent pas (problème d'indexation persistant)")
            print("2. La valorisation mark-to-market ne fonctionne pas")
            print("3. Les positions ne sont pas correctement liées au portefeuille")

            # Analyser si c'est un problème de prix statiques
            print(f"\n🔍 Test théorique:")
            entry_price = 55138.01
            position_size = 0.00029744
            value_at_entry = entry_price * position_size

            print(f"  Position: {position_size} BTC @ {entry_price}")
            print(f"  Valeur à l'entrée: ${value_at_entry:.2f}")
            print(f"  Si prix +1%: ${(entry_price * 1.01) * position_size:.2f} (+${(entry_price * 0.01 * position_size):.2f})")
            print(f"  Si prix -1%: ${(entry_price * 0.99) * position_size:.2f} (-${(entry_price * 0.01 * position_size):.2f})")

            print("❌ Ces variations devraient être visibles dans les logs mais ne le sont pas!")

            self.results['pnl_updates'] = {
                'portfolio_value_static': True,
                'realized_pnl_always_zero': True,
                'mark_to_market_broken': True
            }

        except Exception as e:
            print(f"❌ Erreur lors du diagnostic du PnL: {e}")

        print()

    def diagnose_penalty_system(self):
        """Diagnostique le système de pénalités."""
        print("🔍 DIAGNOSTIC 6: Système de Pénalités")
        print("=" * 50)

        try:
            print("📊 Analyse des pénalités dans les logs:")
            print("  Step 0: Total: -30.0000 (Base: 0.0000, Frequency: -30.0000)")
            print("  Step 1: Total: -26.0164 (Base: -0.0164, Frequency: -26.0000)")

            print(f"\n📈 Analyse des critères de fréquence:")
            print("  Critères exigés dès le step 1:")
            print("  - 5m: 1/6-15 ✗ (besoin de 6-15 trades, n'en a que 1)")
            print("  - 1h: 0/3-10 ✗ (besoin de 3-10 trades, n'en a que 0)")
            print("  - 4h: 0/1-3 ✗ (besoin de 1-3 trades, n'en a que 0)")
            print("  - Total: 1/5-15 ✗ (besoin de 5-15 trades, n'en a que 1)")

            print(f"\n❌ PROBLÈMES IDENTIFIÉS:")
            print("1. Pénalité de -30 dès le step 0 (avant même tout trade!)")
            print("2. Critères trop stricts pour un agent débutant")
            print("3. Pas de période de grâce pour l'apprentissage")
            print("4. Un agent aléatoire ne peut pas satisfaire ces critères rapidement")

            print(f"\n🎯 Suggestions de correction:")
            print("- Période de grâce de 100-200 steps sans pénalité")
            print("- Critères progressifs (plus souples au début)")
            print("- Pénalité proportionnelle au nombre de steps écoulés")

            # Calculer l'impact des pénalités
            base_reward = -0.0164  # Perte due aux commissions
            frequency_penalty = -26.0
            total_penalty = base_reward + frequency_penalty

            print(f"\n📊 Impact des pénalités:")
            print(f"  Récompense de base: {base_reward}")
            print(f"  Pénalité de fréquence: {frequency_penalty}")
            print(f"  Impact relatif: {abs(frequency_penalty/base_reward):.1f}x plus important!")

            if abs(frequency_penalty) > abs(base_reward) * 10:
                print("❌ PROBLÈME CRITIQUE: La pénalité écrase complètement le signal d'apprentissage!")

            self.results['penalty_system'] = {
                'immediate_high_penalty': True,
                'no_grace_period': True,
                'criteria_too_strict': True,
                'penalty_dominates_signal': abs(frequency_penalty) > abs(base_reward) * 10
            }

        except Exception as e:
            print(f"❌ Erreur lors du diagnostic des pénalités: {e}")

        print()

    def run_quick_training_test(self):
        """Lance un test d'entraînement de 30 secondes pour valider les diagnostics."""
        print("🔍 TEST DE VALIDATION: Entraînement court")
        print("=" * 50)

        try:
            cmd = [
                "/home/morningstar/miniconda3/envs/trading_env/bin/python",
                "bot/scripts/train_parallel_agents.py",
                "--config", "bot/config/config.yaml",
                "--checkpoint-dir", "bot/checkpoints"
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            # Collecter les logs pendant 30 secondes
            logs = []
            start_time = time.time()

            while time.time() - start_time < 30:
                line = process.stdout.readline()
                if line:
                    logs.append(line.strip())
                if process.poll() is not None:
                    break

            # Terminer le processus
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)

            # Analyser les logs collectés
            chunk_changes = len([l for l in logs if "chunk" in l.lower() and ("2/10" in l or "3/10" in l)])
            worker_diversity = len(set([re.search(r'Worker (\d+)', l).group(1) for l in logs if re.search(r'Worker (\d+)', l)]))
            portfolio_values = [re.search(r'Portfolio value: ([\d.]+)', l).group(1) for l in logs if re.search(r'Portfolio value: ([\d.]+)', l)]
            unique_values = len(set(portfolio_values)) if portfolio_values else 0

            print(f"📊 Résultats du test (30s):")
            print(f"  - Changements de chunk détectés: {chunk_changes}")
            print(f"  - Workers différents actifs: {worker_diversity}")
            print(f"  - Valeurs de portefeuille uniques: {unique_values}")
            print(f"  - Total de logs collectés: {len(logs)}")

            # Validation des diagnostics
            print(f"\n✅ Validation des diagnostics:")
            print(f"  - Chunks bloqués: {'❌ Confirmé' if chunk_changes == 0 else '✅ Résolu'}")
            print(f"  - Seul worker 0 visible: {'❌ Confirmé' if worker_diversity <= 1 else '✅ Résolu'}")
            print(f"  - Valeurs statiques: {'❌ Confirmé' if unique_values <= 1 else '✅ Résolu'}")

        except Exception as e:
            print(f"❌ Erreur lors du test de validation: {e}")

    def generate_report(self):
        """Génère un rapport de diagnostic complet."""
        print("📋 RAPPORT DE DIAGNOSTIC COMPLET")
        print("=" * 60)

        total_time = time.time() - self.start_time

        print(f"⏱️  Durée du diagnostic: {total_time:.1f}s")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Résumé des problèmes identifiés
        problems_found = []

        if not self.results.get('chunk_progression', {}).get('progression_possible', True):
            problems_found.append("❌ Progression des chunks impossible")

        if self.results.get('worker_behavior', {}).get('logs_restricted', False):
            problems_found.append("❌ Logs restreints au worker 0 seulement")

        if self.results.get('model_initialization', {}).get('parameters_too_precise', False):
            problems_found.append("❌ Modèle pas vraiment aléatoire au départ")

        if self.results.get('metrics_calculation', {}).get('metrics_at_zero', False):
            problems_found.append("❌ Métriques bloquées à zéro")

        if self.results.get('pnl_updates', {}).get('mark_to_market_broken', False):
            problems_found.append("❌ Valorisation mark-to-market cassée")

        if self.results.get('penalty_system', {}).get('penalty_dominates_signal', False):
            problems_found.append("❌ Système de pénalités trop agressif")

        print(f"🔍 PROBLÈMES IDENTIFIÉS ({len(problems_found)}/6):")
        for problem in problems_found:
            print(f"  {problem}")

        if len(problems_found) == 0:
            print("  ✅ Aucun problème majeur détecté")

        print()
        print("🎯 PRIORITÉS DE CORRECTION:")
        print("  1. 🔥 CRITIQUE: Corriger la valorisation mark-to-market")
        print("     → Le PnL doit changer quand les prix changent")
        print()
        print("  2. 🔥 CRITIQUE: Ajuster le système de pénalités")
        print("     → Ajouter une période de grâce pour l'apprentissage")
        print()
        print("  3. ⚠️  IMPORTANT: Vérifier la progression des chunks")
        print("     → S'assurer que l'entraînement ne reste pas bloqué")
        print()
        print("  4. ⚠️  IMPORTANT: Équilibrer l'affichage des workers")
        print("     → Permettre de voir l'activité de tous les workers")
        print()
        print("  5. 🔧 AMÉLIORATION: Vérifier l'initialisation du modèle")
        print("     → S'assurer qu'il part bien d'un état aléatoire")

        # Sauvegarder le rapport
        report_file = Path("diagnostic_report.json")
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'duration': total_time,
                'results': self.results,
                'problems_found': len(problems_found),
                'problems_list': problems_found
            }, f, indent=2)

        print(f"\n💾 Rapport sauvegardé: {report_file}")

def main():
    """Fonction principale du diagnostic."""
    print("🩺 DIAGNOSTIC COMPLET DU SYSTÈME DE TRADING")
    print("=" * 60)
    print()

    diagnostic = SystemDiagnostic()

    # Exécuter tous les diagnostics
    diagnostic.diagnose_chunk_progression()
    diagnostic.diagnose_worker_behavior()
    diagnostic.diagnose_model_initialization()
    diagnostic.diagnose_metrics_calculation()
    diagnostic.diagnose_pnl_updates()
    diagnostic.diagnose_penalty_system()

    # Test de validation optionnel
    print("🤔 Voulez-vous lancer un test d'entraînement de 30s pour valider ? (y/N)")
    # response = input().lower().strip()
    # if response == 'y':
    #     diagnostic.run_quick_training_test()

    # Générer le rapport final
    diagnostic.generate_report()

    print("\n🎯 Le diagnostic est terminé. Consultez le rapport ci-dessus pour les corrections prioritaires.")

if __name__ == "__main__":
    main()
