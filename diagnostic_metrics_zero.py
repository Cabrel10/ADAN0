#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnostic du Problème #5 : Métriques Bloquées à Zéro
====================================================

Ce script diagnostique pourquoi les métriques (win_rate, total_trades, etc.)
restent bloquées à zéro malgré l'activité de trading.

Problèmes potentiels identifiés :
1. Les métriques ne comptent que les positions FERMÉES
2. Ignorent les positions OUVERTES qui contribuent au PnL
3. Logique de calcul défaillante dans update_trade()
4. Différence entre trades ouverts et trades fermés

Auteur: Trading Bot Team
Date: 2024
"""

import sys
import json
import time
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List

# Ajouter le chemin du bot pour les imports
sys.path.insert(0, str(Path(__file__).parent / "bot" / "src"))

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetricsZeroDiagnostic:
    """Diagnostic complet du problème des métriques à zéro."""

    def __init__(self):
        self.results = {}
        self.issues_found = []

    def print_header(self, title):
        """Affiche un en-tête de section."""
        print(f"\n{'='*80}")
        print(f"🔍 {title}")
        print(f"{'='*80}")

    def print_issue(self, issue_type, description, severity="CRITIQUE"):
        """Enregistre un problème trouvé."""
        self.issues_found.append({
            'type': issue_type,
            'description': description,
            'severity': severity
        })
        status_emoji = "🔴" if severity == "CRITIQUE" else "🟡" if severity == "ATTENTION" else "🔵"
        print(f"{status_emoji} [{severity}] {issue_type}: {description}")

    def analyze_metrics_system(self):
        """Analyse le système de métriques pour identifier les problèmes."""
        self.print_header("ANALYSE DU SYSTÈME DE MÉTRIQUES")

        try:
            from adan_trading_bot.performance.metrics import PerformanceMetrics
            from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

            print("✅ Imports réussis - Modules de métriques disponibles")

            # Analyser la classe PerformanceMetrics
            self.analyze_performance_metrics_class()

            # Analyser l'intégration dans PortfolioManager
            self.analyze_portfolio_manager_integration()

        except ImportError as e:
            self.print_issue("IMPORT_ERROR", f"Impossible d'importer les modules de métriques: {e}")
            return False

        return True

    def analyze_performance_metrics_class(self):
        """Analyse la classe PerformanceMetrics."""
        self.print_header("ANALYSE PERFORMANCE METRICS CLASS")

        try:
            from adan_trading_bot.performance.metrics import PerformanceMetrics

            # Créer une instance pour tester
            metrics = PerformanceMetrics(worker_id=0)

            print(f"📊 Métriques initialisées:")
            print(f"   - trades: {len(metrics.trades)}")
            print(f"   - returns: {len(metrics.returns)}")
            print(f"   - equity_curve: {len(metrics.equity_curve)}")
            print(f"   - closed_positions: {len(metrics.closed_positions)}")

            # Tester update_trade
            test_trade = {
                'action': 'close',
                'asset': 'BTCUSDT',
                'pnl': 15.50,
                'pnl_pct': 2.5,
                'equity': 1015.50
            }

            print(f"\n🧪 Test update_trade avec: {test_trade}")
            metrics.update_trade(test_trade)

            print(f"📈 Après update_trade:")
            print(f"   - trades: {len(metrics.trades)}")
            print(f"   - returns: {len(metrics.returns)}")
            print(f"   - equity_curve: {len(metrics.equity_curve)}")

            # Tester get_metrics_summary
            summary = metrics.get_metrics_summary()
            print(f"\n📋 Métriques calculées:")
            for key, value in summary.items():
                print(f"   - {key}: {value}")

            # Identifier les problèmes
            if len(metrics.trades) == 0:
                self.print_issue("NO_TRADES", "Aucun trade n'est enregistré dans le système")

            if summary.get('win_rate', 0) == 0 and len(metrics.trades) > 0:
                self.print_issue("ZERO_WIN_RATE", "Win rate à zéro malgré des trades existants")

        except Exception as e:
            self.print_issue("METRICS_CLASS_ERROR", f"Erreur dans PerformanceMetrics: {e}")

    def analyze_portfolio_manager_integration(self):
        """Analyse l'intégration des métriques dans PortfolioManager."""
        self.print_header("ANALYSE INTÉGRATION PORTFOLIO MANAGER")

        try:
            # Analyser le code source pour les patterns problématiques
            portfolio_file = Path(__file__).parent / "bot/src/adan_trading_bot/portfolio/portfolio_manager.py"

            if not portfolio_file.exists():
                self.print_issue("FILE_NOT_FOUND", f"Fichier portfolio_manager.py introuvable: {portfolio_file}")
                return

            with open(portfolio_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Chercher les patterns problématiques
            patterns_to_check = [
                ("update_trade", "Appels à update_trade"),
                ("close_position", "Méthodes close_position"),
                ("open_position", "Méthodes open_position"),
                ("total_trades", "Références à total_trades"),
                ("win_rate", "Références à win_rate"),
                ("closed_positions", "Références à closed_positions"),
            ]

            print("🔍 Analyse des patterns dans portfolio_manager.py:")
            for pattern, description in patterns_to_check:
                count = content.count(pattern)
                print(f"   - {description}: {count} occurrences")

            # Vérifier la logique update_trade
            if "update_trade" in content:
                print("\n✅ update_trade trouvé dans portfolio_manager.py")
                # Extraire les contextes d'appel
                self.extract_update_trade_contexts(content)
            else:
                self.print_issue("NO_UPDATE_TRADE", "update_trade non trouvé dans portfolio_manager")

        except Exception as e:
            self.print_issue("INTEGRATION_ANALYSIS_ERROR", f"Erreur analyse intégration: {e}")

    def extract_update_trade_contexts(self, content):
        """Extrait les contextes d'appel d'update_trade."""
        lines = content.split('\n')
        update_trade_lines = []

        for i, line in enumerate(lines):
            if "update_trade" in line:
                # Récupérer le contexte (3 lignes avant et après)
                start = max(0, i-3)
                end = min(len(lines), i+4)
                context = '\n'.join([f"{j:4d}: {lines[j]}" for j in range(start, end)])
                update_trade_lines.append({
                    'line_number': i+1,
                    'line': line.strip(),
                    'context': context
                })

        print(f"\n📝 Contextes d'appel update_trade ({len(update_trade_lines)} trouvés):")
        for idx, call in enumerate(update_trade_lines):
            print(f"\n   Call #{idx+1} (ligne {call['line_number']}):")
            print(f"   {call['line']}")
            print(f"   Contexte:\n{call['context']}")

        # Analyser les patterns problématiques
        if len(update_trade_lines) == 0:
            self.print_issue("NO_UPDATE_TRADE_CALLS", "Aucun appel à update_trade trouvé")
        elif len(update_trade_calls := [call for call in update_trade_lines if "'action': 'close'" in call['context']]) == 0:
            self.print_issue("NO_CLOSE_TRADES", "Aucun appel update_trade pour 'close' trouvé")

    def analyze_trade_flow_logic(self):
        """Analyse la logique de flux des trades."""
        self.print_header("ANALYSE LOGIQUE FLUX DES TRADES")

        print("🔄 Simulation du flux de trading:")

        # Simuler un scénario de trading complet
        try:
            from adan_trading_bot.performance.metrics import PerformanceMetrics

            metrics = PerformanceMetrics(worker_id=0)
            initial_equity = 1000.0

            print(f"💰 Capital initial: {initial_equity}")

            # Simulation d'ouverture de position
            open_trade = {
                'action': 'open',
                'asset': 'BTCUSDT',
                'size': 0.001,
                'entry_price': 45000,
                'equity': initial_equity
            }

            print(f"\n📈 Ouverture position: {open_trade}")
            metrics.update_trade(open_trade)

            summary_after_open = metrics.get_metrics_summary()
            print(f"📊 Métriques après ouverture:")
            print(f"   - Total trades: {len(metrics.trades)}")
            print(f"   - Win rate: {summary_after_open.get('win_rate', 0):.2f}%")

            # Simulation de fermeture position (gagnante)
            close_trade = {
                'action': 'close',
                'asset': 'BTCUSDT',
                'pnl': 25.0,
                'pnl_pct': 2.5,
                'equity': initial_equity + 25.0
            }

            print(f"\n📉 Fermeture position (gagnante): {close_trade}")
            metrics.update_trade(close_trade)

            summary_after_close = metrics.get_metrics_summary()
            print(f"📊 Métriques après fermeture:")
            print(f"   - Total trades: {len(metrics.trades)}")
            print(f"   - Win rate: {summary_after_close.get('win_rate', 0):.2f}%")
            print(f"   - Profit factor: {summary_after_close.get('profit_factor', 0):.2f}")

            # Analyser le problème
            if summary_after_close.get('win_rate', 0) == 0:
                self.print_issue("ZERO_WIN_RATE_AFTER_WIN",
                               "Win rate reste à zéro après un trade gagnant")

            if len(metrics.trades) == 0:
                self.print_issue("NO_TRADES_RECORDED",
                               "Aucun trade enregistré après update_trade")

        except Exception as e:
            self.print_issue("SIMULATION_ERROR", f"Erreur simulation trading: {e}")

    def analyze_metrics_calculation_logic(self):
        """Analyse la logique de calcul des métriques."""
        self.print_header("ANALYSE LOGIQUE CALCUL MÉTRIQUES")

        try:
            # Examiner le code source de get_metrics_summary
            metrics_file = Path(__file__).parent / "bot/src/adan_trading_bot/performance/metrics.py"

            if not metrics_file.exists():
                self.print_issue("METRICS_FILE_NOT_FOUND", f"Fichier metrics.py introuvable: {metrics_file}")
                return

            with open(metrics_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Rechercher la logique de get_metrics_summary
            if "def get_metrics_summary" in content:
                print("✅ get_metrics_summary trouvé dans metrics.py")
                self.analyze_win_rate_calculation(content)
            else:
                self.print_issue("NO_METRICS_SUMMARY", "get_metrics_summary non trouvé")

        except Exception as e:
            self.print_issue("METRICS_LOGIC_ERROR", f"Erreur analyse logique métriques: {e}")

    def analyze_win_rate_calculation(self, content):
        """Analyse spécifiquement le calcul du win_rate."""
        lines = content.split('\n')

        # Trouver les lignes relatives au win_rate
        win_rate_lines = []
        for i, line in enumerate(lines):
            if "win_rate" in line.lower() or "winning_trades" in line or "total_trades" in line:
                win_rate_lines.append({
                    'line_number': i+1,
                    'line': line.strip()
                })

        print(f"\n🧮 Lignes relatives au calcul win_rate ({len(win_rate_lines)} trouvées):")
        for line_info in win_rate_lines:
            print(f"   {line_info['line_number']:4d}: {line_info['line']}")

        # Identifier les patterns problématiques
        problematic_patterns = [
            ("if t.get('pnl', 0) > 0", "Condition gagnant basée sur PnL"),
            ("len(self.trades)", "Comptage basé sur self.trades"),
            ("len(self.closed_positions)", "Comptage basé sur closed_positions"),
        ]

        print(f"\n🔍 Analyse des patterns de calcul:")
        for pattern, description in problematic_patterns:
            if pattern in content:
                print(f"   ✅ {description}: Présent")
            else:
                print(f"   ❌ {description}: Absent")
                self.print_issue("MISSING_PATTERN", f"Pattern manquant: {description}")

    def identify_root_cause(self):
        """Identifie la cause racine du problème."""
        self.print_header("IDENTIFICATION CAUSE RACINE")

        print("🎯 Hypothèses principales:")

        hypotheses = [
            {
                'name': "Positions ouvertes ignorées",
                'description': "Les métriques ne comptent que les positions fermées",
                'likelihood': "ÉLEVÉE"
            },
            {
                'name': "update_trade appelé incorrectement",
                'description': "Les appels à update_trade ne transmettent pas les bonnes données",
                'likelihood': "MOYENNE"
            },
            {
                'name': "Logique de calcul défaillante",
                'description': "La logique dans get_metrics_summary a un bug",
                'likelihood': "MOYENNE"
            },
            {
                'name': "Données PnL incorrectes",
                'description': "Les données de PnL ne sont pas correctement calculées",
                'likelihood': "FAIBLE"
            }
        ]

        for hypothesis in hypotheses:
            likelihood_emoji = "🔴" if hypothesis['likelihood'] == "ÉLEVÉE" else "🟡" if hypothesis['likelihood'] == "MOYENNE" else "🟢"
            print(f"{likelihood_emoji} [{hypothesis['likelihood']}] {hypothesis['name']}")
            print(f"    {hypothesis['description']}")

    def propose_solutions(self):
        """Propose des solutions pour résoudre le problème."""
        self.print_header("SOLUTIONS PROPOSÉES")

        solutions = [
            {
                'priority': 1,
                'title': "Inclure positions ouvertes dans métriques",
                'description': "Modifier le calcul pour inclure les positions ouvertes avec leur PnL non réalisé",
                'implementation': [
                    "Ajouter méthode calculate_unrealized_pnl()",
                    "Modifier get_metrics_summary() pour inclure positions ouvertes",
                    "Créer métriques séparées: realized_trades vs total_positions"
                ]
            },
            {
                'priority': 2,
                'title': "Corriger logique update_trade",
                'description': "S'assurer que tous les trades (ouverts ET fermés) sont correctement enregistrés",
                'implementation': [
                    "Vérifier tous les appels update_trade dans portfolio_manager",
                    "Ajouter validation des données dans update_trade",
                    "Logger tous les trades pour debug"
                ]
            },
            {
                'priority': 3,
                'title': "Métriques temps réel",
                'description': "Implémenter un système de métriques temps réel incluant positions actives",
                'implementation': [
                    "Créer classe RealTimeMetrics",
                    "Mise à jour continue des métriques",
                    "Dashboard temps réel des performances"
                ]
            }
        ]

        for solution in solutions:
            priority_emoji = "🔴" if solution['priority'] == 1 else "🟡" if solution['priority'] == 2 else "🟢"
            print(f"{priority_emoji} PRIORITÉ {solution['priority']}: {solution['title']}")
            print(f"    📝 {solution['description']}")
            print(f"    🛠️ Implémentation:")
            for step in solution['implementation']:
                print(f"       - {step}")
            print()

    def generate_report(self):
        """Génère un rapport complet du diagnostic."""
        self.print_header("RAPPORT FINAL")

        total_issues = len(self.issues_found)
        critical_issues = len([i for i in self.issues_found if i['severity'] == 'CRITIQUE'])

        print(f"📊 RÉSUMÉ:")
        print(f"   - Total problèmes identifiés: {total_issues}")
        print(f"   - Problèmes critiques: {critical_issues}")
        print(f"   - Statut: {'🔴 ACTION REQUISE' if critical_issues > 0 else '🟡 OPTIMISATION RECOMMANDÉE'}")

        if self.issues_found:
            print(f"\n🔍 PROBLÈMES IDENTIFIÉS:")
            for i, issue in enumerate(self.issues_found, 1):
                severity_emoji = "🔴" if issue['severity'] == "CRITIQUE" else "🟡" if issue['severity'] == "ATTENTION" else "🔵"
                print(f"   {i}. {severity_emoji} {issue['type']}: {issue['description']}")

        # Sauvegarder le rapport
        report_data = {
            'timestamp': time.time(),
            'total_issues': total_issues,
            'critical_issues': critical_issues,
            'issues': self.issues_found,
            'status': 'CRITICAL' if critical_issues > 0 else 'WARNING'
        }

        report_file = Path(__file__).parent / "diagnostic_metrics_zero_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Rapport sauvegardé: {report_file}")

        return critical_issues == 0

    def run_full_diagnostic(self):
        """Lance le diagnostic complet."""
        print("🚀 DIAGNOSTIC PROBLÈME #5 - MÉTRIQUES BLOQUÉES À ZÉRO")
        print("=" * 80)

        start_time = time.time()

        # Étapes du diagnostic
        steps = [
            ("Analyse système métriques", self.analyze_metrics_system),
            ("Analyse logique flux trades", self.analyze_trade_flow_logic),
            ("Analyse calcul métriques", self.analyze_metrics_calculation_logic),
            ("Identification cause racine", self.identify_root_cause),
            ("Proposition solutions", self.propose_solutions),
            ("Génération rapport", self.generate_report)
        ]

        success = True
        for step_name, step_func in steps:
            try:
                print(f"\n🔄 {step_name}...")
                result = step_func()
                if result is False:
                    success = False
            except Exception as e:
                print(f"❌ Erreur dans {step_name}: {e}")
                success = False

        duration = time.time() - start_time

        print(f"\n⏱️ Diagnostic terminé en {duration:.2f}s")
        print(f"✅ Statut: {'SUCCÈS' if success else 'PARTIEL'}")

        return success


def main():
    """Point d'entrée principal."""
    diagnostic = MetricsZeroDiagnostic()
    success = diagnostic.run_full_diagnostic()

    if success:
        print("\n🎉 Diagnostic terminé avec succès!")
        print("📋 Consultez le rapport pour les solutions recommandées.")
    else:
        print("\n⚠️ Diagnostic terminé avec des erreurs.")
        print("🔍 Vérifiez les messages d'erreur ci-dessus.")

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
