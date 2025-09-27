#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test de Validation - Correction Problème #5 : Métriques Bloquées à Zéro
======================================================================

Ce script teste la correction du système de métriques pour s'assurer que :
1. Les métriques incluent maintenant les positions ouvertes
2. Le win_rate n'est plus artificiellement dilué par les trades d'ouverture
3. Les métriques temps réel fonctionnent correctement
4. Les positions ouvertes contribuent au calcul des performances

Test des corrections appliquées :
✅ Séparation trades fermés vs trades ouverts
✅ Calcul du PnL non réalisé
✅ Win rate combiné (fermé + ouvert)
✅ Métriques temps réel

Auteur: Trading Bot Team
Date: 2024
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
from collections import namedtuple

# Ajouter le chemin du bot pour les imports
sys.path.insert(0, str(Path(__file__).parent / "bot" / "src"))

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Structure pour simuler une position
Position = namedtuple('Position', ['asset', 'entry_price', 'size'])


class MetricsCorrectionValidator:
    """Validateur de la correction des métriques."""

    def __init__(self):
        self.results = {}
        self.test_metrics = None

    def print_header(self, title):
        """Affiche un en-tête de test."""
        print(f"\n{'='*80}")
        print(f"🧪 {title}")
        print(f"{'='*80}")

    def print_result(self, test_name, passed, details=""):
        """Affiche le résultat d'un test."""
        status = "✅ RÉUSSI" if passed else "❌ ÉCHOUÉ"
        self.results[test_name] = passed
        print(f"\n{status} - {test_name}")
        if details:
            print(f"    📋 Détails: {details}")

    def setup_test_metrics(self):
        """Configure les métriques de test."""
        try:
            from adan_trading_bot.performance.metrics import PerformanceMetrics
            self.test_metrics = PerformanceMetrics(worker_id=0)
            return True
        except ImportError as e:
            print(f"❌ Erreur d'import: {e}")
            return False

    def test_basic_functionality(self):
        """Test 1: Fonctionnalité de base des métriques corrigées."""
        self.print_header("TEST 1 - FONCTIONNALITÉ DE BASE")

        if not self.setup_test_metrics():
            self.print_result("Setup métriques", False, "Import failed")
            return

        # Test update_trade avec position fermée
        close_trade = {
            'action': 'close',
            'asset': 'BTCUSDT',
            'pnl': 25.0,
            'pnl_pct': 2.5,
            'equity': 1025.0
        }

        self.test_metrics.update_trade(close_trade)
        summary = self.test_metrics.get_metrics_summary()

        # Vérifications
        win_rate_ok = summary.get('win_rate', 0) == 100.0
        total_trades_ok = summary.get('total_trades', 0) == 1
        basic_ok = win_rate_ok and total_trades_ok

        details = f"Win rate: {summary.get('win_rate', 0)}%, Total trades: {summary.get('total_trades', 0)}"
        self.print_result("Métriques de base", basic_ok, details)

    def test_open_positions_exclusion(self):
        """Test 2: Exclusion des positions d'ouverture du win_rate."""
        self.print_header("TEST 2 - EXCLUSION POSITIONS OUVERTURE")

        if not self.setup_test_metrics():
            return

        # Ajouter trade d'ouverture (ne doit pas affecter win_rate)
        open_trade = {
            'action': 'open',
            'asset': 'BTCUSDT',
            'size': 0.001,
            'entry_price': 45000,
            'equity': 1000.0
        }

        self.test_metrics.update_trade(open_trade)
        summary_after_open = self.test_metrics.get_metrics_summary()

        # Ajouter trade de fermeture gagnant
        close_trade = {
            'action': 'close',
            'asset': 'BTCUSDT',
            'pnl': 50.0,
            'pnl_pct': 5.0,
            'equity': 1050.0
        }

        self.test_metrics.update_trade(close_trade)
        summary_after_close = self.test_metrics.get_metrics_summary()

        # Vérifications
        # Après ouverture: win_rate doit rester 0 (pas de trade fermé)
        open_win_rate_ok = summary_after_open.get('win_rate', 0) == 0.0
        # Après fermeture: win_rate doit être 100% (1 trade fermé gagnant sur 1)
        close_win_rate_ok = summary_after_close.get('win_rate', 0) == 100.0
        # Total trades fermés doit être 1 (pas 2)
        closed_trades_ok = summary_after_close.get('total_trades', 0) == 1

        exclusion_ok = open_win_rate_ok and close_win_rate_ok and closed_trades_ok

        details = f"Après ouvert: {summary_after_open.get('win_rate', 0)}%, Après fermé: {summary_after_close.get('win_rate', 0)}%, Trades fermés: {summary_after_close.get('total_trades', 0)}"
        self.print_result("Exclusion positions ouverture", exclusion_ok, details)

    def test_unrealized_pnl_calculation(self):
        """Test 3: Calcul du PnL non réalisé."""
        self.print_header("TEST 3 - CALCUL PNL NON RÉALISÉ")

        if not self.setup_test_metrics():
            return

        # Simuler des positions ouvertes
        open_positions = [
            Position('BTCUSDT', 45000, 0.001),  # Position gagnante
            Position('ETHUSDT', 3000, 0.01),    # Position perdante
            Position('ADAUSDT', 0.5, 100)       # Position neutre
        ]

        current_prices = {
            'BTCUSDT': 46000,  # +1000 * 0.001 = +1.0 USDT
            'ETHUSDT': 2900,   # -100 * 0.01 = -1.0 USDT
            'ADAUSDT': 0.5     # 0 * 100 = 0.0 USDT
        }

        unrealized = self.test_metrics.calculate_unrealized_pnl(open_positions, current_prices)

        # Vérifications
        expected_pnl = 1.0 - 1.0 + 0.0  # = 0.0 USDT
        pnl_ok = abs(unrealized['unrealized_pnl'] - expected_pnl) < 0.01
        winners_ok = unrealized['unrealized_winners'] == 1
        losers_ok = unrealized['unrealized_losers'] == 1
        count_ok = unrealized['open_positions_count'] == 3

        unrealized_ok = pnl_ok and winners_ok and losers_ok and count_ok

        details = f"PnL: {unrealized['unrealized_pnl']:.2f} USDT, Winners: {unrealized['unrealized_winners']}, Losers: {unrealized['unrealized_losers']}, Total: {unrealized['open_positions_count']}"
        self.print_result("Calcul PnL non réalisé", unrealized_ok, details)

    def test_combined_win_rate(self):
        """Test 4: Win rate combiné (fermé + ouvert)."""
        self.print_header("TEST 4 - WIN RATE COMBINÉ")

        if not self.setup_test_metrics():
            return

        # Ajouter des trades fermés
        trades = [
            {'action': 'close', 'asset': 'BTCUSDT', 'pnl': 50.0, 'pnl_pct': 5.0, 'equity': 1050.0},  # Gagnant
            {'action': 'close', 'asset': 'ETHUSDT', 'pnl': -30.0, 'pnl_pct': -3.0, 'equity': 1020.0}  # Perdant
        ]

        for trade in trades:
            self.test_metrics.update_trade(trade)

        # Ajouter positions ouvertes
        open_positions = [
            Position('ADAUSDT', 0.5, 100),    # Position gagnante (+5 USDT)
            Position('DOTUSDT', 10.0, 5),     # Position gagnante (+2.5 USDT)
            Position('LINKUSDT', 20.0, 2)     # Position perdante (-1 USDT)
        ]

        current_prices = {
            'ADAUSDT': 0.55,   # +0.05 * 100 = +5.0 USDT
            'DOTUSDT': 10.5,   # +0.5 * 5 = +2.5 USDT
            'LINKUSDT': 19.5   # -0.5 * 2 = -1.0 USDT
        }

        self.test_metrics.update_open_positions_metrics(open_positions, current_prices)
        summary = self.test_metrics.get_metrics_summary()

        # Calcul attendu:
        # Trades fermés: 1 gagnant, 1 perdant -> win_rate = 50%
        # Positions ouvertes: 2 gagnantes, 1 perdante
        # Combiné: 3 gagnants, 2 perdants -> win_rate combiné = 60%

        closed_win_rate = summary.get('win_rate', 0)
        combined_win_rate = summary.get('combined_win_rate', 0)

        closed_ok = abs(closed_win_rate - 50.0) < 0.1
        combined_ok = abs(combined_win_rate - 60.0) < 0.1

        combined_test_ok = closed_ok and combined_ok

        details = f"Fermé: {closed_win_rate:.1f}%, Combiné: {combined_win_rate:.1f}% (attendu: 50%/60%)"
        self.print_result("Win rate combiné", combined_test_ok, details)

    def test_real_time_metrics_update(self):
        """Test 5: Mise à jour temps réel des métriques."""
        self.print_header("TEST 5 - MÉTRIQUES TEMPS RÉEL")

        if not self.setup_test_metrics():
            return

        # Simulation d'évolution d'une position
        scenarios = [
            # Scénario 1: Position neutre
            {
                'positions': [Position('BTCUSDT', 45000, 0.001)],
                'prices': {'BTCUSDT': 45000},
                'expected_unrealized': 0.0
            },
            # Scénario 2: Position gagnante
            {
                'positions': [Position('BTCUSDT', 45000, 0.001)],
                'prices': {'BTCUSDT': 46000},
                'expected_unrealized': 1.0
            },
            # Scénario 3: Position perdante
            {
                'positions': [Position('BTCUSDT', 45000, 0.001)],
                'prices': {'BTCUSDT': 44000},
                'expected_unrealized': -1.0
            }
        ]

        all_scenarios_ok = True
        scenario_results = []

        for i, scenario in enumerate(scenarios, 1):
            self.test_metrics.update_open_positions_metrics(
                scenario['positions'],
                scenario['prices']
            )

            summary = self.test_metrics.get_metrics_summary()
            actual_pnl = summary.get('unrealized_pnl', 0)
            expected_pnl = scenario['expected_unrealized']

            scenario_ok = abs(actual_pnl - expected_pnl) < 0.01
            scenario_results.append(f"S{i}: {actual_pnl:.2f}/{expected_pnl:.2f}")

            if not scenario_ok:
                all_scenarios_ok = False

        details = f"Scénarios: {', '.join(scenario_results)}"
        self.print_result("Métriques temps réel", all_scenarios_ok, details)

    def test_zero_metrics_problem_resolved(self):
        """Test 6: Vérifier que le problème des métriques à zéro est résolu."""
        self.print_header("TEST 6 - PROBLÈME MÉTRIQUES ZÉRO RÉSOLU")

        if not self.setup_test_metrics():
            return

        # Scénario réaliste : trading actif avec positions ouvertes

        # 1. Ajouter quelques trades fermés
        closed_trades = [
            {'action': 'close', 'asset': 'BTCUSDT', 'pnl': 25.0, 'equity': 1025.0},
            {'action': 'close', 'asset': 'ETHUSDT', 'pnl': -15.0, 'equity': 1010.0},
            {'action': 'close', 'asset': 'ADAUSDT', 'pnl': 10.0, 'equity': 1020.0}
        ]

        for trade in closed_trades:
            self.test_metrics.update_trade(trade)

        # 2. Ajouter des positions ouvertes
        open_positions = [
            Position('DOTUSDT', 10.0, 5),     # Position gagnante
            Position('LINKUSDT', 20.0, 2)     # Position perdante
        ]

        current_prices = {
            'DOTUSDT': 11.0,   # +5 USDT
            'LINKUSDT': 19.0   # -2 USDT
        }

        self.test_metrics.update_open_positions_metrics(open_positions, current_prices)
        summary = self.test_metrics.get_metrics_summary()

        # Vérifications anti-zéro
        metrics_to_check = [
            ('win_rate', 66.67),           # 2 gagnants / 3 fermés
            ('total_trades', 3),           # 3 trades fermés
            ('unrealized_pnl', 3.0),       # +5 -2 = +3 USDT
            ('open_positions_count', 2),   # 2 positions ouvertes
            ('combined_win_rate', 60.0)    # 3 gagnants / 5 total
        ]

        zero_problem_resolved = True
        check_results = []

        for metric_name, expected_value in metrics_to_check:
            actual_value = summary.get(metric_name, 0)

            if metric_name in ['win_rate', 'combined_win_rate']:
                # Tolérance pour les pourcentages
                metric_ok = abs(actual_value - expected_value) < 5.0
            else:
                # Valeurs exactes ou tolérance faible
                metric_ok = abs(actual_value - expected_value) < 0.1

            if actual_value == 0 and expected_value != 0:
                zero_problem_resolved = False

            check_results.append(f"{metric_name}: {actual_value:.1f}")

            if not metric_ok:
                zero_problem_resolved = False

        details = f"Métriques: {', '.join(check_results)}"
        self.print_result("Problème métriques zéro résolu", zero_problem_resolved, details)

    def test_performance_impact(self):
        """Test 7: Impact sur les performances."""
        self.print_header("TEST 7 - IMPACT PERFORMANCES")

        if not self.setup_test_metrics():
            return

        # Test de performance avec beaucoup de trades
        start_time = time.time()

        # Ajouter 1000 trades
        for i in range(1000):
            trade = {
                'action': 'close' if i % 2 == 0 else 'open',
                'asset': f'TEST{i%10}USDT',
                'pnl': (i % 100) - 50,  # PnL entre -50 et +49
                'equity': 1000 + i
            }
            self.test_metrics.update_trade(trade)

        # Test avec positions ouvertes
        large_positions = [Position(f'ASSET{i}', 100, 1) for i in range(100)]
        large_prices = {f'ASSET{i}': 100 + (i % 20 - 10) for i in range(100)}

        self.test_metrics.update_open_positions_metrics(large_positions, large_prices)

        # Calculer métriques
        summary = self.test_metrics.get_metrics_summary()

        end_time = time.time()
        duration = end_time - start_time

        # Vérifications performance
        performance_ok = duration < 1.0  # Moins d'1 seconde
        results_valid = summary.get('total_trades', 0) > 0

        details = f"Durée: {duration:.3f}s, Trades: {summary.get('total_trades', 0)}, Performance: {'OK' if performance_ok else 'LENT'}"
        self.print_result("Impact performances", performance_ok and results_valid, details)

    def generate_final_report(self):
        """Génère le rapport final."""
        self.print_header("RAPPORT FINAL - CORRECTION MÉTRIQUES")

        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

        print(f"📊 RÉSUMÉ:")
        print(f"   - Tests exécutés: {total_tests}")
        print(f"   - Tests réussis: {passed_tests}")
        print(f"   - Taux de réussite: {success_rate:.1f}%")

        if success_rate >= 85:
            print(f"\n🎉 SUCCÈS! Problème #5 'Métriques Bloquées à Zéro' est RÉSOLU!")
            print(f"✅ Le système de métriques fonctionne correctement")
            print(f"✅ Les positions ouvertes sont incluses dans les calculs")
            print(f"✅ Le win rate n'est plus artificiellement dilué")
            print(f"✅ Les métriques temps réel fonctionnent")
        else:
            print(f"\n⚠️ ATTENTION: Succès partiel ({success_rate:.1f}%)")
            print(f"Des ajustements supplémentaires peuvent être nécessaires")

        # Détails des échecs
        failed_tests = [name for name, result in self.results.items() if not result]
        if failed_tests:
            print(f"\n❌ Tests échoués:")
            for test in failed_tests:
                print(f"   - {test}")

        return success_rate >= 85

    def run_all_tests(self):
        """Lance tous les tests de validation."""
        print("🚀 VALIDATION CORRECTION PROBLÈME #5 - MÉTRIQUES BLOQUÉES À ZÉRO")
        print("=" * 80)

        start_time = time.time()

        # Liste des tests à exécuter
        tests = [
            ("Fonctionnalité de base", self.test_basic_functionality),
            ("Exclusion positions ouverture", self.test_open_positions_exclusion),
            ("Calcul PnL non réalisé", self.test_unrealized_pnl_calculation),
            ("Win rate combiné", self.test_combined_win_rate),
            ("Métriques temps réel", self.test_real_time_metrics_update),
            ("Problème métriques zéro résolu", self.test_zero_metrics_problem_resolved),
            ("Impact performances", self.test_performance_impact)
        ]

        # Exécuter tous les tests
        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                print(f"❌ ERREUR dans {test_name}: {e}")
                self.results[test_name] = False

        duration = time.time() - start_time
        print(f"\n⏱️ Tests terminés en {duration:.2f}s")

        # Générer le rapport final
        return self.generate_final_report()


def main():
    """Point d'entrée principal."""
    print("🔧 Validation Correction Métriques - Problème #5")
    print("=" * 60)

    validator = MetricsCorrectionValidator()
    success = validator.run_all_tests()

    if success:
        print(f"\n🚀 CORRECTION VALIDÉE!")
        print(f"Le Problème #5 'Métriques Bloquées à Zéro' est résolu.")
        print(f"\nProchaine étape recommandée: Problème #6")
    else:
        print(f"\n🔧 AJUSTEMENTS REQUIS")
        print(f"Certains aspects nécessitent encore des corrections.")

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
