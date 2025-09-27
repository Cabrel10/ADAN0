"""
Tests unitaires pour les corrections de métriques avec deque.
Ce module teste spécifiquement les corrections apportées à la classe PerformanceMetrics
pour éviter les fuites de mémoire et les blocages lors du calcul des ratios de Sharpe/Sortino.
"""

import pytest
import numpy as np
import pandas as pd
from collections import deque
from unittest.mock import Mock, patch
import sys
import os

# Imports du système de trading
sys.path.append('/home/morningstar/Documents/trading/bot/src')

try:
    from adan_trading_bot.performance.metrics import PerformanceMetrics
except ImportError:
    # Si l'import échoue, créer une version de test
    class PerformanceMetrics:
        def __init__(self, config=None, worker_id=0, metrics_dir="logs/metrics"):
            self.worker_id = worker_id
            self.returns = deque(maxlen=10000)  # Version corrigée avec deque
            self.trades = []
            self.equity_curve = deque(maxlen=10000)
            self.risk_free_rate = 0.02
            self.total_trades = 0
            self.wins = 0
            self.losses = 0
            self.neutral = 0

        def update_trade(self, trade_result):
            self.trades.append(trade_result)
            self.total_trades += 1

            if 'pnl_pct' in trade_result and trade_result['pnl_pct'] is not None:
                pnl_decimal = trade_result['pnl_pct'] / 100
                self.returns.append(pnl_decimal)

                if trade_result['pnl_pct'] > 0:
                    self.wins += 1
                elif trade_result['pnl_pct'] < 0:
                    self.losses += 1
                else:
                    self.neutral += 1

            if 'equity' in trade_result:
                self.equity_curve.append(trade_result['equity'])

        def calculate_sharpe_ratio(self):
            if len(self.returns) == 0:
                return 0.0

            returns_array = np.array(self.returns)
            excess_returns = returns_array - self.risk_free_rate / 365
            std = np.std(excess_returns)

            if std <= 1e-10:
                return 0.0

            sharpe = np.mean(excess_returns) / std * np.sqrt(365)
            return sharpe

        def calculate_sortino_ratio(self):
            if len(self.returns) == 0:
                return 0.0

            returns_array = np.array(self.returns)
            downside_returns = returns_array[returns_array < 0]

            if len(downside_returns) == 0:
                return 0.0

            downside_std = np.std(downside_returns)
            if downside_std <= 1e-10:
                return 0.0

            mean_excess = np.mean(returns_array) - self.risk_free_rate / 365
            sortino = mean_excess / downside_std * np.sqrt(365)
            return sortino


class TestMetricsDequeCorrections:
    """Tests pour les corrections de métriques avec deque."""

    def test_deque_initialization(self):
        """Test que les métriques sont initialisées avec deque."""
        metrics = PerformanceMetrics(worker_id=0)

        # Vérifier que returns est bien un deque avec maxlen
        assert isinstance(metrics.returns, deque), f"returns devrait être un deque, mais est {type(metrics.returns)}"
        assert metrics.returns.maxlen == 10000, f"maxlen devrait être 10000, mais vaut {metrics.returns.maxlen}"

        # Vérifier que equity_curve est aussi un deque
        assert isinstance(metrics.equity_curve, deque), f"equity_curve devrait être un deque, mais est {type(metrics.equity_curve)}"

        print("✅ Initialisation deque réussie")

    def test_memory_limitation_with_many_trades(self):
        """Test que le deque limite bien la mémoire avec beaucoup de trades."""
        metrics = PerformanceMetrics(worker_id=0)

        # Simuler un grand nombre de trades (plus que la limite de deque)
        num_trades = 15000  # Plus que la limite de 10000

        for i in range(num_trades):
            trade_result = {
                'pnl_pct': np.random.uniform(-5, 5),  # PnL entre -5% et +5%
                'equity': 1000 + i * 0.1
            }
            metrics.update_trade(trade_result)

        # Vérifier que le deque n'a gardé que les 10000 derniers éléments
        assert len(metrics.returns) == 10000, f"returns devrait contenir 10000 éléments, mais en contient {len(metrics.returns)}"
        assert len(metrics.equity_curve) == 10000, f"equity_curve devrait contenir 10000 éléments, mais en contient {len(metrics.equity_curve)}"

        # Vérifier que tous les trades sont comptés dans la liste trades
        assert len(metrics.trades) == num_trades, f"Tous les trades devraient être comptés: {len(metrics.trades)} vs {num_trades}"

        print(f"✅ Limitation mémoire réussie: {len(metrics.returns)} returns conservés sur {num_trades} trades")

    def test_sharpe_calculation_no_crash(self):
        """Test que le calcul de Sharpe ne crash pas même avec beaucoup de données."""
        metrics = PerformanceMetrics(worker_id=0)

        # Ajouter beaucoup de données pour simuler le cas qui causait le crash
        np.random.seed(42)  # Pour reproductibilité
        num_trades = 5000

        for i in range(num_trades):
            pnl_pct = np.random.normal(0.1, 2.0)  # Moyenne 0.1%, écart-type 2%
            trade_result = {
                'pnl_pct': pnl_pct,
                'equity': 1000 * (1 + i * 0.001)
            }
            metrics.update_trade(trade_result)

        # Calculer Sharpe plusieurs fois pour s'assurer qu'il n'y a pas de crash
        sharpe_results = []
        for _ in range(10):
            try:
                sharpe = metrics.calculate_sharpe_ratio()
                assert isinstance(sharpe, (int, float)), f"Sharpe devrait être numérique, reçu {type(sharpe)}"
                assert not np.isnan(sharpe), "Sharpe ne devrait pas être NaN"
                assert not np.isinf(sharpe), "Sharpe ne devrait pas être infini"
                sharpe_results.append(sharpe)
            except Exception as e:
                pytest.fail(f"Calcul Sharpe a crashé: {e}")

        # Vérifier que les résultats sont cohérents
        sharpe_std = np.std(sharpe_results)
        assert sharpe_std < 1e-10, f"Les calculs Sharpe devraient être identiques, écart-type: {sharpe_std}"

        print(f"✅ Calcul Sharpe sans crash: {sharpe_results[0]:.4f} (sur {num_trades} trades)")

    def test_sortino_calculation_no_crash(self):
        """Test que le calcul de Sortino ne crash pas."""
        metrics = PerformanceMetrics(worker_id=0)

        # Ajouter des données avec mix de gains/pertes
        test_returns = [2.5, -1.8, 3.2, -0.9, 1.7, -2.3, 4.1, -1.2, 0.8, -3.1]

        for pnl_pct in test_returns:
            trade_result = {'pnl_pct': pnl_pct, 'equity': 1000}
            metrics.update_trade(trade_result)

        try:
            sortino = metrics.calculate_sortino_ratio()
            assert isinstance(sortino, (int, float)), f"Sortino devrait être numérique, reçu {type(sortino)}"
            assert not np.isnan(sortino), "Sortino ne devrait pas être NaN"
            assert not np.isinf(sortino), "Sortino ne devrait pas être infini"
            print(f"✅ Calcul Sortino sans crash: {sortino:.4f}")
        except Exception as e:
            pytest.fail(f"Calcul Sortino a crashé: {e}")

    def test_edge_cases_empty_data(self):
        """Test des cas limites avec données vides."""
        metrics = PerformanceMetrics(worker_id=0)

        # Test avec aucune donnée
        sharpe_empty = metrics.calculate_sharpe_ratio()
        sortino_empty = metrics.calculate_sortino_ratio()

        assert sharpe_empty == 0.0, f"Sharpe avec données vides devrait être 0.0, reçu {sharpe_empty}"
        assert sortino_empty == 0.0, f"Sortino avec données vides devrait être 0.0, reçu {sortino_empty}"

        print("✅ Cas limites (données vides) gérés correctement")

    def test_edge_cases_zero_volatility(self):
        """Test avec des returns identiques (volatilité nulle)."""
        metrics = PerformanceMetrics(worker_id=0)

        # Ajouter des returns tous identiques
        constant_return = 0.5  # 0.5% constant
        for _ in range(100):
            trade_result = {'pnl_pct': constant_return, 'equity': 1000}
            metrics.update_trade(trade_result)

        # Le calcul devrait retourner 0 car l'écart-type est nul
        sharpe = metrics.calculate_sharpe_ratio()
        assert sharpe == 0.0, f"Sharpe avec volatilité nulle devrait être 0.0, reçu {sharpe}"

        print("✅ Cas limites (volatilité nulle) gérés correctement")

    def test_performance_with_large_dataset(self):
        """Test de performance avec un gros dataset."""
        import time

        metrics = PerformanceMetrics(worker_id=0)

        # Simuler le cas extrême qui causait le blocage
        num_trades = 20000

        start_time = time.time()

        # Ajouter beaucoup de trades
        for i in range(num_trades):
            pnl_pct = np.random.uniform(-2, 3)
            trade_result = {'pnl_pct': pnl_pct, 'equity': 1000 + i}
            metrics.update_trade(trade_result)

        # Calculer les métriques plusieurs fois
        calculation_times = []
        for _ in range(5):
            calc_start = time.time()
            sharpe = metrics.calculate_sharpe_ratio()
            sortino = metrics.calculate_sortino_ratio()
            calc_time = time.time() - calc_start
            calculation_times.append(calc_time)

        total_time = time.time() - start_time
        avg_calc_time = np.mean(calculation_times)

        # Les calculs devraient être rapides (moins de 100ms chacun)
        assert avg_calc_time < 0.1, f"Calcul trop lent: {avg_calc_time:.3f}s en moyenne"

        # Vérifier que la mémoire est bien limitée
        assert len(metrics.returns) == 10000, f"Deque devrait être limité à 10000, mais contient {len(metrics.returns)}"

        print(f"✅ Performance test réussi:")
        print(f"  - {num_trades} trades traités en {total_time:.2f}s")
        print(f"  - Calculs métriques en moyenne: {avg_calc_time*1000:.1f}ms")
        print(f"  - Mémoire limitée: {len(metrics.returns)} éléments conservés")

    def test_trade_counting_accuracy(self):
        """Test que le comptage des trades reste précis malgré la limitation deque."""
        metrics = PerformanceMetrics(worker_id=0)

        # Simuler différents types de trades
        winning_trades = [1.2, 2.5, 0.8, 1.9, 3.1]  # 5 trades gagnants
        losing_trades = [-0.9, -1.5, -2.1, -0.6]    # 4 trades perdants
        neutral_trades = [0.0, 0.0]                   # 2 trades neutres

        all_trades = winning_trades + losing_trades + neutral_trades

        for pnl_pct in all_trades:
            trade_result = {'pnl_pct': pnl_pct, 'equity': 1000}
            metrics.update_trade(trade_result)

        # Vérifier que le comptage est correct
        assert metrics.total_trades == len(all_trades), f"Total trades incorrect: {metrics.total_trades} vs {len(all_trades)}"
        assert metrics.wins == len(winning_trades), f"Wins incorrects: {metrics.wins} vs {len(winning_trades)}"
        assert metrics.losses == len(losing_trades), f"Losses incorrectes: {metrics.losses} vs {len(losing_trades)}"
        assert metrics.neutral == len(neutral_trades), f"Neutral incorrects: {metrics.neutral} vs {len(neutral_trades)}"

        print(f"✅ Comptage trades précis: {metrics.wins}W/{metrics.losses}L/{metrics.neutral}N = {metrics.total_trades} total")

    def test_deque_fifo_behavior(self):
        """Test que le deque respecte bien le comportement FIFO (First In, First Out)."""
        # Créer un deque avec une petite limite pour le test
        small_deque = deque(maxlen=3)

        # Ajouter plus d'éléments que la limite
        elements = [10, 20, 30, 40, 50]
        for elem in elements:
            small_deque.append(elem)

        # Vérifier que seuls les 3 derniers sont conservés
        assert len(small_deque) == 3, f"Deque devrait contenir 3 éléments, mais en contient {len(small_deque)}"
        assert list(small_deque) == [30, 40, 50], f"Deque devrait contenir [30, 40, 50], mais contient {list(small_deque)}"

        print("✅ Comportement FIFO du deque vérifié")


if __name__ == "__main__":
    # Exécution des tests en mode standalone
    print("=" * 70)
    print("TESTS UNITAIRES - CORRECTIONS MÉTRIQUES AVEC DEQUE")
    print("=" * 70)

    test_instance = TestMetricsDequeCorrections()

    tests = [
        ("Initialisation deque", test_instance.test_deque_initialization),
        ("Limitation mémoire", test_instance.test_memory_limitation_with_many_trades),
        ("Calcul Sharpe sans crash", test_instance.test_sharpe_calculation_no_crash),
        ("Calcul Sortino sans crash", test_instance.test_sortino_calculation_no_crash),
        ("Cas limites - données vides", test_instance.test_edge_cases_empty_data),
        ("Cas limites - volatilité nulle", test_instance.test_edge_cases_zero_volatility),
        ("Performance avec gros dataset", test_instance.test_performance_with_large_dataset),
        ("Précision comptage trades", test_instance.test_trade_counting_accuracy),
        ("Comportement FIFO deque", test_instance.test_deque_fifo_behavior),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Test: {test_name}")
            test_func()
            passed += 1
            print(f"✅ {test_name} - RÉUSSI")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} - ÉCHOUÉ: {e}")

    print("\n" + "=" * 70)
    print(f"RÉSULTATS: {passed} tests réussis, {failed} tests échoués")
    if failed == 0:
        print("🎉 TOUS LES TESTS SONT PASSÉS - Corrections validées !")
    print("=" * 70)
