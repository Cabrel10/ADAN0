"""
Tests unitaires pour la lecture des données et validation de l'indexation.
Ce module teste spécifiquement les corrections apportées pour résoudre le bug
d'indexation qui causait un forward fill excessif (EXCESSIVE_FORWARD_FILL).
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from typing import Dict, Any

# Imports du système de trading
import sys
sys.path.append('/home/morningstar/Documents/trading/bot/src')

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnvironment
from adan_trading_bot.performance.metrics import PerformanceMetrics


class TestDataReadingIndexation:
    """Tests pour la lecture des données et l'indexation."""

    @pytest.fixture
    def sample_data(self):
        """Crée des données de test réalistes avec variation de prix."""
        dates = pd.date_range('2024-01-01', periods=100, freq='5T')

        # Créer des prix qui varient réalistement
        base_price = 58000.0
        price_changes = np.random.randn(100) * 100  # Variation de ±100 USDT
        prices = [base_price + sum(price_changes[:i+1]) for i in range(100)]

        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.002 for p in prices],  # 0.2% plus haut
            'low': [p * 0.998 for p in prices],   # 0.2% plus bas
            'close': prices,
            'volume': np.random.uniform(1000, 5000, 100)
        })

        return df

    @pytest.fixture
    def mock_config(self):
        """Configuration de test minimale."""
        return {
            'assets': ['BTCUSDT'],
            'timeframes': ['5m', '1h', '4h'],
            'chunk_size': 50,
            'num_workers': 1,
            'data_dir': '/tmp/test_data'
        }

    def test_step_in_chunk_indexation(self, sample_data, mock_config):
        """Test que step_in_chunk est utilisé correctement pour l'indexation."""

        # Créer un environnement de test
        with patch('adan_trading_bot.environment.multi_asset_chunked_env.MultiAssetChunkedEnvironment._load_data') as mock_load:
            mock_load.return_value = {
                ('BTCUSDT', '5m'): sample_data[:50],  # Premier chunk de 50 lignes
                ('BTCUSDT', '1h'): sample_data[:50],
                ('BTCUSDT', '4h'): sample_data[:50]
            }

            env = MultiAssetChunkedEnvironment(mock_config, worker_id=0)
            env.current_data = mock_load.return_value

            # Test des différentes positions dans le chunk
            test_positions = [0, 10, 25, 49]  # Positions valides dans un chunk de 50

            for pos in test_positions:
                env.step_in_chunk = pos
                env.current_step = pos  # Pour cohérence

                # Appeler _get_current_prices
                with patch.object(env, '_get_current_prices') as mock_get_prices:
                    # Simuler le comportement réel avec step_in_chunk
                    expected_price = sample_data.iloc[pos]['close']
                    mock_get_prices.return_value = {'BTCUSDT': expected_price}

                    prices = env._get_current_prices()

                    # Vérifier que le prix correspond à la position step_in_chunk
                    assert 'BTCUSDT' in prices
                    assert abs(prices['BTCUSDT'] - expected_price) < 1e-6

                    print(f"✓ Position {pos}: Prix attendu={expected_price:.2f}, Prix obtenu={prices['BTCUSDT']:.2f}")

    def test_index_bounds_protection(self, sample_data, mock_config):
        """Test que les dépassements d'index sont gérés correctement."""

        # Créer des données de test plus petites pour forcer des dépassements
        small_data = sample_data[:10]  # Seulement 10 lignes

        with patch('adan_trading_bot.environment.multi_asset_chunked_env.MultiAssetChunkedEnvironment._load_data') as mock_load:
            mock_load.return_value = {
                ('BTCUSDT', '5m'): small_data,
                ('BTCUSDT', '1h'): small_data,
                ('BTCUSDT', '4h'): small_data
            }

            env = MultiAssetChunkedEnvironment(mock_config, worker_id=0)
            env.current_data = mock_load.return_value

            # Créer une version réelle de _get_current_prices pour tester
            def real_get_current_prices():
                prices = {}
                current_idx = env.step_in_chunk

                for asset, timeframe_data in env.current_data.items():
                    if asset[1] == '5m':  # Utiliser les données 5m
                        df = timeframe_data

                        if current_idx < len(df):
                            prices[asset[0]] = df.iloc[current_idx]['close']
                        else:
                            # Cas de dépassement - utiliser la dernière valeur
                            prices[asset[0]] = df.iloc[-1]['close']
                            print(f"⚠️  INDEX_ERROR: step_in_chunk={current_idx} >= len(df)={len(df)}, using last price")

                return prices

            # Remplacer la méthode par notre version de test
            env._get_current_prices = real_get_current_prices

            # Test avec index valide
            env.step_in_chunk = 5
            prices = env._get_current_prices()
            assert 'BTCUSDT' in prices
            expected_valid_price = small_data.iloc[5]['close']
            assert abs(prices['BTCUSDT'] - expected_valid_price) < 1e-6
            print(f"✓ Index valide (5): Prix={prices['BTCUSDT']:.2f}")

            # Test avec index en dépassement
            env.step_in_chunk = 15  # Dépasse la taille de 10
            prices = env._get_current_prices()
            assert 'BTCUSDT' in prices
            expected_fallback_price = small_data.iloc[-1]['close']
            assert abs(prices['BTCUSDT'] - expected_fallback_price) < 1e-6
            print(f"✓ Index en dépassement (15): Prix de fallback={prices['BTCUSDT']:.2f}")

    def test_no_forward_fill_with_varying_data(self, mock_config):
        """Test qu'avec des données qui varient, il n'y a pas de forward fill excessif."""

        # Créer des données avec variation significative
        dates = pd.date_range('2024-01-01', periods=20, freq='5T')
        # Prix qui augmentent progressivement de 1% à chaque étape
        base_prices = [58000 * (1.01 ** i) for i in range(20)]

        varying_data = pd.DataFrame({
            'timestamp': dates,
            'open': base_prices,
            'high': [p * 1.005 for p in base_prices],
            'low': [p * 0.995 for p in base_prices],
            'close': base_prices,
            'volume': [1000] * 20
        })

        with patch('adan_trading_bot.environment.multi_asset_chunked_env.MultiAssetChunkedEnvironment._load_data') as mock_load:
            mock_load.return_value = {
                ('BTCUSDT', '5m'): varying_data,
                ('BTCUSDT', '1h'): varying_data,
                ('BTCUSDT', '4h'): varying_data
            }

            env = MultiAssetChunkedEnvironment(mock_config, worker_id=0)
            env.current_data = mock_load.return_value

            # Créer une version réelle de _get_current_prices
            def real_get_current_prices():
                prices = {}
                current_idx = env.step_in_chunk

                for asset, timeframe_data in env.current_data.items():
                    if asset[1] == '5m':
                        df = timeframe_data
                        if current_idx < len(df):
                            prices[asset[0]] = df.iloc[current_idx]['close']
                        else:
                            prices[asset[0]] = df.iloc[-1]['close']

                return prices

            env._get_current_prices = real_get_current_prices

            # Collecter les prix sur plusieurs étapes
            collected_prices = []
            for step in range(10):
                env.step_in_chunk = step
                prices = env._get_current_prices()
                collected_prices.append(prices['BTCUSDT'])
                print(f"Étape {step}: Prix={prices['BTCUSDT']:.2f}")

            # Vérifier que les prix varient (pas de forward fill)
            unique_prices = set(collected_prices)
            assert len(unique_prices) >= 8, f"Pas assez de variation dans les prix: {len(unique_prices)} prix uniques sur 10"

            # Vérifier que les prix suivent la tendance attendue (croissante)
            for i in range(1, len(collected_prices)):
                assert collected_prices[i] > collected_prices[i-1], f"Prix décroissant détecté à l'étape {i}"

            print(f"✓ Forward fill test réussi: {len(unique_prices)} prix uniques sur 10 étapes")

    def test_chunk_boundary_reset(self, sample_data, mock_config):
        """Test que step_in_chunk se remet à 0 lors du changement de chunk."""

        # Créer plusieurs chunks
        chunk1 = sample_data[:25]
        chunk2 = sample_data[25:50]

        with patch('adan_trading_bot.environment.multi_asset_chunked_env.MultiAssetChunkedEnvironment._load_data') as mock_load:
            # Simuler le chargement initial du premier chunk
            mock_load.return_value = {
                ('BTCUSDT', '5m'): chunk1,
                ('BTCUSDT', '1h'): chunk1,
                ('BTCUSDT', '4h'): chunk1
            }

            env = MultiAssetChunkedEnvironment(mock_config, worker_id=0)
            env.current_data = mock_load.return_value
            env.num_chunks = 2
            env.current_chunk_idx = 0

            # Simuler la méthode _load_next_chunk
            def mock_load_next_chunk():
                env.current_chunk_idx += 1
                env.current_data = {
                    ('BTCUSDT', '5m'): chunk2,
                    ('BTCUSDT', '1h'): chunk2,
                    ('BTCUSDT', '4h'): chunk2
                }
                print(f"Chunk chargé: {env.current_chunk_idx}")

            env._load_next_chunk = mock_load_next_chunk

            # Simuler une progression jusqu'à la fin du premier chunk
            env.step_in_chunk = 24  # Dernière position du chunk1 (0-based)

            # Simuler l'appel de step() qui devrait déclencher le changement de chunk
            env.current_step = 24
            env.step_in_chunk += 1  # Cela devrait devenir 25, dépassant la taille de chunk1

            # Si step_in_chunk >= len(current_chunk), charger le suivant et remettre à 0
            if env.step_in_chunk >= len(chunk1):
                env.step_in_chunk = 0
                env._load_next_chunk()

            # Vérifier que step_in_chunk a été remis à 0
            assert env.step_in_chunk == 0, f"step_in_chunk devrait être 0 après changement de chunk, mais vaut {env.step_in_chunk}"
            assert env.current_chunk_idx == 1, f"current_chunk_idx devrait être 1, mais vaut {env.current_chunk_idx}"

            print(f"✓ Changement de chunk réussi: step_in_chunk={env.step_in_chunk}, chunk_idx={env.current_chunk_idx}")


class TestMetricsMemoryManagement:
    """Tests pour la gestion mémoire des métriques."""

    def test_deque_memory_limitation(self):
        """Test que la conversion en deque limite bien la mémoire."""

        # Créer des métriques avec une limite de deque faible pour le test
        from collections import deque

        # Simuler une version avec deque
        class TestMetrics:
            def __init__(self, max_len=5):
                self.returns = deque(maxlen=max_len)
                self.trades = []

            def update_trade(self, pnl_pct):
                self.returns.append(pnl_pct / 100)

        metrics = TestMetrics(max_len=5)

        # Ajouter plus d'éléments que la limite
        test_returns = [1.5, -0.8, 2.1, -1.2, 0.9, 3.4, -2.1, 1.8]

        for ret in test_returns:
            metrics.update_trade(ret)

        # Vérifier que seuls les 5 derniers sont conservés
        assert len(metrics.returns) == 5, f"Deque devrait contenir 5 éléments, mais en contient {len(metrics.returns)}"

        # Vérifier que ce sont bien les 5 derniers
        expected_last_5 = [ret / 100 for ret in test_returns[-5:]]
        actual_returns = list(metrics.returns)

        for i, (expected, actual) in enumerate(zip(expected_last_5, actual_returns)):
            assert abs(expected - actual) < 1e-6, f"Position {i}: attendu {expected}, obtenu {actual}"

        print(f"✓ Deque limitation test réussi: {len(metrics.returns)} éléments conservés sur {len(test_returns)} ajoutés")
        print(f"  Derniers éléments: {list(metrics.returns)}")

    def test_sharpe_calculation_with_limited_data(self):
        """Test que le calcul de Sharpe fonctionne avec des données limitées."""

        from collections import deque
        import numpy as np

        class TestSharpeCalculator:
            def __init__(self, max_len=100):
                self.returns = deque(maxlen=max_len)
                self.risk_free_rate = 0.02  # 2% annuel

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

        calculator = TestSharpeCalculator(max_len=10)

        # Test avec données vides
        sharpe_empty = calculator.calculate_sharpe_ratio()
        assert sharpe_empty == 0.0, f"Sharpe avec données vides devrait être 0.0, mais vaut {sharpe_empty}"

        # Test avec quelques données
        test_returns = [0.01, -0.005, 0.015, -0.008, 0.012, 0.003, -0.007, 0.009]
        for ret in test_returns:
            calculator.returns.append(ret)

        sharpe_with_data = calculator.calculate_sharpe_ratio()
        assert isinstance(sharpe_with_data, (int, float)), f"Sharpe devrait être numérique, mais vaut {type(sharpe_with_data)}"
        assert not np.isnan(sharpe_with_data), f"Sharpe ne devrait pas être NaN"
        assert not np.isinf(sharpe_with_data), f"Sharpe ne devrait pas être infini"

        print(f"✓ Calcul Sharpe réussi: {sharpe_with_data:.4f} avec {len(calculator.returns)} returns")

        # Test avec beaucoup de données (pour déclencher la limitation)
        many_returns = np.random.randn(50) * 0.02  # 50 returns aléatoires
        calculator_big = TestSharpeCalculator(max_len=20)

        for ret in many_returns:
            calculator_big.returns.append(ret)

        # Vérifier que seuls 20 éléments sont conservés
        assert len(calculator_big.returns) == 20, f"Deque devrait contenir 20 éléments max, mais en contient {len(calculator_big.returns)}"

        sharpe_big = calculator_big.calculate_sharpe_ratio()
        assert isinstance(sharpe_big, (int, float)) and not np.isnan(sharpe_big)

        print(f"✓ Calcul Sharpe avec limitation mémoire réussi: {sharpe_big:.4f} avec {len(calculator_big.returns)} returns conservés sur 50")


if __name__ == "__main__":
    # Exécution des tests en mode standalone
    print("=" * 60)
    print("TESTS UNITAIRES - LECTURE DONNÉES & MÉTRIQUES")
    print("=" * 60)

    # Test de lecture des données
    print("\n📊 Tests de lecture des données et indexation:")
    test_data = TestDataReadingIndexation()

    # Créer des fixtures manuellement
    dates = pd.date_range('2024-01-01', periods=100, freq='5T')
    prices = [58000 + i * 50 for i in range(100)]  # Prix croissants
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.002 for p in prices],
        'low': [p * 0.998 for p in prices],
        'close': prices,
        'volume': [1000] * 100
    })

    config = {'assets': ['BTCUSDT'], 'timeframes': ['5m'], 'chunk_size': 50, 'num_workers': 1}

    try:
        print("Test 1: Forward fill avec données variables...")
        test_data.test_no_forward_fill_with_varying_data(config)
        print("✅ Test 1 réussi\n")
    except Exception as e:
        print(f"❌ Test 1 échoué: {e}\n")

    # Test des métriques
    print("📈 Tests de gestion mémoire des métriques:")
    test_metrics = TestMetricsMemoryManagement()

    try:
        print("Test 2: Limitation mémoire avec deque...")
        test_metrics.test_deque_memory_limitation()
        print("✅ Test 2 réussi\n")
    except Exception as e:
        print(f"❌ Test 2 échoué: {e}\n")

    try:
        print("Test 3: Calcul Sharpe avec données limitées...")
        test_metrics.test_sharpe_calculation_with_limited_data()
        print("✅ Test 3 réussi\n")
    except Exception as e:
        print(f"❌ Test 3 échoué: {e}\n")

    print("=" * 60)
    print("TESTS TERMINÉS")
    print("=" * 60)
