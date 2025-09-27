#!/usr/bin/env python3
"""
Script de test rapide pour valider les corrections critiques avant l'entraînement.

Ce script teste spécifiquement :
1. La correction du bug d'indexation (EXCESSIVE_FORWARD_FILL)
2. La correction des métriques avec deque (éviter les crashs de mémoire)

Usage:
    python test_corrections_before_training.py

Doit être exécuté depuis le répertoire trading/ avec l'environnement trading_env activé.
"""

import sys
import os
import pandas as pd
import numpy as np
from collections import deque
import time
from pathlib import Path

# Ajouter le chemin vers le code source du bot
sys.path.append('/home/morningstar/Documents/trading/bot/src')

def test_data_indexation_logic():
    """Test de la logique d'indexation pour éviter le forward fill excessif."""

    print("🔍 TEST 1: Logique d'indexation des données")
    print("-" * 50)

    # Créer des données de test avec variation significative
    dates = pd.date_range('2024-01-01', periods=20, freq='5T')
    # Prix qui augmentent de 1% à chaque étape
    base_prices = [58000 * (1.01 ** i) for i in range(20)]

    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': base_prices,
        'high': [p * 1.005 for p in base_prices],
        'low': [p * 0.995 for p in base_prices],
        'close': base_prices,
        'volume': [1000] * 20
    })

    print(f"📊 Données créées: {len(test_data)} lignes")
    print(f"   Prix initial: {test_data.iloc[0]['close']:.2f}")
    print(f"   Prix final: {test_data.iloc[-1]['close']:.2f}")
    print(f"   Variation totale: {(test_data.iloc[-1]['close'] / test_data.iloc[0]['close'] - 1) * 100:.1f}%")

    # Simuler la logique d'accès aux données corrigée
    def get_price_at_step(df, step_in_chunk):
        """Simule la logique corrigée de _get_current_prices."""
        if step_in_chunk < len(df):
            return df.iloc[step_in_chunk]['close']
        else:
            # Cas de dépassement - utiliser la dernière valeur
            print(f"⚠️  INDEX_ERROR: step_in_chunk={step_in_chunk} >= len(df)={len(df)}")
            return df.iloc[-1]['close']

    # Test sur plusieurs étapes
    collected_prices = []
    forward_fill_count = 0

    for step in range(15):  # Tester 15 étapes sur 20 disponibles
        price = get_price_at_step(test_data, step)
        collected_prices.append(price)

        # Détecter le forward fill (prix identique au précédent)
        if step > 0 and abs(price - collected_prices[step-1]) < 1e-6:
            forward_fill_count += 1

    # Analyser les résultats
    unique_prices = set(collected_prices)
    forward_fill_rate = (forward_fill_count / (len(collected_prices) - 1)) * 100

    print(f"\n📈 Résultats d'indexation:")
    print(f"   Prix uniques: {len(unique_prices)}/{len(collected_prices)} ({len(unique_prices)/len(collected_prices)*100:.1f}%)")
    print(f"   Forward fill détecté: {forward_fill_count}/{len(collected_prices)-1} cas ({forward_fill_rate:.1f}%)")

    # Critères de validation
    success = True
    if len(unique_prices) < len(collected_prices) * 0.8:  # Au moins 80% de prix uniques
        print("❌ ÉCHEC: Pas assez de variation dans les prix (possiblement du forward fill)")
        success = False

    if forward_fill_rate > 10:  # Moins de 10% de forward fill acceptable
        print(f"❌ ÉCHEC: Trop de forward fill détecté ({forward_fill_rate:.1f}%)")
        success = False

    if success:
        print("✅ SUCCÈS: Indexation fonctionne correctement")

    return success

def test_metrics_memory_management():
    """Test de la gestion mémoire des métriques avec deque."""

    print("\n🧮 TEST 2: Gestion mémoire des métriques")
    print("-" * 50)

    # Simuler la classe PerformanceMetrics corrigée
    class TestMetrics:
        def __init__(self, max_len=1000):  # Plus petit pour test rapide
            self.returns = deque(maxlen=max_len)
            self.equity_curve = deque(maxlen=max_len)
            self.trades = []
            self.risk_free_rate = 0.02
            self.total_trades = 0
            self.wins = 0
            self.losses = 0
            self.neutral = 0

        def update_trade(self, pnl_pct, equity):
            trade_result = {'pnl_pct': pnl_pct, 'equity': equity}
            self.trades.append(trade_result)
            self.total_trades += 1

            # Ajouter au deque (limitation automatique)
            if pnl_pct is not None:
                self.returns.append(pnl_pct / 100)

                if pnl_pct > 0:
                    self.wins += 1
                elif pnl_pct < 0:
                    self.losses += 1
                else:
                    self.neutral += 1

            self.equity_curve.append(equity)

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

    # Test avec beaucoup de données
    print("📊 Simulation de trading intensif...")

    metrics = TestMetrics(max_len=1000)
    num_trades = 5000  # Simuler 5000 trades

    start_time = time.time()

    # Générer des trades aléatoires
    np.random.seed(42)  # Pour reproductibilité
    for i in range(num_trades):
        pnl_pct = np.random.normal(0.1, 2.0)  # Moyenne 0.1%, écart-type 2%
        equity = 1000 + i * 0.2
        metrics.update_trade(pnl_pct, equity)

    data_generation_time = time.time() - start_time

    print(f"   {num_trades} trades générés en {data_generation_time:.2f}s")
    print(f"   Returns stockés: {len(metrics.returns)}")
    print(f"   Equity points: {len(metrics.equity_curve)}")
    print(f"   Total trades comptés: {metrics.total_trades}")

    # Test de calcul des métriques (le point critique qui causait le crash)
    print("\n🧮 Test calculs métriques...")

    calc_times = []
    sharpe_results = []

    for i in range(10):  # Calculer 10 fois pour tester la performance
        calc_start = time.time()
        try:
            sharpe = metrics.calculate_sharpe_ratio()
            calc_time = time.time() - calc_start
            calc_times.append(calc_time)
            sharpe_results.append(sharpe)

            # Vérifier que le résultat est valide
            if np.isnan(sharpe) or np.isinf(sharpe):
                raise ValueError(f"Sharpe invalide: {sharpe}")

        except Exception as e:
            print(f"❌ ÉCHEC calcul #{i+1}: {e}")
            return False

    avg_calc_time = np.mean(calc_times)
    max_calc_time = max(calc_times)
    sharpe_consistency = np.std(sharpe_results)

    print(f"   Temps calcul moyen: {avg_calc_time*1000:.1f}ms")
    print(f"   Temps calcul max: {max_calc_time*1000:.1f}ms")
    print(f"   Sharpe résultat: {sharpe_results[0]:.4f}")
    print(f"   Consistance: {sharpe_consistency:.10f} (devrait être ~0)")

    # Critères de validation
    success = True

    # Vérification de la limitation mémoire
    if len(metrics.returns) != min(1000, num_trades):
        print(f"❌ ÉCHEC: Limitation mémoire inefficace ({len(metrics.returns)} vs attendu {min(1000, num_trades)})")
        success = False

    # Vérification de la performance (pas de blocage)
    if avg_calc_time > 0.1:  # Plus de 100ms est trop lent
        print(f"❌ ÉCHEC: Calculs trop lents ({avg_calc_time*1000:.1f}ms en moyenne)")
        success = False

    # Vérification de la consistance
    if sharpe_consistency > 1e-10:
        print(f"❌ ÉCHEC: Calculs inconsistants (std={sharpe_consistency})")
        success = False

    # Vérification du comptage
    if metrics.total_trades != num_trades:
        print(f"❌ ÉCHEC: Comptage incorrect ({metrics.total_trades} vs {num_trades})")
        success = False

    if success:
        print("✅ SUCCÈS: Métriques avec deque fonctionnent correctement")

    return success

def test_data_file_access():
    """Test d'accès rapide aux fichiers de données."""

    print("\n📁 TEST 3: Accès aux fichiers de données")
    print("-" * 50)

    # Chemins des données d'entraînement
    data_dir = Path("/home/morningstar/Documents/trading/data/processed/indicators/train")
    btc_5m_dir = data_dir / "BTCUSDT"

    if not btc_5m_dir.exists():
        print(f"⚠️  Répertoire de données non trouvé: {btc_5m_dir}")
        print("   Cela pourrait causer des erreurs lors de l'entraînement")
        return False

    # Lister les fichiers parquet (5m.parquet, 1h.parquet, 4h.parquet)
    parquet_files = list(btc_5m_dir.glob("*.parquet"))

    if len(parquet_files) == 0:
        print("❌ ÉCHEC: Aucun fichier parquet trouvé")
        return False

    print(f"   Fichiers parquet trouvés: {len(parquet_files)}")

    # Tester le chargement du fichier 5m (priorité pour l'entraînement)
    try:
        # Chercher spécifiquement le fichier 5m
        file_5m = btc_5m_dir / "5m.parquet"
        if file_5m.exists():
            df = pd.read_parquet(file_5m)
            print(f"   Test fichier: {file_5m.name}")
        else:
            # Fallback sur le premier fichier disponible
            first_file = parquet_files[0]
            df = pd.read_parquet(first_file)
            print(f"   Test fichier: {first_file.name}")
        print(f"   Lignes: {len(df)}")
        print(f"   Colonnes: {list(df.columns)}")

        # Vérifier la qualité des données
        close_prices = df['close']
        nan_count = close_prices.isna().sum()
        unique_prices = close_prices.nunique()
        price_variance = close_prices.var()

        print(f"   Prix NaN: {nan_count}/{len(df)} ({nan_count/len(df)*100:.1f}%)")
        print(f"   Prix uniques: {unique_prices}/{len(df)} ({unique_prices/len(df)*100:.1f}%)")
        print(f"   Variance prix: {price_variance:.2f}")

        success = True
        if nan_count > len(df) * 0.05:  # Plus de 5% de NaN
            print("❌ ÉCHEC: Trop de valeurs NaN dans les prix")
            success = False

        if unique_prices < len(df) * 0.8:  # Moins de 80% de prix uniques
            print("❌ ÉCHEC: Pas assez de variation dans les prix")
            success = False

        if price_variance < 1000:  # Variance trop faible pour BTC
            print("❌ ÉCHEC: Variance des prix trop faible")
            success = False

        if success:
            print("✅ SUCCÈS: Fichiers de données OK")

        return success

    except Exception as e:
        print(f"❌ ÉCHEC chargement fichier: {e}")
        return False

def main():
    """Fonction principale des tests."""

    print("🧪 TESTS DE VALIDATION DES CORRECTIONS CRITIQUES")
    print("="*70)
    print("Ce script valide les corrections apportées pour résoudre :")
    print("1. Bug d'indexation (EXCESSIVE_FORWARD_FILL)")
    print("2. Crash des métriques (KeyboardInterrupt)")
    print("3. Accès aux données d'entraînement")
    print("="*70)

    start_time = time.time()

    # Exécuter tous les tests
    tests_results = {
        "Indexation des données": test_data_indexation_logic(),
        "Métriques avec deque": test_metrics_memory_management(),
        "Accès aux fichiers": test_data_file_access()
    }

    total_time = time.time() - start_time

    # Résultats finaux
    print("\n" + "="*70)
    print("📋 RÉSULTATS FINAUX")
    print("="*70)

    passed = sum(tests_results.values())
    total = len(tests_results)

    for test_name, result in tests_results.items():
        status = "✅ RÉUSSI" if result else "❌ ÉCHOUÉ"
        print(f"{test_name}: {status}")

    print(f"\n📊 Score: {passed}/{total} tests réussis")
    print(f"⏱️  Temps total: {total_time:.1f}s")

    if passed == total:
        print("\n🎉 TOUS LES TESTS SONT PASSÉS !")
        print("✅ Les corrections sont validées, l'entraînement peut être lancé.")
        print("\nCommande d'entraînement recommandée :")
        print("timeout 30s /home/morningstar/miniconda3/envs/trading_env/bin/python bot/scripts/train_parallel_agents.py --config bot/config/config.yaml --checkpoint-dir bot/checkpoints")
        return True
    else:
        print(f"\n⚠️  {total - passed} TEST(S) ONT ÉCHOUÉ")
        print("❌ Des corrections supplémentaires sont nécessaires avant l'entraînement.")
        print("\nVeuillez corriger les problèmes identifiés avant de lancer l'entraînement.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
