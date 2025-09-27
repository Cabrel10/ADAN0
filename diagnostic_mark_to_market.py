#!/usr/bin/env python3
"""
Script de diagnostic spécifique pour le problème de valorisation mark-to-market.

Ce script trace précisément :
1. Les prix lus à chaque step (changent-ils vraiment ?)
2. Les appels à update_market_price
3. Les calculs de valorisation du portefeuille
4. L'évolution de la valeur totale

Usage:
    cd trading/
    python diagnostic_mark_to_market.py
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
from datetime import datetime
from pathlib import Path

# Ajouter le chemin du bot
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bot', 'src'))

try:
    from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
    from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager
except ImportError as e:
    print(f"Erreur d'import: {e}")
    sys.exit(1)

class MarkToMarketDiagnostic:
    """Diagnostic de la valorisation mark-to-market."""

    def __init__(self):
        self.price_history = []
        self.portfolio_value_history = []
        self.position_history = []
        self.update_calls = 0

    def create_test_environment(self):
        """Crée un environnement de test avec des données contrôlées."""

        # Configuration minimale
        config = {
            'data': {
                'assets': ['BTC'],
                'timeframes': ['5m'],
                'data_dir': 'data',
                'chunk_size': 100
            },
            'trading': {
                'initial_balance': 100.0,
                'max_positions': 1,
                'commission_pct': 0.001
            },
            'rewards': {
                'base_reward_multiplier': 1.0,
                'frequency_weight': 0.0  # Désactiver les pénalités pour ce test
            },
            'model': {
                'observation_space': {
                    'shape': [3, 20, 15]
                }
            }
        }

        return config

    def create_mock_price_data(self, steps=50):
        """Crée des données de prix contrôlées qui varient clairement."""

        print("📊 Création de données de prix contrôlées")

        # Prix de base BTC
        base_price = 50000.0
        prices = []

        for i in range(steps):
            # Variation sinusoïdale pour être sûr que ça change
            variation = 1000 * np.sin(i / 10.0)  # Variation de ±1000$
            price = base_price + variation
            prices.append(price)

        # Créer un DataFrame
        dates = pd.date_range('2024-01-01', periods=steps, freq='5min')
        df = pd.DataFrame({
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices],
            'close': prices,
            'volume': [1000.0] * steps
        }, index=dates)

        # Afficher quelques prix pour vérification
        print(f"  Prix créés: {len(prices)} points")
        print(f"  Prix initial: ${prices[0]:.2f}")
        print(f"  Prix à step 10: ${prices[10]:.2f}")
        print(f"  Prix à step 20: ${prices[20]:.2f}")
        print(f"  Variation max: ±${max(prices) - base_price:.2f}")

        return {'BTC': {'5m': df}}

    def trace_price_reading(self, env, step):
        """Trace la lecture des prix à un step donné."""

        print(f"\n🔍 TRACE STEP {step}: Lecture des prix")
        print("-" * 40)

        # Lire les prix via la méthode corrigée
        try:
            prices = env._get_current_prices()
            print(f"  step_in_chunk: {env.step_in_chunk}")
            print(f"  Prix lus: {prices}")

            # Enregistrer l'historique
            self.price_history.append({
                'step': step,
                'step_in_chunk': env.step_in_chunk,
                'prices': prices.copy() if prices else {}
            })

            return prices

        except Exception as e:
            print(f"  ❌ Erreur lecture prix: {e}")
            return {}

    def trace_portfolio_update(self, portfolio_manager, prices, step):
        """Trace la mise à jour du portefeuille."""

        print(f"\n💰 TRACE STEP {step}: Mise à jour portefeuille")
        print("-" * 45)

        # État avant mise à jour
        value_before = portfolio_manager.portfolio_value
        cash_before = portfolio_manager.cash

        print(f"  AVANT - Valeur: ${value_before:.2f}, Cash: ${cash_before:.2f}")

        # Positions ouvertes
        open_positions = {k: v for k, v in portfolio_manager.positions.items() if v.is_open}
        print(f"  Positions ouvertes: {len(open_positions)}")

        for asset, position in open_positions.items():
            print(f"    {asset}: {position.size:.8f} @ ${position.entry_price:.2f}")

        # Appel de mise à jour
        self.update_calls += 1
        try:
            portfolio_manager.update_market_price(prices)
            print(f"  ✅ update_market_price appelé (#{self.update_calls})")
        except Exception as e:
            print(f"  ❌ Erreur update_market_price: {e}")
            return

        # État après mise à jour
        value_after = portfolio_manager.portfolio_value
        cash_after = portfolio_manager.cash
        unrealized_pnl = portfolio_manager.unrealized_pnl

        print(f"  APRÈS - Valeur: ${value_after:.2f}, Cash: ${cash_after:.2f}")
        print(f"  PnL non réalisé: ${unrealized_pnl:.2f}")

        # Calculer la différence
        value_change = value_after - value_before
        print(f"  Variation: ${value_change:.2f}")

        if abs(value_change) < 0.01:
            print("  ⚠️  PROBLÈME: Aucune variation de valeur détectée!")
        else:
            print("  ✅ Valorisation mise à jour correctement")

        # Enregistrer l'historique
        self.portfolio_value_history.append({
            'step': step,
            'value_before': value_before,
            'value_after': value_after,
            'cash': cash_after,
            'unrealized_pnl': unrealized_pnl,
            'change': value_change
        })

        # Détails des positions après update
        for asset, position in open_positions.items():
            if asset in prices:
                expected_value = position.size * prices[asset]
                print(f"    {asset} après update:")
                print(f"      Prix courant: ${prices[asset]:.2f}")
                print(f"      Valeur attendue: ${expected_value:.2f}")
                print(f"      Valeur réelle: ${getattr(position, 'current_value', 'N/A')}")

                self.position_history.append({
                    'step': step,
                    'asset': asset,
                    'size': position.size,
                    'entry_price': position.entry_price,
                    'current_price': prices[asset],
                    'expected_value': expected_value,
                    'actual_value': getattr(position, 'current_value', 0)
                })

    def run_controlled_simulation(self):
        """Lance une simulation contrôlée pour diagnostiquer le problème."""

        print("🧪 SIMULATION CONTRÔLÉE - DIAGNOSTIC MARK-TO-MARKET")
        print("=" * 60)

        # Créer des données de test avec variation claire
        mock_data = self.create_mock_price_data(30)
        config = self.create_test_environment()

        print("\n🏗️  Création de l'environnement de test")

        # Simuler directement les composants critiques
        try:
            # Créer un gestionnaire de portefeuille de test
            portfolio_manager = PortfolioManager(config, worker_id=0)

            print(f"✅ Portfolio manager créé")
            print(f"  Capital initial: ${portfolio_manager.cash:.2f}")

            # Simuler l'ouverture d'une position
            print("\n📈 Simulation d'ouverture de position")

            # Prix initial
            initial_price = mock_data['BTC']['5m'].iloc[0]['close']
            position_size_usdt = 50.0  # 50$ de BTC
            btc_size = position_size_usdt / initial_price

            print(f"  Prix BTC initial: ${initial_price:.2f}")
            print(f"  Taille position: {btc_size:.8f} BTC (${position_size_usdt:.2f})")

            # Simuler l'ouverture (normalement fait par execute_trades)
            from adan_trading_bot.portfolio.portfolio_manager import Position

            position = Position()
            position.open(
                entry_price=initial_price,
                size=btc_size,
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
                open_step=0
            )

            portfolio_manager.positions['BTCUSDT'] = position
            portfolio_manager.cash -= position_size_usdt  # Réduire le cash

            print(f"✅ Position simulée ouverte")
            print(f"  Cash restant: ${portfolio_manager.cash:.2f}")

            # Maintenant tester la mise à jour sur plusieurs steps
            print("\n📊 Test de valorisation sur 10 steps")
            print("=" * 50)

            for step in range(10):
                print(f"\n--- STEP {step} ---")

                # Simuler la lecture du prix à ce step
                current_price = mock_data['BTC']['5m'].iloc[step]['close']
                prices = {'BTCUSDT': current_price}

                print(f"Prix BTC: ${current_price:.2f}")

                # Calculer la valeur attendue manuellement
                expected_position_value = btc_size * current_price
                expected_total_value = portfolio_manager.cash + expected_position_value

                print(f"Valeur position attendue: ${expected_position_value:.2f}")
                print(f"Valeur totale attendue: ${expected_total_value:.2f}")

                # Trace la mise à jour
                self.trace_portfolio_update(portfolio_manager, prices, step)

                # Comparer avec l'attendu
                actual_value = portfolio_manager.portfolio_value
                difference = abs(actual_value - expected_total_value)

                if difference > 0.01:
                    print(f"❌ ÉCART DÉTECTÉ: ${difference:.2f}")
                    print("   Le calcul de valorisation a un problème!")
                else:
                    print("✅ Valorisation correcte")

            # Analyse finale
            self.analyze_results()

        except Exception as e:
            print(f"❌ Erreur durant la simulation: {e}")
            import traceback
            traceback.print_exc()

    def analyze_results(self):
        """Analyse les résultats du diagnostic."""

        print("\n📋 ANALYSE DES RÉSULTATS")
        print("=" * 40)

        # Analyser les variations de prix
        if len(self.price_history) > 1:
            price_changes = []
            for i in range(1, len(self.price_history)):
                prev_price = list(self.price_history[i-1]['prices'].values())[0] if self.price_history[i-1]['prices'] else 0
                curr_price = list(self.price_history[i]['prices'].values())[0] if self.price_history[i]['prices'] else 0
                change = curr_price - prev_price
                price_changes.append(change)

            print(f"📈 Variations de prix:")
            print(f"  Nombre de changements: {len([c for c in price_changes if abs(c) > 0.01])}")
            print(f"  Variation max: ${max(price_changes) if price_changes else 0:.2f}")
            print(f"  Variation min: ${min(price_changes) if price_changes else 0:.2f}")

            if all(abs(c) < 0.01 for c in price_changes):
                print("❌ PROBLÈME: Les prix ne changent pas!")
            else:
                print("✅ Les prix varient correctement")

        # Analyser les variations de valeur du portefeuille
        if len(self.portfolio_value_history) > 1:
            value_changes = [h['change'] for h in self.portfolio_value_history]
            non_zero_changes = [c for c in value_changes if abs(c) > 0.01]

            print(f"\n💰 Variations de valeur du portefeuille:")
            print(f"  Total d'updates: {len(value_changes)}")
            print(f"  Changements significatifs: {len(non_zero_changes)}")
            print(f"  Appels update_market_price: {self.update_calls}")

            if len(non_zero_changes) == 0:
                print("❌ PROBLÈME CRITIQUE: La valeur du portefeuille ne change jamais!")
                print("   Causes possibles:")
                print("   1. update_market_price ne fonctionne pas")
                print("   2. Le calcul de valorisation est cassé")
                print("   3. Les positions ne sont pas correctement liées")
            else:
                print("✅ La valorisation fonctionne")

        # Diagnostic des positions
        if self.position_history:
            print(f"\n📊 Analyse des positions:")
            for pos in self.position_history[:3]:  # Afficher les 3 premiers
                expected = pos['expected_value']
                actual = pos['actual_value']
                diff = abs(expected - actual) if isinstance(actual, (int, float)) else float('inf')

                print(f"  Step {pos['step']}: Attendu ${expected:.2f}, Réel ${actual}, Diff ${diff:.2f}")

                if diff > 0.01:
                    print("    ❌ Position mal valorisée!")

        # Recommandations
        print(f"\n🎯 RECOMMANDATIONS:")

        if self.update_calls == 0:
            print("  1. 🔥 CRITIQUE: update_market_price n'est jamais appelé!")
            print("     → Vérifier l'intégration dans step()")
        elif len(self.portfolio_value_history) > 0 and all(abs(h['change']) < 0.01 for h in self.portfolio_value_history):
            print("  1. 🔥 CRITIQUE: update_market_price appelé mais inefficace!")
            print("     → Vérifier la logique interne de calcul")
        else:
            print("  1. ✅ La valorisation semble fonctionner")

        # Sauvegarder les résultats
        results = {
            'timestamp': datetime.now().isoformat(),
            'price_history': self.price_history,
            'portfolio_history': self.portfolio_value_history,
            'position_history': self.position_history,
            'update_calls': self.update_calls
        }

        with open('diagnostic_mark_to_market_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Résultats sauvegardés dans diagnostic_mark_to_market_results.json")

def main():
    """Fonction principale."""

    print("🩺 DIAGNOSTIC MARK-TO-MARKET - VALORISATION DU PORTEFEUILLE")
    print("=" * 70)

    diagnostic = MarkToMarketDiagnostic()
    diagnostic.run_controlled_simulation()

    print("\n🎯 Diagnostic terminé. Vérifiez les résultats ci-dessus.")

if __name__ == "__main__":
    main()
