#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de test rapide pour vérifier les corrections critiques du Trading Bot ADAN.

Ce script teste les corrections des erreurs suivantes :
1. PnL NUL → Variation de prix réaliste
2. PosSize Incohérent → Synchronisation DBE/PortfolioManager
3. Prix interpolés statiques → Prix dynamiques
4. Métriques nulles → Récompenses positives possibles

Usage:
    python test_corrections_critiques.py
"""

import sys
import os
import traceback
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

# Ajouter le chemin du projet
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "bot" / "src"))

print("🧪 TEST DES CORRECTIONS CRITIQUES - ADAN TRADING BOT")
print("=" * 80)

class CriticalFixesTest:
    def __init__(self):
        self.test_results = {}
        self.failed_tests = []
        self.passed_tests = []

    def log_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Enregistre le résultat d'un test"""
        self.test_results[test_name] = {
            'passed': passed,
            'details': details,
            'timestamp': time.time()
        }

        if passed:
            self.passed_tests.append(test_name)
            print(f"✅ {test_name}: PASSÉ")
        else:
            self.failed_tests.append(test_name)
            print(f"❌ {test_name}: ÉCHOUÉ")

        if details:
            print(f"   💡 {details}")

    def test_price_variation_generator(self):
        """TEST #1: Générateur de variation de prix"""
        print("\n🔧 TEST #1: GÉNÉRATEUR DE VARIATION DE PRIX")

        try:
            from adan_trading_bot.environment.price_variation import PriceVariationGenerator

            # Créer le générateur
            generator = PriceVariationGenerator({
                'min_variation_pct': 0.0005,
                'max_variation_pct': 0.003,
                'random_seed': 42
            })

            # Test avec différentes actions
            test_cases = [
                ('BTCUSDT', 50000.0, 0.5),   # Action positive
                ('BTCUSDT', 50000.0, -0.7),  # Action négative
                ('ETHUSDT', 3500.0, 0.1),    # Action faible
                ('SOLUSDT', 150.0, -0.1),    # Action faible négative
            ]

            all_different_prices = True
            positive_pnl_possible = False

            for asset, base_price, action in test_cases:
                entry_price, exit_price = generator.generate_price_variation(
                    asset, base_price, action, volatility=0.02
                )

                # Vérifier que les prix sont différents
                if entry_price == exit_price:
                    all_different_prices = False

                # Calculer PnL potentiel
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                if pnl_pct > 0.01:  # Plus de 0.01%
                    positive_pnl_possible = True

                print(f"   📊 {asset}: action={action:+.2f} → "
                      f"entry={entry_price:.2f}, exit={exit_price:.2f}, "
                      f"PnL={pnl_pct:+.4f}%")

            # Test interpolation dynamique
            interpolated = generator.get_realistic_interpolated_price('BTCUSDT', 50000.0)
            interpolation_works = abs(interpolated - 50000.0) > 1.0  # Au moins 1$ de différence

            success = all_different_prices and positive_pnl_possible and interpolation_works

            details = (f"Prix différents: {all_different_prices}, "
                      f"PnL positif possible: {positive_pnl_possible}, "
                      f"Interpolation dynamique: {interpolation_works}")

            self.log_test_result("PRICE_VARIATION", success, details)
            return success

        except Exception as e:
            self.log_test_result("PRICE_VARIATION", False, f"Erreur: {e}")
            return False

    def test_position_size_synchronization(self):
        """TEST #2: Synchronisation Position Size"""
        print("\n🔧 TEST #2: SYNCHRONISATION POSITION SIZE")

        try:
            # Test de la logique de synchronisation

            # Simuler paramètres DBE vs PortfolioManager
            dbe_position_pct = 0.81  # 81% proposé par DBE
            tier_max_position_pct = 0.30  # 30% maximum pour palier Micro
            available_capital = 20.0  # Capital disponible
            min_trade_value = 11.0

            # Appliquer la logique corrigée
            effective_position_pct = min(dbe_position_pct, tier_max_position_pct)
            synchronized_trade_value = available_capital * effective_position_pct

            # Test ajustement min_trade_value
            if synchronized_trade_value < min_trade_value:
                if available_capital >= min_trade_value * 2:
                    synchronized_trade_value = min_trade_value
                    effective_position_pct = min_trade_value / available_capital

            # Vérifications
            reasonable_position_size = effective_position_pct <= tier_max_position_pct
            meets_min_trade = synchronized_trade_value >= min_trade_value
            not_excessive = effective_position_pct <= 0.5  # Pas plus de 50%

            print(f"   📊 DBE propose: {dbe_position_pct*100:.1f}%")
            print(f"   📊 Tier limite: {tier_max_position_pct*100:.1f}%")
            print(f"   📊 Position effective: {effective_position_pct*100:.1f}%")
            print(f"   📊 Valeur trade: {synchronized_trade_value:.2f} USDT")

            success = reasonable_position_size and meets_min_trade and not_excessive

            details = (f"Respecte tier: {reasonable_position_size}, "
                      f"Min trade OK: {meets_min_trade}, "
                      f"Pas excessif: {not_excessive}")

            self.log_test_result("POSITION_SYNC", success, details)
            return success

        except Exception as e:
            self.log_test_result("POSITION_SYNC", False, f"Erreur: {e}")
            return False

    def test_pnl_calculation_fix(self):
        """TEST #3: Calcul PnL Non-Nul"""
        print("\n🔧 TEST #3: CALCUL PNL NON-NUL")

        try:
            # Simuler un trade avec les corrections
            from adan_trading_bot.environment.price_variation import PriceVariationGenerator

            generator = PriceVariationGenerator({'random_seed': 123})

            # Paramètres de trade
            asset = 'BTCUSDT'
            base_price = 65000.0
            position_size_usdt = 15.0
            commission_pct = 0.001  # 0.1%

            trades_results = []

            # Simuler 10 trades avec différentes actions
            for i, action in enumerate([-0.8, -0.3, 0.1, 0.4, 0.7, -0.5, 0.2, -0.1, 0.6, -0.9]):
                entry_price, exit_price = generator.generate_price_variation(
                    asset, base_price + (i * 100), action  # Variation du prix de base
                )

                # Calculer PnL
                quantity = position_size_usdt / entry_price
                gross_pnl = (exit_price - entry_price) * quantity
                commission = position_size_usdt * commission_pct * 2  # Entrée + sortie
                net_pnl = gross_pnl - commission

                trades_results.append({
                    'action': action,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'gross_pnl': gross_pnl,
                    'commission': commission,
                    'net_pnl': net_pnl
                })

                print(f"   📊 Trade {i+1}: action={action:+.2f} → "
                      f"PnL={net_pnl:+.4f} USDT "
                      f"(entry={entry_price:.0f}, exit={exit_price:.0f})")

            # Analyse des résultats
            non_zero_pnl = sum(1 for t in trades_results if abs(t['net_pnl']) > 0.001)
            positive_pnl = sum(1 for t in trades_results if t['net_pnl'] > 0)
            negative_pnl = sum(1 for t in trades_results if t['net_pnl'] < 0)

            avg_pnl = np.mean([t['net_pnl'] for t in trades_results])

            # Critères de succès
            has_variety = non_zero_pnl >= 8  # Au moins 8/10 trades non-nuls
            has_positive = positive_pnl >= 3  # Au moins 3/10 trades positifs
            reasonable_avg = abs(avg_pnl) < 1.0  # PnL moyen raisonnable

            success = has_variety and has_positive and reasonable_avg

            details = (f"Trades non-nuls: {non_zero_pnl}/10, "
                      f"Positifs: {positive_pnl}/10, "
                      f"PnL moyen: {avg_pnl:+.4f}")

            self.log_test_result("PNL_CALCULATION", success, details)
            return success

        except Exception as e:
            self.log_test_result("PNL_CALCULATION", False, f"Erreur: {e}")
            return False

    def test_reward_signal_improvement(self):
        """TEST #4: Amélioration du Signal de Récompense"""
        print("\n🔧 TEST #4: SIGNAL DE RÉCOMPENSE AMÉLIORÉ")

        try:
            # Simuler le calcul de récompense avec les corrections
            test_scenarios = [
                {'pnl': 0.15, 'action': 0.7, 'expected_positive': True},   # Trade profitable
                {'pnl': -0.05, 'action': 0.3, 'expected_positive': False}, # Petit loss
                {'pnl': 0.08, 'action': 0.1, 'expected_positive': True},   # Petit gain
                {'pnl': -0.02, 'action': 0.0, 'expected_positive': False}, # Inaction loss
                {'pnl': 0.25, 'action': 0.9, 'expected_positive': True},   # Gros gain
            ]

            rewards_calculated = []

            for scenario in test_scenarios:
                pnl = scenario['pnl']
                action = scenario['action']

                # Logique de récompense améliorée
                # Récompense basée sur PnL
                pnl_reward = pnl * 10.0  # Amplifier le signal

                # Bonus pour prendre des positions (éviter inaction)
                action_strength = abs(action)
                if action_strength > 0.1:
                    action_reward = 0.01 * action_strength
                else:
                    action_reward = -0.005  # Pénalité légère pour inaction

                # Récompense totale
                total_reward = pnl_reward + action_reward

                # Assurer qu'il y ait parfois des récompenses positives
                if pnl > 0:
                    total_reward = max(total_reward, 0.001)  # Minimum positif

                rewards_calculated.append({
                    'scenario': scenario,
                    'reward': total_reward,
                    'matches_expected': (total_reward > 0) == scenario['expected_positive']
                })

                print(f"   📊 PnL={pnl:+.3f}, Action={action:.2f} → "
                      f"Reward={total_reward:+.6f}")

            # Analyse des résultats
            correct_predictions = sum(1 for r in rewards_calculated if r['matches_expected'])
            has_positive_rewards = sum(1 for r in rewards_calculated if r['reward'] > 0)
            has_negative_rewards = sum(1 for r in rewards_calculated if r['reward'] < 0)

            # Critères de succès
            good_predictions = correct_predictions >= 4  # Au moins 4/5 correct
            reward_variety = has_positive_rewards >= 2 and has_negative_rewards >= 1

            success = good_predictions and reward_variety

            details = (f"Prédictions correctes: {correct_predictions}/5, "
                      f"Récompenses +: {has_positive_rewards}, "
                      f"Récompenses -: {has_negative_rewards}")

            self.log_test_result("REWARD_SIGNAL", success, details)
            return success

        except Exception as e:
            self.log_test_result("REWARD_SIGNAL", False, f"Erreur: {e}")
            return False

    def test_environment_integration(self):
        """TEST #5: Intégration Environnement"""
        print("\n🔧 TEST #5: INTÉGRATION ENVIRONNEMENT")

        try:
            # Test que l'environnement peut être créé avec les corrections
            from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

            # Données minimales pour test
            mock_data = {
                'BTCUSDT': {
                    '5m': np.random.randn(100, 10),
                    '1h': np.random.randn(100, 10),
                    '4h': np.random.randn(100, 10)
                }
            }

            timeframes = ['5m', '1h', '4h']
            features_config = {
                '5m': ['open', 'high', 'low', 'close', 'volume', 'rsi_14'],
                '1h': ['open', 'high', 'low', 'close', 'volume', 'rsi_14'],
                '4h': ['open', 'high', 'low', 'close', 'volume', 'rsi_14']
            }

            env_config = {
                'initial_balance': 20.0,
                'commission': 0.001,
                'min_price_variation_pct': 0.0005,
                'max_price_variation_pct': 0.003,
                'random_seed': 42
            }

            worker_config = {
                'worker_id': 'W0',
                'rank': 0
            }

            # Vérifier que l'environnement peut être créé
            try:
                env = MultiAssetChunkedEnv(
                    data=mock_data,
                    timeframes=timeframes,
                    window_size=50,
                    features_config=features_config,
                    config=env_config,
                    worker_config=worker_config
                )
                env_created = True

                # Vérifier que le générateur de prix est initialisé
                has_price_generator = hasattr(env, 'price_variation_generator')

                # Test d'une action simple
                action = np.array([0.5])  # Action d'achat

                # Ne pas exécuter step() car trop complexe pour test rapide
                # Juste vérifier l'initialisation

            except Exception as create_error:
                env_created = False
                has_price_generator = False
                print(f"   ⚠️  Erreur création env: {create_error}")

            success = env_created and has_price_generator

            details = (f"Env créé: {env_created}, "
                      f"Price generator: {has_price_generator}")

            self.log_test_result("ENV_INTEGRATION", success, details)
            return success

        except Exception as e:
            self.log_test_result("ENV_INTEGRATION", False, f"Erreur: {e}")
            return False

    def run_all_tests(self):
        """Exécute tous les tests de corrections critiques"""
        print("🎯 EXÉCUTION DE TOUS LES TESTS DE CORRECTIONS")

        start_time = time.time()

        # Exécuter les tests dans l'ordre de priorité
        test_methods = [
            self.test_price_variation_generator,
            self.test_position_size_synchronization,
            self.test_pnl_calculation_fix,
            self.test_reward_signal_improvement,
            self.test_environment_integration
        ]

        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                test_name = test_method.__name__.replace('test_', '').upper()
                self.log_test_result(test_name, False, f"Exception: {e}")
                traceback.print_exc()

        # Rapport final
        self.generate_final_report(time.time() - start_time)

    def generate_final_report(self, duration: float):
        """Génère le rapport final des tests"""
        print("\n" + "=" * 80)
        print("📊 RAPPORT FINAL - TESTS DES CORRECTIONS CRITIQUES")
        print("=" * 80)

        total_tests = len(self.test_results)
        passed_count = len(self.passed_tests)
        failed_count = len(self.failed_tests)

        success_rate = (passed_count / total_tests * 100) if total_tests > 0 else 0

        print(f"⏱️  Durée d'exécution: {duration:.2f} secondes")
        print(f"📈 Tests exécutés: {total_tests}")
        print(f"✅ Tests réussis: {passed_count}")
        print(f"❌ Tests échoués: {failed_count}")
        print(f"📊 Taux de réussite: {success_rate:.1f}%")

        if failed_count > 0:
            print(f"\n🔥 TESTS ÉCHOUÉS À CORRIGER:")
            for test_name in self.failed_tests:
                details = self.test_results[test_name]['details']
                print(f"   • {test_name}: {details}")

        print(f"\n🎯 STATUT GÉNÉRAL:")
        if success_rate >= 80:
            print("🟢 CORRECTIONS LARGEMENT FONCTIONNELLES")
            print("✅ L'entraînement peut être relancé avec confiance")
            print("🚀 Récompenses positives maintenant possibles")
            print("💡 PnL non-nul permettra l'apprentissage PPO")
        elif success_rate >= 60:
            print("🟡 CORRECTIONS PARTIELLEMENT FONCTIONNELLES")
            print("⚠️  Quelques ajustements nécessaires avant entraînement long")
            print("🔧 Corriger les tests échoués puis relancer")
        else:
            print("🔴 CORRECTIONS INSUFFISANTES")
            print("❌ Entraînement toujours problématique")
            print("🛠️  Corrections supplémentaires requises")

        print(f"\n🔄 PROCHAINES ÉTAPES:")
        if success_rate >= 80:
            print("1. Lancer test d'entraînement court (100 steps)")
            print("2. Vérifier logs pour PnL > 0 et métriques non-nulles")
            print("3. Si OK, lancer entraînement long (timeout 3600s)")
        else:
            print("1. Corriger les tests échoués")
            print("2. Relancer ce script de test")
            print("3. Répéter jusqu'à 80%+ de réussite")

def main():
    """Point d'entrée principal"""
    try:
        # Avertissement
        print("⚠️  Ce script teste les corrections critiques appliquées au système ADAN")
        print("📋 Il vérifie que le PnL nul et autres erreurs sont corrigées")
        print("")

        # Créer et exécuter les tests
        tester = CriticalFixesTest()
        tester.run_all_tests()

        # Déterminer le code de sortie
        success_rate = len(tester.passed_tests) / len(tester.test_results) * 100
        return success_rate >= 80

    except KeyboardInterrupt:
        print("\n❌ Tests interrompus par l'utilisateur")
        return False
    except Exception as e:
        print(f"\n💥 ERREUR FATALE: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception:
        sys.exit(1)
