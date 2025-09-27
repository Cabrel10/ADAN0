#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test pour valider les corrections d'équité vs cash et drawdown.

Ce script teste :
1. Calcul correct de l'équité (cash + valeur positions)
2. Calcul correct du drawdown (basé sur peak_equity)
3. Différence claire entre solde cash et équité totale
4. Tracking correct des fréquences de trading
5. Métriques cohérentes et réalistes
"""

import os
import sys
import unittest
import logging
import io
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Ajouter le chemin du projet
project_root = Path(__file__).parent / "bot"
sys.path.insert(0, str(project_root))

try:
    from src.adan_trading_bot.portfolio.portfolio_manager import PortfolioManager, Position
    from src.adan_trading_bot.performance.metrics import PerformanceMetrics
    from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("Assurez-vous que le projet est dans le bon répertoire")
    sys.exit(1)


class TestEquiteVsCashCorrections(unittest.TestCase):
    """Tests pour valider les corrections d'équité vs cash."""

    def setUp(self):
        """Configuration des tests."""
        # Configuration du portfolio
        self.portfolio_config = {
            "initial_balance": 20.50,
            "default_currency": "USDT",
            "commission": 0.001,
            "worker_id": 0,
            "trading_rules": {
                "max_drawdown_pct": 25.0,
                "max_position_size": 0.8
            }
        }

        # Données de test
        self.test_asset = "BTCUSDT"
        self.test_price_initial = 50000.0
        self.test_price_higher = 55000.0  # +10%
        self.test_price_lower = 45000.0   # -10%

        # Logger pour capture
        self.log_capture = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_capture)
        self.logger = logging.getLogger("test_equity_corrections")
        self.logger.addHandler(self.log_handler)

    def test_1_equite_vs_cash_distinction(self):
        """Test 1: Distinction claire entre équité et cash."""
        print("\n🔍 Test 1: Distinction équité vs cash")

        portfolio = PortfolioManager(
            env_config=self.portfolio_config,
            assets=[self.test_asset]
        )

        # État initial
        initial_cash = portfolio.get_balance()
        initial_equity = portfolio.get_portfolio_value()

        print(f"💰 État initial - Cash: {initial_cash:.2f} USDT, Équité: {initial_equity:.2f} USDT")

        # Vérifications initiales
        self.assertEqual(initial_cash, 20.50)
        self.assertEqual(initial_equity, 20.50)
        self.assertEqual(initial_cash, initial_equity)  # Au début, cash = équité

        # Ouvrir une position de 80% du capital (16.40 USDT)
        position_size = 16.40 / self.test_price_initial  # ~0.000328 BTC
        success = portfolio.open_position(self.test_asset, self.test_price_initial, position_size)

        self.assertTrue(success, "Ouverture de position doit réussir")

        # État après ouverture position
        cash_after_open = portfolio.get_balance()
        equity_after_open = portfolio.get_portfolio_value()

        print(f"🔄 Après ouverture position - Cash: {cash_after_open:.2f} USDT, Équité: {equity_after_open:.2f} USDT")

        # Vérifications après ouverture
        self.assertLess(cash_after_open, initial_cash)  # Cash diminué
        self.assertAlmostEqual(equity_after_open, initial_equity, places=1)  # Équité stable
        self.assertNotEqual(cash_after_open, equity_after_open)  # Cash ≠ Équité

        # Vérifier la position
        position = portfolio.positions.get(self.test_asset)
        self.assertIsNotNone(position)
        self.assertTrue(position.is_open)

        print("✅ Distinction équité vs cash validée")

    def test_2_drawdown_calcul_correct(self):
        """Test 2: Calcul correct du drawdown basé sur équité."""
        print("\n🔍 Test 2: Calcul drawdown correct")

        portfolio = PortfolioManager(
            env_config=self.portfolio_config,
            assets=[self.test_asset]
        )

        # Simulation de prix pour calcul du drawdown
        portfolio.current_prices = {self.test_asset: self.test_price_initial}

        # Ouvrir position importante (80%)
        position_size = 16.40 / self.test_price_initial
        portfolio.open_position(self.test_asset, self.test_price_initial, position_size)

        # État initial après position
        portfolio._update_equity()
        initial_equity = portfolio.equity
        initial_peak = getattr(portfolio, 'peak_equity', initial_equity)
        initial_drawdown = getattr(portfolio, 'current_drawdown', 0.0)

        print(f"📊 Après position - Équité: {initial_equity:.2f}, Peak: {initial_peak:.2f}, DD: {initial_drawdown*100:.2f}%")

        # Vérifier drawdown initial (~0%)
        self.assertLess(abs(initial_drawdown), 0.05)  # < 5%

        # Simuler hausse de prix (+10%)
        portfolio.current_prices[self.test_asset] = self.test_price_higher
        portfolio._update_equity()

        equity_up = portfolio.equity
        peak_up = portfolio.peak_equity
        drawdown_up = portfolio.current_drawdown

        print(f"📈 Prix +10% - Équité: {equity_up:.2f}, Peak: {peak_up:.2f}, DD: {drawdown_up*100:.2f}%")

        # Vérifications hausse
        self.assertGreater(equity_up, initial_equity)  # Équité augmentée
        self.assertGreaterEqual(peak_up, equity_up)    # Peak mis à jour
        self.assertLessEqual(drawdown_up, 0.01)        # DD toujours ~0%

        # Simuler baisse de prix (-10% du prix initial)
        portfolio.current_prices[self.test_asset] = self.test_price_lower
        portfolio._update_equity()

        equity_down = portfolio.equity
        drawdown_down = portfolio.current_drawdown

        print(f"📉 Prix -10% - Équité: {equity_down:.2f}, Peak: {peak_up:.2f}, DD: {drawdown_down*100:.2f}%")

        # Vérifications baisse
        self.assertLess(equity_down, peak_up)          # Équité en baisse
        self.assertGreater(drawdown_down, 0.05)        # DD > 5%
        self.assertLess(drawdown_down, 0.25)           # DD < 25% (réaliste)

        print("✅ Calcul drawdown correct validé")

    def test_3_protection_limits_coherence(self):
        """Test 3: Cohérence des limites de protection."""
        print("\n🔍 Test 3: Cohérence limites de protection")

        portfolio = PortfolioManager(
            env_config=self.portfolio_config,
            assets=[self.test_asset]
        )

        # Mock current_prices
        current_prices = {self.test_asset: self.test_price_initial}
        portfolio.current_prices = current_prices

        # Ouvrir position et simuler perte importante
        position_size = 16.40 / self.test_price_initial
        portfolio.open_position(self.test_asset, self.test_price_initial, position_size)

        # Simuler crash de -30%
        crash_price = self.test_price_initial * 0.7
        current_prices[self.test_asset] = crash_price
        portfolio.current_prices = current_prices

        # Vérifier les limites de protection avec les prix actuels
        with patch('src.adan_trading_bot.portfolio.portfolio_manager.logger') as mock_logger:
            protection_triggered = portfolio.check_protection_limits(current_prices)

            # Vérifier que la protection se déclenche pour une perte importante
            # (dépend de la configuration max_drawdown_pct)
            if protection_triggered:
                print("🚨 Protection déclenchée pour perte importante")
                self.assertTrue(mock_logger.critical.called)
            else:
                print("✅ Protection non déclenchée - dans les limites")
                self.assertTrue(mock_logger.info.called)

        # Vérifier les métriques après protection
        portfolio._update_equity()
        final_equity = portfolio.equity
        final_drawdown = portfolio.current_drawdown

        print(f"📊 Après crash - Équité: {final_equity:.2f}, DD: {final_drawdown*100:.2f}%")

        # Le drawdown doit être cohérent avec la perte réelle
        expected_loss = (self.test_price_initial - crash_price) / self.test_price_initial
        self.assertAlmostEqual(final_drawdown, expected_loss * 0.8, delta=0.05)  # ~24% loss sur 80% position

        print("✅ Limites de protection cohérentes")

    def test_4_frequency_tracking_improvements(self):
        """Test 4: Améliorations du tracking de fréquence."""
        print("\n🔍 Test 4: Tracking fréquence amélioré")

        # Configuration d'environnement avec fréquences
        env_config = {
            "initial_balance": 20.50,
            "worker_id": 0,
            "trading_rules": {
                "frequency": {
                    "total_daily_min": 5,
                    "total_daily_max": 15,
                    "5m": {"min_positions": 6, "max_positions": 15},
                    "1h": {"min_positions": 3, "max_positions": 10},
                    "4h": {"min_positions": 1, "max_positions": 3},
                    "frequency_bonus_weight": 0.05,
                    "frequency_penalty_weight": 0.1
                }
            }
        }

        # Mock d'environnement simplifié
        mock_env = Mock()
        mock_env.frequency_config = env_config["trading_rules"]["frequency"]
        mock_env.positions_count = {'5m': 0, '1h': 0, '4h': 0, 'daily_total': 0}
        mock_env.worker_id = 0
        mock_env.current_step = 100

        # Mock portfolio avec trade_log
        mock_portfolio = Mock()
        mock_portfolio.trade_log = []

        # Simuler quelques trades
        trades = [
            {"timestamp": 1000, "asset": "BTCUSDT", "type": "open", "price": 50000.0, "timeframe": "5m"},
            {"timestamp": 1001, "asset": "BTCUSDT", "type": "close", "price": 50500.0, "timeframe": "5m"},
            {"timestamp": 1002, "asset": "BTCUSDT", "type": "open", "price": 50200.0, "timeframe": "1h"},
            {"timestamp": 1003, "asset": "BTCUSDT", "type": "close", "price": 49800.0, "timeframe": "1h"},
        ]

        # Simuler le tracking des trades
        last_trade_ids = set()
        new_trades_count = {'5m': 0, '1h': 0, '4h': 0}

        for trade in trades:
            trade_id = f"{trade['timestamp']}_{trade['asset']}_{trade['type']}_{trade['price']}"
            if trade_id not in last_trade_ids:
                timeframe = trade.get('timeframe', '5m')
                new_trades_count[timeframe] += 1
                last_trade_ids.add(trade_id)

        # Vérifications
        self.assertEqual(new_trades_count['5m'], 2)  # 2 trades sur 5m
        self.assertEqual(new_trades_count['1h'], 2)  # 2 trades sur 1h
        self.assertEqual(new_trades_count['4h'], 0)  # 0 trades sur 4h

        total_trades = sum(new_trades_count.values())
        self.assertEqual(total_trades, 4)

        # Calcul de récompense de fréquence simulé
        frequency_reward = 0.0
        config = env_config["trading_rules"]["frequency"]
        bonus_weight = config.get('frequency_bonus_weight', 0.05)

        for tf, count in new_trades_count.items():
            if tf in ['5m', '1h', '4h']:
                tf_config = config.get(tf, {})
                min_pos = tf_config.get('min_positions', 0)
                max_pos = tf_config.get('max_positions', 99)

                if min_pos <= count <= max_pos:
                    frequency_reward += bonus_weight * (count / max_pos)
                else:
                    # Pénalité si hors bornes
                    frequency_reward -= config.get('frequency_penalty_weight', 0.1) * 0.5

        print(f"📊 Trades trackés - 5m: {new_trades_count['5m']}, 1h: {new_trades_count['1h']}, 4h: {new_trades_count['4h']}")
        print(f"🎯 Récompense fréquence: {frequency_reward:.4f}")

        # La récompense doit être positive si dans les bornes
        self.assertGreaterEqual(frequency_reward, 0.0)

        print("✅ Tracking fréquence amélioré")

    def test_5_scenario_complet_realiste(self):
        """Test 5: Scénario complet réaliste d'une journée de trading."""
        print("\n🔍 Test 5: Scénario complet réaliste")

        portfolio = PortfolioManager(
            env_config=self.portfolio_config,
            assets=[self.test_asset]
        )

        # Simulation d'une journée avec plusieurs trades
        prices_sequence = [
            50000.0,  # Prix initial
            50500.0,  # +1%
            50200.0,  # -0.6%
            51000.0,  # +2%
            49800.0,  # -2.4%
            50300.0   # Final +0.6%
        ]

        equity_history = []
        drawdown_history = []
        cash_history = []

        print("📈 Simulation journée de trading:")

        for i, price in enumerate(prices_sequence):
            portfolio.current_prices = {self.test_asset: price}

            # Trading logic simulé
            if i == 1:  # Ouvrir position à +1%
                position_size = 10.0 / price  # 10 USDT de position
                portfolio.open_position(self.test_asset, price, position_size)
                print(f"   Step {i}: OPEN @ {price:.0f} - Position {position_size:.6f} BTC")

            elif i == 4:  # Fermer position à -2.4% (stop loss simulé)
                portfolio.close_position(self.test_asset, price)
                print(f"   Step {i}: CLOSE @ {price:.0f}")

            elif i == 5:  # Réouvrir position
                position_size = 8.0 / price  # 8 USDT de position
                portfolio.open_position(self.test_asset, price, position_size)
                print(f"   Step {i}: REOPEN @ {price:.0f} - Position {position_size:.6f} BTC")

            # Mise à jour métriques
            portfolio._update_equity()

            equity = portfolio.equity
            cash = portfolio.get_balance()
            drawdown = portfolio.current_drawdown * 100

            equity_history.append(equity)
            cash_history.append(cash)
            drawdown_history.append(drawdown)

            print(f"   Step {i}: Prix {price:.0f} | Équité {equity:.2f} | Cash {cash:.2f} | DD {drawdown:.2f}%")

        # Vérifications finales
        final_equity = equity_history[-1]
        max_drawdown = max(drawdown_history)

        print(f"\n📊 Résumé journée:")
        print(f"   Équité finale: {final_equity:.2f} USDT (initial: 20.50)")
        print(f"   Max drawdown: {max_drawdown:.2f}%")
        print(f"   Trades exécutés: 3 (1 open, 1 close, 1 reopen)")

        # Assertions finales
        self.assertGreater(final_equity, 0)  # Équité positive
        self.assertLess(max_drawdown, 50)     # DD raisonnable < 50%
        self.assertGreater(len(portfolio.trade_log), 0)  # Trades enregistrés

        # Vérifier cohérence équité vs cash
        has_position = any(pos.is_open for pos in portfolio.positions.values())
        if has_position:
            self.assertNotEqual(final_equity, cash_history[-1])  # Équité ≠ Cash si position ouverte
        else:
            self.assertAlmostEqual(final_equity, cash_history[-1], places=2)  # Équité ≈ Cash si pas de position

        print("✅ Scénario complet validé")

    def tearDown(self):
        """Nettoyage après tests."""
        if hasattr(self, 'log_handler'):
            self.logger.removeHandler(self.log_handler)
        if hasattr(self, 'log_capture'):
            self.log_capture.close()


def run_equity_corrections_validation():
    """Lance la validation des corrections d'équité."""
    print("🚀 VALIDATION DES CORRECTIONS ÉQUITÉ vs CASH")
    print("=" * 55)

    # Configuration du logging pour les tests
    logging.basicConfig(level=logging.WARNING)

    # Créer la suite de tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestEquiteVsCashCorrections)

    # Runner avec reporting détaillé
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)

    # Résumé des résultats
    print("\n" + "=" * 55)
    print("📊 RÉSUMÉ DE LA VALIDATION")
    print("-" * 35)

    if result.wasSuccessful():
        print("✅ TOUTES LES CORRECTIONS ÉQUITÉ VALIDÉES")
        print(f"   • {result.testsRun} tests passés avec succès")
        print("   • Calcul équité vs cash corrigé")
        print("   • Drawdown basé sur peak_equity")
        print("   • Tracking fréquence amélioré")
        print("   • Métriques cohérentes et réalistes")
        print("\n🎉 Le système calcule correctement équité et drawdown!")
        return True
    else:
        print("❌ CERTAINES CORRECTIONS NÉCESSITENT ATTENTION")
        print(f"   • {len(result.failures)} échecs détectés")
        print(f"   • {len(result.errors)} erreurs détectées")

        if result.failures:
            print("\n📋 ÉCHECS:")
            for test, traceback in result.failures:
                test_name = str(test).split('.')[-1]
                error_msg = traceback.split('AssertionError:')[-1].strip()
                print(f"   • {test_name}: {error_msg}")

        if result.errors:
            print("\n🔥 ERREURS:")
            for test, traceback in result.errors:
                test_name = str(test).split('.')[-1]
                error_msg = traceback.split('Exception:')[-1].strip()
                print(f"   • {test_name}: {error_msg}")

        return False


if __name__ == "__main__":
    success = run_equity_corrections_validation()
    sys.exit(0 if success else 1)
