#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de test pour les fonctions de gestion des fonds du PortfolioManager.
"""

import sys
import os
from datetime import datetime
import numpy as np

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager


def test_fund_management():
    """Test complet des fonctions de gestion des fonds."""

    print("🧪 DÉBUT DES TESTS DE GESTION DES FONDS")
    print("=" * 60)

    # Configuration de test
    config = {
        "environment": {"initial_balance": 100.0},
        "assets": ["BTCUSDT", "XRPUSDT"],
        "risk_management": {"max_positions": 5},
    }

    # Initialiser le PortfolioManager
    pm = PortfolioManager(config, worker_id=0)

    print(f"💰 État initial:")
    print(f"   - Cash: ${pm.cash:.2f}")
    print(f"   - Equity: ${pm.equity:.2f}")
    print(f"   - Portfolio Value: ${pm.portfolio_value:.2f}")

    # Test du state vector initial
    initial_state = pm.get_state_vector()
    print(f"   - State Vector Length: {len(initial_state)}")
    print(f"   - Trading PnL %: {initial_state[2]:.4f}")
    print(f"   - External Flow %: {initial_state[3]:.4f}")
    print()

    # TEST 1: Dépôt de fonds
    print("📈 TEST 1: Dépôt de fonds")
    try:
        deposit_result = pm.deposit_funds(50.0, "Test deposit #1")
        print(f"   ✅ Dépôt réussi: {deposit_result['amount']}$")
        print(f"   - Nouveau Cash: ${pm.cash:.2f}")
        print(f"   - Nouvelle Equity: ${pm.equity:.2f}")

        # Vérifier le state vector après dépôt
        state_after_deposit = pm.get_state_vector()
        print(f"   - Trading PnL %: {state_after_deposit[2]:.4f}")
        print(f"   - External Flow %: {state_after_deposit[3]:.4f}")
        print(f"   - Total Deposits %: {state_after_deposit[4]:.4f}")

    except Exception as e:
        print(f"   ❌ Erreur lors du dépôt: {e}")
    print()

    # TEST 2: Deuxième dépôt
    print("📈 TEST 2: Deuxième dépôt")
    try:
        deposit_result2 = pm.deposit_funds(25.0, "Test deposit #2")
        print(f"   ✅ Dépôt réussi: {deposit_result2['amount']}$")
        print(f"   - Nouveau Cash: ${pm.cash:.2f}")
        print(f"   - Nouvelle Equity: ${pm.equity:.2f}")

    except Exception as e:
        print(f"   ❌ Erreur lors du dépôt: {e}")
    print()

    # TEST 3: Retrait de fonds (valide)
    print("📉 TEST 3: Retrait de fonds valide")
    try:
        withdrawal_result = pm.withdraw_funds(30.0, "Test withdrawal #1")
        print(f"   ✅ Retrait réussi: {withdrawal_result['amount']}$")
        print(f"   - Nouveau Cash: ${pm.cash:.2f}")
        print(f"   - Nouvelle Equity: ${pm.equity:.2f}")

        # Vérifier le state vector après retrait
        state_after_withdrawal = pm.get_state_vector()
        print(f"   - Trading PnL %: {state_after_withdrawal[2]:.4f}")
        print(f"   - External Flow %: {state_after_withdrawal[3]:.4f}")
        print(f"   - Total Withdrawals %: {state_after_withdrawal[5]:.4f}")

    except Exception as e:
        print(f"   ❌ Erreur lors du retrait: {e}")
    print()

    # TEST 4: Retrait avec fonds insuffisants
    print("📉 TEST 4: Retrait avec fonds insuffisants")
    try:
        withdrawal_result = pm.withdraw_funds(200.0, "Test withdrawal excessive")
        print(f"   ❌ Retrait inapproprié réussi: {withdrawal_result['amount']}$")

    except ValueError as e:
        print(f"   ✅ Erreur attendue: {e}")
    except Exception as e:
        print(f"   ❌ Erreur inattendue: {e}")
    print()

    # TEST 5: Retrait forcé
    print("📉 TEST 5: Retrait forcé (découvert)")
    try:
        withdrawal_result = pm.withdraw_funds(
            200.0, "Test forced withdrawal", force=True
        )
        print(f"   ✅ Retrait forcé réussi: {withdrawal_result['amount']}$")
        print(f"   - Nouveau Cash: ${pm.cash:.2f}")
        print(f"   - Nouvelle Equity: ${pm.equity:.2f}")
        print(f"   - Forced: {withdrawal_result['forced']}")

    except Exception as e:
        print(f"   ❌ Erreur lors du retrait forcé: {e}")
    print()

    # TEST 6: Résumé des opérations
    print("📊 TEST 6: Résumé des opérations de fonds")
    try:
        summary = pm.get_fund_operations_summary()
        print(f"   - Total Dépôts: ${summary['total_deposits']:.2f}")
        print(f"   - Total Retraits: ${summary['total_withdrawals']:.2f}")
        print(f"   - Flux Net: ${summary['net_external_flow']:.2f}")
        print(f"   - Nombre d'opérations: {summary['operations_count']}")
        print(
            f"   - Dernière opération: {summary['last_operation']['type']} de ${summary['last_operation']['amount']:.2f}"
        )

    except Exception as e:
        print(f"   ❌ Erreur lors du résumé: {e}")
    print()

    # TEST 7: Analyse Trading PnL vs Flux Externes
    print("🔍 TEST 7: Analyse Trading PnL vs Flux Externes")
    try:
        analysis = pm.get_trading_pnl_vs_external_flows()
        print(f"   - Capital Initial: ${analysis['initial_capital']:.2f}")
        print(f"   - Equity Actuelle: ${analysis['current_equity']:.2f}")
        print(f"   - Capital Ajusté: ${analysis['adjusted_initial_capital']:.2f}")
        print(f"   - PnL Total: ${analysis['total_pnl']:.2f}")
        print(f"   - PnL Trading Pur: ${analysis['trading_pnl']:.2f}")
        print(f"   - Impact Flux Externes: ${analysis['external_flow_impact']:.2f}")

    except Exception as e:
        print(f"   ❌ Erreur lors de l'analyse: {e}")
    print()

    # TEST 8: Historique des opérations
    print("📜 TEST 8: Historique des opérations")
    try:
        operations = pm.get_fund_operations_log(limit=3)
        print(f"   - Nombre d'opérations récentes: {len(operations)}")
        for i, op in enumerate(operations, 1):
            print(f"   {i}. {op['type']}: ${op['amount']:.2f} - {op['reason']}")
            print(f"      ID: {op['id'][:8]}... | Worker: {op['worker_id']}")

    except Exception as e:
        print(f"   ❌ Erreur lors de l'historique: {e}")
    print()

    # TEST 9: Comparaison State Vector
    print("🔢 TEST 9: Comparaison des State Vectors")
    try:
        final_state = pm.get_state_vector()
        print(f"   État Initial vs Final:")
        print(f"   - Cash: {initial_state[0]:.2f} -> {final_state[0]:.2f}")
        print(f"   - Total Value: {initial_state[1]:.2f} -> {final_state[1]:.2f}")
        print(f"   - Trading PnL %: {initial_state[2]:.4f} -> {final_state[2]:.4f}")
        print(f"   - External Flow %: {initial_state[3]:.4f} -> {final_state[3]:.4f}")
        print(f"   - Total Deposits %: {initial_state[4]:.4f} -> {final_state[4]:.4f}")
        print(
            f"   - Total Withdrawals %: {initial_state[5]:.4f} -> {final_state[5]:.4f}"
        )

        print(f"\n   🧠 Le modèle peut maintenant différencier:")
        print(f"      - Perte de trading: {final_state[2]:.4f}")
        print(f"      - Impact des retraits: {final_state[5]:.4f}")

    except Exception as e:
        print(f"   ❌ Erreur lors de la comparaison: {e}")
    print()

    print("=" * 60)
    print("🎉 TESTS TERMINÉS!")

    # État final
    print(f"\n💼 ÉTAT FINAL DU PORTEFEUILLE:")
    print(f"   - Cash: ${pm.cash:.2f}")
    print(f"   - Equity: ${pm.equity:.2f}")
    print(f"   - Portfolio Value: ${pm.portfolio_value:.2f}")

    summary = pm.get_fund_operations_summary()
    print(f"\n📊 RÉSUMÉ DES FLUX:")
    print(f"   - Dépôts: ${summary['total_deposits']:.2f}")
    print(f"   - Retraits: ${summary['total_withdrawals']:.2f}")
    print(f"   - Net: ${summary['net_external_flow']:.2f}")


if __name__ == "__main__":
    test_fund_management()
