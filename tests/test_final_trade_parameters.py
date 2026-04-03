#!/usr/bin/env python3
"""
Test de la fonction centralisée calculate_final_trade_parameters()

Valide que la hiérarchie Environnement > DBE > Optuna est appliquée correctement.
"""

import sys
import os
import yaml
import logging
from typing import Dict, Any

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

# Configuration du logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Charge la configuration depuis config/config.yaml"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def test_hierarchy_applied_correctly():
    """Vérifier que la hiérarchie est appliquée correctement."""
    print("\n🔬 TEST 1 : HIÉRARCHIE APPLIQUÉE CORRECTEMENT")
    print("=" * 60)
    
    config = load_config()
    pm = PortfolioManager(config, worker_id=1)
    
    # Scénarios de test
    scenarios = [
        {'worker': 1, 'capital': 50, 'regime': 'bear', 'expected_tier': 'Small Capital'},
        {'worker': 1, 'capital': 150, 'regime': 'bull', 'expected_tier': 'Medium Capital'},
        {'worker': 2, 'capital': 25, 'regime': 'volatile', 'expected_tier': 'Micro Capital'},
        {'worker': 4, 'capital': 500, 'regime': 'sideways', 'expected_tier': 'High Capital'},
    ]
    
    all_passed = True
    
    for scenario in scenarios:
        print(f"\n📋 Scénario: W{scenario['worker']} | {scenario['capital']} USDT | {scenario['regime']}")
        
        result = pm.calculate_final_trade_parameters(
            worker_id=scenario['worker'],
            capital=scenario['capital'],
            market_regime=scenario['regime'],
            current_step=0,
        )
        
        if result is None:
            print(f"   ⚠️  Trade rejeté (notional < min_trade)")
            continue
        
        tier_name = result.get('tier_name', 'Unknown')
        pos_pct = result.get('position_size_pct', 0)
        sl_pct = result.get('stop_loss_pct', 0)
        tp_pct = result.get('take_profit_pct', 0)
        notional = result.get('notional_usdt', 0)
        
        print(f"   Tier: {tier_name}")
        print(f"   Position: {pos_pct*100:.2f}%")
        print(f"   SL: {sl_pct*100:.2f}%, TP: {tp_pct*100:.2f}%")
        print(f"   Notional: {notional:.2f} USDT")
        
        # Vérifier que le tier est correct
        if tier_name == scenario['expected_tier']:
            print(f"   ✅ Tier correct")
        else:
            print(f"   ❌ Tier incorrect (attendu: {scenario['expected_tier']})")
            all_passed = False
        
        # Vérifier que notional >= 11 USDT
        if notional >= 11.0:
            print(f"   ✅ Notional >= 11 USDT")
        else:
            print(f"   ❌ Notional < 11 USDT")
            all_passed = False
    
    return all_passed

def test_min_trade_guarantee():
    """Vérifier que min_trade=11 est garanti."""
    print("\n💰 TEST 2 : GARANTIE MIN_TRADE = 11 USDT")
    print("=" * 60)
    
    config = load_config()
    pm = PortfolioManager(config, worker_id=1)
    
    # Tester avec capital très faible
    result = pm.calculate_final_trade_parameters(
        worker_id=1,
        capital=15,  # Très faible
        market_regime='bull',
        current_step=0,
    )
    
    if result is None:
        print("   ✅ Trade rejeté (impossible d'atteindre min_trade)")
        return True
    
    notional = result.get('notional_usdt', 0)
    print(f"   Notional: {notional:.2f} USDT")
    
    if notional >= 11.0:
        print(f"   ✅ Min trade garanti")
        return True
    else:
        print(f"   ❌ Min trade non garanti")
        return False

def test_tier_constraints():
    """Vérifier que les paliers sont respectés."""
    print("\n🏗️ TEST 3 : CONTRAINTES DES PALIERS")
    print("=" * 60)
    
    config = load_config()
    pm = PortfolioManager(config, worker_id=1)
    
    # Tester chaque palier
    capital_tiers = config.get('capital_tiers', [])
    all_passed = True
    
    for tier in capital_tiers:
        tier_name = tier.get('name', 'Unknown')
        min_cap = tier.get('min_capital', 0)
        max_cap = tier.get('max_capital', None)
        max_pos_pct = tier.get('max_position_size_pct', 90) / 100.0
        
        # Tester avec capital au milieu du palier
        if max_cap is None:
            test_capital = min_cap + 100
        else:
            test_capital = (min_cap + max_cap) / 2
        
        result = pm.calculate_final_trade_parameters(
            worker_id=1,
            capital=test_capital,
            market_regime='bull',
            current_step=0,
        )
        
        if result is None:
            print(f"   ⚠️  {tier_name}: Trade rejeté")
            continue
        
        pos_pct = result.get('position_size_pct', 0)
        
        print(f"   {tier_name}: Pos={pos_pct*100:.2f}% (max={max_pos_pct*100:.0f}%)")
        
        if pos_pct <= max_pos_pct + 1e-6:  # Tolérance numérique
            print(f"      ✅ Palier respecté")
        else:
            print(f"      ❌ Palier dépassé")
            all_passed = False
    
    return all_passed

def test_dbe_bounds():
    """Vérifier que DBE est limité à ±15%."""
    print("\n🔧 TEST 4 : LIMITES DBE (±15%)")
    print("=" * 60)
    
    config = load_config()
    pm = PortfolioManager(config, worker_id=1)
    
    # Lire les valeurs Optuna de base
    base_pos = config['workers']['w1']['trading_parameters']['position_size_pct']
    base_sl = config['workers']['w1']['trading_parameters']['stop_loss_pct']
    base_tp = config['workers']['w1']['trading_parameters']['take_profit_pct']
    
    print(f"   Optuna base: Pos={base_pos*100:.2f}%, SL={base_sl*100:.2f}%, TP={base_tp*100:.2f}%")
    
    # Tester chaque régime
    regimes = ['bull', 'bear', 'sideways', 'volatile']
    all_passed = True
    
    for regime in regimes:
        result = pm.calculate_final_trade_parameters(
            worker_id=1,
            capital=150,
            market_regime=regime,
            current_step=0,
        )
        
        if result is None:
            print(f"   ⚠️  {regime}: Trade rejeté")
            continue
        
        pos_pct = result.get('position_size_pct', 0)
        sl_pct = result.get('stop_loss_pct', 0)
        tp_pct = result.get('take_profit_pct', 0)
        
        # Calculer les ajustements
        pos_adj = (pos_pct - base_pos) / base_pos if base_pos > 0 else 0
        sl_adj = (sl_pct - base_sl) / base_sl if base_sl > 0 else 0
        tp_adj = (tp_pct - base_tp) / base_tp if base_tp > 0 else 0
        
        print(f"   {regime}: Pos={pos_adj:+.1%}, SL={sl_adj:+.1%}, TP={tp_adj:+.1%}")
        
        # Vérifier que les ajustements sont dans ±15% (avec tolérance numérique)
        tolerance = 1e-6
        if abs(pos_adj) <= 0.15 + tolerance and abs(sl_adj) <= 0.15 + tolerance and abs(tp_adj) <= 0.15 + tolerance:
            print(f"      ✅ DBE dans les limites ±15%")
        else:
            print(f"      ❌ DBE dépasse ±15%")
            all_passed = False
    
    return all_passed

def main():
    """Lance tous les tests"""
    print("🚀 TESTS DE LA FONCTION CENTRALISÉE calculate_final_trade_parameters()")
    print("=" * 70)
    
    tests = [
        test_hierarchy_applied_correctly,
        test_min_trade_guarantee,
        test_tier_constraints,
        test_dbe_bounds,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Erreur dans {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅" if result else "❌"
        print(f"{status} Test {i+1}: {test.__name__}")
    
    print(f"\n🎯 RÉSULTAT GLOBAL: {passed}/{total} tests passés ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("✅ TOUS LES TESTS PASSENT")
        return True
    else:
        print("❌ CERTAINS TESTS ÉCHOUENT")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
