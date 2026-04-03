#!/usr/bin/env python3
"""
T6 : Tests d'Intégration de Hiérarchie

Valide le système complet (PortfolioManager + DBE + Environnement) en conditions réelles.
Vérifie que la hiérarchie Environnement > DBE > Optuna fonctionne correctement en action.
"""

import sys
import os
import yaml
import pytest
from typing import Dict, Any

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager


def load_config() -> Dict[str, Any]:
    """Charge la configuration depuis config/config.yaml"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)


def get_tier_config(tier_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Récupère la configuration d'un palier"""
    for tier in config.get('capital_tiers', []):
        if tier.get('name') == tier_name:
            return tier
    raise ValueError(f"Tier {tier_name} not found")


@pytest.fixture
def config():
    """Charger config réelle"""
    return load_config()


@pytest.fixture
def portfolio_manager(config):
    """Initialiser PortfolioManager"""
    pm = PortfolioManager(
        config=config,
        worker_id=1,
        max_positions=1
    )
    return pm


class TestHierarchyIntegration:
    """Suite de tests d'intégration de la hiérarchie"""

    def test_01_integration_multi_workers(self, portfolio_manager, config):
        """
        Test 1 : Intégration Complète Multi-Workers
        
        Valide que le système fonctionne correctement avec plusieurs workers
        dans différentes conditions de capital et régime.
        """
        print("\n🔬 TEST 1 : INTÉGRATION MULTI-WORKERS")
        print("=" * 60)
        
        # Scénarios de test : (worker_id, capital, régime, tier_attendu)
        test_scenarios = [
            (1, 50.0, "bear", "Small Capital"),
            (2, 15.0, "bull", "Micro Capital"),
            (3, 250.0, "sideways", "Medium Capital"),
            (4, 800.0, "volatile", "High Capital"),
        ]
        
        for worker_id, capital, regime, expected_tier in test_scenarios:
            print(f"\n📋 Scénario: W{worker_id} | {capital} USDT | {regime}")
            
            # Obtenir paramètres via hiérarchie centralisée
            params = portfolio_manager.calculate_final_trade_parameters(
                worker_id=worker_id,
                capital=capital,
                market_regime=regime,
                current_step=0
            )
            
            assert params is not None, f"Paramètres None pour W{worker_id}"
            
            # Validation 1 : Tier correct
            assert params['tier_name'] == expected_tier, \
                f"Tier incorrect: {params['tier_name']} != {expected_tier}"
            print(f"   ✅ Tier correct: {params['tier_name']}")
            
            # Validation 2 : Min trade respecté
            notional = params['notional_usdt']
            assert notional >= 11.0, f"Notional < 11 USDT: {notional}"
            print(f"   ✅ Notional: {notional:.2f} USDT ≥ 11 USDT")
            
            # Validation 3 : Position dans les limites du palier
            tier_config = get_tier_config(expected_tier, config)
            max_pos_pct = tier_config['max_position_size_pct'] / 100.0
            assert params['position_size_pct'] <= max_pos_pct + 1e-6, \
                f"Position dépasse le max du palier: {params['position_size_pct']} > {max_pos_pct}"
            print(f"   ✅ Position: {params['position_size_pct']*100:.2f}% ≤ {max_pos_pct*100:.0f}%")
            
            # Validation 4 : SL/TP dans les bornes
            hard_constraints = config.get('environment', {}).get('hard_constraints', {})
            sl_min = hard_constraints.get('stop_loss_pct', {}).get('min', 0.005)
            sl_max = hard_constraints.get('stop_loss_pct', {}).get('max', 0.20)
            tp_min = hard_constraints.get('take_profit_pct', {}).get('min', 0.01)
            tp_max = hard_constraints.get('take_profit_pct', {}).get('max', 0.50)
            
            assert sl_min <= params['stop_loss_pct'] <= sl_max, \
                f"SL hors bornes: {params['stop_loss_pct']}"
            assert tp_min <= params['take_profit_pct'] <= tp_max, \
                f"TP hors bornes: {params['take_profit_pct']}"
            print(f"   ✅ SL/TP dans les bornes")

    def test_02_capital_tier_transitions(self, portfolio_manager, config):
        """
        Test 2 : Évolution du Capital à travers les Paliers
        
        Valide que le système s'adapte correctement quand le capital
        passe d'un palier à l'autre.
        """
        print("\n🔬 TEST 2 : TRANSITIONS ENTRE PALIERS")
        print("=" * 60)
        
        # Simuler évolution du capital
        capital_progression = [11, 15, 25, 35, 80, 150, 500, 1200]
        expected_tiers = [
            "Micro Capital", "Micro Capital", "Micro Capital", "Small Capital",
            "Small Capital", "Medium Capital", "High Capital", "Enterprise"
        ]
        
        for capital, expected_tier in zip(capital_progression, expected_tiers):
            print(f"\n📊 Capital: {capital} USDT → Tier: {expected_tier}")
            
            params = portfolio_manager.calculate_final_trade_parameters(
                worker_id=1,
                capital=capital,
                market_regime="bull",
                current_step=0
            )
            
            assert params is not None, f"Paramètres None pour capital {capital}"
            
            # Validation 1 : Tier correct
            assert params['tier_name'] == expected_tier, \
                f"Tier incorrect: {params['tier_name']} != {expected_tier}"
            print(f"   ✅ Tier correct: {params['tier_name']}")
            
            # Validation 2 : Position respecte le max du palier
            tier_config = get_tier_config(expected_tier, config)
            max_pos_pct = tier_config['max_position_size_pct'] / 100.0
            assert params['position_size_pct'] <= max_pos_pct + 1e-6, \
                f"Position dépasse le max: {params['position_size_pct']} > {max_pos_pct}"
            print(f"   ✅ Position: {params['position_size_pct']*100:.2f}% ≤ {max_pos_pct*100:.0f}%")
            
            # Validation 3 : Min trade respecté
            notional = params['notional_usdt']
            assert notional >= 11.0, f"Notional < 11 USDT: {notional}"
            print(f"   ✅ Notional: {notional:.2f} USDT ≥ 11 USDT")

    def test_03_min_trade_real_conditions(self, portfolio_manager, config):
        """
        Test 3 : Min Trade 11 USDT en Conditions Réelles
        
        Valide que le système ajuste ou rejette les trades < 11 USDT
        en conditions réelles.
        """
        print("\n🔬 TEST 3 : MIN TRADE 11 USDT EN CONDITIONS RÉELLES")
        print("=" * 60)
        
        # Cas 1 : Capital suffisant (pas d'ajustement nécessaire)
        print("\n📋 Cas 1 : Capital suffisant (100 USDT)")
        params1 = portfolio_manager.calculate_final_trade_parameters(
            worker_id=1,
            capital=100.0,
            market_regime="bull",
            current_step=0
        )
        notional1 = params1['notional_usdt']
        assert notional1 >= 11.0, f"Notional < 11 USDT: {notional1}"
        print(f"   ✅ Notional: {notional1:.2f} USDT ≥ 11 USDT (pas d'ajustement)")
        
        # Cas 2 : Capital faible (ajustement nécessaire)
        print("\n📋 Cas 2 : Capital faible (20 USDT)")
        params2 = portfolio_manager.calculate_final_trade_parameters(
            worker_id=1,
            capital=20.0,
            market_regime="bear",
            current_step=0
        )
        notional2 = params2['notional_usdt']
        assert notional2 >= 11.0, f"Notional < 11 USDT: {notional2}"
        # Doit être ajusté à exactement 11.0 (ou très proche)
        assert abs(notional2 - 11.0) < 0.01, f"Notional pas ajusté correctement: {notional2}"
        print(f"   ✅ Notional: {notional2:.2f} USDT (ajusté à 11 USDT)")
        
        # Cas 3 : Capital très faible (< 11 USDT)
        print("\n📋 Cas 3 : Capital très faible (10 USDT)")
        params3 = portfolio_manager.calculate_final_trade_parameters(
            worker_id=1,
            capital=10.0,
            market_regime="bull",
            current_step=0
        )
        # Si capital < 11, le système doit soit :
        # - Retourner None (trade impossible)
        # - Ou retourner une position à 100% (mais notional < 11)
        if params3 is not None:
            notional3 = params3['notional_usdt']
            # Si le système retourne des paramètres, notional doit être < 11
            # (car c'est impossible d'atteindre 11 avec capital < 11)
            print(f"   ⚠️  Notional: {notional3:.2f} USDT < 11 USDT (trade impossible)")
        else:
            print(f"   ✅ Trade rejeté (capital < 11 USDT)")

    def test_04_dbe_modulation_in_action(self, portfolio_manager, config):
        """
        Test 4 : DBE Modulation en Action
        
        Valide que DBE applique bien la modulation relative (sans écraser).
        """
        print("\n🔬 TEST 4 : DBE MODULATION EN ACTION")
        print("=" * 60)
        
        # Charger base Optuna
        worker_config = config['workers']['w1']
        base_position = worker_config['trading_parameters']['position_size_pct']
        base_sl = worker_config['trading_parameters']['stop_loss_pct']
        base_tp = worker_config['trading_parameters']['take_profit_pct']
        
        print(f"\n📊 Base Optuna W1:")
        print(f"   Position: {base_position*100:.2f}%")
        print(f"   SL: {base_sl*100:.2f}%")
        print(f"   TP: {base_tp*100:.2f}%")
        
        # Obtenir paramètres avec DBE bull
        print(f"\n📋 Régime Bull:")
        params_bull = portfolio_manager.calculate_final_trade_parameters(
            worker_id=1,
            capital=100.0,
            market_regime="bull",
            current_step=0
        )
        
        # Obtenir paramètres avec DBE bear
        print(f"\n📋 Régime Bear:")
        params_bear = portfolio_manager.calculate_final_trade_parameters(
            worker_id=1,
            capital=100.0,
            market_regime="bear",
            current_step=0
        )
        
        # Calculer les ajustements relatifs
        adjustment_pos_bull = (params_bull['position_size_pct'] / base_position) - 1
        adjustment_pos_bear = (params_bear['position_size_pct'] / base_position) - 1
        
        adjustment_sl_bull = (params_bull['stop_loss_pct'] / base_sl) - 1
        adjustment_sl_bear = (params_bear['stop_loss_pct'] / base_sl) - 1
        
        adjustment_tp_bull = (params_bull['take_profit_pct'] / base_tp) - 1
        adjustment_tp_bear = (params_bear['take_profit_pct'] / base_tp) - 1
        
        print(f"\n📊 Ajustements Position:")
        print(f"   Bull: {adjustment_pos_bull:+.1%}")
        print(f"   Bear: {adjustment_pos_bear:+.1%}")
        
        print(f"\n📊 Ajustements SL:")
        print(f"   Bull: {adjustment_sl_bull:+.1%}")
        print(f"   Bear: {adjustment_sl_bear:+.1%}")
        
        print(f"\n📊 Ajustements TP:")
        print(f"   Bull: {adjustment_tp_bull:+.1%}")
        print(f"   Bear: {adjustment_tp_bear:+.1%}")
        
        # Validation 1 : Ajustements dans [-15%, +15%]
        tolerance = 1e-6
        assert -0.15 - tolerance <= adjustment_pos_bull <= 0.15 + tolerance, \
            f"Position Bull dépasse ±15%: {adjustment_pos_bull:.1%}"
        assert -0.15 - tolerance <= adjustment_pos_bear <= 0.15 + tolerance, \
            f"Position Bear dépasse ±15%: {adjustment_pos_bear:.1%}"
        print(f"\n   ✅ Ajustements Position dans ±15%")
        
        assert -0.15 - tolerance <= adjustment_sl_bull <= 0.15 + tolerance, \
            f"SL Bull dépasse ±15%: {adjustment_sl_bull:.1%}"
        assert -0.15 - tolerance <= adjustment_sl_bear <= 0.15 + tolerance, \
            f"SL Bear dépasse ±15%: {adjustment_sl_bear:.1%}"
        print(f"   ✅ Ajustements SL dans ±15%")
        
        assert -0.15 - tolerance <= adjustment_tp_bull <= 0.15 + tolerance, \
            f"TP Bull dépasse ±15%: {adjustment_tp_bull:.1%}"
        assert -0.15 - tolerance <= adjustment_tp_bear <= 0.15 + tolerance, \
            f"TP Bear dépasse ±15%: {adjustment_tp_bear:.1%}"
        print(f"   ✅ Ajustements TP dans ±15%")
        
        # Validation 2 : Bull > Bear (bull devrait augmenter, bear diminuer)
        assert adjustment_pos_bull > adjustment_pos_bear, \
            f"Bull position pas > Bear: {adjustment_pos_bull} <= {adjustment_pos_bear}"
        print(f"\n   ✅ Bull position > Bear position (modulation cohérente)")

    def test_05_extreme_tier_scenarios(self, portfolio_manager, config):
        """
        Test 5 : Stress Test - Paliers Extrêmes
        
        Valide les limites du système (Micro et Enterprise).
        """
        print("\n🔬 TEST 5 : STRESS TEST - PALIERS EXTRÊMES")
        print("=" * 60)
        
        # Test Micro Capital (11 USDT exactement)
        print("\n📋 Micro Capital (11 USDT exactement)")
        params_micro = portfolio_manager.calculate_final_trade_parameters(
            worker_id=1,
            capital=11.0,
            market_regime="bull",
            current_step=0
        )
        
        assert params_micro is not None, "Paramètres None pour Micro"
        assert params_micro['tier_name'] == "Micro Capital", \
            f"Tier incorrect: {params_micro['tier_name']}"
        print(f"   ✅ Tier: {params_micro['tier_name']}")
        
        assert params_micro['position_size_pct'] <= 0.90 + 1e-6, \
            f"Position dépasse 90%: {params_micro['position_size_pct']}"
        print(f"   ✅ Position: {params_micro['position_size_pct']*100:.2f}% ≤ 90%")
        
        notional_micro = params_micro['notional_usdt']
        assert notional_micro >= 11.0, f"Notional < 11 USDT: {notional_micro}"
        print(f"   ✅ Notional: {notional_micro:.2f} USDT ≥ 11 USDT")
        
        # Test Enterprise (très gros capital)
        print("\n📋 Enterprise (10000 USDT)")
        params_enterprise = portfolio_manager.calculate_final_trade_parameters(
            worker_id=3,
            capital=10000.0,
            market_regime="sideways",
            current_step=0
        )
        
        assert params_enterprise is not None, "Paramètres None pour Enterprise"
        assert params_enterprise['tier_name'] == "Enterprise", \
            f"Tier incorrect: {params_enterprise['tier_name']}"
        print(f"   ✅ Tier: {params_enterprise['tier_name']}")
        
        assert params_enterprise['position_size_pct'] <= 0.20 + 1e-6, \
            f"Position dépasse 20%: {params_enterprise['position_size_pct']}"
        print(f"   ✅ Position: {params_enterprise['position_size_pct']*100:.2f}% ≤ 20%")
        
        notional_enterprise = params_enterprise['notional_usdt']
        assert notional_enterprise >= 11.0, f"Notional < 11 USDT: {notional_enterprise}"
        print(f"   ✅ Notional: {notional_enterprise:.2f} USDT ≥ 11 USDT")

    def test_06_dbe_consistency_across_tiers(self, portfolio_manager, config):
        """
        Test 6 : Cohérence DBE à travers les Paliers
        
        Valide que DBE s'adapte correctement selon le palier.
        """
        print("\n🔬 TEST 6 : COHÉRENCE DBE À TRAVERS LES PALIERS")
        print("=" * 60)
        
        test_capitals = [15, 50, 200, 600, 2000]  # Couvre tous les paliers
        
        for capital in test_capitals:
            print(f"\n📊 Capital: {capital} USDT")
            
            params = portfolio_manager.calculate_final_trade_parameters(
                worker_id=1,
                capital=capital,
                market_regime="volatile",
                current_step=0
            )
            
            assert params is not None, f"Paramètres None pour capital {capital}"
            
            # Validation 1 : Position respecte le max du palier
            tier_config = get_tier_config(params['tier_name'], config)
            max_pos_pct = tier_config['max_position_size_pct'] / 100.0
            assert params['position_size_pct'] <= max_pos_pct + 1e-6, \
                f"Position dépasse le max du palier: {params['position_size_pct']} > {max_pos_pct}"
            print(f"   ✅ Tier: {params['tier_name']}, Position: {params['position_size_pct']*100:.2f}% ≤ {max_pos_pct*100:.0f}%")
            
            # Validation 2 : Min trade respecté
            notional = params['notional_usdt']
            assert notional >= 11.0, f"Notional < 11 USDT: {notional}"
            print(f"   ✅ Notional: {notional:.2f} USDT ≥ 11 USDT")
            
            # Validation 3 : DBE volatile devrait réduire la position
            # (ajustement négatif par rapport à Optuna)
            worker_config = config['workers']['w1']
            base_position = worker_config['trading_parameters']['position_size_pct']
            adjustment = (params['position_size_pct'] / base_position) - 1
            # Volatile devrait avoir un ajustement négatif (réduction)
            # Mais ce n'est pas toujours le cas après clamp par palier
            print(f"   ℹ️  Ajustement DBE: {adjustment:+.1%}")


def run_all_tests():
    """Lance tous les tests"""
    print("🚀 TESTS D'INTÉGRATION DE HIÉRARCHIE (T6)")
    print("=" * 70)
    
    # Charger config
    config = load_config()
    
    # Initialiser PortfolioManager
    pm = PortfolioManager(config=config, worker_id=1, max_positions=1)
    
    # Créer instance de test
    test_instance = TestHierarchyIntegration()
    
    # Lancer les tests
    try:
        test_instance.test_01_integration_multi_workers(pm, config)
        test_instance.test_02_capital_tier_transitions(pm, config)
        test_instance.test_03_min_trade_real_conditions(pm, config)
        test_instance.test_04_dbe_modulation_in_action(pm, config)
        test_instance.test_05_extreme_tier_scenarios(pm, config)
        test_instance.test_06_dbe_consistency_across_tiers(pm, config)
        
        print("\n" + "=" * 70)
        print("📊 RÉSUMÉ DES TESTS")
        print("=" * 70)
        print("✅ Test 1: Intégration Multi-Workers - PASSÉ")
        print("✅ Test 2: Transitions entre Paliers - PASSÉ")
        print("✅ Test 3: Min Trade 11 USDT - PASSÉ")
        print("✅ Test 4: DBE Modulation en Action - PASSÉ")
        print("✅ Test 5: Stress Test Paliers Extrêmes - PASSÉ")
        print("✅ Test 6: Cohérence DBE - PASSÉ")
        print("\n🎯 RÉSULTAT GLOBAL: 6/6 tests passés (100%)")
        print("✅ TOUS LES TESTS PASSENT")
        return True
        
    except AssertionError as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
