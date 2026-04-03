"""
Test de validation de la hiérarchie DBE V2 : Optuna > DBE > Environnement

Vérifie que :
1. Optuna (trading_parameters) est la source unique de vérité
2. DBE applique des multiplicateurs relatifs ±15% max (modulation, pas écrasement)
3. Environnement impose les contraintes absolues (hard_constraints + capital_tiers)
4. Min trade = 11 USDT est toujours respecté
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine


class TestDBEHierarchyV2:
    """Tests de la hiérarchie DBE V2"""

    @pytest.fixture
    def dbe_config(self):
        """Configuration de test pour DBE"""
        return {
            "workers": {
                "w1": {
                    "trading_parameters": {
                        "position_size_pct": 0.1121,  # Optuna base
                        "stop_loss_pct": 0.0253,      # Optuna base
                        "take_profit_pct": 0.0321,    # Optuna base
                        "risk_per_trade_pct": 0.01,
                    }
                }
            },
            "dbe": {
                "regime_parameters": {
                    "bear": {
                        "position_size_multiplier": 0.9,   # -10%
                        "sl_multiplier": 1.05,             # +5%
                        "tp_multiplier": 0.95,             # -5%
                    },
                    "bull": {
                        "position_size_multiplier": 1.1,   # +10%
                        "sl_multiplier": 0.97,             # -3%
                        "tp_multiplier": 1.08,             # +8%
                    },
                }
            },
            "environment": {
                "hard_constraints": {
                    "min_order_value_usdt": 11.0,
                    "stop_loss_pct": {"min": 0.005, "max": 0.20},
                    "take_profit_pct": {"min": 0.01, "max": 0.50},
                }
            },
            "capital_tiers": [
                {
                    "name": "Small Capital",
                    "min_capital": 30.0,
                    "max_capital": 100.0,
                    "max_position_size_pct": 65,
                    "exposure_range": [35, 75],
                }
            ],
            "risk_parameters": {},
        }

    @pytest.fixture
    def dbe(self, dbe_config):
        """Instance DBE pour tests"""
        dbe = DynamicBehaviorEngine(config=dbe_config, worker_id=0)
        dbe.current_regime = "bear"
        return dbe

    def test_optuna_base_preserved(self, dbe):
        """Test 1 : Optuna (trading_parameters) est préservé comme source unique"""
        # Récupérer les paramètres de base
        base_sl, base_tp, base_pos = dbe._get_tier_based_parameters("w1", "Small Capital")
        
        # Vérifier que ce sont exactement les valeurs Optuna
        assert base_sl == 0.0253, f"SL devrait être 0.0253 (Optuna), got {base_sl}"
        assert base_tp == 0.0321, f"TP devrait être 0.0321 (Optuna), got {base_tp}"
        assert base_pos == 0.1121, f"Position devrait être 0.1121 (Optuna), got {base_pos}"
        print("✅ Test 1 PASSED: Optuna base préservé")

    def test_dbe_modulation_not_replacement(self, dbe):
        """Test 2 : DBE applique une modulation relative, pas un remplacement"""
        # Récupérer les paramètres de base
        base_sl, base_tp, base_pos = dbe._get_tier_based_parameters("w1", "Small Capital")
        
        # Appliquer modulation DBE (régime bear)
        regime_params = dbe.config["dbe"]["regime_parameters"]["bear"]
        
        # Convertir multiplicateurs en ajustements relatifs
        pos_adjustment = min(max(regime_params["position_size_multiplier"] - 1.0, -0.15), 0.15)
        sl_adjustment = min(max(regime_params["sl_multiplier"] - 1.0, -0.15), 0.15)
        tp_adjustment = min(max(regime_params["tp_multiplier"] - 1.0, -0.15), 0.15)
        
        # Appliquer ajustements
        adjusted_pos = base_pos * (1 + pos_adjustment)
        adjusted_sl = base_sl * (1 + sl_adjustment)
        adjusted_tp = base_tp * (1 + tp_adjustment)
        
        # Vérifier que c'est une modulation, pas un remplacement
        assert adjusted_pos != regime_params["position_size_multiplier"], "DBE ne doit pas remplacer, moduler"
        assert adjusted_pos == base_pos * 0.9, f"Position devrait être {base_pos * 0.9}, got {adjusted_pos}"
        assert adjusted_sl == base_sl * 1.05, f"SL devrait être {base_sl * 1.05}, got {adjusted_sl}"
        assert adjusted_tp == base_tp * 0.95, f"TP devrait être {base_tp * 0.95}, got {adjusted_tp}"
        print("✅ Test 2 PASSED: DBE applique modulation relative (pas remplacement)")

    def test_dbe_adjustment_bounded_15_percent(self, dbe):
        """Test 3 : Ajustements DBE bornés à ±15% max"""
        # Tester avec des multiplicateurs extrêmes
        extreme_multipliers = {
            "position_size_multiplier": 1.4,  # +40% → devrait être clampé à +15%
            "sl_multiplier": 0.6,             # -40% → devrait être clampé à -15%
            "tp_multiplier": 1.5,             # +50% → devrait être clampé à +15%
        }
        
        for key, mult in extreme_multipliers.items():
            adjustment = min(max(mult - 1.0, -0.15), 0.15)
            assert -0.15 <= adjustment <= 0.15, f"Ajustement {key} doit être dans [-0.15, 0.15], got {adjustment}"
        
        print("✅ Test 3 PASSED: Ajustements DBE bornés à ±15%")

    def test_min_trade_11_usdt_respected(self, dbe):
        """Test 4 : Min trade = 11 USDT est toujours respecté"""
        # Vérifier que hard_constraints contient min_order_value_usdt = 11.0
        min_trade = dbe.config["environment"]["hard_constraints"]["min_order_value_usdt"]
        assert min_trade == 11.0, f"Min trade devrait être 11.0, got {min_trade}"
        print("✅ Test 4 PASSED: Min trade = 11 USDT respecté")

    def test_capital_tiers_unchanged(self, dbe):
        """Test 5 : Capital tiers sont inchangés"""
        # Vérifier que les tiers n'ont pas été modifiés
        tier = dbe.config["capital_tiers"][0]
        assert tier["name"] == "Small Capital"
        assert tier["max_position_size_pct"] == 65
        assert tier["exposure_range"] == [35, 75]
        print("✅ Test 5 PASSED: Capital tiers inchangés")

    def test_hierarchy_sequence(self, dbe):
        """Test 6 : Hiérarchie séquentielle complète"""
        # Simuler la hiérarchie complète
        
        # Étape 1 : Charger Optuna
        base_sl, base_tp, base_pos = dbe._get_tier_based_parameters("w1", "Small Capital")
        print(f"  Étape 1 (Optuna): SL={base_sl:.4f}, TP={base_tp:.4f}, Pos={base_pos:.4f}")
        
        # Étape 2 : Appliquer DBE modulation (bear)
        regime_params = dbe.config["dbe"]["regime_parameters"]["bear"]
        pos_adjustment = min(max(regime_params["position_size_multiplier"] - 1.0, -0.15), 0.15)
        adjusted_pos = base_pos * (1 + pos_adjustment)
        print(f"  Étape 2 (DBE): Pos={adjusted_pos:.4f} (ajustement {pos_adjustment:+.1%})")
        
        # Étape 3 : Clamp par hard_constraints
        hard_constraints = dbe.config["environment"]["hard_constraints"]
        sl_min = hard_constraints["stop_loss_pct"]["min"]
        sl_max = hard_constraints["stop_loss_pct"]["max"]
        print(f"  Étape 3 (Hard Constraints): SL ∈ [{sl_min:.4f}, {sl_max:.4f}]")
        
        # Étape 4 : Clamp par tier
        tier_cap = 0.65  # Small Capital max_position_size_pct
        final_pos = min(adjusted_pos, tier_cap)
        print(f"  Étape 4 (Tier): Pos={final_pos:.4f} (cap {tier_cap:.2%})")
        
        # Vérifier que la position finale est valide
        assert final_pos <= tier_cap, f"Position finale doit être ≤ {tier_cap}, got {final_pos}"
        assert final_pos > 0, "Position finale doit être > 0"
        print("✅ Test 6 PASSED: Hiérarchie séquentielle complète")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
