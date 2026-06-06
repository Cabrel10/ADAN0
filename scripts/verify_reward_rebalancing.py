#!/usr/bin/env python3
"""
Vérification que les corrections de rééquilibrage ont été appliquées correctement.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def verify_config_changes():
    """Vérifie les changements dans config.yaml"""
    logger.info("=" * 100)
    logger.info("VÉRIFICATION 1: config.yaml")
    logger.info("=" * 100)
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    freq_weights = config.get('reward_shaping', {}).get('frequency_weights', {})
    
    logger.info(f"\nfrequency_weights:")
    logger.info(f"  in_range: {freq_weights.get('in_range')}")
    logger.info(f"  out_of_range: {freq_weights.get('out_of_range')}")
    
    if freq_weights.get('in_range') == 0.05 and freq_weights.get('out_of_range') == 0.02:
        logger.info("✓ CORRECT: Fréquence réduite de 40x (2.0 → 0.05, 0.2 → 0.02)")
        return True
    else:
        logger.error("✗ INCORRECT: Fréquence n'a pas été réduite correctement")
        return False

def verify_code_changes():
    """Vérifie les changements dans le code"""
    logger.info("\n" + "=" * 100)
    logger.info("VÉRIFICATION 2: src/adan_trading_bot/environment/multi_asset_chunked_env.py")
    logger.info("=" * 100)
    
    with open('src/adan_trading_bot/environment/multi_asset_chunked_env.py', 'r') as f:
        code = f.read()
    
    # Vérifier le changement de drawdown_penalty
    if "drawdown_penalty = (abs(dd) - 0.02) ** 2 * 150.0" in code:
        logger.info("\n✓ CORRECT: Drawdown penalty multiplier augmenté de 50.0 à 150.0")
        drawdown_ok = True
    else:
        logger.error("\n✗ INCORRECT: Drawdown penalty multiplier n'a pas été changé")
        drawdown_ok = False
    
    # Vérifier que l'ancien code n'existe plus
    if "drawdown_penalty = (abs(dd) - 0.02) ** 2 * 50.0" in code:
        logger.error("✗ INCORRECT: L'ancien code (50.0) existe toujours")
        drawdown_ok = False
    
    return drawdown_ok

def simulate_new_scenarios():
    """Simule les nouveaux scénarios avec les poids rééquilibrés"""
    logger.info("\n" + "=" * 100)
    logger.info("SIMULATION: Nouveaux scénarios avec poids rééquilibrés")
    logger.info("=" * 100)
    
    import numpy as np
    
    def symlog(x):
        return np.sign(x) * np.log1p(np.abs(x))
    
    scenarios = {
        "Trade gagnant (TP)": {
            "pnl_net_scaled": 0.12,
            "trade_cost": -0.001,
            "drawdown_penalty": 0.0,
            "inaction": 0.0,
            "invalid_penalty": 0.0,
            "capacity_reward": 0.0,
            "frequency_reward": 0.005,  # Réduit de 0.2 à 0.005
        },
        "Trade perdant (SL)": {
            "pnl_net_scaled": -0.15,
            "trade_cost": -0.001,
            "drawdown_penalty": 0.0,
            "inaction": 0.0,
            "invalid_penalty": 0.0,
            "capacity_reward": 0.0,
            "frequency_reward": 0.005,  # Réduit de 0.2 à 0.005
        },
        "HOLD (pas de trade)": {
            "pnl_net_scaled": 0.0,
            "trade_cost": 0.0,
            "drawdown_penalty": 0.0,
            "inaction": -0.001,
            "invalid_penalty": 0.0,
            "capacity_reward": 0.0,
            "frequency_reward": 0.005,  # Réduit de 0.214 à 0.005
        },
        "Drawdown 5%": {
            "pnl_net_scaled": 0.0,
            "trade_cost": 0.0,
            "drawdown_penalty": -(0.05 - 0.02) ** 2 * 150.0,  # Triplé de 50.0 à 150.0
            "inaction": 0.0,
            "invalid_penalty": 0.0,
            "capacity_reward": 0.0,
            "frequency_reward": 0.0,
        },
    }
    
    logger.info("\n[RÉSULTATS AVEC POIDS RÉÉQUILIBRÉS]")
    logger.info(f"{'Scénario':<30} {'Raw':>12} {'Final':>12} {'Hiérarchie':>15}")
    logger.info("-" * 75)
    
    results = []
    for scenario_name, components in scenarios.items():
        raw = (
            components["pnl_net_scaled"]
            - components["trade_cost"]
            - components["drawdown_penalty"]
            + components["inaction"]
            + components["invalid_penalty"]
            + components["capacity_reward"]
            + components["frequency_reward"]
        )
        final = symlog(raw)
        
        if final > 0.15:
            hierarchy = "Excellent"
        elif final > 0.05:
            hierarchy = "Bon"
        elif final > -0.05:
            hierarchy = "Neutre"
        elif final > -0.15:
            hierarchy = "Mauvais"
        else:
            hierarchy = "Très mauvais"
        
        results.append((scenario_name, raw, final, hierarchy))
        logger.info(f"{scenario_name:<30} {raw:>+12.6f} {final:>+12.6f} {hierarchy:>15}")
    
    # Vérifier la hiérarchie
    logger.info("\n[VÉRIFICATION DE LA HIÉRARCHIE]")
    
    winning = [r for r in results if r[0] == "Trade gagnant (TP)"][0]
    losing = [r for r in results if r[0] == "Trade perdant (SL)"][0]
    hold = [r for r in results if r[0] == "HOLD (pas de trade)"][0]
    
    logger.info(f"  Trade gagnant:   {winning[2]:+.6f}")
    logger.info(f"  HOLD:            {hold[2]:+.6f}")
    logger.info(f"  Trade perdant:   {losing[2]:+.6f}")
    
    if winning[2] > hold[2] > losing[2]:
        logger.info("✓ CORRECT: Hiérarchie respectée (gagnant > HOLD > perdant)")
        return True
    else:
        logger.error("✗ INCORRECT: Hiérarchie non respectée")
        return False

def main():
    logger.info("\n" + "=" * 100)
    logger.info("VÉRIFICATION DU RÉÉQUILIBRAGE DE LA RÉCOMPENSE")
    logger.info("=" * 100)
    
    config_ok = verify_config_changes()
    code_ok = verify_code_changes()
    hierarchy_ok = simulate_new_scenarios()
    
    logger.info("\n" + "=" * 100)
    logger.info("RÉSUMÉ")
    logger.info("=" * 100)
    
    if config_ok and code_ok and hierarchy_ok:
        logger.info("\n✓ TOUS LES CHANGEMENTS APPLIQUÉS CORRECTEMENT")
        logger.info("\nProchaines étapes:")
        logger.info("  1. Lancer un test rapide (500 steps) pour vérifier")
        logger.info("  2. Vérifier que [REWARD] Realized PnL affiche des valeurs non-nulles")
        logger.info("  3. Vérifier que mean_reward augmente au fil du temps")
        logger.info("  4. Lancer l'entraînement complet (500k+ steps)")
        return 0
    else:
        logger.error("\n✗ CERTAINS CHANGEMENTS N'ONT PAS ÉTÉ APPLIQUÉS")
        return 1

if __name__ == '__main__':
    sys.exit(main())
