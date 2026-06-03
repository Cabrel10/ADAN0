#!/usr/bin/env python3
"""
Radar statistique pour le rééquilibrage de la fonction de récompense.
Analyse Master 2 niveau: Optimisation multi-objectif avec contraintes.

Objectif: Trouver l'équilibre optimal entre:
1. Maximiser PnL (objectif financier)
2. Minimiser Drawdown (gestion du risque)
3. Maximiser Win Rate (qualité des trades)
4. Respecter les contraintes (fréquence, capacité)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class RewardRadar:
    """Analyse multi-dimensionnelle de la fonction de récompense."""
    
    def __init__(self):
        self.metrics = {}
        self.constraints = {}
        self.pareto_frontier = []
        
    def define_objectives(self):
        """Définit les objectifs d'optimisation."""
        logger.info("=" * 120)
        logger.info("RADAR STATISTIQUE: ANALYSE MULTI-OBJECTIF")
        logger.info("=" * 120)
        
        objectives = {
            "maximize_pnl": {
                "description": "Maximiser le PnL net (objectif financier)",
                "weight": 0.50,  # 50% du signal
                "current_contribution": 0.12,  # Trade gagnant
                "target_contribution": 0.25,  # Doit doubler
            },
            "minimize_drawdown": {
                "description": "Minimiser le drawdown (gestion du risque)",
                "weight": 0.25,  # 25% du signal
                "current_contribution": -0.045,  # Pénalité faible
                "target_contribution": -0.15,  # Doit tripler
            },
            "maximize_win_rate": {
                "description": "Maximiser le win rate (qualité des trades)",
                "weight": 0.15,  # 15% du signal
                "current_contribution": 0.0,  # Pas de bonus actuellement
                "target_contribution": 0.10,  # Nouveau bonus
            },
            "respect_constraints": {
                "description": "Respecter les contraintes (fréquence, capacité)",
                "weight": 0.10,  # 10% du signal
                "current_contribution": 0.2,  # Bonus de fréquence
                "target_contribution": 0.05,  # Réduit drastiquement
            },
        }
        
        logger.info("\n[OBJECTIFS D'OPTIMISATION]")
        for obj_name, obj_info in objectives.items():
            logger.info(f"\n{obj_name}:")
            logger.info(f"  Description: {obj_info['description']}")
            logger.info(f"  Poids: {obj_info['weight']:.0%}")
            logger.info(f"  Contribution actuelle: {obj_info['current_contribution']:+.6f}")
            logger.info(f"  Contribution cible: {obj_info['target_contribution']:+.6f}")
        
        self.objectives = objectives
        return objectives
    
    def define_constraints(self):
        """Définit les contraintes d'optimisation."""
        logger.info("\n" + "=" * 120)
        logger.info("CONTRAINTES D'OPTIMISATION")
        logger.info("=" * 120)
        
        constraints = {
            "max_concurrent_positions": {
                "description": "Max positions ouvertes simultanément",
                "type": "hard",  # Contrainte dure (bloque l'action)
                "current_enforcement": "risk_gate",
                "reward_impact": "Aucun (contrainte dure)",
            },
            "frequency_min_max": {
                "description": "Min/max positions par timeframe",
                "type": "soft",  # Contrainte souple (bonus/pénalité)
                "current_enforcement": "frequency_reward",
                "reward_impact": "Bonus 0.2 (TROP ÉLEVÉ)",
            },
            "drawdown_limit": {
                "description": "Limite de drawdown (40% pour Micro)",
                "type": "hard",  # Contrainte dure (kill-switch)
                "current_enforcement": "drawdown_penalty + kill_switch",
                "reward_impact": "Pénalité -0.045 (TROP FAIBLE)",
            },
            "min_order_value": {
                "description": "Valeur minimale d'une commande ($11)",
                "type": "hard",  # Contrainte dure (bloque l'action)
                "current_enforcement": "size_gate",
                "reward_impact": "Aucun (contrainte dure)",
            },
        }
        
        logger.info("\n[CONTRAINTES]")
        for const_name, const_info in constraints.items():
            logger.info(f"\n{const_name}:")
            logger.info(f"  Description: {const_info['description']}")
            logger.info(f"  Type: {const_info['type']}")
            logger.info(f"  Enforcement: {const_info['current_enforcement']}")
            logger.info(f"  Impact sur récompense: {const_info['reward_impact']}")
        
        self.constraints = constraints
        return constraints
    
    def analyze_current_state(self):
        """Analyse l'état actuel de la fonction de récompense."""
        logger.info("\n" + "=" * 120)
        logger.info("ANALYSE DE L'ÉTAT ACTUEL")
        logger.info("=" * 120)
        
        # Décomposition du signal de récompense
        logger.info("\n[DÉCOMPOSITION DU SIGNAL DE RÉCOMPENSE]")
        logger.info("\nScénario: Trade gagnant (TP)")
        logger.info("  PnL net:           +0.120  (37.4% du signal)")
        logger.info("  Frequency reward:  +0.200  (62.3% du signal)")
        logger.info("  Trade cost:        -0.001  (0.3% du signal)")
        logger.info("  ─────────────────────────────────")
        logger.info("  Total:             +0.321  (100%)")
        
        logger.info("\n[PROBLÈME IDENTIFIÉ]")
        logger.info("  La fréquence (62.3%) domine le PnL (37.4%)")
        logger.info("  → L'agent optimise la fréquence, pas le profit")
        logger.info("  → Reward Hacking: L'agent trade souvent mais perd de l'argent")
        
        logger.info("\n[HIÉRARCHIE ACTUELLE]")
        logger.info("  1. Trade gagnant:     +0.278 ✓ (meilleur)")
        logger.info("  2. HOLD:              +0.193 ✗ (trop proche du trade gagnant)")
        logger.info("  3. Trade aléatoire:   +0.183 ✗ (presque aussi bon que HOLD)")
        logger.info("  4. Trade perdant:     +0.050 ✗ (presque aussi bon que HOLD!)")
        logger.info("  5. Rejet fee_gate:    -0.005 ✓ (pénalisé correctement)")
        
        logger.info("\n[HIÉRARCHIE SOUHAITÉE]")
        logger.info("  1. Trade gagnant:     +0.25  (meilleur)")
        logger.info("  2. HOLD:              +0.05  (neutre)")
        logger.info("  3. Trade aléatoire:   -0.05  (pénalisé)")
        logger.info("  4. Trade perdant:     -0.10  (fortement pénalisé)")
        logger.info("  5. Rejet fee_gate:    -0.01  (pénalisé)")
    
    def propose_optimal_weights(self):
        """Propose les poids optimaux pour chaque composante."""
        logger.info("\n" + "=" * 120)
        logger.info("PROPOSITION DE POIDS OPTIMAUX")
        logger.info("=" * 120)
        
        proposals = {
            "frequency_weights": {
                "current": {"in_range": 2.0, "out_of_range": 0.2},
                "proposed": {"in_range": 0.05, "out_of_range": 0.02},
                "rationale": "Réduire de 40x: fréquence = contrainte dure, pas bonus",
                "impact": "Frequency reward passe de 0.2 à 0.005 par trade",
            },
            "drawdown_penalty_multiplier": {
                "current": 50.0,
                "proposed": 150.0,
                "rationale": "Tripler la pénalité: drawdown > 2% doit être très dissuasif",
                "impact": "Drawdown penalty passe de -0.045 à -0.135 pour 5% DD",
            },
            "time_decay": {
                "current": -0.001,
                "proposed": -0.0005,
                "rationale": "Réduire de moitié: inaction moins pénalisée",
                "impact": "Inaction penalty passe de -0.001 à -0.0005 par step",
            },
            "invalid_trade_penalty_weight": {
                "current": 0.005,
                "proposed": 0.01,
                "rationale": "Doubler: rejets de trades doivent être plus pénalisés",
                "impact": "Invalid penalty passe de -0.005 à -0.01 par rejet",
            },
            "win_rate_bonus": {
                "current": 0.0,
                "proposed": 0.05,
                "rationale": "Nouveau bonus: récompenser la qualité des trades",
                "impact": "Bonus +0.05 si win_rate > 55%, -0.05 si < 45%",
            },
            "capacity_reward": {
                "current": "Inclus dans raw",
                "proposed": "Supprimé (contrainte dure)",
                "rationale": "Max concurrent positions est une limite, pas un objectif",
                "impact": "Capacity reward = 0 (risk_gate bloque l'action)",
            },
        }
        
        logger.info("\n[POIDS PROPOSÉS]")
        for component, proposal in proposals.items():
            logger.info(f"\n{component}:")
            logger.info(f"  Actuel:    {proposal['current']}")
            logger.info(f"  Proposé:   {proposal['proposed']}")
            logger.info(f"  Rationale: {proposal['rationale']}")
            logger.info(f"  Impact:    {proposal['impact']}")
        
        return proposals
    
    def simulate_rebalanced_scenarios(self):
        """Simule les scénarios avec les poids proposés."""
        logger.info("\n" + "=" * 120)
        logger.info("SIMULATION DES SCÉNARIOS RÉÉQUILIBRÉS")
        logger.info("=" * 120)
        
        scenarios = {
            "Trade gagnant (TP)": {
                "pnl_net_scaled": 0.12,
                "trade_cost": -0.001,
                "drawdown_penalty": 0.0,
                "inaction": 0.0,
                "invalid_penalty": 0.0,
                "capacity_reward": 0.0,
                "frequency_reward_old": 0.2,
                "frequency_reward_new": 0.005,
                "win_rate_bonus": 0.05,  # Nouveau
            },
            "Trade perdant (SL)": {
                "pnl_net_scaled": -0.15,
                "trade_cost": -0.001,
                "drawdown_penalty": 0.0,
                "inaction": 0.0,
                "invalid_penalty": 0.0,
                "capacity_reward": 0.0,
                "frequency_reward_old": 0.2,
                "frequency_reward_new": 0.005,
                "win_rate_bonus": -0.05,  # Nouveau
            },
            "HOLD (pas de trade)": {
                "pnl_net_scaled": 0.0,
                "trade_cost": 0.0,
                "drawdown_penalty": 0.0,
                "inaction": -0.0005,  # Réduit
                "invalid_penalty": 0.0,
                "capacity_reward": 0.0,
                "frequency_reward_old": 0.214,
                "frequency_reward_new": 0.005,
                "win_rate_bonus": 0.0,
            },
            "Drawdown 5%": {
                "pnl_net_scaled": 0.0,
                "trade_cost": 0.0,
                "drawdown_penalty": -(0.05 - 0.02) ** 2 * 150.0,  # Triplé
                "inaction": 0.0,
                "invalid_penalty": 0.0,
                "capacity_reward": 0.0,
                "frequency_reward_old": 0.0,
                "frequency_reward_new": 0.0,
                "win_rate_bonus": 0.0,
            },
        }
        
        results = []
        
        logger.info("\n[RÉSULTATS]")
        logger.info(f"{'Scénario':<30} {'Avant':>12} {'Après':>12} {'Δ':>12} {'Hiérarchie':>15}")
        logger.info("-" * 85)
        
        for scenario_name, components in scenarios.items():
            # Avant
            raw_before = (
                components["pnl_net_scaled"]
                - components["trade_cost"]
                - components["drawdown_penalty"]
                + components["inaction"]
                + components["invalid_penalty"]
                + components["capacity_reward"]
                + components["frequency_reward_old"]
            )
            reward_before = np.sign(raw_before) * np.log1p(np.abs(raw_before))
            
            # Après
            raw_after = (
                components["pnl_net_scaled"]
                - components["trade_cost"]
                - components["drawdown_penalty"]
                + components["inaction"]
                + components["invalid_penalty"]
                + components["capacity_reward"]
                + components["frequency_reward_new"]
                + components["win_rate_bonus"]
            )
            reward_after = np.sign(raw_after) * np.log1p(np.abs(raw_after))
            
            delta = reward_after - reward_before
            
            results.append({
                "scenario": scenario_name,
                "before": reward_before,
                "after": reward_after,
                "delta": delta,
            })
            
            # Déterminer la hiérarchie
            if reward_after > 0.2:
                hierarchy = "Excellent"
            elif reward_after > 0.1:
                hierarchy = "Bon"
            elif reward_after > 0.0:
                hierarchy = "Neutre"
            elif reward_after > -0.1:
                hierarchy = "Mauvais"
            else:
                hierarchy = "Très mauvais"
            
            logger.info(f"{scenario_name:<30} {reward_before:>+12.6f} {reward_after:>+12.6f} {delta:>+12.6f} {hierarchy:>15}")
        
        return results
    
    def generate_recommendations(self):
        """Génère les recommandations finales."""
        logger.info("\n" + "=" * 120)
        logger.info("RECOMMANDATIONS FINALES")
        logger.info("=" * 120)
        
        logger.info("\n[PRIORITÉ 1: CORRIGER LE LOG (Cosmétique)]")
        logger.info("  Fichier: src/adan_trading_bot/environment/multi_asset_chunked_env.py")
        logger.info("  Ligne: ~3171")
        logger.info("  Problème: logger.info(f'[REWARD] Realized PnL for step: ${realized_pnl:.2f}')")
        logger.info("  Solution: Vérifier que realized_pnl est la bonne variable (pas réinitialisée)")
        logger.info("  Impact: Aucun sur l'apprentissage, mais améliore la visibilité")
        
        logger.info("\n[PRIORITÉ 2: RÉÉQUILIBRER LA RÉCOMPENSE (Critique)]")
        logger.info("  Fichier: config/config.yaml")
        logger.info("  Section: reward_shaping.frequency_weights")
        logger.info("  Changement:")
        logger.info("    in_range: 2.0 → 0.05")
        logger.info("    out_of_range: 0.2 → 0.02")
        logger.info("  Rationale: Fréquence doit être une contrainte dure, pas un bonus")
        logger.info("  Impact: Frequency reward passe de 0.2 à 0.005 par trade")
        
        logger.info("\n[PRIORITÉ 3: AUGMENTER LA PÉNALITÉ DE DRAWDOWN (Important)]")
        logger.info("  Fichier: src/adan_trading_bot/environment/multi_asset_chunked_env.py")
        logger.info("  Ligne: ~5950 (dans _calculate_reward)")
        logger.info("  Changement:")
        logger.info("    drawdown_penalty = (abs(dd) - 0.02) ** 2 * 50.0")
        logger.info("    → drawdown_penalty = (abs(dd) - 0.02) ** 2 * 150.0")
        logger.info("  Rationale: Drawdown > 2% doit être très dissuasif")
        logger.info("  Impact: Drawdown penalty passe de -0.045 à -0.135 pour 5% DD")
        
        logger.info("\n[PRIORITÉ 4: AJOUTER UN BONUS DE WIN_RATE (Optionnel)]")
        logger.info("  Fichier: src/adan_trading_bot/environment/multi_asset_chunked_env.py")
        logger.info("  Ligne: ~5950 (dans _calculate_reward)")
        logger.info("  Changement:")
        logger.info("    if win_rate > 0.55:")
        logger.info("        win_rate_bonus = 0.05")
        logger.info("    elif win_rate < 0.45:")
        logger.info("        win_rate_bonus = -0.05")
        logger.info("  Rationale: Récompenser la qualité des trades, pas la quantité")
        logger.info("  Impact: Encourage l'agent à améliorer sa sélectivité")
        
        logger.info("\n[PRIORITÉ 5: SUPPRIMER CAPACITY_REWARD (Optionnel)]")
        logger.info("  Fichier: src/adan_trading_bot/environment/multi_asset_chunked_env.py")
        logger.info("  Ligne: ~5891 (dans _calculate_reward)")
        logger.info("  Changement:")
        logger.info("    raw = (...  + capacity_reward + ...)")
        logger.info("    → raw = (...  + ...)")
        logger.info("  Rationale: Max concurrent positions est une limite, pas un objectif")
        logger.info("  Impact: Capacity reward = 0 (risk_gate bloque l'action)")
        
        logger.info("\n[ORDRE D'IMPLÉMENTATION]")
        logger.info("  1. Corriger le log (5 min)")
        logger.info("  2. Rééquilibrer la fréquence (5 min)")
        logger.info("  3. Augmenter drawdown penalty (5 min)")
        logger.info("  4. Lancer un test rapide (500 steps) pour vérifier")
        logger.info("  5. Si OK, ajouter win_rate bonus et supprimer capacity_reward")
        logger.info("  6. Lancer l'entraînement complet (500k+ steps)")
    
    def run_full_radar(self):
        """Exécute l'analyse complète du radar."""
        self.define_objectives()
        self.define_constraints()
        self.analyze_current_state()
        self.propose_optimal_weights()
        self.simulate_rebalanced_scenarios()
        self.generate_recommendations()
        
        logger.info("\n" + "=" * 120)
        logger.info("RADAR STATISTIQUE TERMINÉ")
        logger.info("=" * 120)

if __name__ == '__main__':
    radar = RewardRadar()
    radar.run_full_radar()
