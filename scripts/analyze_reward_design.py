#!/usr/bin/env python3
"""
Analyse statistique rigoureuse de la fonction de récompense.
Identifie le Reward Hacking et propose un rééquilibrage fondé sur des données.

Objectif: Trouver l'équilibre optimal entre:
- PnL net (signal financier réel)
- Fréquence de trading (contrainte de liquidité)
- Capacité (gestion du risque)
- Pénalités (discipline)
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

class RewardAnalyzer:
    """Analyse la fonction de récompense et identifie les biais d'optimisation."""
    
    def __init__(self):
        self.components = {}
        self.weights = {}
        self.scenarios = []
        
    def extract_reward_components(self):
        """Extrait les composantes de récompense du code."""
        logger.info("=" * 100)
        logger.info("ÉTAPE 1: EXTRACTION DES COMPOSANTES DE RÉCOMPENSE")
        logger.info("=" * 100)
        
        # Basé sur _calculate_reward() ligne 5891
        components = {
            "pnl_net_scaled": {
                "description": "PnL net après frais, scalé par 100/initial_capital",
                "typical_range": (-0.5, 0.5),  # $-10 à $+10 sur $20 capital
                "formula": "realized_pnl * (100 / initial_capital)",
                "source": "Trades fermés (SL/TP/Agent)",
            },
            "trade_cost": {
                "description": "Pénalité de slippage (notional * 0.15%)",
                "typical_range": (-0.01, 0.0),
                "formula": "notional * 0.0015 * reward_scale",
                "source": "Trades exécutés",
            },
            "drawdown_penalty": {
                "description": "Pénalité quadratique si drawdown > 2%",
                "typical_range": (-5.0, 0.0),
                "formula": "(|dd| - 0.02)^2 * 50.0",
                "source": "Portfolio metrics",
            },
            "inaction": {
                "description": "Pénalité pour inaction (time decay)",
                "typical_range": (-0.001, 0.0),
                "formula": "-0.001 si realized_pnl == 0",
                "source": "Config: reward_shaping.time_decay",
            },
            "invalid_penalty": {
                "description": "Pénalité pour trades rejetés (fee_gate, risk_gate, cooldown)",
                "typical_range": (-0.1, 0.0),
                "formula": "sum(gate_rejections * -0.005)",
                "source": "_step_invalid_penalty",
            },
            "capacity_reward": {
                "description": "Bonus pour respecter max_concurrent_positions",
                "typical_range": (-0.2, 0.2),
                "formula": "Tier-dependent",
                "source": "calculate_capacity_based_reward()",
            },
            "frequency_reward": {
                "description": "Bonus pour respecter min/max positions par timeframe",
                "typical_range": (-0.5, 0.5),
                "formula": "in_range_weight * (count/max) ou -out_of_range_weight * delta",
                "source": "_calculate_frequency_reward()",
            },
        }
        
        for name, info in components.items():
            logger.info(f"\n[{name}]")
            logger.info(f"  Description: {info['description']}")
            logger.info(f"  Plage typique: {info['typical_range']}")
            logger.info(f"  Formule: {info['formula']}")
            logger.info(f"  Source: {info['source']}")
            self.components[name] = info
        
        return components
    
    def extract_weights_from_config(self):
        """Extrait les poids de la config."""
        logger.info("\n" + "=" * 100)
        logger.info("ÉTAPE 2: EXTRACTION DES POIDS DE LA CONFIG")
        logger.info("=" * 100)
        
        # Basé sur config/config.yaml
        weights = {
            "frequency_weights": {
                "in_range": 2.0,
                "out_of_range": 0.2,
            },
            "time_decay": -0.001,
            "invalid_trade_penalty_weight": 0.005,
            "drawdown_penalty_multiplier": 50.0,
            "trade_cost_multiplier": 0.0015,  # 0.15% slippage
        }
        
        for category, values in weights.items():
            logger.info(f"\n[{category}]")
            if isinstance(values, dict):
                for key, val in values.items():
                    logger.info(f"  {key}: {val}")
            else:
                logger.info(f"  {values}")
        
        self.weights = weights
        return weights
    
    def simulate_scenarios(self) -> pd.DataFrame:
        """Simule différents scénarios pour quantifier le Reward Hacking."""
        logger.info("\n" + "=" * 100)
        logger.info("ÉTAPE 3: SIMULATION DE SCÉNARIOS (Reward Hacking Detection)")
        logger.info("=" * 100)
        
        scenarios = []
        
        # Scénario 1: Trade gagnant (TP atteint)
        scenarios.append({
            "name": "Trade gagnant (TP)",
            "pnl_net_scaled": 0.12,  # $0.12 PnL sur $20 = 0.6% → symlog(0.006) ≈ 0.006 * 100 = 0.12
            "trade_cost": -0.001,
            "drawdown_penalty": 0.0,
            "inaction": 0.0,
            "invalid_penalty": 0.0,
            "capacity_reward": 0.0,
            "frequency_reward": 0.2,  # Bonus pour avoir ouvert un trade
        })
        
        # Scénario 2: Trade perdant (SL atteint)
        scenarios.append({
            "name": "Trade perdant (SL)",
            "pnl_net_scaled": -0.15,  # -$0.30 PnL sur $20 = -1.5%
            "trade_cost": -0.001,
            "drawdown_penalty": 0.0,
            "inaction": 0.0,
            "invalid_penalty": 0.0,
            "capacity_reward": 0.0,
            "frequency_reward": 0.2,  # Bonus pour avoir ouvert un trade (même perdant!)
        })
        
        # Scénario 3: HOLD (pas de trade)
        scenarios.append({
            "name": "HOLD (pas de trade)",
            "pnl_net_scaled": 0.0,
            "trade_cost": 0.0,
            "drawdown_penalty": 0.0,
            "inaction": -0.001,
            "invalid_penalty": 0.0,
            "capacity_reward": 0.0,
            "frequency_reward": 0.214,  # Bonus pour respecter la fréquence (même sans trade!)
        })
        
        # Scénario 4: Trade aléatoire (50% win rate)
        scenarios.append({
            "name": "Trade aléatoire (50% win)",
            "pnl_net_scaled": 0.0,  # Moyenne: 50% * 0.12 + 50% * (-0.15) = -0.015
            "trade_cost": -0.001,
            "drawdown_penalty": 0.0,
            "inaction": 0.0,
            "invalid_penalty": 0.0,
            "capacity_reward": 0.0,
            "frequency_reward": 0.2,  # Bonus pour avoir tradé
        })
        
        # Scénario 5: Drawdown > 2%
        scenarios.append({
            "name": "Drawdown > 2% (pénalité quadratique)",
            "pnl_net_scaled": 0.0,
            "trade_cost": 0.0,
            "drawdown_penalty": -(0.05 - 0.02) ** 2 * 50.0,  # 5% DD → -0.045
            "inaction": 0.0,
            "invalid_penalty": 0.0,
            "capacity_reward": 0.0,
            "frequency_reward": 0.0,
        })
        
        # Scénario 6: Rejet par fee_gate (EV négatif)
        scenarios.append({
            "name": "Rejet fee_gate (EV < 0)",
            "pnl_net_scaled": 0.0,
            "trade_cost": 0.0,
            "drawdown_penalty": 0.0,
            "inaction": 0.0,
            "invalid_penalty": -0.005,  # Pénalité pour rejet
            "capacity_reward": 0.0,
            "frequency_reward": 0.0,
        })
        
        df = pd.DataFrame(scenarios)
        df["raw_reward"] = (
            df["pnl_net_scaled"] 
            - df["trade_cost"] 
            - df["drawdown_penalty"] 
            + df["inaction"] 
            + df["invalid_penalty"] 
            + df["capacity_reward"] 
            + df["frequency_reward"]
        )
        
        # Appliquer symlog
        df["final_reward"] = np.sign(df["raw_reward"]) * np.log1p(np.abs(df["raw_reward"]))
        
        logger.info("\n" + "-" * 100)
        logger.info("RÉSULTATS DES SCÉNARIOS:")
        logger.info("-" * 100)
        
        for idx, row in df.iterrows():
            logger.info(f"\n[{row['name']}]")
            logger.info(f"  PnL net:           {row['pnl_net_scaled']:+.6f}")
            logger.info(f"  Trade cost:        {row['trade_cost']:+.6f}")
            logger.info(f"  Drawdown penalty:  {row['drawdown_penalty']:+.6f}")
            logger.info(f"  Inaction:          {row['inaction']:+.6f}")
            logger.info(f"  Invalid penalty:   {row['invalid_penalty']:+.6f}")
            logger.info(f"  Capacity reward:   {row['capacity_reward']:+.6f}")
            logger.info(f"  Frequency reward:  {row['frequency_reward']:+.6f}")
            logger.info(f"  ─────────────────────────────────")
            logger.info(f"  Raw reward:        {row['raw_reward']:+.6f}")
            logger.info(f"  Final reward:      {row['final_reward']:+.6f}")
        
        self.scenarios = df
        return df
    
    def identify_reward_hacking(self):
        """Identifie les biais d'optimisation (Reward Hacking)."""
        logger.info("\n" + "=" * 100)
        logger.info("ÉTAPE 4: DIAGNOSTIC DU REWARD HACKING")
        logger.info("=" * 100)
        
        df = self.scenarios
        
        # Comparaison clé: Trade gagnant vs Trade perdant vs HOLD
        winning_trade = df[df["name"] == "Trade gagnant (TP)"]["final_reward"].values[0]
        losing_trade = df[df["name"] == "Trade perdant (SL)"]["final_reward"].values[0]
        hold = df[df["name"] == "HOLD (pas de trade)"]["final_reward"].values[0]
        random_trade = df[df["name"] == "Trade aléatoire (50% win)"]["final_reward"].values[0]
        
        logger.info(f"\n[COMPARAISON CLÉS]")
        logger.info(f"  Trade gagnant:     {winning_trade:+.6f}")
        logger.info(f"  Trade perdant:     {losing_trade:+.6f}")
        logger.info(f"  HOLD:              {hold:+.6f}")
        logger.info(f"  Trade aléatoire:   {random_trade:+.6f}")
        
        logger.info(f"\n[PROBLÈMES IDENTIFIÉS]")
        
        # Problème 1: Trade perdant > HOLD
        if losing_trade > hold:
            logger.warning(
                f"  ⚠️  REWARD HACKING DÉTECTÉ: Trade perdant ({losing_trade:+.6f}) > HOLD ({hold:+.6f})"
            )
            logger.warning(f"      → L'agent est incité à perdre de l'argent plutôt que de ne rien faire!")
        
        # Problème 2: Trade aléatoire > HOLD
        if random_trade > hold:
            logger.warning(
                f"  ⚠️  REWARD HACKING DÉTECTÉ: Trade aléatoire ({random_trade:+.6f}) > HOLD ({hold:+.6f})"
            )
            logger.warning(f"      → L'agent est incité à trader au hasard!")
        
        # Problème 3: Fréquence domine PnL
        freq_component = df[df["name"] == "Trade gagnant (TP)"]["frequency_reward"].values[0]
        pnl_component = df[df["name"] == "Trade gagnant (TP)"]["pnl_net_scaled"].values[0]
        
        if freq_component > pnl_component:
            logger.warning(
                f"  ⚠️  REWARD HACKING DÉTECTÉ: Fréquence ({freq_component:+.6f}) > PnL ({pnl_component:+.6f})"
            )
            logger.warning(f"      → L'agent optimise la fréquence, pas le profit!")
        
        # Problème 4: Drawdown penalty trop faible
        dd_scenario = df[df["name"] == "Drawdown > 2% (pénalité quadratique)"]["final_reward"].values[0]
        if dd_scenario > -0.01:
            logger.warning(
                f"  ⚠️  DRAWDOWN PENALTY FAIBLE: {dd_scenario:+.6f}"
            )
            logger.warning(f"      → La pénalité de drawdown n'est pas assez dissuasive!")
        
        return {
            "winning_trade": winning_trade,
            "losing_trade": losing_trade,
            "hold": hold,
            "random_trade": random_trade,
        }
    
    def propose_rebalancing(self):
        """Propose un rééquilibrage fondé sur des principes quantitatifs."""
        logger.info("\n" + "=" * 100)
        logger.info("ÉTAPE 5: PROPOSITION DE RÉÉQUILIBRAGE")
        logger.info("=" * 100)
        
        logger.info("\n[PRINCIPES DE RÉÉQUILIBRAGE]")
        logger.info("  1. PnL net doit dominer nettement (>70% du signal)")
        logger.info("  2. Fréquence doit être une contrainte dure, pas un bonus")
        logger.info("  3. Capacité doit être une contrainte dure, pas un bonus")
        logger.info("  4. Pénalités doivent être proportionnelles au risque")
        logger.info("  5. Trade perdant < HOLD < Trade gagnant (hiérarchie claire)")
        
        logger.info("\n[PROPOSITION 1: Réduction drastique de la fréquence]")
        logger.info("  Avant: in_range=2.0, out_of_range=0.2")
        logger.info("  Après: in_range=0.1, out_of_range=0.05")
        logger.info("  Rationale: Fréquence devient une contrainte, pas un bonus")
        
        logger.info("\n[PROPOSITION 2: Suppression de capacity_reward]")
        logger.info("  Avant: capacity_reward inclus dans raw")
        logger.info("  Après: capacity_reward = 0 (contrainte dure via risk_gate)")
        logger.info("  Rationale: Max concurrent positions est une limite, pas un objectif")
        
        logger.info("\n[PROPOSITION 3: Augmentation de la pénalité de drawdown]")
        logger.info("  Avant: (|dd| - 0.02)^2 * 50.0")
        logger.info("  Après: (|dd| - 0.02)^2 * 100.0 (ou exponentielle)")
        logger.info("  Rationale: Drawdown > 2% doit être très pénalisé")
        
        logger.info("\n[PROPOSITION 4: Pénalité progressive pour inaction]")
        logger.info("  Avant: -0.001 constant")
        logger.info("  Après: -0.001 * (steps_since_trade / 100)")
        logger.info("  Rationale: Encourage l'action, mais pas l'action aléatoire")
        
        logger.info("\n[PROPOSITION 5: Bonus pour win_rate > 50%]")
        logger.info("  Avant: Aucun bonus de win_rate")
        logger.info("  Après: +0.05 si win_rate > 55%, -0.05 si win_rate < 45%")
        logger.info("  Rationale: Récompense la qualité des trades, pas la quantité")
        
        # Simuler le rééquilibrage
        logger.info("\n" + "-" * 100)
        logger.info("SIMULATION DU RÉÉQUILIBRAGE:")
        logger.info("-" * 100)
        
        df_rebalanced = self.scenarios.copy()
        
        # Appliquer les changements
        df_rebalanced.loc[df_rebalanced["name"] == "Trade gagnant (TP)", "frequency_reward"] = 0.05
        df_rebalanced.loc[df_rebalanced["name"] == "Trade perdant (SL)", "frequency_reward"] = 0.05
        df_rebalanced.loc[df_rebalanced["name"] == "HOLD (pas de trade)", "frequency_reward"] = 0.05
        df_rebalanced.loc[df_rebalanced["name"] == "Trade aléatoire (50% win)", "frequency_reward"] = 0.05
        
        # Augmenter drawdown penalty
        df_rebalanced.loc[df_rebalanced["name"] == "Drawdown > 2% (pénalité quadratique)", "drawdown_penalty"] = \
            -(0.05 - 0.02) ** 2 * 100.0
        
        df_rebalanced["raw_reward_new"] = (
            df_rebalanced["pnl_net_scaled"] 
            - df_rebalanced["trade_cost"] 
            - df_rebalanced["drawdown_penalty"] 
            + df_rebalanced["inaction"] 
            + df_rebalanced["invalid_penalty"] 
            + df_rebalanced["frequency_reward"]
        )
        
        df_rebalanced["final_reward_new"] = np.sign(df_rebalanced["raw_reward_new"]) * np.log1p(np.abs(df_rebalanced["raw_reward_new"]))
        
        logger.info("\n[COMPARAISON AVANT/APRÈS]")
        logger.info(f"{'Scénario':<40} {'Avant':>12} {'Après':>12} {'Δ':>12}")
        logger.info("-" * 80)
        
        for idx, row in df_rebalanced.iterrows():
            before = self.scenarios.iloc[idx]["final_reward"]
            after = row["final_reward_new"]
            delta = after - before
            logger.info(f"{row['name']:<40} {before:>+12.6f} {after:>+12.6f} {delta:>+12.6f}")
        
        return df_rebalanced
    
    def run_full_analysis(self):
        """Exécute l'analyse complète."""
        self.extract_reward_components()
        self.extract_weights_from_config()
        self.simulate_scenarios()
        self.identify_reward_hacking()
        self.propose_rebalancing()
        
        logger.info("\n" + "=" * 100)
        logger.info("ANALYSE COMPLÈTE TERMINÉE")
        logger.info("=" * 100)

if __name__ == '__main__':
    analyzer = RewardAnalyzer()
    analyzer.run_full_analysis()
