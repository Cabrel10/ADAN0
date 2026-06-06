#!/usr/bin/env python3
"""
Test rapide: 500 steps avec les nouveaux poids de récompense.
Objectif: Vérifier que l'agent trade toujours et que les métriques s'améliorent.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import yaml
import logging
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Charge la config."""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_data():
    """Charge les données de test."""
    try:
        df = pd.read_parquet('data/processed/indicators/test/BTCUSDT/5m.parquet')
        logger.info(f"✓ Données chargées: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"✗ Erreur chargement données: {e}")
        return None

def create_env(config, data):
    """Crée l'environnement."""
    try:
        from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
        
        env = MultiAssetChunkedEnv(
            config=config,
            worker_id=0,
            worker_config=config.get('workers', {}).get('w1', {}),
        )
        logger.info("✓ Environnement créé")
        return env
    except Exception as e:
        logger.error(f"✗ Erreur création env: {e}")
        return None

def run_test(env, steps=500):
    """Lance le test."""
    logger.info(f"\n{'='*80}")
    logger.info(f"TEST: 500 STEPS AVEC NOUVEAUX POIDS")
    logger.info(f"{'='*80}\n")
    
    # Métriques
    rewards = deque(maxlen=100)
    pnls = deque(maxlen=100)
    trades_count = 0
    winning_trades = 0
    losing_trades = 0
    
    try:
        obs, info = env.reset()
        logger.info(f"✓ Environnement reset")
        logger.info(f"  Initial capital: ${env.portfolio_manager.initial_capital:.2f}")
        logger.info(f"  Initial equity: ${env.portfolio_manager.get_portfolio_value():.2f}\n")
        
        for step in range(steps):
            # Action aléatoire pour le test
            action = np.random.uniform(-1, 1, size=15).astype(np.float32)
            
            obs, reward, done, truncated, info = env.step(action)
            
            rewards.append(reward)
            
            # Extraire PnL du dernier trade
            if hasattr(env, '_step_closed_receipts') and env._step_closed_receipts:
                for receipt in env._step_closed_receipts:
                    if isinstance(receipt, dict):
                        pnl = receipt.get('pnl', 0.0)
                        pnls.append(pnl)
                        if pnl > 0:
                            winning_trades += 1
                        elif pnl < 0:
                            losing_trades += 1
                        trades_count += 1
            
            # Log tous les 50 steps
            if (step + 1) % 50 == 0:
                avg_reward = np.mean(list(rewards)) if rewards else 0
                portfolio_value = env.portfolio_manager.get_portfolio_value()
                equity = env.portfolio_manager.get_equity()
                
                logger.info(f"[Step {step+1:3d}] Reward: {reward:+.6f} | "
                           f"Avg(50): {avg_reward:+.6f} | "
                           f"Portfolio: ${portfolio_value:.2f} | "
                           f"Equity: ${equity:.2f} | "
                           f"Trades: {trades_count}")
            
            if done or truncated:
                logger.info(f"\n✓ Episode terminé au step {step+1}")
                break
        
        # Résumé final
        logger.info(f"\n{'='*80}")
        logger.info(f"RÉSUMÉ FINAL")
        logger.info(f"{'='*80}\n")
        
        final_portfolio = env.portfolio_manager.get_portfolio_value()
        final_equity = env.portfolio_manager.get_equity()
        pnl_total = final_portfolio - env.portfolio_manager.initial_capital
        return_pct = (pnl_total / env.portfolio_manager.initial_capital) * 100
        
        logger.info(f"Steps exécutés:        {step+1}")
        logger.info(f"Initial capital:       ${env.portfolio_manager.initial_capital:.2f}")
        logger.info(f"Final portfolio:       ${final_portfolio:.2f}")
        logger.info(f"Final equity:          ${final_equity:.2f}")
        logger.info(f"Total PnL:             ${pnl_total:+.2f}")
        logger.info(f"Return:                {return_pct:+.2f}%")
        logger.info(f"\nTrades exécutés:       {trades_count}")
        logger.info(f"  Gagnants:            {winning_trades}")
        logger.info(f"  Perdants:            {losing_trades}")
        
        if trades_count > 0:
            win_rate = (winning_trades / trades_count) * 100
            avg_pnl = np.mean(list(pnls)) if pnls else 0
            logger.info(f"  Win rate:            {win_rate:.1f}%")
            logger.info(f"  Avg PnL/trade:       ${avg_pnl:+.4f}")
        
        avg_reward_final = np.mean(list(rewards)) if rewards else 0
        logger.info(f"\nMoyenne récompense:    {avg_reward_final:+.6f}")
        
        # Vérifications
        logger.info(f"\n{'='*80}")
        logger.info(f"VÉRIFICATIONS")
        logger.info(f"{'='*80}\n")
        
        checks = []
        
        # Check 1: Agent trade
        if trades_count > 0:
            logger.info(f"✓ Agent trade: {trades_count} trades exécutés")
            checks.append(True)
        else:
            logger.warning(f"✗ Agent ne trade pas (0 trades)")
            checks.append(False)
        
        # Check 2: Récompense moyenne positive
        if avg_reward_final > 0:
            logger.info(f"✓ Récompense moyenne positive: {avg_reward_final:+.6f}")
            checks.append(True)
        else:
            logger.warning(f"✗ Récompense moyenne négative: {avg_reward_final:+.6f}")
            checks.append(False)
        
        # Check 3: Portfolio ne s'effondre pas
        if final_portfolio > env.portfolio_manager.initial_capital * 0.5:
            logger.info(f"✓ Portfolio stable: ${final_portfolio:.2f} (> 50% initial)")
            checks.append(True)
        else:
            logger.warning(f"✗ Portfolio effondré: ${final_portfolio:.2f} (< 50% initial)")
            checks.append(False)
        
        # Check 4: Win rate > 40%
        if trades_count > 0:
            win_rate = (winning_trades / trades_count) * 100
            if win_rate > 40:
                logger.info(f"✓ Win rate acceptable: {win_rate:.1f}%")
                checks.append(True)
            else:
                logger.warning(f"✗ Win rate faible: {win_rate:.1f}%")
                checks.append(False)
        
        logger.info(f"\n{'='*80}")
        if all(checks):
            logger.info(f"✓ TEST RÉUSSI: Tous les critères satisfaits")
        else:
            logger.warning(f"⚠️  TEST PARTIEL: {sum(checks)}/{len(checks)} critères satisfaits")
        logger.info(f"{'='*80}\n")
        
        return all(checks)
        
    except Exception as e:
        logger.error(f"✗ Erreur pendant le test: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    logger.info("Chargement de la config...")
    config = load_config()
    
    logger.info("Chargement des données...")
    data = load_data()
    
    if data is None:
        sys.exit(1)
    
    logger.info("Création de l'environnement...")
    env = create_env(config, data)
    
    if env is None:
        sys.exit(1)
    
    success = run_test(env, steps=500)
    sys.exit(0 if success else 1)
