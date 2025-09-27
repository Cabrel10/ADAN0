#!/usr/bin/env python3
"""
Test pour tracer précisément le problème de timing des positions.

Ce script suit une position spécifique à travers plusieurs steps pour comprendre
quand exactement elle passe de is_open=True à is_open=False.
"""

import sys
import logging
import time
import threading
from typing import Dict, Any

# Ajouter le chemin du projet
sys.path.insert(0, '/home/morningstar/Documents/trading')
sys.path.insert(0, '/home/morningstar/Documents/trading/bot/src')

from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager, Position

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PositionTracker:
    """Classe pour tracer précisément l'état des positions"""

    def __init__(self):
        self.position_history = []
        self.portfolio_history = []

    def track_position(self, step: int, asset: str, position: Position, context: str = ""):
        """Tracer l'état d'une position à un moment donné"""
        record = {
            'step': step,
            'asset': asset,
            'position_id': id(position),
            'is_open': position.is_open,
            'size': position.size,
            'entry_price': position.entry_price,
            'context': context,
            'timestamp': time.time()
        }
        self.position_history.append(record)
        logger.info(f"[TRACK] Step {step} {context}: {asset} position_id={id(position)}, is_open={position.is_open}, size={position.size}")

    def track_portfolio(self, step: int, portfolio: PortfolioManager, context: str = ""):
        """Tracer l'état du portfolio"""
        record = {
            'step': step,
            'portfolio_id': id(portfolio),
            'context': context,
            'timestamp': time.time()
        }
        self.portfolio_history.append(record)
        logger.info(f"[TRACK] Step {step} {context}: portfolio_id={id(portfolio)}")

    def analyze_changes(self):
        """Analyser les changements dans l'historique"""
        logger.info("=== ANALYSE DES CHANGEMENTS ===")

        # Grouper par asset
        by_asset = {}
        for record in self.position_history:
            asset = record['asset']
            if asset not in by_asset:
                by_asset[asset] = []
            by_asset[asset].append(record)

        for asset, records in by_asset.items():
            logger.info(f"\n--- Analyse pour {asset} ---")
            prev_record = None

            for record in records:
                if prev_record:
                    # Vérifier les changements
                    id_changed = prev_record['position_id'] != record['position_id']
                    state_changed = prev_record['is_open'] != record['is_open']

                    if id_changed:
                        logger.warning(f"🚨 ID CHANGÉ: Step {prev_record['step']} → {record['step']}: {prev_record['position_id']} → {record['position_id']}")

                    if state_changed:
                        logger.warning(f"🚨 ÉTAT CHANGÉ: Step {prev_record['step']} → {record['step']}: is_open={prev_record['is_open']} → {record['is_open']}")

                prev_record = record

def test_position_timing():
    """Test pour tracer le timing des positions"""

    logger.info("🔍 TEST: Tracing Position Timing")

    # Configuration
    config = {
        'portfolio': {'initial_balance': 100.0},
        'environment': {'min_capital_before_reset': 11.0},
        'trading_fees': 0.001,
        'min_order_value_usdt': 11.0,
        'capital_tiers': [
            {
                'name': 'base',
                'min_capital': 0.0,
                'max_balance': 200.0,
                'max_position_size_pct': 0.10,
                'max_concurrent_positions': 3,
                'leverage': 1.0,
                'risk_per_trade_pct': 2.0,
                'max_drawdown_pct': 25.0
            }
        ]
    }

    tracker = PositionTracker()

    # Étape 1: Créer PortfolioManager
    logger.info("=== ÉTAPE 1: Création PortfolioManager ===")
    portfolio = PortfolioManager(env_config=config, assets=['BTCUSDT'])
    tracker.track_portfolio(0, portfolio, "CREATION")

    # Suivre la position initiale
    btc_position = portfolio.positions.get('BTCUSDT')
    tracker.track_position(0, 'BTCUSDT', btc_position, "INITIAL")

    # Étape 2: Ouvrir position
    logger.info("=== ÉTAPE 2: Ouverture position ===")
    btc_position.open(entry_price=50000.0, size=0.001)
    tracker.track_position(1, 'BTCUSDT', btc_position, "AFTER_OPEN")

    # Simuler ce qui se passe dans l'environnement
    logger.info("=== ÉTAPE 3: Simulation steps environment ===")

    for step in range(2, 6):
        logger.info(f"\n--- Step {step} ---")

        # Vérifier état avant toute opération
        current_position = portfolio.positions.get('BTCUSDT')
        tracker.track_position(step, 'BTCUSDT', current_position, f"START_STEP_{step}")

        # Simuler diverses opérations qui pourraient affecter la position

        # 1. reset() avec continuité
        if step == 3:
            logger.info(f"  → Calling portfolio.reset() with continuity")
            portfolio.reset(new_epoch=False, force=False, min_capital_before_reset=11.0)
            tracker.track_portfolio(step, portfolio, "AFTER_RESET")
            current_position = portfolio.positions.get('BTCUSDT')
            tracker.track_position(step, 'BTCUSDT', current_position, f"AFTER_RESET")

        # 2. Mise à jour des prix de marché
        if step == 4:
            logger.info(f"  → Calling update_market_price()")
            portfolio.update_market_price('BTCUSDT', 51000.0)
            current_position = portfolio.positions.get('BTCUSDT')
            tracker.track_position(step, 'BTCUSDT', current_position, f"AFTER_PRICE_UPDATE")

        # 3. soft reset
        if step == 5:
            logger.info(f"  → Calling _perform_soft_reset()")
            portfolio._perform_soft_reset(current_value=120.0)
            tracker.track_portfolio(step, portfolio, "AFTER_SOFT_RESET")
            current_position = portfolio.positions.get('BTCUSDT')
            tracker.track_position(step, 'BTCUSDT', current_position, f"AFTER_SOFT_RESET")

        # Vérifier état final du step
        final_position = portfolio.positions.get('BTCUSDT')
        tracker.track_position(step, 'BTCUSDT', final_position, f"END_STEP_{step}")

        time.sleep(0.1)  # Petit délai pour séparer les logs

    # Analyse finale
    tracker.analyze_changes()

    return tracker

def test_position_recreation_scenario():
    """Test du scénario de recréation de positions"""

    logger.info("\n🔍 TEST: Scénario de recréation")

    config = {
        'portfolio': {'initial_balance': 50.0},
        'environment': {'min_capital_before_reset': 11.0},
        'trading_fees': 0.001,
        'min_order_value_usdt': 11.0,
        'capital_tiers': [
            {
                'name': 'base',
                'min_capital': 0.0,
                'max_balance': 100.0,
                'max_position_size_pct': 0.10,
                'max_concurrent_positions': 3,
                'leverage': 1.0,
                'risk_per_trade_pct': 2.0,
                'max_drawdown_pct': 25.0
            }
        ]
    }

    # Créer premier portfolio
    logger.info("--- Portfolio 1 ---")
    portfolio1 = PortfolioManager(env_config=config, assets=['BTCUSDT'])
    btc_pos1 = portfolio1.positions.get('BTCUSDT')
    btc_pos1.open(entry_price=50000.0, size=0.001)

    logger.info(f"Portfolio1 ID: {id(portfolio1)}")
    logger.info(f"Position1 ID: {id(btc_pos1)}, is_open: {btc_pos1.is_open}")

    # Simuler ce qui se passe quand un nouvel environnement est créé (nouveau worker)
    logger.info("--- Portfolio 2 (nouveau worker) ---")
    portfolio2 = PortfolioManager(env_config=config, assets=['BTCUSDT'])
    btc_pos2 = portfolio2.positions.get('BTCUSDT')

    logger.info(f"Portfolio2 ID: {id(portfolio2)}")
    logger.info(f"Position2 ID: {id(btc_pos2)}, is_open: {btc_pos2.is_open}")

    # Comparaison
    logger.info("--- Comparaison ---")
    logger.info(f"Même Portfolio? {id(portfolio1) == id(portfolio2)}")
    logger.info(f"Même Position? {id(btc_pos1) == id(btc_pos2)}")
    logger.info(f"Position1 ouverte? {btc_pos1.is_open}")
    logger.info(f"Position2 ouverte? {btc_pos2.is_open}")

    if btc_pos2 and not btc_pos2.is_open:
        logger.error("🚨 PROBLÈME REPRODUIT: Nouvelle instance a position_exists=True, position_open=False")

    return {
        'portfolio1_id': id(portfolio1),
        'portfolio2_id': id(portfolio2),
        'position1_id': id(btc_pos1),
        'position2_id': id(btc_pos2),
        'position1_open': btc_pos1.is_open,
        'position2_open': btc_pos2.is_open
    }

def main():
    """Fonction principale"""

    try:
        logger.info("🚀 DÉBUT DES TESTS DE TIMING")
        logger.info("=" * 60)

        # Test 1: Timing des opérations
        tracker = test_position_timing()

        logger.info("\n" + "=" * 60)

        # Test 2: Scénario de recréation
        recreation_results = test_position_recreation_scenario()

        logger.info("\n" + "=" * 60)
        logger.info("📊 RÉSUMÉ FINAL")
        logger.info(f"Recreation test results: {recreation_results}")

        # Recommandations
        logger.info("\n🔧 RECOMMANDATIONS:")
        if recreation_results['position1_open'] and not recreation_results['position2_open']:
            logger.info("1. ✅ Problème confirmé: nouvelles instances ont positions fermées")
            logger.info("2. 🎯 Solution: éviter création multiples PortfolioManager")
            logger.info("3. 🔧 Action: vérifier que _initialize_components() réutilise portfolio existant")

        logger.info("\n✅ Tests terminés avec succès")

    except Exception as e:
        logger.error(f"Erreur durant les tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
