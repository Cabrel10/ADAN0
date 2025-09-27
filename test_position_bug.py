#!/usr/bin/env python3
"""
Test simple pour reproduire le problème position_exists=True, position_open=False
"""

import sys
import logging
import yaml

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

def test_position_bug():
    """Test pour reproduire le problème position_exists=True, position_open=False"""

    logger.info("🔍 TEST: Reproduction du problème position_exists=True, position_open=False")

    # Configuration de base
    config = {
        'portfolio': {'initial_balance': 50.0},
        'environment': {'min_capital_before_reset': 11.0},
        'trading_fees': 0.001,
        'min_order_value_usdt': 11.0,
        'capital_tiers': [
            {
                'name': 'base',
                'min_balance': 0.0,
                'max_balance': 100.0,
                'max_position_size_pct': 0.10,
                'max_concurrent_positions': 3
            }
        ]
    }

    assets = ['BTCUSDT', 'ETHUSDT']

    logger.info("=== ÉTAPE 1: Créer PortfolioManager initial ===")
    portfolio1 = PortfolioManager(env_config=config, assets=assets)
    portfolio1_id = id(portfolio1)

    logger.info(f"PortfolioManager créé: id={portfolio1_id}")
    logger.info(f"Positions initiales: {list(portfolio1.positions.keys())}")

    # Vérifier les IDs des positions initiales
    initial_position_ids = {}
    for asset, position in portfolio1.positions.items():
        initial_position_ids[asset] = id(position)
        logger.info(f"Position {asset}: id={id(position)}, is_open={position.is_open}")

    logger.info("=== ÉTAPE 2: Ouvrir une position ===")
    btc_position = portfolio1.positions.get('BTCUSDT')
    if btc_position:
        btc_position.open(entry_price=50000.0, size=0.001)
        logger.info(f"✅ Position BTCUSDT ouverte: id={id(btc_position)}, is_open={btc_position.is_open}, size={btc_position.size}")

    logger.info("=== ÉTAPE 3: Vérification état AVANT problème ===")
    for asset, position in portfolio1.positions.items():
        exists = position is not None
        is_open = position.is_open if position else False
        logger.info(f"Asset {asset}: position_exists={exists}, position_open={is_open}, position_id={id(position)}")

    logger.info("=== ÉTAPE 4: Simuler recréation du PortfolioManager ===")
    # Ceci simule ce qui se passe dans _initialize_components()
    portfolio2 = PortfolioManager(env_config=config, assets=assets)
    portfolio2_id = id(portfolio2)

    logger.info(f"Nouveau PortfolioManager créé: id={portfolio2_id}")
    logger.info(f"IDs différents? {portfolio1_id != portfolio2_id}")

    logger.info("=== ÉTAPE 5: Vérification état APRÈS recréation ===")
    for asset, position in portfolio2.positions.items():
        exists = position is not None
        is_open = position.is_open if position else False
        position_id = id(position) if position else None
        original_id = initial_position_ids.get(asset)

        logger.info(f"Asset {asset}: position_exists={exists}, position_open={is_open}")
        logger.info(f"  Position ID: original={original_id}, nouveau={position_id}, différent={original_id != position_id}")

        if exists and not is_open:
            logger.error(f"🚨 PROBLÈME REPRODUIT: {asset} position_exists=True, position_open=False")

    logger.info("=== ÉTAPE 6: Test avec _initialize_components() mockée ===")
    # Simuler ce qui devrait se passer avec nos corrections
    class MockEnvironment:
        def __init__(self):
            self.portfolio = None
            self.config = config
            self.assets = assets

        def _initialize_components_old(self):
            """Ancienne version qui cause le problème"""
            self.portfolio = PortfolioManager(env_config=self.config, assets=self.assets)
            logger.info(f"OLD: Nouveau PortfolioManager créé: id={id(self.portfolio)}")

        def _initialize_components_new(self):
            """Nouvelle version avec nos corrections"""
            if not hasattr(self, 'portfolio') or self.portfolio is None:
                self.portfolio = PortfolioManager(env_config=self.config, assets=self.assets)
                logger.info(f"NEW: Nouveau PortfolioManager créé: id={id(self.portfolio)}")
            else:
                logger.info(f"NEW: Réutilisation PortfolioManager existant: id={id(self.portfolio)}")

    logger.info("--- Test ancienne version ---")
    env_old = MockEnvironment()
    env_old._initialize_components_old()

    # Ouvrir position
    btc_pos_old = env_old.portfolio.positions.get('BTCUSDT')
    btc_pos_old.open(entry_price=50000.0, size=0.001)
    logger.info(f"Position ouverte: is_open={btc_pos_old.is_open}")

    # Recréer (simule le problème)
    old_id = id(env_old.portfolio)
    env_old._initialize_components_old()
    new_pos_old = env_old.portfolio.positions.get('BTCUSDT')
    logger.info(f"Après recréation: position_exists={new_pos_old is not None}, position_open={new_pos_old.is_open}")
    logger.info(f"PortfolioManager recréé? {old_id != id(env_old.portfolio)}")

    logger.info("--- Test nouvelle version ---")
    env_new = MockEnvironment()
    env_new._initialize_components_new()

    # Ouvrir position
    btc_pos_new = env_new.portfolio.positions.get('BTCUSDT')
    btc_pos_new.open(entry_price=50000.0, size=0.001)
    logger.info(f"Position ouverte: is_open={btc_pos_new.is_open}")

    # Essayer de recréer (devrait être évité)
    old_id_new = id(env_new.portfolio)
    env_new._initialize_components_new()
    preserved_pos = env_new.portfolio.positions.get('BTCUSDT')
    logger.info(f"Après tentative recréation: position_exists={preserved_pos is not None}, position_open={preserved_pos.is_open}")
    logger.info(f"PortfolioManager préservé? {old_id_new == id(env_new.portfolio)}")

    logger.info("=== RÉSULTATS ===")
    if old_id != id(env_old.portfolio) and new_pos_old and not new_pos_old.is_open:
        logger.info("✅ Problème reproduit avec ancienne version")
    if old_id_new == id(env_new.portfolio) and preserved_pos and preserved_pos.is_open:
        logger.info("✅ Problème résolu avec nouvelle version")
    else:
        logger.warning("❌ Correction non confirmée")

def test_soft_reset_preservation():
    """Test pour vérifier que _perform_soft_reset préserve les positions"""

    logger.info("🔍 TEST: Vérification _perform_soft_reset")

    config = {
        'portfolio': {'initial_balance': 100.0},
        'environment': {'min_capital_before_reset': 11.0},
        'trading_fees': 0.001,
        'min_order_value_usdt': 11.0,
        'capital_tiers': [
            {
                'name': 'base',
                'min_balance': 0.0,
                'max_balance': 200.0,
                'max_position_size_pct': 0.10,
                'max_concurrent_positions': 3
            }
        ]
    }

    portfolio = PortfolioManager(env_config=config, assets=['BTCUSDT', 'ETHUSDT'])

    # Ouvrir position
    btc_position = portfolio.positions.get('BTCUSDT')
    btc_position.open(entry_price=50000.0, size=0.002)

    logger.info(f"Avant soft reset: position_open={btc_position.is_open}, size={btc_position.size}")
    logger.info(f"Position ID avant: {id(btc_position)}")

    # Effectuer soft reset
    portfolio._perform_soft_reset(current_value=120.0)

    # Vérifier après
    btc_after = portfolio.positions.get('BTCUSDT')
    logger.info(f"Après soft reset: position_open={btc_after.is_open}, size={btc_after.size}")
    logger.info(f"Position ID après: {id(btc_after)}")
    logger.info(f"Même objet? {id(btc_position) == id(btc_after)}")

    if btc_after.is_open and id(btc_position) == id(btc_after):
        logger.info("✅ Soft reset préserve correctement les positions")
    else:
        logger.error("❌ Soft reset ne préserve pas les positions")

def main():
    """Fonction principale"""
    try:
        test_position_bug()
        print("\n" + "="*60 + "\n")
        test_soft_reset_preservation()

    except Exception as e:
        logger.error(f"Erreur durant les tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
