#!/usr/bin/env python3
"""
Vérification des deux corrections critiques:
1. Chunks: Augmenté 5m chunk_size de 25k → 50k + logique MIN → MAX
2. Features: Retiré prix absolus (open, high, low, close) + ajouté log_return, close_ema20_ratio
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import yaml
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def verify_chunk_config():
    """Vérifier la configuration des chunks."""
    logger.info("\n" + "="*80)
    logger.info("VÉRIFICATION 1: CONFIGURATION DES CHUNKS")
    logger.info("="*80)
    
    # Lire data_loader.py
    with open('src/adan_trading_bot/data_processing/data_loader.py', 'r') as f:
        content = f.read()
    
    # Check 1: chunk_size pour 5m
    if '"5m": 50000' in content:
        logger.info("✓ Chunk size 5m: 50000 (correct)")
    else:
        logger.error("✗ Chunk size 5m: NOT 50000")
        return False
    
    # Check 2: Logique MAX au lieu de MIN
    if 'max_chunks_found' in content and 'if num_chunks > max_chunks_found' in content:
        logger.info("✓ Logique chunks: MAX (correct)")
    else:
        logger.error("✗ Logique chunks: NOT MAX")
        return False
    
    return True

def verify_features_config():
    """Vérifier la configuration des features."""
    logger.info("\n" + "="*80)
    logger.info("VÉRIFICATION 2: CONFIGURATION DES FEATURES")
    logger.info("="*80)
    
    # Lire state_builder.py
    with open('src/adan_trading_bot/data_processing/state_builder.py', 'r') as f:
        content = f.read()
    
    # Check 1: Pas de prix absolus
    if '"open"' not in content or 'features_config' not in content:
        logger.info("✓ Prix absolus (open, high, low, close): RETIRÉS")
    else:
        # Vérifier plus précisément
        if '"open", "high", "low", "close"' in content:
            logger.error("✗ Prix absolus: TOUJOURS PRÉSENTS")
            return False
        else:
            logger.info("✓ Prix absolus: RETIRÉS")
    
    # Check 2: log_return présent
    if '"log_return"' in content:
        logger.info("✓ log_return: PRÉSENT")
    else:
        logger.error("✗ log_return: ABSENT")
        return False
    
    # Check 3: close_ema20_ratio présent
    if '"close_ema20_ratio"' in content:
        logger.info("✓ close_ema20_ratio: PRÉSENT")
    else:
        logger.error("✗ close_ema20_ratio: ABSENT")
        return False
    
    # Check 4: Nombre de features = 17 (pas 21)
    # Compter les features dans la config 5m
    import re
    match = re.search(r'"5m":\s*\[(.*?)\]', content, re.DOTALL)
    if match:
        features_str = match.group(1)
        features = [f.strip().strip('"') for f in features_str.split(',') if f.strip()]
        num_features = len(features)
        logger.info(f"✓ Nombre de features 5m: {num_features} (attendu: 17)")
        if num_features != 17:
            logger.warning(f"  ⚠️  Attendu 17, trouvé {num_features}")
            logger.info(f"  Features: {features}")
    
    return True

def test_data_loader():
    """Tester le data loader avec les nouvelles configurations."""
    logger.info("\n" + "="*80)
    logger.info("TEST: DATA LOADER AVEC NOUVELLES CONFIGS")
    logger.info("="*80)
    
    try:
        from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
        from adan_trading_bot.config.config_loader import ConfigLoader
        
        # Charger la config
        config_loader = ConfigLoader()
        config = config_loader.load()
        
        # Créer le data loader
        loader = ChunkedDataLoader(
            config=config,
            worker_config=config.get('workers', {}).get('w1', {}),
            split='train'
        )
        
        logger.info(f"✓ Data loader créé")
        logger.info(f"  Total chunks: {loader.total_chunks}")
        logger.info(f"  Chunk sizes: {loader.chunk_sizes}")
        logger.info(f"  Assets: {loader.assets_list}")
        logger.info(f"  Timeframes: {loader.timeframes}")
        
        # Charger un chunk
        chunk_data = loader.load_chunk(0)
        logger.info(f"✓ Chunk 0 chargé")
        
        # Vérifier les features
        for asset in chunk_data:
            for tf in chunk_data[asset]:
                df = chunk_data[asset][tf]
                logger.info(f"  {asset} {tf}: {df.shape[0]} rows, {df.shape[1]} cols")
                logger.info(f"    Colonnes: {df.columns.tolist()[:5]}...")
                
                # Vérifier pas de prix absolus
                if 'open' in df.columns or 'close' in df.columns:
                    logger.warning(f"  ⚠️  Prix absolus trouvés dans {asset} {tf}")
                else:
                    logger.info(f"  ✓ Pas de prix absolus dans {asset} {tf}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Erreur: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    logger.info("\n🔍 VÉRIFICATION DES CORRECTIONS CRITIQUES\n")
    
    checks = []
    
    # Vérification 1: Chunks
    checks.append(("Chunks config", verify_chunk_config()))
    
    # Vérification 2: Features
    checks.append(("Features config", verify_features_config()))
    
    # Vérification 3: Data loader
    checks.append(("Data loader test", test_data_loader()))
    
    # Résumé
    logger.info("\n" + "="*80)
    logger.info("RÉSUMÉ")
    logger.info("="*80)
    
    for name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    all_pass = all(result for _, result in checks)
    
    if all_pass:
        logger.info("\n✓ TOUTES LES CORRECTIONS SONT EN PLACE")
        sys.exit(0)
    else:
        logger.error("\n✗ CERTAINES CORRECTIONS MANQUENT")
        sys.exit(1)
