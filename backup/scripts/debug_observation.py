#!/usr/bin/env python3
"""
Script de débogage pour identifier les problèmes de construction d'observations.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Configuration du chemin d'accès aux modules
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

# Importer les modules après avoir configuré le PYTHONPATH
from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from src.adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from src.adan_trading_bot.common.config_loader import ConfigLoader

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('debug_observation.log')
    ]
)
logger = logging.getLogger(__name__)

def debug_observation():
    # Charger la configuration
    config_path = os.path.join(project_root, 'config/config.yaml')
    config_loader = ConfigLoader()
    config = config_loader.load_config(config_path)

    # Configuration du worker pour le débogage
    worker_config = {
        'id': 'debug_worker',
        'assets': ['BTC', 'ETH'],  # Utiliser les noms de dossiers sans la paire de trading
        'timeframes': ['5m', '1h'],  # Utiliser uniquement les timeframes valides de la configuration
        'data_path': config['paths']['data_dir'],  # Utiliser le chemin des données depuis la configuration
        'chunk_size': 1000,
        'window_size': 100,
        'episode_length': 100,
        'reward_config': config.get('reward', {})
    }

    # Initialiser le chargeur de données
    data_loader = ChunkedDataLoader(config, worker_config)

    # Initialiser l'environnement
    env = MultiAssetChunkedEnv(config, worker_config, data_loader_instance=data_loader)

    # Réinitialiser l'environnement
    logger.info("Réinitialisation de l'environnement...")
    observation = env.reset()

    if observation is None:
        logger.error("L'observation est None après la réinitialisation")
        return

    logger.info(f"Taille de l'observation: {observation.shape}")
    logger.info("Valeurs de l'observation:")
    logger.info(observation)

    # Tester quelques étapes
    for i in range(5):
        logger.info(f"\n=== Étape {i+1} ===")
        action = env.action_space.sample()  # Action aléatoire
        observation, reward, done, info = env.step(action)

        logger.info(f"Taille de l'observation: {observation.shape}")
        logger.info(f"Récompense: {reward}")
        logger.info(f"Terminé: {done}")
        logger.info(f"Infos: {info}")

        if done:
            logger.info("L'épisode est terminé")
            break

if __name__ == "__main__":
    try:
        debug_observation()
    except Exception as e:
        logger.exception("Erreur lors du débogage de l'observation:")
        raise
