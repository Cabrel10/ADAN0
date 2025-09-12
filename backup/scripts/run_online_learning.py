#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour lancer l'apprentissage en ligne du modèle de trading.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
project_root = Path(__file__).parent.parent
src_dir = project_root / 'src'
sys.path.insert(0, str(src_dir))

# Vérification du chemin
if not (src_dir / 'adan_trading_bot').exists():
    raise ImportError(f"Impossible de trouver le module adan_trading_bot dans {src_dir}")

from adan_trading_bot.online_learning_agent import create_online_learning_agent, load_config
from adan_trading_bot.common.utils import get_project_root

# Obtenir le chemin racine du projet
project_root = get_project_root()


def parse_args():
    """Parse les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(description='Lancer l\'apprentissage en ligne du modèle de trading')

    # Arguments obligatoires
    parser.add_argument('--model-path', type=str, required=True,
                      help='Chemin vers le modèle pré-entraîné')

    # Arguments optionnels
    parser.add_argument('--config', type=str, default='config/online_learning_config.yaml',
                      help='Chemin vers le fichier de configuration (défaut: config/online_learning_config.yaml)')
    parser.add_argument('--env-config', type=str, default='config/environment_config.yaml',
                      help='Chemin vers le fichier de configuration de l\'environnement (défaut: config/environment_config.yaml)')
    parser.add_argument('--save-path', type=str, default=None,
                      help='Répertoire de sauvegarde des modèles (surcharge la configuration)')
    parser.add_argument('--tensorboard-log', type=str, default='logs/online',
                      help='Répertoire pour les logs TensorBoard (défaut: logs/online)')
    parser.add_argument('--steps', type=int, default=None,
                      help='Nombre d\'étapes à exécuter (défaut: illimité)')
    parser.add_argument('--verbose', type=int, default=1,
                      help='Niveau de verbosité (0: aucun, 1: info, 2: debug)')

    return parser.parse_args()


def load_full_config(env_config_path):
    """Charge et structure la configuration complète pour AdanTradingEnv."""
    from pathlib import Path

    # Charger toutes les configurations nécessaires
    config_dir = Path(project_root) / 'config'
    main_config = load_config(str(config_dir / 'main_config.yaml'))
    env_config = load_config(str(env_config_path))
    data_config = load_config(str(config_dir / 'data_config.yaml'))

    # Configuration pour data_loader
    data_pipeline_config = data_config.get('data_pipeline', {})
    feature_engineering_config = data_config.get('feature_engineering', {})

    # Valeurs par défaut si manquantes
    if 'ccxt_download' not in data_pipeline_config:
        data_pipeline_config['ccxt_download'] = {'symbol': 'BTC/USDT'}
    if 'timeframes' not in feature_engineering_config:
        feature_engineering_config['timeframes'] = ['1h']

    # Configuration complète pour l'environnement
    config = {
        **main_config,  # Configuration principale
        'data_pipeline': data_pipeline_config,
        'feature_engineering': feature_engineering_config,
        'environment': env_config,
        'trading_rules': env_config.get('trading_rules', {})
    }

    # Afficher la structure de configuration pour le débogage
    import yaml
    logger = logging.getLogger(__name__)
    logger.debug("Configuration chargée :\n%s", yaml.dump(config, default_flow_style=False))

    # Créer une configuration spéciale pour le data_loader qui combine les deux sections
    data_loader_config = {
        'data_pipeline': data_pipeline_config,
        'feature_engineering': feature_engineering_config
    }

    # Remplacer la configuration du data_pipeline par la configuration complète
    # que le data_loader attend
    config['data_pipeline'] = data_loader_config

    # S'assurer que feature_engineering est directement dans la configuration
    # pour FeatureEngineer
    config['feature_engineering'] = feature_engineering_config

    return config

def main():
    """Fonction principale."""
    # Configurer le logging
    logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Parser les arguments
    args = parse_args()

    # Charger la configuration complète
    full_config = load_full_config(args.env_config)
    online_config = load_config(args.config)

    # Surcharger le chemin de sauvegarde si spécifié
    if args.save_path is not None:
        online_config['save_path'] = args.save_path

    # Créer l'agent d'apprentissage en ligne
    agent = create_online_learning_agent(
        model_path=args.model_path,
        env_config=full_config,  # Utiliser la configuration complète
        online_config=online_config,
        tensorboard_log=args.tensorboard_log,
        verbose=args.verbose
    )

    # Lancer l'apprentissage en ligne
    agent.run(num_steps=args.steps)


if __name__ == "__main__":
    main()
