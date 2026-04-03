#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Point d'entrée principal pour l'agent ADAN.

Ce script peut être utilisé pour lancer l'agent en mode entraînement,
backtest, paper trading ou live trading selon la configuration.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

# Ajouter le répertoire parent au PYTHONPATH pour pouvoir importer les modules du package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from adan_trading_bot.common.custom_logger import setup_logging as setup_logger


def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="ADAN - Agent de Décision Algorithmique Neuronal")
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "backtest", "paper", "live"],
        default="train",
        help="Mode d'exécution: entraînement, backtest, paper trading ou live trading",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/main_config.yaml",
        help="Chemin vers le fichier de configuration principal",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Chemin vers un modèle pré-entraîné (pour les modes backtest, paper et live)",
    )
    
    parser.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="Niveau de verbosité (0: silencieux, 3: debug)",
    )
    
    return parser.parse_args()


def load_config(config_path):
    """Charge la configuration depuis un fichier YAML."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Fonction principale."""
    # Parsing des arguments
    args = parse_arguments()
    
    # Chargement de la configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Erreur lors du chargement de la configuration: {e}")
        sys.exit(1)
    
    # Configuration du logger
    log_levels = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    logger = setup_logger(
        config_path=os.path.join(os.path.dirname(__file__), "..", "config", "logging_config.yaml"),
        default_level=log_levels[args.verbose]
    )
    
    logger.info(f"Démarrage d'ADAN en mode {args.mode}")
    
    # Exécution selon le mode
    if args.mode == "train":
        logger.info("Mode entraînement sélectionné")
        logger.info("Utiliser scripts/train_rl_agent.py pour l'entraînement")
    
    elif args.mode == "backtest":
        if not args.model:
            logger.error("Un modèle pré-entraîné doit être spécifié pour le mode backtest")
            sys.exit(1)
        logger.info(f"Mode backtest sélectionné avec le modèle {args.model}")
        logger.info("Module de backtest à implémenter dans une version future")
    
    elif args.mode == "paper":
        if not args.model:
            logger.error("Un modèle pré-entraîné doit être spécifié pour le mode paper trading")
            sys.exit(1)
        logger.info(f"Mode paper trading sélectionné avec le modèle {args.model}")
        logger.info("Module de paper trading à implémenter dans une version future")
    
    elif args.mode == "live":
        if not args.model:
            logger.error("Un modèle pré-entraîné doit être spécifié pour le mode live trading")
            sys.exit(1)
        logger.info(f"Mode live trading sélectionné avec le modèle {args.model}")
        logger.info("Module de live trading à implémenter dans une version future")
    
    logger.info("Exécution terminée")


if __name__ == "__main__":
    main()
