#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dynamic Backtester pour ADAN avec intégration du Dynamic Behavior Engine (DBE).

Ce script effectue un backtest en simulant le comportement en temps réel,
y compris le chargement par chunks et l'adaptation dynamique des paramètres
via le DBE.
"""
import argparse
import copy
import logging
import os
import re
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantstats as qs
import torch
from stable_baselines3 import PPO

from src.adan_trading_bot.common.utils import load_config
from src.adan_trading_bot.data_processing.state_builder import (  # Import TimeframeConfig
    TimeframeConfig,
)
from src.adan_trading_bot.environment.multi_asset_chunked_env import (
    MultiAssetChunkedEnv,
)


# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("dynamic_backtest.log"), logging.StreamHandler()],
)
logger = logging.getLogger("DynamicBacktester")

# Assurer que le package src est dans le PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))


# Configuration des chemins
REPORTS_DIR = Path("reports/backtests")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class DynamicBacktester:
    """
    Classe pour effectuer des backtests dynamiques avec intégration du DBE.
    """

    def __init__(self, config_path: str, model_path: str):
        """
        Initialise le backtester dynamique.

        Args:
            config_path: Chemin vers le fichier de configuration
            model_path: Chemin vers le modèle entraîné
        """
        self.config = load_config(config_path)
        self.model_path = Path(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Determine if it's a single worker model or a merged model
        self.is_merged_model = (
            "merged_model" in self.model_path.name
        )  # Assuming merged model name contains "merged_model"

        self.worker_config = {}
        if not self.is_merged_model:
            # Extract worker ID from model path for single worker models
            match = re.search(r"instance_(\d+)_", self.model_path.name)
            if not match:
                raise ValueError(
                    "Le nom du modèle doit contenir 'instance_<id>_' "
                    f"(ex: instance_1_model.zip) ou 'merged_model'. Fichier reçu : {self.model_path.name}"
                )
            instance_id = int(match.group(1))
            worker_id_str = f"w{instance_id}"

            # Check if the worker exists in the configuration
            if (
                "workers" not in self.config
                or worker_id_str not in self.config["workers"]
            ):
                raise ValueError(
                    f"Worker '{worker_id_str}' non trouvé dans le fichier de configuration."
                )
            self.worker_config = self.config["workers"][worker_id_str]
            logger.info(
                f"Configuration du worker '{worker_id_str}' ({self.worker_config.get('name')}) chargée pour le backtest."
            )
        else:
            # For merged model, create a generic worker config that covers all assets and timeframes
            logger.info(
                "Backtesting a merged model. Using a generic configuration covering all assets and timeframes."
            )
            self.worker_config = {
                "assets": self.config["data"]["assets"],
                "timeframes": self.config["data"]["timeframes"],
                "name": "Merged Model",
                "data_split": "train",  # Assuming merged model is tested on train data
                "trading_mode": "spot",
                "trading_config": self.config[
                    "trading_rules"
                ],  # Use general trading rules
                "reward_config": self.config[
                    "reward_shaping"
                ],  # Use general reward shaping
                "dbe_config": self.config.get("dbe", {}),  # Use general DBE config
                "agent_config": self.config["agent"],  # Use general agent config
            }

        # Create a deep copy of the main config to modify for environment initialization
        env_config_for_env = copy.deepcopy(self.config)

        # Ensure 'environment.observation' section exists in the config passed to the environment
        if "environment" not in env_config_for_env:
            env_config_for_env["environment"] = {}
        if "observation" not in env_config_for_env["environment"]:
            env_config_for_env["environment"]["observation"] = {}

        # Set the correct window_size for the environment's observation
        env_config_for_env["environment"]["observation"][
            "window_size"
        ] = 100  # Hardcoded as per training config
        logger.info(
            f"Backtest window_size forcée à 100 pour correspondre à la configuration de l'environnement."
        )

        # Prepare features_config for StateBuilder based on worker_config's timeframes
        # This ensures StateBuilder gets the correct features for the worker's timeframes
        features_per_timeframe_for_env = {}
        for tf in self.worker_config["timeframes"]:
            if tf in self.config["data"]["features_per_timeframe"]:
                features_per_timeframe_for_env[tf] = self.config["data"][
                    "features_per_timeframe"
                ][tf]
            else:
                logger.warning(
                    f"No features defined for timeframe {tf} in main config. Using default features."
                )
                features_per_timeframe_for_env[tf] = [
                    "OPEN",
                    "HIGH",
                    "LOW",
                    "CLOSE",
                    "VOLUME",
                ]  # Fallback

        # Ensure features_per_timeframe is correctly set in the config passed to the environment
        if "data" not in env_config_for_env:
            env_config_for_env["data"] = {}
        env_config_for_env["data"][
            "features_per_timeframe"
        ] = features_per_timeframe_for_env

        from src.adan_trading_bot.environment.compat import SB3GymCompatibilityWrapper
        raw_env = MultiAssetChunkedEnv(
            config=env_config_for_env, worker_config=self.worker_config
        )
        self.env = SB3GymCompatibilityWrapper(raw_env)

        logger.info("DynamicBacktester initialisé avec succès")

    def _update_metrics(self, info: Dict[str, Any], action: np.ndarray):
        """Met à jour les métriques avec les informations du pas actuel."""
        self.metrics["portfolio_value"].append(
            info.get("portfolio", {}).get(
                "total_value", 0
            )  # Corrected path to portfolio value
        )
        self.metrics["returns"].append(
            info.get("portfolio", {}).get("returns", 0)
        )  # Corrected path to returns
        self.metrics["timestamp"].append(
            info.get("performance", {}).get("timestamp", datetime.now())
        )
        self.metrics["actions"].append(action.tolist() if action is not None else None)

        # Ajouter les métriques du DBE si disponibles
        if hasattr(self.env, "dbe") and self.env.dbe is not None:
            self.metrics["dbe_metrics"].append(
                {
                    "sl_pct": self.env.dbe.state.get("sl_pct", 0),
                    "tp_pct": self.env.dbe.state.get("tp_pct", 0),
                    "risk_mode": self.env.dbe.state.get("risk_mode", "NORMAL"),
                    "position_size": self.env.dbe.state.get("position_size", 0),
                    "market_regime": self.env.dbe.state.get("market_regime", "UNKNOWN"),
                    "volatility": self.env.dbe.state.get("volatility", 0),
                }
            )

    def _generate_report(self, output_dir: Path):
        """
        Génère un rapport de backtest complet."""
        logger.info("Génération du rapport de backtest...")

        # Créer le répertoire de sortie s'il n'existe pas
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convertir les métriques en DataFrame pour l'analyse
        df_metrics = pd.DataFrame(
            {
                "timestamp": self.metrics["timestamp"],
                "portfolio_value": self.metrics["portfolio_value"],
                "returns": self.metrics["returns"],
                "action": self.metrics["actions"],
            }
        )

        # Ajouter les métriques du DBE si disponibles
        if self.metrics["dbe_metrics"]:
            df_dbe = pd.DataFrame(self.metrics["dbe_metrics"])
            df_metrics = pd.concat([df_metrics, df_dbe], axis=1)

        # Sauvegarder les données brutes
        raw_data_path = output_dir / "backtest_metrics.csv"
        df_metrics.to_csv(raw_data_path, index=False)

        returns = pd.Series(
            df_metrics["returns"].values, index=pd.to_datetime(df_metrics["timestamp"])
        )

        # Générer le rapport HTML avec quantstats
        if (returns.index.max() - returns.index.min()).days > 0:
            self._generate_quantstats_report(df_metrics, output_dir)
        else:
            logger.warning(
                "Durée du backtest trop courte pour générer un rapport quantstats."
            )

        # Générer des graphiques supplémentaires
        self._generate_plots(df_metrics, output_dir)

        logger.info(f"Rapport de backtest généré dans {output_dir}")

    def _generate_quantstats_report(self, df_metrics: pd.DataFrame, output_dir: Path):
        """
        Génère un rapport quantstats à partir des métriques.
        """
        returns = pd.Series(
            df_metrics["returns"].values, index=pd.to_datetime(df_metrics["timestamp"])
        )

        # Générer le rapport HTML
        report_path = output_dir / "quantstats_report.html"
        qs.reports.html(
            returns,
            output=report_path,
            title="ADAN Dynamic Backtest Report",
            download_filename=report_path.name,
        )

    def _generate_plots(self, df_metrics: pd.DataFrame, output_dir: Path):
        """
        Génère des graphiques supplémentaires.
        """
        plt.figure(figsize=(15, 10))

        # Graphique de la valeur du portefeuille
        plt.subplot(2, 1, 1)
        plt.plot(df_metrics["timestamp"], df_metrics["portfolio_value"])
        plt.title("Valeur du portefeuille au fil du temps")
        plt.xlabel("Date")
        plt.ylabel("Valeur (USD)")
        plt.grid(True)

        # Graphique des paramètres du DBE
        if "sl_pct" in df_metrics.columns:
            plt.subplot(2, 1, 2)
            plt.plot(
                df_metrics["timestamp"],
                df_metrics["sl_pct"],
                label="Stop Loss %",
            )
            plt.plot(
                df_metrics["timestamp"],
                df_metrics["tp_pct"],
                label="Take Profit %",
            )
            plt.title("Évolution des paramètres de risque")
            plt.xlabel("Date")
            plt.ylabel("Valeur (%)")
            plt.legend()
            plt.grid(True)

        # Sauvegarder la figure
        plot_path = output_dir / "backtest_plots.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    def run(self, num_episodes: int = 10, render: bool = False):
        """
        Exécute le backtest dynamique.

        Args:
            num_episodes: Nombre d'épisodes à exécuter
            render: Si True, affiche la progression
        """
        logger.info(f"Démarrage du backtest dynamique sur {num_episodes} épisodes")

        # Charger le modèle ici, une seule fois
        logger.info(f"Chargement du modèle depuis {self.model_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modèle non trouvé à l'emplacement {self.model_path}"
            )

        try:
            model = PPO.load(self.model_path, device=self.device)
            logger.info("Modèle chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            raise

        for episode in range(1, num_episodes + 1):
            logger.info(f"Début de l'épisode {episode}/{num_episodes}")

            # Réinitialiser l'environnement
            obs = self.env.reset()
            done = False
            episode_reward = 0

            # Boucle sur les pas de temps
            while not done:
                # Sélectionner une action
                action, _ = model.predict(obs, deterministic=True)

                # Exécuter l'action
                next_obs, reward, terminated, truncated, info_step = self.env.step(
                    action
                )

                # Mettre à jour les métriques
                # info_step contient les métriques du portefeuille et autres
                # infos de l'environnement
                self._update_metrics(info_step, action)

                # Mettre à jour l'observation
                obs = next_obs
                episode_reward += reward
                done = terminated or truncated

                # Afficher la progression si demandé
                if render:
                    self.env.render()

            logger.info(f"Épisode {episode} terminé - Récompense: {episode_reward:.2f}")

        # Générer le rapport final
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = REPORTS_DIR / f"backtest_{timestamp}"
        self._generate_report(report_dir)

        logger.info(f"Backtest terminé. Rapport généré dans {report_dir}")
        return report_dir


def main():
    """Fonction principale pour exécuter le backtest dynamique."""
    parser = argparse.ArgumentParser(
        description="Exécute un backtest dynamique avec DBE"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Chemin vers le fichier de configuration",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Chemin vers le modèle entraîné",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Nombre d'épisodes à exécuter",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Afficher la progression du backtest",
    )

    args = parser.parse_args()

    try:
        # Initialiser le backtester
        backtester = DynamicBacktester(config_path=args.config, model_path=args.model)

        # Exécuter le backtest
        report_dir = backtester.run(num_episodes=args.episodes, render=args.render)

        # Ouvrir le rapport dans le navigateur
        report_path = report_dir / "quantstats_report.html"
        if report_path.exists():
            url = f"file://{report_path.absolute()}"
            webbrowser.open(url)

        return 0
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du backtest: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
