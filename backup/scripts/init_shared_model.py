#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Initialisation du modèle partagé pour l'entraînement parallèle.

Ce script crée et initialise un modèle partagé pour l'entraînement parallèle
des agents de trading ADAN. Le modèle utilise une architecture personnalisée
compatible avec l'environnement MultiAssetChunkedEnv.
"""

import os
import sys
from pathlib import Path
from typing import Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.policies import register_policy

# Ajouter le chemin du projet au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class CustomNetwork(nn.Module):
    """Réseau de neurones personnalisé pour traiter les observations.

    Ce réseau est conçu pour traiter à la fois les données de marché
    (séries temporelles) et l'état du portefeuille, puis les combiner
    pour produire une représentation unifiée.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 64
    ) -> None:  # noqa: D107
        """
        Initialise le réseau personnalisé.

        Args:
            observation_space: Espace d'observation (doit être un espace Dict)
            features_dim: Dimension de sortie des caractéristiques
        """
        super().__init__()
        self.features_dim = features_dim

        # Réseau pour les données de marché (2, 50, 15)
        self.market_net = nn.Sequential(
            nn.Conv2d(
                in_channels=2,  # Nombre d'actifs
                out_channels=32,
                kernel_size=(3, 3),
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            # (2,50,15) -> (32,25,7) après conv et maxpool
            nn.Linear(32 * 25 * 7, 256),
            nn.ReLU()
        )

        # Réseau pour l'état du portefeuille (17,)
        self.portfolio_net = nn.Sequential(
            nn.Linear(17, 64),  # 17 métriques de portefeuille
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Réseau commun pour combiner les caractéristiques
        self.combined_net = nn.Sequential(
            nn.Linear(256 + 64, 256),  # Concaténation des sorties
            nn.ReLU(),
            nn.Linear(256, features_dim)
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Passe avant du réseau.

        Args:
            observations: Dictionnaire contenant:
                - 'observation': Données de marché (batch_size, 2, 50, 15)
                - 'portfolio_state': État du portefeuille (batch_size, 17)

        Returns:
            Vecteur de caractéristiques (batch_size, features_dim)
        """
        # Extraire et traiter les données de marché
        market_data = observations["observation"]
        # Transposer pour avoir (batch, channels, height, width)
        if len(market_data.shape) == 3:  # Si pas de dimension de batch
            market_data = market_data.unsqueeze(0)
        market_features = self.market_net(market_data)

        # Extraire et traiter l'état du portefeuille
        portfolio_state = observations["portfolio_state"]
        if len(portfolio_state.shape) == 1:  # Si pas de dimension de batch
            portfolio_state = portfolio_state.unsqueeze(0)
        portfolio_features = self.portfolio_net(portfolio_state)

        # Combiner les caractéristiques
        combined = torch.cat([market_features, portfolio_features], dim=1)
        return self.combined_net(combined)


# Enregistrer la politique personnalisée
register_policy(
    "MultiInputPolicy",
    CustomNetwork,
    CustomNetwork,  # Réutiliser la même classe
    features_extractor_kwargs={"features_dim": 64}
)


def main(config_path: str, output_model_path: str) -> None:
    """
    Initialise et sauvegarde un modèle partagé pour l'entraînement distribué.

    Args:
        config_path: Chemin vers le fichier de configuration
        output_model_path: Chemin où sauvegarder le modèle initialisé
    """
    # Charger la configuration
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Créer le répertoire de sortie si nécessaire
    output_dir = os.path.dirname(output_model_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Enregistrer la politique personnalisée
    register_policy(
        "MultiInputPolicy",
        CustomNetwork,
        features_extractor_class=CustomNetwork,
        features_extractor_kwargs={"features_dim": 64}
    )

    # Créer un environnement factice pour l'initialisation
    class DummyEnv(gym.Env):
        """Environnement factice pour l'initialisation du modèle."""

        def __init__(self):
            super().__init__()
            # Espace d'observation
            self.observation_space = gym.spaces.Dict({
                "observation": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(2, 50, 15),  # (indicateurs, fenêtre, features)
                    dtype=np.float32
                ),
                "portfolio_state": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(17,),  # État du portefeuille
                    dtype=np.float32
                )
            })
            # Espace d'action
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(len(config["trading"]["symbols"]),),  # Une action/actif
                dtype=np.float32
            )

        def reset(self):
            """Réinitialise l'environnement."""
            return {
                "observation": np.zeros((2, 50, 15), dtype=np.float32),
                "portfolio_state": np.zeros(17, dtype=np.float32)
            }

        def step(self, action):
            """Exécute une étape de l'environnement."""
            return self.reset(), 0.0, False, {}

    # Créer l'environnement
    env = DummyEnv()

    # Configuration du modèle
    policy_kwargs = {
        "features_extractor_class": CustomNetwork,
        "features_extractor_kwargs": {"features_dim": 64},
        "net_arch": [{"pi": [256, 256], "vf": [256, 256]}],
        "activation_fn": nn.ReLU
    }

    # Créer le modèle
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join("logs", "shared_model"),
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
    )

    # Sauvegarder le modèle
    model.save(output_model_path)
    print(f"Modèle sauvegardé avec succès dans {output_model_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Initialisation du modèle partagé"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Chemin vers le fichier de configuration"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/shared/shared_model_init.zip",
        help="Chemin de sortie pour le modèle initialisé"
    )

    args = parser.parse_args()

    # Créer et sauvegarder le modèle partagé
    main(config_path=args.config, output_model_path=args.output)
