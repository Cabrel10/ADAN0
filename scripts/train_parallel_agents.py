#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Script d'entraînement parallèle pour instances ADAN."""

import logging
import os
import signal
import sys
import psutil
import time
import traceback
import warnings
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed
)
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import pandas as pd
import torch
import torch as th
import torch.nn as nn
import yaml
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure as sb3_logger_configure
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from adan_trading_bot.training.callbacks import CustomTrainingInfoCallback
from stable_baselines3.common.vec_env import VecEnv
from typing import Dict, Any, Optional, Union, List, Tuple
import time
import numpy as np
from datetime import datetime, timedelta
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize
)
import contextlib

# Timeout and environment validation
from adan_trading_bot.utils.timeout_manager import TimeoutManager, TimeoutException as TMTimeoutException
from adan_trading_bot.training.trainer import validate_environment

# Configuration du logger de base
from adan_trading_bot.common.custom_logger import setup_logging

# Configurer le logger avec la configuration personnalisée
logger = setup_logging(
    default_level=logging.INFO,
    enable_console_logs=True,
    enable_json_logs=False,  # Désactiver les logs JSON par défaut
    force_plain_console=True  # Forcer un affichage simple de la console
)

# Désactiver les avertissements spécifiques
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    module='stable_baselines3'
)
warnings.filterwarnings(
    action="ignore",
    category=FutureWarning
)

class GymnasiumToGymWrapper(gym.Wrapper):
    """
    Wrapper minimal pour adapter un env Gymnasium (reset -> (obs,info), step -> (obs,rew,term,trunc,info))
    à l'API Gym attendue par certains composants (SB3 DummyVecEnv / Monitor).
    """

    def __init__(self, env, rank=0):
        """Initialise le wrapper avec un rank pour éviter les logs dupliqués."""
        super().__init__(env)
        self.rank = rank
        self.log_prefix = f"[WORKER-{rank}]"

    def reset(self, *, seed=None, options=None):
        """Garantit que reset() retourne toujours un tuple (obs, info)."""
        reset_result = super().reset(seed=seed, options=options)

        # Journalisation pour le débogage
        logger.debug(
            "GymnasiumToGymWrapper.reset - Type de sortie: %s",
            type(reset_result)
        )

        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            # Format gymnasium : (obs, info)
            obs, info = reset_result
            obs = self._validate_observation(obs)
            return obs, info
        # Cas où seul obs est retourné (ancienne API gym)
        else:
            logger.debug(
                "GymnasiumToGymWrapper.reset - Format obs unique détecté, "
                "conversion en (obs, {})"
            )
            obs = self._validate_observation(reset_result)
            return obs, {}

    def _validate_observation(self, obs):
        """Valide et convertit l'observation dans le format attendu.

        Returns:
            dict: Un dictionnaire avec les clés 'observation' (shape: 3, 20, 16)
                  et 'portfolio_state' (shape: 17,)
        """
        try:
            # Format de sortie attendu (15 caractéristiques par fenêtre)
            expected_obs_shape = (3, 20, 15)  # 15 caractéristiques par fenêtre
            expected_portfolio_shape = (17,)

            # Si c'est déjà un dictionnaire avec les bonnes clés, on le valide
            if (isinstance(obs, dict) and
                'observation' in obs and
                'portfolio_state' in obs):

                # Valider la forme de l'observation
                obs_array = np.asarray(obs['observation'], dtype=np.float32)
                portfolio_array = np.asarray(obs['portfolio_state'], dtype=np.float32)

                # Vérifier et ajuster les formes si nécessaire
                if obs_array.shape == expected_obs_shape and portfolio_array.shape == expected_portfolio_shape:
                    # Format parfait, pas besoin de modification
                    logger.debug(f"Observation validée avec succès: {obs_array.shape}")
                    return {'observation': obs_array, 'portfolio_state': portfolio_array}

                # Ajuster l'observation si nécessaire
                if len(obs_array.shape) == 3:
                    # Redimensionner vers la forme attendue
                    adjusted_obs = np.zeros(expected_obs_shape, dtype=np.float32)

                    # Copier les données en respectant les dimensions minimales
                    min_timeframes = min(obs_array.shape[0], expected_obs_shape[0])
                    min_steps = min(obs_array.shape[1], expected_obs_shape[1])
                    min_features = min(obs_array.shape[2], expected_obs_shape[2])

                    adjusted_obs[:min_timeframes, :min_steps, :min_features] = obs_array[:min_timeframes, :min_steps, :min_features]
                    obs_array = adjusted_obs
                else:
                    # Si le format est complètement différent, on crée une observation vide
                    obs_array = np.zeros(expected_obs_shape, dtype=np.float32)

                # Ajuster la forme du portefeuille de manière robuste
                if portfolio_array.shape != expected_portfolio_shape:
                    logger.debug(
                        f"Ajustement de l'état du portefeuille: {portfolio_array.shape} -> {expected_portfolio_shape}"
                    )
                    adjusted_portfolio = np.zeros(expected_portfolio_shape, dtype=np.float32)

                    if len(portfolio_array.shape) == 1:
                        # Copier les données disponibles jusqu'à la limite
                        copy_size = min(portfolio_array.size, expected_portfolio_shape[0])
                        adjusted_portfolio[:copy_size] = portfolio_array[:copy_size]
                    elif portfolio_array.size > 0:
                        # Aplatir et copier si nécessaire
                        flattened = portfolio_array.flatten()
                        copy_size = min(flattened.size, expected_portfolio_shape[0])
                        adjusted_portfolio[:copy_size] = flattened[:copy_size]

                    portfolio_array = adjusted_portfolio

                return {
                    'observation': obs_array.astype(np.float32),
                    'portfolio_state': portfolio_array.astype(np.float32)
                }

            # Si c'est un tuple (obs, info), on extrait l'observation
            if isinstance(obs, (tuple, list)) and len(obs) >= 2 and isinstance(obs[1], dict):
                return self._validate_observation(obs[0])

            # Si c'est un dictionnaire mais pas au bon format
            if isinstance(obs, dict):
                logger.warning("Format d'observation inattendu: %s", list(obs.keys()))
                # Essayer de construire une observation valide à partir des clés disponibles
                obs_array = np.asarray(obs.get('observation', np.zeros(expected_obs_shape)), dtype=np.float32)
                portfolio_array = np.asarray(obs.get('portfolio_state', np.zeros(expected_portfolio_shape)), dtype=np.float32)

                # Redimensionner si nécessaire
                if obs_array.shape != expected_obs_shape:
                    obs_array = np.zeros(expected_obs_shape, dtype=np.float32)
                if portfolio_array.shape != expected_portfolio_shape:
                    portfolio_array = np.zeros(expected_portfolio_shape, dtype=np.float32)

                return {
                    'observation': obs_array,
                    'portfolio_state': portfolio_array
                }

            # Si c'est un ndarray, essayer de le convertir au format attendu
            if isinstance(obs, np.ndarray):
                obs_array = obs.astype(np.float32)

                # Si c'est déjà la forme attendue pour l'observation
                if obs_array.shape == expected_obs_shape:
                    return {
                        'observation': obs_array,
                        'portfolio_state': np.zeros(expected_portfolio_shape, dtype=np.float32)
                    }
                # Si c'est une observation plate
                elif obs_array.size == np.prod(expected_obs_shape):
                    return {
                        'observation': obs_array.reshape(expected_obs_shape),
                        'portfolio_state': np.zeros(expected_portfolio_shape, dtype=np.float32)
                    }
                # Si c'est juste l'état du portefeuille
                elif obs_array.size == expected_portfolio_shape[0]:
                    return {
                        'observation': np.zeros(expected_obs_shape, dtype=np.float32),
                        'portfolio_state': obs_array.reshape(expected_portfolio_shape)
                    }

            # Si c'est un objet avec une méthode items(), essayer de le convertir en dict
            if hasattr(obs, 'items') and callable(obs.items):
                try:
                    obs_dict = dict(obs.items())
                    return self._validate_observation(obs_dict)
                except Exception as e:
                    logger.error("Échec de la conversion en dictionnaire: %s", str(e))

            # Si on arrive ici, on crée une observation par défaut
            # Log seulement pour le worker principal pour éviter la duplication
            if getattr(self, 'rank', 0) == 0:
                logger.warning(
                    f"{getattr(self, 'log_prefix', '[WORKER-0]')} Format d'observation non reconnu, création d'une observation par défaut. "
                    f"Type: {type(obs)}"
                )
            return {
                'observation': np.zeros(expected_obs_shape, dtype=np.float32),
                'portfolio_state': np.zeros(expected_portfolio_shape, dtype=np.float32)
            }

        except Exception as e:
            # Log seulement pour le worker principal
            if getattr(self, 'rank', 0) == 0:
                logger.error(f"{getattr(self, 'log_prefix', '[WORKER-0]')} Erreur lors de la validation de l'observation: %s", str(e))
            # En cas d'erreur, on retourne une observation vide mais valide
            return {
                'observation': np.zeros(expected_obs_shape, dtype=np.float32),
                'portfolio_state': np.zeros(expected_portfolio_shape, dtype=np.float32)
            }

    def step(self, action):
        """Convertit le retour de step() de gymnasium (5 valeurs) au format SB3."""
        out = super().step(action)

        # Journalisation pour le débogage
        logger.debug(
            "GymnasiumToGymWrapper.step - Type de sortie: %s",
            type(out)
        )
        if isinstance(out, tuple):
            logger.debug("GymnasiumToGymWrapper.step - Longueur du tuple: %d",
                        len(out))

        if isinstance(out, tuple) and len(out) == 5:
            # Format gymnasium : (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)

            # Valider le format de l'observation
            obs = self._validate_observation(obs)

            # Retourner le format attendu par SB3 avec 5 valeurs
            return obs, float(reward), done, info, {}

        # Si le format est déjà correct (4 valeurs), on ajoute un dict vide à la fin
        elif isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            obs = self._validate_observation(obs)
            return obs, float(reward), done, info, {}

        # Si le format est inattendu, essayer de le convertir
        logger.error(
            "GymnasiumToGymWrapper.step - Format de retour inattendu: %s",
            out
        )

        # Si c'est un tuple avec 3 éléments, supposer que c'est (obs, reward, done)
        if isinstance(out, tuple) and len(out) == 3:
            obs, reward, done = out
            obs = self._validate_observation(obs)
            return obs, float(reward), done, {}, {}

        # Si c'est un tuple avec 2 éléments, supposer que c'est (obs, reward)
        if isinstance(out, tuple) and len(out) == 2:
            obs, reward = out
            obs = self._validate_observation(obs)
            return obs, float(reward), False, {}, {}

        # Si c'est juste une observation, retourner avec des valeurs par défaut
        obs = self._validate_observation(out)
        return obs, 0.0, False, {}, {}

# Gestion des exceptions
class TimeoutException(Exception):
    """Exception levée quand le timeout est atteint (legacy)."""
    pass

# Import local
from adan_trading_bot.environment.checkpoint_manager import CheckpointManager

# Import local
from adan_trading_bot.utils.caching_utils import DataCacheManager

# Configuration du logger
logger = logging.getLogger(__name__)

# Définir le répertoire racine du projet
PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))
sys.path.append(PROJECT_ROOT)

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Extrait les caractéristiques à partir des observations de type Dict.
    Hérite de BaseFeaturesExtractor pour une meilleure intégration avec SB3.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,  # Dimension de sortie requise par SB3
        cnn_output_dim: int = 64,
        mlp_extractor_net_arch: Optional[List[int]] = None
    ) -> None:
        # Appel au constructeur parent avec la dimension de sortie
        super().__init__(observation_space, features_dim=features_dim)

        extractors = {}
        total_concat_size = 0

        # Pour chaque clé de l'espace d'observation
        for key, subspace in observation_space.spaces.items():
            if key == "observation":  # Traitement des données d'image
                # Calcul de la taille après aplatissement
                n_flatten = 1
                for i in range(len(subspace.shape)):
                    n_flatten *= subspace.shape[i]

                # Réseau pour traiter les données d'image
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(n_flatten, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU()
                )
                total_concat_size += 128
            else:  # Traitement des données vectorielles
                # Utilisation d'un MLP simple pour les données vectorielles
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU()
                )
                total_concat_size += 32

        self.extractors = nn.ModuleDict(extractors)

        # Couche linéaire finale pour adapter à la dimension de sortie souhaitée
        self.fc = nn.Sequential(
            nn.Linear(total_concat_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        encoded_tensor_list = []

        # Extraire les caractéristiques pour chaque clé
        for key, extractor in self.extractors.items():
            if key in observations:
                # S'assurer que les observations sont au bon format
                x = observations[key]
                if isinstance(x, np.ndarray):
                    x = th.as_tensor(x, device=self.device, dtype=th.float32)
                encoded_tensor_list.append(extractor(x))

        # Concaténer toutes les caractéristiques et appliquer la couche finale
        return self.fc(th.cat(encoded_tensor_list, dim=1))

class GymnasiumToSB3Wrapper(gym.Wrapper):
    """Wrapper pour convertir un environnement Gymnasium en un format compatible avec Stable Baselines 3."""
    def __init__(self, env: gym.Env) -> None:
        """Initialize the Gymnasium to SB3 wrapper.

        Args:
            env: The Gymnasium environment to wrap
        """
        super().__init__(env)

        # Convertir l'espace d'observation en un format compatible
        if isinstance(env.observation_space, gym.spaces.Dict):
            # Pour les espaces de type Dict, on conserve la structure
            self.observation_space = env.observation_space
        else:
            # Pour les autres types d'espaces, on essaie de les convertir
            self.observation_space = env.observation_space

        self.action_space = env.action_space
        self.metadata = getattr(env, 'metadata', {'render_modes': []})

        # Activer le mode vectorisé si nécessaire
        self.is_vector_env = hasattr(env, 'num_envs')

    def reset(self, **kwargs):
        """Reset the environment and return the initial observation and info."""
        obs, info = self.env.reset(**kwargs)

        # S'assurer que l'observation est au bon format
        if isinstance(obs, dict) and 'observation' in obs and 'portfolio_state' in obs:
            # Déjà au bon format
            return obs, info

        # Gérer les autres formats d'observation si nécessaire
        return obs, info

    def step(self, action):
        """Take an action in the environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # S'assurer que l'observation est au bon format
        if isinstance(obs, dict) and 'observation' in obs and 'portfolio_state' in obs:
            # Déjà au bon format
            return obs, reward, terminated or truncated, False, info

        # Gérer les autres formats d'observation si nécessaire
        return obs, reward, terminated or truncated, False, info

    def render(self, mode: str = "human"):
        return self.env.render(mode)


# Local application imports
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
# Import déjà effectué plus haut
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecTransposeImage, VecEnv
from stable_baselines3.common.utils import set_random_seed

def _normalize_obs_for_sb3(obs: Any) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Normalize observation for Stable Baselines 3 compatibility.

    Handles Gym's tuple (obs, info) and ensures proper numpy arrays.
    Converts observations to the format expected by SB3.

    Args:
        obs: Observation to normalize (can be tuple, dict, numpy array).
            L'observation à normaliser (peut être un tuple, un dict,
            un tableau numpy).

    Returns:
        Union[np.ndarray, Dict[str, np.ndarray]]: Normalized observation for SB3.
    """
    # Handle case where obs is a tuple (obs, info) from Gymnasium
    if isinstance(obs, tuple) and len(obs) >= 1:
        obs = obs[0]  # Take only the observation part

    # Handle dict observations (for MultiInputPolicy)
    if isinstance(obs, dict):
        # Ensure all values are numpy arrays and handle potential tuples
        normalized_obs = {}
        for k, v in obs.items():
            if isinstance(v, tuple):
                v = v[0]  # Take first element if it's a tuple
            normalized_obs[k] = np.asarray(v, dtype=np.float32)
        return normalized_obs

    # Handle numpy arrays
    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32, copy=False)

    # Handle PyTorch tensors
    if hasattr(obs, 'numpy'):
        return obs.detach().cpu().numpy()

    # Handle lists and other array-like objects
    try:
        return np.asarray(obs, dtype=np.float32)
    except Exception as e:
        error_msg = f"Could not convert observation to numpy array: {obs}, error: {e}"
        raise ValueError(error_msg) from e


class ResetObsAdapter:
    """
    Adapter minimal pour rendre un env compatible avec Stable-Baselines3 (Gym API).
    - Si env.reset() renvoie (obs, info) (Gymnasium), on renvoie obs uniquement.
    - Si env.step() renvoie 5 éléments (obs, reward, terminated, truncated, info),
      on convertit en (obs, reward, done, info) où done = terminated or truncated.
    Utiliser : env = ResetObsAdapter(env)
    """
    def __init__(self, env):
        self.env = env
        self.observation_space = getattr(env, "observation_space", None)
        self.action_space = getattr(env, "action_space", None)
        self.metadata = getattr(env, "metadata", {})
        # Propriété unwrapped pour la compatibilité avec SB3
        self.unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env

    def reset(self, **kwargs):
        # Appel à la méthode reset de l'environnement sous-jacent
        reset_result = self.env.reset(**kwargs)

        # Journalisation pour le débogage
        logger.debug(f"ResetObsAdapter.reset - Type de sortie: {type(reset_result)}")
        if isinstance(reset_result, tuple):
            logger.debug(f"ResetObsAdapter.reset - Longueur du tuple: {len(reset_result)}")

        # Gérer le cas où l'environnement retourne un tuple (obs, info)
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
            logger.debug("ResetObsAdapter.reset - Tuple (obs, info) détecté, retourne obs uniquement")

            # Vérifier que l'observation est dans le bon format
            if not isinstance(obs, (np.ndarray, dict)):
                logger.warning(f"ResetObsAdapter.reset - Type d'observation inattendu: {type(obs)}")

            return obs

        # Si ce n'est pas un tuple, retourner tel quel
        logger.debug("ResetObsAdapter.reset - Sortie simple détectée")
        return reset_result

    def step(self, action):
        out = self.env.step(action)
        # gymnasium step: (obs, reward, terminated, truncated, info)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
            return obs, reward, done, info
        return out

    def render(self, *args, **kwargs):
        return getattr(self.env, "render")(*args, **kwargs)

    def close(self):
        return getattr(self.env, "close")()

    def seed(self, seed=None):
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return None


class TrainingProgressCallback(BaseCallback):
    """
    A custom callback that prints a detailed training progress table.
    Displays training metrics, performance, portfolio value, and learning stats.
    """

    def __init__(self,
                 check_freq: int = 1000,  # Check every N time steps
                 verbose: int = 1):
        super(TrainingProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.start_time = time.time()
        self.last_time = self.start_time
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.last_log_steps = 0

    def _on_training_start(self) -> None:
        """Print the header when training starts."""
        self.start_time = time.time()
        self.last_time = self.start_time
        print("\n" + "="*100)
        print("DÉMARRAGE DE L'ENTRAÎNEMENT")
        print("="*100)
        self._print_header()

    def _on_step(self) -> bool:
        """Called at each environment step."""
        # Update counters
        self.total_steps = self.num_timesteps

        # Log every check_freq steps
        if self.n_calls % self.check_freq == 0:
            self._log_progress()

        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        # Get rollout information
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            ep_info = self.model.ep_info_buffer[0][0]  # Get first env's first episode
            if 'r' in ep_info and 'l' in ep_info:
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
                self.episode_times.append(time.time() - self.last_time)
                self.last_time = time.time()

    def _log_progress(self) -> None:
        """Log training progress with detailed metrics."""
        # Calculate metrics
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        steps_per_sec = (self.total_steps - self.last_log_steps) / (current_time - self.last_time) if current_time > self.last_time else 0

        # Get environment statistics
        if hasattr(self.model.get_env(), 'envs') and len(self.model.get_env().envs) > 0:
            env = self.model.get_env().envs[0]
            if hasattr(env, 'env'):  # Handle potential wrappers
                env = env.env

            # Get portfolio metrics if available
            portfolio_value = getattr(env, 'portfolio_value', 0)
            initial_balance = getattr(env, 'initial_balance', 1)  # Avoid division by zero
            roi = ((portfolio_value - initial_balance) / initial_balance) * 100 if initial_balance > 0 else 0
        else:
            portfolio_value = 0
            roi = 0

        # Calculate episode statistics
        avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
        avg_length = np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0
        fps = self.total_steps / elapsed_time if elapsed_time > 0 else 0

        # Get learning statistics
        learning_stats = {}
        if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            learning_stats = self.model.logger.name_to_value

        # Calculate ETA
        total_steps = self.model._total_timesteps
        progress = (self.total_steps / total_steps) * 100 if total_steps > 0 else 0
        remaining_steps = max(0, total_steps - self.total_steps)
        eta_seconds = (remaining_steps / steps_per_sec) if steps_per_sec > 0 else 0
        eta_str = str(timedelta(seconds=int(eta_seconds)))

        # Print progress table
        print("\n" + "-"*100)
        print(f"PROGRESSION: {progress:.1f}% | ETA: {eta_str} | FPS: {fps:.1f}")
        print("-"*100)

        # Print status table
        print("\nSTATUT:")
        print(f"Étape: {self.total_steps:,}/{total_steps:,} ({progress:.1f}%)")
        print(f"Temps écoulé: {str(timedelta(seconds=int(elapsed_time)))}")
        print(f"Temps restant estimé: {eta_str}")
        print(f"Vitesse: {steps_per_sec:.1f} étapes/seconde")

        # Print performance metrics
        print("\nPERFORMANCE:")
        print(f"Récompense moyenne (10 épisodes): {avg_reward:,.2f}")
        print(f"Longueur moyenne des épisodes: {avg_length:.1f} pas")
        print(f"Portefeuille: ${portfolio_value:,.2f} (ROI: {roi:+.2f}%)")

        # Print learning metrics if available
        if learning_stats:
            print("\nAPPRENTISSAGE:")
            for key, value in learning_stats.items():
                if 'loss' in key.lower() or 'entropy' in key.lower() or 'value' in key.lower():
                    print(f"{key}: {value:.4f}")

        # Print system info
        print("\nSYSTÈME:")
        try:
            import psutil
            print(f"Utilisation CPU: {psutil.cpu_percent()}%")
            try:
                process = psutil.Process()
                mem_info = process.memory_info()
                print(f"Utilisation mémoire: {mem_info.rss / 1024 / 1024:.1f} MB")
            except Exception as e:
                print(f"Erreur lors de la lecture de l'utilisation mémoire: {str(e)}")
        except ImportError:
            print("Utilisation CPU: Non disponible (psutil non installé)")
        except Exception as e:
            print(f"Erreur lors de la lecture des informations système: {str(e)}")

        # Update last log time
        self.last_log_steps = self.total_steps
        self.last_time = current_time

    def _print_header(self) -> None:
        """Print the table header."""
        print("\n" + "="*100)
        print("SUIVI DE L'ENTRAÎNEMENT - ADAN TRADING BOT")
        print("="*100)
        print("Démarrage à:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("-"*100)


class SB3GymCompatibilityWrapper(gym.Wrapper):
    """
    Wrapper pour assurer la compatibilité entre Gymnasium et Stable Baselines 3.
    Gère spécifiquement les espaces d'observation de type Dict pour les environnements de trading.
    """
    def __init__(self, env):
        super().__init__(env)

        # Enregistrer l'espace d'observation original
        self.original_obs_space = env.observation_space

        # S'assurer que l'espace d'action est correctement défini
        if not isinstance(env.action_space, (
            spaces.Discrete,
            spaces.Box,
            spaces.MultiDiscrete,
            spaces.MultiBinary
        )):
            raise ValueError(
                f"Type d'espace d'action non supporté: {type(env.action_space)}"
            )

        # Pour les environnements de trading avec espace d'observation de type Dict
        if isinstance(env.observation_space, spaces.Dict):
            # Vérifier si nous avons les clés attendues
            if 'observation' in env.observation_space.spaces and 'portfolio_state' in env.observation_space.spaces:
                # Enregistrer l'espace d'observation original
                self.observation_space = env.observation_space

                # Extraire les espaces pour le traitement
                obs_space = env.observation_space.spaces['observation']
                portfolio_space = env.observation_space.spaces['portfolio_state']

                # Vérifier les dimensions
                expected_obs_shape = (3, 20, 15)  # Forme attendue: (timeframes, window_size, features)
                expected_portfolio_shape = (17,)   # Forme attendue pour le portefeuille

                # Vérifier le type et la forme de l'observation
                if not (isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 3):
                    raise ValueError(
                        f"Format d'observation non supporté. Attendu: Box 3D, obtenu: {type(obs_space)} avec forme {getattr(obs_space, 'shape', 'N/A')}"
                    )

                # Vérifier le type et la forme du portefeuille
                if not (isinstance(portfolio_space, spaces.Box) and len(portfolio_space.shape) == 1):
                    raise ValueError(
                        f"Format d'état du portefeuille non supporté. Attendu: Box 1D, obtenu: {type(portfolio_space)} avec forme {getattr(portfolio_space, 'shape', 'N/A')}"
                    )

                # Avertissement si les formes ne correspondent pas exactement à ce qui est attendu
                if obs_space.shape != expected_obs_shape:
                    logger.warning(
                        f"Forme d'observation non standard: {obs_space.shape}, attendu {expected_obs_shape}. "
                        "Le modèle peut nécessiter un ajustement."
                    )

                if portfolio_space.shape != expected_portfolio_shape:
                    logger.warning(
                        f"Forme de l'état du portefeuille non standard: {portfolio_space.shape}, "
                        f"attendu {expected_portfolio_shape}. Le modèle peut nécessiter un ajustement."
                    )

                # Enregistrer les dimensions pour le traitement des observations
                self.obs_shape = obs_space.shape
                self.portfolio_dim = portfolio_space.shape[0]

                logger.info(
                    "SB3GymCompatibilityWrapper: Espace d'observation "
                    "configuré avec succès. Forme de l'observation: %s, "
                    "Dimension du portefeuille: %s",
                    obs_space.shape,
                    portfolio_space.shape
                )
            else:
                raise ValueError("L'espace d'observation Dict doit contenir les clés 'observation' et 'portfolio_state'")
        else:
            # Pour les autres types d'espaces, utiliser l'espace d'observation tel quel
            self.observation_space = env.observation_space

    def reset(self, **kwargs):
        try:
            # Appeler reset sur l'environnement sous-jacent
            reset_result = self.env.reset(**kwargs)

            # Journalisation pour le débogage
            logger.debug(f"SB3GymCompatibilityWrapper.reset - Type de sortie: {type(reset_result)}")
            if isinstance(reset_result, tuple):
                logger.debug(f"SB3GymCompatibilityWrapper.reset - Longueur du tuple: {len(reset_result)}")

            # Extraire l'observation du résultat de reset
            if isinstance(reset_result, tuple):
                if len(reset_result) == 2:
                    obs, info = reset_result  # Format (obs, info) de Gymnasium
                else:
                    # Si le tuple a une longueur différente, prendre le premier élément comme observation
                    obs = reset_result[0] if len(reset_result) > 0 else {}
                    info = reset_result[1] if len(reset_result) > 1 else {}
            else:
                obs = reset_result
                info = {}

            # Journalisation supplémentaire
            logger.debug(f"Type d'observation après extraction: {type(obs)}")
            if hasattr(obs, 'shape'):
                logger.debug(f"Forme de l'observation: {obs.shape}")
            elif isinstance(obs, dict):
                logger.debug(f"Clés de l'observation: {list(obs.keys())}")

            # Traiter l'observation pour la compatibilité avec SB3
            processed_obs = self._process_obs(obs)

            # Journalisation pour le débogage
            logger.debug(f"SB3GymCompatibilityWrapper.reset - Type d'observation traité: {type(processed_obs)}")
            if hasattr(processed_obs, 'shape'):
                logger.debug(f"SB3GymCompatibilityWrapper.reset - Forme de l'observation: {processed_obs.shape}")

            # Pour la compatibilité avec SB3, retourner un tuple (observation, info)
            return processed_obs, info

        except Exception as e:
            logger.error(f"Erreur dans SB3GymCompatibilityWrapper.reset: {e}")
            logger.error(traceback.format_exc())
            # Retourner une observation par défaut en cas d'erreur
            default_shape = (3 * 20 * 15) + 17
            return np.zeros(default_shape, dtype=np.float32), {}

    def step(self, action):
        try:
            # Appeler step sur l'environnement sous-jacent
            step_result = self.env.step(action)

            # Gérer les différents formats de retour
            if isinstance(step_result, tuple):
                if len(step_result) == 5:
                    # Format gymnasium : (obs, reward, terminated, truncated, info)
                    obs, reward, terminated, truncated, info = step_result
                    done = bool(terminated or truncated)
                elif len(step_result) == 4:
                    # Format gym ancien : (obs, reward, done, info)
                    obs, reward, done, info = step_result
                else:
                    raise ValueError(f"Format de retour de step() non supporté: {step_result}")
            else:
                raise ValueError(f"Le résultat de step() doit être un tuple, reçu: {type(step_result)}")

            # Traiter l'observation pour la compatibilité avec SB3
            processed_obs = self._process_obs(obs)

            # Journalisation pour le débogage
            logger.debug(
                f"Step - Type d'observation: {type(obs)}, "
                f"Type traité: {type(processed_obs)}, "
                f"Forme: {getattr(processed_obs, 'shape', 'N/A')}, "
                f"Récompense: {reward:.4f}, "
                f"Terminé: {done}"
            )

            # Retourner le format attendu par SB3 (4 valeurs)
            return processed_obs, float(reward), done, info

        except Exception as e:
            logger.error(f"Erreur dans SB3GymCompatibilityWrapper.step: {e}")
            logger.error(traceback.format_exc())
            # Retourner une observation par défaut en cas d'erreur
            default_shape = (3 * 20 * 15) + 17
            return np.zeros(default_shape, dtype=np.float32), 0.0, True, {}

    def _process_obs(self, obs):
        """
        Traite l'observation pour la compatibilité avec SB3.
        Convertit un dictionnaire d'observations en un tableau NumPy 1D.

        Gère spécifiquement les observations avec la forme (3, 20, 15) pour les données de marché
        et (17,) pour l'état du portefeuille.
        """
        # Définir les formes attendues
        expected_market_shape = (3, 20, 15)  # Forme attendue pour les données de marché
        expected_portfolio_shape = (17,)     # Forme attendue pour l'état du portefeuille

        # Fonction utilitaire pour créer une observation par défaut
        def create_default_observation():
            market_obs = np.zeros(expected_market_shape, dtype=np.float32)
            portfolio_obs = np.zeros(expected_portfolio_shape, dtype=np.float32)
            return np.concatenate([market_obs.reshape(-1), portfolio_obs.reshape(-1)])

        try:
            # Si l'observation est déjà un tableau numpy, la retourner telle quelle
            if isinstance(obs, np.ndarray):
                return obs.astype(np.float32)

            # Si c'est un dictionnaire avec les clés attendues
            if isinstance(obs, dict) and 'observation' in obs and 'portfolio_state' in obs:
                # Valider le type et la forme des données d'observation
                if not isinstance(obs['observation'], (np.ndarray, list, tuple)) or \
                   not isinstance(obs['portfolio_state'], (np.ndarray, list, tuple)):
                    logger.warning("Type d'observation invalide. Utilisation d'une observation par défaut.")
                    return create_default_observation()

                # Convertir les observations en tableaux NumPy
                try:
                    market_obs = np.array(obs['observation'], dtype=np.float32)
                    portfolio_obs = np.array(obs['portfolio_state'], dtype=np.float32)
                except Exception as e:
                    logger.error(f"Erreur lors de la conversion des observations: {e}")
                    return create_default_observation()

                # Valider et redimensionner les données de marché si nécessaire
                if market_obs.shape != expected_market_shape:
                    logger.warning(
                        f"Forme des données de marché non standard: {market_obs.shape}, "
                        f"attendu {expected_market_shape}. Redimensionnement en cours."
                    )
                    if len(market_obs.shape) == 3:
                        market_obs = market_obs[:3, :20, :15]  # Tronquer aux dimensions attendues
                        if market_obs.shape[1] < 20:  # Si la dimension temporelle est trop petite
                            pad_width = [(0, 0), (0, 20 - market_obs.shape[1]), (0, 0)]
                            market_obs = np.pad(market_obs, pad_width, mode='constant')
                    else:
                        market_obs = np.zeros(expected_market_shape, dtype=np.float32)

                # Valider et redimensionner l'état du portefeuille si nécessaire
                if portfolio_obs.shape != expected_portfolio_shape:
                    logger.warning(
                        f"Forme de l'état du portefeuille non standard: {portfolio_obs.shape}, "
                        f"attendu {expected_portfolio_shape}. Redimensionnement en cours."
                    )
                    if len(portfolio_obs.shape) == 1 and portfolio_obs.size >= 17:
                        portfolio_obs = portfolio_obs[:17]  # Tronquer si trop grand
                    else:
                        portfolio_obs = np.zeros(expected_portfolio_shape, dtype=np.float32)

                # Vérifier la forme finale avant de concaténer
                expected_market_size = np.prod(expected_market_shape)
                if market_obs.size != expected_market_size:
                    logger.warning(
                        f"Taille des données de marché incorrecte: {market_obs.size}, "
                        f"attendu {expected_market_size}. Redimensionnement en cours."
                    )
                    market_obs = market_obs.reshape(-1)[:expected_market_size]
                    if market_obs.size < expected_market_size:
                        market_obs = np.pad(
                            market_obs,
                            (0, expected_market_size - market_obs.size),
                            mode='constant'
                        )
                    market_obs = market_obs.reshape(expected_market_shape)

                # Aplatir et concaténer les observations
                market_flat = market_obs.reshape(-1)
                portfolio_flat = portfolio_obs.reshape(-1)

                try:
                    processed_obs = np.concatenate([market_flat, portfolio_flat])

                    logger.debug(
                        f"Observation traitée - Marché: {market_obs.shape} -> {market_flat.shape}, "
                        f"Portefeuille: {portfolio_obs.shape} -> {portfolio_flat.shape}, "
                        f"Sortie: {processed_obs.shape}"
                    )
                except Exception as e:
                    logger.error(f"Erreur lors de la concaténation des observations: {e}")
                    return create_default_observation()

                return processed_obs.astype(np.float32)

            # Si c'est un tuple (obs, info) de Gymnasium, extraire l'observation
            elif isinstance(obs, tuple) and len(obs) >= 2:
                logger.debug("Tuple d'observation détecté, extraction de l'élément d'observation")
                return self._process_obs(obs[0])  # Traiter uniquement l'observation

            # Si c'est une séquence (liste, tuple, etc.), essayer de la convertir en tableau
            elif isinstance(obs, (list, tuple)):
                logger.debug(f"Séquence d'observation détectée, conversion en tableau: {type(obs)}")
                try:
                    arr = np.array(obs, dtype=np.float32)
                    expected_size = np.prod(expected_market_shape) + len(expected_portfolio_shape)

                    if arr.size == expected_size:
                        return arr.reshape(-1).astype(np.float32)
                    elif arr.size > expected_size:
                        logger.warning(f"Troncature de l'observation de taille {arr.size} à {expected_size}")
                        return arr.flat[:expected_size].astype(np.float32)
                    else:
                        logger.warning(f"Remplissage de l'observation de taille {arr.size} à {expected_size}")
                        return np.pad(arr.reshape(-1),
                           (0, expected_size - arr.size),
                           mode='constant').astype(np.float32)
                except Exception as e:
                    logger.error(f"Erreur lors de la conversion de la séquence: {e}")
                    raise

            # Pour les autres types, essayer une conversion directe
            logger.warning(f"Type d'observation non standard, tentative de conversion: {type(obs)}")
            try:
                arr = np.array(obs, dtype=np.float32)
                expected_size = np.prod(expected_market_shape) + len(expected_portfolio_shape)

                if arr.size == expected_size:
                    return arr.reshape(-1).astype(np.float32)
                elif arr.size > expected_size:
                    logger.warning(f"Troncature de l'observation de taille {arr.size} à {expected_size}")
                    return arr.flat[:expected_size].astype(np.float32)
                else:
                    logger.warning(f"Remplissage de l'observation de taille {arr.size} à {expected_size}")
                    return np.pad(arr.reshape(-1),
                           (0, expected_size - arr.size),
                           mode='constant').astype(np.float32)
            except Exception as e:
                logger.error(f"Échec de la conversion de l'observation: {e}")
                return create_default_observation()

        except Exception as e:
            logger.error(f"Erreur critique lors du traitement de l'observation: {e}")
            logger.error(f"Type d'observation: {type(obs).__name__}")

            # Journalisation détaillée pour le débogage
            if hasattr(obs, 'shape'):
                logger.error(f"Forme de l'observation: {obs.shape}")
            elif hasattr(obs, '__len__'):
                logger.error(f"Longueur de l'observation: {len(obs)}")

            if isinstance(obs, dict):
                logger.error("Clés de l'observation:")
                for k, v in obs.items():
                    type_info = type(v).__name__
                    shape_info = f", shape={v.shape}" if hasattr(v, 'shape') else ""
                    len_info = f", len={len(v)}" if hasattr(v, '__len__') else ""
                    logger.error(f"  {k}: type={type_info}{shape_info}{len_info}")

            logger.error("Traceback complet de l'erreur:", exc_info=True)

        # En cas d'erreur, retourner un tableau de zéros de la bonne dimension
        default_market = np.zeros(expected_market_shape, dtype=np.float32)
        default_portfolio = np.zeros(expected_portfolio_shape, dtype=np.float32)
        default_obs = np.concatenate([default_market.reshape(-1), default_portfolio])

        logger.warning(
            f"Retour d'une observation par défaut de forme {default_obs.shape} "
            f"(marché: {default_market.shape}, portefeuille: {default_portfolio.shape})"
        )

        return default_obs


def make_env(rank: int = 0, seed: Optional[int] = None, config: Dict = None, worker_config: Dict = None) -> gym.Env:
    """
    Crée et configure un environnement pour un worker donné.

    Args:
        rank: Identifiant unique du worker
        seed: Graine pour la reproductibilité
        config: Configuration de l'environnement
        worker_config: Configuration spécifique au worker

    Returns:
        Un environnement Gym valide
    """
    # Configuration par défaut si config n'est pas fourni
    if config is None:
        config = {}

    # Configuration du worker si non fournie
    if worker_config is None:
        # Récupérer la liste des actifs depuis la configuration
        assets = [a.lower() for a in config.get("data", {}).get("assets", ["btcusdt"])]
        timeframes = [str(tf).lower() for tf in config.get("data", {}).get("timeframes", ["5m", "1h", "4h"])]
        data_split = config.get("data", {}).get("data_split", "val").lower()

        worker_config = {
            "rank": rank,
            "worker_id": f"W{rank}",  # ID unique pour logs synchronisés
            "num_workers": config.get("num_workers", 1),
            "assets": assets,
            "timeframes": timeframes,
            "data_split": data_split,
            "data_loader": {
                "batch_size": config.get("batch_size", 32),
                "shuffle": True,
                "num_workers": 0  # Désactiver le multithreading pour éviter les problèmes
            }
        }
        logger.info(f"[WORKER-{rank}] Configuration worker créée: assets={assets}, timeframes={timeframes}, data_split={data_split}")

    # Extraire les paramètres nécessaires de la configuration
    data_config = config.get("data", {})
    env_config = config.get("environment", {})

    # Récupérer les paramètres du worker
    assets = [a.lower() for a in worker_config.get("assets", [])]
    timeframes = [str(tf).lower() for tf in worker_config.get("timeframes", [])]

    # Récupérer le data_split de la configuration du worker, avec une valeur par défaut
    data_split = worker_config.get("data_split", "val").lower()

    # Dossier de base des données (utiliser data_dirs[data_split] s'il existe, sinon data_dirs.base)
    data_dirs = config.get("data", {}).get("data_dirs", {})

    # Utiliser le dossier spécifique au split s'il existe, sinon utiliser le dossier de base
    if data_split in data_dirs:
        data_dir = Path(data_dirs[data_split])
    elif "base" in data_dirs:
        data_dir = Path(data_dirs["base"]) / data_split
    else:
        data_dir = Path("data/processed/indicators") / data_split

    logger.info(f"Configuration data_dirs: {data_dirs}")
    logger.info(f"Data split utilisé: {data_split}")
    logger.info(f"Dossier de données final: {data_dir}")

    # Vérifier que le répertoire existe
    if not data_dir.exists():
        raise FileNotFoundError(f"Le répertoire de données {data_dir} n'existe pas")

    # Vérifier que le répertoire existe
    if not data_dir.exists():
        raise FileNotFoundError(f"Le répertoire de données {data_dir} n'existe pas")

    logger.info(f"Chargement des données depuis : {data_dir}")
    logger.info(f"Actifs: {assets}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Split de données: {data_split}")

    # Dictionnaire pour stocker les données chargées
    data = {}
    data_found = False

    # Charger les données pour chaque actif et chaque timeframe
    for asset in assets:
        # Nettoyer le nom de l'actif (supprimer / et -) et forcer en minuscules
        clean_asset = asset.replace("/", "").replace("-", "").lower()
        data[clean_asset.upper()] = {}
        asset_data_found = False

        for tf in timeframes:
            # Construire le chemin du fichier dans le format: {split}/{asset}/{timeframe}.parquet
            file_path = data_dir / clean_asset / f"{tf}.parquet"

            # Vérifier si le fichier existe
            if not file_path.exists():
                logger.warning(f"Fichier non trouvé: {file_path}")
                continue

            try:
                # Charger les données Parquet
                df = pd.read_parquet(file_path)

                # Vérifier que le DataFrame n'est pas vide
                if df.empty:
                    logger.warning(f"Le fichier {file_path} est vide.")
                    continue

                # Stocker les données dans la structure attendue
                data[clean_asset.upper()][tf] = df
                logger.info(f"Données chargées: {clean_asset.upper()}/{tf} - {len(df)} lignes")
                data_found = True
                asset_data_found = True

            except Exception as e:
                logger.error(f"Erreur lors du chargement de {file_path}: {str(e)}")

        if not asset_data_found:
            logger.warning(f"Aucune donnée valide trouvée pour l'actif {clean_asset.upper()}")
            del data[clean_asset.upper()]

    if not data:
        raise ValueError(
            "Aucune donnée valide n'a pu être chargée. "
            f"Vérifiez les chemins dans la configuration et assurez-vous que les fichiers existent dans {base_data_dir}."
        )

    # Définir la taille de la fenêtre et la configuration des caractéristiques
    window_size = config.get("environment", {}).get("window_size", 50)
    features_config = config.get("environment", {}).get("features_config", {})

    logger.info(f"Taille de la fenêtre: {window_size}")
    logger.info(f"Configuration des caractéristiques: {features_config}")

    # Créer l'environnement avec les données chargées
    env = MultiAssetChunkedEnv(
        data=data,
        timeframes=timeframes,
        window_size=window_size,
        features_config=features_config,
        max_steps=env_config.get("max_steps", 1000),
        initial_balance=env_config.get("initial_balance", 10000.0),
        commission=env_config.get("commission", 0.001),
        reward_scaling=env_config.get("reward_scaling", 1.0),
        render_mode=None,
        enable_logging=config.get("enable_logging", True),
        log_dir=config.get("log_dir", "logs"),
        worker_config=worker_config,
        config=config
    )

    # Appliquer le wrapper pour la compatibilité et éviter les logs dupliqués
    env = GymnasiumToGymWrapper(env, rank=rank)

    # Configurer la graine pour la reproductibilité
    if seed is not None:
        try:
            # Essayer d'utiliser np.random pour la graine
            import numpy as np
            np.random.seed(seed)

            # Configurer la graine de l'environnement si possible
            base = getattr(env, 'unwrapped', env)
            if hasattr(base, 'seed'):
                base.seed(seed)

            # Configurer la graine de l'espace d'action
            if hasattr(env, 'action_space') and hasattr(env.action_space, 'seed'):
                env.action_space.seed(seed)

            # Configurer la graine de l'espace d'observation
            if hasattr(env, 'observation_space') and hasattr(env.observation_space, 'seed'):
                env.observation_space.seed(seed)

        except Exception as e:
            logger.warning(f"Impossible de configurer la graine pour l'environnement {rank}: {str(e)}")

    logger.info(f"Environnement {rank} créé avec succès")
    return env

def main(
    config_path: str = "bot/config/config.yaml",
    timeout: Optional[int] = None,
    checkpoint_dir: str = "checkpoints",
    shared_model_path: Optional[str] = None,
    resume: bool = False,
    num_envs: int = 4,
    use_subproc: bool = False,
) -> bool:
    """
    Fonction principale pour l'entraînement parallèle des agents ADAN.

    Args:
        config_path: Chemin vers le fichier de configuration YAML
        timeout: Délai maximum d'entraînement en secondes
        checkpoint_dir: Répertoire pour enregistrer les points de contrôle
        shared_model_path: Chemin vers un modèle partagé pour l'entraînement distribué
        resume: Reprendre l'entraînement à partir du dernier point de contrôle
        num_envs: Nombre d'environnements parallèles à exécuter
        use_subproc: Si True, utilise SubprocVecEnv au lieu de DummyVecEnv

    Returns:
        bool: True si l'entraînement s'est terminé avec succès, False sinon
    """
    try:
        # Validate environment (Python version, deps, etc.)
        try:
            validate_environment()
        except Exception as e:
            logger.error("Environment validation failed: %s", e)
            raise
        # Charger la configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Configuration chargée depuis {config_path}")
        logger.info(f"Démarrage de l'entraînement avec un timeout de {timeout} secondes")
        # Activer ou non la barre de progression pendant l'entraînement (config training.progress_bar)
        progress_bar = bool(config.get('training', {}).get('progress_bar', False))

        # Récupérer la liste des actifs et timeframes depuis la configuration
        data_config = config.get("data", {})
        file_structure = data_config.get("file_structure", {})
        assets = file_structure.get("assets", ["BTCUSDT"])
        timeframes = file_structure.get("timeframes", ["5m", "1h", "4h"])
        seed = config.get('seed', 42)

        # Configuration commune pour tous les workers
        base_worker_config = {
            "num_workers": num_envs,
            "assets": assets,  # Liste des actifs
            "timeframes": timeframes,  # Liste des timeframes
            "chunk_sizes": {tf: 1000 for tf in timeframes}, # Chunk size augmentée pour le warm-up
            "data_loader": {
                "batch_size": config.get("batch_size", 32),
                "shuffle": True,
                "num_workers": 0  # Important: laisser à 0 pour éviter les problèmes de fork
            },
            "use_subproc": use_subproc,  # Passer l'info au worker
            "enable_worker_id_logging": True  # Activer les IDs workers pour réduire répétitions
        }

        # Choisir la classe d'environnement vectorisé
        VecEnvClass = SubprocVecEnv if use_subproc else DummyVecEnv

        # Configurer les arguments pour SubprocVecEnv
        vec_env_kwargs = {}
        if use_subproc:
            # Configuration spécifique pour le mode multiprocessus
            vec_env_kwargs.update({
                'start_method': 'forkserver',  # Meilleur que 'spawn' pour les performances
                'daemon': False  # Permet aux processus enfants de se terminer correctement
            })
            logger.info(f"Utilisation de SubprocVecEnv avec {num_envs} processus parallèles")
        else:
            logger.info(f"Utilisation de DummyVecEnv (sans parallélisme réel)")

        # Créer les fonctions de création d'environnement
        def make_env_fn(rank: int, seed_val: int) -> Callable[[], gym.Env]:
            """Crée une fonction d'initialisation d'environnement pour un rang donné."""
            def _init() -> gym.Env:
                try:
                    # Configuration spécifique au worker pour éviter les logs dupliqués
                    worker_config = {
                        "rank": rank,
                        "worker_id": f"w{rank}",
                        "log_prefix": f"[WORKER-{rank}]",
                        **base_worker_config
                    }

                    env = make_env(
                        rank=rank,
                        seed=seed_val,
                        config=config,
                        worker_config=worker_config
                    )

                    # Log seulement pour le worker principal pour éviter la duplication
                    if rank == 0 or not use_subproc:
                        logger.info(f"[WORKER-{rank}] Environnement {rank} initialisé avec succès")
                    return env
                except Exception as e:
                    logger.error(f"[WORKER-{rank}] Erreur lors de la création de l'environnement {rank}: {str(e)}")
                    raise
            return _init

        # Créer les environnements avec des seeds uniques
        env_fns = []
        for i in range(num_envs):
            # Chaque environnement a une seed unique basée sur la seed de base + son rang
            env_seed = seed + i * 1000 if seed is not None else None
            env_fns.append(make_env_fn(i, env_seed))

        # Créer l'environnement vectorisé avec la configuration appropriée
        try:
            env = VecEnvClass(env_fns, **vec_env_kwargs)
            logger.info(f"Environnement vectorisé créé avec succès: {env}")
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'environnement vectorisé: {str(e)}")
            raise

        # Ajouter la normalisation des observations si nécessaire
        if config.get('normalize_observations', True):
            env = VecNormalize(env, norm_obs=True, norm_reward=True)

        # Le bloc de validation de l'environnement a été supprimé car il était défectueux
        # et incompatible avec la nouvelle structure d'environnement simplifiée.

        logger.info(f"Environnement vectorisé créé avec {num_envs} environnements")

        # Configuration optimisée de l'agent PPO avec MultiInputPolicy pour gérer les espaces d'observation de type dictionnaire

        # Vérifier l'espace d'observation
        logger.info(f"Espace d'observation de l'environnement: {env.observation_space}")

        # Configuration du réseau de neurones
        policy_kwargs = {
            'net_arch': [dict(pi=[64, 64], vf=[64, 64])],
            'activation_fn': torch.nn.ReLU,
            'ortho_init': True
        }

        # Créer ou charger le modèle PPO
        model = None
        latest_checkpoint = None

        # Afficher des informations sur le parallélisme
        logger.info("\n" + "="*80)
        logger.info(f"Configuration de l'entraînement:")
        logger.info(f"- Nombre d'environnements parallèles: {num_envs}")
        logger.info(f"- Type d'environnement: {'SubprocVecEnv' if use_subproc else 'DummyVecEnv'}")
        logger.info(f"- Seed de base: {config.get('seed', 42)}")
        logger.info(f"- Device: {'auto'}")
        logger.info("="*80 + "\n")

        # Vérifier si on doit reprendre depuis un checkpoint
        if resume and checkpoint_manager.list_checkpoints():
            latest_checkpoint = checkpoint_manager.list_checkpoints()[-1]
            logger.info(f"Reprise de l'entraînement depuis le checkpoint: {latest_checkpoint}")

            # Créer un modèle minimal pour le chargement
            model = PPO(
                policy="MultiInputPolicy",
                env=env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                seed=config.get('seed', 42),
                device='auto'
            )
            # Configure SB3 logger for resume case
            try:
                sb3_log_dir = Path("bot/config/logs/sb3")
                sb3_log_dir.mkdir(parents=True, exist_ok=True)
                new_logger = sb3_logger_configure(str(sb3_log_dir), ["stdout", "csv", "tensorboard"])
                model.set_logger(new_logger)
                logger.info(f"SB3 logger configured at {sb3_log_dir}")
            except Exception as e:
                logger.warning(f"Failed to configure SB3 logger: {e}")

            # Charger le checkpoint
            model, _, metadata = checkpoint_manager.load_checkpoint(
                checkpoint_path=latest_checkpoint,
                model=model,
                map_location='auto'
            )

            if metadata:
                logger.info(
                    "Checkpoint chargé - Épisode: %d, Steps: %d",
                    metadata.episode, metadata.total_steps
                )
                start_timesteps = metadata.total_steps
            else:
                logger.warning("Métadonnées du checkpoint non trouvées, démarrage à zéro")
                start_timesteps = 0

        # Si pas de reprise ou échec de chargement, créer un nouveau modèle
        if model is None:
            logger.info("Création d'un nouveau modèle")
            model = PPO(
                policy="MultiInputPolicy",
                env=env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=policy_kwargs,
                verbose=1,
                seed=config.get('seed', 42),
                device='auto'
            )
            # Configure SB3 logger for fresh model
            try:
                sb3_log_dir = Path("bot/config/logs/sb3")
                sb3_log_dir.mkdir(parents=True, exist_ok=True)
                new_logger = sb3_logger_configure(str(sb3_log_dir), ["stdout", "csv", "tensorboard"])
                model.set_logger(new_logger)
                logger.info(f"SB3 logger configured at {sb3_log_dir}")
            except Exception as e:
                logger.warning(f"Failed to configure SB3 logger: {e}")
            start_timesteps = 0

        # Créer le répertoire de checkpoints si nécessaire
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialiser le CheckpointManager
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=5,  # Garder les 5 derniers checkpoints
            checkpoint_interval=10000,  # Sauvegarder tous les 10 000 steps
            logger=logger
        )

        # Vérifier si on doit reprendre depuis un checkpoint
        if resume:
            try:
                checkpoints = checkpoint_manager.list_checkpoints()
                if not checkpoints:
                    logger.info("[TRAINING] Aucun checkpoint trouvé, démarrage d'un nouvel entraînement")
                    start_timesteps = 0
                else:
                    latest_checkpoint = checkpoints[-1]
                    logger.info(f"[TRAINING] Tentative de reprise depuis le checkpoint: {latest_checkpoint}")

                    # Vérifier si le checkpoint existe toujours
                    if not os.path.exists(latest_checkpoint):
                        logger.warning(f"[TRAINING] Le checkpoint {latest_checkpoint} n'existe plus")
                        start_timesteps = 0
                    else:
                        # Créer un modèle minimal pour le chargement
                        model = PPO(
                            policy="MultiInputPolicy",
                            env=env,
                            policy_kwargs=policy_kwargs,
                            verbose=1,
                            seed=config.get('seed', 42),
                            device='auto'
                        )

                        try:
                            # Charger le checkpoint
                            model, optimizer, metadata = checkpoint_manager.load_checkpoint(
                                checkpoint_path=latest_checkpoint,
                                model=model,
                                map_location='auto'
                            )

                            if metadata is not None:
                                logger.info(
                                    "[TRAINING] Checkpoint chargé - Épisode: %d, Steps: %d",
                                    metadata.episode, metadata.total_steps
                                )
                                start_timesteps = metadata.total_steps

                                # Vérifier la cohérence des métadonnées
                                if not hasattr(metadata, 'total_steps') or not isinstance(metadata.total_steps, int):
                                    logger.warning("[TRAINING] Métadonnées de checkpoint invalides, réinitialisation du compteur d'étapes")
                                    start_timesteps = 0
                            else:
                                logger.warning("[TRAINING] Aucune métadonnée trouvée dans le checkpoint")
                                start_timesteps = 0

                        except Exception as e:
                            logger.error(f"[TRAINING] Erreur lors du chargement du checkpoint: {str(e)}", exc_info=True)
                            logger.warning("[TRAINING] Démarrage d'un nouvel entraînement")
                            start_timesteps = 0

            except Exception as e:
                logger.error(f"[TRAINING] Erreur lors de la vérification des checkpoints: {str(e)}", exc_info=True)
                logger.warning("[TRAINING] Démarrage d'un nouvel entraînement")
                start_timesteps = 0
        else:
            logger.info("[TRAINING] Démarrage d'un nouvel entraînement (sans reprise)")
            start_timesteps = 0

        class CustomCheckpointCallback(BaseCallback):
            """
            Callback personnalisé pour la sauvegarde des checkpoints.
            """

            def __init__(
                self,
                checkpoint_manager: CheckpointManager,
                verbose: int = 0
            ):
                """
                Initialise le callback de sauvegarde.

                Args:
                    checkpoint_manager: Gestionnaire de checkpoints
                    verbose: Niveau de verbosité
                """
                super().__init__(verbose)
                self.checkpoint_manager = checkpoint_manager
                self.last_save = 0
                self.last_checkpoint = None
                self.episode_rewards = []
                self._last_checkpoint_step = 0  # Pour suivre la dernière étape de sauvegarde

            def _on_step(self) -> bool:
                """
                Appelé à chaque étape d'entraînement pour gérer la sauvegarde des checkpoints.

                Returns:
                    bool: True pour continuer l'entraînement, False pour l'arrêter
                """
                # Vérifier si c'est le moment de sauvegarder un checkpoint
                if (self.num_timesteps - self._last_checkpoint_step) >= self.checkpoint_manager.checkpoint_interval:
                    self._save_checkpoint()
                    self._last_checkpoint_step = self.num_timesteps
                return True

            def _save_checkpoint(self):
                """
                Sauvegarde un checkpoint du modèle.
                """
                try:
                    # Récupérer les métriques actuelles
                    metrics = {}
                    if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                        # Calculer les statistiques de récompense sur les derniers épisodes
                        rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer if 'r' in ep_info]
                        if rewards:
                            metrics['mean_reward'] = float(np.mean(rewards))
                            metrics['min_reward'] = float(np.min(rewards))
                            metrics['max_reward'] = float(np.max(rewards))
                            metrics['num_episodes'] = len(rewards)

                    # Récupérer le numéro d'épisode actuel si disponible
                    episode = getattr(self.model, '_episode_num', 0)

                    # Sauvegarder le checkpoint
                    checkpoint_path = self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.model.policy.optimizer if hasattr(self.model.policy, 'optimizer') else None,
                        episode=episode,
                        total_steps=self.num_timesteps,
                        metrics=metrics,
                        custom_data={
                            'config': config,
                            'policy_kwargs': policy_kwargs,
                            'env_config': config.get('env', {})
                        },
                        is_final=False
                    )

                    if checkpoint_path:
                        self.last_checkpoint = checkpoint_path
                        self.last_save = self.num_timesteps
                        logger.info(
                            "Checkpoint sauvegardé (étape %d, épisode %d, récompense moyenne: %s)",
                            self.num_timesteps,
                            episode,
                            metrics.get('mean_reward', 'N/A')
                        )

                        # Nettoyer les anciens checkpoints si nécessaire
                        self.checkpoint_manager._cleanup_old_checkpoints()

                except Exception as e:
                    logger.error("Erreur lors de la sauvegarde du checkpoint: %s", str(e), exc_info=True)

            def _on_training_end(self) -> None:
                """
                Appelé à la fin de l'entraînement pour sauvegarder un checkpoint final.
                """
                try:
                    # Sauvegarder un checkpoint final
                    metrics = {}
                    if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                        rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer if 'r' in ep_info]
                        if rewards:
                            metrics['final_mean_reward'] = float(np.mean(rewards))

                    episode = getattr(self.model, '_episode_num', 0)

                    checkpoint_path = self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.model.policy.optimizer if hasattr(self.model.policy, 'optimizer') else None,
                        episode=episode,
                        total_steps=self.num_timesteps,
                        metrics=metrics,
                        custom_data={
                            'config': config,
                            'policy_kwargs': policy_kwargs,
                            'env_config': config.get('env', {}),
                            'training_completed': True
                        },
                        is_final=True
                    )

                    if checkpoint_path:
                        logger.info(
                            "[TRAINING] Checkpoint final sauvegardé: %s (étape %d, épisode %d)",
                            checkpoint_path, self.num_timesteps, episode
                        )

                except Exception as e:
                    logger.error("[TRAINING] Erreur lors de la sauvegarde du checkpoint final: %s", str(e), exc_info=True)

        # Créer le callback de sauvegarde des checkpoints
        checkpoint_callback = CustomCheckpointCallback(
            checkpoint_manager=checkpoint_manager,
            verbose=1
        )

        # Initialiser la liste des callbacks avec le checkpoint
        callbacks = [checkpoint_callback]

        # Ajouter le callback de progression personnalisé si activé dans la config
        use_custom_progress = config.get('training', {}).get('use_custom_progress', False)
        if use_custom_progress:
            # Créer une instance de notre callback personnalisé
            progress_callback = CustomTrainingInfoCallback(check_freq=1000, verbose=1)
            callbacks.append(progress_callback)
            logger.info("[TRAINING] Barre de progression personnalisée activée pour le suivi de l'entraînement")

        # Créer un CallbackList pour gérer plusieurs callbacks
        from stable_baselines3.common.callbacks import CallbackList
        callback = CallbackList(callbacks)

        logger.info(f"[TRAINING] {len(callbacks)} callback(s) configuré(s) pour l'entraînement")

        # Prepare timeout manager with cleanup callback
        tm = None
        if timeout is not None and timeout > 0:
            def _cleanup_on_timeout():
                logger.info("[TRAINING] Timeout reached, attempting to save checkpoint before exit...")
                try:
                    # Ensure checkpoint directory exists
                    os.makedirs(checkpoint_manager.checkpoint_dir, exist_ok=True)
                except Exception:
                    pass
                try:
                    optimizer = getattr(model.policy, 'optimizer', None) if hasattr(model, 'policy') else None
                    episode = getattr(model, '_episode_num', 0)
                    total_steps = getattr(model, 'num_timesteps', 0)
                    checkpoint_manager.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        episode=episode,
                        total_steps=total_steps,
                        is_final=False
                    )
                    logger.info("[TRAINING] Checkpoint saved on timeout.")
                except Exception as e:
                    logger.error("[TRAINING] Failed to save checkpoint on timeout: %s", e)

            tm = TimeoutManager(timeout=float(timeout), cleanup_callback=_cleanup_on_timeout)

        try:
            # Entraînement avec gestion des interruptions
            try:
                # Calculer le nombre d'étapes restantes
                total_timesteps = config.get('training', {}).get('total_timesteps', 1000000)
                remaining_timesteps = total_timesteps - start_timesteps

                if remaining_timesteps <= 0:
                    logger.info("[TRAINING] L'entraînement est déjà terminé selon le nombre d'étapes total configuré.")
                    return True

                logger.info("[TRAINING] Démarrage de l'entraînement pour %d étapes supplémentaires...", remaining_timesteps)

                # Démarrer l'entraînement avec les callbacks
                if tm:
                    with tm.limit():
                        model.learn(
                            total_timesteps=remaining_timesteps,
                            callback=callback,  # Utilisation du CallbackList
                            reset_num_timesteps=False,  # Ne pas réinitialiser le compteur d'étapes
                            progress_bar=False  # Désactivé pour éviter les conflits avec notre callback personnalisé
                        )
                else:
                    model.learn(
                        total_timesteps=remaining_timesteps,
                        callback=callback,  # Utilisation du CallbackList
                        reset_num_timesteps=False,  # Ne pas réinitialiser le compteur d'étapes
                        progress_bar=progress_bar  # Utiliser la barre de progression configurée
                    )
            except KeyboardInterrupt:
                logger.info(
                    "[TRAINING] \nInterruption de l'utilisateur détectée. "
                    "Sauvegarde du dernier état..."
                )
                # Sauvegarder un checkpoint final
                optimizer = None
                if hasattr(model.policy, 'optimizer'):
                    optimizer = model.policy.optimizer

                episode = 0
                if hasattr(model, '_episode_num'):
                    episode = model._episode_num

                total_steps = 0
                if hasattr(model, 'num_timesteps'):
                    total_steps = model.num_timesteps

                checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    episode=episode,
                    total_steps=total_steps,
                    is_final=True
                )
                logger.info("Dernier état sauvegardé avec succès.")
                raise
            logger.info("Entraînement terminé avec succès!")

        except (TimeoutException, TMTimeoutException):
            logger.info(
                "Temps d'entraînement écoulé. Arrêt de l'entraînement..."
            )
        except Exception as e:
            logger.error(
                "Erreur lors de l'entraînement: %s",
                str(e)
            )
            raise

        finally:
            # Sauvegarder le modèle final
            final_metrics = {}
            if hasattr(model, 'ep_info_buffer'):
                rewards = [ep_info['r'] for ep_info in model.ep_info_buffer]
                if rewards:  # Vérifier si la liste n'est pas vide
                    final_metrics['episode_reward'] = np.mean(rewards)

            optimizer = None
            if hasattr(model.policy, 'optimizer'):
                optimizer = model.policy.optimizer

            episode = 0
            if hasattr(model, '_episode_num'):
                episode = model._episode_num

            total_steps = 0
            if hasattr(model, 'num_timesteps'):
                total_steps = model.num_timesteps

            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                episode=episode,
                total_steps=total_steps,
                metrics=final_metrics,
                custom_data={
                    'config': config,
                    'policy_kwargs': policy_kwargs,
                    'training_complete': True
                },
                is_final=True
            )
            logger.info("Modèle final sauvegardé avec succès.")

            # Afficher un message de fin
            logger.info("Entraînement terminé. Nettoyage des ressources...")

        # Nettoyer
        env.close()
        return True

    except Exception as e:
        logger.error(
            "Erreur lors de l'exécution de l'entraînement: %s",
            str(e)
        )
        logger.error(traceback.format_exc())

        # Fermer les environnements en cas d'erreur
        if 'env' in locals():
            env.close()

        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Entraîne un bot de trading ADAN avec support du timeout "
            "et des points de contrôle"
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="bot/config/config.yaml",
        help="Chemin vers le fichier de configuration"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Temps maximum d'entraînement en secondes"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Répertoire pour enregistrer les points de contrôle",
    )
    parser.add_argument(
        "--shared-model",
        type=str,
        default=None,
        help=(
            "Chemin vers un modèle partagé pour l'entraînement "
            "distribué"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reprend l'entraînement à partir du dernier point de contrôle"
    )

    # Ajout des arguments supplémentaires mentionnés dans le patch
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (optional)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    # Appel robuste de la fonction main()
    try:
        # 1) Appel avec les arguments nommés
        success = main(
            config_path=args.config,
            timeout=args.timeout,
            checkpoint_dir=args.checkpoint_dir,
            shared_model_path=args.shared_model,
            resume=args.resume
        )
    except Exception as e:
        print(f"Erreur lors de l'exécution: {e}", file=sys.stderr)
        raise

    sys.exit(0 if success else 1)
