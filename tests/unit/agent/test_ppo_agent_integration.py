"""
Tests d'intégration réels pour PPOAgent avec environnements Gymnasium fonctionnels.
Ces tests n'utilisent PAS de mocks - ils testent avec de vrais environnements.
"""

import pytest
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tempfile
import os
import torch
import yaml

from adan_trading_bot.agent.ppo_agent import PPOAgent


class SimpleTestTradingEnv(gym.Env):
    """
    Environnement de test simple mais réel qui respecte l'interface Gymnasium.
    Simule un environnement de trading minimal pour tester PPOAgent.
    """

    def __init__(
        self, timeframes=["5m", "1h", "4h"], sequence_length=20, n_features=16
    ):
        super().__init__()

        self.timeframes = timeframes
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.current_step = 0
        self.max_steps = 100

        # Espace d'observation: Dict avec chaque timeframe + portfolio_state
        obs_space = {
            tf: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(sequence_length, n_features),
                dtype=np.float32,
            )
            for tf in timeframes
        }
        # Ajouter portfolio_state (20 dimensions)
        obs_space["portfolio_state"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(obs_space)

        # Espace d'action: 4 actions (hold, buy, sell, close) pour chaque timeframe
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(timeframes) * 4,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Générer des données synthétiques réalistes
        observation = {}
        for tf in self.timeframes:
            # Simuler des données de marché (prix + indicateurs)
            data = np.random.randn(
                self.sequence_length, self.n_features
            ).astype(np.float32)
            # Ajouter une tendance légère pour simuler des données réelles
            trend = np.linspace(0, 0.1, self.sequence_length).reshape(-1, 1)
            data[:, :4] += trend  # Prix OHLC avec tendance
            observation[tf] = data

        # Ajouter portfolio_state (20 dimensions)
        observation["portfolio_state"] = np.random.randn(20).astype(
            np.float32
        )

        info = {"step": self.current_step}
        return observation, info

    def step(self, action):
        self.current_step += 1

        # Générer nouvelle observation
        observation = {}
        for tf in self.timeframes:
            data = np.random.randn(
                self.sequence_length, self.n_features
            ).astype(np.float32)
            # Simuler une réaction légère à l'action
            action_effect = np.mean(action) * 0.01
            data[:, :4] += action_effect
            observation[tf] = data

        # Ajouter portfolio_state (20 dimensions)
        observation["portfolio_state"] = np.random.randn(20).astype(
            np.float32
        )

        # Reward simple basé sur l'action
        reward = -0.001 + np.random.randn() * 0.01  # Petit bruit

        # Terminal conditions
        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "step": self.current_step,
            "action_taken": action,
            "portfolio_value": 1000 + np.random.randn() * 10,
        }

        return observation, reward, terminated, truncated, info


class TestPPOAgentIntegration:
    """Tests d'intégration réels pour PPOAgent - SANS mocks."""

    @pytest.fixture
    def simple_config(self):
        """Configuration minimale mais fonctionnelle pour PPO."""
        return {
            "agent": {
                "algorithm": "PPO",
                "learning_rate": 0.0003,
                "n_steps": 64,  # Plus petit pour les tests
                "batch_size": 32,
                "n_epochs": 2,  # Moins d'epochs pour les tests
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "policy": "MultiInputPolicy",
                "verbose": 0,
                "seed": 42,
            }
        }

    @pytest.fixture
    def simple_env(self):
        """Environnement de test simple mais réel."""
        return SimpleTestTradingEnv()

    def test_ppo_agent_initialization_with_real_env(self, simple_env, simple_config):
        """Test que PPOAgent s'initialise correctement avec un vrai environnement."""
        try:
            agent = PPOAgent(simple_env, simple_config)

            # Vérifications de base
            assert agent.model is not None
            assert hasattr(agent, "config")
            assert agent.config == simple_config

            # Vérifier que l'environnement est correctement wrappe
            assert hasattr(agent.model, "env")

        except Exception as e:
            pytest.fail(
                f"PPOAgent n'a pas pu s'initialiser avec un vrai environnement: {e}"
            )

    def test_ppo_predict_with_real_observation(self, simple_env, simple_config):
        """Test que PPO peut prédire des actions avec de vraies observations."""
        agent = PPOAgent(simple_env, simple_config)

        # Obtenir une vraie observation
        observation, _ = simple_env.reset()

        # Prédire une action
        action, _states = agent.predict(observation, deterministic=True)

        # Vérifications
        assert action is not None
        assert len(action) == simple_env.action_space.shape[0]
        assert np.all(action >= -1.0) and np.all(action <= 1.0)  # Dans les bounds

    def test_ppo_learn_functionality(self, simple_env, simple_config):
        """Test que PPO peut apprendre (même sur peu d'étapes)."""
        # Configuration avec très peu d'steps pour test rapide
        simple_config["agent"]["n_steps"] = 16
        simple_config["agent"]["batch_size"] = 8

        agent = PPOAgent(simple_env, simple_config)

        # Apprendre sur quelques steps seulement
        try:
            agent.learn(total_timesteps=32, log_interval=1)
        except Exception as e:
            pytest.fail(f"L'apprentissage PPO a échoué avec une vraie observation: {e}")
        except Exception as e:
            pytest.fail(f"L'apprentissage PPO a échoué avec une vraie observation: {e}")
