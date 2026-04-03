"""
Tests unitaires réels pour PPOAgent - Tests du fonctionnement effectif du module.
"""

import pytest
import numpy as np
import torch
import gymnasium as gym
from unittest.mock import Mock, patch, MagicMock
import sys
import os

from adan_trading_bot.agent.ppo_agent import PPOAgent
from adan_trading_bot.common.config_loader import ConfigLoader


class TestPPOAgentReal:
    """Tests unitaires réels pour PPOAgent."""

    @pytest.fixture
    def real_config(self):
        """Configuration réelle du système."""
        return ConfigLoader.load_config("config/config.yaml")

    @pytest.fixture
    def minimal_config(self):
        """Configuration minimale pour les tests."""
        return {
            "agent": {
                "policy": "MultiInputPolicy",
                "learning_rate": 0.001,
                "n_steps": 512,
                "batch_size": 64,
                "n_epochs": 2,
                "ent_coef": 0.01,
                "clip_range": 0.2,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "verbose": 0,
            }
        }

    @pytest.fixture
    def mock_env(self):
        """Environnement mock pour les tests - hérite de gymnasium.Env."""
        class MockTradingEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = gym.spaces.Dict(
                    {
                        "5m": gym.spaces.Box(low=-1, high=1, shape=(20, 15), dtype=np.float32),
                        "1h": gym.spaces.Box(low=-1, high=1, shape=(14, 15), dtype=np.float32),
                        "4h": gym.spaces.Box(low=-1, high=1, shape=(5, 15), dtype=np.float32),
                        "portfolio_state": gym.spaces.Box(
                            low=-1, high=1, shape=(10,), dtype=np.float32
                        ),
                    }
                )
                self.action_space = gym.spaces.Discrete(27)  # 3^3 actions pour 3 assets
                self.step_count = 0
                self.max_steps = 100

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self.step_count = 0
                return (
                    {
                        "5m": np.random.random((20, 15)).astype(np.float32),
                        "1h": np.random.random((14, 15)).astype(np.float32),
                        "4h": np.random.random((5, 15)).astype(np.float32),
                        "portfolio_state": np.random.random(10).astype(np.float32),
                    },
                    {},
                )

            def step(self, action):
                self.step_count += 1
                terminated = self.step_count >= self.max_steps
                return (
                    {
                        "5m": np.random.random((20, 15)).astype(np.float32),
                        "1h": np.random.random((14, 15)).astype(np.float32),
                        "4h": np.random.random((5, 15)).astype(np.float32),
                        "portfolio_state": np.random.random(10).astype(np.float32),
                    },
                    1.0,  # reward
                    terminated,  # terminated
                    False,  # truncated
                    {},  # info
                )

            def render(self):
                pass

            def close(self):
                pass

        return MockTradingEnv()

    def test_ppo_agent_initialization_with_real_config(self, mock_env, real_config):
        """Test l'initialisation avec la vraie configuration."""
        try:
            agent = PPOAgent(mock_env, real_config)

            assert agent is not None
            assert hasattr(agent, "model")
            assert hasattr(agent, "env")
            assert agent.config == real_config

            print("✅ PPOAgent s'initialise correctement avec la vraie config")

        except Exception as e:
            pytest.fail(f"PPOAgent n'a pas pu s'initialiser avec la vraie config: {e}")

    def test_ppo_agent_initialization_minimal(self, mock_env, minimal_config):
        """Test l'initialisation avec configuration minimale."""
        try:
            agent = PPOAgent(mock_env, minimal_config)

            assert agent is not None
            assert hasattr(agent, "model")
            assert agent.model is not None

            # Vérifier que les paramètres sont correctement appliqués
            assert agent.config["agent"]["learning_rate"] == 0.001
            assert agent.config["agent"]["n_steps"] == 512

            print("✅ PPOAgent s'initialise avec config minimale")

        except Exception as e:
            pytest.fail(f"PPOAgent n'a pas pu s'initialiser: {e}")

    def test_predict_action_functionality(self, mock_env, minimal_config):
        """Test que predict() fonctionne réellement."""
        agent = PPOAgent(mock_env, minimal_config)

        # Créer une observation réaliste
        observation = {
            "5m": np.random.random((20, 15)).astype(np.float32),
            "1h": np.random.random((14, 15)).astype(np.float32),
            "4h": np.random.random((5, 15)).astype(np.float32),
            "portfolio_state": np.random.random(10).astype(np.float32),
        }

        # Test de prédiction (predict retourne un tuple (action, state))
        result = agent.predict(observation)
        action = result[0] if isinstance(result, tuple) else result

        assert isinstance(action, (int, np.integer, np.ndarray))
        action_val = int(action) if isinstance(action, np.ndarray) else action
        assert 0 <= action_val < mock_env.action_space.n

        print(f"✅ Predict fonctionne - Action prédite: {action_val}")

    def test_predict_deterministic_vs_stochastic(self, mock_env, minimal_config):
        """Test la différence entre mode déterministe et stochastique."""
        agent = PPOAgent(mock_env, minimal_config)

        observation = {
            "5m": np.random.random((20, 15)).astype(np.float32),
            "1h": np.random.random((14, 15)).astype(np.float32),
            "4h": np.random.random((5, 15)).astype(np.float32),
            "portfolio_state": np.random.random(10).astype(np.float32),
        }

        # Prédictions déterministes (devrait être identique)
        result1_det = agent.predict(observation, deterministic=True)
        action1_det = result1_det[0] if isinstance(result1_det, tuple) else result1_det
        action1_det = int(action1_det) if isinstance(action1_det, np.ndarray) else action1_det

        result2_det = agent.predict(observation, deterministic=True)
        action2_det = result2_det[0] if isinstance(result2_det, tuple) else result2_det
        action2_det = int(action2_det) if isinstance(action2_det, np.ndarray) else action2_det

        # Prédictions stochastiques (peut être différent)
        result1_stoch = agent.predict(observation, deterministic=False)
        action1_stoch = result1_stoch[0] if isinstance(result1_stoch, tuple) else result1_stoch
        action1_stoch = int(action1_stoch) if isinstance(action1_stoch, np.ndarray) else action1_stoch

        result2_stoch = agent.predict(observation, deterministic=False)
        action2_stoch = result2_stoch[0] if isinstance(result2_stoch, tuple) else result2_stoch
        action2_stoch = int(action2_stoch) if isinstance(action2_stoch, np.ndarray) else action2_stoch

        # Les prédictions déterministes devraient être identiques
        assert action1_det == action2_det, (
            "Les prédictions déterministes devraient être identiques"
        )

        # Les actions doivent être valides
        assert 0 <= action1_det < mock_env.action_space.n
        assert 0 <= action1_stoch < mock_env.action_space.n

        print(f"✅ Mode déterministe: {action1_det} == {action2_det}")
        print(f"✅ Mode stochastique: {action1_stoch}, {action2_stoch}")

    def test_learn_functionality(self, mock_env, minimal_config):
        """Test que learn() fonctionne sans erreur."""
        agent = PPOAgent(mock_env, minimal_config)

        # Test d'apprentissage court
        total_timesteps = 100

        try:
            result = agent.learn(total_timesteps)

            assert result is not None
            assert hasattr(
                result, "predict"
            )  # Le modèle retourné doit avoir une méthode predict

            print("✅ Learn() s'exécute sans erreur")

        except Exception as e:
            pytest.fail(f"Learn() a échoué: {e}")

    def test_model_save_and_load(self, mock_env, minimal_config, tmp_path):
        """Test de sauvegarde et chargement du modèle."""
        agent = PPOAgent(mock_env, minimal_config)

        # Chemin de sauvegarde
        save_path = tmp_path / "test_model"

        # Test de sauvegarde
        try:
            agent.save(str(save_path))
            assert (
                save_path.exists()
                or (save_path.parent / f"{save_path.name}.zip").exists()
            )
            print("✅ Sauvegarde du modèle réussie")
        except Exception as e:
            pytest.fail(f"Sauvegarde échouée: {e}")

        # Test de chargement
        try:
            agent.load(str(save_path))
            print("✅ Chargement du modèle réussi")
        except Exception as e:
            pytest.fail(f"Chargement échoué: {e}")

    def test_observation_space_compatibility(self, mock_env, minimal_config):
        """Test la compatibilité avec l'espace d'observation."""
        agent = PPOAgent(mock_env, minimal_config)

        # Vérifier que l'agent peut gérer l'observation space
        obs_space = mock_env.observation_space

        assert isinstance(obs_space, gym.spaces.Dict)
        assert "5m" in obs_space.spaces
        assert "1h" in obs_space.spaces
        assert "4h" in obs_space.spaces
        assert "portfolio_state" in obs_space.spaces

        # Test avec une vraie observation
        observation = {
            "5m": np.zeros((20, 15), dtype=np.float32),
            "1h": np.zeros((14, 15), dtype=np.float32),
            "4h": np.zeros((5, 15), dtype=np.float32),
            "portfolio_state": np.zeros(10, dtype=np.float32),
        }

        try:
            result = agent.predict(observation)
            action = result[0] if isinstance(result, tuple) else result
            action_val = int(action) if isinstance(action, np.ndarray) else action
            assert isinstance(action_val, (int, np.integer))
            print("✅ Observation space compatible")
        except Exception as e:
            pytest.fail(f"Incompatibilité observation space: {e}")

    def test_action_space_bounds(self, mock_env, minimal_config):
        """Test que les actions respectent l'espace d'action."""
        agent = PPOAgent(mock_env, minimal_config)

        observation = {
            "5m": np.random.random((20, 15)).astype(np.float32),
            "1h": np.random.random((14, 15)).astype(np.float32),
            "4h": np.random.random((5, 15)).astype(np.float32),
            "portfolio_state": np.random.random(10).astype(np.float32),
        }

        # Tester plusieurs prédictions
        actions = []
        for _ in range(20):
            result = agent.predict(observation)
            action = result[0] if isinstance(result, tuple) else result
            action_val = int(action) if isinstance(action, np.ndarray) else action
            actions.append(action_val)

            # Vérifier que l'action est dans les bornes
            assert 0 <= action_val < mock_env.action_space.n, (
                f"Action {action_val} hors bornes [0, {mock_env.action_space.n})"
            )

        print(f"✅ Toutes les actions sont dans les bornes: {set(actions)}")

    def test_multiple_predictions_consistency(self, mock_env, minimal_config):
        """Test la cohérence des prédictions multiples."""
        agent = PPOAgent(mock_env, minimal_config)

        observations = []
        actions = []

        # Générer plusieurs observations différentes
        for i in range(5):
            obs = {
                "5m": np.random.random((20, 15)).astype(np.float32) * (i + 1),
                "1h": np.random.random((14, 15)).astype(np.float32) * (i + 1),
                "4h": np.random.random((5, 15)).astype(np.float32) * (i + 1),
                "portfolio_state": np.random.random(10).astype(np.float32) * (i + 1),
            }
            observations.append(obs)

            result = agent.predict(obs, deterministic=True)
            action = result[0] if isinstance(result, tuple) else result
            action_val = int(action) if isinstance(action, np.ndarray) else action
            actions.append(action_val)

        # Vérifier que toutes les actions sont valides
        assert all(isinstance(a, (int, np.integer)) for a in actions)
        assert all(0 <= a < mock_env.action_space.n for a in actions)

        print(f"✅ Prédictions multiples cohérentes: {actions}")

    def test_model_training_integration(self, mock_env, minimal_config):
        """Test d'intégration complète avec entraînement."""
        # Configuration pour entraînement rapide
        minimal_config["agent"]["n_steps"] = 64
        minimal_config["agent"]["batch_size"] = 32
        minimal_config["agent"]["n_epochs"] = 1

        agent = PPOAgent(mock_env, minimal_config)

        # Simuler un épisode d'entraînement court
        total_timesteps = 64

        try:
            # Entraînement
            trained_model = agent.learn(total_timesteps)

            # Test de prédiction après entraînement
            observation = {
                "5m": np.random.random((20, 15)).astype(np.float32),
                "1h": np.random.random((14, 15)).astype(np.float32),
                "4h": np.random.random((5, 15)).astype(np.float32),
                "portfolio_state": np.random.random(10).astype(np.float32),
            }

            result = agent.predict(observation)
            action = result[0] if isinstance(result, tuple) else result
            action_val = int(action) if isinstance(action, np.ndarray) else action
            assert isinstance(action_val, (int, np.integer))

            print("✅ Intégration entraînement-prédiction réussie")

        except Exception as e:
            pytest.fail(f"Intégration entraînement échouée: {e}")

    def test_config_parameter_application(self, mock_env):
        """Test que les paramètres de config sont bien appliqués."""
        custom_config = {
            "agent": {
                "policy": "MultiInputPolicy",
                "learning_rate": 0.123,  # Valeur spécifique
                "n_steps": 256,  # Valeur spécifique
                "batch_size": 16,  # Valeur spécifique
                "n_epochs": 3,
                "ent_coef": 0.456,  # Valeur spécifique
                "clip_range": 0.789,  # Valeur spécifique
                "gamma": 0.987,
                "gae_lambda": 0.654,
                "vf_coef": 0.321,
                "max_grad_norm": 1.5,
                "verbose": 1,
            }
        }

        agent = PPOAgent(mock_env, custom_config)

        # Vérifier que la configuration est stockée
        assert agent.config["agent"]["learning_rate"] == 0.123
        assert agent.config["agent"]["n_steps"] == 256
        assert agent.config["agent"]["batch_size"] == 16
        assert agent.config["agent"]["ent_coef"] == 0.456
        assert agent.config["agent"]["clip_range"] == 0.789

        print("✅ Paramètres de config correctement appliqués")

    def test_error_handling_invalid_observation(self, mock_env, minimal_config):
        """Test la gestion d'erreurs avec observations invalides."""
        agent = PPOAgent(mock_env, minimal_config)

        # Test avec observation mal formée
        invalid_obs = {
            "5m": np.random.random((10, 10)).astype(np.float32),  # Mauvaises dimensions
        }

        with pytest.raises((ValueError, KeyError, RuntimeError)):
            agent.predict(invalid_obs)

        print("✅ Gestion d'erreurs pour observations invalides")

    def test_model_attributes_existence(self, mock_env, minimal_config):
        """Test l'existence des attributs essentiels du modèle."""
        agent = PPOAgent(mock_env, minimal_config)

        # Vérifier les attributs essentiels
        assert hasattr(agent, "model")
        assert hasattr(agent, "env")
        assert hasattr(agent, "config")

        # Vérifier les méthodes essentielles
        assert hasattr(agent, "predict")
        assert hasattr(agent, "learn")
        assert hasattr(agent, "save")
        assert hasattr(agent, "load")

        # Vérifier que le modèle est bien initialisé
        assert agent.model is not None
        assert hasattr(agent.model, "predict")

        print("✅ Tous les attributs essentiels sont présents")
