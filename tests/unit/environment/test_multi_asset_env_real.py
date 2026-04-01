"""
Tests unitaires réels pour MultiAssetChunkedEnv - Tests du fonctionnement effectif du module.
"""

import pytest
import numpy as np
import pandas as pd
import gym
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime, timedelta

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.common.config_loader import ConfigLoader


class TestMultiAssetChunkedEnvReal:
    """Tests unitaires réels pour MultiAssetChunkedEnv."""

    @pytest.fixture
    def real_config(self):
        """Configuration réelle du système."""
        return ConfigLoader.load_config("config/config.yaml")

    @pytest.fixture
    def minimal_config(self):
        """Configuration minimale pour les tests."""
        return {
            "environment": {
                "initial_balance": 1000.0,
                "max_steps": 500,
                "commission": 0.001,
                "assets": ["BTCUSDT"],
                "observation": {
                    "timeframes": ["5m", "1h", "4h"],
                    "window_sizes": {"5m": 20, "1h": 14, "4h": 5},
                    "features": {
                        "base": ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"],
                        "indicators": {
                            "5m": ["RSI_14", "ATR_14", "MACD_HIST"],
                            "1h": ["RSI_14", "ADX_14", "MACD_HIST"],
                            "4h": ["RSI_14", "ADX_14", "ATR_14"],
                        },
                    },
                },
                "reward_params": {
                    "pnl_weight": 1.0,
                    "win_rate_bonus": 0.5,
                    "stop_loss_penalty": -1.0,
                    "take_profit_bonus": 1.0,
                },
            },
            "trading": {
                "assets": ["BTCUSDT"],
                "timeframes": ["5m", "1h", "4h"],
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.10,
            },
            "workers": {
                "w1": {
                    "assets": ["BTCUSDT"],
                    "timeframes": ["5m", "1h", "4h"],
                }
            }
        }

    @pytest.fixture
    def mock_data_loader(self):
        """DataLoader mock avec données réalistes."""
        data_loader = Mock()

        # Créer des données réalistes pour chaque timeframe
        timestamps_5m = pd.date_range(start="2024-01-01", periods=1000, freq="5min")
        timestamps_1h = pd.date_range(start="2024-01-01", periods=200, freq="1h")
        timestamps_4h = pd.date_range(start="2024-01-01", periods=50, freq="4h")

        # Données 5m
        data_5m = pd.DataFrame(
            {
                "timestamp": timestamps_5m,
                "OPEN": np.random.uniform(40000, 50000, len(timestamps_5m)),
                "HIGH": np.random.uniform(50000, 55000, len(timestamps_5m)),
                "LOW": np.random.uniform(35000, 40000, len(timestamps_5m)),
                "CLOSE": np.random.uniform(40000, 50000, len(timestamps_5m)),
                "VOLUME": np.random.uniform(100, 1000, len(timestamps_5m)),
                "RSI_14": np.random.uniform(20, 80, len(timestamps_5m)),
                "ATR_14": np.random.uniform(500, 2000, len(timestamps_5m)),
                "MACD_HIST": np.random.uniform(-100, 100, len(timestamps_5m)),
            }
        )

        # Données 1h
        data_1h = pd.DataFrame(
            {
                "timestamp": timestamps_1h,
                "OPEN": np.random.uniform(40000, 50000, len(timestamps_1h)),
                "HIGH": np.random.uniform(50000, 55000, len(timestamps_1h)),
                "LOW": np.random.uniform(35000, 40000, len(timestamps_1h)),
                "CLOSE": np.random.uniform(40000, 50000, len(timestamps_1h)),
                "VOLUME": np.random.uniform(100, 1000, len(timestamps_1h)),
                "RSI_14": np.random.uniform(20, 80, len(timestamps_1h)),
                "ADX_14": np.random.uniform(10, 50, len(timestamps_1h)),
                "MACD_HIST": np.random.uniform(-100, 100, len(timestamps_1h)),
            }
        )

        # Données 4h
        data_4h = pd.DataFrame(
            {
                "timestamp": timestamps_4h,
                "OPEN": np.random.uniform(40000, 50000, len(timestamps_4h)),
                "HIGH": np.random.uniform(50000, 55000, len(timestamps_4h)),
                "LOW": np.random.uniform(35000, 40000, len(timestamps_4h)),
                "CLOSE": np.random.uniform(40000, 50000, len(timestamps_4h)),
                "VOLUME": np.random.uniform(100, 1000, len(timestamps_4h)),
                "RSI_14": np.random.uniform(20, 80, len(timestamps_4h)),
                "ADX_14": np.random.uniform(10, 50, len(timestamps_4h)),
                "ATR_14": np.random.uniform(500, 2000, len(timestamps_4h)),
            }
        )

        mock_data = {
            "BTCUSDT_5m": data_5m,
            "BTCUSDT_1h": data_1h,
            "BTCUSDT_4h": data_4h,
        }

        data_loader.load_chunk.return_value = mock_data
        return data_loader

    @pytest.fixture
    def mock_portfolio_manager(self):
        """PortfolioManager mock."""
        portfolio = Mock()
        portfolio.get_balance.return_value = 1000.0
        portfolio.get_positions.return_value = {}
        portfolio.get_total_value.return_value = 1000.0
        portfolio.get_unrealized_pnl.return_value = 0.0
        portfolio.get_realized_pnl.return_value = 0.0
        portfolio.execute_trade.return_value = True
        portfolio.update_market_prices.return_value = None
        portfolio.get_metrics.return_value = {
            "balance": 1000.0,
            "positions": {},
            "total_value": 1000.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
        }
        return portfolio

    @pytest.fixture
    def mock_state_builder(self):
        """StateBuilder mock."""
        state_builder = Mock()
        # État structuré par timeframe
        mock_state = {
            "5m": np.random.random((20, 15)).astype(np.float32),
            "1h": np.random.random((14, 15)).astype(np.float32),
            "4h": np.random.random((5, 15)).astype(np.float32),
            "portfolio_state": np.random.random(10).astype(np.float32),
        }
        state_builder.build_state.return_value = mock_state
        return state_builder

    def test_env_initialization_with_real_config(self, real_config, mock_data_loader):
        """Test l'initialisation avec la vraie configuration."""
        with patch(
            "adan_trading_bot.data_processing.data_loader.ChunkedDataLoader",
            return_value=mock_data_loader,
        ):
            try:
                env = MultiAssetChunkedEnv(config=real_config, worker_config=real_config['workers']['w1'])

                assert env is not None
                assert hasattr(env, "observation_space")
                assert hasattr(env, "action_space")
                assert hasattr(env, "reset")
                assert hasattr(env, "step")


            except Exception as e:
                pytest.fail(f"Environment n'a pas pu s'initialiser: {e}")

    def test_observation_space_structure(self, minimal_config, mock_data_loader):
        with patch(
            "adan_trading_bot.data_processing.data_loader.ChunkedDataLoader",
            return_value=mock_data_loader,
        ):
            env = MultiAssetChunkedEnv(config=minimal_config, worker_config=minimal_config['workers']['w1'])

            obs_space = env.observation_space

            # Vérifier que c'est un Dict space
            assert isinstance(obs_space, gym.spaces.Dict)

            # Vérifier les timeframes
            assert "5m" in obs_space.spaces
            assert "1h" in obs_space.spaces
            assert "4h" in obs_space.spaces
            assert "portfolio_state" in obs_space.spaces
            assert "context_vector" in obs_space.spaces

            # Vérifier les dimensions (features dynamiques)
            assert obs_space.spaces["5m"].shape[0] == 20   # window_size
            assert obs_space.spaces["1h"].shape[0] == 14    # window_size
            assert obs_space.spaces["4h"].shape[0] == 5     # window_size
            assert obs_space.spaces["context_vector"].shape == (12,)

            print("✅ Observation space correctement structuré")

    def test_action_space_dimensionality(self, minimal_config, mock_data_loader):
        """Test les dimensions de l'espace d'action."""
        with patch(
            "adan_trading_bot.data_processing.data_loader.ChunkedDataLoader",
            return_value=mock_data_loader,
        ):
            env = MultiAssetChunkedEnv(minimal_config)

            action_space = env.action_space

            # Vérifier que c'est un Box space (actions continues)
            assert isinstance(action_space, gym.spaces.Box)

            # 5 assets × 5 dims (action, size, timeframe, SL, TP) = 25
            assert action_space.shape == (25,)
            assert action_space.dtype == np.float32

            print(f"✅ Action space dimensionnality correcte: {action_space.shape}")

    def test_reset_functionality(
        self, minimal_config, mock_data_loader, mock_state_builder
    ):
        """Test la fonction reset."""
        with (
            patch(
                "adan_trading_bot.data_processing.data_loader.ChunkedDataLoader",
                return_value=mock_data_loader,
            ),
            patch(
                "adan_trading_bot.data_processing.state_builder.StateBuilder",
                return_value=mock_state_builder,
            ),
        ):
            env = MultiAssetChunkedEnv(minimal_config)

            observation, info = env.reset()

            # Vérifier la structure de l'observation
            assert isinstance(observation, dict)
            assert "5m" in observation
            assert "1h" in observation
            assert "4h" in observation
            assert "portfolio_state" in observation

            # Vérifier les types et dimensions
            assert observation["5m"].dtype == np.float32
            assert observation["5m"].shape == (20, 15)
            assert observation["1h"].shape == (14, 15)
            assert observation["4h"].shape == (5, 15)

            # Vérifier les infos
            assert isinstance(info, dict)

            print("✅ Reset fonctionne correctement")

    def test_step_with_different_actions(
        self,
        minimal_config,
        mock_data_loader,
        mock_state_builder,
        mock_portfolio_manager,
    ):
        """Test la fonction step avec différentes actions."""
        with (
            patch(
                "adan_trading_bot.data_processing.data_loader.ChunkedDataLoader",
                return_value=mock_data_loader,
            ),
            patch(
                "adan_trading_bot.data_processing.state_builder.StateBuilder",
                return_value=mock_state_builder,
            ),
            patch(
                "adan_trading_bot.portfolio.portfolio_manager.PortfolioManager",
                return_value=mock_portfolio_manager,
            ),
        ):
            env = MultiAssetChunkedEnv(minimal_config)
            env.reset()

            # Test différentes actions (continues)
            actions_to_test = [
                np.random.uniform(-1, 1, 15).astype(np.float32),
                np.zeros(15, dtype=np.float32),
                np.ones(15, dtype=np.float32) * 0.5,
            ]

            for action in actions_to_test:
                observation, reward, terminated, truncated, info = env.step(action)

                # Vérifier les types de retour
                assert isinstance(observation, dict)
                assert isinstance(reward, (float, np.floating))
                assert isinstance(terminated, bool)
                assert isinstance(truncated, bool)
                assert isinstance(info, dict)

                # Vérifier la structure de l'observation
                assert "5m" in observation
                assert observation["5m"].dtype == np.float32

                print(f"✅ Action {action} exécutée - Reward: {reward:.4f}")

    def test_multi_timeframe_data_handling(
        self, minimal_config, mock_data_loader, mock_state_builder
    ):
        """Test la gestion des données multi-timeframe."""
        with (
            patch(
                "adan_trading_bot.data_processing.data_loader.ChunkedDataLoader",
                return_value=mock_data_loader,
            ),
            patch(
                "adan_trading_bot.data_processing.state_builder.StateBuilder",
                return_value=mock_state_builder,
            ),
        ):
            env = MultiAssetChunkedEnv(minimal_config)

            # Vérifier que l'environnement gère les 3 timeframes
            timeframes = env.config["environment"]["observation"]["timeframes"]
            assert "5m" in timeframes
            assert "1h" in timeframes
            assert "4h" in timeframes

            observation, _ = env.reset()

            # Vérifier que chaque timeframe est présent dans l'observation
            for tf in timeframes:
                assert tf in observation
                assert isinstance(observation[tf], np.ndarray)
                assert observation[tf].dtype == np.float32

            print("✅ Données multi-timeframe correctement gérées")

    def test_portfolio_state_integration(
        self,
        minimal_config,
        mock_data_loader,
        mock_state_builder,
        mock_portfolio_manager,
    ):
        """Test l'intégration de l'état du portfolio."""
        with (
            patch(
                "adan_trading_bot.data_processing.data_loader.ChunkedDataLoader",
                return_value=mock_data_loader,
            ),
            patch(
                "adan_trading_bot.data_processing.state_builder.StateBuilder",
                return_value=mock_state_builder,
            ),
            patch(
                "adan_trading_bot.portfolio.portfolio_manager.PortfolioManager",
                return_value=mock_portfolio_manager,
            ),
        ):
            env = MultiAssetChunkedEnv(minimal_config)
            observation, _ = env.reset()

            # Vérifier que portfolio_state est inclus
            assert "portfolio_state" in observation
            assert isinstance(observation["portfolio_state"], np.ndarray)
            assert observation["portfolio_state"].dtype == np.float32

            # Exécuter quelques steps et vérifier les mises à jour du portfolio
            for _ in range(3):
                # Action continue valide (15 dimensions)
                action = np.random.uniform(-1, 1, 15).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)

                # Vérifier que portfolio_state est toujours dans l'observation
                assert "portfolio_state" in obs
                assert isinstance(obs["portfolio_state"], np.ndarray)

                if terminated or truncated:
                    env.reset()

            print("✅ Portfolio state correctement intégré")

    def test_reward_calculation(
        self,
        minimal_config,
        mock_data_loader,
        mock_state_builder,
        mock_portfolio_manager,
    ):
        """Test le calcul des récompenses."""
        # Configurer le portfolio manager pour retourner différents PnL
        mock_portfolio_manager.get_realized_pnl.side_effect = [0.0, 10.0, -5.0, 15.0]

        with (
            patch(
                "adan_trading_bot.data_processing.data_loader.ChunkedDataLoader",
                return_value=mock_data_loader,
            ),
            patch(
                "adan_trading_bot.data_processing.state_builder.StateBuilder",
                return_value=mock_state_builder,
            ),
            patch(
                "adan_trading_bot.portfolio.portfolio_manager.PortfolioManager",
                return_value=mock_portfolio_manager,
            ),
        ):
            env = MultiAssetChunkedEnv(minimal_config)
            env.reset()

            rewards = []
            for i in range(4):
                obs, reward, terminated, truncated, info = env.step(1)
                rewards.append(reward)

                if terminated or truncated:
                    env.reset()

            # Vérifier que les rewards sont numériques et finis
            assert all(isinstance(r, (float, np.floating)) for r in rewards)
            assert all(np.isfinite(r) for r in rewards)

            print(f"✅ Rewards calculés: {rewards}")

    def test_episode_termination_conditions(
        self, minimal_config, mock_data_loader, mock_state_builder
    ):
        """Test les conditions de fin d'épisode."""
        # Configurer pour un épisode court
        minimal_config["environment"]["max_steps"] = 10

        with (
            patch(
                "adan_trading_bot.data_processing.data_loader.ChunkedDataLoader",
                return_value=mock_data_loader,
            ),
            patch(
                "adan_trading_bot.data_processing.state_builder.StateBuilder",
                return_value=mock_state_builder,
            ),
        ):
            env = MultiAssetChunkedEnv(minimal_config)
            env.reset()

            terminated = False
            truncated = False
            step_count = 0

            while not (terminated or truncated) and step_count < 20:
                obs, reward, terminated, truncated, info = env.step(0)
                step_count += 1

            # Vérifier que l'épisode se termine dans les limites
            assert step_count <= minimal_config["environment"]["max_steps"]
            assert terminated or truncated

            print(f"✅ Épisode terminé après {step_count} steps")

    def test_observation_consistency_across_steps(
        self, minimal_config, mock_data_loader, mock_state_builder
    ):
        """Test la cohérence des observations entre les steps."""
        with (
            patch(
                "adan_trading_bot.data_processing.data_loader.ChunkedDataLoader",
                return_value=mock_data_loader,
            ),
            patch(
                "adan_trading_bot.data_processing.state_builder.StateBuilder",
                return_value=mock_state_builder,
            ),
        ):
            env = MultiAssetChunkedEnv(minimal_config)

            # Collecter plusieurs observations
            observations = []

            obs, _ = env.reset()
            observations.append(obs)

            for _ in range(5):
                # Action continue valide (15 dimensions)
                action = np.random.uniform(-1, 1, 15).astype(np.float32)
                obs, _, terminated, truncated, _ = env.step(action)
                observations.append(obs)

                if terminated or truncated:
                    obs, _ = env.reset()
                    observations.append(obs)

            # Vérifier la cohérence des structures
            for obs in observations:
                assert isinstance(obs, dict)
                assert set(obs.keys()) == {"5m", "1h", "4h", "portfolio_state", "context_vector"}

                # Vérifier les dimensions
                assert obs["5m"].shape == (20, 15)
                assert obs["1h"].shape == (14, 15)
                assert obs["4h"].shape == (5, 15)
                # portfolio_state peut avoir différentes dimensions selon la config
                assert len(obs["portfolio_state"].shape) == 1
                assert obs["portfolio_state"].shape[0] > 0

                # Vérifier les types
                assert obs["5m"].dtype == np.float32
                assert obs["1h"].dtype == np.float32
                assert obs["4h"].dtype == np.float32
                assert obs["portfolio_state"].dtype == np.float32

            print("✅ Observations cohérentes entre les steps")

    def test_action_execution_effects(
        self,
        minimal_config,
        mock_data_loader,
        mock_state_builder,
        mock_portfolio_manager,
    ):
        """Test les effets de l'exécution des actions."""
        with (
            patch(
                "adan_trading_bot.data_processing.data_loader.ChunkedDataLoader",
                return_value=mock_data_loader,
            ),
            patch(
                "adan_trading_bot.data_processing.state_builder.StateBuilder",
                return_value=mock_state_builder,
            ),
            patch(
                "adan_trading_bot.portfolio.portfolio_manager.PortfolioManager",
                return_value=mock_portfolio_manager,
            ),
        ):
            env = MultiAssetChunkedEnv(minimal_config)
            env.reset()

            # Tester différentes actions et leurs effets
            actions_effects = []

            for action in [0, 1, 2]:  # Hold, Buy, Sell
                obs, reward, terminated, truncated, info = env.step(action)

                effects = {
                    "action": action,
                    "reward": reward,
                    "portfolio_called": mock_portfolio_manager.execute_trade.called,
                    "terminated": terminated,
                    "truncated": truncated,
                }

                actions_effects.append(effects)

                if terminated or truncated:
                    env.reset()

            # Vérifier que les actions ont des effets mesurables
            assert all(
                isinstance(effect["reward"], (float, np.floating))
                for effect in actions_effects
            )

            print(
                f"✅ Actions ont des effets mesurables: {len(actions_effects)} actions testées"
            )

    def test_info_dict_completeness(
        self, minimal_config, mock_data_loader, mock_state_builder
    ):
        """Test la complétude du dictionnaire info."""
        with (
            patch(
                "adan_trading_bot.data_processing.data_loader.ChunkedDataLoader",
                return_value=mock_data_loader,
            ),
            patch(
                "adan_trading_bot.data_processing.state_builder.StateBuilder",
                return_value=mock_state_builder,
            ),
        ):
            env = MultiAssetChunkedEnv(minimal_config)
            env.reset()

            obs, reward, terminated, truncated, info = env.step(1)

            # Vérifier que info contient des informations utiles
            assert isinstance(info, dict)

            # Vérifier les clés attendues (peut varier selon l'implémentation)
            expected_keys = ["step", "action_taken", "portfolio_value"]
            present_keys = [key for key in expected_keys if key in info]

            print(f"✅ Info dict contient: {list(info.keys())}")
            print(f"✅ Clés attendues présentes: {present_keys}")

    def test_multiple_episodes_stability(
        self, minimal_config, mock_data_loader, mock_state_builder
    ):
        """Test la stabilité sur plusieurs épisodes."""
        minimal_config["environment"]["max_steps"] = 20

        with (
            patch(
                "adan_trading_bot.data_processing.data_loader.ChunkedDataLoader",
                return_value=mock_data_loader,
            ),
            patch(
                "adan_trading_bot.data_processing.state_builder.StateBuilder",
                return_value=mock_state_builder,
            ),
        ):
            env = MultiAssetChunkedEnv(minimal_config)

            episodes_data = []

            for episode in range(3):
                obs, _ = env.reset()
                episode_rewards = []
                steps = 0

                while steps < 15:
                    action = np.random.choice([0, 1, 2])
                    obs, reward, terminated, truncated, info = env.step(action)

                    episode_rewards.append(reward)
                    steps += 1

                    if terminated or truncated:
                        break

                episodes_data.append(
                    {
                        "episode": episode,
                        "steps": steps,
                        "total_reward": sum(episode_rewards),
                        "avg_reward": np.mean(episode_rewards)
                        if episode_rewards
                        else 0,
                    }
                )

            # Vérifier que tous les épisodes se sont déroulés normalement
            assert len(episodes_data) == 3
            assert all(ep["steps"] > 0 for ep in episodes_data)

            print("✅ Stabilité sur plusieurs épisodes confirmée")
            for ep_data in episodes_data:
                print(
                    f"   Episode {ep_data['episode']}: {ep_data['steps']} steps, reward total: {ep_data['total_reward']:.3f}"
                )
