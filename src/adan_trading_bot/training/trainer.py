#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training module for the ADAN trading bot with support for CNN feature extraction.
"""
import argparse
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback, EvalCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv, VecFrameStack, VecNormalize
)

import adan_trading_bot.agent.ppo_agent as ppo_agent
import adan_trading_bot.common.utils as utils
from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from adan_trading_bot.environment.multi_asset_chunked_env import (
    MultiAssetChunkedEnv
)
from adan_trading_bot.models.feature_extractors import CustomCNNFeatureExtractor
from adan_trading_bot.utils.timeout_manager import (
    TimeoutManager,
    TimeoutException,
)

# Configure logger
logger = utils.get_logger()


def validate_environment() -> None:
    """Basic environment validation before training starts."""
    import sys
    try:
        import gymnasium  # noqa: F401
        import stable_baselines3  # noqa: F401
    except Exception as e:
        raise RuntimeError(f"Missing training dependency: {e}")
    if sys.version_info < (3, 11):
        raise RuntimeError("Python >= 3.11 is required for training")


class CustomActorCriticPolicy(ActorCriticPolicy):
    """Custom policy with CNN feature extractor for 3D observations."""

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs
    ):
        # Custom feature extractor for 3D observations
        features_extractor_class = kwargs.pop(
            'features_extractor_class', CustomCNNFeatureExtractor
        )
        features_extractor_kwargs = kwargs.pop('features_extractor_kwargs', {})

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch or [dict(pi=[256, 128], vf=[256, 128])],
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            *args,
            **kwargs
        )


class TrainingConfig:
    """Configuration class for model training."""

    def __init__(self, config: Dict[str, Any]):
        # Training parameters
        self.total_timesteps = config.get('total_timesteps', 1_000_000)
        self.n_steps = config.get('n_steps', 2048)
        self.batch_size = config.get('batch_size', 64)
        self.n_epochs = config.get('n_epochs', 10)
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_range = config.get('clip_range', 0.2)
        self.ent_coef = config.get('ent_coef', 0.0)
        self.vf_coef = config.get('vf_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)

        # Environment parameters
        self.n_envs = config.get('n_envs', 1)
        self.use_frame_stack = config.get('use_frame_stack', True)
        self.frame_stack = config.get('frame_stack', 4)
        self.normalize = config.get('normalize', True)

        # Model saving
        self.save_freq = config.get('save_freq', 10000)
        self.eval_freq = config.get('eval_freq', 10000)
        self.best_model_save_path = config.get(
            'best_model_save_path', 'models/best'
        )
        self.log_path = config.get('log_path', 'logs')
        self.tensorboard_log = config.get('tensorboard_log', 'logs/tensorboard')

        # Random seed for reproducibility
        self.seed = config.get('seed', 42)

        # Device (GPU if available, else CPU)
        self.device = 'cuda' if th.cuda.is_available() and config.get(
            'use_gpu', True
        ) else 'cpu'

        # Create necessary directories
        utils.create_directories([
            self.best_model_save_path, self.log_path, self.tensorboard_log
        ])

        # Set random seeds for reproducibility
        set_random_seed(self.seed)
        th.manual_seed(self.seed)
        np.random.seed(self.seed)

        if th.cuda.is_available():
            th.backends.cudnn.deterministic = True
            th.backends.cudnn.benchmark = False


def create_envs(
    config: TrainingConfig,
    env_config: Dict[str, Any]
) -> Tuple[GymEnv, GymEnv]:
    """
    Create training and evaluation environments.

    Args:
        config: Training configuration.
        env_config: Environment configuration.

    Returns:
        A tuple of (train_env, eval_env).
    """
    # Create the training data loader
    train_loader = ChunkedDataLoader(
        data_dir=env_config['data']['data_dir'],
        assets_list=env_config['data']['assets'],
        timeframes=env_config['data']['timeframes'],
        features_by_timeframe=env_config['data']['features_per_timeframe'],
        split='train',
        chunk_size=env_config['data']['chunk_size']
    )

    # Create the training environment
    train_env = DummyVecEnv([
        lambda: Monitor(
            MultiAssetChunkedEnv(
                data_loader=train_loader, config=env_config
            )
        )
    ] * config.n_envs)

    # Create the evaluation data loader
    eval_loader = ChunkedDataLoader(
        data_dir=env_config['data']['data_dir'],
        assets_list=env_config['data']['assets'],
        timeframes=env_config['data']['timeframes'],
        features_by_timeframe=env_config['data']['features_per_timeframe'],
        split='validation',
        chunk_size=env_config['data']['chunk_size']
    )

    # Create the evaluation environment
    eval_env = DummyVecEnv([
        lambda: Monitor(
            MultiAssetChunkedEnv(
                data_loader=eval_loader, config=env_config
            )
        )
    ])

    # Normalize observations and rewards with improved stability settings
    if config.normalize:
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,  # Clip observations to prevent extreme values
            clip_reward=10.0,  # Clip rewards to prevent extreme updates
            gamma=config.gamma,  # Use the same gamma as in training
            norm_obs_keys=None,  # Normalize all observations
            training=True  # Track running statistics during training
        )

        # Use the same normalization parameters for evaluation
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=config.gamma,
            norm_obs_keys=None,
            training=False  # Don't update running stats during evaluation
        )

        # Sync the observation normalization parameters
        eval_env.obs_rms = train_env.obs_rms
        eval_env.ret_rms = train_env.ret_rms

    # Add frame stacking if requested
    if config.use_frame_stack:
        train_env = VecFrameStack(train_env, n_stack=config.frame_stack)
        eval_env = VecFrameStack(eval_env, n_stack=config.frame_stack)

    return train_env, eval_env


def train_agent(
    config_path: str,
    custom_config: Optional[Dict[str, Any]] = None,
    callbacks: Optional[List[BaseCallback]] = None,
    timeout: Optional[float] = None,
) -> PPO:
    """
    Train a PPO agent with the given configuration.

    Args:
        config_path: Path to the configuration file.
        custom_config: Optional dictionary to override config values.
        callbacks: Optional list of callbacks for training.

    Returns:
        The trained PPO agent.
    """
    # Validate environment
    validate_environment()

    # Load and merge configuration
    config = utils.load_config(config_path)
    if custom_config:
        config.update(custom_config)

    # Create training configuration
    training_config = TrainingConfig(config)

    # Get environment configuration
    env_config = config.get('environment', {})

    # Create environments
    train_env, eval_env = create_envs(training_config, env_config)

    # Define policy kwargs for custom feature extractor
    policy_kwargs = config.get('policy_kwargs', {})

    # Create PPO agent
    agent = ppo_agent.create_ppo_agent(
        env=train_env,
        config=training_config,
        tensorboard_log=training_config.tensorboard_log,
        ent_coef=training_config.ent_coef,
        vf_coef=training_config.vf_coef,
        max_grad_norm=training_config.max_grad_norm,
        device=training_config.device,
        policy_kwargs=policy_kwargs,
        verbose=1
    )

    # Create callbacks
    callback_list = []

    # Add checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=training_config.save_freq,
        save_path=training_config.best_model_save_path,
        name_prefix='rl_model'
    )
    callback_list.append(checkpoint_callback)

    # Add evaluation callback if we have an evaluation environment
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=training_config.best_model_save_path,
            log_path=training_config.log_path,
            eval_freq=training_config.eval_freq,
            deterministic=True,
            render=False
        )
        callback_list.append(eval_callback)

    # Add any additional callbacks
    if callbacks:
        callback_list.extend(callbacks)

    # Train the agent (with optional timeout and graceful checkpoint)
    def _cleanup_on_timeout() -> None:
        try:
            save_dir = training_config.best_model_save_path
            utils.create_directories([save_dir])
            agent.save(os.path.join(save_dir, 'timeout_checkpoint'))
            logger.info("Checkpoint saved on timeout")
        except Exception:
            logger.exception("Failed to save checkpoint on timeout")

    if timeout and timeout > 0:
        tm = TimeoutManager(timeout=float(timeout), cleanup_callback=_cleanup_on_timeout)
        try:
            with tm.limit():
                agent.learn(
                    total_timesteps=training_config.total_timesteps,
                    callback=CallbackList(callback_list)
                )
        except TimeoutException:
            logger.warning("Training stopped due to timeout")
    else:
        agent.learn(
            total_timesteps=training_config.total_timesteps,
            callback=CallbackList(callback_list)
        )

    # Save the environment stats if normalization was used
    if training_config.normalize and hasattr(train_env, 'save'):
        vec_normalize_path = os.path.join(
            training_config.best_model_save_path, 'vec_normalize.pkl'
        )
        train_env.save(vec_normalize_path)
        logger.info(f"VecNormalize stats saved to {vec_normalize_path}")

    # Close environments
    train_env.close()
    if eval_env is not None:
        eval_env.close()

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train the ADAN trading bot with PPO'
    )
    parser.add_argument(
        '--config', type=str, default='config/train_config.yaml',
        help='Path to the training configuration file'
    )
    parser.add_argument(
        '--model-path', type=str, default='models/ppo_cnn',
        help='Path to save the trained model'
    )
    parser.add_argument(
        '--timesteps', type=int, default=1_000_000,
        help='Number of timesteps to train for'
    )
    parser.add_argument(
        '--gpu', action='store_true',
        help='Use GPU for training if available'
    )
    parser.add_argument(
        '--timeout', type=float, default=None,
        help='Maximum training duration in seconds (graceful stop with checkpoint)'
    )

    args = parser.parse_args()

    # Start training
    train_agent(
        config_path=args.config,
        custom_config={
            'total_timesteps': args.timesteps,
            'use_gpu': args.gpu,
            'models_dir': args.model_path
        },
        timeout=args.timeout
    )
