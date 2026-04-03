#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Online Learning Agent for continuous improvement of the trading model.
"""
import os
import time
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Union, Deque, Any, NamedTuple, Callable, Type, TypeVar
from collections import deque, namedtuple, defaultdict
import gymnasium as gym
import random
import math
import copy
from dataclasses import dataclass, field
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv as AdanTradingEnv
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.buffers import ReplayBuffer, RolloutBuffer, DictReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file
from stable_baselines3.common.utils import get_schedule_fn, update_learning_rate, get_linear_fn, safe_mean

from .common.utils import get_logger, load_config, create_directories
from .environment.multi_asset_chunked_env import MultiAssetChunkedEnv

try:
    from .agent.feature_extractors import CustomCNNFeatureExtractor
except ImportError:
    CustomCNNFeatureExtractor = None

logger = get_logger()

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer for efficient learning.

    Supports both flat numpy arrays *and* complex dict observations
    (e.g. {'5m': ..., '1h': ..., '4h': ..., 'portfolio_state': ..., 'context_vector': ...}).
    Dict states are stored as-is; sampling stacks each key independently.
    """

    def __init__(self, buffer_size: int, batch_size: int, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001, epsilon: float = 1e-6):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done", "td_error"])
        self.memory: list = []
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.pos = 0
        self.full = False

    # ------------------------------------------------------------------
    # helpers for dict observations
    # ------------------------------------------------------------------
    @staticmethod
    def _to_numpy(obj):
        """Convert a state (dict of arrays or plain array) to a storable form."""
        if isinstance(obj, dict):
            return {k: np.asarray(v) for k, v in obj.items()}
        return np.asarray(obj)

    @staticmethod
    def _stack_states(states):
        """Stack a list of states.  If dict, stack per-key; otherwise np.vstack."""
        if len(states) == 0:
            return states
        if isinstance(states[0], dict):
            keys = states[0].keys()
            return {k: np.stack([s[k] for s in states]) for k in keys}
        return np.vstack(states)

    # ------------------------------------------------------------------
    def add(self, state, action, reward, next_state, done,
            td_error: Optional[float] = None) -> None:
        """Add a new experience to memory (supports dict states)."""
        if td_error is None:
            max_priority = float(self.priorities.max()) if not self.is_empty() else 1.0
            td_error = max_priority

        e = self.experience(
            self._to_numpy(state),
            np.asarray(action),
            float(reward),
            self._to_numpy(next_state),
            bool(done),
            float(td_error),
        )

        if len(self.memory) < self.buffer_size:
            self.memory.append(e)
        else:
            self.memory[self.pos] = e

        priority = (abs(td_error) + self.epsilon) ** self.alpha
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True

    def sample(self):
        """Sample a batch.  Returns (states, actions, rewards, next_states, dones, indices, weights)."""
        if self.is_empty():
            raise ValueError("Cannot sample from an empty memory buffer")

        self.beta = min(1.0, self.beta + self.beta_increment)
        n = len(self.memory)
        priorities = self.priorities[:n] if not self.full else self.priorities
        probs = priorities / (priorities.sum() + 1e-12)

        batch_n = min(self.batch_size, n)
        indices = np.random.choice(n, size=batch_n, p=probs, replace=False)

        weights = (n * probs[indices]) ** (-self.beta)
        weights = weights / (weights.max() + 1e-12)

        experiences = [self.memory[idx] for idx in indices]

        states = self._stack_states([e.state for e in experiences])
        actions = np.vstack([np.atleast_1d(e.action) for e in experiences])
        rewards = np.array([e.reward for e in experiences], dtype=np.float32).reshape(-1, 1)
        next_states = self._stack_states([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences], dtype=np.float32).reshape(-1, 1)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + self.epsilon) ** self.alpha

    def is_empty(self) -> bool:
        return len(self.memory) == 0

    def __len__(self) -> int:
        return len(self.memory)

class ExperienceBuffer:
    """Buffer to store and sample experiences for online learning."""
    
    def __init__(self, buffer_size: int = 10000, batch_size: int = 64):
        """
        Initialize the experience buffer.
        
        Args:
            buffer_size: Maximum number of experiences to store
            batch_size: Number of experiences to sample in a batch
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer: Deque[Tuple] = deque(maxlen=buffer_size)
    
    def add(self, experience: Tuple) -> None:
        """Add a new experience to the buffer."""
        self.buffer.append(experience)
    
    def sample(self) -> List[Tuple]:
        """Sample a batch of experiences from the buffer."""
        if len(self.buffer) < self.batch_size:
            return list(self.buffer)
        return np.random.choice(self.buffer, self.batch_size, replace=False)
    
    def __len__(self) -> int:
        """Return the current number of experiences in the buffer."""
        return len(self.buffer)

class EWCRegularizer:
    """Elastic Weight Consolidation (EWC) regularizer to prevent catastrophic forgetting.

    Implements the diagonal Fisher approximation from Kirkpatrick et al. (2017).
    Explicitly covers **all** trainable parameters, including the new
    HierarchicalCrossAttention and FiLM layers in ContextualTemporalFusionExtractor.
    """

    # Layer-name substrings that belong to the cross-attention / FiLM modules.
    CROSS_ATTN_KEYS = ("cross_attention", "film_layer", "film_gen")

    def __init__(self, model: nn.Module, ewc_lambda: float = 0.1, fisher_samples: int = 100):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples

        # Snapshot *all* trainable params (including cross-attn + FiLM)
        self.params = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters() if p.requires_grad
        }
        self.fisher = {
            n: th.zeros_like(p)
            for n, p in self.model.named_parameters() if p.requires_grad
        }
        self.task_params = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters() if p.requires_grad
        }

        # Verify that cross-attention / FiLM layers are captured
        ca_count = sum(
            1 for n in self.params if any(k in n for k in self.CROSS_ATTN_KEYS)
        )
        logger.info(
            f"[EWC] Tracking {len(self.params)} parameters "
            f"({ca_count} from cross-attention/FiLM layers)"
        )

    # ------------------------------------------------------------------
    def _obs_to_tensor(self, obs):
        """Convert obs (dict or array) to the tensor format the model expects."""
        device = next(self.model.parameters()).device
        if isinstance(obs, dict):
            return {k: th.as_tensor(v, device=device).float().unsqueeze(0)
                    if not isinstance(v, th.Tensor) else v.float().unsqueeze(0)
                    for k, v in obs.items()}
        t = th.as_tensor(obs, device=device).float()
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t

    def compute_fisher(self, env: GymEnv, num_samples: Optional[int] = None) -> None:
        """Compute diagonal Fisher information using the current policy."""
        if num_samples is None:
            num_samples = self.fisher_samples

        for n in self.fisher:
            self.fisher[n].zero_()

        self.model.eval()
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result

        for _ in range(num_samples):
            obs_tensor = self._obs_to_tensor(obs)
            try:
                dist = self.model.get_distribution(obs_tensor)
            except Exception:
                break
            action = dist.sample()
            log_prob = dist.log_prob(action)
            if log_prob.dim() > 1:
                log_prob = log_prob.sum(dim=-1)
            log_prob = log_prob.mean()

            self.model.zero_grad()
            log_prob.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None and n in self.fisher:
                    self.fisher[n] += p.grad.detach() ** 2 / num_samples

            try:
                result = env.step(action.detach().cpu().numpy())
                obs = result[0] if isinstance(result, tuple) else result
                done = result[2] if isinstance(result, tuple) else False
                if done if isinstance(done, bool) else bool(np.any(done)):
                    result = env.reset()
                    obs = result[0] if isinstance(result, tuple) else result
            except Exception:
                break

    def penalty(self) -> th.Tensor:
        """EWC penalty — covers cross-attention + FiLM parameters automatically."""
        penalty = th.tensor(0.0, device=next(self.model.parameters()).device)
        for n, p in self.model.named_parameters():
            if n in self.fisher and n in self.task_params:
                penalty = penalty + (self.fisher[n] * (p - self.task_params[n]) ** 2).sum()
        return 0.5 * self.ewc_lambda * penalty

    def update_task_params(self) -> None:
        """Snapshot current parameters as reference for the next penalty computation."""
        self.task_params = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters() if p.requires_grad
        }

class OnlineLearningCallback(BaseCallback):
    """Callback for online learning that updates the model with new experiences."""
    
    def __init__(
        self,
        experience_buffer: 'PrioritizedReplayBuffer',
        learn_every: int = 100,  # Learn every n steps
        learning_rate: float = 1e-5,  # Small learning rate for fine-tuning
        ewc_lambda: float = 0.1,  # EWC regularization strength
        use_ewc: bool = True,  # Whether to use EWC
        target_update_freq: int = 1000,  # Update target network every n steps
        verbose: int = 0
    ):
        """
        Initialize the online learning callback.
        
        Args:
            experience_buffer: Buffer containing experiences
            learn_every: Number of steps between learning updates
            learning_rate: Learning rate for fine-tuning
            ewc_lambda: Strength of EWC regularization
            use_ewc: Whether to use EWC to prevent catastrophic forgetting
            target_update_freq: Frequency of target network updates
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.experience_buffer = experience_buffer
        self.learn_every = learn_every
        self.learning_rate = learning_rate
        self.ewc_lambda = ewc_lambda
        self.use_ewc = use_ewc
        self.target_update_freq = target_update_freq
        self.num_updates = 0
        self.optimizer = None
        self.ewc = None
        self.target_network = None
    
    def _on_step(self) -> bool:
        """Called at each environment step."""
        # Add experience to buffer
        if len(self.model.rollout_buffer.observations) > 0:
            # Get the last transition
            obs = self.model.rollout_buffer.observations[-1:]
            next_obs = self.model.rollout_buffer.next_observations[-1:]
            actions = self.model.rollout_buffer.actions[-1:]
            rewards = self.model.rollout_buffer.rewards[-1:]
            dones = self.model.rollout_buffer.dones[-1:]
            
            # Compute TD error for prioritized experience replay
            with th.no_grad():
                obs_tensor = th.FloatTensor(obs).to(self.model.device)
                next_obs_tensor = th.FloatTensor(next_obs).to(self.model.device)
                actions_tensor = th.FloatTensor(actions).to(self.model.device)
                
                # Get current Q values
                current_q = th.min(
                    *self.model.critic(obs_tensor, actions_tensor)
                )
                
                # Get next actions and Q values from target network
                next_actions, _ = self.model.policy.actor.action_log_prob(next_obs_tensor)
                next_q = th.min(
                    *self.model.critic_target(next_obs_tensor, next_actions)
                )
                
                # Compute target Q values
                target_q = th.FloatTensor(rewards).to(self.model.device) + \
                          (1 - th.FloatTensor(dones).to(self.model.device)) * self.model.gamma * next_q
                
                # Compute TD error
                td_error = (current_q - target_q).abs().cpu().numpy()
            
            # Add to experience buffer with TD error
            self.experience_buffer.add(obs[0], actions[0], rewards[0], next_obs[0], dones[0], td_error[0])
        
        # Learn from experiences periodically
        if self.num_timesteps % self.learn_every == 0 and len(self.experience_buffer) > 0:
            self._learn_from_experiences()
        
        # Update target network periodically
        if self.num_timesteps % self.target_update_freq == 0 and hasattr(self.model, '_update_target_networks'):
            self.model._update_target_networks()
        
        return True
    
    def _init_callback(self) -> None:
        """Initialize the callback."""
        super()._init_callback()
        
        # Initialize optimizer if not already done
        if self.optimizer is None and hasattr(self.model, 'policy') and hasattr(self.model.policy, 'parameters'):
            self.optimizer = optim.Adam(self.model.policy.parameters(), lr=self.learning_rate)
        
        # Initialize EWC if enabled
        if self.use_ewc and self.ewc is None and hasattr(self.model, 'policy'):
            self.ewc = EWCRegularizer(self.model.policy, ewc_lambda=self.ewc_lambda)
            # Compute Fisher information on the current task
            if hasattr(self, 'training_env'):
                self.ewc.compute_fisher(self.training_env)
            self.ewc.update_task_params()
    
    def _learn_from_experiences(self) -> None:
        """Update the model using experiences from the buffer."""
        try:
            # Sample a batch of experiences
            batch = self.experience_buffer.sample()
            if not batch:
                return
                
            # Unpack the batch
            obs, actions, rewards, next_obs, dones = zip(*batch)
            obs = np.array(obs)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_obs = np.array(next_obs)
            dones = np.array(dones)
            
            # Convert to tensors
            obs_tensor = th.FloatTensor(obs).to(self.model.device)
            actions_tensor = th.FloatTensor(actions).to(self.model.device)
            rewards_tensor = th.FloatTensor(rewards).to(self.model.device).unsqueeze(1)
            next_obs_tensor = th.FloatTensor(next_obs).to(self.model.device)
            dones_tensor = th.FloatTensor(dones).to(self.model.device).unsqueeze(1)
            
            # Store current learning rate
            current_lr = self.model.learning_rate
            
            try:
                # Temporarily set learning rate for fine-tuning
                self.model.learning_rate = self.learning_rate
                
                # Forward pass
                values, log_prob, entropy = self.model.policy.evaluate_actions(
                    obs_tensor, actions_tensor
                )
                
                # Calculate advantages (simple TD(0) for simplicity)
                with th.no_grad():
                    next_values = self.model.policy.predict_values(next_obs_tensor)
                    advantages = rewards_tensor + (1 - dones_tensor) * self.model.gamma * next_values - values
                
                # Calculate policy loss
                policy_loss = -(log_prob * advantages.detach()).mean()
                
                # Value function loss
                value_loss = advantages.pow(2).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # EWC penalty if enabled
                ewc_penalty = 0.0
                if self.use_ewc and self.ewc is not None:
                    ewc_penalty = self.ewc.penalty()
                
                # Total loss
                loss = (
                    policy_loss 
                    + self.model.vf_coef * value_loss 
                    + self.model.ent_coef * entropy_loss
                    + ewc_penalty
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                th.nn.utils.clip_grad_norm_(
                    self.model.policy.parameters(), 
                    self.model.max_grad_norm
                )
                
                # Update parameters
                self.optimizer.step()
                
                self.num_updates += 1
                
                if self.verbose >= 1:
                    logger.info(
                        f"Online update {self.num_updates} - "
                        f"Policy Loss: {policy_loss.item():.4f}, "
                        f"Value Loss: {value_loss.item():.4f}, "
                        f"Entropy: {entropy.mean().item():.4f}"
                    )
                
                # Log to TensorBoard if available
                if self.model.logger is not None:
                    self.model.logger.record("online/policy_loss", policy_loss.item())
                    self.model.logger.record("online/value_loss", value_loss.item())
                    self.model.logger.record("online/entropy", entropy.mean().item())
                    self.model.logger.record("online/learning_rate", self.learning_rate)
                    if self.use_ewc and self.ewc is not None:
                        self.model.logger.record("online/ewc_penalty", ewc_penalty.item())
                    self.model.logger.dump(step=self.num_updates)
                
            except Exception as e:
                logger.error(f"Error during online learning: {str(e)}", exc_info=True)
                
            finally:
                # Restore original learning rate
                self.model.learning_rate = current_lr
                
        except Exception as e:
            logger.error(f"Error in _learn_from_experiences: {str(e)}", exc_info=True)

    
class OnlineLearningAgent:
    """
    Agent for online learning that can continuously improve a pre-trained model.
    
    This agent implements several advanced techniques for stable and efficient online learning:
    - Prioritized Experience Replay: Samples important transitions more frequently
    - Elastic Weight Consolidation (EWC): Prevents catastrophic forgetting
    - Target Network Updates: Stabilizes learning by using a separate target network
    
    The agent can be configured with various hyperparameters to balance exploration,
    exploitation, and computational efficiency.
    """
    
    def __init__(
        self,
        model: PPO,
        env: GymEnv,
        config: Dict[str, Any],
        experience_buffer: Optional[Union[PrioritizedReplayBuffer, ReplayBuffer]] = None,
        experience_buffer_size: int = 10000,
        batch_size: int = 64,
        learn_every: int = 100,
        learning_rate: float = 1e-5,
        save_freq: int = 1000,
        save_path: Optional[str] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 1,
        use_prioritized_replay: bool = True,
        prioritized_replay_alpha: float = 0.6,
        prioritized_replay_beta: float = 0.4,
        prioritized_replay_eps: float = 1e-6,
        use_ewc: bool = True,
        ewc_lambda: float = 0.1,
        target_update_freq: int = 1000
    ):
        """
        Initialize the online learning agent.
        
        Args:
            model: Pre-trained PPO model
            env: Environment to interact with
            config: Configuration dictionary
            experience_buffer_size: Maximum size of the experience replay buffer
            batch_size: Number of experiences to sample for each update
            learn_every: Number of steps between learning updates
            learning_rate: Learning rate for online updates
            save_freq: Frequency (in steps) to save the model
            save_path: Directory to save models and logs
            tensorboard_log: Directory for TensorBoard logs
            verbose: Verbosity level (0: no output, 1: info, 2: debug)
            use_prioritized_replay: Whether to use prioritized experience replay
            prioritized_replay_alpha: Alpha parameter for prioritized replay (controls prioritization, 0=uniform)
            prioritized_replay_beta: Beta parameter for prioritized replay (controls importance sampling)
            prioritized_replay_eps: Epsilon to add to priorities to avoid zero probabilities
            use_ewc: Whether to use Elastic Weight Consolidation
            ewc_lambda: Strength of the EWC regularization
            target_update_freq: Frequency (in steps) to update the target network
        """
        self.model = model
        self.env = env
        self.config = config
        self.learn_every = learn_every
        self.learning_rate = learning_rate
        self.save_freq = save_freq
        self.save_path = save_path
        self.verbose = verbose
        
        # Create save directory if it doesn't exist
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        
        # Store parameters
        self.model = model
        self.env = env
        self.config = config
        self.learn_every = learn_every
        self.learning_rate = learning_rate
        self.save_freq = save_freq
        self.save_path = save_path
        self.verbose = verbose
        self.use_prioritized_replay = use_prioritized_replay
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta = prioritized_replay_beta
        self.prioritized_replay_eps = prioritized_replay_eps
        self.use_ewc = use_ewc
        self.ewc_lambda = ewc_lambda
        self.target_update_freq = target_update_freq
        self.experience_buffer = experience_buffer or self._create_experience_buffer(experience_buffer_size, batch_size, use_prioritized_replay, prioritized_replay_alpha, prioritized_replay_beta, prioritized_replay_eps, env, model)
        
        # Create save directory if it doesn't exist
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        
        
        
        # Setup callbacks
        self.callbacks = []
        
        # Add online learning callback
        self.online_learning_callback = OnlineLearningCallback(
            experience_buffer=self.experience_buffer,
            learn_every=self.learn_every,
            learning_rate=self.learning_rate,
            ewc_lambda=self.ewc_lambda,
            use_ewc=self.use_ewc,
            target_update_freq=self.target_update_freq,
            verbose=verbose
        )
        self.callbacks.append(self.online_learning_callback)
        
        # Add checkpoint callback if save_path is provided
        if self.save_path is not None:
            checkpoint_callback = CheckpointCallback(
                save_freq=self.save_freq,
                save_path=self.save_path,
                name_prefix="online_model",
                save_replay_buffer=True,
                save_vecnormalize=True
            )
            self.callbacks.append(checkpoint_callback)
            
        # Initialize training state
        self.episode_reward = 0
        self.episode_length = 0
        self.total_steps = 0
        self.episode_count = 0
        self.start_time = time.time()
        
        # Initialize environment state
        self.current_obs = self.env.reset()
        self.episode_reward = 0
        self.episode_length = 0
        self.total_steps = 0
        self.episode_count = 0
        self.start_time = time.time()
    
    def run(self, num_steps: Optional[int] = None) -> None:
        """
        Run the online learning loop.
        
        Args:
            num_steps: Number of steps to run for (None for infinite loop)
        """
        try:
            step = 0
            
            while num_steps is None or step < num_steps:
                # Get action from policy
                action, _ = self.model.predict(
                    self.current_obs,
                    deterministic=False
                )
                
                # Take a step in the environment
                next_obs, reward, done, info = self.env.step(action)
                
                # Store experience in buffer
                self.experience_buffer.add(
                    (self.current_obs, action, reward, next_obs, done)
                )
                
                # Update statistics
                self.episode_reward += reward
                self.episode_length += 1
                self.total_steps += 1
                
                # Call callbacks
                for callback in self.callbacks:
                    callback.locals.update({
                        'self': self,
                        'action': action,
                        'reward': reward,
                        'done': done,
                        'info': info
                    })
                    callback.on_step()
                
                # Update current observation
                self.current_obs = next_obs
                
                # Handle episode end
                if done:
                    if self.verbose >= 1:
                        logger.info(
                            f"Episode {self.episode_count + 1} - "
                            f"Reward: {self.episode_reward:.2f}, "
                            f"Length: {self.episode_length}, "
                            f"Total Steps: {self.total_steps}"
                        )
                    
                    # Log to TensorBoard if available
                    if self.model.logger is not None:
                        self.model.logger.record("rollout/ep_rew_mean", self.episode_reward)
                        self.model.logger.record("rollout/ep_len_mean", self.episode_length)
                        self.model.logger.record("time/episodes", self.episode_count)
                        self.model.logger.record("time/steps_per_second", 
                                              self.total_steps / (time.time() - self.start_time))
                        self.model.logger.dump(step=self.total_steps)
                    
                    # Reset environment
                    self.current_obs = self.env.reset()
                    self.episode_reward = 0
                    self.episode_length = 0
                    self.episode_count += 1
                
                step += 1
                
        except KeyboardInterrupt:
            logger.info("Online learning interrupted by user.")
        
        finally:
            # Save final model
            if self.save_path is not None:
                final_path = os.path.join(self.save_path, "final_model")
                self.model.save(final_path)
                logger.info(f"Final model saved to {final_path}")
            
            # Close environment
            self.env.close()

    def _create_experience_buffer(self, buffer_size, batch_size, use_prioritized_replay, alpha, beta, eps, env, model):
        if use_prioritized_replay:
            return PrioritizedReplayBuffer(
                buffer_size=buffer_size,
                batch_size=batch_size,
                alpha=alpha,
                beta=beta,
                epsilon=eps
            )
        else:
            return ReplayBuffer(
                buffer_size=buffer_size,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=model.device,
                n_envs=env.num_envs if hasattr(env, 'num_envs') else 1,
                optimize_memory_usage=True
            )

def create_online_learning_agent(
    model_path: str,
    env_config: Dict[str, Any],
    online_config: Dict[str, Any],
    tensorboard_log: Optional[str] = None,
    verbose: int = 1
) -> OnlineLearningAgent:
    """
    Create an online learning agent from a pre-trained model.
    
    Args:
        model_path: Path to the pre-trained model
        env_config: Environment configuration
        online_config: Online learning configuration
        tensorboard_log: Directory for TensorBoard logs
        verbose: Verbosity level
        
    Returns:
        Configured OnlineLearningAgent instance
    """
    # Create the environment
    env = DummyVecEnv([lambda: AdanTradingEnv(config=env_config)])
    
    # Load pre-trained model
    model = PPO.load(model_path, env=env, tensorboard_log=tensorboard_log, verbose=verbose)
    
    # Create the agent with enhanced configuration
    agent = OnlineLearningAgent(
        model=model,
        env=env,
        config=online_config,
        experience_buffer_size=online_config.get('experience_buffer_size', 10000),
        batch_size=online_config.get('batch_size', 64),
        learn_every=online_config.get('learn_every', 100),
        learning_rate=online_config.get('learning_rate', 1e-5),
        save_freq=online_config.get('save_freq', 1000),
        save_path=online_config.get('save_path'),
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        use_prioritized_replay=online_config.get('use_prioritized_replay', True),
        prioritized_replay_alpha=online_config.get('prioritized_replay_alpha', 0.6),
        prioritized_replay_beta=online_config.get('prioritized_replay_beta', 0.4),
        prioritized_replay_eps=online_config.get('prioritized_replay_eps', 1e-6),
        use_ewc=online_config.get('use_ewc', True),
        ewc_lambda=online_config.get('ewc_lambda', 0.1),
        target_update_freq=online_config.get('target_update_freq', 1000)
    )
    
    return agent

def main():
    """Main function to run the online learning agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run online learning for the ADAN trading bot')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to the pre-trained model')
    parser.add_argument('--config', type=str, default='config/online_config.yaml',
                      help='Path to the online learning configuration file')
    parser.add_argument('--env-config', type=str, default='config/environment_config.yaml',
                      help='Path to the environment configuration file')
    parser.add_argument('--save-path', type=str, default='models/online',
                      help='Directory to save the fine-tuned models')
    parser.add_argument('--tensorboard-log', type=str, default='logs/online',
                      help='Directory for TensorBoard logs')
    parser.add_argument('--steps', type=int, default=None,
                      help='Number of steps to run (default: run until interrupted)')
    parser.add_argument('--verbose', type=int, default=1,
                      help='Verbosity level (0: no output, 1: info, 2: debug)')
    
    args = parser.parse_args()
    
    # Load configurations
    online_config = load_config(args.config)
    env_config = load_config(args.env_config)
    
    # Ensure save path exists
    os.makedirs(args.save_path, exist_ok=True)
    
    # Ensure tensorboard log directory exists
    if args.tensorboard_log:
        os.makedirs(args.tensorboard_log, exist_ok=True)
    
    # Update online config with command line arguments
    online_config['save_path'] = args.save_path
    
    # Create and run the online learning agent
    agent = create_online_learning_agent(
        model_path=args.model_path,
        env_config=env_config,
        online_config=online_config,
        tensorboard_log=args.tensorboard_log,
        verbose=args.verbose
    )
    
    try:
        logger.info("Starting online learning...")
        logger.info(f"Configuration: {online_config}")
        
        # Run the agent
        agent.run(num_steps=args.steps)
        
    except KeyboardInterrupt:
        logger.info("Online learning stopped by user.")
        
    except Exception as e:
        logger.error(f"Error during online learning: {str(e)}", exc_info=True)
        
    finally:
        try:
            # Save final model
            final_path = os.path.join(args.save_path, "final_model")
            agent.model.save(final_path)
            logger.info(f"Final model saved to {final_path}")
            
            # Save the experience buffer if it exists
            if hasattr(agent, 'experience_buffer') and hasattr(agent.experience_buffer, 'save'):
                buffer_path = os.path.join(args.save_path, "experience_buffer.pkl")
                agent.experience_buffer.save(buffer_path)
                logger.info(f"Experience buffer saved to {buffer_path}")
                
        except Exception as e:
            logger.error(f"Error saving final model or experience buffer: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
