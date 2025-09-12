"""Module implémentant un agent PPO pour le trading algorithmique."""
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import MaybeCallback, Schedule
from stable_baselines3.common.vec_env import VecEnv

# Suppress unused imports for type checking
if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv  # noqa: F401


class CustomPPOPolicy(ActorCriticPolicy):
    """
    Politique personnalisée pour PPO avec vérification des NaN/Inf et journalisation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Sera mis à jour par l'agent
        self._num_timesteps = 0

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Passe avant avec vérification des NaN/Inf.
        """
        # Appel à la méthode forward du parent
        actions, values, log_prob = super().forward(obs, deterministic)

        # Vérification des NaN/Inf dans les sorties
        tensors_to_check = [
            ("features", self.features_extractor(obs)
             if hasattr(self, 'features_extractor') else None),
            ("logits", self.action_net(obs)
             if hasattr(self, 'action_net') else None),
            ("value", values),
            ("actions", actions),
            ("log_prob", log_prob)
        ]

        for name, tensor in tensors_to_check:
            if tensor is None:
                continue

            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                logger.error(
                    f"Policy produced NaN/Inf in {name} at step "
                    f"(num_timesteps {getattr(self, '_num_timesteps', 'N/A')})"
                )

                # Sauvegarde des tenseurs pour débogage
                try:
                    os.makedirs("nan_debug", exist_ok=True)
                    timestamp = int(time.time())
                    np.savez(
                        f"nan_debug/policy_nan_{name}_{timestamp}.npz",
                        arr=tensor.detach().cpu().numpy(),
                        obs=obs.detach().cpu().numpy() if hasattr(obs, 'detach') else obs
                    )
                    logger.info(f"Saved debug data to nan_debug/policy_nan_{name}_{timestamp}.npz")
                except Exception as e:
                    logger.error(f"Failed to save debug data: {e}")

                # Nettoyage pour éviter les plantages
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)

        return actions, values, log_prob

from ..common.utils import (
    ensure_dir_exists,
    get_logger,
    get_path,
    safe_serialize,
    sanitize_ppo_params,
)
from .feature_extractors import CustomCNNFeatureExtractor

logger = get_logger(__name__)

def _validate_param(
    value: Any,
    name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    default: Optional[float] = None,
) -> float:
    """Valide et nettoie un paramètre numérique.

    Args:
        value: Valeur à valider
        name: Nom du paramètre pour les messages d'erreur
        min_val: Valeur minimale autorisée (inclusive)
        max_val: Valeur maximale autorisée (inclusive)
        default: Valeur par défaut si la validation échoue

    Returns:
        float: La valeur validée ou la valeur par défaut
    """
    if value is None and default is not None:
        return float(default)
    if value is None:
        raise ValueError(f"{name} cannot be None and no default provided")
    try:
        value_float = float(value)
        if not np.isfinite(value_float):
            raise ValueError(f"{name} must be a finite number")
        if min_val is not None and value_float < min_val:
            logger.warning(
                "%s (%s) is below minimum value (%s), using %s",
                name, value_float, min_val, min_val
            )
            return float(min_val)
        if max_val is not None and value_float > max_val:
            logger.warning(
                "%s (%s) is above maximum value (%s), using %s",
                name, value_float, max_val, max_val
            )
            return float(max_val)
        return value_float
    except (TypeError, ValueError) as e:
        if default is not None:
            logger.warning(
                "Invalid %s: %s. Using default: %s",
                name, str(e), default
            )
            return float(default)
        raise ValueError(f"Invalid {name}: {e}")



def _safe_numpy(x: Any) -> Optional[np.ndarray]:
    """Convert input to numpy array safely, handling various types and NaN/Inf values.

    Args:
        x: Input to convert (tensor, numpy array, or other)
    Returns:
        Numpy array with NaN/Inf values replaced, or None if conversion fails
    """
    if x is None:
        return None
    try:
        if isinstance(x, (list, tuple)):
            x = np.array(x)
        elif torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        elif not isinstance(x, np.ndarray):
            x = np.array([x])
        # Replace NaN and Inf with finite values
        if (np.issubdtype(x.dtype, np.floating) or
                np.issubdtype(x.dtype, np.complexfloating)):
            x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        return x
    except Exception as e:
        logger.warning("Failed to convert to numpy: %s", str(e))
        return None

class NaNMonitorCallback(BaseCallback):
    """Callback to monitor and debug NaN/Inf values during training.
    This callback will:
    1. Check for NaN/Inf in observations, actions, rewards, and model outputs
    2. Save a snapshot of the training state when NaN/Inf is detected
    3. Optionally stop training when an issue is found
    """
    def __init__(self, save_dir: Union[str, Path], verbose: int = 1, stop_on_nan: bool = False):
        """Initialize the NaN monitor callback.
        Args:
            save_dir: Directory to save debug snapshots
            verbose: Verbosity level (0: no output, 1: warnings, 2: debug info)
            stop_on_nan: Whether to stop training when NaN is detected
        """
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.stop_on_nan = stop_on_nan
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # Track NaN sources for better error reporting
        self.nan_sources = set()

    def _check_tensor(self, name: str, tensor: torch.Tensor, check_finite: bool = True) -> bool:
        """Vérifie un tenseur pour les valeurs NaN/Inf.

        Args:
            name: Nom du tenseur pour les logs
            tensor: Tenseur à vérifier
            check_finite: Si True, vérifie aussi les valeurs infinies

        Returns:
            bool: True si le tenseur est valide, False sinon
        """
        if not isinstance(tensor, torch.Tensor):
            self.logger.warning(
                f"{name} is not a torch.Tensor, got {type(tensor)}"
            )
            return True
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item() if check_finite else False
        if has_nan or has_inf:
            issues = []
            if has_nan:
                issues.append("NaN")
            if has_inf:
                issues.append("Inf")
            self.logger.error(
                f"Invalid values in {name} at step {self.num_timesteps}: "
                f"{' and '.join(issues)}"
            )
            try:
                os.makedirs("nan_debug", exist_ok=True)
                dump_path = os.path.join(
                    "nan_debug", f"{name}_step{self.num_timesteps}.pt"
                )
                torch.save({
                    'tensor': tensor,
                    'step': self.num_timesteps,
                    'name': name
                }, dump_path)
                self.logger.info(
                    f"Dumped {name} state to {dump_path}"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to dump {name} state: {e}"
                )
            return False
        return True

    def _save_debug_snapshot(self, timestep: int) -> None:
        """Sauvegarde un instantané de débogage lorsqu'une valeur NaN/Inf est détectée.
        Args:
            timestep: Numéro du pas de temps actuel
        """
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"nan_debug_{timestamp}_step{timestep}.npz"
            filepath = os.path.join(self.save_dir, filename)
            # Préparer les données à sauvegarder
            data = {
                'timestep': timestep,
                'nan_sources': list(self.nan_sources)
            }
            # Ajouter les tenseurs disponibles
            for name in ['obs', 'actions', 'rewards', 'values', 'log_probs']:
                tensor = self.locals.get(name)
                tensor_np = _safe_numpy(tensor)
                if tensor_np is not None:
                    data[name] = tensor_np
            # Sauvegarder dans un fichier
            np.savez_compressed(filepath, **data)
            if self.verbose > 0:
                logger.warning(
                    "NaN/Inf detected in %s. Saved debug snapshot to %s",
                    ', '.join(self.nan_sources),
                    filepath
                )
        except Exception as e:
            logger.error("Failed to save debug snapshot: %s", e)

    def _on_step(self) -> bool:
        """Called at each training step to check for NaN/Inf values."""
        # Reset nan sources for this step
        self.nan_sources = set()
        # Check tensors in locals
        for name in ['obs', 'actions', 'rewards', 'values', 'log_probs']:
            tensor = self.locals.get(name)
            self._check_tensor(name, tensor)
        # Check model parameters and gradients
        if hasattr(self.model, 'policy'):
            for name, param in self.model.policy.named_parameters():
                if param.grad is not None:
                    if self._check_tensor(f"grad_{name}", param.grad):
                        self.nan_sources.add(f"gradient {name}")
                if self._check_tensor(f"param_{name}", param):
                    self.nan_sources.add(f"parameter {name}")
        # If we found any issues, handle them
        if self.nan_sources:
            # Get infos from vectorized envs if available
            infos = self.locals.get('infos', [])
            if not isinstance(infos, (list, tuple)):
                infos = [infos]
            # Add NaN info to environment infos for logging
            for info in infos:
                if isinstance(info, dict):
                    info['nan_detected'] = True
                    info['nan_sources'] = list(self.nan_sources)
            # Save debug information
            self._save_snapshot(infos)
            # Optionally stop training
            if self.stop_on_nan:
                logger.error(f"Stopping training due to NaN/Inf in {', '.join(self.nan_sources)}")
                return False
        return True

class LearningRateMonitor(BaseCallback):
    """Callback pour surveiller les taux d'apprentissage pendant l'entraînement."""

    def __init__(self, verbose: int = 0):
        """Initialise le callback de monitoring du taux d'apprentissage.

        Args:
            verbose: Niveau de verbosité (0: pas de sortie, 1: avertissements, 2: debug)
        """
        super(LearningRateMonitor, self).__init__(verbose)

    def _on_step(self) -> bool:
        """Appelé à chaque étape d'entraînement pour logger le taux d'apprentissage.

        Returns:
            bool: Toujours True pour continuer l'entraînement
        """
        if self.n_calls % 100 == 0 and hasattr(self.model, 'policy') and \
           hasattr(self.model.policy, 'optimizer'):
            for i, g in enumerate(self.model.policy.optimizer.param_groups):
                logger.debug("LR group %d: %f", i, g['lr'])
        return True


def create_ppo_agent(
    env: VecEnv,
    config: Dict[str, Any],
    policy_kwargs: Optional[Dict[str, Any]] = None,
    verbose: int = 1,
    device: str = "auto",
) -> PPO:
    """Create and configure a PPO agent with enhanced stability features.

    Args:
        env: The training environment
        config: Configuration dictionary containing agent parameters
        policy_kwargs: Additional arguments to pass to the policy
        verbose: Verbosity level (0: no output, 1: training info, 2: debug)
        device: Device to run on ('cpu', 'cuda', 'auto')

    Returns:
        PPO: Configured PPO agent instance with enhanced stability
    """
    # Default policy kwargs with stability enhancements
    if policy_kwargs is None:
        policy_kwargs = {}
    # Get agent config with conservative defaults for stability
    agent_config = config.get("agent", {})
    ppo_config = agent_config.get("ppo", {})
    fx_kwargs_cfg = agent_config.get("features_extractor_kwargs", {})

    # Features extractor configuration
    features_dim = int(agent_config.get("features_dim", 256))
    # Pass through cnn_config from YAML if present
    cnn_config = fx_kwargs_cfg if isinstance(fx_kwargs_cfg, dict) else {}
    num_input_channels = int(cnn_config.get("input_channels", 3))

    # Ensure policy_kwargs carries extractor settings and sharing
    policy_kwargs.setdefault("net_arch", {
        "pi": ppo_config.get("policy_net_arch", [256, 256]),
        "vf": ppo_config.get("value_net_arch", [256, 256]),
    })
    policy_kwargs.setdefault("activation_fn", torch.nn.LeakyReLU)
    policy_kwargs.setdefault("ortho_init", True)
    policy_kwargs.setdefault("log_std_init", -0.5)
    policy_kwargs.setdefault("use_sde", True)
    policy_kwargs["features_extractor_class"] = CustomCNNFeatureExtractor
    policy_kwargs["features_extractor_kwargs"] = {
        "features_dim": features_dim,
        "num_input_channels": num_input_channels,
        "cnn_config": cnn_config,
    }
    # Unifier: un seul extracteur partagé pour acteur & critique
    policy_kwargs["share_features_extractor"] = True
    policy_kwargs.setdefault("optimizer_class", torch.optim.Adam)
    policy_kwargs.setdefault("optimizer_kwargs", {"eps": 1e-5})
    policy_kwargs.setdefault("squash_output", False)

    # Get policy type from config (default to MultiInputPolicy for Dict obs)
    policy_type = agent_config.get('policy_type', None)
    if policy_type is None:
        try:
            from gym import spaces as _spaces
            if hasattr(env, 'observation_space') and isinstance(env.observation_space, _spaces.Dict):
                policy_type = 'MultiInputPolicy'
            else:
                policy_type = 'MlpPolicy'
        except Exception:
            policy_type = 'MultiInputPolicy'

    # Use our custom policy class
    policy_kwargs["policy_class"] = CustomPPOPolicy

    # Learning rate warmup and scheduling with validation
    base_learning_rate = _validate_param(
        ppo_config.get("learning_rate"),
        name="learning_rate",
        min_val=1e-7,
        max_val=1e-2,
        default=1e-4
    )
    if isinstance(base_learning_rate, (int, float)):
        # Linear schedule from 1e-5 to learning_rate over first 10% of training
        def learning_rate_schedule(progress_remaining):
            warmup_start = 1e-5
            warmup_end = float(base_learning_rate)
            progress = 1 - min(progress_remaining, 0.9) / 0.9
            return warmup_start + (warmup_end - warmup_start) * progress
        learning_rate = learning_rate_schedule
    # Clip range annealing with validation
    clip_range_val = _validate_param(
        ppo_config.get("clip_range"),
        name="clip_range",
        min_val=0.01,
        max_val=0.3,
        default=0.1
    )
    if isinstance(clip_range_val, (int, float)):
        # Anneal clip range from initial value to half over training
        initial_clip = min(0.2, clip_range_val * 2)
        def clip_range_schedule(progress_remaining):
            return initial_clip * (0.5 + 0.5 * progress_remaining)
        clip_range = clip_range_schedule
    else:
        clip_range = clip_range_val

    # Create PPO parameters with validation
    ppo_params = {
        "policy": policy_type,
        "env": env,
        "learning_rate": learning_rate,
        "n_steps": _validate_param(
            ppo_config.get("n_steps"),
            name="n_steps",
            min_val=1,
            max_val=None,
            default=2048
        ),
        "batch_size": _validate_param(
            ppo_config.get("batch_size"),
            name="batch_size",
            min_val=1,
            max_val=None,
            default=64
        ),
        "n_epochs": _validate_param(
            ppo_config.get("n_epochs"),
            name="n_epochs",
            min_val=1,
            max_val=50,
            default=10
        ),
        "gamma": _validate_param(
            ppo_config.get("gamma"),
            name="gamma",
            min_val=0.8,
            max_val=0.9999,
            default=0.99
        ),
        "gae_lambda": _validate_param(
            ppo_config.get("gae_lambda"),
            name="gae_lambda",
            min_val=0.9,
            max_val=1.0,
            default=0.95
        ),
        "ent_coef": _validate_param(
            ppo_config.get("ent_coef"),
            name="ent_coef",
            min_val=0.0,
            max_val=0.5,
            default=0.0
        ),
        "vf_coef": _validate_param(
            ppo_config.get("vf_coef"),
            name="vf_coef",
            min_val=0.1,
            max_val=1.0,
            default=0.5
        ),
        "max_grad_norm": _validate_param(
            ppo_config.get("max_grad_norm"),
            name="max_grad_norm",
            min_val=0.1,
            max_val=5.0,
            default=0.5
        ),
        "sde_sample_freq": _validate_param(
            ppo_config.get("sde_sample_freq"),
            name="sde_sample_freq",
            min_val=-1,
            max_val=128,
            default=-1
        ),
        "target_kl": ppo_config.get("target_kl"),
        "tensorboard_log": os.path.join(get_path('reports'), 'tensorboard_logs'),
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
        "device": device,
        # Additional stability parameters
        "create_eval_env": False,  # Don't create separate env for evaluation
        "monitor_wrapper": True,  # Use Monitor wrapper by default
        "stats_window_size": 100,  # Larger window for stable statistics
    }
    # Sanitize all PPO parameters before agent creation
    sanitized_params = sanitize_ppo_params(ppo_params)
    # Log the sanitized parameters for debugging
    logger.debug("Sanitized PPO parameters:")
    for key, value in sanitized_params.items():
        if key not in ['policy_kwargs', 'env']:  # Skip large objects in logs
            logger.debug(f"  {key}: {value}")
    # Create and return the PPO agent with sanitized parameters
    agent = PPO(**sanitized_params)
    # Add NaN checks and gradient clipping callbacks
    agent._setup_model()
    if hasattr(agent, 'policy'):
        # Enable gradient clipping in the optimizer
        for param_group in agent.policy.optimizer.param_groups:
            if 'max_grad_norm' not in param_group:
                param_group['max_grad_norm'] = 0.5  # Default gradient clipping
    logger.info(
        "Created PPO agent with learning_rate=%s, "
        "batch_size=%s, n_epochs=%s",
        learning_rate, ppo_params['batch_size'], ppo_params['n_epochs']
    )
    return agent

def save_agent(agent, save_path):
    """
    Save a trained agent.
    Args:
        agent: Trained agent.
        save_path: Path to save the agent.
    Returns:
        str: Path where the agent was saved.
    """
    # Ensure directory exists
    ensure_dir_exists(os.path.dirname(save_path))
    # Save the agent (PPO n'accepte pas l'argument save_replay_buffer)
    agent.save(save_path)
    logger.info("Agent saved to %s", save_path)
    return save_path

def load_agent(load_path, env=None):
    """
    Load a trained agent.
    Args:
        load_path: Path to load the agent from.
        env: Environment to use with the loaded agent.
    Returns:
        PPO: Loaded agent.
    """
    try:
        agent = PPO.load(load_path, env=env)
        logger.info("Agent loaded from %s", load_path)
        return agent
    except Exception as e:
        logger.error("Error loading agent from %s: %s", load_path, e)
        raise

class TradingCallback(BaseCallback):
    """
    Callback for saving the agent during training.
    """
    def __init__(self, check_freq, save_path, verbose=1):
        """
        Initialize the callback.
        Args:
            check_freq: Frequency to check for saving.
            save_path: Path to save the agent.
            verbose: Verbosity level.
        """
        super(TradingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _init_callback(self):
        """
        Initialize the callback.
        """
        # Create folder if needed
        if self.save_path is not None:
            ensure_dir_exists(os.path.dirname(self.save_path))

    def _on_step(self):
        """
        Called at each step of training.
        Returns:
            bool: Whether to continue training.
        """
        if self.n_calls % self.check_freq == 0:
            # Get current reward
            try:
                # Méthode 1: Utiliser les valeurs du logger SB3
                mean_reward = -np.inf
                if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                    latest_values = self.model.logger.name_to_value
                    if 'rollout/ep_rew_mean' in latest_values:
                        mean_reward = latest_values['rollout/ep_rew_mean']
                # Méthode 2: Fallback - calculer manuellement si ep_info_buffer existe
                if mean_reward == -np.inf and hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer:
                    rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
                    if rewards:
                        mean_reward = np.mean(rewards)
                if mean_reward != -np.inf:
                    if self.verbose > 0:
                        logger.info(f"Num timesteps: {self.num_timesteps}")
                        logger.info(f"Mean reward: {mean_reward:.2f}")
                    # Save if better than previous best
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        if self.verbose > 0:
                            logger.info(f"Saving new best model to {self.save_path}")
                        # Nous supprimons explicitement l'argument save_replay_buffer
                        self.model.save(self.save_path)
            except Exception as e:
                import traceback
                logger.warning(f"Erreur lors de la récupération des métriques SB3: {e}")
                logger.warning(f"Trace: {traceback.format_exc()}")
        return True
