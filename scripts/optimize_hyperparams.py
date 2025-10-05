import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import optuna
import copy

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import necessary components from the bot
from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from src.adan_trading_bot.common.config_loader import ConfigLoader
from src.adan_trading_bot.common.custom_logger import setup_logging
from src.adan_trading_bot.data_processing.data_loader import ChunkedDataLoader # Needed for data loading

# Setup basic logging for the optimization script
setup_logging()
logger = logging.getLogger(__name__)

# --- Global Configuration ---
CONFIG_PATH = os.path.join(project_root, "config", "config.yaml")
GLOBAL_CONFIG = ConfigLoader.load_config(CONFIG_PATH)
logger.info(f"Resolved data_dir.train: {GLOBAL_CONFIG['data']['data_dirs']['train']}")

# --- Helper function to run a short training ---
def run_short_training(hyperparams: dict, trial_number: int, worker_id: int = 0) -> float:
    """
    Runs a short training episode with the given hyperparameters and returns the Sharpe Ratio.
    """
    logger.info(f"Running short training for trial {trial_number}, worker {worker_id} with hyperparams: {hyperparams}")
    
    # 1. Create a temporary configuration
    temp_config = copy.deepcopy(GLOBAL_CONFIG)

    # Update PPO agent parameters in the temporary config
    for param, value in hyperparams.items():
        if param in ["learning_rate", "n_steps", "ent_coef", "clip_range", "gamma", "batch_size", "n_epochs"]:
            temp_config["agent"][param] = value
        # Update reward shaping parameters for a specific worker's profile
        elif param in ["stop_loss_penalty", "take_profit_bonus", "missed_opportunity_penalty", "overstay_penalty", "multi_traque_bonus", "pnl_weight"]:
            worker_name_map = {0: 'Conservative', 1: 'Moderate', 2: 'Aggressive', 3: 'Adaptive'}
            target_worker_name = worker_name_map.get(worker_id, 'Default')
            
            if target_worker_name in temp_config["reward_shaping"]["profiles"]:
                temp_config["reward_shaping"]["profiles"][target_worker_name][param] = value
            else:
                # Fallback to global reward_shaping if not worker-specific
                temp_config["reward_shaping"][param] = value
        elif param in ["dbe_aggressiveness_decay", "dbe_volatility_guard", "dbe_volatility_threshold", "dbe_trend_threshold"]:
            # Update DBE parameters
            if "dbe" not in temp_config:
                temp_config["dbe"] = {}
            if param == "dbe_aggressiveness_decay":
                temp_config["dbe"]["aggressiveness_decay"] = value
            elif param == "dbe_volatility_guard":
                temp_config["dbe"]["volatility_guard"] = value
            elif param == "dbe_volatility_threshold":
                if "market_regime_detection" not in temp_config["dbe"]:
                    temp_config["dbe"]["market_regime_detection"] = {}
                temp_config["dbe"]["market_regime_detection"]["volatility_threshold"] = value
            elif param == "dbe_trend_threshold":
                if "market_regime_detection" not in temp_config["dbe"]:
                    temp_config["dbe"]["market_regime_detection"] = {}
                temp_config["dbe"]["market_regime_detection"]["trend_threshold"] = value
        elif param in ["trading_rules_stop_loss_pct", "trading_rules_take_profit_pct", "trading_rules_max_position_steps"]:
            # Update trading rules parameters
            if "trading_rules" not in temp_config:
                temp_config["trading_rules"] = {}
            if param == "trading_rules_stop_loss_pct":
                temp_config["trading_rules"]["stop_loss_pct"] = value
            elif param == "trading_rules_take_profit_pct":
                temp_config["trading_rules"]["take_profit_pct"] = value
            elif param == "trading_rules_max_position_steps":
                temp_config["trading_rules"]["max_position_steps"] = value

    # Update trading_rules.frequency parameters
    if "frequency_min_positions_5m" in hyperparams:
        temp_config["trading_rules"]["frequency"]["min_positions"]["5m"] = hyperparams["frequency_min_positions_5m"]
    if "frequency_max_positions_5m" in hyperparams:
        temp_config["trading_rules"]["frequency"]["max_positions"]["5m"] = hyperparams["frequency_max_positions_5m"]
    if "frequency_min_positions_1h" in hyperparams:
        temp_config["trading_rules"]["frequency"]["min_positions"]["1h"] = hyperparams["frequency_min_positions_1h"]
    if "frequency_max_positions_1h" in hyperparams:
        temp_config["trading_rules"]["frequency"]["max_positions"]["1h"] = hyperparams["frequency_max_positions_1h"]
    if "frequency_min_positions_4h" in hyperparams:
        temp_config["trading_rules"]["frequency"]["min_positions"]["4h"] = hyperparams["frequency_min_positions_4h"]
    if "frequency_max_positions_4h" in hyperparams:
        temp_config["trading_rules"]["frequency"]["max_positions"]["4h"] = hyperparams["frequency_max_positions_4h"]
    if "frequency_total_daily_min" in hyperparams:
        temp_config["trading_rules"]["frequency"]["total_daily_min"] = hyperparams["frequency_total_daily_min"]
    if "frequency_total_daily_max" in hyperparams:
        temp_config["trading_rules"]["frequency"]["total_daily_max"] = hyperparams["frequency_total_daily_max"]
    if "frequency_bonus_weight" in hyperparams:
        temp_config["trading_rules"]["frequency"]["frequency_bonus_weight"] = hyperparams["frequency_bonus_weight"]
    if "frequency_penalty_weight" in hyperparams:
        temp_config["trading_rules"]["frequency"]["penalty_weight"] = hyperparams["frequency_penalty_weight"]
    if "frequency_grace_period_steps" in hyperparams:
        temp_config["trading_rules"]["frequency"]["grace_period_steps"] = hyperparams["frequency_grace_period_steps"]
    if "force_trade_steps" in hyperparams:
        temp_config["trading_rules"]["frequency"]["force_trade_steps"] = hyperparams["force_trade_steps"]

    # Update trading_rules.duration_tracking parameters
    if "duration_tracking_5m_max_duration_steps" in hyperparams:
        temp_config["trading_rules"]["duration_tracking"]["5m"]["max_duration_steps"] = hyperparams["duration_tracking_5m_max_duration_steps"]
    if "duration_tracking_5m_optimal_duration" in hyperparams:
        temp_config["trading_rules"]["duration_tracking"]["5m"]["optimal_duration"] = hyperparams["duration_tracking_5m_optimal_duration"]
    if "duration_tracking_1h_max_duration_steps" in hyperparams:
        temp_config["trading_rules"]["duration_tracking"]["1h"]["max_duration_steps"] = hyperparams["duration_tracking_1h_max_duration_steps"]
    if "duration_tracking_4h_max_duration_steps" in hyperparams:
        temp_config["trading_rules"]["duration_tracking"]["4h"]["max_duration_steps"] = hyperparams["duration_tracking_4h_max_duration_steps"]

    # Update window sizes
    if "window_size_5m" in hyperparams:
        temp_config["environment"]["observation"]["window_sizes"]["5m"] = hyperparams["window_size_5m"]
    if "window_size_1h" in hyperparams:
        temp_config["environment"]["observation"]["window_sizes"]["1h"] = hyperparams["window_size_1h"]
    if "window_size_4h" in hyperparams:
        temp_config["environment"]["observation"]["window_sizes"]["4h"] = hyperparams["window_size_4h"]

    # TODO: Handle other parameters if they are added to hyperparams

    # 2. Data Loading (simplified for short training)
    temp_config["data"]["data_split_override"] = "train"
    temp_config["environment"]["max_chunks_per_episode"] = 1 # Train on a single chunk
    temp_config["environment"]["max_steps"] = 50000 # Short training duration

    # Initialize ChunkedDataLoader
    worker_config = temp_config["workers"][f"w{worker_id+1}"] # Assuming worker_id 0 corresponds to w1
    data_loader = ChunkedDataLoader(config=temp_config, worker_config=worker_config, worker_id=worker_id)
    
    # Load the first chunk of data
    data = data_loader.load_chunk(0) # Load the first chunk

    # 3. Environment Setup
    n_envs = temp_config["agent"].get("n_envs", 1) # Get n_envs from config, default to 1

    # Create multiple environment instances
    env_fns = []
    for i in range(n_envs):
        # Each env needs a unique log_dir to avoid conflicts
        env_log_dir = os.path.join(project_root, "logs", f"optuna_trial_{trial_number}_env_{i}")
        os.makedirs(env_log_dir, exist_ok=True)

        # Create a copy of worker_config for each env to ensure isolation
        env_worker_config = copy.deepcopy(worker_config)
        env_worker_config["worker_id"] = i # Assign unique worker_id for each sub-environment

        env_kwargs = {
            "data": data,
            "timeframes": temp_config["data"]["timeframes"],
            "window_size": temp_config["environment"]["window_size"],
            "features_config": temp_config["data"]["features_config"]["timeframes"],
            "max_steps": temp_config["environment"]["max_steps"],
            "initial_balance": temp_config["portfolio"]["initial_balance"],
            "commission": temp_config["environment"]["commission"],
            "reward_scaling": temp_config["environment"]["reward_scaling"],
            "enable_logging": False, # Disable verbose logging for sub-environments
            "log_dir": env_log_dir, 
            "worker_config": env_worker_config,
            "config": temp_config, # Pass the modified global config
        }
        env_fns.append(lambda: MultiAssetChunkedEnv(**env_kwargs))
    
    # Wrap in SubprocVecEnv
    env = SubprocVecEnv(env_fns)

    # Create policy_kwargs for PPO agent
    policy_kwargs_for_ppo = copy.deepcopy(temp_config["agent"]["features_extractor_kwargs"]["policy_kwargs"])

    # Convert activation_fn string to callable if it exists
    if "activation_fn" in policy_kwargs_for_ppo:
        activation_fn_str = policy_kwargs_for_ppo["activation_fn"]
        import torch.nn as nn # Import torch.nn
        if hasattr(nn, activation_fn_str.split('.')[-1]):
            policy_kwargs_for_ppo["activation_fn"] = getattr(nn, activation_fn_str.split('.')[-1])
        else:
            logger.warning(f"Unknown activation function: {activation_fn_str}. Using ReLU as default.")
            policy_kwargs_for_ppo["activation_fn"] = nn.ReLU

    # 4. Agent Setup
    model = PPO(
        "MultiInputPolicy", # Policy is fixed for now
        env,
        learning_rate=hyperparams["learning_rate"],
        n_steps=hyperparams["n_steps"],
        batch_size=hyperparams["batch_size"],
        n_epochs=hyperparams["n_epochs"],
        gamma=hyperparams["gamma"],
        gae_lambda=temp_config["agent"]["gae_lambda"], # Use global for now
        clip_range=hyperparams["clip_range"],
        ent_coef=hyperparams["ent_coef"],
        vf_coef=temp_config["agent"]["vf_coef"], # Use global for now
        max_grad_norm=temp_config["agent"]["max_grad_norm"], # Use global for now
        tensorboard_log=None, # Disable tensorboard for trials
        policy_kwargs=policy_kwargs_for_ppo, # Pass the prepared policy_kwargs
        verbose=0, # Suppress SB3 verbose output
        seed=temp_config["agent"]["seed"],
    )

    # 5. Training
    model.learn(total_timesteps=temp_config["environment"]["max_steps"], progress_bar=True)

    # 6. Metrics Extraction
    # Get the final Sharpe Ratio from the environment
    # Access the unwrapped env to get portfolio metrics
    final_sharpe_ratio = env.get_attr("portfolio_manager")[0].metrics.get_metrics_summary()["sharpe_ratio"]
    
    # Ensure Sharpe is not NaN or Inf
    if np.isnan(final_sharpe_ratio) or np.isinf(final_sharpe_ratio):
        return -np.inf # Penalize NaN/Inf Sharpe

    logger.info(f"Trial {trial_number} finished with Sharpe Ratio: {final_sharpe_ratio:.4f}")
    return final_sharpe_ratio

# --- Optuna Objective Function ---
def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna to optimize.
    """
    # Suggest hyperparameters for PPO agent
    ppo_params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
        "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]), # Added batch_size
        "n_epochs": trial.suggest_categorical("n_epochs", [5, 10, 20]), # Added n_epochs
    }

    # Suggest hyperparameters for reward shaping (worker-specific)
    # For simplicity, let's optimize for a single worker's profile first (e.g., Conservative)
    reward_params = {
        "stop_loss_penalty": trial.suggest_float("stop_loss_penalty", -2.0, -0.5),
        "take_profit_bonus": trial.suggest_float("take_profit_bonus", 0.2, 1.5),
        "missed_opportunity_penalty": trial.suggest_float("missed_opportunity_penalty", -1.0, -0.1),
        "overstay_penalty": trial.suggest_float("overstay_penalty", -2.0, -0.5),
        "multi_traque_bonus": trial.suggest_float("multi_traque_bonus", 0.5, 3.0),
        "pnl_weight": trial.suggest_float("pnl_weight", 1.0, 10.0),
        
        # New parameters from trading_rules.frequency
        "frequency_min_positions_5m": trial.suggest_int("frequency_min_positions_5m", 1, 10),
        "frequency_max_positions_5m": trial.suggest_int("frequency_max_positions_5m", 10, 30),
        "frequency_min_positions_1h": trial.suggest_int("frequency_min_positions_1h", 1, 5),
        "frequency_max_positions_1h": trial.suggest_int("frequency_max_positions_1h", 5, 15),
        "frequency_min_positions_4h": trial.suggest_int("frequency_min_positions_4h", 0, 3),
        "frequency_max_positions_4h": trial.suggest_int("frequency_max_positions_4h", 1, 5),
        "frequency_total_daily_min": trial.suggest_int("frequency_total_daily_min", 1, 10),
        "frequency_total_daily_max": trial.suggest_int("frequency_total_daily_max", 10, 25),
        "frequency_bonus_weight": trial.suggest_float("frequency_bonus_weight", 0.1, 1.0),
        "frequency_penalty_weight": trial.suggest_float("frequency_penalty_weight", 0.01, 0.5),
        "frequency_grace_period_steps": trial.suggest_int("frequency_grace_period_steps", 50, 200),

        # New parameters from trading_rules.duration_tracking (for 5m timeframe for now)
        "duration_tracking_5m_max_duration_steps": trial.suggest_int("duration_tracking_5m_max_duration_steps", 24, 96), # 2-8 hours for 5m
        "duration_tracking_5m_optimal_duration": trial.suggest_int("duration_tracking_5m_optimal_duration", 12, 48), # 1-4 hours for 5m
        "duration_tracking_1h_max_duration_steps": trial.suggest_int("duration_tracking_1h_max_duration_steps", 12, 48), # 12-48 hours for 1h
        "duration_tracking_4h_max_duration_steps": trial.suggest_int("duration_tracking_4h_max_duration_steps", 6, 24), # 24-96 hours for 4h
        "force_trade_steps": trial.suggest_int("force_trade_steps", 50, 200),
    }

    dbe_params = {
        "dbe_aggressiveness_decay": trial.suggest_float("dbe_aggressiveness_decay", 0.9, 0.999),
        "dbe_volatility_guard": trial.suggest_float("dbe_volatility_guard", 0.1, 0.5),
        "dbe_volatility_threshold": trial.suggest_float("dbe_volatility_threshold", 0.01, 0.05),
        "dbe_trend_threshold": trial.suggest_float("dbe_trend_threshold", 0.01, 0.1),
    }

    trading_rules_params = {
        "trading_rules_stop_loss_pct": trial.suggest_float("trading_rules_stop_loss_pct", 0.01, 0.05),
        "trading_rules_take_profit_pct": trial.suggest_float("trading_rules_take_profit_pct", 0.02, 0.1),
        "trading_rules_max_position_steps": trial.suggest_int("trading_rules_max_position_steps", 100, 1000),
    }

    window_size_params = {
        "window_size_5m": trial.suggest_int("window_size_5m", 10, 50),
        "window_size_1h": trial.suggest_int("window_size_1h", 5, 20),
        "window_size_4h": trial.suggest_int("window_size_4h", 3, 10),
    }
    
    # Combine all hyperparameters
    all_hyperparams = {**ppo_params, **reward_params, **dbe_params, **trading_rules_params, **window_size_params}

    # Run a short training with these hyperparameters
    # For now, let's assume we are optimizing for worker 'w1' (Conservative)
    sharpe_ratio = run_short_training(all_hyperparams, trial_number=trial.number, worker_id=0) # Assuming worker_id 0 for w1

    return sharpe_ratio

# --- Main Optuna Study ---
if __name__ == "__main__":
    logger.info("Starting Optuna hyperparameter optimization study...")
    
    # Create an Optuna study to maximize Sharpe Ratio
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///hyperparam_optimization.db",
        study_name="adan_trading_bot_hyperparam_optimization",
        load_if_exists=True,
    )
    
    # Run the optimization for a specified number of trials
    n_trials = 100 # Start with a small number for testing
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    
    logger.info("\nOptimization finished.")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    
    logger.info("Best trial:")
    trial = study.best_trial
    
    logger.info(f"  Value (Sharpe Ratio): {trial.value:.4f}")
    logger.info("  Best hyperparameters:")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # Save study results (optional)
    # study.trials_dataframe().to_csv("optuna_study_results.csv")