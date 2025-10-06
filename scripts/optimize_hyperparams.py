import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import optuna
import copy
import gc

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from optuna.pruners import MedianPruner

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

class OptunaPruningCallback(BaseCallback):
    """Stop training if the trial is pruned."""

    def __init__(self, trial: optuna.Trial, eval_env: SubprocVecEnv, eval_freq: int = 10000):
        super().__init__(verbose=0)
        self.trial = trial
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.last_eval_step = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_eval_step) >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            # Evaluate the model
            sharpe_ratio = self._evaluate_sharpe()
            
            # Report the intermediate value to Optuna
            self.trial.report(sharpe_ratio, self.num_timesteps)

            # Prune trial if need be
            if self.trial.should_prune():
                logger.info(f"Trial {self.trial.number} pruned at step {self.num_timesteps}.")
                return False  # Stop training
        return True

    def _evaluate_sharpe(self) -> float:
        """A simplified evaluation to get an intermediate Sharpe ratio."""
        try:
            # In a SubprocVecEnv, we need to get the attribute from the underlying envs
            portfolio_manager = self.eval_env.get_attr("portfolio_manager")[0]
            metrics = portfolio_manager.metrics.get_metrics_summary()
            sharpe = metrics.get("sharpe_ratio", 0.0)
            if np.isnan(sharpe) or np.isinf(sharpe):
                return -1.0 # Return a bad score for invalid metrics
            return sharpe
        except Exception as e:
            logger.warning(f"Could not evaluate intermediate Sharpe ratio: {e}")
            return -1.0

logger = logging.getLogger(__name__)

# --- Global Configuration ---
CONFIG_PATH = os.path.join(project_root, "config", "config.yaml")
GLOBAL_CONFIG = ConfigLoader.load_config(CONFIG_PATH)
logger.info(f"Resolved data_dir.train: {GLOBAL_CONFIG['data']['data_dirs']['train']}")

# --- Optuna Objective Function ---
def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna to optimize, with pruning and memory management.
    """
    env = None
    model = None
    try:
        # Suggest hyperparameters for PPO agent
        ppo_params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
            "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
            "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "n_epochs": trial.suggest_categorical("n_epochs", [5, 10, 20]),
        }

        # Suggest hyperparameters for reward shaping
        reward_params = {
            "stop_loss_penalty": trial.suggest_float("stop_loss_penalty", -2.0, -0.5),
            "take_profit_bonus": trial.suggest_float("take_profit_bonus", 0.2, 1.5),
            "missed_opportunity_penalty": trial.suggest_float("missed_opportunity_penalty", -1.0, -0.1),
            "overstay_penalty": trial.suggest_float("overstay_penalty", -2.0, -0.5),
            "multi_traque_bonus": trial.suggest_float("multi_traque_bonus", 0.5, 3.0),
            "pnl_weight": trial.suggest_float("pnl_weight", 1.0, 10.0),
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
            "duration_tracking_5m_max_duration_steps": trial.suggest_int("duration_tracking_5m_max_duration_steps", 24, 96),
            "duration_tracking_5m_optimal_duration": trial.suggest_int("duration_tracking_5m_optimal_duration", 12, 48),
            "duration_tracking_1h_max_duration_steps": trial.suggest_int("duration_tracking_1h_max_duration_steps", 12, 48),
            "duration_tracking_4h_max_duration_steps": trial.suggest_int("duration_tracking_4h_max_duration_steps", 6, 24),
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

        position_sizing_params = {
            "initial_position_size": trial.suggest_float("initial_position_size", 0.9, 0.95)
        }
        
        all_hyperparams = {**ppo_params, **reward_params, **dbe_params, **trading_rules_params, **window_size_params, **position_sizing_params}

        temp_config = copy.deepcopy(GLOBAL_CONFIG)
        for param, value in all_hyperparams.items():
            # This logic can be simplified, but we keep it for now
            # A better approach would be to use a nested dictionary update utility
            if param in ["learning_rate", "n_steps", "ent_coef", "clip_range", "gamma", "batch_size", "n_epochs"]:
                temp_config["agent"][param] = value
            elif param == "initial_position_size":
                temp_config["environment"]["risk_management"]["position_sizing"]["initial_position_size"] = value
            elif param in ["stop_loss_penalty", "take_profit_bonus", "missed_opportunity_penalty", "overstay_penalty", "multi_traque_bonus", "pnl_weight"]:
                temp_config["reward_shaping"]["profiles"]['Conservative'][param] = value
            elif param.startswith("dbe_"):
                # Simplified DBE param handling
                key = param.replace("dbe_", "")
                if "dbe" not in temp_config: temp_config["dbe"] = {}
                temp_config["dbe"][key] = value
            elif param.startswith("trading_rules_"):
                key = param.replace("trading_rules_", "")
                if "trading_rules" not in temp_config: temp_config["trading_rules"] = {}
                temp_config["trading_rules"][key] = value
            elif param.startswith("frequency_") or param.startswith("duration_") or param == "force_trade_steps":
                 # Simplified frequency/duration handling
                pass # This part of the logic was complex and might need a better mapping
            elif param.startswith("window_size_"):
                tf = param.split('_')[-1]
                temp_config["environment"]["observation"]["window_sizes"][tf] = value

        temp_config["data"]["data_split_override"] = "train"
        temp_config["environment"]["max_chunks_per_episode"] = 1
        temp_config["environment"]["max_steps"] = 50000

        worker_config = temp_config["workers"]["w3"]
        data_loader = ChunkedDataLoader(config=temp_config, worker_config=worker_config, worker_id=0)
        data = data_loader.load_chunk(0)

        n_envs = temp_config["agent"].get("n_envs", 1)
        env_fns = []
        for i in range(n_envs):
            env_log_dir = os.path.join(project_root, "logs", f"optuna_trial_{trial.number}_env_{i}")
            os.makedirs(env_log_dir, exist_ok=True)
            env_worker_config = copy.deepcopy(worker_config)
            env_worker_config["worker_id"] = i
            env_kwargs = {
                "data": data,
                "timeframes": temp_config["data"]["timeframes"],
                "window_size": temp_config["environment"]["window_size"],
                "features_config": temp_config["data"]["features_config"]["timeframes"],
                "max_steps": temp_config["environment"]["max_steps"],
                "initial_balance": temp_config["portfolio"]["initial_balance"],
                "commission": temp_config["environment"]["commission"],
                "reward_scaling": temp_config["environment"]["reward_scaling"],
                "enable_logging": False,
                "log_dir": env_log_dir, 
                "worker_config": env_worker_config,
                "config": temp_config,
            }
            env_fns.append(lambda: MultiAssetChunkedEnv(**env_kwargs))
        
        env = SubprocVecEnv(env_fns)

        pruning_callback = OptunaPruningCallback(trial, eval_env=env, eval_freq=10000)

        policy_kwargs_for_ppo = copy.deepcopy(temp_config["agent"]["features_extractor_kwargs"]["policy_kwargs"])
        activation_fn_map = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "LeakyReLU": nn.LeakyReLU}
        if "activation_fn" in policy_kwargs_for_ppo:
            activation_fn_str = policy_kwargs_for_ppo["activation_fn"]
            act_fn_name = activation_fn_str.split('.')[-1]
            activation_fn = activation_fn_map.get(act_fn_name)
            if activation_fn:
                policy_kwargs_for_ppo["activation_fn"] = activation_fn
            else:
                policy_kwargs_for_ppo["activation_fn"] = nn.ReLU

        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=all_hyperparams["learning_rate"],
            n_steps=all_hyperparams["n_steps"],
            batch_size=all_hyperparams["batch_size"],
            n_epochs=all_hyperparams["n_epochs"],
            gamma=all_hyperparams["gamma"],
            gae_lambda=temp_config["agent"]["gae_lambda"],
            clip_range=all_hyperparams["clip_range"],
            ent_coef=all_hyperparams["ent_coef"],
            vf_coef=temp_config["agent"]["vf_coef"],
            max_grad_norm=temp_config["agent"]["max_grad_norm"],
            tensorboard_log=None,
            policy_kwargs=policy_kwargs_for_ppo,
            verbose=0,
            seed=temp_config["agent"]["seed"],
        )

        model.learn(
            total_timesteps=temp_config["environment"]["max_steps"],
            callback=pruning_callback,
            progress_bar=True
        )

        # Retrieve and log final metrics
        metrics_summary = env.get_attr("portfolio_manager")[0].metrics.get_metrics_summary()
        final_sharpe_ratio = metrics_summary.get("sharpe_ratio", 0.0)
        
        invalid_attempts = metrics_summary.get("invalid_trade_attempts", 0)
        valid_attempts = metrics_summary.get("valid_trade_attempts", 0)
        executed_trades = metrics_summary.get("executed_trades_opened", 0)

        logger.info(f"Trial {trial.number} finished. Valid Attempts: {valid_attempts}, Invalid Attempts: {invalid_attempts}, Executed Trades: {executed_trades}")

        # Save counters as user attributes
        trial.set_user_attr("invalid_trade_attempts", invalid_attempts)
        trial.set_user_attr("valid_trade_attempts", valid_attempts)
        trial.set_user_attr("executed_trades_opened", executed_trades)

        if np.isnan(final_sharpe_ratio) or np.isinf(final_sharpe_ratio):
            return -np.inf

        return final_sharpe_ratio

    except optuna.exceptions.TrialPruned as e:
        logger.info(f"Trial {trial.number} was pruned successfully.")
        raise e
    finally:
        logger.info(f"Cleaning up resources for trial {trial.number}.")
        if env is not None:
            try:
                env.close()
            except Exception as e:
                logger.error(f"Error closing environment: {e}")
        if model is not None:
            del model
        del env
        gc.collect()

# --- Main Optuna Study ---
if __name__ == "__main__":
    logger.info("Starting Optuna hyperparameter optimization study with Pruning...")
    
    # Create an Optuna study to maximize Sharpe Ratio with a Pruner
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///hyperparam_optimization.db",
        study_name="adan_trading_bot_hyperparam_optimization",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10000),
        load_if_exists=True,
    )
    
    # Run the optimization for a specified number of trials
    n_trials = 100 # Start with a small number for testing
    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=2, gc_after_trial=True)
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
    
    logger.info("\nOptimization finished.")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    
    logger.info("Best trial:")
    trial = study.best_trial
    
    logger.info(f"  Value (Sharpe Ratio): {trial.value:.4f}")
    logger.info("  Best hyperparameters:")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # Generate and save visualization plots
    try:
        import optuna.visualization as vis

        # 1. Optimization History Plot
        fig_history = vis.plot_optimization_history(study)
        fig_history.write_html("optuna_history.html")

        # 2. Hyperparameter Importances Plot
        fig_importance = vis.plot_param_importances(study)
        fig_importance.write_html("optuna_importance.html")

        logger.info("\nGenerated optimization plots: optuna_history.html, optuna_importance.html")

    except ImportError:
        logger.warning("\nCould not generate plots. Please install plotly and kaleido: `pip install plotly kaleido`")
    except Exception as e:
        logger.error(f"\nAn error occurred during plot generation: {e}")