import argparse
import os
import copy
from typing import Optional

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.common.custom_logger import setup_logging
import logging
from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv


def main(config_path: str, resume: bool, num_envs: int, use_subproc: bool, progress_bar: bool, timeout: Optional[int]):
    logger = logging.getLogger(__name__)
    """Main training function."""
    try:
        # --- Configuration ---
        config = ConfigLoader.load_config(config_path)
        total_timesteps = config["training"]["timesteps_per_instance"]
        checkpoint_dir = config["paths"]["trained_models_dir"]
        
        # --- Environment Setup ---
        # Use a single data loader for all environments
        data_loader = ChunkedDataLoader(config=config, worker_config=config["workers"]["w1"], worker_id=0)
        data = data_loader.load_chunk(0)

        env_fns = []
        for i in range(num_envs):
            env_worker_config = copy.deepcopy(config["workers"].get(f"w{i+1}", config["workers"]["w1"]))
            env_worker_config["worker_id"] = i
            env_kwargs = {
                "data": data,
                "timeframes": config["data"]["timeframes"],
                "window_size": config["environment"]["window_size"],
                "features_config": config["data"]["features_config"]["timeframes"],
                "max_steps": config["environment"]["max_steps"],
                "initial_balance": config["portfolio"]["initial_balance"],
                "commission": config["environment"]["commission"],
                "reward_scaling": config["environment"]["reward_scaling"],
                "enable_logging": False,
                "log_dir": os.path.join(config["paths"]["logs_dir"], f"env_{i}"),
                "worker_config": env_worker_config,
                "config": config,
            }
            env_fns.append(lambda: MultiAssetChunkedEnv(**env_kwargs))

        if use_subproc:
            env = SubprocVecEnv(env_fns, start_method="spawn")
        else:
            env = DummyVecEnv(env_fns)

        # --- Model Instantiation ---
        policy_kwargs = copy.deepcopy(config["agent"]["features_extractor_kwargs"]["policy_kwargs"])
        
        # Convert activation function string to class
        activation_fn_map = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "LeakyReLU": nn.LeakyReLU}
        if "activation_fn" in policy_kwargs:
            activation_fn_str = policy_kwargs["activation_fn"]
            act_fn_name = activation_fn_str.split(".")[-1]
            activation_fn = activation_fn_map.get(act_fn_name)
            if activation_fn:
                policy_kwargs["activation_fn"] = activation_fn
            else:
                policy_kwargs["activation_fn"] = nn.ReLU

        # --- Callbacks ---
        # (Callbacks can be added here later)
        callbacks = []

        # --- Training ---
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=config["agent"]["learning_rate"],
            n_steps=config["agent"]["n_steps"],
            batch_size=config["agent"]["batch_size"],
            n_epochs=config["agent"]["n_epochs"],
            gamma=config["agent"]["gamma"],
            gae_lambda=config["agent"]["gae_lambda"],
            clip_range=config["agent"]["clip_range"],
            ent_coef=config["agent"]["ent_coef"],
            vf_coef=config["agent"]["vf_coef"],
            max_grad_norm=config["agent"]["max_grad_norm"],
            tensorboard_log=os.path.join(config["paths"]["logs_dir"], "tensorboard"),
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=config["agent"]["seed"],
        )

        logger.info("Starting model training...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
            progress_bar=progress_bar,
        )

        # --- Save final model ---
        final_model_path = os.path.join(checkpoint_dir, "final_model.zip")
        model.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")

        env.close()
        return True

    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        if "env" in locals():
            env.close()
        return False

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Train ADAN trading bot in parallel.")
    parser.add_argument("-c", "--config-path", type=str, default="config/config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    parser.add_argument("--no-subproc", action="store_true", help="Use DummyVecEnv instead of SubprocVecEnv.")
    parser.add_argument("--no-progress-bar", action="store_true", help="Disable the progress bar.")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds for the training run.")
    
    args = parser.parse_args()

    # Load the main config to get the number of workers
    config = ConfigLoader.load_config(args.config_path)
    num_workers = args.workers if args.workers is not None else config["agent"].get("n_envs", 4)

    main(
        config_path=args.config_path,
        resume=args.resume,
        num_envs=num_workers,
        use_subproc=not args.no_subproc,
        progress_bar=not args.no_progress_bar,
        timeout=args.timeout,
    )