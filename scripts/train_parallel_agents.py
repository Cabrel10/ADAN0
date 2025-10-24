import argparse
import os
import copy
from typing import Optional
import json
import time
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.common.custom_logger import setup_logging
import logging
from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.model.model_ensemble import ModelEnsemble


class CapitalTierTracker:
    """Tracks capital tier progression for each worker."""

    TIERS = {
        "Micro": {"min": 0, "max": 100},
        "Small": {"min": 100, "max": 1000},
        "Medium": {"min": 1000, "max": 10000},
        "High": {"min": 10000, "max": 100000},
        "Enterprise": {"min": 100000, "max": float("inf")},
    }

    def __init__(self, initial_balance=20):
        self.initial_balance = initial_balance
        self.current_tier = "Micro"
        self.tier_history = [("Micro", 0, initial_balance)]
        self.progression_log = []

    def get_tier_from_balance(self, balance):
        """Determine tier based on current balance."""
        for tier_name, limits in self.TIERS.items():
            if limits["min"] <= balance < limits["max"]:
                return tier_name
        return "Enterprise"  # Fallback for very high balances

    def update(self, step, balance, pnl=0.0):
        """Update tier tracking."""
        new_tier = self.get_tier_from_balance(balance)

        if new_tier != self.current_tier:
            # Tier upgrade/downgrade detected
            self.tier_history.append((new_tier, step, balance))
            self.progression_log.append(
                {
                    "step": step,
                    "from_tier": self.current_tier,
                    "to_tier": new_tier,
                    "balance": balance,
                    "pnl": pnl,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            self.current_tier = new_tier

    def get_progression_summary(self):
        """Get summary of tier progression."""
        return {
            "current_tier": self.current_tier,
            "tier_history": self.tier_history,
            "total_progressions": len(self.progression_log),
            "progression_log": self.progression_log,
            "reached_enterprise": self.current_tier == "Enterprise",
        }


class MetricsMonitor(BaseCallback):
    """
    Enhanced callback to monitor each worker's performance and capital tier progression.
    Generates real-time portfolio curves and tracks tier advancement.
    """

    def __init__(self, config, num_workers=4, log_interval=1000):
        super().__init__()
        self.config = config
        self.num_workers = num_workers
        self.log_interval = log_interval
        self.worker_metrics = {}
        self.portfolio_curves = {i: [] for i in range(num_workers)}
        self.tier_trackers = {
            i: CapitalTierTracker(config["portfolio"]["initial_balance"])
            for i in range(num_workers)
        }
        self.step_count = 0
        self.start_time = time.time()

        # Initialize worker-specific tracking
        for i in range(num_workers):
            self.worker_metrics[i] = {
                "total_steps": 0,
                "total_rewards": [],
                "portfolio_values": [],
                "realized_pnls": [],
                "sharpe_ratios": [],
                "drawdowns": [],
                "trade_counts": [],
                "win_rates": [],
                "tier_progressions": [],
            }

    def _on_step(self) -> bool:
        """Called at each step of training."""
        self.step_count += 1

        if self.step_count % self.log_interval == 0:
            self._collect_worker_metrics()

        return True

    def _collect_worker_metrics(self):
        """Collect metrics from all workers with daily tracking."""
        try:
            # Get portfolio managers and environments
            portfolio_managers = self.training_env.get_attr("portfolio_manager")
            environments = self.training_env.get_attr("data")

            for worker_id, pm in enumerate(portfolio_managers):
                if pm is None:
                    continue

                # Get current metrics
                metrics = pm.metrics.get_metrics_summary()
                current_balance = getattr(
                    pm.portfolio, "balance", self.config["portfolio"]["initial_balance"]
                )
                current_pnl = (
                    metrics.get("total_return", 0.0)
                    * self.config["portfolio"]["initial_balance"]
                    / 100.0
                )

                # Get current day from environment data
                current_day = 0
                trade_info = {}
                if worker_id < len(environments) and environments[worker_id]:
                    env_data = environments[worker_id]
                    if "TIMESTAMP" in env_data and len(env_data["TIMESTAMP"]) > 0:
                        # Calculate day from timestamp (assuming milliseconds)
                        timestamp = env_data["TIMESTAMP"].iloc[-1] if hasattr(env_data["TIMESTAMP"], 'iloc') else env_data["TIMESTAMP"][-1]
                        current_day = int(timestamp // (24 * 60 * 60 * 1000))

                    # Check for trade information in recent metrics
                    recent_trades = metrics.get("recent_trades", [])
                    if recent_trades:
                        last_trade = recent_trades[-1] if isinstance(recent_trades, list) else recent_trades
                        if isinstance(last_trade, dict):
                            trade_info = {
                                "trade_closed": last_trade.get("closed", False),
                                "trade_opened": last_trade.get("opened", False),
                                "trade_pnl": last_trade.get("pnl", 0.0)
                            }

                # Update tier tracker
                self.tier_trackers[worker_id].update(
                    self.step_count, current_balance, current_pnl
                )

                # Update daily tracker
                return_value = metrics.get("last_return", 0.0)
                self.daily_trackers[worker_id].update(
                    current_day, current_balance, trade_info, return_value
                )

                # Get daily performance summary
                daily_summary = self.daily_trackers[worker_id].get_current_day_summary()

                # Store worker metrics
                worker_data = {
                    "step": self.step_count,
                    "balance": current_balance,
                    "pnl": current_pnl,
                    "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                    "drawdown": metrics.get("max_drawdown", 0.0),
                    "trade_count": metrics.get("executed_trades_closed", 0),
                    "win_rate": metrics.get("win_rate", 0.0),
                    "tier": self.tier_trackers[worker_id].current_tier,
                    "timestamp": time.time() - self.start_time,
                    "current_day": current_day,
                    "daily_performance": daily_summary,
                }

                self.portfolio_curves[worker_id].append(worker_data)

                # Update aggregated metrics
                # Update aggregated metrics
                self.worker_metrics[worker_id]["total_steps"] = self.step_count
                self.worker_metrics[worker_id]["portfolio_values"].append(
                    current_balance
                )
                self.worker_metrics[worker_id]["realized_pnls"].append(current_pnl)
                self.worker_metrics[worker_id]["sharpe_ratios"].append(
                    metrics.get("sharpe_ratio", 0.0)
                )
                self.worker_metrics[worker_id]["drawdowns"].append(
                    metrics.get("max_drawdown", 0.0)
                )
                self.worker_metrics[worker_id]["trade_counts"].append(
                    metrics.get("executed_trades_closed", 0)
                )
                self.worker_metrics[worker_id]["win_rates"].append(
                    metrics.get("win_rate", 0.0)
                )
                self.worker_metrics[worker_id]["daily_performance"].append(daily_summary)

                # Log worker progress including daily metrics
                if worker_id == 0 or self.step_count % (self.log_interval * 5) == 0:
                    self.logger.record(f"worker_{worker_id}/balance", current_balance)
                    self.logger.record(f"worker_{worker_id}/pnl", current_pnl)
                    self.logger.record(
                        f"worker_{worker_id}/tier",
                        self.tier_trackers[worker_id].current_tier,
                    )
                    self.logger.record(
                        f"worker_{worker_id}/sharpe", metrics.get("sharpe_ratio", 0.0)
                    )

                    # Log daily metrics
                    if daily_summary:
                        self.logger.record(f"worker_{worker_id}/daily_pnl", daily_summary.get("daily_pnl", 0.0))
                        self.logger.record(f"worker_{worker_id}/daily_return_pct", daily_summary.get("daily_return_pct", 0.0))
                        self.logger.record(f"worker_{worker_id}/daily_trades", daily_summary.get("trades_closed", 0))
                        self.logger.record(f"worker_{worker_id}/daily_win_rate", daily_summary.get("win_rate", 0.0))
                        self.logger.record(f"worker_{worker_id}/daily_profit_factor", min(daily_summary.get("profit_factor", 0.0), 10.0))  # Cap for logging

        except Exception as e:
            self.logger.warn(f"Error collecting worker metrics: {e}")

    def get_final_summary(self):
        """Generate comprehensive training summary with daily metrics."""
        # Finalize daily tracking for all workers
        for worker_id in range(self.num_workers):
            self.daily_trackers[worker_id].finalize_current_day()

        summary = {
            "training_duration": time.time() - self.start_time,
            "total_steps": self.step_count,
            "workers": {},
            "overall_daily_performance": {},
        }

        all_avg_daily_returns = []
        all_avg_trades_per_day = []
        all_avg_win_rates = []
        all_avg_profit_factors = []
        all_avg_sharpe_ratios = []

        for worker_id in range(self.num_workers):
            if not self.worker_metrics[worker_id]["portfolio_values"]:
                continue

            final_balance = self.worker_metrics[worker_id]["portfolio_values"][-1]
            total_return = (
                (final_balance - self.config["portfolio"]["initial_balance"])
                / self.config["portfolio"]["initial_balance"]
                * 100
            )

            # Tier progression summary
            tier_summary = self.tier_trackers[worker_id].get_progression_summary()

            # Daily performance summary
            daily_avg_performance = self.daily_trackers[worker_id].get_average_daily_performance()

            worker_summary = {
                "final_balance": final_balance,
                "total_return_pct": total_return,
                "final_tier": tier_summary["current_tier"],
                "tier_progressions": tier_summary["total_progressions"],
                "final_sharpe": (
                    self.worker_metrics[worker_id]["sharpe_ratios"][-1]
                    if self.worker_metrics[worker_id]["sharpe_ratios"]
                    else 0.0
                ),
                "max_drawdown": (
                    max(self.worker_metrics[worker_id]["drawdowns"])
                    if self.worker_metrics[worker_id]["drawdowns"]
                    else 0.0
                ),
                "total_trades": (
                    self.worker_metrics[worker_id]["trade_counts"][-1]
                    if self.worker_metrics[worker_id]["trade_counts"]
                    else 0
                ),
                "final_win_rate": (
                    self.worker_metrics[worker_id]["win_rates"][-1]
                    if self.worker_metrics[worker_id]["win_rates"]
                    else 0.0
                ),
                "reached_enterprise": tier_summary["reached_enterprise"],
                # Daily performance metrics
                "daily_performance": daily_avg_performance,
            }

            summary["workers"][worker_id] = worker_summary

            # Collect for overall averages
            if daily_avg_performance:
                all_avg_daily_returns.append(daily_avg_performance.get("avg_daily_return_pct", 0))
                all_avg_trades_per_day.append(daily_avg_performance.get("avg_trades_per_day", 0))
                all_avg_win_rates.append(daily_avg_performance.get("avg_win_rate", 0))
                profit_factor = daily_avg_performance.get("avg_profit_factor", 0)
                if profit_factor != float("inf") and profit_factor > 0:
                    all_avg_profit_factors.append(profit_factor)
                all_avg_sharpe_ratios.append(daily_avg_performance.get("avg_daily_sharpe", 0))

        # Calculate overall daily performance across all workers
        if all_avg_daily_returns:
            summary["overall_daily_performance"] = {
                "avg_daily_return_pct": np.mean(all_avg_daily_returns),
                "avg_trades_per_day": np.mean(all_avg_trades_per_day),
                "avg_win_rate": np.mean(all_avg_win_rates),
                "avg_profit_factor": np.mean(all_avg_profit_factors) if all_avg_profit_factors else 0,
                "avg_daily_sharpe": np.mean(all_avg_sharpe_ratios),
                "best_daily_return_pct": max(all_avg_daily_returns),
                "worst_daily_return_pct": min(all_avg_daily_returns),
                "consistency_score": 1.0 - (np.std(all_avg_daily_returns) / max(abs(np.mean(all_avg_daily_returns)), 0.01)),
            }

        return summary

    def generate_portfolio_curves(self, output_dir):
        """Generate portfolio progression curves for each worker."""
        os.makedirs(output_dir, exist_ok=True)

        for worker_id in range(self.num_workers):
            if not self.portfolio_curves[worker_id]:
                continue

            df = pd.DataFrame(self.portfolio_curves[worker_id])
            worker_name = f"w{worker_id + 1}"

            # Create portfolio progression chart
            fig = go.Figure()

            # Portfolio balance line
            fig.add_trace(
                go.Scatter(
                    x=df["step"],
                    y=df["balance"],
                    mode="lines",
                    name=f"{worker_name} Portfolio Balance",
                    line=dict(color="blue", width=2),
                )
            )

            # Add tier progression markers
            tier_changes = self.tier_trackers[worker_id].progression_log
            if tier_changes:
                tier_steps = [tc["step"] for tc in tier_changes]
                tier_balances = [tc["balance"] for tc in tier_changes]
                tier_labels = [
                    f"{tc['from_tier']} → {tc['to_tier']}" for tc in tier_changes
                ]

                fig.add_trace(
                    go.Scatter(
                        x=tier_steps,
                        y=tier_balances,
                        mode="markers+text",
                        name=f"{worker_name} Tier Upgrades",
                        text=tier_labels,
                        textposition="top center",
                        marker=dict(color="red", size=10, symbol="diamond"),
                    )
                )

            # Add tier zones as background
            tier_colors = {
                "Micro": "lightgray",
                "Small": "lightblue",
                "Medium": "lightgreen",
                "High": "lightyellow",
                "Enterprise": "lightcoral",
            }

            for tier_name, limits in CapitalTierTracker.TIERS.items():
                if limits["max"] != float("inf"):
                    fig.add_hrect(
                        y0=limits["min"],
                        y1=limits["max"],
                        fillcolor=tier_colors.get(tier_name, "lightgray"),
                        opacity=0.2,
                        line_width=0,
                        annotation_text=tier_name,
                        annotation_position="top left",
                    )

            fig.update_layout(
                title=f"Portfolio Progression - {worker_name.upper()} (Capital Tier Advancement)",
                xaxis_title="Training Steps",
                yaxis_title="Portfolio Balance ($)",
                yaxis_type="log",
                showlegend=True,
            )

            # Save chart
            chart_path = os.path.join(
                output_dir, f"portfolio_progression_{worker_name}.html"
            )
            fig.write_html(chart_path)
            print(f"✅ Generated portfolio chart: {chart_path}")

    def get_final_summary(self):
        """Get final training summary with tier progression."""
        summary = {
            "training_duration_minutes": (time.time() - self.start_time) / 60,
            "total_steps": self.step_count,
            "workers": {},
        }

        for worker_id in range(self.num_workers):
            worker_name = f"w{worker_id + 1}"
            tier_summary = self.tier_trackers[worker_id].get_progression_summary()

            if self.portfolio_curves[worker_id]:
                final_data = self.portfolio_curves[worker_id][-1]
                initial_balance = self.config["portfolio"]["initial_balance"]

                summary["workers"][worker_name] = {
                    "initial_balance": initial_balance,
                    "final_balance": final_data["balance"],
                    "total_return_pct": (
                        (final_data["balance"] - initial_balance) / initial_balance
                    )
                    * 100,
                    "final_pnl": final_data["pnl"],
                    "final_sharpe": final_data["sharpe_ratio"],
                    "max_drawdown": max(self.worker_metrics[worker_id]["drawdowns"])
                    if self.worker_metrics[worker_id]["drawdowns"]
                    else 0,
                    "total_trades": final_data["trade_count"],
                    "tier_progression": tier_summary,
                    "reached_enterprise": tier_summary["reached_enterprise"],
                }

        return summary


def main(
    config_path: str,
    resume: bool,
    num_envs: int,
    use_subproc: bool,
    progress_bar: bool,
    timeout: Optional[int],
    checkpoint_dir: str = None,
):
    logger = logging.getLogger(__name__)
    """Main training function."""
    try:
        # --- Configuration ---
        config = ConfigLoader.load_config(config_path)
        logger.info("📋 Utilisation des paramètres de config.yaml")
        total_timesteps = config["training"]["timesteps_per_instance"]

        # Utiliser checkpoint_dir fourni ou celui du config
        if checkpoint_dir is None:
            checkpoint_dir = config["paths"]["trained_models_dir"]

        # Créer les répertoires nécessaires
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_export_dir = os.path.join(checkpoint_dir, "final")
        os.makedirs(final_export_dir, exist_ok=True)

        # --- Environment Setup (Matching Optuna Configuration) ---
        # Force 4 workers to match Optuna optimization
        if num_envs != 4:
            logger.warning(
                f"Forcing num_envs to 4 workers to match Optuna configuration (was {num_envs})"
            )
            num_envs = 4

        # Create individual environments for each worker with their specific configs
        env_fns = []
        worker_ids = ["w1", "w2", "w3", "w4"]

        for i in range(num_envs):
            worker_id = worker_ids[i]
            worker_config = config["workers"][worker_id]

            # Create data loader for this specific worker
            data_loader = ChunkedDataLoader(
                config=config, worker_config=worker_config, worker_id=i
            )
            data = data_loader.load_chunk(0)

            env_worker_config = copy.deepcopy(worker_config)
            env_worker_config["worker_id"] = i

            env_log_dir = os.path.join(config["paths"]["logs_dir"], f"{worker_id}_env")
            os.makedirs(env_log_dir, exist_ok=True)

            env_kwargs = {
                "data": data,
                "timeframes": config["data"]["timeframes"],
                "window_size": config["environment"]["window_size"],
                "features_config": config["data"]["features_config"]["timeframes"],
                "max_steps": config["environment"]["max_steps"],
                "initial_balance": config["portfolio"]["initial_balance"],
                "commission": config["environment"]["commission"],
                "reward_scaling": config["environment"]["reward_scaling"],
                "enable_logging": True,  # Enable logging for better tracking
                "log_dir": env_log_dir,
                "worker_config": env_worker_config,
                "config": config,
                "exploration_tutor": config.get("reward_shaping", {}).get(
                    "exploration_tutor", {}
                ),
            }
            # Fix lambda capture issue
            env_fns.append(lambda kwargs=env_kwargs: MultiAssetChunkedEnv(**kwargs))

            logger.info(
                f"✅ Configured {worker_id}: {worker_config['name']} - {worker_config['description']}"
            )

        # Force SubprocVecEnv for true parallelism
        logger.info("🔄 Using SubprocVecEnv for TRUE PARALLEL execution (4 workers)")
        env = SubprocVecEnv(env_fns, start_method="spawn")

        logger.info(
            f"✅ Created parallel environment with {num_envs} workers matching Optuna configuration"
        )

        # Environment already created above with forced SubprocVecEnv

        # --- Model Instantiation ---
        policy_kwargs = copy.deepcopy(
            config["agent"]["features_extractor_kwargs"]["policy_kwargs"]
        )

        # Convert activation function string to class
        activation_fn_map = {
            "ReLU": nn.ReLU,
            "Tanh": nn.Tanh,
            "LeakyReLU": nn.LeakyReLU,
        }
        if "activation_fn" in policy_kwargs:
            activation_fn_str = policy_kwargs["activation_fn"]
            act_fn_name = activation_fn_str.split(".")[-1]
            activation_fn = activation_fn_map.get(act_fn_name)
            if activation_fn:
                policy_kwargs["activation_fn"] = activation_fn
            else:
                policy_kwargs["activation_fn"] = nn.ReLU

        # --- Callbacks with Metrics Monitoring ---
        callbacks = []

        # Checkpoint callback pour sauvegardes régulières
        checkpoint_callback = CheckpointCallback(
            save_freq=config["training"]["checkpointing"]["save_freq"],
            save_path=checkpoint_dir,
            name_prefix="adan_model_checkpoint",
        )
        callbacks.append(checkpoint_callback)

        # Enhanced metrics monitor for capital tier tracking
        metrics_monitor = MetricsMonitor(
            config=config,
            num_workers=num_envs,
            log_interval=max(
                1000, config["training"]["checkpointing"]["save_freq"] // 10
            ),
        )
        callbacks.append(metrics_monitor)

        logger.info("✅ Added MetricsMonitor for capital tier progression tracking")

        # --- Training ---
        model_path = None
        if resume:
            # Chercher le dernier checkpoint
            checkpoint_files = [
                f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")
            ]
            if checkpoint_files:
                # Trier par date de modification
                checkpoint_files.sort(
                    key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
                    reverse=True,
                )
                model_path = os.path.join(checkpoint_dir, checkpoint_files[0])
                logger.info(f"📂 Resuming from checkpoint: {model_path}")
            else:
                logger.warning(
                    "⚠️ --resume specified but no checkpoint found, starting from scratch"
                )

        if model_path and os.path.exists(model_path):
            # Charger depuis checkpoint
            model = PPO.load(
                model_path,
                env=env,
                tensorboard_log=os.path.join(
                    config["paths"]["logs_dir"], "tensorboard"
                ),
            )
            logger.info("✅ Model loaded from checkpoint successfully")
        else:
            # Créer nouveau modèle
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
                tensorboard_log=os.path.join(
                    config["paths"]["logs_dir"], "tensorboard"
                ),
                policy_kwargs=policy_kwargs,
                verbose=1,
                seed=config["agent"]["seed"],
            )
            logger.info("✅ New model created successfully")

        logger.info("🚀 Starting ADAN model training on FULL train dataset...")
        logger.info(f"📊 Total timesteps: {total_timesteps:,}")
        logger.info(f"💾 Checkpoints will be saved to: {checkpoint_dir}")

        # Entraînement avec timeout handler si spécifié
        if timeout:
            with TimeoutHandler(timeout) as timeout_handler:
                try:
                    model.learn(
                        total_timesteps=total_timesteps,
                        callback=callbacks if callbacks else None,
                        progress_bar=progress_bar,
                        reset_num_timesteps=not resume,
                    )
                except TimeoutError:
                    logger.warning(f"⏰ Training timed out after {timeout}s")
        else:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks if callbacks else None,
                progress_bar=progress_bar,
                reset_num_timesteps=not resume,
            )

        # --- Save final models ---
        logger.info("💾 Saving final models...")

        # Modèle principal PyTorch
        final_model_path = os.path.join(final_export_dir, "adan_final_model.zip")
        model.save(final_model_path)
        logger.info(f"✅ PyTorch model saved: {final_model_path}")

        # Export ONNX pour portabilité et fine-tuning
        try:
            onnx_model_path = os.path.join(final_export_dir, "adan_final_model.onnx")

            # Obtenir observation de sample pour export
            sample_obs = env.observation_space.sample()
            if hasattr(sample_obs, "shape"):
                sample_obs = np.expand_dims(sample_obs, axis=0)  # Add batch dimension

            # Export vers ONNX (format portable)
            logger.info("🔄 Exporting to ONNX format for portability...")

            # Note: L'export ONNX complet nécessiterait une conversion manuelle
            # Pour l'instant, on sauve les poids dans un format accessible
            torch.save(
                {
                    "model_state_dict": model.policy.state_dict(),
                    "hyperparameters": {
                        "learning_rate": config["agent"]["learning_rate"],
                        "n_steps": config["agent"]["n_steps"],
                        "batch_size": config["agent"]["batch_size"],
                        "n_epochs": config["agent"]["n_epochs"],
                        "gamma": config["agent"]["gamma"],
                        "ent_coef": config["agent"]["ent_coef"],
                    },
                    "fusion_weights": ModelEnsemble().get_fusion_weights(),
                    "training_config": config,
                },
                os.path.join(final_export_dir, "adan_model_for_finetuning.pth"),
            )

            logger.info(
                f"✅ Model weights saved for fine-tuning: adan_model_for_finetuning.pth"
            )

        except Exception as e:
            logger.warning(f"⚠️ Could not export to ONNX: {e}")

        # --- Generate Performance Reports ---
        logger.info("📈 Training completed successfully!")

        # Generate portfolio progression curves
        charts_dir = os.path.join(checkpoint_dir, "progression_charts")
        metrics_monitor.generate_portfolio_curves(charts_dir)

        # Get final summary with tier progression
        final_summary = metrics_monitor.get_final_summary()

        # Save training summary
        summary_path = os.path.join(final_export_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(final_summary, f, indent=2)

        logger.info(f"📊 FINAL TRAINING SUMMARY:")
        logger.info(
            f"⏱️  Duration: {final_summary['training_duration_minutes']:.1f} minutes"
        )
        logger.info(f"📈 Total Steps: {final_summary['total_steps']:,}")

        # Report tier progression for each worker
        enterprise_count = 0
        for worker_name, worker_data in final_summary["workers"].items():
            tier_info = worker_data["tier_progression"]
            logger.info(f"")
            logger.info(f"🏆 {worker_name.upper()} PERFORMANCE:")
            logger.info(
                f"   💰 Balance: ${worker_data['initial_balance']:.2f} → ${worker_data['final_balance']:.2f}"
            )
            logger.info(f"   📈 Return: {worker_data['total_return_pct']:+.2f}%")
            logger.info(f"   🎯 Final Tier: {tier_info['current_tier']}")
            logger.info(f"   🚀 Tier Progressions: {tier_info['total_progressions']}")
            logger.info(
                f"   🏢 Reached Enterprise: {'✅ YES' if tier_info['reached_enterprise'] else '❌ NO'}"
            )
            logger.info(f"   📊 Sharpe: {worker_data['final_sharpe']:.4f}")
            logger.info(f"   📉 Max DD: {worker_data['max_drawdown']:.2f}%")
            logger.info(f"   🔄 Trades: {worker_data['total_trades']}")

            if tier_info["reached_enterprise"]:
                enterprise_count += 1

        # Overall success metrics
        logger.info(f"")
        logger.info(f"🎯 OVERALL SUCCESS METRICS:")
        logger.info(f"   🏢 Workers reaching Enterprise tier: {enterprise_count}/4")
        logger.info(f"   ✅ Training Success Rate: {(enterprise_count / 4) * 100:.1f}%")
        logger.info(f"   📁 Models location: {final_export_dir}")
        logger.info(f"   📊 Summary saved: {summary_path}")
        logger.info(f"   📈 Charts generated: {charts_dir}")
        logger.info(f"🔧 Models ready for fine-tuning and deployment")

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
    parser.add_argument(
        "-c",
        "--config-path",
        type=str,
        default="config/config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save/load checkpoints from.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers."
    )
    parser.add_argument(
        "--no-subproc",
        action="store_true",
        help="Use DummyVecEnv instead of SubprocVecEnv.",
    )
    parser.add_argument(
        "--no-progress-bar", action="store_true", help="Disable the progress bar."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for the training run.",
    )

    args = parser.parse_args()

    # Load the main config to get the number of workers
    config = ConfigLoader.load_config(args.config_path)
    # Force 4 workers to match Optuna configuration
    num_workers = 4  # Always use 4 workers to match Optuna optimization

    main(
        config_path=args.config_path,
        resume=args.resume,
        num_envs=num_workers,
        use_subproc=not args.no_subproc,
        progress_bar=not args.no_progress_bar,
        timeout=args.timeout,
        checkpoint_dir=args.checkpoint_dir,
    )
