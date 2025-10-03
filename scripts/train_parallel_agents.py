# -*- coding: utf-8 -*-

"""Script d'entraînement parallèle pour instances ADAN."""

# Standard Library Imports
import argparse
import json
import logging
import os
import sys
import time
import traceback
import uuid
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# Third-Party Imports
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import pandas as pd
import torch
import torch as th
import torch.nn as nn
import yaml
from rich.console import Console
try:
    from sb3_contrib import RecurrentPPO
except Exception as _e:
    RecurrentPPO = None  # Will raise at runtime if used without sb3-contrib installed
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure as sb3_logger_configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecNormalize
)

# Local Application Imports
from adan_trading_bot.common.custom_logger import setup_logging
from adan_trading_bot.environment.checkpoint_manager import CheckpointManager
from adan_trading_bot.environment.multi_asset_chunked_env import (
    MultiAssetChunkedEnv
)
from adan_trading_bot.training.trainer import (
    validate_environment,
    TimeoutManager,
    TimeoutException as TMTimeoutException,
)
from adan_trading_bot.agent.custom_recurrent_policy import CustomRecurrentPolicy




# ==============================================================================
# SECTION: Configuration et exécution du script
# ==============================================================================
# Configure logging to be less verbose
logging.getLogger().setLevel(logging.ERROR)
for logger_name in [
    "root", "matplotlib", "numpy", "pandas", "torch",
    "stable_baselines3", "gymnasium", "gym", "adan_trading_bot"
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Suppress all warnings for a cleaner output
warnings.filterwarnings("ignore")

# Initialize Rich Console
console = Console()

class HierarchicalTrainingCallback(BaseCallback):
    """Callback pour affichage hiérarchique de l'entraînement avec métriques détaillées."""

    def __init__(
        self,
        verbose=1,
        display_freq=1000,
        total_timesteps=1000000,
        initial_capital=20.50,
    ):
        super().__init__(verbose)
        self.display_freq = display_freq
        self.total_timesteps = total_timesteps
        self.initial_capital = initial_capital
        self.correlation_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.last_step_summary = 0
        self.episode_rewards = []
        self.episode_count = 0
        self.positions = {}
        self.metrics = {
            "sharpe": 0.0,
            "sortino": 0.0,
            "profit_factor": 0.0,
            "max_dd": 0.0,
            "cagr": 0.0,
            "win_rate": 0.0,
            "trades": 0,
        }

    def _on_training_start(self):
        """Démarrage de l'entraînement avec affichage de la configuration."""
        logger.info("╭" + "─" * 60 + "╮")
        logger.info("│" + " " * 15 + "🚀 DÉMARRAGE ADAN TRAINING" + " " * 15 + "│")
        logger.info("╰" + "─" * 60 + "╯")
        logger.info(f"[TRAINING START] Correlation ID: {self.correlation_id}")
        logger.info(f"[TRAINING START] Total timesteps: {self.total_timesteps:,}")
        logger.info(f"[TRAINING START] Capital initial: ${self.initial_capital:.2f}")

    def _on_step(self) -> bool:
        """Appelé à chaque étape pour mettre à jour l'affichage."""
        if self.num_timesteps % self.display_freq == 0 and self.num_timesteps > 0:
            self._log_detailed_metrics()
        return True

    def _log_detailed_metrics(self):
        """Affichage détaillé des métriques pour chaque worker (compatible SubprocVecEnv)."""
        try:
            logger.info("╭" + "─" * 90 + "╮")
            logger.info(
                "│" + " " * 30 + f"ÉTAPE {self.num_timesteps:,}" + " " * 30 + "│"
            )
            logger.info("╰" + "─" * 90 + "╯")

            self._display_model_metrics()

            infos = self.locals.get("infos")
            if isinstance(infos, list) and all(isinstance(i, dict) for i in infos):
                logger.info(f"📊 WORKERS ANALYSIS | Total: {len(infos)} workers")
                logger.info("=" * 92)
                for i, info in enumerate(infos):
                    final_info = info.get('final_info', info)
                    self._display_worker_summary(i, final_info)
            else:
                logger.info("Impossible de récupérer les 'infos' des workers depuis self.locals.")

            elapsed = time.time() - self.start_time
            steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
            logger.info("=" * 92)
            logger.info(
                f"⏱️  GLOBAL TIMING | Elapsed: {elapsed / 60:.1f}min | Speed: {steps_per_sec:.1f} steps/s"
            )
            logger.info("─" * 92)

        except Exception as e:
            logger.error(f"Erreur lors de l'affichage des métriques: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

            self._display_model_metrics()

            # Utiliser self.locals['infos'] qui est fourni par SB3 à chaque step
            # C'est la méthode la plus sûre pour le multiprocessing
            infos = self.locals.get("infos")
            if isinstance(infos, list) and all(isinstance(i, dict) for i in infos):
                logger.info(f"📊 WORKERS ANALYSIS | Total: {len(infos)} workers")
                logger.info("=" * 92)
                for i, info in enumerate(infos):
                    # L'info de l'environnement final (non-wrappé) est souvent sous la clé 'final_info'
                    final_info = info.get('final_info', info)
                    self._display_worker_summary(i, final_info)
            else:
                logger.info("Impossible de récupérer les 'infos' des workers depuis self.locals.")

            elapsed = time.time() - self.start_time
            steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
            logger.info("=" * 92)
            logger.info(
                f"⏱️  GLOBAL TIMING | Elapsed: {elapsed / 60:.1f}min | Speed: {steps_per_sec:.1f} steps/s"
            )
            logger.info("─" * 92)

        except Exception as e:
            logger.error(f"Erreur lors de l'affichage des métriques: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _display_individual_worker_metrics(self, worker_id: int, worker_env_wrapper):
        """Afficher les métriques détaillées d'un worker spécifique."""
        try:
            # Naviguer à travers les wrappers pour trouver l'environnement réel et ses métriques
            current_env = worker_env_wrapper
            metrics = None
            info = None

            # Essayer de trouver les métriques à travers les couches de wrappers
            while hasattr(current_env, "env") or hasattr(
                current_env, "get_portfolio_metrics"
            ):
                if hasattr(current_env, "get_portfolio_metrics"):
                    try:
                        metrics = current_env.get_portfolio_metrics()
                        break
                    except:
                        pass

                if hasattr(current_env, "last_info"):
                    info = current_env.last_info

                current_env = getattr(current_env, "env", None)
                if current_env is None:
                    break

            # Utiliser info comme fallback
            if not metrics and info:
                metrics = info

            if metrics:
                self._display_worker_summary(worker_id, metrics)
            else:
                logger.info(
                    f"│ WORKER {worker_id} | ❌ Impossible de récupérer les métriques."
                )

        except Exception as e:
            logger.error(
                f"Erreur lors de l'affichage des métriques du worker {worker_id}: {e}"
            )

    def _display_worker_summary(self, worker_id: int, metrics: dict):
        """Afficher le résumé complet des métriques d'un worker."""
        try:
            # Métriques de base (avec fallback pour compatibilité)
            portfolio_value = metrics.get("total_value", metrics.get("portfolio_value", self.initial_capital))
            cash = metrics.get("cash", self.initial_capital)
            roi = (
                ((portfolio_value - self.initial_capital) / self.initial_capital) * 100
                if self.initial_capital > 0
                else 0
            )
            drawdown = metrics.get("drawdown", 0.0)
            max_dd = metrics.get("max_drawdown", metrics.get("max_dd", 0.0))
            sharpe = metrics.get("sharpe_ratio", metrics.get("sharpe", 0.0))
            win_rate = metrics.get("win_rate", 0.0)
            total_trades = metrics.get("total_trades", metrics.get("trades", 0))
            trade_attempts = metrics.get("trade_attempts")
            invalid_trade_attempts = metrics.get("invalid_trade_attempts")

            # Métriques de trading détaillées
            valid_trades = metrics.get("valid_trades", 0)
            invalid_trades = (
                total_trades - valid_trades if total_trades > valid_trades else 0
            )
            current_positions = metrics.get("positions", {})
            closed_positions = metrics.get("closed_positions", [])

            # Informations de récompense et pénalités
            last_reward = metrics.get("last_reward", 0.0)
            last_penalty = metrics.get("last_penalty", 0.0)
            cumulative_reward = metrics.get("cumulative_reward", 0.0)

            # Dates et actifs
            current_date = metrics.get("current_date", "N/A")
            active_assets = list(current_positions.keys()) if current_positions else []

            # En-tête du worker
            logger.info(
                f"╭─── WORKER {worker_id} ────────────────────────────────────────────────────────────────╮"
            )

            # Ligne 1: Portfolio et Performance
            logger.info(
                f"│ 📊 PORTFOLIO  | Valeur: ${portfolio_value:>10.2f} | Cash: ${cash:>10.2f} | ROI: {roi:>+7.2f}% │"
            )

            # Ligne 2: Risk Management
            logger.info(
                f"│ ⚠️  RISK      | Drawdown: {drawdown:>6.2f}% | Max DD: {max_dd:>6.2f}% | Sharpe: {sharpe:>6.2f}     │"
            )

            # Ligne 3: Trading Statistics
            logger.info(
                f"│ 📈 TRADING    | Total: {total_trades:>3d} | Valid: {valid_trades:>3d} | Invalid: {invalid_trades:>3d} | Win Rate: {win_rate:>5.1f}% │"
            )

            # Ligne 3b: Actions de l'agent
            if trade_attempts is not None and invalid_trade_attempts is not None:
                invalid_rate = (invalid_trade_attempts / trade_attempts * 100) if trade_attempts > 0 else 0
                logger.info(
                    f"│ 🎯 ACTIONS    | Tentatives: {trade_attempts:>4d} | Invalides: {invalid_trade_attempts:>4d} ({invalid_rate:5.1f}%)             │"
                )

            # Ligne 4: Rewards & Penalties
            logger.info(
                f"│ 🎯 REWARDS    | Last: {last_reward:>+8.4f} | Penalty: {last_penalty:>+8.4f} | Cumul: {cumulative_reward:>+8.2f}   │"
            )

            # Ligne 5: Temporal & Assets
            date_str = str(current_date)[:10] if current_date != "N/A" else "N/A"
            assets_str = ", ".join(active_assets[:3]) if active_assets else "Aucun"
            if len(active_assets) > 3:
                assets_str += f"+{len(active_assets) - 3}"
            logger.info(
                f"│ 📅 CONTEXT    | Date: {date_str:>10s} | Active Assets: {assets_str:<25s}              │"
            )

            # Positions ouvertes détaillées (si présentes)
            if current_positions:
                logger.info("│ ├─ POSITIONS OUVERTES:" + " " * 54 + "│")
                for asset, pos in list(current_positions.items())[
                    :3
                ]:  # Max 3 pour l'affichage
                    if isinstance(pos, dict):
                        size = pos.get("size", 0)
                        entry_price = pos.get("entry_price", 0)
                        current_value = pos.get("value", 0)
                        pnl = pos.get("unrealized_pnl", 0)
                        logger.info(
                            f"│ │  {asset:<8s} | Size: {size:>6.2f} @ {entry_price:>8.4f} | Val: ${current_value:>7.2f} | PnL: {pnl:>+6.2f} │"
                        )

                remaining = len(current_positions) - 3
                if remaining > 0:
                    logger.info(
                        f"│ │  ... et {remaining} autres positions" + " " * 44 + "│"
                    )

            # Derniers trades fermés (si disponibles)
            if closed_positions:
                recent_closed = (
                    closed_positions[-2:]
                    if len(closed_positions) >= 2
                    else closed_positions
                )
                logger.info("│ ├─ DERNIERS TRADES FERMÉS:" + " " * 48 + "│")
                for trade in recent_closed:
                    if isinstance(trade, dict):
                        asset = trade.get("asset", "N/A")
                        profit = trade.get("profit", 0)
                        duration = trade.get("duration", "N/A")
                        close_reason = trade.get("reason", "N/A")[:8]
                        logger.info(
                            f"│ │  {asset:<8s} | Profit: {profit:>+8.2f} | Durée: {str(duration):<6s} | Raison: {close_reason:<8s}  │"
                        )

            logger.info(
                f"╰──────────────────────────────────────────────────────────────────────────────────╯"
            )

        except Exception as e:
            logger.error(
                f"Erreur lors de l'affichage des métriques du worker {worker_id}: {e}"
            )
            logger.info(f"│ WORKER {worker_id} | ❌ Erreur d'affichage des métriques.")

    def _display_model_metrics(self):
        """Afficher les métriques globales du modèle PPO."""
        try:
            # Récupérer les métriques du modèle PPO
            model_metrics = {}
            total_loss = 0.0
            policy_loss = 0.0
            value_loss = 0.0
            entropy = 0.0

            if hasattr(self.model, "logger") and hasattr(
                self.model.logger, "name_to_value"
            ):
                model_metrics = self.model.logger.name_to_value
                total_loss = model_metrics.get("train/loss", 0.0)
                policy_loss = model_metrics.get("train/policy_loss", 0.0)
                value_loss = model_metrics.get("train/value_loss", 0.0)
                entropy = model_metrics.get("train/entropy_loss", 0.0)

            # Model Learning Metrics
            logger.info(
                f"🧠 MODEL | Loss: {total_loss:.4f} | Policy: {policy_loss:+.4f} | "
                f"Value: {value_loss:.4f} | Entropy: {entropy:.4f}"
            )

        except Exception as e:
            logger.error(f"Erreur lors de l'affichage des métriques du modèle: {e}")

    def _on_rollout_end(self):
        """Appelé à la fin de chaque rollout pour capturer les positions fermées."""
        try:
            # Essayer de récupérer les positions fermées via les wrappers améliorés
            if hasattr(self.model, "get_env"):
                env = self.model.get_env()
                closed_positions = []

                try:
                    # Si c'est un environnement vectorisé, essayer d'accéder au premier environnement
                    if hasattr(env, "envs") and len(env.envs) > 0:
                        first_env = env.envs[0]
                        # Naviguer à travers les wrappers pour trouver notre GymnasiumToGymWrapper
                        current_env = first_env
                        while hasattr(current_env, "env"):
                            if isinstance(current_env, GymnasiumToGymWrapper):
                                metrics = current_env.get_portfolio_metrics()
                                closed_positions = metrics.get("closed_positions", [])
                                break
                            current_env = (
                                current_env.env if hasattr(current_env, "env") else None
                            )
                            if current_env is None:
                                break

                    # Note: ne pas utiliser get_attr("last_info") en mode Subproc (non picklable / attribut absent)

                    if closed_positions:
                        logger.info(
                            "╭" + "─" * 25 + " Positions Fermées " + "─" * 25 + "╮"
                        )
                        for pos in closed_positions:
                            if isinstance(pos, dict):
                                asset = pos.get("asset", "Unknown")
                                size = pos.get("size", 0)
                                entry_price = pos.get("entry_price", 0)
                                exit_price = pos.get("exit_price", 0)
                                pnl = pos.get("pnl", 0)
                                pnl_pct = pos.get("pnl_pct", 0)
                                logger.info(
                                    f"│ {asset}: Taille: {size:.2f} | Entrée: {entry_price:.4f} | "
                                    f"Sortie: {exit_price:.4f} | PnL: ${pnl:.2f} ({pnl_pct:.2f}%)"
                                    + " " * 5
                                    + "│"
                                )
                        logger.info("╰" + "─" * 68 + "╯")
                except Exception as e:
                    logger.debug(f"Impossible de récupérer les positions fermées: {e}")
        except Exception as e:
            logger.error(f"Erreur lors du traitement des positions fermées: {e}")

    def _on_training_end(self):
        """Fin de l'entraînement avec résumé complet."""
        elapsed = time.time() - self.start_time
        logger.info("╭" + "─" * 60 + "╮")
        logger.info("│" + " " * 15 + "✅ ENTRAÎNEMENT TERMINÉ" + " " * 15 + "│")
        logger.info("╰" + "─" * 60 + "╯")
        logger.info(f"[TRAINING END] Total steps: {self.num_timesteps:,}")
        logger.info(f"[TRAINING END] Duration: {elapsed / 60:.1f} minutes")
        logger.info(f"[TRAINING END] Episodes: {self.episode_count}")

        # Résumé final des performances
        if self.episode_rewards:
            final_reward = (
                np.mean(self.episode_rewards[-10:])
                if len(self.episode_rewards) >= 10
                else np.mean(self.episode_rewards)
            )
            logger.info(f"[TRAINING END] Final Mean Reward: {final_reward:.2f}")

        logger.info(f"[TRAINING END] Correlation ID: {self.correlation_id}")

class WorkerMetricsLogger(BaseCallback):
    """Journalise les métriques par worker vers TensorBoard et JSONL."""

    def __init__(
        self,
        log_dir: str,
        initial_balance: float,
        log_interval: int = 500,
        tensorboard_prefix: str = "workers",
    ) -> None:
        super().__init__(verbose=0)
        self.log_dir = log_dir
        self.initial_balance = float(initial_balance)
        self.log_interval = max(1, int(log_interval))
        self.tensorboard_prefix = tensorboard_prefix.rstrip("/")
        self.log_path: Optional[str] = None
        self._log_file = None
        self._last_worker_step: Dict[str, int] = {}

    def _ensure_log_file(self) -> None:
        if self._log_file is None and self.log_path:
            self._log_file = open(self.log_path, "a", encoding="utf-8")

    def _close_log_file(self) -> None:
        if self._log_file:
            try:
                self._log_file.close()
            finally:
                self._log_file = None

    def _unwrap_envs(self):
        env = getattr(self, "training_env", None)
        if env is None:
            return []
        current = env
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if hasattr(current, "envs"):
                return getattr(current, "envs", [])
            if hasattr(current, "venv"):
                current = getattr(current, "venv")
                continue
            if hasattr(current, "env"):
                current = getattr(current, "env")
                continue
            break
        return []

    def _extract_metrics(self, env) -> Optional[Dict[str, Any]]:
        depth = 0
        current = env
        while current is not None and depth < 10:
            if hasattr(current, "get_portfolio_metrics"):
                try:
                    metrics = current.get_portfolio_metrics()
                    if isinstance(metrics, dict):
                        return metrics
                except Exception:
                    pass
            current = getattr(current, "env", None)
            depth += 1
        return None

    @staticmethod
    def _extract_worker_id(env, fallback: int) -> int:
        depth = 0
        current = env
        while current is not None and depth < 10:
            worker_id = getattr(current, "worker_id", None)
            if worker_id is not None:
                try:
                    return int(worker_id)
                except (TypeError, ValueError):
                    return fallback
            current = getattr(current, "env", None)
            depth += 1
        return fallback

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _to_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: WorkerMetricsLogger._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [WorkerMetricsLogger._sanitize(v) for v in obj]
        if isinstance(obj, tuple):
            return [WorkerMetricsLogger._sanitize(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    def _write_record(self, record: Dict[str, Any]) -> None:
        if not self._log_file:
            return
        try:
            self._log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._log_file.flush()
        except Exception as exc:
            logger.error("[WorkerMetricsLogger] Échec d'écriture JSON: %s", exc)

    def _log_tensorboard(
        self,
        worker_label: str,
        portfolio_value: float,
        penalty: float,
        inference_count: int,
        drawdown_pct: float,
        sharpe_ratio: float,
        trades: int,
    ) -> None:
        if self.logger is None:
            return
        base = f"{self.tensorboard_prefix}/{worker_label}"
        self.logger.record(f"{base}/capital", portfolio_value)
        self.logger.record(f"{base}/penalty", penalty)
        self.logger.record(f"{base}/inference_count", float(inference_count))
        self.logger.record(f"{base}/drawdown_pct", drawdown_pct)
        self.logger.record(f"{base}/sharpe", sharpe_ratio)
        self.logger.record(f"{base}/trades", float(trades))

    def _on_training_start(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(
            self.log_dir, f"parallel_training_results_{timestamp}.jsonl"
        )
        self._ensure_log_file()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_interval != 0:
            return True
        envs = self._unwrap_envs()
        if not envs:
            return True
        timestamp = datetime.utcnow().isoformat()
        total_workers = len(envs)

        for idx, worker_env in enumerate(envs):
            metrics = self._extract_metrics(worker_env)
            if not metrics:
                continue
            worker_id = self._extract_worker_id(worker_env, idx)
            worker_label = f"worker_{worker_id}"

            current_step = self._to_int(metrics.get("step", self.num_timesteps), self.num_timesteps)
            if self._last_worker_step.get(worker_label) == current_step:
                continue
            self._last_worker_step[worker_label] = current_step

            reward_components = metrics.get("reward_components", {}) or {}
            penalty = self._to_float(
                reward_components.get("frequency_penalty"), metrics.get("last_penalty", 0.0)
            )
            portfolio_value = self._to_float(metrics.get("portfolio_value"), self.initial_balance)
            cash = self._to_float(metrics.get("cash"), self.initial_balance)
            drawdown_pct = self._to_float(metrics.get("drawdown"))
            sharpe_ratio = self._to_float(metrics.get("sharpe"))
            trades = self._to_int(metrics.get("trades", 0))
            positions = self._to_int(metrics.get("num_positions", 0))
            inference_count = self._to_int(metrics.get("step", self.num_timesteps))
            action_stats = metrics.get("action_stats", {}) or {}

            record = {
                "timestamp": timestamp,
                "instance_id": worker_label,
                "worker_id": worker_id,
                "step": current_step,
                "episode": self._to_int(metrics.get("chunk", 0)),
                "reward": self._to_float(metrics.get("last_reward", 0.0)),
                "total_reward": self._to_float(metrics.get("cumulative_reward", 0.0)),
                "penalty": penalty,
                "positions": positions,
                "trades": trades,
                "parallel": {"workers": total_workers, "active": total_workers},
                "inference": {
                    "count": inference_count,
                    "action_mean": self._to_float(action_stats.get("action_mean")),
                    "action_std": self._to_float(action_stats.get("action_std")),
                    "action_min": self._to_float(action_stats.get("action_min")),
                    "action_max": self._to_float(action_stats.get("action_max")),
                },
                "portfolio": {
                    "portfolio_value": portfolio_value,
                    "cash": cash,
                    "drawdown": drawdown_pct,
                    "max_drawdown": self._to_float(metrics.get("max_dd")),
                    "sharpe": sharpe_ratio,
                    "sortino": self._to_float(metrics.get("sortino")),
                    "win_rate": self._to_float(metrics.get("win_rate")),
                    "leverage": self._to_float(metrics.get("leverage")),
                },
                "risk": self._sanitize(metrics.get("risk_metrics", {})),
                "initial_balance": self.initial_balance,
            }

            self._write_record(record)
            self._log_tensorboard(
                worker_label,
                portfolio_value,
                penalty,
                inference_count,
                drawdown_pct,
                sharpe_ratio,
                trades,
            )

        return True

    def _on_training_end(self) -> None:
        self._close_log_file()


# Timeout and environment validation
from adan_trading_bot.utils.timeout_manager import (
    TimeoutManager,
    TimeoutException as TMTimeoutException,
)
from adan_trading_bot.training.trainer import validate_environment

# Configuration du logger de base
from adan_trading_bot.common.custom_logger import setup_logging


def resolve_config_variables(config):
    """
    Résout les variables de configuration de type ${variable.path}.
    Version silencieuse optimisée sans logs debug.
    """
    import copy

    # Utiliser des chemins fixes pour éviter les problèmes de résolution
    base_dir = "/home/morningstar/Documents/trading/bot"

    # Dictionnaire de substitution complet
    substitutions = {
        "${paths.base_dir}": base_dir,
        "${paths.data_dir}": f"{base_dir}/data",
        "${paths.raw_data_dir}": f"{base_dir}/data/raw",
        "${paths.processed_data_dir}": f"{base_dir}/data/processed",
        "${paths.indicators_data_dir}": f"{base_dir}/data/processed/indicators",
        "${paths.final_data_dir}": f"{base_dir}/data/final",
        "${paths.models_dir}": f"{base_dir}/models",
        "${paths.trained_models_dir}": f"{base_dir}/models/rl_agents",
        "${paths.logs_dir}": f"{base_dir}/logs",
        "${paths.reports_dir}": f"{base_dir}/reports",
        "${paths.figures_dir}": f"{base_dir}/reports/figures",
        "${paths.metrics_dir}": f"{base_dir}/reports/metrics",
        "${data.data_dirs.base}": f"{base_dir}/data/processed/indicators",
    }

    def simple_resolve(obj):
        if isinstance(obj, str):
            result = obj
            for var, value in substitutions.items():
                result = result.replace(var, value)
            return result
        elif isinstance(obj, dict):
            return {k: simple_resolve(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [simple_resolve(item) for item in obj]
        else:
            return obj

    return simple_resolve(copy.deepcopy(config))


def clean_worker_id(worker_id):
    """
    Nettoie l'ID du worker pour éviter les erreurs JSONL.
    Convertit 'w0' en 0, 'w1' en 1, etc.

    Args:
        worker_id: ID du worker (peut être string ou int)

    Returns:
        int: ID du worker nettoyé
    """
    if isinstance(worker_id, str):
        # Supprimer le préfixe 'w' si présent
        if worker_id.startswith("w"):
            try:
                return int(worker_id[1:])
            except ValueError:
                return 0
        # Essayer de convertir directement en int
        try:
            return int(worker_id)
        except ValueError:
            return 0
    elif isinstance(worker_id, int):
        return worker_id
    else:
        return 0


def log_worker_comparison(envs, n_workers):
    """
    Log comparison metrics between workers for debugging and analysis.

    Args:
        envs: List or VecEnv containing environment instances
        n_workers: Number of workers to compare
    """
    try:
        logger.info("=" * 80)
        logger.info("[WORKER COMPARISON] Performance Analysis")
        logger.info("=" * 80)

        for worker_id in range(n_workers):
            try:
                # Get environment instance
                if hasattr(envs, "get_attr"):
                    # VecEnv case
                    worker_env = (
                        envs.get_attr("envs")[0][worker_id]
                        if hasattr(envs.get_attr("envs")[0], "__getitem__")
                        else None
                    )
                elif isinstance(envs, list):
                    # List of envs
                    worker_env = envs[worker_id] if worker_id < len(envs) else None
                else:
                    worker_env = None

                if worker_env is None:
                    logger.warning(
                        f"[COMPARISON] Worker {worker_id}: Unable to access environment"
                    )
                    continue

                # Get performance metrics
                if hasattr(worker_env, "portfolio_manager") and hasattr(
                    worker_env.portfolio_manager, "metrics"
                ):
                    metrics = (
                        worker_env.portfolio_manager.metrics.calculate_metrics()
                        if hasattr(
                            worker_env.portfolio_manager.metrics, "calculate_metrics"
                        )
                        else {}
                    )
                    equity = (
                        worker_env.portfolio_manager.get_equity()
                        if hasattr(worker_env.portfolio_manager, "get_equity")
                        else 0.0
                    )
                    positions_count = getattr(worker_env, "positions_count", {})

                    logger.info(
                        f"[COMPARISON Worker {worker_id}] "
                        f"Trades: {metrics.get('total_trades', 0)}, "
                        f"Winrate: {metrics.get('winrate', 0.0):.1f}%, "
                        f"Equity: {equity:.2f} USDT, "
                        f"Counts: {positions_count}"
                    )
                else:
                    logger.info(f"[COMPARISON Worker {worker_id}] No metrics available")

            except Exception as e:
                logger.warning(
                    f"[COMPARISON Worker {worker_id}] Error accessing metrics: {e}"
                )

        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error in worker comparison: {e}")


# Configurer le logger avec la configuration personnalisée
logger = setup_logging(
    default_level=logging.INFO,
    enable_console_logs=True,
    enable_json_logs=False,  # Désactiver les logs JSON par défaut
    force_plain_console=True,  # Forcer un affichage simple de la console
)

# Désactiver les avertissements spécifiques
warnings.filterwarnings(
    action="ignore", category=UserWarning, module="stable_baselines3"
)
warnings.filterwarnings(action="ignore", category=FutureWarning)


class GymnasiumToGymWrapper(gym.Wrapper):
    """
    Wrapper minimal pour adapter un env Gymnasium (reset -> (obs,info), step -> (obs,rew,term,trunc,info))
    à l'API Gym attendue par certains composants (SB3 DummyVecEnv / Monitor).
    """

    def __init__(self, env, rank=0):
        """Initialise le wrapper avec un rank pour éviter les logs dupliqués."""
        super().__init__(env)
        self.rank = rank
        self.log_prefix = f"[WORKER-{rank}]"
        self.last_info = {}
        self.last_obs = None
        self.episode_rewards = []
        self.episode_count = 0

    def reset(self, *, seed=None, options=None):
        """Garantit que reset() retourne toujours un tuple (obs, info)."""
        reset_result = super().reset(seed=seed, options=options)
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
            return obs, info
        else:
            return reset_result, {}

    def get_metrics(self):
        """Retourne les métriques actuelles de l'environnement."""
        return {
            "last_info": self.last_info,
            "episode_count": getattr(self, "episode_count", 0),
            "episode_rewards": getattr(self, "episode_rewards", []),
            "last_obs": self.last_obs,
        }

    def get_portfolio_metrics(self):
        """Retourne spécifiquement les métriques de portfolio."""
        if hasattr(self, "last_info") and self.last_info:
            return {
                "portfolio_value": self.last_info.get("portfolio_value", 0),
                "cash": self.last_info.get("cash", 0),
                "drawdown": self.last_info.get("drawdown", 0),
                "positions": self.last_info.get("positions", {}),
                "closed_positions": self.last_info.get("closed_positions", []),
                "sharpe": self.last_info.get("sharpe", 0),
                "sortino": self.last_info.get("sortino", 0),
                "profit_factor": self.last_info.get("profit_factor", 0),
                "max_dd": self.last_info.get("max_dd", 0),
                "cagr": self.last_info.get("cagr", 0),
                "win_rate": self.last_info.get("win_rate", 0),
                "trades": self.last_info.get("trades", 0),
            }
        return {}

    def step(self, action):
        """Convertit le retour de step() de gymnasium (5 valeurs) au format SB3."""
        obs, reward, terminated, truncated, info = super().step(action)
        self.last_info = info.copy() if isinstance(info, dict) else {}
        self.last_obs = obs
        return obs, float(reward), terminated, truncated, info


# Gestion des exceptions
class TimeoutException(Exception):
    """Exception levée quand le timeout est atteint (legacy)."""

    pass


# Import local
from adan_trading_bot.environment.checkpoint_manager import CheckpointManager

# Import local
from adan_trading_bot.utils.caching_utils import DataCacheManager

# Configuration du logger
logger = logging.getLogger(__name__)

# Définir le répertoire racine du projet
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(PROJECT_ROOT)

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Extrait les caractéristiques à partir des observations de type Dict.
    Hérite de BaseFeaturesExtractor pour une meilleure intégration avec SB3.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,  # Dimension de sortie requise par SB3
        cnn_output_dim: int = 64,
        mlp_extractor_net_arch: Optional[List[int]] = None,
    ) -> None:
        # Appel au constructeur parent avec la dimension de sortie
        super().__init__(observation_space, features_dim=features_dim)

        extractors = {}
        total_concat_size = 0

        # Pour chaque clé de l'espace d'observation
        for key, subspace in observation_space.spaces.items():
            if key == "observation":  # Traitement des données d'image
                # Calcul de la taille après aplatissement
                n_flatten = 1
                for i in range(len(subspace.shape)):
                    n_flatten *= subspace.shape[i]

                # Réseau pour traiter les données d'image
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(n_flatten, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                )
                total_concat_size += 128
            else:  # Traitement des données vectorielles
                # Utilisation d'un MLP simple pour les données vectorielles
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                )
                total_concat_size += 32

        self.extractors = nn.ModuleDict(extractors)

        # Couche linéaire finale pour adapter à la dimension de sortie souhaitée
        self.fc = nn.Sequential(nn.Linear(total_concat_size, features_dim), nn.ReLU())

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        encoded_tensor_list = []

        # Extraire les caractéristiques pour chaque clé
        for key, extractor in self.extractors.items():
            if key in observations:
                # S'assurer que les observations sont au bon format
                x = observations[key]
                if isinstance(x, np.ndarray):
                    x = th.as_tensor(x, device=self.device, dtype=th.float32)
                encoded_tensor_list.append(extractor(x))

        # Concaténer toutes les caractéristiques et appliquer la couche finale
        return self.fc(th.cat(encoded_tensor_list, dim=1))


class GymnasiumToSB3Wrapper(gym.Wrapper):
    """Wrapper pour convertir un environnement Gymnasium en un format compatible avec Stable Baselines 3."""

    def __init__(self, env: gym.Env) -> None:
        """Initialize the Gymnasium to SB3 wrapper.

        Args:
            env: The Gymnasium environment to wrap
        """
        super().__init__(env)

        # Convertir l'espace d'observation en un format compatible
        if isinstance(env.observation_space, gym.spaces.Dict):
            # Pour les espaces de type Dict, on conserve la structure
            self.observation_space = env.observation_space
        else:
            # Pour les autres types d'espaces, on essaie de les convertir
            self.observation_space = env.observation_space

        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {"render_modes": []})

        # Activer le mode vectorisé si nécessaire
        self.is_vector_env = hasattr(env, "num_envs")

    def reset(self, **kwargs):
        """Reset the environment and return the initial observation and info."""
        obs, info = self.env.reset(**kwargs)

        # S'assurer que l'observation est au bon format
        if isinstance(obs, dict) and "observation" in obs and "portfolio_state" in obs:
            # Déjà au bon format
            return obs, info

        # Gérer les autres formats d'observation si nécessaire
        return obs, info

    def step(self, action):
        """Take an action in the environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # S'assurer que l'observation est au bon format
        if isinstance(obs, dict) and "observation" in obs and "portfolio_state" in obs:
            # Déjà au bon format
            return obs, reward, terminated or truncated, False, info

        # Gérer les autres formats d'observation si nécessaire
        return obs, reward, terminated or truncated, False, info

    def render(self, mode: str = "human"):
        return self.env.render(mode)


# Local application imports
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine

# Import déjà effectué plus haut
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
    VecTransposeImage,
    VecEnv,
)
from stable_baselines3.common.utils import set_random_seed

# SOLUTION IMMORTALITÉ ADAN: Registre global des DBE pour survivre aux recréations d'environnement
_GLOBAL_DBE_REGISTRY = {}


def _normalize_obs_for_sb3(obs: Any) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Normalize observation for Stable Baselines 3 compatibility.

    Handles Gym's tuple (obs, info) and ensures proper numpy arrays.
    Converts observations to the format expected by SB3.

    Args:
        obs: Observation to normalize (can be tuple, dict, numpy array).
            L'observation à normaliser (peut être un tuple, un dict,
            un tableau numpy).

    Returns:
        Union[np.ndarray, Dict[str, np.ndarray]]: Normalized observation for SB3.
    """
    # Handle case where obs is a tuple (obs, info) from Gymnasium
    if isinstance(obs, tuple) and len(obs) >= 1:
        obs = obs[0]  # Take only the observation part

    # Handle dict observations (for MultiInputPolicy)
    if isinstance(obs, dict):
        # Ensure all values are numpy arrays and handle potential tuples
        normalized_obs = {}
        for k, v in obs.items():
            if isinstance(v, tuple):
                v = v[0]  # Take first element if it's a tuple
            normalized_obs[k] = np.asarray(v, dtype=np.float32)
        return normalized_obs

    # Handle numpy arrays
    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32, copy=False)

    # Handle PyTorch tensors
    if hasattr(obs, "numpy"):
        return obs.detach().cpu().numpy()

    # Handle lists and other array-like objects
    try:
        return np.asarray(obs, dtype=np.float32)
    except Exception as e:
        error_msg = f"Could not convert observation to numpy array: {obs}, error: {e}"
        raise ValueError(error_msg) from e


class ResetObsAdapter:
    """
    Adapter minimal pour rendre un env compatible avec Stable-Baselines3 (Gym API).
    - Si env.reset() renvoie (obs, info) (Gymnasium), on renvoie obs uniquement.
    - Si env.step() renvoie 5 éléments (obs, reward, terminated, truncated, info),
      on convertit en (obs, reward, done, info) où done = terminated or truncated.
    Utiliser : env = ResetObsAdapter(env)
    """

    def __init__(self, env):
        self.env = env
        self.observation_space = getattr(env, "observation_space", None)
        self.action_space = getattr(env, "action_space", None)
        self.metadata = getattr(env, "metadata", {})
        # Propriété unwrapped pour la compatibilité avec SB3
        self.unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env

    def reset(self, **kwargs):
        # Appel à la méthode reset de l'environnement sous-jacent
        reset_result = self.env.reset(**kwargs)

        # Journalisation pour le débogage
        logger.debug(f"ResetObsAdapter.reset - Type de sortie: {type(reset_result)}")
        if isinstance(reset_result, tuple):
            logger.debug(
                f"ResetObsAdapter.reset - Longueur du tuple: {len(reset_result)}"
            )

        # Gérer le cas où l'environnement retourne un tuple (obs, info)
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
            logger.debug(
                "ResetObsAdapter.reset - Tuple (obs, info) détecté, retourne obs uniquement"
            )

            # Vérifier que l'observation est dans le bon format
            if not isinstance(obs, (np.ndarray, dict)):
                logger.warning(
                    f"ResetObsAdapter.reset - Type d'observation inattendu: {type(obs)}"
                )

            return obs

        # Si ce n'est pas un tuple, retourner tel quel
        logger.debug("ResetObsAdapter.reset - Sortie simple détectée")
        return reset_result

    def step(self, action):
        out = self.env.step(action)
        # gymnasium step: (obs, reward, terminated, truncated, info)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
            return obs, reward, done, info
        return out

    def render(self, *args, **kwargs):
        return getattr(self.env, "render")(*args, **kwargs)

    def close(self):
        return getattr(self.env, "close")()

    def seed(self, seed=None):
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return None


class TrainingProgressCallback(BaseCallback):
    """
    A custom callback that prints a detailed training progress table.
    Displays training metrics, performance, portfolio value, and learning stats.
    """

    def __init__(
        self,
        check_freq: int = 1000,  # Check every N time steps
        verbose: int = 1,
    ):
        super(TrainingProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.start_time = time.time()
        self.last_time = self.start_time
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.last_log_steps = 0

    def _on_training_start(self) -> None:
        """Print the header when training starts."""
        self.start_time = time.time()
        self.last_time = self.start_time
        print("\n" + "=" * 100)
        print("DÉMARRAGE DE L'ENTRAÎNEMENT")
        print("=" * 100)
        self._print_header()

    def _on_step(self) -> bool:
        """Called at each environment step."""
        # Update counters
        self.total_steps = self.num_timesteps

        # Log every check_freq steps
        if self.n_calls % self.check_freq == 0:
            self._log_progress()

        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        # Get rollout information
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            ep_info = self.model.ep_info_buffer[0][0]  # Get first env's first episode
            if "r" in ep_info and "l" in ep_info:
                self.episode_rewards.append(ep_info["r"])
                self.episode_lengths.append(ep_info["l"])
                self.episode_times.append(time.time() - self.last_time)
                self.last_time = time.time()

    def _log_progress(self) -> None:
        """Log training progress with detailed metrics."""
        # Calculate metrics
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        steps_per_sec = (
            (self.total_steps - self.last_log_steps) / (current_time - self.last_time)
            if current_time > self.last_time
            else 0
        )

        # Get environment statistics
        if hasattr(self.model.get_env(), "envs") and len(self.model.get_env().envs) > 0:
            env = self.model.get_env().envs[0]
            if hasattr(env, "env"):  # Handle potential wrappers
                env = env.env

            # Get portfolio metrics if available
            portfolio_value = getattr(env, "portfolio_value", 0)
            initial_balance = getattr(
                env, "initial_balance", 1
            )  # Avoid division by zero
            roi = (
                ((portfolio_value - initial_balance) / initial_balance) * 100
                if initial_balance > 0
                else 0
            )
        else:
            portfolio_value = 0
            roi = 0

        # Calculate episode statistics
        avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
        avg_length = np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0
        fps = self.total_steps / elapsed_time if elapsed_time > 0 else 0

        # Get learning statistics
        learning_stats = {}
        if hasattr(self.model, "logger") and hasattr(
            self.model.logger, "name_to_value"
        ):
            learning_stats = self.model.logger.name_to_value

        # Calculate ETA
        total_steps = self.model._total_timesteps
        progress = (self.total_steps / total_steps) * 100 if total_steps > 0 else 0
        remaining_steps = max(0, total_steps - self.total_steps)
        eta_seconds = (remaining_steps / steps_per_sec) if steps_per_sec > 0 else 0
        eta_str = str(timedelta(seconds=int(eta_seconds)))

        # Print progress table
        print("\n" + "-" * 100)
        print(f"PROGRESSION: {progress:.1f}% | ETA: {eta_str} | FPS: {fps:.1f}")
        print("-" * 100)

        # Print status table
        print("\nSTATUT:")
        print(f"Étape: {self.total_steps:,}/{total_steps:,} ({progress:.1f}%)")
        print(f"Temps écoulé: {str(timedelta(seconds=int(elapsed_time)))}")
        print(f"Temps restant estimé: {eta_str}")
        print(f"Vitesse: {steps_per_sec:.1f} étapes/seconde")

        # Print performance metrics
        print("\nPERFORMANCE:")
        print(f"Récompense moyenne (10 épisodes): {avg_reward:,.2f}")
        print(f"Longueur moyenne des épisodes: {avg_length:.1f} pas")
        print(f"Portefeuille: ${portfolio_value:,.2f} (ROI: {roi:+.2f}%)")

        # Print learning metrics if available
        if learning_stats:
            print("\nAPPRENTISSAGE:")
            for key, value in learning_stats.items():
                if (
                    "loss" in key.lower()
                    or "entropy" in key.lower()
                    or "value" in key.lower()
                ):
                    print(f"{key}: {value:.4f}")

        # Print system info
        print("\nSYSTÈME:")
        try:
            print(f"Utilisation CPU: {psutil.cpu_percent()}%")
            try:
                process = psutil.Process()
                mem_info = process.memory_info()
                print(f"Utilisation mémoire: {mem_info.rss / 1024 / 1024:.1f} MB")
            except Exception as e:
                print(f"Erreur lors de la lecture de l'utilisation mémoire: {str(e)}")
        except ImportError:
            print("Utilisation CPU: Non disponible (psutil non installé)")
        except Exception as e:
            print(f"Erreur lors de la lecture des informations système: {str(e)}")

        # Update last log time
        self.last_log_steps = self.total_steps
        self.last_time = current_time

    def _print_header(self) -> None:
        """Print the table header."""
        print("\n" + "=" * 100)
        print("SUIVI DE L'ENTRAÎNEMENT - ADAN TRADING BOT")
        print("=" * 100)
        print("Démarrage à:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("-" * 100)


class SB3GymCompatibilityWrapper(gym.Wrapper):
    """
    Wrapper pour assurer la compatibilité entre Gymnasium et Stable Baselines 3.
    Gère spécifiquement les espaces d'observation de type Dict pour les environnements de trading.
    """

    def __init__(self, env):
        super().__init__(env)

        # Enregistrer l'espace d'observation original
        self.original_obs_space = env.observation_space

        # S'assurer que l'espace d'action est correctement défini
        if not isinstance(
            env.action_space,
            (spaces.Discrete, spaces.Box, spaces.MultiDiscrete, spaces.MultiBinary),
        ):
            raise ValueError(
                f"Type d'espace d'action non supporté: {type(env.action_space)}"
            )

        # Pour les environnements de trading avec espace d'observation de type Dict
        if isinstance(env.observation_space, spaces.Dict):
            # Vérifier si nous avons les clés attendues
            if (
                "observation" in env.observation_space.spaces
                and "portfolio_state" in env.observation_space.spaces
            ):
                # Enregistrer l'espace d'observation original
                self.observation_space = env.observation_space

                # Extraire les espaces pour le traitement
                obs_space = env.observation_space.spaces["observation"]
                portfolio_space = env.observation_space.spaces["portfolio_state"]

                # Vérifier les dimensions
                expected_obs_shape = (
                    3,
                    20,
                    15,
                )  # Forme attendue: (timeframes, window_size, features)
                expected_portfolio_shape = (17,)  # Forme attendue pour le portefeuille

                # Vérifier le type et la forme de l'observation
                if not (
                    isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 3
                ):
                    raise ValueError(
                        f"Format d'observation non supporté. Attendu: Box 3D, obtenu: {type(obs_space)} avec forme {getattr(obs_space, 'shape', 'N/A')}"
                    )

                # Vérifier le type et la forme du portefeuille
                if not (
                    isinstance(portfolio_space, spaces.Box)
                    and len(portfolio_space.shape) == 1
                ):
                    raise ValueError(
                        f"Format d'état du portefeuille non supporté. Attendu: Box 1D, obtenu: {type(portfolio_space)} avec forme {getattr(portfolio_space, 'shape', 'N/A')}"
                    )

                # Avertissement si les formes ne correspondent pas exactement à ce qui est attendu
                if obs_space.shape != expected_obs_shape:
                    logger.warning(
                        f"Forme d'observation non standard: {obs_space.shape}, attendu {expected_obs_shape}. "
                        "Le modèle peut nécessiter un ajustement."
                    )

                if portfolio_space.shape != expected_portfolio_shape:
                    logger.warning(
                        f"Forme de l'état du portefeuille non standard: {portfolio_space.shape}, "
                        f"attendu {expected_portfolio_shape}. Le modèle peut nécessiter un ajustement."
                    )

                # Enregistrer les dimensions pour le traitement des observations
                self.obs_shape = obs_space.shape
                self.portfolio_dim = portfolio_space.shape[0]

                logger.info(
                    "SB3GymCompatibilityWrapper: Espace d'observation "
                    "configuré avec succès. Forme de l'observation: %s, "
                    "Dimension du portefeuille: %s",
                    obs_space.shape,
                    portfolio_space.shape,
                )
            else:
                raise ValueError(
                    "L'espace d'observation Dict doit contenir les clés 'observation' et 'portfolio_state'"
                )
        else:
            # Pour les autres types d'espaces, utiliser l'espace d'observation tel quel
            self.observation_space = env.observation_space

    def reset(self, **kwargs):
        try:
            # Appeler reset sur l'environnement sous-jacent
            reset_result = self.env.reset(**kwargs)

            # Journalisation pour le débogage
            logger.debug(
                f"SB3GymCompatibilityWrapper.reset - Type de sortie: {type(reset_result)}"
            )
            if isinstance(reset_result, tuple):
                logger.debug(
                    f"SB3GymCompatibilityWrapper.reset - Longueur du tuple: {len(reset_result)}"
                )

            # Extraire l'observation du résultat de reset
            if isinstance(reset_result, tuple):
                if len(reset_result) == 2:
                    obs, info = reset_result  # Format (obs, info) de Gymnasium
                else:
                    # Si le tuple a une longueur différente, prendre le premier élément comme observation
                    obs = reset_result[0] if len(reset_result) > 0 else {}
                    info = reset_result[1] if len(reset_result) > 1 else {}
            else:
                obs = reset_result
                info = {}

            # Journalisation supplémentaire
            logger.debug(f"Type d'observation après extraction: {type(obs)}")
            if hasattr(obs, "shape"):
                logger.debug(f"Forme de l'observation: {obs.shape}")
            elif isinstance(obs, dict):
                logger.debug(f"Clés de l'observation: {list(obs.keys())}")

            # Traiter l'observation pour la compatibilité avec SB3
            processed_obs = self._process_obs(obs)

            # Journalisation pour le débogage
            logger.debug(
                f"SB3GymCompatibilityWrapper.reset - Type d'observation traité: {type(processed_obs)}"
            )
            if hasattr(processed_obs, "shape"):
                logger.debug(
                    f"SB3GymCompatibilityWrapper.reset - Forme de l'observation: {processed_obs.shape}"
                )

            # Pour la compatibilité avec SB3, retourner un tuple (observation, info)
            return processed_obs, info

        except Exception as e:
            logger.error(f"Erreur dans SB3GymCompatibilityWrapper.reset: {e}")
            logger.error(traceback.format_exc())
            # Retourner une observation par défaut en cas d'erreur
            default_shape = (3 * 20 * 15) + 17
            return np.zeros(default_shape, dtype=np.float32), {}

    def step(self, action):
        try:
            # Appeler step sur l'environnement sous-jacent
            step_result = self.env.step(action)

            # Gérer les différents formats de retour
            if isinstance(step_result, tuple):
                if len(step_result) == 5:
                    # Format gymnasium : (obs, reward, terminated, truncated, info)
                    obs, reward, terminated, truncated, info = step_result
                    done = bool(terminated or truncated)
                elif len(step_result) == 4:
                    # Format gym ancien : (obs, reward, done, info)
                    obs, reward, done, info = step_result
                else:
                    raise ValueError(
                        f"Format de retour de step() non supporté: {step_result}"
                    )
            else:
                raise ValueError(
                    f"Le résultat de step() doit être un tuple, reçu: {type(step_result)}"
                )

            # Traiter l'observation pour la compatibilité avec SB3
            processed_obs = self._process_obs(obs)

            # Journalisation pour le débogage
            logger.debug(
                f"Step - Type d'observation: {type(obs)}, "
                f"Type traité: {type(processed_obs)}, "
                f"Forme: {getattr(processed_obs, 'shape', 'N/A')}, "
                f"Récompense: {reward:.4f}, "
                f"Terminé: {done}"
            )

            # Retourner le format attendu par SB3 (4 valeurs)
            return processed_obs, float(reward), done, info

        except Exception as e:
            logger.error(f"Erreur dans SB3GymCompatibilityWrapper.step: {e}")
            logger.error(traceback.format_exc())
            # Retourner une observation par défaut en cas d'erreur
            default_shape = (3 * 20 * 15) + 17
            return np.zeros(default_shape, dtype=np.float32), 0.0, True, {}

    def _process_obs(self, obs):
        """
        Traite l'observation pour la compatibilité avec SB3.
        Convertit un dictionnaire d'observations en un tableau NumPy 1D.

        Gère spécifiquement les observations avec la forme (3, 20, 15) pour les données de marché
        et (17,) pour l'état du portefeuille.
        """
        # Définir les formes attendues
        expected_market_shape = (3, 20, 15)  # Forme attendue pour les données de marché
        expected_portfolio_shape = (17,)  # Forme attendue pour l'état du portefeuille

        # Fonction utilitaire pour créer une observation par défaut
        def create_default_observation():
            market_obs = np.zeros(expected_market_shape, dtype=np.float32)
            portfolio_obs = np.zeros(expected_portfolio_shape, dtype=np.float32)
            return np.concatenate([market_obs.reshape(-1), portfolio_obs.reshape(-1)])

        try:
            # Si l'observation est déjà un tableau numpy, la retourner telle quelle
            if isinstance(obs, np.ndarray):
                return obs.astype(np.float32)

            # Si c'est un dictionnaire avec les clés attendues
            if (
                isinstance(obs, dict)
                and "observation" in obs
                and "portfolio_state" in obs
            ):
                # Valider le type et la forme des données d'observation
                if not isinstance(
                    obs["observation"], (np.ndarray, list, tuple)
                ) or not isinstance(obs["portfolio_state"], (np.ndarray, list, tuple)):
                    logger.warning(
                        "Type d'observation invalide. Utilisation d'une observation par défaut."
                    )
                    return create_default_observation()

                # Convertir les observations en tableaux NumPy
                try:
                    market_obs = np.array(obs["observation"], dtype=np.float32)
                    portfolio_obs = np.array(obs["portfolio_state"], dtype=np.float32)
                except Exception as e:
                    logger.error(f"Erreur lors de la conversion des observations: {e}")
                    return create_default_observation()

                # Valider et redimensionner les données de marché si nécessaire
                if market_obs.shape != expected_market_shape:
                    logger.warning(
                        f"Forme des données de marché non standard: {market_obs.shape}, "
                        f"attendu {expected_market_shape}. Redimensionnement en cours."
                    )
                    if len(market_obs.shape) == 3:
                        market_obs = market_obs[
                            :3, :20, :15
                        ]  # Tronquer aux dimensions attendues
                        if (
                            market_obs.shape[1] < 20
                        ):  # Si la dimension temporelle est trop petite
                            pad_width = [(0, 0), (0, 20 - market_obs.shape[1]), (0, 0)]
                            market_obs = np.pad(market_obs, pad_width, mode="constant")
                    else:
                        market_obs = np.zeros(expected_market_shape, dtype=np.float32)

                # Valider et redimensionner l'état du portefeuille si nécessaire
                if portfolio_obs.shape != expected_portfolio_shape:
                    logger.warning(
                        f"Forme de l'état du portefeuille non standard: {portfolio_obs.shape}, "
                        f"attendu {expected_portfolio_shape}. Redimensionnement en cours."
                    )
                    if len(portfolio_obs.shape) == 1 and portfolio_obs.size >= 17:
                        portfolio_obs = portfolio_obs[:17]  # Tronquer si trop grand
                    else:
                        portfolio_obs = np.zeros(
                            expected_portfolio_shape, dtype=np.float32
                        )

                # Vérifier la forme finale avant de concaténer
                expected_market_size = np.prod(expected_market_shape)
                if market_obs.size != expected_market_size:
                    logger.warning(
                        f"Taille des données de marché incorrecte: {market_obs.size}, "
                        f"attendu {expected_market_size}. Redimensionnement en cours."
                    )
                    market_obs = market_obs.reshape(-1)[:expected_market_size]
                    if market_obs.size < expected_market_size:
                        market_obs = np.pad(
                            market_obs,
                            (0, expected_market_size - market_obs.size),
                            mode="constant",
                        )
                    market_obs = market_obs.reshape(expected_market_shape)

                # Aplatir et concaténer les observations
                market_flat = market_obs.reshape(-1)
                portfolio_flat = portfolio_obs.reshape(-1)

                try:
                    processed_obs = np.concatenate([market_flat, portfolio_flat])

                    logger.debug(
                        f"Observation traitée - Marché: {market_obs.shape} -> {market_flat.shape}, "
                        f"Portefeuille: {portfolio_obs.shape} -> {portfolio_flat.shape}, "
                        f"Sortie: {processed_obs.shape}"
                    )
                except Exception as e:
                    logger.error(
                        f"Erreur lors de la concaténation des observations: {e}"
                    )
                    return create_default_observation()

                return processed_obs.astype(np.float32)

            # Si c'est un tuple (obs, info) de Gymnasium, extraire l'observation
            elif isinstance(obs, tuple) and len(obs) >= 2:
                logger.debug(
                    "Tuple d'observation détecté, extraction de l'élément d'observation"
                )
                return self._process_obs(obs[0])  # Traiter uniquement l'observation

            # Si c'est une séquence (liste, tuple, etc.), essayer de la convertir en tableau
            elif isinstance(obs, (list, tuple)):
                logger.debug(
                    f"Séquence d'observation détectée, conversion en tableau: {type(obs)}"
                )
                try:
                    arr = np.array(obs, dtype=np.float32)
                    expected_size = np.prod(expected_market_shape) + len(
                        expected_portfolio_shape
                    )

                    if arr.size == expected_size:
                        return arr.reshape(-1).astype(np.float32)
                    elif arr.size > expected_size:
                        logger.warning(
                            f"Troncature de l'observation de taille {arr.size} à {expected_size}"
                        )
                        return arr.flat[:expected_size].astype(np.float32)
                    else:
                        logger.warning(
                            f"Remplissage de l'observation de taille {arr.size} à {expected_size}"
                        )
                        return np.pad(
                            arr.reshape(-1),
                            (0, expected_size - arr.size),
                            mode="constant",
                        ).astype(np.float32)
                except Exception as e:
                    logger.error(f"Erreur lors de la conversion de la séquence: {e}")
                    raise

            # Pour les autres types, essayer une conversion directe
            logger.warning(
                f"Type d'observation non standard, tentative de conversion: {type(obs)}"
            )
            try:
                arr = np.array(obs, dtype=np.float32)
                expected_size = np.prod(expected_market_shape) + len(
                    expected_portfolio_shape
                )

                if arr.size == expected_size:
                    return arr.reshape(-1).astype(np.float32)
                elif arr.size > expected_size:
                    logger.warning(
                        f"Troncature de l'observation de taille {arr.size} à {expected_size}"
                    )
                    return arr.flat[:expected_size].astype(np.float32)
                else:
                    logger.warning(
                        f"Remplissage de l'observation de taille {arr.size} à {expected_size}"
                    )
                    return np.pad(
                        arr.reshape(-1), (0, expected_size - arr.size), mode="constant"
                    ).astype(np.float32)
            except Exception as e:
                logger.error(f"Échec de la conversion de l'observation: {e}")
                return create_default_observation()

        except Exception as e:
            logger.error(f"Erreur critique lors du traitement de l'observation: {e}")
            logger.error(f"Type d'observation: {type(obs).__name__}")

            # Journalisation détaillée pour le débogage
            if hasattr(obs, "shape"):
                logger.error(f"Forme de l'observation: {obs.shape}")
            elif hasattr(obs, "__len__"):
                logger.error(f"Longueur de l'observation: {len(obs)}")

            if isinstance(obs, dict):
                logger.error("Clés de l'observation:")
                for k, v in obs.items():
                    type_info = type(v).__name__
                    shape_info = f", shape={v.shape}" if hasattr(v, "shape") else ""
                    len_info = f", len={len(v)}" if hasattr(v, "__len__") else ""
                    logger.error(f"  {k}: type={type_info}{shape_info}{len_info}")

            logger.error("Traceback complet de l'erreur:", exc_info=True)

        # En cas d'erreur, retourner un tableau de zéros de la bonne dimension
        default_market = np.zeros(expected_market_shape, dtype=np.float32)
        default_portfolio = np.zeros(expected_portfolio_shape, dtype=np.float32)
        default_obs = np.concatenate([default_market.reshape(-1), default_portfolio])

        logger.warning(
            f"Retour d'une observation par défaut de forme {default_obs.shape} "
            f"(marché: {default_market.shape}, portefeuille: {default_portfolio.shape})"
        )

        return default_obs


def make_env(
    rank: int = 0,
    seed: Optional[int] = None,
    config: Dict = None,
    worker_config: Dict = None,
    dbe_registry: Dict = None,
) -> gym.Env:
    """
    Crée et configure un environnement pour un worker donné.

    Args:
        rank: Identifiant unique du worker
        seed: Graine pour la reproductibilité
        config: Configuration de l'environnement
        worker_config: Configuration spécifique au worker

    Returns:
        Un environnement Gym valide
    """
    # Configuration par défaut si config n'est pas fourni
    if config is None:
        config = {}

    # Configuration du worker si non fournie
    if worker_config is None:
        # Récupérer la liste des actifs depuis la configuration
        assets = [a.lower() for a in config.get("data", {}).get("assets", ["btcusdt"])]
        timeframes = [
            str(tf).lower()
            for tf in config.get("data", {}).get("timeframes", ["5m", "1h", "4h"])
        ]
        data_split = config.get("data", {}).get("data_split", "val").lower()

        worker_config = {
            "rank": rank,
            "worker_id": clean_worker_id(
                rank
            ),  # ID nettoyé pour éviter les erreurs JSONL
            "num_workers": config.get("num_workers", 1),
            "assets": assets,
            "timeframes": timeframes,
            "data_split": data_split,
            "data_loader": {
                "batch_size": config.get("batch_size", 32),
                "shuffle": True,
                "num_workers": 0,  # Désactiver le multithreading pour éviter les problèmes
            },
        }
        logger.info(
            f"[WORKER-{rank}] Configuration worker créée: assets={assets}, timeframes={timeframes}, data_split={data_split}"
        )

    # Extraire les paramètres nécessaires de la configuration
    data_config = config.get("data", {})
    env_config = config.get("environment", {})

    # Récupérer les paramètres du worker
    assets = [a for a in worker_config.get("assets", [])]
    timeframes = [str(tf).lower() for tf in worker_config.get("timeframes", [])]

    # Récupérer le data_split de la configuration du worker, avec une valeur par défaut
    data_split = worker_config.get("data_split", "val").lower()

    # Utiliser des chemins absolus avec fallbacks intelligents
    possible_paths = [
        Path("/home/morningstar/Documents/trading/data/processed/indicators")
        / data_split,
        Path("data/processed/indicators") / data_split,
        Path(__file__).parent.parent.parent
        / "data"
        / "processed"
        / "indicators"
        / data_split,
        Path(__file__).parent.parent / "data" / "processed" / "indicators" / data_split,
    ]

    data_dir = None
    for path in possible_paths:
        if path.exists():
            data_dir = path
            break

    # Si aucun chemin n'existe, utiliser le premier comme défaut
    if data_dir is None:
        data_dir = possible_paths[0]

    logger.info(f"Data split utilisé: {data_split}")
    logger.info(f"Dossier de données final: {data_dir}")

    # Créer le répertoire s'il n'existe pas (pour les tests)
    if not data_dir.exists():
        logger.warning(
            f"Répertoire de données {data_dir} non trouvé, création en cours..."
        )
        data_dir.mkdir(parents=True, exist_ok=True)

        # Créer des données factices pour les tests si aucune donnée n'existe
        if pd is None or np is None:
            logger.error(
                "Pandas ou NumPy non disponible pour créer des données factices"
            )
            raise ValueError(
                f"Le répertoire de données {data_dir} n'existe pas et impossible de créer des données factices"
            )

        for asset in assets:
            clean_asset = asset.replace("/", "").replace("-", "")
            asset_dir = data_dir / clean_asset.upper()
            asset_dir.mkdir(exist_ok=True)

            for tf in timeframes:
                file_path = asset_dir / f"{tf}.parquet"
                if not file_path.exists():
                    # Créer des données factices pour le test
                    dates = pd.date_range("2024-01-01", periods=1000, freq="5min")
                    fake_data = pd.DataFrame(
                        {
                            "timestamp": dates,
                            "open": np.random.uniform(0.5, 1.0, 1000),
                            "high": np.random.uniform(0.5, 1.0, 1000),
                            "low": np.random.uniform(0.5, 1.0, 1000),
                            "close": np.random.uniform(0.5, 1.0, 1000),
                            "volume": np.random.uniform(1000, 10000, 1000),
                        }
                    )

                    # Ajouter des indicateurs techniques factices
                    for i in range(10):  # 10 indicateurs factices
                        fake_data[f"indicator_{i}"] = np.random.uniform(-1, 1, 1000)

                    fake_data.to_parquet(file_path, index=False)
                    logger.info(f"Données factices créées: {file_path}")

    logger.info(f"Chargement des données depuis : {data_dir}")
    logger.info(f"Actifs: {assets}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Split de données: {data_split}")

    # Dictionnaire pour stocker les données chargées
    data = {}
    data_found = False

    # Charger les données pour chaque actif et chaque timeframe
    for asset in assets:
        # Nettoyer le nom de l'actif (supprimer / et -) et forcer en minuscules
        clean_asset = asset.replace("/", "").replace("-", "")
        data[clean_asset] = {}
        asset_data_found = False

        for tf in timeframes:
            # Construire le chemin du fichier dans le format: {split}/{asset}/{timeframe}.parquet
            file_path = data_dir / clean_asset / f"{tf}.parquet"

            # Vérifier si le fichier existe
            if not file_path.exists():
                logger.warning(f"Fichier non trouvé: {file_path}")
                continue

            try:
                # Charger les données Parquet
                df = pd.read_parquet(file_path)

                # Vérifier que le DataFrame n'est pas vide
                if df.empty:
                    logger.warning(f"Le fichier {file_path} est vide.")
                    continue

                # Stocker les données dans la structure attendue
                data[clean_asset.upper()][tf] = df
                logger.info(
                    f"Données chargées: {clean_asset.upper()}/{tf} - {len(df)} lignes"
                )
                data_found = True
                asset_data_found = True

            except Exception as e:
                logger.error(f"Erreur lors du chargement de {file_path}: {str(e)}")

        if not asset_data_found:
            logger.warning(
                f"Aucune donnée valide trouvée pour l'actif {clean_asset.upper()}"
            )
            del data[clean_asset.upper()]

    if not data:
        raise ValueError(
            "Aucune donnée valide n'a pu être chargée. "
            f"Vérifiez les chemins dans la configuration et assurez-vous que les fichiers existent dans {data_dir}."
        )

    # Définir la taille de la fenêtre et la configuration des caractéristiques
    window_size = config.get("environment", {}).get("window_size", 50)
    features_config = config.get("environment", {}).get("features_config", {})

    logger.info(f"Taille de la fenêtre: {window_size}")
    logger.info(f"Configuration des caractéristiques: {features_config}")

    # SOLUTION IMMORTALITÉ ADAN: Récupérer ou créer le DBE pour ce worker
    worker_id = worker_config.get("worker_id", f"w{rank}")
    existing_dbe = None

    if dbe_registry is not None:
        existing_dbe = dbe_registry.get(worker_id)
        if existing_dbe is not None:
            logger.critical(
                f"👑 RÉUTILISATION DBE IMMORTEL réussie pour Worker {worker_id}, DBE_ID={id(existing_dbe)}"
            )
        else:
            logger.critical(f"🆕 CRÉATION PREMIER DBE IMMORTEL pour Worker {worker_id}")

    # Créer l'environnement avec les données chargées
    env = MultiAssetChunkedEnv(
        data=data,
        timeframes=timeframes,
        window_size=window_size,
        features_config=features_config,
        max_steps=env_config.get("max_steps", 1000),
        initial_balance=env_config.get("initial_balance", 10000.0),
        commission=env_config.get("commission", 0.001),
        reward_scaling=env_config.get("reward_scaling", 1.0),
        render_mode=None,
        enable_logging=config.get("enable_logging", True),
        log_dir=config.get("log_dir", "logs"),
        worker_config=worker_config,
        config=config,
        external_dbe=existing_dbe,  # Passer le DBE existant s'il y en a un
    )

    # SOLUTION IMMORTALITÉ ADAN: Enregistrer le DBE nouvellement créé dans le registre
    if dbe_registry is not None and existing_dbe is None:
        if hasattr(env, "dbe") and env.dbe is not None:
            dbe_registry[worker_id] = env.dbe
            logger.critical(
                f"💾 DBE IMMORTEL SAUVEGARDÉ dans registre pour Worker {worker_id}, DBE_ID={id(env.dbe)}"
            )
        else:
            logger.error(
                f"❌ ÉCHEC sauvegarde DBE pour Worker {worker_id} - DBE non trouvé dans l'environnement"
            )

    # Appliquer le wrapper pour la compatibilité et éviter les logs dupliqués
    env = GymnasiumToGymWrapper(env, rank=rank)

    # Configurer la graine pour la reproductibilité
    if seed is not None:
        try:
            # Essayer d'utiliser np.random pour la graine
            import numpy as np

            np.random.seed(seed)

            # Configurer la graine de l'environnement si possible
            base = getattr(env, "unwrapped", env)
            if hasattr(base, "seed"):
                base.seed(seed)

            # Configurer la graine de l'espace d'action
            if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
                env.action_space.seed(seed)

            # Configurer la graine de l'espace d'observation
            if hasattr(env, "observation_space") and hasattr(
                env.observation_space, "seed"
            ):
                env.observation_space.seed(seed)

        except Exception as e:
            logger.warning(
                f"Impossible de configurer la graine pour l'environnement {rank}: {str(e)}"
            )

    logger.info(f"Environnement {rank} créé avec succès")
    return env


def main(
    config_path: str = "bot/config/config.yaml",
    timeout: Optional[int] = None,
    checkpoint_dir: str = "/mnt/new_data/trading_bot_checkpoints",
    shared_model_path: Optional[str] = None,
    resume: bool = False,
    num_envs: int = 4,
    use_subproc: bool = False,
) -> bool:
    """
    Fonction principale pour l'entraînement parallèle des agents ADAN.

    Args:
        config_path: Chemin vers le fichier de configuration YAML
        timeout: Délai maximum d'entraînement en secondes
        checkpoint_dir: Répertoire pour enregistrer les points de contrôle
        shared_model_path: Chemin vers un modèle partagé pour l'entraînement distribué
        resume: Reprendre l'entraînement à partir du dernier point de contrôle
        num_envs: Nombre d'environnements parallèles à exécuter
        use_subproc: Si True, utilise SubprocVecEnv au lieu de DummyVecEnv

    Returns:
        bool: True si l'entraînement s'est terminé avec succès, False sinon
    """
    try:
        # Supprimer tous les logs de debug pour un affichage plus propre
        logging.getLogger().setLevel(logging.ERROR)

        # Validate environment (Python version, deps, etc.)
        try:
            validate_environment()
        except Exception as e:
            print(f"Environment validation failed: {e}")
            raise

        # Charger la configuration
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Résoudre les variables de configuration (version simplifiée)
        config = resolve_config_variables(config)

        print("🚀 ADAN Training Bot - Configuration chargée")
        print(f"⏱️  Timeout configuré: {timeout} secondes")
        # Activer ou non la barre de progression pendant l'entraînement (config training.progress_bar)
        progress_bar = bool(config.get("training", {}).get("progress_bar", False))

        # Récupérer la liste des actifs et timeframes depuis la configuration
        data_config = config.get("data", {})
        file_structure = data_config.get("file_structure", {})
        assets = data_config.get("assets", ["BTCUSDT"])
        timeframes = file_structure.get("timeframes", ["5m", "1h", "4h"])
        seed = config.get("seed", 42)

        # Configuration commune pour tous les workers
        base_worker_config = {
            "num_workers": num_envs,
            "assets": assets,  # Liste des actifs
            "timeframes": timeframes,  # Liste des timeframes
            "chunk_sizes": {
                tf: 1000 for tf in timeframes
            },  # Chunk size augmentée pour le warm-up
            "data_loader": {
                "batch_size": config.get("batch_size", 32),
                "shuffle": True,
                "num_workers": 0,  # Important: laisser à 0 pour éviter les problèmes de fork
            },
            "use_subproc": use_subproc,  # Passer l'info au worker
            "enable_worker_id_logging": True,  # Activer les IDs workers pour réduire répétitions
        }

        # Choisir la classe d'environnement vectorisé
        VecEnvClass = SubprocVecEnv if use_subproc else DummyVecEnv

        # SOLUTION IMMORTALITÉ ADAN: Initialiser le registre global des DBE
        logger.critical(f"🏛️ INITIALISATION REGISTRE GLOBAL DBE pour {num_envs} workers")

        # Configurer les arguments pour SubprocVecEnv
        vec_env_kwargs = {}
        if use_subproc:
            # Activer les logs de debug du multiprocessing pour voir les erreurs des workers
            try:
                import multiprocessing.util as mp_util
                mp_util.log_to_stderr(logging.INFO)
            except Exception:
                pass

            # Utiliser 'spawn' pour éviter l'héritage d'état problématique avec certaines libs
            vec_env_kwargs.update({"start_method": "spawn"})
            logger.info(
                f"Utilisation de SubprocVecEnv avec {num_envs} processus parallèles (start_method='spawn')"
            )
        else:
            logger.info(f"Utilisation de DummyVecEnv (sans parallélisme réel)")

        # Créer les fonctions de création d'environnement
        def make_env_fn(rank: int, seed_val: int) -> Callable[[], gym.Env]:
            """Crée une fonction d'initialisation d'environnement pour un rang donné."""
            def _init() -> gym.Env:
                # Worker-specific config
                worker_config = base_worker_config.copy()
                worker_config.update({
                    "rank": rank,
                    "seed": seed_val,
                    "worker_id": f"worker_{rank}"
                })

                # Build required constructor args for MultiAssetChunkedEnv
                try:
                    # Assets and placeholder data dict
                    assets = config.get("data", {}).get("assets", ["BTCUSDT"]) or ["BTCUSDT"]
                    data_placeholder = {str(asset): {} for asset in assets}

                    # Timeframes
                    tfs = (
                        config.get("environment", {})
                              .get("observation", {})
                              .get("timeframes", config.get("data", {}).get("timeframes", ["5m", "1h", "4h"]))
                    )

                    # Window size
                    window_size = config.get("environment", {}).get("window_size", 20)

                    # Features config per timeframe (use observation features lists)
                    obs_features = (
                        config.get("environment", {})
                              .get("observation", {})
                              .get("features", {})
                    )
                    base_feats = obs_features.get("base", []) or ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
                    tf_indicators = obs_features.get("indicators", {})
                    features_config = {}
                    for tf in tfs:
                        tf_inds = tf_indicators.get(tf, [])
                        # Combine base + tf indicators (keep as-is; only length is used early)
                        features_config[str(tf)] = list(base_feats) + list(tf_inds)

                    # Environment scalars
                    env_cfg = config.get("environment", {})
                    max_steps = env_cfg.get("max_steps", 100000)
                    initial_balance = env_cfg.get("initial_balance", 10000.0)
                    commission = env_cfg.get("commission", 0.001)
                    reward_scaling = env_cfg.get("reward_scaling", 1.0)

                    return MultiAssetChunkedEnv(
                        data=data_placeholder,
                        timeframes=tfs,
                        window_size=int(window_size),
                        features_config=features_config,
                        max_steps=int(max_steps),
                        initial_balance=float(initial_balance),
                        commission=float(commission),
                        reward_scaling=float(reward_scaling),
                        worker_config=worker_config,
                        config=config,
                        total_workers=num_envs,
                    )
                except Exception as e:
                    logger.error(f"[WORKER-{rank}] Failed to build env args: {e}", exc_info=True)
                    raise
            return _init

        # Créer l'environnement vectorisé
        try:
            env = SubprocVecEnv(
                [make_env_fn(i, seed + i * 1000 if seed is not None else None) for i in range(num_envs)],
                start_method='spawn'
            )
            logger.info(f"Environnement vectorisé 'SubprocVecEnv' créé avec succès.")
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'environnement vectorisé: {e}", exc_info=True)
            raise

        # Ajouter la normalisation des observations si nécessaire
        if config.get("normalize_observations", True):
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
        # Configuration optimisée de l'agent PPO avec MultiInputPolicy pour gérer les espaces d'observation de type dictionnaire

        # Vérifier l'espace d'observation
        logger.info(f"Espace d'observation de l'environnement: {env.observation_space}")

        # Configuration du réseau de neurones pour la politique récurrente
        policy_kwargs = {
            # La dimension des features est définie dans la politique (128 par défaut)
            # et passée à l'extracteur TemporalFusionExtractor.
            "lstm_hidden_size": 256, # Taille de l'état caché du LSTM
            "n_lstm_layers": 2, # Nombre de couches LSTM
            "net_arch": dict(pi=[128, 64], vf=[128, 64]), # Couches après le LSTM
            "activation_fn": th.nn.ReLU,
        }

        # Créer ou charger le modèle PPO
        model = None
        latest_checkpoint = None

        # Afficher des informations sur le parallélisme
        logger.info("\n" + "=" * 80)
        logger.info(f"Configuration de l'entraînement:")
        logger.info(f"- Nombre d'environnements parallèles: {num_envs}")
        logger.info(
            f"- Type d'environnement: {'SubprocVecEnv' if use_subproc else 'DummyVecEnv'}"
        )
        logger.info(f"- Seed de base: {config.get('seed', 42)}")
        logger.info(f"- Device: {'auto'}")
        logger.info("=" * 80 + "\n")

        # Vérifier si on doit reprendre depuis un checkpoint
        # Créer le répertoire de checkpoints si nécessaire
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialiser le CheckpointManager
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=5,  # Garder les 5 derniers checkpoints
            checkpoint_interval=10000,  # Sauvegarder tous les 10 000 steps
            logger=logger,
        )

        # Vérifier si on doit reprendre depuis un checkpoint
        if resume and checkpoint_manager.list_checkpoints():
            latest_checkpoint = checkpoint_manager.list_checkpoints()[-1]
            logger.info(
                f"Reprise de l'entraînement depuis le checkpoint: {latest_checkpoint}"
            )

            # Créer un modèle minimal pour le chargement (RecurrentPPO)
            if RecurrentPPO is None:
                raise ImportError("sb3-contrib (RecurrentPPO) est requis. Installez-le: pip install sb3-contrib")
            model = RecurrentPPO(
                policy=CustomRecurrentPolicy,
                env=env,
                policy_kwargs=policy_kwargs if 'policy_kwargs' in locals() else None,
                verbose=1,
                seed=config.get("seed", 42),
                device="auto",
            )
            # Configure SB3 logger for resume case
            try:
                sb3_log_dir = Path("reports/tensorboard_logs")
                sb3_log_dir.mkdir(parents=True, exist_ok=True)
                new_logger = sb3_logger_configure(
                    str(sb3_log_dir), ["stdout", "csv", "tensorboard"]
                )
                model.set_logger(new_logger)
                logger.info(f"SB3 logger configured at {sb3_log_dir}")
            except Exception as e:
                logger.warning(f"Failed to configure SB3 logger: {e}")

            # Charger le checkpoint
            model, _, metadata = checkpoint_manager.load_checkpoint(
                checkpoint_path=latest_checkpoint, model=model, map_location="auto"
            )

            if metadata:
                logger.info(
                    "Checkpoint chargé - Épisode: %d, Steps: %d",
                    metadata.episode,
                    metadata.total_steps,
                )
                start_timesteps = metadata.total_steps
            else:
                logger.warning(
                    "Métadonnées du checkpoint non trouvées, démarrage à zéro"
                )
                start_timesteps = 0

        # Si pas de reprise ou échec de chargement, créer un nouveau modèle
        if model is None:
            logger.info("Création d'un nouveau modèle")
            if RecurrentPPO is None:
                raise ImportError("sb3-contrib (RecurrentPPO) est requis. Installez-le: pip install sb3-contrib")
            model = RecurrentPPO(
                policy=CustomRecurrentPolicy,
                env=env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=policy_kwargs if 'policy_kwargs' in locals() else None,
                verbose=1,
                seed=config.get("seed", 42),
                device="auto",
            )
            # Configure SB3 logger for fresh model
            try:
                sb3_log_dir = Path("reports/tensorboard_logs")
                sb3_log_dir.mkdir(parents=True, exist_ok=True)
                new_logger = sb3_logger_configure(
                    str(sb3_log_dir), ["stdout", "csv", "tensorboard"]
                )
                model.set_logger(new_logger)
                logger.info(f"SB3 logger configured at {sb3_log_dir}")
            except Exception as e:
                logger.warning(f"Failed to configure SB3 logger: {e}")
            start_timesteps = 0

        # Vérifier si on doit reprendre depuis un checkpoint
        if resume:
            try:
                checkpoints = checkpoint_manager.list_checkpoints()
                if not checkpoints:
                    logger.info(
                        "[TRAINING] Aucun checkpoint trouvé, démarrage d'un nouvel entraînement"
                    )
                    start_timesteps = 0
                else:
                    latest_checkpoint = checkpoints[-1]
                    logger.info(
                        f"[TRAINING] Tentative de reprise depuis le checkpoint: {latest_checkpoint}"
                    )

                    # Vérifier si le checkpoint existe toujours
                    if not os.path.exists(latest_checkpoint):
                        logger.warning(
                            f"[TRAINING] Le checkpoint {latest_checkpoint} n'existe plus"
                        )
                        start_timesteps = 0
                    else:
                        # Créer un modèle minimal pour le chargement (RecurrentPPO)
                        if RecurrentPPO is None:
                            raise ImportError("sb3-contrib (RecurrentPPO) est requis. Installez-le: pip install sb3-contrib")
                        model = RecurrentPPO(
                            policy=CustomRecurrentPolicy,
                            env=env,
                            policy_kwargs=policy_kwargs if 'policy_kwargs' in locals() else None,
                            verbose=1,
                            seed=config.get("seed", 42),
                            device="auto",
                        )

                        try:
                            # Charger le checkpoint
                            model, optimizer, metadata = (
                                checkpoint_manager.load_checkpoint(
                                    checkpoint_path=latest_checkpoint,
                                    model=model,
                                    map_location="auto",
                                )
                            )

                            if metadata is not None:
                                logger.info(
                                    "[TRAINING] Checkpoint chargé - Épisode: %d, Steps: %d",
                                    metadata.episode,
                                    metadata.total_steps,
                                )
                                start_timesteps = metadata.total_steps

                                # Vérifier la cohérence des métadonnées
                                if not hasattr(
                                    metadata, "total_steps"
                                ) or not isinstance(metadata.total_steps, int):
                                    logger.warning(
                                        "[TRAINING] Métadonnées de checkpoint invalides, réinitialisation du compteur d'étapes"
                                    )
                                    start_timesteps = 0
                            else:
                                logger.warning(
                                    "[TRAINING] Aucune métadonnée trouvée dans le checkpoint"
                                )
                                start_timesteps = 0

                        except Exception as e:
                            logger.error(
                                f"[TRAINING] Erreur lors du chargement du checkpoint: {str(e)}",
                                exc_info=True,
                            )
                            logger.warning(
                                "[TRAINING] Démarrage d'un nouvel entraînement"
                            )
                            start_timesteps = 0

            except Exception as e:
                logger.error(
                    f"[TRAINING] Erreur lors de la vérification des checkpoints: {str(e)}",
                    exc_info=True,
                )
                logger.warning("[TRAINING] Démarrage d'un nouvel entraînement")
                start_timesteps = 0
        else:
            logger.info("[TRAINING] Démarrage d'un nouvel entraînement (sans reprise)")
            start_timesteps = 0

        class CustomCheckpointCallback(BaseCallback):
            """
            Callback personnalisé pour la sauvegarde des checkpoints.
            """

            def __init__(self, checkpoint_manager: CheckpointManager, verbose: int = 0):
                """
                Initialise le callback de sauvegarde.

                Args:
                    checkpoint_manager: Gestionnaire de checkpoints
                    verbose: Niveau de verbosité
                """
                super().__init__(verbose)
                self.checkpoint_manager = checkpoint_manager
                self.last_save = 0
                self.last_checkpoint = None
                self.episode_rewards = []
                self._last_checkpoint_step = (
                    0  # Pour suivre la dernière étape de sauvegarde
                )

            def _on_step(self) -> bool:
                """
                Appelé à chaque étape d'entraînement pour gérer la sauvegarde des checkpoints.

                Returns:
                    bool: True pour continuer l'entraînement, False pour l'arrêter
                """
                # Vérifier si c'est le moment de sauvegarder un checkpoint
                if (
                    self.num_timesteps - self._last_checkpoint_step
                ) >= self.checkpoint_manager.checkpoint_interval:
                    self._save_checkpoint()
                    self._last_checkpoint_step = self.num_timesteps
                return True

            def _save_checkpoint(self):
                """
                Sauvegarde un checkpoint du modèle.
                """
                try:
                    # Récupérer les métriques actuelles
                    metrics = {}
                    if (
                        hasattr(self.model, "ep_info_buffer")
                        and len(self.model.ep_info_buffer) > 0
                    ):
                        # Calculer les statistiques de récompense sur les derniers épisodes
                        rewards = [
                            ep_info["r"]
                            for ep_info in self.model.ep_info_buffer
                            if "r" in ep_info
                        ]
                        if rewards:
                            metrics["mean_reward"] = float(np.mean(rewards))
                            metrics["min_reward"] = float(np.min(rewards))
                            metrics["max_reward"] = float(np.max(rewards))
                            metrics["num_episodes"] = len(rewards)

                    # Récupérer le numéro d'épisode actuel si disponible
                    episode = getattr(self.model, "_episode_num", 0)

                    # Sauvegarder le checkpoint
                    checkpoint_path = self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.model.policy.optimizer
                        if hasattr(self.model.policy, "optimizer")
                        else None,
                        episode=episode,
                        total_steps=self.num_timesteps,
                        metrics=metrics,
                        custom_data={
                            "config": config,
                            "policy_kwargs": policy_kwargs,
                            "env_config": config.get("env", {}),
                        },
                        is_final=False,
                    )

                    if checkpoint_path:
                        self.last_checkpoint = checkpoint_path
                        self.last_save = self.num_timesteps
                        logger.info(
                            "Checkpoint sauvegardé (étape %d, épisode %d, récompense moyenne: %s)",
                            self.num_timesteps,
                            episode,
                            metrics.get("mean_reward", "N/A"),
                        )

                        # Nettoyer les anciens checkpoints si nécessaire
                        self.checkpoint_manager._cleanup_old_checkpoints()

                except Exception as e:
                    logger.error(
                        "Erreur lors de la sauvegarde du checkpoint: %s",
                        str(e),
                        exc_info=True,
                    )

            def _on_training_end(self) -> None:
                """
                Appelé à la fin de l'entraînement pour sauvegarder un checkpoint final.
                """
                try:
                    # Sauvegarder un checkpoint final
                    metrics = {}
                    if (
                        hasattr(self.model, "ep_info_buffer")
                        and len(self.model.ep_info_buffer) > 0
                    ):
                        rewards = [
                            ep_info["r"]
                            for ep_info in self.model.ep_info_buffer
                            if "r" in ep_info
                        ]
                        if rewards:
                            metrics["final_mean_reward"] = float(np.mean(rewards))

                    episode = getattr(self.model, "_episode_num", 0)

                    checkpoint_path = self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.model.policy.optimizer
                        if hasattr(self.model.policy, "optimizer")
                        else None,
                        episode=episode,
                        total_steps=self.num_timesteps,
                        metrics=metrics,
                        custom_data={
                            "config": config,
                            "policy_kwargs": policy_kwargs,
                            "env_config": config.get("env", {}),
                            "training_completed": True,
                        },
                        is_final=True,
                    )

                    if checkpoint_path:
                        logger.info(
                            "[TRAINING] Checkpoint final sauvegardé: %s (étape %d, épisode %d)",
                            checkpoint_path,
                            self.num_timesteps,
                            episode,
                        )

                except Exception as e:
                    logger.error(
                        "[TRAINING] Erreur lors de la sauvegarde du checkpoint final: %s",
                        str(e),
                        exc_info=True,
                    )

        # Créer le callback de sauvegarde des checkpoints
        checkpoint_callback = CustomCheckpointCallback(
            checkpoint_manager=checkpoint_manager, verbose=1
        )

        # Initialiser la liste des callbacks avec le checkpoint
        callbacks = [checkpoint_callback]

        # Ajouter le callback hiérarchique pour l'affichage structuré (maintenant compatible Subproc)
        total_timesteps = config.get("training", {}).get("total_timesteps", 1000000)
        initial_capital = config.get("environment", {}).get("initial_balance", 10000.0)
        hierarchical_callback = HierarchicalTrainingCallback(
            verbose=1,
            display_freq=1000,
            total_timesteps=total_timesteps,
            initial_capital=initial_capital,
        )
        callbacks.append(hierarchical_callback)
        logger.info(
            "[TRAINING] Affichage hiérarchique activé avec métriques détaillées"
        )

        # En mode non-Subproc, ajouter les autres callbacks intrusifs
        if not use_subproc:
            # Ajouter le callback pour la visualisation de l'attention si activé
            if config.get('model', {}).get('diagnostics', {}).get('save_attention_maps', False):
                attention_cb = AttentionVisualizerCallback(
                    viz_cfg=config['model']['diagnostics'],
                    verbose=1
                )
                callbacks.append(attention_cb)

            # Ajouter le callback de progression personnalisé si activé dans la config
            use_custom_progress = config.get("training", {}).get(
                "use_custom_progress", False
            )
            if use_custom_progress:
                progress_callback = CustomTrainingInfoCallback(check_freq=1000, verbose=1)
                callbacks.append(progress_callback)
                logger.info(
                    "[TRAINING] Barre de progression personnalisée activée pour le suivi de l'entraînement"
                )


            metrics_log_dir = os.path.join(
                config.get("paths", {}).get("logs_dir", "logs"),
                "training_metrics",
            )
            metrics_log_interval = (
                config.get("monitoring", {}).get("metrics_log_interval")
                or config.get("training", {}).get("metrics_log_interval")
                or 500
            )
            worker_metrics_callback = WorkerMetricsLogger(
                log_dir=metrics_log_dir,
                initial_balance=initial_capital,
                log_interval=metrics_log_interval,
                tensorboard_prefix="workers",
            )
            callbacks.append(worker_metrics_callback)
            logger.info(
                "[TRAINING] Journalisation des métriques par worker activée (%s)",
                metrics_log_dir,
            )

        # Créer un CallbackList pour gérer plusieurs callbacks
        callback = CallbackList(callbacks)

        logger.info(
            f"[TRAINING] {len(callbacks)} callback(s) configuré(s) pour l'entraînement"
        )

        # Prepare timeout manager with cleanup callback
        tm = None
        if timeout is not None and timeout > 0:

            def _cleanup_on_timeout():
                logger.info(
                    "[TRAINING] Timeout reached, attempting to save checkpoint before exit..."
                )
                try:
                    # Ensure checkpoint directory exists
                    os.makedirs(checkpoint_manager.checkpoint_dir, exist_ok=True)
                except Exception:
                    pass
                try:
                    optimizer = (
                        getattr(model.policy, "optimizer", None)
                        if hasattr(model, "policy")
                        else None
                    )
                    episode = getattr(model, "_episode_num", 0)
                    total_steps = getattr(model, "num_timesteps", 0)
                    checkpoint_manager.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        episode=episode,
                        total_steps=total_steps,
                        is_final=False,
                    )
                    logger.info("[TRAINING] Checkpoint saved on timeout.")
                except Exception as e:
                    logger.error(
                        "[TRAINING] Failed to save checkpoint on timeout: %s", e
                    )

            tm = TimeoutManager(
                timeout=float(timeout), cleanup_callback=_cleanup_on_timeout
            )

        try:
            # Entraînement avec gestion des interruptions
            try:
                # Calculer le nombre d'étapes restantes
                total_timesteps = config.get("training", {}).get(
                    "total_timesteps", 1000000
                )
                remaining_timesteps = total_timesteps - start_timesteps

                if remaining_timesteps <= 0:
                    logger.info(
                        "[TRAINING] L'entraînement est déjà terminé selon le nombre d'étapes total configuré."
                    )
                    return True

                logger.info(
                    "[TRAINING] Démarrage de l'entraînement pour %d étapes supplémentaires...",
                    remaining_timesteps,
                )

                # Démarrer l'entraînement avec les callbacks
                if tm:
                    with tm.limit():
                        model.learn(
                            total_timesteps=remaining_timesteps,
                            callback=callback,  # Utilisation du CallbackList
                            reset_num_timesteps=False,  # Ne pas réinitialiser le compteur d'étapes
                            progress_bar=False,  # Désactivé pour éviter les conflits avec notre callback personnalisé
                            tb_log_name="PPO",  # Nom pour les logs TensorBoard
                        )
                else:
                    model.learn(
                        total_timesteps=remaining_timesteps,
                        callback=callback,  # Utilisation du CallbackList
                        reset_num_timesteps=False,  # Ne pas réinitialiser le compteur d'étapes
                        progress_bar=progress_bar,  # Utiliser la barre de progression configurée
                        tb_log_name="PPO",  # Nom pour les logs TensorBoard
                    )
            except KeyboardInterrupt:
                logger.info(
                    "[TRAINING] \nInterruption de l'utilisateur détectée. "
                    "Sauvegarde du dernier état..."
                )
                # Sauvegarder un checkpoint final
                optimizer = None
                if hasattr(model.policy, "optimizer"):
                    optimizer = model.policy.optimizer

                episode = 0
                if hasattr(model, "_episode_num"):
                    episode = model._episode_num

                total_steps = 0
                if hasattr(model, "num_timesteps"):
                    total_steps = model.num_timesteps

                checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    episode=episode,
                    total_steps=total_steps,
                    is_final=True,
                )
                logger.info("Dernier état sauvegardé avec succès.")
                raise
            logger.info("Entraînement terminé avec succès!")

        except (TimeoutException, TMTimeoutException):
            logger.info("Temps d'entraînement écoulé. Arrêt de l'entraînement...")
        except Exception as e:
            logger.error("Erreur lors de l'entraînement: %s", str(e))
            raise

        finally:
            # Sauvegarder le modèle final
            final_metrics = {}
            if hasattr(model, "ep_info_buffer"):
                rewards = [ep_info["r"] for ep_info in model.ep_info_buffer]
                if rewards:  # Vérifier si la liste n'est pas vide
                    final_metrics["episode_reward"] = np.mean(rewards)

            optimizer = None
            if hasattr(model.policy, "optimizer"):
                optimizer = model.policy.optimizer

            episode = 0
            if hasattr(model, "_episode_num"):
                episode = model._episode_num

            total_steps = 0
            if hasattr(model, "num_timesteps"):
                total_steps = model.num_timesteps

            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                episode=episode,
                total_steps=total_steps,
                metrics=final_metrics,
                custom_data={
                    "config": config,
                    "policy_kwargs": policy_kwargs,
                    "training_complete": True,
                },
                is_final=True,
            )
            logger.info("Modèle final sauvegardé avec succès.")

            # Afficher un message de fin
            logger.info("Entraînement terminé. Nettoyage des ressources...")

        # Nettoyer
        env.close()
        return True

    except Exception as e:
        logger.error("Erreur lors de l'exécution de l'entraînement: %s", str(e))
        logger.error(traceback.format_exc())

        # Fermer les environnements en cas d'erreur
        if "env" in locals():
            env.close()

        return False


if __name__ == "__main__":
    class AttentionVisualizerCallback(BaseCallback):
        """
        Callback pour la visualisation des cartes d'attention.
        """
        def __init__(self, viz_cfg, verbose=0):
            super().__init__(verbose)
            self.viz_cfg = viz_cfg
            self.output_dir = viz_cfg.get('attention_map_dir', 'attention_maps')
            os.makedirs(self.output_dir, exist_ok=True)
            self.interval = viz_cfg.get('interval', 1000)

        def _on_step(self) -> bool:
            if self.num_timesteps % self.interval == 0:
                try:
                    # Récupérer l'environnement
                    env = self.training_env.envs[0] if hasattr(self.training_env, 'envs') else self.training_env

                    # Récupérer l'observation actuelle
                    obs = env.render(mode='rgb_array')

                    # Récupérer la carte d'attention du modèle
                    if hasattr(self.model.policy.features_extractor, 'get_attention_map'):
                        with torch.no_grad():
                            # Convertir l'observation en tenseur
                            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.model.device)
                            attention_map = self.model.policy.features_extractor.get_attention_map(obs_tensor)

                            # Sauvegarder la visualisation
                            output_path = os.path.join(
                                self.output_dir,
                                f'attention_step_{self.num_timesteps}.png'
                            )
                            self._save_attention_visualization(obs, attention_map, output_path)

                except Exception as e:
                    print(f"Erreur lors de la génération de la visualisation d'attention: {e}")

            return True

        def _save_attention_visualization(self, obs, attention_map, output_path):
            """
            Sauvegarde la visualisation de l'attention superposée à l'observation.
            """
            import matplotlib.pyplot as plt
            from matplotlib import cm
            import cv2

            # Créer une figure avec deux sous-graphiques
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Afficher l'observation originale
            ax1.imshow(obs)
            ax1.set_title('Observation originale')
            ax1.axis('off')

            # Afficher la carte d'attention
            if len(attention_map.shape) > 2:
                attention_map = attention_map.mean(dim=1).squeeze()

            # Redimensionner la carte d'attention pour correspondre à l'observation
            attention_map_np = attention_map.cpu().numpy()
            resized_attention = cv2.resize(
                attention_map_np,
                (obs.shape[1], obs.shape[0]),
                interpolation=cv2.INTER_CUBIC
            )

            # Afficher la carte d'attention
            im = ax2.imshow(resized_attention, cmap='viridis')
            ax2.set_title('Carte d\'attention')
            ax2.axis('off')
            plt.colorbar(im, ax=ax2)

            # Sauvegarder la figure
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()

            if self.verbose > 0:
                print(f"Carte d'attention sauvegardée: {output_path}")

    parser = argparse.ArgumentParser(
        description=(
            "Entraîne un bot de trading ADAN avec support du timeout "
            "et des points de contrôle"
        )
    )
    parser.add_argument(
        "-c",
        "--config-path",
        type=str,
        default="bot/config/config.yaml",
        help="Chemin vers le fichier de configuration YAML",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Répertoire pour enregistrer les points de contrôle",
    )
    parser.add_argument(
        "--shared-model",
        type=str,
        default=None,
        help=("Chemin vers un modèle partagé pour l'entraînement distribué"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reprend l'entraînement à partir du dernier point de contrôle",
    )

    # Ajout des arguments supplémentaires mentionnés dans le patch
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (optional)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Appel robuste de la fonction main()
    try:
        # 1) Appel avec les arguments nommés
        success = main(
            config_path=args.config_path,
            timeout=args.timeout,
            checkpoint_dir=args.checkpoint_dir,
            shared_model_path=args.shared_model,
            resume=args.resume,
            num_envs=(args.workers if args.workers is not None else 4),
            use_subproc=True,
        )
    except Exception as e:
        print(f"Erreur lors de l'exécution: {e}", file=sys.stderr)
        raise

    sys.exit(0 if success else 1)
