#!/usr/bin/env python3
"""
Optimisation Optuna améliorée avec progression par paliers et analyse comportementale.
Version avancée avec évaluation multi-palier et détection de comportements.
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import optuna
import copy
import gc
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import necessary components from the bot
from src.adan_trading_bot.environment.multi_asset_chunked_env import (
    MultiAssetChunkedEnv,
)
from src.adan_trading_bot.common.config_loader import ConfigLoader
from src.adan_trading_bot.common.custom_logger import setup_logging
from src.adan_trading_bot.data_processing.data_loader import ChunkedDataLoader

# Setup advanced logging
setup_logging()
logger = logging.getLogger(__name__)


class CapitalTier(Enum):
    """Énumération des paliers de capital."""

    MICRO = "Micro Capital"
    SMALL = "Small Capital"
    MEDIUM = "Medium Capital"
    HIGH = "High Capital"
    ENTERPRISE = "Enterprise"


@dataclass
class TradingBehavior:
    """Structure pour analyser le comportement de trading."""

    tier: CapitalTier
    avg_trade_duration: float
    win_rate: float
    avg_win_size: float
    avg_loss_size: float
    risk_reward_ratio: float
    max_consecutive_losses: int
    trading_frequency: float
    volatility_adaptation: float
    profit_consistency: float
    drawdown_recovery: float

    def get_behavior_score(self) -> float:
        """Calcule un score comportemental global."""
        # Poids pour chaque critère comportemental
        weights = {
            "win_rate": 0.25,  # Très important
            "risk_reward": 0.20,  # Très important
            "profit_consistency": 0.15,  # Important
            "drawdown_recovery": 0.15,  # Important
            "volatility_adaptation": 0.10,  # Modéré
            "frequency_balance": 0.10,  # Modéré
            "consecutive_losses": 0.05,  # Modéré
        }

        # Normalisation des métriques (0-1)
        win_rate_norm = min(self.win_rate / 0.7, 1.0)  # Target: 70%
        risk_reward_norm = min(self.risk_reward_ratio / 2.0, 1.0)  # Target: 2.0
        consistency_norm = self.profit_consistency
        recovery_norm = self.drawdown_recovery
        adaptation_norm = self.volatility_adaptation

        # Fréquence optimale (ni trop ni trop peu)
        optimal_freq = 0.3  # 30% des steps
        freq_penalty = abs(self.trading_frequency - optimal_freq) / optimal_freq
        frequency_norm = max(0, 1 - freq_penalty)

        # Pénalité pour pertes consécutives
        max_acceptable_losses = 5
        losses_penalty = min(self.max_consecutive_losses / max_acceptable_losses, 1.0)
        consecutive_norm = 1 - losses_penalty

        # Score final pondéré
        behavior_score = (
            weights["win_rate"] * win_rate_norm
            + weights["risk_reward"] * risk_reward_norm
            + weights["profit_consistency"] * consistency_norm
            + weights["drawdown_recovery"] * recovery_norm
            + weights["volatility_adaptation"] * adaptation_norm
            + weights["frequency_balance"] * frequency_norm
            + weights["consecutive_losses"] * consecutive_norm
        )

        return behavior_score

    def get_behavior_description(self) -> str:
        """Génère une description textuelle du comportement."""
        behavior_type = "UNKNOWN"

        if self.win_rate >= 0.6 and self.risk_reward_ratio >= 1.5:
            behavior_type = "EXCELLENCE - Trader optimal"
        elif self.win_rate >= 0.5 and self.risk_reward_ratio >= 1.2:
            behavior_type = "QUALITY - Trader solide"
        elif self.win_rate >= 0.4 and self.avg_loss_size < 0.08:  # Pertes <8%
            behavior_type = "DEFENSIVE - Trader conservateur"
        elif self.trading_frequency > 0.5:
            behavior_type = "OVERACTIVE - Sur-trading"
        elif self.max_consecutive_losses > 7:
            behavior_type = "RISKY - Gestion risque insuffisante"
        else:
            behavior_type = "DEVELOPING - En apprentissage"

        return (
            f"{behavior_type} | WR:{self.win_rate:.1%} RR:{self.risk_reward_ratio:.2f}"
        )


class TierProgressionManager:
    """Gestionnaire de progression par paliers de capital."""

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.tier_progression = self._calculate_tier_progression()

    def _calculate_tier_progression(self) -> List[CapitalTier]:
        """Calcule la progression par palier sur les époques."""
        tiers = list(CapitalTier)
        epochs_per_tier = max(1, self.total_epochs // len(tiers))

        progression = []
        for i, tier in enumerate(tiers):
            start_epoch = i * epochs_per_tier
            end_epoch = min((i + 1) * epochs_per_tier, self.total_epochs)

            # Ajouter le tier pour chaque époque dans sa plage
            for epoch in range(start_epoch, end_epoch):
                if epoch < self.total_epochs:
                    progression.append(tier)

        # Compléter si nécessaire
        while len(progression) < self.total_epochs:
            progression.append(CapitalTier.ENTERPRISE)

        return progression[: self.total_epochs]

    def get_tier_for_epoch(self, epoch: int) -> CapitalTier:
        """Retourne le palier pour une époque donnée."""
        if epoch < len(self.tier_progression):
            return self.tier_progression[epoch]
        return CapitalTier.ENTERPRISE

    def get_tier_config(self, tier: CapitalTier) -> Dict[str, Any]:
        """Retourne la configuration spécifique au palier."""
        tier_configs = {
            CapitalTier.MICRO: {
                "initial_balance": 20.5,
                "max_position_size_pct": 0.90,
                "risk_per_trade_pct": 0.05,
                "max_concurrent_positions": 1,
                "behavior_focus": "growth",
            },
            CapitalTier.SMALL: {
                "initial_balance": 65.0,
                "max_position_size_pct": 0.65,
                "risk_per_trade_pct": 0.02,
                "max_concurrent_positions": 2,
                "behavior_focus": "stability",
            },
            CapitalTier.MEDIUM: {
                "initial_balance": 200.0,
                "max_position_size_pct": 0.60,
                "risk_per_trade_pct": 0.02,
                "max_concurrent_positions": 3,
                "behavior_focus": "diversification",
            },
            CapitalTier.HIGH: {
                "initial_balance": 650.0,
                "max_position_size_pct": 0.35,
                "risk_per_trade_pct": 0.025,
                "max_concurrent_positions": 4,
                "behavior_focus": "optimization",
            },
            CapitalTier.ENTERPRISE: {
                "initial_balance": 2000.0,
                "max_position_size_pct": 0.20,
                "risk_per_trade_pct": 0.03,
                "max_concurrent_positions": 5,
                "behavior_focus": "institutional",
            },
        }
        return tier_configs.get(tier, tier_configs[CapitalTier.MICRO])


class BehaviorAnalyzer:
    """Analyseur de comportement de trading."""

    def __init__(self):
        self.trade_history = []
        self.performance_history = []

    def analyze_trial_behavior(
        self, trial_results: Dict, tier: CapitalTier
    ) -> TradingBehavior:
        """Analyse le comportement d'un trial."""

        # Extraction des métriques de base
        trades = trial_results.get("trades", [])
        returns = trial_results.get("returns", [])
        portfolio_values = trial_results.get("portfolio_values", [])

        if not trades or not returns:
            # Comportement par défaut si pas de données
            return TradingBehavior(
                tier=tier,
                avg_trade_duration=0,
                win_rate=0,
                avg_win_size=0,
                avg_loss_size=0,
                risk_reward_ratio=0,
                max_consecutive_losses=10,
                trading_frequency=0,
                volatility_adaptation=0,
                profit_consistency=0,
                drawdown_recovery=0,
            )

        # Calcul des métriques comportementales
        winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in trades if t.get("pnl", 0) <= 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0

        avg_win_size = (
            np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
        )
        avg_loss_size = (
            abs(np.mean([t["pnl"] for t in losing_trades])) if losing_trades else 0
        )

        risk_reward_ratio = avg_win_size / avg_loss_size if avg_loss_size > 0 else 0

        # Durée moyenne des trades
        durations = [t.get("duration", 0) for t in trades]
        avg_trade_duration = np.mean(durations) if durations else 0

        # Pertes consécutives maximum
        consecutive_losses = 0
        max_consecutive_losses = 0
        for trade in trades:
            if trade.get("pnl", 0) <= 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0

        # Fréquence de trading
        total_steps = trial_results.get("total_steps", 1)
        trading_frequency = len(trades) / total_steps if total_steps > 0 else 0

        # Adaptation à la volatilité
        volatility_adaptation = self._calculate_volatility_adaptation(trades, returns)

        # Consistance des profits
        profit_consistency = self._calculate_profit_consistency(returns)

        # Récupération après drawdown
        drawdown_recovery = self._calculate_drawdown_recovery(portfolio_values)

        return TradingBehavior(
            tier=tier,
            avg_trade_duration=avg_trade_duration,
            win_rate=win_rate,
            avg_win_size=avg_win_size,
            avg_loss_size=avg_loss_size,
            risk_reward_ratio=risk_reward_ratio,
            max_consecutive_losses=max_consecutive_losses,
            trading_frequency=trading_frequency,
            volatility_adaptation=volatility_adaptation,
            profit_consistency=profit_consistency,
            drawdown_recovery=drawdown_recovery,
        )

    def _calculate_volatility_adaptation(self, trades: List, returns: List) -> float:
        """Calcule la capacité d'adaptation à la volatilité."""
        if not returns or len(returns) < 10:
            return 0.0

        # Calculer la volatilité roulante
        returns_array = np.array(returns)
        volatility = np.std(returns_array)

        # Mesurer si les trades s'adaptent à la volatilité
        # Plus la volatilité est haute, plus les trades devraient être prudents
        high_vol_periods = returns_array[np.abs(returns_array) > volatility]
        adaptation_score = 1.0 - (len(high_vol_periods) / len(returns_array))

        return max(0.0, min(1.0, adaptation_score))

    def _calculate_profit_consistency(self, returns: List) -> float:
        """Calcule la consistance des profits."""
        if not returns or len(returns) < 5:
            return 0.0

        returns_array = np.array(returns)
        positive_returns = returns_array[returns_array > 0]

        if len(positive_returns) < 2:
            return 0.0

        # Coefficient de variation inverse (plus bas = plus consistent)
        mean_positive = np.mean(positive_returns)
        std_positive = np.std(positive_returns)

        if mean_positive <= 0:
            return 0.0

        cv = std_positive / mean_positive
        consistency = 1.0 / (1.0 + cv)  # Normalise entre 0 et 1

        return min(1.0, consistency)

    def _calculate_drawdown_recovery(self, portfolio_values: List) -> float:
        """Calcule la capacité de récupération après drawdown."""
        if not portfolio_values or len(portfolio_values) < 10:
            return 0.0

        values = np.array(portfolio_values)

        # Calculer les drawdowns
        cummax = np.maximum.accumulate(values)
        drawdowns = (values - cummax) / cummax

        # Trouver les périodes de récupération
        recovery_times = []
        in_drawdown = False
        drawdown_start = 0

        for i, dd in enumerate(drawdowns):
            if dd < -0.02 and not in_drawdown:  # Drawdown >2%
                in_drawdown = True
                drawdown_start = i
            elif dd >= -0.01 and in_drawdown:  # Récupération
                recovery_time = i - drawdown_start
                recovery_times.append(recovery_time)
                in_drawdown = False

        if not recovery_times:
            return 1.0  # Pas de drawdown = parfait

        # Score basé sur la vitesse de récupération moyenne
        avg_recovery = np.mean(recovery_times)
        max_acceptable_recovery = len(portfolio_values) * 0.1  # 10% du temps total

        recovery_score = max(0.0, 1.0 - (avg_recovery / max_acceptable_recovery))
        return min(1.0, recovery_score)


class OptunaPruningCallback(BaseCallback):
    """Callback avec progression par paliers et analyse comportementale."""

    def __init__(
        self,
        trial: optuna.Trial,
        eval_env: VecEnv,
        tier_manager: TierProgressionManager,
        eval_freq: int = 5000,
        total_timesteps: int = 25000,
    ):
        super().__init__(verbose=0)
        self.trial = trial
        self.eval_env = eval_env
        self.tier_manager = tier_manager
        self.eval_freq = eval_freq
        self.total_timesteps = total_timesteps
        self.start_time = time.time()
        self.progress_bar = None
        self.last_update_time = time.time()
        self.current_epoch = 0
        self.tier_performances = {}

    def _on_training_start(self) -> None:
        """Initialize progress bar with tier information."""
        self.progress_bar = tqdm(
            total=self.total_timesteps,
            desc=f"Trial {self.trial.number} [Tier: {self.tier_manager.get_tier_for_epoch(0).value}]",
            unit="steps",
            leave=False,
            position=1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc} Sharpe: {postfix}",
        )

    def _on_step(self) -> bool:
        current_time = time.time()

        # Gestion de la progression par paliers
        expected_epoch = int(
            (self.num_timesteps / self.total_timesteps) * self.tier_manager.total_epochs
        )
        if expected_epoch != self.current_epoch:
            self.current_epoch = expected_epoch
            current_tier = self.tier_manager.get_tier_for_epoch(self.current_epoch)

            # Mettre à jour la description avec le nouveau tier
            if self.progress_bar:
                new_desc = f"Trial {self.trial.number} [Tier: {current_tier.value}]"
                self.progress_bar.set_description(new_desc)

        # Update progress bar
        if current_time - self.last_update_time >= 2.0:  # Update every 2 seconds
            if self.progress_bar:
                # Get current performance
                try:
                    sharpe = self._evaluate_sharpe()
                    current_tier = self.tier_manager.get_tier_for_epoch(
                        self.current_epoch
                    )
                    sharpe_str = f"{sharpe:.3f} ({current_tier.name})"
                except:
                    sharpe_str = "calculating..."

                self.progress_bar.set_postfix_str(sharpe_str)
                self.progress_bar.update(self.num_timesteps - self.progress_bar.n)

            self.last_update_time = current_time

        # Evaluate and report for pruning
        if (self.num_timesteps - getattr(self, "last_eval_step", 0)) >= self.eval_freq:
            self.last_eval_step = self.num_timesteps

            try:
                # Evaluer avec le tier actuel
                current_tier = self.tier_manager.get_tier_for_epoch(self.current_epoch)
                performance = self._evaluate_tier_performance(current_tier)

                # Stocker la performance du tier
                tier_key = current_tier.name
                if tier_key not in self.tier_performances:
                    self.tier_performances[tier_key] = []
                self.tier_performances[tier_key].append(performance)

                # Report pour pruning
                step_value = self.num_timesteps / self.total_timesteps
                self.trial.report(performance["composite_score"], step_value)

                # Vérifier si le trial doit être élagué
                if self.trial.should_prune():
                    logger.info(
                        f"Trial {self.trial.number} pruned at step {self.num_timesteps}"
                    )
                    return False

            except Exception as e:
                logger.warning(f"Evaluation failed at step {self.num_timesteps}: {e}")

        return True

    def _on_training_end(self) -> None:
        """Complete tier analysis at training end."""
        if self.progress_bar:
            self.progress_bar.close()

        # Store final tier performances in trial attributes
        self.trial.set_user_attr("tier_performances", self.tier_performances)

        # Calculate overall multi-tier performance
        overall_score = self._calculate_multi_tier_score()
        self.trial.set_user_attr("multi_tier_score", overall_score)

    def _evaluate_sharpe(self) -> float:
        """Quick Sharpe ratio evaluation."""
        try:
            obs = self.eval_env.reset()
            returns = []

            for _ in range(min(500, self.eval_freq // 5)):  # Quick evaluation
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)

                if hasattr(info, "__iter__") and len(info) > 0:
                    env_info = info[0] if isinstance(info[0], dict) else {}
                    if "return" in env_info:
                        returns.append(env_info["return"])

                if done.any():
                    obs = self.eval_env.reset()

            if len(returns) > 10:
                returns_array = np.array(returns)
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                sharpe = (mean_return / std_return) if std_return > 0 else 0.0
                return sharpe

        except Exception as e:
            logger.debug(f"Sharpe evaluation error: {e}")

        return 0.0

    def _evaluate_tier_performance(self, tier: CapitalTier) -> Dict[str, float]:
        """Évalue la performance spécifique à un palier."""
        try:
            # Configurer l'environnement pour le tier
            tier_config = self.tier_manager.get_tier_config(tier)

            obs = self.eval_env.reset()
            episode_returns = []
            trade_history = []
            portfolio_values = []
            steps = 0
            max_eval_steps = 1000  # Evaluation rapide mais suffisante

            for step in range(max_eval_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                steps += 1

                if hasattr(info, "__iter__") and len(info) > 0:
                    env_info = info[0] if isinstance(info[0], dict) else {}

                    if "return" in env_info:
                        episode_returns.append(env_info["return"])

                    if "portfolio_value" in env_info:
                        portfolio_values.append(env_info["portfolio_value"])

                    if "trade_completed" in env_info and env_info["trade_completed"]:
                        trade_history.append(
                            {
                                "pnl": env_info.get("trade_pnl", 0),
                                "duration": env_info.get("trade_duration", 0),
                            }
                        )

                if done.any():
                    obs = self.eval_env.reset()

            # Calcul des métriques
            if episode_returns:
                returns_array = np.array(episode_returns)
                sharpe_ratio = self._calculate_sharpe(returns_array)
                max_drawdown = self._calculate_max_drawdown(portfolio_values)
                win_rate = self._calculate_win_rate(trade_history)
                profit_factor = self._calculate_profit_factor(trade_history)

                # Score composite adapté au palier
                behavior_weight = self._get_tier_behavior_weight(tier)
                composite_score = (
                    sharpe_ratio * behavior_weight["sharpe"]
                    + (1 - max_drawdown) * behavior_weight["drawdown"]
                    + win_rate * behavior_weight["win_rate"]
                    + min(profit_factor / 2.0, 1.0) * behavior_weight["profit_factor"]
                )

                return {
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "win_rate": win_rate,
                    "profit_factor": profit_factor,
                    "composite_score": composite_score,
                    "trade_count": len(trade_history),
                    "total_steps": steps,
                }

        except Exception as e:
            logger.warning(f"Tier {tier.name} evaluation failed: {e}")

        return {
            "sharpe_ratio": 0.0,
            "max_drawdown": 1.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "composite_score": 0.0,
            "trade_count": 0,
            "total_steps": 0,
        }

    def _get_tier_behavior_weight(self, tier: CapitalTier) -> Dict[str, float]:
        """Retourne les poids d'évaluation selon le palier."""
        tier_weights = {
            CapitalTier.MICRO: {
                "sharpe": 0.25,
                "drawdown": 0.35,
                "win_rate": 0.3,
                "profit_factor": 0.1,
            },
            CapitalTier.SMALL: {
                "sharpe": 0.3,
                "drawdown": 0.25,
                "win_rate": 0.25,
                "profit_factor": 0.2,
            },
            CapitalTier.MEDIUM: {
                "sharpe": 0.35,
                "drawdown": 0.2,
                "win_rate": 0.25,
                "profit_factor": 0.2,
            },
            CapitalTier.HIGH: {
                "sharpe": 0.4,
                "drawdown": 0.15,
                "win_rate": 0.2,
                "profit_factor": 0.25,
            },
            CapitalTier.ENTERPRISE: {
                "sharpe": 0.45,
                "drawdown": 0.1,
                "win_rate": 0.15,
                "profit_factor": 0.3,
            },
        }
        return tier_weights.get(tier, tier_weights[CapitalTier.MICRO])

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calcule le ratio de Sharpe."""
        if len(returns) < 2:
            return 0.0
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        return (mean_return / std_return) if std_return > 0 else 0.0

    def _calculate_max_drawdown(self, portfolio_values: List) -> float:
        """Calcule le drawdown maximum."""
        if not portfolio_values:
            return 1.0
        values = np.array(portfolio_values)
        cummax = np.maximum.accumulate(values)
        drawdowns = (values - cummax) / cummax
        return abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    def _calculate_win_rate(self, trades: List) -> float:
        """Calcule le win rate."""
        if not trades:
            return 0.0
        winning_trades = sum(1 for trade in trades if trade.get("pnl", 0) > 0)
        return winning_trades / len(trades)

    def _calculate_profit_factor(self, trades: List) -> float:
        """Calcule le profit factor."""
        if not trades:
            return 0.0
        gross_profit = sum(trade["pnl"] for trade in trades if trade.get("pnl", 0) > 0)
        gross_loss = abs(
            sum(trade["pnl"] for trade in trades if trade.get("pnl", 0) < 0)
        )
        return gross_profit / gross_loss if gross_loss > 0 else 0.0

    def _calculate_multi_tier_score(self) -> float:
        """Calcule un score global multi-paliers."""
        if not self.tier_performances:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        # Pondération progressive : plus le palier est élevé, plus il compte
        tier_weights = {
            "MICRO": 0.15,
            "SMALL": 0.2,
            "MEDIUM": 0.25,
            "HIGH": 0.3,
            "ENTERPRISE": 0.35,
        }

        for tier_name, performances in self.tier_performances.items():
            if performances:
                avg_performance = np.mean(
                    [p.get("composite_score", 0) for p in performances]
                )
                weight = tier_weights.get(tier_name, 0.1)
                total_score += avg_performance * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0


def setup_database(study_name: str = "adan_progressive_hyperopt") -> optuna.Study:
    """Setup Optuna database with enhanced configuration."""
    db_path = f"{study_name}.db"
    storage = f"sqlite:///{db_path}"

    # Pruner plus sophistiqué
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=5)

    # Sampler avec meilleurs paramètres
    sampler = TPESampler(
        n_startup_trials=5, n_ei_candidates=24, multivariate=True, constant_liar=True
    )

    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    return study


# Configuration globale
CONFIG_PATH = os.path.join(project_root, "config", "config.yaml")
GLOBAL_CONFIG = ConfigLoader.load_config(CONFIG_PATH)


def objective(trial: optuna.Trial) -> float:
    """
    Fonction objective améliorée avec progression par paliers et analyse comportementale.
    """
    env = None
    model = None

    try:
        start_time = datetime.now()
        trial.set_user_attr("start_time", start_time.isoformat())
        logger.info(f"=== TRIAL {trial.number} - PROGRESSIVE TIER TRAINING ===")

        # Calculer le nombre d'époques pour la progression
        n_epochs = trial.suggest_categorical("n_epochs", [3, 4, 5, 10, 20])

        # Initialiser le gestionnaire de progression par paliers
        tier_manager = TierProgressionManager(total_epochs=n_epochs)

        # Nouveaux intervalles SL/TP selon vos spécifications
        stop_loss_pct = trial.suggest_float("stop_loss_pct", 0.05, 0.13)  # 5-13%
        take_profit_pct = trial.suggest_float("take_profit_pct", 0.05, 0.15)  # 5-15%

        # Hyperparamètres PPO
        ppo_params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
            "ent_coef": trial.suggest_float("ent_coef", 0.001, 0.1),
            "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "n_epochs": n_epochs,
        }

        # Poids de récompenses optimisés pour trading de qualité
        reward_params = {
            # Très important : Favoriser win rate et qualité
            "win_rate_bonus": trial.suggest_float("win_rate_bonus", 0.5, 2.0),
            "pnl_weight": trial.suggest_float("pnl_weight", 50.0, 100.0),
            # Pénalités pour mauvais comportements
            "stop_loss_penalty": trial.suggest_float("stop_loss_penalty", -2.0, -0.1),
            "consecutive_loss_penalty": trial.suggest_float(
                "consecutive_loss_penalty", -5.0, -1.0
            ),
            "overtrading_penalty": trial.suggest_float(
                "overtrading_penalty", -3.0, -0.5
            ),
            # Bonus pour bon comportement
            "take_profit_bonus": trial.suggest_float("take_profit_bonus", 1.0, 5.0),
            "consistency_bonus": trial.suggest_float("consistency_bonus", 0.5, 3.0),
            "patience_bonus": trial.suggest_float("patience_bonus", 0.2, 1.5),
        }

        # Paramètres de trading focalisés sur la durabilité
        trading_params = {
            "max_consecutive_losses": trial.suggest_int("max_consecutive_losses", 3, 7),
            "min_trade_quality_score": trial.suggest_float(
                "min_trade_quality_score", 0.6, 0.9
            ),
            "position_hold_min": trial.suggest_int("position_hold_min", 5, 30),
            "position_hold_max": trial.suggest_int("position_hold_max", 50, 500),
        }

        # Configuration temporaire avec les nouveaux paramètres
        temp_config = copy.deepcopy(GLOBAL_CONFIG)

        # Appliquer les nouveaux paramètres SL/TP à tous les workers
        for worker_key in ["w1", "w2", "w3", "w4"]:
            if worker_key in temp_config["workers"]:
                # Mettre à jour SL/TP pour tous les tiers
                for tier in ["Micro", "Small", "Medium", "High", "Enterprise"]:
                    temp_config["workers"][worker_key]["stop_loss_pct_by_tier"][
                        tier
                    ] = stop_loss_pct
                    temp_config["workers"][worker_key]["take_profit_pct_by_tier"][
                        tier
                    ] = take_profit_pct

                # Appliquer les poids de récompenses
                temp_config["workers"][worker_key]["agent_config"]["pnl_weight"] = (
                    reward_params["pnl_weight"]
                )
                temp_config["workers"][worker_key]["reward_config"][
                    "win_rate_bonus"
                ] = reward_params["win_rate_bonus"]

        # Mettre à jour les paramètres globaux
        temp_config["risk_parameters"]["base_sl_pct"] = stop_loss_pct
        temp_config["risk_parameters"]["base_tp_pct"] = take_profit_pct
        temp_config["environment"]["risk_management"]["position_sizing"][
            "initial_sl_pct"
        ] = stop_loss_pct
        temp_config["environment"]["risk_management"]["position_sizing"][
            "initial_tp_pct"
        ] = take_profit_pct

        # Appliquer les paramètres PPO
        for param, value in ppo_params.items():
            temp_config["agent"][param] = value

        # Configuration pour entraînement progressif par paliers
        temp_config["environment"]["max_steps"] = 30000  # Plus long pour progression
        temp_config["environment"]["max_chunks_per_episode"] = 3
        temp_config["progressive_training"] = {
            "enabled": True,
            "tier_progression": True,
            "epochs_per_tier": max(1, n_epochs // len(CapitalTier)),
        }

        # Créer l'environnement avec le premier worker
        worker_config = temp_config["workers"]["w1"]
        data_loader = ChunkedDataLoader(
            config=temp_config, worker_config=worker_config, worker_id=0
        )
        data = data_loader.load_chunk(0)

        if data is None or not data:
            logger.error(f"Trial {trial.number}: No data available")
            return -np.inf

        # Check if any timeframe has non-empty data
        has_valid_data = any(
            not df.empty
            for asset_data in data.values()
            if isinstance(asset_data, dict)
            for df in asset_data.values()
            if isinstance(df, pd.DataFrame)
        )

        if not has_valid_data:
            logger.error(f"Trial {trial.number}: No valid data available")
            return -np.inf

        # Configuration de l'environnement
        env_kwargs = {
            "data": data,
            "timeframes": temp_config["data"]["timeframes"],
            "window_size": temp_config["environment"]["window_size"],
            "features_config": temp_config["environment"]["features_config"],
            "max_steps": temp_config["environment"]["max_steps"],
            "initial_balance": temp_config["environment"]["initial_balance"],
            "commission": temp_config["environment"]["commission"],
            "reward_scaling": temp_config["environment"]["reward_scaling"],
            "worker_config": worker_config,
            "config": temp_config,
        }

        env = MultiAssetChunkedEnv(**env_kwargs)
        env = DummyVecEnv([lambda: env])

        # Callback avec progression par paliers
        callback = OptunaPruningCallback(
            trial=trial,
            eval_env=env,
            tier_manager=tier_manager,
            eval_freq=3000,
            total_timesteps=temp_config["environment"]["max_steps"],
        )

        # Créer le modèle PPO
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=ppo_params["learning_rate"],
            n_steps=ppo_params["n_steps"],
            batch_size=ppo_params["batch_size"],
            n_epochs=ppo_params["n_epochs"],
            gamma=ppo_params["gamma"],
            clip_range=ppo_params["clip_range"],
            ent_coef=ppo_params["ent_coef"],
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            seed=42,
        )

        # Entraînement avec progression par paliers
        model.learn(
            total_timesteps=temp_config["environment"]["max_steps"],
            callback=callback,
            progress_bar=False,
        )

        # Analyse comportementale finale
        analyzer = BehaviorAnalyzer()

        # Évaluer sur chaque palier
        final_behaviors = {}
        tier_scores = []

        for tier in CapitalTier:
            # Simuler quelques épisodes pour ce palier
            tier_config = tier_manager.get_tier_config(tier)

            # Évaluation rapide
            obs = env.reset()
            episode_data = {
                "trades": [],
                "returns": [],
                "portfolio_values": [tier_config["initial_balance"]],
                "total_steps": 0,
            }

            for step in range(1000):  # Évaluation rapide
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_data["total_steps"] += 1

                if hasattr(info, "__iter__") and len(info) > 0:
                    env_info = info[0] if isinstance(info[0], dict) else {}

                    if "return" in env_info:
                        episode_data["returns"].append(env_info["return"])

                    if "portfolio_value" in env_info:
                        episode_data["portfolio_values"].append(
                            env_info["portfolio_value"]
                        )

                    if "trade_completed" in env_info:
                        episode_data["trades"].append(
                            {
                                "pnl": env_info.get("trade_pnl", 0),
                                "duration": env_info.get("trade_duration", 0),
                            }
                        )

                if done.any():
                    obs = env.reset()

            # Analyser le comportement pour ce palier
            behavior = analyzer.analyze_trial_behavior(episode_data, tier)
            final_behaviors[tier.name] = behavior

            # Score comportemental
            behavior_score = behavior.get_behavior_score()
            tier_scores.append(behavior_score)

            logger.info(
                f"Tier {tier.name}: {behavior.get_behavior_description()}, Score: {behavior_score:.3f}"
            )

        # Score final multi-palier avec emphase sur qualité
        if tier_scores:
            # Favoriser les paliers supérieurs mais exiger qualité sur tous
            weights = [0.15, 0.2, 0.25, 0.3, 0.35]  # Progressive (5 tiers)

            # Bonus si tous les paliers sont acceptables (>0.4)
            all_acceptable = all(score > 0.4 for score in tier_scores)
            quality_bonus = 0.2 if all_acceptable else 0.0

            # Score pondéré
            weighted_score = sum(
                score * weight
                for score, weight in zip(tier_scores, weights[: len(tier_scores)])
            )

            # Pénalité si trop de pertes consécutives sur n'importe quel palier
            max_consecutive_penalty = 0
            for behavior in final_behaviors.values():
                if (
                    behavior.max_consecutive_losses
                    > trading_params["max_consecutive_losses"]
                ):
                    max_consecutive_penalty += 0.1

            final_score = weighted_score + quality_bonus - max_consecutive_penalty
        else:
            final_score = -1.0

        # Sauvegarder les résultats comportementaux
        behavior_summary = {}
        for tier_name, behavior in final_behaviors.items():
            behavior_summary[tier_name] = {
                "win_rate": behavior.win_rate,
                "risk_reward_ratio": behavior.risk_reward_ratio,
                "trading_frequency": behavior.trading_frequency,
                "behavior_description": behavior.get_behavior_description(),
                "behavior_score": behavior.get_behavior_score(),
            }

        trial.set_user_attr("behavior_analysis", behavior_summary)
        trial.set_user_attr("final_behavior_score", final_score)
        trial.set_user_attr("tier_progression", [tier.name for tier in CapitalTier])

        # Paramètres importants pour l'analyse
        trial.set_user_attr("stop_loss_pct", stop_loss_pct)
        trial.set_user_attr("take_profit_pct", take_profit_pct)
        trial.set_user_attr("win_rate_bonus", reward_params["win_rate_bonus"])
        trial.set_user_attr("pnl_weight", reward_params["pnl_weight"])

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60.0
        trial.set_user_attr("duration_minutes", duration)

        logger.info(
            f"Trial {trial.number} completed - Score: {final_score:.4f} - Duration: {duration:.1f}min"
        )

        # Log du meilleur comportement
        best_tier = max(
            final_behaviors.keys(),
            key=lambda k: final_behaviors[k].get_behavior_score(),
        )
        logger.info(
            f"Best behavior: {final_behaviors[best_tier].get_behavior_description()}"
        )

        return final_score

    except optuna.exceptions.TrialPruned:
        logger.info(f"Trial {trial.number} was pruned")
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
        return -np.inf
    finally:
        if env is not None:
            try:
                env.close()
            except:
                pass
        if model is not None:
            del model
        gc.collect()


def print_comprehensive_results(study: optuna.Study) -> None:
    """Affichage des résultats avec analyse comportementale."""
    if not study.best_trial:
        logger.warning("No successful trials completed!")
        return

    best_trial = study.best_trial
    logger.info("=" * 80)
    logger.info("🏆 MEILLEURS RÉSULTATS - ANALYSE COMPORTEMENTALE")
    logger.info("=" * 80)

    # Informations générales
    logger.info(f"Score final: {best_trial.value:.4f}")
    logger.info(
        f"Durée: {best_trial.user_attrs.get('duration_minutes', 0):.1f} minutes"
    )

    # Paramètres clés
    logger.info("\n📊 PARAMÈTRES OPTIMAUX:")
    logger.info(
        f"  Stop Loss: {best_trial.user_attrs.get('stop_loss_pct', 0) * 100:.1f}%"
    )
    logger.info(
        f"  Take Profit: {best_trial.user_attrs.get('take_profit_pct', 0) * 100:.1f}%"
    )
    logger.info(
        f"  Win Rate Bonus: {best_trial.user_attrs.get('win_rate_bonus', 0):.2f}"
    )
    logger.info(f"  PnL Weight: {best_trial.user_attrs.get('pnl_weight', 0):.1f}")

    # PPO Params
    logger.info(f"  Learning Rate: {best_trial.params.get('learning_rate', 0):.2e}")
    logger.info(f"  N Steps: {best_trial.params.get('n_steps', 0)}")
    logger.info(f"  Batch Size: {best_trial.params.get('batch_size', 0)}")
    logger.info(f"  Epochs: {best_trial.params.get('n_epochs', 0)}")

    # Analyse comportementale par palier
    behavior_analysis = best_trial.user_attrs.get("behavior_analysis", {})

    if behavior_analysis:
        logger.info("\n🎯 ANALYSE COMPORTEMENTALE PAR PALIER:")
        for tier_name, behavior in behavior_analysis.items():
            logger.info(f"\n  {tier_name}:")
            logger.info(f"    Win Rate: {behavior['win_rate']:.1%}")
            logger.info(f"    Risk/Reward: {behavior['risk_reward_ratio']:.2f}")
            logger.info(f"    Fréquence: {behavior['trading_frequency']:.1%}")
            logger.info(f"    Comportement: {behavior['behavior_description']}")
            logger.info(f"    Score: {behavior['behavior_score']:.3f}")

    # Recommandations
    logger.info("\n💡 RECOMMANDATIONS POUR DÉPLOIEMENT:")

    best_behavior = (
        max(behavior_analysis.values(), key=lambda x: x["behavior_score"])
        if behavior_analysis
        else None
    )

    if best_behavior:
        if best_behavior["win_rate"] >= 0.6:
            logger.info("  ✅ Excellent win rate - Prêt pour déploiement")
        elif best_behavior["win_rate"] >= 0.4:
            logger.info("  ⚠️ Win rate acceptable - Surveiller attentivement")
        else:
            logger.info("  ❌ Win rate insuffisant - Plus d'optimisation requise")

        if best_behavior["risk_reward_ratio"] >= 1.5:
            logger.info("  ✅ Excellent ratio risque/récompense")
        else:
            logger.info("  ⚠️ Ratio risque/récompense à améliorer")

    logger.info("\n🚀 PROCHAINES ÉTAPES:")
    logger.info("  1. Valider sur données de test")
    logger.info("  2. Paper trading 48h")
    logger.info("  3. Déploiement progressif par palier")


def main():
    """Fonction principale d'optimisation."""
    import argparse

    parser = argparse.ArgumentParser(description="Optimisation Optuna avancée")
    parser.add_argument("--n-trials", type=int, default=100, help="Nombre de trials")
    parser.add_argument("--timeout", type=int, default=7200, help="Timeout en secondes")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Chemin vers config.yaml")

    args = parser.parse_args()

    # Mise à jour de la config globale
    global GLOBAL_CONFIG, CONFIG_PATH
    CONFIG_PATH = args.config
    GLOBAL_CONFIG = ConfigLoader.load_config(CONFIG_PATH)

    logger.info("🚀 DÉMARRAGE OPTIMISATION OPTUNA AVANCÉE")
    logger.info("=" * 80)
    logger.info("✨ FONCTIONNALITÉS:")
    logger.info("  🎯 Progression par paliers de capital")
    logger.info("  🧠 Analyse comportementale automatique")
    logger.info("  📊 Stop Loss: 5-13% | Take Profit: 5-15%")
    logger.info("  🏆 Focus sur win rate et qualité des trades")
    logger.info("  ⏱️ Optimisation pour trading durable")
    logger.info("=" * 80)

    try:
        # Setup de l'étude
        study = setup_database("adan_progressive_optimization")

        logger.info(
            f"Démarrage de {args.n_trials} trials avec timeout de {args.timeout}s"
        )

        # Optimisation
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            n_jobs=3,  # Parallélisme avec 3 workers
            gc_after_trial=True,
        )

        # Affichage des résultats
        print_comprehensive_results(study)

        # Sauvegarde des résultats
        results_file = (
            f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        if study.best_trial:
            results = {
                "best_score": study.best_trial.value,
                "best_params": study.best_trial.params,
                "best_attributes": study.best_trial.user_attrs,
                "n_trials": len(study.trials),
                "optimization_time": args.timeout,
            }

            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"📄 Résultats sauvegardés: {results_file}")

        logger.info("✅ OPTIMISATION TERMINÉE AVEC SUCCÈS!")

    except KeyboardInterrupt:
        logger.info("⏹️ Optimisation interrompue par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur d'optimisation: {e}", exc_info=True)


if __name__ == "__main__":
    main()
