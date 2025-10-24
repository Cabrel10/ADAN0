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
# CapitalTierTracker class definition
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

# DailyMetricsTracker class definition
class DailyMetricsTracker:
    """Simple daily metrics tracker for optimization."""
    
    def __init__(self, initial_balance):
        self.initial_balance = initial_balance
        self.current_day = 0
        self.daily_data = {}
        self.total_days = 0
        
    def update(self, current_day, balance, trade_info, return_value):
        """Update daily metrics."""
        if current_day != self.current_day:
            self.current_day = current_day
            self.total_days += 1
            
        if current_day not in self.daily_data:
            self.daily_data[current_day] = {
                'trades_closed': 0,
                'daily_pnl': 0.0,
                'daily_return_pct': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'daily_sharpe': 0.0,
                'profitable_days_pct': 0.0
            }
            
        # Update trade info
        if trade_info.get('trade_closed', False):
            self.daily_data[current_day]['trades_closed'] += 1
            pnl = trade_info.get('trade_pnl', 0.0)
            self.daily_data[current_day]['daily_pnl'] += pnl
            
    def get_current_day_summary(self):
        """Get current day summary."""
        if self.current_day in self.daily_data:
            return self.daily_data[self.current_day]
        return {}
        
    def get_average_daily_performance(self):
        """Get average daily performance."""
        if not self.daily_data:
            return {}
            
        total_trades = sum(day['trades_closed'] for day in self.daily_data.values())
        total_pnl = sum(day['daily_pnl'] for day in self.daily_data.values())
        
        return {
            'total_days': self.total_days,
            'avg_trades_per_day': total_trades / max(self.total_days, 1),
            'avg_daily_return_pct': (total_pnl / self.initial_balance) * 100,
            'avg_win_rate': 0.5,  # Placeholder
            'avg_profit_factor': 1.0,  # Placeholder
            'avg_daily_sharpe': 0.0,  # Placeholder
            'profitable_days_pct': 50.0  # Placeholder
        }
        
    def finalize_current_day(self):
        """Finalize current day tracking."""
        pass

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
    """
    Callback amélioré pour Optuna qui intègre :
    1. Le pruning (élagage) des essais non prometteurs.
    2. Un suivi détaillé des métriques de chaque worker, similaire à `train_parallel_agents.py`.
    """

    def __init__(
        self,
        trial: optuna.Trial,
        eval_env: VecEnv,
        tier_manager: TierProgressionManager,
        eval_freq: int = 5000,
        total_timesteps: int = 25000,
        log_interval: int = 2500,
        initial_balance: float = 20,
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

        # --- Ajouts pour le monitoring détaillé ---
        self.log_interval = log_interval
        self.last_log_step = 0
        self.num_workers = self.eval_env.num_envs

        # Add daily metrics trackers for each worker
        self.daily_trackers = {
            i: DailyMetricsTracker(initial_balance) for i in range(self.num_workers)
        }
        
        # Add capital tier trackers for each worker
        self.tier_trackers = {
            i: CapitalTierTracker(initial_balance) for i in range(self.num_workers)
        }

    def _collect_and_log_metrics(self):
        """Collecte et affiche les métriques de chaque worker avec tracking journalier."""
        try:
            portfolio_managers = self.eval_env.get_attr("portfolio_manager")
            environments = self.eval_env.get_attr("data")

            for worker_id, pm in enumerate(portfolio_managers):
                if pm is None:
                    continue

                metrics = pm.get_metrics()
                balance = metrics.get("total_value", 0)
                pnl = balance - pm.initial_equity

                # Get current day from environment data
                current_day = 0
                trade_info = {}
                if worker_id < len(environments) and environments[worker_id]:
                    env_data = environments[worker_id]
                    if "TIMESTAMP" in env_data and len(env_data["TIMESTAMP"]) > 0:
                        timestamp = (
                            env_data["TIMESTAMP"].iloc[-1]
                            if hasattr(env_data["TIMESTAMP"], "iloc")
                            else env_data["TIMESTAMP"][-1]
                        )
                        current_day = int(timestamp // (24 * 60 * 60 * 1000))

                    # Check for trade information
                    recent_trades = metrics.get("recent_trades", [])
                    if recent_trades:
                        last_trade = (
                            recent_trades[-1]
                            if isinstance(recent_trades, list)
                            else recent_trades
                        )
                        if isinstance(last_trade, dict):
                            trade_info = {
                                "trade_closed": last_trade.get("closed", False),
                                "trade_opened": last_trade.get("opened", False),
                                "trade_pnl": last_trade.get("pnl", 0.0),
                            }

                # Update daily tracker
                return_value = metrics.get("last_return", 0.0)
                self.daily_trackers[worker_id].update(
                    current_day, balance, trade_info, return_value
                )
                
                # Update tier tracker
                self.tier_trackers[worker_id].update(
                    self.num_timesteps, balance, pnl
                )

                # Get daily performance summary
                daily_summary = self.daily_trackers[
                    worker_id
                ].get_average_daily_performance()

                # Enhanced logging with daily metrics
                daily_info = ""
                if daily_summary and daily_summary.get("total_days", 0) > 0:
                    daily_info = (
                        f" | Daily: Return={daily_summary.get('avg_daily_return_pct', 0):.2f}% "
                        f"Trades={daily_summary.get('avg_trades_per_day', 0):.1f} "
                        f"WinRate={daily_summary.get('avg_win_rate', 0):.1f}% "
                        f"PF={min(daily_summary.get('avg_profit_factor', 0), 9.99):.2f}"
                    )

                # Get tier information
                tier_info = self.tier_trackers[worker_id].get_progression_summary()
                tier_str = f" | Tier={tier_info['current_tier']}"
                
                logger.info(
                    f"TRIAL {self.trial.number} | Step {self.num_timesteps:<6} | "
                    f"Worker {worker_id}: Balance=${balance:,.2f} | PnL=${pnl:,.2f} | "
                    f"Sharpe={metrics.get('sharpe_ratio', 0):.2f} | "
                    f"Trades={metrics.get('total_trades', 0)}{tier_str}{daily_info}"
                )
        except Exception as e:
            logger.warning(f"Could not collect metrics during trial: {e}")

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

        # --- NOUVEAU : Suivi détaillé des métriques ---
        if (self.num_timesteps - self.last_log_step) >= self.log_interval:
            self._collect_and_log_metrics()
            self.last_log_step = self.num_timesteps

        # --- Logique existante ---
        # Gestion de la progression par paliers
        expected_epoch = int(
            (self.num_timesteps / self.total_timesteps) * self.tier_manager.total_epochs
        )
        if expected_epoch != self.current_epoch:
            self.current_epoch = expected_epoch
            current_tier = self.tier_manager.get_tier_for_epoch(self.current_epoch)
            if self.progress_bar:
                new_desc = f"Trial {self.trial.number} [Tier: {current_tier.value}]"
                self.progress_bar.set_description(new_desc)

        # Mise à jour de la barre de progression
        if current_time - self.last_update_time >= 2.0:
            if self.progress_bar:
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

        # Évaluation et élagage (pruning)
        if (self.num_timesteps - getattr(self, "last_eval_step", 0)) >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            try:
                current_tier = self.tier_manager.get_tier_for_epoch(self.current_epoch)
                performance = self._evaluate_tier_performance(current_tier)
                tier_key = current_tier.name
                if tier_key not in self.tier_performances:
                    self.tier_performances[tier_key] = []
                self.tier_performances[tier_key].append(performance)

                step_value = self.num_timesteps / self.total_timesteps
                self.trial.report(performance["composite_score"], step_value)

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
        
        # Generate portfolio progression summary
        self._generate_trial_summary()

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
        """Calcule un score global multi-paliers avec focus sur les performances journalières."""
        if not self.tier_performances:
            return 0.0

        # Finalize daily tracking for all workers
        for worker_id in range(self.num_workers):
            self.daily_trackers[worker_id].finalize_current_day()

        # Calculate daily performance scores
        daily_scores = []
        for worker_id in range(self.num_workers):
            daily_perf = self.daily_trackers[worker_id].get_average_daily_performance()
            if daily_perf and daily_perf.get("total_days", 0) > 0:
                # Daily performance composite score
                daily_score = (
                    daily_perf.get("avg_daily_return_pct", 0) * 0.3
                    + min(daily_perf.get("avg_profit_factor", 0), 5.0) * 0.25
                    + daily_perf.get("avg_win_rate", 0) * 0.2
                    + daily_perf.get("avg_daily_sharpe", 0) * 0.15
                    + daily_perf.get("profitable_days_pct", 0) * 0.1
                )
                if (
                    daily_perf.get("avg_trades_per_day", 0) >= 1
                ):  # Ensure sufficient trading activity
                    daily_scores.append(daily_score)

        # Calculate traditional multi-tier score
        traditional_score = 0.0
        total_weight = 0.0
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
                traditional_score += avg_performance * weight
                total_weight += weight

        traditional_score = (
            traditional_score / total_weight if total_weight > 0 else 0.0
        )

        # Combine daily and traditional scores with heavy weight on daily performance
        if daily_scores:
            daily_score_avg = np.mean(daily_scores)
            final_score = daily_score_avg * 0.7 + traditional_score * 0.3

            # Bonus for consistency across workers
            if len(daily_scores) > 1:
                consistency_bonus = max(
                    0, 1.0 - (np.std(daily_scores) / max(abs(daily_score_avg), 0.01))
                )
                final_score *= 1.0 + consistency_bonus * 0.1

            return final_score
        else:
            return traditional_score * 0.5  # Penalty if no sufficient daily data

    def _generate_trial_summary(self):
        """Generate trial summary with tier progression."""
        summary = {
            "trial_number": self.trial.number,
            "total_steps": self.num_timesteps,
            "workers": {},
            "overall_enterprise_count": 0,
        }
        
        for worker_id in range(self.num_workers):
            tier_summary = self.tier_trackers[worker_id].get_progression_summary()
            daily_perf = self.daily_trackers[worker_id].get_average_daily_performance()
            
            worker_summary = {
                "tier_progression": tier_summary,
                "reached_enterprise": tier_summary["reached_enterprise"],
                "daily_performance": daily_perf,
            }
            
            summary["workers"][f"w{worker_id + 1}"] = worker_summary
            
            if tier_summary["reached_enterprise"]:
                summary["overall_enterprise_count"] += 1
        
        # Store in trial attributes
        self.trial.set_user_attr("trial_summary", summary)
        
        logger.info(f"📊 TRIAL {self.trial.number} SUMMARY:")
        logger.info(f"   🏢 Workers reaching Enterprise: {summary['overall_enterprise_count']}/4")
        logger.info(f"   ✅ Enterprise Success Rate: {(summary['overall_enterprise_count'] / 4) * 100:.1f}%")


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
        n_epochs = 1

        # Initialiser le gestionnaire de progression par paliers
        tier_manager = TierProgressionManager(total_epochs=n_epochs)

        # Nouveaux intervalles SL/TP selon vos spécifications
        stop_loss_pct = trial.suggest_float("stop_loss_pct", 0.05, 0.13)  # 5-13%
        take_profit_pct = trial.suggest_float("take_profit_pct", 0.05, 0.15)  # 5-15%

        # Hyperparamètres PPO
        ppo_params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
            "ent_coef": trial.suggest_float("ent_coef", 0.02, 0.1),
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

        # NOUVEAU: force_trade_steps adaptatif par timeframe (pris du config.yaml)
        force_trade_steps_params = (
            GLOBAL_CONFIG.get("trading_rules", {})
            .get("frequency", {})
            .get("force_trade_steps", {})
        )

        # NOUVEAU: Intégrer min_tracking_steps dans l'optimisation
        min_tracking_steps_params = {
            "5m": trial.suggest_int("mts_5m", 1, 10),
            "1h": trial.suggest_int("mts_1h", 1, 8),
            "4h": trial.suggest_int("mts_4h", 1, 6),
        }

        # NOUVEAU: Définition statique des hyperparamètres par worker pour l'optimiseur multivarié
        # --- Worker 1 (Conservative) ---
        w1_position_size_pct = trial.suggest_float("w1_position_size", 0.05, 0.25)
        w1_risk_multiplier = trial.suggest_float("w1_risk_mult", 0.8, 1.5)
        w1_patience_steps = trial.suggest_int("w1_patience", 10, 50)
        w1_min_confidence = trial.suggest_float("w1_min_conf", 0.01, 0.08)
        w1_mts_5m = trial.suggest_int("w1_mts_5m", 1, 12)
        w1_mts_1h = trial.suggest_int("w1_mts_1h", 1, 8)
        w1_mts_4h = trial.suggest_int("w1_mts_4h", 1, 12)

        # --- Worker 2 (Moderate) ---
        w2_position_size_pct = trial.suggest_float("w2_position_size", 0.05, 0.25)
        w2_risk_multiplier = trial.suggest_float("w2_risk_mult", 0.8, 1.5)
        w2_patience_steps = trial.suggest_int("w2_patience", 10, 50)
        w2_min_confidence = trial.suggest_float("w2_min_conf", 0.01, 0.08)
        w2_mts_5m = trial.suggest_int("w2_mts_5m", 1, 12)
        w2_mts_1h = trial.suggest_int("w2_mts_1h", 1, 8)
        w2_mts_4h = trial.suggest_int("w2_mts_4h", 1, 12)

        # --- Worker 3 (Aggressive) ---
        w3_position_size_pct = trial.suggest_float("w3_position_size", 0.05, 0.25)
        w3_risk_multiplier = trial.suggest_float("w3_risk_mult", 0.8, 1.5)
        w3_patience_steps = trial.suggest_int("w3_patience", 10, 50)
        w3_min_confidence = trial.suggest_float("w3_min_conf", 0.01, 0.08)
        w3_mts_5m = trial.suggest_int("w3_mts_5m", 1, 12)
        w3_mts_1h = trial.suggest_int("w3_mts_1h", 1, 8)
        w3_mts_4h = trial.suggest_int("w3_mts_4h", 1, 12)

        # --- Worker 4 (Adaptive) ---
        w4_position_size_pct = trial.suggest_float("w4_position_size", 0.05, 0.25)
        w4_risk_multiplier = trial.suggest_float("w4_risk_mult", 0.8, 1.5)
        w4_patience_steps = trial.suggest_int("w4_patience", 10, 50)
        w4_min_confidence = trial.suggest_float("w4_min_conf", 0.01, 0.08)
        w4_mts_5m = trial.suggest_int("w4_mts_5m", 1, 12)
        w4_mts_1h = trial.suggest_int("w4_mts_1h", 1, 8)
        w4_mts_4h = trial.suggest_int("w4_mts_4h", 1, 12)

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

                # NOUVEAU: Application statique des hyperparamètres par worker
                # Regrouper les variables unrollées pour un accès plus facile
                unrolled_params = {
                    "w1": {
                        "position_size_pct": w1_position_size_pct,
                        "risk_multiplier": w1_risk_multiplier,
                        "patience_steps": w1_patience_steps,
                        "min_confidence": w1_min_confidence,
                        "mts_5m": w1_mts_5m,
                        "mts_1h": w1_mts_1h,
                        "mts_4h": w1_mts_4h,
                    },
                    "w2": {
                        "position_size_pct": w2_position_size_pct,
                        "risk_multiplier": w2_risk_multiplier,
                        "patience_steps": w2_patience_steps,
                        "min_confidence": w2_min_confidence,
                        "mts_5m": w2_mts_5m,
                        "mts_1h": w2_mts_1h,
                        "mts_4h": w2_mts_4h,
                    },
                    "w3": {
                        "position_size_pct": w3_position_size_pct,
                        "risk_multiplier": w3_risk_multiplier,
                        "patience_steps": w3_patience_steps,
                        "min_confidence": w3_min_confidence,
                        "mts_5m": w3_mts_5m,
                        "mts_1h": w3_mts_1h,
                        "mts_4h": w3_mts_4h,
                    },
                    "w4": {
                        "position_size_pct": w4_position_size_pct,
                        "risk_multiplier": w4_risk_multiplier,
                        "patience_steps": w4_patience_steps,
                        "min_confidence": w4_min_confidence,
                        "mts_5m": w4_mts_5m,
                        "mts_1h": w4_mts_1h,
                        "mts_4h": w4_mts_4h,
                    },
                }

        # Store worker-specific parameters for later use
        worker_specific_params = unrolled_params

        # Apply worker-specific parameters to temp_config
        for worker_key in ["w1", "w2", "w3", "w4"]:
            if worker_key in temp_config["workers"]:
                if worker_key in unrolled_params:
                    wsp = unrolled_params[worker_key]
                    temp_config["workers"][worker_key]["position_size_pct"] = wsp[
                        "position_size_pct"
                    ]
                    temp_config["workers"][worker_key]["risk_multiplier"] = wsp[
                        "risk_multiplier"
                    ]
                    temp_config["workers"][worker_key]["patience_steps"] = wsp[
                        "patience_steps"
                    ]
                    temp_config["workers"][worker_key]["min_confidence"] = wsp[
                        "min_confidence"
                    ]

                    # Appliquer les MTS optimisés pour ce worker MAIS forcer à 1 minimum
                    if (
                        "specialization" in temp_config["workers"][worker_key]
                        and "tracking_periods"
                        in temp_config["workers"][worker_key]["specialization"]
                    ):
                        temp_config["workers"][worker_key]["specialization"][
                            "tracking_periods"
                        ]["5m"]["min_tracking_steps"] = max(1, min(wsp["mts_5m"], 2))
                        temp_config["workers"][worker_key]["specialization"][
                            "tracking_periods"
                        ]["1h"]["min_tracking_steps"] = max(1, min(wsp["mts_1h"], 2))
                        temp_config["workers"][worker_key]["specialization"][
                            "tracking_periods"
                        ]["4h"]["min_tracking_steps"] = max(1, min(wsp["mts_4h"], 2))

                # NOUVEAU: Appliquer les min_tracking_steps optimisés MAIS plafonner à 2
                for timeframe, min_steps in min_tracking_steps_params.items():
                    if timeframe in temp_config["workers"][worker_key].get(
                        "specialization", {}
                    ).get("tracking_periods", {}):
                        temp_config["workers"][worker_key]["specialization"][
                            "tracking_periods"
                        ][timeframe]["min_tracking_steps"] = max(1, min(min_steps, 2))

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

        # Appliquer les force_trade_steps adaptatifs
        if "trading_rules" not in temp_config:
            temp_config["trading_rules"] = {}
        if "frequency" not in temp_config["trading_rules"]:
            temp_config["trading_rules"]["frequency"] = {}
        temp_config["trading_rules"]["frequency"]["force_trade_steps"] = (
            force_trade_steps_params
        )

        # Configuration pour entraînement optimisé pour générer des trades
        temp_config["environment"]["max_steps"] = (
            50000  # 50k steps au lieu de 8k pour permettre l'apprentissage
        )
        temp_config["environment"]["max_chunks_per_episode"] = 2  # Réduit pour focus

        # OPTIMISATION COHÉRENTE - Respecter la configuration de base
        for worker_key in ["w1", "w2", "w3", "w4"]:
            if worker_key in temp_config["workers"]:
                # Utiliser les valeurs optimisées mais cohérentes
                temp_config["workers"][worker_key]["min_confidence"] = max(
                    0.01, worker_specific_params[worker_key]["min_confidence"]
                )

                # Utiliser les patience_steps optimisés
                temp_config["workers"][worker_key]["patience_steps"] = worker_specific_params[worker_key]["patience_steps"]

                # Appliquer les min_tracking_steps optimisés de manière cohérente
                if "specialization" in temp_config["workers"][worker_key]:
                    if (
                        "tracking_periods"
                        in temp_config["workers"][worker_key]["specialization"]
                    ):
                        for tf in ["5m", "1h", "4h"]:
                            if (
                                tf
                                in temp_config["workers"][worker_key]["specialization"][
                                    "tracking_periods"
                                ]
                            ):
                                # Utiliser les valeurs optimisées mais avec des limites raisonnables
                                optimized_mts = worker_specific_params[worker_key].get(f"mts_{tf}", 1)
                                temp_config["workers"][worker_key]["specialization"][
                                    "tracking_periods"
                                ][tf]["min_tracking_steps"] = max(1, min(optimized_mts, 3))
                                # Garder grace_period raisonnable
                                temp_config["workers"][worker_key]["specialization"][
                                    "tracking_periods"
                                ][tf]["grace_period"] = max(1, min(optimized_mts, 2))

        # Ajustements globaux pour favoriser les trades
        if "trading_rules" not in temp_config:
            temp_config["trading_rules"] = {}

        # Utiliser action_threshold cohérent avec la configuration
        temp_config["trading_rules"]["action_threshold"] = 0.01  # Valeur du config.yaml

        # Forcer frequency rules très permissives
        if "frequency" not in temp_config["trading_rules"]:
            temp_config["trading_rules"]["frequency"] = {}

        # Désactiver complètement les garde-fous de fréquence
        temp_config["trading_rules"]["frequency"]["min_steps_between_trades"] = 1
        temp_config["trading_rules"]["frequency"]["position_size_decay"] = 1.0
        temp_config["progressive_training"] = {
            "enabled": True,
            "tier_progression": True,
            "epochs_per_tier": max(1, n_epochs // len(CapitalTier)),
        }

        # --- Copied Env Creation Logic ---
        num_envs = 4
        env_fns = []
        worker_ids = ["w1", "w2", "w3", "w4"]

        for i in range(num_envs):
            worker_id = worker_ids[i]
            worker_config = temp_config["workers"][worker_id]

            data_loader = ChunkedDataLoader(
                config=temp_config, worker_config=worker_config, worker_id=i
            )
            data = data_loader.load_chunk(0)  # Utiliser le premier chunk avec plus d'entraînement

            env_worker_config = copy.deepcopy(worker_config)
            env_worker_config["worker_id"] = i

            env_log_dir = os.path.join(
                temp_config["paths"]["logs_dir"],
                f"{worker_id}_env_trial_{trial.number}",
            )
            os.makedirs(env_log_dir, exist_ok=True)

            env_kwargs = {
                "data": data,
                "timeframes": temp_config["data"]["timeframes"],
                "window_size": temp_config["environment"]["window_size"],
                "features_config": temp_config["data"]["features_config"]["timeframes"],
                "max_steps": temp_config["environment"]["max_steps"],
                "initial_balance": temp_config["portfolio"]["initial_balance"],
                "commission": temp_config["environment"]["commission"],
                "reward_scaling": temp_config["environment"]["reward_scaling"],
                "enable_logging": True,
                "log_dir": env_log_dir,
                "worker_config": env_worker_config,
                "config": temp_config,
                "exploration_tutor": temp_config.get("reward_shaping", {}).get(
                    "exploration_tutor", {}
                ),
            }
            env_fns.append(lambda kwargs=env_kwargs: MultiAssetChunkedEnv(**kwargs))

        logger.info(
            f"🔄 Trial {trial.number}: Using SubprocVecEnv for TRUE PARALLEL execution ({num_envs} workers)"
        )
        env = SubprocVecEnv(env_fns, start_method="spawn")

        # Callback avec progression par paliers
        callback = OptunaPruningCallback(
            trial=trial,
            eval_env=env,
            tier_manager=tier_manager,
            eval_freq=3000,
            total_timesteps=temp_config["environment"]["max_steps"],
            log_interval=5000,
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

        # NOUVEAU: Évaluation individuelle par worker
        analyzer = BehaviorAnalyzer()
        worker_scores = {}
        worker_behaviors = {}

        # Évaluation INTENSIVE par worker avec LOGGING COMPLET
        EVALUATION_STEPS = 5000  # Beaucoup plus d'étapes pour générer des trades

        for worker_idx, worker_id in enumerate(["w1", "w2", "w3", "w4"]):
            logger.info(
                f"🔍 DÉBUT ÉVALUATION INTENSIVE {worker_id.upper()} - {EVALUATION_STEPS} steps"
            )

            worker_data = {
                "trades": [],
                "returns": [],
                "portfolio_values": [20.50],
                "total_steps": 0,
                "timeframe_trades": {"5m": 0, "1h": 0, "4h": 0},
                "actions_taken": 0,
                "positions_opened": 0,
                "positions_closed": 0,
                "trade_attempts": 0,
            }

            # Reset environnement pour ce worker
            obs = env.reset()
            logger.info(f"📊 {worker_id}: Environnement reset, début évaluation...")

            # Compteurs pour monitoring temps réel
            step_log_interval = 200
            last_trade_count = 0

            for step in range(EVALUATION_STEPS):
                # Prédiction action
                action, _ = model.predict(obs, deterministic=False)
                worker_data["actions_taken"] += 1

                # Exécution action
                obs, reward, done, info = env.step(action)
                worker_data["total_steps"] += 1

                # Analyse INFO pour ce worker
                env_info = {}
                if hasattr(info, "__iter__") and len(info) > worker_idx:
                    env_info = (
                        info[worker_idx] if isinstance(info[worker_idx], dict) else {}
                    )

                    # Collecte portfolio value
                    if "portfolio_value" in env_info:
                        worker_data["portfolio_values"].append(
                            float(env_info["portfolio_value"])
                        )

                    # Collecte returns
                    if "return" in env_info:
                        worker_data["returns"].append(float(env_info["return"]))

                    # DÉTECTION DE TRADE COMPLÉTÉ - LOG IMMÉDIAT
                    if "trade_completed" in env_info and env_info["trade_completed"]:
                        trade_info = {
                            "pnl": env_info.get("trade_pnl", 0),
                            "duration": env_info.get("trade_duration", 0),
                            "timeframe": env_info.get("timeframe", "5m"),
                            "step": step,
                        }
                        worker_data["trades"].append(trade_info)
                        worker_data["timeframe_trades"][trade_info["timeframe"]] += 1

                        logger.info(
                            f"🎯 TRADE COMPLET {worker_id}: Step {step}, TF={trade_info['timeframe']}, "
                            f"PnL=${trade_info['pnl']:.2f}, Duration={trade_info['duration']}"
                        )

                    # DÉTECTION POSITION OUVERTE
                    if "position_opened" in env_info and env_info["position_opened"]:
                        worker_data["positions_opened"] += 1
                        logger.info(
                            f"🔓 POSITION OUVERTE {worker_id}: Step {step}, "
                            f"TF={env_info.get('timeframe', '?')}"
                        )

                    # DÉTECTION POSITION FERMÉE
                    if "position_closed" in env_info and env_info["position_closed"]:
                        worker_data["positions_closed"] += 1
                        logger.info(
                            f"🔒 POSITION FERMÉE {worker_id}: Step {step}, "
                            f"PnL=${env_info.get('trade_pnl', 0):.2f}"
                        )

                    # COMPTEURS DE POSITIONS PAR TIMEFRAME
                    if "positions_count" in env_info:
                        counts = env_info["positions_count"]
                        if isinstance(counts, dict):
                            for tf in ["5m", "1h", "4h"]:
                                if tf in counts:
                                    worker_data["timeframe_trades"][tf] = int(
                                        counts[tf]
                                    )

                    # TENTATIVES DE TRADE
                    if "trade_attempt" in env_info and env_info["trade_attempt"]:
                        worker_data["trade_attempts"] += 1

                    # Log activité de trading périodique
                    current_trade_count = len(worker_data["trades"])
                    if current_trade_count > last_trade_count:
                        logger.info(
                            f"📈 {worker_id}: NOUVEAU TRADE! Total={current_trade_count}"
                        )
                        last_trade_count = current_trade_count

                # Reset si episode terminé
                if done.any():
                    obs = env.reset()

                # Log de progression périodique
                if step % 1000 == 0 and step > 0:
                    logger.info(
                        f"⏱️ {worker_id}: Step {step}/{EVALUATION_STEPS}, "
                        f"Trades={len(worker_data['trades'])}, "
                        f"Positions={worker_data['positions_opened']}, "
                        f"Tentatives={worker_data['trade_attempts']}"
                    )

            # LOG FINAL pour ce worker
            total_trades = len(worker_data["trades"])
            logger.info(f"🏁 FIN ÉVALUATION {worker_id.upper()}:")
            logger.info(f"   📊 Total steps: {worker_data['total_steps']}")
            logger.info(f"   🎯 Actions prises: {worker_data['actions_taken']}")
            logger.info(f"   📈 Trades complétés: {total_trades}")
            logger.info(f"   🔓 Positions ouvertes: {worker_data['positions_opened']}")
            logger.info(f"   🔒 Positions fermées: {worker_data['positions_closed']}")
            logger.info(f"   ⚡ Tentatives trades: {worker_data['trade_attempts']}")
            logger.info(f"   ⏰ Répartition TF: {worker_data['timeframe_trades']}")

            if worker_data["portfolio_values"]:
                final_portfolio = worker_data["portfolio_values"][-1]
                initial_portfolio = worker_data["portfolio_values"][0]
                growth = (
                    (final_portfolio - initial_portfolio) / initial_portfolio
                ) * 100
                logger.info(
                    f"   💼 Portfolio: ${initial_portfolio:.2f} → ${final_portfolio:.2f} ({growth:+.2f}%)"
                )

            # CALCUL MÉTRIQUES DÉTAILLÉES
            trades_list = worker_data["trades"]
            total_pnl = sum(trade.get("pnl", 0) for trade in trades_list)
            portfolio_final = (
                worker_data["portfolio_values"][-1]
                if worker_data["portfolio_values"]
                else 20.50
            )
            total_trades = len(trades_list)

            # PÉNALITÉ SÉVÈRE si aucun trade
            if total_trades == 0:
                logger.warning(
                    f"❌ {worker_id}: AUCUN TRADE - Score pénalisé sévèrement!"
                )
                worker_score = -1.0  # Score négatif mais pas aussi punitif que -2.0
                win_rate = 0.0
                portfolio_growth = -0.5  # Moins punitif que -1.0
                tf_diversity = 0
                tf_bonus = 0.0
                trade_frequency = 0.0
            else:
                # Score multi-timeframe : bonus si le worker utilise plusieurs TF
                tf_trades = worker_data["timeframe_trades"]
                tf_diversity = sum(1 for count in tf_trades.values() if count > 0)
                tf_bonus = tf_diversity * 0.15

                # Win rate
                winning_trades = sum(
                    1 for trade in trades_list if trade.get("pnl", 0) > 0
                )
                win_rate = winning_trades / total_trades

                # Portfolio growth
                portfolio_growth = (
                    (portfolio_final - 20.50) / 20.50 if portfolio_final > 0 else -0.2  # Moins punitif que -1.0
                )

                # Trade frequency
                trade_frequency = total_trades / EVALUATION_STEPS

                # Score composite
                base_score = (
                    portfolio_growth * 0.35
                    + win_rate * 0.25
                    + tf_bonus * 0.25
                    + trade_frequency * 0.15
                )

                # Bonus pour activité
                activity_bonus = 0.0
                if total_trades >= 5:
                    activity_bonus += 0.1
                if total_trades >= 10:
                    activity_bonus += 0.1
                if worker_data["positions_opened"] >= 10:
                    activity_bonus += 0.05

                worker_score = base_score + activity_bonus

            # Get tier progression summary
            tier_summary = self.tier_trackers[worker_idx].get_progression_summary()
            
            # Stockage des comportements
            worker_behaviors[worker_id] = {
                "total_trades": total_trades,
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "portfolio_final": portfolio_final,
                "portfolio_growth": portfolio_growth,
                "timeframe_trades": tf_trades,
                "tf_diversity": tf_diversity,
                "score": worker_score,
                "params": worker_specific_params[worker_id],
                "positions_opened": worker_data["positions_opened"],
                "positions_closed": worker_data["positions_closed"],
                "trade_attempts": worker_data["trade_attempts"],
                "actions_taken": worker_data["actions_taken"],
                "evaluation_steps": EVALUATION_STEPS,
                "trade_frequency": trade_frequency,
                "activity_score": worker_data["positions_opened"] + total_trades,
                "tier_progression": tier_summary,
                "reached_enterprise": tier_summary["reached_enterprise"],
            }

            worker_scores[worker_id] = worker_score

            # LOG FINAL DÉTAILLÉ pour ce worker
            logger.info(f"🎯 RÉSULTAT FINAL {worker_id.upper()}:")
            logger.info(f"   Score: {worker_score:.4f}")
            logger.info(f"   Trades: {total_trades}")
            logger.info(f"   PnL: ${total_pnl:.2f}")
            logger.info(f"   Win Rate: {win_rate:.1%}")
            logger.info(f"   Portfolio Growth: {portfolio_growth:.2%}")
            logger.info(f"   TF Diversity: {tf_diversity}/3")
            logger.info(f"   TF Distribution: {tf_trades}")
            logger.info(f"   Positions Opened: {worker_data['positions_opened']}")
            logger.info(f"   Trade Attempts: {worker_data['trade_attempts']}")
            logger.info(f"   Trade Frequency: {trade_frequency:.4f}")
            logger.info(f"   Tier Progression: {tier_summary['current_tier']}")
            logger.info(f"   Reached Enterprise: {'✅ YES' if tier_summary['reached_enterprise'] else '❌ NO'}")
            logger.info("   " + "=" * 50)

        # Score final : moyenne des workers avec bonus si tous sont positifs
        individual_scores = list(worker_scores.values())
        if not individual_scores:
            # Si aucun worker n'a de score valide, retourner un score très négatif mais pas -inf
            final_score = -5.0
            logger.warning("No valid worker scores available, returning penalty score")
        else:
            final_score = sum(individual_scores) / len(individual_scores)

        # Bonus si tous les workers sont rentables
        all_positive = all(score > 0 for score in individual_scores)
        if all_positive:
            final_score += 0.2
            
        # Bonus Enterprise : si des workers atteignent Enterprise tier
        enterprise_count = sum(1 for worker_id in ["w1", "w2", "w3", "w4"] 
                              if worker_behaviors.get(worker_id, {}).get("reached_enterprise", False))
        if enterprise_count > 0:
            enterprise_bonus = enterprise_count * 0.1  # 0.1 par worker Enterprise
            final_score += enterprise_bonus
            logger.info(f"🏢 Enterprise Bonus: {enterprise_count}/4 workers reached Enterprise tier (+{enterprise_bonus:.2f})")

        logger.info(f"Individual scores: {worker_scores}")
        logger.info(f"Final combined score: {final_score:.3f}")

        # NOUVEAU: Sauvegarder les résultats par worker
        trial.set_user_attr("worker_scores", worker_scores)
        trial.set_user_attr("worker_behaviors", worker_behaviors)
        trial.set_user_attr("worker_specific_params", worker_specific_params)

        # Sauvegarder les meilleurs paramètres par worker
        best_w1_score = worker_scores.get("w1", -1.0)
        best_w2_score = worker_scores.get("w2", -1.0)
        best_w3_score = worker_scores.get("w3", -1.0)
        best_w4_score = worker_scores.get("w4", -1.0)

        trial.set_user_attr("w1_score", best_w1_score)
        trial.set_user_attr("w2_score", best_w2_score)
        trial.set_user_attr("w3_score", best_w3_score)
        trial.set_user_attr("w4_score", best_w4_score)

        trial.set_user_attr("w1_params", worker_specific_params["w1"])
        trial.set_user_attr("w2_params", worker_specific_params["w2"])
        trial.set_user_attr("w3_params", worker_specific_params["w3"])
        trial.set_user_attr("w4_params", worker_specific_params["w4"])

        # Paramètres globaux pour compatibilité
        trial.set_user_attr("final_behavior_score", final_score)
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

        # Log du meilleur worker
        best_worker = max(worker_scores.keys(), key=lambda k: worker_scores[k])
        logger.info(
            f"Best worker: {best_worker} with score {worker_scores[best_worker]:.3f}"
        )
        logger.info(f"Best worker params: {worker_specific_params[best_worker]}")

        return final_score

    except optuna.exceptions.TrialPruned:
        logger.info(f"Trial {trial.number} was pruned")
        raise
    except Exception as e:
        import traceback

        logger.error(f"Trial {trial.number} failed critically. Full traceback below:")
        logger.error(traceback.format_exc())
        # We still return a very low score so Optuna can log the trial as failed and continue
        # but the traceback will be visible in the logs. Using -10.0 instead of -inf for better optimization.
        return -10.0
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
    parser.add_argument("--n-trials", type=int, default=1, help="Nombre de trials")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout en secondes (10 min pour plus d'apprentissage)")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Chemin vers config.yaml",
    )

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
            n_jobs=1,  # MODIFIÉ: Forcer 1 worker pour le débogage
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
