#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Agent PPO avec callback d'affichage hiérarchique pour ADAN Trading Bot."""

import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as sb3_configure


def setup_logger(log_file: str = "training_log.txt", logger_name: str = 'PPOTraining') -> logging.Logger:
    """
    Configuration du logger pour l'entraînement PPO.

    Args:
        log_file: Chemin vers le fichier de log
        logger_name: Nom du logger

    Returns:
        Logger configuré
    """
    logger = logging.getLogger(logger_name)

    # Éviter la duplication des handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Handler pour la console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Handler pour le fichier
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)

    # Format des messages
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


class HierarchicalTrainingDisplayCallback(BaseCallback):
    """
    Callback personnalisé pour un affichage hiérarchique complet de l'entraînement.
    Affiche les métriques de portfolio, positions, métriques financières et modèle.
    """

    def __init__(
        self,
        verbose: int = 0,
        display_freq: int = 1000,
        total_timesteps: int = 1000000,
        initial_capital: float = 20.50,
        log_file: str = "training_log.txt"
    ):
        """
        Initialise le callback d'affichage hiérarchique.

        Args:
            verbose: Niveau de verbosité
            display_freq: Fréquence d'affichage (en steps)
            total_timesteps: Nombre total de timesteps pour l'entraînement
            initial_capital: Capital initial du portfolio
            log_file: Fichier de log
        """
        super(HierarchicalTrainingDisplayCallback, self).__init__(verbose)
        self.display_freq = display_freq
        self.total_timesteps = total_timesteps
        self.initial_capital = initial_capital
        self.correlation_id = str(uuid.uuid4())[:8]

        # Configurer le logger
        self.logger = setup_logger(log_file)

        # Métriques de suivi
        self.episode_rewards = []
        self.episode_count = 0
        self.start_time = time.time()
        self.last_display_time = time.time()

        # Métriques financières
        self.metrics = {
            "sharpe": 0.0,
            "sortino": 0.0,
            "profit_factor": 0.0,
            "max_dd": 0.0,
            "cagr": 0.0,
            "win_rate": 0.0,
            "trades": 0,
            "volatility": 0.0
        }

        # Positions et portfolio
        self.positions = {}
        self.closed_positions = []
        self.portfolio_value = initial_capital
        self.drawdown = 0.0
        self.cash = initial_capital

    def _on_training_start(self) -> None:
        """Démarrage de l'entraînement avec affichage de la configuration."""
        self.start_time = time.time()
        self.last_display_time = self.start_time

        self.logger.info("╭" + "─" * 70 + "╮")
        self.logger.info("│" + " " * 20 + "🚀 DÉMARRAGE ADAN TRAINING" + " " * 20 + "│")
        self.logger.info("╰" + "─" * 70 + "╯")

        self.logger.info(f"[TRAINING START] Correlation ID: {self.correlation_id}")
        self.logger.info(f"[TRAINING START] Total timesteps: {self.total_timesteps:,}")
        self.logger.info(f"[TRAINING START] Display frequency: {self.display_freq:,} steps")
        self.logger.info(f"[TRAINING START] Capital initial: ${self.initial_capital:.2f}")

        # Section Configuration Flux Monétaires
        self.logger.info("╭" + "─" * 25 + " Configuration Flux Monétaires " + "─" * 25 + "╮")
        self.logger.info(f"│ 💰 Capital Initial: ${self.initial_capital:<40.2f}│")
        self.logger.info("│ 🎯 Gestion Dynamique des Flux Activée" + " " * 32 + "│")
        self.logger.info("│ 📊 Monitoring en Temps Réel" + " " * 39 + "│")
        self.logger.info("│ 🔄 Corrélation ID: " + self.correlation_id + " " * 42 + "│")
        self.logger.info("╰" + "─" * 82 + "╯")

    def _on_step(self) -> bool:
        """
        Appelé à chaque étape pour mettre à jour l'affichage et collecter les métriques.
        """
        # Collecter les récompenses depuis les informations locales
        try:
            # Récupérer les récompenses depuis les buffers du modèle
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                recent_episodes = self.model.ep_info_buffer[-10:]  # 10 derniers épisodes
                rewards = [ep_info.get('r', 0) for ep_info in recent_episodes if 'r' in ep_info]
                if rewards:
                    self.episode_rewards.extend(rewards[-5:])  # Garder les 5 dernières récompenses
                    if len(self.episode_rewards) > 50:  # Limiter la taille du buffer
                        self.episode_rewards = self.episode_rewards[-50:]
        except Exception as e:
            self.logger.debug(f"Erreur lors de la collecte des récompenses: {e}")

        # Vérifier si un épisode est terminé et afficher la progression
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            current_episodes = len(self.model.ep_info_buffer)
            if current_episodes > self.episode_count:
                self.episode_count = current_episodes
                self._display_episode_progress()

        # Affichage hiérarchique périodique
        if self.num_timesteps % self.display_freq == 0 and self.num_timesteps > 0:
            self._log_detailed_metrics()

        return True

    def _display_episode_progress(self) -> None:
        """Affiche la barre de progression à la fin de chaque épisode."""
        try:
            progress = (self.num_timesteps / self.total_timesteps) * 100

            # Barre de progression visuelle
            progress_bar_length = 30
            filled_length = int(progress_bar_length * progress / 100)
            bar = "━" * filled_length + "─" * (progress_bar_length - filled_length)

            # Calculer la récompense moyenne récente
            mean_reward = 0.0
            if self.episode_rewards:
                recent_rewards = self.episode_rewards[-10:]
                mean_reward = np.mean(recent_rewards)

            # Temps écoulé et ETA
            elapsed = time.time() - self.start_time
            if progress > 0:
                eta = (elapsed / progress * 100) - elapsed
                eta_str = str(timedelta(seconds=int(eta)))
            else:
                eta_str = "N/A"

            self.logger.info(
                f"🚀 ADAN Training {bar} {progress:.1f}% ({self.num_timesteps:,}/{self.total_timesteps:,}) • "
                f"Episode {self.episode_count} • Mean Reward: {mean_reward:.3f} • ETA: {eta_str}"
            )

        except Exception as e:
            self.logger.error(f"Erreur lors de l'affichage de la progression: {e}")

    def _log_detailed_metrics(self) -> None:
        """Affichage détaillé des métriques avec structure hiérarchique."""
        try:
            self._update_environment_metrics()
            self._update_model_metrics()

            # Temps et vitesse
            elapsed = time.time() - self.start_time
            steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0.0
            current_time = time.time()
            recent_steps_per_sec = self.display_freq / (current_time - self.last_display_time) if current_time > self.last_display_time else 0.0
            self.last_display_time = current_time

            # En-tête de section
            self.logger.info("╭" + "─" * 80 + "╮")
            self.logger.info("│" + " " * 25 + f"ÉTAPE {self.num_timesteps:,}" + " " * 25 + "│")
            self.logger.info("╰" + "─" * 80 + "╯")

            # Métriques de portfolio
            roi = ((self.portfolio_value - self.initial_capital) / self.initial_capital) * 100 if self.initial_capital > 0 else 0
            self.logger.info(
                f"📊 PORTFOLIO | Valeur: ${self.portfolio_value:.2f} | Cash: ${self.cash:.2f} | "
                f"ROI: {roi:+.2f}%"
            )

            # Métriques de risque
            self.logger.info(
                f"⚠️  RISK | Drawdown: {self.drawdown:.2f}% | Max DD: {self.metrics['max_dd']:.2f}% | "
                f"Volatilité: {self.metrics['volatility']:.2f}%"
            )

            # Métriques de performance
            self.logger.info(
                f"📈 METRICS | Sharpe: {self.metrics['sharpe']:.2f} | Sortino: {self.metrics['sortino']:.2f} | "
                f"Profit Factor: {self.metrics['profit_factor']:.2f}"
            )

            self.logger.info(
                f"📊 TRADING | CAGR: {self.metrics['cagr']:.2f}% | Win Rate: {self.metrics['win_rate']:.1f}% | "
                f"Trades: {self.metrics['trades']}"
            )

            # Positions ouvertes
            self._display_open_positions()

            # Métriques du modèle
            self._display_model_metrics()

            # Informations temporelles
            self.logger.info(
                f"⏱️  TIMING | Elapsed: {elapsed/60:.1f}min | Speed: {steps_per_sec:.1f} steps/s | "
                f"Recent: {recent_steps_per_sec:.1f} steps/s"
            )

            self.logger.info("─" * 80)

        except Exception as e:
            self.logger.error(f"Erreur lors de l'affichage des métriques détaillées: {e}")

    def _update_environment_metrics(self) -> None:
        """Met à jour les métriques depuis l'environnement."""
        try:
            # Essayer de récupérer les métriques via l'environnement vectorisé
            if hasattr(self.model, 'get_env'):
                env = self.model.get_env()

                # Pour les environnements vectorisés
                if hasattr(env, 'envs') and len(env.envs) > 0:
                    first_env = env.envs[0]

                    # Naviguer à travers les wrappers pour trouver les métriques
                    current_env = first_env
                    while hasattr(current_env, 'env'):
                        if hasattr(current_env, 'get_portfolio_metrics'):
                            metrics = current_env.get_portfolio_metrics()
                            if metrics:
                                self.portfolio_value = metrics.get('portfolio_value', self.initial_capital)
                                self.drawdown = metrics.get('drawdown', 0.0)
                                self.cash = metrics.get('cash', self.initial_capital)
                                self.positions = metrics.get('positions', {})
                                self.closed_positions = metrics.get('closed_positions', [])

                                # Mettre à jour les métriques financières
                                self.metrics.update({
                                    "sharpe": metrics.get('sharpe', 0.0),
                                    "sortino": metrics.get('sortino', 0.0),
                                    "profit_factor": metrics.get('profit_factor', 0.0),
                                    "max_dd": metrics.get('max_dd', 0.0),
                                    "cagr": metrics.get('cagr', 0.0),
                                    "win_rate": metrics.get('win_rate', 0.0),
                                    "trades": metrics.get('trades', 0),
                                    "volatility": metrics.get('volatility', 0.0)
                                })
                                break
                        current_env = getattr(current_env, 'env', None)
                        if current_env is None:
                            break

                # Méthode de fallback avec get_attr si disponible
                elif hasattr(env, 'get_attr'):
                    try:
                        env_infos = env.get_attr('last_info')
                        if env_infos and len(env_infos) > 0 and env_infos[0]:
                            info = env_infos[0]
                            self._extract_info_metrics(info)
                    except Exception as e:
                        self.logger.debug(f"Fallback get_attr failed: {e}")

        except Exception as e:
            self.logger.debug(f"Erreur lors de la mise à jour des métriques d'environnement: {e}")

    def _extract_info_metrics(self, info: Dict[str, Any]) -> None:
        """Extrait les métriques depuis le dictionnaire info."""
        self.portfolio_value = info.get('portfolio_value', self.initial_capital)
        self.drawdown = info.get('drawdown', 0.0)
        self.cash = info.get('cash', self.initial_capital)
        self.positions = info.get('positions', {})
        self.closed_positions = info.get('closed_positions', [])

        # Mettre à jour les métriques financières
        self.metrics.update({
            "sharpe": info.get('sharpe', 0.0),
            "sortino": info.get('sortino', 0.0),
            "profit_factor": info.get('profit_factor', 0.0),
            "max_dd": info.get('max_dd', 0.0),
            "cagr": info.get('cagr', 0.0),
            "win_rate": info.get('win_rate', 0.0),
            "trades": info.get('trades', 0),
            "volatility": info.get('volatility', 0.0)
        })

    def _update_model_metrics(self) -> None:
        """Met à jour les métriques du modèle PPO."""
        try:
            self.model_metrics = {}
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                self.model_metrics = self.model.logger.name_to_value
        except Exception as e:
            self.logger.debug(f"Erreur lors de la mise à jour des métriques du modèle: {e}")

    def _display_open_positions(self) -> None:
        """Affiche les positions ouvertes si disponibles."""
        try:
            if self.positions and any(self.positions.values()):
                self.logger.info("╭" + "─" * 28 + " Positions Ouvertes " + "─" * 28 + "╮")

                for asset, pos in self.positions.items():
                    if isinstance(pos, dict) and pos:
                        size = pos.get('size', 0)
                        entry_price = pos.get('entry_price', 0)
                        current_price = pos.get('current_price', entry_price)
                        value = pos.get('value', 0)
                        sl = pos.get('sl', 0)
                        tp = pos.get('tp', 0)
                        pnl_unrealized = pos.get('pnl_unrealized', 0)

                        self.logger.info(
                            f"│ {asset}: Size: {size:.2f} @ {entry_price:.4f} | "
                            f"Current: {current_price:.4f} | Value: ${value:.2f}"
                            + " " * (80 - len(f"│ {asset}: Size: {size:.2f} @ {entry_price:.4f} | Current: {current_price:.4f} | Value: ${value:.2f}")) + "│"
                        )

                        if sl > 0 or tp > 0:
                            self.logger.info(
                                f"│   └─ SL: {sl:.4f} | TP: {tp:.4f} | P&L: ${pnl_unrealized:.2f}"
                                + " " * (80 - len(f"│   └─ SL: {sl:.4f} | TP: {tp:.4f} | P&L: ${pnl_unrealized:.2f}")) + "│"
                            )

                self.logger.info("╰" + "─" * 78 + "╯")
            else:
                self.logger.info("📝 POSITIONS | Aucune position ouverte")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'affichage des positions: {e}")

    def _display_model_metrics(self) -> None:
        """Affiche les métriques du modèle PPO."""
        try:
            if hasattr(self, 'model_metrics') and self.model_metrics:
                total_loss = self.model_metrics.get("train/loss", 0.0)
                policy_loss = self.model_metrics.get("train/policy_loss", 0.0)
                value_loss = self.model_metrics.get("train/value_loss", 0.0)
                entropy = self.model_metrics.get("train/entropy_loss", 0.0)
                clip_fraction = self.model_metrics.get("train/clip_fraction", 0.0)

                self.logger.info(
                    f"🧠 MODEL | Loss: {total_loss:.4f} | Policy: {policy_loss:.4f} | "
                    f"Value: {value_loss:.4f} | Entropy: {entropy:.4f}"
                )

                if clip_fraction > 0:
                    self.logger.info(f"🎯 LEARNING | Clip Fraction: {clip_fraction:.3f}")
            else:
                self.logger.info("🧠 MODEL | Métriques non disponibles")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'affichage des métriques du modèle: {e}")

    def _on_rollout_end(self) -> None:
        """Appelé à la fin de chaque rollout pour capturer les positions fermées."""
        try:
            self._display_closed_positions()
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement du rollout: {e}")

    def _display_closed_positions(self) -> None:
        """Affiche les positions fermées récemment."""
        try:
            if self.closed_positions:
                recent_closed = self.closed_positions[-5:]  # 5 dernières positions fermées

                if recent_closed:
                    self.logger.info("╭" + "─" * 28 + " Positions Fermées " + "─" * 28 + "╮")

                    for pos in recent_closed:
                        if isinstance(pos, dict):
                            asset = pos.get('asset', 'Unknown')
                            size = pos.get('size', 0)
                            entry_price = pos.get('entry_price', 0)
                            exit_price = pos.get('exit_price', 0)
                            pnl = pos.get('pnl', 0)
                            pnl_pct = pos.get('pnl_pct', 0)
                            duration = pos.get('duration_minutes', 0)

                            status_emoji = "🟢" if pnl > 0 else "🔴"

                            self.logger.info(
                                f"│ {status_emoji} {asset}: Size: {size:.2f} | "
                                f"Entry: {entry_price:.4f} | Exit: {exit_price:.4f}"
                                + " " * (80 - len(f"│ {status_emoji} {asset}: Size: {size:.2f} | Entry: {entry_price:.4f} | Exit: {exit_price:.4f}")) + "│"
                            )

                            self.logger.info(
                                f"│   └─ P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) | Duration: {duration}min"
                                + " " * (80 - len(f"│   └─ P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) | Duration: {duration}min")) + "│"
                            )

                    self.logger.info("╰" + "─" * 78 + "╯")

                    # Clear the closed positions after displaying
                    self.closed_positions = []
        except Exception as e:
            self.logger.error(f"Erreur lors de l'affichage des positions fermées: {e}")

    def _on_training_end(self) -> None:
        """Fin de l'entraînement avec résumé complet."""
        try:
            elapsed = time.time() - self.start_time

            self.logger.info("╭" + "─" * 70 + "╮")
            self.logger.info("│" + " " * 20 + "✅ ENTRAÎNEMENT TERMINÉ" + " " * 20 + "│")
            self.logger.info("╰" + "─" * 70 + "╯")

            self.logger.info(f"[TRAINING END] Correlation ID: {self.correlation_id}")
            self.logger.info(f"[TRAINING END] Total steps: {self.num_timesteps:,}")
            self.logger.info(f"[TRAINING END] Duration: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
            self.logger.info(f"[TRAINING END] Episodes completed: {self.episode_count}")

            # Résumé final des performances
            if self.episode_rewards:
                final_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
                max_reward = np.max(self.episode_rewards)
                min_reward = np.min(self.episode_rewards)

                self.logger.info(f"[TRAINING END] Final Mean Reward: {final_reward:.3f}")
                self.logger.info(f"[TRAINING END] Best Episode Reward: {max_reward:.3f}")
                self.logger.info(f"[TRAINING END] Worst Episode Reward: {min_reward:.3f}")

            # Résumé du portfolio
            final_roi = ((self.portfolio_value - self.initial_capital) / self.initial_capital) * 100 if self.initial_capital > 0 else 0
            self.logger.info(f"[TRAINING END] Final Portfolio Value: ${self.portfolio_value:.2f}")
            self.logger.info(f"[TRAINING END] Final ROI: {final_roi:+.2f}%")
            self.logger.info(f"[TRAINING END] Max Drawdown: {self.metrics['max_dd']:.2f}%")
            self.logger.info(f"[TRAINING END] Total Trades: {self.metrics['trades']}")

            # Stats de performance
            avg_steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
            self.logger.info(f"[TRAINING END] Average Speed: {avg_steps_per_sec:.1f} steps/second")

        except Exception as e:
            self.logger.error(f"Erreur lors du résumé final: {e}")


class PPOAgent:
    """
    Agent PPO avec callback d'affichage hiérarchique intégré.
    """

    def __init__(self, env, config: Dict[str, Any]):
        """
        Initialise l'agent PPO avec la configuration donnée.

        Args:
            env: Environnement d'entraînement
            config: Configuration de l'agent
        """
        self.env = env
        self.config = config

        # Configuration du réseau de neurones
        policy_kwargs = {
            "net_arch": config.get("net_arch", [256, 128]),
            "features_dim": config.get("features_dim", 256),
            "activation_fn": torch.nn.ReLU,
            "ortho_init": True
        }

        # Configuration PPO
        ppo_params = config.get("ppo_params", {})
        default_ppo_params = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }

        # Fusionner les paramètres par défaut avec ceux fournis
        final_ppo_params = {**default_ppo_params, **ppo_params}

        # Créer le modèle PPO
        self.model = PPO(
            policy="MultiInputPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=config.get('seed', 42),
            device='auto',
            **final_ppo_params
        )

        # Configurer le logger SB3 si spécifié
        if config.get('enable_sb3_logging', True):
            log_dir = config.get('log_dir', 'logs/sb3')
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            new_logger = sb3_configure(str(log_dir), ["stdout", "csv", "tensorboard"])
            self.model.set_logger(new_logger)

    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 1000,
        initial_capital: float = 20.50,
        callback=None,
        **kwargs
    ):
        """
        Lance l'entraînement avec le callback hiérarchique.

        Args:
            total_timesteps: Nombre total de timesteps
            log_interval: Intervalle d'affichage des logs
            initial_capital: Capital initial du portfolio
            callback: Callback additionnel (optionnel)
            **kwargs: Arguments supplémentaires pour model.learn()
        """
        # Créer le callback hiérarchique
        hierarchical_callback = HierarchicalTrainingDisplayCallback(
            display_freq=log_interval,
            total_timesteps=total_timesteps,
            initial_capital=initial_capital,
            log_file=self.config.get('log_file', 'training_log.txt')
        )

        # Combiner avec d'autres callbacks si fournis
        if callback is not None:
            from stable_baselines3.common.callbacks import CallbackList
            if isinstance(callback, list):
                all_callbacks = [hierarchical_callback] + callback
            else:
                all_callbacks = [hierarchical_callback, callback]
            final_callback = CallbackList(all_callbacks)
        else:
            final_callback = hierarchical_callback

        # Lancer l'entraînement
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=final_callback,
            progress_bar=False,  # Utiliser notre callback personnalisé
            **kwargs
        )

        return self.model

    def predict(self, observation, **kwargs):
        """Prédiction avec le modèle entraîné."""
        return self.model.predict(observation, **kwargs)

    def save(self, path: str):
        """Sauvegarde le modèle."""
        self.model.save(path)

    @classmethod
    def load(cls, path: str, env=None, **kwargs):
        """Charge un modèle sauvegardé."""
        model = PPO.load(path, env=env, **kwargs)

        # Créer une instance de la classe avec le modèle chargé
        agent = cls.__new__(cls)
        agent.model = model
        agent.env = env
        agent.config = {}

        return agent


# Exemple d'utilisation
if __name__ == "__main__":
    """Exemple d'utilisation de l'agent PPO avec callback hiérarchique."""
    import gym
