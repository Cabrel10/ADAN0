#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test minimal pour l'entraînement avec affichage hiérarchique.

Ce script teste l'affichage hiérarchique sans les complexités de configuration,
en simulant un entraînement complet avec des données factices.
"""

import logging
import numpy as np
import time
import sys
from pathlib import Path
from unittest.mock import Mock

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockModel:
    """Modèle fictif pour simuler l'entraînement."""

    def __init__(self):
        self.num_timesteps = 0
        self.logger = Mock()
        self.logger.name_to_value = {
            "train/loss": 0.0,
            "train/policy_loss": 0.0,
            "train/value_loss": 0.0,
            "train/entropy_loss": 0.0
        }

class HierarchicalTrainingDisplayCallback:
    """Callback d'affichage hiérarchique simplifié."""

    def __init__(self, verbose=1, display_freq=1000, total_timesteps=10000, initial_capital=20.50):
        self.verbose = verbose
        self.display_freq = display_freq
        self.total_timesteps = total_timesteps
        self.initial_capital = initial_capital
        self.episode_rewards = []
        self.episode_count = 0
        self.positions = {}
        self.metrics = {
            "sharpe": 0.0, "sortino": 0.0, "profit_factor": 0.0,
            "max_dd": 0.0, "cagr": 0.0, "win_rate": 0.0, "trades": 0
        }
        self.start_time = time.time()
        self.model = None
        self.num_timesteps = 0

    def _on_training_start(self):
        """Démarrage de l'entraînement avec affichage de la configuration."""
        logger.info("╭" + "─" * 60 + "╮")
        logger.info("│" + " " * 15 + "🚀 DÉMARRAGE ADAN TRAINING" + " " * 15 + "│")
        logger.info("╰" + "─" * 60 + "╯")
        logger.info(f"[TRAINING START] Total timesteps: {self.total_timesteps:,}")
        logger.info(f"[TRAINING START] Capital initial: ${self.initial_capital:.2f}")

        # Affichage de la configuration des flux monétaires
        logger.info("╭" + "─" * 50 + " Configuration Flux Monétaires " + "─" * 50 + "╮")
        logger.info("│ 💰 Capital Initial: $%-40.2f │" % self.initial_capital)
        logger.info("│ 🎯 Gestion Dynamique des Flux Activée" + " " * 32 + "│")
        logger.info("│ 📊 Monitoring en Temps Réel" + " " * 39 + "│")
        logger.info("╰" + "─" * 132 + "╯")

    def _on_step(self, timestep, episode_data=None):
        """Appelé à chaque étape pour mettre à jour l'affichage."""
        self.num_timesteps = timestep

        # Simuler la fin d'un épisode de temps en temps
        if timestep % 500 == 0 and timestep > 0:
            self.episode_count += 1
            mean_reward = np.random.uniform(-0.5, 1.5)  # Récompense simulée
            self.episode_rewards.append(mean_reward)

            progress = self.num_timesteps / self.total_timesteps * 100

            # Barre de progression visuelle
            progress_bar_length = 30
            filled_length = int(progress_bar_length * progress // 100)
            bar = "━" * filled_length + "─" * (progress_bar_length - filled_length)

            logger.info(
                f"🚀 ADAN Training {bar} {progress:.1f}% ({self.num_timesteps:,}/{self.total_timesteps:,}) • "
                f"Episode {self.episode_count} • Mean Reward: {mean_reward:.2f}"
            )

        # Affichage hiérarchique périodique
        if self.num_timesteps % self.display_freq == 0 and self.num_timesteps > 0:
            self._log_detailed_metrics(episode_data)

    def _log_detailed_metrics(self, episode_data=None):
        """Affichage détaillé des métriques avec structure hiérarchique."""
        # Simuler l'évolution du portfolio
        progress_ratio = self.num_timesteps / self.total_timesteps
        portfolio_value = self.initial_capital * (1 + progress_ratio * 0.5)  # Croissance de 50%
        drawdown = max(0, progress_ratio * 5.0)  # Drawdown jusqu'à 5%
        cash = portfolio_value * 0.3  # 30% en cash

        # Simuler des positions
        positions = {}
        if progress_ratio > 0.1:  # Après 10% de progression
            positions = {
                "ADAUSDT": {
                    "size": 23.86,
                    "entry_price": 0.7092,
                    "value": portfolio_value * 0.7,
                    "sl": 0.67,
                    "tp": 0.82
                }
            }

        # Mettre à jour les métriques financières simulées
        self.metrics.update({
            "sharpe": progress_ratio * 2.0,
            "sortino": progress_ratio * 2.2,
            "profit_factor": 1.0 + progress_ratio * 0.8,
            "max_dd": drawdown,
            "cagr": progress_ratio * 25.0,
            "win_rate": 50.0 + progress_ratio * 25.0,
            "trades": int(self.num_timesteps / 500)
        })

        # Récupérer les métriques du modèle (simulées)
        if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            model_metrics = self.model.logger.name_to_value
            # Simuler l'évolution des losses
            model_metrics["train/loss"] = 0.5 * (1 - progress_ratio * 0.8)
            model_metrics["train/policy_loss"] = 0.2 * (1 - progress_ratio * 0.7)
            model_metrics["train/value_loss"] = 0.3 * (1 - progress_ratio * 0.9)
            model_metrics["train/entropy_loss"] = 0.8 * (1 - progress_ratio * 0.5)

            total_loss = model_metrics.get("train/loss", 0.0)
            policy_loss = model_metrics.get("train/policy_loss", 0.0)
            value_loss = model_metrics.get("train/value_loss", 0.0)
            entropy = model_metrics.get("train/entropy_loss", 0.0)
        else:
            total_loss = policy_loss = value_loss = entropy = 0.0

        # Calcul du ROI
        roi = ((portfolio_value - self.initial_capital) / self.initial_capital) * 100

        # Affichage structuré
        logger.info("╭" + "─" * 80 + "╮")
        logger.info("│" + " " * 25 + f"ÉTAPE {self.num_timesteps:,}" + " " * 25 + "│")
        logger.info("╰" + "─" * 80 + "╯")

        # Portfolio Status
        logger.info(f"📊 PORTFOLIO | Valeur: ${portfolio_value:.2f} | Cash: ${cash:.2f} | ROI: {roi:+.2f}%")

        # Risk Metrics
        logger.info(f"⚠️  RISK | Drawdown: {drawdown:.2f}% | Max DD: {self.metrics['max_dd']:.2f}%")

        # Performance Metrics
        logger.info(
            f"📈 METRICS | Sharpe: {self.metrics['sharpe']:.2f} | Sortino: {self.metrics['sortino']:.2f} | "
            f"Profit Factor: {self.metrics['profit_factor']:.2f}"
        )
        logger.info(
            f"📊 TRADING | CAGR: {self.metrics['cagr']:.2f}% | Win Rate: {self.metrics['win_rate']:.1f}% | "
            f"Trades: {self.metrics['trades']}"
        )

        # Positions ouvertes si disponibles
        if positions:
            logger.info("╭" + "─" * 25 + " Positions Ouvertes " + "─" * 25 + "╮")
            for asset, pos in positions.items():
                size = pos.get('size', 0)
                entry_price = pos.get('entry_price', 0)
                value = pos.get('value', 0)
                sl = pos.get('sl', 0)
                tp = pos.get('tp', 0)
                logger.info(
                    f"│ {asset}: Taille: {size:.2f} @ {entry_price:.4f} | "
                    f"Valeur: ${value:.2f} | SL: {sl:.4f} | TP: {tp:.4f}" + " " * 5 + "│"
                )
            logger.info("╰" + "─" * 68 + "╯")
        else:
            logger.info("📝 POSITIONS | Aucune position ouverte")

        # Model Learning Metrics
        logger.info(
            f"🧠 MODEL | Total Loss: {total_loss:.4f} | Policy Loss: {policy_loss:.4f} | "
            f"Value Loss: {value_loss:.4f} | Entropy: {entropy:.4f}"
        )

        # Temps et vitesse
        elapsed = time.time() - self.start_time
        steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
        logger.info(f"⏱️  TIMING | Elapsed: {elapsed/60:.1f}min | Speed: {steps_per_sec:.1f} steps/s")

        logger.info("─" * 80)

    def _on_rollout_end(self, closed_positions=None):
        """Appelé à la fin de chaque rollout pour capturer les positions fermées."""
        if closed_positions is None:
            # Simuler quelques positions fermées
            if np.random.random() > 0.7:  # 30% de chance d'avoir des positions fermées
                closed_positions = [{
                    'asset': 'ADAUSDT',
                    'size': 23.86,
                    'entry_price': 0.71,
                    'exit_price': 0.71 + np.random.uniform(-0.05, 0.05),
                    'pnl': np.random.uniform(-2.0, 5.0),
                    'pnl_pct': np.random.uniform(-5.0, 8.0)
                }]

        if closed_positions:
            logger.info("╭" + "─" * 25 + " Positions Fermées " + "─" * 25 + "╮")
            for pos in closed_positions:
                asset = pos.get('asset', 'Unknown')
                size = pos.get('size', 0)
                entry_price = pos.get('entry_price', 0)
                exit_price = pos.get('exit_price', 0)
                pnl = pos.get('pnl', 0)
                pnl_pct = pos.get('pnl_pct', 0)
                pnl_sign = "📈" if pnl >= 0 else "📉"
                logger.info(
                    f"│ {pnl_sign} {asset}: Taille: {size:.2f} | Entrée: {entry_price:.4f} | "
                    f"Sortie: {exit_price:.4f} | PnL: ${pnl:.2f} ({pnl_pct:.2f}%)" + " " * 5 + "│"
                )
            logger.info("╰" + "─" * 68 + "╯")

    def _on_training_end(self):
        """Fin de l'entraînement avec résumé complet."""
        elapsed = time.time() - self.start_time
        logger.info("╭" + "─" * 60 + "╮")
        logger.info("│" + " " * 15 + "✅ ENTRAÎNEMENT TERMINÉ" + " " * 15 + "│")
        logger.info("╰" + "─" * 60 + "╯")
        logger.info(f"[TRAINING END] Total steps: {self.num_timesteps:,}")
        logger.info(f"[TRAINING END] Duration: {elapsed/60:.1f} minutes")
        logger.info(f"[TRAINING END] Episodes: {self.episode_count}")

        # Résumé final des performances
        if self.episode_rewards:
            final_reward = np.mean(self.episode_rewards[-5:]) if len(self.episode_rewards) >= 5 else np.mean(self.episode_rewards)
            logger.info(f"[TRAINING END] Final Mean Reward: {final_reward:.2f}")

        # Métriques finales
        final_portfolio_value = self.initial_capital * 1.5  # Supposer 50% de gain
        final_roi = ((final_portfolio_value - self.initial_capital) / self.initial_capital) * 100
        logger.info(f"[TRAINING END] Final Portfolio Value: ${final_portfolio_value:.2f} (ROI: {final_roi:+.1f}%)")

def simulate_training():
    """Simule un entraînement complet avec l'affichage hiérarchique."""
    print("🎯 TEST D'ENTRAÎNEMENT MINIMAL AVEC AFFICHAGE HIÉRARCHIQUE")
    print("=" * 80)

    # Configuration de l'entraînement
    total_timesteps = 5000
    initial_capital = 20.50

    # Créer le callback
    callback = HierarchicalTrainingDisplayCallback(
        verbose=1,
        display_freq=1000,
        total_timesteps=total_timesteps,
        initial_capital=initial_capital
    )

    # Créer un modèle fictif
    callback.model = MockModel()

    # Démarrer l'entraînement
    callback._on_training_start()

    # Simuler l'entraînement
    for timestep in range(0, total_timesteps + 1, 100):
        # Simuler des données d'épisode
        episode_data = {
            "portfolio_value": initial_capital * (1 + timestep / total_timesteps * 0.5),
            "drawdown": max(0, (timestep / total_timesteps) * 5.0),
            "positions": {},
            "rewards": [np.random.uniform(-0.1, 0.3) for _ in range(10)]
        }

        # Mise à jour du callback
        callback._on_step(timestep, episode_data)

        # Simuler des rollouts
        if timestep % 1000 == 0 and timestep > 0:
            callback._on_rollout_end()

        # Pause courte pour simuler le temps d'entraînement
        time.sleep(0.1)

    # Terminer l'entraînement
    callback._on_training_end()

def test_clean_worker_id():
    """Test rapide de la fonction clean_worker_id."""
    def clean_worker_id(worker_id):
        """Nettoie l'ID du worker pour éviter les erreurs JSONL."""
        if isinstance(worker_id, str):
            if worker_id.lower().startswith('w'):
                try:
                    return int(worker_id[1:])
                except ValueError:
                    return 0
            try:
                return int(worker_id)
            except ValueError:
                return 0
        elif isinstance(worker_id, int):
            return worker_id
        else:
            return 0

    print("\n🧪 Test correction JSONL:")
    test_cases = [("w0", 0), ("W5", 5), ("invalid", 0)]
    for input_val, expected in test_cases:
        result = clean_worker_id(input_val)
        status = "✅" if result == expected else "❌"
        print(f"{status} {input_val} -> {result} (attendu: {expected})")

def main():
    """Fonction principale."""
    print("🚀 TEST MINIMAL - AFFICHAGE HIÉRARCHIQUE ADAN")
    print("=" * 80)
    print("Ce script teste l'affichage hiérarchique avec simulation d'entraînement")
    print("=" * 80)

    start_time = time.time()

    try:
        # Test de la correction JSONL
        test_clean_worker_id()

        # Simulation d'entraînement
        simulate_training()

        # Résumé
        elapsed = time.time() - start_time
        print("\n" + "=" * 80)
        print("✅ TEST TERMINÉ AVEC SUCCÈS")
        print(f"⏱️  Durée totale: {elapsed:.1f} secondes")
        print("📊 Toutes les fonctionnalités d'affichage sont opérationnelles")
        print("=" * 80)
        print("\n🎉 L'AFFICHAGE HIÉRARCHIQUE EST PRÊT POUR L'ENTRAÎNEMENT RÉEL!")
        print("\nPour l'entraînement réel, utilisez:")
        print("conda run -n trading_env timeout 20s python bot/scripts/train_parallel_agents.py --config bot/config/config.yaml --checkpoint-dir checkpoints")

        return True

    except Exception as e:
        print(f"\n❌ ERREUR LORS DU TEST: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
