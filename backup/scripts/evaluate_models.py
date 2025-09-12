#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Évaluation comparative des modèles avant et après apprentissage en ligne.

Ce script compare les performances d'un modèle de base et d'un modèle affiné
en ligne sur un ensemble de test commun.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Configuration
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
EVAL_DIR = REPORTS_DIR / "evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Configuration du style
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

# Import de l'environnement personnalisé
import sys
sys.path.append(str(BASE_DIR / "src"))
from adan_trading_bot.environment.multi_asset_env import AdanTradingEnv
from adan_trading_bot.config_loader import load_full_config

def load_model(model_path, env):
    """Charge un modèle PPO sauvegardé."""
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle non trouvé : {model_path}")

    print(f"Chargement du modèle : {model_path}")
    return PPO.load(model_path, env=env)

def evaluate_model(model, env, n_eval_episodes=10):
    """Évalue un modèle sur l'environnement donné."""
    print(f"Évaluation sur {n_eval_episodes} épisodes...")

    # Réinitialisation de l'environnement
    obs = env.reset()

    # Variables de suivi
    episode_rewards = []
    episode_lengths = []
    portfolio_values = []

    for _ in range(n_eval_episodes):
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            # Suivi des valeurs du portefeuille
            if 'portfolio' in info[0]:
                portfolio_values.append(info[0]['portfolio']['total_value'])

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        obs = env.reset()

    # Calcul des métriques
    metrics = {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'mean_episode_length': float(np.mean(episode_lengths)),
        'final_portfolio_value': float(portfolio_values[-1]) if portfolio_values else 0.0,
        'max_drawdown': calculate_max_drawdown(portfolio_values) if portfolio_values else 0.0,
        'sharpe_ratio': calculate_sharpe_ratio(portfolio_values) if portfolio_values else 0.0
    }

    return metrics, portfolio_values

def calculate_max_drawdown(portfolio_values):
    """Calcule le drawdown maximum."""
    if not portfolio_values:
        return 0.0

    peak = portfolio_values[0]
    max_drawdown = 0.0

    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return max_drawdown

def calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0.0):
    """Calcule le ratio de Sharpe annualisé."""
    if len(portfolio_returns) < 2:
        return 0.0

    returns = np.diff(portfolio_returns) / portfolio_returns[:-1]
    excess_returns = returns - risk_free_rate / 252  # 252 jours de trading par an

    if np.std(excess_returns) == 0:
        return 0.0

    return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

def plot_comparison(metrics_baseline, metrics_online, output_dir):
    """Génère des graphiques comparatifs."""
    # Préparation des données
    models = ['Baseline', 'Online']

    # Métriques à comparer
    comparison_metrics = {
        'mean_reward': 'Récompense Moyenne',
        'final_portfolio_value': 'Valeur Finale du Portefeuille',
        'max_drawdown': 'Drawdown Maximum',
        'sharpe_ratio': 'Ratio de Sharpe'
    }

    # Création des graphiques
    for metric, title in comparison_metrics.items():
        plt.figure(figsize=(10, 6))
        values = [metrics_baseline[metric], metrics_online[metric]]

        # Ajustement de l'échelle pour le drawdown (on veut des valeurs positives)
        if metric == 'max_drawdown':
            values = [abs(v) for v in values]

        bars = plt.bar(models, values, color=['#1f77b4', '#ff7f0e'])

        # Ajout des valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom')

        plt.title(f'Comparaison des Modèles - {title}')
        plt.ylabel(metric)
        plt.tight_layout()

        # Sauvegarde du graphique
        output_path = output_dir / f'comparison_{metric}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Graphique sauvegardé : {output_path}")

def main():
    # Chargement de la configuration
    config = load_full_config()

    # Création de l'environnement d'évaluation
    env = DummyVecEnv([lambda: Monitor(AdanTradingEnv(config))])

    # Chemins des modèles
    baseline_model_path = MODELS_DIR / "rl_agents" / "adan_ppo_v1.0.zip"
    online_model_path = MODELS_DIR / "online" / "adan_ppo_v1.1.zip"

    try:
        # Évaluation du modèle de base
        print("=== Évaluation du Modèle de Base ===")
        baseline_model = load_model(baseline_model_path, env)
        metrics_baseline, _ = evaluate_model(baseline_model, env)

        # Évaluation du modèle avec apprentissage en ligne
        print("\n=== Évaluation du Modèle avec Apprentissage en Ligne ===")
        online_model = load_model(online_model_path, env)
        metrics_online, portfolio_values = evaluate_model(online_model, env)

        # Affichage des résultats
        print("\n=== Résultats de l'Évaluation ===")
        print(f"{'Métrique':<30} {'Base':<15} {'En Ligne':<15} 'Différence'")
        print("-" * 60)

        for metric in metrics_baseline.keys():
            base_val = metrics_baseline[metric]
            online_val = metrics_online[metric]
            diff = ((online_val - base_val) / base_val * 100) if base_val != 0 else float('inf')
            print(f"{metric:<30} {base_val:>10.4f} {online_val:>15.4f} {diff:>15.2f}%")

        # Génération des graphiques
        print("\nGénération des graphiques comparatifs...")
        plot_comparison(metrics_baseline, metrics_online, EVAL_DIR)

        # Sauvegarde des résultats
        results = {
            'timestamp': datetime.now().isoformat(),
            'baseline_metrics': metrics_baseline,
            'online_metrics': metrics_online,
            'portfolio_values': portfolio_values
        }

        results_file = EVAL_DIR / f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nÉvaluation terminée. Résultats sauvegardés dans : {results_file}")

    except Exception as e:
        print(f"Erreur lors de l'évaluation : {str(e)}")
        raise

    finally:
        env.close()

if __name__ == "__main__":
    main()
