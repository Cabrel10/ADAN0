#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script d'évaluation final optimisé pour ADAN.
Évalue les performances d'un modèle entraîné avec métriques détaillées.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adan_trading_bot.common.utils import get_path, load_config
from src.adan_trading_bot.common.custom_logger import setup_logging
from src.adan_trading_bot.data_processing.feature_engineer import prepare_data_pipeline
from src.adan_trading_bot.environment.multi_asset_env import MultiAssetEnv
from stable_baselines3 import PPO

def calculate_performance_metrics(history, initial_capital):
    """
    Calcule les métriques de performance détaillées.

    Args:
        history: Historique des trades
        initial_capital: Capital initial

    Returns:
        dict: Métriques de performance
    """
    if not history or len(history) == 0:
        return {}

    # Convertir en DataFrame pour faciliter les calculs
    df = pd.DataFrame(history)

    # Calculs de base
    final_capital = df['capital'].iloc[-1] if 'capital' in df.columns else initial_capital
    total_return = (final_capital - initial_capital) / initial_capital * 100

    # Calculs avancés
    portfolio_values = df['portfolio_value'].values if 'portfolio_value' in df.columns else [initial_capital] * len(df)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # Sharpe Ratio (annualisé, en supposant 525600 minutes par an pour 1m timeframe)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(525600)
    else:
        sharpe_ratio = 0

    # Maximum Drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak * 100
    max_drawdown = np.min(drawdown)

    # Win Rate (pour les trades avec reward > 0)
    rewards = df['reward'].values if 'reward' in df.columns else []
    positive_rewards = [r for r in rewards if r > 0]
    win_rate = len(positive_rewards) / len(rewards) * 100 if len(rewards) > 0 else 0

    # Durée moyenne des épisodes
    episodes = df.groupby(df.index // 1000)  # Approximation
    avg_episode_length = len(df) / len(episodes) if len(episodes) > 0 else len(df)

    return {
        'total_return_pct': total_return,
        'final_capital': final_capital,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown,
        'win_rate_pct': win_rate,
        'avg_episode_length': avg_episode_length,
        'total_steps': len(df),
        'avg_reward': np.mean(rewards) if len(rewards) > 0 else 0,
        'cumulative_reward': df['cumulative_reward'].iloc[-1] if 'cumulative_reward' in df.columns else 0
    }

def classify_performance(metrics):
    """
    Classifie la performance du modèle.

    Args:
        metrics: Métriques de performance

    Returns:
        str: Classification (Excellent, Bon, Acceptable, Problématique)
    """
    total_return = metrics.get('total_return_pct', 0)
    sharpe_ratio = metrics.get('sharpe_ratio', 0)
    max_drawdown = abs(metrics.get('max_drawdown_pct', 100))
    win_rate = metrics.get('win_rate_pct', 0)

    # Critères de classification
    if total_return > 15 and sharpe_ratio > 1.5 and max_drawdown < 10 and win_rate > 60:
        return "🏆 EXCELLENT"
    elif total_return > 8 and sharpe_ratio > 1.0 and max_drawdown < 20 and win_rate > 50:
        return "🥈 BON"
    elif total_return > 2 and sharpe_ratio > 0.5 and max_drawdown < 35 and win_rate > 40:
        return "🥉 ACCEPTABLE"
    else:
        return "⚠️ PROBLÉMATIQUE"

def main():
    """
    Fonction principale d'évaluation.
    """
    parser = argparse.ArgumentParser(description='ADAN - Évaluation Final des Performances')

    # Paramètres principaux
    parser.add_argument('--model_path', type=str, required=True,
                        help='Chemin vers le modèle à évaluer (.zip)')
    parser.add_argument('--profile', type=str, default='cpu',
                        choices=['cpu', 'gpu'],
                        help='Profil de configuration à utiliser')

    # Paramètres d'évaluation
    parser.add_argument('--episodes', type=int, default=10,
                        help='Nombre d\'épisodes d\'évaluation (défaut: 10)')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Maximum steps par épisode (défaut: 1000)')
    parser.add_argument('--initial_capital', type=float, default=15000,
                        help='Capital initial pour l\'évaluation (défaut: 15000)')

    # Options de sortie
    parser.add_argument('--save_report', action='store_true',
                        help='Sauvegarder le rapport détaillé')
    parser.add_argument('--verbose', action='store_true',
                        help='Mode verbose avec logs détaillés')

    args = parser.parse_args()

    # Configuration des logs
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logging('config/logging_config.yaml', level=log_level)

    # Affichage de démarrage
    print("📊 ADAN - Évaluation des Performances")
    print("=" * 50)
    print(f"🤖 Modèle: {args.model_path}")
    print(f"📋 Profil: {args.profile.upper()}")
    print(f"🎯 Épisodes: {args.episodes}")
    print(f"💰 Capital initial: ${args.initial_capital:,.2f}")
    print("=" * 50)

    # Vérifier l'existence du modèle
    if not os.path.exists(args.model_path):
        logger.error(f"❌ Modèle non trouvé: {args.model_path}")
        sys.exit(1)

    # Charger les configurations
    config_paths = {
        'main': 'config/main_config.yaml',
        'data': f'config/data_config_{args.profile}.yaml',
        'environment': 'config/environment_config.yaml',
        'agent': f'config/agent_config_{args.profile}.yaml'
    }

    try:
        # Charger le modèle
        logger.info(f"🔄 Chargement du modèle: {args.model_path}")
        model = PPO.load(args.model_path)

        # Préparer l'environnement de test
        logger.info("🔄 Préparation de l'environnement de test...")

        # Charger les données de test
        config = {}
        for key, path in config_paths.items():
            config[key] = load_config(path)

        # Préparer les données
        df_test, scaler, encoder = prepare_data_pipeline(
            config['data'],
            split='test',
            scaler=None,
            encoder=None
        )

        logger.info(f"📈 Données de test chargées: {df_test.shape}")

        # Créer l'environnement
        env = MultiAssetEnv(
            df_received=df_test,
            config=config,
            scaler=scaler,
            encoder=encoder,
            max_episode_steps_override=args.max_steps
        )

        # Override du capital initial
        env.initial_capital = args.initial_capital

        # Évaluation
        logger.info(f"🎯 Début de l'évaluation ({args.episodes} épisodes)...")

        all_rewards = []
        all_history = []
        episode_results = []

        for episode in range(args.episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False

            while not done and episode_steps < args.max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_steps += 1

                if done or truncated:
                    break

            all_rewards.append(episode_reward)
            all_history.extend(env.history)

            final_portfolio = info.get('portfolio_value', args.initial_capital)
            return_pct = (final_portfolio - args.initial_capital) / args.initial_capital * 100

            episode_results.append({
                'episode': episode + 1,
                'reward': episode_reward,
                'steps': episode_steps,
                'final_portfolio': final_portfolio,
                'return_pct': return_pct
            })

            print(f"📈 Épisode {episode+1}/{args.episodes}: "
                  f"Reward={episode_reward:.4f}, "
                  f"Steps={episode_steps}, "
                  f"Return={return_pct:.2f}%")

        # Calcul des métriques globales
        metrics = calculate_performance_metrics(all_history, args.initial_capital)
        classification = classify_performance(metrics)

        # Affichage des résultats
        print("\n" + "=" * 60)
        print("📊 RÉSULTATS DE L'ÉVALUATION")
        print("=" * 60)

        print(f"\n🎯 PERFORMANCE GLOBALE: {classification}")
        print("-" * 40)
        print(f"💰 Rendement Total: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"💎 Capital Final: ${metrics.get('final_capital', args.initial_capital):,.2f}")
        print(f"📈 Ratio de Sharpe: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"📉 Drawdown Max: {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"🎯 Taux de Victoire: {metrics.get('win_rate_pct', 0):.1f}%")
        print(f"🔄 Récompense Moyenne: {metrics.get('avg_reward', 0):.4f}")

        print(f"\n📊 STATISTIQUES D'ÉPISODES:")
        print("-" * 40)
        rewards_array = np.array(all_rewards)
        print(f"🏆 Meilleure Récompense: {np.max(rewards_array):.4f}")
        print(f"📉 Pire Récompense: {np.min(rewards_array):.4f}")
        print(f"📊 Récompense Moyenne: {np.mean(rewards_array):.4f}")
        print(f"📏 Écart-Type: {np.std(rewards_array):.4f}")
        print(f"⏱️  Durée Moyenne d'Épisode: {metrics.get('avg_episode_length', 0):.0f} steps")

        # Sauvegarde du rapport si demandé
        if args.save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = os.path.basename(args.model_path).replace('.zip', '')
            report_path = f"reports/evaluation_{model_name}_{timestamp}.json"

            os.makedirs("reports", exist_ok=True)

            report_data = {
                'model_path': args.model_path,
                'evaluation_date': timestamp,
                'parameters': vars(args),
                'metrics': metrics,
                'classification': classification,
                'episode_results': episode_results,
                'episode_stats': {
                    'mean_reward': float(np.mean(rewards_array)),
                    'std_reward': float(np.std(rewards_array)),
                    'min_reward': float(np.min(rewards_array)),
                    'max_reward': float(np.max(rewards_array))
                }
            }

            import json
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)

            print(f"\n💾 Rapport sauvegardé: {report_path}")

        print("\n" + "=" * 60)

        # Recommandations
        print("💡 RECOMMANDATIONS:")
        if "EXCELLENT" in classification:
            print("✅ Modèle prêt pour le trading en production!")
            print("🚀 Considérez un entraînement plus long pour optimiser davantage.")
        elif "BON" in classification:
            print("✅ Modèle acceptable pour le trading prudent.")
            print("🔧 Ajustez les paramètres d'entraînement pour améliorer.")
        elif "ACCEPTABLE" in classification:
            print("⚠️ Modèle nécessite des améliorations avant production.")
            print("🔧 Augmentez le nombre de timesteps d'entraînement.")
        else:
            print("❌ Modèle non recommandé pour le trading.")
            print("🔄 Re-entraînement nécessaire avec paramètres optimisés.")

    except Exception as e:
        logger.error(f"❌ Erreur pendant l'évaluation: {str(e)}")
        print(f"\n💥 ERREUR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
