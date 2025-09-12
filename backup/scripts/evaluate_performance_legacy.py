#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script d'évaluation des performances pour les modèles ADAN entraînés.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import quantstats as qs
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import json
import webbrowser
from typing import Dict, List, Tuple, Optional, Any

# Assurer que le package src est dans le PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '..'))

from src.adan_trading_bot.common.utils import load_config, get_logger
from src.adan_trading_bot.environment.multi_asset_env import MultiAssetEnv
from stable_baselines3 import PPO
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_performance_metrics(env, episode_rewards, episode_capitals, trades_history):
    """
    Calcule les métriques de performance trading.

    Args:
        env: Environnement de trading
        episode_rewards: Liste des récompenses par épisode
        episode_capitals: Liste des valeurs de capital par épisode
        trades_history: Historique des trades

    Returns:
        Dictionnaire contenant les métriques de performance
    """
    if not episode_capitals or len(episode_capitals) < 2:
        return {}

    initial_capital = episode_capitals[0]
    final_capital = episode_capitals[-1]

    # Rendement total
    total_return = (final_capital - initial_capital) / initial_capital * 100

    # Calcul des rendements quotidiens
    returns = np.diff(episode_capitals) / episode_capitals[:-1]

    # Sharpe Ratio (annualisé, 252 jours de trading par an)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # Maximum Drawdown
    peak = np.maximum.accumulate(episode_capitals)
    drawdown = (episode_capitals - peak) / peak * 100
    max_drawdown = np.min(drawdown)

    # Volatilité annualisée
    volatility = np.std(returns) * np.sqrt(365) * 100 if len(returns) > 1 else 0

    # Analyse des trades
    total_trades = len(trades_history) if trades_history else 0
    winning_trades = 0
    losing_trades = 0
    total_pnl = 0

    if trades_history:
        for trade in trades_history:
            pnl = trade.get('pnl', 0)
            total_pnl += pnl
            if pnl > 0:
                winning_trades += 1
            elif pnl < 0:
                losing_trades += 1

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    # Rendement moyen par épisode
    avg_episode_return = np.mean(episode_rewards) if episode_rewards else 0

    return {
        'total_return_percent': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_percent': max_drawdown,
        'volatility_percent': volatility,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate_percent': win_rate,
        'avg_episode_return': avg_episode_return,
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_pnl': total_pnl
    }

def load_test_data(config):
    """Charge les données de test en fonction de la configuration."""
    try:
        project_root = config.get('paths', {}).get('base_project_dir_local', '.')
        data_dir_name = config.get('paths', {}).get('data_dir_name', 'data')
        processed_dir_name = config.get('data', {}).get('processed_data_dir', 'processed')

        # This will reflect the command-line override if one was provided and set in main()
        timeframe_to_load = config.get('data', {}).get('training_timeframe', '1h')
        lot_id = config.get('data', {}).get('lot_id', None)

        logger.info(f"Attempting to load test data for timeframe: {timeframe_to_load}")

        base_merged_path = os.path.join(project_root, data_dir_name, processed_dir_name, 'merged')
        unified_segment = 'unified'

        if lot_id:
            merged_dir = os.path.join(base_merged_path, lot_id, unified_segment)
        else:
            merged_dir = os.path.join(base_merged_path, unified_segment)

        file_name = f"{timeframe_to_load}_test_merged.parquet"
        test_file_path = os.path.join(merged_dir, file_name)

        logger.info(f"Constructed test data path: {test_file_path}")

        if os.path.exists(test_file_path):
            df = pd.read_parquet(test_file_path)
            logger.info(f"✅ Données de test chargées depuis {test_file_path}: {df.shape}")
            return df
        else:
            logger.error(f"❌ Fichier de test introuvable: {test_file_path}")
            # Provide more context if file not found
            if not os.path.exists(merged_dir):
                logger.error(f"  Le répertoire merged/unified ({merged_dir}) n'existe pas.")
                parent_merged_dir = os.path.dirname(merged_dir)
                if os.path.exists(parent_merged_dir):
                    logger.error(f"  Contenu de {parent_merged_dir}: {os.listdir(parent_merged_dir)}")
            else:
                logger.error(f"  Contenu de {merged_dir}: {os.listdir(merged_dir)}")
            return None
    except Exception as e:
        logger.error(f"❌ Erreur chargement données test: {e}", exc_info=True)
        return None

def run_backtest(
    model_path: str,
    config: Dict[str, Any],
    num_episodes: int = 10,
    max_steps_per_episode: int = 1000,
    output_dir: str = 'reports'
) -> Dict[str, Any]:
    """
    Exécute un backtest complet du modèle sur les données de test.

    Args:
        model_path: Chemin vers le modèle entraîné
        config: Configuration du backtest
        num_episodes: Nombre d'épisodes à exécuter
        max_steps_per_episode: Nombre maximum de pas par épisode
        output_dir: Répertoire de sortie pour les rapports

    Returns:
        Dictionnaire contenant les résultats du backtest et les métriques
    """
    logger.info(f"🔍 DÉMARRAGE DU BACKTEST: {model_path}")
    logger.info("=" * 80)

    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Charger les données de test
    test_df = load_test_data(config)
    if test_df is None:
        raise ValueError("Impossible de charger les données de test")

    # Préparer les données pour quantstats
    timestamps = pd.to_datetime(test_df['timestamp'])
    price_data = test_df['close'].values

    # Créer l'environnement de backtest
    try:
        env = MultiAssetEnv(
            data=test_df,
            config=config,
            mode='backtest',
            max_episode_steps_override=max_steps_per_episode
        )
        logger.info(f"✅ Environnement créé avec {len(env.assets)} actifs")
    except Exception as e:
        logger.error(f"❌ Erreur création environnement: {e}")
        raise

    # Charger le modèle
    try:
        model = PPO.load(model_path)
        logger.info(f"✅ Modèle chargé: {model_path}")
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle: {e}")
        raise

    # Initialiser les structures de données pour le suivi
    results = {
        'episode_rewards': [],
        'episode_returns': [],
        'portfolio_values': [],
        'trades': [],
        'timestamps': [],
        'equity_curve': [],
        'daily_returns': pd.Series(dtype=float),
        'metrics': {}
    }

    # Exécuter le backtest
    logger.info(f"🎯 Démarrage du backtest: {num_episodes} épisodes")

    for episode in range(num_episodes):
        try:
            obs, info = env.reset()
            episode_reward = 0
            episode_steps = 0

            # Enregistrer la valeur initiale du portefeuille
            initial_equity = env.portfolio_manager.equity
            results['equity_curve'].append(initial_equity)

            # Exécuter l'épisode
            for step in range(max_steps_per_episode):
                # Prédiction avec le modèle (mode déterministe pour évaluation)
                action, _ = model.predict(obs, deterministic=True)

                # Exécuter l'action et obtenir la nouvelle observation
                obs, reward, terminated, truncated, info = env.step(action)

                # Mettre à jour les métriques
                episode_reward += reward
                episode_steps += 1

                # Enregistrer les informations de trading
                if 'trade_info' in info:
                    trade_info = info['trade_info']
                    trade_info.update({
                        'episode': episode,
                        'step': step,
                        'timestamp': timestamps[step] if step < len(timestamps) else timestamps[-1]
                    })
                    results['trades'].append(trade_info)

                # Enregistrer la valeur du portefeuille à chaque pas
                current_equity = env.portfolio_manager.equity
                results['equity_curve'].append(current_equity)

                if terminated or truncated:
                    break

            # Calculer le rendement de l'épisode
            final_equity = env.portfolio_manager.equity
            episode_return = (final_equity / initial_equity - 1) * 100

            # Enregistrer les résultats de l'épisode
            results['episode_rewards'].append(episode_reward)
            results['episode_returns'].append(episode_return)
            results['portfolio_values'].append(final_equity)

            logger.info(
                f"Episode {episode+1:2d}/{num_episodes}: "
                f"Reward={episode_reward:8.2f}, "
                f"Return={episode_return:6.2f}%, "
                f"Equity=${final_equity:,.2f}, "
                f"Steps={episode_steps}"
            )

        except Exception as e:
            logger.error(f"❌ Erreur lors de l'épisode {episode}: {e}")
            continue

    # Calculer les métriques de performance avec quantstats
    if results['equity_curve']:
        # Créer une série de rendements pour quantstats
        equity_series = pd.Series(
            results['equity_curve'],
            index=pd.date_range(
                start=datetime.now() - timedelta(days=len(results['equity_curve'])),
                periods=len(results['equity_curve']),
                freq='D'
            )
        )

        # Calculer les rendements quotidiens
        returns = equity_series.pct_change().dropna()
        results['daily_returns'] = returns

        # Sauvegarder les données brutes
        results['equity_curve_series'] = equity_series

        # Calculer les métriques avec quantstats
        results['metrics'] = {
            'sharpe_ratio': qs.stats.sharpe(returns, periods=252),
            'sortino_ratio': qs.stats.sortino(returns, periods=252),
            'max_drawdown': qs.stats.max_drawdown(returns),
            'cagr': qs.stats.cagr(returns, periods=252),
            'volatility': qs.stats.volatility(returns, periods=252),
            'calmar_ratio': qs.stats.calmar(returns, periods=252),
            'value_at_risk': qs.stats.value_at_risk(returns),
            'expected_return': returns.mean() * 252,
            'total_return': (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100,
            'win_rate': qs.stats.win_rate(returns) if not returns.empty else 0,
            'profit_factor': qs.stats.profit_factor(returns) if not returns.empty else 0,
            'total_trades': len(results['trades']),
            'winning_trades': sum(1 for t in results['trades'] if t.get('pnl', 0) > 0),
            'losing_trades': sum(1 for t in results['trades'] if t.get('pnl', 0) < 0),
        }

        # Générer le rapport HTML avec quantstats
        generate_quantstats_report(
            returns=returns,
            benchmark=None,  # Vous pourriez ajouter un benchmark ici
            output_dir=output_dir,
            model_name=os.path.basename(model_path).replace('.zip', '')
        )

    return results

def generate_quantstats_report(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    output_dir: str = 'reports',
    model_name: str = 'model',
    title: str = 'ADAN Trading Bot - Rapport de Performance'
) -> str:
    """
    Génère un rapport HTML complet avec quantstats.

    Args:
        returns: Série des rendements du portefeuille
        benchmark: Série des rendements du benchmark (optionnel)
        output_dir: Répertoire de sortie
        model_name: Nom du modèle pour le nom du fichier
        title: Titre du rapport

    Returns:
        Chemin vers le rapport HTML généré
    """
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Nom du fichier de sortie
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f'backtest_report_{model_name}_{timestamp}.html')

    # Configuration de quantstats
    qs.reports.html(
        returns=returns,
        benchmark=benchmark,
        output=report_path,
        title=title,
        # Personnalisation des métriques affichées
        rf=0.0,  # Taux sans risque
        grayscale=False,
        figsize=(12, 8),
        # Désactiver certaines métriques pour alléger le rapport
        display=False,
        compounded=True,
        periods_per_year=252,
        download_filename=report_path,
        template_path=None
    )

    logger.info(f"📊 Rapport de performance généré: {report_path}")
    return report_path

def save_backtest_results(
    results: Dict[str, Any],
    output_dir: str = 'reports',
    model_name: str = 'model'
) -> Dict[str, str]:
    """
    Sauvegarde les résultats du backtest dans des fichiers.

    Args:
        results: Dictionnaire contenant les résultats du backtest
        output_dir: Répertoire de sortie
        model_name: Nom du modèle pour les noms de fichiers

    Returns:
        Dictionnaire avec les chemins des fichiers générés
    """
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Fichiers de sortie
    output_files = {}

    # 1. Sauvegarder les métriques au format JSON
    metrics_file = os.path.join(output_dir, f'metrics_{model_name}_{timestamp}.json')
    with open(metrics_file, 'w') as f:
        json.dump(results.get('metrics', {}), f, indent=2, default=str)
    output_files['metrics'] = metrics_file

    # 2. Sauvegarder l'historique des trades au format CSV
    if results.get('trades'):
        trades_file = os.path.join(output_dir, f'trades_{model_name}_{timestamp}.csv')
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv(trades_file, index=False)
        output_files['trades'] = trades_file

    # 3. Sauvegarder la courbe d'équité au format CSV
    if 'equity_curve_series' in results:
        equity_file = os.path.join(output_dir, f'equity_curve_{model_name}_{timestamp}.csv')
        results['equity_curve_series'].to_csv(equity_file, header=['equity'])
        output_files['equity_curve'] = equity_file

    # 4. Générer un rapport texte récapitulatif
    report_file = os.path.join(output_dir, f'summary_{model_name}_{timestamp}.txt')
    with open(report_file, 'w') as f:
        f.write(generate_summary_report(results, model_name))
    output_files['summary'] = report_file

    return output_files

def generate_summary_report(results: Dict[str, Any], model_name: str) -> str:
    """
    Génère un rapport texte récapitulatif des performances.

    Args:
        results: Dictionnaire contenant les résultats du backtest
        model_name: Nom du modèle

    Returns:
        Chaîne de caractères formatée contenant le rapport
    """
    metrics = results.get('metrics', {})

    report = [
        "=" * 80,
        f"ADAN TRADING BOT - RAPPORT DE PERFORMANCE",
        "=" * 80,
        f"Modèle: {model_name}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 80,
        "MÉTRIQUES DE PERFORMANCE",
        "-" * 80,
        f"Rendement total: {metrics.get('total_return', 0):.2f}%",
        f"Ratio de Sharpe: {metrics.get('sharpe_ratio', 0):.2f}",
        f"Ratio de Sortino: {metrics.get('sortino_ratio', 0):.2f}",
        f"Maximum Drawdown: {abs(metrics.get('max_drawdown', 0) * 100):.2f}%",
        f"Volatilité annualisée: {metrics.get('volatility', 0) * 100:.2f}%",
        f"Rendement annualisé (CAGR): {metrics.get('cagr', 0) * 100:.2f}%",
        f"Ratio Calmar: {metrics.get('calmar_ratio', 0):.2f}",
        f"VaR (95%): {metrics.get('value_at_risk', 0) * 100:.2f}%",
        "-" * 80,
        "STATISTIQUES DE TRADING",
        "-" * 80,
        f"Nombre total de trades: {metrics.get('total_trades', 0)}",
        f"Trades gagnants: {metrics.get('winning_trades', 0)}",
        f"Trades perdants: {metrics.get('losing_trades', 0)}",
        f"Taux de réussite: {metrics.get('win_rate', 0) * 100:.1f}%",
        f"Profit factor: {metrics.get('profit_factor', 0):.2f}",
        "=" * 80
    ]

    return "\n".join(report)

def print_performance_report(metrics, model_path):
    """Affiche un rapport de performance formaté."""

    logger.info("=" * 60)
    logger.info("📊 RAPPORT DE PERFORMANCE")
    logger.info("=" * 60)
    logger.info(f"Modèle évalué: {model_path}")
    logger.info(f"Date d'évaluation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    logger.info("💰 PERFORMANCE FINANCIÈRE")
    logger.info("-" * 30)
    logger.info(f"Capital initial:        ${metrics['initial_capital']:,.0f}")
    logger.info(f"Capital final:          ${metrics['final_capital']:,.0f}")
    logger.info(f"Rendement total:        {metrics['total_return_percent']:+.2f}%")
    logger.info(f"PnL total:              ${metrics['total_pnl']:+,.2f}")
    logger.info("")

    logger.info("📈 MÉTRIQUES DE RISQUE")
    logger.info("-" * 30)
    logger.info(f"Sharpe Ratio:           {metrics['sharpe_ratio']:.3f}")
    logger.info(f"Maximum Drawdown:       {metrics['max_drawdown_percent']:.2f}%")
    logger.info(f"Volatilité annuelle:    {metrics['volatility_percent']:.2f}%")
    logger.info("")

    logger.info("🎯 ANALYSE DES TRADES")
    logger.info("-" * 30)
    logger.info(f"Total trades:           {metrics['total_trades']}")
    logger.info(f"Trades gagnants:        {metrics['winning_trades']}")
    logger.info(f"Trades perdants:        {metrics['losing_trades']}")
    logger.info(f"Taux de réussite:       {metrics['win_rate_percent']:.1f}%")
    logger.info("")

    logger.info("⭐ RENDEMENT PAR ÉPISODE")
    logger.info("-" * 30)
    logger.info(f"Reward moyen:           {metrics['avg_episode_return']:.3f}")
    logger.info("")

    # Classification de la performance
    if metrics['total_return_percent'] > 10:
        performance_grade = "🎉 EXCELLENT"
    elif metrics['total_return_percent'] > 5:
        performance_grade = "✅ BON"
    elif metrics['total_return_percent'] > 0:
        performance_grade = "🟡 MODÉRÉ"
    else:
        performance_grade = "❌ FAIBLE"

    logger.info(f"🏆 ÉVALUATION GLOBALE: {performance_grade}")
    logger.info("=" * 60)

def main():
    """Fonction principale pour exécuter le backtest."""
    parser = argparse.ArgumentParser(
        description='Exécute un backtest complet d\'un modèle ADAN entraîné',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Arguments principaux
    parser.add_argument('--model', type=str, required=True,
                      help='Chemin vers le modèle entraîné')
    parser.add_argument('--config', type=str, default='config/main_config.yaml',
                      help='Chemin vers le fichier de configuration principal')
    parser.add_argument('--output-dir', type=str, default='reports',
                      help='Répertoire de sortie pour les rapports')

    # Paramètres de backtest
    parser.add_argument('--episodes', type=int, default=10,
                      help='Nombre d\'épisodes pour le backtest')
    parser.add_argument('--steps', type=int, default=1000,
                      help='Nombre maximum de pas par épisode')

    # Options de rapport
    parser.add_argument('--open-report', action='store_true',
                      help='Ouvre automatiquement le rapport HTML dans le navigateur')
    parser.add_argument('--no-html', action='store_true',
                      help='Désactive la génération du rapport HTML')
    parser.add_argument('--benchmark', type=str,
                      help='Chemin vers un fichier CSV/Parquet contenant les données de benchmark')

    args = parser.parse_args()

    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

    logger.info("=" * 80)
    logger.info(f"DÉMARRAGE DU BACKTEST - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # Vérifier que le modèle existe
    if not os.path.exists(args.model):
        logger.error(f"Le fichier du modèle n'existe pas: {args.model}")
        return 1

    # Charger la configuration
    try:
        config = load_config(args.config)
        if not config:
            raise ValueError("La configuration est vide")
        logger.info(f"Configuration chargée depuis {args.config}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {e}")
        return 1

    # Charger les données du benchmark si spécifié
    benchmark_data = None
    if args.benchmark and os.path.exists(args.benchmark):
        try:
            if args.benchmark.endswith('.parquet'):
                benchmark_data = pd.read_parquet(args.benchmark)
            else:  # CSV par défaut
                benchmark_data = pd.read_csv(args.benchmark, parse_dates=['date'], index_col='date')
            logger.info(f"Données de benchmark chargées: {args.benchmark}")
        except Exception as e:
            logger.warning(f"Impossible de charger le benchmark: {e}")

    # Exécuter le backtest
    try:
        results = run_backtest(
            model_path=args.model,
            config=config,
            num_episodes=args.episodes,
            max_steps_per_episode=args.steps,
            output_dir=args.output_dir
        )

        if not results:
            raise ValueError("Aucun résultat n'a été généré par le backtest")

        # Sauvegarder les résultats
        output_files = save_backtest_results(
            results=results,
            output_dir=args.output_dir,
            model_name=os.path.basename(args.model).replace('.zip', '')
        )

        # Afficher un résumé
        logger.info("\n" + "=" * 80)
        logger.info("RÉCAPITULATIF DU BACKTEST")
        logger.info("=" * 80)
        logger.info("Modèle: %s", args.model)
        logger.info("Épisodes: %d", args.episodes)
        logger.info("Fichiers générés:")
        for name, path in output_files.items():
            logger.info("  • %s: %s", name, os.path.abspath(path))

        # Affichage du rapport détaillé
        if 'metrics' in results:
            print_performance_report(results['metrics'], args.model)

        # Sauvegarde optionnelle des résultats bruts
        if hasattr(args, 'save_results') and args.save_results:
            results_file = os.path.join(args.output_dir, f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df_results = pd.DataFrame([results['metrics']])
            df_results.to_csv(results_file, index=False)
            logger.info(f"💾 Résultats bruts sauvegardés: {results_file}")

        # Ouvrir le rapport HTML si demandé
        if args.open_report and 'html_report' in output_files:
            webbrowser.open(f'file://{os.path.abspath(output_files["html_report"])}')

        # Code de sortie basé sur la performance
        if 'metrics' in results and results['metrics'].get('total_return', 0) > 0:
            return 0
        return 1

    except Exception as e:
        logger.error("Erreur lors de l'exécution du backtest: %s", str(e), exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
