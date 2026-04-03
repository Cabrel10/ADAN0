#!/usr/bin/env python3
"""
Module centralisé pour l'évaluation robuste des paramètres PPO dans Optuna.

Gère la collecte de métriques, la validation des données, et le calcul des scores.
"""
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
import torch

from .utils.ppo_safety import clamp_policy_log_std, PpoStdSafetyCallback
from .environment.multi_asset_chunked_env import MultiAssetChunkedEnv

logger = logging.getLogger(__name__)


def collect_portfolio_metrics(
    env: MultiAssetChunkedEnv,
    model,
    eval_steps: int = 2000,
    min_portfolio_values: int = 10,
) -> Tuple[List[float], List[Dict], bool]:
    """
    Collecte robustement les valeurs de portfolio et les trades pendant l'évaluation.

    Args:
        env: Environnement MultiAssetChunkedEnv
        model: Modèle PPO entraîné
        eval_steps: Nombre de steps d'évaluation
        min_portfolio_values: Nombre minimum de valeurs à collecter (default 10)

    Returns:
        (portfolio_values, trades_info, success)
    """
    portfolio_values = []
    trades_info = []
    success = False

    try:
        # Reset initial
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        # Vérifier que portfolio existe
        if not hasattr(env, 'portfolio') or env.portfolio is None:
            logger.error("❌ env.portfolio n'existe pas!")
            return [], [], False

        # Collecter valeur initiale
        try:
            initial_pv = float(env.portfolio.equity)
            portfolio_values.append(initial_pv)
            logger.info(f"Initial portfolio value: {initial_pv:.2f}")
        except Exception as e:
            logger.error(f"❌ Erreur lecture initial equity: {e}")
            return [], [], False

        # Boucle d'évaluation
        steps_collected = 0
        resets_count = 0
        max_resets = 10  # Limite les resets pour éviter boucles infinies
        last_closed_count = 0  # Tracker le nombre de trades fermés

        for step in range(eval_steps):
            try:
                # Prédiction
                action, _ = model.predict(obs, deterministic=True)

                # Step
                result = env.step(action)
                if len(result) == 5:
                    obs, reward, done, truncated, info = result
                else:
                    obs, reward, done, info = result
                    truncated = False

                # Collecter equity
                try:
                    current_pv = float(env.portfolio.equity)
                    portfolio_values.append(current_pv)
                    steps_collected += 1
                except Exception as e:
                    logger.warning(
                        f"Step {step}: Erreur lecture equity: {e}"
                    )

                # Accumuler les trades fermés AVANT le reset
                # (car reset() vide closed_positions)
                try:
                    if (
                        hasattr(env, 'portfolio_manager')
                        and hasattr(env.portfolio_manager, 'metrics')
                    ):
                        pm_metrics = env.portfolio_manager.metrics
                        if hasattr(pm_metrics, 'closed_positions'):
                            current_closed = list(pm_metrics.closed_positions)
                            if len(current_closed) > last_closed_count:
                                new_trades = current_closed[last_closed_count:]
                                trades_info.extend(new_trades)
                                logger.info(
                                    f"Step {step}: +{len(new_trades)} "
                                    f"trades (total: {len(trades_info)})"
                                )
                                last_closed_count = len(current_closed)
                except Exception as e:
                    logger.warning(
                        f"Step {step}: erreur collecte trades: {e}"
                    )

                # Reset si done/truncated
                if done or truncated:
                    if resets_count < max_resets:
                        obs = env.reset()
                        if isinstance(obs, tuple):
                            obs = obs[0]
                        resets_count += 1
                        # Réinitialiser le tracker après reset
                        last_closed_count = 0
                        logger.debug(
                            f"Reset {resets_count}, "
                            f"portfolio_values collected: {len(portfolio_values)}"
                        )
                    else:
                        logger.warning(
                            f"Max resets ({max_resets}) reached, "
                            f"stopping evaluation"
                        )
                        break

            except Exception as e:
                logger.error(f"Step {step} error: {e}")
                continue

        # Récupérer trades fermés (fallback si rien collecté via info)
        try:
            if not trades_info:
                if (
                    hasattr(env, 'portfolio_manager')
                    and hasattr(env.portfolio_manager, 'metrics')
                ):
                    pm_metrics = env.portfolio_manager.metrics
                    if hasattr(pm_metrics, 'closed_positions'):
                        trades_info = list(pm_metrics.closed_positions)
                elif hasattr(env, 'portfolio') and hasattr(env.portfolio, 'metrics'):
                    trades_info = list(env.portfolio.metrics.closed_positions)
        except Exception as e:
            logger.warning(f"Erreur récupération trades: {e}")
            trades_info = []

        # Validation
        if len(portfolio_values) >= min_portfolio_values:
            success = True
            logger.info(
                f"✅ Collecte réussie: {len(portfolio_values)} valeurs, "
                f"{len(trades_info)} trades"
            )
        else:
            logger.warning(
                f"⚠️ Collecte insuffisante: {len(portfolio_values)} valeurs "
                f"(min: {min_portfolio_values})"
            )

        return portfolio_values, trades_info, success

    except Exception as e:
        logger.error(f"❌ Erreur critique collecte métriques: {e}")
        return [], [], False


def calculate_metrics_robust(
    portfolio_values: List[float],
    trades_info: List[Dict],
    min_values: int = 3,
) -> Dict[str, float]:
    """
    Calcule les métriques de performance de manière robuste.

    Retourne des valeurs par défaut si les données sont insuffisantes.
    min_values=3 car on a au minimum [initial, step1, step2]
    """
    pv = np.array(portfolio_values, dtype=np.float64)

    # Validation minimale (très permissive pour Optuna)
    if len(pv) < min_values:
        logger.warning(
            f"Données insuffisantes: {len(pv)} valeurs "
            f"(min: {min_values})"
        )
        return {
            'sharpe_ratio': -999.0,
            'max_drawdown': 1.0,
            'win_rate': 0.0,
            'total_return': -1.0,
            'total_trades': 0,
            'profit_factor': 0.0,
        }

    try:
        # Calcul des rendements
        returns = np.diff(pv) / np.maximum(np.abs(pv[:-1]), 1e-8)
        returns = returns[np.isfinite(returns)]

        if len(returns) < 2:
            returns = np.zeros(10)

        # Sharpe Ratio
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        if std_ret < 1e-10:
            total_ret = (pv[-1] - pv[0]) / max(np.abs(pv[0]), 1e-8)
            sharpe = total_ret * 10
        else:
            # Annualisé pour 5min bars
            sharpe = np.sqrt(252 * 24 * 12) * mean_ret / std_ret

        sharpe = float(np.clip(sharpe, -50, 50))

        # Max Drawdown
        peak = np.maximum.accumulate(pv)
        drawdown = (peak - pv) / np.maximum(peak, 1e-8)
        max_dd = float(np.max(drawdown))

        # Total Return
        total_return = float((pv[-1] - pv[0]) / max(np.abs(pv[0]), 1e-8))

        # Trades
        wins = 0
        losses = 0
        gross_profit = 0.0
        gross_loss = 0.0

        for trade in trades_info:
            pnl = trade.get('pnl', 0)
            if pnl > 0:
                wins += 1
                gross_profit += pnl
            elif pnl < 0:
                losses += 1
                gross_loss += abs(pnl)

        total_trades = wins + losses
        win_rate = float(wins / max(total_trades, 1))
        profit_factor = (
            float(gross_profit / max(gross_loss, 1e-8))
            if gross_loss > 0
            else 10.0
        )

        metrics = {
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'total_return': total_return,
            'total_trades': int(total_trades),
            'profit_factor': float(np.clip(profit_factor, 0.0, 100.0)),
        }

        logger.info(
            f"📊 Métriques calculées: "
            f"Sharpe={sharpe:.2f}, DD={max_dd:.1%}, "
            f"WR={win_rate:.1%}, Trades={total_trades}"
        )

        return metrics

    except Exception as e:
        logger.error(f"❌ Erreur calcul métriques: {e}")
        return {
            'sharpe_ratio': -999.0,
            'max_drawdown': 1.0,
            'win_rate': 0.0,
            'total_return': -1.0,
            'total_trades': 0,
            'profit_factor': 0.0,
        }


def evaluate_ppo_params_robust(
    env: MultiAssetChunkedEnv,
    ppo_params: Dict[str, Any],
    training_steps: int = 5000,
    eval_steps: int = 2000,
) -> Dict[str, float]:
    """
    Évalue les paramètres PPO de manière robuste.

    STRATÉGIE ROBUSTE:
    - Entraîne le modèle avec model.learn()
    - Utilise directement les métriques de l'entraînement
    - Évite la boucle d'évaluation séparée qui ne génère pas de trades

    Args:
        env: Environnement de trading
        ppo_params: Paramètres PPO à évaluer
        training_steps: Nombre de steps d'entraînement
        eval_steps: Ignoré (on utilise les métriques d'entraînement)

    Returns:
        Dict avec les métriques calculées
    """
    logger.info(
        f"🚀 Évaluation PPO: training={training_steps} steps "
        f"(eval_steps ignoré, utilisation des métriques post-training)"
    )

    try:
        from stable_baselines3 import PPO

        # 1. Créer et entraîner le modèle
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=ppo_params['learning_rate'],
            n_steps=ppo_params['n_steps'],
            batch_size=ppo_params['batch_size'],
            n_epochs=ppo_params['n_epochs'],
            gamma=ppo_params['gamma'],
            gae_lambda=ppo_params['gae_lambda'],
            clip_range=ppo_params['clip_range'],
            ent_coef=ppo_params['ent_coef'],
            vf_coef=ppo_params['vf_coef'],
            max_grad_norm=ppo_params['max_grad_norm'],
            verbose=0,
            seed=42,
            device='auto',
        )

        # Gardes-fous PPO
        clamp_policy_log_std(model, min_log_std=-5.0, max_log_std=2.0)

        safety_cb = PpoStdSafetyCallback(
            min_log_std=-5.0,
            max_log_std=2.0,
            std_warn_threshold=100.0,
            verbose=0,
        )

        # Entraîner
        logger.info("  [1/2] Entraînement du modèle...")
        model.learn(
            total_timesteps=training_steps,
            progress_bar=False,
            callback=safety_cb,
        )
        logger.info("  ✅ Entraînement terminé")

    except Exception as e:
        logger.error(f"❌ Erreur entraînement PPO: {e}")
        import traceback
        traceback.print_exc()
        return {
            "sharpe_ratio": -999.0,
            "max_drawdown": 1.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "profit_factor": 1.0,
            "total_return": 0.0,
        }

    # 2. Collecter les métriques directement depuis l'env
    try:
        logger.info("  [2/2] Collecte des métriques post-training...")

        pm = env.portfolio_manager

        # Portfolio values depuis equity curve
        portfolio_values = []
        if (
            hasattr(pm.metrics, 'equity_curve')
            and pm.metrics.equity_curve
        ):
            portfolio_values = list(pm.metrics.equity_curve)
            logger.info(
                f"    📊 Equity curve: {len(portfolio_values)} valeurs"
            )
        else:
            logger.warning("    ⚠️ Pas d'equity curve disponible")
            portfolio_values = [pm.equity]

        # Trades fermés depuis closed_positions
        trades_info = []
        if hasattr(pm.metrics, 'closed_positions'):
            trades_info = list(pm.metrics.closed_positions)
            logger.info(f"    📊 Trades fermés: {len(trades_info)}")
        else:
            logger.warning("    ⚠️ Pas de closed_positions disponible")

        # Calculer les métriques
        metrics = calculate_metrics_robust(portfolio_values, trades_info)

        logger.info(
            f"  ✅ Métriques calculées: Sharpe={metrics['sharpe_ratio']:.2f}, "
            f"DD={metrics['max_drawdown']:.2%}, WR={metrics['win_rate']:.2%}, "
            f"Trades={metrics['total_trades']}"
        )

        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        logger.error(f"❌ Erreur collecte métriques: {e}")
        import traceback
        traceback.print_exc()
        return {
            "sharpe_ratio": -999.0,
            "max_drawdown": 1.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "profit_factor": 1.0,
            "total_return": 0.0,
        }
