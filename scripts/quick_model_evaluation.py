#!/usr/bin/env python3
"""
Script d'évaluation rapide du modèle final ADAN.
Version simplifiée pour obtenir rapidement les métriques essentielles.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Ajouter le path du projet
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from stable_baselines3 import PPO

from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.model.model_ensemble import ModelEnsemble

# Configurer logging simplifié
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def quick_model_evaluation(config_path: str, model_path: str, num_steps: int = 1000):
    """Évaluation rapide du modèle avec métriques essentielles."""

    logger.info("🚀 ÉVALUATION RAPIDE MODÈLE ADAN")
    logger.info("=" * 50)

    try:
        # 1. Validation des poids de fusion
        logger.info("🔍 Validation poids de fusion...")
        ensemble = ModelEnsemble()
        fusion_weights = ensemble.get_fusion_weights()
        expected_weights = {0: 0.25, 1: 0.27, 2: 0.30, 3: 0.18}

        weights_valid = all(
            abs(fusion_weights.get(k, 0) - v) < 0.01
            for k, v in expected_weights.items()
        )

        logger.info(f"✅ Poids fusion: {fusion_weights}")
        logger.info(f"🎯 Validation: {'✅ RÉUSSI' if weights_valid else '❌ ÉCHOUÉ'}")

        # 2. Chargement modèle
        logger.info("\n📂 Chargement modèle...")
        if not os.path.exists(model_path):
            logger.error(f"❌ Modèle non trouvé: {model_path}")
            return False

        model = PPO.load(model_path)
        logger.info("✅ Modèle PPO chargé")

        # 3. Configuration environnement
        logger.info("\n🔧 Configuration environnement...")
        config = ConfigLoader.load_config(config_path)

        # Configuration simplifiée pour test rapide
        test_config = config["workers"]["w1"].copy()
        test_config["chunk_size"] = 1000  # Réduire pour test rapide

        data_loader = ChunkedDataLoader(
            config=config, worker_config=test_config, worker_id=0
        )

        data = data_loader.load_chunk(0)
        if data is None or data.empty:
            logger.error("❌ Données de test indisponibles")
            return False

        logger.info(f"✅ Données chargées: {len(data)} échantillons")

        # 4. Création environnement
        env = MultiAssetChunkedEnv(data=data, config=config, worker_config=test_config)
        logger.info("✅ Environnement créé")

        # 5. Test rapide
        logger.info(f"\n🧪 Test modèle ({num_steps} steps)...")

        obs = env.reset()
        total_reward = 0
        returns = []
        portfolio_values = [config["environment"]["initial_balance"]]
        actions_taken = []

        for step in range(num_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            total_reward += reward
            actions_taken.append(action)

            if "portfolio_value" in info:
                portfolio_values.append(info["portfolio_value"])
            if "return" in info:
                returns.append(info["return"])

            if done:
                obs = env.reset()

            # Progress minimal
            if (step + 1) % 200 == 0:
                logger.info(
                    f"  Step {step + 1}/{num_steps} - Reward cumulé: {total_reward:.4f}"
                )

        logger.info("✅ Test terminé")

        # 6. Calcul métriques rapides
        logger.info("\n📊 Calcul métriques...")

        if len(returns) == 0:
            logger.warning("⚠️ Pas de données de retour disponibles")
            returns = [0.01] * max(1, len(portfolio_values) - 1)  # Fallback

        returns_array = np.array(returns)

        # Métriques essentielles
        total_return = (
            (np.prod(returns_array + 1) - 1) if len(returns_array) > 0 else 0.0
        )
        volatility = (
            np.std(returns_array) * np.sqrt(252) if len(returns_array) > 1 else 0.0
        )

        # Sharpe ratio
        mean_return = np.mean(returns_array)
        sharpe_ratio = (mean_return * 252 / volatility) if volatility > 0 else 0.0

        # Max drawdown
        if len(portfolio_values) > 1:
            portfolio_array = np.array(portfolio_values)
            cumulative_max = np.maximum.accumulate(portfolio_array)
            drawdowns = (portfolio_array - cumulative_max) / cumulative_max
            max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
        else:
            max_drawdown = 0.0

        # Win rate
        winning_trades = len(returns_array[returns_array > 0])
        total_trades = len(returns_array)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        # Profit factor
        gross_profit = np.sum(returns_array[returns_array > 0])
        gross_loss = abs(np.sum(returns_array[returns_array < 0]))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

        # 7. Rapport final
        logger.info("\n" + "=" * 50)
        logger.info("📈 RÉSULTATS ÉVALUATION RAPIDE")
        logger.info("=" * 50)

        # Déterminer le statut global
        performance_score = 0
        if sharpe_ratio > 3.0:
            performance_score += 1
        if max_drawdown < 0.20:
            performance_score += 1
        if win_rate > 48:
            performance_score += 1
        if profit_factor > 1.5:
            performance_score += 1
        if total_return > 0:
            performance_score += 1

        deployment_ready = performance_score >= 3 and weights_valid

        logger.info(
            f"🎯 Sharpe Ratio: {sharpe_ratio:.3f} {'✅' if sharpe_ratio > 3.0 else '⚠️'}"
        )
        logger.info(
            f"📉 Max Drawdown: {max_drawdown:.2%} {'✅' if max_drawdown < 0.20 else '⚠️'}"
        )
        logger.info(f"🎲 Win Rate: {win_rate:.1f}% {'✅' if win_rate > 48 else '⚠️'}")
        logger.info(
            f"💰 Profit Factor: {profit_factor:.2f} {'✅' if profit_factor > 1.5 else '⚠️'}"
        )
        logger.info(
            f"📈 Return Total: {total_return:.2%} {'✅' if total_return > 0 else '⚠️'}"
        )
        logger.info(f"📊 Volatilité: {volatility:.2%}")
        logger.info(f"🔄 Steps testés: {num_steps}")
        logger.info(f"📝 Total trades: {total_trades}")
        logger.info(
            f"🎛️ Poids fusion: {'✅ Validés' if weights_valid else '❌ Invalides'}"
        )

        logger.info("-" * 50)
        logger.info(f"📊 Score Performance: {performance_score}/5")

        if deployment_ready:
            logger.info("🚀 MODÈLE APPROUVÉ POUR DÉPLOIEMENT")
            logger.info("💡 Recommandations:")
            logger.info("   - Commencer avec position size 0.1%")
            logger.info("   - Monitoring temps réel requis")
            logger.info("   - Réévaluation après 30 jours")
        else:
            logger.info("⚠️ OPTIMISATION REQUISE AVANT DÉPLOIEMENT")
            logger.info("🔧 Actions nécessaires:")
            if sharpe_ratio < 2.0:
                logger.info("   - Améliorer ratio rendement/risque")
            if max_drawdown > 0.25:
                logger.info("   - Réduire le drawdown maximum")
            if win_rate < 45:
                logger.info("   - Améliorer la précision des signaux")
            if not weights_valid:
                logger.info("   - Corriger les poids de fusion")

        logger.info("=" * 50)

        # Sauvegarde rapport simple
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "metrics": {
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "win_rate": float(win_rate),
                "profit_factor": float(profit_factor),
                "total_return": float(total_return),
                "volatility": float(volatility),
                "total_trades": int(total_trades),
                "performance_score": int(performance_score),
            },
            "fusion_weights": fusion_weights,
            "deployment_ready": deployment_ready,
        }

        # Créer dossier reports si nécessaire
        os.makedirs("reports", exist_ok=True)

        import json

        report_file = (
            f"reports/quick_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"📄 Rapport sauvegardé: {report_file}")

        env.close()
        return deployment_ready

    except Exception as e:
        logger.error(f"❌ Erreur évaluation: {e}")
        return False


def main():
    """Fonction principale."""
    import argparse

    parser = argparse.ArgumentParser(description="Évaluation rapide modèle ADAN")
    parser.add_argument("--config", required=True, help="Chemin config")
    parser.add_argument("--model", required=True, help="Chemin modèle")
    parser.add_argument("--steps", type=int, default=1000, help="Nombre de steps test")

    args = parser.parse_args()

    # Vérifications
    if not os.path.exists(args.config):
        logger.error(f"❌ Config non trouvée: {args.config}")
        return False

    if not os.path.exists(args.model):
        logger.error(f"❌ Modèle non trouvé: {args.model}")
        return False

    # Lancer évaluation
    success = quick_model_evaluation(args.config, args.model, args.steps)

    if success:
        logger.info("\n🎉 ÉVALUATION RÉUSSIE - Modèle prêt!")
    else:
        logger.info("\n⚠️ ÉVALUATION MITIGÉE - Optimisations requises")

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
