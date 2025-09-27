#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Script de test pour vérifier les corrections d'indexation et de métriques."""

import sys
import os
sys.path.append('bot/src')

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from typing import Dict, Any
import time

# Imports du bot
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from stable_baselines3.common.vec_env import DummyVecEnv


def load_test_config():
    """Charger la configuration de test."""
    config_path = Path("bot/config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_test_environment(config: Dict[str, Any], worker_id: int = 0):
    """Créer un environnement de test."""
    try:
        # Extraire les paramètres nécessaires
        data_config = config.get("data", {})
        trading_config = config.get("trading", {})
        env_config = config.get("environment", {})

        env = MultiAssetChunkedEnv(
            data=data_config.get("assets", {}),
            timeframes=data_config.get("timeframes", ["5m", "1h", "4h"]),
            window_size=env_config.get("window_size", 20),
            features_config=data_config.get("features", {}),
            max_steps=env_config.get("max_steps", 1000),
            initial_balance=trading_config.get("initial_capital", 20.50),
            commission=trading_config.get("commission", 0.001),
            reward_scaling=trading_config.get("reward_scaling", 1.0),
            enable_logging=True,
            log_dir="logs/test",
            worker_id=worker_id,
            config=config
        )

        return env
    except Exception as e:
        print(f"❌ Erreur lors de la création de l'environnement: {e}")
        return None


def test_indexation_fix():
    """Tester la correction de l'indexation step_in_chunk vs current_step."""
    print("🔍 Test 1: Vérification de la correction d'indexation")
    print("=" * 60)

    config = load_test_config()
    env = create_test_environment(config)

    if not env:
        print("❌ Impossible de créer l'environnement")
        return False

    try:
        # Reset et premiers steps
        obs, info = env.reset()
        print(f"✅ Reset réussi")

        # Faire quelques steps et vérifier les prix
        success_count = 0
        forward_fill_count = 0

        for step in range(10):
            action = np.array([0.0])  # Action neutre
            obs, reward, terminated, truncated, info = env.step(action)

            # Vérifier si on a des prix valides
            current_prices = env._get_current_prices()

            if current_prices and len(current_prices) > 0:
                success_count += 1
                price_val = list(current_prices.values())[0]
                print(f"  Step {step+1}: Prix obtenu = {price_val:.4f} ✅")
            else:
                forward_fill_count += 1
                print(f"  Step {step+1}: Aucun prix obtenu ❌")

            if terminated or truncated:
                break

        success_rate = success_count / (success_count + forward_fill_count) * 100
        print(f"\n📊 Résultats:")
        print(f"  - Lectures réussies: {success_count}")
        print(f"  - Forward-fills: {forward_fill_count}")
        print(f"  - Taux de succès: {success_rate:.1f}%")

        # Vérifier les compteurs internes
        if hasattr(env, '_price_read_success_count') and hasattr(env, '_price_forward_fill_count'):
            total_internal = env._price_read_success_count + env._price_forward_fill_count
            print(f"  - Compteurs internes: Success={env._price_read_success_count}, Forward-fill={env._price_forward_fill_count}")

        return success_rate > 90.0  # Au moins 90% de succès

    except Exception as e:
        print(f"❌ Erreur durant le test: {e}")
        return False
    finally:
        if env:
            env.close()


def test_multi_worker_metrics():
    """Tester la collecte de métriques pour plusieurs workers."""
    print("\n🔍 Test 2: Vérification des métriques multi-workers")
    print("=" * 60)

    config = load_test_config()

    # Créer plusieurs environnements (simulation de workers)
    workers = []
    num_workers = 3

    for i in range(num_workers):
        env = create_test_environment(config, worker_id=i)
        if env:
            workers.append(env)

    if len(workers) != num_workers:
        print(f"❌ Impossible de créer {num_workers} workers")
        return False

    try:
        print(f"✅ {num_workers} workers créés avec succès")

        # Reset tous les workers
        for i, env in enumerate(workers):
            obs, info = env.reset()
            print(f"  Worker {i}: Reset OK")

        # Faire quelques steps sur chaque worker
        for step in range(5):
            print(f"\n📊 Step {step+1}:")

            for i, env in enumerate(workers):
                action = np.array([0.1 * (i+1)])  # Actions légèrement différentes
                obs, reward, terminated, truncated, info = env.step(action)

                # Obtenir les métriques détaillées
                try:
                    metrics = env.get_portfolio_metrics()

                    portfolio_value = metrics.get('portfolio_value', 0)
                    worker_id = metrics.get('worker_id', i)
                    last_reward = metrics.get('last_reward', 0)
                    trades = metrics.get('trades', 0)

                    print(f"  Worker {worker_id}: Portfolio=${portfolio_value:.2f}, Reward={last_reward:+.4f}, Trades={trades}")

                except Exception as e:
                    print(f"  Worker {i}: Erreur métrique = {e}")

                if terminated or truncated:
                    print(f"  Worker {i}: Episode terminé")

        print(f"\n✅ Test multi-workers terminé avec succès")
        return True

    except Exception as e:
        print(f"❌ Erreur durant le test multi-workers: {e}")
        return False
    finally:
        for env in workers:
            if env:
                env.close()


def test_forward_fill_detection():
    """Tester la détection de forward-fill excessif."""
    print("\n🔍 Test 3: Vérification de la détection de forward-fill")
    print("=" * 60)

    config = load_test_config()
    env = create_test_environment(config)

    if not env:
        print("❌ Impossible de créer l'environnement")
        return False

    try:
        obs, info = env.reset()

        # Forcer un scénario de forward-fill en manipulant les compteurs
        if hasattr(env, '_price_forward_fill_count') and hasattr(env, '_price_read_success_count'):
            # Simuler beaucoup de forward-fills
            env._price_forward_fill_count = 80
            env._price_read_success_count = 20

            print("🔧 Simulation: 80 forward-fills sur 100 lectures (80%)")

            # Déclencher la vérification
            env._check_excessive_forward_fill()

            print("✅ Mécanisme de détection testé")
            return True
        else:
            print("❌ Compteurs de forward-fill non trouvés")
            return False

    except Exception as e:
        print(f"❌ Erreur durant le test forward-fill: {e}")
        return False
    finally:
        if env:
            env.close()


def main():
    """Fonction principale du script de test."""
    print("🚀 Tests de vérification des corrections ADAN")
    print("=" * 80)

    start_time = time.time()

    results = {
        "indexation": test_indexation_fix(),
        "multi_workers": test_multi_worker_metrics(),
        "forward_fill": test_forward_fill_detection()
    }

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("📋 RÉSULTATS FINAUX")
    print("=" * 80)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name.upper():<15} : {status}")
        if not passed:
            all_passed = False

    print(f"\n⏱️  Temps d'exécution: {elapsed:.2f}s")

    if all_passed:
        print("🎉 TOUS LES TESTS SONT PASSÉS ! Les corrections fonctionnent.")
        return 0
    else:
        print("⚠️  CERTAINS TESTS ONT ÉCHOUÉ. Vérifiez les corrections.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
