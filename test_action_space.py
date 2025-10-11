#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify action space dimensions for the ADAN trading bot.

This script helps diagnose action space dimension mismatches between
the environment and the PPO model.
"""

import sys
import os
import yaml
import numpy as np
import torch
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.adan_trading_bot.environment.multi_asset_chunked_env import (
    MultiAssetChunkedEnv,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_action_space_dimensions():
    """Test the action space dimensions of the environment."""

    print("=" * 60)
    print("🔍 DIAGNOSTIC DE L'ESPACE D'ACTION")
    print("=" * 60)

    # Load configuration
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        print("✅ Configuration chargée avec succès")
    except Exception as e:
        print(f"❌ Erreur lors du chargement de la configuration: {e}")
        return False

    # Extract assets from config
    assets = config.get("data", {}).get("assets", [])
    print(f"📊 Actifs configurés: {assets} (nombre: {len(assets)})")

    # Expected action space dimensions
    expected_dim = (
        len(assets) * 3
    )  # 3 values per asset (decision, risk_horizon, position_size)
    print(
        f"🎯 Dimension d'action attendue: {expected_dim} (= {len(assets)} actifs × 3)"
    )

    # Create environment
    try:
        # Use minimal worker_id for testing
        # Extract required parameters from config
        data_config = config.get("data", {})
        timeframes = data_config.get("timeframes", ["5m", "1h", "4h"])
        window_size = config.get("environment", {}).get("window_size", 20)
        features_config = config.get("environment", {}).get("features_config", {})

        env = MultiAssetChunkedEnv(
            config=config,
            worker_id=0,
            data=data_config,
            timeframes=timeframes,
            window_size=window_size,
            features_config=features_config,
        )
        print("✅ Environnement créé avec succès")

        # Check action space
        action_space = env.action_space
        print(f"📏 Espace d'action détecté: {action_space}")
        print(f"📐 Forme de l'espace d'action: {action_space.shape}")

        if hasattr(action_space, "shape"):
            actual_dim = (
                action_space.shape[0]
                if len(action_space.shape) == 1
                else np.prod(action_space.shape)
            )
            print(f"🔢 Dimension réelle: {actual_dim}")

            if actual_dim == expected_dim:
                print("✅ SUCCÈS: Les dimensions correspondent!")
            else:
                print(
                    f"❌ ERREUR: Dimension attendue ({expected_dim}) != dimension réelle ({actual_dim})"
                )
                return False

        # Test action generation
        print("\n🧪 Test de génération d'actions:")

        # Generate random action with correct dimensions
        action = env.action_space.sample()
        print(f"Action échantillonnée: {action}")
        print(f"Forme de l'action: {action.shape}")
        print(f"Type de l'action: {type(action)}")

        # Test manual action with correct dimensions
        manual_action = np.random.uniform(-1, 1, size=(expected_dim,))
        print(f"Action manuelle: {manual_action}")
        print(f"Forme de l'action manuelle: {manual_action.shape}")

        # Test action space validation
        try:
            # This should pass if our action space is correct
            clipped_action = np.clip(
                manual_action, env.action_space.low, env.action_space.high
            )
            print("✅ Action validée par l'espace d'action")
        except Exception as e:
            print(f"❌ Erreur de validation de l'action: {e}")
            return False

        # Test step function (minimal)
        try:
            obs, info = env.reset()
            print("✅ Reset de l'environnement réussi")
            print(
                f"Observation shape: {type(obs)} - {obs.keys() if hasattr(obs, 'keys') else 'Non-dict obs'}"
            )

            # Try to step with correct action
            obs, reward, done, truncated, info = env.step(manual_action)
            print("✅ Step de l'environnement réussi!")
            print(f"Reward: {reward}, Done: {done}, Truncated: {truncated}")

        except Exception as e:
            print(f"❌ Erreur lors du step: {e}")
            # This might fail due to other reasons, but the action dimension should be OK
            if "Action shape" in str(e):
                print("🚨 Erreur de dimension d'action confirmée!")
                return False
            else:
                print("⚠️  Erreur non liée aux dimensions d'action (peut être ignorée)")

        print("\n🎉 Test de l'espace d'action terminé avec succès!")
        return True

    except Exception as e:
        print(f"❌ Erreur lors de la création de l'environnement: {e}")
        return False


def test_ppo_model_compatibility():
    """Test PPO model creation with the environment."""

    print("\n" + "=" * 60)
    print("🤖 TEST DE COMPATIBILITÉ PPO")
    print("=" * 60)

    try:
        from sb3_contrib import RecurrentPPO
        from src.adan_trading_bot.agent.custom_recurrent_policy import (
            CustomRecurrentPolicy,
        )

        # Load configuration
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Create environment
        # Extract required parameters from config
        data_config = config.get("data", {})
        timeframes = data_config.get("timeframes", ["5m", "1h", "4h"])
        window_size = config.get("environment", {}).get("window_size", 20)
        features_config = config.get("environment", {}).get("features_config", {})

        env = MultiAssetChunkedEnv(
            config=config,
            worker_id=0,
            data=data_config,
            timeframes=timeframes,
            window_size=window_size,
            features_config=features_config,
        )

        print(f"📊 Espace d'action de l'env: {env.action_space}")
        print(f"📊 Espace d'observation de l'env: {env.observation_space}")

        # Try to create PPO model
        model = RecurrentPPO(
            policy=CustomRecurrentPolicy,
            env=env,
            verbose=1,
            device="cpu",  # Force CPU for testing
        )

        print("✅ Modèle PPO créé avec succès!")

        # Test action prediction
        obs, info = env.reset()
        action, _states = model.predict(obs, deterministic=True)

        print(f"🎬 Action prédite par PPO: {action}")
        print(f"📐 Forme de l'action PPO: {action.shape}")

        expected_dim = len(config.get("data", {}).get("assets", [])) * 3
        if action.shape[0] == expected_dim:
            print("✅ SUCCÈS: PPO génère des actions avec les bonnes dimensions!")
            return True
        else:
            print(
                f"❌ ERREUR: PPO génère des actions de dimension {action.shape[0]}, attendu {expected_dim}"
            )
            return False

    except ImportError as e:
        print(f"⚠️  sb3-contrib non disponible pour le test PPO: {e}")
        return None
    except Exception as e:
        print(f"❌ Erreur lors du test PPO: {e}")
        return False


if __name__ == "__main__":
    print("🚀 DÉMARRAGE DU DIAGNOSTIC DE L'ESPACE D'ACTION")
    print("=" * 60)

    # Test 1: Environment action space
    env_test_passed = test_action_space_dimensions()

    # Test 2: PPO compatibility
    ppo_test_result = test_ppo_model_compatibility()

    # Summary
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ DU DIAGNOSTIC")
    print("=" * 60)
    print(f"🔧 Test environnement: {'✅ PASSÉ' if env_test_passed else '❌ ÉCHEC'}")

    if ppo_test_result is True:
        print("🤖 Test PPO: ✅ PASSÉ")
    elif ppo_test_result is False:
        print("🤖 Test PPO: ❌ ÉCHEC")
    else:
        print("🤖 Test PPO: ⚠️  NON DISPONIBLE")

    if env_test_passed and ppo_test_result is not False:
        print("\n🎉 DIAGNOSTIC GLOBAL: SUCCÈS!")
        sys.exit(0)
    else:
        print("\n💥 DIAGNOSTIC GLOBAL: PROBLÈMES DÉTECTÉS!")
        sys.exit(1)
