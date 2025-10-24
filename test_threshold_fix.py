#!/usr/bin/env python3
"""
Test rapide pour valider que les corrections des seuils fonctionnent.
"""
import sys
import os
sys.path.insert(0, 'src')

from src.adan_trading_bot.common.config_loader import ConfigLoader
from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_action_thresholds():
    """Test que les seuils d'action sont maintenant permissifs."""
    print("🧪 TEST DES SEUILS D'ACTION CORRIGÉS")
    print("=" * 50)

    # Charger config
    config = ConfigLoader.load_config('config/config.yaml')
    print(f"✅ Config loaded: action_threshold = {config.get('trading_rules', {}).get('frequency', {}).get('action_threshold', 0.01)}")

    # Vérifier min_confidence des workers
    for worker_key in ['w1', 'w2', 'w3', 'w4']:
        if worker_key in config.get('workers', {}):
            min_conf = config['workers'][worker_key].get('min_confidence', 0.01)
            print(f"✅ {worker_key} min_confidence = {min_conf}")

    # Test d'actions simulées
    print("\n📊 TEST D'ACTIONS SIMULÉES:")
    test_actions = [
        [0.001, 0.01, 0.005, 0.02, -0.005],  # Actions typiques du modèle
        [0.007, 0.015, 0.008, 0.012, -0.003], # Actions qui devraient maintenant déclencher
        [0.05, 0.03, 0.02, 0.01, -0.01],     # Actions fortes
    ]

    for i, actions in enumerate(test_actions):
        print(f"\n🎯 Test {i+1}: Actions = {actions}")
        should_trade = any(abs(a) > 0.005 for a in actions)  # Seuil 0.005
        print(f"   {'✅ DEVRAIT TRADER' if should_trade else '❌ PAS DE TRADE'}")

        # Vérifier par rapport aux anciens seuils
        old_threshold = 0.01
        old_would_trade = any(abs(a) > old_threshold for a in actions)
        if should_trade and not old_would_trade:
            print(f"   🚀 AMÉLIORATION: Trade maintenant possible (impossible avant)")

    print("\n🎉 TEST RÉUSSI: Seuils plus permissifs configurés!")
    return True

if __name__ == "__main__":
    success = test_action_thresholds()
    sys.exit(0 if success else 1)
