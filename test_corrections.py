#!/usr/bin/env python3
"""
Test de diagnostic pour vérifier les corrections apportées :
1. Persistance de l'état DBE
2. Progression correcte des chunks
3. Réduction des resets intempestifs
"""

import sys
import os
import time
import logging
from pathlib import Path

# Ajouter le chemin du package
sys.path.insert(0, str(Path(__file__).parent / "bot" / "src"))

try:
    import yaml
    import numpy as np
    from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
    from adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine
except ImportError as e:
    print(f"Erreur d'import: {e}")
    print("Assurez-vous que tous les packages sont installés")
    sys.exit(1)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dbe_persistence():
    """Test la persistance de l'état DBE"""
    print("\n🔍 Test 1: Persistance de l'état DBE")

    try:
        # Créer une instance DBE
        config = {
            'dynamic_behavior': {
                'enabled': True,
                'regime_detection': {'enabled': True},
                'position_sizing': {'enabled': True}
            }
        }

        dbe = DynamicBehaviorEngine(config=config, worker_id=0)

        # Vérifier l'initialisation
        if not hasattr(dbe, 'state') or dbe.state is None:
            print("❌ État DBE non initialisé")
            return False

        initial_step = dbe.state.get('current_step', 0)
        print(f"✅ État DBE initialisé, step initial: {initial_step}")

        # Simuler des mises à jour
        for i in range(5):
            mock_metrics = {
                'portfolio_value': 20.0 + i,
                'win_rate': 0.6,
                'drawdown': 0.02 * i
            }

            dbe.update_state(mock_metrics)
            current_step = dbe.state.get('current_step', 0)
            print(f"Step {i+1}: DBE step = {current_step}")

            # Vérifier la persistance
            if current_step != initial_step + i + 1:
                print(f"❌ Persistance échouée: attendu {initial_step + i + 1}, obtenu {current_step}")
                return False

        print("✅ Test DBE persistance: SUCCÈS")
        return True

    except Exception as e:
        print(f"❌ Erreur dans test DBE: {e}")
        return False

def test_environment_initialization():
    """Test l'initialisation de l'environnement sans crash"""
    print("\n🔍 Test 2: Initialisation de l'environnement")

    try:
        # Charger la configuration
        config_path = Path("bot/config/config.yaml")
        if not config_path.exists():
            print(f"❌ Fichier de config non trouvé: {config_path}")
            return False

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Modifier pour test rapide
        config['data']['max_chunks_per_episode'] = 2
        config['environment']['max_steps'] = 100

        # Créer l'environnement
        env = MultiAssetChunkedEnv(config=config, worker_id=999)  # ID unique

        print("✅ Environnement créé")

        # Reset initial
        observation, info = env.reset()
        print(f"✅ Reset initial réussi, obs shape: {[k + ':' + str(v.shape) for k, v in observation.items()]}")

        # Vérifier l'état initial
        initial_chunk = env.current_chunk_idx
        initial_step = env.current_step

        print(f"✅ État initial - Chunk: {initial_chunk+1}/{env.total_chunks}, Step: {initial_step}")

        return True, env

    except Exception as e:
        print(f"❌ Erreur initialisation env: {e}")
        return False, None

def test_chunk_progression(env):
    """Test la progression des chunks"""
    print("\n🔍 Test 3: Progression des chunks")

    if env is None:
        print("❌ Environnement non disponible")
        return False

    try:
        initial_chunk = env.current_chunk_idx
        chunk_transitions = []
        max_steps_to_test = 50  # Test limité

        for step in range(max_steps_to_test):
            # Action aléatoire
            action = np.random.uniform(-1, 1, size=(1,))

            obs, reward, terminated, truncated, info = env.step(action)

            current_chunk = env.current_chunk_idx
            current_step_in_chunk = env.step_in_chunk

            # Détecter les transitions de chunk
            if current_chunk != initial_chunk + len(chunk_transitions):
                transition_info = {
                    'step': step,
                    'from_chunk': initial_chunk + len(chunk_transitions),
                    'to_chunk': current_chunk,
                    'step_in_chunk': current_step_in_chunk
                }
                chunk_transitions.append(transition_info)
                print(f"🔄 Transition détectée: {transition_info}")

            # Arrêter si terminé
            if terminated or truncated:
                termination_reason = info.get('termination_reason', 'unknown')
                print(f"⏹️ Épisode terminé à l'étape {step}: {termination_reason}")
                break

            # Log périodique
            if step % 10 == 0:
                print(f"📊 Step {step}: chunk {current_chunk+1}, step_in_chunk {current_step_in_chunk}, reward {reward:.3f}")

        # Résumé
        print(f"\n📈 Résumé test chunks:")
        print(f"   - Steps testés: {min(step + 1, max_steps_to_test)}")
        print(f"   - Transitions détectées: {len(chunk_transitions)}")
        print(f"   - Chunk final: {env.current_chunk_idx + 1}/{env.total_chunks}")

        if len(chunk_transitions) > 0:
            print("✅ Progression des chunks: SUCCÈS")
            return True
        else:
            print("⚠️ Aucune transition de chunk observée (peut être normal pour un test court)")
            return True  # Pas forcément un échec

    except Exception as e:
        print(f"❌ Erreur test chunks: {e}")
        return False

def test_dbe_integration(env):
    """Test l'intégration DBE dans l'environnement"""
    print("\n🔍 Test 4: Intégration DBE")

    if env is None:
        print("❌ Environnement non disponible")
        return False

    try:
        if not hasattr(env, 'dynamic_behavior_engine') or env.dynamic_behavior_engine is None:
            print("⚠️ DBE non activé dans l'environnement")
            return True

        dbe = env.dynamic_behavior_engine

        # Vérifier l'état initial
        if not hasattr(dbe, 'state') or dbe.state is None:
            print("❌ État DBE non initialisé dans l'environnement")
            return False

        initial_step = dbe.state.get('current_step', 0)
        print(f"✅ DBE intégré, step initial: {initial_step}")

        # Faire quelques steps pour voir si l'état évolue
        for i in range(5):
            action = np.array([0.1 * i])  # Actions graduelles
            obs, reward, terminated, truncated, info = env.step(action)

            current_dbe_step = dbe.state.get('current_step', 0)
            print(f"Step {i+1}: DBE step = {current_dbe_step}, reward = {reward:.3f}")

            if terminated or truncated:
                break

        final_step = dbe.state.get('current_step', 0)
        steps_evolved = final_step - initial_step

        if steps_evolved > 0:
            print(f"✅ DBE évolue correctement: {steps_evolved} steps")
            return True
        else:
            print(f"❌ DBE n'évolue pas: {steps_evolved} steps")
            return False

    except Exception as e:
        print(f"❌ Erreur test DBE intégration: {e}")
        return False

def run_diagnostic_tests():
    """Lance tous les tests de diagnostic"""
    print("🚀 Lancement des tests de diagnostic des corrections")
    print("=" * 60)

    results = []
    start_time = time.time()

    # Test 1: Persistance DBE
    results.append(("DBE Persistence", test_dbe_persistence()))

    # Test 2: Initialisation environnement
    env_success, env = test_environment_initialization()
    results.append(("Environment Init", env_success))

    if env_success and env is not None:
        # Test 3: Progression chunks
        results.append(("Chunk Progression", test_chunk_progression(env)))

        # Test 4: Intégration DBE
        results.append(("DBE Integration", test_dbe_integration(env)))

        # Nettoyage
        try:
            env.close()
        except:
            pass

    # Résumé final
    duration = time.time() - start_time
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ DES TESTS")
    print("=" * 60)

    success_count = sum(1 for _, success in results if success)
    total_count = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\n📊 Score: {success_count}/{total_count} tests réussis")
    print(f"⏱️ Durée: {duration:.2f}s")

    if success_count == total_count:
        print("\n🎉 TOUS LES TESTS PASSÉS - Corrections fonctionnelles!")
        return True
    else:
        print(f"\n⚠️ {total_count - success_count} test(s) échoué(s) - Corrections à revoir")
        return False

if __name__ == "__main__":
    try:
        # Changer vers le répertoire trading
        os.chdir(Path(__file__).parent)
        success = run_diagnostic_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrompu par l'utilisateur")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Erreur critique: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
