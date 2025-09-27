#!/usr/bin/env python3
"""
Test simplifié pour vérifier la progression des chunks dans l'environnement ADAN.
Utilise l'infrastructure d'entraînement existante.
"""

import os
import sys
import logging
from pathlib import Path

# Configuration du chemin
current_dir = Path(__file__).parent
bot_dir = current_dir / "bot"
sys.path.insert(0, str(bot_dir / "src"))

def setup_simple_logging():
    """Configuration simple du logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    return logger

def test_chunk_progression_simple():
    """Test simplifié de progression des chunks"""
    logger = setup_simple_logging()
    logger.info("🧪 TEST CHUNK PROGRESSION - Version Simplifiée")

    try:
        # Importer directement depuis le script d'entraînement
        sys.path.insert(0, str(bot_dir / "scripts"))

        # Import de la fonction de création d'environnement
        from train_parallel_agents import create_env_from_config
        import yaml

        # Charger la configuration
        config_path = bot_dir / "config" / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info("✅ Configuration chargée")

        # Créer un environnement de test
        logger.info("🔨 Création de l'environnement...")
        env = create_env_from_config(config, worker_id=0, total_workers=1)
        logger.info("✅ Environnement créé")

        # Reset initial
        obs, info = env.reset()
        logger.info("✅ Environnement resetté")

        # Obtenir les informations sur les chunks
        current_chunk = getattr(env, 'current_chunk_idx', 0)
        total_chunks = getattr(env, 'total_chunks', 1)
        step_in_chunk = getattr(env, 'step_in_chunk', 0)

        logger.info(f"📊 État initial:")
        logger.info(f"   - Chunk actuel: {current_chunk + 1}/{total_chunks}")
        logger.info(f"   - Step dans chunk: {step_in_chunk}")

        # Obtenir la taille du chunk pour diagnostic
        if hasattr(env, 'current_data') and env.current_data:
            first_asset = next(iter(env.current_data))
            first_timeframe = next(iter(env.current_data[first_asset]))
            chunk_size = len(env.current_data[first_asset][first_timeframe])
            logger.info(f"   - Taille du chunk: {chunk_size} steps")
        else:
            chunk_size = 1000  # Valeur par défaut observée
            logger.info(f"   - Taille du chunk: {chunk_size} steps (estimée)")

        # Variables de suivi
        step_count = 0
        max_test_steps = min(chunk_size + 100, 1200)  # Un peu plus qu'un chunk
        chunk_transitions = 0
        last_chunk = current_chunk

        logger.info(f"🚀 DÉBUT DU TEST - Maximum {max_test_steps} steps")

        while step_count < max_test_steps:
            # Action neutre
            action = [0.0] * len(getattr(env, 'assets', ['BTCUSDT']))

            try:
                obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1

                # Vérifier l'état actuel
                current_chunk = getattr(env, 'current_chunk_idx', 0)
                step_in_chunk = getattr(env, 'step_in_chunk', 0)

                # Détecter les transitions de chunks
                if current_chunk != last_chunk:
                    chunk_transitions += 1
                    logger.info(f"🎉 TRANSITION DÉTECTÉE! Chunk {last_chunk + 1} → {current_chunk + 1}")
                    logger.info(f"   Step global: {step_count}, step_in_chunk: {step_in_chunk}")
                    last_chunk = current_chunk

                # Log périodique
                if step_count % 50 == 0:
                    logger.info(f"📈 Step {step_count}: chunk={current_chunk + 1}/{total_chunks}, step_in_chunk={step_in_chunk}")

                # Log quand on approche de la fin du chunk
                if step_in_chunk > chunk_size - 50 and step_in_chunk <= chunk_size - 45:
                    logger.info(f"🔥 APPROCHE FIN CHUNK: step_in_chunk={step_in_chunk}/{chunk_size-1}")

                # Arrêter si l'épisode est terminé
                if terminated or truncated:
                    logger.info(f"⏹️ Épisode terminé: terminated={terminated}, truncated={truncated}")
                    break

            except Exception as e:
                logger.error(f"❌ Erreur au step {step_count}: {e}")
                break

        # Résultats
        logger.info(f"🏁 FIN DU TEST")
        logger.info(f"📊 RÉSULTATS:")
        logger.info(f"   - Steps exécutés: {step_count}")
        logger.info(f"   - Transitions de chunks: {chunk_transitions}")
        logger.info(f"   - Chunk final: {current_chunk + 1}/{total_chunks}")
        logger.info(f"   - Step final dans chunk: {step_in_chunk}")

        # Évaluation
        if chunk_transitions > 0:
            logger.info(f"✅ SUCCÈS! Les chunks progressent correctement.")
            return True
        elif step_count >= chunk_size - 10:
            logger.info(f"⚠️ Pas de transition observée, mais nous avons atteint presque la fin du chunk.")
            logger.info(f"   Cela peut indiquer un problème dans la logique de transition.")
            return False
        else:
            logger.info(f"⚠️ Test incomplet: arrêt prématuré avant la fin du chunk.")
            return False

    except Exception as e:
        logger.error(f"❌ ÉCHEC CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chunk_progression_simple()
    print(f"\n{'='*60}")
    if success:
        print("✅ TEST RÉUSSI: La progression des chunks fonctionne!")
    else:
        print("❌ TEST ÉCHOUÉ: Problème avec la progression des chunks.")
    print(f"{'='*60}")
    sys.exit(0 if success else 1)
