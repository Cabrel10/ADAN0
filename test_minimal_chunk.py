#!/usr/bin/env python3
"""
Test minimal pour vérifier la progression des chunks.
Utilise la fonction make_env existante du script d'entraînement.
"""

import os
import sys
import logging
import yaml
from pathlib import Path

# Configuration des chemins
current_dir = Path(__file__).parent
bot_dir = current_dir / "bot"
sys.path.insert(0, str(bot_dir / "src"))
sys.path.insert(0, str(bot_dir / "scripts"))

def setup_logging():
    """Configuration du logging pour le test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    """Test principal de progression des chunks"""
    logger = setup_logging()
    logger.info("🧪 TEST MINIMAL - PROGRESSION DES CHUNKS")

    try:
        # Charger la configuration
        config_path = bot_dir / "config" / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Forcer l'utilisation des données d'entraînement au lieu de validation
        if 'data_loader' not in config:
            config['data_loader'] = {}
        config['data_loader']['split'] = 'train'

        logger.info("✅ Configuration chargée (modifiée pour utiliser les données train)")

        # Importer la fonction make_env
        from train_parallel_agents import make_env
        logger.info("✅ Fonction make_env importée")

        # Créer l'environnement
        worker_config = {
            "worker_id": 0,
            "rank": 0,
            "log_prefix": "[TEST]"
        }

        env = make_env(rank=0, seed=42, config=config, worker_config=worker_config)
        logger.info("✅ Environnement créé")

        # Reset initial
        obs = env.reset()
        logger.info("✅ Environnement resetté")

        # Obtenir les informations initiales
        if hasattr(env, 'env'):  # Si c'est un wrapper
            base_env = env.env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
        else:
            base_env = env

        current_chunk = getattr(base_env, 'current_chunk_idx', 0)
        total_chunks = getattr(base_env, 'total_chunks', 1)
        step_in_chunk = getattr(base_env, 'step_in_chunk', 0)

        logger.info(f"📊 État initial:")
        logger.info(f"   - Chunk: {current_chunk + 1}/{total_chunks}")
        logger.info(f"   - Step dans chunk: {step_in_chunk}")

        # Estimer la taille du chunk
        chunk_size = 1000  # D'après nos observations précédentes
        logger.info(f"   - Taille estimée du chunk: {chunk_size}")

        # Variables de suivi
        step_count = 0
        max_steps = chunk_size + 50  # Un peu plus qu'un chunk complet
        chunk_transitions = 0
        last_chunk = current_chunk

        logger.info(f"🚀 DÉBUT DU TEST - {max_steps} steps maximum")

        # Boucle de test
        while step_count < max_steps:
            # Action neutre (pas de trading)
            action = 0.0  # Action simple

            try:
                result = env.step(action)
                step_count += 1

                # Gérer les différents formats de retour
                if len(result) == 4:
                    obs, reward, done, info = result
                    terminated = done
                    truncated = False
                elif len(result) == 5:
                    obs, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:
                    logger.error(f"Format de retour step() inattendu: {len(result)} éléments")
                    break

                # Vérifier l'état actuel des chunks
                current_chunk = getattr(base_env, 'current_chunk_idx', 0)
                step_in_chunk = getattr(base_env, 'step_in_chunk', 0)

                # Détecter les transitions
                if current_chunk != last_chunk:
                    chunk_transitions += 1
                    logger.info(f"🎉 TRANSITION RÉUSSIE!")
                    logger.info(f"   Chunk {last_chunk + 1} → {current_chunk + 1}")
                    logger.info(f"   Step global: {step_count}")
                    logger.info(f"   Nouveau step_in_chunk: {step_in_chunk}")
                    last_chunk = current_chunk

                # Log périodique
                if step_count % 100 == 0:
                    logger.info(f"📈 Step {step_count}: chunk={current_chunk + 1}, step_in_chunk={step_in_chunk}")

                # Log quand on approche de la fin du chunk
                if step_in_chunk > chunk_size - 20 and step_in_chunk % 5 == 0:
                    logger.info(f"🔥 PROCHE FIN CHUNK: step_in_chunk={step_in_chunk}/{chunk_size-1}")

                # Arrêter si l'épisode est terminé
                if terminated or truncated or done:
                    logger.info(f"⏹️ Épisode terminé: step={step_count}")
                    logger.info(f"   terminated={terminated}, truncated={truncated}")
                    break

            except Exception as e:
                logger.error(f"❌ Erreur au step {step_count}: {e}")
                import traceback
                traceback.print_exc()
                break

        # Résultats finaux
        logger.info(f"🏁 FIN DU TEST")
        logger.info(f"📊 RÉSULTATS FINAUX:")
        logger.info(f"   - Steps exécutés: {step_count}")
        logger.info(f"   - Transitions de chunks: {chunk_transitions}")
        logger.info(f"   - Chunk final: {current_chunk + 1}/{total_chunks}")
        logger.info(f"   - Step final dans chunk: {step_in_chunk}")

        # Évaluation du succès
        success = False
        if chunk_transitions > 0:
            logger.info(f"✅ SUCCÈS COMPLET! Les chunks progressent correctement.")
            success = True
        elif step_in_chunk >= chunk_size - 10:
            logger.info(f"⚠️ SUCCÈS PARTIEL: Atteint la fin du chunk mais pas de transition.")
            logger.info(f"   Cela peut indiquer un problème dans la logique de transition.")
        else:
            logger.info(f"❌ ÉCHEC: Arrêt prématuré, chunk non complété.")

        return success

    except Exception as e:
        logger.error(f"❌ ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 TEST MINIMAL - PROGRESSION DES CHUNKS ADAN")
    print("=" * 70)

    success = main()

    print("\n" + "=" * 70)
    if success:
        print("✅ RÉSULTAT: TEST RÉUSSI - Les chunks progressent!")
        print("   L'environnement peut passer d'un chunk à l'autre.")
    else:
        print("❌ RÉSULTAT: TEST ÉCHOUÉ - Problème de progression.")
        print("   L'environnement ne progresse pas entre les chunks.")
    print("=" * 70)

    sys.exit(0 if success else 1)
