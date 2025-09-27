#!/usr/bin/env python3
"""
Test simple pour vérifier la progression des chunks dans l'environnement ADAN.
Ce test vérifie si l'environnement peut correctement passer d'un chunk à l'autre.
"""

import sys
import os
import yaml
import logging
from pathlib import Path

# Ajouter le chemin du bot au sys.path
bot_path = Path(__file__).parent / "bot" / "src"
sys.path.insert(0, str(bot_path))

def setup_logging():
    """Configure le logging pour le test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_config():
    """Charge la configuration"""
    config_path = Path(__file__).parent / "bot" / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_chunk_progression():
    """Test principal de progression des chunks"""
    logger = setup_logging()
    logger.info("🧪 DÉBUT DU TEST DE PROGRESSION DES CHUNKS")

    try:
        # Charger la configuration
        config = load_config()
        logger.info(f"✅ Configuration chargée")

        # Importer les modules nécessaires
        from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
        from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

        # Créer le data loader
        data_loader_config = config.get('data_loader', {})
        data_loader = ChunkedDataLoader(config=config, worker_config={'worker_id': 0}, worker_id=0)
        logger.info(f"✅ DataLoader créé")

        # Charger les données
        data = data_loader.load_all_data()
        logger.info(f"✅ Données chargées: {list(data.keys())}")

        # Créer l'environnement
        env_config = config.get('environment', {})
        env = MultiAssetChunkedEnv(
            data=data,
            timeframes=config.get('data', {}).get('timeframes', ['5m', '1h']),
            window_size=env_config.get('window_size', 50),
            features_config=env_config.get('features_config', {}),
            config=config,
            worker_config={'worker_id': 0, 'rank': 0},
            total_workers=1
        )
        logger.info(f"✅ Environnement créé")

        # Reset initial
        obs, info = env.reset()
        logger.info(f"✅ Environnement resetté")
        logger.info(f"📊 Chunk initial: {env.current_chunk_idx + 1}/{getattr(env, 'total_chunks', 'unknown')}")

        # Obtenir la taille du chunk
        first_asset = next(iter(env.current_data))
        first_timeframe = next(iter(env.current_data[first_asset]))
        chunk_size = len(env.current_data[first_asset][first_timeframe])
        logger.info(f"📏 Taille du chunk: {chunk_size} steps")

        # Variables de suivi
        step_count = 0
        chunk_transitions = 0
        max_steps = min(chunk_size * 3, 3000)  # Maximum 3 chunks ou 3000 steps
        last_chunk_idx = env.current_chunk_idx

        logger.info(f"🚀 DÉBUT DES STEPS (max: {max_steps})")

        while step_count < max_steps:
            # Action simple (pas d'action)
            action = [0.0] * len(env.assets)

            try:
                obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1

                # Vérifier si le chunk a changé
                current_chunk = env.current_chunk_idx
                if current_chunk != last_chunk_idx:
                    chunk_transitions += 1
                    logger.info(f"🎉 TRANSITION RÉUSSIE! Passage du chunk {last_chunk_idx + 1} au chunk {current_chunk + 1}")
                    logger.info(f"📈 step_in_chunk resetté: {env.step_in_chunk}")
                    last_chunk_idx = current_chunk

                # Log périodique
                if step_count % 100 == 0:
                    logger.info(f"📊 Step {step_count}: chunk={current_chunk + 1}, step_in_chunk={env.step_in_chunk}/{chunk_size-1}")

                # Vérifier si nous approchons de la fin du chunk
                if env.step_in_chunk > chunk_size - 20:
                    logger.info(f"🔥 APPROCHE DE LA FIN DU CHUNK: step_in_chunk={env.step_in_chunk}, seuil={chunk_size-1}")

                # Si l'épisode est terminé
                if terminated or truncated:
                    logger.info(f"⏹️ Épisode terminé à step {step_count}: terminated={terminated}, truncated={truncated}")
                    break

            except Exception as e:
                logger.error(f"❌ Erreur pendant step {step_count}: {e}")
                import traceback
                traceback.print_exc()
                break

        # Résultats du test
        logger.info(f"🏁 FIN DU TEST")
        logger.info(f"📊 RÉSULTATS:")
        logger.info(f"   - Steps exécutés: {step_count}")
        logger.info(f"   - Transitions de chunks: {chunk_transitions}")
        logger.info(f"   - Chunk final: {env.current_chunk_idx + 1}/{getattr(env, 'total_chunks', 'unknown')}")
        logger.info(f"   - step_in_chunk final: {env.step_in_chunk}")

        # Évaluation du succès
        if chunk_transitions > 0:
            logger.info(f"✅ TEST RÉUSSI! L'environnement peut progresser entre les chunks.")
            return True
        else:
            logger.warning(f"⚠️ TEST PARTIELLEMENT RÉUSSI: Aucune transition de chunk observée.")
            logger.info(f"   Cela peut être normal si le chunk est très grand ou si l'épisode termine avant.")
            return False

    except Exception as e:
        logger.error(f"❌ ÉCHEC DU TEST: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chunk_progression()
    sys.exit(0 if success else 1)
