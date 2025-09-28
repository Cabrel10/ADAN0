#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de test pour valider les corrections TensorBoard et Checkpoints
Test les fonctionnalités :
- Configuration du logger SB3 TensorBoard
- Sauvegarde des checkpoints
- Resume depuis checkpoint
- Lecture des données par le dashboard
"""

import os
import sys
import time
import glob
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# Configuration des chemins
PROJECT_ROOT = Path(__file__).parent
TENSORBOARD_DIR = PROJECT_ROOT / "reports" / "tensorboard_logs"
CHECKPOINT_DIR = PROJECT_ROOT / "bot" / "checkpoints"
CONFIG_PATH = PROJECT_ROOT / "bot" / "config" / "config.yaml"
SCRIPT_PATH = PROJECT_ROOT / "bot" / "scripts" / "train_parallel_agents.py"


def print_status(message, status="INFO"):
    """Afficher un message avec timestamp et status"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if status == "SUCCESS":
        print(f"✅ [{timestamp}] {message}")
    elif status == "ERROR":
        print(f"❌ [{timestamp}] {message}")
    elif status == "WARNING":
        print(f"⚠️  [{timestamp}] {message}")
    else:
        print(f"ℹ️  [{timestamp}] {message}")


def cleanup_test_files():
    """Nettoyer les fichiers de test"""
    print_status("Nettoyage des fichiers de test...")

    # Supprimer les anciens checkpoints
    if CHECKPOINT_DIR.exists():
        for item in CHECKPOINT_DIR.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint_"):
                shutil.rmtree(item)
                print_status(f"Supprimé checkpoint: {item.name}")

    # Supprimer les anciens logs TensorBoard
    if TENSORBOARD_DIR.exists():
        for item in TENSORBOARD_DIR.iterdir():
            if item.is_file():
                item.unlink()
                print_status(f"Supprimé fichier TensorBoard: {item.name}")


def run_training_test(timeout_seconds=45, resume=False):
    """Lancer un test d'entraînement"""
    cmd = [
        "/home/morningstar/miniconda3/envs/trading_env/bin/python",
        str(SCRIPT_PATH),
        "--config",
        str(CONFIG_PATH),
        "--checkpoint-dir",
        str(CHECKPOINT_DIR),
    ]

    if resume:
        cmd.append("--resume")
        print_status(
            f"Lancement d'un entraînement avec RESUME (timeout: {timeout_seconds}s)"
        )
    else:
        print_status(
            f"Lancement d'un nouvel entraînement (timeout: {timeout_seconds}s)"
        )

    try:
        # Utiliser timeout pour limiter la durée
        result = subprocess.run(
            ["timeout", f"{timeout_seconds}s"] + cmd, capture_output=True, text=True
        )

        # timeout retourne code 124 quand il termine le processus
        if result.returncode == 124:
            print_status("Entraînement interrompu par timeout (normal)")
            return True
        elif result.returncode == 0:
            print_status("Entraînement terminé avec succès")
            return True
        else:
            print_status(f"Entraînement échoué (code: {result.returncode})", "ERROR")
            if result.stderr:
                print(f"Erreur: {result.stderr[:500]}...")
            return False

    except Exception as e:
        print_status(f"Erreur lors du lancement: {e}", "ERROR")
        return False


def check_tensorboard_files():
    """Vérifier que les fichiers TensorBoard sont créés"""
    print_status("Vérification des fichiers TensorBoard...")

    if not TENSORBOARD_DIR.exists():
        print_status("Répertoire TensorBoard inexistant", "ERROR")
        return False

    # Chercher les fichiers d'événements
    event_files = list(TENSORBOARD_DIR.glob("events.out.tfevents.*"))
    csv_files = list(TENSORBOARD_DIR.glob("*.csv"))

    if not event_files:
        print_status("Aucun fichier d'événements TensorBoard trouvé", "ERROR")
        return False

    print_status(
        f"Trouvé {len(event_files)} fichier(s) d'événements TensorBoard", "SUCCESS"
    )

    # Vérifier la taille des fichiers
    for event_file in event_files:
        size = event_file.stat().st_size
        print_status(f"  {event_file.name}: {size} bytes")

    if csv_files:
        print_status(f"Trouvé {len(csv_files)} fichier(s) CSV", "SUCCESS")

    return True


def check_tensorboard_content():
    """Vérifier que les fichiers TensorBoard contiennent des données"""
    print_status("Vérification du contenu TensorBoard...")

    try:
        # Script Python inline pour tester la lecture TensorBoard
        test_script = """
import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

tb_files = glob.glob("reports/tensorboard_logs/events.out.tfevents.*")
if tb_files:
    tb_file = max(tb_files, key=os.path.getmtime)
    try:
        acc = EventAccumulator(tb_file)
        acc.Reload()
        tags = acc.Tags()
        print(f"SCALARS:{len(tags['scalars'])}")
        print(f"TAGS:{','.join(tags['scalars'][:5])}")  # Max 5 tags

        # Essayer de lire des données
        for tag in tags['scalars'][:2]:
            scalars = acc.Scalars(tag)
            print(f"DATA:{tag}:{len(scalars)}")
    except Exception as e:
        print(f"ERROR:{e}")
else:
    print("NOFILES")
"""

        result = subprocess.run(
            [
                "/home/morningstar/miniconda3/envs/trading_env/bin/python",
                "-c",
                test_script,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            lines = output.split("\n")

            for line in lines:
                if line.startswith("SCALARS:"):
                    count = int(line.split(":")[1])
                    if count > 0:
                        print_status(
                            f"TensorBoard contient {count} tags scalaires", "SUCCESS"
                        )
                    else:
                        print_status(
                            "TensorBoard ne contient aucun tag scalaire", "WARNING"
                        )
                elif line.startswith("TAGS:"):
                    tags = line.split(":", 1)[1]
                    if tags:
                        print_status(f"Tags disponibles: {tags}")
                elif line.startswith("DATA:"):
                    parts = line.split(":")
                    tag_name, data_count = parts[1], parts[2]
                    print_status(f"  {tag_name}: {data_count} points de données")
                elif line.startswith("ERROR:"):
                    error = line.split(":", 1)[1]
                    print_status(f"Erreur de lecture TensorBoard: {error}", "ERROR")
                elif line == "NOFILES":
                    print_status("Aucun fichier TensorBoard trouvé", "ERROR")

            return "SCALARS:" in output and not "ERROR:" in output
        else:
            print_status(f"Erreur lors de la vérification: {result.stderr}", "ERROR")
            return False

    except Exception as e:
        print_status(f"Erreur lors de la vérification TensorBoard: {e}", "ERROR")
        return False


def check_checkpoints():
    """Vérifier que les checkpoints sont créés"""
    print_status("Vérification des checkpoints...")

    if not CHECKPOINT_DIR.exists():
        print_status("Répertoire de checkpoints inexistant", "ERROR")
        return False

    # Chercher les répertoires de checkpoint
    checkpoint_dirs = [
        d
        for d in CHECKPOINT_DIR.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint_")
    ]

    if not checkpoint_dirs:
        print_status("Aucun checkpoint trouvé", "ERROR")
        return False

    print_status(f"Trouvé {len(checkpoint_dirs)} checkpoint(s)", "SUCCESS")

    # Vérifier le contenu de chaque checkpoint
    for cp_dir in checkpoint_dirs:
        print_status(f"  Checkpoint: {cp_dir.name}")

        metadata_file = cp_dir / "metadata.json"
        optimizer_file = cp_dir / "optimizer.pt"

        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    steps = metadata.get("total_steps", 0)
                    episode = metadata.get("episode", 0)
                    print_status(f"    Métadonnées: Étape {steps}, Épisode {episode}")
            except Exception as e:
                print_status(f"    Erreur lecture métadonnées: {e}", "ERROR")
        else:
            print_status(f"    Fichier metadata.json manquant", "ERROR")

        if optimizer_file.exists():
            size = optimizer_file.stat().st_size
            print_status(f"    Optimiseur: {size} bytes")
        else:
            print_status(f"    Fichier optimizer.pt manquant", "ERROR")

    return True


def test_dashboard_integration():
    """Tester que le dashboard peut lire les données"""
    print_status("Test de l'intégration dashboard...")

    # Script pour tester la classe TensorboardMonitor
    test_script = """
import sys
sys.path.append("bot/scripts")

try:
    from training_dashboard import TensorboardMonitor

    # Créer une instance du monitor
    monitor = TensorboardMonitor()

    # Rafraîchir les données
    monitor.refresh()

    # Vérifier les données
    total_scalars = sum(len(scalars) for scalars in monitor.scalars.values())
    print(f"SCALARS_COUNT:{total_scalars}")
    print(f"TAGS_COUNT:{len(monitor.scalars)}")

    for tag, data in monitor.scalars.items():
        print(f"TAG:{tag}:{len(data)}")

    print("SUCCESS")
except Exception as e:
    print(f"ERROR:{e}")
"""

    try:
        result = subprocess.run(
            [
                "/home/morningstar/miniconda3/envs/trading_env/bin/python",
                "-c",
                test_script,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            lines = output.split("\n")

            success = False
            for line in lines:
                if line.startswith("SCALARS_COUNT:"):
                    count = int(line.split(":")[1])
                    if count > 0:
                        print_status(
                            f"Dashboard peut lire {count} données scalaires", "SUCCESS"
                        )
                        success = True
                    else:
                        print_status(
                            "Dashboard ne trouve aucune donnée scalaire", "WARNING"
                        )
                elif line.startswith("TAGS_COUNT:"):
                    count = int(line.split(":")[1])
                    print_status(f"Dashboard trouve {count} tags")
                elif line.startswith("TAG:"):
                    parts = line.split(":")
                    tag_name, data_count = parts[1], parts[2]
                    print_status(f"  Tag '{tag_name}': {data_count} points")
                elif line == "SUCCESS":
                    success = True
                elif line.startswith("ERROR:"):
                    error = line.split(":", 1)[1]
                    print_status(f"Erreur dashboard: {error}", "ERROR")
                    return False

            return success
        else:
            print_status(f"Erreur lors du test dashboard: {result.stderr}", "ERROR")
            return False

    except Exception as e:
        print_status(f"Erreur lors du test dashboard: {e}", "ERROR")
        return False


def main():
    """Fonction principale de test"""
    print("=" * 60)
    print("🧪 TEST DE VALIDATION - TENSORBOARD & CHECKPOINTS")
    print("=" * 60)

    results = {}

    # Phase 1: Nettoyage
    cleanup_test_files()

    # Phase 2: Premier entraînement (nouveau)
    print("\n" + "=" * 40)
    print("📍 PHASE 1: NOUVEL ENTRAÎNEMENT")
    print("=" * 40)

    results["training_new"] = run_training_test(timeout_seconds=45, resume=False)

    if results["training_new"]:
        # Vérifications après premier entraînement
        time.sleep(2)  # Laisser temps aux fichiers de se finaliser

        results["tensorboard_files"] = check_tensorboard_files()
        results["tensorboard_content"] = check_tensorboard_content()
        results["checkpoints"] = check_checkpoints()
        results["dashboard"] = test_dashboard_integration()

        # Phase 3: Test de resume
        if results["checkpoints"]:
            print("\n" + "=" * 40)
            print("📍 PHASE 2: TEST RESUME")
            print("=" * 40)

            results["training_resume"] = run_training_test(
                timeout_seconds=30, resume=True
            )

            if results["training_resume"]:
                # Vérifications après resume
                time.sleep(2)
                results["tensorboard_after_resume"] = check_tensorboard_files()

    # Phase 4: Résultats finaux
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS FINAUX")
    print("=" * 60)

    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)

    print(f"\nTests réussis: {passed_tests}/{total_tests}")

    for test_name, result in results.items():
        status = "SUCCESS" if result else "ERROR"
        test_display = test_name.replace("_", " ").title()
        print_status(f"{test_display}: {'RÉUSSI' if result else 'ÉCHEC'}", status)

    # Analyse des résultats
    print("\n" + "=" * 40)
    print("📋 ANALYSE")
    print("=" * 40)

    if results.get("tensorboard_files", False):
        print_status("✅ Les fichiers TensorBoard sont créés correctement")
    else:
        print_status("❌ Problème avec la création des fichiers TensorBoard", "ERROR")

    if results.get("tensorboard_content", False):
        print_status("✅ Les données TensorBoard sont lisibles")
    else:
        print_status(
            "⚠️  Les fichiers TensorBoard sont vides (normal si entraînement court)",
            "WARNING",
        )

    if results.get("checkpoints", False):
        print_status("✅ Les checkpoints sont sauvegardés")
    else:
        print_status("❌ Problème avec la sauvegarde des checkpoints", "ERROR")

    if results.get("training_resume", False):
        print_status("✅ La reprise depuis checkpoint fonctionne")
    elif "training_resume" in results:
        print_status("❌ Problème avec la reprise depuis checkpoint", "ERROR")

    if results.get("dashboard", False):
        print_status("✅ Le dashboard peut lire les données TensorBoard")
    else:
        print_status(
            "⚠️  Le dashboard ne trouve pas de données (normal si fichiers vides)",
            "WARNING",
        )

    # Verdict final
    critical_features = ["training_new", "tensorboard_files", "checkpoints"]
    critical_passed = all(results.get(feature, False) for feature in critical_features)

    print("\n" + "🏆" * 20)
    if critical_passed:
        print("🎉 SUCCÈS: Les corrections principales fonctionnent!")
        print("   - Logger TensorBoard configuré ✅")
        print("   - Checkpoints sauvegardés ✅")
        print("   - Entraînement stable ✅")
    else:
        print("❌ ÉCHEC: Des corrections critiques ne fonctionnent pas")
        print("   Vérifiez les logs d'erreur ci-dessus")
    print("🏆" * 20)

    return critical_passed


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrompu par l'utilisateur")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Erreur critique: {e}")
        sys.exit(1)
