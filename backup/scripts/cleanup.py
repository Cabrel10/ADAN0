#!/usr/bin/env python3
"""
Script de nettoyage du projet.
Supprime les fichiers temporaires, les caches et nettoie l'environnement.
"""
import os
import shutil
from pathlib import Path

def clean_pycache():
    """Supprime les dossiers __pycache__ et fichiers .pyc."""
    print("🧹 Nettoyage des fichiers Python compilés...")
    for root, dirs, files in os.walk('.'):
        # Supprimer les dossiers __pycache__
        if "__pycache__" in dirs:
            dir_path = os.path.join(root, "__pycache__")
            print(f"  Suppression de {dir_path}")
            shutil.rmtree(dir_path, ignore_errors=True)
            dirs.remove("__pycache__")

        # Supprimer les fichiers .pyc
        for file in files:
            if file.endswith('.pyc') or file.endswith('.pyo') or file.endswith('.pyd'):
                file_path = os.path.join(root, file)
                print(f"  Suppression de {file_path}")
                try:
                    os.unlink(file_path)
                except Exception as e:
                    print(f"  ❌ Erreur lors de la suppression de {file_path}: {e}")

def clean_build_dirs():
    """Supprime les dossiers de build et de distribution."""
    print("\n🧹 Nettoyage des dossiers de build...")
    build_dirs = ["build", "dist", "*.egg-info"]

    for pattern in build_dirs:
        for path in Path('.').glob(f"**/{pattern}"):
            if path.is_dir():
                print(f"  Suppression de {path}")
                shutil.rmtree(path, ignore_errors=True)

def clean_test_outputs():
    """Supprime les fichiers de sortie des tests."""
    print("\n🧹 Nettoyage des sorties de tests...")
    test_dirs = ["tests/outputs", "tests/__pycache__"]

    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            print(f"  Suppression de {test_dir}")
            shutil.rmtree(test_dir, ignore_errors=True)

def clean_logs():
    """Supprime les fichiers de logs."""
    print("\n🧹 Nettoyage des logs...")
    log_dirs = ["logs"]

    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            print(f"  Suppression des logs dans {log_dir}")
            for filename in os.listdir(log_dir):
                file_path = os.path.join(log_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"  ❌ Erreur lors de la suppression de {file_path}: {e}")

def main():
    print("🚀 Démarrage du nettoyage du projet...")

    # Exécution des différentes étapes de nettoyage
    clean_pycache()
    clean_build_dirs()
    clean_test_outputs()
    clean_logs()

    print("\n✅ Nettoyage terminé avec succès !")

if __name__ == "__main__":
    main()
