#!/usr/bin/env python3
"""
Script pour nettoyer le projet en déplaçant les fichiers non essentiels vers un dossier de sauvegarde.
"""

import os
import shutil
from pathlib import Path

# Dossier racine du projet
PROJECT_ROOT = Path("/home/morningstar/Documents/trading/bot")
BACKUP_DIR = PROJECT_ROOT / "backup"

# Dossiers et fichiers essentiels à conserver
ESSENTIAL_PATHS = [
    # Fichiers racine
    "main_app.py",
    "run_adan.py",
    "cli.py",
    "config/",
    "scripts/train_parallel_agents.py",
    "scripts/training_dashboard.py",
    "src/adan_trading_bot/",
    "api/",

    # Fichiers de configuration
    "requirements.txt",
    "setup.py",
    "pyproject.toml",
    "README.md",
]

def create_backup_dir():
    """Crée le dossier de sauvegarde s'il n'existe pas."""
    BACKUP_DIR.mkdir(exist_ok=True, parents=True)
    print(f"Dossier de sauvegarde : {BACKUP_DIR}")

def is_essential(file_path: Path) -> bool:
    """Vérifie si un fichier est essentiel."""
    # Convertir en chemin relatif par rapport au projet
    rel_path = file_path.relative_to(PROJECT_ROOT)
    rel_path_str = str(rel_path).replace('\\', '/')

    # Vérifier si le fichier ou son dossier parent est dans les chemins essentiels
    for essential in ESSENTIAL_PATHS:
        essential = essential.rstrip('/')
        if rel_path_str == essential or rel_path_str.startswith(essential + '/'):
            return True
    return False

def move_to_backup(file_path: Path) -> None:
    """Déplace un fichier vers le dossier de sauvegarde."""
    rel_path = file_path.relative_to(PROJECT_ROOT)
    target_path = BACKUP_DIR / rel_path

    # Créer les dossiers parents si nécessaire
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Déplacer le fichier
    shutil.move(str(file_path), str(target_path))
    print(f"Déplacé : {rel_path}")

def clean_project():
    """Nettoie le projet en déplaçant les fichiers non essentiels."""
    create_backup_dir()

    # Parcourir tous les fichiers Python
    for root, _, files in os.walk(PROJECT_ROOT):
        root_path = Path(root)

        # Ignorer certains dossiers
        if any(part.startswith(('.', '__pycache__', 'venv', '.venv', 'env', '.env', 'backup'))
               for part in root_path.parts):
            continue

        for file in files:
            if file.endswith('.py'):
                file_path = root_path / file
                if not is_essential(file_path):
                    move_to_backup(file_path)

    print("\nNettoyage terminé. Les fichiers non essentiels ont été déplacés vers le dossier 'backup'.")
    print("Les fichiers suivants sont considérés comme essentiels :")
    for path in ESSENTIAL_PATHS:
        print(f"- {path}")

if __name__ == "__main__":
    print("Nettoyage du projet...")
    clean_project()
