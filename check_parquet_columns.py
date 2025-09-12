#!/usr/bin/env python3
"""
Script pour vérifier le nombre de colonnes dans les fichiers Parquet.
Affiche les fichiers qui n'ont pas le nombre attendu de colonnes.
"""
import os
import pandas as pd
from pathlib import Path

def check_parquet_columns(directory, expected_columns=14):
    """Vérifie le nombre de colonnes dans tous les fichiers Parquet du répertoire."""
    print(f"Vérification des fichiers dans: {directory}")
    print("-" * 80)

    # Parcourir tous les fichiers Parquet
    parquet_files = list(Path(directory).rglob('*.parquet'))
    print(f"Nombre total de fichiers Parquet trouvés: {len(parquet_files)}")

    # Dictionnaire pour stocker les résultats
    results = {
        'correct': [],
        'incorrect': []
    }

    for file_path in parquet_files:
        try:
            # Lire uniquement les métadonnées pour plus d'efficacité
            df = pd.read_parquet(file_path, engine='pyarrow')
            num_columns = len(df.columns)

            if num_columns == expected_columns:
                results['correct'].append(str(file_path))
            else:
                results['incorrect'].append({
                    'path': str(file_path),
                    'columns': num_columns,
                    'column_names': list(df.columns)
                })

        except Exception as e:
            print(f"Erreur avec le fichier {file_path}: {str(e)}")

    return results

def main():
    # Dossiers à vérifier
    base_dir = "/home/morningstar/Documents/trading/bot/data/processed"
    directories = [
        os.path.join(base_dir, "indicators"),
        os.path.join(base_dir, "backups")
    ]

    for directory in directories:
        if not os.path.exists(directory):
            print(f"Le dossier {directory} n'existe pas.")
            continue

        print("\n" + "="*80)
        print(f"Vérification du dossier: {directory}")
        print("="*80)

        results = check_parquet_columns(directory)

        print(f"\nRésultats pour {directory}:")
        print(f"- Fichiers avec le bon nombre de colonnes: {len(results['correct'])}")
        print(f"- Fichiers avec un nombre incorrect de colonnes: {len(results['incorrect'])}")

        if results['incorrect']:
            print("\nFichiers avec un nombre incorrect de colonnes:")
            for item in results['incorrect']:
                print(f"\nFichier: {item['path']}")
                print(f"Nombre de colonnes: {item['columns']}")
                print(f"Colonnes: {item['column_names']}")

        print("\n" + "-"*80 + "\n")

if __name__ == "__main__":
    main()
