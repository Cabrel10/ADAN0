#!/usr/bin/env python3
"""
Script pour examiner la structure des fichiers CSV.
"""
import os
import pandas as pd
from pathlib import Path

def check_csv_structure(filepath):
    """Affiche les premières lignes et les métadonnées d'un fichier CSV."""
    try:
        # Essayer de lire le fichier avec différentes options d'encodage
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                # Lire uniquement la première ligne pour vérifier l'encodage
                with open(filepath, 'r', encoding=encoding) as f:
                    first_line = f.readline().strip()
                    print(f"\nEncodage réussi avec {encoding}:")
                    print(f"Première ligne: {first_line}")
                
                # Lire tout le fichier avec l'encodage qui fonctionne
                df = pd.read_csv(filepath, encoding=encoding)
                print("\nAperçu du DataFrame:")
                print(df.head())
                print("\nColonnes:", df.columns.tolist())
                print("\nTypes de données:")
                print(df.dtypes)
                print("\nNombre de lignes:", len(df))
                return True
                
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Erreur avec l'encodage {encoding}: {e}")
                continue
        
        print("Aucun encodage standard n'a fonctionné. Tentative avec des options avancées...")
        
        # Essayer avec différentes options de séparateur
        separators = [',', ';', '\t', '|']
        for sep in separators:
            try:
                df = pd.read_csv(filepath, sep=sep, engine='python')
                print(f"\nSéparateur détecté: {sep}")
                print("Aperçu du DataFrame:")
                print(df.head())
                print("\nColonnes:", df.columns.tolist())
                return True
            except Exception as e:
                print(f"Échec avec le séparateur {sep}: {e}")
                continue
        
        # Si on arrive ici, tous les essais ont échoué
        print("\nImpossible de lire le fichier avec les paramètres standards.")
        return False
        
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier {filepath}: {e}")
        return False

def main():
    # Dossier contenant les fichiers CSV
    data_dir = Path("/home/morningstar/Documents/trading/data/raw/")
    
    # Vérifier si le dossier existe
    if not data_dir.exists():
        print(f"Le dossier {data_dir} n'existe pas.")
        return
    
    # Lister tous les fichiers CSV
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"Aucun fichier CSV trouvé dans {data_dir}")
        return
    
    print(f"Fichiers CSV trouvés: {len(csv_files)}")
    
    # Vérifier chaque fichier
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n{'='*50}")
        print(f"Fichier {i}/{len(csv_files)}: {csv_file.name}")
        print(f"Taille: {os.path.getsize(csv_file) / 1024:.2f} KB")
        
        # Vérifier la structure du fichier
        check_csv_structure(csv_file)

if __name__ == "__main__":
    main()
