#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
from pathlib import Path

def show_parquet_heads(data_dir):
    """Affiche les deux premières lignes de chaque fichier parquet."""
    pattern = str(Path(data_dir) / '**' / '*.parquet')
    parquet_files = glob.glob(pattern, recursive=True)
    
    if not parquet_files:
        print(f"Aucun fichier parquet trouvé dans {data_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"ANALYSE DES FICHIERS PARQUET ({len(parquet_files)} fichiers trouvés)")
    print(f"{'='*80}\n")
    
    for filepath in sorted(parquet_files):
        try:
            # Lire uniquement les deux premières lignes
            df = pd.read_parquet(filepath).head(2)
            
            print(f"\n{'='*60}")
            print(f"FICHIER: {filepath}")
            print(f"COLONNES: {', '.join(df.columns)}")
            print(f"LIGNES: {len(df)}")
            print(f"{'='*60}")
            print(df)
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"\n{'!'*60}")
            print(f"ERREUR avec le fichier: {filepath}")
            print(f"Erreur: {str(e)}")
            print(f"{'!'*60}\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print(f"Utilisation: {sys.argv[0]} <répertoire_des_données>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    if not os.path.isdir(data_dir):
        print(f"Erreur: Le répertoire {data_dir} n'existe pas")
        sys.exit(1)
    
    show_parquet_heads(data_dir)
