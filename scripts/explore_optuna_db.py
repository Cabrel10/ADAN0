"""
Script pour explorer le contenu d'une base de données Optuna.
"""

import sqlite3
import pandas as pd
import os

def explore_database(db_path: str):
    """Explore le contenu de la base de données Optuna."""
    try:
        # Connexion à la base de données
        conn = sqlite3.connect(db_path)
        
        # Récupérer la liste des tables
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
        print("Tables dans la base de données:")
        print(tables)
        
        # Afficher le contenu de chaque table
        for table in tables['name']:
            print(f"\nContenu de la table '{table}':")
            try:
                df = pd.read_sql(f"SELECT * FROM {table} LIMIT 5;", conn)
                print(df.head())
                print(f"Nombre total d'entrées: {len(pd.read_sql(f'SELECT * FROM {table}', conn))}")
            except Exception as e:
                print(f"Erreur lors de la lecture de la table {table}: {e}")
        
        # Vérifier les études disponibles
        if 'studies' in tables['name'].values:
            print("\nÉtudes disponibles:")
            studies = pd.read_sql("SELECT * FROM studies;", conn)
            print(studies)
            
            if not studies.empty:
                study_id = studies['study_id'].iloc[0]
                print(f"\nDétails des essais pour l'étude ID {study_id}:")
                trials = pd.read_sql(f"SELECT * FROM trials WHERE study_id = {study_id};", conn)
                print(trials.head())
        
        conn.close()
        
    except Exception as e:
        print(f"Erreur lors de l'exploration de la base de données: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Explorer une base de données Optuna')
    parser.add_argument('--db-path', type=str, default='hyperparam_optimization.db',
                      help='Chemin vers la base de données Optuna')
    
    args = parser.parse_args()
    
    # Vérifier si le fichier existe
    if not os.path.exists(args.db_path):
        print(f"Erreur: Le fichier {args.db_path} n'existe pas.")
        print("Recherche dans le répertoire scripts/...")
        alt_path = os.path.join('scripts', args.db_path)
        if os.path.exists(alt_path):
            print(f"Fichier trouvé à {alt_path}")
            args.db_path = alt_path
        else:
            print("Aucune base de données trouvée.")
            exit(1)
    
    print(f"Exploration de la base de données: {args.db_path}")
    explore_database(args.db_path)
