"""Script pour examiner la structure de la base de données Optuna."""

import sqlite3
import pandas as pd

def check_db_structure(db_path):
    """Affiche les tables et leur structure dans la base de données."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Obtenir la liste des tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print("Tables dans la base de données:")
        for table in tables:
            table_name = table[0]
            print(f"\nTable: {table_name}")
            print("-" * (len(table_name) + 7))
            
            # Obtenir les colonnes de la table
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            if columns:
                print("Colonnes:")
                for col in columns:
                    print(f"  - {col[1]} ({col[2]})")
            
            # Afficher un aperçu des données
            try:
                df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 3", conn)
                print("\nAperçu des données (3 premières lignes):")
                print(df)
                print(f"\nNombre total d'entrées: {len(pd.read_sql_query(f'SELECT * FROM {table_name}', conn))}")
            except Exception as e:
                print(f"Impossible d'afficher les données: {e}")
        
        conn.close()
        
    except Exception as e:
        print(f"Erreur lors de l'analyse de la base de données: {e}")

if __name__ == "__main__":
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'scripts/hyperparam_optimization.db'
    check_db_structure(db_path)
