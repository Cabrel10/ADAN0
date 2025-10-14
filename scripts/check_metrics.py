"""Script pour examiner les métriques stockées dans la base de données."""

import sqlite3
import pandas as pd
import json

def check_metrics(db_path):
    """Affiche les métriques disponibles dans la base de données."""
    try:
        conn = sqlite3.connect(db_path)
        
        # Vérifier les clés uniques dans trial_user_attributes
        query = """
        SELECT DISTINCT key 
        FROM trial_user_attributes 
        WHERE key LIKE '%sharpe%' 
           OR key LIKE '%win_rate%' 
           OR key LIKE '%profit_factor%'
        """
        metrics = pd.read_sql_query(query, conn)
        print("Métriques disponibles:")
        print(metrics)
        
        # Afficher un exemple de données pour chaque métrique
        if not metrics.empty:
            for _, row in metrics.iterrows():
                metric = row['key']
                print(f"\nValeurs pour la métrique '{metric}':")
                query = f"""
                SELECT trial_id, value_json 
                FROM trial_user_attributes 
                WHERE key = ? 
                LIMIT 3
                """
                values = pd.read_sql_query(query, conn, params=(metric,))
                print(values)
        
        conn.close()
        
    except Exception as e:
        print(f"Erreur lors de l'analyse des métriques: {e}")

if __name__ == "__main__":
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'scripts/hyperparam_optimization.db'
    check_metrics(db_path)
