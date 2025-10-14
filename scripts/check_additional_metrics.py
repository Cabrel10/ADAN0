"""Script pour vérifier les métriques supplémentaires comme le drawdown, le capital et le P&L."""

import sqlite3
import pandas as pd

def check_additional_metrics(db_path: str):
    """Vérifie la présence de métriques supplémentaires dans la base de données."""
    try:
        conn = sqlite3.connect(db_path)
        
        # Vérifier les clés uniques dans trial_user_attributes
        query = """
        SELECT DISTINCT key 
        FROM trial_user_attributes 
        WHERE key LIKE '%drawdown%' 
           OR key LIKE '%capital%' 
           OR key LIKE '%pnl%'
           OR key LIKE '%profit%loss%'
           OR key LIKE '%equity%'
        """
        metrics = pd.read_sql_query(query, conn)
        
        print("Métriques supplémentaires disponibles:")
        if metrics.empty:
            print("Aucune métrique supplémentaire trouvée (drawdown, capital, P&L, equity)")
        else:
            print(metrics)
            
            # Afficher un exemple de données pour chaque métrique trouvée
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
    check_additional_metrics(db_path)
