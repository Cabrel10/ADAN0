"""Script pour lister toutes les métriques disponibles dans la base de données."""

import sqlite3
import pandas as pd

def list_all_metrics(db_path: str):
    """Liste toutes les métriques uniques disponibles dans la base de données."""
    try:
        conn = sqlite3.connect(db_path)
        
        # Obtenir toutes les clés uniques des attributs utilisateur
        query = """
        SELECT DISTINCT key 
        FROM trial_user_attributes 
        ORDER BY key
        """
        metrics = pd.read_sql_query(query, conn)
        
        print("Toutes les métriques disponibles dans la base de données:")
        print("-" * 50)
        
        if metrics.empty:
            print("Aucune métrique trouvée.")
        else:
            # Afficher les 10 premières métriques
            print("\nQuelques exemples de métriques:")
            for i, row in metrics.head(10).iterrows():
                print(f"- {row['key']}")
            
            # Compter et afficher le nombre total de métriques uniques
            print(f"\nTotal des métriques uniques: {len(metrics)}")
            
            # Vérifier les catégories de métriques
            print("\nCatégories de métriques détectées:")
            categories = set()
            for _, row in metrics.iterrows():
                key = row['key']
                if '_' in key:
                    # Essayer d'extraire la catégorie (partie après le dernier _)
                    base = key.rsplit('_', 1)[0]
                    categories.add(base)
            
            print("\nQuelques exemples de catégories de métriques:")
            for i, cat in enumerate(sorted(categories)[:10], 1):
                print(f"{i}. {cat}")
            print(f"... et {len(categories) - 10} autres catégories")
        
        conn.close()
        
    except Exception as e:
        print(f"Erreur lors de l'analyse des métriques: {e}")

if __name__ == "__main__":
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'scripts/hyperparam_optimization.db'
    list_all_metrics(db_path)
