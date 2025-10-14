"""
Script pour trouver les meilleurs paramètres individuels pour chaque worker
selon les critères : Sharpe > 0, Win Rate > 50%, Profit Factor > 1.5
"""

import sqlite3
import pandas as pd
from typing import Dict, List, Any


def get_best_params_for_worker(db_path: str, worker_id: int) -> pd.DataFrame:
    """Trouve les meilleurs paramètres pour un worker spécifique."""
    try:
        conn = sqlite3.connect(db_path)
        
        # Requête pour obtenir les essais qui répondent aux critères pour ce worker
        query = f"""
        SELECT 
            t.trial_id,
            t.value as objective_value,
            GROUP_CONCAT(p.param_name || '=' || p.param_value, '|') as params,
            MAX(CASE WHEN tua.key = 'worker_{worker_id}_sharpe_ratio' THEN tua.value_json ELSE NULL END) as sharpe_ratio,
            MAX(CASE WHEN tua.key = 'worker_{worker_id}_win_rate' THEN tua.value_json ELSE NULL END) as win_rate,
            MAX(CASE WHEN tua.key = 'worker_{worker_id}_profit_factor' THEN tua.value_json ELSE NULL END) as profit_factor
        FROM trial_values t
        JOIN trial_params p ON t.trial_id = p.trial_id
        JOIN trial_user_attributes tua ON t.trial_id = tua.trial_id
        WHERE tua.key IN (
            'worker_{worker_id}_sharpe_ratio',
            'worker_{worker_id}_win_rate',
            'worker_{worker_id}_profit_factor'
        )
        GROUP BY t.trial_id
        HAVING sharpe_ratio > 0 
           AND win_rate > 50 
           AND profit_factor > 1.5
        ORDER BY sharpe_ratio DESC
        LIMIT 1
        """
        
        df = pd.read_sql_query(query, conn)
        
        if not df.empty:
            # Convertir la chaîne de paramètres en dictionnaire
            params_str = df['params'].iloc[0]
            params_dict = {}
            if params_str:
                for param in params_str.split('|'):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        params_dict[key] = value
            
            # Créer un nouveau DataFrame avec les paramètres
            result = pd.DataFrame({
                'worker_id': [f'worker_{worker_id}'],
                'trial_id': df['trial_id'].iloc[0],
                'sharpe_ratio': float(df['sharpe_ratio'].iloc[0]),
                'win_rate': float(df['win_rate'].iloc[0]),
                'profit_factor': float(df['profit_factor'].iloc[0])
            })
            
            # Ajouter les paramètres comme colonnes
            for key, value in params_dict.items():
                result[key] = value
                
            return result
        
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Erreur lors de la recherche des paramètres pour worker_{worker_id}: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Trouve les meilleurs paramètres pour chaque worker")
    parser.add_argument('--db-path', type=str, default='scripts/hyperparam_optimization.db',
                      help='Chemin vers la base de données Optuna')
    parser.add_argument('--output', type=str, default='analysis_results/best_worker_params.csv',
                      help='Fichier de sortie pour les résultats')
    
    args = parser.parse_args()
    
    print("Recherche des meilleurs paramètres pour chaque worker...")
    print("Critères: Sharpe > 0, Win Rate > 50%, Profit Factor > 1.5\n")
    
    # Créer un DataFrame pour stocker les résultats
    all_results = []
    
    # Analyser chaque worker
    for worker_id in [0, 1, 2, 3]:
        print(f"Traitement du worker_{worker_id}...")
        result = get_best_params_for_worker(args.db_path, worker_id)
        if not result.empty:
            all_results.append(result)
    
    if not all_results:
        print("\nAucun essai ne répond aux critères pour aucun worker.")
        return
    
    # Fusionner tous les résultats
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Afficher les résultats
    print("\n" + "="*80)
    print("MEILLEURS PARAMÈTRES PAR WORKER")
    print("="*80)
    
    for _, row in final_df.iterrows():
        worker = row['worker_id']
        print(f"\n{worker.upper()} (Essai #{row['trial_id']}):")
        print(f"- Ratio de Sharpe: {row['sharpe_ratio']:.2f}")
        print(f"- Taux de réussite: {row['win_rate']:.1f}%")
        print(f"- Profit Factor: {row['profit_factor']:.2f}")
        
        # Afficher les paramètres spécifiques
        print("\nParamètres optimaux:")
        param_columns = [col for col in row.index if col not in 
                        ['worker_id', 'trial_id', 'sharpe_ratio', 'win_rate', 'profit_factor']]
        
        for param in param_columns:
            if not pd.isna(row[param]):
                print(f"- {param}: {row[param]}")
    
    # Sauvegarder les résultats
    import os
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    final_df.to_csv(args.output, index=False)
    print(f"\nRésultats sauvegardés dans: {args.output}")


if __name__ == "__main__":
    main()
