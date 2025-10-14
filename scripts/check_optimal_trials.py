"""Script pour vérifier les essais optimaux pour chaque worker."""

import sqlite3
from typing import Dict, List, Any

import pandas as pd


def get_trial_details(db_path: str, trial_ids: List[int]) -> pd.DataFrame:
    """Récupère les détails des essais spécifiés."""
    try:
        conn = sqlite3.connect(db_path)
        
        # Récupérer les paramètres des essais
        placeholders = ','.join(['?'] * len(trial_ids))
        params_query = f"""
        SELECT 
            trial_id,
            param_name,
            param_value
        FROM trial_params
        WHERE trial_id IN ({placeholders})
        ORDER BY trial_id, param_name
        """
        
        params_df = pd.read_sql_query(params_query, conn, params=trial_ids)
        
        # Récupérer les valeurs des essais
        values_query = f"""
        SELECT 
            trial_id,
            value as objective_value
        FROM trial_values
        WHERE trial_id IN ({placeholders})
        """
        
        values_df = pd.read_sql_query(values_query, conn, params=trial_ids)
        
        # Récupérer les métriques des essais
        metrics_query = f"""
        SELECT 
            trial_id,
            key,
            value_json
        FROM trial_user_attributes
        WHERE trial_id IN ({placeholders})
        AND (key LIKE '%sharpe%' 
             OR key LIKE '%win_rate%' 
             OR key LIKE '%profit_factor%')
        """
        
        metrics_df = pd.read_sql_query(metrics_query, conn, params=trial_ids)
        
        # Fusionner les données
        if params_df.empty:
            conn.close()
            return pd.DataFrame()
            
        # Pivoter les paramètres
        params_pivot = params_df.pivot(
            index='trial_id', 
            columns='param_name', 
            values='param_value'
        )
        
        # Fusionner avec les valeurs des essais
        result_df = pd.merge(values_df, params_pivot, on='trial_id', how='left')
        
        # Fusionner avec les métriques si disponibles
        if not metrics_df.empty:
            metrics_pivot = metrics_df.pivot(
                index='trial_id', 
                columns='key', 
                values='value_json'
            )
            result_df = pd.merge(
                result_df, 
                metrics_pivot, 
                on='trial_id', 
                how='left'
            )
        
        conn.close()
        return result_df
        
    except Exception as e:
        print(f"Erreur lors de la récupération des détails des essais: {e}")
        return pd.DataFrame()

def get_best_trials(db_path: str) -> Dict[str, Dict[str, Any]]:
    """Récupère les meilleurs essais pour chaque worker selon nos critères."""
    try:
        conn = sqlite3.connect(db_path)
        
        # Dictionnaire pour stocker les meilleurs essais par worker
        best_trials = {}
        
        # Pour chaque worker, trouver les essais qui répondent aux critères
        for worker_id in [0, 1, 2, 3]:
            query = f"""
            SELECT 
                tv.trial_id,
                tv.value as objective_value,
                MAX(CASE WHEN tua.key = 'worker_{0}_sharpe_ratio' THEN tua.value_json ELSE NULL END) as sharpe_ratio,
                MAX(CASE WHEN tua.key = 'worker_{0}_win_rate' THEN tua.value_json ELSE NULL END) as win_rate,
                MAX(CASE WHEN tua.key = 'worker_{0}_profit_factor' THEN tua.value_json ELSE NULL END) as profit_factor
            FROM trial_values tv
            LEFT JOIN trial_user_attributes tua ON tv.trial_id = tua.trial_id
            WHERE (tua.key = 'worker_{0}_sharpe_ratio' AND CAST(tua.value_json AS REAL) > 0)
               OR (tua.key = 'worker_{0}_win_rate' AND CAST(tua.value_json AS REAL) > 50)
               OR (tua.key = 'worker_{0}_profit_factor' AND CAST(tua.value_json AS REAL) > 1.5)
            GROUP BY tv.trial_id
            HAVING sharpe_ratio > 0 AND win_rate > 50 AND profit_factor > 1.5
            ORDER BY sharpe_ratio DESC
            LIMIT 1
            """.format(worker_id)
            
            df = pd.read_sql_query(query, conn)
            
            if not df.empty:
                best_trials[f'worker_{worker_id}'] = df.iloc[0].to_dict()
        
        conn.close()
        return best_trials
        
    except Exception as e:
        print(f"Erreur lors de la recherche des meilleurs essais: {e}")
        return {}

def print_trial_comparison(trial_details: pd.DataFrame, worker_id: int, trial_id: int):
    """Affiche une comparaison des paramètres et métriques d'un essai."""
    trial = trial_details[trial_details['trial_id'] == trial_id]
    
    if trial.empty:
        print(f"\nAucune donnée trouvée pour l'essai #{trial_id} du WORKER_{worker_id}")
        return
    
    print(f"\n{'='*80}")
    print(f"ANALYSE DE L'ESSAI #{trial_id} POUR WORKER_{worker_id}")
    print(f"{'='*80}")
    
    # Afficher les paramètres
    print("\nPARAMÈTRES:")
    params = trial.drop(columns=['trial_id', 'objective_value'], errors='ignore')
    params = params.loc[:, ~params.columns.str.contains('sharpe|win_rate|profit_factor')]
    
    for col in params.columns:
        if not pd.isna(params[col].iloc[0]):
            print(f"- {col}: {params[col].iloc[0]}")
    
    # Afficher les métriques
    print("\nMÉTRIQUES:")
    
    # Vérifier et afficher les métriques pour chaque worker
    for w_id in [0, 1, 2, 3]:
        sharpe_col = f'worker_{w_id}_sharpe_ratio'
        win_rate_col = f'worker_{w_id}_win_rate'
        pf_col = f'worker_{w_id}_profit_factor'
        
        if sharpe_col in trial.columns and not pd.isna(trial[sharpe_col].iloc[0]):
            print(f"\nWORKER {w_id}:")
            print(f"  - Ratio de Sharpe: {trial[sharpe_col].iloc[0]}")
            print(f"  - Taux de réussite: {trial[win_rate_col].iloc[0] if win_rate_col in trial.columns else 'N/A'}%")
            print(f"  - Profit Factor: {trial[pf_col].iloc[0] if pf_col in trial.columns else 'N/A'}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vérifie les essais optimaux pour chaque worker")
    parser.add_argument('--db-path', type=str, default='scripts/hyperparam_optimization.db',
                      help='Chemin vers la base de données Optuna')
    
    args = parser.parse_args()
    
    # 1. Obtenir les meilleurs essais selon nos critères
    print("\nRECHERCHE DES MEILLEURS ESSAIS PAR WORKER...")
    best_trials = get_best_trials(args.db_path)
    
    if not best_trials:
        print("Aucun essai optimal trouvé selon les critères (Sharpe > 0, Win Rate > 50%, Profit Factor > 1.5)")
    else:
        print("\nMEILLEURS ESSAIS TROUVÉS PAR WORKER:")
        for worker, trial in best_trials.items():
            print(f"\n{worker.upper()}:")
            print(f"- Essai #{trial['trial_id']}")
            print(f"- Ratio de Sharpe: {trial['sharpe_ratio']}")
            print(f"- Taux de réussite: {trial['win_rate']}%")
            print(f"- Profit Factor: {trial['profit_factor']}")
    
    # 2. Vérifier les essais spécifiques mentionnés
    mentioned_trials = {
        'worker_0': 61,
        'worker_1': 26,
        'worker_2': 61,
        'worker_3': 61
    }
    
    print("\n" + "="*80)
    print("VÉRIFICATION DES ESSAIS MENTIONNÉS")
    print("="*80)
    
    # Récupérer les détails des essais mentionnés
    trial_ids = list(mentioned_trials.values())
    trial_details = get_trial_details(args.db_path, trial_ids)
    
    if trial_details.empty:
        print("\nImpossible de récupérer les détails des essais mentionnés.")
    else:
        # Afficher les détails de chaque essai mentionné
        for worker, trial_id in mentioned_trials.items():
            print_trial_comparison(trial_details, worker.split('_')[1], trial_id)
    
    # 3. Comparaison avec les meilleurs essais trouvés
    if best_trials:
        print("\n" + "="*80)
        print("COMPARAISON AVEC LES MEILLEURS ESSAIS TROUVÉS")
        print("="*80)
        
        for worker, best_trial in best_trials.items():
            best_trial_id = best_trial['trial_id']
            worker_id = int(worker.split('_')[1])  # Extraire l'ID du worker (0-3)
            mentioned_trial_id = mentioned_trials.get(worker, None)
            
            if mentioned_trial_id and best_trial_id != mentioned_trial_id:
                mentioned_trial = trial_details[trial_details['trial_id'] == mentioned_trial_id]
                
                if not mentioned_trial.empty:
                    sharpe_col = f'worker_{worker_id}_sharpe_ratio'
                    win_rate_col = f'worker_{worker_id}_win_rate'
                    pf_col = f'worker_{worker_id}_profit_factor'
                    
                    mentioned_sharpe = mentioned_trial[sharpe_col].iloc[0] if sharpe_col in mentioned_trial.columns else 'N/A'
                    mentioned_win_rate = mentioned_trial[win_rate_col].iloc[0] if win_rate_col in mentioned_trial.columns else 'N/A'
                    mentioned_pf = mentioned_trial[pf_col].iloc[0] if pf_col in mentioned_trial.columns else 'N/A'
                    
                    print(f"\nPour {worker.upper()}, l'essai #{best_trial_id} semble meilleur que l'essai mentionné #{mentioned_trial_id}:")
                    print(f"- Meilleur Sharpe: {best_trial['sharpe_ratio']} (vs {mentioned_sharpe})")
                    print(f"- Meilleur Win Rate: {best_trial['win_rate']}% (vs {mentioned_win_rate}%)")
                    print(f"- Meilleur Profit Factor: {best_trial['profit_factor']} (vs {mentioned_pf})")
