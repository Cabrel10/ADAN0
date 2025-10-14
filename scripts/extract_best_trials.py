"""
Extraction des meilleurs essais d'optimisation Optuna par worker.
"""

import sqlite3
import pandas as pd
import json
from typing import Dict, List, Any
from datetime import datetime
import os

def extract_best_trials(db_path: str, output_dir: str = 'optuna_analysis'):
    """
    Extrait les meilleurs essais d'optimisation par worker.
    
    Args:
        db_path: Chemin vers la base de données Optuna
        output_dir: Répertoire de sortie pour les résultats
    """
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Connexion à la base de données
        conn = sqlite3.connect(db_path)
        
        # Requête pour récupérer les essais avec leurs paramètres et attributs
        # D'abord, on récupère les essais avec leurs valeurs
        query = """
        WITH trial_data AS (
            SELECT 
                t.trial_id,
                t.state,
                tv.value,
                t.datetime_start,
                t.datetime_complete
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.state = 'COMPLETE' AND tv.value IS NOT NULL
        ),
        params_agg AS (
            SELECT 
                trial_id,
                GROUP_CONCAT(param_name || ':' || param_value, '|') as params_str
            FROM trial_params
            GROUP BY trial_id
        ),
        attrs_agg AS (
            SELECT 
                trial_id,
                GROUP_CONCAT(key || ':' || value_json, '|') as attrs_str
            FROM trial_user_attributes
            GROUP BY trial_id
        )
        SELECT 
            td.trial_id,
            td.state,
            td.value,
            pa.params_str as params,
            aa.attrs_str as attributes
        FROM trial_data td
        LEFT JOIN params_agg pa ON td.trial_id = pa.trial_id
        LEFT JOIN attrs_agg aa ON td.trial_id = aa.trial_id
        ORDER BY td.value DESC
        """
        
        # Exécuter la requête
        df = pd.read_sql(query, conn)
        
        if df.empty:
            print("Aucun essai complet trouvé dans la base de données.")
            return
        
        # Traiter les paramètres et attributs
        trials_data = []
        for _, row in df.iterrows():
            # Extraire les paramètres
            params = {}
            if row['params']:
                for param in row['params'].split('|'):
                    if ':' in param:
                        key, value = param.split(':', 1)
                        try:
                            # Essayer de convertir les valeurs numériques
                            if '.' in value:
                                params[key] = float(value)
                            else:
                                params[key] = int(value)
                        except ValueError:
                            params[key] = value
            
            # Extraire les attributs
            attributes = {}
            if pd.notna(row['attributes']):
                for attr in row['attributes'].split('|'):
                    if ':' in attr:
                        key, value = attr.split(':', 1)
                        # Nettoyer les guillemets en trop
                        value = value.strip('"')
                        try:
                            # Essayer de parser le JSON
                            attributes[key] = json.loads(value)
                        except json.JSONDecodeError:
                            attributes[key] = value
            
            # Ajouter les données du trial
            trial_data = {
                'trial_id': row['trial_id'],
                'value': row['value'],
                'params': params,
                'attributes': attributes
            }
            trials_data.append(trial_data)
        
        # Grouper les essais par worker
        trials_by_worker = {}
        for trial in trials_data:
            worker = trial['attributes'].get('worker', 'unknown')
            if worker not in trials_by_worker:
                trials_by_worker[worker] = []
            trials_by_worker[worker].append(trial)
        
        # Trier les essais par valeur (meilleur en premier) pour chaque worker
        best_trials = {}
        for worker, trials in trials_by_worker.items():
            sorted_trials = sorted(trials, key=lambda x: x['value'], reverse=True)
            best_trials[worker] = sorted_trials[0]  # Meilleur essai par worker
        
        # Afficher les résultats
        print("\n" + "="*80)
        print("MEILLEURS HYPERPARAMÈTRES PAR WORKER")
        print("="*80)
        
        for worker, trial in best_trials.items():
            print(f"\n{'='*40}")
            print(f"WORKER: {worker.upper()}")
            print(f"Valeur de l'objectif: {trial['value']:.4f}")
            print(f"ID de l'essai: {trial['trial_id']}")
            
            print("\nParamètres:")
            for key, value in trial['params'].items():
                print(f"  {key}: {value}")
            
            print("\nAttributs:")
            for key, value in trial['attributes'].items():
                if key not in ['worker']:  # On a déjà affiché le worker
                    print(f"  {key}: {value}")
        
        # Sauvegarder les résultats dans un fichier JSON
        output_path = os.path.join(output_dir, 'best_hyperparameters.json')
        with open(output_path, 'w') as f:
            json.dump(best_trials, f, indent=2, default=str)
        
        print(f"\nRésultats sauvegardés dans: {output_path}")
        
        # Fermer la connexion
        conn.close()
        
    except Exception as e:
        print(f"Erreur lors de l'extraction des essais: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extraction des meilleurs essais Optuna par worker')
    parser.add_argument('--db-path', type=str, default='hyperparam_optimization.db',
                      help='Chemin vers la base de données Optuna')
    parser.add_argument('--output-dir', type=str, default='optuna_analysis',
                      help='Répertoire de sortie pour les résultats')
    
    args = parser.parse_args()
    
    # Vérifier si le fichier existe
    if not os.path.exists(args.db_path):
        alt_path = os.path.join('scripts', args.db_path)
        if os.path.exists(alt_path):
            args.db_path = alt_path
        else:
            print(f"Erreur: Le fichier {args.db_path} n'existe pas.")
            exit(1)
    
    print(f"Extraction des meilleurs essais depuis: {args.db_path}")
    extract_best_trials(args.db_path, args.output_dir)
