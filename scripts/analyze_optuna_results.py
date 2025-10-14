"""
Analyse des résultats d'optimisation Optuna pour extraire les meilleurs hyperparamètres.
Ce script analyse la base de données d'Optuna et extrait les 4 meilleurs jeux d'hyperparamètres,
un pour chaque profil de worker (conservateur, modéré, agressif, adaptatif).
"""

import sqlite3
import pandas as pd
import json
from typing import Dict, List, Any
import optuna
from optuna.storages import RDBStorage
from pathlib import Path
import os

def get_best_trials_per_worker(study_name: str, storage_url: str, n_trials: int = 1) -> Dict[str, List[Dict[str, Any]]]:
    """
    Récupère les meilleurs essais pour chaque worker à partir d'une étude Optuna.
    
    Args:
        study_name: Nom de l'étude Optuna
        storage_url: URL de stockage de la base de données
        n_trials: Nombre de meilleurs essais à récupérer par worker
        
    Returns:
        Dictionnaire avec les meilleurs essais par worker
    """
    # Connexion à la base de données
    storage = RDBStorage(url=storage_url)
    
    # Charger l'étude
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception as e:
        print(f"Erreur lors du chargement de l'étude {study_name}: {e}")
        return {}
    
    # Récupérer tous les essais
    trials = study.trials
    
    # Grouper les essais par worker
    trials_by_worker = {}
    for trial in trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
            
        worker = trial.user_attrs.get('worker', 'unknown')
        if worker not in trials_by_worker:
            trials_by_worker[worker] = []
        trials_by_worker[worker].append(trial)
    
    # Trier les essais par valeur (meilleur en premier) pour chaque worker
    best_trials = {}
    for worker, worker_trials in trials_by_worker.items():
        # Trier par valeur (meilleur en premier)
        sorted_trials = sorted(worker_trials, key=lambda x: x.value, reverse=True)
        best_trials[worker] = sorted_trials[:n_trials]
    
    return best_trials

def format_trial_params(trial) -> Dict[str, Any]:
    """Formate les paramètres d'un essai pour l'affichage."""
    params = {
        'trial_number': trial.number,
        'value': trial.value,
        'params': trial.params,
        'user_attrs': trial.user_attrs,
        'datetime_start': trial.datetime_start,
        'datetime_complete': trial.datetime_complete
    }
    return params

def save_results_to_json(results: Dict[str, List[Dict[str, Any]]], output_dir: str):
    """Sauvegarde les résultats dans un fichier JSON."""
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Formater les résultats pour la sérialisation
    serializable_results = {}
    for worker, trials in results.items():
        serializable_results[worker] = [format_trial_params(trial) for trial in trials]
    
    # Sauvegarder dans un fichier JSON
    output_path = os.path.join(output_dir, 'best_hyperparameters.json')
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"Résultats sauvegardés dans {output_path}")
    return output_path

def analyze_optuna_database(db_path: str, study_name: str = 'adan_study', output_dir: str = 'optuna_analysis'):
    """
    Analyse la base de données Optuna et extrait les meilleurs hyperparamètres.
    
    Args:
        db_path: Chemin vers la base de données Optuna
        study_name: Nom de l'étude Optuna à analyser
        output_dir: Répertoire de sortie pour les résultats
    """
    # Vérifier si le fichier de base de données existe
    if not os.path.exists(db_path):
        print(f"Erreur: Le fichier de base de données {db_path} n'existe pas.")
        return
    
    # URL de connexion à la base de données
    storage_url = f"sqlite:///{os.path.abspath(db_path)}"
    
    print(f"Analyse de la base de données Optuna: {db_path}")
    print(f"Nom de l'étude: {study_name}")
    
    # Récupérer les meilleurs essais par worker
    best_trials = get_best_trials_per_worker(study_name, storage_url, n_trials=1)
    
    if not best_trials:
        print("Aucun essai valide trouvé dans la base de données.")
        return
    
    # Afficher les résultats
    print("\n" + "="*80)
    print("MEILLEURS HYPERPARAMÈTRES PAR WORKER")
    print("="*80)
    
    for worker, trials in best_trials.items():
        print(f"\n{'='*40}")
        print(f"WORKER: {worker.upper()}")
        print(f"Nombre d'essais valides: {len(trials)}")
        
        if not trials:
            print("Aucun essai valide pour ce worker.")
            continue
            
        # Afficher les détails du meilleur essai
        best_trial = trials[0]
        print(f"\nMeilleur essai (valeur: {best_trial.value:.4f}):")
        print(f"Numéro d'essai: {best_trial.number}")
        print("\nParamètres:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        
        print("\nAttributs utilisateur:")
        for key, value in best_trial.user_attrs.items():
            print(f"  {key}: {value}")
    
    # Sauvegarder les résultats dans un fichier JSON
    output_path = save_results_to_json(best_trials, output_dir)
    print(f"\nAnalyse terminée. Résultats sauvegardés dans: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyse des résultats d\'optimisation Optuna')
    parser.add_argument('--db-path', type=str, default='hyperparam_optimization.db',
                       help='Chemin vers la base de données Optuna')
    parser.add_argument('--study-name', type=str, default='adan_study',
                       help='Nom de l\'étude Optuna à analyser')
    parser.add_argument('--output-dir', type=str, default='optuna_analysis',
                       help='Répertoire de sortie pour les résultats')
    
    args = parser.parse_args()
    
    # Exécuter l'analyse
    analyze_optuna_database(
        db_path=args.db_path,
        study_name=args.study_name,
        output_dir=args.output_dir
    )
