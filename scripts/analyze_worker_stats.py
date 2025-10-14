"""
Analyse des statistiques des workers à partir de la base de données Optuna.
Extrait les statistiques de Sharpe Ratio, Win Rate (>50%) et Profit Factor (>1.5) par worker.
"""

import argparse
import sqlite3
import pandas as pd
import json
import os
from typing import Dict, Any

def get_worker_statistics(db_path: str) -> Dict[str, Dict[str, Any]]:
    """Récupère les statistiques des workers depuis la base de données Optuna."""
    conn = sqlite3.connect(db_path)
    
    # Dictionnaire pour stocker les statistiques par worker
    stats = {}
    
    # Liste des workers
    workers = [0, 1, 2, 3]
    
    for worker_id in workers:
        # Requête pour obtenir les métriques du worker
        query = f"""
        SELECT 
            tv.trial_id,
            MAX(CASE WHEN tua.key = 'worker_{worker_id}_sharpe_ratio' THEN tua.value_json ELSE NULL END) as sharpe_ratio,
            MAX(CASE WHEN tua.key = 'worker_{worker_id}_win_rate' THEN tua.value_json ELSE NULL END) as win_rate,
            MAX(CASE WHEN tua.key = 'worker_{worker_id}_profit_factor' THEN tua.value_json ELSE NULL END) as profit_factor
        FROM trial_user_attributes tua
        JOIN trial_values tv ON tua.trial_id = tv.trial_id
        WHERE tua.key IN (
            'worker_{worker_id}_sharpe_ratio',
            'worker_{worker_id}_win_rate',
            'worker_{worker_id}_profit_factor'
        )
        GROUP BY tv.trial_id
        """
        
        df = pd.read_sql_query(query, conn)
        
        # Convertir les types de données
        df['sharpe_ratio'] = pd.to_numeric(df['sharpe_ratio'], errors='coerce')
        df['win_rate'] = pd.to_numeric(df['win_rate'], errors='coerce')
        df['profit_factor'] = pd.to_numeric(df['profit_factor'], errors='coerce')
        
        # Filtrer uniquement les résultats positifs et rentables
        filtered = df[
            (df['sharpe_ratio'] > 0) & 
            (df['win_rate'] > 50) & 
            (df['profit_factor'] > 1.5)
        ]
        
        total_filtered = len(filtered)
        total_trials = len(df)
        
        if total_filtered > 0:  # Ne pas inclure les workers sans résultats positifs
            # Calculer les statistiques sur les résultats filtrés
            stats[f'worker_{worker_id}'] = {
                'total_trials': total_trials,
                'successful_trials': total_filtered,
                'success_rate': (total_filtered / total_trials) * 100 if total_trials > 0 else 0,
                'avg_sharpe_ratio': filtered['sharpe_ratio'].mean(),
                'min_sharpe_ratio': filtered['sharpe_ratio'].min(),
                'max_sharpe_ratio': filtered['sharpe_ratio'].max(),
                'avg_win_rate': filtered['win_rate'].mean(),
                'min_win_rate': filtered['win_rate'].min(),
                'max_win_rate': filtered['win_rate'].max(),
                'avg_profit_factor': filtered['profit_factor'].mean(),
                'min_profit_factor': filtered['profit_factor'].min(),
                'max_profit_factor': filtered['profit_factor'].max()
            }
    
    conn.close()
    return stats


def save_statistics(stats: Dict[str, Dict[str, Any]], output_dir: str) -> str:
    """Sauvegarde les statistiques dans un fichier JSON."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'worker_positive_statistics.json')
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return output_path


def print_statistics(stats: Dict[str, Dict[str, Any]]):
    """Affiche les statistiques de manière lisible."""
    if not stats:
        print("\nAucun résultat positif et rentable trouvé.")
        return
        
    print("\n" + "="*80)
    print("STATISTIQUES PAR WORKER (résultats positifs et rentables)")
    print("="*80)
    print("Critères: Sharpe Ratio > 0, Win Rate > 50%, Profit Factor > 1.5\n")
    
    # Trier par Sharpe Ratio moyen décroissant
    sorted_workers = sorted(
        stats.items(), 
        key=lambda x: x[1]['avg_sharpe_ratio'], 
        reverse=True
    )
    
    for worker, data in sorted_workers:
        print(f"\n{worker.upper()}")
        print("-" * len(worker) + "----")
        print(f"Essais réussis: {data['successful_trials']}/"
              f"{data['total_trials']} ({data['success_rate']:.1f}%)")
        
        print("\nRatio de Sharpe:")
        print(f"  - Moyen: {data['avg_sharpe_ratio']:.2f}")
        print(f"  - Min: {data['min_sharpe_ratio']:.2f}")
        print(f"  - Max: {data['max_sharpe_ratio']:.2f}")
        
        print("\nWin Rate (en %):")
        print(f"  - Moyen: {data['avg_win_rate']:.1f}%")
        print(f"  - Min: {data['min_win_rate']:.1f}%")
        print(f"  - Max: {data['max_win_rate']:.1f}%")
        
        print("\nProfit Factor:")
        print(f"  - Moyen: {data['avg_profit_factor']:.2f}")
        print(f"  - Min: {data['min_profit_factor']:.2f}")
        print(f"  - Max: {data['max_profit_factor']:.2f}")
        
        print("\n" + "-"*40)


def main():
    parser = argparse.ArgumentParser(
        description='Analyse des statistiques des workers Optuna',
        epilog='Filtre les résultats pour ne garder que les essais avec Sharpe > 0, Win Rate > 50% et Profit Factor > 1.5'
    )
    parser.add_argument(
        '--db-path', 
        type=str, 
        default='scripts/hyperparam_optimization.db',
        help='Chemin vers la base de données Optuna (défaut: scripts/hyperparam_optimization.db)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='analysis_results',
        help='Répertoire de sortie pour les résultats (défaut: analysis_results)'
    )
    
    args = parser.parse_args()
    
    print(f"Analyse des statistiques des workers depuis: {args.db_path}")
    print("Filtrage des résultats avec: Sharpe Ratio > 0, "
          "Win Rate > 50%, Profit Factor > 1.5\n")
    
    # Vérifier si le fichier de base de données existe
    if not os.path.exists(args.db_path):
        print(f"Erreur: Le fichier {args.db_path} n'existe pas.")
        return
    
    # Récupérer les statistiques
    stats = get_worker_statistics(args.db_path)
    
    # Afficher les statistiques
    print_statistics(stats)
    
    # Sauvegarder les résultats
    if stats:
        output_path = save_statistics(stats, args.output_dir)
        print(f"\nStatistiques détaillées sauvegardées dans: {output_path}")


if __name__ == "__main__":
    main()
