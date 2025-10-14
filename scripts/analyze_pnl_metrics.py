"""Script pour analyser les métriques de P&L par worker."""

import sqlite3
import pandas as pd
import json
import os
from typing import Dict, Any, List

def get_pnl_metrics(db_path: str) -> Dict[str, Dict[str, Any]]:
    """Récupère et analyse les métriques de P&L par worker."""
    try:
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
                MAX(CASE WHEN tua.key = 'worker_{worker_id}_total_pnl' THEN tua.value_json ELSE NULL END) as total_pnl,
                MAX(CASE WHEN tua.key = 'worker_{worker_id}_sharpe_ratio' THEN tua.value_json ELSE NULL END) as sharpe_ratio,
                MAX(CASE WHEN tua.key = 'worker_{worker_id}_win_rate' THEN tua.value_json ELSE NULL END) as win_rate,
                MAX(CASE WHEN tua.key = 'worker_{worker_id}_profit_factor' THEN tua.value_json ELSE NULL END) as profit_factor
            FROM trial_user_attributes tua
            JOIN trial_values tv ON tua.trial_id = tv.trial_id
            WHERE tua.key IN (
                'worker_{worker_id}_total_pnl',
                'worker_{worker_id}_sharpe_ratio',
                'worker_{worker_id}_win_rate',
                'worker_{worker_id}_profit_factor'
            )
            GROUP BY tv.trial_id
            """
            
            df = pd.read_sql_query(query, conn)
            
            # Convertir les types de données
            for col in ['total_pnl', 'sharpe_ratio', 'win_rate', 'profit_factor']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Filtrer les essais avec P&L valide
            valid_pnl = df[df['total_pnl'].notna()]
            
            if not valid_pnl.empty:
                # Trier par P&L décroissant
                sorted_by_pnl = valid_pnl.sort_values('total_pnl', ascending=False)
                
                # Calculer les statistiques
                stats[f'worker_{worker_id}'] = {
                    'total_trials': len(valid_pnl),
                    'avg_pnl': valid_pnl['total_pnl'].mean(),
                    'max_pnl': valid_pnl['total_pnl'].max(),
                    'min_pnl': valid_pnl['total_pnl'].min(),
                    'positive_pnl_count': len(valid_pnl[valid_pnl['total_pnl'] > 0]),
                    'negative_pnl_count': len(valid_pnl[valid_pnl['total_pnl'] < 0]),
                    'top_pnl_trials': sorted_by_pnl[['trial_id', 'total_pnl', 'sharpe_ratio', 'win_rate', 'profit_factor']].head(5).to_dict('records')
                }
        
        conn.close()
        return stats
        
    except Exception as e:
        print(f"Erreur lors de l'analyse des métriques P&L: {e}")
        return {}

def save_to_json(data: Dict, output_path: str):
    """Sauvegarde les données au format JSON."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def print_pnl_statistics(stats: Dict[str, Dict[str, Any]]):
    """Affiche les statistiques de P&L de manière lisible."""
    if not stats:
        print("Aucune donnée P&L valide trouvée.")
        return
    
    print("\n" + "="*80)
    print("ANALYSE DU PROFIT & LOSS (P&L) PAR WORKER")
    print("="*80)
    
    # Trier les workers par P&L moyen décroissant
    sorted_workers = sorted(
        stats.items(), 
        key=lambda x: x[1]['avg_pnl'], 
        reverse=True
    )
    
    for worker, data in sorted_workers:
        print(f"\n{worker.upper()}")
        print("-" * len(worker) + "----")
        print(f"Essais avec P&L valide: {data['total_trials']}")
        print(f"P&L moyen: {data['avg_pnl']:.2f}")
        print(f"Meilleur P&L: {data['max_pnl']:.2f}")
        print(f"Pire P&L: {data['min_pnl']:.2f}")
        print(f"Essais avec P&L positif: {data['positive_pnl_count']} ({(data['positive_pnl_count']/data['total_trials'])*100:.1f}%)")
        print(f"Essais avec P&L négatif: {data['negative_pnl_count']} ({(data['negative_pnl_count']/data['total_trials'])*100:.1f}%)")
        
        # Afficher les 3 meilleurs essais
        print("\nMeilleurs essais (par P&L):")
        for i, trial in enumerate(data['top_pnl_trials'][:3], 1):
            print(f"  {i}. Essai #{trial['trial_id']}:")
            print(f"     P&L: {trial['total_pnl']:.2f}")
            print(f"     Ratio de Sharpe: {trial['sharpe_ratio']:.2f}")
            print(f"     Taux de réussite: {trial['win_rate']:.1f}%")
            print(f"     Profit Factor: {trial['profit_factor']:.2f}")
        
        print("\n" + "-"*40)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyse des métriques P&L par worker')
    parser.add_argument('--db-path', type=str, default='scripts/hyperparam_optimization.db',
                      help='Chemin vers la base de données Optuna')
    parser.add_argument('--output-dir', type=str, default='analysis_results',
                      help='Répertoire de sortie pour les résultats')
    
    args = parser.parse_args()
    
    print(f"Analyse des métriques P&L depuis: {args.db_path}")
    
    # Récupérer les statistiques P&L
    pnl_stats = get_pnl_metrics(args.db_path)
    
    # Afficher les résultats
    print_pnl_statistics(pnl_stats)
    
    # Sauvegarder les résultats
    if pnl_stats:
        output_path = os.path.join(args.output_dir, 'pnl_analysis.json')
        save_to_json(pnl_stats, output_path)
        print(f"\nAnalyse P&L sauvegardée dans: {output_path}")
