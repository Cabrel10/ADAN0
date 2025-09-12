#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyse des logs TensorBoard pour l'apprentissage en ligne.

Ce script analyse les logs générés pendant l'apprentissage en ligne pour évaluer
la stabilité et l'efficacité de l'entraînement.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tensorboard.backend.event_processing import event_accumulator

# Configuration
BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"
ONLINE_LOGS_DIR = LOGS_DIR / "online"
REPORTS_DIR = BASE_DIR / "reports"
TENSORBOARD_DIR = REPORTS_DIR / "tensorboard_analysis"
TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

# Configuration du style
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

def load_tensorboard_logs(log_dir):
    """Charge les logs TensorBoard depuis le répertoire spécifié."""
    if not log_dir.exists():
        raise FileNotFoundError(f"Répertoire de logs non trouvé : {log_dir}")

    # Recherche des fichiers d'événements TensorBoard
    event_files = list(log_dir.glob("**/events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"Aucun fichier d'événements TensorBoard trouvé dans {log_dir}")

    print(f"Chargement des logs depuis : {event_files[0]}")

    # Chargement des données
    ea = event_accumulator.EventAccumulator(
        str(event_files[0]),
        size_guidance={
            event_accumulator.SCALARS: 0,  # Tous les scalaires
            event_accumulator.IMAGES: 0,
            event_accumulator.AUDIO: 0,
            event_accumulator.GRAPH: 0,
            event_accumulator.HISTOGRAMS: 0,
        }
    )
    ea.Reload()

    # Extraction des données
    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        data[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events],
            'wall_times': [e.wall_time for e in events]
        }

    return data

def plot_learning_curves(data, output_dir):
    """Génère des graphiques des courbes d'apprentissage."""
    # Métriques à tracer
    metrics = {
        'train/rollout/ep_rew_mean': 'Récompense Moyenne par Épisode',
        'train/loss': 'Perte d\'Entraînement',
        'train/entropy_loss': 'Perte d\'Entropie',
        'train/policy_gradient_loss': 'Perte de Gradient de Politique',
        'train/value_loss': 'Perte de Valeur',
        'train/learning_rate': 'Taux d\'Apprentissage'
    }

    for metric_key, title in metrics.items():
        if metric_key not in data:
            print(f"Avertissement : Métrique non trouvée : {metric_key}")
            continue

        plt.figure(figsize=(12, 6))

        # Tracé de la métrique
        plt.plot(data[metric_key]['steps'], data[metric_key]['values'],
                label=title, linewidth=2)

        # Configuration du graphique
        plt.title(title)
        plt.xlabel('Étapes d\'Entraînement')
        plt.ylabel('Valeur')
        plt.grid(True, alpha=0.3)

        # Ajout d'une ligne de tendance si suffisamment de points
        if len(data[metric_key]['steps']) > 5:
            z = np.polyfit(data[metric_key]['steps'], data[metric_key]['values'], 1)
            p = np.poly1d(z)
            plt.plot(data[metric_key]['steps'],
                    p(data[metric_key]['steps']),
                    'r--',
                    label='Tendance')

        plt.legend()
        plt.tight_layout()

        # Sauvegarde du graphique
        filename = f"learning_{metric_key.replace('/', '_')}.png"
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Graphique sauvegardé : {output_path}")

def analyze_learning_stability(data, output_dir):
    """Analyse la stabilité de l'apprentissage."""
    if 'train/rollout/ep_rew_mean' not in data:
        print("Avertissement : Données de récompense non disponibles pour l'analyse de stabilité")
        return

    rewards = data['train/rollout/ep_rew_mean']['values']
    steps = data['train/rollout/ep_rew_mean']['steps']

    if len(rewards) < 2:
        print("Pas assez de données pour l'analyse de stabilité")
        return

    # Calcul des métriques de stabilité
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    max_reward = max(rewards)
    min_reward = min(rewards)

    # Détection des tendances
    z = np.polyfit(steps, rewards, 1)
    trend_slope = z[0]  # Pente de la droite de tendance

    # Calcul du coefficient de variation
    cv = (std_reward / mean_reward) * 100 if mean_reward != 0 else float('inf')

    # Création du rapport
    stability_report = {
        'timestamp': datetime.now().isoformat(),
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'max_reward': float(max_reward),
        'min_reward': float(min_reward),
        'trend_slope': float(trend_slope),
        'coefficient_of_variation': float(cv),
        'is_stable': abs(trend_slope) < 0.01 and cv < 30.0  # Critères de stabilité
    }

    # Sauvegarde du rapport
    report_path = output_dir / 'stability_report.json'
    with open(report_path, 'w') as f:
        json.dump(stability_report, f, indent=2)

    print(f"Rapport de stabilité sauvegardé : {report_path}")

    # Affichage des résultats
    print("\n=== Analyse de Stabilité de l'Apprentissage ===")
    print(f"Récompense moyenne: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Plage de récompenses: [{min_reward:.2f}, {max_reward:.2f}]")
    print(f"Pente de la tendance: {trend_slope:.6f}")
    print(f"Coefficient de variation: {cv:.2f}%")
    print(f"Apprentissage stable: {'Oui' if stability_report['is_stable'] else 'Non'}")

def main():
    try:
        # Chargement des logs TensorBoard
        print("Chargement des logs TensorBoard...")
        logs_data = load_tensorboard_logs(ONLINE_LOGS_DIR)

        # Création des graphiques d'apprentissage
        print("\nGénération des graphiques d'apprentissage...")
        plot_learning_curves(logs_data, TENSORBOARD_DIR)

        # Analyse de la stabilité
        print("\nAnalyse de la stabilité de l'apprentissage...")
        analyze_learning_stability(logs_data, TENSORBOARD_DIR)

        print("\nAnalyse terminée avec succès!")
        print(f"Résultats sauvegardés dans : {TENSORBOARD_DIR.absolute()}")

    except Exception as e:
        print(f"Erreur lors de l'analyse des logs TensorBoard : {str(e)}")
        raise

if __name__ == "__main__":
    main()
