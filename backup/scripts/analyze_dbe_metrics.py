#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyse des métriques du Dynamic Behavior Engine (DBE)

Ce script analyse les logs générés par le test d'endurance du DBE.
Il génère des visualisations pour comprendre le comportement du moteur.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Configuration des styles
plt.style.use('default')
sns.set_palette("viridis")

# Chemins des fichiers
LOGS_DIR = Path("./logs")
OUTPUT_DIR = Path("../reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics(log_file: str = "endurance_metrics.jsonl") -> pd.DataFrame:
    """Charge les métriques depuis le fichier JSONL."""
    filepath = LOGS_DIR / log_file

    if not filepath.exists():
        raise FileNotFoundError(f"Fichier de logs non trouvé : {filepath}")

    # Chargement des données
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Erreur de décodage JSON : {e}")

    if not data:
        raise ValueError("Aucune donnée valide trouvée dans le fichier de logs.")

    # Conversion en DataFrame
    df = pd.DataFrame(data)

    # Conversion des timestamps
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

    return df


def plot_risk_metrics(df: pd.DataFrame) -> None:
    """Affiche l'évolution des paramètres de risque."""
    plt.figure(figsize=(14, 7))

    # SL/TP
    plt.subplot(2, 1, 1)
    plt.plot(df['timestamp'], df['sl_pct'], label='Stop Loss (%)')
    plt.plot(df['timestamp'], df['tp_pct'], label='Take Profit (%)')
    plt.title('Évolution des niveaux de SL/TP')
    plt.ylabel('Pourcentage (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Mode de risque
    plt.subplot(2, 1, 2)
    risk_colors = {'NORMAL': 'green', 'DEFENSIVE': 'red', 'AGGRESSIVE': 'blue'}
    for mode, color in risk_colors.items():
        mask = df['risk_mode'] == mode
        plt.scatter(df.loc[mask, 'timestamp'],
                   df.loc[mask, 'drawdown'],
                   c=color, label=mode, alpha=0.6)

    plt.title('Drawdown par mode de risque')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'risk_metrics.png', dpi=300, bbox_inches='tight')


def plot_capital_evolution(df: pd.DataFrame) -> None:
    """Affiche l'évolution du capital et du drawdown."""
    plt.figure(figsize=(14, 7))

    # Évolution du capital
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df['timestamp'], df['capital'], 'b-')
    ax1.set_ylabel('Capital ($)', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_title('Évolution du capital')
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = ax1.twinx()
    ax2.fill_between(df['timestamp'], df['drawdown'],
                    color='r', alpha=0.2, label='Drawdown')
    ax2.set_ylabel('Drawdown (%)', color='r')
    ax2.tick_params('y', colors='r')

    # Distribution des rendements
    plt.subplot(2, 1, 2)
    returns = df['capital'].pct_change().dropna() * 100
    sns.histplot(returns, kde=True, bins=50)
    plt.axvline(returns.mean(), color='r', linestyle='--',
                label=f'Moyenne: {returns.mean():.2f}%')
    plt.title('Distribution des rendements journaliers')
    plt.xlabel('Rendement (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'capital_evolution.png', dpi=300, bbox_inches='tight')


def plot_behavior_analysis(df: pd.DataFrame) -> None:
    """Analyse du comportement du DBE."""
    plt.figure(figsize=(14, 10))

    # Fréquence des modes de risque
    plt.subplot(2, 2, 1)
    risk_counts = df['risk_mode'].value_counts()
    plt.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%',
            startangle=90, colors=sns.color_palette('viridis', len(risk_counts)))
    plt.title('Répartition des modes de risque')

    # Drawdown par mode de risque
    plt.subplot(2, 2, 2)
    sns.boxplot(x='risk_mode', y='drawdown', data=df,
                order=['AGGRESSIVE', 'NORMAL', 'DEFENSIVE'])
    plt.title('Distribution du drawdown par mode de risque')
    plt.xticks(rotation=45)

    # Évolution des paramètres de risque
    if 'reward_boost' in df.columns:
        plt.subplot(2, 2, 3)
        sns.histplot(df['reward_boost'], kde=True, bins=30)
        plt.title('Distribution des récompenses (reward_boost)')
        plt.xlabel('Valeur de récompense')

    if 'penalty_inaction' in df.columns:
        plt.subplot(2, 2, 4)
        sns.histplot(df['penalty_inaction'], kde=True, bins=30, color='r')
        plt.title('Distribution des pénalités d\'inaction')
        plt.xlabel('Valeur de pénalité')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'behavior_analysis.png', dpi=300, bbox_inches='tight')


def generate_report(df: pd.DataFrame) -> None:
    """Génère un rapport d'analyse complet."""
    # Conversion des types pour JSON
    def to_py(o):
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        if isinstance(o, (np.floating, np.float64, np.float32)):
            return float(o)
        if isinstance(o, pd.Timestamp):
            return o.strftime('%Y-%m-%d %H:%M:%S')
        return o

    report = {
        'date_analyse': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'periode': {
            'debut': to_py(df['timestamp'].min()),
            'fin': to_py(df['timestamp'].max()),
            'duree_jours': int((df['timestamp'].max() - df['timestamp'].min()).days)
        },
        'performance': {
            'rendement_total_pct': float(((df['capital'].iloc[-1] / df['capital'].iloc[0]) - 1) * 100),
            'drawdown_max_pct': float(df['drawdown'].max()),
            'volatilite_annuelle_pct': float(df['capital'].pct_change().std() * (252 ** 0.5) * 100),
            'sharpe_ratio': float((df['capital'].pct_change().mean() / df['capital'].pct_change().std()) * (252 ** 0.5))
        },
        'comportement': {
            'mode_risque_principal': str(df['risk_mode'].mode()[0]),
            'nb_changements_risque': int((df['risk_mode'] != df['risk_mode'].shift(1)).sum() - 1),
            'moyenne_sl_pct': float(df['sl_pct'].mean()),
            'moyenne_tp_pct': float(df['tp_pct'].mean())
        }
    }

    # Sauvegarde du rapport
    report_path = OUTPUT_DIR / 'dbe_analysis_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    return report


def main():
    try:
        print("Chargement des données...")
        df = load_metrics()

        print("Génération des graphiques...")
        plot_risk_metrics(df)
        plot_capital_evolution(df)
        plot_behavior_analysis(df)

        print("Génération du rapport...")
        report = generate_report(df)

        print("\n=== Rapport d'analyse DBE ===")
        print(f"Période: {report['periode']['debut']} -> {report['periode']['fin']}")
        print(f"Rendement total: {report['performance']['rendement_total_pct']:.2f}%")
        print(f"Drawdown max: {report['performance']['drawdown_max_pct']:.2f}%")
        print(f"Mode de risque principal: {report['comportement']['mode_risque_principal']}")

        print(f"\nLes graphiques ont été sauvegardés dans: {OUTPUT_DIR.absolute()}")

    except Exception as e:
        print(f"Erreur lors de l'analyse: {str(e)}")
        raise


if __name__ == "__main__":
    main()
