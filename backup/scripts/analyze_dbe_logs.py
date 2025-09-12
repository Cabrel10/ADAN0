#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyse des logs du Dynamic Behavior Engine (DBE).

Ce script permet d'analyser et de visualiser les décisions prises par le DBE
au cours d'une session de trading.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional
import typer
from rich.console import Console
from rich.table import Table
import numpy as np

# Configuration de la console Rich pour une meilleure sortie
console = Console()

class DBELogAnalyzer:
    """Classe pour analyser les logs du Dynamic Behavior Engine."""

    def __init__(self, log_file: str):
        """
        Initialise l'analyseur de logs.

        Args:
            log_file: Chemin vers le fichier de log à analyser
        """
        self.log_file = Path(log_file)
        self.log_data = self._load_logs()

    def _load_logs(self) -> List[Dict[str, Any]]:
        """Charge les logs depuis le fichier JSONL."""
        logs = []
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        logs.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        console.print(f"[yellow]Avertissement: Ligne mal formatée ignorée: {line[:100]}...[/]")
        except FileNotFoundError:
            console.print(f"[red]Erreur: Fichier non trouvé: {self.log_file}[/]")
            raise

        if not logs:
            console.print("[yellow]Avertissement: Aucune donnée de log valide trouvée.[/]")

        return logs

    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des logs."""
        if not self.log_data:
            return {}

        # Filtrer uniquement les entrées de décision
        decisions = [log for log in self.log_data if log.get('type') == 'decision']

        if not decisions:
            return {"error": "Aucune décision trouvée dans les logs"}

        # Statistiques de base
        summary = {
            "total_decisions": len(decisions),
            "time_period": {
                "start": decisions[0].get('timestamp'),
                "end": decisions[-1].get('timestamp')
            },
            "modulation_stats": {
                "sl_pct": self._calculate_stats(decisions, 'modulation.sl_pct'),
                "tp_pct": self._calculate_stats(decisions, 'modulation.tp_pct'),
                "position_size": self._calculate_stats(decisions, 'modulation.position_size')
            },
            "context_stats": {
                "portfolio_value": self._calculate_stats(decisions, 'context.portfolio_value'),
                "drawdown": self._calculate_stats(decisions, 'context.drawdown'),
                "volatility": self._calculate_stats(decisions, 'context.volatility')
            }
        }

        return summary

    def _calculate_stats(self, decisions: List[Dict], field: str) -> Dict[str, float]:
        """Calcule les statistiques pour un champ donné."""
        values = []
        for d in decisions:
            # Gestion des champs imbriqués (ex: 'modulation.sl_pct')
            parts = field.split('.')
            try:
                value = d
                for part in parts:
                    value = value.get(part, {})
                if isinstance(value, (int, float)):
                    values.append(value)
            except (KeyError, AttributeError):
                continue

        if not values:
            return {"count": 0, "min": None, "max": None, "mean": None, "std": None}

        return {
            "count": len(values),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)) if len(values) > 1 else 0.0
        }

    def plot_metrics(self, output_file: Optional[str] = None) -> None:
        """Génère des graphiques des métriques clés."""
        if not self.log_data:
            console.print("[red]Aucune donnée à afficher.[/]")
            return

        # Préparer les données pour le tracé
        decisions = [log for log in self.log_data if log.get('type') == 'decision']
        if not decisions:
            console.print("[red]Aucune décision trouvée à tracer.[/]")
            return

        # Créer un DataFrame pour faciliter l'analyse
        df = pd.DataFrame([
            {
                'step': d.get('step'),
                'timestamp': pd.to_datetime(d.get('timestamp')),
                'sl_pct': d.get('modulation', {}).get('sl_pct'),
                'tp_pct': d.get('modulation', {}).get('tp_pct'),
                'position_size': d.get('modulation', {}).get('position_size'),
                'portfolio_value': d.get('context', {}).get('portfolio_value'),
                'drawdown': d.get('context', {}).get('drawdown'),
                'volatility': d.get('context', {}).get('volatility')
            }
            for d in decisions
        ])

        if df.empty:
            console.print("[red]Aucune donnée valide à tracer.[/]")
            return

        # Créer une figure avec plusieurs sous-graphiques
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

        # Graphique 1: SL/TP
        ax1 = axes[0]
        ax1.plot(df['timestamp'], df['sl_pct'] * 100, label='Stop-Loss (%)', color='r')
        ax1.plot(df['timestamp'], df['tp_pct'] * 100, label='Take-Profit (%)', color='g')
        ax1.set_title('Évolution des paramètres de risque')
        ax1.set_ylabel('Valeur (%)')
        ax1.legend()
        ax1.grid(True)

        # Graphique 2: Taille de position et volatilité
        ax2 = axes[1]
        ax2_twin = ax2.twinx()

        color = 'tab:blue'
        ax2.set_ylabel('Taille de position', color=color)
        ax2.plot(df['timestamp'], df['position_size'] * 100, color=color, label='Taille de position')
        ax2.tick_params(axis='y', labelcolor=color)

        color = 'tab:orange'
        ax2_twin.set_ylabel('Volatilité', color=color)
        ax2_twin.plot(df['timestamp'], df['volatility'] * 100, color=color, linestyle='--', label='Volatilité')
        ax2_twin.tick_params(axis='y', labelcolor=color)

        ax2.set_title('Taille de position et volatilité')

        # Graphique 3: Valeur du portefeuille et drawdown
        ax3 = axes[2]
        ax3_twin = ax3.twinx()

        color = 'tab:blue'
        ax3.set_ylabel('Valeur du portefeuille (USDT)', color=color)
        ax3.plot(df['timestamp'], df['portfolio_value'], color=color, label='Valeur du portefeuille')
        ax3.tick_params(axis='y', labelcolor=color)

        color = 'tab:red'
        ax3_twin.set_ylabel('Drawdown (%)', color=color)
        ax3_twin.plot(df['timestamp'], df['drawdown'], color=color, linestyle='--', label='Drawdown')
        ax3_twin.tick_params(axis='y', labelcolor=color)

        ax3.set_title('Performance du portefeuille')

        # Ajuster l'espacement et afficher
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file)
            console.print(f"[green]Graphique enregistré sous {output_file}[/]")
        else:
            plt.show()

def display_summary(summary: Dict[str, Any]) -> None:
    """Affiche un résumé des logs dans la console."""
    if not summary:
        console.print("[red]Aucun résumé à afficher.[/]")
        return

    # Tableau récapitulatif
    table = Table(title="Résumé des décisions du DBE")
    table.add_column("Métrique", style="cyan")
    table.add_column("Valeur", style="green")

    # Informations générales
    table.add_row("Nombre total de décisions", str(summary['total_decisions']))
    table.add_row("Période de début", summary['time_period']['start'])
    table.add_row("Période de fin", summary['time_period']['end'])

    console.print(table)

    # Statistiques détaillées
    for category, stats in summary.items():
        if category in ['total_decisions', 'time_period']:
            continue

        console.print(f"\n[bold]{category.upper()}[/]")
        cat_table = Table(show_header=True, header_style="bold magenta")
        cat_table.add_column("Paramètre")
        cat_table.add_column("Min")
        cat_table.add_column("Max")
        cat_table.add_column("Moyenne")
        cat_table.add_column("Écart-type")

        for param, values in stats.items():
            if values['count'] > 0:
                cat_table.add_row(
                    param,
                    f"{values['min']:.4f}" if values['min'] is not None else "N/A",
                    f"{values['max']:.4f}" if values['max'] is not None else "N/A",
                    f"{values['mean']:.4f}" if values['mean'] is not None else "N/A",
                    f"{values['std']:.4f}" if values['std'] is not None else "N/A"
                )

        console.print(cat_table)

def main(
    log_file: str = typer.Argument(..., help="Chemin vers le fichier de log à analyser"),
    plot: bool = typer.Option(False, "--plot", "-p", help="Génère un graphique des métriques"),
    output: str = typer.Option(None, "--output", "-o", help="Fichier de sortie pour le graphique")
):
    """Analyse les logs du Dynamic Behavior Engine."""
    try:
        analyzer = DBELogAnalyzer(log_file)

        # Afficher le résumé
        summary = analyzer.get_summary()
        display_summary(summary)

        # Générer les graphiques si demandé
        if plot:
            analyzer.plot_metrics(output)

    except Exception as e:
        console.print(f"[red]Erreur lors de l'analyse des logs: {e}[/]")
        raise typer.Exit(1)

if __name__ == "__main__":
    typer.run(main)
