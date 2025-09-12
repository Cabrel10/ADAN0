#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de monitoring en temps réel pour l'entraînement ADAN.
Surveille les performances, flux monétaires et métriques d'entraînement.
"""
import os
import sys
import time
import json
import argparse
import threading
from datetime import datetime, timedelta
from pathlib import Path
import signal

# Rich imports pour interface
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.align import Align

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

console = Console()

class TrainingMonitor:
    """Moniteur temps réel des performances d'entraînement."""

    def __init__(self, log_dir="reports", refresh_rate=2.0):
        self.log_dir = Path(log_dir)
        self.refresh_rate = refresh_rate
        self.running = False
        self.start_time = datetime.now()

        # Métriques de suivi
        self.metrics_history = []
        self.current_metrics = {}
        self.alerts = []

        # Fichiers à surveiller
        self.tensorboard_dir = self.log_dir / "tensorboard_logs"
        self.training_logs = []

    def find_training_logs(self):
        """Trouve les fichiers de logs d'entraînement récents."""
        log_patterns = [
            "training_*.log",
            "*.log"
        ]

        recent_logs = []
        for pattern in log_patterns:
            for log_file in Path(".").glob(pattern):
                # Vérifier si le fichier a été modifié récemment (dernières 24h)
                if log_file.stat().st_mtime > time.time() - 86400:
                    recent_logs.append(log_file)

        return sorted(recent_logs, key=lambda x: x.stat().st_mtime, reverse=True)

    def parse_log_metrics(self, log_file):
        """Parse les métriques depuis un fichier de log."""
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()

            metrics = {}
            for line in reversed(lines[-100:]):  # Dernières 100 lignes
                line = line.strip()

                # Parser différents formats de métriques
                if "Capital:" in line and "$" in line:
                    try:
                        capital_part = line.split("Capital:")[1].split()[0]
                        capital = float(capital_part.replace("$", "").replace(",", ""))
                        metrics['capital'] = capital
                    except:
                        pass

                if "ROI:" in line and "%" in line:
                    try:
                        roi_part = line.split("ROI:")[1].split("%")[0].strip()
                        roi = float(roi_part.replace("+", ""))
                        metrics['roi_pct'] = roi
                    except:
                        pass

                if "Step" in line and "/" in line:
                    try:
                        step_part = line.split("Step")[1].split("/")[0].strip()
                        current_step = int(step_part.replace(",", ""))
                        metrics['current_step'] = current_step
                    except:
                        pass

                if "Reward" in line:
                    try:
                        if ":" in line:
                            reward_part = line.split(":")[-1].strip()
                            reward = float(reward_part)
                            metrics['reward'] = reward
                    except:
                        pass

            return metrics
        except Exception as e:
            return {}

    def check_model_files(self):
        """Vérifie l'existence et la taille des modèles."""
        model_info = {}

        models_dir = Path("models")
        if models_dir.exists():
            for model_file in models_dir.glob("*.zip"):
                stat = model_file.stat()
                model_info[model_file.name] = {
                    'size_mb': stat.st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                    'age_minutes': (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).total_seconds() / 60
                }

        return model_info

    def generate_alerts(self):
        """Génère des alertes basées sur les métriques actuelles."""
        alerts = []

        if 'capital' in self.current_metrics:
            capital = self.current_metrics['capital']
            if capital < 5.0:
                alerts.append("🚨 CAPITAL CRITIQUE: ${:.2f}".format(capital))
            elif capital < 10.0:
                alerts.append("⚠️ Capital faible: ${:.2f}".format(capital))

        if 'roi_pct' in self.current_metrics:
            roi = self.current_metrics['roi_pct']
            if roi < -50:
                alerts.append("📉 ROI très négatif: {:.1f}%".format(roi))
            elif roi > 50:
                alerts.append("📈 ROI excellent: +{:.1f}%".format(roi))

        # Vérifier si l'entraînement semble bloqué
        if len(self.metrics_history) > 5:
            recent_steps = [m.get('current_step', 0) for m in self.metrics_history[-5:]]
            if len(set(recent_steps)) == 1 and recent_steps[0] > 0:
                alerts.append("⏸️ Entraînement possiblement bloqué")

        return alerts

    def create_dashboard(self):
        """Crée le tableau de bord principal."""
        layout = Layout()

        # Division principale
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=6)
        )

        # Division du contenu principal
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )

        # Header avec titre et temps
        elapsed = datetime.now() - self.start_time
        header_text = Text.assemble(
            ("🚀 ADAN Training Monitor", "bold blue"),
            (" | ", "dim"),
            (f"Uptime: {str(elapsed).split('.')[0]}", "green"),
            (" | ", "dim"),
            (f"Refresh: {self.refresh_rate}s", "dim")
        )
        layout["header"] = Panel(Align.center(header_text), style="blue")

        # Métriques principales (gauche)
        metrics_table = Table(title="📊 Métriques Temps Réel", title_style="bold cyan")
        metrics_table.add_column("Métrique", style="dim")
        metrics_table.add_column("Valeur", style="bright_white")
        metrics_table.add_column("Status", style="green")

        # Ajouter les métriques actuelles
        if 'capital' in self.current_metrics:
            capital = self.current_metrics['capital']
            status = "🟢" if capital >= 15.0 else "🟡" if capital >= 10.0 else "🔴"
            metrics_table.add_row("💰 Capital", f"${capital:.2f}", status)

        if 'roi_pct' in self.current_metrics:
            roi = self.current_metrics['roi_pct']
            status = "🟢" if roi >= 0 else "🟡" if roi >= -20 else "🔴"
            metrics_table.add_row("📈 ROI", f"{roi:+.2f}%", status)

        if 'current_step' in self.current_metrics:
            step = self.current_metrics['current_step']
            metrics_table.add_row("🎯 Step Actuel", f"{step:,}", "🟢")

        if 'reward' in self.current_metrics:
            reward = self.current_metrics['reward']
            status = "🟢" if reward >= 0 else "🟡" if reward >= -1 else "🔴"
            metrics_table.add_row("⭐ Récompense", f"{reward:.4f}", status)

        layout["left"] = Panel(metrics_table, title="Métriques", border_style="cyan")

        # Informations système (droite)
        system_table = Table(title="🖥️ Système", title_style="bold yellow")
        system_table.add_column("Composant", style="dim")
        system_table.add_column("Status", style="bright_white")

        # Vérifier les modèles
        model_info = self.check_model_files()
        if model_info:
            latest_model = max(model_info.items(), key=lambda x: x[1]['modified'])
            system_table.add_row(
                "📁 Dernier Modèle",
                f"{latest_model[0]} ({latest_model[1]['size_mb']:.1f}MB)"
            )
            system_table.add_row(
                "⏰ Dernière Sauvegarde",
                f"{latest_model[1]['age_minutes']:.0f}min ago"
            )

        # Logs actifs
        training_logs = self.find_training_logs()
        if training_logs:
            system_table.add_row("📝 Logs Actifs", f"{len(training_logs)} fichiers")

        # Historique des métriques
        if len(self.metrics_history) > 1:
            system_table.add_row("📊 Historique", f"{len(self.metrics_history)} points")

        layout["right"] = Panel(system_table, title="Système", border_style="yellow")

        # Footer construction starts here
        alerts_panel = None
        # Use self.generate_alerts() as it's a method of the class
        current_alerts = self.generate_alerts() # Call it once
        if current_alerts: # Check if the list of alert strings is non-empty
            alerts_text = "\n".join(current_alerts)
            if alerts_text: # Check if the joined string is not empty
                alerts_panel = Panel(alerts_text, title="🚨 Alertes", border_style="red")

        instructions_text = (
            "[bold cyan]Contrôles:[/bold cyan] Ctrl+C pour quitter | "
            "[dim]Moniteur automatique des performances ADAN[/dim]"
        )
        instructions_panel = Panel(instructions_text, border_style="dim")

        if alerts_panel:
            # Both alerts and instructions need to be displayed.
            # Create a new Layout for the footer that will contain these two, potentially side-by-side.
            footer_combined_layout = Layout(name="footer_split")

            # Split this new layout into two columns (adjust ratio/size as needed)
            # Using ratio based split as an example
            footer_combined_layout.split_row(
                Layout(name="alerts_col", ratio=1),
                Layout(name="instructions_col", ratio=2) # Example: instructions take more space
            )

            # Populate these new columns with the panel content
            footer_combined_layout["alerts_col"].update(alerts_panel)
            footer_combined_layout["instructions_col"].update(instructions_panel)

            # Update the main layout's "footer" region with this composite layout
            layout["footer"].update(footer_combined_layout)
        else:
            # Only instructions panel needs to be displayed.
            # Update the main layout's "footer" region directly with the instructions_panel.
            layout["footer"].update(instructions_panel)

        return layout

    def update_metrics(self):
        """Met à jour les métriques depuis les logs."""
        training_logs = self.find_training_logs()

        if not training_logs:
            return

        # Parser le log le plus récent
        latest_log = training_logs[0]
        new_metrics = self.parse_log_metrics(latest_log)

        if new_metrics:
            # Ajouter timestamp
            new_metrics['timestamp'] = datetime.now()

            # Mettre à jour les métriques actuelles
            self.current_metrics.update(new_metrics)

            # Ajouter à l'historique
            self.metrics_history.append(new_metrics.copy())

            # Garder seulement les 100 derniers points
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]

    def run(self):
        """Lance le monitoring en temps réel."""
        self.running = True

        with Live(self.create_dashboard(), refresh_per_second=1/self.refresh_rate, console=console) as live:
            try:
                while self.running:
                    # Mettre à jour les métriques
                    self.update_metrics()

                    # Mettre à jour l'affichage
                    live.update(self.create_dashboard())

                    # Attendre avant la prochaine mise à jour
                    time.sleep(self.refresh_rate)

            except KeyboardInterrupt:
                self.running = False
                console.print("\n[yellow]⏹️ Monitoring arrêté par l'utilisateur[/yellow]")

    def stop(self):
        """Arrête le monitoring."""
        self.running = False

def signal_handler(signum, frame, monitor):
    """Gestionnaire de signal pour arrêt propre."""
    monitor.stop()

def main():
    parser = argparse.ArgumentParser(description='ADAN Training Monitor - Surveillance Temps Réel')

    parser.add_argument('--refresh', type=float, default=2.0,
                        help='Taux de rafraîchissement en secondes (défaut: 2.0)')
    parser.add_argument('--log-dir', type=str, default='reports',
                        help='Répertoire des logs (défaut: reports)')

    args = parser.parse_args()

    # Affichage de démarrage
    console.print(Panel.fit(
        "[bold green]🚀 ADAN Training Monitor[/bold green]\n"
        f"[cyan]Taux de rafraîchissement: {args.refresh}s[/cyan]\n"
        "[yellow]Appuyez sur Ctrl+C pour arrêter[/yellow]",
        title="Démarrage du Monitoring"
    ))

    # Créer et lancer le moniteur
    monitor = TrainingMonitor(log_dir=args.log_dir, refresh_rate=args.refresh)

    # Configurer le gestionnaire de signal
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, monitor))

    try:
        monitor.run()
    except Exception as e:
        console.print(f"[red]❌ Erreur du monitoring: {str(e)}[/red]")
        return 1

    console.print("[green]✅ Monitoring terminé[/green]")
    return 0

if __name__ == "__main__":
    sys.exit(main())
