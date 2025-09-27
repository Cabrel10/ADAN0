#!/usr/bin/env python3
# -*- coding: utf-8

"""
Training dashboard for monitoring trading bot performance.

This dashboard provides real-time visualization of training metrics
from the trading bot's log files.
"""

# Standard library imports
import json
import logging
import json
import logging
import os
import re
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, Optional, Deque

# Third-party imports
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Palette de couleurs pour les graphiques
COLOR_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

"""LogMonitor class for processing and monitoring training logs."""


class LogMonitor:
    """Monitors and processes training logs in real-time."""

    def __init__(self, log_dir: str = None, max_points: int = 1000) -> None:
        """Initialize the log monitor.

        Args:
            log_dir: Directory containing log files
            max_points: Maximum number of points to store in the metrics
        """
        if log_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.log_dir = os.path.join(
                os.path.dirname(current_dir), 'logs'
            )
        else:
            self.log_dir = log_dir

        self.max_points = max_points
        self.data: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: {
                'capital': deque(maxlen=max_points),
                'reward': deque(maxlen=max_points),
                'steps': deque(maxlen=max_points),
                'drawdown': deque(maxlen=max_points),
                'trades': deque(maxlen=max_points),
                'sharpe': deque(maxlen=max_points),
                'episode_length': deque(maxlen=max_points),
                'episode_reward': deque(maxlen=max_points),
                'episode_return': deque(maxlen=max_points),
                'episode_time': deque(maxlen=max_points),
                'episode_trades': deque(maxlen=max_points),
                'risk_drawdown_value': deque(maxlen=max_points),
                'risk_drawdown_limit_value': deque(maxlen=max_points),
                'risk_drawdown_pct': deque(maxlen=max_points),
                'risk_drawdown_limit_pct': deque(maxlen=max_points),
                'risk_equity': deque(maxlen=max_points),
                'risk_cash': deque(maxlen=max_points),
                'risk_solde_dispo': deque(maxlen=max_points),
                'parallel_workers': 0,
                'last_update': time.time()
            }
        )
        self.global_stats = {
            'total_episodes': 0,
            'total_trades': 0,
            'avg_return': 0,
            'best_worker': None,
            'worst_worker': None,
            'parallel_workers': 0,
            'active_workers': 0,
            'avg_episode_length': 0,
            'avg_episode_reward': 0,
            'total_episode_time': 0,
            'episodes_per_second': 0,
            'last_update': time.time()
        }
        self.last_modified = 0
        self.processed_files: set = set()

        # TensorBoard and checkpoint monitors
        self.tensorboard = TensorboardMonitor()
        self.checkpoints = CheckpointMonitor()

    def parse_json_log(self, log_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a JSON log entry and extract relevant metrics.

        Args:
            log_entry: Dictionary containing log entry data

        Returns:
            Dictionary of extracted metrics or None if entry is invalid
        """
        if not isinstance(log_entry, dict):
            logger.warning("Skipping invalid log entry (expected dict): %s",
                         str(log_entry)[:100])
            return None

        if 'status' in log_entry and log_entry['status'] in ['error', 'failed']:
            error_msg = log_entry.get('message', 'Unknown error')
            instance_id = log_entry.get('instance_id', 'unknown')
            logger.warning("Error in instance %s: %s", instance_id, error_msg)
            return None

        metrics = {}
        metrics['timestamp'] = log_entry.get('timestamp', datetime.now().isoformat())

        # Détection du mode parallèle
        if 'parallel' in log_entry:
            metrics['parallel_workers'] = log_entry.get('parallel', {}).get('workers', 0)
            metrics['active_workers'] = log_entry.get('parallel', {}).get('active', 0)

        # Détection des métriques d'épisode
        if 'episode' in log_entry:
            ep = log_entry['episode']
            metrics.update({
                'episode_length': ep.get('length', 0),
                'episode_reward': ep.get('reward', 0),
                'episode_return': ep.get('return', 0),
                'episode_time': ep.get('time', 0),
                'episode_trades': ep.get('trades', 0)
            })

            # Mise à jour des statistiques globales
            self.global_stats['total_episodes'] += 1
            self.global_stats['total_trades'] += metrics['episode_trades']
            self.global_stats['total_episode_time'] += metrics['episode_time']

            # Calcul du taux d'épisodes par seconde
            time_diff = time.time() - self.global_stats['last_update']
            if time_diff > 0:
                self.global_stats['episodes_per_second'] = 1.0 / time_diff
            self.global_stats['last_update'] = time.time()
        metrics['instance_id'] = log_entry.get('instance_id', 'unknown')

        # Extract portfolio metrics
        portfolio = log_entry.get('portfolio', {})
        metrics['portfolio_value'] = float(portfolio.get('portfolio_value', 0.0))
        metrics['cash'] = float(portfolio.get('cash', 0.0))
        metrics['shares'] = float(portfolio.get('shares', 0.0))
        metrics['asset_value'] = float(portfolio.get('asset_value', 0.0))

        # Extract episode and training metrics
        metrics['episode'] = int(log_entry.get('episode', 0))
        metrics['step'] = int(log_entry.get('step', 0))
        metrics['reward'] = float(log_entry.get('reward', 0.0))
        metrics['total_reward'] = float(log_entry.get('total_reward', 0.0))
        metrics['action'] = log_entry.get('action', 0)
        metrics['done'] = bool(log_entry.get('done', False))

        # Extract risk metrics from log message if available
        if 'message' in log_entry and '[RISK]' in log_entry['message']:
            risk_message = log_entry['message']
            # Example: [Worker 0] [RISK] Drawdown actuel: 0.02/0.82 USDT (0.1%/4.0%), Équité: 20.48 USDT, Cash: 4.08 USDT, Solde dispo: 4.08 USDT
            match = re.search(r'Drawdown actuel: ([\d.]+) / ([\d.]+) USDT \(([\d.]+)% / ([\d.]+)%\), Équité: ([\d.]+) USDT, Cash: ([\d.]+) USDT, Solde dispo: ([\d.]+) USDT', risk_message)
            if match:
                metrics['risk_drawdown_value'] = float(match.group(1))
                metrics['risk_drawdown_limit_value'] = float(match.group(2))
                metrics['risk_drawdown_pct'] = float(match.group(3))
                metrics['risk_drawdown_limit_pct'] = float(match.group(4))
                metrics['risk_equity'] = float(match.group(5))
                metrics['risk_cash'] = float(match.group(6))
                metrics['risk_solde_dispo'] = float(match.group(7))
            else:
                # Fallback if regex doesn't match, try to get equity from portfolio
                metrics['risk_equity'] = float(portfolio.get('portfolio_value', 0.0))

        # Extract training metrics if available
        training = log_entry.get('training', {})
        metrics['loss'] = float(training.get('loss', 0.0))
        metrics['learning_rate'] = float(training.get('learning_rate', 0.0))
        metrics['epsilon'] = float(training.get('epsilon', 0.0))

        # Add additional training metrics if available
        try:
            metrics.update({
                'policy_loss': float(training.get('policy_loss', 0.0)),
                'value_loss': float(training.get('value_loss', 0.0)),
                'entropy_loss': float(training.get('entropy_loss', 0.0)),
                'explained_variance': float(training.get('explained_variance', 0.0))
            })
        except (ValueError, TypeError, AttributeError) as e:
            logger.error("Error parsing training metrics: %s", e, exc_info=True)
            # Continue with basic metrics if additional ones fail

        return metrics

    def update_from_json_file(self, file_path: str):
        """Update metrics from a JSON or JSONL log file.

        Args:
            file_path: Path to the log file to process
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return

                try:
                    # Try to parse as JSON array first
                    entries = json.loads(content)
                    if not isinstance(entries, list):
                        entries = [entries]
                except json.JSONDecodeError:
                    # If not a JSON array, try parsing as JSONL
                    entries = []
                    for line in content.split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            entries.append(entry)
                        except json.JSONDecodeError as e:
                            logger.warning(
                                "Failed to parse JSON line: %s\nError: %s",
                                line, str(e)
                            )
                            continue

                valid_entries = 0
                for entry in entries:
                    if not isinstance(entry, dict):
                        logger.warning("Skipping non-dict entry: %s", entry)
                        continue

                    metrics = self.parse_json_log(entry)
                    if metrics is not None:
                        valid_entries += 1
                        # Mise à jour des métriques par worker
                        for key, value in metrics.items():
                            if key in self.data[metrics['instance_id']]:
                                self.data[metrics['instance_id']][key].append(value)

                        # Mettre à jour les statistiques globales
                        self._update_global_stats(metrics['instance_id'], metrics)

                        logger.debug(f"Métriques mises à jour pour {metrics['instance_id']}: {metrics}")

                if valid_entries > 0:
                    logger.info(
                        "Processed %d valid entries from %s",
                        valid_entries, os.path.basename(file_path)
                    )

        except (IOError, OSError) as e:
            logger.error("Error reading file %s: %s", file_path, str(e))
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "Unexpected error processing file %s: %s",
                file_path, str(e), exc_info=True
            )

            logger.error(f"Erreur lors de la lecture du fichier {file_path}: {str(e)}")

    def _update_global_stats(self, worker_id: str, metrics: Dict[str, Any]):
        """Met à jour les statistiques globales."""
        # Mettre à jour le nombre total d'épisodes
        if 'episode' in metrics:
            self.global_stats['total_episodes'] = max(
                self.global_stats['total_episodes'],
                metrics['episode']
            )

        # Mettre à jour le nombre total de trades
        if 'trades' in metrics:
            self.global_stats['total_trades'] = max(
                self.global_stats['total_trades'],
                metrics['trades']
            )

        # Mettre à jour le meilleur et le pire worker
        if 'reward' in metrics:
            current_reward = metrics['reward']

            if (self.global_stats['best_worker'] is None or
                current_reward > self.data[self.global_stats['best_worker']]['reward'][-1]):
                self.global_stats['best_worker'] = worker_id

            if (self.global_stats['worst_worker'] is None or
                current_reward < self.data[self.global_stats['worst_worker']]['reward'][-1]):
                self.global_stats['worst_worker'] = worker_id

    def check_for_new_logs(self):
        """Vérifie les nouveaux fichiers de logs et les traite."""
        try:
            # Vérifier si le répertoire de logs existe
            if not os.path.exists(self.log_dir):
                logger.warning(f"Le répertoire de logs {self.log_dir} n'existe pas")
                return

            # Parcourir les fichiers de logs
            for filename in os.listdir(self.log_dir):
                if filename.startswith('parallel_training_results_') and filename.endswith('.json'):
                    file_path = os.path.join(self.log_dir, filename)

                    # Vérifier si le fichier a été modifié depuis la dernière vérification
                    mtime = os.path.getmtime(file_path)
                    if mtime > self.last_modified:
                        self.update_from_json_file(file_path)
                        self.last_modified = mtime

                        # Ajouter le fichier à la liste des fichiers traités
                        if file_path not in self.processed_files:
                            self.processed_files.add(file_path)
                            logger.info(f"Nouveau fichier de log détecté: {filename}")

            # Mettre à jour TensorBoard et checkpoints
            self.tensorboard.refresh()
            self.checkpoints.refresh()

        except Exception as e:
            logger.error(f"Erreur lors de la vérification des logs: {str(e)}")


class TensorboardMonitor:
    """Parcourt les fichiers TensorBoard pour extraire les scalaires utiles."""

    def __init__(self, tb_dir: Optional[str] = None) -> None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Align with PPO tensorboard_log in ppo_agent.py (reports/tensorboard_logs)
        default_tb_dir = os.path.join(project_root, 'reports', 'tensorboard_logs')
        self.tb_dir = tb_dir or default_tb_dir
        self.tags = {
            'ep_rew_mean': 'rollout/ep_rew_mean',
            'policy_loss': 'train/policy_loss',
            'value_loss': 'train/value_loss',
            'entropy_loss': 'train/entropy_loss',
            'approx_kl': 'train/approx_kl',
            'clip_fraction': 'train/clip_fraction',
            'explained_variance': 'train/explained_variance',
            'learning_rate': 'train/learning_rate',
        }
        self.scalars: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=2000))
        self._event_accs: Dict[str, EventAccumulator] = {}

    def _load_event_acc(self, path: str) -> Optional[EventAccumulator]:
        try:
            acc = EventAccumulator(path)
            acc.Reload()
            return acc
        except Exception:
            return None

    def refresh(self) -> None:
        if not os.path.isdir(self.tb_dir):
            return
        # Find event files recursively
        for root, _, files in os.walk(self.tb_dir):
            for f in files:
                if f.startswith('events.out.tfevents'):
                    fp = os.path.join(root, f)
                    if fp not in self._event_accs:
                        acc = self._load_event_acc(fp)
                        if acc:
                            self._event_accs[fp] = acc
                    acc = self._event_accs.get(fp)
                    if not acc:
                        continue
                    # Extract all desired tags
                    for key, tag in self.tags.items():
                        try:
                            if tag in acc.Tags().get('scalars', []):
                                events = acc.Scalars(tag)
                                if events:
                                    # Append last point only to keep lightweight refresh
                                    self.scalars[key].append((events[-1].step, events[-1].value))
                        except Exception:
                            continue


class CheckpointMonitor:
    """Surveille le dossier checkpoints pour afficher les derniers résultats."""

    def __init__(self, ckpt_dir: Optional[str] = None) -> None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_ckpt = os.path.join(project_root, 'models', 'rl_agents', 'checkpoints')
        self.ckpt_dir = ckpt_dir or default_ckpt
        self.entries: Deque[Dict[str, Any]] = deque(maxlen=100)
        self._last_seen: set = set()

    def refresh(self) -> None:
        if not os.path.isdir(self.ckpt_dir):
            return
        for root, dirs, files in os.walk(self.ckpt_dir):
            if 'metadata.json' in files:
                meta_path = os.path.join(root, 'metadata.json')
                if meta_path in self._last_seen:
                    continue
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    meta['path'] = root
                    self.entries.append(meta)
                    self._last_seen.add(meta_path)
                except Exception:
                    continue

# Initialisation du moniteur de logs
log_monitor = LogMonitor()

# Données partagées pour la communication avec les workers
shared_stats = {
    "last_update": time.time(),
    "workers": {},
    "global": {
        "total_episodes": 0,
        "active_workers": 0,
        "avg_portfolio_value": 0,
        "total_trades": 0
    }
}

# Initialisation de l'application Dash
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True
)
app.title = "ADAN Trading - Dashboard"

# Styles personnalisés
card_style = {
    "background-color": "#1e1e1e",
    "border": "1px solid #333",
    "border-radius": "10px",
    "box-shadow": "0 4px 6px rgba(0, 0, 0, 0.3)"
}

# Layout principal du dashboard
app.layout = dbc.Container([
    # Composant d'intervalle pour les mises à jour en temps réel (toutes les 5 secondes)
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # en millisecondes
        n_intervals=0
    ),

    # En-tête avec indicateurs globaux
    dbc.Row([
        dbc.Col([
            html.H1([
                html.Img(src="/assets/adan.jpg", style={'height':'50px', 'margin-right':'15px'}),
                "l'entrainement de adan"
            ], className="text-center mb-4", style={"color": "#00ff88"})
        ], width=12)
    ]),

    dcc.Tabs(id="tabs", value='overview', children=[
        dcc.Tab(label='Overview', value='overview'),
        dcc.Tab(label='Per-worker', value='per-worker'),
        dcc.Tab(label='RL Internals', value='rl-internals'),
        dcc.Tab(label='Trades / Performance', value='trades-performance'),
        dcc.Tab(label='CNN & Perceptual', value='cnn-perceptual'),
    ]),
    html.Div(id='tabs-content')

], fluid=True, className="p-3", style={"background-image": "url(/assets/istock.jpg)", "background-size": "contain", "background-repeat": "no-repeat"})

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'overview':
        return html.Div([
            # Cartes de statut global
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Workers Actifs", className="card-title"),
                            html.H2(id="active-workers", children="0", className="text-success"),
                            html.Small("En cours d'entraînement", className="text-muted")
                        ])
                    ], style=card_style)
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Capital Moyen", className="card-title"),
                            html.H2(id="avg-capital", children="$0", className="text-info"),
                            html.Small("Tous workers confondus", className="text-muted")
                        ])
                    ], style=card_style)
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Best Performer", className="card-title"),
                            html.H2(id="best-worker", children="-", className="text-warning"),
                            html.Small("Meilleur worker", className="text-muted")
                        ])
                    ], style=card_style)
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Model Health", className="card-title"),
                            html.H2(id="model-health", children="🟡", style={"font-size": "3rem"}),
                            html.Small("État général", className="text-muted")
                        ])
                    ], style=card_style)
                ], md=3)
            ], className="mb-4"),
        ])
    elif tab == 'per-worker':
        return html.Div([
            # Graphiques principaux
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📈 Évolution du Capital par Worker"),
                        dbc.CardBody(dcc.Graph(id='capital-graph'))
                    ], style=card_style)
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🎯 Histogramme des Rewards Cumulés"),
                        dbc.CardBody(dcc.Graph(id='reward-histogram'))
                    ], style=card_style)
                ], md=6)
            ], className="mb-4"),

            # Graphiques secondaires
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📉 Drawdown Analysis"),
                        dbc.CardBody(dcc.Graph(id='drawdown-graph'))
                    ], style=card_style)
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Sharpe Ratio Evolution"),
                        dbc.CardBody(dcc.Graph(id='sharpe-graph'))
                    ], style=card_style)
                ], md=6)
            ], className="mb-4"),
        ])
    elif tab == 'rl-internals':
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Policy Loss, Value Loss, Entropy"),
                        dbc.CardBody(dcc.Graph(id='loss-entropy-graph'))
                    ], style=card_style)
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Approx KL and Clip Fraction"),
                        dbc.CardBody(dcc.Graph(id='kl-clip-graph'))
                    ], style=card_style)
                ], md=6)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Explained Variance"),
                        dbc.CardBody(dcc.Graph(id='explained-variance-graph'))
                    ], style=card_style)
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Learning Rate Schedule"),
                        dbc.CardBody(dcc.Graph(id='lr-graph'))
                    ], style=card_style)
                ], md=6)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Derniers Checkpoints"),
                        dbc.CardBody(dash_table.DataTable(
                            id='checkpoint-table',
                            columns=[
                                {"name": "Timestamp", "id": "timestamp"},
                                {"name": "Episode", "id": "episode"},
                                {"name": "Total Steps", "id": "total_steps"},
                                {"name": "Mean Reward", "id": "final_mean_reward"},
                                {"name": "Path", "id": "path"},
                            ],
                            style_cell={'textAlign': 'center', 'backgroundColor': '#1e1e1e', 'color': 'white'},
                            style_header={'backgroundColor': '#333', 'fontWeight': 'bold'},
                        ))
                    ], style=card_style)
                ], width=12)
            ])
        ])
    elif tab == 'trades-performance':
        return html.Div([
            # Tableau de performance
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🏆 Performance Leaderboard"),
                        dbc.CardBody([
                            dash_table.DataTable(
                                id='performance-table',
                                columns=[
                                    {"name": "Worker", "id": "worker"},
                                    {"name": "Capital", "id": "capital", "type": "numeric", "format": {"specifier": ".2f"}},
                                    {"name": "PnL %", "id": "pnl_pct", "type": "numeric", "format": {"specifier": ".2f"}},
                                    {"name": "Drawdown %", "id": "drawdown", "type": "numeric", "format": {"specifier": ".2f"}},
                                    {"name": "Sharpe", "id": "sharpe", "type": "numeric", "format": {"specifier": ".2f"}},
                                    {"name": "Reward Moyen", "id": "avg_reward", "type": "numeric", "format": {"specifier": ".4f"}},
                                    {"name": "Trades", "id": "trades", "type": "numeric"},
                                    {"name": "Win/Loss Ratio", "id": "win_loss_ratio", "type": "numeric", "format": {"specifier": ".2f"}},
                                    {"name": "Status", "id": "status"}
                                ],
                                style_cell={'textAlign': 'center', 'backgroundColor': '#1e1e1e', 'color': 'white'},
                                style_header={'backgroundColor': '#333', 'fontWeight': 'bold'},
                                style_data_conditional=[
                                    {
                                        'if': {'filter_query': '{pnl_pct} > 1'},
                                        'backgroundColor': '#1e4d32',
                                        'color': '#4caf50'
                                    },
                                    {
                                        'if': {'filter_query': '{pnl_pct} < -1'},
                                        'backgroundColor': '#4d1e1e',
                                        'color': '#f44336'
                                    }
                                ]
                            )
                        ])
                    ], style=card_style)
                ], width=12)
            ], className="mb-4")
        ])
    elif tab == 'cnn-perceptual':
        return html.Div([html.H3('CNN & Perceptual')])

def create_enhanced_line_chart(data: Dict, metric: str, title: str,
                              y_title: str, color_discrete_map: Dict = None):
    """Crée un graphique amélioré avec des couleurs et styles personnalisés."""
    fig = go.Figure()

    # Trier les workers pour un ordre cohérent
    sorted_workers = sorted(data.items(), key=lambda x: x[0])

    for worker_id, worker_data in sorted_workers:
        if metric in worker_data and worker_data[metric]:
            # Calculer la moyenne mobile pour un affichage plus lisse
            values = list(worker_data[metric])
            window_size = max(1, len(values) // 10)  # Taille de fenêtre adaptative

            if window_size > 1:
                df = pd.Series(values)
                smoothed = df.rolling(window=window_size, min_periods=1).mean()
                y_values = smoothed.values
            else:
                y_values = values

            # Choisir une couleur en fonction de l'ID du worker
            color_idx = hash(worker_id) % len(COLOR_PALETTE)

            fig.add_trace(go.Scatter(
                x=list(range(len(y_values))),
                y=y_values,
                name=f'Worker {worker_id}',
                mode='lines+markers',
                line=dict(width=2, color=COLOR_PALETTE[color_idx]),
                marker=dict(size=6, color=COLOR_PALETTE[color_idx]),
                opacity=0.8,
                hovertemplate=f'<b>Worker {worker_id}</b><br>' +
                            f'{y_title}: %{{y:.2f}}<br>' +
                            'Étape: %{x}<extra></extra>'
            ))

    # Ajouter une ligne de tendance si plus d'un point
    if len(fig.data) > 0 and len(fig.data[0].y) > 1:
        for trace in fig.data:
            x = np.array(list(range(len(trace.y))))
            y = np.array(trace.y)
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)

            fig.add_trace(go.Scatter(
                x=x,
                y=p(x),
                name=f'{trace.name} (tendance)',
                line=dict(
                    color=trace.line.color,
                    width=1,
                    dash='dash'
                ),
                showlegend=False,
                hoverinfo='skip'
            ))

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=16, family='Arial')
        ),
        xaxis_title='Étapes',
        yaxis_title=y_title,
        showlegend=True,
        template='plotly_dark',
        margin=dict(l=50, r=50, t=60, b=50),
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    # Ajouter des annotations pour les valeurs min/max
    for trace in fig.data:
        if len(trace.y) > 0 and 'tendance' not in trace.name:
            max_idx = np.argmax(trace.y)
            min_idx = np.argmin(trace.y)

            fig.add_annotation(
                x=max_idx,
                y=trace.y[max_idx],
                text=f'Max: {trace.y[max_idx]:.2f}',
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )

            if max_idx != min_idx:  # Ne pas ajouter d'annotation si min = max
                fig.add_annotation(
                    x=min_idx,
                    y=trace.y[min_idx],
                    text=f'Min: {trace.y[min_idx]:.2f}',
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=40
                )

    return fig

@app.callback(
    [Output('capital-graph', 'figure'),
     Output('reward-histogram', 'figure'),
     Output('drawdown-graph', 'figure'),
     Output('sharpe-graph', 'figure'),
     Output('active-workers', 'children'),
     Output('avg-capital', 'children'),
     Output('best-worker', 'children'),
     Output('model-health', 'children'),
     Output('performance-table', 'data')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n: int):
    """Met à jour tous les éléments du dashboard."""

    # Simuler des données si pas de vraies données disponibles
    if not log_monitor.data:
        # Générer des données de test
        import random
        for i in range(3):
            worker_name = f"Worker_{i}"
            for _ in range(50):
                log_monitor.data[worker_name]['capital'].append(20 + random.uniform(-2, 2))
                log_monitor.data[worker_name]['reward'].append(random.uniform(-0.1, 0.1))
                log_monitor.data[worker_name]['drawdown'].append(random.uniform(0, 5))
                log_monitor.data[worker_name]['sharpe'].append(random.uniform(-3, 1))
                log_monitor.data[worker_name]['trades'].append(random.randint(0, 10))
                log_monitor.data[worker_name]['win_loss_ratio'].append(random.uniform(0, 2))

    # Rafraîchir les données de logs, tensorboard et checkpoints
    log_monitor.check_for_new_logs()
    data = dict(log_monitor.data)

    # Créer les graphiques
    capital_fig = create_enhanced_line_chart(data, 'capital', 'Capital au fil du temps', 'Capital ($)')
    reward_fig = create_enhanced_line_chart(data, 'reward', 'Reward cumulé', 'Reward')

    # Créer un graphique de l'utilisation des workers
    worker_fig = go.Figure()
    for worker_id, worker_data in data.items():
        if 'parallel_workers' in worker_data and len(worker_data['parallel_workers']) > 0:
            worker_fig.add_trace(go.Scatter(
                x=list(range(len(worker_data['parallel_workers']))),
                y=list(worker_data['parallel_workers']),
                name=f'Worker {worker_id}'
            ))

    worker_fig.update_layout(
        title='Utilisation des workers parallèles',
        xaxis_title='Étapes',
        yaxis_title='Nombre de workers',
        showlegend=True,
        template='plotly_dark',
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    # Créer des graphiques supplémentaires
    drawdown_fig = create_enhanced_line_chart(data, 'drawdown', 'Drawdown Analysis', 'Drawdown (%)')
    sharpe_fig = create_enhanced_line_chart(data, 'sharpe', 'Sharpe Ratio Evolution', 'Sharpe Ratio')

    # Mettre à jour les statistiques globales
    stats = log_monitor.global_stats

    # Get initial balance from config (assuming it's consistent across workers)
    initial_balance = log_monitor.config.get("environment", {}).get("initial_balance", 20.50) # Default to 20.50 if not found

    # Calculer les performances des workers
    table_data = []
    active_workers = 0
    avg_capital = 0
    best_worker = None
    best_reward = float('-inf')
    for worker, worker_data in data.items():
        if worker_data['reward']:
            current_reward = worker_data['reward'][-1]
            if current_reward > best_reward:
                best_reward = current_reward
                best_worker = worker
            active_workers += 1

            # Use risk_equity for current capital if available, otherwise fallback to portfolio_value
            current_capital = worker_data['risk_equity'][-1] if worker_data['risk_equity'] else worker_data['capital'][-1]

            avg_capital += current_capital

            # Calculate PnL % dynamically
            pnl_pct = ((current_capital - initial_balance) / initial_balance) * 100 if initial_balance > 0 else 0

            if pnl_pct > 1:
                status = "🟢 Performant"
            elif pnl_pct < -1:
                status = "🔴 En Perte"
            else:
                status = "🟡 Neutre"

            # Données pour le tableau
            table_data.append({
                'worker': worker,
                'capital': current_capital,
                'pnl_pct': pnl_pct,
                'drawdown': worker_data['risk_drawdown_pct'][-1] if worker_data['risk_drawdown_pct'] else 0,
                'sharpe': worker_data['sharpe'][-1] if worker_data['sharpe'] else 0,
                'avg_reward': np.mean(worker_data['reward']) if worker_data['reward'] else 0,
                'trades': worker_data['trades'][-1] if worker_data['trades'] else 0,
                'win_loss_ratio': np.mean(worker_data['win_loss_ratio']) if worker_data['win_loss_ratio'] else 0, # Assuming win_loss_ratio is tracked
                'status': status
            })

    if active_workers > 0:
        avg_capital /= active_workers

    # Déterminer la santé du modèle
    if avg_capital > 20:
        model_health = "🟢"  # Bon
    elif avg_capital > 19:
        model_health = "🟡"  # Moyen
    else:
        model_health = "🔴"  # Mauvais

    return (
        capital_fig,
        reward_fig,
        drawdown_fig,
        sharpe_fig,
        str(active_workers),
        f"${avg_capital:.2f}",
        best_worker,
        model_health,
        table_data
    )

@app.callback(
    Output('log-status', 'children'),
    [Input('upload-data', 'contents')],
    prevent_initial_call=True
)
def handle_log_upload(contents):
    """Gère l'upload de fichiers de logs."""
    if contents is not None:
        # Décoder le contenu du fichier
        import base64
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string).decode('utf-8')

        # Mettre à jour le moniteur avec le contenu des logs
        log_monitor.update_from_logs(decoded, "Uploaded_Worker")

        return dbc.Alert("Logs chargés avec succès! 📁", color="success")

    return ""

@app.callback(
    [Output('loss-entropy-graph', 'figure'),
     Output('kl-clip-graph', 'figure'),
     Output('explained-variance-graph', 'figure'),
     Output('lr-graph', 'figure'),
     Output('checkpoint-table', 'data')],
    [Input('interval-component', 'n_intervals')]
)
def update_rl_internals(n: int):
    tb = log_monitor.tensorboard
    # Build figures from TensorBoard scalars
    def build_series(deq):
        xs = [s for s, _ in deq]
        ys = [v for _, v in deq]
        return xs, ys

    # Loss/Entropy
    x_pl, y_pl = build_series(tb.scalars['policy_loss'])
    x_vl, y_vl = build_series(tb.scalars['value_loss'])
    x_el, y_el = build_series(tb.scalars['entropy_loss'])
    loss_entropy_fig = go.Figure()
    if y_pl:
        loss_entropy_fig.add_trace(go.Scatter(x=x_pl, y=y_pl, name='Policy Loss'))
    if y_vl:
        loss_entropy_fig.add_trace(go.Scatter(x=x_vl, y=y_vl, name='Value Loss'))
    if y_el:
        loss_entropy_fig.add_trace(go.Scatter(x=x_el, y=y_el, name='Entropy'))
    loss_entropy_fig.update_layout(template='plotly_dark', title='Policy/Value/Entropy')

    # KL / Clip
    x_kl, y_kl = build_series(tb.scalars['approx_kl'])
    x_cf, y_cf = build_series(tb.scalars['clip_fraction'])
    kl_clip_fig = go.Figure()
    if y_kl:
        kl_clip_fig.add_trace(go.Scatter(x=x_kl, y=y_kl, name='Approx KL'))
    if y_cf:
        kl_clip_fig.add_trace(go.Scatter(x=x_cf, y=y_cf, name='Clip Fraction'))
    kl_clip_fig.update_layout(template='plotly_dark', title='Approx KL / Clip Fraction')

    # Explained variance
    x_ev, y_ev = build_series(tb.scalars['explained_variance'])
    explained_variance_fig = go.Figure()
    if y_ev:
        explained_variance_fig.add_trace(go.Scatter(x=x_ev, y=y_ev, name='Explained Variance'))
    explained_variance_fig.update_layout(template='plotly_dark', title='Explained Variance')

    # Learning rate
    x_lr, y_lr = build_series(tb.scalars['learning_rate'])
    lr_fig = go.Figure()
    if y_lr:
        lr_fig.add_trace(go.Scatter(x=x_lr, y=y_lr, name='Learning Rate'))
    lr_fig.update_layout(template='plotly_dark', title='Learning Rate')

    # Checkpoints table
    ckpt_rows = []
    for e in list(log_monitor.checkpoints.entries)[-20:][::-1]:
        ckpt_rows.append({
            'timestamp': e.get('timestamp', ''),
            'episode': e.get('episode', ''),
            'total_steps': e.get('total_steps', ''),
            'final_mean_reward': e.get('final_mean_reward', e.get('mean_reward', '')),
            'path': e.get('path', ''),
        })

    return loss_entropy_fig, kl_clip_fig, explained_variance_fig, lr_fig, ckpt_rows

def run_dashboard(host: str = "0.0.0.0", port: int = 8050, debug: bool = False) -> None:
    """Lance le serveur de dashboard."""
    print("=" * 60)
    print("🚀 ADAN Trading Dashboard")
    print("=" * 60)
    print(f"📡 Serveur: http://{host}:{port}")
    print(f"🔧 Mode debug: {'Activé' if debug else 'Désactivé'}")
    print("📊 Fonctionnalités:")
    print("   • Monitoring en temps réel")
    print("   • Comparaison multi-workers")
    print("   • Analyse de performance")
    print("   • Export des données")
    print("=" * 60)
    print("Appuyez sur Ctrl+C pour arrêter")

    app.run_server(host=host, port=port, debug=debug)

# Styles pour les onglets
tab_style = {
    'borderBottom': '1px solid #444',
    'padding': '10px',
    'fontWeight': 'bold',
    'color': '#aaa',
    'backgroundColor': '#2d2d2d'
}

tab_selected_style = {
    'borderTop': '3px solid #00ff88',
    'borderBottom': '1px solid #444',
    'backgroundColor': '#1e1e1e',
    'color': 'white',
    'padding': '10px',
    'fontWeight': 'bold'
}

# Styles pour les cartes de métriques
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

# Définir le style des cartes de métriques
app.layout = html.Div([
    dcc.Store(id='store-data'),
    html.Div(id='page-content')
])

# Appliquer des styles CSS personnalisés
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>ADAN Trading Bot Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            .metric-card {
                background-color: #2d2d2d;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border-left: 4px solid #00ff88;
                height: 100%;
            }
            .metric-card h4 {
                color: #00ff88;
                margin-top: 0;
                border-bottom: 1px solid #444;
                padding-bottom: 10px;
            }
            .metric-card p {
                margin: 5px 0;
                color: #ddd;
            }
            .nav-tabs .nav-link.active {
                color: #00ff88 !important;
                background-color: #1e1e1e !important;
                border-color: #444 #444 #1e1e1e !important;
            }
            .nav-tabs .nav-link {
                color: #aaa !important;
            }
            .tab-content {
                background-color: #1e1e1e;
                padding: 15px;
                border: 1px solid #444;
                border-top: none;
                border-radius: 0 0 5px 5px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    logger.info("Démarrage du tableau de bord d'entraînement...")
    run_dashboard(debug=True)
