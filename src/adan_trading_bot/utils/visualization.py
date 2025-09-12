"""
Module de visualisation pour le projet ADAN Trading Bot.

Fournit des fonctions pour visualiser les performances de trading,
les décisions de l'agent et les métriques d'entraînement.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style
from typing import Dict, List, Optional, Tuple, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Configuration du style des graphiques
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('ggplot')  # Fallback si seaborn n'est pas disponible
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 12

class TradingVisualizer:
    """Classe pour visualiser les performances de trading."""
    
    def __init__(self, results_dir: str = 'results'):
        """
        Initialise le visualiseur avec le répertoire de sauvegarde des résultats.
        
        Args:
            results_dir: Chemin vers le répertoire de sauvegarde des graphiques
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def plot_portfolio_value(
        self, 
        portfolio_values: List[float],
        benchmark_values: Optional[List[float]] = None,
        title: str = 'Évolution de la valeur du portefeuille',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Trace l'évolution de la valeur du portefeuille au fil du temps.
        
        Args:
            portfolio_values: Liste des valeurs du portefeuille
            benchmark_values: Valeurs de référence (ex: marché)
            title: Titre du graphique
            save_path: Chemin pour sauvegarder le graphique
            
        Returns:
            Figure Plotly
        """
        fig = go.Figure()
        
        # Ajout de la courbe du portefeuille
        fig.add_trace(go.Scatter(
            y=portfolio_values,
            mode='lines',
            name='Portefeuille',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Ajout du benchmark si fourni
        if benchmark_values is not None:
            fig.add_trace(go.Scatter(
                y=benchmark_values,
                mode='lines',
                name='Benchmark',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ))
        
        # Mise en forme du graphique
        fig.update_layout(
            title=title,
            xaxis_title='Périodes',
            yaxis_title='Valeur',
            template='plotly_white',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        # Sauvegarde si un chemin est fourni
        if save_path:
            fig.write_html(os.path.join(self.results_dir, f'{save_path}.html'))
            fig.write_image(os.path.join(self.results_dir, f'{save_path}.png'))
        
        return fig
    
    def plot_actions(
        self,
        actions: List[int],
        prices: List[float],
        timestamps: Optional[List] = None,
        title: str = 'Décisions de trading',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Visualise les décisions d'achat/vente par rapport au prix.
        
        Args:
            actions: Liste des actions (0: vente, 1: neutre, 2: achat)
            prices: Liste des prix
            timestamps: Liste des horodatages (optionnel)
            title: Titre du graphique
            save_path: Chemin pour sauvegarder le graphique
            
        Returns:
            Figure Plotly
        """
        if timestamps is None:
            timestamps = list(range(len(prices)))
        
        # Création de la figure avec deux sous-graphiques
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # Graphique des prix
        fig.add_trace(
            go.Scatter(x=timestamps, y=prices, name='Prix', line=dict(color='#1f77b4')),
            row=1, col=1
        )
        
        # Points d'achat (action = 2)
        buy_indices = [i for i, a in enumerate(actions) if a == 2]
        if buy_indices:
            fig.add_trace(
                go.Scatter(
                    x=[timestamps[i] for i in buy_indices],
                    y=[prices[i] for i in buy_indices],
                    mode='markers',
                    name='Achat',
                    marker=dict(symbol='triangle-up', size=10, color='green')
                ),
                row=1, col=1
            )
        
        # Points de vente (action = 0)
        sell_indices = [i for i, a in enumerate(actions) if a == 0]
        if sell_indices:
            fig.add_trace(
                go.Scatter(
                    x=[timestamps[i] for i in sell_indices],
                    y=[prices[i] for i in sell_indices],
                    mode='markers',
                    name='Vente',
                    marker=dict(symbol='triangle-down', size=10, color='red')
                ),
                row=1, col=1
            )
        
        # Graphique des actions (en bas)
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=actions,
                mode='lines+markers',
                name='Action',
                line=dict(color='purple', width=1),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
        
        # Mise en forme du graphique
        fig.update_layout(
            title=title,
            xaxis_title='Temps',
            yaxis_title='Prix',
            yaxis2_title='Action',
            template='plotly_white',
            hovermode='x unified',
            height=800,
            showlegend=True
        )
        
        # Configuration des axes Y
        fig.update_yaxes(title_text='Prix', row=1, col=1)
        fig.update_yaxes(title_text='Action', row=2, col=1)
        
        # Sauvegarde si un chemin est fourni
        if save_path:
            fig.write_html(os.path.join(self.results_dir, f'{save_path}.html'))
            fig.write_image(os.path.join(self.results_dir, f'{save_path}.png'))
        
        return fig
    
    def plot_returns_distribution(
        self,
        returns: List[float],
        title: str = 'Distribution des rendements',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Affiche la distribution des rendements.
        
        Args:
            returns: Liste des rendements
            title: Titre du graphique
            save_path: Chemin pour sauvegarder le graphique
            
        Returns:
            Figure Plotly
        """
        fig = px.histogram(
            x=returns,
            nbins=100,
            title=title,
            labels={'x': 'Rendement', 'y': 'Fréquence'},
            opacity=0.7,
            color_discrete_sequence=['#1f77b4']
        )
        
        # Ajout d'une courbe de densité
        fig.add_trace(
            go.Scatter(
                x=np.linspace(min(returns), max(returns), 100),
                y=np.histogram(returns, bins=100, density=True)[0],
                mode='lines',
                name='Densité',
                line=dict(color='#ff7f0e', width=2)
            )
        )
        
        # Ligne verticale à zéro
        fig.add_shape(
            type='line',
            x0=0, y0=0, x1=0, y1=1,
            yref='paper',
            line=dict(color='red', width=1, dash='dash')
        )
        
        # Mise en forme
        fig.update_layout(
            template='plotly_white',
            showlegend=True,
            bargap=0.01
        )
        
        # Sauvegarde si un chemin est fourni
        if save_path:
            fig.write_html(os.path.join(self.results_dir, f'{save_path}.html'))
            fig.write_image(os.path.join(self.results_dir, f'{save_path}.png'))
        
        return fig
    
    def plot_training_metrics(
        self,
        metrics: Dict[str, List[float]],
        title: str = 'Métriques d\'entraînement',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Affiche les métriques d'entraînement au fil du temps.
        
        Args:
            metrics: Dictionnaire des métriques à afficher
            title: Titre du graphique
            save_path: Chemin pour sauvegarder le graphique
            
        Returns:
            Figure Plotly
        """
        fig = go.Figure()
        
        # Ajout de chaque métrique
        for name, values in metrics.items():
            if values:  # Ne pas tracer les listes vides
                fig.add_trace(go.Scatter(
                    y=values,
                    mode='lines',
                    name=name,
                    opacity=0.8
                ))
        
        # Mise en forme
        fig.update_layout(
            title=title,
            xaxis_title='Étapes d\'entraînement',
            yaxis_title='Valeur',
            template='plotly_white',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        # Sauvegarde si un chemin est fourni
        if save_path:
            fig.write_html(os.path.join(self.results_dir, f'{save_path}.html'))
            fig.write_image(os.path.join(self.results_dir, f'{save_path}.png'))
        
        return fig

# Fonction utilitaire pour générer un rapport complet
def generate_training_report(
    training_metrics: Dict[str, List[float]],
    portfolio_values: List[float],
    actions: List[int],
    prices: List[float],
    returns: List[float],
    output_dir: str = 'reports'
) -> None:
    """
    Génère un rapport complet de l'entraînement.
    
    Args:
        training_metrics: Métriques d'entraînement
        portfolio_values: Valeurs du portefeuille au fil du temps
        actions: Actions prises par l'agent
        prices: Prix de l'actif
        returns: Rendements
        output_dir: Répertoire de sortie
    """
    # Création du visualiseur
    visualizer = TradingVisualizer(output_dir)
    
    # Génération des graphiques
    visualizer.plot_training_metrics(
        training_metrics,
        save_path='training_metrics'
    )
    
    visualizer.plot_portfolio_value(
        portfolio_values,
        save_path='portfolio_value'
    )
    
    visualizer.plot_actions(
        actions,
        prices,
        save_path='trading_actions'
    )
    
    visualizer.plot_returns_distribution(
        returns,
        save_path='returns_distribution'
    )
    
    # Génération d'un rapport HTML
    report_path = os.path.join(output_dir, 'training_report.html')
    
    with open(report_path, 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport d'entraînement ADAN Trading Bot</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #1f77b4; }
                .container { margin-bottom: 40px; }
                .plot { margin: 20px 0; }
                img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <h1>Rapport d'entraînement ADAN Trading Bot</h1>
            <div class="container">
                <h2>Métriques d'entraînement</h2>
                <div class="plot">
                    <iframe src="training_metrics.html" width="100%" height="500"></iframe>
                </div>
            </div>
            <div class="container">
                <h2>Valeur du portefeuille</h2>
                <div class="plot">
                    <iframe src="portfolio_value.html" width="100%" height="500"></iframe>
                </div>
            </div>
            <div class="container">
                <h2>Décisions de trading</h2>
                <div class="plot">
                    <iframe src="trading_actions.html" width="100%" height="600"></iframe>
                </div>
            </div>
            <div class="container">
                <h2>Distribution des rendements</h2>
                <div class="plot">
                    <iframe src="returns_distribution.html" width="100%" height="500"></iframe>
                </div>
            </div>
        </body>
        </html>
        """)
    
    print(f"Rapport généré avec succès : {report_path}")
