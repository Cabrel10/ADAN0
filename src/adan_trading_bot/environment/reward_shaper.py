"""Module de façonnage des récompenses pour l'environnement de trading.

Ce module fournit des fonctionnalités pour façonner et adapter les récompenses
en fonction de divers facteurs et métriques de performance.
"""
# Standard library imports
from typing import Any, Dict, Optional

# Third-party imports
import numpy as np


class RewardShaper:
    """Classe pour façonner les récompenses en fonction de divers facteurs."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialise le façonneur de récompenses.

        Args:
            config: Configuration pour le façonnage des récompenses
        """
        self.config = config or {}
        self._setup_shaping_factors()

    def _setup_shaping_factors(self) -> None:
        """Configure les facteurs de façonnage à partir de la configuration."""
        self.risk_factor = self.config.get('risk_factor', 1.0)
        self.volatility_factor = self.config.get('volatility_factor', 1.0)
        self.drawdown_penalty = self.config.get('drawdown_penalty', 0.0)
        self.sharpe_factor = self.config.get('sharpe_factor', 1.0)

    def shape_reward(
        self,
        raw_reward: float,
        metrics: Dict[str, float]
    ) -> float:
        """Modifie la récompense brute en fonction des métriques fournies.

        Applique différents facteurs de façonnage pour ajuster la récompense
        en fonction du risque, du drawdown, du ratio de Sharpe et de la volatilité.

        Args:
            raw_reward: Récompense brute à façonner
            metrics: Dictionnaire de métriques pour le façonnage, qui peut
                contenir les clés suivantes :
                - drawdown: Taux de drawdown actuel (entre 0 et 1)
                - sharpe_ratio: Ratio de Sharpe actuel
                - volatility: Volatilité actuelle

        Returns:
            float: Récompense façonnée, bornée entre -1e6 et 1e6

        Example:
            >>> shaper = RewardShaper()
            >>> reward = shaper.shape_reward(1.0, {'drawdown': 0.05, 'sharpe_ratio': 1.5, 'volatility': 0.1})
        """
        try:
            shaped_reward = float(raw_reward) * self.risk_factor

            # Pénalité pour drawdown élevé
            if 'drawdown' in metrics and metrics['drawdown'] > 0.1:
                shaped_reward -= self.drawdown_penalty * metrics['drawdown']

            # Récompense pour un bon ratio de Sharpe
            if 'sharpe_ratio' in metrics:
                shaped_reward *= (1 + self.sharpe_factor * metrics['sharpe_ratio'])

            # Réduction de la récompense en cas de forte volatilité
            if 'volatility' in metrics:
                volatility_impact = 1 + (metrics['volatility'] * self.volatility_factor)
                shaped_reward /= max(1e-10, volatility_impact)  # Éviter la division par zéro

            return float(np.clip(shaped_reward, -1e6, 1e6))

        except (TypeError, ValueError) as e:
            # En cas d'erreur, retourner une récompense neutre
            print(f"Erreur lors du façonnage de la récompense: {e}")
            return 0.0
