"""Module de façonnage des récompenses pour l'environnement de trading.

Ce module fournit des fonctionnalités pour façonner et adapter les récompenses
en fonction de divers facteurs et métriques de performance.
"""
from typing import Dict, Any, Optional
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

        Args:
            raw_reward: Récompense brute à façonner
            metrics: Dictionnaire de métriques pour le façonnage

        Returns:
            Récompense façonnée
        """
        shaped_reward = raw_reward
        shaped_reward *= self.risk_factor

        if 'drawdown' in metrics and metrics['drawdown'] > 0.1:
            shaped_reward -= self.drawdown_penalty * metrics['drawdown']

        if 'sharpe_ratio' in metrics:
            shaped_reward *= (1 + self.sharpe_factor * metrics['sharpe_ratio'])

        if 'volatility' in metrics:
            shaped_reward /= (1 + metrics['volatility'] * self.volatility_factor)

        return float(np.clip(shaped_reward, -1e6, 1e6))
