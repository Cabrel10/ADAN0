"""
Dynamic Behavior Engine avec méta-apprentissage adaptatif pour ADAN Trading Bot.
Implémente la tâche 9.2.1 - Évolution paramètres DBE.
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import deque

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Types de régimes de marché"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"


@dataclass
class DBEParameters:
    """Paramètres du DBE avec valeurs par défaut"""
    # Seuils de risque
    risk_threshold_low: float = 0.3
    risk_threshold_medium: float = 0.6
    risk_threshold_high: float = 0.8

    # Paramètres de volatilité
    volatility_window: int = 20
    volatility_threshold_low: float = 0.01
    volatility_threshold_high: float = 0.05

    # Paramètres de drawdown
    max_drawdown_threshold: float = 0.15
    drawdown_recovery_factor: float = 0.5

    # Paramètres de performance
    min_sharpe_ratio: float = 0.5
    performance_window: int = 100

    # Paramètres d'adaptation
    adaptation_rate: float = 0.1
    learning_rate: float = 0.01
    momentum: float = 0.9

    # Paramètres de régime
    regime_detection_window: int = 50
    trend_threshold: float = 0.02

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DBEParameters':
        """Crée depuis un dictionnaire"""
        return cls(**data)


class ParameterEvolution:
    """Système d'évolution des paramètres DBE"""

    def __init__(self, initial_params: DBEParameters):
        self.current_params = initial_params
        self.param_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        self.gradient_estimates = {}
        self.momentum_terms = {}

        # Initialize gradient estimates and momentum
        for param_name in asdict(initial_params).keys():
            self.gradient_estimates[param_name] = 0.0
            self.momentum_terms[param_name] = 0.0

    def update_parameters(self, performance_metrics: Dict[str, float],
                          market_conditions: Dict[str, float]) -> DBEParameters:
        """
        Met à jour les paramètres basé sur la performance et les conditions de marché.

        Args:
            performance_metrics: Métriques de performance récentes
            market_conditions: Conditions actuelles du marché

        Returns:
            Nouveaux paramètres DBE
        """
        # Enregistrer l'état actuel
        self.param_history.append(self.current_params.to_dict())
        self.performance_history.append(performance_metrics.copy())

        # Calculer les gradients estimés
        gradients = self._estimate_gradients(
            performance_metrics, market_conditions)

        # Mettre à jour les paramètres avec momentum
        new_params_dict = self.current_params.to_dict()

        for param_name, gradient in gradients.items():
            if param_name in new_params_dict:
                # Mise à jour du momentum
                self.momentum_terms[param_name] = (
                    self.current_params.momentum * self.momentum_terms[param_name] +
                    self.current_params.learning_rate * gradient
                )

                # Mise à jour du paramètre
                old_value = new_params_dict[param_name]
                new_value = old_value + self.momentum_terms[param_name]

                # Contraintes sur les valeurs
                new_value = self._apply_constraints(param_name, new_value)
                new_params_dict[param_name] = new_value

                logger.debug(f"Parameter {param_name}: {old_value:.4f} -> {new_value:.4f} "
                             f"(gradient: {gradient:.6f})")

        # Créer nouveaux paramètres
        self.current_params = DBEParameters.from_dict(new_params_dict)

        return self.current_params

    def _estimate_gradients(self, performance_metrics: Dict[str, float],
                            market_conditions: Dict[str, float]) -> Dict[str, float]:
        """Estime les gradients pour chaque paramètre"""
        gradients = {}

        if len(self.performance_history) < 2:
            return {param: 0.0 for param in asdict(self.current_params).keys()}

        # Performance actuelle vs précédente
        current_perf = performance_metrics.get('sharpe_ratio', 0.0)
        previous_perf = self.performance_history[-2].get('sharpe_ratio', 0.0)
        perf_delta = current_perf - previous_perf

        # Gradients basés sur la performance et les conditions de marché
        volatility = market_conditions.get('volatility', 0.02)
        drawdown = market_conditions.get('current_drawdown', 0.0)
        trend_strength = market_conditions.get('trend_strength', 0.0)

        # Adaptation des seuils de risque
        if perf_delta > 0 and drawdown < 0.05:
            # Performance positive, on peut être plus agressif
            gradients['risk_threshold_low'] = 0.01
            gradients['risk_threshold_medium'] = 0.01
            gradients['risk_threshold_high'] = 0.01
        elif perf_delta < 0 or drawdown > 0.1:
            # Performance négative ou drawdown élevé, être plus conservateur
            gradients['risk_threshold_low'] = -0.01
            gradients['risk_threshold_medium'] = -0.01
            gradients['risk_threshold_high'] = -0.01
        else:
            gradients['risk_threshold_low'] = 0.0
            gradients['risk_threshold_medium'] = 0.0
            gradients['risk_threshold_high'] = 0.0

        # Adaptation des seuils de volatilité
        if volatility > 0.04:  # Haute volatilité
            gradients['volatility_threshold_low'] = 0.001
            gradients['volatility_threshold_high'] = 0.002
        elif volatility < 0.01:  # Basse volatilité
            gradients['volatility_threshold_low'] = -0.001
            gradients['volatility_threshold_high'] = -0.001
        else:
            gradients['volatility_threshold_low'] = 0.0
            gradients['volatility_threshold_high'] = 0.0

        # Adaptation du seuil de drawdown
        if drawdown > 0.1:
            gradients['max_drawdown_threshold'] = -0.005  # Plus strict
        elif drawdown < 0.02 and perf_delta > 0:
            gradients['max_drawdown_threshold'] = 0.002   # Plus permissif
        else:
            gradients['max_drawdown_threshold'] = 0.0

        # Adaptation du Sharpe ratio minimum
        if current_perf > 1.0:
            gradients['min_sharpe_ratio'] = 0.01  # Augmenter les standards
        elif current_perf < 0.2:
            gradients['min_sharpe_ratio'] = -0.01  # Réduire les standards
        else:
            gradients['min_sharpe_ratio'] = 0.0

        # Adaptation du taux d'apprentissage
        if abs(perf_delta) > 0.1:  # Changements importants
            gradients['learning_rate'] = -0.001  # Réduire le learning rate
        elif abs(perf_delta) < 0.01:  # Changements faibles
            gradients['learning_rate'] = 0.0005  # Augmenter légèrement
        else:
            gradients['learning_rate'] = 0.0

        # Paramètres par défaut pour les autres
        for param_name in asdict(self.current_params).keys():
            if param_name not in gradients:
                gradients[param_name] = 0.0

        return gradients

    def _apply_constraints(self, param_name: str, value: float) -> float:
        """Applique les contraintes sur les valeurs des paramètres"""
        constraints = {
            'risk_threshold_low': (0.1, 0.5),
            'risk_threshold_medium': (0.3, 0.8),
            'risk_threshold_high': (0.6, 0.95),
            'volatility_threshold_low': (0.005, 0.02),
            'volatility_threshold_high': (0.02, 0.1),
            'max_drawdown_threshold': (0.05, 0.3),
            'drawdown_recovery_factor': (0.1, 0.9),
            'min_sharpe_ratio': (0.1, 2.0),
            'adaptation_rate': (0.01, 0.5),
            'learning_rate': (0.001, 0.1),
            'momentum': (0.5, 0.99),
            'trend_threshold': (0.005, 0.05),
            'volatility_window': (10, 100),
            'performance_window': (50, 500),
            'regime_detection_window': (20, 200)
        }

        if param_name in constraints:
            min_val, max_val = constraints[param_name]
            return np.clip(value, min_val, max_val)

        return value

    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'adaptation"""
        if len(self.param_history) < 2:
            return {'adaptation_count': 0}

        # Calculer les changements de paramètres
        current = self.current_params.to_dict()
        initial = self.param_history[0]

        changes = {}
        for param_name in current.keys():
            if param_name in initial:
                change = abs(current[param_name] - initial[param_name])
                changes[param_name] = change

        return {
            'adaptation_count': len(self.param_history),
            'parameter_changes': changes,
            'total_adaptation': sum(changes.values()),
            'most_adapted_param': max(changes.keys(), key=lambda k: changes[k]) if changes else None,
            'current_learning_rate': self.current_params.learning_rate,
            'current_momentum': self.current_params.momentum
        }


class MarketRegimeDetector:
    """
    Détecteur de régimes de marché avec volatilité EWMA et scores exponentiels.

    Utilise une approche non-linéaire pour la détection des régimes avec :
    - Volatilité EWMA (Exponentially Weighted Moving Average)
    - Scores de régime exponentiels
    - Détection adaptative des seuils
    """

    def __init__(self, window_size: int = 50, ewma_lambda: float = 0.94):
        """
        Initialise le détecteur de régimes.

        Args:
            window_size: Taille de la fenêtre d'observation
            ewma_lambda: Facteur de lissage pour la volatilité EWMA (0 < λ < 1)
        """
        self.window_size = window_size
        self.ewma_lambda = ewma_lambda
        self.ewma_variance = None
        self.price_history = deque(maxlen=window_size)
        self.volume_history = deque(maxlen=window_size)
        self.volatility_history = deque(maxlen=window_size)

    def update(self, price: float, volume: float, volatility: float) -> None:
        """
        Met à jour l'historique des données et calcule la volatilité EWMA.

        Args:
            price: Prix actuel de l'actif
            volume: Volume échangé
            volatility: Volatilité instantanée
        """
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.volatility_history.append(volatility)

        # Calcul des rendements logarithmiques
        if len(self.price_history) > 1:
            log_returns = np.log(np.array(self.price_history)[
                                 1:] / np.array(self.price_history)[:-1])

            # Initialisation de la variance EWMA si nécessaire
            if self.ewma_variance is None:
                self.ewma_variance = np.var(log_returns)

            # Mise à jour de la variance EWMA
            for r in log_returns[-1:]:  # Mise à jour avec le dernier rendement
                self.ewma_variance = (self.ewma_lambda * self.ewma_variance +
                                      (1 - self.ewma_lambda) * r**2)

    def get_ewma_volatility(self) -> float:
        """Retourne la volatilité EWMA actuelle."""
        if self.ewma_variance is None:
            return 0.0
        return np.sqrt(self.ewma_variance)

    def calculate_regime_score(self) -> float:
        """
        Calcule un score de régime exponentiel.

        Returns:
            Score numérique où :
            - > 1.5 : Tendance haussière
            - < -1.5 : Tendance baissière
            - Entre -0.5 et 0.5 : Marché plat
            - > 2.0 : Haute volatilité
            - < -2.0 : Crise
        """
        if len(self.price_history) < 2:
            return 0.0

        prices = np.array(self.price_history)
        log_returns = np.log(prices[1:] / prices[:-1])

        # Calcul des métriques
        trend_strength = np.mean(log_returns) * 100  # En pourcentage
        ewma_vol = self.get_ewma_volatility() * 100  # En pourcentage

        # Score de tendance (fonction tanh pour borner entre -1 et 1)
        # Plus sensible aux faibles tendances
        trend_score = np.tanh(trend_strength / 2.0)

        # Score de volatilité (fonction sigmoïde pour une réponse non-linéaire)
        vol_score = 2 / (1 + np.exp(-10 * (ewma_vol - 0.03))) - 1  # Seuil à 3%

        # Score composite avec pondération non-linéaire
        composite_score = trend_score * \
            np.exp(-0.5 * vol_score**2) + 0.5 * vol_score

        return composite_score

    def detect_regime(self) -> MarketRegime:
        """
        Détecte le régime de marché actuel avec une approche non-linéaire.

        Returns:
            MarketRegime: Le régime de marché détecté
        """
        if len(self.price_history) < self.window_size // 2:  # Moins de données nécessaires avec EWMA
            return MarketRegime.SIDEWAYS

        score = self.calculate_regime_score()
        ewma_vol = self.get_ewma_volatility()

        # Détection de crise (score très bas et volatilité élevée)
        if score < -2.0 and ewma_vol > 0.08:
            return MarketRegime.CRISIS

        # Détection de haute volatilité
        if ewma_vol > 0.06:  # Seuil plus bas que l'ancienne version
            return MarketRegime.HIGH_VOLATILITY

        # Détection de tendance
        if score > 1.5:
            return MarketRegime.TRENDING_UP
        elif score < -1.5:
            return MarketRegime.TRENDING_DOWN

        # Détection de faible volatilité
        if ewma_vol < 0.008:  # Seuil plus bas pour éviter les faux positifs
            return MarketRegime.LOW_VOLATILITY

        # Par défaut, marché plat
        return MarketRegime.SIDEWAYS

    def get_market_conditions(self) -> Dict[str, float]:
        """Retourne les conditions actuelles du marché"""
        if len(self.price_history) < 2:
            return {
                'volatility': 0.02,
                'trend_strength': 0.0,
                'current_drawdown': 0.0,
                'regime_stability': 1.0
            }

        prices = np.array(self.price_history)
        volatilities = np.array(self.volatility_history)

        # Calculs
        returns = np.diff(prices) / prices[:-1]
        trend_strength = np.mean(returns)
        volatility = np.mean(volatilities)

        # Drawdown
        peak = np.maximum.accumulate(prices)
        drawdown = (peak - prices) / peak
        current_drawdown = drawdown[-1]

        # Stabilité du régime (variance des rendements)
        regime_stability = 1.0 / (1.0 + np.std(returns))

        return {
            'volatility': volatility,
            'trend_strength': trend_strength,
            'current_drawdown': current_drawdown,
            'regime_stability': regime_stability
        }


class AdaptiveDBE:
    """
    Dynamic Behavior Engine avec adaptation automatique des paramètres
    et gestion non-linéaire des positions basée sur la volatilité.
    """

    def __init__(self, initial_params: Optional[DBEParameters] = None,
                 adaptation_enabled: bool = True,
                 save_path: str = "logs/adaptive_dbe",
                 tau: float = 0.03,  # Paramètre de sensibilité à la volatilité
                 ewma_lambda: float = 0.94):  # Facteur de lissage EWMA
        """
        Initialise le DBE avec adaptation des paramètres.

        Args:
            initial_params: Paramètres initiaux du DBE
            adaptation_enabled: Active/désactive l'adaptation
            save_path: Chemin pour sauvegarder les logs
            tau: Paramètre de sensibilité à la volatilité (plus petit = plus sensible)
            ewma_lambda: Facteur de lissage pour la volatilité EWMA (0 < λ < 1)
        """
        self.params = initial_params or DBEParameters()
        self.adaptation_enabled = adaptation_enabled
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.tau = tau

        # Composants d'adaptation avec paramètres non-linéaires
        self.param_evolution = ParameterEvolution(self.params)
        self.regime_detector = MarketRegimeDetector(ewma_lambda=ewma_lambda)

        # Historique et métriques
        self.performance_history = deque(maxlen=1000)
        self.regime_history = deque(maxlen=100)
        self.adaptation_log = []

        # Paramètres de position dynamiques
        self.position_multiplier = 1.0  # Multiplicateur de position basé sur la volatilité
        self.volatility_adjustment = 1.0  # Ajustement basé sur la volatilité

        # Threading pour sauvegarde asynchrone
        self.save_lock = threading.Lock()

        logger.info(
            "AdaptiveDBE initialisé avec adaptation non-linéaire des paramètres")

    def update(self, market_data: Dict[str, Any],
               performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Met à jour le DBE avec adaptation des paramètres.

        Args:
            market_data: Données de marché actuelles
            performance_metrics: Métriques de performance

        Returns:
            Modulation DBE mise à jour
        """
        # Mise à jour du détecteur de régime
        price = market_data.get('price', 50000)
        volume = market_data.get('volume', 1000000)
        volatility = market_data.get('volatility', 0.02)

        self.regime_detector.update(price, volume, volatility)
        current_regime = self.regime_detector.detect_regime()
        market_conditions = self.regime_detector.get_market_conditions()

        # Enregistrer l'historique
        self.performance_history.append(performance_metrics.copy())
        self.regime_history.append(current_regime)

        # Adaptation des paramètres si activée
        if self.adaptation_enabled and len(self.performance_history) > 1:
            old_params = self.params.to_dict()
            self.params = self.param_evolution.update_parameters(
                performance_metrics, market_conditions
            )
            new_params = self.params.to_dict()

            # Log des changements significatifs
            significant_changes = []
            for param_name in old_params.keys():
                if param_name in new_params:
                    change = abs(
                        new_params[param_name] - old_params[param_name])
                    if change > 0.001:  # Seuil de changement significatif
                        significant_changes.append({
                            'parameter': param_name,
                            'old_value': old_params[param_name],
                            'new_value': new_params[param_name],
                            'change': change
                        })

            if significant_changes:
                self.adaptation_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'regime': current_regime.value,
                    'performance': performance_metrics.copy(),
                    'changes': significant_changes
                })

                logger.info(f"DBE parameters adapted for regime {current_regime.value}: "
                            f"{len(significant_changes)} parameters changed")

        # Calcul de la modulation basée sur les paramètres actuels
        modulation = self._calculate_modulation(
            market_conditions, current_regime)

        # Sauvegarde périodique
        if len(self.adaptation_log) % 10 == 0:
            self._save_adaptation_state()

        return modulation

    def _calculate_modulation(self, market_conditions: Dict[str, float],
                              regime: MarketRegime) -> Dict[str, float]:
        """
        Calcule la modulation DBE avec des ajustements non linéaires basés sur la volatilité.

        Args:
            market_conditions: Dictionnaire contenant les conditions actuelles du marché
            regime: Régime de marché détecté

        Returns:
            Dictionnaire contenant les paramètres modulés
        """
        # Récupération de la volatilité EWMA et autres métriques
        ewma_vol = self.regime_detector.get_ewma_volatility()
        drawdown = market_conditions.get('current_drawdown', 0.0)
        trend_strength = market_conditions.get('trend_strength', 0.0)

        # 1. Calcul du multiplicateur de position basé sur la volatilité (fonction exponentielle décroissante)
        # Plus la volatilité est élevée, plus le multiplicateur diminue rapidement
        self.volatility_adjustment = np.exp(-ewma_vol / self.tau)

        # 2. Ajustement de la taille de position (non linéaire)
        base_position_size = 1.0  # Taille de base à 100%
        max_position_size = 2.0   # Taille maximale (200%)
        min_position_size = 0.2   # Taille minimale (20%)

        # Calcul de la taille de position avec une fonction sigmoïde pour un lissage progressif
        position_size = min_position_size + (max_position_size - min_position_size) / \
            (1 + np.exp(10 * (ewma_vol - 0.05)))  # Point d'inflexion à 5%

        # 3. Ajustement des niveaux de stop-loss et take-profit
        base_sl = 0.02  # 2% de stop-loss de base
        base_tp = 0.04  # 4% de take-profit de base

        # Ajustement non linéaire des stops (racine carrée pour réduire l'impact des fortes volatilités)
        sl_adjustment = np.sqrt(ewma_vol / 0.02)  # Normalisé par rapport à 2%
        tp_adjustment = np.sqrt(ewma_vol / 0.02)  # Même ajustement pour TP

        # Application des ajustements avec des limites
        stop_loss = min(0.10, base_sl * sl_adjustment)  # Maximum 10%
        take_profit = min(0.20, base_tp * tp_adjustment)  # Maximum 20%

        # 4. Ajustement supplémentaire basé sur le régime de marché
        regime_multiplier = 1.0
        if regime == MarketRegime.CRISIS:
            regime_multiplier = 0.3  # Réduction drastique en période de crise
        elif regime == MarketRegime.HIGH_VOLATILITY:
            regime_multiplier = 0.6  # Réduction en haute volatilité
        elif regime == MarketRegime.LOW_VOLATILITY:
            regime_multiplier = 1.2  # Augmentation en faible volatilité

        # Application du multiplicateur de régime
        position_size = min(max_position_size,
                            position_size * regime_multiplier)

        # 5. Calcul du niveau de risque global (0.0 à 2.0)
        risk_level = 1.0  # Niveau de risque de base

        # Ajustement du risque en fonction de la volatilité (fonction exponentielle inverse)
        # Réduction douce jusqu'à 5% de vol
        risk_level *= np.exp(-0.5 * (ewma_vol / 0.05) ** 2)

        # 6. Ajustement basé sur le drawdown
        if drawdown > self.params.max_drawdown_threshold * 1.5:
            risk_level *= 0.5  # Réduction forte du risque en cas de gros drawdown
        elif drawdown > self.params.max_drawdown_threshold:
            risk_level *= 0.8  # Réduction modérée du risque

        # 7. Ajustement basé sur la force de la tendance
        # Augmente le risque en tendance forte
        risk_level *= (1 + 0.5 * np.tanh(trend_strength))

        # 8. Application des limites finales
        risk_level = max(0.1, min(2.0, risk_level))  # Entre 10% et 200%
        position_size = max(min_position_size, min(
            max_position_size, position_size))

        # Journalisation des paramètres pour le débogage
        logger.debug(
            f"DBE Modulation - Vol: {ewma_vol:.4f}, "
            f"PosSize: {position_size:.2f}, "
            f"SL: {stop_loss:.2%}, TP: {take_profit:.2%}, "
            f"Risk: {risk_level:.2f}, Drawdown: {drawdown:.2%}"
        )

        # Construction du dictionnaire de retour avec tous les paramètres
        result = {
            'risk_level': float(risk_level),
            'position_size': float(position_size),
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'volatility': float(ewma_vol),
            'regime': regime.value,
            'drawdown': float(drawdown),
            'trend_strength': float(trend_strength),
            'parameters_snapshot': self.params.to_dict(),
            'adaptation_enabled': self.adaptation_enabled
        }

        return result

    def _save_adaptation_state(self) -> None:
        """Sauvegarde l'état d'adaptation"""
        with self.save_lock:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                state = {
                    'timestamp': timestamp,
                    'current_parameters': self.params.to_dict(),
                    'adaptation_stats': self.param_evolution.get_adaptation_stats(),
                    # 10 dernières adaptations
                    'recent_adaptations': self.adaptation_log[-10:],
                    'regime_distribution': self._get_regime_distribution(),
                    'performance_summary': self._get_performance_summary()
                }

                filepath = self.save_path / \
                    f"adaptive_dbe_state_{timestamp}.json"
                with open(filepath, 'w') as f:
                    json.dump(state, f, indent=2)

                logger.debug(f"Adaptive DBE state saved to {filepath}")

            except Exception as e:
                logger.error(f"Failed to save adaptive DBE state: {e}")

    def _get_regime_distribution(self) -> Dict[str, float]:
        """Calcule la distribution des régimes récents"""
        if not self.regime_history:
            return {}

        regime_counts = {}
        for regime in self.regime_history:
            regime_counts[regime.value] = regime_counts.get(
                regime.value, 0) + 1

        total = len(self.regime_history)
        return {regime: count / total for regime, count in regime_counts.items()}

    def _get_performance_summary(self) -> Dict[str, float]:
        """Calcule un résumé des performances récentes"""
        if not self.performance_history:
            return {}

        recent_perfs = list(self.performance_history)[-50:]  # 50 dernières

        sharpe_ratios = [p.get('sharpe_ratio', 0.0) for p in recent_perfs]
        returns = [p.get('total_return', 0.0) for p in recent_perfs]

        return {
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'avg_return': np.mean(returns),
            'sharpe_std': np.std(sharpe_ratios),
            'return_std': np.std(returns),
            'performance_trend': np.polyfit(range(len(sharpe_ratios)), sharpe_ratios, 1)[0] if len(sharpe_ratios) > 1 else 0.0
        }

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Retourne des statistiques complètes du système adaptatif"""
        return {
            'current_parameters': self.params.to_dict(),
            'adaptation_stats': self.param_evolution.get_adaptation_stats(),
            'regime_distribution': self._get_regime_distribution(),
            'performance_summary': self._get_performance_summary(),
            'adaptation_log_size': len(self.adaptation_log),
            'total_updates': len(self.performance_history),
            'adaptation_enabled': self.adaptation_enabled
        }

    def reset_adaptation(self) -> None:
        """Remet à zéro le système d'adaptation"""
        self.params = DBEParameters()  # Paramètres par défaut
        self.param_evolution = ParameterEvolution(self.params)
        self.performance_history.clear()
        self.regime_history.clear()
        self.adaptation_log.clear()

        logger.info("Adaptive DBE system reset to default parameters")
