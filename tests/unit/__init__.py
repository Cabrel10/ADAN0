#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests unitaires pour le trading bot ADAN.

Ce package contient les tests unitaires pour valider individuellement
chaque composant du système de trading :

- test_portfolio_tier_logic.py: Tests de la logique des paliers et CVaR
- Futures tests pour data_loader, state_builder, environment, etc.

Les tests unitaires se concentrent sur :
- Fonctions individuelles et méthodes de classe
- Logique métier isolée (CVaR, normalisation, paliers)
- Gestion d'erreur et cas limites
- Performance des algorithmes
"""

__version__ = "1.0.0"

# Importations des modules de test
try:
    from .test_portfolio_tier_logic import (
        TestCapitalTiers,
        TestCVaRPositionSizing,
        TestTierConstraintsNormalization,
        TestCVaRIntegrationWithTiers,
        TestErrorHandling
    )
except ImportError:
    # Gérer gracieusement si les modules ne sont pas encore disponibles
    pass

# Utilitaires spécifiques aux tests unitaires
import logging
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Données de test standardisées
UNIT_TEST_CONFIG = {
    "test_capitals": [15.0, 25.0, 75.0, 300.0, 1500.0, 8000.0],  # Couvre tous les paliers
    "test_assets": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "test_risk_levels": [0.01, 0.02, 0.03, 0.05],  # 1% à 5%
    "test_confidence_levels": [0.01, 0.05, 0.10],  # 1%, 5%, 10%
    "test_timeframes": ["5m", "1h", "4h"]
}

# Constantes pour validation
VALIDATION_CONSTANTS = {
    "min_binance_order": 11.0,
    "max_position_pct_absolute": 100.0,
    "min_risk_pct": 0.001,  # 0.1%
    "max_risk_pct": 10.0,   # 10%
    "cvar_confidence_range": (0.01, 0.10),  # 1% à 10%
    "normalization_tolerance": 0.001  # Tolérance d'erreur pour normalisation
}

def create_mock_returns(n_samples: int = 1000, volatility: float = 0.02,
                       mean_return: float = 0.0001, seed: int = 42) -> np.ndarray:
    """
    Génère des rendements simulés pour les tests CVaR.

    Args:
        n_samples: Nombre d'échantillons
        volatility: Volatilité quotidienne (défaut: 2%)
        mean_return: Rendement moyen quotidien (défaut: 0.01%)
        seed: Seed pour reproductibilité

    Returns:
        Array numpy de rendements simulés
    """
    np.random.seed(seed)
    returns = np.random.normal(mean_return, volatility, n_samples)

    # Ajouter quelques événements extrêmes (5% de chances)
    extreme_events = np.random.random(n_samples) < 0.05
    returns[extreme_events] *= 3  # Amplifier les événements extrêmes

    return returns

def create_tier_test_scenario(capital: float) -> Dict[str, Any]:
    """
    Crée un scénario de test pour un montant de capital donné.

    Args:
        capital: Montant de capital à tester

    Returns:
        Dictionnaire avec le scénario de test complet
    """
    # Déterminer le palier attendu
    if capital <= 30:
        tier_name = "Micro Capital"
        max_pos_pct = 90
        risk_pct = 5.0
        max_concurrent = 1
        exposure_range = [75, 90]
    elif capital <= 100:
        tier_name = "Small Capital"
        max_pos_pct = 75
        risk_pct = 4.0
        max_concurrent = 2
        exposure_range = [65, 80]
    elif capital <= 500:
        tier_name = "Medium Capital"
        max_pos_pct = 60
        risk_pct = 3.0
        max_concurrent = 3
        exposure_range = [55, 70]
    elif capital <= 2000:
        tier_name = "Large Capital"
        max_pos_pct = 45
        risk_pct = 2.5
        max_concurrent = 4
        exposure_range = [45, 60]
    else:
        tier_name = "Enterprise"
        max_pos_pct = 30
        risk_pct = 2.0
        max_concurrent = 5
        exposure_range = [35, 50]

    return {
        "capital": capital,
        "expected_tier": tier_name,
        "max_position_pct": max_pos_pct,
        "risk_per_trade_pct": risk_pct,
        "max_concurrent_positions": max_concurrent,
        "exposure_range": exposure_range,
        "max_position_value": capital * (max_pos_pct / 100)
    }

def assert_cvar_mathematical_properties(cvar_value: float, returns: np.ndarray,
                                      confidence_level: float = 0.05) -> None:
    """
    Valide les propriétés mathématiques du CVaR calculé.

    Args:
        cvar_value: Valeur CVaR calculée
        returns: Rendements utilisés pour le calcul
        confidence_level: Niveau de confiance utilisé
    """
    # CVaR doit être négatif (perte) ou nul
    assert cvar_value <= 0, f"CVaR doit être ≤ 0, obtenu: {cvar_value}"

    # CVaR doit être plus extrême que le VaR
    var = np.percentile(returns, confidence_level * 100)
    assert cvar_value <= var, f"CVaR ({cvar_value}) doit être ≤ VaR ({var})"

    # CVaR ne doit pas être excessivement extrême (sanity check)
    min_return = returns.min()
    assert cvar_value >= min_return, f"CVaR ({cvar_value}) ne peut pas être < min_return ({min_return})"

def assert_normalization_properties(original_value: float, normalized_value: float,
                                  min_bound: float, max_bound: float,
                                  method: str = 'linear') -> None:
    """
    Valide les propriétés de la normalisation.

    Args:
        original_value: Valeur originale
        normalized_value: Valeur normalisée
        min_bound: Borne minimale
        max_bound: Borne maximale
        method: Méthode de normalisation utilisée
    """
    # La valeur normalisée doit être dans les bornes
    assert min_bound <= normalized_value <= max_bound, \
        f"Valeur normalisée {normalized_value} hors bornes [{min_bound}, {max_bound}]"

    # Si la valeur originale était dans les bornes, elle ne doit pas être trop altérée
    if min_bound <= original_value <= max_bound:
        if method == 'linear':
            # Normalisation linéaire ne doit pas changer les valeurs dans les bornes
            assert abs(normalized_value - original_value) < VALIDATION_CONSTANTS['normalization_tolerance'], \
                f"Normalisation linéaire a altéré valeur dans bornes: {original_value} -> {normalized_value}"
        elif method == 'sigmoid':
            # Normalisation sigmoïde peut légèrement altérer, mais pas trop
            relative_change = abs(normalized_value - original_value) / abs(original_value) if original_value != 0 else 0
            assert relative_change < 0.05, \
                f"Normalisation sigmoïde a trop altéré valeur dans bornes: {relative_change:.1%}"

# Fixtures spécifiques aux tests unitaires
def get_unit_test_portfolio_configs() -> List[Dict[str, Any]]:
    """
    Retourne les configurations de portfolio pour les tests unitaires.

    Returns:
        Liste des configurations de test
    """
    configs = []

    for capital in UNIT_TEST_CONFIG["test_capitals"]:
        scenario = create_tier_test_scenario(capital)

        config = {
            "portfolio": {"initial_balance": capital},
            "assets": UNIT_TEST_CONFIG["test_assets"],
            "capital_tiers": [
                {
                    "name": "Micro Capital",
                    "min_capital": 11.0,
                    "max_capital": 30.0,
                    "max_position_size_pct": 90,
                    "risk_per_trade_pct": 5.0,
                    "max_drawdown_pct": 50.0,
                    "leverage": 1,
                    "max_concurrent_positions": 1,
                    "exposure_range": [75, 90]
                },
                {
                    "name": "Small Capital",
                    "min_capital": 31.0,
                    "max_capital": 100.0,
                    "max_position_size_pct": 75,
                    "risk_per_trade_pct": 4.0,
                    "max_drawdown_pct": 40.0,
                    "leverage": 1,
                    "max_concurrent_positions": 2,
                    "exposure_range": [65, 80]
                },
                {
                    "name": "Medium Capital",
                    "min_capital": 101.0,
                    "max_capital": 500.0,
                    "max_position_size_pct": 60,
                    "risk_per_trade_pct": 3.0,
                    "max_drawdown_pct": 30.0,
                    "leverage": 1,
                    "max_concurrent_positions": 3,
                    "exposure_range": [55, 70]
                },
                {
                    "name": "Large Capital",
                    "min_capital": 501.0,
                    "max_capital": 2000.0,
                    "max_position_size_pct": 45,
                    "risk_per_trade_pct": 2.5,
                    "max_drawdown_pct": 25.0,
                    "leverage": 1,
                    "max_concurrent_positions": 4,
                    "exposure_range": [45, 60]
                },
                {
                    "name": "Enterprise",
                    "min_capital": 2001.0,
                    "max_capital": None,
                    "max_position_size_pct": 30,
                    "risk_per_trade_pct": 2.0,
                    "max_drawdown_pct": 20.0,
                    "leverage": 1,
                    "max_concurrent_positions": 5,
                    "exposure_range": [35, 50]
                }
            ],
            "expected_scenario": scenario
        }

        configs.append(config)

    return configs

# Messages de statut pour tests unitaires
UNIT_TEST_STATUS = {
    'TIER_DETECTION': "Détection palier",
    'CVAR_CALCULATION': "Calcul CVaR",
    'NORMALIZATION': "Normalisation",
    'CONSTRAINT_APPLICATION': "Application contraintes",
    'ERROR_HANDLING': "Gestion erreurs",
    'MATHEMATICAL_PROPERTIES': "Propriétés mathématiques"
}

def log_unit_test_progress(test_category: str, test_name: str, status: str, details: str = ""):
    """
    Log spécialisé pour le suivi des tests unitaires.

    Args:
        test_category: Catégorie de test (ex: 'TIER_DETECTION')
        test_name: Nom spécifique du test
        status: Statut (SUCCESS, FAILED, etc.)
        details: Détails additionnels
    """
    category_desc = UNIT_TEST_STATUS.get(test_category, test_category)
    message = f"[{category_desc}] {test_name}"

    if details:
        message += f" - {details}"

    # Log avec niveau approprié
    if status == 'SUCCESS':
        logger.info(f"✅ {message}")
    elif status == 'FAILED':
        logger.error(f"❌ {message}")
    elif status == 'WARNING':
        logger.warning(f"⚠️ {message}")
    else:
        logger.info(f"ℹ️ {message}")
