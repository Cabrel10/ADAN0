#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Package de tests pour le trading bot ADAN.

Ce package contient tous les tests unitaires et d'intégration pour valider
le bon fonctionnement du système de trading Phase 1 et suivantes.

Structure:
- tests/unit/: Tests unitaires des composants individuels
- tests/integration/: Tests d'intégration du système complet
- tests/conftest.py: Configuration partagée et fixtures pytest
"""

__version__ = "1.0.0"
__author__ = "ADAN Trading Bot Team"

# Importations communes pour faciliter les tests
import logging
import pytest
import numpy as np
import pandas as pd

# Configuration du logging pour les tests
logging.basicConfig(
    level=logging.INFO,
    format='[TEST] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Constantes de test
TEST_ASSETS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT']
TEST_TIMEFRAMES = ['5m', '1h', '4h']
TEST_CAPITAL_AMOUNTS = [20.0, 50.0, 250.0, 1000.0, 5000.0]

# Utilitaires de test
def setup_test_logging():
    """Configure le logging pour les tests."""
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)  # Réduire le bruit pendant les tests
    return logger

def assert_position_valid(position_size: float, capital: float, min_order: float = 11.0):
    """
    Assertion standard pour valider une position de trading.

    Args:
        position_size: Taille de la position calculée
        capital: Capital disponible
        min_order: Valeur minimale d'ordre (défaut Binance: 11$)
    """
    assert position_size > 0, f"Position doit être positive: {position_size}"
    assert position_size >= min_order, f"Position ${position_size:.2f} < minimum ${min_order}"
    assert position_size <= capital, f"Position ${position_size:.2f} > capital ${capital:.2f}"

def assert_tier_consistency(tier_data: dict):
    """
    Assertion pour valider la cohérence d'un palier de capital.

    Args:
        tier_data: Données du palier à valider
    """
    required_keys = [
        'name', 'min_capital', 'max_position_size_pct',
        'risk_per_trade_pct', 'max_drawdown_pct', 'max_concurrent_positions'
    ]

    for key in required_keys:
        assert key in tier_data, f"Clé manquante dans palier: {key}"

    assert 0 < tier_data['max_position_size_pct'] <= 100, "max_position_size_pct invalide"
    assert 0 < tier_data['risk_per_trade_pct'] <= 10, "risk_per_trade_pct invalide"
    assert tier_data['max_concurrent_positions'] >= 1, "max_concurrent_positions invalide"

# Messages de test standardisés
TEST_MESSAGES = {
    'PHASE1_START': "🚀 Début des tests Phase 1",
    'PHASE1_SUCCESS': "🎉 Phase 1 entièrement validée",
    'COMPONENT_OK': "✅ Composant validé",
    'INTEGRATION_OK': "✅ Intégration validée",
    'ERROR': "❌ Erreur de test",
    'WARNING': "⚠️ Avertissement de test"
}

def log_test_result(test_name: str, status: str, details: str = ""):
    """
    Log standardisé pour les résultats de test.

    Args:
        test_name: Nom du test
        status: Statut (SUCCESS, FAILED, WARNING)
        details: Détails additionnels
    """
    icon = {
        'SUCCESS': '✅',
        'FAILED': '❌',
        'WARNING': '⚠️',
        'INFO': 'ℹ️'
    }.get(status, 'ℹ️')

    message = f"{icon} {test_name}"
    if details:
        message += f" - {details}"

    logger = logging.getLogger(__name__)
    if status == 'FAILED':
        logger.error(message)
    elif status == 'WARNING':
        logger.warning(message)
    else:
        logger.info(message)
