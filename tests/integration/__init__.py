#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests d'intégration pour le trading bot ADAN.

Ce package contient les tests d'intégration pour valider le fonctionnement
du système complet et les interactions entre composants :

- test_phase1_complete_system.py: Tests d'intégration Phase 1 complets
- Futures tests pour workflows complets, performance, stress testing

Les tests d'intégration se concentrent sur :
- Flux de données complet (data loading → processing → trading)
- Intégration entre portfolio manager, data loader, et environment
- Validation des configurations complexes
- Performance du système sous charge
- Cohérence des résultats à travers les composants
- Scenarios de trading réalistes
"""

__version__ = "1.0.0"

# Importations des modules de test d'intégration
try:
    from .test_phase1_complete_system import (
        TestPhase1CompleteIntegration,
        TestDataIntegration
    )
except ImportError:
    # Gérer gracieusement si les modules ne sont pas encore disponibles
    pass

# Importations pour tests d'intégration
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Configuration des tests d'intégration
INTEGRATION_TEST_CONFIG = {
    # Scénarios de capital pour tests complets
    "capital_scenarios": [
        {"amount": 20.0, "tier": "Micro Capital", "description": "Débutant avec capital minimal"},
        {"amount": 75.0, "tier": "Small Capital", "description": "Petit investisseur"},
        {"amount": 300.0, "tier": "Medium Capital", "description": "Investisseur intermédiaire"},
        {"amount": 1200.0, "tier": "Large Capital", "description": "Investisseur avancé"},
        {"amount": 5000.0, "tier": "Enterprise", "description": "Investisseur institutionnel"}
    ],

    # Assets pour tests d'intégration complets
    "comprehensive_assets": [
        "BTCUSDT",  # Crypto majeur stable
        "ETHUSDT",  # Crypto majeur avec plus de volatilité
        "SOLUSDT",  # Altcoin populaire volatile
        "ADAUSDT",  # Altcoin avec différents patterns
        "XRPUSDT"   # Altcoin avec régulations
    ],

    # Timeframes pour tests multi-échelle
    "timeframe_hierarchy": {
        "scalping": "1m",
        "short_term": "5m",
        "medium_term": "1h",
        "long_term": "4h",
        "position": "1d"
    },

    # Scénarios de marché pour stress testing
    "market_scenarios": {
        "bull_market": {"trend": 1, "volatility": 0.02, "description": "Marché haussier stable"},
        "bear_market": {"trend": -1, "volatility": 0.025, "description": "Marché baissier"},
        "sideways": {"trend": 0, "volatility": 0.015, "description": "Marché latéral"},
        "high_volatility": {"trend": 0, "volatility": 0.08, "description": "Haute volatilité"},
        "crash": {"trend": -3, "volatility": 0.15, "description": "Crash market"}
    }
}

# Métriques de performance pour validation
PERFORMANCE_BENCHMARKS = {
    "cvar_calculation_max_time_ms": 100,  # Temps max pour calcul CVaR
    "tier_detection_max_time_ms": 10,     # Temps max pour détection palier
    "normalization_max_time_ms": 5,       # Temps max pour normalisation
    "complete_flow_max_time_ms": 1000,    # Temps max pour flux complet
    "memory_usage_max_mb": 100,           # Usage mémoire maximum acceptable
    "cpu_usage_max_percent": 80           # Usage CPU maximum acceptable
}

# Seuils de validation pour les tests d'intégration
INTEGRATION_VALIDATION_THRESHOLDS = {
    "position_size_variance_max": 0.20,   # Variance max 20% entre calculs identiques
    "tier_transition_smooth_factor": 1.5, # Facteur de lissage entre paliers
    "multi_timeframe_coherence_min": 0.7, # Cohérence min entre timeframes
    "asset_selection_stability": 0.8,     # Stabilité sélection d'actifs
    "risk_management_compliance": 1.0,    # Compliance 100% aux règles de risque
    "sharpe_momentum_correlation": 0.6    # Corrélation min avec performance réelle
}

def create_integration_test_environment(capital: float, assets: List[str],
                                      market_scenario: str = "bull_market") -> Dict[str, Any]:
    """
    Crée un environnement de test d'intégration complet.

    Args:
        capital: Capital initial pour le test
        assets: Liste des actifs à inclure
        market_scenario: Scénario de marché à simuler

    Returns:
        Environnement de test configuré
    """
    scenario = INTEGRATION_TEST_CONFIG["market_scenarios"][market_scenario]

    # Configuration de l'environnement
    env_config = {
        "capital": capital,
        "assets": assets,
        "market_scenario": scenario,
        "test_duration_steps": 1000,
        "data_points_per_asset": 5000,
        "performance_tracking": True,
        "detailed_logging": True
    }

    # Configuration des paliers (copie de la config principale)
    env_config["capital_tiers"] = INTEGRATION_TEST_CONFIG.get("capital_tiers", [])

    return env_config

def simulate_realistic_market_data(assets: List[str], n_periods: int = 5000,
                                 scenario: str = "bull_market") -> Dict[str, pd.DataFrame]:
    """
    Simule des données de marché réalistes pour les tests d'intégration.

    Args:
        assets: Liste des actifs à simuler
        n_periods: Nombre de périodes à générer
        scenario: Scénario de marché

    Returns:
        Dictionnaire avec DataFrames de données de marché par actif
    """
    market_config = INTEGRATION_TEST_CONFIG["market_scenarios"][scenario]
    trend = market_config["trend"]
    volatility = market_config["volatility"]

    market_data = {}

    # Prix de base par actif (réalistes)
    base_prices = {
        "BTCUSDT": 45000,
        "ETHUSDT": 3000,
        "SOLUSDT": 100,
        "ADAUSDT": 0.5,
        "XRPUSDT": 0.6
    }

    for asset in assets:
        np.random.seed(hash(asset) % 2**32)  # Seed reproductible par actif

        base_price = base_prices.get(asset, 100)

        # Générer des rendements avec trend et volatilité
        daily_trend = trend * 0.001  # 0.1% par période si trend = 1
        returns = np.random.normal(daily_trend, volatility, n_periods)

        # Ajouter de la persistance (momentum effect)
        for i in range(1, len(returns)):
            returns[i] += 0.15 * returns[i-1]  # Autocorrélation

        # Ajouter des événements extrêmes
        extreme_events = np.random.random(n_periods) < 0.02  # 2% chance
        returns[extreme_events] *= 3

        # Calculer les prix
        prices = base_price * np.exp(np.cumsum(returns))

        # Créer OHLCV data
        high_low_spread = volatility * 0.5
        volume_base = np.random.lognormal(15, 1, n_periods)

        timestamps = pd.date_range(start='2023-01-01', periods=n_periods, freq='1H')

        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices * np.random.uniform(0.999, 1.001, n_periods),
            'high': prices * (1 + np.random.uniform(0, high_low_spread, n_periods)),
            'low': prices * (1 - np.random.uniform(0, high_low_spread, n_periods)),
            'close': prices,
            'volume': volume_base,
            'returns': returns
        })

        # Calculer indicateurs techniques
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(24*365)
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        market_data[asset] = df.dropna()

    return market_data

def measure_integration_performance(func, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
    """
    Mesure les performances d'une fonction d'intégration.

    Args:
        func: Fonction à mesurer
        *args, **kwargs: Arguments de la fonction

    Returns:
        Tuple (résultat, métriques de performance)
    """
    import psutil
    import tracemalloc

    # Démarrer le monitoring
    tracemalloc.start()
    process = psutil.Process()
    cpu_before = process.cpu_percent()

    start_time = time.perf_counter()

    # Exécuter la fonction
    result = func(*args, **kwargs)

    # Mesurer les performances
    end_time = time.perf_counter()
    execution_time_ms = (end_time - start_time) * 1000

    # Mémoire
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_usage_mb = peak / 1024 / 1024

    # CPU (approximatif)
    cpu_after = process.cpu_percent()
    cpu_usage = max(cpu_after - cpu_before, 0)

    metrics = {
        "execution_time_ms": execution_time_ms,
        "memory_usage_mb": memory_usage_mb,
        "cpu_usage_percent": cpu_usage
    }

    return result, metrics

def validate_integration_results(results: Dict[str, Any],
                               expected_metrics: Dict[str, Any]) -> Dict[str, bool]:
    """
    Valide les résultats d'un test d'intégration.

    Args:
        results: Résultats du test d'intégration
        expected_metrics: Métriques attendues

    Returns:
        Dictionnaire de validation (True/False par métrique)
    """
    validation = {}

    # Validation des performances
    if 'performance' in results:
        perf = results['performance']
        for metric, threshold in PERFORMANCE_BENCHMARKS.items():
            if metric.replace('_max_', '_') in perf:
                actual = perf[metric.replace('_max_', '_')]
                validation[f"performance_{metric}"] = actual <= threshold

    # Validation des seuils d'intégration
    for threshold_name, threshold_value in INTEGRATION_VALIDATION_THRESHOLDS.items():
        if threshold_name in results:
            actual = results[threshold_name]
            if "max" in threshold_name:
                validation[threshold_name] = actual <= threshold_value
            elif "min" in threshold_name:
                validation[threshold_name] = actual >= threshold_value
            else:
                # Pour les valeurs exactes (comme compliance)
                validation[threshold_name] = abs(actual - threshold_value) < 0.01

    return validation

def log_integration_test_summary(test_name: str, results: Dict[str, Any],
                                validation: Dict[str, bool]) -> None:
    """
    Log un résumé détaillé des résultats d'intégration.

    Args:
        test_name: Nom du test d'intégration
        results: Résultats détaillés
        validation: Résultats de validation
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"RÉSUMÉ TEST D'INTÉGRATION: {test_name}")
    logger.info(f"{'='*60}")

    # Statut général
    all_passed = all(validation.values()) if validation else False
    status_icon = "🎉" if all_passed else "⚠️"
    logger.info(f"{status_icon} STATUT GÉNÉRAL: {'SUCCÈS' if all_passed else 'PARTIELLEMENT RÉUSSI'}")

    # Détails des validations
    if validation:
        logger.info(f"\n📊 VALIDATIONS ({sum(validation.values())}/{len(validation)} réussies):")
        for metric, passed in validation.items():
            icon = "✅" if passed else "❌"
            logger.info(f"  {icon} {metric}")

    # Métriques de performance si disponibles
    if 'performance' in results:
        perf = results['performance']
        logger.info(f"\n⚡ MÉTRIQUES DE PERFORMANCE:")
        for metric, value in perf.items():
            logger.info(f"  • {metric}: {value}")

    # Résultats clés
    if 'summary' in results:
        summary = results['summary']
        logger.info(f"\n🔍 RÉSULTATS CLÉS:")
        for key, value in summary.items():
            logger.info(f"  • {key}: {value}")

    logger.info(f"{'='*60}\n")

# Classes d'exception spécialisées pour les tests d'intégration
class IntegrationTestError(Exception):
    """Exception de base pour les tests d'intégration."""
    pass

class PerformanceBenchmarkError(IntegrationTestError):
    """Exception levée quand les benchmarks de performance ne sont pas atteints."""
    pass

class ValidationThresholdError(IntegrationTestError):
    """Exception levée quand les seuils de validation ne sont pas respectés."""
    pass

# Messages standardisés pour les tests d'intégration
INTEGRATION_TEST_MESSAGES = {
    'WORKFLOW_START': "🚀 Démarrage test workflow complet",
    'COMPONENT_INTEGRATION': "🔗 Test intégration composants",
    'PERFORMANCE_CHECK': "⚡ Vérification performance",
    'DATA_FLOW_VALIDATION': "📊 Validation flux de données",
    'CONFIGURATION_TEST': "⚙️ Test configuration système",
    'STRESS_TEST': "💪 Test de stress",
    'SCENARIO_SIMULATION': "🎭 Simulation scénario",
    'INTEGRATION_SUCCESS': "🎉 Intégration réussie",
    'INTEGRATION_WARNING': "⚠️ Intégration partiellement réussie",
    'INTEGRATION_FAILED': "❌ Échec intégration"
}
