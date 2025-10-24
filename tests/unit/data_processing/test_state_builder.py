"""Tests unitaires pour le StateBuilder."""

import numpy as np
import pandas as pd
from unittest.mock import patch
from src.adan_trading_bot.data_processing.state_builder import (
    StateBuilder, TimeframeConfig, _calculate_technical_indicators
)


def test_timeframe_config_initialization():
    """Test l'initialisation de TimeframeConfig."""
    config = TimeframeConfig(
        timeframe="5m",
        features=["open", "close", "volume"],
        window_size=100,
        normalize=True
    )

    if not (config.timeframe == "5m" and
            config.features == ["open", "close", "volume"] and
            config.window_size == 100 and
            config.normalize is True):
        raise AssertionError("La configuration du timeframe est incorrecte")


def test_state_builder_initialization():
    """Test l'initialisation du StateBuilder."""
    features_config = {
        "5m": ["open", "close", "volume"],
        "1h": ["open", "high", "low", "close"]
    }

    # Créer l'instance avec les timeframes spécifiés
    builder = StateBuilder(
        features_config=features_config,
        timeframes=["5m", "1h"],
        window_size=100
    )

    # Vérifier que les timeframes sont correctement configurés
    if not (hasattr(builder, 'timeframes') and
            "5m" in builder.timeframes and
            "1h" in builder.timeframes):
        raise AssertionError("Les timeframes n'ont pas été configurés")

    # Vérifier que les fonctionnalités sont correctement configurées
    if not (hasattr(builder, 'features_config') and
            "5m" in builder.features_config and
            "1h" in builder.features_config):
        raise AssertionError("Les fonctionnalités n'ont pas été configurées")


def test_calculate_technical_indicators():
    """Test le calcul des indicateurs techniques."""
    np.random.seed(42)
    df = pd.DataFrame({
        'OPEN': np.random.rand(100) * 100 + 100,
        'HIGH': np.random.rand(100) * 10 + 105,
        'LOW': np.random.rand(100) * 10 + 95,
        'CLOSE': np.random.rand(100) * 5 + 100,
        'VOLUME': np.random.randint(100, 1000, 100)
    })

    # Tester le calcul des indicateurs
    result = _calculate_technical_indicators(df)

    # Vérifier que les colonnes ont été ajoutées
    required_indicators = {
        'RSI_14', 'MACD_HIST_12_26_9', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER',
        'ATR_14', 'EMA_12', 'EMA_26', 'SMA_20', 'ADX_14'
    }

    # Vérifier que tous les indicateurs requis sont présents
    missing_indicators = required_indicators - set(result.columns)
    if missing_indicators:
        raise AssertionError(f"Indicateurs manquants: {missing_indicators}")

    # Vérifier qu'il n'y a pas de valeurs NaN dans les indicateurs
    indicator_cols = list(required_indicators)
    if result[indicator_cols].isnull().any().any():
        raise AssertionError("Valeurs NaN détectées dans les indicateurs")


@patch('src.adan_trading_bot.data_processing.state_builder._calculate_technical_indicators')
def test_build_state(mock_calc_indicators):
    """Test la construction de l'état."""
    mock_calc_indicators.return_value = pd.DataFrame({
        'OPEN': [100, 101, 102],
        'HIGH': [101, 102, 103],
        'LOW': [99, 100, 101],
        'CLOSE': [101, 102, 103],
        'VOLUME': [1000, 2000, 3000],
        'RSI_14': [50, 55, 60],
        'MACD_HIST_12_26_9': [0.1, 0.2, 0.3],
        'BB_UPPER': [102, 103, 104],
        'BB_MIDDLE': [101, 102, 103],
        'BB_LOWER': [100, 101, 102],
        'ATR_14': [1.5, 1.6, 1.7],
        'EMA_12': [100.5, 101.5, 102.5],
        'EMA_26': [100.2, 101.2, 102.2],
        'SMA_20': [100.8, 101.8, 102.8],
        'ADX_14': [25, 26, 27]
    })

    builder = StateBuilder(
        features_config={
            "5m": [
                "open", "high", "low", "close", "volume",
                "rsi_14", "macd_hist_12_26_9", "bb_upper",
                "bb_middle", "bb_lower", "atr_14", "ema_12",
                "ema_26", "sma_20", "adx_14"
            ]
        },
        timeframes=["5m"],
        window_size=3
    )

    market_data = {
        "asset1": {
            "5m": pd.DataFrame({
                'TIMESTAMP': pd.date_range('2023-01-01', periods=3, freq='5min'),
                'OPEN': [100, 101, 102],
                'HIGH': [101, 102, 103],
                'LOW': [99, 100, 101],
                'CLOSE': [101, 102, 103],
                'VOLUME': [1000, 2000, 3000]
            }).set_index('TIMESTAMP')
        }
    }

    try:
        builder.expected_features = {
            "5m": [
                "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME",
                "RSI_14", "MACD_HIST_12_26_9", "BB_UPPER",
                "BB_MIDDLE", "BB_LOWER", "ATR_14", "EMA_12",
                "EMA_26", "SMA_20", "ADX_14"
            ]
        }

        tf, processed = builder._process_timeframe_data(
            market_data, "5m", current_idx=2, max_features=15, verbose_log=False
        )

        if processed is None:
            raise AssertionError("Le traitement du timeframe a échoué")

        if not isinstance(processed, np.ndarray):
            raise AssertionError("Le résultat devrait être un tableau numpy")

        # Vérifier la forme du résultat
        exp_shape = (builder.window_size, len(builder.expected_features["5m"]))
        if processed.shape != exp_shape:
            msg = f"Forme incorrecte: attendu {exp_shape}, obtenu {processed.shape}"
            raise AssertionError(msg)

    except Exception as e:
        raise AssertionError(f"Erreur lors du traitement: {str(e)}")

    # Vérifier que les indicateurs ont été calculés
    if not mock_calc_indicators.called:
        raise AssertionError("Les indicateurs n'ont pas été calculés")

def test_set_timeframe_config():
    """Test la configuration des timeframes."""
    builder = StateBuilder(
        features_config={"5m": ["open", "close"]},
        timeframes=["5m"],
        window_size=100
    )

    if not hasattr(builder, 'timeframes') or "5m" not in builder.timeframes:
        raise AssertionError("Le timeframe 5m est manquant")

    if not hasattr(builder, 'features_config') or "5m" not in builder.features_config:
        raise AssertionError("Configuration des fonctionnalités manquante")

    if hasattr(builder, 'features_config'):
        builder.features_config["1h"] = ["open", "high", "low", "close"]
        builder.timeframes.append("1h")

        if "1h" not in builder.timeframes:
            raise AssertionError("Échec de l'ajout du timeframe 1h")
        if "1h" not in builder.features_config:
            raise AssertionError("Fonctionnalités manquantes pour 1h")
