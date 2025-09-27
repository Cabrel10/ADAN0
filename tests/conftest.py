#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration de test partagée pour les tests de la Phase 1.

Ce fichier contient les fixtures pytest partagées pour tester :
- CVaR Position Sizing
- Sharpe Momentum Ratio
- Logique des paliers (capital_tiers)
- Configuration des workers
- Système multi-timeframe
"""

import pytest
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Any

# Configuration du logging pour les tests
logging.basicConfig(level=logging.INFO)

@pytest.fixture(scope="session")
def config_data():
    """Configuration complète du bot pour les tests."""
    return {
        "default_currency": "USDT",
        "trading_fees": 0.001,
        "min_order_value_usdt": 11.0,

        # Assets de test
        "assets": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT"],

        # Configuration des paliers de capital
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

        # Configuration des workers spécialisés
        "workers": [
            {
                "id": 0,
                "name": "Worker-0 (Pilier Stable)",
                "assets": ["BTCUSDT", "ETHUSDT"],
                "data_split": "train"
            },
            {
                "id": 1,
                "name": "Worker-1 (Explorateur Alts)",
                "assets": ["SOLUSDT", "ADAUSDT", "XRPUSDT"],
                "data_split": "train"
            },
            {
                "id": 2,
                "name": "Worker-2 (Validation Croisée)",
                "assets": ["BTCUSDT", "SOLUSDT"],
                "data_split": "val"
            },
            {
                "id": 3,
                "name": "Worker-3 (Stratège Global)",
                "assets": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"],
                "data_split": "test"
            }
        ],

        # Timeframes multi-échelles
        "timeframes": {
            "fast": "5m",
            "medium": "1h",
            "slow": "4h"
        },

        # Paths
        "paths": {
            "data_dir": "data",
            "metrics_dir": "logs/metrics",
            "models_dir": "models"
        },

        # Logging
        "logging": {
            "metrics_interval": 100
        },

        # Portfolio
        "portfolio": {
            "initial_balance": 20.0  # Teste palier Micro Capital
        },

        # Environment
        "environment": {
            "chunk_size": 250,
            "warmup_steps": 50
        }
    }

@pytest.fixture
def mock_data_loader():
    """Mock du DataLoader avec données simulées réalistes."""
    mock_loader = Mock()

    def generate_market_data(asset: str, timeframe: str, split: str) -> pd.DataFrame:
        """Génère des données de marché réalistes pour les tests."""
        np.random.seed(42)  # Reproductibilité

        # Prix de base selon l'actif
        base_prices = {
            "BTCUSDT": 45000,
            "ETHUSDT": 3000,
            "SOLUSDT": 100,
            "ADAUSDT": 0.5,
            "XRPUSDT": 0.6
        }

        base_price = base_prices.get(asset, 100)
        n_periods = 1000

        # Génération de prix avec tendance et volatilité
        returns = np.random.normal(0.0001, 0.02, n_periods)  # 0.01% moyenne, 2% volatilité

        # Ajouter de la persistance (momentum)
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]  # Facteur d'autocorrélation

        prices = base_price * np.exp(np.cumsum(returns))

        # Créer le DataFrame avec OHLCV
        high_low_spread = 0.005  # 0.5% spread
        volume_base = np.random.lognormal(15, 1, n_periods)

        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=n_periods, freq='1H'),
            'open': prices * np.random.uniform(0.999, 1.001, n_periods),
            'high': prices * (1 + np.random.uniform(0, high_low_spread, n_periods)),
            'low': prices * (1 - np.random.uniform(0, high_low_spread, n_periods)),
            'close': prices,
            'volume': volume_base
        })

        # Calculer indicateurs techniques
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(24*365)  # Volatilité annualisée

        # RSI simulé
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD simulé
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        return df.dropna()

    mock_loader.load_data.side_effect = generate_market_data
    mock_loader.get_available_assets.return_value = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT"]

    return mock_loader

@pytest.fixture
def portfolio_manager_micro(config_data, mock_data_loader):
    """PortfolioManager configuré pour palier Micro Capital (20$ de capital)."""
    from bot.src.adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

    # Configuration spécifique au test
    config_data["portfolio"]["initial_balance"] = 20.0  # Palier Micro Capital

    pm = PortfolioManager(config_data, assets=config_data["assets"])
    pm.data_loader = mock_data_loader
    pm.initial_capital = 20.0
    pm.cash = 20.0

    return pm

@pytest.fixture
def portfolio_manager_medium(config_data, mock_data_loader):
    """PortfolioManager configuré pour palier Medium Capital (250$ de capital)."""
    from bot.src.adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

    # Configuration spécifique au test
    config_data["portfolio"]["initial_balance"] = 250.0  # Palier Medium Capital

    pm = PortfolioManager(config_data, assets=config_data["assets"])
    pm.data_loader = mock_data_loader
    pm.initial_capital = 250.0
    pm.cash = 250.0

    return pm

@pytest.fixture
def sample_market_data():
    """Données de marché samples pour tests spécialisés."""
    np.random.seed(42)
    n_periods = 500

    # Prix BTC simulé avec tendance haussière
    btc_returns = np.random.normal(0.0002, 0.025, n_periods)  # Tendance légèrement haussière
    btc_prices = 45000 * np.exp(np.cumsum(btc_returns))

    # Prix ETH simulé avec plus de volatilité
    eth_returns = np.random.normal(0.0001, 0.035, n_periods)  # Plus volatil
    eth_prices = 3000 * np.exp(np.cumsum(eth_returns))

    # Prix SOL simulé avec forte volatilité (altcoin)
    sol_returns = np.random.normal(0.0003, 0.055, n_periods)  # Très volatil, tendance haussière
    sol_prices = 100 * np.exp(np.cumsum(sol_returns))

    timestamps = pd.date_range(start='2023-01-01', periods=n_periods, freq='1H')

    return {
        'BTCUSDT': pd.DataFrame({
            'timestamp': timestamps,
            'close': btc_prices,
            'returns': btc_returns,
            'volatility': pd.Series(btc_returns).rolling(20).std() * np.sqrt(24*365)
        }),
        'ETHUSDT': pd.DataFrame({
            'timestamp': timestamps,
            'close': eth_prices,
            'returns': eth_returns,
            'volatility': pd.Series(eth_returns).rolling(20).std() * np.sqrt(24*365)
        }),
        'SOLUSDT': pd.DataFrame({
            'timestamp': timestamps,
            'close': sol_prices,
            'returns': sol_returns,
            'volatility': pd.Series(sol_returns).rolling(20).std() * np.sqrt(24*365)
        })
    }

@pytest.fixture
def mock_environment():
    """Mock de l'environnement de trading pour tests d'intégration."""
    from unittest.mock import Mock

    env = Mock()
    env.observation_space = Mock()
    env.observation_space.shape = (100,)  # State vector size
    env.action_space = Mock()
    env.action_space.n = 3  # Hold, Buy, Sell

    # État initial
    env.current_step = 0
    env.chunk_id = 0
    env.asset = "BTCUSDT"

    # Méthodes mock
    env.reset.return_value = np.random.random(100), {}
    env.step.return_value = (np.random.random(100), 0.0, False, False, {})

    return env

@pytest.fixture
def sample_cvar_scenarios():
    """Scénarios de test pour la validation CVaR."""
    return {
        # Scénario 1: Marché normal (faible volatilité)
        "normal_market": {
            "returns": np.random.normal(0.0001, 0.015, 1000),  # 1.5% volatilité quotidienne
            "expected_position_range": (50, 90),  # % du capital (ajusté pour palier Micro)
            "description": "Marché stable, position sizing modéré"
        },

        # Scénario 2: Marché volatil (forte volatilité)
        "volatile_market": {
            "returns": np.random.normal(0.0002, 0.045, 1000),  # 4.5% volatilité
            "expected_position_range": (20, 60),  # % du capital (plus conservateur)
            "description": "Marché volatil, position sizing réduit"
        },

        # Scénario 3: Marché avec queues épaisses (black swans)
        "tail_risk_market": {
            "returns": np.concatenate([
                np.random.normal(0.0001, 0.02, 950),  # 95% normal
                np.random.normal(-0.15, 0.05, 50)    # 5% crashes extrêmes
            ]),
            "expected_position_range": (10, 40),  # % du capital (très conservateur)
            "description": "Marché avec risques de queue, position sizing minimal"
        }
    }

@pytest.fixture
def tier_transition_scenarios():
    """Scénarios pour tester les transitions entre paliers."""
    return [
        {"capital": 20.0, "expected_tier": "Micro Capital", "max_position_pct": 90},
        {"capital": 50.0, "expected_tier": "Small Capital", "max_position_pct": 75},
        {"capital": 300.0, "expected_tier": "Medium Capital", "max_position_pct": 60},
        {"capital": 1000.0, "expected_tier": "Large Capital", "max_position_pct": 45},
        {"capital": 5000.0, "expected_tier": "Enterprise", "max_position_pct": 30}
    ]

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Configuration automatique pour chaque test."""
    # Configurer numpy pour la reproductibilité
    np.random.seed(42)

    # Configurer le logging pour les tests
    logging.getLogger().setLevel(logging.WARNING)  # Réduire le bruit dans les tests

    yield

    # Cleanup après chaque test si nécessaire
    pass

# Helpers pour les assertions communes
class TestHelpers:
    """Utilitaires d'assertion pour les tests."""

    @staticmethod
    def assert_in_range(value: float, min_val: float, max_val: float, msg: str = ""):
        """Vérifie qu'une valeur est dans un intervalle."""
        assert min_val <= value <= max_val, f"{msg}: {value} not in [{min_val}, {max_val}]"

    @staticmethod
    def assert_cvar_properties(position_size: float, capital: float, cvar: float):
        """Vérifie les propriétés attendues du CVaR position sizing."""
        position_pct = (position_size / capital) * 100

        # Position doit être positive
        assert position_size > 0, f"Position size must be positive: {position_size}"

        # Position ne doit pas dépasser 100% du capital
        assert position_pct <= 100, f"Position exceeds capital: {position_pct}%"

        # Si CVaR élevé, position doit être réduite
        if abs(cvar) > 0.05:  # CVaR > 5%
            assert position_pct < 50, f"High CVaR should reduce position: {position_pct}% with CVaR {cvar}"

    @staticmethod
    def assert_tier_compliance(portfolio_manager, capital: float):
        """Vérifie que le portfolio respecte les contraintes du palier."""
        tier = portfolio_manager.get_current_tier()

        # Vérifier que le capital correspond au bon palier
        assert tier['min_capital'] <= capital, f"Capital {capital} below tier minimum {tier['min_capital']}"

        if tier.get('max_capital'):
            assert capital <= tier['max_capital'], f"Capital {capital} above tier maximum {tier['max_capital']}"

@pytest.fixture
def helpers():
    """Fixture pour accéder aux helpers de test."""
    return TestHelpers
