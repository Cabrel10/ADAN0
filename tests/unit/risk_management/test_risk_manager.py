"""Tests unitaires pour le gestionnaire de risques."""

import numpy as np
from src.adan_trading_bot.risk_management.risk_manager import RiskManager


def test_risk_manager_initialization():
    """Test l'initialisation du gestionnaire de risques."""
    # Configuration de test
    config = {
        "risk_management": {
            "max_drawdown": 0.2,
            "max_position_risk": 0.02,
            "max_portfolio_risk": 0.1,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.1,
            "volatility_lookback": 20,
            "max_volatility_pct": 0.5
        }
    }
    
    # Initialiser le gestionnaire
    risk_manager = RiskManager(config)
    
    # Vérifications
    if not (risk_manager.max_drawdown == 0.2 and
            risk_manager.max_position_risk == 0.02 and
            risk_manager.max_portfolio_risk == 0.1 and
            risk_manager.stop_loss_pct == 0.05 and
            risk_manager.take_profit_pct == 0.1 and
            risk_manager.volatility_lookback == 20 and
            risk_manager.max_volatility_pct == 0.5):
        raise AssertionError("L'initialisation du gestionnaire de risques a échoué")

def test_validate_trade():
    """Test la validation d'une opération de trading."""
    # Configuration de test
    config = {
        "risk_management": {
            "max_drawdown": 0.2,
            "max_position_risk": 0.02,
            "max_portfolio_risk": 0.1,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.1,
            "volatility_lookback": 20,
            "max_volatility_pct": 0.5
        }
    }
    
    # Initialiser le gestionnaire
    risk_manager = RiskManager(config)
    
    # Données de marché simulées
    market_data = {
        'close': np.linspace(100, 200, 50).tolist(),
        'high': np.linspace(105, 205, 50).tolist(),
        'low': np.linspace(95, 195, 50).tolist(),
        'volume': np.ones(50) * 1000
    }
    
    # Tester une opération valide
    is_valid, reason = risk_manager.validate_trade(
        asset="BTCUSDT",
        position_type="long",
        entry_price=150,
        size=0.1,
        portfolio_value=10000,
        market_data=market_data
    )
    
    assert is_valid is True
    assert reason == ""
    
    # Tester une opération avec un risque trop élevé
    is_valid, reason = risk_manager.validate_trade(
        asset="BTCUSDT",
        position_type="long",
        entry_price=150,
        size=0.5,  # Taille de position trop importante
        portfolio_value=10000,
        market_data=market_data
    )
    
    assert is_valid is False
    assert "position size" in reason.lower()

def test_calculate_position_size():
    """Test le calcul de la taille de position."""
    # Configuration de test
    config = {
        "risk_management": {
            "max_position_risk": 0.02,
            "stop_loss_pct": 0.05
        }
    }
    
    # Initialiser le gestionnaire
    risk_manager = RiskManager(config)
    
    # Tester le calcul
    portfolio_value = 10000
    entry_price = 100
    stop_loss_price = 95  # 5% de stop loss
    
    position_size = risk_manager.calculate_position_size(
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        portfolio_value=portfolio_value
    )
    
    # Vérifier que la taille de position ne dépasse pas le risque maximum
    max_risk_amount = portfolio_value * 0.02  # 2% de risque
    risk_per_share = entry_price - stop_loss_price
    expected_size = min(max_risk_amount / risk_per_share, 1.0)  # Ne pas dépasser 100%
    
    assert position_size == pytest.approx(expected_size)

def test_calculate_volatility():
    """Test le calcul de la volatilité."""
    # Configuration de test
    config = {"risk_management": {"volatility_lookback": 20}}
    
    # Initialiser le gestionnaire
    risk_manager = RiskManager(config)
    
    # Données de test (prix constants)
    market_data = {
        'close': [100] * 30,
        'high': [101] * 30,
        'low': [99] * 30
    }
    
    # Volatilité nulle car les prix ne changent pas
    volatility = risk_manager.calculate_volatility(market_data)
    assert volatility == pytest.approx(0.0)
    
    # Données avec volatilité
    market_data_volatile = {
        'close': [100, 105, 95, 110, 90],
        'high': [101, 106, 96, 111, 91],
        'low': [99, 104, 94, 109, 89]
    }
    
    volatility = risk_manager.calculate_volatility(market_data_volatile)
    assert volatility > 0.0
