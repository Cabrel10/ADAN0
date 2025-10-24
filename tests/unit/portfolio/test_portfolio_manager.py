"""Tests unitaires pour le gestionnaire de portefeuille."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

def test_portfolio_manager_initialization():
    """Test l'initialisation du gestionnaire de portefeuille."""
    # Configuration de test
    config = {
        "trading": {
            "initial_balance": 10000,
            "max_position_size": 0.1,
            "max_leverage": 10,
            "trading_fee": 0.001
        }
    }
    
    # Initialiser le gestionnaire
    portfolio = PortfolioManager(config)
    
    # Vérifications
    assert portfolio.balance == 10000
    assert portfolio.initial_balance == 10000
    assert portfolio.max_position_size == 0.1
    assert portfolio.max_leverage == 10
    assert portfolio.trading_fee == 0.001

def test_open_position():
    """Test l'ouverture d'une position."""
    # Configuration de test
    config = {
        "trading": {
            "initial_balance": 10000,
            "max_position_size": 0.5,
            "max_leverage": 10,
            "trading_fee": 0.001
        }
    }
    
    # Initialiser le gestionnaire
    portfolio = PortfolioManager(config)
    
    # Ouvrir une position longue
    success = portfolio.open_position(
        asset="BTCUSDT",
        position_type="long",
        entry_price=50000,
        size=0.1,  # 10% du portefeuille
        leverage=2,
        current_time=pd.Timestamp("2023-01-01 00:00:00")
    )
    
    # Vérifications
    assert success is True
    assert len(portfolio.positions) == 1
    assert portfolio.positions[0]["asset"] == "BTCUSDT"
    assert portfolio.positions[0]["position_type"] == "long"
    assert portfolio.positions[0]["entry_price"] == 50000
    assert portfolio.positions[0]["size"] == 0.1
    assert portfolio.positions[0]["leverage"] == 2
    assert portfolio.balance < 10000  # Les frais ont été déduits

def test_close_position():
    """Test la fermeture d'une position."""
    # Configuration de test
    config = {
        "trading": {
            "initial_balance": 10000,
            "max_position_size": 0.5,
            "max_leverage": 10,
            "trading_fee": 0.001
        }
    }
    
    # Initialiser le gestionnaire et ouvrir une position
    portfolio = PortfolioManager(config)
    portfolio.open_position(
        asset="BTCUSDT",
        position_type="long",
        entry_price=50000,
        size=0.1,
        leverage=2,
        current_time=pd.Timestamp("2023-01-01 00:00:00")
    )
    
    # Fermer la position
    initial_balance = portfolio.balance
    success, pnl = portfolio.close_position(
        position_id=0,
        exit_price=55000,  # Prix de sortie plus élevé que l'entrée
        current_time=pd.Timestamp("2023-01-02 00:00:00")
    )
    
    # Vérifications
    assert success is True
    assert pnl > 0  # PnL positif car le prix a augmenté
    assert len(portfolio.positions) == 0
    assert portfolio.balance > initial_balance  # Le solde a augmenté

def test_calculate_pnl():
    """Test le calcul du PnL d'une position."""
    # Configuration de test
    config = {
        "trading": {
            "initial_balance": 10000,
            "max_position_size": 0.5,
            "max_leverage": 10,
            "trading_fee": 0.001
        }
    }
    
    # Initialiser le gestionnaire
    portfolio = PortfolioManager(config)
    
    # Tester une position longue avec profit
    pnl_long_profit = portfolio._calculate_pnl(
        entry_price=50000,
        exit_price=55000,
        size=0.1,
        leverage=2,
        position_type="long"
    )
    assert pnl_long_profit > 0  # Profit car le prix a augmenté
    
    # Tester une position longue avec perte
    pnl_long_loss = portfolio._calculate_pnl(
        entry_price=50000,
        exit_price=45000,
        size=0.1,
        leverage=2,
        position_type="long"
    )
    assert pnl_long_loss < 0  # Perte car le prix a baissé
    
    # Tester une position courte avec profit
    pnl_short_profit = portfolio._calculate_pnl(
        entry_price=50000,
        exit_price=45000,
        size=0.1,
        leverage=2,
        position_type="short"
    )
    assert pnl_short_profit > 0  # Profit car le prix a baissé
    
    # Tester une position courte avec perte
    pnl_short_loss = portfolio._calculate_pnl(
        entry_price=50000,
        exit_price=55000,
        size=0.1,
        leverage=2,
        position_type="short"
    )
    assert pnl_short_loss < 0  # Perte car le prix a augmenté
