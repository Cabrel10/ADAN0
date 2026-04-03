"""
Module d'API d'exchange pour le système ADAN Trading Bot.
Gère les connexions aux exchanges via CCXT pour le paper trading et le trading live.
"""

from .connector import (
    get_exchange_client,
    test_exchange_connection,
    get_market_info,
    validate_exchange_config,
    ExchangeConnectionError,
    ExchangeConfigurationError
)

__all__ = [
    'get_exchange_client',
    'test_exchange_connection', 
    'get_market_info',
    'validate_exchange_config',
    'ExchangeConnectionError',
    'ExchangeConfigurationError'
]

__version__ = '1.0.0'
__author__ = 'ADAN Trading Bot Team'