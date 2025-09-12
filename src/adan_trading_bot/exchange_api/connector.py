"""
Module de connexion aux exchanges via CCXT pour le système ADAN.
Gère l'initialisation des clients d'exchange et la connexion aux APIs.
"""

import ccxt
import os
import time
from typing import Optional, Dict, Any
from ..common.utils import get_logger

logger = get_logger(__name__)


class ExchangeConnectionError(Exception):
    """Exception levée lors d'erreurs de connexion à l'exchange."""
    pass


class ExchangeConfigurationError(Exception):
    """Exception levée lors d'erreurs de configuration de l'exchange."""
    pass


def get_exchange_client(config: Dict[str, Any]) -> ccxt.Exchange:
    """
    Initialise et retourne un client d'exchange CCXT basé sur la configuration.
    Lit les clés API depuis les variables d'environnement.
    
    Args:
        config: Configuration complète du système contenant la section paper_trading
        
    Returns:
        ccxt.Exchange: Client d'exchange initialisé et configuré
        
    Raises:
        ExchangeConfigurationError: Si la configuration est invalide
        ExchangeConnectionError: Si la connexion à l'exchange échoue
        ValueError: Si les clés API sont manquantes
    """
    paper_config = config.get('paper_trading', {})
    exchange_id = paper_config.get('exchange_id')
    use_testnet = paper_config.get('use_testnet', False)

    if not exchange_id:
        logger.error("L'ID de l'exchange (exchange_id) n'est pas spécifié dans la configuration paper_trading.")
        raise ExchangeConfigurationError("exchange_id manquant dans la configuration paper_trading.")

    # Déterminer les noms des variables d'environnement pour les clés API
    # Convention: {EXCHANGE_ID_UPPER}_API_KEY, {EXCHANGE_ID_UPPER}_SECRET_KEY
    # Pour Testnet: {EXCHANGE_ID_UPPER}_TESTNET_API_KEY, {EXCHANGE_ID_UPPER}_TESTNET_SECRET_KEY
    
    env_key_prefix = exchange_id.upper()
    if use_testnet:
        env_key_prefix += "_TESTNET"
        
    api_key_env_var = f"{env_key_prefix}_API_KEY"
    secret_key_env_var = f"{env_key_prefix}_SECRET_KEY"
    
    api_key = os.environ.get(api_key_env_var)
    secret_key = os.environ.get(secret_key_env_var)

    if not api_key or not secret_key:
        logger.error(f"Clés API ({api_key_env_var}, {secret_key_env_var}) non trouvées pour {exchange_id} {'Testnet' if use_testnet else 'Live'}.")
        raise ValueError(f"Clés API manquantes pour {exchange_id} {'Testnet' if use_testnet else 'Live'}. "
                        f"Vérifiez les variables d'environnement {api_key_env_var} et {secret_key_env_var}.")

    try:
        # Vérifier que l'exchange est supporté par CCXT
        if not hasattr(ccxt, exchange_id):
            logger.error(f"L'ID d'exchange '{exchange_id}' n'est pas supporté par CCXT.")
            raise ExchangeConfigurationError(f"Exchange ID '{exchange_id}' non supporté par CCXT.")
        
        exchange_class = getattr(ccxt, exchange_id)
        
        # Configuration de base du client
        client_config = {
            'apiKey': api_key,
            'secret': secret_key,
            'timeout': 30000,  # 30 secondes
            'rateLimit': 1200,  # Limite de taux par défaut
            'options': {
                'defaultType': 'spot',  # Trading spot par défaut
            },
        }
        
        # Initialiser le client d'exchange
        exchange = exchange_class(client_config)

        # Activer le mode testnet si demandé
        if use_testnet:
            if hasattr(exchange, 'set_sandbox_mode'):
                exchange.set_sandbox_mode(True)
                logger.info(f"Client CCXT pour '{exchange_id}' initialisé en mode TESTNET.")
            else:
                logger.warning(f"L'exchange '{exchange_id}' ne supporte peut-être pas set_sandbox_mode(). "
                             "Vérifiez la configuration manuelle des URLs si nécessaire.")
        else:
            logger.info(f"Client CCXT pour '{exchange_id}' initialisé en mode LIVE (PRODUCTION).")
            logger.warning("⚠️  MODE LIVE ACTIVÉ - ATTENTION AUX ORDRES RÉELS ⚠️")
        
        # Test de connectivité de base
        try:
            server_time = exchange.fetch_time()
            logger.info(f"Connexion à {exchange_id} réussie. Heure du serveur: {exchange.iso8601(server_time)}")
        except Exception as e_conn:
            logger.error(f"Échec du test de connexion à {exchange_id}: {e_conn}")
            raise ExchangeConnectionError(f"Impossible de se connecter à {exchange_id}") from e_conn
        
        return exchange
        
    except AttributeError:
        logger.error(f"L'ID d'exchange '{exchange_id}' n'est pas un attribut valide de ccxt.")
        raise ExchangeConfigurationError(f"Exchange ID '{exchange_id}' non supporté par CCXT.") from None
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du client d'exchange CCXT pour '{exchange_id}': {e}")
        raise ExchangeConnectionError(f"Erreur d'initialisation CCXT pour {exchange_id}.") from e


def test_exchange_connection(exchange: ccxt.Exchange) -> Dict[str, Any]:
    """
    Teste la connexion à l'exchange et retourne des informations de diagnostic.
    
    Args:
        exchange: Client d'exchange CCXT initialisé
        
    Returns:
        Dict contenant les résultats des tests de connexion
    """
    results = {
        'exchange_id': exchange.id,
        'testnet_mode': getattr(exchange, 'sandbox', False),
        'server_time': None,
        'markets_loaded': False,
        'balance_accessible': False,
        'market_count': 0,
        'errors': []
    }
    
    try:
        # Test 1: Heure du serveur
        server_time = exchange.fetch_time()
        results['server_time'] = exchange.iso8601(server_time)
        logger.info(f"✅ Heure du serveur: {results['server_time']}")
    except Exception as e:
        error_msg = f"❌ Erreur lors de la récupération de l'heure du serveur: {e}"
        logger.error(error_msg)
        results['errors'].append(error_msg)
    
    try:
        # Test 2: Chargement des marchés
        markets = exchange.load_markets()
        results['markets_loaded'] = True
        results['market_count'] = len(markets)
        logger.info(f"✅ Marchés chargés: {results['market_count']} paires disponibles")
        
        # Vérifier quelques paires importantes
        important_pairs = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
        available_pairs = [pair for pair in important_pairs if pair in markets]
        if available_pairs:
            logger.info(f"✅ Paires importantes disponibles: {', '.join(available_pairs)}")
        else:
            logger.warning("⚠️ Aucune paire importante trouvée dans les marchés disponibles")
            
    except Exception as e:
        error_msg = f"❌ Erreur lors du chargement des marchés: {e}"
        logger.error(error_msg)
        results['errors'].append(error_msg)
    
    try:
        # Test 3: Accès au solde
        balance = exchange.fetch_balance()
        results['balance_accessible'] = True
        
        # Afficher les soldes non nuls
        non_zero_balances = {currency: amount for currency, amount in balance['total'].items() if amount > 0}
        if non_zero_balances:
            logger.info(f"✅ Soldes disponibles: {non_zero_balances}")
        else:
            logger.info("ℹ️ Aucun solde disponible ou tous les soldes sont à zéro")
            
    except Exception as e:
        error_msg = f"❌ Erreur lors de l'accès au solde: {e}"
        logger.error(error_msg)
        results['errors'].append(error_msg)
    
    # Résumé du test
    if not results['errors']:
        logger.info("🎉 Tous les tests de connexion ont réussi !")
    else:
        logger.warning(f"⚠️ {len(results['errors'])} erreur(s) détectée(s) lors des tests de connexion")
    
    return results


def get_market_info(exchange: ccxt.Exchange, symbol: str) -> Optional[Dict[str, Any]]:
    """
    Récupère les informations détaillées d'un marché spécifique.
    
    Args:
        exchange: Client d'exchange CCXT
        symbol: Symbole du marché (ex: 'BTC/USDT')
        
    Returns:
        Dict contenant les informations du marché ou None si non trouvé
    """
    try:
        markets = exchange.load_markets()
        if symbol not in markets:
            logger.warning(f"Marché '{symbol}' non trouvé sur {exchange.id}")
            return None
        
        market = markets[symbol]
        
        # Extraire les informations importantes
        market_info = {
            'symbol': market['symbol'],
            'base': market['base'],
            'quote': market['quote'],
            'active': market.get('active', True),
            'type': market.get('type', 'spot'),
            'spot': market.get('spot', True),
            'margin': market.get('margin', False),
            'future': market.get('future', False),
            'limits': market.get('limits', {}),
            'precision': market.get('precision', {}),
            'fees': market.get('fees', {}),
        }
        
        logger.info(f"Informations du marché {symbol}: {market_info}")
        return market_info
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des informations du marché {symbol}: {e}")
        return None


def validate_exchange_config(config: Dict[str, Any]) -> bool:
    """
    Valide la configuration de l'exchange avant l'initialisation.
    
    Args:
        config: Configuration complète du système
        
    Returns:
        bool: True si la configuration est valide, False sinon
    """
    paper_config = config.get('paper_trading', {})
    
    # Vérifications de base
    if not paper_config:
        logger.error("Section 'paper_trading' manquante dans la configuration")
        return False
    
    exchange_id = paper_config.get('exchange_id')
    if not exchange_id:
        logger.error("'exchange_id' manquant dans la configuration paper_trading")
        return False
    
    # Vérifier que l'exchange est supporté par CCXT
    if not hasattr(ccxt, exchange_id):
        logger.error(f"Exchange '{exchange_id}' non supporté par CCXT")
        return False
    
    # Vérifier la présence des variables d'environnement
    use_testnet = paper_config.get('use_testnet', False)
    env_key_prefix = exchange_id.upper()
    if use_testnet:
        env_key_prefix += "_TESTNET"
    
    api_key_env_var = f"{env_key_prefix}_API_KEY"
    secret_key_env_var = f"{env_key_prefix}_SECRET_KEY"
    
    if not os.environ.get(api_key_env_var):
        logger.error(f"Variable d'environnement '{api_key_env_var}' non définie")
        return False
    
    if not os.environ.get(secret_key_env_var):
        logger.error(f"Variable d'environnement '{secret_key_env_var}' non définie")
        return False
    
    logger.info("✅ Configuration de l'exchange validée avec succès")
    return True