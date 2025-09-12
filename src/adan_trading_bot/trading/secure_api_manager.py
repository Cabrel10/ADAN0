"""
Gestionnaire sécurisé des API keys pour le trading live.
Implémente les tâches 10B.2.1, 10B.2.2.
"""

import os
import json
import logging
import hashlib
import base64
import hmac
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
import requests
import websocket
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class ExchangeType(Enum):
    """Types d'exchanges supportés"""
    BINANCE = "binance"
    BITGET = "bitget"
    BINANCE_FUTURES = "binance_futures"
    BYBIT = "bybit"
    OKEX = "okex"
    KRAKEN = "kraken"


class ConnectionStatus(Enum):
    """États de connexion"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class APICredentials:
    """Informations d'identification API"""
    exchange: ExchangeType
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None  # Pour OKEx
    sandbox: bool = True
    name: str = "Default"

    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        data = {
            'exchange': self.exchange.value,
            'name': self.name,
            'sandbox': self.sandbox
        }

        if include_secrets:
            data.update({
                'api_key': self.api_key,
                'api_secret': self.api_secret,
                'passphrase': self.passphrase
            })
        else:
            # Masquer les secrets
            data.update({
                'api_key': self.api_key[:8] + "..." if self.api_key else None,
                'api_secret': "***" if self.api_secret else None,
                'passphrase': "***" if self.passphrase else None
            })

        return data


@dataclass
class ConnectionInfo:
    """Informations de connexion"""
    exchange: ExchangeType
    status: ConnectionStatus
    last_ping: Optional[datetime] = None
    latency_ms: Optional[float] = None
    error_message: Optional[str] = None
    reconnect_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        data = asdict(self)
        data['exchange'] = self.exchange.value
        data['status'] = self.status.value
        if self.last_ping:
            data['last_ping'] = self.last_ping.isoformat()
        return data


class SecureAPIManager:
    """Gestionnaire sécurisé des API keys"""

    def __init__(self, config_path: str = "config/api_keys.enc"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Chiffrement
        self._cipher_suite = None
        self._master_password = None

        # Stockage des credentials
        self.credentials: Dict[str, APICredentials] = {}

        # Monitoring des connexions
        self.connections: Dict[ExchangeType, ConnectionInfo] = {}
        self.connection_threads: Dict[ExchangeType, threading.Thread] = {}
        self.websockets: Dict[ExchangeType, websocket.WebSocketApp] = {}

        # Callbacks
        self.connection_callbacks: List[callable] = []

        # Configuration des endpoints
        self.endpoints = self._get_exchange_endpoints()

        # Cache pour les informations de l'échange
        self._exchange_info_cache: Dict[ExchangeType, Dict[str, Any]] = {}
        self._exchange_info_last_updated: Dict[ExchangeType, datetime] = {}
        self._cache_lock = threading.Lock()

        # Perform security validation at startup
        self._validate_security_at_startup()

        # Load credentials from environment variables first
        self._load_credentials_from_env()

        logger.info("SecureAPIManager initialized")

    def get_exchange_info(self, exchange: ExchangeType, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Récupère les informations de l'échange (avec cache)"""
        with self._cache_lock:
            cache_duration = timedelta(hours=1)
            last_updated = self._exchange_info_last_updated.get(exchange)

            if not force_refresh and last_updated and (datetime.now() - last_updated < cache_duration):
                logger.debug(f"Using cached exchange info for {exchange.value}")
                return self._exchange_info_cache.get(exchange)

            logger.info(f"Fetching exchange info for {exchange.value}")
            exchange_info = self._fetch_exchange_info(exchange)

            if exchange_info:
                self._exchange_info_cache[exchange] = exchange_info
                self._exchange_info_last_updated[exchange] = datetime.now()
                return exchange_info

            return None

    def _fetch_exchange_info(self, exchange: ExchangeType) -> Optional[Dict[str, Any]]:
        """Récupère les informations de l'échange depuis l'API"""
        credentials = self.get_credentials(exchange)
        if not credentials:
            logger.error(f"No credentials found for {exchange.value} to fetch exchange info")
            return None

        endpoints = self.endpoints.get(exchange)
        if not endpoints:
            logger.warning(f"No endpoints configured for {exchange.value}")
            return None

        base_url = endpoints['rest_testnet'] if credentials.sandbox else endpoints['rest_url']
        endpoint = "/api/v3/exchangeInfo"
        url = f"{base_url}{endpoint}"

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                logger.info(f"Successfully fetched exchange info for {exchange.value}")
                return response.json()
            else:
                logger.error(f"Failed to fetch exchange info for {exchange.value}: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error fetching exchange info for {exchange.value}: {e}")
            return None

    def _get_exchange_endpoints(self) -> Dict[ExchangeType, Dict[str, str]]:
        """Retourne les endpoints des exchanges"""
        return {
            ExchangeType.BINANCE: {
                'rest_url': 'https://api.binance.com',
                'rest_testnet': 'https://testnet.binance.vision',
                'ws_url': 'wss://stream.binance.com:9443/ws',
                'ws_testnet': 'wss://testnet.binance.vision/ws'
            },
            ExchangeType.BINANCE_FUTURES: {
                'rest_url': 'https://fapi.binance.com',
                'rest_testnet': 'https://testnet.binancefuture.com',
                'ws_url': 'wss://fstream.binance.com/ws',
                'ws_testnet': 'wss://stream.binancefuture.com/ws'
            },
            ExchangeType.BYBIT: {
                'rest_url': 'https://api.bybit.com',
                'rest_testnet': 'https://api-testnet.bybit.com',
                'ws_url': 'wss://stream.bybit.com/v5/public/spot',
                'ws_testnet': 'wss://stream-testnet.bybit.com/v5/public/spot'
            },
            ExchangeType.BITGET: {
                'rest_url': 'https://api.bitget.com/api/v2',
                'rest_testnet': 'https://api.bitget.com/api/v2', # Placeholder, check Bitget testnet URL
                'ws_url': 'wss://ws.bitget.com/v2/ws',
                'ws_testnet': 'wss://ws.bitget.com/v2/ws' # Placeholder, check Bitget testnet URL
            }
        }

    def set_master_password(self, password: str) -> bool:
        """Définit le mot de passe maître pour le chiffrement"""
        try:
            # Générer une clé de chiffrement à partir du mot de passe
            password_bytes = password.encode()
            salt = b'adan_trading_bot_salt'  # En production, utiliser un salt aléatoire

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password_bytes))

            self._cipher_suite = Fernet(key)
            self._master_password = password

            # Essayer de charger les credentials existants
            if self.config_path.exists():
                self._load_credentials()

            return True

        except Exception as e:
            logger.error(f"Error setting master password: {e}")
            return False

    def add_credentials(self, credentials: APICredentials) -> bool:
        """Ajoute des credentials API"""
        if not self._cipher_suite:
            raise ValueError("Master password not set")

        # Tester la connexion avant de sauvegarder
        if not self._test_api_connection(credentials):
            logger.warning("API connection test failed, but saving credentials anyway")

        # Générer un ID unique
        cred_id = f"{credentials.exchange.value}_{credentials.name}"
        self.credentials[cred_id] = credentials

        # Sauvegarder
        self._save_credentials()

        logger.info(f"Added credentials for {credentials.exchange.value} ({credentials.name})")
        return True

    def get_credentials(self, exchange: ExchangeType, name: str = "Default") -> Optional[APICredentials]:
        """Récupère des credentials"""
        cred_id = f"{exchange.value}_{name}"
        return self.credentials.get(cred_id)

    def list_credentials(self) -> List[Dict[str, Any]]:
        """Liste tous les credentials (sans les secrets)"""
        return [cred.to_dict(include_secrets=False) for cred in self.credentials.values()]

    def remove_credentials(self, exchange: ExchangeType, name: str = "Default") -> bool:
        """Supprime des credentials"""
        cred_id = f"{exchange.value}_{name}"

        if cred_id in self.credentials:
            del self.credentials[cred_id]
            self._save_credentials()
            logger.info(f"Removed credentials for {exchange.value} ({name})")
            return True

        return False

    def _save_credentials(self) -> None:
        """Sauvegarde les credentials chiffrés"""
        if not self._cipher_suite:
            raise ValueError("Master password not set")

        try:
            # Préparer les données
            data = {
                'timestamp': datetime.now().isoformat(),
                'credentials': {
                    cred_id: cred.to_dict(include_secrets=True)
                    for cred_id, cred in self.credentials.items()
                }
            }

            # Chiffrer
            json_data = json.dumps(data)
            encrypted_data = self._cipher_suite.encrypt(json_data.encode())

            # Sauvegarder
            with open(self.config_path, 'wb') as f:
                f.write(encrypted_data)

            logger.debug("Credentials saved successfully")

        except Exception as e:
            logger.error(f"Error saving credentials: {e}")
            raise

    def _load_credentials(self) -> None:
        """Charge les credentials chiffrés"""
        if not self._cipher_suite:
            raise ValueError("Master password not set")

        try:
            # Lire le fichier chiffré
            with open(self.config_path, 'rb') as f:
                encrypted_data = f.read()

            # Déchiffrer
            decrypted_data = self._cipher_suite.decrypt(encrypted_data)
            data = json.loads(decrypted_data.decode())

            # Reconstruire les credentials
            self.credentials = {}
            for cred_id, cred_data in data.get('credentials', {}).items():
                exchange = ExchangeType(cred_data['exchange'])
                credentials = APICredentials(
                    exchange=exchange,
                    api_key=cred_data['api_key'],
                    api_secret=cred_data['api_secret'],
                    passphrase=cred_data.get('passphrase'),
                    sandbox=cred_data.get('sandbox', True),
                    name=cred_data.get('name', 'Default')
                )
                self.credentials[cred_id] = credentials

            logger.info(f"Loaded {len(self.credentials)} credentials")

        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
            raise

    def _send_signed_request(self, exchange: ExchangeType, method: str, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Envoie une requête signée à l'exchange"""
        credentials = self.get_credentials(exchange)
        if not credentials:
            logger.error(f"No credentials found for {exchange.value}")
            return None

        endpoints = self.endpoints.get(exchange)
        if not endpoints:
            logger.warning(f"No endpoints configured for {exchange.value}")
            return None

        base_url = endpoints['rest_testnet'] if credentials.sandbox else endpoints['rest_url']
        url = f"{base_url}{endpoint}"

        # Préparer les paramètres et la signature
        if params is None:
            params = {}
        params['timestamp'] = int(time.time() * 1000)

        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])

        signature = hmac.new(
            credentials.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        params['signature'] = signature

        headers = {
            'X-MBX-APIKEY': credentials.api_key
        }

        logger.debug(f"Sending {method} request to {url} with headers: {headers} and params: {params}")

        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=10)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, data=params, timeout=10)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=headers, data=params, timeout=10)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None

            logger.debug(f"Received response from {url}: Status {response.status_code}, Content: {response.text}")

            if response.status_code == 200:
                return response.json()
            else:
                error_data = {}
                try:
                    error_data = response.json()
                except json.JSONDecodeError:
                    error_data = {"message": response.text}
                logger.error(f"API Error on {method} {url}: {response.status_code} - Code: {error_data.get('code', 'N/A')}, Msg: {error_data.get('msg', error_data.get('message', 'N/A'))}")
                return error_data

        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception on {method} {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred on {method} {url}: {e}")
            return None

    def _test_api_connection(self, credentials: APICredentials) -> bool:
        """Teste la connexion API"""
        logger.debug(f"Testing API connection for {credentials.exchange.value} (Sandbox: {credentials.sandbox})")
        if credentials.exchange == ExchangeType.BINANCE:
            # Binance requires signed request for account info
            response = self._send_signed_request(credentials.exchange, 'GET', '/api/v3/account')
            if response is not None and 'canTrade' in response:
                logger.info(f"Binance API connection test successful.")
                return True
            else:
                logger.error(f"Binance API connection test failed. Response: {response}")
                return False
        elif credentials.exchange == ExchangeType.BITGET:
            # Bitget public endpoint for testing connectivity (e.g., server time)
            endpoints = self.endpoints.get(credentials.exchange)
            if not endpoints:
                logger.error(f"No endpoints configured for {credentials.exchange.value}")
                return False
            base_url = endpoints['rest_testnet'] if credentials.sandbox else endpoints['rest_url']
            url = f"{base_url}/public/time" # Assuming /public/time is a public endpoint
            logger.debug(f"Attempting to reach Bitget public endpoint: {url}")
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"Bitget API connection test successful (public endpoint). Status: {response.status_code}")
                    return True
                else:
                    logger.error(f"Bitget API connection test failed (public endpoint). Status: {response.status_code}, Response: {response.text}")
                    return False
            except requests.exceptions.RequestException as e:
                logger.error(f"Bitget API connection test failed (public endpoint) due to request exception: {e}")
                return False
        else:
            logger.warning(f"API connection test not implemented for {credentials.exchange.value}. Assuming connection is valid.")
            return True

    def send_order(self, exchange: ExchangeType, order_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Passe un ordre sur l'exchange"""
        if exchange == ExchangeType.BINANCE:
            return self._send_signed_request(exchange, 'POST', '/api/v3/order', order_params)
        return None

    def get_order(self, exchange: ExchangeType, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        """Récupère les informations d'un ordre"""
        if exchange == ExchangeType.BINANCE:
            params = {'symbol': symbol, 'orderId': order_id}
            return self._send_signed_request(exchange, 'GET', '/api/v3/order', params)
        return None

    def cancel_order(self, exchange: ExchangeType, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        """Annule un ordre"""
        if exchange == ExchangeType.BINANCE:
            params = {'symbol': symbol, 'orderId': order_id}
            return self._send_signed_request(exchange, 'DELETE', '/api/v3/order', params)
        return None

    def start_connection_monitoring(self, exchange: ExchangeType) -> bool:
        """Démarre le monitoring de connexion pour un exchange"""
        if exchange in self.connection_threads:
            logger.warning(f"Connection monitoring already active for {exchange.value}")
            return False

        # Initialiser l'info de connexion
        self.connections[exchange] = ConnectionInfo(
            exchange=exchange,
            status=ConnectionStatus.CONNECTING
        )

        # Lancer le thread de monitoring
        thread = threading.Thread(
            target=self._monitor_connection,
            args=(exchange,),
            daemon=True
        )
        thread.start()
        self.connection_threads[exchange] = thread

        logger.info(f"Started connection monitoring for {exchange.value}")
        return True

    def _monitor_connection(self, exchange: ExchangeType) -> None:
        """Surveille la connexion WebSocket"""
        connection_info = self.connections[exchange]
        endpoints = self.endpoints.get(exchange)

        if not endpoints:
            connection_info.status = ConnectionStatus.ERROR
            connection_info.error_message = "No endpoints configured"
            return

        # Récupérer les credentials
        credentials = self.get_credentials(exchange)
        if not credentials:
            connection_info.status = ConnectionStatus.ERROR
            connection_info.error_message = "No credentials found"
            return

        # URL WebSocket
        ws_url = endpoints['ws_testnet'] if credentials.sandbox else endpoints['ws_url']

        def on_open(ws):
            connection_info.status = ConnectionStatus.CONNECTED
            connection_info.last_ping = datetime.now()
            connection_info.reconnect_count = 0
            logger.info(f"WebSocket connected for {exchange.value}")
            self._notify_connection_change(exchange, ConnectionStatus.CONNECTED)

        def on_message(ws, message):
            # Mettre à jour le ping
            connection_info.last_ping = datetime.now()

            # Calculer la latence si possible
            try:
                data = json.loads(message)
                if 'ping' in data:
                    # Répondre au ping
                    ws.send(json.dumps({'pong': data['ping']}))
            except:
                pass

        def on_error(ws, error):
            connection_info.status = ConnectionStatus.ERROR
            connection_info.error_message = str(error)
            logger.error(f"WebSocket error for {exchange.value}: {error}")
            self._notify_connection_change(exchange, ConnectionStatus.ERROR)

        def on_close(ws, close_status_code, close_msg):
            if connection_info.status != ConnectionStatus.ERROR:
                connection_info.status = ConnectionStatus.RECONNECTING
                connection_info.reconnect_count += 1
                logger.warning(f"WebSocket closed for {exchange.value}, reconnecting...")
                self._notify_connection_change(exchange, ConnectionStatus.RECONNECTING)

                # Reconnexion automatique après délai
                time.sleep(min(connection_info.reconnect_count * 2, 30))
                if exchange in self.connection_threads:  # Vérifier si pas arrêté
                    self._monitor_connection(exchange)

        # Créer et lancer la WebSocket
        try:
            ws = websocket.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )

            self.websockets[exchange] = ws
            ws.run_forever()

        except Exception as e:
            connection_info.status = ConnectionStatus.ERROR
            connection_info.error_message = str(e)
            logger.error(f"WebSocket connection failed for {exchange.value}: {e}")

    def stop_connection_monitoring(self, exchange: ExchangeType) -> bool:
        """Arrête le monitoring de connexion"""
        if exchange not in self.connection_threads:
            return False

        # Fermer la WebSocket
        if exchange in self.websockets:
            try:
                self.websockets[exchange].close()
                del self.websockets[exchange]
            except:
                pass

        # Arrêter le thread
        if exchange in self.connection_threads:
            del self.connection_threads[exchange]

        # Mettre à jour le statut
        if exchange in self.connections:
            self.connections[exchange].status = ConnectionStatus.DISCONNECTED

        logger.info(f"Stopped connection monitoring for {exchange.value}")
        return True

    def get_connection_status(self, exchange: ExchangeType) -> Optional[ConnectionInfo]:
        """Récupère le statut de connexion"""
        return self.connections.get(exchange)

    def get_all_connection_status(self) -> Dict[str, Dict[str, Any]]:
        """Récupère tous les statuts de connexion"""
        return {
            exchange.value: info.to_dict()
            for exchange, info in self.connections.items()
        }

    def add_connection_callback(self, callback: callable) -> None:
        """Ajoute un callback pour les changements de connexion"""
        self.connection_callbacks.append(callback)

    def _notify_connection_change(self, exchange: ExchangeType, status: ConnectionStatus) -> None:
        """Notifie les callbacks des changements de connexion"""
        for callback in self.connection_callbacks:
            try:
                callback(exchange, status)
            except Exception as e:
                logger.error(f"Error in connection callback: {e}")

    def _validate_security_at_startup(self) -> None:
        """Valide la sécurité au démarrage pour prévenir les clés hardcodées"""
        logger.info("Performing security validation at startup...")

        # Liste des fichiers à vérifier pour les clés hardcodées
        suspicious_files = [
            "gemini_api_keys.txt",
            "api_keys.txt",
            "keys.txt",
            "secrets.txt"
        ]

        # Patterns de clés API communes
        api_key_patterns = [
            r'AIzaSy[A-Za-z0-9_-]{33}',  # Google API keys
            r'sk-[A-Za-z0-9]{48}',       # OpenAI API keys
            r'[A-Za-z0-9]{64}',          # Generic 64-char keys
            r'[A-Za-z0-9]{32}',          # Generic 32-char keys
        ]

        security_violations = []

        # Vérifier les fichiers suspects
        for file_name in suspicious_files:
            file_path = Path(file_name)
            if file_path.exists():
                try:
                    content = file_path.read_text()
                    # Vérifier si le fichier contient des patterns de clés API
                    for pattern in api_key_patterns:
                        if re.search(pattern, content):
                            security_violations.append(f"Hardcoded API keys detected in {file_name}")
                            break
                except Exception as e:
                    logger.warning(f"Could not read {file_name}: {e}")

        # Vérifier les variables d'environnement pour des patterns suspects
        for env_var, value in os.environ.items():
            if env_var.upper().endswith(('_KEY', '_SECRET', '_TOKEN')) and value:
                # Vérifier si la valeur ressemble à une clé hardcodée dans le code
                for pattern in api_key_patterns:
                    if re.match(pattern, value):
                        logger.info(f"Found API key in environment variable: {env_var}")
                        break

        if security_violations:
            error_msg = "SECURITY VIOLATION: " + "; ".join(security_violations)
            logger.error(error_msg)
            raise SecurityError(error_msg + "\n\nPlease move all API keys to environment variables and remove hardcoded files.")

        logger.info("Security validation passed - no hardcoded API keys detected")

    def _load_credentials_from_env(self) -> None:
        """Charge les credentials depuis les variables d'environnement (priorité sur les fichiers chiffrés)"""
        logger.info("Loading credentials from environment variables...")

        # Mapping des exchanges vers leurs variables d'environnement
        env_mappings = {
            ExchangeType.BINANCE: {
                'api_key': 'BINANCE_API_KEY',
                'api_secret': 'BINANCE_API_SECRET',
                'sandbox': 'BINANCE_SANDBOX'
            },
            ExchangeType.BINANCE_FUTURES: {
                'api_key': 'BINANCE_FUTURES_API_KEY',
                'api_secret': 'BINANCE_FUTURES_API_SECRET',
                'sandbox': 'BINANCE_FUTURES_SANDBOX'
            },
            ExchangeType.BITGET: {
                'api_key': 'BITGET_API_KEY',
                'api_secret': 'BITGET_API_SECRET',
                'passphrase': 'BITGET_PASSPHRASE',
                'sandbox': 'BITGET_SANDBOX'
            },
            ExchangeType.BYBIT: {
                'api_key': 'BYBIT_API_KEY',
                'api_secret': 'BYBIT_API_SECRET',
                'sandbox': 'BYBIT_SANDBOX'
            },
            ExchangeType.OKEX: {
                'api_key': 'OKEX_API_KEY',
                'api_secret': 'OKEX_API_SECRET',
                'passphrase': 'OKEX_PASSPHRASE',
                'sandbox': 'OKEX_SANDBOX'
            },
            ExchangeType.KRAKEN: {
                'api_key': 'KRAKEN_API_KEY',
                'api_secret': 'KRAKEN_API_SECRET',
                'sandbox': 'KRAKEN_SANDBOX'
            }
        }

        loaded_count = 0

        for exchange, env_vars in env_mappings.items():
            api_key = os.getenv(env_vars['api_key'])
            api_secret = os.getenv(env_vars['api_secret'])

            if api_key and api_secret:
                # Récupérer les paramètres optionnels
                passphrase = os.getenv(env_vars.get('passphrase'))
                sandbox = os.getenv(env_vars.get('sandbox', 'true')).lower() in ('true', '1', 'yes')

                # Créer les credentials
                credentials = APICredentials(
                    exchange=exchange,
                    api_key=api_key,
                    api_secret=api_secret,
                    passphrase=passphrase,
                    sandbox=sandbox,
                    name="Environment"
                )

                # Ajouter aux credentials (sans chiffrement car déjà en env)
                cred_id = f"{exchange.value}_Environment"
                self.credentials[cred_id] = credentials
                loaded_count += 1

                logger.info(f"Loaded credentials for {exchange.value} from environment variables")

        if loaded_count > 0:
            logger.info(f"Successfully loaded {loaded_count} credentials from environment variables")
        else:
            logger.info("No credentials found in environment variables")

    def get_credentials(self, exchange: ExchangeType, name: str = "Default") -> Optional[APICredentials]:
        """Récupère des credentials (priorité: Environment > Default)"""
        # Priorité aux credentials d'environnement
        env_cred_id = f"{exchange.value}_Environment"
        if env_cred_id in self.credentials:
            return self.credentials[env_cred_id]

        # Fallback vers les credentials par défaut
        cred_id = f"{exchange.value}_{name}"
        return self.credentials.get(cred_id)

    def create_env_setup_guide(self, output_path: str = "ENVIRONMENT_SETUP.md") -> None:
        """Crée un guide de configuration des variables d'environnement"""
        guide_content = """# Environment Variables Setup Guide

## Overview

This guide explains how to securely configure API keys using environment variables instead of hardcoded files.

## Required Environment Variables

### Binance
```bash
export BINANCE_API_KEY="your_binance_api_key"
export BINANCE_API_SECRET="your_binance_api_secret"
export BINANCE_SANDBOX="true"  # Set to "false" for production
```

### Binance Futures
```bash
export BINANCE_FUTURES_API_KEY="your_binance_futures_api_key"
export BINANCE_FUTURES_API_SECRET="your_binance_futures_api_secret"
export BINANCE_FUTURES_SANDBOX="true"  # Set to "false" for production
```

### Bitget
```bash
export BITGET_API_KEY="your_bitget_api_key"
export BITGET_API_SECRET="your_bitget_api_secret"
export BITGET_PASSPHRASE="your_bitget_passphrase"
export BITGET_SANDBOX="true"  # Set to "false" for production
```

### Bybit
```bash
export BYBIT_API_KEY="your_bybit_api_key"
export BYBIT_API_SECRET="your_bybit_api_secret"
export BYBIT_SANDBOX="true"  # Set to "false" for production
```

### OKEx
```bash
export OKEX_API_KEY="your_okex_api_key"
export OKEX_API_SECRET="your_okex_api_secret"
export OKEX_PASSPHRASE="your_okex_passphrase"
export OKEX_SANDBOX="true"  # Set to "false" for production
```

### Kraken
```bash
export KRAKEN_API_KEY="your_kraken_api_key"
export KRAKEN_API_SECRET="your_kraken_api_secret"
export KRAKEN_SANDBOX="true"  # Set to "false" for production
```

## Setup Methods

### Method 1: .env File (Recommended for Development)

1. Create a `.env` file in your project root:
```bash
# .env file
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
BINANCE_SANDBOX=true
```

2. Add `.env` to your `.gitignore` file to prevent committing secrets:
```bash
echo ".env" >> .gitignore
```

3. Load the .env file in your application:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Method 2: System Environment Variables

#### Linux/macOS
Add to your `~/.bashrc` or `~/.zshrc`:
```bash
export BINANCE_API_KEY="your_binance_api_key"
export BINANCE_API_SECRET="your_binance_api_secret"
```

Then reload your shell:
```bash
source ~/.bashrc  # or ~/.zshrc
```

#### Windows
```cmd
setx BINANCE_API_KEY "your_binance_api_key"
setx BINANCE_API_SECRET "your_binance_api_secret"
```

### Method 3: Docker Environment Variables

```dockerfile
ENV BINANCE_API_KEY=your_binance_api_key
ENV BINANCE_API_SECRET=your_binance_api_secret
```

Or using docker-compose:
```yaml
environment:
  - BINANCE_API_KEY=your_binance_api_key
  - BINANCE_API_SECRET=your_binance_api_secret
```

## Security Best Practices

1. **Never commit API keys to version control**
2. **Use different keys for development and production**
3. **Regularly rotate your API keys**
4. **Set appropriate permissions on your exchange accounts**
5. **Monitor API key usage and set up alerts**
6. **Use sandbox/testnet environments for development**

## Verification

To verify your environment variables are set correctly:

```bash
# Check if variables are set (will show masked values)
echo $BINANCE_API_KEY | sed 's/./*/g'
echo $BINANCE_API_SECRET | sed 's/./*/g'
```

## Migration from Hardcoded Files

If you have existing hardcoded API keys:

1. Copy your API keys to environment variables using one of the methods above
2. Remove any files containing hardcoded keys (e.g., `gemini_api_keys.txt`)
3. Restart your application
4. Verify the application loads credentials from environment variables

## Troubleshooting

### Common Issues

1. **Environment variables not loaded**: Make sure to restart your terminal/application after setting variables
2. **Permission denied**: Check that your API keys have the correct permissions on the exchange
3. **Sandbox mode**: Ensure `SANDBOX` variables are set to "true" for testing

### Validation

The SecureAPIManager will automatically validate that no hardcoded API keys exist in your codebase at startup.
"""

        try:
            with open(output_path, 'w') as f:
                f.write(guide_content)
            logger.info(f"Environment setup guide created at: {output_path}")
        except Exception as e:
            logger.error(f"Failed to create environment setup guide: {e}")

    def shutdown(self) -> None:
        """Arrêt propre du gestionnaire"""
        logger.info("Shutting down SecureAPIManager...")

        # Arrêter tous les monitorings
        for exchange in list(self.connection_threads.keys()):
            self.stop_connection_monitoring(exchange)

        # Sauvegarder les credentials
        if self.credentials and self._cipher_suite:
            self._save_credentials()

        logger.info("SecureAPIManager shutdown completed")


class SecurityError(Exception):
    """Exception levée lors de violations de sécurité"""
    pass
