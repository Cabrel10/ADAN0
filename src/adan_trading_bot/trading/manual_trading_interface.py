"""
Interface de trading manuel pour ADAN Trading Bot.
Implémente les tâches 10B.2.3, 10B.2.4.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import uuid
from decimal import Decimal, ROUND_DOWN
import yaml

from .secure_api_manager import SecureAPIManager, ExchangeType, APICredentials

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Types d'ordres"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Côtés d'ordre"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """États des ordres"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class RiskOverrideType(Enum):
    """Types d'override de risque"""
    FORCE_DEFENSIVE = "force_defensive"
    FORCE_AGGRESSIVE = "force_aggressive"
    DISABLE_DBE = "disable_dbe"
    CUSTOM_PARAMS = "custom_params"


@dataclass
class ManualOrder:
    """Ordre manuel"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancelled

    # Métadonnées
    exchange: Optional[ExchangeType] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    # Exécution
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    commission: float = 0.0
    commission_asset: Optional[str] = None

    # Tracking
    exchange_order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.client_order_id is None:
            self.client_order_id = f"ADAN_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        data = asdict(self)
        data['side'] = self.side.value
        data['order_type'] = self.order_type.value
        data['status'] = self.status.value
        if self.exchange:
            data['exchange'] = self.exchange.value
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.submitted_at:
            data['submitted_at'] = self.submitted_at.isoformat()
        if self.filled_at:
            data['filled_at'] = self.filled_at.isoformat()
        return data


@dataclass
class RiskOverride:
    """Override de risque"""
    override_id: str
    override_type: RiskOverrideType
    parameters: Dict[str, Any]
    reason: str
    created_by: str = "manual"
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    active: bool = True

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.expires_at is None and self.override_type != RiskOverrideType.CUSTOM_PARAMS:
            # Override temporaire par défaut (1 heure)
            self.expires_at = datetime.now() + timedelta(hours=1)

    def is_expired(self) -> bool:
        """Vérifie si l'override a expiré"""
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        data = asdict(self)
        data['override_type'] = self.override_type.value
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        return data


class ManualTradingInterface:
    """Interface de trading manuel"""

    def __init__(self, api_manager: SecureAPIManager):
        self.api_manager = api_manager

        # Stockage des ordres
        self.orders: Dict[str, ManualOrder] = {}
        self.order_history: List[ManualOrder] = []

        # Gestion des overrides de risque
        self.risk_overrides: Dict[str, RiskOverride] = {}
        self.override_history: List[RiskOverride] = []

        # Callbacks
        self.order_callbacks: List[Callable] = []
        self.risk_override_callbacks: List[Callable] = []

        # Configuration
        self.default_exchange = ExchangeType.BINANCE
        self.confirmation_required = True
        self.max_order_value_usd = 10000  # Limite de sécurité
        self.default_order_type = OrderType.MARKET # New attribute

        # Trading parameters from config.yaml
        self.config = self._load_config()
        self.stop_loss_pct = self.config['trading_rules']['stop_loss_pct']
        self.take_profit_pct = self.config['trading_rules']['take_profit_pct']
        self.trailing_stop = self.config['trading_rules']['trailing_stop']
        self.commission_pct = self.config['trading_rules']['commission_pct']
        self.min_order_value_usdt = self.config['trading_rules']['min_order_value_usdt']
        self.slippage_pct = self.config['trading_rules']['slippage_pct']
        self.futures_enabled = self.config['trading_rules']['futures_enabled']
        self.leverage = self.config['trading_rules']['leverage']

        # Strategy weights
        self._strategy_weights: Dict[str, float] = {'w1': 2.5, 'w2': 2.5, 'w3': 2.5, 'w4': 2.5}

        # Trading pairs
        self._selected_trading_pairs: List[str] = []

        # Threading
        self.order_monitor_thread = None
        self.stop_monitoring = False

        logger.info("ManualTradingInterface initialized")

    def _load_config(self) -> Dict[str, Any]:
        config_path = "/home/morningstar/Documents/trading/bot/config/config.yaml"
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found at {config_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            return {}

    def set_trading_pairs(self, pairs: List[str]) -> None:
        """Définit les paires de trading sélectionnées"""
        self._selected_trading_pairs = [pair.upper() for pair in pairs]
        logger.info(f"Trading pairs set to: {', '.join(self._selected_trading_pairs)}")

    def get_trading_pairs(self) -> List[str]:
        """Récupère les paires de trading sélectionnées"""
        return self._selected_trading_pairs

    def set_default_exchange(self, exchange: ExchangeType) -> None:
        """Définit l'exchange par défaut"""
        self.default_exchange = exchange
        logger.info(f"Default exchange set to {exchange.value}")

    def set_stop_loss_pct(self, value: float) -> None:
        """Définit le pourcentage de stop loss"""
        self.stop_loss_pct = value
        logger.info(f"Stop Loss Percentage set to: {value}")

    def set_take_profit_pct(self, value: float) -> None:
        """Définit le pourcentage de take profit"""
        self.take_profit_pct = value
        logger.info(f"Take Profit Percentage set to: {value}")

    def set_trailing_stop(self, value: float) -> None:
        """Définit le pourcentage de trailing stop"""
        self.trailing_stop = value
        logger.info(f"Trailing Stop Percentage set to: {value}")

    def set_initial_balance(self, value: float) -> None:
        """Définit le capital initial"""
        self.config['portfolio']['initial_balance'] = value
        logger.info(f"Initial balance set to: {value}")

    def set_model_config(self, algorithm: str, policy: str) -> None:
        """Définit l'algorithme et la politique du modèle"""
        self.config['agent']['algorithm'] = algorithm
        self.config['agent']['policy'] = policy
        logger.info(f"Model config set to Algorithm: {algorithm}, Policy: {policy}")

    def set_strategy_weights(self, weights: Dict[str, float]) -> None:
        """Définit les poids des stratégies des workers"""
        self._strategy_weights = weights
        logger.info(f"Strategy weights set to: {weights}")

    def get_strategy_weights(self) -> Dict[str, float]:
        """Récupère les poids des stratégies des workers"""
        return self._strategy_weights

    def create_market_order(self, symbol: str, side: OrderSide, quantity: float,
                          exchange: Optional[ExchangeType] = None) -> str:
        """
        Crée un ordre au marché.

        Args:
            symbol: Symbole de trading (ex: BTCUSDT)
            side: Côté de l'ordre (BUY/SELL)
            quantity: Quantité
            exchange: Exchange à utiliser

        Returns:
            ID de l'ordre
        """
        order = ManualOrder(
            order_id=str(uuid.uuid4()),
            symbol=symbol.upper(),
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            exchange=exchange or self.default_exchange
        )

        return self._process_order(order)

    def create_limit_order(self, symbol: str, side: OrderSide, quantity: float, price: float,
                          exchange: Optional[ExchangeType] = None) -> str:
        """
        Crée un ordre limite.

        Args:
            symbol: Symbole de trading
            side: Côté de l'ordre
            quantity: Quantité
            price: Prix limite
            exchange: Exchange à utiliser

        Returns:
            ID de l'ordre
        """
        order = ManualOrder(
            order_id=str(uuid.uuid4()),
            symbol=symbol.upper(),
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            exchange=exchange or self.default_exchange
        )

        return self._process_order(order)

    def create_stop_loss_order(self, symbol: str, side: OrderSide, quantity: float,
                              stop_price: float, limit_price: Optional[float] = None,
                              exchange: Optional[ExchangeType] = None) -> str:
        """
        Crée un ordre stop-loss.

        Args:
            symbol: Symbole de trading
            side: Côté de l'ordre
            quantity: Quantité
            stop_price: Prix de déclenchement
            limit_price: Prix limite (si None, ordre stop-market)
            exchange: Exchange à utiliser

        Returns:
            ID de l'ordre
        """
        order_type = OrderType.STOP_LIMIT if limit_price else OrderType.STOP_LOSS

        order = ManualOrder(
            order_id=str(uuid.uuid4()),
            symbol=symbol.upper(),
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=limit_price,
            stop_price=stop_price,
            exchange=exchange or self.default_exchange
        )

        return self._process_order(order)

    def _process_order(self, order: ManualOrder) -> str:
        """Traite un ordre"""
        try:
            print(f"Processing order with type: {order.order_type}")
            # Validation de base
            if not self._validate_order(order):
                order.status = OrderStatus.REJECTED
                order.error_message = "Order validation failed"
                self.orders[order.order_id] = order
                return order.order_id

            # Vérification des credentials
            credentials = self.api_manager.get_credentials(order.exchange)
            if not credentials:
                order.status = OrderStatus.REJECTED
                order.error_message = f"No credentials for {order.exchange.value}"
                self.orders[order.order_id] = order
                return order.order_id

            # Confirmation si requise
            if self.confirmation_required:
                order.status = OrderStatus.PENDING
                self.orders[order.order_id] = order
                self._notify_order_update(order)
                logger.info(f"Order {order.order_id} pending confirmation")
                return order.order_id

            # Soumettre directement
            return self._submit_order(order)

        except Exception as e:
            logger.error(f"Error processing order: {e}")
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            self.orders[order.order_id] = order
            return order.order_id

    def _validate_order(self, order: ManualOrder) -> bool:
        """Valide un ordre contre les filtres de l'exchange"""
        try:
            exchange_info = self.api_manager.get_exchange_info(order.exchange)
            if not exchange_info:
                logger.error(f"Could not retrieve exchange info for {order.exchange.value}")
                return False

            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == order.symbol), None)
            if not symbol_info:
                logger.error(f"Could not find symbol info for {order.symbol}")
                return False

            # Convertir les valeurs de l'ordre en Decimal pour la précision
            quantity = Decimal(str(order.quantity))
            price = Decimal(str(order.price)) if order.price is not None else None

            for f in symbol_info['filters']:
                if f['filterType'] == 'PRICE_FILTER':
                    if price:
                        min_price = Decimal(f['minPrice'])
                        max_price = Decimal(f['maxPrice'])
                        tick_size = Decimal(f['tickSize'])
                        if price < min_price:
                            logger.error(f"Price {price} is below minPrice {min_price}")
                            return False
                        if max_price > 0 and price > max_price:
                            logger.error(f"Price {price} is above maxPrice {max_price}")
                            return False
                        if tick_size > 0 and (price - min_price) % tick_size != 0:
                            logger.error(f"Price {price} does not match tickSize {tick_size}")
                            return False

                elif f['filterType'] == 'LOT_SIZE':
                    min_qty = Decimal(f['minQty'])
                    max_qty = Decimal(f['maxQty'])
                    step_size = Decimal(f['stepSize'])
                    if quantity < min_qty:
                        logger.error(f"Quantity {quantity} is below minQty {min_qty}")
                        return False
                    if quantity > max_qty:
                        logger.error(f"Quantity {quantity} is above maxQty {max_qty}")
                        return False
                    if (quantity - min_qty) % step_size != 0:
                        logger.error(f"Quantity {quantity} does not match stepSize {step_size}")
                        return False

                elif f['filterType'] == 'MIN_NOTIONAL':
                    min_notional = Decimal(f['minNotional'])
                    if price and quantity * price < min_notional:
                        logger.error(f"Notional value {quantity * price} is below minNotional {min_notional}")
                        return False

            # Vérifications spécifiques au type d'ordre
            if order.order_type == OrderType.LIMIT and order.price is None:
                logger.error("Limit order requires price")
                return False

            if order.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LIMIT] and order.stop_price is None:
                logger.error("Stop order requires stop price")
                return False

            return True

        except Exception as e:
            logger.error(f"Order validation error: {e}")
            return False

    def confirm_order(self, order_id: str) -> bool:
        """Confirme un ordre en attente"""
        if order_id not in self.orders:
            logger.error(f"Order {order_id} not found")
            return False

        order = self.orders[order_id]
        if order.status != OrderStatus.PENDING:
            logger.error(f"Order {order_id} not pending")
            return False

        return self._submit_order(order) == order_id

    def _submit_order(self, order: ManualOrder) -> str:
        """Soumet un ordre à l'exchange"""
        try:
            order_params = {
                'symbol': order.symbol,
                'side': order.side.value,
                'type': order.order_type.value,
                'quantity': order.quantity,
                'newClientOrderId': order.client_order_id,
            }
            if order.price is not None:
                order_params['price'] = order.price
            if order.stop_price is not None:
                order_params['stopPrice'] = order.stop_price
            if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                order_params['timeInForce'] = order.time_in_force

            response = self.api_manager.send_order(order.exchange, order_params)

            if response and 'orderId' in response:
                order.status = OrderStatus.SUBMITTED
                order.submitted_at = datetime.now()
                order.exchange_order_id = response['orderId']
                self.orders[order.order_id] = order
                logger.info(f"Order {order.order_id} submitted successfully with exchange ID {order.exchange_order_id}")
                self._notify_order_update(order)

                if not self.order_monitor_thread or not self.order_monitor_thread.is_alive():
                    self._start_order_monitoring()
            else:
                order.status = OrderStatus.REJECTED
                order.error_message = response.get('msg') if response else "Submission failed"
                self.orders[order.order_id] = order # Add rejected order to the dictionary
                logger.error(f"Order {order.order_id} submission failed: {order.error_message}")

            return order.order_id

        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            self.orders[order.order_id] = order # Add rejected order to the dictionary
            return order.order_id

    def cancel_order(self, order_id: str) -> bool:
        """Annule un ordre"""
        if order_id not in self.orders:
            logger.error(f"Order {order_id} not found")
            return False

        order = self.orders[order_id]

        if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            logger.error(f"Cannot cancel order {order_id} with status {order.status.value}")
            return False

        try:
            if order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                response = self.api_manager.cancel_order(order.exchange, order.symbol, order.exchange_order_id)
                if not response or 'orderId' not in response:
                    logger.error(f"Failed to cancel order {order_id} on exchange: {response.get('msg') if response else 'Unknown error'}")
                    return False

            order.status = OrderStatus.CANCELLED
            self._notify_order_update(order)
            logger.info(f"Order {order_id} cancelled")
            return True

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def _start_order_monitoring(self) -> None:
        """Démarre le monitoring des ordres"""
        if self.order_monitor_thread and self.order_monitor_thread.is_alive():
            return

        self.stop_monitoring = False
        self.order_monitor_thread = threading.Thread(
            target=self._monitor_orders,
            daemon=True
        )
        self.order_monitor_thread.start()
        logger.info("Order monitoring started")

    def _monitor_orders(self) -> None:
        """Surveille les ordres actifs"""
        while not self.stop_monitoring:
            try:
                active_orders = [
                    order for order in self.orders.values()
                    if order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
                ]

                for order in active_orders:
                    self._check_order_status(order)

                time.sleep(1)  # Vérifier toutes les secondes

            except Exception as e:
                logger.error(f"Error in order monitoring: {e}")
                time.sleep(5)

    def _check_order_status(self, order: ManualOrder) -> None:
        """Vérifie le statut d'un ordre sur l'exchange"""
        try:
            if not order.exchange_order_id:
                return

            response = self.api_manager.get_order(order.exchange, order.symbol, order.exchange_order_id)

            if response and 'status' in response:
                new_status_str = response['status'].lower()
                if new_status_str not in [s.value for s in OrderStatus]:
                    logger.warning(f"Unknown order status '{new_status_str}' for order {order.order_id}")
                    return

                new_status = OrderStatus(new_status_str)

                if new_status != order.status:
                    order.status = new_status
                    order.filled_quantity = float(response.get('executedQty', 0))
                    if order.filled_quantity > 0:
                        order.average_price = float(response.get('cummulativeQuoteQty', 0)) / order.filled_quantity

                    if new_status == OrderStatus.FILLED:
                        order.filled_at = datetime.fromtimestamp(response['updateTime'] / 1000)

                    self._notify_order_update(order)
                    logger.info(f"Order {order.order_id} status updated to {new_status.value}")

            elif response:
                logger.warning(f"Could not update status for order {order.order_id}: {response.get('msg')}")

        except Exception as e:
            logger.error(f"Error checking order status for {order.order_id}: {e}")

    def get_order(self, order_id: str) -> Optional[ManualOrder]:
        """Récupère un ordre"""
        return self.orders.get(order_id)

    def get_active_orders(self) -> List[ManualOrder]:
        """Récupère les ordres actifs"""
        return [
            order for order in self.orders.values()
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
        ]

    def get_order_history(self, limit: int = 100) -> List[ManualOrder]:
        """Récupère l'historique des ordres"""
        all_orders = list(self.orders.values()) + self.order_history
        all_orders.sort(key=lambda x: x.created_at, reverse=True)
        return all_orders[:limit]

    # ==================== RISK OVERRIDE MANAGEMENT ====================

    def create_risk_override(self, override_type: RiskOverrideType, parameters: Dict[str, Any],
                           reason: str, duration_hours: Optional[int] = None) -> str:
        """
        Crée un override de risque.

        Args:
            override_type: Type d'override
            parameters: Paramètres spécifiques
            reason: Raison de l'override
            duration_hours: Durée en heures (None = permanent)

        Returns:
            ID de l'override
        """
        override_id = str(uuid.uuid4())

        expires_at = None
        if duration_hours:
            expires_at = datetime.now() + timedelta(hours=duration_hours)

        override = RiskOverride(
            override_id=override_id,
            override_type=override_type,
            parameters=parameters,
            reason=reason,
            expires_at=expires_at
        )

        self.risk_overrides[override_id] = override
        self._notify_risk_override_update(override)

        logger.info(f"Risk override created: {override_type.value} - {reason}")
        return override_id

    def force_defensive_mode(self, reason: str, duration_hours: int = 1) -> str:
        """Force le mode défensif"""
        return self.create_risk_override(
            RiskOverrideType.FORCE_DEFENSIVE,
            {'mode': 'defensive', 'risk_multiplier': 0.5},
            reason,
            duration_hours
        )

    def force_aggressive_mode(self, reason: str, duration_hours: int = 1) -> str:
        """Force le mode agressif"""
        return self.create_risk_override(
            RiskOverrideType.FORCE_AGGRESSIVE,
            {'mode': 'aggressive', 'risk_multiplier': 1.5},
            reason,
            duration_hours
        )

    def disable_dbe(self, reason: str, duration_hours: int = 1) -> str:
        """Désactive temporairement le DBE"""
        return self.create_risk_override(
            RiskOverrideType.DISABLE_DBE,
            {'dbe_enabled': False},
            reason,
            duration_hours
        )

    def set_custom_risk_params(self, params: Dict[str, Any], reason: str) -> str:
        """Définit des paramètres de risque personnalisés"""
        return self.create_risk_override(
            RiskOverrideType.CUSTOM_PARAMS,
            params,
            reason,
            None  # Permanent jusqu'à révocation manuelle
        )

    def revoke_risk_override(self, override_id: str) -> bool:
        """Révoque un override de risque"""
        if override_id not in self.risk_overrides:
            logger.error(f"Risk override {override_id} not found")
            return False

        override = self.risk_overrides[override_id]
        override.active = False

        # Déplacer vers l'historique
        self.override_history.append(override)
        del self.risk_overrides[override_id]

        self._notify_risk_override_update(override)

        logger.info(f"Risk override {override_id} revoked")
        return True

    def get_active_risk_overrides(self) -> List[RiskOverride]:
        """Récupère les overrides actifs"""
        # Nettoyer les overrides expirés
        expired_ids = []
        for override_id, override in self.risk_overrides.items():
            if override.is_expired():
                expired_ids.append(override_id)

        for override_id in expired_ids:
            self.revoke_risk_override(override_id)

        return list(self.risk_overrides.values())

    def get_risk_override_history(self, limit: int = 50) -> List[RiskOverride]:
        """Récupère l'historique des overrides"""
        all_overrides = list(self.risk_overrides.values()) + self.override_history
        all_overrides.sort(key=lambda x: x.created_at, reverse=True)
        return all_overrides[:limit]

    def start_live_trading(self) -> None:
        """Démarre la simulation de trading en direct"""
        logger.info("Starting live trading simulation...")
        logger.info(f"Current Trading Pairs: {self._selected_trading_pairs}")
        logger.info(f"Trading Parameters: SL={self.stop_loss_pct}, TP={self.take_profit_pct}, Trailing={self.trailing_stop}")
        logger.info(f"Model Algorithm: {self.config['agent']['algorithm']}")
        logger.info(f"Model Policy: {self.config['agent']['policy']}")
        logger.info(f"Initial Capital: {self.config['portfolio']['initial_balance']}")
        logger.info(f"Strategy Weights: {self._strategy_weights}")

        if not self._selected_trading_pairs:
            logger.warning("No trading pairs selected. Cannot start trading simulation.")
            return

        # Simulate some trading activity
        import random
        for i in range(3): # Simulate 3 trading actions
            symbol = random.choice(self._selected_trading_pairs)
            side = random.choice([OrderSide.BUY, OrderSide.SELL])
            quantity = round(random.uniform(0.001, 0.1), 4) # Random quantity

            logger.info(f"Simulating order {i+1}: {side.value} {quantity} of {symbol}")
            order_id = self.create_market_order(symbol, side, quantity)
            logger.info(f"Simulated order {order_id} created.")
            time.sleep(1) # Simulate some processing time

        logger.info("Live trading simulation finished.")

    # ==================== CALLBACKS ====================

    def add_order_callback(self, callback: Callable[[ManualOrder], None]) -> None:
        """Ajoute un callback pour les mises à jour d'ordres"""
        self.order_callbacks.append(callback)

    def add_risk_override_callback(self, callback: Callable[[RiskOverride], None]) -> None:
        """Ajoute un callback pour les overrides de risque"""
        self.risk_override_callbacks.append(callback)

    def _notify_order_update(self, order: ManualOrder) -> None:
        """Notifie les callbacks des mises à jour d'ordres"""
        for callback in self.order_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Error in order callback: {e}")

    def _notify_risk_override_update(self, override: RiskOverride) -> None:
        """Notifie les callbacks des overrides de risque"""
        for callback in self.risk_override_callbacks:
            try:
                callback(override)
            except Exception as e:
                logger.error(f"Error in risk override callback: {e}")

    def get_trading_summary(self) -> Dict[str, Any]:
        """Récupère un résumé de l'activité de trading"""
        active_orders = self.get_active_orders()
        recent_orders = [o for o in self.orders.values() if o.created_at > datetime.now() - timedelta(hours=24)]
        active_overrides = self.get_active_risk_overrides()

        return {
            'active_orders_count': len(active_orders),
            'recent_orders_count': len(recent_orders),
            'active_risk_overrides_count': len(active_overrides),
            'total_orders': len(self.orders),
            'order_success_rate': self._calculate_success_rate(),
            'active_overrides': [o.to_dict() for o in active_overrides]
        }

    def _calculate_success_rate(self) -> float:
        """Calcule le taux de succès des ordres"""
        if not self.orders:
            return 0.0

        completed_orders = [
            o for o in self.orders.values()
            if o.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]
        ]

        if not completed_orders:
            return 0.0

        successful_orders = [o for o in completed_orders if o.status == OrderStatus.FILLED]
        return len(successful_orders) / len(completed_orders)

    def shutdown(self) -> None:
        """Arrêt propre de l'interface"""
        logger.info("Shutting down ManualTradingInterface...")

        # Arrêter le monitoring
        self.stop_monitoring = True
        if self.order_monitor_thread and self.order_monitor_thread.is_alive():
            self.order_monitor_thread.join(timeout=5.0)

        # Sauvegarder l'historique si nécessaire
        # (implémentation selon les besoins)

        logger.info("ManualTradingInterface shutdown completed")
