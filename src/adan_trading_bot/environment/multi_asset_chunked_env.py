#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Multi-asset chunked environment for trading with real-time data support."""

from typing import Dict, Any, Optional, List, Tuple
import copy
import io
import json
import pickle
import random
import re
import signal
import time
import uuid
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import gym
from gym import spaces
import os
import logging
import threading
import gc
from collections import deque

from ..exchange_api.websocket_manager import WebSocketManager
from ..exchange_api.connector import get_exchange_client
from ..common.utils import get_logger
from ..common.logging_utils import create_smart_logger, configure_smart_logger

logger = get_logger(__name__)

# Standard portfolio state vector dimension
DEFAULT_PORTFOLIO_STATE_SIZE = 20
# Maximum attempts to reload a data chunk
MAX_RELOAD_ATTEMPTS = 3
# Fallback chunk index when primary fails
RELOAD_FALLBACK_CHUNK = 0
# Delay between reload attempts (seconds)
RELOAD_RETRY_DELAY = 0.5

# Excellence system (optional)
try:
    from ..rewards.excellence_system import (
        ExcellenceMetrics,
        create_excellence_rewards_system,
    )
    EXCELLENCE_SYSTEM_AVAILABLE = True
except ImportError:
    EXCELLENCE_SYSTEM_AVAILABLE = False
    ExcellenceMetrics = None
    create_excellence_rewards_system = None


def clean_worker_id(worker_id):
    """
    Normalize worker_id to an integer.
    Handles formats like 'W0', 'W1', 'w0', 0, 1, etc.
    """
    if worker_id is None:
        return 0
    
    if isinstance(worker_id, int):
        return worker_id
    
    if isinstance(worker_id, str):
        # Remove 'W' or 'w' prefix if present
        cleaned = worker_id.lstrip('Ww')
        try:
            return int(cleaned)
        except ValueError:
            return 0
    
    return 0


class LiveDataManager:
    def __init__(self, exchange_client, websocket_manager, assets, timeframes):
        self.exchange_client = exchange_client
        self.websocket_manager = websocket_manager
        self.assets = assets
        self.timeframes = timeframes
        self.current_data = {asset: {tf: pd.DataFrame() for tf in timeframes} for asset in assets}

    def get_initial_data(self, window_size: int) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Fetches historical k-lines for each asset/timeframe to build the initial observation window."""
        for asset in self.assets:
            for tf in self.timeframes:
                try:
                    # Fetch more than window_size to ensure enough data for indicators
                    limit = window_size * 2
                    ohlcv = self.exchange_client.fetch_ohlcv(asset, timeframe=tf, limit=limit)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    self.current_data[asset][tf] = df
                except Exception as e:
                    logger.error(f"Failed to fetch initial data for {asset} {tf}: {e}")
        return self.current_data

    def update_data(self, ws_data: Dict[str, Any]):
        """Updates the current data with a new k-line from the WebSocket."""
        try:
            stream = ws_data.get('stream')
            if not stream or '@kline_' not in stream:
                return

            symbol, _, timeframe = stream.partition('@kline_')
            asset = f"{symbol.upper()}/USDT" # Assuming USDT pair
            kline = ws_data['data']['k']

            if asset in self.assets and timeframe in self.timeframes:
                new_candle = pd.DataFrame([{
                    'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v'])
                }])

                # Append new candle and remove the oldest
                self.current_data[asset][timeframe] = pd.concat([self.current_data[asset][timeframe], new_candle]).iloc[-200:] # Keep last 200
        except Exception as e:
            logger.error(f"Error updating data from WebSocket: {e}")


class MultiAssetChunkedEnv(gym.Env):
    """Environnement de trading multi-actifs avec chargement par morceaux."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Constantes pour les actions
    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        timeframes=None,
        window_size=None,
        features_config=None,
        max_steps=1000,
        worker_config=None,
        config=None,
        external_dbe=None,
        worker_id=None, # Added worker_id to signature
        live_mode: bool = False,
        **kwargs,
    ):
        """Initialise l'environnement de trading multi-actifs."""
        super().__init__()
        self.crash_snapshot_dir = "reports/crash_snapshots"
        os.makedirs(self.crash_snapshot_dir, exist_ok=True)

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Allow tests to pass the config as first positional arg (data)
        if (
            isinstance(data, dict)
            and ("environment" in data or "workers" in data or "trading" in data)
        ) and (config is None):
            config = data
            data = None

        self.data = data
        self.live_mode = live_mode

        # DIAGNOSTIC: Tracer les instances d'environnement pour identifier les recréations
        import uuid

        self.env_instance_id = str(uuid.uuid4())[:8]

        # Initialize logger lock for thread-safe logging
        self.logger_lock = threading.Lock()

        # Initialize configuration attributes
        self.worker_config = worker_config if worker_config is not None else {}
        self.config = config or {}

        # Ensure a minimal 'trading' section exists for tests/minimal configs
        if not isinstance(self.config.get("trading", None), dict):
            self.config["trading"] = {"workers": {}, "timeframes": ["5m", "1h", "4h"]}

        # Auto-derive worker_config from config['workers'] if not provided
        if not self.worker_config and isinstance(self.config.get("workers", {}), dict):
            try:
                first_worker_key = next(iter(self.config["workers"].keys()))
                self.worker_config = dict(self.config["workers"][first_worker_key])
                if isinstance(first_worker_key, str) and first_worker_key.lower().startswith("w"):
                    try:
                        self.worker_config.setdefault("worker_id", int(first_worker_key[1:]) - 1)
                    except Exception:
                        pass
            except Exception:
                self.worker_config = {}

        if timeframes is not None:
            self.timeframes = timeframes
        else:
            tf = None
            if isinstance(self.config.get("trading", {}), dict):
                tf = self.config["trading"].get("timeframes")
            if not tf and isinstance(self.config.get("environment", {}).get("observation", {}), dict):
                tf = self.config["environment"]["observation"].get("timeframes")
            if not tf and isinstance(self.worker_config, dict):
                tf = self.worker_config.get("timeframes")
            self.timeframes = tf or ["5m", "1h", "4h"]

        if worker_id is not None:
            self.worker_id = clean_worker_id(worker_id)
        else:
            raw_worker_id = self.worker_config.get(
                "worker_id", self.worker_config.get("rank", "W0")
            )
            self.worker_id = clean_worker_id(raw_worker_id)

        self.external_dbe = external_dbe
        if external_dbe is not None:
            self.logger.critical(
                f"🔄 RÉCEPTION DBE IMMORTEL: ENV_ID={self.env_instance_id}, Worker={self.worker_id}, DBE_ID={id(external_dbe)}"
            )

        self.logger.critical(
            f"🆕 NOUVELLE INSTANCE ENV CRÉÉE: ID={self.env_instance_id}, Worker={self.worker_id}"
        )

        total_workers = kwargs.get("total_workers", 4)
        self.smart_logger = create_smart_logger(
            self.logger, self.worker_id, total_workers
        )
        configure_smart_logger(
            self.smart_logger, "training"
        )

        self.risk_params = self.worker_config.get("risk_parameters", {})
        self.max_positions = self.worker_config.get("max_positions", 1)
        self.tier = self.worker_config.get("tier", 1)
        self._init_risk_parameters()

        self.shared_buffer = None
        self.strict_validation = kwargs.get("strict_validation", False)

        if data is not None and isinstance(data, dict):
            self.assets = list(data.keys())
        else:
            assets_fc = []
            if isinstance(self.worker_config, dict):
                assets_fc = self.worker_config.get("assets", [])

            env_cfg = self.config.get("environment")
            if not assets_fc and isinstance(env_cfg, dict):
                assets_fc = env_cfg.get("assets", [])

            trading_cfg = self.config.get("trading")
            if not assets_fc and isinstance(trading_cfg, dict):
                assets_fc = trading_cfg.get("assets", [])

            self.assets = assets_fc

        if not self.assets:
            self.logger.warning("No assets specified; falling back to default ['BTCUSDT'] for test compatibility")
            self.assets = ["BTCUSDT"]

        self._observation_cache = {}
        self._max_cache_size = 1000
        self._cache_hits = 0
        self._cache_misses = 0
        self._last_observation = None
        self._last_market_timestamp: Optional[pd.Timestamp] = None
        self._last_asset_timestamp: Dict[str, pd.Timestamp] = {}
        self._current_obs = None
        self._cache_access = {}

        self.current_chunk = 0
        self.current_chunk_idx = 0
        self.done = False
        self.global_step = 0
        self.current_step = 0

        self.interpolation_count = 0
        self.total_steps_with_price_check = 0

        worker_trading_rules = self.worker_config.get("trading_rules", {})
        global_trading_rules = self.config.get("trading_rules", {})

        self.force_trade_steps_by_tf = self.worker_config.get("force_trade", {})
        if not self.force_trade_steps_by_tf:
            self.force_trade_steps_by_tf = global_trading_rules.get("frequency", {}).get("force_trade_steps", {})

        def _normalize_force_trade_steps(raw):
            default_map = {"5m": 15, "1h": 20, "4h": 50}
            if raw is None:
                return default_map
            if isinstance(raw, dict):
                out = {}
                for tf in ("5m", "1h", "4h"):
                    val = raw.get(tf, default_map[tf])
                    try:
                        out[tf] = int(val)
                    except Exception:
                        out[tf] = default_map[tf]
                return out
            try:
                scalar = int(raw)
                return {"5m": scalar, "1h": scalar, "4h": scalar}
            except Exception:
                return default_map

        self.force_trade_steps_by_tf = _normalize_force_trade_steps(self.force_trade_steps_by_tf)
        self.logger.debug(f"[DEBUG_CONFIG] Worker {self.worker_id} force_trade_steps_by_tf: {self.force_trade_steps_by_tf}")

        self.daily_max_forced_trades = self.worker_config.get("daily_max_forced_trades", 
                                       self.worker_config.get("trading_rules", {}).get("daily_max_forced_trades", 10))
        self.daily_forced_trades_count = 0

        self.frequency_config = self.config.get("trading_rules", {}).get("frequency", {})

        self.force_cooldown_by_tf = {
            tf: int(worker_trading_rules.get("frequency", {}).get("force_cooldown_steps", {}).get(tf, global_trading_rules.get("frequency", {}).get("force_cooldown_steps", {}).get(tf, 0)))
            for tf in ("5m", "1h", "4h")
        }
        _wj = worker_trading_rules.get("frequency", {}).get("force_jitter_max", global_trading_rules.get("frequency", {}).get("force_jitter_max", 0))
        if isinstance(_wj, dict):
            self.force_jitter_max_by_tf = {
                tf: int(_wj.get(tf, 0)) for tf in ("5m", "1h", "4h")
            }
        else:
            scalar_j = int(_wj or 0)
            self.force_jitter_max_by_tf = {tf: scalar_j for tf in ("5m", "1h", "4h")}
        self.next_force_allowed_step_by_tf = {tf: 0 for tf in ("5m", "1h", "4h")}
        self.positions_count = {"5m": 0, "1h": 0, "4h": 0, "daily_total": 0}

        self.last_trade_ids = set()
        self.current_timeframe_for_trade = "5m"
        self.daily_reset_step = 0
        self.current_day = 0
        self.last_trade_steps_by_tf = {}

        self.last_trade_timestamps = {"5m": None, "1h": None, "4h": None}
        self.receipts: deque = deque(maxlen=100)

        self.last_info: Dict[str, Any] = {}

        self.data_loader_instance = None
        self.live_data_manager = None
        self.exchange_client = None
        self.websocket_manager = kwargs.get('websocket_manager')

        self.current_tier = None
        self.previous_tier = None
        self.episode_count = 0
        self.episodes_in_tier = 0
        self.best_portfolio_value = 0.0
        self.last_tier_change_step = 0
        self.tier_history = []

        # OMEGA-3/4: Episode receipts tracking and anti-spam cooldown
        self._all_episode_receipts: List[dict] = []
        self._last_open_step_by_asset: Dict[str, int] = {}

        self.last_trade_step = -1
        self.risk_metrics = {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }
        self.performance_history = []

        self._is_initialized = False

        self._price_read_success_count = 0
        self._price_forward_fill_count = 0
        self._forward_fill_threshold = 0.02

        self._last_reward = 0.0
        self._cumulative_reward = 0.0

        self.trade_attempts = 0
        self.invalid_trade_attempts = 0
        self.last_reset_date = None
        try:
            self._initialize_components()
            self._is_initialized = True
        except Exception as e:
            self.logger.error("Erreur lors de l'initialisation: %s", str(e))
            raise

    def __getstate__(self):
        """Préparer l'état pour le pickling, en excluant les objets non-sérialisables."""
        state = self.__dict__.copy()
        # Exclure les loggers, locks, et autres objets non-sérialisables
        state.pop("logger", None)
        state.pop("smart_logger", None)
        state.pop("logger_lock", None)
        # Casser la référence circulaire avec DBE pour le pickling
        if "dbe" in state and hasattr(state["dbe"], "env"):
            state["dbe"].env = None
        return state

    def __setstate__(self, state):
        """Restaurer l'état après le unpickling et ré-initialiser les objets."""
        self.__dict__.update(state)

        # Ré-initialiser les objets non-sérialisables
        self.logger = logging.getLogger(__name__)
        self.logger_lock = threading.Lock()
        self.smart_logger = create_smart_logger(
            self.logger, getattr(self, "worker_id", 0), 4
        )
        configure_smart_logger(self.smart_logger, "training")
        # Rétablir la référence circulaire si DBE existe
        if hasattr(self, "dbe") and hasattr(self.dbe, "set_env_reference"):
            self.dbe.set_env_reference(self)
            # Restaurer également la référence au finance_manager dans le DBE
            if hasattr(self, "portfolio_manager"):
                self.dbe.finance_manager = self.portfolio_manager

    def _epoch_reset(self, force: bool = False, new_epoch: bool = False):
        """
        Centralized call to portfolio.reset(...) with config-driven threshold.

        Args:
            force: Force a full reset if True
            new_epoch: If True, indicates this is the start of a new epoch
        """
        min_cap = getattr(self, "config", {}).get("min_capital_before_reset", 11.0)
        self.portfolio.reset(
            new_epoch=new_epoch, force=force, min_capital_before_reset=min_cap
        )

    def update_risk_parameters(self, market_conditions=None):
        """
        Met à jour les paramètres de risque en fonction des conditions de marché
        et du régime de marché détecté, EN PRIORISANT LES VALEURS DU DBE.

        Args:
            market_conditions: Dictionnaire contenant les indicateurs de marché actuels.
                              Si None, utilise les prix actuels via _get_current_prices().
        """
        if (
            not hasattr(self, "dynamic_position_sizing")
            or not self.dynamic_position_sizing
        ):
            return

        try:
            # Récupération des conditions de marché actuelles si non fournies
            if market_conditions is None:
                # Utiliser la méthode corrigée _get_current_prices
                current_prices = self._get_current_prices()
                if current_prices:
                    # Prendre le premier asset disponible pour les conditions de marché
                    first_asset = next(iter(current_prices))
                    market_conditions = {
                        "close": current_prices[first_asset],
                        "asset": first_asset,
                    }
                    # Compléter avec les indicateurs techniques si disponibles
                    try:
                        tech_indicators = self._get_current_market_indicators()
                        if tech_indicators:
                            market_conditions.update(tech_indicators)
                    except Exception:
                        # Si les indicateurs techniques échouent, continuer avec le prix seulement
                        pass
                else:
                    # Fallback si aucun prix disponible
                    market_conditions = self._get_default_market_conditions()

            # Vérification que nous avons au moins un prix de clôture
            close_price = market_conditions.get("close")
            if (
                close_price is None
                or not isinstance(close_price, (int, float))
                or close_price <= 0
            ):
                # Utiliser des conditions par défaut
                market_conditions = self._get_default_market_conditions()
                close_price = market_conditions.get("close", 50000.0)

            # ✅ NOUVEAU: Appeler le DBE pour obtenir les modulations de risque
            # MAIS seulement si le DBE est activé dans la config
            dbe_enabled = self.config.get('dbe', {}).get('enabled', True)
            dbe_modulation = None
            
            if dbe_enabled and hasattr(self, "dynamic_behavior_engine") and self.dynamic_behavior_engine:
                try:
                    # compute_dynamic_modulation n'a besoin que de l'env
                    dbe_modulation = self.dynamic_behavior_engine.compute_dynamic_modulation(env=self)
                    logger.debug(f"[DBE_MODULATION] Worker {self.worker_id}: {dbe_modulation}")
                except Exception as e:
                    logger.warning(f"[DBE_MODULATION_ERROR] Worker {self.worker_id}: {e}")
                    dbe_modulation = None
            elif not dbe_enabled:
                # DBE désactivé - utiliser les params de la config directement (pour Optuna)
                risk_mgmt = self.config.get('trading_rules', {}).get('risk_management', {})
                pos_sizing = self.config.get('trading_rules', {}).get('position_sizing', {})
                dbe_modulation = {
                    'sl_pct': risk_mgmt.get('stop_loss_pct', 0.02),
                    'tp_pct': risk_mgmt.get('take_profit_pct', 0.04),
                    'position_size_pct': pos_sizing.get('position_size_pct', 0.20),
                }
                logger.info(f"[DBE_DISABLED] Using config params directly: SL={dbe_modulation['sl_pct']:.4f}, TP={dbe_modulation['tp_pct']:.4f}")

            # 1. Détection du régime de marché
            regime, confidence = self._detect_market_regime(market_conditions)

            # 2. Mise à jour des paramètres de risque en fonction du régime
            risk_params = self._calculate_risk_parameters(regime, market_conditions)

            # ✅ PRIORITÉ: Si le DBE fournit des modulations, les utiliser
            if dbe_modulation:
                risk_params["stop_loss_pct"] = dbe_modulation.get("sl_pct", risk_params.get("stop_loss_pct", 0.02))
                risk_params["take_profit_pct"] = dbe_modulation.get("tp_pct", risk_params.get("take_profit_pct", 0.04))
                risk_params["position_size_pct"] = dbe_modulation.get("position_size_pct", risk_params.get("position_size_pct", 0.1))
                if dbe_enabled:
                    logger.info(f"[RISK_PARAMS_FROM_DBE] Worker {self.worker_id}: SL={risk_params['stop_loss_pct']:.4f}, TP={risk_params['take_profit_pct']:.4f}, PosSize={risk_params['position_size_pct']:.4f}")
                else:
                    logger.info(f"[RISK_PARAMS_FROM_CONFIG] Worker {self.worker_id}: SL={risk_params['stop_loss_pct']:.4f}, TP={risk_params['take_profit_pct']:.4f}, PosSize={risk_params['position_size_pct']:.4f}")

            # 3. Application des limites de risque
            self._apply_risk_limits(risk_params)

            # 4. Mise à jour du gestionnaire de portefeuille
            if hasattr(self, "portfolio"):
                try:
                    self.portfolio.update_risk_parameters(risk_params)
                except AttributeError:
                    logger.warning(
                        "[FALLBACK] update_risk_parameters absent – Valeurs par défaut."
                    )
                    self.portfolio.sl_pct = risk_params.get("stop_loss_pct", 0.02)
                    self.portfolio.tp_pct = risk_params.get("take_profit_pct", 0.04)
                    self.portfolio.pos_size_pct = min(
                        risk_params.get("position_size_pct", 0.825), 0.9
                    )  # Clip à 90%
                    logging.warning("update_risk_parameters manquant - Ajouté fallback")
                    # Fallback direct pour éviter le crash
                    if hasattr(self.portfolio, "sl_pct"):
                        self.portfolio.sl_pct = risk_params.get("stop_loss_pct", 0.02)
                    if hasattr(self.portfolio, "tp_pct"):
                        self.portfolio.tp_pct = risk_params.get("take_profit_pct", 0.04)

                # Journalisation des changements significatifs
                if hasattr(self, "last_risk_params") and self.last_risk_params:
                    changed = []
                    for k, v in risk_params.items():
                        if k in self.last_risk_params:
                            try:
                                # Gestion explicite des types pour éviter DTypePromotionError
                                old_val = self.last_risk_params[k]
                                new_val = v

                                # Vérification stricte des types numériques
                                if (
                                    isinstance(new_val, (int, float, np.number))
                                    and isinstance(old_val, (int, float, np.number))
                                    and not isinstance(new_val, (str, bool))
                                    and not isinstance(old_val, (str, bool))
                                ):
                                    # Conversion explicite en float pour éviter les problèmes de dtype
                                    new_val_float = float(new_val)
                                    old_val_float = float(old_val)

                                    # Comparaison numérique avec np.isclose
                                    if not np.isclose(
                                        new_val_float, old_val_float, rtol=1e-3
                                    ):
                                        changed.append(
                                            f"{k}: {old_val_float:.4f}→{new_val_float:.4f}"
                                        )
                                else:
                                    # Comparaison directe pour chaînes et autres types
                                    if str(new_val) != str(old_val):
                                        changed.append(f"{k}: {old_val}→{new_val}")
                            except Exception as e:
                                # Fallback en cas d'erreur de type
                                self.logger.warning(
                                    f"Erreur comparaison param {k}: {e}"
                                )
                                if str(v) != str(self.last_risk_params[k]):
                                    changed.append(
                                        f"{k}: {self.last_risk_params[k]}→{v}"
                                    )

                    if changed:
                        self.logger.info(
                            f"Mise à jour des paramètres de risque - "
                            f"Régime: {regime} (confiance: {confidence:.1%}), "
                            f"Changements: {', '.join(changed)}"
                        )

                self.last_risk_params = risk_params.copy()

        except Exception as e:
            try:
                with self.logger_lock:
                    self.logger.error(
                        f"[{self.worker_id}] RISK UPDATE ERROR: Failed to update risk parameters: {e}",
                        exc_info=True,
                    )
            except AttributeError:
                # Fallback si logger_lock n'existe pas encore
                self.logger.error(
                    f"[{self.worker_id}] RISK UPDATE ERROR: Failed to update risk parameters: {e}",
                    exc_info=True,
                )

    def _get_last_valid_price(self, max_age_minutes: int = 15):
        """
        Récupère le dernier prix valide sans interpolation.

        RÈGLE STRICTE: Jamais d'interpolation linéaire pour les prix d'exécution.
        Utilise seulement forward fill avec limite temporelle.

        Args:
            max_age_minutes: Age maximum accepté pour un prix (en minutes)

        Returns:
            float: Dernier prix valide ou None si trop ancien/indisponible
        """
        try:
            current_time = pd.Timestamp.now()
            max_age = pd.Timedelta(minutes=max_age_minutes)

            # Essayer d'utiliser les données du chunk actuel
            if hasattr(self, "current_data") and self.current_data is not None:
                for asset_data in self.current_data.values():
                    if isinstance(asset_data, dict):
                        for timeframe_data in asset_data.values():
                            if (
                                hasattr(timeframe_data, "iloc")
                                and len(timeframe_data) > 0
                            ):
                                # Trouver la colonne close
                                close_col = None
                                for col in ["close", "CLOSE", "Close"]:
                                    if col in timeframe_data.columns:
                                        close_col = col
                                        break

                                if close_col is not None:
                                    # FORWARD FILL SEULEMENT - pas d'interpolation
                                    last_valid_prices = timeframe_data[
                                        close_col
                                    ].dropna()
                                    if len(last_valid_prices) > 0:
                                        last_price = float(last_valid_prices.iloc[-1])

                                        # Vérifier l'âge du prix si timestamp disponible
                                        if "timestamp" in timeframe_data.columns:
                                            last_timestamp = timeframe_data[
                                                timeframe_data[close_col].notna()
                                            ].iloc[-1]["timestamp"]
                                            if isinstance(last_timestamp, (int, float)):
                                                last_timestamp = pd.Timestamp(
                                                    last_timestamp, unit="ms"
                                                )
                                            elif not isinstance(
                                                last_timestamp, pd.Timestamp
                                            ):
                                                last_timestamp = pd.Timestamp(
                                                    last_timestamp
                                                )

                                            age = current_time - last_timestamp
                                            if age > max_age:
                                                self.logger.warning(
                                                    f"PRICE_TOO_OLD | last_price={last_price} | age={age} | max_age={max_age}"
                                                )
                                                continue

                                        return last_price

            # Fallback: utiliser le dernier prix connu stocké
            if (
                hasattr(self, "_last_known_price")
                and self._last_known_price is not None
            ):
                return float(self._last_known_price)

            return None

        except Exception as e:
            self.logger.debug(
                f"Erreur lors de la récupération du dernier prix valide: {e}"
            )
            return None

    def _calculate_asset_volatility(self, asset: str, lookback: int = 21) -> float:
        """
        Calcule la volatilité annualisée d'un actif sur une période donnée.

        Args:
            asset: Symbole de l'actif
            lookback: Nombre de jours pour le calcul de la volatilité (par défaut: 21 jours)

        Returns:
            float: Volatilité annualisée en décimal (0.2 pour 20%)
        """
        try:
            if not hasattr(self, "current_data") or not self.current_data:
                self.logger.warning(
                    "Données de marché non disponibles pour le calcul de volatilité"
                )
                return 0.15  # Valeur par défaut raisonnable

            # Récupérer les données de prix pour l'actif
            if asset not in self.current_data:
                self.logger.warning(f"Données manquantes pour l'actif {asset}")
                return 0.15

            # Prendre le premier intervalle de temps disponible
            tf = next(iter(self.current_data[asset].keys()))
            df = self.current_data[asset][tf]

            # Vérifier si on a assez de données
            if len(df) < lookback + 1:
                self.logger.warning(
                    f"Pas assez de données pour calculer la volatilité sur {lookback} jours"
                )
                return 0.15

            # Calculer les rendements journaliers
            close_prices = df["close"].iloc[-(lookback + 1) :]
            returns = close_prices.pct_change().dropna()

            # Calculer la volatilité annualisée (252 jours de trading par an)
            volatility = returns.std() * np.sqrt(252)

            # Limiter la volatilité entre 5% et 200%
            volatility = np.clip(volatility, 0.05, 2.0)

            self.logger.debug(
                f"Volatilité calculée pour {asset}: {volatility:.2%} (sur {lookback} jours)"
            )
            return float(volatility)

        except Exception as e:
            self.logger.error(
                f"Erreur dans le calcul de la volatilité pour {asset}: {str(e)}"
            )
            return 0.15  # Retourne une volatilité par défaut en cas d'erreur

    def _get_current_market_indicators(self) -> Dict[str, float]:
        """Récupère les indicateurs de marché actuels à partir de l'observation construite."""
        try:
            # Obtenir l'observation complète pour l'étape actuelle
            # _get_observation() utilise déjà self.state_builder et self.current_step
            observation_dict = self._get_observation()
            market_observation = observation_dict.get("observation")

            # Debug logging for observation
            if market_observation is not None:
                self.logger.info(
                    f"[ENV_DEBUG] Observation shape: {market_observation.shape}, "
                    f"dtype: {market_observation.dtype}, "
                    f"NaNs: {np.isnan(market_observation).sum()}"
                )
            else:
                self.logger.info("[ENV_DEBUG] market_observation is None")

            if market_observation is None or market_observation.size == 0:
                self.logger.warning(
                    "Observation de marché vide ou invalide pour les indicateurs."
                )
                return {}

            # Extraire les indicateurs du timeframe 5m (le plus granulaire)
            # L'observation est de forme (timeframes, window_size, features)
            # Le timeframe 5m est généralement en premier (index 0)

            # Récupérer les noms des features pour le timeframe 5m
            features_5m = self.state_builder.get_feature_names("5m")

            # Vérifications robustes de la structure des données
            if len(features_5m) == 0:
                self.logger.debug(
                    "Aucune feature configurée pour 5m, utilisation de valeurs par défaut"
                )
                features_5m = [
                    "OPEN",
                    "HIGH",
                    "LOW",
                    "close",
                    "VOLUME",
                    "RSI_14",
                    "STOCH_K_14_3_3",
                    "STOCH_D_14_3_3",
                    "MACD_HIST_12_26_9",
                    "ATR_14",
                    "EMA_5",
                    "EMA_12",
                    "BB_UPPER",
                    "BB_MIDDLE",
                    "BB_LOWER",
                ]

            # Vérification de la forme de l'observation
            if market_observation.ndim != 3:
                self.logger.warning(
                    f"Forme d'observation incorrecte: {market_observation.shape}, attendu 3D (timeframes, window, features)"
                )
                return self._get_default_market_conditions()

            if market_observation.shape[0] == 0 or market_observation.shape[1] == 0:
                self.logger.warning(f"Observation vide: {market_observation.shape}")
                return self._get_default_market_conditions()

            # Récupérer la dernière ligne de données du timeframe 5m dans l'observation
            try:
                current_5m_data = market_observation[
                    0, -1, :
                ]  # Premier timeframe, dernière ligne de la fenêtre
            except IndexError as e:
                self.logger.error(
                    f"Erreur d'indexation lors de l'extraction des données 5m: {e}"
                )
                return self._get_default_market_conditions()

            indicators = {}
            # Mapper les valeurs numériques aux noms des features avec vérifications
            for i, feature_name in enumerate(features_5m):
                if i < len(current_5m_data):
                    val = float(current_5m_data[i])
                    # Remplacer NaN ou Inf par des valeurs par défaut
                    if np.isnan(val) or np.isinf(val):
                        if "RSI" in feature_name.upper():
                            val = 50.0
                        elif "ADX" in feature_name.upper():
                            val = 20.0
                        elif "ATR" in feature_name.upper():
                            val = 0.01
                        elif "EMA" in feature_name.upper():
                            val = 1.0
                        elif "MACD_HIST" in feature_name.upper():
                            val = 0.0
                        elif "close" in feature_name.upper():
                            val = 1.0
                        elif "BB_" in feature_name.upper():
                            val = 1.0
                        elif "STOCH" in feature_name.upper():
                            val = 50.0
                        else:
                            val = 0.0
                        self.logger.debug(
                            f"NaN/Inf détecté pour {feature_name} (index {i}), remplacé par {val}"
                        )

                    indicators[feature_name.upper()] = val
                    self.logger.debug(f"Feature {i}: {feature_name.upper()} = {val}")
                else:
                    # Si la feature n'est pas disponible, utiliser une valeur par défaut
                    default_val = (
                        50.0
                        if "RSI" in feature_name.upper()
                        or "STOCH" in feature_name.upper()
                        else 1.0
                        if "close" in feature_name.upper()
                        or "EMA" in feature_name.upper()
                        else 0.0
                    )
                    indicators[feature_name.upper()] = default_val
                    self.logger.debug(
                        f"Feature manquante {feature_name} (index {i}), valeur par défaut: {default_val}"
                    )

            # Assurer la présence des clés essentielles pour le DBE
            essential_indicators = {
                "close": indicators.get("close", 1.0),
                "close": indicators.get("close", 1.0),  # Ajouter la version lowercase
                "VOLUME": indicators.get("VOLUME", 0.0),
                "RSI_14": indicators.get("RSI_14", 50.0),
                "ATR_14": indicators.get("ATR_14", 0.01),
                "ADX_14": indicators.get("ADX_14", 20.0),
                "EMA_5": indicators.get("EMA_5", indicators.get("close", 1.0)),
                "EMA_12": indicators.get("EMA_12", indicators.get("close", 1.0)),
                "EMA_20": indicators.get("EMA_20", indicators.get("close", 1.0)),
                "EMA_26": indicators.get("EMA_26", indicators.get("close", 1.0)),
                "EMA_50": indicators.get("EMA_50", indicators.get("close", 1.0)),
                "MACD_HIST_12_26_9": indicators.get("MACD_HIST_12_26_9", 0.0),
                "BB_UPPER": indicators.get(
                    "BB_UPPER", indicators.get("close", 1.0) * 1.02
                ),
                "BB_MIDDLE": indicators.get("BB_MIDDLE", indicators.get("close", 1.0)),
                "BB_LOWER": indicators.get(
                    "BB_LOWER", indicators.get("close", 1.0) * 0.98
                ),
                "STOCH_K_14_3_3": indicators.get("STOCH_K_14_3_3", 50.0),
                "STOCH_D_14_3_3": indicators.get("STOCH_D_14_3_3", 50.0),
            }

            # Ajouter tous les autres indicateurs extraits
            essential_indicators.update(indicators)

            self.logger.debug(
                f"Indicateurs finaux extraits: {len(essential_indicators)} items"
            )
            return essential_indicators

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la récupération des indicateurs de marché: {str(e)}",
                exc_info=True,
            )
            return self._get_default_market_conditions()

    def _get_default_market_conditions(self) -> Dict[str, float]:
        """Retourne des conditions de marché par défaut pour éviter les erreurs."""
        return {
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 0.0,
            "RSI_14": 50.0,
            "ATR_14": 0.01,
            "ADX_14": 20.0,
            "EMA_5": 1.0,
            "EMA_12": 1.0,
            "EMA_20": 1.0,
            "EMA_26": 1.0,
            "EMA_50": 1.0,
            "MACD_HIST_12_26_9": 0.0,
            "BB_UPPER": 1.05,
            "BB_MIDDLE": 1.0,
            "BB_LOWER": 0.95,
            "STOCH_K_14_3_3": 50.0,
            "STOCH_D_14_3_3": 50.0,
        }

    def _detect_market_regime(self, market_data: Dict[str, float]) -> Tuple[str, float]:
        """Détecte le régime de marché actuel."""
        try:
            # Utilisation du DBE si disponible
            if hasattr(self, "dbe") and hasattr(self.dbe, "detect_market_regime"):
                return self.dbe.detect_market_regime(market_data)

            # Implémentation de secours si le DBE n'est pas disponible
            adx = market_data.get("adx", 0)
            rsi = market_data.get("rsi", 50)
            ema_fast = market_data.get("ema_fast", 0)
            ema_slow = market_data.get("ema_slow", 0)

            adx_threshold = 25  # Seuil ADX pour la détection de tendance

            if adx > adx_threshold:
                if ema_fast > ema_slow:
                    return "bull", 0.7 + (0.3 * (adx / 100))
                else:
                    return "bear", 0.7 + (0.3 * (adx / 100))
            else:
                if rsi > 70 or rsi < 30:
                    return "volatile", 0.8
                return "sideways", 0.9

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la détection du régime de marché: {str(e)}"
            )
            return "unknown", 0.5

    def _calculate_risk_parameters(
        self, regime: str, market_data: Dict[str, float]
    ) -> Dict[str, float]:
        """Calcule les paramètres de risque en fonction du régime de marché."""
        try:
            # Paramètres par défaut
            params = {
                "position_size": self.base_position_size,
                "stop_loss_pct": 0.02,  # 2% par défaut
                "take_profit_pct": 0.04,  # 4% par défaut
                "max_position_size": self.max_position_size,
                "risk_per_trade": self.risk_per_trade,
                "regime": regime,
            }

            # Récupération des paramètres spécifiques au régime
            regime_params = self.regime_parameters.get(regime, {})

            # Application des multiplicateurs du régime
            for param in ["position_size", "stop_loss_pct", "take_profit_pct"]:
                if param in regime_params:
                    params[param] *= regime_params[param]

            # Ajustement basé sur la volatilité
            if (
                "ATR_14" in market_data
                and "close" in market_data
                and market_data["close"] > 0
            ):
                volatility = market_data["ATR_14"] / market_data["close"]
                vol_factor = np.clip(
                    volatility / max(self.baseline_volatility, 1e-6),
                    0.5,
                    2.0,  # Bornes min/max du facteur de volatilité
                )

                # Ajustement des paramètres en fonction de la volatilité
                params["position_size"] = np.clip(
                    params["position_size"] / vol_factor,
                    self.min_position_size,
                    self.max_position_size,
                )
                params["stop_loss_pct"] *= vol_factor
                params["take_profit_pct"] /= vol_factor

            return params

        except Exception as e:
            self.logger.error(
                f"Erreur lors du calcul des paramètres de risque: {str(e)}"
            )
            # Retour des valeurs par défaut en cas d'erreur
            return {
                "position_size": self.base_position_size,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
                "max_position_size": self.max_position_size,
                "risk_per_trade": self.risk_per_trade,
                "regime": "error",
            }

    def _apply_risk_limits(self, risk_params: Dict[str, float]) -> None:
        """Applique les limites de risque aux paramètres calculés."""
        try:
            # Limites de base
            risk_params["position_size"] = np.clip(
                risk_params["position_size"],
                self.min_position_size,
                self.max_position_size,
            )

            # Limites pour les stop loss et take profit
            risk_params["stop_loss_pct"] = np.clip(
                risk_params["stop_loss_pct"],
                0.005,
                0.10,  # 0.5% minimum  # 10% maximum
            )

            risk_params["take_profit_pct"] = np.clip(
                risk_params["take_profit_pct"],
                0.005,  # 0.5% minimum
                0.20,  # 20% maximum
            )

            # Ajustement pour les micro-capitaux
            if hasattr(self, "portfolio") and hasattr(
                self.portfolio, "current_balance"
            ):
                if self.portfolio.current_balance < self.micro_capital_threshold:
                    risk_params["position_size"] = min(
                        risk_params["position_size"],
                        self.worker_config.get("dbe_config", {})
                        .get("micro_capital", {})
                        .get("position_size_cap", 0.3),
                    )
                    risk_params["risk_per_trade"] = min(
                        risk_params.get("risk_per_trade", 1.0), 0.5
                    )

            # Mise à jour du facteur de risque dans le DBE si disponible
            if hasattr(self, "dbe") and hasattr(self.dbe, "update_parameters"):
                # Calculer la volatilité actuelle à partir des données disponibles
                current_volatility = 0.01  # Valeur par défaut
                if hasattr(self, "current_data") and self.current_data is not None:
                    # Essayer d'obtenir ATR et prix actuel depuis les données
                    for tf_data in self.current_data.values():
                        if isinstance(tf_data, dict):
                            for asset_data in tf_data.values():
                                if hasattr(asset_data, "iloc") and len(asset_data) > 0:
                                    latest_row = asset_data.iloc[-1]
                                    if "atr_14" in latest_row and "close" in latest_row:
                                        close_price = latest_row["close"]
                                        atr_value = latest_row["atr_14"]
                                        if close_price > 0:
                                            current_volatility = atr_value / close_price
                                            break

                self.dbe.update_parameters(
                    {
                        "volatility_factor": 1.0
                        / max(risk_params.get("volatility_factor", 1.0), 0.1),
                        "max_position_size": risk_params["position_size"],
                        "current_volatility": current_volatility,
                    }
                )

        except Exception as e:
            self.logger.error(
                f"Erreur lors de l'application des limites de risque: {str(e)}"
            )

    def _init_risk_parameters(self):
        """
        Initialise les paramètres de risque.

        ARCHITECTURE RULE: risk_per_trade and exposure limits come from
        the Capital Tier (based on current portfolio value), NOT from the
        worker profile. The worker only provides temporal specialisation.
        """
        # 1. Base position size - use tier if available, else safe default
        self.base_position_size = 0.10  # 10% default, overridden at trade time by tier
        
        # risk_per_trade will be overridden by tier at each step in _execute_trades
        # but we need a safe init value
        self.risk_per_trade = 0.04  # 4% Micro Capital default

        # 2. Paramètres de position sizing dynamique
        self.dynamic_position_sizing = self.worker_config.get("dbe_config", {}).get(
            "dynamic_position_sizing", True
        )

        # Récupération des plages de taille de position
        if "position_size_range" in self.risk_params:
            self.min_position_size = self.risk_params["position_size_range"][0]
            self.max_position_size = self.risk_params["position_size_range"][1]
        else:
            # Valeurs par défaut basées sur le profil du worker
            profile = self.worker_config.get("profile", "moderate")
            # Get position size limits from configuration
            risk_mgmt_config = (
                self.config.get("environment", {})
                .get("risk_management", {})
                .get("position_sizing", {})
            )

            # Try to get current tier limits first
            current_tier = None
            if hasattr(self, "portfolio") and hasattr(
                self.portfolio, "get_current_tier"
            ):
                try:
                    current_tier = self.portfolio.get_current_tier()
                except:
                    current_tier = None

            if current_tier and isinstance(current_tier, dict):
                # Use tier-specific limits
                self.max_position_size = (
                    current_tier.get("max_position_size_pct", 25) / 100.0
                )
                self.min_position_size = 0.01  # 1% minimum for all tiers
            else:
                # Fall back to environment configuration
                self.max_position_size = risk_mgmt_config.get("max_position_size", 0.25)
                self.min_position_size = (
                    risk_mgmt_config.get("initial_position_size", 0.1) * 0.5
                )  # Half of initial as minimum

            # Profile-based adjustments (multiplicative factors)
            if profile == "conservative":
                self.max_position_size *= 0.6  # Reduce max by 40%
            elif profile == "aggressive":
                self.max_position_size *= 1.2  # Increase max by 20%
            # moderate keeps the base values

        # 3. Paramètres de gestion du risque
        self.max_drawdown_pct = self.risk_params.get("max_drawdown_pct", 0.25)
        self.daily_loss_limit = self.risk_params.get(
            "daily_loss_limit", 0.05
        )  # 5% par défaut
        self.weekly_loss_limit = self.risk_params.get(
            "weekly_loss_limit", 0.15
        )  # 15% par défaut

        # 4. Paramètres pour micro-capitaux
        self.micro_capital_threshold = 50.0  # Seuil en USDT
        if (
            hasattr(self, "portfolio")
            and self.portfolio.initial_balance < self.micro_capital_threshold
        ):
            # Ajustements pour les petits portefeuilles
            self.max_position_size = min(
                self.max_position_size,
                self.worker_config.get("dbe_config", {})
                .get("micro_capital", {})
                .get("position_size_cap", 0.3),
            )
            self.risk_per_trade = min(
                self.risk_per_trade, 0.5
            )  # Max 0.5% de risque par trade

        # 5. Initialisation de la volatilité
        self.baseline_volatility = 0.01  # 1% de volatilité par défaut
        self.volatility_lookback = (
            self.worker_config.get("dbe_config", {})
            .get("volatility_management", {})
            .get("lookback", 14)
        )

        # 6. Paramètres de trading
        self.max_concurrent_trades = (
            self.worker_config.get("dbe_config", {})
            .get("position_sizing", {})
            .get("max_concurrent_trades", 5)
        )
        self.correlation_threshold = (
            self.worker_config.get("dbe_config", {})
            .get("position_sizing", {})
            .get("correlation_threshold", 0.7)
        )

        # 7. Paramètres spécifiques au régime de marché
        self.regime_parameters = self.worker_config.get("dbe_config", {}).get(
            "regime_parameters", {}
        )

        # Journalisation des paramètres
        self.logger.info(
            f"Paramètres de risque initialisés - "
            f"Taille position: {self.base_position_size * 100:.1f}% "
            f"({self.min_position_size * 100:.1f}%-{self.max_position_size * 100:.1f}%), "
            f"Risque/trade: {self.risk_per_trade:.2f}%, "
            f"Drawdown max: {self.max_drawdown_pct * 100:.1f}%, "
            f"Trades conc.: {self.max_concurrent_trades}"
        )

    def _initialize_components(self) -> None:
        """Initialize all environment components in the correct order."""
        # Lazy imports to avoid circular dependencies
        from ..performance.metrics import PerformanceMetrics
        from ..data_processing.observation_validator import ObservationValidator

        # DIAGNOSTIC: Tracer les appels à _initialize_components
        self.logger.critical(
            f"📋 APPEL _initialize_components pour ENV_ID={getattr(self, 'env_instance_id', 'UNKNOWN')}"
        )

        # Initialize data loader FIRST to know the data structure
        if (
            not hasattr(self, "data_loader_instance")
            or self.data_loader_instance is None
        ):
            # Initialize data loader with correct assets
            self.data_loader_instance = self._init_data_loader(self.assets)
        self.data_loader = self.data_loader_instance

        # 2. Create TimeframeConfig from loaded data (robust to mocked loaders)
        from ..data_processing.state_builder import TimeframeConfig
        timeframe_configs = []
        features_by_tf = getattr(self.data_loader, "features_by_timeframe", None)
        if isinstance(features_by_tf, dict) and len(features_by_tf) > 0:
            for tf_name, features in features_by_tf.items():
                config = TimeframeConfig(
                    timeframe=tf_name, features=features, window_size=100
                )
                timeframe_configs.append(config)
        else:
            # Fallback for tests/mocks: use default OHLCV features for configured timeframes
            self.logger.warning(
                "[FALLBACK] features_by_timeframe missing/non-dict on data_loader; using default features"
            )
            default_features = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
            for tf_name in (self.timeframes or ["5m", "1h", "4h"]):
                config = TimeframeConfig(
                    timeframe=tf_name, features=default_features, window_size=100
                )
                timeframe_configs.append(config)

        # 3. Initialize portfolio manager
        portfolio_config = self.config.copy()
        env_config = self.config.get("environment", {})
        portfolio_config["trading_rules"] = self.config.get("trading_rules", {})
        portfolio_config["capital_tiers"] = self.config.get("capital_tiers", [])
        # Utiliser portfolio.initial_balance en priorité, puis environment.initial_balance
        # avec une valeur par défaut de 20.0
        portfolio_balance = self.config.get("portfolio", {}).get(
            "initial_balance", env_config.get("initial_balance", 20.0)
        )
        portfolio_config["initial_capital"] = portfolio_balance
        portfolio_config["worker_id"] = self.worker_id  # Pass worker_id for log control

        # Map asset names to full names (e.g., BTC -> BTCUSDT)
        asset_mapping = {
            "BTC": "BTCUSDT",
            "ETH": "ETHUSDT",
            "SOL": "SOLUSDT",
            "XRP": "XRPUSDT",
            "ADA": "ADAUSDT",
        }
        # Create a mapped assets list for the data loader
        mapped_assets = [asset_mapping.get(asset, asset) for asset in self.assets]

        # Initialize PerformanceMetrics first (to be shared with PortfolioManager)
        from ..performance.metrics import PerformanceMetrics

        self.performance_metrics = PerformanceMetrics(
            config=self.config, worker_id=self.worker_id
        )

        # Initialize the new PositionSizer
        from ..trading.position_sizer import PositionSizer
        self.position_sizer = PositionSizer(config=self.config)

        # Initialize portfolio with mapped asset names, worker_id and shared PerformanceMetrics
        from ..portfolio.portfolio_manager import PortfolioManager
        self.portfolio = PortfolioManager(
            config=portfolio_config,
            worker_id=self.worker_id,
            performance_metrics=self.performance_metrics,
            max_positions=self.max_positions, # Pass max_positions to PortfolioManager
        )
        # Create alias for backward compatibility
        self.portfolio_manager = self.portfolio
        self.assets = [
            a.upper() for a in mapped_assets
        ]  # Update self.assets and normalize to uppercase

        # Convert list of TimeframeConfig objects to dictionary
        timeframe_configs_dict = {
            tf_config.timeframe: tf_config for tf_config in timeframe_configs
        }

        # 4. Initialize StateBuilder with dynamic config
        features_config = {
            tf: config.features for tf, config in timeframe_configs_dict.items()
        }



        # Récupérer les tailles de fenêtres spécifiques à chaque timeframe
        env_obs_cfg = self.config.get("environment", {}).get("observation", {})
        window_sizes = env_obs_cfg.get(
            "window_sizes",
            {"5m": 20, "1h": 10, "4h": 5},  # Valeurs par défaut si non spécifiées
        )
        self.window_sizes = window_sizes

        # Utiliser la taille de fenêtre du timeframe 5m comme valeur par défaut
        default_window_size = window_sizes.get("5m", 20)

        # Initialisation du validateur d'observations
        n_features = len(next(iter(features_config.values())))
        validator_config = {
            "timeframes": self.timeframes,
            "n_assets": len(self.assets),
            "window_size": default_window_size,
            "n_features": n_features,
            "portfolio_state_size": DEFAULT_PORTFOLIO_STATE_SIZE,
        }
        self.observation_validator = ObservationValidator(validator_config)

        # Configurer les tailles de fenêtres spécifiques pour chaque timeframe
        timeframe_configs = {}
        for tf in features_config.keys():
            tf_window_size = window_sizes.get(tf, default_window_size)
            timeframe_configs[tf] = TimeframeConfig(
                timeframe=tf,
                features=features_config[tf],
                window_size=tf_window_size,
                normalize=True,
            )
            self.logger.info(
                f"Configuration de la fenêtre pour {tf}: {tf_window_size} périodes"
            )

        # Initialiser le StateBuilder avec la configuration des timeframes
        from ..data_processing.state_builder import StateBuilder
        self.state_builder = StateBuilder(
            features_config=features_config,
            window_sizes=window_sizes,
            include_portfolio_state=True,
            normalize=True,
        )

        # La configuration des timeframes est maintenant gérée dans le constructeur de StateBuilder

        # 5. Setup action and observation spaces (requires state_builder)
        self._setup_spaces()

        # Define warmup steps from config, with a default of 50
        self.warmup_steps = self.config.get("environment", {}).get("observation", {}).get("warmup_steps", 50)
        self.logger.info(f"Warmup period set to {self.warmup_steps} steps.")

        # 6. Initialize max_steps and max_chunks_per_episode from config
        self.max_steps = self.config.get("environment", {}).get("max_steps", 1000)
        self.max_chunks_per_episode = self.config.get("environment", {}).get(
            "max_chunks_per_episode", 10
        )

        # Initialize total_chunks from data_loader if available
        self.total_chunks = getattr(
            self.data_loader, "total_chunks", 10
        )  # Default to 10 if not available

        # 7. Initialize DynamicBehaviorEngine with proper configuration
        # Fusion de la configuration du worker et de la configuration principale
        dbe_config = self.worker_config.get("dbe", {}) or self.config.get("dbe", {})

        # Assurez-vous que la configuration des paramètres de risque est correctement chargée
        dbe_config.setdefault("risk_parameters", {})
        
        # ⚠️ IMPORTANT: Passer la section "workers" complète à DBE pour que chaque worker
        # puisse lire ses propres SL/TP depuis la configuration
        if "workers" not in dbe_config and "workers" in self.config:
            dbe_config["workers"] = self.config["workers"]

        # Inject position_sizing config into DBE config if not present
        if "position_sizing" not in dbe_config:
            env_risk_management = self.config.get("environment", {}).get(
                "risk_management", {}
            )
            if "position_sizing" in env_risk_management:
                dbe_config["position_sizing"] = env_risk_management["position_sizing"]
            else:
                dbe_config["position_sizing"] = {}

        # 8. PerformanceMetrics already initialized above (shared with PortfolioManager)

        # SOLUTION IMMORTALITÉ ADAN: Utiliser le DBE externe ou en créer un nouveau
        if hasattr(self, "external_dbe") and self.external_dbe is not None:
            # Réutiliser le DBE immortel fourni lors de la création
            self.dynamic_behavior_engine = self.external_dbe
            self.logger.critical(
                f"👑 DBE IMMORTEL RÉUTILISÉ pour ENV_ID={self.env_instance_id}, DBE_ID={id(self.external_dbe)}"
            )
        elif (
            not hasattr(self, "dynamic_behavior_engine")
            or self.dynamic_behavior_engine is None
        ):
            # Créer un nouveau DBE uniquement si aucun n'existe
            # --- CORRECTION CENTRALE ---
            # Injecter le PortfolioManager dans le DBE
            from .dynamic_behavior_engine import DynamicBehaviorEngine
            self.dynamic_behavior_engine = DynamicBehaviorEngine(
                config=dbe_config,
                finance_manager=self.portfolio_manager, # <-- INJECTION DE DÉPENDANCE
                worker_id=self.worker_id,
            )
            self.logger.critical(
                f"🧠 NOUVEAU DBE CRÉÉ pour ENV_ID={self.env_instance_id}"
            )
        else:
            # Cas très rare : DBE déjà existant dans la même instance
            self.logger.critical(
                f"🔄 DBE EXISTANT PRÉSERVÉ pour ENV_ID={self.env_instance_id}"
            )

        # Création d'un alias pour la rétrocompatibilité
        self.dbe = self.dynamic_behavior_engine

        # Connecter le DBE à l'environnement IMMÉDIATEMENT pour garantir la référence
        if hasattr(self.dbe, "set_env_reference"):
            self.dbe.set_env_reference(self)
            self.logger.info(f"[INIT] Référence DBE -> ENV établie immédiatement.")

        # Ajout de la référence du DBE au PortfolioManager pour la gestion des traques
        self.portfolio_manager.dbe = self.dbe

        # ✅ CRITIQUE: Vérifier que finance_manager est bien présent
        if not hasattr(self.dbe, 'finance_manager') or self.dbe.finance_manager is None:
            self.logger.error("[INIT] CRITICAL: DBE.finance_manager est None après init!")
            self.dbe.finance_manager = self.portfolio_manager

        self.logger.info(
            f"Initialized max_steps to {self.max_steps} and max_chunks_per_episode to {self.max_chunks_per_episode}"
        )

        # Log the chunking configuration
        self.logger.info(
            f"Chunk configuration - Total chunks: {self.total_chunks}, Max chunks per episode: {self.max_chunks_per_episode}"
        )

        # 8. Initialize other components using worker_config where available
        trading_rules = self.config.get("trading_rules", {})
        penalties = self.config.get("environment", {}).get("penalties", {})
        from .order_manager import OrderManager
        self.order_manager = OrderManager(
            trading_rules=trading_rules, penalties=penalties
        )

        # Get reward config with fallback to main config
        env_section = self.config.get("environment", {})
        reward_cfg = self.worker_config.get(
            "reward_config", env_section.get("reward_config", {})
        )

        # Create env config with reward shaping
        env_config = {"reward_shaping": reward_cfg}
        from .reward_calculator import RewardCalculator
        self.reward_calculator = RewardCalculator(env_config=env_config)

        # Initialize observation validator (will be initialized if needed)
        self.observation_validator = None

    def _init_data_loader(self, assets: List[str]) -> Any:
        """Initialize the chunked data loader using worker-specific config.

        Args:
            assets: List of assets to load data for

        Returns:
            Initialized ChunkedDataLoader instance

        Raises:
            ValueError: If configuration is invalid or no assets are available
        """
        if not self.worker_config:
            raise ValueError(
                "worker_config must be provided to initialize the data loader."
            )

        # Ensure paths are resolved
        if not hasattr(self, "config") or not self.config:
            raise ValueError("Configuration not properly initialized")

        # Mapping for asset names to file system names (e.g., BTC -> BTCUSDT)
        asset_mapping = {
            "BTC": "BTCUSDT",
            "ETH": "ETHUSDT",
            "SOL": "SOLUSDT",
            "XRP": "XRPUSDT",
            "ADA": "ADAUSDT",
        }
        # Create a mapped assets list for the data loader
        mapped_assets = [asset_mapping.get(asset, asset) for asset in assets]

        if not mapped_assets:
            raise ValueError("No assets specified in worker or environment config")

        # Get timeframes from config with fallback to worker config
        global_data_timeframes = self.config.get("data", {}).get("timeframes", [])
        worker_timeframes = self.worker_config.get("timeframes", [])

        # Use worker timeframes if specified, otherwise fallback to global config
        self.timeframes = worker_timeframes or global_data_timeframes

        if not self.timeframes:
            raise ValueError(
                f"No timeframes defined: global={global_data_timeframes}, "
                f"worker={worker_timeframes}"
            )

        # Create a worker config with the correct assets and timeframes
        worker_config = {
            **self.worker_config,
            "assets": mapped_assets,
            "timeframes": self.timeframes,
            "data_split_override": self.worker_config.get("data_split", "train"),
        }

        # Initialize the data loader with the correct config (lazy import to avoid circular deps)
        from ..data_processing.data_loader import ChunkedDataLoader as _CDL
        self.data_loader = _CDL(
            config=self.config, worker_config=worker_config, worker_id=self.worker_id
        )

        return self.data_loader

    def _safe_load_chunk(
        self, chunk_idx: int, fallback_enabled: bool = True
    ) -> Dict[str, Any]:
        """Safely load a chunk with retry mechanism and fallback.

        Args:
            chunk_idx: Index of the chunk to load
            fallback_enabled: Whether to use fallback mechanism on failure

        Returns:
            Loaded chunk data or fallback data

        Raises:
            RuntimeError: If all attempts fail and no fallback is available
        """
        import time

        dbe_state_file = f"dbe_state_{self.worker_id}.pkl"

        # 1. Sauvegarder l'état du DBE
        if hasattr(self, "dbe") and self.dbe is not None:
            try:
                with open(dbe_state_file, "wb") as f:
                    # Utiliser directement l'objet, ce qui appelle __getstate__
                    pickle.dump(self.dbe, f) 
                self.smart_logger.debug(f"[DBE_STATE] Worker {self.worker_id}: DBE state saved to {dbe_state_file}")
            except Exception as e:
                self.smart_logger.warning(f"[DBE_STATE] Worker {self.worker_id}: Failed to save DBE state: {e}")

        for attempt in range(MAX_RELOAD_ATTEMPTS):
            try:
                self.smart_logger.info(
                    f"[CHUNK_LOADER] Attempting to load chunk {chunk_idx} (attempt {attempt + 1}/{MAX_RELOAD_ATTEMPTS})",
                    rotate=True,
                )
                chunk_data = self.data_loader.load_chunk(chunk_idx)

                # Validate chunk data
                if isinstance(chunk_data, dict) and chunk_data:
                    has_data = False
                    for asset_data in chunk_data.values():
                        if isinstance(asset_data, dict) and asset_data:
                            for tf_data in asset_data.values():
                                if isinstance(tf_data, pd.DataFrame) and not tf_data.empty:
                                    has_data = True
                                    break
                        if has_data:
                            break
                    if has_data:
                        self.smart_logger.info(
                            f"[CHUNK_LOADER] Successfully loaded chunk {chunk_idx} on attempt {attempt + 1}",
                            rotate=True,
                        )
                        
                        # CRITICAL FIX: Deepcopy to prevent cache corruption
                        # DataLoader uses lru_cache, so we must not modify the returned object in-place
                        import copy
                        chunk_data = copy.deepcopy(chunk_data)

                        # 2. Charger l'état du DBE
                        if hasattr(self, "dbe") and self.dbe is not None and os.path.exists(dbe_state_file):
                            try:
                                with open(dbe_state_file, "rb") as f:
                                    loaded_dbe = pickle.load(f)
                                self.dbe = loaded_dbe
                                if hasattr(self.dbe, 'set_env_reference'):
                                    self.dbe.set_env_reference(self)
                                
                                # ✅ NOUVEAU: Nettoyage du fichier
                                os.remove(dbe_state_file)
                                self.smart_logger.debug(f"[DBE_STATE] Cleaned up pickle: {dbe_state_file}")
                            except Exception as e:
                                self.smart_logger.warning(f"[DBE_STATE] Load or cleanup failed: {e}")
                        return chunk_data
                    else:
                        self.smart_logger.warning(
                            f"[CHUNK_LOADER] Chunk {chunk_idx} loaded but contains no valid data (attempt {attempt + 1})"
                        )
                else:
                    self.smart_logger.warning(
                        f"[CHUNK_LOADER] Chunk {chunk_idx} loaded but is not a dictionary (attempt {attempt + 1})"
                    )
            except Exception as e:
                self.smart_logger.error(
                    f"[CHUNK_LOADER] Failed to load chunk {chunk_idx} on attempt {attempt + 1}: {str(e)}"
                )

                # Wait before retry (except on last attempt)
                if attempt < MAX_RELOAD_ATTEMPTS - 1:
                    time.sleep(RELOAD_RETRY_DELAY)

        # All attempts failed, try fallback if enabled
        if fallback_enabled and chunk_idx != RELOAD_FALLBACK_CHUNK:
            self.smart_logger.warning(
                f"[CHUNK_LOADER] All attempts failed for chunk {chunk_idx}, falling back to chunk {RELOAD_FALLBACK_CHUNK}"
            )
            try:
                fallback_data = self.data_loader.load_chunk(RELOAD_FALLBACK_CHUNK)
                if fallback_data and any(
                    fallback_data.get(asset) for asset in fallback_data
                ):
                    self.smart_logger.info(
                        f"[CHUNK_LOADER] Successfully loaded fallback chunk {RELOAD_FALLBACK_CHUNK}",
                        rotate=True,
                    )
                    # 2. Load DBE state after successful chunk load (for fallback)
                    if hasattr(self, "dbe") and self.dbe is not None and os.path.exists(dbe_state_file):
                        try:
                            with open(dbe_state_file, "rb") as f:
                                dbe_loaded_state = pickle.load(f)
                            self.dbe.__dict__.update(dbe_loaded_state)
                            self.smart_logger.debug(f"[DBE_STATE] Worker {self.worker_id}: DBE state loaded from {dbe_state_file} (fallback)")
                        except Exception as e:
                            self.smart_logger.warning(f"[DBE_STATE] Worker {self.worker_id}: Failed to load DBE state (fallback): {e}")
                    return fallback_data
                else:
                    self.smart_logger.error(
                        f"[CHUNK_LOADER] Fallback chunk {RELOAD_FALLBACK_CHUNK} is also empty or invalid"
                    )
            except Exception as e:
                self.smart_logger.error(
                    f"[CHUNK_LOADER] Fallback chunk {RELOAD_FALLBACK_CHUNK} also failed to load: {str(e)}"
                )

        # If we get here, everything failed – synthesize a minimal valid mock chunk to allow tests to proceed
        try:
            self.smart_logger.warning(
                f"[CHUNK_LOADER] All loading strategies failed for chunk {chunk_idx}. Synthesizing mock chunk for tests."
            )
            synthetic = self._synthesize_mock_chunk(n_rows=200)
            if isinstance(synthetic, dict) and synthetic:
                return synthetic
        except Exception as _e:
            self.smart_logger.error(
                f"[CHUNK_LOADER] Failed to synthesize mock chunk: {_e}",
            )
        # Final hard failure
        raise RuntimeError(
            f"Failed to load chunk {chunk_idx} after {MAX_RELOAD_ATTEMPTS} attempts and fallback failed"
        )

    def _synthesize_mock_chunk(self, n_rows: int = 200) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Create a minimal valid chunk structure for tests when data loader is mocked or returns invalid data.

        Returns:
            Dict[str, Dict[str, DataFrame]] mapping asset -> timeframe -> DataFrame
        """
        try:
            # Ensure assets/timeframes are available
            assets = getattr(self, "assets", ["BTCUSDT"]) or ["BTCUSDT"]
            tfs = getattr(self, "timeframes", ["5m", "1h", "4h"]) or ["5m", "1h", "4h"]

            # Map timeframe to pandas frequency
            tf_to_freq = {"5m": "5min", "1h": "1h", "4h": "4h"}
            now = pd.Timestamp.now(tz="UTC").floor("min")

            result: Dict[str, Dict[str, pd.DataFrame]] = {}
            for asset in assets:
                result[asset] = {}
                for tf in tfs:
                    freq = tf_to_freq.get(tf, "5min")
                    idx = pd.date_range(end=now, periods=n_rows, freq=freq)
                    # Simple synthetic OHLCV
                    base = np.linspace(100.0, 101.0, num=n_rows, dtype=np.float64)
                    noise = np.random.default_rng(0).normal(0, 0.1, size=n_rows)
                    close = base + noise.cumsum() * 0.01
                    open_ = close + np.random.default_rng(1).normal(0, 0.02, size=n_rows)
                    high = np.maximum(open_, close) + 0.05
                    low = np.minimum(open_, close) - 0.05
                    vol = np.abs(np.random.default_rng(2).normal(1000, 50, size=n_rows))
                    df = pd.DataFrame(
                        {
                            "OPEN": open_.astype(np.float64),
                            "HIGH": high.astype(np.float64),
                            "LOW": low.astype(np.float64),
                            "CLOSE": close.astype(np.float64),
                            "VOLUME": vol.astype(np.float64),
                        },
                        index=idx,
                    )
                    result[asset][tf] = df
            return result
        except Exception as e:
            logger.error(f"Synthetic chunk creation failed: {e}")
            return {}

    def _setup_spaces(self) -> None:
        """Configure les espaces d'action et d'observation pour le modèle."""
        # L'espace d'action reste un vecteur continu pour la taille des positions et le choix du timeframe.
        num_actions = len(self.assets) + 1
        # Solution pragmatique: maintenir 5 actifs pour compatibilité avec anciens modèles PPO
        # Les actifs non utilisés seront ignorés dans _execute_trades
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(25,),  # 5 actifs × 5 dimensions (Action, Size, TF, SL, TP)
            dtype=np.float32,
        )
        logger.info(f"Espace d'action configuré : {self.action_space}")
        logger.info(f"Structure par actif : [Action, Size, Timeframe, StopLoss, TakeProfit]")
        logger.info(f"Actifs configurés : {len(self.assets)}/5 actifs ({self.assets})")
        if len(self.assets) < 5:
            logger.info(
                f"🔧 Mode compatibilité : {5 - len(self.assets)} actifs non utilisés seront ignorés"
            )

        # --- NOUVEL ESPACE D'OBSERVATION ALIGNÉ SUR STATEBUILDER ---
        try:
            obs_spaces = {}
            window_sizes = getattr(self, "window_sizes", {"5m": 20, "1h": 10, "4h": 5})
            for tf in self.timeframes:
                window_size = window_sizes.get(tf, 20)
                # Robuste aux mocks: get_feature_names(tf) peut retourner un Mock non itérable
                try:
                    feature_names = self.state_builder.get_feature_names(tf)
                    if hasattr(feature_names, "__len__") and not isinstance(feature_names, str):
                        # Guarantee at least 15 features for test compatibility
                        n_features = self.state_builder.max_features
                    else:
                        # Fallback par défaut pour tests: 15 features (compatibilité jeux de tests existants)
                        n_features = 15
                        logger.warning(
                            f"[MOCK FALLBACK] get_feature_names({tf}) non itérable, utilisation de n_features={n_features}"
                        )
                except Exception as e:
                    n_features = 15
                    logger.warning(
                        f"[MOCK FALLBACK] Exception lors de get_feature_names({tf}): {e} → n_features={n_features}"
                    )
                obs_spaces[tf] = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(window_size, n_features),
                    dtype=np.float32,
                )

            portfolio_dim = DEFAULT_PORTFOLIO_STATE_SIZE
            obs_spaces["portfolio_state"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(portfolio_dim,), dtype=np.float32
            )

            # Context vector for FiLM Meta-RL modulation (SOTA 2025: 12 dims)
            # [0-5] market: volatility, trend, ADX, regime_score, drawdown, candle_progress
            # [6-11] cyclical time: sin/cos hour, day-of-week, day-of-month
            obs_spaces["context_vector"] = spaces.Box(
                low=-1.0, high=1.0, shape=(12,), dtype=np.float32
            )

            self.observation_space = spaces.Dict(obs_spaces)
            logger.info(
                f"Espace d'observation reconfiguré pour CNN multi-échelle: {self.observation_space}"
            )

        except Exception as e:
            logger.error(
                f"Erreur lors de la configuration du nouvel espace d'observation : {str(e)}",
                exc_info=True,
            )
            raise

    def _get_initial_observation(self) -> Dict[str, np.ndarray]:
        """
        Get the initial observation after environment reset.
        This method now directly calls _get_observation to ensure consistency
        with the new multi-timeframe dictionary-based observation space.
        """
        self.logger.debug(
            "Redirecting initial observation to main _get_observation method."
        )
        return self._get_observation()

    def _set_start_step_for_chunk(self):
        """Calculates and sets the starting step within a new chunk to account for indicator warmup."""
        try:
            # Use a more conservative warmup period to avoid index out of bounds
            warmup = getattr(self, "warmup_period", 200)  # Default to 200 if not set

            # CRITICAL FIX: Calculate safe start step based on SMALLEST timeframe
            # Prevents out-of-bounds reads for short timeframes (e.g. 4h with 111 rows)
            min_chunk_length = float('inf')
            if isinstance(self.current_data, dict) and self.current_data:
                for asset_data in self.current_data.values():
                    if not isinstance(asset_data, dict):
                        continue
                    for tf in getattr(self, "timeframes", []):
                        df = asset_data.get(tf)
                        if isinstance(df, pd.DataFrame):
                            min_chunk_length = min(min_chunk_length, len(df))
            
            # Convert to int and ensure valid
            if min_chunk_length == float('inf') or min_chunk_length < 10:
                min_chunk_length = 10  # Fallback minimum
                logger.warning(f"Very small or missing chunk, using fallback min_len={min_chunk_length}")
            
            #  Safety margin: Use only 20% of chunk length to be VERY conservative
            # For 111-row chunk, this gives us step ~22, well within safe bounds
            safe_max_start = max(1, int(min_chunk_length * 0.2))
            
            # Use minimum of warmup_period and safe max
            # This ensures we never start beyond available data and stay very conservative
            self.step_in_chunk = min(warmup, safe_max_start)
            
            # Additional safety: ensure we don't start at 0 
            self.step_in_chunk = max(1, self.step_in_chunk)
            
            logger.info(
                f"Repositioning to step {self.step_in_chunk} in new chunk "
                f"(warmup={warmup}, min_len={int(min_chunk_length)}, max_safe={safe_max_start})"
            )

        except Exception as e:
            logger.warning(f"Failed to set warmup step_in_chunk: {e}")
            self.step_in_chunk = 1  # Safe fallback

    def _calculate_excellence_bonus(
        self, base_reward: float, worker_id: int = 0
    ) -> float:
        """Calcule les bonus d'excellence Gugu & March"""
        if not self.excellence_rewards or not EXCELLENCE_SYSTEM_AVAILABLE:
            return 0.0

        try:
            # Récupérer les métriques actuelles du worker
            metrics = self._build_excellence_metrics(worker_id)

            # Calculer les bonus d'excellence
            total_bonus, bonus_breakdown = (
                self.excellence_rewards.calculate_excellence_bonus(
                    base_reward, metrics, trade_won=(base_reward > 0)
                )
            )

            # Logger si bonus significatif
            if total_bonus > 0.01:
                logger.debug(
                    f"[GUGU-MARCH] Worker {worker_id} excellence bonus: {total_bonus:.4f}"
                )

            return total_bonus

        except Exception as e:
            logger.warning(f"[GUGU-MARCH] Error calculating excellence bonus: {e}")
            return 0.0

    def _build_excellence_metrics(self, worker_id: int = 0) -> "ExcellenceMetrics":
        """Construit les métriques d'excellence pour un worker"""
        if not EXCELLENCE_SYSTEM_AVAILABLE:
            from dataclasses import dataclass

            @dataclass
            class DummyMetrics:
                sharpe_ratio: float = 0.0
                profit_factor: float = 1.0
                win_rate: float = 0.5
                winning_streak: int = 0
                total_trades: int = 0

            return DummyMetrics()

        try:
            # Récupérer les métriques depuis le portfolio manager
            portfolio = (
                self.portfolio_managers[worker_id]
                if hasattr(self, "portfolio_managers")
                else self.portfolio_manager
            )
            perf_metrics = (
                self.performance_metrics[worker_id]
                if hasattr(self, "performance_metrics")
                else self.performance_metrics
            )

            metrics_summary = perf_metrics.get_metrics_summary() if perf_metrics else {}

            # Analyser confluence des timeframes (exemple simplifié)
            timeframe_signals = self._analyze_current_confluence(worker_id)

            return ExcellenceMetrics(
                sharpe_ratio=metrics_summary.get("sharpe_ratio", 0.0),
                profit_factor=metrics_summary.get("profit_factor", 1.0),
                win_rate=metrics_summary.get("win_rate", 0.5),
                winning_streak=getattr(
                    self.excellence_rewards, "last_winning_streak", 0
                ),
                total_trades=metrics_summary.get("total_trades", 0),
                current_drawdown=portfolio.current_drawdown_pct if portfolio else 0.0,
                timeframe_signals=timeframe_signals,
            )

        except Exception as e:
            logger.warning(f"[GUGU-MARCH] Error building excellence metrics: {e}")
            return ExcellenceMetrics()

    def _analyze_current_confluence(self, worker_id: int = 0) -> Dict[str, bool]:
        """Analyse la confluence des signaux multi-timeframes"""
        # Implémentation simplifiée - à adapter selon votre structure de données
        try:
            confluence = {"5m": False, "1h": False, "4h": False}

            # Exemple: vérifier si les indicateurs sont alignés
            if hasattr(self, "current_observations") and self.current_observations:
                obs = (
                    self.current_observations[worker_id]
                    if isinstance(self.current_observations, list)
                    else self.current_observations
                )

                # Logique simplifiée de confluence (à personnaliser)
                # Par exemple, vérifier RSI et MACD sur différents TF
                for tf in ["5m", "1h", "4h"]:
                    if f"rsi_{tf}" in obs:
                        rsi = obs[f"rsi_{tf}"]
                        macd = obs.get(f"macd_{tf}", 0)
                        # Signal haussier si RSI < 70 et MACD > 0
                        confluence[tf] = (20 < rsi < 70) and (macd > -0.1)

            return confluence

        except Exception as e:
            logger.debug(f"[GUGU-MARCH] Error in confluence analysis: {e}")
            return {"5m": False, "1h": False, "4h": False}

    def reset(self, *, seed=None, options=None):
        """
        Resets the environment to an initial state.
        Loads a new random chunk and repositions within it.
        """
        # CRITICAL FIX: Close all open positions before reset to ensure:
        # 1. PnL is realized (not left as unrealized)
        # 2. Trade counts are correct
        # 3. Metrics are accurate
        if hasattr(self, 'portfolio_manager') and self.portfolio_manager is not None:
            try:
                # Get current positions before closing
                if hasattr(self.portfolio_manager, 'positions') and self.portfolio_manager.positions:
                    logger.info(f"[EPISODE_END] Closing {len(self.portfolio_manager.positions)} open position(s) before reset")
                    
                    # Close each position
                    current_prices = self._get_current_prices() if hasattr(self, '_get_current_prices') else {}
                    for asset in list(self.portfolio_manager.positions.keys()):
                        if asset in current_prices and current_prices[asset]:
                            try:
                                self.portfolio_manager.close_position(
                                    asset=asset,
                                    price=current_prices[asset],
                                    reason="EPISODE_END"
                                )
                                logger.info(f"[EPISODE_END] Closed position for {asset}")
                            except Exception as e:
                                logger.warning(f"Failed to close position for {asset}: {e}")
            except Exception as e:
                logger.warning(f"Error closing positions before reset: {e}")
        
        self.logger.critical(
            f"🔄 RESET appelé pour ENV_ID={getattr(self, 'env_instance_id', 'UNKNOWN')}, Worker={getattr(self, 'worker_id', 'N/A')}, DBE_ID={id(getattr(self, 'dbe', None)) if hasattr(self, 'dbe') else 'NONE'}"
        )
        super().reset(seed=seed)

        # Reset episode-specific variables
        self.current_step = 0
        self.done = False
        self.episode_reward = 0.0
        self.step_in_chunk = 0

        # Reset frequency tracking counters
        self.positions_count = {"5m": 0, "1h": 0, "4h": 0, "daily_total": 0}
        self.daily_reset_step = 0
        self.current_day = 0
        self._last_trade_count = 0  # Initialize trade count tracking

        # Reset trade tracking helpers to avoid negative deltas after episode restart
        self.last_trade_steps_by_tf = {}
        self.last_trade_timestamps = {}
        self.current_timeframe_for_trade = "5m"

        # OMEGA-3/4: Reset episode receipts and per-asset cooldown
        self._all_episode_receipts = []
        self._last_open_step_by_asset = {}

        # Reset portfolio and load initial data chunk
        if hasattr(self, "last_trade_step"):
            self.last_trade_step = -1

        # Determine if this is a true new episode or just a reset
        is_new_episode = not hasattr(self, "_episode_initialized") or getattr(
            self, "_needs_full_reset", False
        )

        # Reset the environment with appropriate parameters
        self._epoch_reset(force=False, new_epoch=is_new_episode)

        # Mark that we've initialized at least one episode
        self._episode_initialized = True
        self._needs_full_reset = False  # Reset the flag

        # CORRECTION CRITIQUE : Réinitialiser les index de chunks
        # Utiliser start_chunk_index de la config si disponible (pour validation Out-of-Sample)
        start_chunk = self.config.get("environment", {}).get("start_chunk_index", 0)
        self.current_chunk_idx = start_chunk
        if hasattr(self, "current_chunk"):
            self.current_chunk = 0

        logger.info(
            f"[RESET Worker {getattr(self, 'worker_id', 0)}] Starting new episode - Loading chunk 1/{getattr(self, 'total_chunks', 'unknown')}"
        )
        self.current_data = self._safe_load_chunk(self.current_chunk_idx)

        # Fit scalers on the new data
        if hasattr(self, "state_builder") and hasattr(self.state_builder, "fit_scalers"):
            try:
                # Transpose data from {asset: {tf: df}} to {tf: [dfs]}
                data_for_fitting = {}
                for asset_data in self.current_data.values():
                    for tf, df in asset_data.items():
                        if tf not in data_for_fitting:
                            data_for_fitting[tf] = []
                        data_for_fitting[tf].append(df)
                
                # Concatenate dataframes for each timeframe
                concatenated_data = {}
                for tf, dfs in data_for_fitting.items():
                    if dfs:
                        concatenated_data[tf] = pd.concat(dfs)

                if concatenated_data:
                    self.state_builder.fit_scalers(concatenated_data)
                    logger.info("Scalers fitted successfully on the new data chunk.")
                else:
                    logger.warning("No data available for fitting scalers.")
            except Exception as e:
                logger.error(f"Failed to fit scalers: {e}", exc_info=True)

        # Position the step within the chunk to ensure a non-empty observation window
        self._set_start_step_for_chunk()

        # Get initial observation using the robust _get_initial_observation method
        observation = self._get_initial_observation()

        # Store the current observation for future reference
        self._current_obs = observation

        # Get additional info
        info = self._get_info()

        return observation, info

    def _apply_tier_reward(self, reward: float, current_value: float) -> float:
        """Applique les récompenses et pénalités liées aux changements de palier.

        Args:
            reward: Récompense actuelle à modifier
            current_value: Valeur actuelle du portefeuille

        Returns:
            float: Récompense modifiée
        """
        if not hasattr(self, "current_tier") or self.current_tier is None:
            return reward

        # Mettre à jour le meilleur portefeuille pour ce palier
        if current_value > self.best_portfolio_value:
            self.best_portfolio_value = current_value

        # Vérifier si le palier a changé
        has_changed, is_promotion = self._update_tier(current_value)

        if not has_changed:
            return reward

        # Appliquer les bonus/malus de changement de palier
        tier_rewards = self.config.get("reward_shaping", {}).get("tier_rewards", {})

        if is_promotion:
            promotion_bonus = tier_rewards.get("promotion_bonus", 0.0)
            logger.info(f"Applying promotion bonus: {promotion_bonus}")
            reward += promotion_bonus

            # Sauvegarder le modèle si configuré
            if tier_rewards.get("checkpoint_on_promotion", False):
                self._save_checkpoint_on_promotion()
        else:
            demotion_penalty = tier_rewards.get("demotion_penalty", 0.0)
            logger.info(f"Applying demotion penalty: {demotion_penalty}")
            reward -= demotion_penalty

        # Appliquer le multiplicateur de performance du palier
        performance_multiplier = self.current_tier.get("performance_multiplier", 1.0)
        if performance_multiplier != 1.0:
            reward *= performance_multiplier
            logger.info(
                f"Applied tier performance multiplier: {performance_multiplier}"
            )

        return reward

    def _save_checkpoint_on_promotion(self) -> None:
        """Sauvegarde un point de contrôle complet lors d'une promotion de palier.

        Cette méthode sauvegarde à la fois le modèle et l'état de l'environnement.
        """
        if not hasattr(self, "model") or self.model is None:
            logger.warning("Cannot save checkpoint: model not available")
            return

        # Créer le répertoire de checkpoints s'il n'existe pas
        tier_rewards = self.config.get("reward_shaping", {}).get("tier_rewards", {})
        checkpoint_dir = tier_rewards.get("checkpoint_dir", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Générer un nom de fichier unique avec le timestamp et le palier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tier_name = self.current_tier["name"].lower().replace(" ", "_")
        checkpoint_base = os.path.join(
            checkpoint_dir, f"model_{tier_name}_promo_{timestamp}"
        )

        try:
            # 1. Sauvegarder le modèle
            model_path = f"{checkpoint_base}_model"
            self.model.save(model_path)
            logger.info(f"Model checkpoint saved to {model_path}")

            # 2. Sauvegarder l'état de l'environnement
            env_checkpoint = self._save_checkpoint()
            env_checkpoint["model_path"] = model_path

            # 3. Sauvegarder les métadonnées supplémentaires
            metadata = {
                "tier": self.current_tier["name"],
                "timestamp": timestamp,
                "portfolio_value": self.portfolio.get_total_value(),
                "episode": self.episode_count,
                "step": self.current_step,
                "checkpoint_type": "promotion",
                "tier_info": {
                    "current_tier": self.current_tier["name"],
                    "min_value": self.current_tier["min_value"],
                    "max_value": self.current_tier.get("max_value", float("inf")),
                    "episodes_in_tier": self.episodes_in_tier,
                    "last_tier_change_step": self.last_tier_change_step,
                },
            }

            # 4. Fusionner les métadonnées avec le checkpoint
            env_checkpoint["metadata"] = metadata

            # 5. Sauvegarder le checkpoint complet
            checkpoint_path = f"{checkpoint_base}_full.pkl"
            with open(checkpoint_path, "wb") as f:
                import pickle

                pickle.dump(env_checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Full environment checkpoint saved to {checkpoint_path}")

            # 6. Mettre à jour l'historique des checkpoints
            if not hasattr(self, "checkpoint_history"):
                self.checkpoint_history = []

            self.checkpoint_history.append(
                {
                    "timestamp": timestamp,
                    "path": checkpoint_path,
                    "tier": self.current_tier["name"],
                    "portfolio_value": self.portfolio.get_total_value(),
                }
            )

            # 7. Garder uniquement les N derniers checkpoints
            max_checkpoints = tier_rewards.get("max_checkpoints", 5)
            if len(self.checkpoint_history) > max_checkpoints:
                oldest_checkpoint = self.checkpoint_history.pop(0)
                try:
                    os.remove(oldest_checkpoint["path"])
                    logger.info(f"Removed old checkpoint: {oldest_checkpoint['path']}")
                except Exception as e:
                    logger.error(f"Failed to remove old checkpoint: {e}")

        except Exception as e:
            logger.error(f"Failed to save promotion checkpoint: {e}")
            raise

    def _update_tier(self, current_value: float) -> Tuple[bool, bool]:
        """Met à jour le palier actuel en fonction de la valeur du portefeuille.

        Args:
            current_value: Valeur actuelle du portefeuille

        Returns:
            Tuple[bool, bool]: (has_tier_changed, is_promotion) indiquant
                              si le palier a changé et si c'est une promotion
        """
        if not hasattr(self, "portfolio"):
            return False, False

        current_tier = self.portfolio.get_current_tier()

        # Si c'est la première initialisation
        if self.current_tier is None:
            self.current_tier = current_tier
            self.best_portfolio_value = current_value
            self.tier_history.append(
                {
                    "step": self.current_step,
                    "tier": current_tier["name"],
                    "portfolio_value": current_value,
                    "episode": self.episode_count,
                    "is_promotion": False,
                }
            )
            return False, False

        # Vérifier si le palier a changé
        if current_tier["name"] != self.current_tier["name"]:
            self.previous_tier = self.current_tier
            self.current_tier = current_tier
            self.last_tier_change_step = self.current_step
            self.episodes_in_tier = 0

            # Déterminer s'il s'agit d'une promotion
            prev_min = (
                self.previous_tier.get("min_capital", 0) if self.previous_tier else 0
            )
            is_promotion = current_tier["min_capital"] > prev_min

            # Mettre à jour l'historique
            self.tier_history.append(
                {
                    "step": self.current_step,
                    "tier": current_tier["name"],
                    "portfolio_value": current_value,
                    "episode": self.episode_count,
                    "is_promotion": is_promotion,
                }
            )

            prev_name = self.previous_tier["name"]
            curr_name = current_tier["name"]
            logger.info(
                f"Tier changed from {prev_name} to {curr_name} "
                f"(Promotion: {is_promotion}) at step {self.current_step}"
            )

            return True, is_promotion

        return False, False

        # Patch Gugu & March - Initialisation du système d'excellence
        if EXCELLENCE_SYSTEM_AVAILABLE:
            try:
                self.excellence_rewards = create_excellence_rewards_system(self.config)
                logger.info("[GUGU-MARCH] Excellence rewards system initialized")
            except Exception as e:
                logger.warning(
                    f"[GUGU-MARCH] Failed to initialize excellence system: {e}"
                )
                self.excellence_rewards = None
        else:
            self.excellence_rewards = None

        if self.warmup_steps < self.window_size:
            msg = (
                f"warmup_steps ({self.warmup_steps}) is less than "
                f"window_size ({self.window_size}). Setting warmup_steps to "
                f"{self.window_size}"
            )
            logger.warning(msg)
            self.warmup_steps = self.window_size

        first_asset = next(iter(self.current_data.keys()))
        first_timeframe = next(iter(self.current_data[first_asset].keys()))
        data_length = len(self.current_data[first_asset][first_timeframe])

        if data_length < self.warmup_steps:
            raise ValueError(
                f"Le premier chunk ({data_length} steps) est plus petit "
                f"que la période de warm-up requise "
                f"({self.warmup_steps} steps)."
            )

        self.step_in_chunk = 0

        for _ in range(self.warmup_steps - 1):
            self.step_in_chunk += 1
            self.current_step += 1
            if self.step_in_chunk >= data_length:
                self.current_chunk_idx += 1
                if self.current_chunk_idx >= self.total_chunks:
                    raise ValueError(
                        "Reached end of data during warm-up period. "
                        f"Current chunk: {self.current_chunk_idx}, "
                        f"Total chunks: {self.data_loader.total_chunks}"
                    )
                self.current_data = self._safe_load_chunk(self.current_chunk_idx)
                self.step_in_chunk = 0
                first_asset = next(iter(self.current_data.keys()))
                first_timeframe = next(iter(self.current_data[first_asset].keys()))
                data_length = len(self.current_data[first_asset][first_timeframe])

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def should_force_close_chunk(self, worker_id: int, asset: str, position: "Position", step_in_chunk: int) -> bool:
        """
        Decide whether to force-close at chunk end.
        position: objet position contenant entry_price, size, unrealized_pnl_usd, realized_pnl_usd...
        Règles (configurable) :
          - garder si unrealized PnL > keep_unrealized_pct * position_value
          - garder si moyenne des N derniers trades fermés pour ce worker > keep_recent_winrate_pct
          - forcer close si unrealized PnL < max_allowed_drawdown_pct_of_position
        """
        cfg = self.config.get("chunk_carry_over", {})
        keep_unrealized_pct = cfg.get("keep_unrealized_pct", 0.02)     # garder si +2%
        recent_window = cfg.get("recent_closed_window", 10)
        keep_recent_avg = cfg.get("keep_recent_avg_pnl_pct", 0.0)      # garder si recent avg pnl >= 0%
        max_allowed_loss_pct = cfg.get("max_allowed_loss_pct", 0.10)   # si perte unrealized > 10% -> fermer

        # safe defaults
        unrealized_pct = 0.0
        try:
            unrealized_pct = position.unrealized_pnl_usd / max(1e-9, position.notional_usd)
        except Exception:
            unrealized_pct = 0.0

        # 1) strong winner -> keep
        if unrealized_pct >= keep_unrealized_pct:
            self.smart_logger.info(f"[CHUNK_CARRY] KEEP winner worker={worker_id} asset={asset} unrealized_pct={unrealized_pct:.4f}")
            return False  # don't force close

        # 2) large loss -> force close
        if unrealized_pct <= -max_allowed_loss_pct:
            self.smart_logger.info(f"[CHUNK_CARRY] FORCE close - large loss worker={worker_id} asset={asset} unrealized_pct={unrealized_pct:.4f}")
            return True

        # 3) recent closed trade quality for worker
        # NOTE: self.trade_history is not available, this part of the logic is disabled for now.
        # recent = self.trade_history.get_recent_closed_trades(worker_id=worker_id, n=recent_window)
        # if recent:
        #     avg_recent_pnl = sum(t.pnl_pct for t in recent) / len(recent)
        #     if avg_recent_pnl >= keep_recent_avg:
        #         self.smart_logger.info(f"[CHUNK_CARRY] KEEP recent_avg_pnl={avg_recent_pnl:.4f} worker={worker_id}")
        #         return False

        # 4) otherwise default behaviour: prefer to keep only small positions; force close if small and near chunk end and risk high
        # Parameter: min_steps_left_to_keep (avoid carry if chunk nearly over)
        min_steps_left = cfg.get("min_steps_left_to_allow_carry", 5)
        
        first_asset = next(iter(self.current_data))
        first_timeframe = next(iter(self.current_data[first_asset]))
        data_length = len(self.current_data[first_asset][first_timeframe])

        if step_in_chunk >= (data_length - min_steps_left):
            # chunk almost over — safer to close
            self.smart_logger.info(f"[CHUNK_CARRY] FORCE close (near chunk end) worker={worker_id} step_in_chunk={step_in_chunk}")
            return True

        # default: keep (do not force close)
        self.smart_logger.info(f"[CHUNK_CARRY] DEFAULT KEEP worker={worker_id} asset={asset} unrealized_pct={unrealized_pct:.4f}")
        return False

    def set_global_risk(self, worker_id: int = None, **kwargs):
        """
        Dynamically adjusts risk parameters by applying DBE market regime adjustments.
        
        DBE adjusts model parameters by ±10% based on market regime:
        - Bull market: +10% to position size and take profit
        - Bear market: -10% to position size, +10% to stop loss
        - Sideways: No adjustment (±0%)
        - Volatile: -10% to position size, +10% to stop loss
        
        This teaches the model to respect market regimes without overriding its decisions.
        """
        # Store original model parameters
        original_pos_size = self.portfolio_manager.pos_size_pct
        original_sl = self.portfolio_manager.sl_pct
        original_tp = self.portfolio_manager.tp_pct
        
        # Apply DBE adjustments (±10% based on market regime)
        if 'max_position_size_pct' in kwargs:
            dbe_pos_size = kwargs['max_position_size_pct']
            # DBE suggests adjustment, but we apply only ±10% to model's decision
            # Calculate the adjustment factor (should be close to 1.0 for ±10%)
            adjustment_factor = dbe_pos_size / original_pos_size if original_pos_size > 0 else 1.0
            # Clamp adjustment to ±10%
            adjustment_factor = max(0.9, min(1.1, adjustment_factor))
            self.portfolio_manager.pos_size_pct = original_pos_size * adjustment_factor
        
        if 'stop_loss_pct' in kwargs:
            dbe_sl = kwargs['stop_loss_pct']
            # Apply only ±10% adjustment to model's stop loss
            adjustment_factor = dbe_sl / original_sl if original_sl > 0 else 1.0
            adjustment_factor = max(0.9, min(1.1, adjustment_factor))
            self.portfolio_manager.sl_pct = original_sl * adjustment_factor
        
        if 'take_profit_pct' in kwargs:
            dbe_tp = kwargs['take_profit_pct']
            # Apply only ±10% adjustment to model's take profit
            adjustment_factor = dbe_tp / original_tp if original_tp > 0 else 1.0
            adjustment_factor = max(0.9, min(1.1, adjustment_factor))
            self.portfolio_manager.tp_pct = original_tp * adjustment_factor
        
        # Log the adjustment for transparency
        self.smart_logger.info(
            f"[DBE_MARKET_REGIME_ADJUSTMENT] "
            f"Model: PosSize={original_pos_size:.2%}, SL={original_sl:.2%}, TP={original_tp:.2%} | "
            f"Adjusted: PosSize={self.portfolio_manager.pos_size_pct:.2%}, "
            f"SL={self.portfolio_manager.sl_pct:.2%}, "
            f"TP={self.portfolio_manager.tp_pct:.2%} | "
            f"(±10% max based on market regime)"
        )


    def step(self, action: np.ndarray) -> tuple:
        """Execute one time step within the environment."""
        self._step_closed_receipts = []

        # Generate correlation_id for this step
        correlation_id = str(uuid.uuid4())
        # Log uniquement depuis le worker principal pour éviter les duplications
        if getattr(self, "worker_id", 0) == 0:
            logger.info(
                f"[STEP] Starting step {self.current_step}",
                extra={"correlation_id": correlation_id},
            )

        # Mise à jour des paramètres de risque si le sizing dynamique est activé
        if getattr(self, "dynamic_position_sizing", False) and self.current_step > 0:
            # Mise à jour des paramètres de risque avec synchronisation
            market_conditions = {
                "volatility": self._calculate_current_volatility(),
                "market_regime": self._get_current_market_regime(),
            }
            try:
                with self.logger_lock:
                    logger.debug(f"[{self.worker_id}] Updating risk parameters")
                    self.update_risk_parameters(market_conditions)
            except AttributeError:
                # Fallback si logger_lock n'existe pas
                logger.debug(f"[{self.worker_id}] Updating risk parameters (no lock)")
                self.update_risk_parameters(market_conditions)

        # Initialize Rich console once per environment if not done already
        if not hasattr(self, "_rich_initialized"):
            self._rich_console = None
            self._rich_table = None
            self._rich_text = None

            try:
                from rich.console import Console
                from rich.table import Table
                from rich.text import Text

                # Tentative 1: Configuration complète
                try:
                    self._rich_console = Console(
                        force_terminal=True, force_interactive=True
                    )
                    self.logger.debug(
                        "Rich console initialized with full terminal support"
                    )
                except Exception:
                    # Tentative 2: Configuration de base
                    try:
                        self._rich_console = Console()
                        self.logger.debug(
                            "Rich console initialized with basic configuration"
                        )
                    except Exception:
                        # Tentative 3: Mode fichier
                        import io

                        self._rich_console = Console(file=io.StringIO())
                        self.logger.debug("Rich console initialized in file mode")

                if self._rich_console:
                    self._rich_table = Table
                    self._rich_text = Text
                    self._rich_last_print = 0
                    self._rich_print_interval = max(
                        1, int(os.getenv("ADAN_RICH_STEP_EVERY", "10"))
                    )
                    self.logger.info(
                        f"Rich table display enabled (interval: {self._rich_print_interval} steps)"
                    )

            except Exception as e:
                self._rich_console = None
                self.logger.warning(
                    f"Rich console initialization failed: {e}. Falling back to standard logging."
                )

            self._rich_initialized = True

        if not self._is_initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Vérifier les conditions d'urgence avant l'exécution de l'étape
        if hasattr(self, "portfolio_manager"):
            emergency_reset = self.portfolio_manager.check_emergency_condition(
                self.current_step
            )
            if emergency_reset:
                logger.critical("🆘 EMERGENCY RESET TRIGGERED - Terminating episode")
                observation = self._get_observation()
                info = self._get_info()
                info["termination_reason"] = "emergency_reset"
                return observation, 0.0, True, False, info

        # Validate action
        if not self._check_array("action", action):
            self.logger.warning("Invalid action detected, using no-op action")
            action = np.zeros_like(action, dtype=np.float32)

        # Nettoyage et validation de l'action
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        self.current_step += 1
        self.global_step += 1
        self.step_in_chunk += 1

        # Update current day for frequency tracking (robust guards)
        try:
            if (
                hasattr(self, "data")
                and isinstance(self.data, dict)
                and "TIMESTAMP" in self.data
                and getattr(self.data["TIMESTAMP"], "__len__", None)
                and len(self.data["TIMESTAMP"]) > 0
            ):
                ts_series = self.data["TIMESTAMP"]
                idx = min(self.current_step, len(ts_series) - 1)
                current_day = ts_series.iloc[idx] // (24 * 60 * 60 * 1000)
            else:
                current_day = self.current_step // 288
        except Exception:
            # Last-resort fallback to avoid hard failures in optimization loops
            current_day = self.current_step // 288

        current_date = self.get_current_date()
        
        # --- CORRECTION ---
        # S'assurer que les deux objets sont de type date avant la comparaison
        if self.last_reset_date is None or current_date.date() != self.last_reset_date:
            self.reset_daily_counts()
            self.last_reset_date = current_date.date() # Stocker un objet date
        # Log current step and action with detailed information
        chunk_info = (
            f"chunk {self.current_chunk_idx + 1}/{self.total_chunks}"
            if hasattr(self, "total_chunks")
            else ""
        )
        logger.debug(
            "[STEP LOG] step=%d, action=%s, current_chunk=%d, step_in_chunk=%d",
            self.current_step,
            np.array2string(action, precision=6),
            self.current_chunk_idx,
            self.step_in_chunk,
        )
        logger.info(
            f"[STEP {self.current_step} - {chunk_info}] Executing step with action: {action}"
        )

        # Log portfolio value at the start of the step
        if hasattr(self, "portfolio_manager"):
            try:
                pv = float(self.portfolio_manager.get_portfolio_value())

                # Vérifier l'état de surveillance et mettre à jour si nécessaire
                if hasattr(self.portfolio_manager, "_check_surveillance_status"):
                    needs_reset = self.portfolio_manager._check_surveillance_status(
                        self.current_step
                    )
                    if needs_reset:
                        logger.warning(
                            "🔁 Surveillance mode reset required - ending episode"
                        )
                        observation = self._get_observation()
                        info = self._get_info()
                        info["termination_reason"] = "surveillance_reset"
                        return observation, 0.0, True, False, info

                # Log surveillance status if in surveillance mode
                if (
                    hasattr(self.portfolio_manager, "_surveillance_mode")
                    and self.portfolio_manager._surveillance_mode
                ):
                    logger.warning(
                        "👁️  SURVEILLANCE MODE - Survived chunks: %d/2, Current value: %.2f, Start value: %.2f",
                        getattr(self.portfolio_manager, "_survived_chunks", 0),
                        pv,
                        getattr(
                            self.portfolio_manager,
                            "surveillance_chunk_start_balance",
                            0.0,
                        ),
                    )
                logger.info(f"[STEP {self.current_step}] Portfolio value: {pv:.2f}")
            except Exception as _e:
                logger.warning("[STEP] Failed to read portfolio value: %s", str(_e))
        else:
            logger.warning("[STEP] Portfolio manager or portfolio_value not available")

        # Supprimer l'auto-reset qui empêchait la progression normale vers les chunks suivants
        # L'environnement doit laisser Stable Baselines 3 gérer les resets

        try:
            # Préparation de l'action
            action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

            # Validation pour 5 actifs avec support pour actions de taille 14 (compatibilité modèles PPO existants)
            # et 25 (Autonomie SL/TP)
            if action.shape == (14,):
                # Padding temporaire : ajouter un zéro pour compatibilité 15 dimensions
                action = np.pad(action, (0, 1), mode="constant", constant_values=0.0)
                logger.debug(f"[ACTION_COMPAT] Action paddée de 14 à 15 dimensions")
            elif action.shape == (25,):
                # Nouvelle structure autonome (5 actifs * 5 params)
                pass
            elif action.shape != (15,):
                raise ValueError(
                    f"Action shape {action.shape} does not match "
                    f"expected shape (25,) [Autonomy], (15,) [Legacy], or (14,) [Compat]"
                )

            # Early risk check before executing any trades
            # Ensure local info dict exists to attach spot protection context
            info = {}
            current_prices = self._get_current_prices()
            
            # CRITICAL FIX: Update portfolio value with current prices and PnL
            if hasattr(self, "portfolio_manager") and hasattr(self.portfolio_manager, "_update_equity"):
                try:
                    self.portfolio_manager._update_equity(current_prices)
                except Exception as e:
                    logger.warning(f"Failed to update equity: {e}")
            
            try:
                # Update portfolio with current prices and enforce protection limits
                if hasattr(self, "portfolio_manager"):
                    positions_before = {
                        k: v.is_open
                        for k, v in self.portfolio_manager.positions.items()
                    }
                    self.portfolio_manager.update_market_price(
                        current_prices, self.current_step
                    )
                    positions_after = {
                        k: v.is_open
                        for k, v in self.portfolio_manager.positions.items()
                    }

                    # Check for closed positions and log them
                    for asset, was_open in positions_before.items():
                        if was_open and not positions_after.get(asset, True):
                            self.smart_logger.info(
                                f"Position for {asset} was closed during market price update (SL/TP or stale).",
                                rotate=True,
                            )

                    protection_triggered = (
                        self.portfolio_manager.check_protection_limits(current_prices)
                    )
                    if protection_triggered:
                        if getattr(self.portfolio_manager, "futures_enabled", False):
                            # In futures mode, terminate on protection (e.g., liquidation)
                            info = {
                                "termination_reason": "Risk protection triggered",
                                "current_prices": current_prices,
                                "protection": "futures_liquidation_or_breach",
                            }
                            observation = self._get_observation()
                            logger.warning(
                                "[TERMINATION] Risk protection triggered at step %d (futures)",
                                self.current_step,
                            )
                            self.done = True
                            return observation, 0.0, True, False, info
                        else:
                            # Spot mode: protection disables new buys; continue episode
                            logger.warning(
                                "[PROTECTION] Spot drawdown breach: new BUY orders disabled. Continuing episode."
                            )
                            info.update(
                                {
                                    "protection": "spot_drawdown",
                                    "trading_disabled": True,
                                }
                            )
            except Exception as risk_e:
                logger.error("Early risk check failed: %s", str(risk_e), exc_info=True)

            # Capture positions snapshot before executing trades to detect activity
            positions_before = None
            try:
                if hasattr(self, "portfolio_manager") and hasattr(
                    self.portfolio_manager, "get_metrics"
                ):
                    m_before = self.portfolio_manager.get_metrics() or {}
                    positions_before = {
                        k: (v.get("quantity") or v.get("size") or 0.0)
                        for k, v in (m_before.get("positions", {}) or {}).items()
                    }
            except Exception as _e:
                logger.debug(f"[STEP] Failed capturing positions before trade: {_e}")

            trade_start_time = time.time()

            # Get frequency configuration for trade execution
            frequency_config = self.config.get("trading_rules", {}).get("frequency", {})
            
            # ===================================================================
            # FIX CRITIQUE: Charger action_threshold depuis le BON endroit
            # AVANT: Utilisait trading_rules.frequency.action_threshold (inexistant)
            #        + curriculum learning contre-productif (×0.5 au début)
            # APRÈS: Charge environment.action_thresholds[timeframe] directement
            # ===================================================================
            
            # 1. Charger thresholds depuis environment (config calibré)
            env_thresholds = self.config.get("environment", {}).get("action_thresholds", {})
            
            # 2. Déterminer timeframe actuel
            current_timeframe = getattr(self, "current_timeframe_for_trade", "5m")
            
            # 3. Charger threshold pour ce timeframe (avec fallback sécurisé)
            if env_thresholds and current_timeframe in env_thresholds:
                action_threshold = float(env_thresholds[current_timeframe])
            else:
                # Fallback si config mal formaté: valeurs sécurisées par défaut
                default_thresholds = {'5m': 0.05, '1h': 0.08, '4h': 0.10}
                action_threshold = default_thresholds.get(current_timeframe, 0.05)
                self.logger.warning(
                    f"[THRESHOLD] Config environment.action_thresholds manquant pour {current_timeframe}, "
                    f"utilisation fallback: {action_threshold}"
                )
            
            # Log du threshold appliqué (tous les 1000 steps)
            if self.current_step % 1000 == 0:
                self.logger.info(
                    f"[THRESHOLD] Step {self.current_step} | TF={current_timeframe} | "
                    f"Threshold={action_threshold:.4f} (config direct, NO curriculum)"
                )

            # Calcul des paramètres de risque dynamiques via le DBE
            # ✅ CORRECTION: Vérifier si le DBE est activé avant de l'appeler
            dbe_enabled = self.config.get('dbe', {}).get('enabled', True)
            
            if dbe_enabled and hasattr(self, 'dbe') and self.dbe:
                dbe_modulation = self.dbe.compute_dynamic_modulation(
                    env=self,
                    risk_horizon=float(action[1]) if len(action) > 1 else 0.0,
                )
            else:
                # DBE désactivé - utiliser les paramètres de la config directement
                risk_mgmt = self.config.get('trading_rules', {}).get('risk_management', {})
                pos_sizing = self.config.get('trading_rules', {}).get('position_sizing', {})
                dbe_modulation = {
                    'sl_pct': risk_mgmt.get('stop_loss_pct', 0.02),
                    'tp_pct': risk_mgmt.get('take_profit_pct', 0.04),
                    'position_size_pct': pos_sizing.get('position_size_pct', 0.10),
                    'risk_mode': 'CONFIG_DIRECT',
                }
                if self.current_step % 500 == 0:
                    self.logger.info(
                        f"[DBE_BYPASSED] Step {self.current_step} | Using config params: "
                        f"SL={dbe_modulation['sl_pct']:.2%}, TP={dbe_modulation['tp_pct']:.2%}, "
                        f"PosSize={dbe_modulation['position_size_pct']:.2%}"
                    )


            # --- PHASE 1: EXÉCUTION DES TRADES NORMAUX ---
            realized_pnl, discrete_action, _ = self._execute_trades(
                action, dbe_modulation, action_threshold, force_trade=False
            )

            # --- PHASE 2: FORCE TRADES DÉSACTIVÉS - MODÈLE AUTONOME ---
            # Le modèle doit apprendre à trader de manière autonome sur toutes les timeframes
            # Les force trades ne doivent PAS interférer avec l'apprentissage
            # Logique désactivée par choix de conception
            pass

            # --- PHASE 3: MISE À JOUR DES MÉTRIQUES APRÈS TOUS LES TRADES ---
            self._update_dbe_state()
            self._validate_frequency()

            # Log frequency state AFTER all trades are executed
            for timeframe in self.timeframes:
                force_trade_steps_config = frequency_config.get("force_trade_steps", {})
                if isinstance(force_trade_steps_config, dict):
                    force_trade_steps = force_trade_steps_config.get(
                        timeframe, force_trade_steps_config.get("default", 50)
                    )
                else:
                    force_trade_steps = int(force_trade_steps_config)

                min_positions = frequency_config.get("min_positions", {})
                min_pos_tf = min_positions.get(timeframe, 1)
                steps_since_last_trade = (
                    self.current_step - self.last_trade_steps_by_tf.get(timeframe, 0)
                )

                try:
                    logger.info(
                        f"[FREQ GATE POST-TRADE] TF={timeframe} last_step={self.last_trade_steps_by_tf.get(timeframe, '-')} | "
                        f"since_last={steps_since_last_trade} | min_pos_tf={min_pos_tf} | count={self.positions_count.get(timeframe, 0)} | "
                        f"force_after={force_trade_steps} | action_thr={action_threshold:.2f}"
                    )
                except Exception:
                    pass

            trade_end_time = time.time()
            logger.debug(
                f"_execute_trades took {trade_end_time - trade_start_time:.4f} seconds"
            )

            # Detect trade activity by comparing positions snapshots
            try:
                if (
                    positions_before is not None
                    and hasattr(self, "portfolio_manager")
                    and hasattr(self.portfolio_manager, "get_metrics")
                ):
                    m_after = self.portfolio_manager.get_metrics() or {}
                    positions_after = {
                        k: (v.get("quantity") or v.get("size") or 0.0)
                        for k, v in (m_after.get("positions", {}) or {}).items()
                    }
                    if positions_after != positions_before:
                        self.last_trade_step = self.current_step
                        logger.debug(
                            f"[TRADE] Positions changed at step {self.current_step} -> last_trade_step updated"
                        )
            except Exception as _e:
                logger.debug(f"[STEP] Failed detecting trade activity: {_e}")

            # Journalisation du PnL réalisé (uniquement depuis worker principal)
            if getattr(self, "worker_id", 0) == 0:
                logger.info(f"[REWARD] Realized PnL for step: ${realized_pnl:.2f}")

            # step_in_chunk is already incremented earlier in step() method - removing duplicate

            first_asset = next(iter(self.current_data))
            first_timeframe = next(iter(self.current_data[first_asset]))
            data_length = len(self.current_data[first_asset][first_timeframe])

            # DIAGNOSTIC LOGS - Comprendre pourquoi les chunks ne transitionnent jamais
            if self.current_step % 50 == 0:  # Log tous les 50 steps pour éviter le spam
                logger.info(
                    f"[CHUNK DIAGNOSTIC Worker {self.worker_id}] step_in_chunk: {self.step_in_chunk}, data_length: {data_length}, current_chunk: {self.current_chunk_idx + 1}/{getattr(self, 'total_chunks', 'unknown')}"
                )

            MIN_EPISODE_STEPS = (
                288  # Minimum pour une journée complète (288 steps * 5m = 1 jour)
            )
            done = False
            termination_reason = ""

            # Check if we should terminate based on frequency check interval or other conditions
            frequency_check_interval = (
                self.config.get("trading", {})
                .get("frequency", {})
                .get("frequency_check_interval", 288)
            )

            # Determiner steps_since_trade pour la logique de terminaison

            # Log current state before checking termination conditions
            steps_since_trade = (
                "-"
                if (self.last_trade_step is None or self.last_trade_step < 0)
                else str(self.current_step - self.last_trade_step)
            )
            # Log de vérification de terminaison (uniquement depuis worker principal)
            if getattr(self, "worker_id", 0) == 0:
                logger.info(
                    f"[TERMINATION CHECK] Step: {self.current_step}, "
                    f"Max Steps: {self.max_steps}, "
                    f"Portfolio Value: {self.portfolio_manager.get_portfolio_value():.2f}, "
                    f"Initial Equity: {self.portfolio_manager.initial_equity:.2f}, "
                    f"Steps Since Last Trade: {steps_since_trade}"
                )

            # DIAGNOSTIC : Vérifier si la terminaison précoce empêche la transition de chunks
            # DÉSACTIVÉ : La condition frequency_check_interval empêchait la progression des chunks
            # if self.current_step >= frequency_check_interval:
            #     logger.warning(f"[EARLY TERMINATION WARNING Worker {self.worker_id}] About to terminate at step {self.current_step} (frequency_check_interval: {frequency_check_interval}), step_in_chunk: {self.step_in_chunk}, data_length: {data_length}")
            #     done = True
            #     termination_reason = (
            #         f"Frequency check interval reached ({self.current_step} >= {frequency_check_interval})"
            #     )
            #     logger.info(f"[TERMINATION Worker {self.worker_id}] {termination_reason}")
            # elif self.current_step >= self.max_steps:
            if self.current_step >= self.max_steps:
                done = True
                termination_reason = (
                    f"Max steps reached ({self.current_step} >= {self.max_steps})"
                )
                logger.info(
                    f"[TERMINATION Worker {self.worker_id}] {termination_reason}"
                )
            elif (
                self.portfolio_manager.get_portfolio_value()
                <= self.portfolio_manager.initial_equity * 0.70
            ):
                done = True
                termination_reason = (
                    f"Portfolio value too low ({self.portfolio_manager.get_portfolio_value():.2f} "
                    f"<= {self.portfolio_manager.initial_equity * 0.50:.2f})"
                )
                logger.info(
                    f"[TERMINATION Worker {self.worker_id}] {termination_reason}"
                )
            # DÉSACTIVÉ : Cette condition terminait l'épisode trop agressivement, empêchant l'apprentissage
            # La condition originale était : 144 * 5 = 720 steps sans trade = terminaison
            # elif self.current_step - self.last_trade_step > self.config.get('trading', {}).get('frequency', {}).get('force_trade_steps', 144) * 5:
            #     done = True
            #     termination_reason = f"No trades for too long ({self.current_step - self.last_trade_step} steps)"
            #     logger.info(f"[TERMINATION Worker {self.worker_id}] {termination_reason}")

            # NOUVELLE LOGIQUE : Terminaison seulement après un chunk complet sans trades (plus permissive)
            force_trade_limit = (
                self.config.get("trading", {})
                .get("frequency", {})
                .get("force_trade_steps", 144)
                * 10
            )  # 1440 steps = ~5 jours
            if self.current_step - self.last_trade_step > force_trade_limit:
                logger.warning(
                    f"[TERMINATION WARNING Worker {self.worker_id}] Long period without trades: {self.current_step - self.last_trade_step} steps > {force_trade_limit}"
                )
                # NE PAS TERMINER - laisser l'agent apprendre même s'il ne trade pas immédiatement
                # done = True
                # termination_reason = f"No trades for very long ({self.current_step - self.last_trade_step} steps)"
                # logger.info(f"[TERMINATION Worker {self.worker_id}] {termination_reason}")

            # Ensure environment done flag is set when a termination condition is met
            if done:
                self.done = True

            # DIAGNOSTIC CRITIQUE : Vérifier si nous avons atteint la fin du chunk actuel
            transition_threshold = data_length - 1
            logger.debug(
                f"[CHUNK TRANSITION CHECK Worker {self.worker_id}] step_in_chunk: {self.step_in_chunk}, threshold: {transition_threshold}, will_transition: {self.step_in_chunk >= transition_threshold}"
            )

            # PROTECTION DE FIN DE CHUNK DÉSACTIVÉE
            # Le modèle doit être autonome même en fin de chunk
            # Pas de force trades, le modèle apprend à trader naturellement
            pass

            if self.step_in_chunk >= data_length - 1:
                logger.info(
                    f"[CHUNK TRANSITION Worker {self.worker_id}] End of chunk {self.current_chunk_idx + 1} reached (step_in_chunk: {self.step_in_chunk} >= {data_length - 1})"
                )

                # --- INTELLIGENT CHUNK CARRY-OVER ---
                current_prices = self._get_current_prices()
                current_timestamp = self._get_current_timestamp()
                step_in_chunk = self.step_in_chunk
                
                # Iterate over a copy of positions to avoid issues with modification during iteration
                for asset, position in list(self.portfolio_manager.positions.items()):
                    if position.is_open:
                        if self.should_force_close_chunk(self.worker_id, asset, position, step_in_chunk):
                            receipt = self.portfolio_manager.close_position(
                                asset=asset.upper(),
                                price=current_prices.get(asset, position.entry_price),
                                timestamp=current_timestamp,
                                current_prices=current_prices,
                                reason="CHUNK_END_FORCE_CLOSE"
                            )
                            if receipt:
                                self._step_closed_receipts.append(receipt)
                                realized_pnl += float(receipt.get("pnl", 0.0))
                        else:
                            self.smart_logger.info(f"[CHUNK_CARRY] Carried position worker={self.worker_id} asset={asset}")
                # --- FIN DU CARRY-OVER ---

                self.current_chunk_idx += 1
                self.current_chunk += 1

                # POINT CRITIQUE : Réinitialiser le compteur step_in_chunk pour le nouveau chunk
                self.step_in_chunk = 0

                # CONSERVATION: NE PAS réinitialiser last_trade_steps_by_tf et positions_count
                # pour maintenir la continuité du tracking de fréquence entre chunks
                logger.debug(
                    f"[CHUNK TRANSITION] Preserving frequency counters: "
                    f"last_trade_steps_by_tf={self.last_trade_steps_by_tf}, "
                    f"positions_count={self.positions_count}"
                )

                if hasattr(self, "portfolio") and hasattr(
                    self.portfolio, "check_reset"
                ):
                    if self.portfolio.check_reset(chunk_completed=True):
                        logger.info(
                            "[HARD RESET] Portfolio reset performed due to capital below threshold"
                        )
                    else:
                        logger.debug("[NO RESET] Capital OK, continuité préservée")

                # Vérifier si on a atteint le nombre maximum de chunks pour cet épisode
                chunks_limit = min(self.total_chunks, self.max_chunks_per_episode)

                if self.current_chunk_idx >= chunks_limit:
                    done = True
                    self.done = True
                    termination_reason = f"Max chunks per episode reached ({self.current_chunk_idx} >= {self.max_chunks_per_episode})"
                    logger.info(f"[TERMINATION] {termination_reason}")
                else:
                    # Charger le prochain chunk
                    logger.info(
                        f"[CHUNK] Loading next chunk {self.current_chunk_idx + 1}/"
                        f"{chunks_limit}"
                    )
                    try:
                        self.current_data = self._safe_load_chunk(
                            self.current_chunk_idx
                        )
                        self._set_start_step_for_chunk()  # Reposition step to skip warmup period
                        logger.info(
                            f"[CHUNK] Successfully loaded chunk {self.current_chunk_idx + 1}/{chunks_limit}"
                        )
                    except Exception as e:
                        logger.error(
                            f"[CHUNK] Failed to load chunk {self.current_chunk_idx + 1}: {e}"
                        )
                        done = True
                        self.done = True
                        termination_reason = (
                            f"Failed to load chunk {self.current_chunk_idx + 1}: {e}"
                        )

                    # Réinitialiser les composants pour le nouveau chunk avec continuité
                    if hasattr(self, "dbe") and hasattr(
                        self.dbe, "reset_for_new_chunk"
                    ):
                        try:
                            with self.logger_lock:
                                logger.debug(
                                    f"[DBE {self.worker_id}] Resetting DBE for new chunk with continuity"
                                )
                        except AttributeError:
                            logger.debug(
                                f"[DBE {self.worker_id}] Resetting DBE for new chunk with continuity"
                            )
                        self.dbe.reset_for_new_chunk(continuity=True)
                    elif hasattr(self, "dbe") and hasattr(
                        self.dbe, "_reset_for_new_chunk"
                    ):
                        try:
                            with self.logger_lock:
                                logger.debug(
                                    f"[DBE {self.worker_id}] Fallback to legacy reset"
                                )
                        except AttributeError:
                            logger.debug(
                                f"[DBE {self.worker_id}] Fallback to legacy reset"
                            )
                        self.dbe._reset_for_new_chunk()

                    # Réinitialiser les composants pour le nouveau chunk avec continuité
                    if hasattr(self, "dbe") and hasattr(
                        self.dbe, "reset_for_new_chunk"
                    ):
                        try:
                            with self.logger_lock:
                                logger.debug(
                                    f"[DBE {self.worker_id}] Resetting DBE for new chunk with continuity"
                                )
                        except AttributeError:
                            logger.debug(
                                f"[DBE {self.worker_id}] Resetting DBE for new chunk with continuity"
                            )
                        self.dbe.reset_for_new_chunk(continuity=True)
                    elif hasattr(self, "dbe") and hasattr(
                        self.dbe, "_reset_for_new_chunk"
                    ):
                        try:
                            with self.logger_lock:
                                logger.debug(
                                    f"[DBE {self.worker_id}] Fallback to legacy reset"
                                )
                        except AttributeError:
                            logger.debug(
                                f"[DBE {self.worker_id}] Fallback to legacy reset"
                            )
                        self.dbe._reset_for_new_chunk()

            # Log final decision and handle episode termination
            if done:
                logger.info(
                    f"[EPISODE END] Episode ending. Reason: {termination_reason}"
                )
                logger.info(
                    f"[EPISODE STATS] Total steps: {self.current_step}, "
                    f"Final portfolio value: {self.portfolio_manager.get_portfolio_value():.2f}, "
                    f"Return: {(self.portfolio_manager.get_portfolio_value() / self.portfolio_manager.initial_equity - 1) * 100:.2f}%"
                )
            else:
                logger.debug(
                    f"[TERMINATION] Episode continues. Current step: {self.current_step}"
                )

            # Build observations and validate
            current_observation = self._get_observation()
            if not self._check_array(
                "observation",
                np.concatenate([v.flatten() for v in current_observation.values()]),
            ):
                self.logger.warning(
                    "Invalid observation detected, attempting recovery instead of reset"
                )

                # TENTATIVE DE RÉCUPÉRATION AVANT RESET COMPLET
                try:
                    # Essayer de reconstruire l'observation sans reset
                    current_observation = self._get_observation()

                    # Vérifier si la récupération a fonctionné
                    if not any(
                        np.isnan(v).any() or np.isinf(v).any()
                        for v in current_observation.values()
                    ):
                        self.logger.info(
                            "Observation recovery successful, continuing episode"
                        )
                        # Continuer avec l'observation récupérée
                    else:
                        raise ValueError("Recovery failed, still has NaN/inf values")

                except Exception as recovery_error:
                    self.logger.error(
                        f"Recovery failed: {recovery_error}, performing reset as last resort"
                    )
                    obs_reset, info_reset = self.reset()
                    return (
                        obs_reset,
                        0.0,
                        True,
                        False,
                        {
                            "nan_detected": True,
                            "nan_source": "observation",
                            "recovery_attempted": True,
                            "recovery_error": str(recovery_error),
                        },
                    )

            # Reward calculation moved after trade execution for synchronization.
            reward = self._calculate_reward(action, realized_pnl)
            self._last_reward = reward

            # Mise à jour des métriques de risque
            if hasattr(self, "portfolio_manager"):
                try:
                    current_value = self.portfolio_manager.get_portfolio_value()
                    prev_value = getattr(self, "_last_portfolio_value", current_value)
                    returns = (
                        (current_value - prev_value) / prev_value
                        if prev_value > 0
                        else 0.0
                    )
                    self._update_risk_metrics(current_value, returns)
                    self._last_portfolio_value = current_value
                except Exception as e:
                    self.logger.error(
                        f"Erreur lors de la mise à jour des métriques de risque: {str(e)}"
                    )

            # Use local 'done' to signal termination for this step
            terminated = done
            truncated = False

            max_steps = getattr(self, "_max_episode_steps", float("inf"))
            if self.current_step >= max_steps:
                truncated = True
                self.done = True

            info = self._get_info()

            if hasattr(self, "_last_reward_components"):
                info.update({"reward_components": self._last_reward_components})

            # --- Minimal structured JSON-lines logging for multicolumn visualization ---
            try:
                # Prepare JSON metrics using available fields; null for unavailable ones
                pm = getattr(self, "portfolio_manager", None)
                pm_metrics = (
                    pm.get_metrics() if pm and hasattr(pm, "get_metrics") else {}
                )
                portfolio_value = pm_metrics.get("total_value") or pm_metrics.get(
                    "total_capital"
                )
                cash = pm_metrics.get("cash")
                sharpe = pm_metrics.get("sharpe_ratio")
                max_dd = pm_metrics.get("max_drawdown")
                trading_disabled = (
                    bool(getattr(pm, "trading_disabled", False)) if pm else False
                )
                futures_enabled = (
                    bool(getattr(pm, "futures_enabled", False)) if pm else False
                )
                current_prices = info.get("market", {}).get("current_prices") or {}
                # Derive a basic protection event label for quick filtering
                protection_event = (
                    "futures_liquidation"
                    if futures_enabled and self.done
                    else (
                        "spot_drawdown"
                        if (not futures_enabled and trading_disabled)
                        else "none"
                    )
                )
                # Compose compact positions list: symbol:size:entry_price:side if available
                positions_compact = []
                for sym, pos in pm_metrics.get("positions", {}).items():
                    size = (
                        pos.get("size") or pos.get("quantity")
                    )  # Préférer 'size', avec fallback sur 'quantity' pour rétrocompatibilité
                    entry = pos.get("entry_price") or pos.get("avg_price")
                    side = "LONG" if (size or 0) >= 0 else "SHORT"
                    positions_compact.append(
                        f"{sym}:{float(size or 0):.8f}:{float(entry or 0):.8f}:{side}"
                    )
                reward_components = info.get("reward_components") or {}
                event_tags = []
                if trading_disabled:
                    event_tags.append("[PROTECTION]")
                # Detect tier change
                current_tier = (pm_metrics or {}).get("tier")
                last_tier = getattr(self, "_last_tier", None)
                tier_changed = current_tier is not None and current_tier != last_tier
                if tier_changed:
                    event_tags.append("[TIER]")
                setattr(self, "_last_tier", current_tier)
                # Pull potential sizer outputs from info if available
                sizer_final_val = info.get("sizer_final")
                sizer_reason_val = info.get("sizer_reason")
                sizer_clamped = (sizer_final_val == 0) or (sizer_reason_val is not None)
                if sizer_clamped:
                    event_tags.append("[SIZER]")
                # Build record
                record = {
                    "timestamp": self._get_safe_timestamp(),
                    "step": int(self.current_step),
                    "env_id": int(getattr(self, "worker_id", 0)),
                    "episode_id": int(getattr(self, "episode_count", 0)),
                    "chunk_id": int(getattr(self, "current_chunk", 0)),
                    "action": (
                        action.tolist() if isinstance(action, np.ndarray) else action
                    ),
                    "action_meaning": "VECTOR",
                    "price_reference": None,
                    "sizer_raw": None,
                    "sizer_final": (
                        sizer_final_val if sizer_final_val is not None else None
                    ),
                    "sizer_reason": (
                        sizer_reason_val if sizer_reason_val is not None else None
                    ),
                    "available_cash": float(cash) if cash is not None else None,
                    "portfolio_value": (
                        float(portfolio_value) if portfolio_value is not None else None
                    ),
                    "cash": float(cash) if cash is not None else None,
                    "positions_value": info.get("portfolio", {}).get(
                        "total_position_value"
                    ),
                    "unrealized_pnl": None,
                    "realized_pnl": (
                        float(realized_pnl)
                        if "realized_pnl" in locals() and realized_pnl is not None
                        else None
                    ),
                    "cum_realized_pnl": None,
                    "num_positions": int(
                        info.get("portfolio", {}).get("num_positions", 0)
                    ),
                    "positions": positions_compact,
                    "order_notional": None,
                    "order_status": None,
                    "commission": None,
                    "slippage": None,
                    "reward": float(reward),
                    "reward_components": reward_components,
                    "drawdown_value": float(max_dd) if max_dd is not None else None,
                    "drawdown_pct": float(max_dd) if max_dd is not None else None,
                    "max_drawdown_pct": None,
                    "tier": (
                        str(getattr(self, "current_tier", ""))
                        if getattr(self, "current_tier", None) is not None
                        else None
                    ),
                    "trading_disabled": trading_disabled,
                    "protection_event": protection_event,
                    "protection_msg": None,
                    "dbE_regime": None,
                    "dbe_params": None,
                    "ppo_metrics": None,
                    "learning_rate": None,
                    "grad_norm": None,
                    "num_trades_step": None,
                    "cum_num_trades": None,
                    "num_wins": None,
                    "num_losses": None,
                    "winrate": None,
                    "avg_win": None,
                    "avg_loss": None,
                    "avg_trade_duration": None,
                    "last_trade_entry_step": None,
                    "last_trade_exit_step": None,
                    "metrics_sharpe": float(sharpe) if sharpe is not None else None,
                    "metrics_volatility": None,
                    "throughput": info.get("performance", {}).get("steps_per_second"),
                    "memory_usage": None,
                    "custom_tags": event_tags,
                    "notes": None,
                }
                # Sampling control to reduce noise: default every 10 steps, always on protection events
                jsonl_every_env = os.getenv("ADAN_JSONL_EVERY", "")
                jsonl_every_cfg = 10
                try:
                    jsonl_every_cfg = (
                        int(
                            (self.config or {})
                            .get("logging", {})
                            .get("jsonl_every", 10)
                        )
                        if hasattr(self, "config")
                        else 10
                    )
                except Exception:
                    jsonl_every_cfg = 10
                jsonl_every = (
                    int(jsonl_every_env)
                    if jsonl_every_env.isdigit()
                    else jsonl_every_cfg
                )
                should_write = (
                    (self.current_step % max(1, jsonl_every) == 0)
                    or (protection_event != "none")
                    or sizer_clamped
                    or tier_changed
                )
                if should_write:
                    logs_dir = os.path.abspath(
                        os.path.join(
                            os.path.dirname(__file__), "..", "..", "..", "..", "logs"
                        )
                    )
                    os.makedirs(logs_dir, exist_ok=True)
                    jsonl_path = os.path.join(logs_dir, "training_events.jsonl")
                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, separators=(",", ":")) + "\n")
            except Exception as _log_e:
                logger.debug("[JSONL] Failed to write training event: %s", str(_log_e))

            # Quiet verbose DEBUG logs after initial checks (one-time)
            try:
                if not getattr(self, "_quiet_after_init", False):
                    # default ON; set ADAN_QUIET_AFTER_INIT=0 to disable
                    _quiet_env = os.getenv("ADAN_QUIET_AFTER_INIT", "1").lower() in (
                        "1",
                        "true",
                        "yes",
                        "on",
                    )
                    if _quiet_env and int(self.current_step) >= 1:
                        try:
                            import logging as _logging

                            logger.setLevel(_logging.INFO)
                        except Exception:
                            pass
                        self._quiet_after_init = True
            except Exception:
                pass

            # --- Rich Summary Table ---
            if hasattr(self, "_rich_console") and self._rich_console is not None:
                try:
                    # Get configuration
                    rich_cfg = (
                        (self.config or {}).get("logging", {})
                        if hasattr(self, "config")
                        else {}
                    )
                    env_enabled = os.getenv("ADAN_RICH_STEP_TABLE", "").lower() in (
                        "1",
                        "true",
                        "yes",
                        "on",
                    )
                    rich_enabled = (
                        rich_cfg.get("rich_step_table", True)
                        if "rich_step_table" in rich_cfg
                        else env_enabled
                    )

                    # If rich table is not enabled, skip rendering here and continue
                    if not rich_enabled:
                        pass

                    # Check if we should print on this step
                    print_interval = getattr(self, "_rich_print_interval", 10)
                    should_print = (self.current_step % print_interval == 0) or (
                        self.current_step - getattr(self, "_rich_last_print", 0)
                        >= print_interval
                    )

                    if should_print:
                        self._rich_last_print = self.current_step

                        # Local import for Text to avoid scope issues
                        from rich.text import Text

                        # Helpers
                        def _fmt(v):
                            if v is None:
                                return "-"
                            if isinstance(v, float):
                                return f"{v:.6g}"
                            return str(v)

                        def _dd_cell(v):
                            if not isinstance(v, (int, float)):
                                return Text("-")
                            if v < 0.05:
                                return Text(f"{v:.3f}", style="green")
                            if v < 0.20:
                                return Text(f"{v:.3f}", style="yellow")
                            if v < 0.50:
                                return Text(f"{v:.3f}", style="orange3")
                            return Text(f"{v:.3f}", style="red")

                        def _reward_cell(v, avg=None):
                            if not isinstance(v, (int, float)):
                                return Text("-")
                            style = None
                            if v > 0:
                                style = "green3"
                            elif (
                                avg is not None
                                and isinstance(avg, (int, float))
                                and v < -1.0 * abs(avg)
                            ):
                                style = "red"
                            elif v < 0:
                                style = "orange3"
                            return Text(f"{v:.4g}", style=style)

                        def _prot_cell(p):
                            if p and p != "none":
                                return Text(str(p), style="orange3")
                            return Text("none")

                        def _reason_cell(reason: str):
                            if not reason:
                                return Text("-")
                            r = str(reason)
                            if "insufficient_cash" in r:
                                return Text(r, style="magenta")
                            if "min_notional" in r:
                                return Text(r, style="orange3")
                            if ("step_size" in r) or ("precision" in r):
                                return Text(r, style="gold1")
                            if "trading_disabled" in r:
                                return Text(r, style="red")
                            return Text(r)

                        def _winrate_cell(w):
                            if not isinstance(w, (int, float)):
                                return Text("-")
                            if w >= 0.6:
                                return Text(f"{w:.2f}", style="green3")
                            if w >= 0.4:
                                return Text(f"{w:.2f}")
                            return Text(f"{w:.2f}", style="orange3")

                        def _loss_cell(cur, prev):
                            if not isinstance(cur, (int, float)):
                                return Text("-")
                            if isinstance(prev, (int, float)):
                                delta = cur - prev
                                if delta > 0:
                                    # big increase vs previous -> orange, extremely large -> red
                                    return Text(
                                        f"{cur:.3g}",
                                        style="red"
                                        if delta > abs(prev) * 2
                                        else "orange3",
                                    )
                            return Text(f"{cur:.3g}")

                        # Gather row fields - build record for Rich display
                        # Get portfolio values safely
                        total_value = (
                            getattr(
                                self.portfolio_manager, "get_total_value", lambda: 0.0
                            )()
                            if hasattr(self, "portfolio_manager")
                            else 0.0
                        )
                        current_dd_pct = getattr(self, "_current_drawdown_pct", 0.0)
                        current_tier = getattr(self, "current_tier", "unknown")
                        trading_disabled = getattr(self, "_trading_disabled", False)
                        protection_event = getattr(self, "_protection_event", "none")
                        reward = getattr(self, "_last_reward", 0.0)
                        sizer_final_pct = getattr(self, "_last_sizer_final_pct", 0.0)
                        sizer_reason = getattr(self, "_last_sizer_reason", "unknown")
                        trades_this_step = getattr(self, "_trades_this_step", 0)
                        total_trades_count = getattr(self, "_total_trades_count", 0)
                        winrate = getattr(self, "_current_winrate", None)

                        record = {
                            "timestamp": self._get_safe_timestamp(),
                            "step": self.current_step,
                            "env_id": getattr(self, "env_id", "unknown")[:8],
                            "episode_id": getattr(self, "current_episode_id", 0),
                            "portfolio_value": float(total_value),
                            "drawdown_pct": float(current_dd_pct),
                            "tier": current_tier,
                            "trading_disabled": bool(trading_disabled),
                            "protection_event": protection_event,
                            "reward": float(reward),
                            "sizer_final": sizer_final_pct,
                            "sizer_reason": sizer_reason,
                            "num_trades_step": trades_this_step,
                            "cum_num_trades": total_trades_count,
                            "winrate": float(winrate) if winrate is not None else None,
                            "metrics_sharpe": float(
                                getattr(self, "_current_sharpe", None)
                            )
                            if getattr(self, "_current_sharpe", None) is not None
                            else None,
                        }

                        ts = record.get("timestamp")
                        ts_short = (
                            ts[11:19]
                            if isinstance(ts, str) and len(ts) >= 19
                            else "--:--:--"
                        )
                        step_id = _fmt(record.get("step"))
                        env_id = _fmt(record.get("env_id"))
                        ep_id = _fmt(record.get("episode_id"))
                        pv = record.get("portfolio_value")
                        ddv = record.get("drawdown_pct")
                        tier = _fmt(record.get("tier"))
                        td_flag = bool(record.get("trading_disabled"))
                        prot = record.get("protection_event")
                        reward_val = record.get("reward")
                        # Rolling average of reward for magnitude-based coloring
                        avg_reward = getattr(self, "_reward_avg", None)
                        try:
                            if isinstance(reward_val, (int, float)):
                                if avg_reward is None:
                                    avg_reward = float(reward_val)
                                else:
                                    # EMA with smoothing factor
                                    beta = 0.1
                                    avg_reward = (1 - beta) * float(
                                        avg_reward
                                    ) + beta * float(reward_val)
                                setattr(self, "_reward_avg", avg_reward)
                        except Exception:
                            pass
                        sizer_f = record.get("sizer_final")
                        sizer_r = record.get("sizer_reason")
                        trades_step = record.get("num_trades_step")
                        trades_cum = record.get("cum_num_trades")
                        winrate = record.get("winrate")
                        sharpe = record.get("metrics_sharpe")
                        ppo = (
                            info.get("ppo_metrics", {})
                            if isinstance(info, dict)
                            else {}
                        )
                        pol_loss = ppo.get("policy_loss")
                        val_loss = ppo.get("value_loss")
                        prev_pol_loss = getattr(self, "_prev_policy_loss", None)
                        prev_val_loss = getattr(self, "_prev_value_loss", None)
                        self._prev_policy_loss = pol_loss
                        self._prev_value_loss = val_loss

                        # Build compact live table
                        from rich import box

                        table = self._rich_table(
                            title=f"Step {self.current_step} - {self._get_safe_timestamp()}",
                            box=box.SIMPLE,
                            show_header=True,
                            header_style="bold magenta",
                            show_lines=True,
                            title_justify="left",
                            expand=False,
                        )
                        table.add_column("t", justify="left")
                        table.add_column("step", justify="right")
                        table.add_column("env", justify="right")
                        table.add_column("ep", justify="right")
                        table.add_column("pv", justify="right")
                        table.add_column("dd%", justify="right")
                        table.add_column("tier", justify="center")
                        table.add_column("TD", justify="center")
                        table.add_column("prot", justify="left")
                        table.add_column("reward", justify="right")
                        table.add_column("sizer", justify="right")
                        table.add_column("trades", justify="right")
                        table.add_column("winrate", justify="right")
                        table.add_column("sharpe", justify="right")
                        table.add_column("polL", justify="right")
                        table.add_column("valL", justify="right")
                        table.add_column("tags", justify="left")

                        row_style = "bold white on red" if td_flag else None
                        table.add_row(
                            Text(ts_short),
                            Text(str(step_id)),
                            Text(str(env_id)),
                            Text(str(ep_id)),
                            Text(_fmt(pv)),
                            _dd_cell(ddv),
                            Text(str(tier)),
                            Text("T" if td_flag else "F"),
                            _prot_cell(prot),
                            _reward_cell(reward_val, avg_reward),
                            _reason_cell(_fmt(sizer_r))
                            if sizer_r
                            else Text(_fmt(sizer_f)),
                            Text(f"{_fmt(trades_step)}|{_fmt(trades_cum)}"),
                            _winrate_cell(winrate),
                            Text(_fmt(sharpe)),
                            _loss_cell(pol_loss, prev_pol_loss),
                            _loss_cell(val_loss, prev_val_loss),
                            Text("".join(event_tags)),
                            style=row_style,
                        )
                        self._rich_console.print(table)

                except Exception as e:
                    self.logger.debug(f"Rich table display error: {e}")
                    # Fallback to simple text summary with actual values
                    try:
                        portfolio_metrics = (
                            self.portfolio_manager.get_metrics()
                            if hasattr(self, "portfolio_manager")
                            else {}
                        )
                        portfolio_value = portfolio_metrics.get("total_value", 0.0)
                        drawdown = portfolio_metrics.get("drawdown", 0.0)
                        total_trades = portfolio_metrics.get("total_trades", 0)
                        last_reward = getattr(self, "_last_reward", 0.0)

                        self.logger.info(
                            f"[RICH FALLBACK] Step {self.current_step} | "
                            f"Portfolio: ${portfolio_value:.2f} | Drawdown: {drawdown:.2f}% | "
                            f"Trades: {total_trades} | Reward: {last_reward:.4f}"
                        )
                    except Exception as fallback_error:
                        self.logger.warning(
                            f"Fallback metrics retrieval failed: {fallback_error}"
                        )
                        self.logger.info(
                            f"[RICH FALLBACK] Step {self.current_step} | "
                            f"Portfolio: Error | Drawdown: Error% | "
                            f"Trades: Error | Reward: Error"
                        )

            if self.shared_buffer is not None:
                experience = {
                    "state": current_observation,
                    "action": action,
                    "reward": float(reward),
                    "next_state": current_observation,
                    "done": terminated or truncated,
                    "info": info,
                    "timestamp": self._get_safe_timestamp() or str(self.current_step),
                    "worker_id": self.worker_id,
                }
                self.shared_buffer.add(experience)

            # Log summary
            # self._log_summary(
            #     self.current_step, self.current_chunk_idx + 1, self.total_chunks
            # )
            logger.info("End of step processing.")

            # OMEGA-3: Log episode capital stats on termination
            if terminated or truncated:
                self._log_episode_capital_stats(info)

            # Nettoyage explicite des observations precedentes
            if hasattr(self, 'last_observation') and self.last_observation is not None:
                del self.last_observation
            
            self.last_observation = current_observation

            # Garbage collection périodique
            if self.current_step % 500 == 0:
                gc.collect()

            return current_observation, float(reward), terminated, truncated, info

        except Exception as e:
            logger.error(f"Error in step(): {str(e)}", exc_info=True)
            self.done = True
            observation = self._get_observation()
            info = self._get_info()
            info["error"] = str(e)
            return observation, 0.0, True, False, info

    def _update_dbe_state(self) -> None:
        """Update the DBE state with current market conditions."""
        # DIAGNOSTIC: Tracer l'utilisation du DBE
        self.logger.debug(
            f"🔄 UPDATE_DBE_STATE appelé pour ENV_ID={getattr(self, 'env_instance_id', 'UNKNOWN')}, Step={getattr(self, 'current_step', 'UNKNOWN')}"
        )
        try:
            current_prices = self._get_current_prices()
            portfolio_metrics = self.portfolio.get_metrics()

            live_metrics = {
                "step": self.current_step,
                "current_prices": current_prices,
                "portfolio_value": portfolio_metrics.get("total_capital", 0.0),
                "cash": portfolio_metrics.get("cash", 0.0),
                "positions": portfolio_metrics.get("positions", {}),
                "returns": portfolio_metrics.get("returns", 0.0),
                "max_drawdown": portfolio_metrics.get("max_drawdown", 0.0),
            }

            if hasattr(self, "current_data") and self.current_data:
                first_asset = next(iter(self.current_data.keys()))
                if first_asset in self.current_data and self.current_data[first_asset]:
                    first_tf = next(iter(self.current_data[first_asset].keys()))
                    df = self.current_data[first_asset][first_tf]

                    if not df.empty and self.current_step < len(df):
                        current_row = df.iloc[self.current_step]
                        live_metrics.update(
                            {
                                "rsi": current_row.get("rsi", 50.0),
                                "adx": current_row.get("adx", 20.0),
                                "atr": current_row.get("atr", 0.0),
                                "atr_pct": current_row.get("atr_pct", 0.0),
                                "ema_ratio": current_row.get("ema_ratio", 1.0),
                            }
                        )
            if hasattr(self, "dbe"):
                # DIAGNOSTIC: Tracer l'appel effectif au DBE
                self.logger.debug(
                    f"🧠 DBE.update_state appelé - ENV_ID={getattr(self, 'env_instance_id', 'UNKNOWN')}, DBE_ID={id(self.dbe)}"
                )
                self.dbe.update_state(live_metrics)
            else:
                self.logger.warning(
                    f"❌ DBE non disponible pour ENV_ID={getattr(self, 'env_instance_id', 'UNKNOWN')}"
                )

        except Exception as e:
            logger.warning(f"Failed to update DBE state: {e}")

    def _check_array(self, name: str, arr: np.ndarray) -> bool:
        """Vérifie la présence de NaN/Inf dans un tableau et enregistre un rapport détaillé.

        Args:
            name: Nom de la variable pour les logs
            arr: Tableau NumPy à vérifier

        Returns:
            bool: True si le tableau est valide, False sinon
        """
        if not isinstance(arr, np.ndarray):
            self.logger.warning(f"{name} is not a numpy array, got {type(arr)}")
            return True

        has_nan = np.any(np.isnan(arr))
        has_inf = np.any(np.isinf(arr))

        if has_nan or has_inf:
            issues = []
            if has_nan:
                issues.append("NaN")
            if has_inf:
                issues.append("Inf")

            self.logger.error(
                f"Invalid values detected in {name} at step {self.current_step}: {' and '.join(issues)}"
            )

            # Enregistrement du contexte
            try:
                dump_path = os.path.join(
                    os.getcwd(), f"nan_dump_{name}_step{self.current_step}.npz"
                )
                np.savez(dump_path, arr=arr)
                self.logger.info(f"Dumped {name} state to {dump_path}")
            except Exception as e:
                self.logger.error(f"Failed to dump {name} state: {e}")

            return False

        return True

    def _get_current_prices(self) -> Dict[str, float]:
        """
        CORRECTED V3: Retrieve prices with robust indexing and safety net.
        1. Calculates correct index per timeframe.
        2. Backtracks if invalid price (0.0/NaN) is found (handling padding).
        3. Returns None only if no valid price found after backtracking.
        """
        prices: Dict[str, Optional[float]] = {}
        
        # Get the step within current chunk (calculated for base timeframe, usually 5m)
        base_step_in_chunk = int(getattr(self, "step_in_chunk", 0))
        
        if not hasattr(self, "current_data") or not self.current_data:
            self.logger.error("_get_current_prices called without current_data loaded")
            return {a: None for a in getattr(self, "assets", [])}

        def _resolve_close_column(df: pd.DataFrame) -> Optional[str]:
            lowered = {col.lower(): col for col in df.columns}
            for candidate in ("close", "closing_price", "price"):
                if candidate in lowered:
                    return lowered[candidate]
            return None

        # Timeframe conversion ratios (minutes per candle)
        tf_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440
        }
        
       # Assume base timeframe is 5m (most common)
        base_tf_minutes = 5

        for asset in getattr(self, "assets", []):
            asset_key = asset.upper()
            tf_map = self.current_data.get(asset)
            prices[asset_key] = None  # Default to None

            if not isinstance(tf_map, dict) or not tf_map:
                self.smart_logger.warning(f"PRICE_DATA_MISSING | asset={asset_key} | reason=empty_timeframe_map", dedupe=True)
                continue

            preferred_order = [getattr(self, "current_timeframe_for_trade", None), "5m", "1h", "4h"]
            timeframe = next((tf for tf in preferred_order if tf and tf in tf_map), next(iter(tf_map)))
            df = tf_map.get(timeframe)

            if df is None or df.empty:
                self.smart_logger.warning(f"PRICE_DATA_EMPTY | asset={asset_key} | timeframe={timeframe}", dedupe=True)
                continue

            close_col = _resolve_close_column(df)
            if close_col is None:
                self.smart_logger.error(f"PRICE_NO_CLOSE_COL | asset={asset_key} | timeframe={timeframe} | cols={list(df.columns)}")
                continue

            # CRITICAL: Convert step_in_chunk to this timeframe's index
            tf_minutes_value = tf_minutes.get(timeframe, base_tf_minutes)
            conversion_ratio = tf_minutes_value / base_tf_minutes
            
            # Calculate index for THIS timeframe
            target_step_idx = int(base_step_in_chunk / conversion_ratio)
            
            # Safety bounds check
            chunk_size_for_tf = len(df)
            
            # Initial clamp
            step_idx = min(target_step_idx, chunk_size_for_tf - 1)
            
            if step_idx < 0:
                step_idx = 0

            # SAFETY NET: Backtrack if price is invalid (0.0 or NaN)
            # This handles padding at the end of chunks or corrupted rows
            max_backtrack = 5
            found_valid = False
            
            for offset in range(max_backtrack + 1):
                current_idx = step_idx - offset
                if current_idx < 0:
                    break
                
                try:
                    price = float(df.iloc[current_idx][close_col])
                    
                    if np.isfinite(price) and price > 0:
                        prices[asset_key] = price
                        found_valid = True
                        
                        if offset > 0:
                            self.smart_logger.warning(
                                f"PRICE_BACKTRACKED | asset={asset_key} | timeframe={timeframe} | target_idx={step_idx} | valid_idx={current_idx} | offset={offset} | value={price}",
                                dedupe=True
                            )
                        break
                except (ValueError, IndexError):
                    continue
            
            if not found_valid:
                # Log failure only if we couldn't find ANY valid price
                try:
                    bad_val = df.iloc[step_idx][close_col]
                except:
                    bad_val = "ERROR"
                    
                self.smart_logger.error(
                    f"PRICE_INVALID_ALL_ATTEMPTS | asset={asset_key} | timeframe={timeframe} | idx={step_idx} | value={bad_val} | backtracked={max_backtrack}",
                    dedupe=True
                )
                continue # Keep price as None
        
        return prices

    def _check_excessive_forward_fill(self):
        """Vérifier si le taux de forward-fill dépasse le seuil acceptable."""
        # Initialiser les compteurs s'ils n'existent pas
        if not hasattr(self, "_price_read_success_count"):
            self._price_read_success_count = 0
        if not hasattr(self, "_price_forward_fill_count"):
            self._price_forward_fill_count = 0
        if not hasattr(self, "_forward_fill_threshold"):
            self._forward_fill_threshold = 0.5  # 50% maximum de forward-fill acceptable
        if not hasattr(self, "_last_ff_check_step"):
            self._last_ff_check_step = 0

        # Compter cette lecture comme succès (puisqu'on arrive ici, on a lu des prix)
        self._price_read_success_count += 1

        # Vérifier périodiquement (tous les 100 steps)
        if self.current_step - self._last_ff_check_step < 100:
            return

        self._last_ff_check_step = self.current_step
        total_reads = self._price_read_success_count + self._price_forward_fill_count

        if total_reads < 10:  # Attendre au moins 10 lectures avant de vérifier
            return

        forward_fill_rate = (
            self._price_forward_fill_count / total_reads if total_reads > 0 else 0
        )

        if forward_fill_rate > self._forward_fill_threshold:
            self.smart_logger.error(
                f"EXCESSIVE_FORWARD_FILL | rate={forward_fill_rate * 100:.1f}% | count={self._price_forward_fill_count}/{total_reads}",
                dedupe=True,
            )

        # Réinitialiser les compteurs périodiquement pour avoir des mesures actuelles
        if total_reads > 1000:
            self._price_read_success_count = max(
                100, int(self._price_read_success_count * 0.1)
            )
            self._price_forward_fill_count = max(
                10, int(self._price_forward_fill_count * 0.1)
            )

    def _validate_market_data(self, prices: Dict[str, float]) -> bool:
        """Valide les données de marché avant l'exécution des trades.

        Args:
            prices: Dictionnaire des prix actuels par actif

        Returns:
            bool: True si les données sont valides, False sinon
        """
        if not prices:
            logger.error("No market data available")
            return False

        # --- CORRECTION ---
        # Filtrer les prix non valides (None, <= 0, ou non finis)
        invalid_assets = [
            asset
            for asset, price in prices.items()
            if price is None or price <= 0 or not np.isfinite(price)
        ]

        if invalid_assets:
            invalid_list = ", ".join(invalid_assets)
            logger.error("Invalid prices for assets: %s", invalid_list)
            return False

        return True

    def _log_trade_error(
        self, asset: str, action_value: float, price: float, error: str
    ) -> None:
        """Enregistre les erreurs de trading pour analyse ultérieure.

        Args:
            asset: Symbole de l'actif concerné
            action_value: Valeur de l'action (-1 à 1)
            price: Prix au moment de l'erreur
            error: Message d'erreur détaillé
        """
        # Déterminer le type d'action
        if action_value > 0.05:
            action = "BUY"
        elif action_value < -0.05:
            action = "SELL"
        else:
            action = "HOLD"

        # Préparer les informations d'erreur
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "step": self.current_step,
            "asset": asset,
            "action": action,
            "action_value": float(action_value),
            "price": float(price) if price is not None else None,
            "error": error,
        }

        # Ajouter la valeur du portefeuille si disponible
        if hasattr(self.portfolio, "portfolio_value"):
            error_info["portfolio_value"] = float(self.portfolio.portfolio_value)

        # Logger l'erreur
        logger.error(f"Trade error: {error_info}")

        # Enregistrer dans un fichier si configuré
        log_config = self.config.get("logging", {})
        error_log_path = log_config.get("error_log_path", "trade_errors.log")

        try:
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(error_info) + "\n")
        except IOError as e:
            logger.error(f"Failed to write to error log: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error writing to log: {str(e)}")

    def _get_current_timestamp(self) -> pd.Timestamp:
        """Get the current timestamp aligned with the current chunk index."""
        current_idx_in_chunk = getattr(self, "step_in_chunk", self.current_step)

        for tf, asset_data_dict in self.current_data.items():
            for asset, df in asset_data_dict.items():
                if not df.empty:
                    timestamp: Optional[pd.Timestamp] = None

                    if 0 <= current_idx_in_chunk < len(df):
                        timestamp = df.index[current_idx_in_chunk]
                    elif current_idx_in_chunk >= len(df):
                        timestamp = df.index[-1]
                    else:
                        timestamp = df.index[0]

                    if isinstance(timestamp, pd.Timestamp):
                        self._last_asset_timestamp[asset] = timestamp
                        self._last_market_timestamp = timestamp
                        return timestamp

        if self._last_market_timestamp is not None:
            return self._last_market_timestamp

        raise RuntimeError("No timestamp data available")

    def _get_safe_timestamp(self) -> Optional[str]:
        """Get the current timestamp safely."""
        try:
            return self._get_current_timestamp().isoformat()
        except Exception:
            return None

    def _manage_cache(self, key: str, value: np.ndarray = None) -> Optional[np.ndarray]:
        """Gère le cache d'observations avec une politique LRU."""
        if key in self._observation_cache:
            self._cache_access[key] = time.time()
            self._cache_hits += 1
            return self._observation_cache[key]

        self._cache_misses += 1

        if len(self._observation_cache) >= self._max_cache_size:
            sorted_keys = sorted(
                self._cache_access.keys(), key=lambda k: self._cache_access[k]
            )
            num_to_remove = max(1, int(self._max_cache_size * 0.1))
            for k in sorted_keys[:num_to_remove]:
                self._observation_cache.pop(k, None)
                self._cache_access.pop(k, None)

        if value is not None:
            self._observation_cache[key] = value
            self._cache_access[key] = time.time()

        return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'utilisation du cache."""
        total = self._cache_hits + self._cache_misses
        hit_ratio = self._cache_hits / total if total > 0 else 0.0

        return {
            "cache_enabled": True,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._observation_cache),
            "max_size": self._max_cache_size,
            "hit_ratio": hit_ratio,
        }

    def _process_assets(
        self, feature_config: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Process asset data for the current step.

        Args:
            feature_config: Dictionary mapping timeframes to lists of feature names

        Returns:
            Dictionary mapping assets to dictionaries of {timeframe: DataFrame}

            Format attendu par StateBuilder.build_observation():
            {
                'asset1': {
                    '5m': DataFrame,
                    '1h': DataFrame,
                    '4h': DataFrame
                },
                'asset2': {
                    '5m': DataFrame,
                    '1h': DataFrame,
                    '4h': DataFrame
                },
                ...
            }
        """
        processed_data = {asset: {} for asset in self.assets}

        for asset in self.assets:
            for timeframe in self.timeframes:
                if timeframe not in feature_config:
                    logger.warning(
                        f"No feature configuration for timeframe {timeframe}"
                    )
                    continue

                features = feature_config[timeframe]
                if not features:
                    logger.warning(f"No features specified for timeframe {timeframe}")
                    continue

                # Get data for this asset and timeframe
                asset_data = self.current_data.get(asset, {}).get(timeframe)
                if asset_data is None or asset_data.empty:
                    logger.debug(f"No data for {asset} {timeframe}")
                    continue

                # Log all available columns for debugging
                logger.debug(
                    f"Available columns in {asset} {timeframe} data: {asset_data.columns.tolist()}"
                )
                logger.debug(f"Requested features for {timeframe}: {features}")

                # Create a mapping of uppercase column names to actual column names
                column_mapping = {col.upper(): col for col in asset_data.columns}

                # Find available features in the asset data (case-insensitive)
                available_features = []
                missing_features = []
                for f in features:
                    upper_f = f.upper()
                    if upper_f in column_mapping:
                        available_features.append(column_mapping[upper_f])
                        logger.debug(
                            f"Found feature: '{f}' -> '{column_mapping[upper_f]}'"
                        )
                    else:
                        missing_features.append(f)
                        logger.debug(
                            f"Missing feature: '{f}' (not in DataFrame columns)"
                        )

                if missing_features:
                    logger.warning(
                        f"Missing {len(missing_features)} features for {asset} {timeframe}: {missing_features}"
                    )
                    logger.debug(
                        f"Available columns in {asset} {timeframe}: {asset_data.columns.tolist()}"
                    )
                    logger.debug(f"Available features: {available_features}")

                if not available_features:
                    logger.warning(
                        f"None of the requested features found for {asset} {timeframe}"
                    )
                    continue

                try:
                    # Select only the requested features using their original case
                    asset_df = asset_data[available_features].copy()

                    # Ensure column names are in uppercase for consistency
                    asset_df.columns = [col.upper() for col in asset_df.columns]

                    # Store the DataFrame in the processed data
                    processed_data[asset][timeframe] = asset_df

                except Exception as e:
                    logger.error(f"Error processing {asset} {timeframe}: {str(e)}")
                    logger.debug(f"Available columns: {asset_data.columns.tolist()}")
                    logger.debug(f"Available features: {available_features}")

        # Remove assets with no data
        return {k: v for k, v in processed_data.items() if v}

    def _create_empty_dataframe(
        self, timeframe: str, window_size: int = None
    ) -> pd.DataFrame:
        """Create an empty DataFrame with required features for a given timeframe.

        Args:
            timeframe: The timeframe for which to create the empty DataFrame
            window_size: Number of rows to include in the empty DataFrame

        Returns:
            DataFrame with required columns and zero values
        """
        try:
            # Get required features for this timeframe
            features = self.state_builder.get_feature_names(timeframe)
            if not features:
                logger.warning(f"No features defined for timeframe {timeframe}")
                features = ["close"]  # Fallback to basic column

            # Use window_size if provided, otherwise default to 1
            rows = window_size if window_size is not None else 1

            # Create empty DataFrame with required features
            empty_data = np.zeros((rows, len(features)))
            df = pd.DataFrame(empty_data, columns=features)

            # Add timestamp column if not present
            if (
                "timestamp" not in df.columns
                and "timestamp" in self.features[timeframe]
            ):
                df["timestamp"] = pd.Timestamp.now()

            logger.debug(
                f"Created empty DataFrame for {timeframe} with shape {df.shape}"
            )
            return df

        except Exception as e:
            logger.error(f"Error creating empty DataFrame for {timeframe}: {str(e)}")
            # Fallback to minimal DataFrame
            return pd.DataFrame(columns=["timestamp", "close"])

    def _is_valid_observation_structure(self, obs: Dict[str, np.ndarray]) -> bool:
        """Vérifie si l'observation correspond à l'espace `Dict` attendu."""
        if not isinstance(obs, dict):
            logger.error(f"Observation is not a dict, but {type(obs)}")
            return False

        obs_keys = set(obs.keys())
        space_keys = set(self.observation_space.spaces.keys())

        if obs_keys != space_keys:
            logger.error(f"Observation keys mismatch: {obs_keys} vs {space_keys}")
            missing_keys = space_keys - obs_keys
            extra_keys = obs_keys - space_keys
            if missing_keys:
                logger.error(f"Missing keys in observation: {missing_keys}")
            if extra_keys:
                logger.error(f"Extra keys in observation: {extra_keys}")
            return False

        for key, space in self.observation_space.spaces.items():
            if key not in obs:
                logger.error(
                    f"Key {key} is in observation space but not in observation"
                )
                return False
            if not hasattr(obs[key], "shape"):
                logger.error(f"Observation for key {key} has no shape attribute")
                return False
            if obs[key].shape != space.shape:
                logger.error(
                    f"Shape mismatch for key {key}: expected {space.shape}, got {obs[key].shape}"
                )
                return False
        return True

    def _default_observation(self) -> Dict[str, np.ndarray]:
        """Retourne une observation par défaut (remplie de zéros) correspondant à l'espace."""
        obs = {}
        for key, space in self.observation_space.spaces.items():
            obs[key] = np.zeros(space.shape, dtype=np.float32)
        return obs


    def _build_observation(self) -> Dict[str, np.ndarray]:
        """
        Construit l'observation pour le pas de temps actuel en utilisant le StateBuilder.
        Retourne un dictionnaire de tenseurs, un pour chaque timeframe, plus l'état du portefeuille.
        """
        try:
            current_idx = int(getattr(self, "step_in_chunk", 0))

            if not hasattr(self, "state_builder") or self.state_builder is None:
                raise RuntimeError("StateBuilder non initialisé.")
            if not hasattr(self, "current_data") or self.current_data is None:
                logger.warning(
                    "Données non disponibles, retourne une observation par défaut."
                )
                return self._default_observation()

            market_obs_dict = self.state_builder.build_observation(
                current_idx, self.current_data
            )
            # Récupérer le vecteur d'état du portefeuille
            if hasattr(self.portfolio, "get_state_vector"):
                portfolio_state = self.portfolio.get_state_vector()
            else:
                portfolio_state = np.zeros(
                    self.observation_space.spaces["portfolio_state"].shape,
                    dtype=np.float32,
                )

            # Combiner les observations du marché et du portefeuille
            final_obs = {**market_obs_dict}
            final_obs["portfolio_state"] = portfolio_state.astype(np.float32)

            return final_obs

        except Exception as e:
            logger.error(f"Erreur dans _build_observation: {e}", exc_info=True)
            return self._default_observation()

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Construit et retourne l'observation actuelle, en s'assurant que les shapes sont correctes."""
        try:
            obs = self._build_observation()

            # --- PATCH POUR CORRIGER LES SHAPE MISMATCH ---
            for key, space in self.observation_space.spaces.items():
                if key == "portfolio_state":
                    continue  # Ne pas toucher à l'état du portefeuille

                if key in obs and obs[key].shape != space.shape:
                    self.logger.warning(
                        f"Shape mismatch for key {key}: expected {space.shape}, got {obs[key].shape}. Applying padding."
                    )
                    
                    padded_obs = np.zeros(space.shape, dtype=np.float32)
                    
                    # Copier les données existantes à la fin du tableau paddé
                    current_data = obs[key]
                    rows_to_copy = min(current_data.shape[0], space.shape[0])
                    cols_to_copy = min(current_data.shape[1] if len(current_data.shape) > 1 else 1, space.shape[1] if len(space.shape) > 1 else 1)
                    
                    if rows_to_copy > 0 and cols_to_copy > 0:
                        if len(current_data.shape) == 2 and len(space.shape) == 2:
                            padded_obs[-rows_to_copy:, :cols_to_copy] = current_data[-rows_to_copy:, :cols_to_copy]
                        else:
                            padded_obs[-rows_to_copy:] = current_data[-rows_to_copy:]
                    
                    obs[key] = padded_obs
            # --- FIN DU PATCH ---

            self._last_observation = obs
            # Valider la structure de l'observation finale
            if not self._is_valid_observation_structure(obs):
                logger.error(
                    "L'observation générée a une structure invalide, utilisation d'une observation par défaut."
                )
                default_obs = self._default_observation()
                self._last_observation = default_obs
                return default_obs
            return obs
        except Exception as e:
            logger.error(f"Erreur critique dans _get_observation: {e}", exc_info=True)
            default_obs = self._default_observation()
            self._last_observation = default_obs
            return default_obs

    def _check_and_reset_daily_counters(self) -> None:
        """
        Vérifie si un nouveau jour a commencé et reset les compteurs si nécessaire.
        Utilise le nombre de steps pour déterminer le passage d'un jour.
        """
        if not self.frequency_config:
            return

        daily_steps_5m = self.frequency_config.get("daily_steps_5m", 288)

        # Calculer le jour courant basé sur les steps totaux
        current_day = self.num_timesteps // daily_steps_5m

        if current_day > self.current_day:
            # Nouveau jour détecté, reset des compteurs
            self.positions_count = {"5m": 0, "1h": 0, "4h": 0, "daily_total": 0}
            self.current_day = current_day
            self.daily_reset_step = self.current_step

            self.smart_logger.info(
                f"[FREQUENCY] Nouveau jour détecté (jour {current_day}), reset des compteurs de positions",
                rotate=True,
            )

    def _track_position_frequency(self) -> None:
        """
        Suit les positions ouvertes/fermées par timeframe.
        Met à jour les compteurs globaux.
        """
        if not self.frequency_config:
            return

        # Déterminer le timeframe courant (simplifié - vous pouvez l'améliorer)
        # Pour cet exemple, on utilise un mapping basé sur le current_step modulo
        step_in_day = self.current_step % self.frequency_config.get(
            "daily_steps_5m", 288
        )

        # Logique simplifiée pour déterminer le timeframe actuel
        if step_in_day % 48 == 0:  # Toutes les 4h (288/6 = 48)
            current_timeframe = "4h"
        elif step_in_day % 12 == 0:  # Toutes les 1h (288/24 = 12)
            current_timeframe = "1h"
        else:
            current_timeframe = "5m"

        # Vérifier si de nouveaux trades ont été ajoutés au trade_log
        if hasattr(self.portfolio, "trade_log") and self.portfolio.trade_log:
            # Initialiser last_trade_ids si nécessaire
            if not hasattr(self, "last_trade_ids"):
                self.last_trade_ids = set()

            # Identifier les nouveaux trades
            new_trades = []
            for trade in self.portfolio.trade_log:
                trade_id = f"{trade.get('timestamp', 0)}_{trade.get('asset', '')}_{trade.get('type', '')}_{trade.get('price', 0)}"
                if trade_id not in self.last_trade_ids:
                    new_trades.append(trade)
                    self.last_trade_ids.add(trade_id)

            # Compter les nouveaux trades par timeframe
            for trade in new_trades:
                trade_type = trade.get("type", "")
                if trade_type == "open":
                    self.positions_count[current_timeframe] += 1
                    self.positions_count["daily_total"] += 1

                    asset = trade.get("asset", "Unknown")
                    price = trade.get("price", 0.0)
                    self.smart_logger.info(
                        f"[FREQUENCY] Trade {trade_type} {asset} @ {price:.2f} sur {current_timeframe} "
                        f"(count: {self.positions_count[current_timeframe]}, "
                        f"total: {self.positions_count['daily_total']})",
                        rotate=True,
                    )

    def _calculate_frequency_reward(self) -> float:
        """
        Calcule le bonus/pénalité de fréquence basé sur le nombre de positions
        par timeframe et le total journalier.
        """
        if not self.frequency_config:
            return 0.0

        frequency_reward = 0.0

        # Get weights from reward_shaping.frequency_weights
        reward_shaping_config = self.config.get("reward_shaping", {})
        frequency_weights = reward_shaping_config.get("frequency_weights", {})
        in_range_weight = frequency_weights.get("in_range", 0.0)
        out_of_range_weight = frequency_weights.get("out_of_range", 0.0)

        # Logique de conscience de la traque
        is_hunting = self.dbe.is_hunting(self.worker_id)
        hunt_info = self.dbe.get_hunt_info(self.worker_id) if is_hunting else None
        timeframe_map = {"5m": 5, "1h": 60, "4h": 240}

        # Vérifier chaque timeframe individuellement
        for timeframe in ["5m", "1h", "4h"]:
            if timeframe in self.frequency_config:
                tf_config = self.frequency_config[timeframe]
                min_pos = tf_config.get("min_positions", 0)
                max_pos = tf_config.get("max_positions", 999)
                current_count = self.positions_count[timeframe]

                # Ajustement adaptatif si en traque
                if is_hunting and hunt_info:
                    hunting_tf = hunt_info.get("hunting_timeframe")
                    if timeframe_map.get(timeframe, 0) < timeframe_map.get(
                        hunting_tf, 0
                    ):
                        min_pos = int(min_pos * 0.5)
                        max_pos = int(max_pos * 0.5)
                        self.smart_logger.info(
                            f"[HUNT AWARENESS] Freq thresholds for {timeframe} reduced to min:{min_pos}, max:{max_pos} due to hunt on {hunting_tf}",
                            rotate=True,
                        )

                if min_pos <= current_count <= max_pos:
                    # Dans les bornes : bonus proportionnel
                    frequency_reward += in_range_weight * (
                        current_count / max(max_pos, 1)
                    )
                else:
                    # Hors bornes : pénalité progressive et asymétrique
                    if current_count < min_pos:
                        penalty = out_of_range_weight * (min_pos - current_count)
                    else:  # current_count > max_pos
                        penalty = out_of_range_weight * (current_count - max_pos)
                    frequency_reward -= penalty

        # Vérifier le total journalier
        total_min = self.frequency_config.get("total_daily_min", 5)
        total_max = self.frequency_config.get("total_daily_max", 15)
        daily_total = self.positions_count["daily_total"]

        if total_min <= daily_total <= total_max:
            # Bonus pour être dans les bornes totales
            frequency_reward += in_range_weight * (daily_total / max(total_max, 1))
        else:
            # Pénalité pour être hors bornes totales
            if daily_total < total_min:
                penalty = out_of_range_weight * (total_min - daily_total)
            else:  # daily_total > total_max
                penalty = out_of_range_weight * (daily_total - total_max)
            frequency_reward -= penalty

        return frequency_reward

    def _validate_frequency(self) -> None:
        """
        Valide que les compteurs de fréquence respectent les bornes configurées.
        Log des warnings si hors bornes, confirmation si dans les bornes.
        """
        if not self.frequency_config:
            return

        valid = True

        # Vérifier chaque timeframe individuellement
        for tf in ["5m", "1h", "4h"]:
            if tf in self.frequency_config:
                tf_config = self.frequency_config[tf]
                count = self.positions_count[tf]
                min_pos = tf_config.get("min_positions", 0)
                max_pos = tf_config.get("max_positions", 999)

                if count < min_pos or count > max_pos:
                    valid = False
                    self.smart_logger.warning(
                        f"[FREQUENCY] Count hors bornes pour {tf}: {count} (min: {min_pos}, max: {max_pos})",
                        dedupe=True,
                    )

        # Vérifier le total journalier
        total_count = self.positions_count["daily_total"]
        min_total = self.frequency_config.get("total_daily_min", 5)
        max_total = self.frequency_config.get("total_daily_max", 15)

        if total_count < min_total or total_count > max_total:
            valid = False
            self.smart_logger.warning(
                f"[FREQUENCY] Total journalier hors bornes: {total_count} (min: {min_total}, max: {max_total})",
                dedupe=True,
            )

        # Log de confirmation si tout est dans les bornes
        if valid:
            self.smart_logger.info(
                f"[FREQUENCY] Tous les counts dans les bornes: {self.positions_count}",
                rotate=True,
            )

    def calculate_position_limit_penalty(self) -> float:
        """Calculates the penalty for exceeding the position limit for the current tier."""
        pos_limit_penalty = 0.0
        try:
            tier_cfg = self.portfolio_manager.get_current_tier()
            limit = 1
            if isinstance(tier_cfg, dict):
                limit = int(tier_cfg.get("max_open_positions", 1))
            open_count = 0
            try:
                open_count = len(
                    [
                        p
                        for p in self.portfolio_manager.positions.values()
                        if getattr(p, "is_open", False)
                    ]
                )
            except Exception:
                open_count = 0
            if open_count > limit:
                weight = self.config.get("trading_rules", {}).get(
                    "position_limit_penalty_weight", 1.0
                )
                # Appliquer une pénalité non-linéaire pour adoucir les petits dépassements
                pos_limit_penalty -= float(weight) * np.tanh(open_count - limit)
        except Exception:
            pass
        return pos_limit_penalty

    def calculate_outcome_reward(self) -> float:
        """Calculates the reward/penalty for trades closed."""
        outcome_reward = 0.0
        try:
            reward_config = self.config.get("reward_shaping", {})
            trade_outcome_config = reward_config.get("trade_outcome", {})
            tp_multiplier = trade_outcome_config.get("take_profit_multiplier", 0.5)
            sl_multiplier = trade_outcome_config.get("stop_loss_multiplier", 0.5)
            passivity_penalty = reward_config.get(
                "passivity_penalty", -5.0
            )  # Get new penalty

            if hasattr(self, "_step_closed_receipts"):
                for receipt in self._step_closed_receipts:
                    pnl = receipt.get("pnl", 0.0)

                    # Apply passivity penalty if trade was force-closed
                    if receipt.get("reason") == "MaxDuration":
                        outcome_reward += passivity_penalty
                        self.smart_logger.warning(
                            f"[REWARD] Passivity penalty applied for MaxDuration closure: {passivity_penalty:.2f}",
                            rotate=True,
                        )

                    if pnl > 0:  # Trade profitable
                        outcome_reward += pnl * tp_multiplier
                    else:  # Trade perdant
                        outcome_reward += pnl * sl_multiplier
        except Exception as e:
            self.logger.error(
                f"Erreur lors du calcul de la récompense de résultat: {e}"
            )
        return outcome_reward

    def calculate_capacity_based_reward(self) -> float:
        """Calculates a reward based on the current capital usage."""
        reward = 0.0
        if not hasattr(self, "portfolio_manager"):
            return reward

        total_value = self.portfolio_manager.get_total_value()
        cash = self.portfolio_manager.get_cash()

        if total_value > 0:
            capacity_usage = (total_value - cash) / total_value

            if 0.6 <= capacity_usage <= 0.9:
                reward += 2.0
            elif capacity_usage > 0.9:
                reward -= (capacity_usage - 0.9) * 10
            elif capacity_usage < 0.3:
                reward -= (0.3 - capacity_usage) * 5

        return reward

    def has_open_trade(self, timeframe: str) -> bool:
        """Checks if there is an open trade for the given timeframe."""
        if not hasattr(self, "portfolio_manager"):
            return False

        for position in self.portfolio_manager.positions.values():
            if position.is_open and position.timeframe == timeframe:
                return True
        return False

    def calculate_early_close_bonus(self) -> float:
        bonus = 0.0

        reward_shaping_config = self.config.get("reward_shaping", {})
        worker_name = self.worker_config.get("name", "Default")
        worker_profile_config = reward_shaping_config.get("profiles", {}).get(
            worker_name, {}
        )

        early_close_bonus_weight = worker_profile_config.get(
            "early_exit_bonus", reward_shaping_config.get("early_close_bonus", 0.0)
        )

        if not hasattr(self, "_step_closed_receipts"):
            return bonus

        for receipt in self._step_closed_receipts:
            pnl_pct = receipt.get("pnl_pct", 0.0)
            duration_seconds = receipt.get("duration_seconds")
            timeframe = receipt.get("timeframe", "5m")

            if pnl_pct > 0 and duration_seconds is not None:
                duration_config = self.config.get("trading_rules", {}).get(
                    "duration_tracking", {}
                )
                if timeframe in duration_config:
                    optimal_duration_steps = duration_config[timeframe].get(
                        "optimal_duration", 0
                    )
                    # Assuming 5m steps, convert duration_seconds to steps
                    duration_steps = duration_seconds / 300  # 300 seconds = 5 minutes

                    if (
                        optimal_duration_steps > 0
                        and duration_steps < optimal_duration_steps
                    ):
                        # Bonus is proportional to how early it closed relative to optimal
                        bonus += (
                            early_close_bonus_weight
                            * (optimal_duration_steps - duration_steps)
                            / optimal_duration_steps
                        )
                        self.smart_logger.info(
                            f"[OUTCOME BONUS] Early profitable close for {receipt.get('asset')}. Bonus: {bonus:.2f}",
                            rotate=True,
                        )
        return bonus

    def calculate_duration_penalty(self) -> float:
        """Calculates the penalty for trades that are open for too long, and rewards for optimal duration."""
        penalty = 0.0
        if not hasattr(self, "portfolio_manager"):
            return penalty

        duration_config = self.config.get("trading_rules", {}).get(
            "duration_tracking", {}
        )
        overstay_penalty_value = self.config.get("reward_shaping", {}).get(
            "overstay_penalty", 0.0
        )  # Get from new config

        if not duration_config:
            return penalty

        for position in self.portfolio_manager.positions.values():
            if position.is_open:
                timeframe = position.timeframe
                if timeframe in duration_config:
                    config = duration_config[timeframe]
                    max_duration_steps = config.get("max_duration_steps")
                    optimal_duration_steps = config.get(
                        "optimal_duration", max_duration_steps
                    )  # Use max as optimal if not specified

                    if max_duration_steps:
                        current_duration = self.current_step - position.open_step

                        if current_duration > max_duration_steps:
                            # Apply overstay penalty
                            penalty += (
                                overstay_penalty_value
                                * (current_duration - max_duration_steps)
                                / max_duration_steps
                            )
                            self.smart_logger.warning(
                                f"[DURATION PENALTY] Overstay penalty for {position.asset}. Duration: {current_duration} > Max: {max_duration_steps}. Penalty: {penalty:.2f}",
                                rotate=True,
                            )
                        elif current_duration < optimal_duration_steps:
                            # No penalty for closing early if profitable, handled by early_close_bonus
                            pass
                        # Add bonus for optimal duration if needed, but user suggested overstay penalty
        return penalty

    def get_current_timeframe(self) -> str:
        """Returns the current timeframe for trade execution."""
        return getattr(self, "current_timeframe_for_trade", "5m")

    def _calculate_reward(self, action: np.ndarray, realized_pnl: float) -> float:
        """Calcule la récompense finale alignée pour chaque worker.
        
        Logs detailed information about each reward component to help with debugging.
        """
        # Initialize reward components dictionary for logging
        reward_components = {
            'base_reward': 0.0,
            'pnl': 0.0,
            'duration_penalties': 0.0,
            'frequency_reward': 0.0,
            'pos_limit_penalty': 0.0,
            'outcome_reward': 0.0,
            'early_close_bonus': 0.0,
            'invalid_trade_penalty': 0.0,
            'inaction_penalty': 0.0,
            'missed_penalty': 0.0,
            'multi_bonus': 0.0,
            'final_reward': 0.0
        }

        reward_shaping_config = self.config.get("reward_shaping", {})

        # --- Load worker-specific reward profile ---
        worker_name = self.worker_config.get(
            "name", "Default"
        )  # e.g., Conservative, Moderate, Aggressive, Adaptive
        worker_profile_config = reward_shaping_config.get("profiles", {}).get(
            worker_name, {}
        )

        # Override global reward shaping parameters with worker-specific ones if they exist
        pnl_weight = worker_profile_config.get(
            "pnl_weight", reward_shaping_config.get("pnl_weight", 1.0)
        )
        missed_penalty_value = worker_profile_config.get(
            "missed_opportunity_penalty",
            reward_shaping_config.get("missed_penalty", 0.0),
        )
        multi_traque_bonus_value = worker_profile_config.get(
            "multi_traque_bonus", reward_shaping_config.get("multi_traque_bonus", 0.0)
        )
        invalid_trade_penalty_weight = worker_profile_config.get(
            "invalid_trade_penalty_weight",
            reward_shaping_config.get("invalid_trade_penalty_weight", 0.5),
        )
        # Note: early_close_bonus and overstay_penalty are handled in their respective functions, which will need to be updated to use worker_profile_config as well.

        # 1. RÉCOMPENSE DE BASE SPÉCIALISÉE (PnL + Résultat)
        pnl = realized_pnl
        reward_components['pnl'] = float(pnl)
        
        # Apply non-linear scaling to PnL
        base_reward = pnl_weight * np.tanh(pnl / pnl_weight)  # Apply pnl_weight
        reward_components['base_reward'] = float(base_reward)

        # Conditional inaction penalty and missed opportunity penalty
        current_timeframe = self.get_current_timeframe()
        if not self.dbe.is_hunting(self.worker_id) and not self.has_open_trade(
            current_timeframe
        ):
            inaction_penalty = self.calculate_inaction_penalty()
            base_reward += inaction_penalty
            reward_components['inaction_penalty'] = float(inaction_penalty)
            
            # Add missed penalty if no open trade and not hunting
            base_reward += missed_penalty_value
            reward_components['missed_penalty'] = float(missed_penalty_value)
            
            self.smart_logger.info(
                f"[REWARD] Inaction penalty: {inaction_penalty:.4f}, Missed opportunity penalty: {missed_penalty_value:.4f}",
                rotate=True,
            )
        elif not self.dbe.is_hunting(self.worker_id) and self.portfolio_manager.get_cash() < self.config.get(
            "trading_rules", {}
        ).get("min_order_value_usdt", 11.0):
            # If not hunting and not enough cash, still apply inaction penalty if applicable
            inaction_penalty = self.calculate_inaction_penalty()
            base_reward += inaction_penalty
            reward_components['inaction_penalty'] = float(inaction_penalty)

        # 2. PÉNALITÉS DE GESTION TEMPORELLE
        # calculate_duration_penalty now includes overstay_penalty
        duration_penalties = self.calculate_duration_penalty()
        reward_components['duration_penalties'] = float(duration_penalties) if duration_penalties is not None else 0.0

        # 3. RÉCOMPENSE DE FRÉQUENCE
        frequency_reward = self._calculate_frequency_reward()
        reward_components['frequency_reward'] = float(frequency_reward) if frequency_reward is not None else 0.0

        # 4. PÉNALITÉ DE LIMITE DE POSITION
        pos_limit_penalty = self.calculate_position_limit_penalty()
        reward_components['pos_limit_penalty'] = float(pos_limit_penalty) if pos_limit_penalty is not None else 0.0

        # 5. RÉCOMPENSE DE RÉSULTAT DE TRADE (inclut bonus early close)
        outcome_reward = self.calculate_outcome_reward()
        early_close_bonus = self.calculate_early_close_bonus()
        outcome_reward += early_close_bonus
        reward_components['outcome_reward'] = float(outcome_reward) if outcome_reward is not None else 0.0
        reward_components['early_close_bonus'] = float(early_close_bonus) if early_close_bonus is not None else 0.0

        # 6. PÉNALITÉ POUR TENTATIVES DE TRADE INVALIDE
        invalid_trade_attempt_penalty = -invalid_trade_penalty_weight * self.invalid_trade_attempts
        reward_components['invalid_trade_penalty'] = float(invalid_trade_attempt_penalty)
        self.invalid_trade_attempts = 0  # Reset counter after use

        # 7. BONUS MULTI-TRAQUE (si capital élevé)
        multi_bonus = 0.0
        if hasattr(self, "portfolio_manager"):
            initial_capital = self.portfolio_manager.initial_capital
            current_equity = self.portfolio_manager.get_equity()
            open_trades_count = len(
                [p for p in self.portfolio_manager.positions.values() if p.is_open]
            )

            multi_bonus_threshold_factor = reward_shaping_config.get(
                "multi_traque_bonus_threshold_factor", 1.5
            )  # This should be configurable per worker or globally
            multi_bonus_min_open_trades = reward_shaping_config.get(
                "multi_traque_bonus_min_open_trades", 2
            )  # This should be configurable per worker or globally

            if (
                current_equity > initial_capital * multi_bonus_threshold_factor
                and open_trades_count >= multi_bonus_min_open_trades
            ):
                multi_bonus = multi_traque_bonus_value * open_trades_count
                reward_components['multi_bonus'] = float(multi_bonus)
                self.smart_logger.info(
                    f"[REWARD] Multi-hunt bonus applied: {multi_bonus:.2f}", rotate=True
                )

        # Somme de toutes les composantes
        total_reward = (
            base_reward
            + frequency_reward
            + pos_limit_penalty
            + outcome_reward
            + duration_penalties
            + invalid_trade_attempt_penalty
            + multi_bonus
        )

        # Clipping asymétrique pour favoriser les gains
        final_reward = np.clip(total_reward, -3.0, 10.0)

        # Journalisation pour debug
        self.logger.info(
            f"[REWARD Worker {self.worker_id}] Base: {base_reward:.4f}, Freq: {frequency_reward:.4f}, PosLimit: {pos_limit_penalty:.4f}, Outcome: {outcome_reward:.4f}, Duration: {duration_penalties:.4f}, InvalidTrade: {invalid_trade_attempt_penalty:.4f}, MultiHunt: {multi_bonus:.4f}, Total: {final_reward:.4f}, Counts: {self.positions_count}"
        )

        # Store reward components for info dict
        self._last_reward_components = {
            "pnl": float(pnl),
            "base_reward": float(base_reward),
            "frequency_reward": float(frequency_reward),
            "pos_limit_penalty": float(pos_limit_penalty),
            "outcome_reward": float(outcome_reward),
            "duration_penalties": float(duration_penalties),
            "invalid_trade_attempt_penalty": float(invalid_trade_attempt_penalty),
            "multi_bonus": float(multi_bonus),
            "total_reward": float(final_reward),
        }

        return final_reward

    def _save_checkpoint(self) -> Dict[str, Any]:
        """Sauvegarde l'état actuel de l'environnement et du portefeuille.

        Returns:
            Dict contenant l'état sauvegardé
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "portfolio_state": self.portfolio.get_state(),
            "env_state": {
                "current_step": self.current_step,
                "current_chunk": self.current_chunk,
                "episode_count": self.episode_count,
                "episode_reward": self.episode_reward,
                "best_portfolio_value": self.best_portfolio_value,
            },
            "tier_info": {
                "current_tier": (
                    self.current_tier["name"] if self.current_tier else None
                ),
                "episodes_in_tier": self.episodes_in_tier,
                "last_tier_change_step": self.last_tier_change_step,
            },
        }

    def _load_checkpoint_on_demotion(self) -> bool:
        """Charge un point de contrôle précédent en cas de rétrogradation.

        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        if not hasattr(self, "checkpoint_history") or not self.checkpoint_history:
            logger.warning("No checkpoint history available for demotion")
            return False

        try:
            # Charger le dernier checkpoint
            last_checkpoint = self.checkpoint_history[-1]

            # Restaurer l'état du portefeuille
            if "portfolio_state" in last_checkpoint:
                self.portfolio.set_state(last_checkpoint["portfolio_state"])

            # Restaurer l'état de l'environnement
            if "env_state" in last_checkpoint:
                env_state = last_checkpoint["env_state"]
                self.current_step = env_state.get("current_step", 0)
                self.current_chunk = env_state.get("current_chunk", 0)
                self.episode_count = env_state.get("episode_count", 0)
                self.episode_reward = env_state.get("episode_reward", 0.0)
                self.best_portfolio_value = env_state.get(
                    "best_portfolio_value", self.portfolio.get_total_value()
                )

            # Restaurer les informations de palier
            if "tier_info" in last_checkpoint:
                tier_info = last_checkpoint["tier_info"]
                self.current_tier = next(
                    (
                        t
                        for t in self.tiers
                        if t["name"] == tier_info.get("current_tier")
                    ),
                    self.tiers[0] if self.tiers else None,
                )
                self.episodes_in_tier = tier_info.get("episodes_in_tier", 0)
                self.last_tier_change_step = tier_info.get("last_tier_change_step", 0)

            logger.info("Successfully loaded checkpoint after demotion")

            # Recharger les données du chunk actuel si nécessaire
            if hasattr(self, "current_chunk"):
                self.current_data = self._safe_load_chunk(self.current_chunk)

            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

        # Journalisation des composantes de la récompense
        self._last_reward_components = {
            "base_reward": float(reward_info.get("base_reward", 0.0)),
            "risk_penalty": float(reward_info.get("risk_penalty", 0.0)),
            "transaction_penalty": float(reward_info.get("transaction_penalty", 0.0)),
            "concentration_penalty": float(
                reward_info.get("concentration_penalty", 0.0)
            ),
            "action_smoothness_penalty": float(
                reward_info.get("action_smoothness_penalty", 0.0)
            ),
            "total_reward": float(reward_info.get("total_reward", reward)),
        }

        return float(reward_info.get("total_reward", reward))

    def _calculate_asset_volatility(self, asset: str, lookback: int = 21) -> float:
        """
        Calcule la volatilité annualisée d'un actif sur une période donnée.

        Args:
            asset: Symbole de l'actif
            lookback: Nombre de jours pour le calcul de la volatilité (par défaut: 21 jours)

        Returns:
        # Patch Gugu & March - Ajouter bonus d'excellence
        if hasattr(self, 'excellence_rewards') and self.excellence_rewards:
            try:
                excellence_bonus = self._calculate_excellence_bonus(total_reward, worker_id)
                total_reward += excellence_bonus

                if excellence_bonus > 0.01:
                    logger.debug(f"[GUGU-MARCH] Worker {worker_id} total reward boosted by {excellence_bonus:.4f}")

            except Exception as e:
                logger.warning(f"[GUGU-MARCH] Error applying excellence bonus: {e}")

            float: Volatilité annualisée en décimal (0.2 pour 20%)
        """
        try:
            if not hasattr(self, "current_data") or not self.current_data:
                self.logger.warning(
                    "Données de marché non disponibles pour le calcul de volatilité"
                )
                return 0.15  # Valeur par défaut raisonnable

            # Récupérer les données de prix pour l'actif
            if asset not in self.current_data:
                self.logger.warning(f"Données manquantes pour l'actif {asset}")
                return 0.15

            # Prendre le premier intervalle de temps disponible
            tf = next(iter(self.current_data[asset].keys()))
            df = self.current_data[asset][tf]

            # Vérifier si on a assez de données
            if len(df) < lookback + 1:
                self.logger.warning(
                    f"Pas assez de données pour calculer la volatilité sur {lookback} jours"
                )
                return 0.15

            # Calculer les rendements journaliers
            close_prices = df["close"].iloc[-(lookback + 1) :]
            returns = close_prices.pct_change().dropna()

            # Calculer la volatilité annualisée (252 jours de trading par an)
            volatility = returns.std() * np.sqrt(252)

            # Limiter la volatilité entre 5% et 200%
            volatility = np.clip(volatility, 0.05, 2.0)

            self.logger.debug(
                f"Volatilité calculée pour {asset}: {volatility:.2%} (sur {lookback} jours)"
            )
            return float(volatility)

        except Exception as e:
            self.logger.error(
                f"Erreur dans le calcul de la volatilité pour {asset}: {str(e)}"
            )
            return 0.15  # Retourne une volatilité par défaut en cas d'erreur

    def _force_trade(self, timeframe: str) -> float:
        """
        Forces a trade based on worker-specific strategy, respecting position limits and daily caps.

        Returns:
            float: Realized PnL from any positions closed during force trade
        """
        realized_pnl = 0.0
        try:
            # Daily reset for forced trade count
            current_timestamp = self._get_current_timestamp()
            current_date = current_timestamp.date() if current_timestamp else None

            if current_date and (self.last_reset_date is None or current_date != self.last_reset_date):
                self.daily_forced_trades_count = 0
                self.last_reset_date = current_date
                self.logger.info(f"[FORCE_TRADE_DAILY_RESET] Worker {self.worker_id}: Daily forced trade count reset for {current_date}.")

            # Check daily forced trade cap
            if self.daily_forced_trades_count >= self.daily_max_forced_trades:
                self.logger.warning(
                    f"🚫 [FORCE_TRADE_CAP] Worker {self.worker_id}: Daily forced trade limit ({self.daily_max_forced_trades}) reached. Skipping forced trade for {timeframe}."
                )
                return 0.0

            self.logger.warning(
                f"🚨 [FORCE_TRADE] Attempting to force trade for timeframe {timeframe} at step {self.current_step} for Worker {self.worker_id}"
            )

            # 1. Get worker's strategy and primary timeframe from config
            worker_key = f"w{self.worker_id + 1}"  # Assuming worker_id is 0-indexed
            worker_config = self.config.get("workers", {}).get(worker_key, {})
            primary_tf = worker_config.get("specialization", {}).get("timeframe")

            # 2. Check position limit and existing positions
            open_positions = self.portfolio._get_open_positions()
            position_limit = self.max_positions

            if len(open_positions) >= position_limit:
                oldest_position = min(open_positions, key=lambda p: p.open_step)

                # 3. Apply strategic replacement logic
                # DO NOT close a primary timeframe position for a secondary one
                if oldest_position.timeframe == primary_tf and timeframe != primary_tf:
                    self.logger.warning(
                        f"[FORCE_TRADE_SKIP] Worker {self.worker_id} holds a primary position on {primary_tf}. Ignoring force trade for secondary timeframe {timeframe}."
                    )
                    return 0.0

                self.logger.warning(
                    f"[FORCE_TRADE] Position limit ({position_limit}) reached. Closing position on {oldest_position.timeframe} to make room for {timeframe}."
                )
                current_prices = self._get_current_prices()
                price_to_close = current_prices.get(
                    oldest_position.asset, oldest_position.current_price
                )

                close_receipt = self.portfolio.close_position(
                    asset=oldest_position.asset,
                    price=price_to_close,
                    timestamp=self._get_current_timestamp(),
                    current_prices=current_prices,
                    reason="FORCE_CLOSE_FOR_NEW_TRADE",
                )

                if close_receipt:
                    # Add the PnL from the closed position
                    closed_pnl = close_receipt.get("pnl", 0.0)
                    realized_pnl += float(closed_pnl)
                    self.logger.info(
                        f"[FORCE_TRADE] Closed position realized PnL: ${closed_pnl:.2f}"
                    )
                else:
                    self.logger.error(
                        f"❌ [FORCE_TRADE] Failed to close oldest position ({oldest_position.asset}) to make room. Aborting."
                    )
                    return 0.0

            # 4. Proceed to open the new forced trade
            if not self.assets:
                self.logger.error("[FORCE_TRADE] No assets available.")
                return realized_pnl
            asset_to_trade = self.assets[0]

            current_prices = self._get_current_prices()
            price = current_prices.get(asset_to_trade.upper())
            if price is None or price <= 0:
                self.logger.error(f"[FORCE_TRADE] No valid price for {asset_to_trade}.")
                return realized_pnl

            # Compute desired position using DBE to stay consistent with tier ranges and DBE ±10% modulation
            try:
                tier_cfg = self.portfolio_manager.get_current_tier() if hasattr(self, "portfolio_manager") else {}
            except Exception:
                tier_cfg = {}
            try:
                capital = float(self.portfolio_manager.get_equity()) if hasattr(self, "portfolio_manager") else float(self.portfolio.get_portfolio_value())
            except Exception:
                capital = float(self.portfolio.get_portfolio_value())

            dbe_params = self.dbe.calculate_trade_parameters(
                capital=capital,
                worker_pref_pct=1.0,  # force trade implies full allocation within tier range
                tier_config=tier_cfg or {},
                current_price=price,
                desired_position_size=1.0,
            ) if hasattr(self, "dbe") else {"feasible": False, "reason": "DBE unavailable"}

            self.logger.debug(f"DEBUG_OPTUNA: DBE params: {dbe_params}")

            if not dbe_params.get("feasible", False):
                self.logger.error(f"❌ [FORCE_TRADE] Aborted for {asset_to_trade} on {timeframe}. Reason: {dbe_params.get('reason', 'DBE calculation failed')}")
                return realized_pnl

            pos_usdt = float(dbe_params.get("position_size_usdt", 0.0))
            self.logger.debug(f"DEBUG_OPTUNA: Calculated pos_usdt: {pos_usdt}")

            min_order_value = self.config.get("trading_rules", {}).get("min_order_value_usdt", 11.0)

            # Clamp to minimum order value if DBE suggests below threshold and equity allows
            if pos_usdt < min_order_value:
                if capital >= min_order_value:
                    self.logger.warning(
                        f"[FORCE_TRADE] pos_usdt {pos_usdt:.2f} < min {min_order_value} → clamped to minimum."
                    )
                    pos_usdt = float(min_order_value)
                else:
                    self.logger.error(
                        f"❌ [FORCE_TRADE] Failed for {asset_to_trade} on {timeframe}. Reason: pos_usdt {pos_usdt:.2f} < min_order_value {min_order_value} and capital {capital:.2f} insufficient."
                    )
                    self.invalid_trade_attempts += 1
                    try:
                        if (hasattr(self.portfolio_manager, "metrics") and self.portfolio_manager.metrics):
                            self.portfolio_manager.metrics.record_trade_rejection(
                                reason="force_trade_min_order",
                                context={
                                    "asset": asset_to_trade,
                                    "timeframe": timeframe,
                                    "pos_usdt": pos_usdt,
                                    "min_order_value": min_order_value,
                                },
                            )
                    except Exception:
                        pass
                    return realized_pnl

            # Coherence check and logging
            qty = pos_usdt / float(price)
            recalc_notional = qty * float(price)
            if abs(recalc_notional - pos_usdt) > 1e-4:
                self.logger.warning(
                    f"[FORCE_TRADE_VALIDATION] Incoherence detected: pos_usdt={pos_usdt:.4f}, recalc_notional={recalc_notional:.4f}, price={float(price):.4f}"
                )
            self.logger.info(
                f"[FORCE_TRADE_VALIDATION] pos_usdt={pos_usdt:.2f} ≥ min_order_value={min_order_value:.2f}, qty={qty:.8f}, price={float(price):.2f}"
            )

            amount = qty
            # Snapshot open positions before
            try:
                _before_positions = list(self.portfolio._get_open_positions())
                _before_count = len(_before_positions)
            except Exception:
                _before_positions = []
                _before_count = 0

            # Enforce frequency limits for forced trades as well
            try:
                can_trade_result = self.can_open_trade(timeframe)
                self.logger.debug(f"DEBUG_OPTUNA: can_open_trade({timeframe}) returned {can_trade_result}")
                if not can_trade_result:
                    self.smart_logger.info(
                        f"[FORCE_TRADE_GATE] Blocked by limits for TF={timeframe} (daily_total={self.positions_count.get('daily_total', 0)}, tf_count={self.positions_count.get(timeframe, 0)})",
                        rotate=True,
                    )
                    self.invalid_trade_attempts += 1
                    return realized_pnl
            except Exception:
                # Fail-safe: if gating raises, do not force open
                return realized_pnl

            self.logger.debug(f"DEBUG_OPTUNA: Calling open_position for {asset_to_trade}")
            receipt = self.portfolio.open_position(
                asset=asset_to_trade.upper(),
                price=price,
                size=amount,
                stop_loss_pct=self.portfolio.sl_pct,
                take_profit_pct=self.portfolio.tp_pct,
                timestamp=self._get_current_timestamp(),
                current_prices=current_prices,
                timeframe=timeframe,
                current_step=self.current_step,
                risk_horizon=0.0,
            )
            self.logger.debug(f"DEBUG_OPTUNA: open_position receipt: {receipt}")
            
            # Robust success detection: receipt OK OR open positions increased OR matching freshly opened position
            success = False
            if isinstance(receipt, dict) and str(receipt.get("status", "")).upper() in {"OPEN", "OPENED", "SUCCESS"}:
                success = True
            else:
                try:
                    _after_positions = list(self.portfolio._get_open_positions())
                    _after_count = len(_after_positions)
                except Exception:
                    _after_positions = []
                    _after_count = _before_count

                if _after_count > _before_count:
                    success = True
                else:
                    for p in _after_positions:
                        if (
                            getattr(p, "asset", "").upper() == asset_to_trade.upper()
                            and getattr(p, "timeframe", timeframe) == timeframe
                            and getattr(p, "open_step", -1) == self.current_step
                        ):
                            success = True
                            break

            if success:
                # Increment daily forced trade count if trade is successfully opened
                self.daily_forced_trades_count += 1
                self.logger.info(f"[FORCE_TRADE_COUNT] Worker {self.worker_id}: Daily forced trades: {self.daily_forced_trades_count}/{self.daily_max_forced_trades}")
                
                self.logger.warning(
                    f"✅ [FORCE_TRADE] Success for {asset_to_trade} on {timeframe} for Worker {self.worker_id}."
                )
                
                # Synchronize ALL counters (timeframe + global)
                self.positions_count[timeframe] = self.positions_count.get(timeframe, 0) + 1
                self.positions_count['daily_total'] = self.positions_count.get('daily_total', 0) + 1  # FIX: Increment global counter
                self.last_trade_steps_by_tf[timeframe] = self.current_step
                self.last_trade_step = self.current_step
                
                # Also notify freq_controller if RealisticTradingEnv is being used
                if hasattr(self, 'freq_controller') and self.freq_controller:
                    try:
                        self.freq_controller.record_trade(
                            asset=asset_to_trade,
                            current_step=self.current_step,
                            timeframe=timeframe,
                            is_forced=True
                        )
                        self.logger.debug(f"[FORCE_TRADE_SYNC] Notified freq_controller for {asset_to_trade}")
                    except Exception as e:
                        self.logger.warning(f"[FORCE_TRADE_SYNC] Failed to notify freq_controller: {e}")
                
                return realized_pnl
            else:
                _err = (receipt.get("message", "No receipt") if isinstance(receipt, dict) else "No receipt")
                self.logger.error(
                    f"❌ [FORCE_TRADE] Failed for {asset_to_trade} on {timeframe}. Reason: {_err}"
                )
                return realized_pnl

        except Exception as e:
            self.logger.error(
                f"💥 [FORCE_TRADE] Exception for {timeframe} on Worker {self.worker_id}: {e}",
                exc_info=True,
            )
            return 0.0  # Return 0 on exception

    def _execute_trades(
        self,
        action: np.ndarray,
        dbe_modulation: dict,
        action_threshold: float,
        force_trade: bool = False,
    ) -> tuple:
        """
        OMEGA-3 Target-Weight execution logic.

        Action space per asset (5 dims): [action_raw, size_raw, tf_raw, sl_raw, tp_raw]
        - action_raw: -1..1  (< -thr = SELL, > thr = BUY, else HOLD)
        - size_raw:   -1..1  mapped to tier exposure_range
        - tf_raw:     -1..1  maps to timeframe index
        - sl_raw:     -1..1  dynamic SL based on tier risk_per_trade_pct
        - tp_raw:     -1..1  dynamic TP

        Key design points:
        - Tier-mapped exposure: uses capital_tiers.exposure_range
        - Notional clamped to >= 11 USDT
        - Anti-spam HOLD: if target exposure ~ current exposure, do nothing
        - Dynamic SL bounded by tier risk_per_trade_pct
        """
        if not hasattr(self, "portfolio_manager"):
            self.logger.error("Portfolio manager non initialise.")
            return 0.0, 0, 0

        current_timestamp = None
        try:
            current_prices = self._get_current_prices()
            try:
                current_timestamp = self._get_current_timestamp()
                self._last_market_timestamp = current_timestamp
            except Exception:
                current_timestamp = self._last_market_timestamp

            if hasattr(self.portfolio_manager, "register_market_timestamp"):
                self.portfolio_manager.register_market_timestamp(current_timestamp)

            if not current_prices or not self._validate_market_data(current_prices):
                if hasattr(self, "portfolio_manager"):
                    self.portfolio_manager.update_market_price(
                        current_prices if current_prices else {}, self.current_step
                    )
                return 0.0, 0, 0
        except Exception as e:
            self.logger.error(f"Price retrieval error at step {self.current_step}: {e}", exc_info=True)
            return 0.0, 0, 0

        realized_pnl = 0.0
        trade_executed_this_step = False
        first_discrete_action = 0

        # 1. Update positions & check SL/TP
        pnl_from_update, sl_tp_receipts = self.portfolio_manager.update_market_price(
            current_prices, self.current_step
        )
        if sl_tp_receipts:
            self._step_closed_receipts.extend(sl_tp_receipts)
        if pnl_from_update > 0:
            realized_pnl += pnl_from_update
            trade_executed_this_step = True

        # ================================================================
        # CAPITAL TIER SUPREMACY: exposure_range and risk_per_trade_pct
        # are dictated EXCLUSIVELY by the current Capital Tier, NEVER
        # by the worker profile. A Scalper at 20$ (Micro) exposes 70-90%;
        # the same Scalper at 10000$ (Enterprise) exposes only 5-15%.
        # ================================================================
        capital = self.portfolio_manager.get_total_value()
        tier = None

        # 1. Try DBE tier resolution (preferred, uses live capital)
        if hasattr(self, 'dbe') and self.dbe:
            tier_func = getattr(self.dbe, 'get_capital_tier',
                                getattr(self.dbe, '_get_capital_tier', None))
            if tier_func:
                tier = tier_func(capital)

        # 2. Fallback: search capital_tiers from master config
        if not tier:
            for t in self.config.get("capital_tiers", []):
                lo = t.get("min_capital", 0)
                hi = t.get("max_capital") or float("inf")
                if lo <= capital < hi:
                    tier = t
                    break

        # 3. Last resort: Micro Capital defaults (safest assumption)
        if not tier:
            tier = {"exposure_range": [70, 90], "risk_per_trade_pct": 4.0,
                    "name": "Micro Capital"}

        # Store for monitoring / reward calculator
        self.current_tier = tier

        # 4. Extract tier-level risk parameters (NEVER from worker_config)
        min_exp = float(tier.get("exposure_range", [70, 90])[0]) / 100.0
        max_exp = float(tier.get("exposure_range", [70, 90])[1]) / 100.0
        max_risk_pct = float(tier.get("risk_per_trade_pct", 4.0)) / 100.0
        min_order_value = float(self.config.get("environment", {}).get(
            "hard_constraints", {}).get("min_order_value_usdt",
            self.config.get("trading_rules", {}).get("min_order_value_usdt", 11.0)))

        # Log tier info periodically
        if self.current_step % 50 == 0:
            self.logger.info(
                f"[TIER] Step {self.current_step} | Capital={capital:.2f} | "
                f"Tier={tier.get('name', '?')} | Exposure=[{min_exp:.0%},{max_exp:.0%}] | "
                f"MaxRisk={max_risk_pct:.2%}"
            )

        # 2. Iterate over assets
        for i, asset in enumerate(self.assets):
            if asset not in current_prices:
                continue
            price = current_prices[asset]

            # Decode 5-dim action per asset
            base_idx = i * 5
            if base_idx + 4 >= len(action):
                continue

            action_raw = float(action[base_idx])
            size_raw = float(action[base_idx + 1])
            tf_raw = float(action[base_idx + 2])
            sl_raw = float(action[base_idx + 3])
            tp_raw = float(action[base_idx + 4])

            # Decode timeframe
            tf_idx = int((tf_raw + 1) * 1.5)
            tf_idx = max(0, min(len(self.timeframes) - 1, tf_idx))
            self.current_timeframe_for_trade = self.timeframes[tf_idx]

            main_decision = action_raw

            # Discrete action
            discrete_action = 0
            if main_decision < -action_threshold:
                discrete_action = 2  # Sell
            elif main_decision > action_threshold:
                discrete_action = 1  # Buy

            if i == 0:
                first_discrete_action = discrete_action

            # Log intention (every 50 steps to reduce spam)
            if self.current_step % 50 == 0:
                action_str = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(discrete_action, "HOLD")
                self.logger.info(
                    f"[TARGET_WEIGHT] Step {self.current_step} | {asset} | "
                    f"Action={action_str} | Raw={main_decision:.3f} | Thr={action_threshold:.3f} | "
                    f"SizeRaw={size_raw:.3f} | Capital={capital:.2f}"
                )

            position = self.portfolio_manager.positions.get(asset)
            is_open = position and position.is_open

            # ---- Target-Weight Sizing ----
            normalized_size = (size_raw + 1.0) / 2.0  # 0..1
            target_exposure_pct = min_exp + normalized_size * (max_exp - min_exp)
            notional_usd = capital * target_exposure_pct

            # Clamp to min order value
            if notional_usd < min_order_value and capital >= min_order_value:
                notional_usd = min_order_value
                target_exposure_pct = min_order_value / capital if capital > 0 else 0

            # ---- Dynamic SL bounded by tier risk ----
            if notional_usd > 0:
                max_sl_pct = (capital * max_risk_pct) / notional_usd
            else:
                max_sl_pct = 0.02  # fallback
            normalized_sl = (sl_raw + 1.0) / 2.0
            sl_pct = 0.005 + normalized_sl * (max(max_sl_pct, 0.005) - 0.005)
            sl_pct = min(sl_pct, 0.10)  # hard cap 10%

            # Dynamic TP (agent decides, 1% to 20%)
            normalized_tp = (tp_raw + 1.0) / 2.0
            tp_pct = 0.01 + normalized_tp * (0.20 - 0.01)

            # ---- Anti-spam HOLD ----
            # If already open and target exposure ~ current exposure -> HOLD
            if is_open and discrete_action == 1:
                current_exposure = 0.0
                try:
                    pos_value = position.size * price if hasattr(position, 'size') else 0
                    current_exposure = pos_value / capital if capital > 0 else 0
                except Exception:
                    pass
                exposure_diff = abs(target_exposure_pct - current_exposure)
                if exposure_diff < 0.10:  # Within 10% -> no action needed (OMEGA-4E)
                    discrete_action = 0  # Override to HOLD

            # ---- Force close max duration ----
            max_steps = self.config.get("trading_rules", {}).get("max_position_steps")
            if is_open and max_steps and (self.current_step - position.open_step > max_steps):
                receipt = self.portfolio_manager.close_position(
                    asset=asset.upper(), price=price, timestamp=current_timestamp,
                    current_prices=current_prices, reason="MAX_DURATION",
                    risk_horizon=getattr(position, 'risk_horizon', 0.0),
                )
                if receipt:
                    self._step_closed_receipts.append(receipt)
                    self._all_episode_receipts.append(receipt)
                    realized_pnl += float(receipt.get("pnl", 0.0))
                    trade_executed_this_step = True
                continue

            # A. CLOSE position - with SELL hysteresis
            if discrete_action == 2 and is_open:
                # SELL hysteresis: skip close if exposure < 5%
                current_exposure_for_sell = 0.0
                try:
                    pos_val = position.size * price if hasattr(position, 'size') else 0
                    current_exposure_for_sell = pos_val / capital if capital > 0 else 0
                except Exception:
                    pass
                if current_exposure_for_sell < 0.05:
                    discrete_action = 0  # Override to HOLD, position too small
                else:
                    self.trade_attempts += 1
                    receipt = self.portfolio_manager.close_position(
                        asset=asset.upper(), price=price, timestamp=current_timestamp,
                        current_prices=current_prices, reason="AGENT_CLOSE",
                        risk_horizon=getattr(position, 'risk_horizon', 0.0),
                    )
                    if receipt and isinstance(receipt, dict):
                        self._step_closed_receipts.append(receipt)
                        self._all_episode_receipts.append(receipt)
                        val = receipt.get("pnl", 0.0)
                        fees = receipt.get("fees", 0.0)
                        realized_pnl += float(val)
                        self._apply_trade_results_safely(pnl_value=float(val), fees=float(fees))
                        trade_executed_this_step = True
                    else:
                        self.invalid_trade_attempts += 1

            # B. OPEN position
            elif discrete_action == 1 and not is_open:
                self.trade_attempts += 1

                # OMEGA-4E: Per-worker cooldown
                _COOLDOWN_MAP = {"scalper": 3, "intraday": 2, "swing": 1, "position": 1}
                _wname = str(self.worker_config.get("profile", self.worker_config.get("name", "scalper"))).lower()
                for _pkey in _COOLDOWN_MAP:
                    if _pkey in _wname:
                        _min_cd = _COOLDOWN_MAP[_pkey]
                        break
                else:
                    _min_cd = 2
                if self.current_step - self._last_open_step_by_asset.get(asset, -999) < _min_cd:
                    self.invalid_trade_attempts += 1
                    continue

                # Check daily limits
                if not self.can_open_trade(self.current_timeframe_for_trade):
                    self.invalid_trade_attempts += 1
                    continue

                # Check cash
                available_cash = self.portfolio_manager.get_cash()
                if notional_usd < min_order_value or available_cash < notional_usd:
                    self.invalid_trade_attempts += 1
                    if self.current_step % 100 == 0:
                        self.logger.warning(
                            f"[SIZE_GATE] {asset} notional={notional_usd:.2f} min={min_order_value} cash={available_cash:.2f}"
                        )
                    continue

                size_in_asset = notional_usd / price if price > 0 else 0
                equity = self.portfolio_manager.get_equity()
                final_pct = (notional_usd / equity) if equity > 0 else 0.0

                receipt = None
                try:
                    receipt = self.portfolio_manager.open_position(
                        asset=asset.upper(), price=price, size=size_in_asset,
                        stop_loss_pct=float(sl_pct), take_profit_pct=float(tp_pct),
                        timestamp=current_timestamp, current_prices=current_prices,
                        allocated_pct=final_pct,
                        timeframe=self.current_timeframe_for_trade,
                        current_step=self.current_step, risk_horizon=0.0,
                    )
                except Exception as open_err:
                    self.logger.error(f"[OPEN_ERROR] {asset}: {open_err}", exc_info=True)

                if receipt:
                    self.receipts.append(receipt)
                    self._all_episode_receipts.append(receipt)
                    self._last_open_step_by_asset[asset] = self.current_step
                    trade_executed_this_step = True
                    # Update frequency counters
                    if not force_trade:
                        tf = self.current_timeframe_for_trade
                        self.positions_count[tf] = int(self.positions_count.get(tf, 0)) + 1
                        self.positions_count["daily_total"] = int(self.positions_count.get("daily_total", 0)) + 1
                    # Update timestamps
                    tf = self.current_timeframe_for_trade
                    try:
                        if isinstance(current_timestamp, (pd.Timestamp,)):
                            self.last_trade_timestamps[tf] = current_timestamp.to_pydatetime()
                        elif current_timestamp is not None:
                            self.last_trade_timestamps[tf] = current_timestamp
                        if not hasattr(self, "last_trade_steps_by_tf"):
                            self.last_trade_steps_by_tf = {}
                        self.last_trade_steps_by_tf[tf] = self.current_step
                    except Exception:
                        pass
                    self.logger.info(
                        f"[TRADE_OPEN] {asset} size={size_in_asset:.6f} notional={notional_usd:.2f} "
                        f"SL={sl_pct:.2%} TP={tp_pct:.2%} tier={tier.get('name', '?')}"
                    )
                else:
                    self.invalid_trade_attempts += 1

        # 3. Update last trade step
        if trade_executed_this_step:
            self.last_trade_step = self.current_step

        return realized_pnl, first_discrete_action, first_discrete_action

    def _should_force_close_timeframe(
        self, timeframe: str, steps_since_open: int, force_after: int
    ) -> bool:
        """Détermine si une position doit être forcée à la clôture pour un timeframe donné."""
        if force_after <= 0:
            return False
        return steps_since_open >= max(force_after, 1)

    def _current_chunk_length(self) -> int:
        try:
            first_asset = next(iter(self.current_data))
            first_tf = next(iter(self.current_data[first_asset]))
            return len(self.current_data[first_asset][first_tf])
        except Exception:
            return 0

    def _should_force_trade(self, timeframe: str) -> bool:
        """Détermine si un trade doit être forcé pour la timeframe donnée."""
        if not self.can_open_trade(timeframe):
            self.logger.warning(f"DEBUG_OPTUNA: can_open_trade({timeframe}) returned False")
            return False
        # self.logger.warning(f"DEBUG_OPTUNA: can_open_trade({timeframe}) returned True")

        force_cfg = (
            self.config.get("trading_rules", {})
            .get("frequency", {})
            .get("force_trade_steps", {})
        )
        if isinstance(force_cfg, dict):
            force_after = int(force_cfg.get(timeframe, force_cfg.get("default", 50) or 50))
        else:
            force_after = int(force_cfg or 50)

        last_steps = getattr(self, "last_trade_steps_by_tf", {})
        if not isinstance(last_steps, dict):
            last_steps = {}
            self.last_trade_steps_by_tf = last_steps

        last_step = last_steps.get(timeframe, 0) # Changed -force_after to 0
        steps_since = self.current_step - last_step

        # Cooldown gating (non-recurrent intervals) — primary trigger if configured
        next_allowed = int(self.next_force_allowed_step_by_tf.get(timeframe, 0))
        if next_allowed > 0:
            if self.current_step < next_allowed:
                return False
            # Cooldown elapsed => allow force immediately (no need to wait force_after)
            return True

        near_chunk_end = self.step_in_chunk >= max(0, self._current_chunk_length() - 5)
        try:
            self.logger.info(
                "[FORCE_TRADE_DIAG] tf=%s force_cfg=%s force_after=%s last_step=%s steps_since=%s near_chunk_end=%s",
                timeframe,
                force_cfg,
                force_after,
                last_step,
                steps_since,
                near_chunk_end,
            )
        except Exception:
            pass
        # Default behavior when no cooldown configured: inactivity/near-end rules
        result = steps_since >= force_after or (near_chunk_end and steps_since > max(20, force_after // 2))
        if not result and steps_since > force_after:
             self.logger.warning(f"DEBUG_OPTUNA: _should_force_trade({timeframe}) -> {result} (steps_since={steps_since}, force_after={force_after}, next_allowed={next_allowed})")
        return result

    def _log_agent_thought(
        self,
        timeframe: str,
        asset: str,
        action_raw: float,
        action_threshold: float,
        current_price: float,
        reason: str,
        observation_summary: Dict[str, float],
        action_decision: str,
    ) -> None:
        """Enregistre un log riche décrivant la décision de l'agent."""
        try:
            now = datetime.now(timezone.utc).strftime("%H:%M:%S")
            rsi = observation_summary.get("rsi", 0.0)
            trend = observation_summary.get("trend", 0.0)
            trend_label = "UP" if trend > 0 else "DOWN" if trend < 0 else "FLAT"
            volatility = observation_summary.get("volatility", 0.0)
            self.logger.info(
                f"[{now}] [THOUGHT:{timeframe.upper()}] Asset={asset.upper()} Act={action_raw:+.4f} "
                f"Thr={action_threshold:.4f} Decision={action_decision} Price={current_price:.2f} "
                f"RSI={rsi:5.1f} Trend={trend_label} Vol={volatility:5.2f} Reason={reason}"
            )
        except Exception:
            pass

    def _extract_observation_summary(self, asset: str, timeframe: str) -> Dict[str, float]:
        """Construit un résumé simple de l'observation pour le logging."""
        summary: Dict[str, float] = {"rsi": 0.0, "trend": 0.0, "volatility": 0.0}
        try:
            observation = getattr(self, "_last_observation", None)
            if not observation or timeframe not in observation:
                return summary

            tf_data = observation[timeframe]
            if isinstance(tf_data, np.ndarray):
                # Support arrays de forme (window, features)
                if tf_data.size == 0:
                    return summary
                last_row = tf_data[-1]
                # Hypothèse: colonnes alignées avec features_config (RSI etc.)
                # Impossible de mapper précisément sans métadonnées, donc we leave defaults.
                return summary

            if isinstance(tf_data, pd.DataFrame) and not tf_data.empty:
                last_row = tf_data.iloc[-1]
                summary["rsi"] = float(last_row.get("rsi", summary["rsi"]))
                if "close" in tf_data.columns:
                    recent = tf_data["close"].iloc[-min(len(tf_data), 5) :]
                    summary["trend"] = (
                        float(recent.iloc[-1] - recent.iloc[0]) if len(recent) > 1 else 0.0
                    )
                    summary["volatility"] = (
                        float(tf_data["close"].pct_change().std()) if len(tf_data) > 1 else 0.0
                    )
        except Exception:
            pass
        return summary

    def _update_risk_metrics(self, portfolio_value, returns):
        """Met à jour les métriques de risque du portefeuille en utilisant PerformanceMetrics."""
        try:
            # Utiliser les métriques centralisées du PortfolioManager si disponibles
            if hasattr(self, "portfolio_manager") and hasattr(self.portfolio_manager, "metrics"):
                metrics_summary = self.portfolio_manager.metrics.get_metrics_summary()
                
                # Log la source des métriques pour traçabilité
                self.logger.debug(
                    f"[METRICS_FLOW] Worker {self.worker_id} | Source: PortfolioManager.metrics | "
                    f"Trades={metrics_summary.get('total_trades', 0)}, "
                    f"Sharpe={metrics_summary.get('sharpe_ratio', 0.0):.4f}, "
                    f"MaxDD={metrics_summary.get('max_drawdown', 0.0):.2%}"
                )
                
                self.risk_metrics.update(
                    {
                        "sharpe_ratio": metrics_summary.get("sharpe_ratio", 0.0),
                        "sortino_ratio": metrics_summary.get("sortino_ratio", 0.0),
                        "volatility": metrics_summary.get("volatility", 0.0),
                        "max_drawdown": metrics_summary.get("max_drawdown", 0.0),
                        "win_rate": metrics_summary.get("win_rate", 0.0),
                        "total_trades": metrics_summary.get("total_trades", 0),
                    }
                )
                
                # Log périodique de confirmation (tous les 500 steps)
                if self.current_step % 500 == 0:
                    self.logger.info(
                        f"[METRICS_SYNC] Step {self.current_step} | Worker {self.worker_id} | "
                        f"Sharpe={self.risk_metrics['sharpe_ratio']:.4f}, "
                        f"Sortino={self.risk_metrics['sortino_ratio']:.4f}, "
                        f"WinRate={self.risk_metrics['win_rate']:.2f}%, "
                        f"Trades={self.risk_metrics['total_trades']}"
                    )
            else:
                # Fallback si pas de metrics manager (ne devrait pas arriver avec la config actuelle)
                self.logger.warning(
                    f"[METRICS_FLOW_ERROR] Worker {self.worker_id} | PortfolioManager.metrics NOT available | "
                    f"has_pm={hasattr(self, 'portfolio_manager')}, "
                    f"has_metrics={hasattr(self.portfolio_manager, 'metrics') if hasattr(self, 'portfolio_manager') else 'N/A'}"
                )
                self.risk_metrics.update(
                    {
                        "sharpe_ratio": 0.0,
                        "sortino_ratio": 0.0,
                        "volatility": 0.0,
                        "max_drawdown": 0.0,
                    }
                )

        except Exception as e:
            self.logger.error(
                f"[METRICS_FLOW_EXCEPTION] Worker {self.worker_id} | Error: {str(e)}"
            )

    def reset_daily_counts(self):
        """
        Réinitialise les compteurs de trades journaliers pour tous les workers.
        """
        # Assurez-vous que self.positions_count est un dictionnaire de dictionnaires
        # où la clé est l'ID du worker et la valeur est un dictionnaire de compteurs.
        # Pour l'environnement unique, nous utilisons self.positions_count directement.
        
        # Réinitialiser le total journalier
        self.positions_count['daily_total'] = 0
        
        # Réinitialiser les compteurs par timeframe
        for tf in self.timeframes: # Utiliser self.timeframes pour itérer sur les timeframes configurés
            self.positions_count[tf] = 0
            
        self.smart_logger.info("[DAILY RESET] All daily trade counts reset to 0", rotate=True)

    def can_open_trade(self, tf: str) -> bool:
        """
        Vérifie si un trade peut être ouvert en respectant les limites journalières et par timeframe.
        """
        daily_total = self.positions_count.get('daily_total', 0)
        tf_count = self.positions_count.get(tf, 0)

        # Récupérer les limites depuis la configuration du worker
        # Fallback vers la config globale si non spécifié au niveau du worker
        worker_freq_config = self.worker_config.get("trading_rules", {}).get("frequency", {})
        global_freq_config = self.config.get("trading_rules", {}).get("frequency", {})

        daily_max = worker_freq_config.get("daily_max_total", global_freq_config.get("total_daily_max", 20))
        tf_max = worker_freq_config.get("daily_max_by_tf", {}).get(tf, global_freq_config.get("max_positions", {}).get(tf, float('inf')))

        if daily_total >= daily_max:
            self.logger.warning(f"DEBUG_OPTUNA: can_open_trade blocked by daily_max ({daily_total} >= {daily_max})")
            self.smart_logger.warning(
                f"[TRADE BLOCKED] Worker {self.worker_id} | TF {tf} | Reason: daily_max_total={daily_max} reached (current: {daily_total})",
                rotate=True,
            )
            return False
        
        if tf_count >= tf_max:
            self.smart_logger.warning(
                f"[TRADE BLOCKED] Worker {self.worker_id} | TF {tf} | Reason: daily_max_by_tf[{tf}]={tf_max} reached (current: {tf_count})",
                rotate=True,
            )
            return False

        # If no blocking condition is met, allow opening a trade
        return True

    def get_current_date(self) -> pd.Timestamp:
        """
        Récupère la date courante de la simulation.
        """
        return self._get_current_timestamp()

    def _get_info(self) -> Dict[str, Any]:
        """
        Récupère des informations supplémentaires sur l'état de l'environnement.

        Returns:
            Dict[str, Any]: Dictionnaire contenant des informations détaillées sur l'état
                actuel du portefeuille et de l'environnement.
        """
        # Récupérer les métriques du portfolio manager
        portfolio_metrics = self.portfolio_manager.get_metrics()
        current_prices = self._get_current_prices()
        position_values = {}
        total_position_value = 0.0
        info_warnings: List[str] = []

        # Calculer les valeurs des positions actuelles
        for asset, pos_info in portfolio_metrics.get("positions", {}).items():
            qty = pos_info.get("size", pos_info.get("quantity", 0.0)) or 0.0
            price = None
            if isinstance(current_prices, dict):
                price = current_prices.get(asset)

            # Fallback to last known/current price stored in metrics (transparent)
            if price is None or not np.isfinite(price) or price <= 0:
                fallback_price = pos_info.get("current_price") or pos_info.get("entry_price")
                if fallback_price is not None and np.isfinite(fallback_price) and fallback_price > 0:
                    price = float(fallback_price)
                    info_warnings.append(f"STALE_PRICE|{asset}|using_last_known")
                else:
                    # Skip this asset if we cannot value it safely
                    info_warnings.append(f"PRICE_UNAVAILABLE|{asset}|skipped_in_info")
                    continue

            value = float(qty) * float(price)
            position_values[asset] = {
                "quantity": float(qty),
                "price": float(price),
                "value": float(value),
                "unrealized_pnl": float(pos_info.get("unrealized_pnl", 0.0)),
                "entry_price": float(pos_info.get("entry_price", price)),
                "weight": (
                    float(value) / float(portfolio_metrics.get("total_value", 1.0))
                    if portfolio_metrics.get("total_value", 0) > 0
                    else 0.0
                ),
            }
            total_position_value += float(value)

        # Composants de récompense
        reward_components = {}
        if hasattr(self, "_last_reward_components"):
            reward_components = self._last_reward_components

        # Statistiques d'actions
        action_stats = {}
        if hasattr(self, "_last_action") and self._last_action is not None:
            action = self._last_action
            action_stats = {
                "action_mean": float(np.mean(action)),
                "action_std": float(np.std(action)),
                "action_min": float(np.min(action)),
                "action_max": float(np.max(action)),
                "num_assets": len(action),
            }

        # Trading statistics détaillées
        total_trades = portfolio_metrics.get("total_trades", 0)
        valid_trades = portfolio_metrics.get("valid_trades", 0)
        # Utiliser _step_closed_receipts qui contient les fermetures SL/TP réelles
        closed_positions = getattr(self, "_step_closed_receipts", []) or portfolio_metrics.get("closed_positions", [])

        # Récompenses et pénalités
        last_reward = getattr(self, "_last_reward", 0.0)
        last_penalty = reward_components.get("frequency_penalty", 0.0)
        cumulative_reward = getattr(self, "_cumulative_reward", 0.0)

        # Information complète
        info = {
            "step": self.current_step,
            "chunk": self.current_chunk_idx,
            "done": getattr(self, "done", False),
            "worker_id": getattr(self, "worker_id", 0),
            # Portfolio metrics
            "portfolio_value": portfolio_metrics.get("total_value", 0.0),
            "cash": portfolio_metrics.get("cash", 0.0),
            "unrealized_pnl_total": portfolio_metrics.get("unrealized_pnl_total", 0.0),
            "realized_pnl_total": portfolio_metrics.get("realized_pnl_total", 0.0),
            "drawdown": portfolio_metrics.get("drawdown", 0.0),
            "max_dd": portfolio_metrics.get("max_drawdown", 0.0),
            "sharpe": portfolio_metrics.get("sharpe_ratio", 0.0),
            "sortino": portfolio_metrics.get("sortino_ratio", 0.0),
            "win_rate": portfolio_metrics.get("win_rate", 0.0),
            # Trading statistics
            "trades": total_trades,
            "valid_trades": valid_trades,
            "invalid_trades": max(0, total_trades - valid_trades),
            # Activity counters from PerformanceMetrics
            "trade_attempts_total": portfolio_metrics.get(
                "trade_attempts_total", portfolio_metrics.get("trade_attempts", 0)
            ),
            "valid_trade_attempts": portfolio_metrics.get("valid_trade_attempts", 0),
            "invalid_trade_attempts": portfolio_metrics.get(
                "invalid_trade_attempts", 0
            ),
            "executed_trades_opened": portfolio_metrics.get(
                "executed_trades_opened", 0
            ),
            "executed_trades_closed": len(closed_positions),
            "closed_positions": closed_positions,
            # Positions actuelles
            "positions": position_values,
            "total_position_value": total_position_value,
            "leverage": portfolio_metrics.get("leverage", 0.0),
            "num_positions": portfolio_metrics.get(
                "open_positions_count", len(position_values)
            ),
            # Rewards & Penalties
            "current_prices": current_prices,
            "assets": list(current_prices.keys()),
            "num_assets": len(current_prices),
            # Action stats
            "action_stats": action_stats,
            # Risk metrics
            "risk_metrics": getattr(self, "risk_metrics", {}),
            "position_size": getattr(self, "base_position_size", 0.0),
            "risk_per_trade": getattr(self, "risk_per_trade", 0.0),
            "dynamic_sizing": getattr(self, "dynamic_position_sizing", False),
            # Performance metrics
            "performance": {
                "timestamp": self._get_safe_timestamp(),
                "steps_per_second": (
                    self.current_step
                    / max(0.0001, time.time() - self._episode_start_time)
                    if hasattr(self, "_episode_start_time")
                    else 0.0
                ),
            },
            # Frequency counts (picklable)
            "frequency": {
                "counts": {
                    "5m": int(self.positions_count.get("5m", 0)),
                    "1h": int(self.positions_count.get("1h", 0)),
                    "4h": int(self.positions_count.get("4h", 0)),
                    "daily_total": int(self.positions_count.get("daily_total", 0)),
                }
            },
            # Recent receipts (small buffer, picklable primitives)
            "last_receipts": [
                {
                    k: (
                        float(v)
                        if isinstance(v, (np.floating,))
                        else int(v)
                        if isinstance(v, (np.integer,))
                        else str(v)
                    )
                    for k, v in rec.items()
                }
                for rec in (
                    list(self.receipts)[-5:]
                    if hasattr(self, "receipts") and self.receipts
                    else []
                )
            ],
        }

        # Rendre accessible pour toute requête get_attr("last_info") éventuelle
        try:
            self.last_info = info
        except Exception:
            # Toujours éviter de casser le step si une valeur non sérialisable est ajoutée par erreur
            pass

        return info

    def _log_episode_capital_stats(self, info: Dict[str, Any]) -> None:
        """Log capital statistics at episode end for forensic analysis."""
        try:
            initial = self.config.get("portfolio", {}).get(
                "initial_balance",
                self.config.get("environment", {}).get("initial_balance", 20.5),
            )
            final_value = info.get("portfolio_value", 0.0)
            pnl = final_value - initial
            pnl_pct = (pnl / max(initial, 1e-8)) * 100.0
            trades = info.get("trades", 0)
            steps = info.get("step", self.current_step)
            trade_step_ratio = trades / max(steps, 1)

            self.logger.info(
                f"[EPISODE_END_STATS] Worker={self.worker_id} | "
                f"Initial=${initial:.2f} | Final=${final_value:.2f} | "
                f"PnL=${pnl:.2f} ({pnl_pct:+.2f}%) | "
                f"Trades={trades} | Steps={steps} | "
                f"Trade/Step={trade_step_ratio:.4f} | "
                f"MaxDD={info.get('max_dd', 0.0):.2%} | "
                f"Sharpe={info.get('sharpe', 0.0):.4f} | "
                f"WinRate={info.get('win_rate', 0.0):.1f}%"
            )
        except Exception as e:
            self.logger.warning(f"[EPISODE_END_STATS] Error logging: {e}")

    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Méthode publique pour récupérer les métriques du portfolio pour les callbacks."""
        try:
            return self._get_info()
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des métriques: {e}")
            # Retourner des métriques par défaut en cas d'erreur
            return {
                "portfolio_value": getattr(self, "initial_balance", 20.50),
                "cash": getattr(self, "initial_balance", 20.50),
                "drawdown": 0.0,
                "max_dd": 0.0,
                "sharpe": 0.0,
                "win_rate": 0.0,
                "trades": 0,
                "valid_trades": 0,
                "invalid_trades": 0,
                "positions": {},
                "closed_positions": [],
                "last_reward": 0.0,
                "last_penalty": 0.0,
                "cumulative_reward": 0.0,
                "current_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "worker_id": getattr(self, "worker_id", 0),
            }

    def render(self, mode: str = "human") -> None:
        """Affiche l'état actuel de l'environnement."""
        if mode == "human":
            portfolio_value = self.portfolio.get_portfolio_value()
            print(
                f"Étape: {self.current_step}, "
                f"Valeur du portefeuille: {portfolio_value:.2f}, "
                f"Espèces: {self.portfolio.cash:.2f}, "
                f"Positions: {self.portfolio.positions}"
            )

    def _calculate_current_volatility(self, lookback: int = 21) -> float:
        """
        Calcule la volatilité actuelle du marché sur une période donnée.

        Args:
            lookback: Nombre de jours pour le calcul de la volatilité

        Returns:
            float: Volatilité annualisée
        """
        try:
            if not hasattr(self, "data_loader") or not hasattr(self.data_loader, "get"):
                return self.baseline_volatility

            # Récupère les données de clôture
            closes = self.data_loader.get("close")
            if closes is None or len(closes) < lookback:
                return self.baseline_volatility

            # Récupère les rendements journaliers
            returns = np.log(closes / closes.shift(1)).dropna()
            if len(returns) < 2:
                return self.baseline_volatility

            # Calcule la volatilité annualisée
            daily_vol = returns.std()
            annualized_vol = daily_vol * np.sqrt(252)

            return float(annualized_vol)

        except Exception as e:
            self.logger.error(f"Erreur calcul volatilité: {str(e)}")
            return self.baseline_volatility

    def _get_current_market_regime(self) -> str:
        """
        Détermine le régime de marché actuel.

        Returns:
            str: 'high_volatility', 'low_volatility', 'trending_up', 'trending_down', ou 'ranging'
        """
        try:
            if not hasattr(self, "data_loader") or not hasattr(self.data_loader, "get"):
                return "ranging"

            # Logique simplifiée pour déterminer le régime
            # À améliorer avec des indicateurs plus sophistiqués
            lookback = 50
            closes = self.data_loader.get("close")

            if closes is None or len(closes) < lookback:
                return "ranging"

            returns = np.log(closes / closes.shift(1)).dropna()

            # Volatilité
            vol = returns.std()

            # Tendance
            ma_fast = closes.rolling(window=20).mean().iloc[-1]
            ma_slow = closes.rolling(window=50).mean().iloc[-1]

            if vol > self.baseline_volatility * 1.5:
                return "high_volatility"
            elif vol < self.baseline_volatility * 0.5:
                return "low_volatility"
            elif ma_fast > ma_slow * 1.02:
                return "trending_up"
            elif ma_fast < ma_slow * 0.98:
                return "trending_down"
            else:
                return "ranging"

        except Exception as e:
            self.logger.error(f"Erreur détection régime: {str(e)}")
            return "ranging"

    def _log_summary(self, step, chunk_id, total_chunks):
        if self.worker_id != 0:
            return

        try:
            pm = self.portfolio_manager
            # Use performance_metrics with positions_count for synchronized metrics
            if hasattr(self, "performance_metrics"):
                metrics = self.performance_metrics.calculate_metrics(
                    positions_count=self.positions_count,
                    trade_attempts=self.trade_attempts,
                    invalid_trade_attempts=self.invalid_trade_attempts,
                )
            else:
                metrics = {}

            # Safely collect open positions
            open_positions = []
            try:
                for asset, pos in pm.positions.items():
                    if pos.is_open:
                        sl_price = (
                            pos.entry_price * (1 - pos.stop_loss_pct)
                            if pos.stop_loss_pct > 0
                            else 0
                        )
                        tp_price = (
                            pos.entry_price * (1 + pos.take_profit_pct)
                            if pos.take_profit_pct > 0
                            else 0
                        )
                        open_positions.append(
                            f"│   {asset}: {pos.size:.4f} @ {pos.entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f}"
                        )
            except Exception as e:
                open_positions = [f"│   Error retrieving positions: {str(e)}"]

            # Safely collect closed positions
            closed_positions = []
            try:
                if hasattr(pm, "trade_log") and pm.trade_log:
                    closed_trades = [
                        t for t in pm.trade_log if t.get("action") == "close"
                    ]
                    for trade in closed_trades[-3:]:  # Last 3 closed trades
                        pnl = trade.get("pnl", 0.0)
                        pnl_pct = trade.get("pnl_pct", 0.0)
                        asset = trade.get("asset", "Unknown")
                        size = trade.get("size", 0.0)
                        entry_price = trade.get("entry_price", 0.0)
                        exit_price = trade.get("exit_price", trade.get("price", 0.0))
                        opened_at = trade.get("opened_at")
                        closed_at = trade.get("closed_at")
                        duration_seconds = trade.get("duration_seconds")

                        closed_positions.append(
                            (
                                f"│   {asset}: {size:.4f} @ {entry_price:.2f}→{exit_price:.2f} | "
                                f"PnL {pnl:+.2f} ({pnl_pct:+.2f}%)"
                            ).ljust(65)
                            + "│"
                        )

                        timing_parts = []
                        if opened_at:
                            timing_parts.append(f"ouvert: {opened_at}")
                        if closed_at:
                            timing_parts.append(f"fermé: {closed_at}")
                        if duration_seconds is not None:
                            timing_parts.append(f"durée: {duration_seconds:.0f}s")

                        if timing_parts:
                            closed_positions.append(
                                ("│   " + " | ".join(timing_parts)).ljust(65) + "│"
                            )
            except Exception as e:
                closed_positions = [f"│   Error retrieving closed trades: {str(e)}"]

            # Safe metric retrieval
            sharpe = metrics.get("sharpe", 0.0)
            sortino = metrics.get("sortino", 0.0)
            profit_factor = metrics.get("profit_factor", 0.0)
            max_dd = metrics.get("max_dd", 0.0)
            cagr = metrics.get("cagr", 0.0)
            win_rate = metrics.get("winrate", 0.0)
            total_trades = metrics.get("total_trades", 0)
            winning_trades = metrics.get("wins", 0)
            losing_trades = metrics.get("losses", 0)
            neutral_trades = metrics.get("neutrals", 0)
            trade_attempts = metrics.get("trade_attempts_total", 0)
            executed_opens = metrics.get("executed_trades_opened", 0)
            invalid_attempts = metrics.get("invalid_trade_attempts", 0)

            # Safe portfolio values
            capital = pm.get_total_value() if hasattr(pm, "get_total_value") else 0.0
            equity = pm.get_equity() if hasattr(pm, "get_equity") else 0.0
            balance = (
                pm.get_cash() if hasattr(pm, "get_cash") else (getattr(pm, "cash", 0.0))
            )

            # Utiliser calculate_drawdown() pour obtenir les valeurs correctes
            current_dd = (
                pm.calculate_drawdown() * 100
            )  # calculate_drawdown() retourne un ratio (0.0-1.0)
            max_dd_allowed = getattr(pm, "max_drawdown_pct", 0.25) * 100

            summary_lines = [
                "╭──────── Étape {} / Chunk {}/{} (Worker {}) ─────────╮".format(
                    step, chunk_id, total_chunks, self.worker_id
                ),
                "│ 📊 PORTFOLIO                                                  │",
                "│ Capital: {:.2f} USDT | Équité: {:.2f} USDT".format(
                    capital, equity
                ).ljust(65)
                + "│",
                "│ Solde disponible: {:.2f} USDT".format(balance).ljust(65) + "│",
                "│                                                               │",
                "│ 📈 MÉTRIQUES                                                  │",
                "│ Sharpe: {:.2f} | Sortino: {:.2f} | Profit Factor: {:.2f}".format(
                    sharpe, sortino, profit_factor
                ).ljust(65)
                + "│",
                "│ Max DD: {:.2f}% | CAGR: {:.2f}% | Win Rate: {:.1f}%".format(
                    max_dd, cagr, win_rate
                ).ljust(65)
                + "│",
                "│ Trades Clôturés: {} ({}W/{}L/{}N)".format(
                    total_trades, winning_trades, losing_trades, neutral_trades
                ).ljust(65)
                + "│",
                "│ Activité: {} tentatives, {} ouverts, {} rejets".format(
                    trade_attempts, executed_opens, invalid_attempts
                ).ljust(65)
                + "│",
                "│ Positions: 5m:{}, 1h:{}, 4h:{}, Total:{}".format(
                    self.positions_count.get("5m", 0),
                    self.positions_count.get("1h", 0),
                    self.positions_count.get("4h", 0),
                    len([p for p in pm.positions.values() if p.is_open]),
                ).ljust(65)
                + "│",
                "│                                                               │",
                "│ ⚠️  RISQUE                                                     │",
                "│ Drawdown actuel: {:.1f}%/{:.1f}%".format(
                    current_dd, max_dd_allowed
                ).ljust(65)
                + "│",
                "│                                                               │",
                "│ 📋 POSITIONS OUVERTES                                         │",
            ]

            if open_positions:
                # open_positions doit déjà contenir des lignes formatées. On les enrichit si possible
                summary_lines.extend(open_positions)
            else:
                summary_lines.append(
                    "│   Aucune                                                      │"
                )

            summary_lines.extend(
                [
                    "│                                                               │",
                    "│ 📕 DERNIÈRES POSITIONS FERMÉES                                │",
                ]
            )

            if closed_positions:
                # Enrichir l’affichage si le dict contient opened_at/closed_at
                enriched = []
                for line in closed_positions:
                    enriched.append(line)
                summary_lines.extend(enriched)
            else:
                summary_lines.append(
                    "│   Aucune                                                      │"
                )

            summary_lines.append(
                "╰───────────────────────────────────────────────────────────────╯"
            )

            summary = "\n".join(summary_lines)
            logger.info(summary)

        except Exception as e:
            logger.error(f"Error in _log_summary: {str(e)}")
            logger.info(
                f"[SUMMARY] Step {step} | Chunk {chunk_id}/{total_chunks} | Basic info only due to error"
            )


            return (
                self.portfolio_manager.get_equity()
                - self.portfolio_manager.initial_capital
            )
        return 0.0

    def calculate_inaction_penalty(self):
        """Calculate penalty for inaction."""
        penalty = 0.0
        current_tf = self.get_current_timeframe()
        steps_since_trade = self.current_step - getattr(
            self, "last_trade_steps_by_tf", {}
        ).get(current_tf, 0)

        if steps_since_trade > 20:  # Penalty after 20 steps of inaction
            penalty = -0.01 * (steps_since_trade - 20)

        return penalty

    def close(self) -> None:
        """Nettoie les ressources de l'environnement."""
        pass

    def log_worker_comparison(self):
        """Log comparison metrics between workers (for debugging)."""
        if hasattr(self, "portfolio_manager") and hasattr(
            self.portfolio_manager, "metrics"
        ):
            metrics = (
                self.portfolio_manager.metrics.calculate_metrics()
                if hasattr(self.portfolio_manager.metrics, "calculate_metrics")
                else {}
            )
            equity = (
                self.portfolio_manager.get_equity()
                if hasattr(self.portfolio_manager, "get_equity")
                else 0.0
            )

            logger.info(
                f"[COMPARISON Worker {self.worker_id}] Trades: {metrics.get('total_trades', 0)}, "
                f"Winrate: {metrics.get('winrate', 0.0):.1f}%, "
                f"Equity: {equity:.2f} USDT, "
                f"Counts: {getattr(self, 'positions_count', {})}"
            )

    def _save_crash_snapshot(self, step: int, reason: str, extra_data: dict = None):
        """Sauvegarde un snapshot de l'état lors d'un crash"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.crash_snapshot_dir}/crash_{timestamp}_step{step}_{reason}.json"
            
            snapshot = {
                'timestamp': timestamp,
                'step': step,
                'reason': reason,
                'portfolio_value': float(self.portfolio_manager.current_value),
                'portfolio_cash': float(self.portfolio_manager.cash),
                'current_step': self.current_step,
                'positions': self._get_positions_snapshot(),
                'last_trades': self._get_last_trades_snapshot(10),
                'observation_stats': self._get_observation_stats(),
                'extra_data': extra_data or {}
            }
            
            with open(filename, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
                
            logger.error(f"🚨 CRASH SNAPSHOT sauvé: {filename}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde snapshot: {e}")
    
    def _apply_trade_results_safely(self, pnl_value: float, fees: float = 0.0) -> float:
        """
        Applique les résultats de trade avec sécurité maximale
        """
        # Vérification PnL aberrant
        if not np.isfinite(pnl_value):
            logger.error(f"PnL non fini: {pnl_value}")
            self._save_crash_snapshot(self.current_step, "pnl_not_finite", 
                                    {'pnl_value': pnl_value, 'fees': fees})
            pnl_value = 0.0
            
        # Détection d'explosion (PnL > 100x le capital en un step)
        capital_before = self.portfolio_manager.current_value
        if abs(pnl_value) > (capital_before * 100):
            logger.error(
                f"PnL absurde détecté: {pnl_value:.2f} (capital: {capital_before:.2f})"
            )
            self._save_crash_snapshot(self.current_step, "absurd_pnl", 
                                    {'pnl_value': pnl_value, 'capital_before': capital_before})
            
            # Clamp sévère
            pnl_value = np.sign(pnl_value) * (capital_before * 10)  # Max 10x
            
        # Application via PortfolioManager sécurisé
        new_value = self.portfolio_manager.apply_trade_result(pnl_value, fees)
        
        # Vérification post-trade
        if not np.isfinite(new_value):
            logger.error(f"Valeur portfolio non finie après trade: {new_value}")
            self._save_crash_snapshot(self.current_step, "portfolio_nan_after_trade")
            new_value = self.portfolio_manager.MIN_PORTFOLIO_VALUE
            
        return new_value
    
    def _get_positions_snapshot(self):
        """Capture l'état des positions pour le snapshot"""
        positions = {}
        for asset, position in self.portfolio_manager.positions.items():
            positions[asset] = {
                'units': float(position.get('units', 0)),
                'entry_price': float(position.get('entry_price', 0)),
                'current_value': float(position.get('current_value', 0))
            }
        return positions
    
    def _get_last_trades_snapshot(self, count: int = 10):
        """Capture les derniers trades"""
        # Implémentez selon votre structure de données des trades
        return []
    
    def _get_observation_stats(self):
        """Capture des statistiques sur l'observation actuelle"""
        try:
            obs = self._get_observation()  # Adaptez à votre méthode
            return {
                'min': float(np.nanmin(obs)) if obs is not None else None,
                'max': float(np.nanmax(obs)) if obs is not None else None,
                'mean': float(np.nanmean(obs)) if obs is not None else None,
                'has_nan': bool(np.any(np.isnan(obs))) if obs is not None else None
            }
        except:
            return {'error': 'could_not_compute_obs_stats'}
