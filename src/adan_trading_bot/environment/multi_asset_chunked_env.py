#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Environnement de trading multi-actifs avec chargement par morceaux.

Ce module implémente un environnement de trading pour plusieurs actifs
avec chargement efficace des données par lots.
"""
# Standard library imports
import json
from typing import Dict, List, Optional
import logging
import os
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TypeVar

# Third-party imports
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

# Local application imports
from ..data_processing.data_loader import ChunkedDataLoader
from .dynamic_behavior_engine import DynamicBehaviorEngine
from ..data_processing.observation_validator import ObservationValidator
from .order_manager import OrderManager
from ..portfolio.portfolio_manager import PortfolioManager
from ..common.logging_utils import create_smart_logger, configure_smart_logger
from .reward_calculator import RewardCalculator

try:
    # Import from data_processing (canonical location)
    from adan_trading_bot.data_processing.state_builder import (
        StateBuilder,
        TimeframeConfig,
    )
except Exception:
    # Fallback for compatibility (should not be used under normal conditions)
    pass


def clean_worker_id(worker_id):
    """
    Nettoie l'ID du worker pour éviter les erreurs JSONL.
    Convertit 'w0' en 0, 'W0' en 0, 'worker-1' en 1, '[WORKER-2]' en 2, etc.
    Gère tous les formats possibles d'ID worker.

    Args:
        worker_id: ID du worker (peut être string, int, ou autre)

    Returns:
        int: ID du worker nettoyé (toujours un entier positif)
    """
    if worker_id is None:
        return 0

    if isinstance(worker_id, int):
        return max(0, worker_id)  # Assurer que c'est positif

    if isinstance(worker_id, str):
        import re
        # Nettoyer la chaîne: supprimer les crochets, espaces, etc.
        clean_str = worker_id.strip().upper()

        # Extraire tous les nombres de la chaîne
        numbers = re.findall(r'\d+', clean_str)

        if numbers:
            try:
                # Prendre le premier nombre trouvé
                return int(numbers[0])
            except (ValueError, IndexError):
                pass

        # Si aucun nombre trouvé, essayer des patterns spécifiques
        patterns = [
            r'^W(\d+)$',           # W0, W1, etc.
            r'^WORKER[-_]?(\d+)$', # WORKER0, WORKER-1, WORKER_2, etc.
            r'^RANK[-_]?(\d+)$',   # RANK0, RANK-1, etc.
        ]

        for pattern in patterns:
            match = re.match(pattern, clean_str)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    pass

        # Fallback: essayer de convertir directement
        try:
            return max(0, int(clean_str))
        except ValueError:
            pass

    # Dernier recours: utiliser hash pour avoir un entier déterministe
    try:
        return abs(hash(str(worker_id))) % 1000
    except Exception:
        return 0


    from adan_trading_bot.environment.state_builder import (
        StateBuilder,
        TimeframeConfig,
    )  # pragma: no cover

# Type variables for generics
T = TypeVar("T")

# Constants
DEFAULT_PORTFOLIO_STATE_SIZE = 17
DEFAULT_WINDOW_SIZE = 50
DEFAULT_INITIAL_BALANCE = 10000.0
MAX_STEPS = 10000

# Constants for reload attempts and fallback mechanism
MAX_RELOAD_ATTEMPTS = 3
RELOAD_FALLBACK_CHUNK = 0
RELOAD_RETRY_DELAY = 0.1  # seconds

# Configuration du logger
logger = logging.getLogger(__name__)

# Désactiver la propagation des logs des bibliothèques tierces
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Configure logging for this module
logger.setLevel(logging.DEBUG)

# Add console handler if not already configured
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Ensure basic config is set for other loggers
if not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

# Désactiver la propagation pour éviter les doublons
logger.propagate = False



# Patch Gugu & March - Système d'Excellence
try:
    from ..patches.gugu_march_excellence_rewards import (
        GuguMarchExcellenceRewards,
        ExcellenceMetrics,
        create_excellence_rewards_system
    )
    EXCELLENCE_SYSTEM_AVAILABLE = True
    logger.info("[GUGU-MARCH] Excellence rewards system loaded successfully")
except ImportError as e:
    logger.warning(f"[GUGU-MARCH] Excellence system not available: {e}")
    EXCELLENCE_SYSTEM_AVAILABLE = False


class MultiAssetChunkedEnv(gym.Env):
    """Environnement de trading multi-actifs avec chargement par morceaux.

    Cet environnement gère plusieurs actifs et intervalles de temps, avec
    support pour des espaces d'actions discrets et continus. Il utilise un
    constructeur d'état pour créer l'espace d'observation et un gestionnaire
    de portefeuille pour suivre les positions et le PnL.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Constantes pour les actions
    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(
        self,
        data: Dict[str, Dict[str, pd.DataFrame]],
        timeframes: List[str],
        window_size: int,
        features_config: Dict[str, List[str]],
        max_steps: int = 1000,
        initial_balance: float = 10000.0,
        commission: float = 0.001,
        reward_scaling: float = 1.0,
        render_mode: Optional[str] = None,
        enable_logging: bool = True,
        log_dir: str = "logs",
        external_dbe: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """Initialise l'environnement de trading multi-actifs.

        Args:
            data: Données de trading sous forme de dictionnaire.
            timeframes: Liste des intervalles de temps pour les données.
            window_size: Taille de la fenêtre pour les observations.
            features_config: Configuration des caractéristiques pour chaque intervalle de temps.
            max_steps: Nombre maximum d'étapes par épisode.
            initial_balance: Balance initiale du portefeuille.
            commission: Commission pour chaque transaction.
            reward_scaling: Échelle de récompense pour les récompenses.
            render_mode: Mode de rendu pour l'environnement.
            enable_logging: Activation de la journalisation.
            log_dir: Répertoire pour les fichiers de logs.
            external_dbe: DBE externe pré-existant à réutiliser (pour l'immortalité d'ADAN).
        """
        super().__init__()

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # DIAGNOSTIC: Tracer les instances d'environnement pour identifier les recréations
        import uuid
        self.env_instance_id = str(uuid.uuid4())[:8]

        # Initialize logger lock for thread-safe logging
        self.logger_lock = threading.Lock()

        # Initialize configuration attributes
        self.worker_config = kwargs.get("worker_config", {})
        self.config = kwargs.get("config", {})

        # Get worker ID for synchronized logging (nettoyé pour éviter les erreurs JSONL)
        raw_worker_id = self.worker_config.get("worker_id", self.worker_config.get("rank", "W0"))
        self.worker_id = clean_worker_id(raw_worker_id)

        # SOLUTION IMMORTALITÉ ADAN: Stocker le DBE externe s'il est fourni
        self.external_dbe = external_dbe
        if external_dbe is not None:
            self.logger.critical(f"🔄 RÉCEPTION DBE IMMORTEL: ENV_ID={self.env_instance_id}, Worker={self.worker_id}, DBE_ID={id(external_dbe)}")

        # DIAGNOSTIC: Log critique pour tracer la création d'instances
        self.logger.critical(f"🆕 NOUVELLE INSTANCE ENV CRÉÉE: ID={self.env_instance_id}, Worker={self.worker_id}")

        # Initialize smart logger for intelligent multi-worker logging (after worker_id is set)
        total_workers = kwargs.get('total_workers', 4)  # Default to 4 workers
        self.smart_logger = create_smart_logger(self.logger, self.worker_id, total_workers)
        configure_smart_logger(self.smart_logger, 'training')  # Use training configuration

        # Initialize risk parameters
        self.risk_params = self.worker_config.get("risk_parameters", {})
        self._init_risk_parameters()

        # Initialize shared_buffer to avoid AttributeError in step()
        self.shared_buffer = None

        # Initialize strict_validation flag with a default value
        self.strict_validation = kwargs.get("strict_validation", False)

        # Initialize self.assets from data
        self.assets = list(data.keys())
        if not self.assets:
            raise ValueError("No assets found in the provided data.")

        # Configuration du cache d'observations
        self._observation_cache = {}  # Cache des observations
        self._max_cache_size = 1000  # Taille maximale du cache
        self._cache_hits = 0  # Succès du cache
        self._cache_misses = 0  # Échecs du cache
        self._last_observation = None
        self._last_market_timestamp: Optional[pd.Timestamp] = None
        self._last_asset_timestamp: Dict[str, pd.Timestamp] = {}
        # Observation courante
        # Ensure attribute exists for downstream access in _get_observation
        self._current_obs = None
        self._cache_access = {}  # Suivi de l'utilisation

        # Initialisation du validateur d'observations
        n_features = len(next(iter(features_config.values())))
        validator_config = {
            "timeframes": timeframes,
            "n_assets": len(self.assets),
            "window_size": window_size,
            "n_features": n_features,
            "portfolio_state_size": DEFAULT_PORTFOLIO_STATE_SIZE,
        }
        self.observation_validator = ObservationValidator(validator_config)

        # Initialisation des compteurs et états
        self.current_chunk = 0
        self.current_chunk_idx = 0
        self.done = False  # État done
        self.global_step = 0  # Compteur global d'étapes
        self.current_step = 0  # Étape courante dans l'épisode

        # Compteurs pour surveiller le forward fill de prix
        self.interpolation_count = 0  # Nombre de forward fills effectués (legacy name)
        self.total_steps_with_price_check = 0  # Nombre total d'étapes avec vérification de prix

        # Configuration et compteurs de fréquence des positions
        self.frequency_config = self.config.get("trading_rules", {}).get("frequency", {})
        self.positions_count = {
            '5m': 0,
            '1h': 0,
            '4h': 0,
            'daily_total': 0
        }

        # Tracking des trades pour éviter de compter les duplicatas
        self.last_trade_ids = set()
        self.current_timeframe_for_trade = "5m"  # Default timeframe
        self.daily_reset_step = 0  # Step du dernier reset journalier
        self.current_day = 0  # Jour courant pour suivi
        self.last_trade_steps_by_tf = {}  # Dictionnaire des derniers trades par timeframe

        # Initialisation du chargeur de données
        self.data_loader_instance = None

        # Suivi des paliers
        self.current_tier = None
        self.previous_tier = None
        self.episode_count = 0
        self.episodes_in_tier = 0
        self.best_portfolio_value = 0.0
        self.last_tier_change_step = 0
        self.tier_history = []  # Historique des paliers

        # Suivi des trades et métriques de risque
        self.last_trade_step = (
            -1
        )  # Dernière étape où un trade a été effectué (-1 = aucun trade)
        self.risk_metrics = {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }
        self.performance_history = []  # Historique des performances

        # Initialisation des composants critiques
        self._is_initialized = False  # Standardisation sur _is_initialized

        # Compteurs pour détecter les forward-fill excessifs
        self._price_read_success_count = 0
        self._price_forward_fill_count = 0
        self._forward_fill_threshold = 0.02  # 2% de forward-fill maximum

        # Suivi des récompenses
        self._last_reward = 0.0
        self._cumulative_reward = 0.0
        try:
            self._initialize_components()
            self._is_initialized = True
        except Exception as e:
            self.logger.error("Erreur lors de l'initialisation: %s", str(e))
            raise

    def _epoch_reset(self, force: bool = False, new_epoch: bool = False):
        """
        Centralized call to portfolio.reset(...) with config-driven threshold.

        Args:
            force: Force a full reset if True
            new_epoch: If True, indicates this is the start of a new epoch
        """
        min_cap = getattr(self, "config", {}).get("min_capital_before_reset", 11.0)
        self.portfolio.reset(
            new_epoch=new_epoch,
            force=force,
            min_capital_before_reset=min_cap
        )

    def update_risk_parameters(self, market_conditions=None):
        """
        Met à jour les paramètres de risque en fonction des conditions de marché
        et du régime de marché détecté.

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
                        "asset": first_asset
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
            if close_price is None or not isinstance(close_price, (int, float)) or close_price <= 0:
                # Utiliser des conditions par défaut
                market_conditions = self._get_default_market_conditions()
                close_price = market_conditions.get("close", 50000.0)

            # 1. Détection du régime de marché
            regime, confidence = self._detect_market_regime(market_conditions)

            # 2. Mise à jour des paramètres de risque en fonction du régime
            risk_params = self._calculate_risk_parameters(regime, market_conditions)

            # 3. Application des limites de risque
            self._apply_risk_limits(risk_params)

            # 4. Mise à jour du gestionnaire de portefeuille
            if hasattr(self, "portfolio"):
                try:
                    self.portfolio.update_risk_parameters(risk_params)
                except AttributeError:
                    logger.warning("[FALLBACK] update_risk_parameters absent – Valeurs par défaut.")
                    self.portfolio.sl_pct = risk_params.get('sl', 0.02)
                    self.portfolio.pos_size_pct = min(risk_params.get('pos_size', 0.825), 0.9)  # Clip à 90%
                    logging.warning("update_risk_parameters manquant - Ajouté fallback")
                    # Fallback direct pour éviter le crash
                    if hasattr(self.portfolio, 'sl_pct'):
                        self.portfolio.sl_pct = risk_params.get('sl', 0.02)
                    if hasattr(self.portfolio, 'tp_pct'):
                        self.portfolio.tp_pct = risk_params.get('tp', 0.04)

                # Journalisation des changements significatifs
                if hasattr(self, "last_risk_params") and self.last_risk_params:
                    changed = []
                    for k, v in risk_params.items():
                        if k in self.last_risk_params:
                            # Gestion explicite des types pour éviter DTypePromotionError
                            if isinstance(v, (int, float)) and isinstance(self.last_risk_params[k], (int, float)):
                                # Comparaison numérique avec np.isclose
                                if not np.isclose(v, self.last_risk_params[k], rtol=1e-3):
                                    changed.append(
                                        f"{k}: {self.last_risk_params[k]:.4f}→{v:.4f}"
                                    )
                            else:
                                # Comparaison directe pour chaînes et autres types
                                if v != self.last_risk_params[k]:
                                    changed.append(f"{k}: {self.last_risk_params[k]}→{v}")

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
                        exc_info=True
                    )
            except AttributeError:
                # Fallback si logger_lock n'existe pas encore
                self.logger.error(
                    f"[{self.worker_id}] RISK UPDATE ERROR: Failed to update risk parameters: {e}",
                    exc_info=True
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
            if hasattr(self, 'current_data') and self.current_data is not None:
                for asset_data in self.current_data.values():
                    if isinstance(asset_data, dict):
                        for timeframe_data in asset_data.values():
                            if hasattr(timeframe_data, 'iloc') and len(timeframe_data) > 0:
                                # Trouver la colonne close
                                close_col = None
                                for col in ['close', 'CLOSE', 'Close']:
                                    if col in timeframe_data.columns:
                                        close_col = col
                                        break

                                if close_col is not None:
                                    # FORWARD FILL SEULEMENT - pas d'interpolation
                                    last_valid_prices = timeframe_data[close_col].dropna()
                                    if len(last_valid_prices) > 0:
                                        last_price = float(last_valid_prices.iloc[-1])

                                        # Vérifier l'âge du prix si timestamp disponible
                                        if 'timestamp' in timeframe_data.columns:
                                            last_timestamp = timeframe_data[timeframe_data[close_col].notna()].iloc[-1]['timestamp']
                                            if isinstance(last_timestamp, (int, float)):
                                                last_timestamp = pd.Timestamp(last_timestamp, unit='ms')
                                            elif not isinstance(last_timestamp, pd.Timestamp):
                                                last_timestamp = pd.Timestamp(last_timestamp)

                                            age = current_time - last_timestamp
                                            if age > max_age:
                                                self.logger.warning(f"PRICE_TOO_OLD | last_price={last_price} | age={age} | max_age={max_age}")
                                                continue

                                        return last_price

            # Fallback: utiliser le dernier prix connu stocké
            if hasattr(self, '_last_known_price') and self._last_known_price is not None:
                return float(self._last_known_price)

            return None

        except Exception as e:
            self.logger.debug(f"Erreur lors de la récupération du dernier prix valide: {e}")
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
                self.logger.info(f"[ENV_DEBUG] Observation shape: {market_observation.shape}, "
                              f"dtype: {market_observation.dtype}, "
                              f"NaNs: {np.isnan(market_observation).sum()}")
            else:
                self.logger.info("[ENV_DEBUG] market_observation is None")

            if market_observation is None or market_observation.size == 0:
                self.logger.warning("Observation de marché vide ou invalide pour les indicateurs.")
                return {}

            # Extraire les indicateurs du timeframe 5m (le plus granulaire)
            # L'observation est de forme (timeframes, window_size, features)
            # Le timeframe 5m est généralement en premier (index 0)

            # Récupérer les noms des features pour le timeframe 5m
            features_5m = self.state_builder.get_feature_names('5m')

            # Vérifications robustes de la structure des données
            if len(features_5m) == 0:
                self.logger.debug("Aucune feature configurée pour 5m, utilisation de valeurs par défaut")
                features_5m = ["OPEN", "HIGH", "LOW", "close", "VOLUME", "RSI_14", "STOCH_K_14_3_3",
                              "STOCH_D_14_3_3", "MACD_HIST_12_26_9", "ATR_14", "EMA_5", "EMA_12",
                              "BB_UPPER", "BB_MIDDLE", "BB_LOWER"]

            # Vérification de la forme de l'observation
            if market_observation.ndim != 3:
                self.logger.warning(f"Forme d'observation incorrecte: {market_observation.shape}, attendu 3D (timeframes, window, features)")
                return self._get_default_market_conditions()

            if market_observation.shape[0] == 0 or market_observation.shape[1] == 0:
                self.logger.warning(f"Observation vide: {market_observation.shape}")
                return self._get_default_market_conditions()

            # Récupérer la dernière ligne de données du timeframe 5m dans l'observation
            try:
                current_5m_data = market_observation[0, -1, :] # Premier timeframe, dernière ligne de la fenêtre
            except IndexError as e:
                self.logger.error(f"Erreur d'indexation lors de l'extraction des données 5m: {e}")
                return self._get_default_market_conditions()

            indicators = {}
            # Mapper les valeurs numériques aux noms des features avec vérifications
            for i, feature_name in enumerate(features_5m):
                if i < len(current_5m_data):
                    val = float(current_5m_data[i])
                    # Remplacer NaN ou Inf par des valeurs par défaut
                    if np.isnan(val) or np.isinf(val):
                        if "RSI" in feature_name.upper(): val = 50.0
                        elif "ADX" in feature_name.upper(): val = 20.0
                        elif "ATR" in feature_name.upper(): val = 0.01
                        elif "EMA" in feature_name.upper(): val = 1.0
                        elif "MACD_HIST" in feature_name.upper(): val = 0.0
                        elif "close" in feature_name.upper(): val = 1.0
                        elif "BB_" in feature_name.upper(): val = 1.0
                        elif "STOCH" in feature_name.upper(): val = 50.0
                        else: val = 0.0
                        self.logger.debug(f"NaN/Inf détecté pour {feature_name} (index {i}), remplacé par {val}")

                    indicators[feature_name.upper()] = val
                    self.logger.debug(f"Feature {i}: {feature_name.upper()} = {val}")
                else:
                    # Si la feature n'est pas disponible, utiliser une valeur par défaut
                    default_val = 50.0 if "RSI" in feature_name.upper() or "STOCH" in feature_name.upper() else 1.0 if "close" in feature_name.upper() or "EMA" in feature_name.upper() else 0.0
                    indicators[feature_name.upper()] = default_val
                    self.logger.debug(f"Feature manquante {feature_name} (index {i}), valeur par défaut: {default_val}")

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
                "BB_UPPER": indicators.get("BB_UPPER", indicators.get("close", 1.0) * 1.02),
                "BB_MIDDLE": indicators.get("BB_MIDDLE", indicators.get("close", 1.0)),
                "BB_LOWER": indicators.get("BB_LOWER", indicators.get("close", 1.0) * 0.98),
                "STOCH_K_14_3_3": indicators.get("STOCH_K_14_3_3", 50.0),
                "STOCH_D_14_3_3": indicators.get("STOCH_D_14_3_3", 50.0),
            }

            # Ajouter tous les autres indicateurs extraits
            essential_indicators.update(indicators)

            self.logger.debug(f"Indicateurs finaux extraits: {len(essential_indicators)} items")
            return essential_indicators

        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des indicateurs de marché: {str(e)}", exc_info=True)
            return self._get_default_market_conditions()

    def _get_default_market_conditions(self) -> Dict[str, float]:
        """Retourne des conditions de marché par défaut pour éviter les erreurs."""
        return {
            "close": 1.0,
            "close": 1.0,
            "VOLUME": 0.0,
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
                risk_params["stop_loss_pct"], 0.005, 0.10  # 0.5% minimum  # 10% maximum
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
                self.dbe.update_parameters(
                    {
                        "volatility_factor": 1.0
                        / max(risk_params.get("volatility_factor", 1.0), 0.1),
                        "max_position_size": risk_params["position_size"],
                        "current_volatility": (
                            market_data.get("atr", 0) / market_data.get("close", 1)
                            if "close" in market_data and market_data["close"] > 0
                            else 0.01
                        ),
                    }
                )

        except Exception as e:
            self.logger.error(
                f"Erreur lors de l'application des limites de risque: {str(e)}"
            )

    def _init_risk_parameters(self):
        """
        Initialise les paramètres de risque à partir de la configuration du worker.

        Cette méthode configure les paramètres de gestion des risques en fonction du profil du worker,
        avec une attention particulière aux micro-capitaux (<50 USDT) et aux différents régimes de marché.
        """
        # 1. Configuration de base du risque
        self.base_position_size = self.risk_params.get("position_size_pct", 0.1)
        self.risk_per_trade = self.risk_params.get("risk_per_trade_pct", 1.0)

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
            if profile == "conservative":
                self.min_position_size = 0.03  # 3%
                self.max_position_size = 0.15  # 15%
            elif profile == "aggressive":
                self.min_position_size = 0.08  # 8%
                self.max_position_size = 0.30  # 30%
            else:  # moderate ou par défaut
                self.min_position_size = 0.05  # 5%
                self.max_position_size = 0.25  # 25%

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
            f"Taille position: {self.base_position_size*100:.1f}% "
            f"({self.min_position_size*100:.1f}%-{self.max_position_size*100:.1f}%), "
            f"Risque/trade: {self.risk_per_trade:.2f}%, "
            f"Drawdown max: {self.max_drawdown_pct*100:.1f}%, "
            f"Trades conc.: {self.max_concurrent_trades}"
        )

    def _initialize_components(self) -> None:
        """Initialize all environment components in the correct order."""
        # DIAGNOSTIC: Tracer les appels à _initialize_components
        self.logger.critical(f"📋 APPEL _initialize_components pour ENV_ID={getattr(self, 'env_instance_id', 'UNKNOWN')}")

        # Initialize data loader FIRST to know the data structure
        if (
            not hasattr(self, "data_loader_instance")
            or self.data_loader_instance is None
        ):
            # Initialize data loader with correct assets
            self.data_loader_instance = self._init_data_loader(self.assets)
        self.data_loader = self.data_loader_instance

        # 2. Create TimeframeConfig from loaded data
        timeframe_configs = []
        if self.data_loader.features_by_timeframe:
            for tf_name, features in self.data_loader.features_by_timeframe.items():
                config = TimeframeConfig(
                    timeframe=tf_name, features=features, window_size=100
                )
                timeframe_configs.append(config)
        else:
            raise ValueError("No feature configuration in data loader.")

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

        # Initialize portfolio with mapped asset names, worker_id and shared PerformanceMetrics
        self.portfolio = PortfolioManager(
            config=portfolio_config, worker_id=self.worker_id, performance_metrics=self.performance_metrics
        )
        # Create alias for backward compatibility
        self.portfolio_manager = self.portfolio
        self.assets = mapped_assets  # Update self.assets

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

        # Utiliser la taille de fenêtre du timeframe 5m comme valeur par défaut
        default_window_size = window_sizes.get("5m", 20)

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
        self.state_builder = StateBuilder(
            features_config=features_config,
            window_size=default_window_size,  # Valeur par défaut pour la rétrocompatibilité
            include_portfolio_state=True,
            normalize=True,
        )

        # Configurer les tailles de fenêtres spécifiques dans le StateBuilder
        for tf, config in timeframe_configs.items():
            self.state_builder.set_timeframe_config(
                tf, config.window_size, config.features
            )
            self.logger.info(
                f"Configuration appliquée pour {tf}: fenêtre={config.window_size}, features={len(config.features)}"
            )

        # 5. Setup action and observation spaces (requires state_builder)
        self._setup_spaces()

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

        # Inject position_sizing config into DBE config if not present
        if 'position_sizing' not in dbe_config:
            env_risk_management = self.config.get('environment', {}).get('risk_management', {})
            if 'position_sizing' in env_risk_management:
                dbe_config['position_sizing'] = env_risk_management['position_sizing']
            else:
                dbe_config['position_sizing'] = {}

        # 8. PerformanceMetrics already initialized above (shared with PortfolioManager)

        # SOLUTION IMMORTALITÉ ADAN: Utiliser le DBE externe ou en créer un nouveau
        if hasattr(self, 'external_dbe') and self.external_dbe is not None:
            # Réutiliser le DBE immortel fourni lors de la création
            self.dynamic_behavior_engine = self.external_dbe
            self.logger.critical(f"👑 DBE IMMORTEL RÉUTILISÉ pour ENV_ID={self.env_instance_id}, DBE_ID={id(self.external_dbe)}")
        elif not hasattr(self, 'dynamic_behavior_engine') or self.dynamic_behavior_engine is None:
            # Créer un nouveau DBE uniquement si aucun n'existe
            self.dynamic_behavior_engine = DynamicBehaviorEngine(
                config=dbe_config, finance_manager=self.portfolio_manager, worker_id=self.worker_id
            )
            self.logger.critical(f"🧠 NOUVEAU DBE CRÉÉ pour ENV_ID={self.env_instance_id}")
        else:
            # Cas très rare : DBE déjà existant dans la même instance
            self.logger.critical(f"🔄 DBE EXISTANT PRÉSERVÉ pour ENV_ID={self.env_instance_id}")

        # Création d'un alias pour la rétrocompatibilité
        self.dbe = self.dynamic_behavior_engine

        # Connecter le DBE à l'environnement pour accès aux compteurs de fréquence
        if hasattr(self.dbe, 'set_env_reference'):
            self.dbe.set_env_reference(self)
            self.logger.info("DBE connecté à l'environnement pour suivi de fréquence")

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

        # Initialize the data loader with the correct config
        from ..data_processing.data_loader import ChunkedDataLoader

        self.data_loader = ChunkedDataLoader(
            config=self.config, worker_config=worker_config, worker_id=self.worker_id
        )

        return self.data_loader

    def _safe_load_chunk(self, chunk_idx: int, fallback_enabled: bool = True) -> Dict[str, Any]:
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

        for attempt in range(MAX_RELOAD_ATTEMPTS):
            try:
                self.smart_logger.info(f"[CHUNK_LOADER] Attempting to load chunk {chunk_idx} (attempt {attempt + 1}/{MAX_RELOAD_ATTEMPTS})", rotate=True)
                chunk_data = self.data_loader.load_chunk(chunk_idx)



                # Validate chunk data
                if chunk_data and any(chunk_data.get(asset) for asset in chunk_data):
                    self.smart_logger.info(f"[CHUNK_LOADER] Successfully loaded chunk {chunk_idx} on attempt {attempt + 1}", rotate=True)
                    return chunk_data
                else:
                    self.smart_logger.warning(f"[CHUNK_LOADER] Chunk {chunk_idx} loaded but contains no valid data (attempt {attempt + 1})")

            except Exception as e:
                self.smart_logger.error(f"[CHUNK_LOADER] Failed to load chunk {chunk_idx} on attempt {attempt + 1}: {str(e)}")

                # Wait before retry (except on last attempt)
                if attempt < MAX_RELOAD_ATTEMPTS - 1:
                    time.sleep(RELOAD_RETRY_DELAY)

        # All attempts failed, try fallback if enabled
        if fallback_enabled and chunk_idx != RELOAD_FALLBACK_CHUNK:
            self.smart_logger.warning(f"[CHUNK_LOADER] All attempts failed for chunk {chunk_idx}, falling back to chunk {RELOAD_FALLBACK_CHUNK}")
            try:
                fallback_data = self.data_loader.load_chunk(RELOAD_FALLBACK_CHUNK)
                if fallback_data and any(fallback_data.get(asset) for asset in fallback_data):
                    self.smart_logger.info(f"[CHUNK_LOADER] Successfully loaded fallback chunk {RELOAD_FALLBACK_CHUNK}", rotate=True)
                    return fallback_data
                else:
                    self.smart_logger.error(f"[CHUNK_LOADER] Fallback chunk {RELOAD_FALLBACK_CHUNK} is also empty or invalid")
            except Exception as e:
                self.smart_logger.error(f"[CHUNK_LOADER] Fallback chunk {RELOAD_FALLBACK_CHUNK} also failed to load: {str(e)}")

        # If we get here, everything failed
        raise RuntimeError(f"Failed to load chunk {chunk_idx} after {MAX_RELOAD_ATTEMPTS} attempts and fallback failed")
        logger.info(
            f"Initialized data loader with {len(mapped_assets)} assets: {', '.join(mapped_assets)}"
        )
        logger.debug(f"Available timeframes: {', '.join(self.timeframes)}")

        # Update assets list with mapped assets
        self.assets = mapped_assets

        return self.data_loader

    def _setup_spaces(self) -> None:
        """Set up action and observation spaces.

        Raises:
            ValueError: If the observation space cannot be properly configured
        """
        # Action space: Continuous actions in [-1, 1] for each asset
        # -1 = max sell, 0 = hold, 1 = max buy
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.assets),),  # One action per asset
            dtype=np.float32,
        )

        # Configure observation space with fixed shape (3 timeframes, 20 window_size, 15 features)
        try:
            # Define fixed observation shape
            self.observation_shape = (3, 20, 15)  # (timeframes, window_size, features)
            self.portfolio_state_dim = 17  # Fixed portfolio state dimension

            # Log the dimensions for debugging
            logger.info(f"Using fixed observation shape: {self.observation_shape}")
            logger.info(f"Portfolio state dimension: {self.portfolio_state_dim}")

            # Create observation space dictionary
            self.observation_space = gym.spaces.Dict(
                {
                    "observation": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=self.observation_shape,
                        dtype=np.float32,
                    ),
                    "portfolio_state": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.portfolio_state_dim,),
                        dtype=np.float32,
                    ),
                }
            )

            logger.info(f"Observation space configured: {self.observation_space}")

        except Exception as e:
            logger.error(f"Error setting up observation space: {str(e)}")
            raise

    def _get_initial_observation(self) -> Dict[str, np.ndarray]:
        """Get the initial observation after environment reset.

        This method ensures we have a valid observation before starting the episode,
        with proper error handling and logging. The observation shape is fixed to (3, 20, 15)
        where:
        - 3 timeframes (5m, 1h, 4h)
        - 20 window size
        - 15 features per timeframe

        Returns:
            Dict[str, np.ndarray]: Initial observation dictionary with 'observation' and 'portfolio_state' keys
        """
        # Define the expected observation shape (timeframes, window_size, features)
        expected_shape = (3, 20, 15)

        # Initialize default observation with correct shape
        default_observation = {
            "observation": np.zeros(expected_shape, dtype=np.float32),
            "portfolio_state": np.zeros(
                17, dtype=np.float32
            ),  # Fixed portfolio state size
        }

        try:
            # Get market data for the current chunk using safe loader
            market_data = self._safe_load_chunk(0)

            # Check for empty or invalid market data
            if not market_data or not any(market_data[asset] for asset in market_data):
                logger.error("No valid market data available for initial observation")
                return default_observation

            # Build observation using state_builder
            observation_dict = self.state_builder.build_observation(0, market_data)

            # Validate and extract the observation array
            if (
                not isinstance(observation_dict, dict)
                or "observation" not in observation_dict
            ):
                logger.error(
                    "Invalid observation format from state_builder.build_observation()"
                )
                return default_observation

            observation = observation_dict["observation"]

            # Ensure observation is a numpy array
            if not isinstance(observation, np.ndarray):
                logger.error(
                    f"Observation is not a numpy array, got {type(observation)}"
                )
                return default_observation

            # Log observation statistics
            logger.info(f"Raw observation shape: {observation.shape}")
            logger.info(
                f"Observation min/max/mean: {np.min(observation):.4f}/{np.max(observation):.4f}/{np.mean(observation):.4f}"
            )

            # Check for all zeros after transformation
            if np.all(observation == 0):
                logger.warning(
                    "Initial observation is entirely zero after transformation"
                )

            # Ensure observation has the correct shape (3, 20, 15)
            if observation.shape != expected_shape:
                # Log observation shape adjustments with rotation
                self.smart_logger.info(
                    f"Observation shape auto-adjustment: {observation.shape} -> {expected_shape}. "
                    f"Features available: {observation.shape[2] if len(observation.shape) > 2 else 0}/15. "
                    f"System will pad with zeros or truncate as needed.", rotate=True
                )

                # Create output array with correct shape
                output = np.zeros(expected_shape, dtype=np.float32)

                # Calculate the slices to copy data safely
                slices = [
                    slice(0, min(observation.shape[i], expected_shape[i]))
                    for i in range(len(expected_shape))
                ]

                # Copy data with broadcasting
                output[tuple(slices)] = observation[tuple(slices)]
                observation = output

            # Ensure data type is float32
            observation = observation.astype(np.float32)

            # Validate observation values
            if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
                logger.warning(
                    "Observation contains NaN or Inf values, replacing with zeros"
                )
                observation = np.nan_to_num(
                    observation, nan=0.0, posinf=0.0, neginf=0.0
                )

            return {
                "observation": observation,
                "portfolio_state": np.zeros(
                    17, dtype=np.float32
                ),  # Fixed portfolio state size
            }

        except Exception as e:
            logger.error(f"Error in _get_initial_observation: {str(e)}", exc_info=True)
            return default_observation

    def _set_start_step_for_chunk(self):
        """Calculates and sets the starting step within a new chunk to account for indicator warmup."""
        try:
            # Use a more conservative warmup period to avoid index out of bounds
            warmup = 200
            min_len = None

            if isinstance(self.current_data, dict) and self.current_data:
                lengths = []
                for asset_dict in self.current_data.values():
                    if not isinstance(asset_dict, dict):
                        continue
                    for tf in getattr(self, "timeframes", []):
                        df = asset_dict.get(tf)
                        if isinstance(df, pd.DataFrame):
                            lengths.append(len(df))
                if lengths:
                    min_len = min(lengths)

            if min_len is not None and min_len > 50:
                # Be more conservative: ensure we leave at least 50 steps of buffer
                # and never go beyond 80% of the chunk size
                max_safe_step = min(min_len - 50, int(min_len * 0.8))
                self.step_in_chunk = max(1, min(warmup, max_safe_step))

                logger.info(
                    f"Repositioning to step {self.step_in_chunk} in new chunk "
                    f"(warmup={warmup}, min_len={min_len}, max_safe={max_safe_step})"
                )
            else:
                # For very small chunks, start near the beginning
                self.step_in_chunk = min(10, min_len - 5) if min_len and min_len > 10 else 1
                logger.warning(
                    f"Small chunk detected (min_len={min_len}), starting at step {self.step_in_chunk}"
                )

        except Exception as e:
            logger.warning(f"Failed to set warmup step_in_chunk: {e}")
            self.step_in_chunk = 1  # Safe fallback
    def _calculate_excellence_bonus(self, base_reward: float, worker_id: int = 0) -> float:
        """Calcule les bonus d'excellence Gugu & March"""
        if not self.excellence_rewards or not EXCELLENCE_SYSTEM_AVAILABLE:
            return 0.0

        try:
            # Récupérer les métriques actuelles du worker
            metrics = self._build_excellence_metrics(worker_id)

            # Calculer les bonus d'excellence
            total_bonus, bonus_breakdown = self.excellence_rewards.calculate_excellence_bonus(
                base_reward, metrics, trade_won=(base_reward > 0)
            )

            # Logger si bonus significatif
            if total_bonus > 0.01:
                logger.debug(f"[GUGU-MARCH] Worker {worker_id} excellence bonus: {total_bonus:.4f}")

            return total_bonus

        except Exception as e:
            logger.warning(f"[GUGU-MARCH] Error calculating excellence bonus: {e}")
            return 0.0

    def _build_excellence_metrics(self, worker_id: int = 0) -> 'ExcellenceMetrics':
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
            portfolio = self.portfolio_managers[worker_id] if hasattr(self, 'portfolio_managers') else self.portfolio_manager
            perf_metrics = self.performance_metrics[worker_id] if hasattr(self, 'performance_metrics') else self.performance_metrics

            metrics_summary = perf_metrics.get_metrics_summary() if perf_metrics else {}

            # Analyser confluence des timeframes (exemple simplifié)
            timeframe_signals = self._analyze_current_confluence(worker_id)

            return ExcellenceMetrics(
                sharpe_ratio=metrics_summary.get('sharpe_ratio', 0.0),
                profit_factor=metrics_summary.get('profit_factor', 1.0),
                win_rate=metrics_summary.get('win_rate', 0.5),
                winning_streak=getattr(self.excellence_rewards, 'last_winning_streak', 0),
                total_trades=metrics_summary.get('total_trades', 0),
                current_drawdown=portfolio.current_drawdown_pct if portfolio else 0.0,
                timeframe_signals=timeframe_signals
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
            if hasattr(self, 'current_observations') and self.current_observations:
                obs = self.current_observations[worker_id] if isinstance(self.current_observations, list) else self.current_observations

                # Logique simplifiée de confluence (à personnaliser)
                # Par exemple, vérifier RSI et MACD sur différents TF
                for tf in ['5m', '1h', '4h']:
                    if f'rsi_{tf}' in obs:
                        rsi = obs[f'rsi_{tf}']
                        macd = obs.get(f'macd_{tf}', 0)
                        # Signal haussier si RSI < 70 et MACD > 0
                        confluence[tf] = (20 < rsi < 70) and (macd > -0.1)

            return confluence

        except Exception as e:
            logger.debug(f"[GUGU-MARCH] Error in confluence analysis: {e}")
            return {"5m": False, "1h": False, "4h": False}


    def reset(self, *, seed=None, options=None):
        """Reset the environment to start a new episode.

        Args:
            seed: Optional seed for the random number generator
            options: Additional options for reset

        Returns:
            tuple: (observation, info) containing the initial observation and info
        """
        # DIAGNOSTIC: Tracer les appels à reset
        self.logger.critical(f"🔄 RESET appelé pour ENV_ID={getattr(self, 'env_instance_id', 'UNKNOWN')}, Worker={getattr(self, 'worker_id', 'UNKNOWN')}, DBE_ID={id(self.dbe) if hasattr(self, 'dbe') else 'NONE'}")
        super().reset(seed=seed)

        # Reset episode-specific variables
        self.current_step = 0
        self.done = False
        self.episode_reward = 0.0
        self.step_in_chunk = 0

        # Reset frequency tracking counters
        self.positions_count = {
            '5m': 0,
            '1h': 0,
            '4h': 0,
            'daily_total': 0
        }
        self.daily_reset_step = 0
        self.current_day = 0
        self._last_trade_count = 0  # Initialize trade count tracking

        # Reset portfolio and load initial data chunk
        if hasattr(self, "last_trade_step"):
            self.last_trade_step = -1

        # Determine if this is a true new episode or just a reset
        is_new_episode = not hasattr(self, '_episode_initialized') or getattr(self, '_needs_full_reset', False)

        # Reset the environment with appropriate parameters
        self._epoch_reset(force=False, new_epoch=is_new_episode)

        # Mark that we've initialized at least one episode
        self._episode_initialized = True
        self._needs_full_reset = False  # Reset the flag

        # CORRECTION CRITIQUE : Réinitialiser les index de chunks
        self.current_chunk_idx = 0
        if hasattr(self, 'current_chunk'):
            self.current_chunk = 0

        logger.info(f"[RESET Worker {getattr(self, 'worker_id', 0)}] Starting new episode - Loading chunk 1/{getattr(self, 'total_chunks', 'unknown')}")
        self.current_data = self._safe_load_chunk(self.current_chunk_idx)

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
        logger.info(f"[CONFIG] Window size: {self.window_size}")
        logger.info(f"[CONFIG] Warmup steps: {self.warmup_steps}")

        # Get max steps configuration
        self.max_steps = env_config.get("max_steps", 1000)
        logger.info(f"[CONFIG] Max steps per episode: {self.max_steps}")

        # Initialize last trade step
        self.last_trade_step = 0
        logger.info("[INIT] Last trade step initialized to 0")

        # Patch Gugu & March - Initialisation du système d'excellence
        if EXCELLENCE_SYSTEM_AVAILABLE:
            try:
                self.excellence_rewards = create_excellence_rewards_system(self.config)
                logger.info("[GUGU-MARCH] Excellence rewards system initialized")
            except Exception as e:
                logger.warning(f"[GUGU-MARCH] Failed to initialize excellence system: {e}")
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

    def step(self, action: np.ndarray) -> tuple:
        """Execute one time step within the environment.

        This method handles the main environment loop, including:
        - Action validation and processing
        - Portfolio updates and trading
        - Reward calculation
        - Episode termination conditions
        - Chunk transitions and surveillance mode management
        - Risk management and position sizing

        Args:
            action: Array of actions for each asset in the portfolio

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Generate correlation_id for this step
        correlation_id = str(uuid.uuid4())
        # Log uniquement depuis le worker principal pour éviter les duplications
        if getattr(self, 'worker_id', 0) == 0:
            logger.info(f"[STEP] Starting step {self.current_step}", extra={'correlation_id': correlation_id})

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
            try:
                from rich.console import Console
                from rich.table import Table
                from rich.text import Text

                self._rich_console = Console(
                    force_terminal=True, force_interactive=True
                )
                self._rich_table = Table
                self._rich_text = Text
                self._rich_initialized = True
                self._rich_last_print = 0
                self._rich_print_interval = max(
                    1, int(os.getenv("ADAN_RICH_STEP_EVERY", "10"))
                )
            except Exception as e:
                self._rich_console = None
                self._rich_initialized = True
                self.logger.debug(f"Rich console disabled: {e}")

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

        # Update current day for frequency tracking
        current_day = getattr(self, 'data', {}).get('TIMESTAMP', pd.Series()).iloc[min(self.current_step, len(getattr(self, 'data', {}).get('TIMESTAMP', pd.Series())) - 1)] // (24 * 60 * 60 * 1000) if hasattr(self, 'data') and 'TIMESTAMP' in self.data else self.current_step // 288

        if not hasattr(self, 'current_day') or self.current_day != current_day:
            self.current_day = current_day
            # Reset daily counters
            self.positions_count = {'daily_total': 0, '5m': 0, '1h': 0, '4h': 0}
            if not hasattr(self, 'last_trade_steps_by_tf'):
                self.last_trade_steps_by_tf = {}

        # Determine current timeframe
        timeframe = self.get_current_timeframe() if hasattr(self, 'get_current_timeframe') else '5m'

        # Get frequency configuration
        frequency_config = self.config.get('trading_rules', {}).get('frequency', {})
        action_threshold = frequency_config.get('action_threshold', 0.3)
        force_trade_steps = frequency_config.get('force_trade_steps', 50)
        min_positions = frequency_config.get('min_positions', {})
        min_pos_tf = min_positions.get(timeframe, 1)

        # Check if trade should be forced
        steps_since_last_trade = self.current_step - self.last_trade_steps_by_tf.get(timeframe, 0)
        should_force_trade = (self.positions_count.get(timeframe, 0) < min_pos_tf and
                             steps_since_last_trade >= force_trade_steps)

        # Note: Trade execution is now handled in _execute_trades method
        # This section previously contained duplicate trade counting logic
        # which has been removed to prevent incorrect frequency counting

        # Calculate reward with frequency penalties
        reward = self._calculate_reward(action)

        # Track rewards for metrics
        self._last_reward = reward
        self._cumulative_reward = getattr(self, '_cumulative_reward', 0.0) + reward

        # Validate frequency requirements
        self._validate_frequency()

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

            if action.shape != (len(self.assets),):
                raise ValueError(
                    f"Action shape {action.shape} does not match "
                    f"expected shape (n_assets={len(self.assets)},)"
                )

            # Early risk check before executing any trades
            # Ensure local info dict exists to attach spot protection context
            info = {}
            current_prices = self._get_current_prices()
            try:
                # Update portfolio with current prices and enforce protection limits
                if hasattr(self, "portfolio_manager"):
                    positions_before = {k: v.is_open for k, v in self.portfolio_manager.positions.items()}
                    self.portfolio_manager.update_market_price(current_prices)
                    positions_after = {k: v.is_open for k, v in self.portfolio_manager.positions.items()}

                    # Check for closed positions and log them
                    for asset, was_open in positions_before.items():
                        if was_open and not positions_after.get(asset, True):
                            self.smart_logger.info(f"Position for {asset} was closed during market price update (SL/TP or stale).", rotate=True)

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

            # Mise à jour de l'état DBE et calcul de la modulation
            # DIAGNOSTIC: Tracer l'utilisation du DBE dans step()
            self.logger.debug(f"🎯 STEP utilise DBE - ENV_ID={getattr(self, 'env_instance_id', 'UNKNOWN')}, Step={self.current_step}, DBE_ID={id(self.dbe) if hasattr(self, 'dbe') else 'NONE'}")
            self._update_dbe_state()
            dbe_modulation = self.dbe.compute_dynamic_modulation()

            # Exécution des trades avec modulation DBE et récupération du PnL réalisé
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
            realized_pnl = self._execute_trades(action, dbe_modulation)
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
            if getattr(self, 'worker_id', 0) == 0:
                logger.info(f"[REWARD] Realized PnL for step: ${realized_pnl:.2f}")

            # step_in_chunk is already incremented earlier in step() method - removing duplicate

            first_asset = next(iter(self.current_data))
            first_timeframe = next(iter(self.current_data[first_asset]))
            data_length = len(self.current_data[first_asset][first_timeframe])

            # DIAGNOSTIC LOGS - Comprendre pourquoi les chunks ne transitionnent jamais
            if self.current_step % 50 == 0:  # Log tous les 50 steps pour éviter le spam
                logger.info(f"[CHUNK DIAGNOSTIC Worker {self.worker_id}] step_in_chunk: {self.step_in_chunk}, data_length: {data_length}, current_chunk: {self.current_chunk_idx + 1}/{getattr(self, 'total_chunks', 'unknown')}")

            MIN_EPISODE_STEPS = 288  # Minimum pour une journée complète (288 steps * 5m = 1 jour)
            done = False
            termination_reason = ""

            # Check if we should terminate based on frequency check interval or other conditions
            frequency_check_interval = self.config.get('trading', {}).get('frequency', {}).get('frequency_check_interval', 288)

            # Determiner steps_since_trade pour la logique de terminaison

            # Log current state before checking termination conditions
            steps_since_trade = (
                "-"
                if (self.last_trade_step is None or self.last_trade_step < 0)
                else str(self.current_step - self.last_trade_step)
            )
            # Log de vérification de terminaison (uniquement depuis worker principal)
            if getattr(self, 'worker_id', 0) == 0:
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
                logger.info(f"[TERMINATION Worker {self.worker_id}] {termination_reason}")
            elif (
                self.portfolio_manager.get_portfolio_value()
                <= self.portfolio_manager.initial_equity * 0.70
            ):
                done = True
                termination_reason = (
                    f"Portfolio value too low ({self.portfolio_manager.get_portfolio_value():.2f} "
                    f"<= {self.portfolio_manager.initial_equity * 0.50:.2f})"
                )
                logger.info(f"[TERMINATION Worker {self.worker_id}] {termination_reason}")
            # DÉSACTIVÉ : Cette condition terminait l'épisode trop agressivement, empêchant l'apprentissage
            # La condition originale était : 144 * 5 = 720 steps sans trade = terminaison
            # elif self.current_step - self.last_trade_step > self.config.get('trading', {}).get('frequency', {}).get('force_trade_steps', 144) * 5:
            #     done = True
            #     termination_reason = f"No trades for too long ({self.current_step - self.last_trade_step} steps)"
            #     logger.info(f"[TERMINATION Worker {self.worker_id}] {termination_reason}")

            # NOUVELLE LOGIQUE : Terminaison seulement après un chunk complet sans trades (plus permissive)
            force_trade_limit = self.config.get('trading', {}).get('frequency', {}).get('force_trade_steps', 144) * 10  # 1440 steps = ~5 jours
            if self.current_step - self.last_trade_step > force_trade_limit:
                logger.warning(f"[TERMINATION WARNING Worker {self.worker_id}] Long period without trades: {self.current_step - self.last_trade_step} steps > {force_trade_limit}")
                # NE PAS TERMINER - laisser l'agent apprendre même s'il ne trade pas immédiatement
                # done = True
                # termination_reason = f"No trades for very long ({self.current_step - self.last_trade_step} steps)"
                # logger.info(f"[TERMINATION Worker {self.worker_id}] {termination_reason}")

            # Ensure environment done flag is set when a termination condition is met
            if done:
                self.done = True

            # DIAGNOSTIC CRITIQUE : Vérifier si nous avons atteint la fin du chunk actuel
            transition_threshold = data_length - 1
            logger.debug(f"[CHUNK TRANSITION CHECK Worker {self.worker_id}] step_in_chunk: {self.step_in_chunk}, threshold: {transition_threshold}, will_transition: {self.step_in_chunk >= transition_threshold}")

            if self.step_in_chunk >= data_length - 1:
                logger.info(f"[CHUNK TRANSITION Worker {self.worker_id}] End of chunk {self.current_chunk_idx + 1} reached (step_in_chunk: {self.step_in_chunk} >= {data_length - 1})")
                self.current_chunk_idx += 1
                self.current_chunk += 1

                # POINT CRITIQUE : Réinitialiser le compteur step_in_chunk pour le nouveau chunk
                self.step_in_chunk = 0

                if hasattr(self, 'portfolio') and hasattr(self.portfolio, 'check_reset'):
                    if self.portfolio.check_reset(chunk_completed=True):
                        logger.info("[HARD RESET] Portfolio reset performed due to capital below threshold")
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
                        self.current_data = self._safe_load_chunk(self.current_chunk_idx)
                        self._set_start_step_for_chunk()  # Reposition step to skip warmup period
                        logger.info(f"[CHUNK] Successfully loaded chunk {self.current_chunk_idx + 1}/{chunks_limit}")
                    except Exception as e:
                        logger.error(f"[CHUNK] Failed to load chunk {self.current_chunk_idx + 1}: {e}")
                        done = True
                        self.done = True
                        termination_reason = f"Failed to load chunk {self.current_chunk_idx + 1}: {e}"

                    # Réinitialiser les composants pour le nouveau chunk avec continuité
                    if hasattr(self, 'dbe') and hasattr(self.dbe, "reset_for_new_chunk"):
                        try:
                            with self.logger_lock:
                                logger.debug(f"[DBE {self.worker_id}] Resetting DBE for new chunk with continuity")
                        except AttributeError:
                            logger.debug(f"[DBE {self.worker_id}] Resetting DBE for new chunk with continuity")
                        self.dbe.reset_for_new_chunk(continuity=True)
                    elif hasattr(self, 'dbe') and hasattr(self.dbe, "_reset_for_new_chunk"):
                        try:
                            with self.logger_lock:
                                logger.debug(f"[DBE {self.worker_id}] Fallback to legacy reset")
                        except AttributeError:
                            logger.debug(f"[DBE {self.worker_id}] Fallback to legacy reset")
                        self.dbe._reset_for_new_chunk()

                    # Réinitialiser les composants pour le nouveau chunk avec continuité
                    if hasattr(self, 'dbe') and hasattr(self.dbe, "reset_for_new_chunk"):
                        try:
                            with self.logger_lock:
                                logger.debug(f"[DBE {self.worker_id}] Resetting DBE for new chunk with continuity")
                        except AttributeError:
                            logger.debug(f"[DBE {self.worker_id}] Resetting DBE for new chunk with continuity")
                        self.dbe.reset_for_new_chunk(continuity=True)
                    elif hasattr(self, 'dbe') and hasattr(self.dbe, "_reset_for_new_chunk"):
                        try:
                            with self.logger_lock:
                                logger.debug(f"[DBE {self.worker_id}] Fallback to legacy reset")
                        except AttributeError:
                            logger.debug(f"[DBE {self.worker_id}] Fallback to legacy reset")
                        self.dbe._reset_for_new_chunk()

            # Log final decision and handle episode termination
            if done:
                logger.info(
                    f"[EPISODE END] Episode ending. Reason: {termination_reason}"
                )
                logger.info(
                    f"[EPISODE STATS] Total steps: {self.current_step}, "
                    f"Final portfolio value: {self.portfolio_manager.get_portfolio_value():.2f}, "
                    f"Return: {(self.portfolio_manager.get_portfolio_value()/self.portfolio_manager.initial_equity - 1)*100:.2f}%"
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
                self.logger.warning("Invalid observation detected, attempting recovery instead of reset")

                # TENTATIVE DE RÉCUPÉRATION AVANT RESET COMPLET
                try:
                    # Essayer de reconstruire l'observation sans reset
                    current_observation = self._get_observation()

                    # Vérifier si la récupération a fonctionné
                    if not any(
                        np.isnan(v).any() or np.isinf(v).any()
                        for v in current_observation.values()
                    ):
                        self.logger.info("Observation recovery successful, continuing episode")
                        # Continuer avec l'observation récupérée
                    else:
                        raise ValueError("Recovery failed, still has NaN/inf values")

                except Exception as recovery_error:
                    self.logger.error(f"Recovery failed: {recovery_error}, performing reset as last resort")
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
                            "recovery_error": str(recovery_error)
                        },
                    )

            # Calculate reward using internal shaper (includes risk penalties/tier adjustments)
            reward = self._calculate_reward(action)

            # Track rewards for metrics
            self._last_reward = reward
            self._cumulative_reward = getattr(self, '_cumulative_reward', 0.0) + reward

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
                    size = pos.get("size") or pos.get(
                        "quantity"
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
                                    style="red" if delta > abs(prev) * 2 else "orange3",
                                )
                        return Text(f"{cur:.3g}")

                    # Gather row fields
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
                    ppo = info.get("ppo_metrics", {}) if isinstance(info, dict) else {}
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
                        _reason_cell(_fmt(sizer_r)) if sizer_r else Text(_fmt(sizer_f)),
                        Text(f"{_fmt(trades_step)}|{_fmt(trades_cum)}"),
                        _winrate_cell(winrate),
                        Text(_fmt(sharpe)),
                        _loss_cell(pol_loss, prev_pol_loss),
                        _loss_cell(val_loss, prev_val_loss),
                        Text("".join(event_tags)),
                        style=row_style,
                    )
                    self._rich_console.print(table)

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
            self._log_summary(self.current_step, self.current_chunk_idx + 1, self.total_chunks)

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
        self.logger.debug(f"🔄 UPDATE_DBE_STATE appelé pour ENV_ID={getattr(self, 'env_instance_id', 'UNKNOWN')}, Step={getattr(self, 'current_step', 'UNKNOWN')}")
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
                self.logger.debug(f"🧠 DBE.update_state appelé - ENV_ID={getattr(self, 'env_instance_id', 'UNKNOWN')}, DBE_ID={id(self.dbe)}")
                self.dbe.update_state(live_metrics)
            else:
                self.logger.warning(f"❌ DBE non disponible pour ENV_ID={getattr(self, 'env_instance_id', 'UNKNOWN')}")

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
        """Get the current prices for all assets using correct step_in_chunk indexing."""
        prices: Dict[str, float] = {}

        # Debug logging pour diagnostiquer les problèmes
        if self.current_step % 50 == 0:
            self.smart_logger.debug(f"PRICE_DEBUG | step_in_chunk={self.step_in_chunk} | current_step={self.current_step}", sample_rate=0.1)

        # *** CORRECTION CLÉ : Utiliser step_in_chunk de manière stricte ***
        current_idx_in_chunk = self.step_in_chunk

        for _asset, timeframe_data in self.current_data.items():
            try:
                # Get 5m data for this asset
                asset_data = timeframe_data.get("5m")
                if asset_data is None or asset_data.empty:
                    self.smart_logger.warning(f"MISSING_DATA | asset={_asset}")
                    continue

                # Vérification stricte des limites d'index
                if current_idx_in_chunk >= len(asset_data):
                    # Index hors limites - utiliser la dernière valeur disponible
                    self.smart_logger.error(f"INDEX_ERROR_PRICE | step_in_chunk={current_idx_in_chunk} >= len(df)={len(asset_data)}. Using last price.",
                                          asset=_asset, dedupe=True)

                    last_row = asset_data.iloc[-1]
                    prices[_asset] = float(last_row['close'])
                    if isinstance(last_row.name, pd.Timestamp):
                        self._last_asset_timestamp[_asset] = last_row.name

                    # Compteur de forward-fill pour surveillance
                    if not hasattr(self, '_price_forward_fill_count'):
                        self._price_forward_fill_count = 0
                    self._price_forward_fill_count += 1

                elif current_idx_in_chunk < 0:
                    # Index négatif - utiliser la première valeur
                    self.smart_logger.error(f"NEGATIVE_INDEX_PRICE | step_in_chunk={current_idx_in_chunk}. Using first price.", dedupe=True)

                    prices[_asset] = float(asset_data.iloc[0]['close'])

                    # Compteur de forward-fill pour surveillance
                    if not hasattr(self, '_price_forward_fill_count'):
                        self._price_forward_fill_count = 0
                    self._price_forward_fill_count += 1

                else:
                    # Index valide - lecture normale
                    prices[_asset] = float(asset_data.iloc[current_idx_in_chunk]['close'])

                    if self.current_step % 100 == 0:
                        self.smart_logger.debug(f"PRICE_SUCCESS | asset={_asset} | step_in_chunk={current_idx_in_chunk} | price={prices[_asset]:.4f}", sample_rate=0.05)

            except Exception as e:
                self.smart_logger.error(f"PRICE_READ_ERROR | asset={_asset} | step_in_chunk={current_idx_in_chunk} | error={str(e)}", dedupe=True)

                # Compteur de forward-fill pour les erreurs
                if not hasattr(self, '_price_forward_fill_count'):
                    self._price_forward_fill_count = 0
                self._price_forward_fill_count += 1

                # Utiliser la dernière valeur connue comme fallback
                if hasattr(self, '_last_known_prices') and _asset in self._last_known_prices:
                    prices[_asset] = self._last_known_prices[_asset]
                    self.smart_logger.warning(f"FALLBACK_PRICE | asset={_asset} | using_last_known={prices[_asset]:.4f}", dedupe=True)

        # Mettre à jour les derniers prix connus
        if not hasattr(self, '_last_known_prices'):
            self._last_known_prices = {}
        self._last_known_prices.update(prices)

        # Vérifier le taux de forward-fill excessif
        self._check_excessive_forward_fill()

        return prices

    def _check_excessive_forward_fill(self):
        """Vérifier si le taux de forward-fill dépasse le seuil acceptable."""
        # Initialiser les compteurs s'ils n'existent pas
        if not hasattr(self, '_price_read_success_count'):
            self._price_read_success_count = 0
        if not hasattr(self, '_price_forward_fill_count'):
            self._price_forward_fill_count = 0
        if not hasattr(self, '_forward_fill_threshold'):
            self._forward_fill_threshold = 0.5  # 50% maximum de forward-fill acceptable
        if not hasattr(self, '_last_ff_check_step'):
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

        forward_fill_rate = self._price_forward_fill_count / total_reads if total_reads > 0 else 0

        if forward_fill_rate > self._forward_fill_threshold:
            self.smart_logger.error(
                f"EXCESSIVE_FORWARD_FILL | rate={forward_fill_rate*100:.1f}% | count={self._price_forward_fill_count}/{total_reads}",
                dedupe=True
            )

        # Réinitialiser les compteurs périodiquement pour avoir des mesures actuelles
        if total_reads > 1000:
            self._price_read_success_count = max(100, int(self._price_read_success_count * 0.1))
            self._price_forward_fill_count = max(10, int(self._price_forward_fill_count * 0.1))

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

        invalid_assets = [
            asset
            for asset, price in prices.items()
            if price <= 0 or not np.isfinite(price)
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

        for asset, timeframe_data in self.current_data.items():
            if not timeframe_data:
                continue

            asset_data = timeframe_data.get("5m")
            if asset_data is None or asset_data.empty:
                continue

            timestamp: Optional[pd.Timestamp] = None

            if 0 <= current_idx_in_chunk < len(asset_data):
                timestamp = asset_data.index[current_idx_in_chunk]
            elif current_idx_in_chunk >= len(asset_data):
                timestamp = asset_data.index[-1]
            else:
                timestamp = asset_data.index[0]

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

    def _is_valid_observation_structure(self, obs) -> bool:
        """
        Robustly check whether `obs` looks like a valid observation:
          - must be a dict with keys 'observation' and 'portfolio_state'
          - 'observation' must be array-like; 'portfolio_state' must be 1-D array-like
        Avoid any `if array` boolean checks here.
        """
        if obs is None:
            return False
        if not isinstance(obs, dict):
            return False
        if "observation" not in obs or "portfolio_state" not in obs:
            return False
        try:
            arr = np.asarray(obs["observation"])
            ps = np.asarray(obs["portfolio_state"])
        except Exception:
            return False
        # Loose shape checks (do not rely on exact sizes unless available)
        if arr.size == 0:
            return False
        if ps.ndim != 1:
            return False
        # If we have expected shapes cached, check they are compatible
        if hasattr(self, "observation_shape") and self.observation_shape is not None:
            try:
                # allow broadcasting-compatible but check dims
                if arr.ndim != len(self.observation_shape):
                    return False
                for a_dim, expected in zip(arr.shape, self.observation_shape):
                    # only check when expected is not None
                    if expected is not None and a_dim != expected:
                        return False
            except Exception:
                return False
        if (
            hasattr(self, "portfolio_state_size")
            and self.portfolio_state_size is not None
        ):
            if ps.size != self.portfolio_state_size:
                return False
        return True

    def _default_observation(self):
        """
        Return a standard zero-padded observation consistent with the
        environment's observation_space / shapes previously logged.
        """
        # If we have explicit shape info use it, else fall back to conservative defaults seen in logs.
        obs_shape = getattr(self, "observation_shape", (3, 20, 15))
        portfolio_size = getattr(self, "portfolio_state_size", 17)
        return {
            "observation": np.zeros(obs_shape, dtype=np.float32),
            "portfolio_state": np.zeros((portfolio_size,), dtype=np.float32),
        }

    def _summarize_raw_obs(self, raw_obs) -> str:
        """
        Generate a string summary of the raw observation structure for debugging.

        Args:
            raw_obs: The raw observation to summarize

        Returns:
            str: A string summary of the observation structure
        """
        if raw_obs is None:
            return "None"

        if isinstance(raw_obs, dict):
            summary = []
            for k, v in raw_obs.items():
                if hasattr(v, "shape"):
                    summary.append(
                        f"{k}: array{tuple(v.shape)} ({v.dtype if hasattr(v, 'dtype') else '?'})"
                    )
                elif isinstance(v, (list, tuple)):
                    summary.append(f"{k}: {type(v).__name__} of length {len(v)}")
                else:
                    summary.append(f"{k}: {type(v).__name__}")
            return "{" + ", ".join(summary) + "}"

        if hasattr(raw_obs, "shape"):
            return f"array{tuple(raw_obs.shape)} ({raw_obs.dtype if hasattr(raw_obs, 'dtype') else '?'})"

        if isinstance(raw_obs, (list, tuple)):
            return f"{type(raw_obs).__name__} of length {len(raw_obs)}"

        return str(type(raw_obs))

    def _build_observation(self) -> Dict[str, np.ndarray]:
        """
        Build the current observation using the StateBuilder and current data.

        Returns a dict with keys:
          - 'observation': np.ndarray with shape self.observation_shape (float32)
          - 'portfolio_state': np.ndarray with shape (self.portfolio_state_dim,) (float32)
        """
        try:
            # Determine current index within the chunk. Use step_in_chunk which advances with steps
            current_idx = int(getattr(self, "step_in_chunk", 0))

            # Basic guards
            if not hasattr(self, "state_builder") or self.state_builder is None:
                raise RuntimeError("state_builder not initialized")
            if (
                not hasattr(self, "current_data")
                or self.current_data is None
                or not isinstance(self.current_data, dict)
                or len(self.current_data) == 0
            ):
                logger.warning(
                    "No current_data available in _build_observation, returning default"
                )
                return self._default_observation()

            # Delegate construction to StateBuilder
            raw = self.state_builder.build_observation(current_idx, self.current_data)

            # Normalize to expected dict format
            logger.debug(f"Raw observation type: {type(raw)}")
            if isinstance(raw, dict):
                logger.debug(f"Raw observation keys: {list(raw.keys())}")
                if "observation" in raw:
                    market = np.asarray(raw["observation"], dtype=np.float32)
                    logger.debug(f"Observation shape from dict: {market.shape}")
                else:
                    market = None
                    logger.debug("No 'observation' key in raw dict")
                port = raw.get("portfolio_state", None)
            else:
                logger.debug(f"Raw observation is not a dict, converting to array")
                market = np.asarray(raw, dtype=np.float32)
                logger.debug(f"Converted observation shape: {market.shape}")
                port = None

            # If market is not directly usable, try aligning from per-timeframe dict
            if market is None or market.ndim != 3:
                logger.debug(
                    f"Market data needs alignment. Shape: {getattr(market, 'shape', 'None')}, Type: {type(market)}"
                )
                try:
                    # raw may actually be a dict of timeframe->2D arrays; try to align
                    if (
                        isinstance(raw, dict)
                        and "observation" not in raw
                        and hasattr(self.state_builder, "align_timeframe_dims")
                    ):
                        logger.debug(
                            "Attempting to align timeframes with align_timeframe_dims"
                        )
                        market = self.state_builder.align_timeframe_dims(raw)
                        logger.debug(
                            f"Aligned market data shape: {getattr(market, 'shape', 'None')}"
                        )
                except Exception as e:
                    logger.error(
                        f"Error in align_timeframe_dims: {str(e)}", exc_info=True
                    )
                    market = None

            # Ensure market has expected shape, padding/truncating as needed
            expected_shape = (3, 20, 15)  # Forcer la forme attendue
            self.smart_logger.debug(f"Adjusting observation shape to expected {expected_shape}", sample_rate=0.1)

            try:
                if market is None or market.ndim != 3:
                    self.smart_logger.info(
                        f"Market data unavailable or invalid shape: {getattr(market, 'shape', 'None')}, "
                        f"using zeros with shape {expected_shape}", rotate=True
                    )
                    market = np.zeros(expected_shape, dtype=np.float32)

                # Log la forme actuelle avant ajustement
                self.smart_logger.debug(f"Market data shape before adjustment: {market.shape}", sample_rate=0.1)

                # Créer un nouveau tableau avec la forme exacte attendue
                out = np.zeros(expected_shape, dtype=np.float32)

                # Copier les données existantes en respectant les dimensions
                min_timeframes = min(market.shape[0], expected_shape[0])
                min_steps = min(market.shape[1], expected_shape[1])
                min_features = min(market.shape[2], expected_shape[2])

                out[:min_timeframes, :min_steps, :min_features] = market[
                    :min_timeframes, :min_steps, :min_features
                ]

                market = out.astype(np.float32)
                self.smart_logger.debug(f"Market data shape after adjustment: {market.shape}", sample_rate=0.1)

            except Exception as e:
                self.smart_logger.error(f"Error adjusting market data shape: {str(e)}", dedupe=True)
                market = np.zeros(expected_shape, dtype=np.float32)

            # Normalize portfolio state
            ps_size = int(getattr(self, "portfolio_state_size", 17))
            try:
                ps = np.asarray(
                    (
                        port
                        if port is not None
                        else np.zeros((ps_size,), dtype=np.float32)
                    ),
                    dtype=np.float32,
                ).reshape(-1)
                if ps.size != ps_size:
                    fixed = np.zeros((ps_size,), dtype=np.float32)
                    fixed[: min(ps_size, ps.size)] = ps[: min(ps_size, ps.size)]
                    ps = fixed
            except Exception:
                ps = np.zeros((ps_size,), dtype=np.float32)

            obs = {"observation": market, "portfolio_state": ps}
            logger.debug(
                f"_build_observation -> obs shape={obs['observation'].shape}, ps shape={obs['portfolio_state'].shape}, idx={current_idx}"
            )
            return obs

        except Exception as e:
            logger.error(f"Error in _build_observation: {e}", exc_info=True)
            return self._default_observation()

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Construit et retourne l'observation actuelle sous forme de dictionnaire.

        Cette méthode est robuste à différents formats d'entrée et inclut un système de cache.
        Elle gère automatiquement le padding/truncature pour s'assurer que les dimensions
        correspondent à l'espace d'observation défini.

        Returns:
            Dict[str, np.ndarray]: Dictionnaire contenant :
                - 'observation': np.ndarray de forme (timeframes, window_size, features)
                - 'portfolio_state': np.ndarray de forme (17,) avec les métriques du portefeuille
        """
        try:
            # Try to get cached observation first
            if self._current_obs is not None:
                if self._is_valid_observation_structure(self._current_obs):
                    return self._current_obs
                else:
                    logger.warning(
                        "Cached observation failed validation, regenerating..."
                    )

            # Build new observation
            obs = self._build_observation()

            # Validate the structure
            if not self._is_valid_observation_structure(obs):
                logger.error("Generated observation failed validation, using default")
                obs = self._default_observation()

            self._current_obs = obs
            return obs

        except Exception as e:
            logger.error(f"Error in _get_observation: {str(e)}")
            return self._default_observation()

    def _check_and_reset_daily_counters(self) -> None:
        """
        Vérifie si un nouveau jour a commencé et reset les compteurs si nécessaire.
        Utilise le nombre de steps pour déterminer le passage d'un jour.
        """
        if not self.frequency_config:
            return

        daily_steps_5m = self.frequency_config.get('daily_steps_5m', 288)

        # Calculer le jour courant basé sur les steps totaux
        current_day = self.current_step // daily_steps_5m

        if current_day > self.current_day:
            # Nouveau jour détecté, reset des compteurs
            self.positions_count = {
                '5m': 0,
                '1h': 0,
                '4h': 0,
                'daily_total': 0
            }
            self.current_day = current_day
            self.daily_reset_step = self.current_step

            self.smart_logger.info(f"[FREQUENCY] Nouveau jour détecté (jour {current_day}), reset des compteurs de positions", rotate=True)

    def _track_position_frequency(self) -> None:
        """
        Suit les positions ouvertes/fermées par timeframe.
        Met à jour les compteurs globaux.
        """
        if not self.frequency_config:
            return

        # Déterminer le timeframe courant (simplifié - vous pouvez l'améliorer)
        # Pour cet exemple, on utilise un mapping basé sur le current_step modulo
        step_in_day = self.current_step % self.frequency_config.get('daily_steps_5m', 288)

        # Logique simplifiée pour déterminer le timeframe actuel
        if step_in_day % 48 == 0:  # Toutes les 4h (288/6 = 48)
            current_timeframe = '4h'
        elif step_in_day % 12 == 0:  # Toutes les 1h (288/24 = 12)
            current_timeframe = '1h'
        else:
            current_timeframe = '5m'

        # Vérifier si de nouveaux trades ont été ajoutés au trade_log
        if hasattr(self.portfolio, 'trade_log') and self.portfolio.trade_log:
            # Initialiser last_trade_ids si nécessaire
            if not hasattr(self, 'last_trade_ids'):
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
                trade_type = trade.get('type', '')
                if trade_type in ['open', 'close']:
                    self.positions_count[current_timeframe] += 1
                    self.positions_count['daily_total'] += 1

                    asset = trade.get('asset', 'Unknown')
                    price = trade.get('price', 0.0)
                    self.smart_logger.info(f"[FREQUENCY] Trade {trade_type} {asset} @ {price:.2f} sur {current_timeframe} "
                                       f"(count: {self.positions_count[current_timeframe]}, "
                                       f"total: {self.positions_count['daily_total']})", rotate=True)

    def _calculate_frequency_reward(self) -> float:
        """
        Calcule le bonus/pénalité de fréquence basé sur le nombre de positions
        par timeframe et le total journalier.
        """
        if not self.frequency_config:
            return 0.0

        frequency_reward = 0.0
        bonus_weight = self.frequency_config.get('frequency_bonus_weight', 0.05)
        penalty_weight = self.frequency_config.get('frequency_penalty_weight', 0.1)

        # Vérifier chaque timeframe individuellement
        for timeframe in ['5m', '1h', '4h']:
            if timeframe in self.frequency_config:
                tf_config = self.frequency_config[timeframe]
                min_pos = tf_config.get('min_positions', 0)
                max_pos = tf_config.get('max_positions', 999)
                current_count = self.positions_count[timeframe]

                if min_pos <= current_count <= max_pos:
                    # Dans les bornes : bonus proportionnel
                    frequency_reward += bonus_weight * (current_count / max(max_pos, 1))
                else:
                    # Hors bornes : pénalité progressive
                    if current_count < min_pos:
                        penalty = penalty_weight * (min_pos - current_count)
                    else:  # current_count > max_pos
                        penalty = penalty_weight * (current_count - max_pos)
                    frequency_reward -= penalty

        # Vérifier le total journalier
        total_min = self.frequency_config.get('total_daily_min', 5)
        total_max = self.frequency_config.get('total_daily_max', 15)
        daily_total = self.positions_count['daily_total']

        if total_min <= daily_total <= total_max:
            # Bonus pour être dans les bornes totales
            frequency_reward += bonus_weight * (daily_total / max(total_max, 1))
        else:
            # Pénalité pour être hors bornes totales
            if daily_total < total_min:
                penalty = penalty_weight * (total_min - daily_total)
            else:  # daily_total > total_max
                penalty = penalty_weight * (daily_total - total_max)
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
        for tf in ['5m', '1h', '4h']:
            if tf in self.frequency_config:
                tf_config = self.frequency_config[tf]
                count = self.positions_count[tf]
                min_pos = tf_config.get('min_positions', 0)
                max_pos = tf_config.get('max_positions', 999)

                if count < min_pos or count > max_pos:
                    valid = False
                    self.smart_logger.warning(f"[FREQUENCY] Count hors bornes pour {tf}: {count} (min: {min_pos}, max: {max_pos})", dedupe=True)

        # Vérifier le total journalier
        total_count = self.positions_count['daily_total']
        min_total = self.frequency_config.get('total_daily_min', 5)
        max_total = self.frequency_config.get('total_daily_max', 15)

        if total_count < min_total or total_count > max_total:
            valid = False
            self.smart_logger.warning(f"[FREQUENCY] Total journalier hors bornes: {total_count} (min: {min_total}, max: {max_total})", dedupe=True)

        # Log de confirmation si tout est dans les bornes
        if valid:
            self.smart_logger.info(f"[FREQUENCY] Tous les counts dans les bornes: {self.positions_count}", rotate=True)

    def _calculate_reward(self, action: np.ndarray) -> float:
        """
        Calcule la récompense pour l'étape actuelle.

        La récompense est calculée comme suit :
        - Récompense de base : rendement du portefeuille
        - Pénalité de risque : basée sur le drawdown maximum
        - Pénalité de transaction : basée sur le turnover
        - Pénalité de concentration : pénalise les positions trop importantes
        - Pénalité de régularité des actions : pénalise les changements brusques

        Args:
            action: Vecteur d'actions du modèle

        Returns:
            float: Valeur de la récompense
        """
        if not hasattr(self, "_is_initialized") or not self._is_initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Récupération des métriques du portefeuille
        portfolio_metrics = self.portfolio_manager.get_metrics()
        returns = portfolio_metrics.get("returns", 0.0)
        max_drawdown = portfolio_metrics.get("max_drawdown", 0.0)
        reward_config = self.config.get("reward", {})

        # Configuration des paramètres de récompense
        return_scale = reward_config.get("return_scale", 1.0)
        risk_aversion = reward_config.get("risk_aversion", 1.5)

        # Calcul de la récompense de base
        base_reward = returns * return_scale
        risk_penalty = risk_aversion * max_drawdown

        # Calcul de la pénalité de transaction
        transaction_penalty = 0.0
        if hasattr(self, "_last_portfolio_value"):
            last_value = self._last_portfolio_value
            current_value = portfolio_metrics.get("total_value", 0.0)
            turnover = abs(current_value - last_value) / max(1.0, last_value)
            transaction_penalty = (
                reward_config.get("transaction_cost_penalty", 0.1) * turnover
            )

        # Calcul de la pénalité de concentration
        position_concentration = 0.0
        if portfolio_metrics.get("total_value", 0) > 0:
            positions = portfolio_metrics.get("positions", {})
            position_values = [p.get("value", 0) for p in positions.values()]
            if position_values:
                max_position = max(position_values)
                position_concentration = (
                    max_position / portfolio_metrics["total_value"]
                ) ** 2

        concentration_penalty = (
            reward_config.get("concentration_penalty", 0.5) * position_concentration
        )

        # Calcul de la pénalité de régularité des actions
        action_smoothness_penalty = 0.0
        if hasattr(self, "_last_action") and self._last_action is not None:
            action_diff = np.mean(np.abs(action - self._last_action))
            action_smoothness_penalty = (
                reward_config.get("action_smoothness_penalty", 0.2) * action_diff
            )

        # Vérification et suivi de fréquence des positions
        self._check_and_reset_daily_counters()
        self._track_position_frequency()
        frequency_reward = self._calculate_frequency_reward()

        # Validation des métriques de fréquence
        self._validate_frequency()

        # Calcul de la récompense totale
        total_reward = (
            base_reward
            - risk_penalty
            - transaction_penalty
            - concentration_penalty
            - action_smoothness_penalty
            + frequency_reward
        )

        # Mise à jour de l'état pour la prochaine itération
        self._last_portfolio_value = portfolio_metrics.get("total_value", 0.0)
        self._last_action = action.copy()

        # Journalisation des composantes de la récompense pour le débogage
        self.logger.debug(
            f"Reward components - Base: {base_reward:.4f}, "
            f"Risk: -{risk_penalty:.4f}, "
            f"Transaction: -{transaction_penalty:.4f}, "
            f"Concentration: -{concentration_penalty:.4f}, "
            f"Action Smoothness: -{action_smoothness_penalty:.4f}, "
            f"Frequency: +{frequency_reward:.4f}, "
            f"Total: {total_reward:.4f}"
        )

        return total_reward

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
            "base_reward": float(base_reward),
            "risk_penalty": float(risk_penalty),
            "transaction_penalty": float(transaction_penalty),
            "concentration_penalty": float(concentration_penalty),
            "action_smoothness_penalty": float(action_smoothness_penalty),
            "total_reward": float(reward),
        }

        return float(reward)

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

    def _execute_trades(
        self, action: np.ndarray, dbe_modulation: Dict[str, float]
    ) -> float:
        if not hasattr(self, "portfolio_manager"):
            self.logger.error("Portfolio manager non initialisé, impossible d'exécuter le trade.")
            return 0.0

        current_timestamp = None
        try:
            current_prices = self._get_current_prices()
            try:
                current_timestamp = self._get_current_timestamp()
                self._last_market_timestamp = current_timestamp
            except Exception as timestamp_error:
                self.logger.error(
                    f"Impossible de récupérer l'horodatage marché au step {self.current_step}: {timestamp_error}"
                )
                current_timestamp = self._last_market_timestamp

            if hasattr(self.portfolio_manager, "register_market_timestamp"):
                self.portfolio_manager.register_market_timestamp(current_timestamp)

            if not current_prices or not self._validate_market_data(current_prices):
                self.logger.warning(f"[W:{self.worker_id}] Données de marché invalides pour le step {self.current_step}, aucun trade exécuté.")
                if hasattr(self, "portfolio_manager"):
                    self.portfolio_manager.update_market_price(current_prices if current_prices else {})
                return 0.0
        except Exception as e:
            self.logger.error(f"Erreur critique lors de la récupération des prix au step {self.current_step}: {e}", exc_info=True)
            return 0.0

        realized_pnl = 0.0
        trade_executed_this_step = False

        # 1. Mettre à jour la valeur des positions ouvertes et vérifier les SL/TP
        pnl_from_update = self.portfolio_manager.update_market_price(current_prices)
        if pnl_from_update > 0:
            realized_pnl += pnl_from_update
            trade_executed_this_step = True

        # 2. Itérer sur les actions de l'agent pour ouvrir ou fermer des positions
        for i, asset in enumerate(self.assets):
            if i >= len(action) or asset not in current_prices:
                continue

            action_value = np.clip(action[i], -1.0, 1.0)
            price = current_prices[asset]
            position = self.portfolio_manager.positions.get(asset)
            is_open = position and position.is_open

            # A. L'agent veut VENDRE (fermer une position)
            if action_value < -0.5 and is_open:
                self.logger.info(f"[ACTION] Agent requests CLOSE for {asset} at price {price:.2f}")
                pnl = self.portfolio_manager.close_position(
                    asset=asset.upper(),
                    price=price,
                    timestamp=current_timestamp,
                    current_prices=current_prices,
                )
                if pnl is not None:
                    realized_pnl += pnl
                    trade_executed_this_step = True
                    self.logger.info(f"Position {asset} closed. PnL: {pnl:.2f}")

            # B. L'agent veut ACHETER (ouvrir une position)
            elif action_value > 0.5 and not is_open:
                self.logger.info(f"[ACTION] Agent requests OPEN for {asset} at price {price:.2f}")

                trade_params = self.dbe.calculate_trade_parameters(
                    capital=self.portfolio_manager.get_total_value(),
                    worker_pref_pct=action_value,
                    tier_config=self.portfolio_manager.get_current_tier(),
                    current_price=price,
                    asset_volatility=self._calculate_asset_volatility(asset),
                    dbe_modulation=dbe_modulation
                )

                if not trade_params.get("feasible", False):
                    self.logger.warning(f"Trade non réalisable pour {asset}: {trade_params.get('reason', 'Raison inconnue')}")
                    continue

                position_size_usdt = trade_params.get("position_size_usdt", 0)
                if position_size_usdt < 11.0: # Seuil minimum de trade
                    self.logger.warning(f"Taille de position trop faible pour {asset}: {position_size_usdt:.2f} USDT")
                    continue

                size_in_asset_units = position_size_usdt / price if price > 0 else 0
                if size_in_asset_units > 0:
                    try:
                        was_opened = self.portfolio_manager.open_position(
                            asset=asset.upper(),
                            price=price,
                            size=size_in_asset_units,
                            stop_loss_pct=trade_params.get("sl_pct"),
                            take_profit_pct=trade_params.get("tp_pct"),
                            timestamp=current_timestamp,
                            current_prices=current_prices,
                        )
                        if was_opened:
                            trade_executed_this_step = True
                            self.logger.info(f"Position {asset} opened. Size: {size_in_asset_units:.4f}")
                        else:
                            self.logger.warning(f"Failed to open position for {asset} (insufficient cash or already open)")
                    except Exception as e:
                        self.logger.error(f"Impossible d'ouvrir une position pour {asset}: {e}", exc_info=True)

            # C. L'agent veut CONSERVER (HOLD)
            else:
                pass

        # 3. Mettre à jour l'étape du dernier trade si une action a eu lieu
        if trade_executed_this_step:
            self.last_trade_step = self.current_step

        return realized_pnl

    def _update_risk_metrics(self, portfolio_value, returns):
        """Met à jour les métriques de risque du portefeuille."""
        try:
            # Ajout du rendement à l'historique
            self.performance_history.append(returns)

            # Calcul du ratio de Sharpe (annualisé)
            if len(self.performance_history) > 1:
                returns_series = pd.Series(self.performance_history)
                excess_returns = returns_series - (
                    0.01 / 252
                )  # Taux sans risque journalier (1% annuel)

                # Ratio de Sharpe (annualisé)
                sharpe_ratio = np.sqrt(252) * (
                    excess_returns.mean() / (returns_series.std() + 1e-8)
                )

                # Ratio de Sortino (seulement la volatilité à la baisse)
                downside_returns = returns_series[returns_series < 0]
                downside_std = (
                    np.sqrt((downside_returns**2).mean())
                    if len(downside_returns) > 0
                    else 0
                )
                sortino_ratio = np.sqrt(252) * (
                    returns_series.mean() / (downside_std + 1e-8)
                )

                # Mise à jour des métriques
                self.risk_metrics.update(
                    {
                        "sharpe_ratio": sharpe_ratio,
                        "sortino_ratio": sortino_ratio,
                        "volatility": returns_series.std()
                        * np.sqrt(252),  # Volatilité annualisée
                        "max_drawdown": (
                            self.portfolio.max_drawdown
                            if hasattr(self.portfolio, "max_drawdown")
                            else 0.0
                        ),
                    }
                )

        except Exception as e:
            self.logger.error(
                f"Erreur lors du calcul des métriques de risque: {str(e)}"
            )

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

        # Calculer les valeurs des positions actuelles
        for asset, pos_info in portfolio_metrics.get("positions", {}).items():
            if asset in current_prices:
                qty = pos_info.get("size", pos_info.get("quantity", 0))
                price = current_prices[asset]
                value = qty * price
                position_values[asset] = {
                    "quantity": qty,
                    "price": price,
                    "value": value,
                    "unrealized_pnl": pos_info.get("unrealized_pnl", 0.0),
                    "entry_price": pos_info.get("entry_price", price),
                    "weight": (
                        value / portfolio_metrics.get("total_value", 1.0)
                        if portfolio_metrics.get("total_value", 0) > 0
                        else 0.0
                    ),
                }
                total_position_value += value

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
        closed_positions = portfolio_metrics.get("closed_positions", [])

        # Récompenses et pénalités
        last_reward = getattr(self, "_last_reward", 0.0)
        last_penalty = reward_components.get("frequency_penalty", 0.0)
        cumulative_reward = getattr(self, "_cumulative_reward", 0.0)

        # Information complète
        info = {
            "step": self.current_step,
            "chunk": self.current_chunk_idx,
            "done": getattr(self, 'done', False),
            "worker_id": getattr(self, 'worker_id', 0),

            # Portfolio metrics
            "portfolio_value": portfolio_metrics.get("total_value", 0.0),
            "cash": portfolio_metrics.get("cash", 0.0),
            "drawdown": portfolio_metrics.get("drawdown", 0.0),
            "max_dd": portfolio_metrics.get("max_drawdown", 0.0),
            "sharpe": portfolio_metrics.get("sharpe_ratio", 0.0),
            "sortino": portfolio_metrics.get("sortino_ratio", 0.0),
            "win_rate": portfolio_metrics.get("win_rate", 0.0),

            # Trading statistics
            "trades": total_trades,
            "valid_trades": valid_trades,
            "invalid_trades": max(0, total_trades - valid_trades),
            "closed_positions": closed_positions,

            # Positions actuelles
            "positions": position_values,
            "total_position_value": total_position_value,
            "leverage": portfolio_metrics.get("leverage", 0.0),
            "num_positions": portfolio_metrics.get("open_positions_count", len(position_values)),

            # Rewards & Penalties
            "last_reward": last_reward,
            "last_penalty": last_penalty,
            "cumulative_reward": cumulative_reward,
            "reward_components": reward_components,

            # Market data
            "current_date": self._get_safe_timestamp(),
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
        }

        return info

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
                "current_date": "N/A",
                "worker_id": getattr(self, 'worker_id', 0),
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
            if hasattr(self, 'performance_metrics'):
                metrics = self.performance_metrics.calculate_metrics(positions_count=self.positions_count)
            else:
                metrics = {}

            # Safely collect open positions
            open_positions = []
            try:
                for asset, pos in pm.positions.items():
                    if pos.is_open:
                        sl_price = pos.entry_price * (1 - pos.stop_loss_pct) if pos.stop_loss_pct > 0 else 0
                        tp_price = pos.entry_price * (1 + pos.take_profit_pct) if pos.take_profit_pct > 0 else 0
                        open_positions.append(
                            f"│   {asset}: {pos.size:.4f} @ {pos.entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f}"
                        )
            except Exception as e:
                open_positions = [f"│   Error retrieving positions: {str(e)}"]

            # Safely collect closed positions
            closed_positions = []
            try:
                if hasattr(pm, 'trade_log') and pm.trade_log:
                    closed_trades = [t for t in pm.trade_log if t.get('action') == 'close']
                    for trade in closed_trades[-3:]:  # Last 3 closed trades
                        pnl = trade.get('pnl', 0.0)
                        pnl_pct = trade.get('pnl_pct', 0.0)
                        asset = trade.get('asset', 'Unknown')
                        size = trade.get('size', 0.0)
                        entry_price = trade.get('entry_price', 0.0)
                        exit_price = trade.get('exit_price', trade.get('price', 0.0))
                        opened_at = trade.get('opened_at')
                        closed_at = trade.get('closed_at')
                        duration_seconds = trade.get('duration_seconds')

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
            sharpe = metrics.get('sharpe', 0.0)
            sortino = metrics.get('sortino', 0.0)
            profit_factor = metrics.get('profit_factor', 0.0)
            max_dd = metrics.get('max_dd', 0.0)
            cagr = metrics.get('cagr', 0.0)
            win_rate = metrics.get('winrate', 0.0)
            total_trades = metrics.get('total_trades', 0)
            winning_trades = metrics.get('wins', 0)
            losing_trades = metrics.get('losses', 0)
            neutral_trades = metrics.get('neutrals', 0)

            # Safe portfolio values
            capital = pm.get_total_value() if hasattr(pm, 'get_total_value') else 0.0
            equity = pm.get_equity() if hasattr(pm, 'get_equity') else 0.0
            balance = pm.get_balance() if hasattr(pm, 'get_balance') else 0.0

            # Utiliser calculate_drawdown() pour obtenir les valeurs correctes
            current_dd = pm.calculate_drawdown() * 100  # calculate_drawdown() retourne un ratio (0.0-1.0)
            max_dd_allowed = getattr(pm, 'max_drawdown_pct', 0.25) * 100

            summary_lines = [
                "╭──────── Étape {} / Chunk {}/{} (Worker {}) ─────────╮".format(step, chunk_id, total_chunks, self.worker_id),
                "│ 📊 PORTFOLIO                                                  │",
                "│ Capital: {:.2f} USDT | Équité: {:.2f} USDT".format(capital, equity).ljust(65) + "│",
                "│ Solde disponible: {:.2f} USDT".format(balance).ljust(65) + "│",
                "│                                                               │",
                "│ 📈 MÉTRIQUES                                                  │",
                "│ Sharpe: {:.2f} | Sortino: {:.2f} | Profit Factor: {:.2f}".format(sharpe, sortino, profit_factor).ljust(65) + "│",
                "│ Max DD: {:.2f}% | CAGR: {:.2f}% | Win Rate: {:.1f}%".format(max_dd, cagr, win_rate).ljust(65) + "│",
                "│ Trades: {} ({}W/{}L/{}N)".format(total_trades, winning_trades, losing_trades, neutral_trades).ljust(65) + "│",
                "│ Positions: 5m:{}, 1h:{}, 4h:{}, Total:{}".format(
                    self.positions_count.get('5m', 0),
                    self.positions_count.get('1h', 0),
                    self.positions_count.get('4h', 0),
                    metrics.get('open_positions_count', self.positions_count.get('daily_total', 0))
                ).ljust(65) + "│",
                "│                                                               │",
                "│ ⚠️  RISQUE                                                     │",
                "│ Drawdown actuel: {:.1f}%/{:.1f}%".format(current_dd, max_dd_allowed).ljust(65) + "│",
                "│                                                               │",
                "│ 📋 POSITIONS OUVERTES                                         │"
            ]

            if open_positions:
                # open_positions doit déjà contenir des lignes formatées. On les enrichit si possible
                summary_lines.extend(open_positions)
            else:
                summary_lines.append("│   Aucune                                                      │")

            summary_lines.extend([
                "│                                                               │",
                "│ 📕 DERNIÈRES POSITIONS FERMÉES                                │"
            ])

            if closed_positions:
                # Enrichir l’affichage si le dict contient opened_at/closed_at
                enriched = []
                for line in closed_positions:
                    enriched.append(line)
                summary_lines.extend(enriched)
            else:
                summary_lines.append("│   Aucune                                                      │")

            summary_lines.append("╰───────────────────────────────────────────────────────────────╯")

            summary = "\n".join(summary_lines)
            logger.info(summary)

        except Exception as e:
            logger.error(f"Error in _log_summary: {str(e)}")
            logger.info(f"[SUMMARY] Step {step} | Chunk {chunk_id}/{total_chunks} | Basic info only due to error")

    def _calculate_reward(self, action):
        """Calculate reward with enhanced frequency penalties."""
        # Base reward (PnL, etc.)
        base_reward = self.calculate_pnl() if hasattr(self, 'calculate_pnl') else 0.0
        base_reward += self.calculate_inaction_penalty() if hasattr(self, 'calculate_inaction_penalty') else 0.0

        # Frequency-based reward/penalty with grace period
        frequency_config = self.config.get('trading_rules', {}).get('frequency', {})
        frequency_reward = 0.0

        # Grace period: no frequency penalties for the first 100 steps to allow learning
        grace_period_steps = frequency_config.get('grace_period_steps', 100)
        current_step = getattr(self, 'step_count', 0)

        # Only apply frequency penalties after grace period
        if current_step > grace_period_steps:
            for tf in ['5m', '1h', '4h']:
                count = self.positions_count.get(tf, 0)
                min_pos = frequency_config.get('min_positions', {}).get(tf, 1)
                max_pos = frequency_config.get('max_positions', {}).get(tf, 10)

                if min_pos <= count <= max_pos:
                    # Bonus for being in range
                    frequency_reward += frequency_config.get('frequency_bonus_weight', 0.3) * (count / max_pos)
                elif count < min_pos:
                    # Reduced penalty for insufficient positions (removed 2.0 factor)
                    penalty = frequency_config.get('frequency_penalty_weight', 1.0) * (min_pos - count)
                    frequency_reward -= penalty
                elif count > max_pos:
                    # Standard penalty for excessive positions
                    frequency_reward -= frequency_config.get('frequency_penalty_weight', 1.0) * (count - max_pos)

            # Daily total frequency check
            total_count = self.positions_count.get('daily_total', 0)
            min_total = frequency_config.get('min_positions', {}).get('total_daily', 5)
            max_total = frequency_config.get('max_positions', {}).get('total_daily', 15)

            if min_total <= total_count <= max_total:
                frequency_reward += frequency_config.get('frequency_bonus_weight', 0.3) * (total_count / max_total)
            elif total_count < min_total:
                # Reduced penalty for insufficient total positions (removed 2.0 factor)
                frequency_reward -= frequency_config.get('frequency_penalty_weight', 1.0) * (min_total - total_count)
            elif total_count > max_total:
                frequency_reward -= frequency_config.get('frequency_penalty_weight', 1.0) * (total_count - max_total)
        else:
            # During grace period: only give bonuses, no penalties
            if current_step % 20 == 0:
                self.smart_logger.info(f"[GRACE PERIOD] Step {current_step}/{grace_period_steps} - No frequency penalties applied", rotate=True)

        total_reward = base_reward + frequency_reward

        logger.info(f"[REWARD Worker {self.worker_id}] Base: {base_reward:.4f}, Frequency: {frequency_reward:.4f}, "
                   f"Total: {total_reward:.4f}, Counts: {self.positions_count}")

        return total_reward

    def get_current_timeframe(self):
        """Determine current timeframe based on step or configuration."""
        # Simple logic: cycle through timeframes or use default
        if not hasattr(self, '_timeframe_cycle'):
            self._timeframe_cycle = ['5m', '1h', '4h']
            self._timeframe_index = 0

        # Return current timeframe (can be made more sophisticated)
        current_tf = self._timeframe_cycle[self._timeframe_index % len(self._timeframe_cycle)]

        # Advance to next timeframe occasionally
        if self.current_step % 10 == 0:  # Change timeframe every 10 steps
            self._timeframe_index = (self._timeframe_index + 1) % len(self._timeframe_cycle)

        return current_tf

    def _validate_frequency(self):
        """Log frequency validation information."""
        frequency_config = self.config.get('trading_rules', {}).get('frequency', {})

        validation_info = []
        for tf in ['5m', '1h', '4h']:
            count = self.positions_count.get(tf, 0)
            min_pos = frequency_config.get('min_positions', {}).get(tf, 1)
            max_pos = frequency_config.get('max_positions', {}).get(tf, 10)

            status = "✓" if min_pos <= count <= max_pos else "✗"
            validation_info.append(f"{tf}: {count}/{min_pos}-{max_pos} {status}")

        total_count = self.positions_count.get('daily_total', 0)
        min_total = frequency_config.get('min_positions', {}).get('total_daily', 5)
        max_total = frequency_config.get('max_positions', {}).get('total_daily', 15)
        total_status = "✓" if min_total <= total_count <= max_total else "✗"
        validation_info.append(f"Total: {total_count}/{min_total}-{max_total} {total_status}")

        logger.info(f"[FREQUENCY VALIDATION Worker {self.worker_id}] {' | '.join(validation_info)}")

    def calculate_pnl(self):
        """Calculate basic PnL from portfolio."""
        if hasattr(self, 'portfolio_manager'):
            return self.portfolio_manager.get_equity() - self.portfolio_manager.initial_capital
        return 0.0

    def calculate_inaction_penalty(self):
        """Calculate penalty for inaction."""
        # Simple inaction penalty based on time since last trade
        penalty = 0.0
        current_tf = self.get_current_timeframe()
        steps_since_trade = self.current_step - getattr(self, 'last_trade_steps_by_tf', {}).get(current_tf, 0)

        if steps_since_trade > 20:  # Penalty after 20 steps of inaction
            penalty = -0.01 * (steps_since_trade - 20)

        return penalty

    def close(self) -> None:
        """Nettoie les ressources de l'environnement."""
        pass

    def log_worker_comparison(self):
        """Log comparison metrics between workers (for debugging)."""
        if hasattr(self, 'portfolio_manager') and hasattr(self.portfolio_manager, 'metrics'):
            metrics = self.portfolio_manager.metrics.calculate_metrics() if hasattr(self.portfolio_manager.metrics, 'calculate_metrics') else {}
            equity = self.portfolio_manager.get_equity() if hasattr(self.portfolio_manager, 'get_equity') else 0.0

            logger.info(f"[COMPARISON Worker {self.worker_id}] Trades: {metrics.get('total_trades', 0)}, "
                       f"Winrate: {metrics.get('winrate', 0.0):.1f}%, "
                       f"Equity: {equity:.2f} USDT, "
                       f"Counts: {getattr(self, 'positions_count', {})}")
