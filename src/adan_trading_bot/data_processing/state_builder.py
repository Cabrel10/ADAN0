"""
State builder for creating multi-timeframe observations for the RL agent.

This module provides the StateBuilder class which transforms raw market data
into a structured observation space suitable for reinforcement learning.
"""

import gc
import os
import logging
import traceback
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union

import psutil
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# Import pour le calcul des indicateurs techniques
try:
    import pandas_ta as ta

    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logger.warning(
        "pandas_ta non disponible - calcul manuel des indicateurs techniques"
    )

# Imports pour les fonctionnalités avancées
try:
    from arch import arch_model

    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning("arch package non disponible - fonctionnalités GARCH désactivées")

try:
    from pykalman import KalmanFilter

    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    logger.warning(
        "pykalman package non disponible - fonctionnalités Kalman désactivées"
    )

logger = logging.getLogger(__name__)


def _calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les indicateurs techniques manquants à partir des données OHLCV.

    Args:
        df: DataFrame avec au minimum les colonnes OPEN, HIGH, LOW, CLOSE, VOLUME

    Returns:
        DataFrame avec les indicateurs techniques ajoutés
    """
    try:
        # Copier le DataFrame pour éviter de modifier l'original
        result_df = df.copy()

        # S'assurer que les colonnes de base existent (vérifier d'abord en minuscules puis majuscules)
        required_cols = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
        lowercase_cols = ["open", "high", "low", "close", "volume"]

        # Vérifier les colonnes encore manquantes
        missing_basic = [col for col in required_cols if col not in result_df.columns]

        if missing_basic:
            logger.warning(
                f"Colonnes OHLCV manquantes pour calculer les indicateurs: {missing_basic}"
            )
            # Utiliser CLOSE comme substitut si disponible
            if "CLOSE" in result_df.columns:
                for col in missing_basic:
                    if col != "CLOSE":
                        result_df[col] = result_df["CLOSE"]
            else:
                logger.error(
                    "Impossible de calculer les indicateurs - aucune donnée de prix disponible"
                )
                return result_df

        # Calculer RSI (14 périodes)
        if "RSI_14" not in result_df.columns:
            if PANDAS_TA_AVAILABLE:
                result_df["RSI_14"] = ta.rsi(result_df["CLOSE"], length=14)
            else:
                # Calcul manuel du RSI
                delta = result_df["CLOSE"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                result_df["RSI_14"] = 100 - (100 / (1 + rs))

        # Calculer MACD
        if "MACD_HIST_12_26_9" not in result_df.columns:
            if PANDAS_TA_AVAILABLE:
                macd_data = ta.macd(result_df["CLOSE"], fast=12, slow=26, signal=9)
                result_df["MACD_HIST_12_26_9"] = macd_data["MACDh_12_26_9"]
            else:
                # Calcul manuel MACD
                ema_12 = result_df["CLOSE"].ewm(span=12).mean()
                ema_26 = result_df["CLOSE"].ewm(span=26).mean()
                macd_line = ema_12 - ema_26
                signal_line = macd_line.ewm(span=9).mean()
                result_df["MACD_HIST_12_26_9"] = macd_line - signal_line

        # Calculer ATR (14 périodes)
        if "ATR_14" not in result_df.columns:
            if PANDAS_TA_AVAILABLE:
                result_df["ATR_14"] = ta.atr(
                    result_df["HIGH"], result_df["LOW"], result_df["CLOSE"], length=14
                )
            else:
                # Calcul manuel ATR
                high_low = result_df["HIGH"] - result_df["LOW"]
                high_close_prev = np.abs(
                    result_df["HIGH"] - result_df["CLOSE"].shift(1)
                )
                low_close_prev = np.abs(result_df["LOW"] - result_df["CLOSE"].shift(1))
                true_range = np.maximum(
                    high_low, np.maximum(high_close_prev, low_close_prev)
                )
                result_df["ATR_14"] = true_range.rolling(window=14).mean()

        # Calculer les Bandes de Bollinger (20 périodes, 2 std)
        if not all(
            col in result_df.columns for col in ["BB_UPPER", "BB_MIDDLE", "BB_LOWER"]
        ):
            if PANDAS_TA_AVAILABLE:
                bb_data = ta.bbands(result_df["CLOSE"], length=20, std=2)
                result_df["BB_UPPER"] = bb_data["BBU_20_2.0"]
                result_df["BB_MIDDLE"] = bb_data["BBM_20_2.0"]
                result_df["BB_LOWER"] = bb_data["BBL_20_2.0"]
            else:
                # Calcul manuel des Bandes de Bollinger
                sma_20 = result_df["CLOSE"].rolling(window=20).mean()
                std_20 = result_df["CLOSE"].rolling(window=20).std()
                result_df["BB_MIDDLE"] = sma_20
                result_df["BB_UPPER"] = sma_20 + (2 * std_20)
                result_df["BB_LOWER"] = sma_20 - (2 * std_20)

        # Calculer EMA 12 périodes
        if "EMA_12" not in result_df.columns:
            if PANDAS_TA_AVAILABLE:
                result_df["EMA_12"] = ta.ema(result_df["CLOSE"], length=12)
            else:
                result_df["EMA_12"] = result_df["CLOSE"].ewm(span=12).mean()

        # Calculer EMA 26 périodes
        if "EMA_26" not in result_df.columns:
            if PANDAS_TA_AVAILABLE:
                result_df["EMA_26"] = ta.ema(result_df["CLOSE"], length=26)
            else:
                result_df["EMA_26"] = result_df["CLOSE"].ewm(span=26).mean()

        # Calculer SMA 20 périodes
        if "SMA_20" not in result_df.columns:
            if PANDAS_TA_AVAILABLE:
                result_df["SMA_20"] = ta.sma(result_df["CLOSE"], length=20)
            else:
                result_df["SMA_20"] = result_df["CLOSE"].rolling(window=20).mean()

        # Calculer ADX 14 périodes
        if "ADX_14" not in result_df.columns:
            if PANDAS_TA_AVAILABLE:
                adx_data = ta.adx(
                    result_df["HIGH"], result_df["LOW"], result_df["CLOSE"], length=14
                )
                result_df["ADX_14"] = adx_data["ADX_14"]
            else:
                # Calcul manuel simplifié du ADX (approximation)
                high_low = result_df["HIGH"] - result_df["LOW"]
                high_close = np.abs(result_df["HIGH"] - result_df["CLOSE"].shift(1))
                low_close = np.abs(result_df["LOW"] - result_df["CLOSE"].shift(1))
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))

                plus_dm = np.where(
                    (result_df["HIGH"].diff() > result_df["LOW"].diff().abs())
                    & (result_df["HIGH"].diff() > 0),
                    result_df["HIGH"].diff(),
                    0,
                )
                minus_dm = np.where(
                    (result_df["LOW"].diff().abs() > result_df["HIGH"].diff())
                    & (result_df["LOW"].diff() < 0),
                    result_df["LOW"].diff().abs(),
                    0,
                )

                tr_smooth = pd.Series(true_range).rolling(window=14).mean()
                plus_di = 100 * (
                    pd.Series(plus_dm).rolling(window=14).mean() / tr_smooth
                )
                minus_di = 100 * (
                    pd.Series(minus_dm).rolling(window=14).mean() / tr_smooth
                )

                dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
                result_df["ADX_14"] = dx.rolling(window=14).mean()

        # Remplacer les NaN par forward fill puis backward fill
        result_df = result_df.fillna(method="ffill").fillna(method="bfill")

        # Si il reste encore des NaN, les remplacer par 0
        result_df = result_df.fillna(0)

        logger.debug(
            f"Indicateurs techniques calculés. Colonnes finales: {result_df.columns.tolist()}"
        )
        return result_df

    except Exception as e:
        logger.error(f"Erreur lors du calcul des indicateurs techniques: {str(e)}")
        return df


def _force_canonical_output(
    market_arr, portfolio_arr, expected_market_shape=None, expected_port_shape=None
):
    """Convertit les tableaux d'entrée en un format canonique.

    Args:
        market_arr: Données de marché à convertir (peut être un tableau numpy,
            un tenseur PyTorch, un DataFrame pandas, etc.)
        portfolio_arr: Données de portefeuille à convertir
        expected_market_shape: Forme attendue pour les données de marché
                             (par défaut: None)
        expected_port_shape: Forme attendue pour les données de portefeuille
                           (par défaut: None)

    Returns:
        Tuple de (market_np, portfolio_np) où les deux sont des tableaux numpy
        du bon format
    """

    def to_np(x):
        """Fonction utilitaire pour convertir en numpy."""
        try:
            import torch

            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        try:
            import pandas as _pd

            if isinstance(x, (_pd.DataFrame, _pd.Series)):
                return x.values
        except Exception:
            pass
        try:
            return np.asarray(x)
        except Exception:
            return None

    # Conversion en numpy
    m = to_np(market_arr)
    p = to_np(portfolio_arr)

    # Valeur par défaut si None
    if m is None:
        if expected_market_shape is not None:
            m = np.zeros(expected_market_shape, dtype=np.float32)
        else:
            m = np.zeros((1, 1, 1), dtype=np.float32)

    # Conversion du type et ajustement des dimensions
    m = m.astype(np.float32, copy=False)
    if expected_market_shape is not None:
        # Ajustement à la forme attendue
        target = tuple(expected_market_shape)
        out = np.zeros(target, dtype=np.float32)
        # S'assurer que m a au moins le même nombre de dimensions
        while m.ndim < len(target):
            m = np.expand_dims(m, 0)
        slices = tuple(slice(0, min(m.shape[i], target[i])) for i in range(len(target)))
        out[slices] = m[slices]
        m = out

    # Gestion du portefeuille
    if p is None:
        if expected_port_shape is not None:
            p = np.zeros(expected_port_shape, dtype=np.float32)
        else:
            p = np.zeros((1,), dtype=np.float32)

    p = p.astype(np.float32, copy=False)
    # Aplatir ou redimensionner le portefeuille à la forme attendue
    p = p.reshape(-1)
    if expected_port_shape is not None:
        desired = int(np.prod(expected_port_shape))
        outp = np.zeros(desired, dtype=np.float32)
        outp[: min(desired, p.shape[0])] = p[: min(desired, p.shape[0])]
        p = outp.reshape(expected_port_shape)

    return m, p


class TimeframeConfig:
    """Configuration class for timeframe-specific settings.

    This class encapsulates the configuration for a specific timeframe,
    including its features and any other relevant settings.
    """

    def __init__(
        self,
        timeframe: str,
        features: List[str],
        window_size: int = 100,
        normalize: bool = True,
    ):
        """Initialize timeframe configuration.

        Args:
            timeframe: The timeframe identifier (e.g., '5m', '1h')
            features: List of feature names for this timeframe
            window_size: Number of time steps to include in the window
            normalize: Whether to normalize the data
        """
        self.timeframe = timeframe
        self.features = features if (features is not None and len(features) > 0) else []
        self.window_size = window_size if window_size is not None else 100
        self.normalize = normalize

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            "timeframe": self.timeframe,
            "features": self.features,
            "window_size": self.window_size,
            "normalize": self.normalize,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TimeframeConfig":
        """Create configuration from dictionary."""
        return cls(
            timeframe=config_dict["timeframe"],
            features=config_dict["features"],
            window_size=config_dict.get("window_size", 100),
            normalize=config_dict.get("normalize", True),
        )


class StateBuilder:
    """
    Builds state representations from multi-timeframe market data.

    This class handles the transformation of raw market data into a structured
    observation space that can be used by reinforcement learning agents.
    """

    def __init__(
        self,
        features_config: Dict[str, List[str]] = None,
        timeframes: List[str] = None,  # Ajout du paramètre
        window_size: int = 100,
        include_portfolio_state: bool = True,
        normalize: bool = True,
        scaler_path: Optional[str] = None,
        adaptive_window: bool = True,
        min_window_size: int = 50,
        max_window_size: int = 150,
        memory_config: Optional[Dict[str, Any]] = None,
        target_observation_size: Optional[int] = None,
    ):
        """Initialize the StateBuilder according to design specifications."""
        # Configuration initiale
        if features_config is None:
            features_config = {}
        self.features_config = features_config

        # Utiliser la liste de timeframes fournie de manière autoritaire
        if timeframes:
            self.timeframes = timeframes
        else:
            # Fallback sur l'ancienne logique si non fourni
            self.timeframes = [
                tf for tf in ["5m", "1h", "4h"] if tf in self.features_config
            ]

        if not self.timeframes:
            raise ValueError(
                "Aucun timeframe valide n'a été fourni ou déduit de la configuration."
            )

        # Log de la configuration des caractéristiques
        logger.info("=== Configuration des caractéristiques par timeframe ===")
        for tf in self.timeframes:
            features = self.features_config.get(tf, [])
            logger.info(f"Timeframe {tf}: {len(features)} features - {features}")

        # Définition des fonctionnalités disponibles par timeframe
        # Basé sur les logs, nous avons les indicateurs suivants disponibles :
        # - Données OHLCV de base : OPEN, HIGH, LOW, CLOSE, VOLUME
        # - Indicateurs de momentum disponibles : RSI_14, STOCHk_14_3_3, STOCHd_14_3_3, MACD_HIST
        # - Moyennes mobiles disponibles : EMA_5, EMA_12, EMA_26, EMA_50, EMA_200, SMA_200
        # - Indicateurs de tendance disponibles : SUPERTREND_14_2.0, PSAR_0.02_0.2

        # Configuration des indicateurs composites qui génèrent plusieurs colonnes
        # Cette configuration est utilisée pour la transformation des données
        self.composite_indicators = {
            # STOCH génère %K et %D (les noms réels dans les données)
            "STOCH_14_3_3": ["STOCHk_14_3_3", "STOCHd_14_3_3"],
            "MACD_12_26_9": [
                "MACD_12_26_9",
                "MACD_SIGNAL_12_26_9",
                "MACD_HIST_12_26_9",
            ],
            "ICHIMOKU_9_26_52": [
                "TENKAN_9",
                "KIJUN_26",
                "SENKOU_A",
                "SENKOU_B",
                "CHIKOU_26",
            ],
        }

        # Utilisation directe de la configuration fournie
        self.expected_features = self.features_config

        # Journalisation des fonctionnalités configurées
        for tf in self.expected_features:
            logger.info(
                f"Configuration des fonctionnalités pour {tf}: {self.expected_features[tf]}"
            )

        # Vérifier la cohérence entre la configuration et les fonctionnalités attendues
        for tf in self.timeframes:
            if tf in self.expected_features:
                expected = set(self.expected_features[tf])
                actual = set(self.features_config[tf])
                if expected != actual:
                    logger.warning(
                        f"Configuration des fonctionnalités incohérente pour {tf}."
                        f" Attendu: {len(expected)} features, Reçu: {len(actual)}"
                    )
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"Détails - Manquants: {expected - actual}, "
                            f"Supplémentaires: {actual - expected}"
                        )
                    # Mettre à jour la configuration avec les fonctionnalités attendues
                    self.features_config[tf] = self.expected_features[tf]
                    logger.info(
                        f"Configuration des fonctionnalités mise à jour pour {tf} "
                        f"avec {len(self.features_config[tf])} features"
                    )

        self.nb_features_per_tf = {
            tf: len(features)
            for tf, features in self.features_config.items()
            if tf in self.timeframes
        }

        # Configuration de la gestion de la mémoire
        default_memory_config = {
            # Nettoyage agressif des données intermédiaires
            "aggressive_cleanup": True,
            # Forcer le garbage collection à intervalles réguliers
            "force_gc": True,
            # Activer la surveillance de la mémoire
            "memory_monitoring": True,
            # Seuil d'avertissement mémoire en Mo
            "memory_warning_threshold_mb": 5600,
            # Seuil critique de mémoire en Mo
            "memory_critical_threshold_mb": 6300,
            # Désactiver la mise en cache pour économiser de la mémoire
            "disable_caching": True,
        }
        # Fusionner avec la configuration personnalisée si fournie
        self.memory_config = {**default_memory_config, **(memory_config or {})}

        # Initialisation des métriques de performance
        self.performance_metrics = {
            # Nombre de collections de garbage collection effectuées
            "gc_collections": 0,
            # Pic d'utilisation mémoire en Mo
            "memory_peak_mb": 0,
            # Nombre total d'erreurs rencontrées
            "errors_count": 0,
            # Nombre total d'avertissements
            "warnings_count": 0,
        }

        # Mémoire initiale
        self.initial_memory_mb = 0
        self.memory_peak_mb = 0

        # Initialiser les métriques après la configuration
        self._initialize_memory_metrics()

        # Configuration de la taille de fenêtre
        self.base_window_size = window_size
        # Utilisation de la taille de fenêtre spécifiée (100 par défaut)
        self.window_size = window_size  # Utiliser la taille fournie en paramètre
        self.include_portfolio_state = include_portfolio_state
        self.normalize = normalize

        # Maximum de features défini dans la config
        # Déterminer le nombre maximum de features parmi tous les timeframes
        self.max_features = (
            max(len(features) for features in self.features_config.values())
            if self.features_config
            else 0
        )

        # Forme dynamique : (nombre de timeframes, fenêtre, max_features)
        self.observation_shape = (
            len(self.timeframes),
            self.window_size,
            self.max_features,
        )

        # Configuration adaptative
        self.adaptive_window = adaptive_window
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.volatility_history = []
        self.volatility_window = 20
        self.timeframe_weights = {
            tf: 1.0 for tf in self.timeframes
        }  # Initialisation des poids

        # Configuration des scalers
        self.scaler_path = scaler_path
        self.scalers = {tf: None for tf in self.timeframes}
        self.feature_indices = {}
        self._col_mappings: Dict[str, Dict[str, str]] = {}

        # Initialize scaler cache with LRU behavior
        self.scaler_cache = {}  # (timeframe, tuple(features)) -> fitted scaler
        self.scaler_feature_order = {}  # (timeframe, tuple(features)) -> list(features) (order)
        # stats pour debug / metrics
        self.scaler_cache_hits = 0
        self.scaler_cache_misses = 0
        self._max_scaler_cache_size = 100  # Maximum number of scalers to cache

        # Initialisation des scalers
        self._init_scalers()

        # Calcul de la taille totale de l'observation après flatten
        market_obs_size = len(self.timeframes) * self.window_size * self.max_features

        # Ajout de la taille de l'état du portefeuille
        dummy_portfolio = np.zeros(1)  # Taille ignorée
        portfolio_dim = (
            len(self._build_portfolio_state(dummy_portfolio))
            if hasattr(self, "_build_portfolio_state")
            else 17
        )

        self.total_flattened_observation_size = market_obs_size + (
            portfolio_dim if self.include_portfolio_state else 0
        )

        logger.info(
            f"Observation dimensions - Market: {market_obs_size} "
            f"+ Portfolio: {portfolio_dim if self.include_portfolio_state else 0} = "
            f"Total: {self.total_flattened_observation_size}"
        )

        logger.info(
            f"StateBuilder initialized. Target flattened observation size: {self.total_flattened_observation_size}"
        )
        logger.info(f"Features per timeframe: {self.nb_features_per_tf}")
        logger.info(
            f"StateBuilder initialized with base_window_size={window_size}, "
            f"adaptive_window={adaptive_window}, "
            f"timeframes={self.timeframes}, "
            f"features_per_timeframe={{self.nb_features_per_tf}}"
        )
        self._verbose_logging_done = False

    def set_timeframe_config(
        self, timeframe: str, window_size: int, features: List[str]
    ) -> None:
        """
        Configure les paramètres d'un timeframe spécifique.

        Args:
            timeframe: Identifiant du timeframe (ex: '5m', '1h', '4h')
            window_size: Taille de la fenêtre pour ce timeframe
            features: Liste des noms des features pour ce timeframe
        """
        if timeframe not in self.timeframes:
            logger.warning(
                f"Tentative de configuration d'un timeframe non pris en charge: {timeframe}"
            )
            return

        # Mettre à jour la configuration des features si fournie
        if features:
            self.features_config[timeframe] = features
            self.nb_features_per_tf[timeframe] = len(features)

        # Mettre à jour la taille de la fenêtre pour ce timeframe
        if hasattr(self, "window_sizes"):
            self.window_sizes[timeframe] = window_size
        else:
            self.window_sizes = {tf: self.window_size for tf in self.timeframes}
            self.window_sizes[timeframe] = window_size

        # Mettre à jour la taille maximale de fenêtre si nécessaire
        if window_size > self.window_size:
            self.window_size = window_size
            # Mettre à jour la forme d'observation
            self.observation_shape = (
                len(self.timeframes),
                self.window_size,
                self.max_features,
            )

        logger.info(
            f"Configuration du timeframe {timeframe}: fenêtre={window_size}, features={len(features)}"
        )

    def _initialize_memory_metrics(self):
        """
        Initialize memory metrics after configuration.
        """
        try:
            # Get initial memory usage
            self.initial_memory_mb = self._get_memory_usage_mb()
            self.memory_peak_mb = self.initial_memory_mb

            # Update performance metrics
            self._update_performance_metrics("memory_peak_mb", self.initial_memory_mb)

        except Exception as e:
            logger.error(f"Error initializing memory metrics: {str(e)}")
            self._update_performance_metrics(
                "errors_count",
                self.get_performance_metrics().get("errors_count", 0) + 1,
            )

    def _get_data_hash(self, data: np.ndarray) -> str:
        """
        Generate a hash key for the input data to be used in the scaler cache.

        Args:
            data: Input data array to hash

        Returns:
            str: MD5 hash of the data's content
        """
        # Convert data to bytes and generate MD5 hash
        return hashlib.md5(data.tobytes()).hexdigest()

    def _get_memory_usage_mb(self):
        """
        Get current memory usage in MB with monitoring.
        """
        try:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            memory_mb = mem_info.rss / (1024 * 1024)

            # Vérifier les seuils critiques
            if memory_mb > self.memory_config["memory_critical_threshold_mb"]:
                logger.error(
                    f"CRITICAL: Memory usage exceeds critical threshold: {memory_mb:.1f} MB"
                )
                metrics = self.get_performance_metrics()
                warnings_count = metrics.get("warnings_count", 0)
                self._update_performance_metrics("warnings_count", warnings_count + 1)
            elif memory_mb > self.memory_config["memory_warning_threshold_mb"]:
                logger.warning(f"Memory usage warning: {memory_mb:.1f} MB")
                metrics = self.get_performance_metrics()
                warnings_count = metrics.get("warnings_count", 0)
                self._update_performance_metrics("warnings_count", warnings_count + 1)

            # Mettre à jour le pic de mémoire
            self.memory_peak_mb = max(getattr(self, "memory_peak_mb", 0), memory_mb)
            self._update_performance_metrics("memory_peak_mb", self.memory_peak_mb)

            return memory_mb

        except Exception as e:
            logger.error(f"Error getting memory usage: {str(e)}")
            self._update_performance_metrics(
                "errors_count",
                self.get_performance_metrics().get("errors_count", 0) + 1,
            )
            return 0  # Return 0 on error

    def _cleanup_memory(self):
        """
        Helper method to clean up memory with aggressive cleanup.
        """
        try:
            # Clear cached data
            if hasattr(self, "current_chunk_data"):
                self.current_chunk_data = None

            # Clear scaler caches
            for scaler in self.scalers.values():
                if scaler is not None:
                    if hasattr(scaler, "clear_cache"):
                        scaler.clear_cache()

            # Force garbage collection
            if self.memory_config["force_gc"]:
                gc.collect()

            # Log memory usage
            current_memory = self._get_memory_usage_mb()
            if current_memory > self.memory_peak_mb:
                self.memory_peak_mb = current_memory
                self._update_performance_metrics("memory_peak_mb", self.memory_peak_mb)

            logger.info(
                f"Memory cleanup completed. Current usage: {current_memory:.1f} MB"
            )

        except Exception as e:
            logger.error(f"Error during memory cleanup: {str(e)}")
            self._update_performance_metrics("errors_count", 1)

    def _update_performance_metrics(self, metric: str, value: Any) -> None:
        """
        Update performance metrics safely.

        Args:
            metric: The metric name to update
            value: The new value for the metric
        """
        if not hasattr(self, "_performance_metrics"):
            self._performance_metrics = {
                "gc_collections": 0,
                "memory_peak_mb": self.initial_memory_mb,
                "errors_count": 0,
                "warnings_count": 0,
            }

        self._performance_metrics[metric] = value

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get the current performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        if not hasattr(self, "_performance_metrics"):
            # Utiliser la mémoire initiale si elle est définie, sinon 0
            initial_memory = getattr(self, "initial_memory_mb", 0)
            return {
                "gc_collections": 0,
                "memory_peak_mb": initial_memory,
                "errors_count": 0,
                "warnings_count": 0,
            }
        return self._performance_metrics

    def _init_scalers(self):
        """
        Initialize scalers for each timeframe with advanced normalization.

        Each timeframe gets its own scaler with specific parameters:
        - 5m: MinMaxScaler with feature_range (-1, 1)
        - 1h: StandardScaler with mean=0, std=1
        - 4h: RobustScaler for outlier resistance

        Memory optimizations:
        - Use float32 for scaler parameters
        - Cache scaler parameters efficiently
        """
        # Nettoyer les scalers existants
        if self.scalers:
            for scaler in self.scalers.values():
                if scaler is not None:
                    del scaler
            self.scalers = {tf: None for tf in self.timeframes}
            gc.collect()

        # Initialiser les nouveaux scalers
        for tf in self.timeframes:
            if tf == "5m":
                self.scalers[tf] = MinMaxScaler(feature_range=(0, 1), copy=False)
            elif tf == "1h":
                self.scalers[tf] = StandardScaler(copy=False)
            elif tf == "4h":
                self.scalers[tf] = RobustScaler(copy=False)
            else:
                self.scalers[tf] = StandardScaler(copy=False)

            # Optimiser la mémoire en utilisant float32
            if hasattr(self.scalers[tf], "dtype"):
                self.scalers[tf].dtype = np.float32

        logger.info(f"Initialized scalers for timeframes: {list(self.scalers.keys())}")
        if not self.normalize:
            logger.info("Normalization disabled - no scalers initialized")
            return

        scaler_configs = {
            "5m": {"scaler_type": "minmax", "feature_range": (0, 1)},
            "1h": {"scaler_type": "standard"},
            "4h": {"scaler_type": "robust"},
        }

        for tf in self.timeframes:
            config = scaler_configs.get(tf, {"scaler_type": "standard"})

            if config["scaler_type"] == "minmax":
                scaler = MinMaxScaler(feature_range=config.get("feature_range", (0, 1)))
            elif config["scaler_type"] == "standard":
                scaler = StandardScaler()
            elif config["scaler_type"] == "robust":
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaler type: {config['scaler_type']}")

            self.scalers[tf] = scaler
            logger.info(
                f"Scaler initialized for timeframe {tf}: {config['scaler_type']} "
                f"with params: {config}"
            )

    def fit_scalers(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Fit scalers on the provided data with memory optimization.

        Args:
            data: Dictionary mapping timeframes to DataFrames
        """
        if not self.normalize:
            return

        logger.info("Fitting scalers on provided data...")

        # Vérifier la mémoire avant le fitting
        current_memory = self._get_memory_usage_mb()
        if current_memory > self.memory_config["memory_warning_threshold_mb"]:
            logger.warning(f"Memory usage high before fitting: {current_memory:.1f} MB")

        try:
            for tf in self.timeframes:
                if tf not in self.scalers:
                    raise ValueError(f"Scaler not initialized for timeframe {tf}")
                if self.scalers[tf] is None:
                    if tf == "5m":
                        self.scalers[tf] = MinMaxScaler(
                            feature_range=(-1, 1), copy=False
                        )
                    elif tf == "1h":
                        self.scalers[tf] = StandardScaler(copy=False)
                    elif tf == "4h":
                        self.scalers[tf] = RobustScaler(copy=False)
                    else:
                        self.scalers[tf] = StandardScaler(copy=False)
                    logger.info(f"Initializing scaler for timeframe {tf}")

            for tf, df in data.items():
                if tf not in self.timeframes:
                    logger.warning(f"Skipping unknown timeframe {tf}")
                    continue

                columns = [
                    col for col in self.features_config.get(tf, []) if col in df.columns
                ]
                if not columns:
                    logger.warning(
                        f"No matching feature columns found for timeframe {tf}"
                    )
                    continue

                # Utiliser float32 pour optimiser la mémoire
                timeframe_data = df[columns].values.astype(np.float32)

                if not np.isfinite(timeframe_data).all():
                    logger.warning(
                        f"Non-finite values found in {tf} data. Replacing with zeros."
                    )
                    timeframe_data = np.nan_to_num(
                        timeframe_data, nan=0.0, posinf=0.0, neginf=0.0
                    )

                if self.scalers[tf] is None:
                    raise ValueError(
                        f"Scaler not properly initialized for timeframe {tf}"
                    )

                if len(timeframe_data) < 2:
                    raise ValueError(
                        f"Not enough data samples ({len(timeframe_data)}) to fit scaler for {tf}"
                    )

                # Generate cache key and check cache first
                data_hash = self._get_data_hash(timeframe_data)
                cache_key = f"{tf}_{data_hash}"

                if cache_key in self.scaler_cache:
                    self.scaler_cache_hits += 1
                    self.scalers[tf] = self.scaler_cache[cache_key]
                    logger.debug(f"Using cached scaler for {tf}")
                else:
                    self.scaler_cache_misses += 1
                    # Fit new scaler
                    self.scalers[tf].fit(timeframe_data)

                    # Cache the fitted scaler (LRU eviction)
                    if len(self.scaler_cache) >= self._max_scaler_cache_size:
                        del self.scaler_cache[next(iter(self.scaler_cache))]
                    self.scaler_cache[cache_key] = self.scalers[tf]
                    logger.info(
                        f"Fitted new scaler for {tf} on {len(timeframe_data)} samples"
                    )

            # Sauvegarder les scalers si nécessaire
            if self.scaler_path:
                self.save_scalers()

            # Nettoyer la mémoire après le fitting
            if self.memory_config["aggressive_cleanup"]:
                self._cleanup_memory()

        except Exception as e:
            logger.error(f"Error fitting scalers: {str(e)}")
            self._update_performance_metrics(
                "errors_count",
                self.get_performance_metrics().get("errors_count", 0) + 1,
            )
            raise

        # Update memory metrics
        current_memory = self._get_memory_usage_mb()
        self.memory_peak_mb = max(getattr(self, "memory_peak_mb", 0), current_memory)
        self._update_performance_metrics("memory_peak_mb", self.memory_peak_mb)

        # Log cache statistics
        cache_hit_rate = (
            (
                self.scaler_cache_hits
                / (self.scaler_cache_hits + self.scaler_cache_misses)
            )
            * 100
            if (self.scaler_cache_hits + self.scaler_cache_misses) > 0
            else 0
        )

        logger.info(
            f"Scaler cache stats: {len(self.scaler_cache)} cached scalers, "
            f"{self.scaler_cache_hits} hits, {self.scaler_cache_misses} misses, "
            f"{cache_hit_rate:.1f}% hit rate"
        )

    def build_multi_channel_observation(
        self, current_idx: int, data: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """
        Build a multi-channel observation with all timeframes and memory optimization.

        Args:
            current_idx: Current index in the data
            data: Dictionary mapping timeframes to DataFrames

        Returns:
            3D numpy array of shape (n_timeframes, window_size, n_features)

        Raises:
            ValueError: If data is missing or insufficient
            KeyError: If required features are missing
            RuntimeError: If observation shape mismatch occurs
        """
        try:
            # Vérifier la mémoire avant le traitement
            current_memory = self._get_memory_usage_mb()
            if current_memory > self.memory_config["memory_warning_threshold_mb"]:
                logger.warning(
                    f"Memory usage high before building observation: {current_memory:.1f} MB"
                )

            # Build observations for each timeframe
            observations = self.build_observation(current_idx, data)

            # Initialize output array with fixed shape
            output = np.zeros(self.observation_shape, dtype=np.float32)

            # Fill the output array with observations
            for i, (tf, obs) in enumerate(observations.items()):
                if obs is not None and len(obs) > 0:
                    # Take the most recent window_size observations
                    obs = obs[-self.window_size :]

                    # Ensure correct number of features
                    if obs.shape[1] > self.max_features:
                        obs = obs[:, : self.max_features]
                    elif obs.shape[1] < self.max_features:
                        # Pad with zeros if needed
                        pad_width = ((0, 0), (0, self.max_features - obs.shape[1]))
                        obs = np.pad(obs, pad_width, mode="constant")

                    # Handle window size
                    if obs.shape[0] < self.window_size:
                        # Pad with zeros at the beginning
                        pad_width = ((self.window_size - obs.shape[0], 0), (0, 0))
                        obs = np.pad(obs, pad_width, mode="constant")
                    elif obs.shape[0] > self.window_size:
                        # Take the most recent observations
                        obs = obs[-self.window_size :]

                    # Store in output array
                    output[i] = obs

            # Mettre à jour les métriques de mémoire
            current_memory = self._get_memory_usage_mb()
            self.memory_peak_mb = max(
                getattr(self, "memory_peak_mb", 0), current_memory
            )
            self._update_performance_metrics("memory_peak_mb", self.memory_peak_mb)

            return output

        except Exception as e:
            logger.error(f"Error building multi-channel observation: {str(e)}")
            raise

    def get_observation_shape(self) -> Tuple[int, int, int]:
        """
        Retourne la forme de l'observation (sans la dimension du portefeuille).

        Returns:
            Tuple[int, int, int]: (n_timeframes, window_size, n_features)
        """
        return (len(self.timeframes), self.window_size, self.max_features)

    def get_portfolio_state_dim(self) -> int:
        """
        Retourne la dimension de l'état du portefeuille.

        Returns:
            int: Dimension de l'état du portefeuille
        """
        if hasattr(self, "_build_portfolio_state"):
            dummy_portfolio = np.zeros(1)
            return len(self._build_portfolio_state(dummy_portfolio))
        return 17  # Valeur par défaut si _build_portfolio_state n'existe pas

    def calculate_expected_flat_dimension(
        self, portfolio_included: bool = False
    ) -> int:
        """
        Calculate the expected flattened dimension of the observation state.

        Args:
            portfolio_included: Whether to include the portfolio state in the calculation.

        Returns:
            The total expected number of features in the flattened observation.
        """
        # The shape is (n_timeframes, window_size, max_features)
        n_timeframes, window_size, n_features = self.get_observation_shape()

        # For a flattened vector, the total dimension is channels * time * features
        total_dim = n_timeframes * window_size * n_features

        # This version may also include portfolio state
        if self.include_portfolio_state and portfolio_included:
            # This is a simplified placeholder. A real implementation would get this from a portfolio manager.
            # Based on build_portfolio_state, we have 7 base features + 5*2 position features = 17
            total_dim += 17

        logger.info(
            f"Calculated expected flat dimension: {total_dim} (portfolio included: {portfolio_included})"
        )
        return total_dim

    def validate_dimension(self, data: Dict[str, pd.DataFrame], portfolio_manager=None):
        """
        Validates the actual dimension of a built state against the expected dimension.

        Args:
            data: A sample data slice to build a test observation.
            portfolio_manager: An optional portfolio manager instance.

        Raises:
            ValueError: If the actual dimension does not match the expected dimension.
        """
        # We pass the portfolio_manager to the calculation method to decide if it should be included
        expected_dim = self.calculate_expected_flat_dimension(
            portfolio_manager is not None
        )

        # Build a sample state to get the actual dimension
        # We need a sample index, let's take the last one from the largest dataframe
        if not data:
            logger.warning("Cannot validate dimension without data.")
            return True

        max_len = max(len(df) for df in data.values())
        current_idx = max_len - 1

        test_observation_3d = self.build_multi_channel_observation(current_idx, data)

        if test_observation_3d is None:
            logger.warning(
                "Could not build a sample observation for validation, skipping."
            )
            return True

        actual_dim = (
            test_observation_3d.shape[0]
            * test_observation_3d.shape[1]
            * test_observation_3d.shape[2]
        )

        if actual_dim != expected_dim:
            error_report = self._generate_error_report(
                actual_dim, expected_dim, portfolio_manager is not None
            )
            logger.error(f"Dimension mismatch detected:\n{error_report}")
            return False

        logger.info(f"Dimension validation passed: {actual_dim} == {expected_dim}")
        return True

    def _generate_error_report(
        self, actual_dim: int, expected_dim: int, portfolio_included: bool
    ) -> Dict[str, Any]:
        """Generates a detailed report for a dimension mismatch error."""
        n_timeframes, window_size, n_features = self.get_observation_shape()
        market_contribution = n_timeframes * window_size * n_features

        portfolio_contribution = 0
        if self.include_portfolio_state and portfolio_included:
            portfolio_contribution = expected_dim - market_contribution

        discrepancy = actual_dim - expected_dim
        analysis = f"⚠️ System has a {-discrepancy} dimension discrepancy."

        return {
            "expected_dimension": expected_dim,
            "actual_dimension": actual_dim,
            "discrepancy": discrepancy,
            "discrepancy_analysis": analysis,
            "calculation_breakdown": {
                "observation_shape": self.get_observation_shape(),
                "market_data_contribution": market_contribution,
                "portfolio_contribution": portfolio_contribution,
                "window_size": self.window_size,
                "features_per_timeframe": self.nb_features_per_tf,
            },
        }

    def build_portfolio_state(self, portfolio_manager: Any) -> np.ndarray:
        """
        Build portfolio state information to include in observations.

        Args:
            portfolio_manager: Portfolio manager instance

        Returns:
            Numpy array containing portfolio state information
        """
        if not self.include_portfolio_state or portfolio_manager is None:
            return np.zeros(17, dtype=np.float32)  # Return zero-padded portfolio state

        try:
            metrics = portfolio_manager.get_metrics()
            portfolio_state = [
                metrics.get("cash", 0.0),
                metrics.get("total_capital", 0.0),
                metrics.get("total_pnl_pct", 0.0),  # Using total_pnl_pct as returns
                metrics.get("sharpe_ratio", 0.0),
                metrics.get("drawdown", 0.0),
                len(metrics.get("positions", {})),
                (
                    (metrics.get("total_capital", 0.0) - metrics.get("cash", 0.0))
                    / metrics.get("total_capital", 0.0)
                    if metrics.get("total_capital", 0.0) > 0
                    else 0.0
                ),
            ]

            # Add individual position information (up to 5 largest positions)
            sorted_positions = sorted(
                metrics.get("positions", {}).items(),
                key=lambda x: abs(x[1].get("size", 0.0)),
                reverse=True,
            )[:5]

            for i, (asset, position_obj) in enumerate(sorted_positions):
                portfolio_state.append(position_obj.get("size", 0.0))
                portfolio_state.append(hash(asset) % 1000)  # Simple asset encoding

            # Pad remaining position slots with zeros
            for i in range(len(sorted_positions), 5):
                portfolio_state.append(0.0)
                portfolio_state.append(0.0)

            return np.array(portfolio_state, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error building portfolio state: {e}")
            return np.zeros(17, dtype=np.float32)  # Return zero-padded portfolio state

    def validate_observation(self, observation: np.ndarray) -> bool:
        """
        Validate that an observation meets design specifications.

        Args:
            observation: Observation array to validate

        Returns:
            True if observation is valid, False otherwise
        """
        try:
            # Check shape according to design: (3, window_size, nb_features)
            if observation.shape[0] != len(self.timeframes):
                logger.error(
                    f"Invalid observation shape: expected {len(self.timeframes)} timeframes, got {observation.shape[0]}"
                )
                return False

            if observation.shape[1] != self.window_size:
                logger.error(
                    f"Invalid observation shape: expected window size {self.window_size}, got {observation.shape[1]}"
                )
                return False

            # Check for NaN or infinite values
            if not np.isfinite(observation).all():
                logger.error("Observation contains NaN or infinite values")
                return False

            # Check and fix value ranges (normalized data should be roughly in [-3, 3] range)
            if self.normalize:
                max_abs_val = np.abs(observation).max()
                if max_abs_val > 10:
                    logger.warning(
                        f"[NORMALIZATION FIX] Observation values unnormalized: max={max_abs_val:.1f}, applying clipping"
                    )
                    # Clip extreme values to reasonable range
                    observation = np.clip(observation, -10.0, 10.0)
                    logger.info(
                        f"[NORMALIZATION FIX] Values clipped to [-10, 10], new max: {np.abs(observation).max():.4f}"
                    )

            return True

        except Exception as e:
            logger.error(f"Error validating observation: {e}")
            return False

    def get_feature_names(self, timeframe: str) -> List[str]:
        """
        Get feature names for a specific timeframe.

        Args:
            timeframe: Timeframe to get features for

        Returns:
            List of feature names
        """
        return self.features_config.get(timeframe, [])

    def reset_scalers(self) -> None:
        """Reset all scalers to unfitted state."""
        self._init_scalers()
        logger.info("All scalers have been reset")

    def get_normalization_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get normalization statistics from fitted scalers.

        Returns:
            Dictionary containing mean and scale for each timeframe
        """
        stats = {}

        for tf, scaler in self.scalers.items():
            if scaler is not None and hasattr(scaler, "mean_"):
                stats[tf] = {
                    "mean": scaler.mean_,
                    "scale": scaler.scale_,
                    "var": getattr(scaler, "var_", None),
                }

        return stats

    def detect_market_regime(
        self, data: Dict[str, pd.DataFrame], current_idx: int
    ) -> Dict[str, float]:
        """
        Détecte le régime de marché actuel basé sur plusieurs indicateurs.

        Retourne un dictionnaire contenant:
        - regime: 'trending', 'ranging', 'volatile'
        - confidence: score de confiance [0-1]
        - volatility: niveau de volatilité [0-1]
        - trend_strength: force de la tendance [-1, 1]
        """
        regime_metrics = {
            "regime": "ranging",
            "confidence": 0.5,
            "volatility": 0.5,
            "trend_strength": 0.0,
        }

        try:
            # 1. Calcul de la volatilité
            volatility_scores = []
            trend_strengths = []

            for tf in self.timeframes:
                if tf not in data:
                    continue

                df = data[tf]
                start_idx = max(0, current_idx - 50)  # Fenêtre de 50 périodes
                window = df.iloc[start_idx : current_idx + 1]

                # Fonction utilitaire pour trouver une colonne avec gestion de la casse
                def find_column(possible_names, default=None):
                    for name in possible_names:
                        # Essayer le nom exact d'abord
                        if name in window.columns:
                            return name
                        # Puis essayer en majuscules
                        if name.upper() in window.columns:
                            return name.upper()
                        # Puis essayer en minuscules
                        if name.lower() in window.columns:
                            return name.lower()
                        # Puis essayer avec le préfixe du timeframe
                        tf_prefixed = f"{tf}_{name}"
                        if tf_prefixed in window.columns:
                            return tf_prefixed
                        # Essayer avec le préfixe du timeframe en majuscules
                        if tf_prefixed.upper() in window.columns:
                            return tf_prefixed.upper()
                    return default

                # Recherche des colonnes nécessaires
                close_col = find_column(["close", "CLOSE", "price", "PRICE"], None)
                high_col = find_column(["high", "HIGH"], close_col)
                low_col = find_column(["low", "LOW"], close_col)

                # Vérification des colonnes requises
                if close_col is None:
                    logger.warning(
                        f"Aucune colonne de prix de clôture trouvée pour {tf}. Colonnes disponibles: {list(window.columns)}"
                    )
                    logger.warning(
                        f"Types des colonnes: {[type(c) for c in window.columns]}"
                    )
                    logger.warning(
                        f"Colonnes en minuscules: {[c.lower() for c in window.columns]}"
                    )
                    continue

                # Vérification des autres colonnes
                missing_columns = []
                if high_col is None:
                    missing_columns.append("high")
                if low_col is None:
                    missing_columns.append("low")

                if missing_columns:
                    logger.warning(
                        f"Colonnes manquantes pour {tf}: {', '.join(missing_columns)}. Utilisation des valeurs de clôture comme substitut."
                    )
                    if high_col is None:
                        high_col = close_col
                    if low_col is None:
                        low_col = close_col

                # Vérification finale des colonnes
                try:
                    logger.debug(f"=== DÉBUT VÉRIFICATION COLONNES POUR {tf} ===")
                    logger.debug(f"Colonnes disponibles: {list(window.columns)}")
                    logger.debug(
                        f"Types des colonnes: {[type(c) for c in window.columns]}"
                    )
                    logger.debug(
                        f"Valeurs de close_col: '{close_col}', type: {type(close_col)}"
                    )
                    logger.debug(
                        f"Valeurs de high_col: '{high_col}', type: {type(high_col)}"
                    )
                    logger.debug(
                        f"Valeurs de low_col: '{low_col}', type: {type(low_col)}"
                    )

                    # Vérification des colonnes dans le DataFrame
                    logger.debug(f"close_col in columns: {close_col in window.columns}")
                    logger.debug(f"high_col in columns: {high_col in window.columns}")
                    logger.debug(f"low_col in columns: {low_col in window.columns}")

                    # Vérification des valeurs
                    close_val = (
                        window[close_col].iloc[-1]
                        if not window[close_col].empty
                        else None
                    )
                    high_val = (
                        window[high_col].iloc[-1]
                        if not window[high_col].empty
                        else None
                    )
                    low_val = (
                        window[low_col].iloc[-1] if not window[low_col].empty else None
                    )

                    logger.debug(
                        f"Colonnes sélectionnées pour {tf} - Close: '{close_col}', High: '{high_col}', Low: '{low_col}'"
                    )
                    logger.debug(
                        f"Valeurs de test - Close: {close_val}, High: {high_val}, Low: {low_val}"
                    )
                    logger.debug(
                        f"Types des données - Close: {type(close_val)}, High: {type(high_val)}, Low: {type(low_val)}"
                    )
                    logger.debug(f"=== FIN VÉRIFICATION COLONNES POUR {tf} ===")

                except Exception as e:
                    logger.error(f"Erreur lors de l'accès aux colonnes: {str(e)}")
                    logger.error(f"Colonnes disponibles: {list(window.columns)}")
                    logger.error(
                        f"Types des colonnes: {[type(c) for c in window.columns]}"
                    )
                    raise

                try:
                    # Vérification supplémentaire de la colonne close_col
                    if close_col not in window.columns:
                        logger.error(
                            f"ERREUR CRITIQUE: La colonne '{close_col}' n'existe pas dans le DataFrame. Colonnes disponibles: {list(window.columns)}"
                        )
                        logger.error(
                            f"Types des colonnes: {[type(c) for c in window.columns]}"
                        )
                        continue

                    prices = window[close_col]
                    if len(prices) < 20:  # Minimum 20 périodes
                        logger.warning(
                            f"Pas assez de données pour {tf}: {len(prices)} périodes (minimum 20 requises)"
                        )
                        continue

                    logger.debug(
                        f"Données de prix pour {tf} - Taille: {len(prices)}, Valeurs: {prices.tolist()[-5:]}"
                    )

                except Exception as e:
                    logger.error(
                        f"Erreur lors de l'accès à la colonne {close_col}: {str(e)}"
                    )
                    logger.error(f"Colonnes disponibles: {list(window.columns)}")
                    logger.error(f"Type de window: {type(window)}")
                    logger.error(f"Type de close_col: {type(close_col)}")
                    continue

                try:
                    logger.debug(f"=== DÉBUT CALCUL VOLATILITÉ POUR {tf} ===")

                    # Vérification des données d'entrée
                    logger.debug(f"Type de window: {type(window)}")
                    logger.debug(f"Colonnes dans window: {window.columns.tolist()}")
                    logger.debug(f"high_col: '{high_col}', type: {type(high_col)}")
                    logger.debug(f"low_col: '{low_col}', type: {type(low_col)}")
                    logger.debug(f"prices type: {type(prices)}")
                    logger.debug(f"prices sample: {prices.head(3).tolist()}")

                    # Calcul de la volatilité (ATR sur 14 périodes)
                    high = window[high_col]
                    low = window[low_col]

                    logger.debug(f"high sample: {high.head(3).tolist()}")
                    logger.debug(f"low sample: {low.head(3).tolist()}")

                    # True Range = max(high-low, |high - close_prev|, |low - close_prev|)
                    tr1 = high - low
                    logger.debug(f"tr1 sample: {tr1.head(3).tolist()}")

                    prices_shifted = prices.shift(1)
                    logger.debug(
                        f"prices_shifted sample: {prices_shifted.head(3).tolist()}"
                    )

                    tr2 = (high - prices_shifted).abs()
                    logger.debug(f"tr2 sample: {tr2.head(3).tolist()}")

                    tr3 = (low - prices_shifted).abs()
                    logger.debug(f"tr3 sample: {tr3.head(3).tolist()}")

                    true_range_df = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3})
                    logger.debug(f"true_range_df head:\n{true_range_df.head()}")

                    true_range = true_range_df.max(axis=1)
                    logger.debug(f"true_range sample: {true_range.head(3).tolist()}")

                    # ATR = Moyenne mobile du True Range
                    atr_series = true_range.rolling(window=14, min_periods=1).mean()
                    logger.debug(f"atr_series sample: {atr_series.head(3).tolist()}")

                    atr = atr_series.iloc[-1] if not atr_series.empty else 0
                    logger.debug(f"ATR: {atr}")

                    # Normalisation par le prix moyen
                    prices_mean = prices.mean()
                    logger.debug(f"prices_mean: {prices_mean}")

                    atr_normalized = atr / prices_mean if prices_mean != 0 else 0
                    logger.debug(f"ATR normalisé: {atr_normalized}")

                    volatility_scores.append(atr_normalized)
                    logger.debug(f"=== FIN CALCUL VOLATILITÉ POUR {tf} ===")

                except Exception as e:
                    logger.warning(
                        f"Erreur lors du calcul de la volatilité pour {tf}: {str(e)}"
                    )
                    continue

                # Calcul de la force de tendance (ADX)
                if "ADX_14" in window.columns:
                    adx = window["ADX_14"].iloc[-1] / 100  # Normalisation 0-1
                    trend_strengths.append(adx)

            # Calcul des métriques agrégées
            if volatility_scores is not None and len(volatility_scores) > 0:
                regime_metrics["volatility"] = float(np.mean(volatility_scores))

            if trend_strengths is not None and len(trend_strengths) > 0:
                regime_metrics["trend_strength"] = (
                    float(np.mean(trend_strengths)) * 2 - 1
                )  # -1 à 1

            # Détection du régime
            if regime_metrics["trend_strength"] > 0.3:
                regime_metrics["regime"] = "trending"
                regime_metrics["confidence"] = min(
                    1.0, regime_metrics["trend_strength"]
                )
            elif regime_metrics["volatility"] > 0.7:
                regime_metrics["regime"] = "volatile"
                regime_metrics["confidence"] = min(1.0, regime_metrics["volatility"])
            else:
                regime_metrics["regime"] = "ranging"
                regime_metrics["confidence"] = 1.0 - max(
                    regime_metrics["trend_strength"], regime_metrics["volatility"]
                )

            return regime_metrics

        except Exception as e:
            logger.error(f"Erreur lors de la détection du régime de marché: {e}")
            return regime_metrics

    def calculate_market_volatility(
        self, data: Dict[str, pd.DataFrame], current_idx: int
    ) -> float:
        """
        Calcule la volatilité du marché avec une approche plus robuste.

        Args:
            data: Dictionnaire des données par timeframe
            current_idx: Index actuel dans les données

        Returns:
            Score de volatilité normalisé [0-1]
        """
        try:
            # Utiliser la détection de régime pour obtenir la volatilité
            regime_metrics = self.detect_market_regime(data, current_idx)

            # Mise à jour de l'historique de volatilité
            self.volatility_history.append(regime_metrics["volatility"])
            if len(self.volatility_history) > self.volatility_window:
                self.volatility_history.pop(0)

            # Calcul de la volatilité normalisée par rapport à l'historique
            if len(self.volatility_history) > 1:
                hist_mean = np.mean(self.volatility_history)
                hist_std = np.std(self.volatility_history)

                if hist_std > 0:
                    normalized_vol = (
                        regime_metrics["volatility"] - hist_mean
                    ) / hist_std
                    # Transformation en échelle 0-1 avec saturation
                    normalized_vol = 0.5 + np.tanh(normalized_vol) * 0.5
                else:
                    normalized_vol = 0.5
            else:
                normalized_vol = regime_metrics["volatility"]

            logger.debug(
                f"Volatilité du marché: {normalized_vol:.3f} (régime: {regime_metrics['regime']}, confiance: {regime_metrics['confidence']:.2f})"
            )

            return min(1.0, max(0.0, normalized_vol))

        except Exception as e:
            logger.error(f"Erreur dans le calcul de la volatilité: {e}")
            return 0.5  # Valeur par défaut en cas d'erreur

    def adapt_window_size(self, volatility: float) -> int:
        """
        Adapt window size based on market volatility.

        Args:
            volatility: Normalized volatility score (0.0 to 2.0+)

        Returns:
            Adapted window size

        Raises:
            ValueError: If volatility is out of expected range
        """
        if not self.adaptive_window:
            return self.base_window_size

        if not (0.0 <= volatility <= 2.0):
            raise ValueError(
                f"Volatility score {volatility} out of expected range [0.0, 2.0]"
            )

        # High volatility -> smaller window (more reactive)
        # Low volatility -> larger window (more stable)

        # Calculate window size based on volatility
        if volatility < 0.3:
            # Low volatility: use larger window for stability
            adapted_size = int(self.base_window_size * 1.5)
        elif volatility < 0.7:
            # Medium volatility: use base window size
            adapted_size = self.base_window_size
        else:
            # High volatility: use smaller window for reactivity
            adapted_size = int(self.base_window_size * 0.7)

        # Ensure window size stays within bounds
        adapted_size = max(
            self.min_window_size, min(adapted_size, self.max_window_size)
        )

        # Log the adaptation
        logger.info(
            f"Adapting window size: base={self.base_window_size}, volatility={volatility:.2f}, adapted={adapted_size}"
        )

        return adapted_size

    def _update_timeframe_weights(self, regime_metrics: Dict[str, float]) -> None:
        """
        Met à jour les poids des différents timeframes en fonction du régime de marché détecté.

        Stratégie de pondération :
        - Marché en tendance : Poids plus élevé sur les timeframes plus longs (4h, 1h)
        - Marché range : Poids équilibré entre les timeframes
        - Marché volatile : Poids plus élevé sur les timeframes plus courts (5m, 1h)

        Args:
            regime_metrics: Métriques du régime de marché (récupérées via detect_market_regime)
        """
        regime = regime_metrics.get("regime", "ranging")
        confidence = regime_metrics.get("confidence", 0.5)

        # Poids de base pour chaque régime
        if regime == "trending":
            # Privilégier les timeframes plus longs en tendance
            weights = {
                "5m": 0.7,  # Moins important en tendance établie
                "1h": 1.0,  # Important pour identifier la tendance
                "4h": 1.3,  # Très important pour la tendance à long terme
            }
        elif regime == "volatile":
            # Privilégier les timeframes plus courts en période de volatilité
            weights = {
                "5m": 1.3,  # Très important pour la réactivité
                "1h": 1.0,  # Important pour le contexte
                "4h": 0.7,  # Moins important en période de forte volatilité
            }
        else:  # ranging
            # Poids équilibré en marché range
            weights = {
                "5m": 1.0,  # Important pour les mouvements courts
                "1h": 1.0,  # Contexte moyen terme
                "4h": 1.0,  # Contexte long terme
            }

        # Ajuster les poids en fonction de la confiance
        # Plus la confiance est élevée, plus on applique les poids du régime
        # Avec une confiance faible, on se rapproche de poids neutres (1.0)
        for tf in self.timeframe_weights:
            if tf in weights:
                # Interpolation linéaire entre poids neutre (1.0) et le poids cible
                # en fonction de la confiance
                target_weight = weights[tf]
                self.timeframe_weights[tf] = 1.0 + (target_weight - 1.0) * confidence

        # Normaliser les poids pour que leur somme reste constante
        total_weight = sum(self.timeframe_weights.values())
        num_timeframes = len(self.timeframe_weights)
        if total_weight > 0:
            for tf in self.timeframe_weights:
                self.timeframe_weights[tf] = (
                    self.timeframe_weights[tf] / total_weight
                ) * num_timeframes

        logger.debug(
            f"Mise à jour des poids des timeframes (régime: {regime}, confiance: {confidence:.2f}): {self.timeframe_weights}"
        )

    def update_adaptive_window(
        self, data: Dict[str, pd.DataFrame], current_idx: int
    ) -> None:
        """
        Update the window size based on current market conditions.

        Args:
            data: Dictionary mapping timeframes to DataFrames
            current_idx: Current index in the data

        Raises:
            ValueError: If data is invalid or volatility calculation fails
        """
        if not self.adaptive_window:
            return

        try:
            # Détecter le régime de marché
            regime_metrics = self.detect_market_regime(data, current_idx)

            # Mettre à jour les poids des timeframes en fonction du régime
            self._update_timeframe_weights(regime_metrics)

            # Calculate current market volatility
            volatility = self.calculate_market_volatility(data, current_idx)

            # Adapt window size
            new_window_size = self.adapt_window_size(volatility)

            # Update window size if it changed significantly (threshold of 10%)
            change_threshold = 0.10  # 10% change threshold
            if abs(new_window_size - self.window_size) > (
                self.window_size * change_threshold
            ):
                old_size = self.window_size
                self.window_size = new_window_size
                logger.info(
                    f"Adapted window size from {old_size} to {new_window_size} "
                    f"(volatility: {volatility:.3f}, change: {abs(new_window_size - old_size)} steps)"
                )

        except Exception as e:
            logger.error(f"Error updating adaptive window: {e}")
            raise ValueError(f"Failed to update adaptive window: {e}")

    def apply_timeframe_weighting(
        self, observations: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Apply intelligent weighting to different timeframes based on market conditions.

        Args:
            observations: Dictionary of observations by timeframe

        Returns:
            Weighted observations
        """
        weighted_observations = {}

        for tf, obs in observations.items():
            if tf in self.timeframe_weights:
                weight = self.timeframe_weights[tf]

                # Apply weight to the observation
                # For normalized data, we can scale the values
                weighted_obs = obs * weight

                # Ensure we don't lose important information by applying a minimum weight
                min_weight = 0.3
                if weight < min_weight:
                    weighted_obs = obs * min_weight + weighted_obs * (1 - min_weight)

                weighted_observations[tf] = weighted_obs
            else:
                weighted_observations[tf] = obs

        return weighted_observations

    def get_adaptive_stats(self) -> Dict[str, Union[int, float, List[float]]]:
        """
        Get statistics about the adaptive window system.

        Returns:
            Dictionary containing adaptive window statistics
        """
        return {
            "adaptive_enabled": self.adaptive_window,
            "base_window_size": self.base_window_size,
            "current_window_size": self.window_size,
            "min_window_size": self.min_window_size,
            "max_window_size": self.max_window_size,
            "volatility_history": self.volatility_history.copy(),
            "current_volatility": self.volatility_history[-1]
            if self.volatility_history
            else 0.0,
            "timeframe_weights": self.timeframe_weights.copy(),
        }

    def _get_column_mapping(self, df: pd.DataFrame, tf: str):
        if tf not in self._col_mappings:
            # build once
            m = {col.upper(): col for col in df.columns}
            self._col_mappings[tf] = m
        return self._col_mappings[tf]

    def _pad_features(self, obs: np.ndarray, target_length: int) -> np.ndarray:
        """
        Pad or truncate observation to match target feature length.

        Args:
            obs: Input observation array of shape (window_size, n_features)
            target_length: Target number of features

        Returns:
            Padded/truncated array of shape (window_size, target_length)
        """
        if obs.shape[1] < target_length:
            # Pad with zeros
            padding = ((0, 0), (0, target_length - obs.shape[1]))
            return np.pad(obs, padding, mode="constant")
        elif obs.shape[1] > target_length:
            # Truncate to target length
            return obs[:, :target_length]
        return obs

    def _apply_temporal_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique des transformations temporelles aux données d'entrée selon la configuration.

        Cette méthode ajoute des indicateurs techniques et des caractéristiques temporelles
        aux données de prix et de volume pour améliorer la capacité du modèle à détecter
        des motifs temporels.

        Args:
            df: DataFrame d'entrée avec les données de séries temporelles

        Returns:
            DataFrame avec des caractéristiques temporelles supplémentaires

        Raises:
            ValueError: Si les données d'entrée sont vides ou invalides
        """
        # Vérifier si les transformations temporelles sont activées
        if not hasattr(self, "config") or not self.config.get(
            "temporal_transforms", {}
        ).get("enabled", True):
            logger.debug(
                "Transformations temporelles désactivées dans la configuration"
            )
            return df

        # Vérifier les données d'entrée
        if df.empty or not isinstance(df, pd.DataFrame):
            logger.warning(
                "Données d'entrée vides ou invalides pour les transformations temporelles"
            )
            return pd.DataFrame()

        # Faire une copie pour éviter les effets de bord
        df = df.copy()

        # Journaliser le nombre de lignes et colonnes initiales
        logger.debug(
            "Application des transformations temporelles sur un DataFrame de forme %s",
            df.shape,
        )

        # Identifier les colonnes numériques pour les transformations
        numeric_cols = df.select_dtypes(
            include=["float64", "float32", "int64", "int32"]
        ).columns.tolist()
        if not numeric_cols:
            logger.warning(
                "Aucune colonne numérique trouvée pour les transformations temporelles"
            )
            return df

        # 1. Différences premières
        if "diffs" in self.config.get("temporal_transforms", {}):
            diff_cols = self.config["temporal_transforms"]["diffs"]
            if not isinstance(diff_cols, list):
                logger.warning(
                    "La configuration 'diffs' doit être une liste de noms de colonnes"
                )
            else:
                logger.debug(
                    "Application des différences premières sur les colonnes: %s",
                    diff_cols,
                )
                for col in diff_cols:
                    try:
                        if col in df.columns:
                            # Calculer la différence première
                            diff_series = df[col].diff()

                            # Remplacer les valeurs infinies et NaN
                            if np.isinf(diff_series).any() or diff_series.isna().any():
                                logger.debug(
                                    "Valeurs infinies ou NaN détectées dans les différences pour %s",
                                    col,
                                )
                                diff_series = diff_series.replace(
                                    [np.inf, -np.inf], np.nan
                                ).fillna(0)

                            # Ajouter la colonne différenciée
                            new_col = f"{col}_diff"
                            df[new_col] = diff_series.astype(np.float32)
                            logger.debug(
                                "Colonne ajoutée: %s (min=%.4f, max=%.4f)",
                                new_col,
                                df[new_col].min(),
                                df[new_col].max(),
                            )
                        else:
                            logger.warning(
                                "Colonne non trouvée pour le calcul des différences: %s",
                                col,
                            )
                    except Exception as e:
                        logger.error(
                            "Erreur lors du calcul de la différence pour %s: %s",
                            col,
                            str(e),
                            exc_info=True,
                        )

        # Mettre à jour la liste des colonnes numériques après l'ajout des différences
        numeric_cols = df.select_dtypes(
            include=["float64", "float32", "int64", "int32"]
        ).columns.tolist()

        # 2. Moyennes mobiles
        if "rolling_windows" in self.config.get("temporal_transforms", {}):
            rolling_windows = self.config["temporal_transforms"]["rolling_windows"]
            if not isinstance(rolling_windows, list) or not all(
                isinstance(w, int) and w > 0 for w in rolling_windows
            ):
                logger.warning(
                    "La configuration 'rolling_windows' doit être une liste d'entiers positifs"
                )
            else:
                logger.debug(
                    "Calcul des moyennes mobiles avec les fenêtres: %s", rolling_windows
                )
                for window in rolling_windows:
                    if window < 1:
                        logger.warning(
                            "Fenêtre de moyenne mobile invalide (doit être >= 1): %d",
                            window,
                        )
                        continue

                    for col in numeric_cols:
                        try:
                            # Calculer la moyenne mobile
                            ma_series = (
                                df[col].rolling(window=window, min_periods=1).mean()
                            )

                            # Remplir les valeurs manquantes
                            if ma_series.isna().any():
                                logger.debug(
                                    "Valeurs manquantes détectées dans la moyenne mobile de %s (fenêtre=%d)",
                                    col,
                                    window,
                                )
                                ma_series = ma_series.ffill().bfill().fillna(0)

                            # Ajouter la colonne de moyenne mobile
                            new_col = f"{col}_ma{window}"
                            df[new_col] = ma_series.astype(np.float32)

                            # Journalisation des statistiques
                            logger.debug(
                                "Moyenne mobile ajoutée: %s (min=%.4f, max=%.4f, mean=%.4f)",
                                new_col,
                                df[new_col].min(),
                                df[new_col].max(),
                                df[new_col].mean(),
                            )

                        except Exception as e:
                            logger.error(
                                "Erreur lors du calcul de la moyenne mobile pour %s (fenêtre=%d): %s",
                                col,
                                window,
                                str(e),
                                exc_info=True,
                            )

        # 3. RSI (Relative Strength Index)
        rsi_config = self.config.get("temporal_transforms", {}).get("rsi_window")
        if rsi_config:
            try:
                # Vérifier que la configuration est valide
                if not isinstance(rsi_config, int) or rsi_config < 1:
                    logger.warning(
                        "La configuration RSI doit être un entier positif, reçu: %s",
                        rsi_config,
                    )
                elif "close" not in df.columns:
                    logger.warning("Colonne 'close' non trouvée pour le calcul du RSI")
                else:
                    window = int(rsi_config)
                    logger.debug(
                        "Calcul du RSI avec une fenêtre de %d périodes", window
                    )

                    # Calculer les variations de prix
                    delta = df["close"].diff()

                    # Séparer les gains et les pertes
                    gain = delta.where(delta > 0, 0.0)
                    loss = -delta.where(delta < 0, 0.0)

                    # Calculer les moyennes mobiles des gains et pertes
                    avg_gain = gain.rolling(window=window, min_periods=1).mean()
                    avg_loss = loss.rolling(window=window, min_periods=1).mean()

                    # Éviter la division par zéro
                    avg_loss = avg_loss.replace(0, np.nan)

                    # Calculer le RS et le RSI
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))

                    # Remplacer les valeurs infinies et NaN
                    rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(
                        50.0
                    )  # 50 est la valeur neutre du RSI

                    # Ajouter la colonne RSI
                    df["rsi"] = rsi.astype(np.float32)

                    # Journalisation des statistiques
                    logger.debug(
                        "RSI calculé (min=%.2f, max=%.2f, mean=%.2f)",
                        df["rsi"].min(),
                        df["rsi"].max(),
                        df["rsi"].mean(),
                    )

            except Exception as e:
                logger.error("Erreur lors du calcul du RSI: %s", str(e), exc_info=True)

        # 4. MACD (Moving Average Convergence Divergence)
        macd_config = self.config.get("temporal_transforms", {})
        required_macd_keys = ["macd_fast", "macd_slow", "macd_signal"]

        if all(k in macd_config for k in required_macd_keys):
            try:
                # Vérifier que les paramètres sont valides
                fast = int(macd_config["macd_fast"])
                slow = int(macd_config["macd_slow"])
                signal = int(macd_config["macd_signal"])

                if fast <= 0 or slow <= 0 or signal <= 0:
                    logger.warning(
                        "Les paramètres MACD doivent être des entiers positifs (fast=%d, slow=%d, signal=%d)",
                        fast,
                        slow,
                        signal,
                    )
                elif fast >= slow:
                    logger.warning(
                        "La période MACD rapide (%d) doit être inférieure à la période lente (%d)",
                        fast,
                        slow,
                    )
                elif "close" not in df.columns:
                    logger.warning("Colonne 'close' non trouvée pour le calcul du MACD")
                else:
                    logger.debug(
                        "Calcul du MACD avec fast=%d, slow=%d, signal=%d",
                        fast,
                        slow,
                        signal,
                    )

                    # Calculer les moyennes mobiles exponentielles
                    exp1 = (
                        df["close"].ewm(span=fast, adjust=False, min_periods=1).mean()
                    )
                    exp2 = (
                        df["close"].ewm(span=slow, adjust=False, min_periods=1).mean()
                    )

                    # Calculer la ligne MACD (différence entre les deux EMA)
                    macd_line = exp1 - exp2

                    # Calculer la ligne de signal (EMA de la ligne MACD)
                    signal_line = macd_line.ewm(
                        span=signal, adjust=False, min_periods=1
                    ).mean()

                    # Ajouter les colonnes au DataFrame
                    df["macd"] = macd_line.astype(np.float32)
                    df["macd_signal"] = signal_line.astype(np.float32)

                    # Calculer l'histogramme MACD (différence entre la ligne MACD et la ligne de signal)
                    df["macd_hist"] = (macd_line - signal_line).astype(np.float32)

                    # Journalisation des statistiques
                    logger.debug(
                        "MACD calculé - Ligne: min=%.4f, max=%.4f",
                        df["macd"].min(),
                        df["macd"].max(),
                    )
                    logger.debug(
                        "Signal MACD - min=%.4f, max=%.4f",
                        df["macd_signal"].min(),
                        df["macd_signal"].max(),
                    )
                    logger.debug(
                        "Histogramme MACD - min=%.4f, max=%.4f",
                        df["macd_hist"].min(),
                        df["macd_hist"].max(),
                    )

            except (ValueError, TypeError) as e:
                logger.error("Erreur de type dans les paramètres MACD: %s", str(e))
            except Exception as e:
                logger.error("Erreur lors du calcul du MACD: %s", str(e), exc_info=True)

        # 5. Bandes de Bollinger
        bb_config = self.config.get("temporal_transforms", {})
        if all(k in bb_config for k in ["bollinger_window", "bollinger_std"]):
            try:
                # Récupérer et valider les paramètres
                window = int(bb_config["bollinger_window"])
                std_dev = float(bb_config["bollinger_std"])

                if window <= 0:
                    logger.warning(
                        "La fenêtre des bandes de Bollinger doit être un entier positif, reçu: %d",
                        window,
                    )
                elif std_dev <= 0:
                    logger.warning(
                        "L'écart-type des bandes de Bollinger doit être un nombre positif, reçu: %.2f",
                        std_dev,
                    )
                elif "close" not in df.columns:
                    logger.warning(
                        "Colonne 'close' non trouvée pour le calcul des bandes de Bollinger"
                    )
                else:
                    logger.debug(
                        "Calcul des bandes de Bollinger avec fenêtre=%d, écart-type=%.2f",
                        window,
                        std_dev,
                    )

                    # Calculer la moyenne mobile (ligne médiane)
                    middle_band = (
                        df["close"].rolling(window=window, min_periods=1).mean()
                    )

                    # Calculer l'écart-type
                    rolling_std = (
                        df["close"].rolling(window=window, min_periods=1).std()
                    )

                    # Calculer les bandes supérieure et inférieure
                    upper_band = middle_band + (rolling_std * std_dev)
                    lower_band = middle_band - (rolling_std * std_dev)

                    # Ajouter les colonnes au DataFrame
                    df["bollinger_mid"] = middle_band.astype(np.float32)
                    df["bollinger_upper"] = upper_band.astype(np.float32)
                    df["bollinger_lower"] = lower_band.astype(np.float32)

                    # Calculer la largeur des bandes (pourcentage)
                    df["bollinger_width"] = (
                        (upper_band - lower_band) / middle_band * 100
                    ).astype(np.float32)

                    # Journalisation des statistiques
                    logger.debug(
                        "Bandes de Bollinger calculées - Moyenne: min=%.4f, max=%.4f",
                        df["bollinger_mid"].min(),
                        df["bollinger_mid"].max(),
                    )
                    logger.debug(
                        "Bande supérieure: min=%.4f, max=%.4f",
                        df["bollinger_upper"].min(),
                        df["bollinger_upper"].max(),
                    )
                    logger.debug(
                        "Bande inférieure: min=%.4f, max=%.4f",
                        df["bollinger_lower"].min(),
                        df["bollinger_lower"].max(),
                    )
                    logger.debug(
                        "Largeur des bandes: min=%.2f%%, max=%.2f%%",
                        df["bollinger_width"].min(),
                        df["bollinger_width"].max(),
                    )

            except (ValueError, TypeError) as e:
                logger.error(
                    "Erreur de type dans les paramètres des bandes de Bollinger: %s",
                    str(e),
                )
            except Exception as e:
                logger.error(
                    "Erreur lors du calcul des bandes de Bollinger: %s",
                    str(e),
                    exc_info=True,
                )

        # 6. Volume moyen mobile
        volume_windows = self.config.get("temporal_transforms", {}).get(
            "volume_rolling_windows",
            self.config.get("temporal_transforms", {}).get("rolling_windows", []),
        )

        if volume_windows and "volume" in df.columns:
            try:
                if not isinstance(volume_windows, list) or not all(
                    isinstance(w, int) and w > 0 for w in volume_windows
                ):
                    logger.warning(
                        "La configuration 'volume_rolling_windows' doit être une liste d'entiers positifs"
                    )
                else:
                    logger.debug(
                        "Calcul des moyennes mobiles de volume avec les fenêtres: %s",
                        volume_windows,
                    )

                    for window in volume_windows:
                        try:
                            if window < 1:
                                logger.warning(
                                    "Fenêtre de volume invalide (doit être >= 1): %d",
                                    window,
                                )
                                continue

                            # Calculer la moyenne mobile du volume
                            volume_ma = (
                                df["volume"]
                                .rolling(window=window, min_periods=1)
                                .mean()
                            )

                            # Remplir les valeurs manquantes
                            if volume_ma.isna().any():
                                logger.debug(
                                    "Valeurs manquantes détectées dans la moyenne mobile du volume (fenêtre=%d)",
                                    window,
                                )
                                volume_ma = volume_ma.ffill().bfill().fillna(0)

                            # Ajouter la colonne
                            col_name = f"volume_ma{window}"
                            df[col_name] = volume_ma.astype(np.float32)

                            # Journalisation des statistiques
                            logger.debug(
                                "Moyenne mobile de volume ajoutée: %s (min=%.2f, max=%.2f, mean=%.2f)",
                                col_name,
                                df[col_name].min(),
                                df[col_name].max(),
                                df[col_name].mean(),
                            )

                        except Exception as e:
                            logger.error(
                                "Erreur lors du calcul de la moyenne mobile du volume (fenêtre=%d): %s",
                                window,
                                str(e),
                                exc_info=True,
                            )

            except Exception as e:
                logger.error(
                    "Erreur lors du traitement des fenêtres de volume: %s",
                    str(e),
                    exc_info=True,
                )
        elif "volume" not in df.columns:
            logger.debug(
                "Colonne 'volume' non trouvée, calcul des moyennes mobiles de volume ignoré"
            )

        # Nettoyage final des valeurs manquantes et infinies
        try:
            # Compter les NaN et infinis avant nettoyage
            nan_before = df.isna().sum().sum()
            inf_before = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()

            # Remplacer les infinis par NaN d'abord
            df = df.replace([np.inf, -np.inf], np.nan)

            # Remplir les valeurs manquantes
            df = df.ffill().bfill()

            # Remplacer les valeurs manquantes restantes par 0
            df = df.fillna(0)

            # Vérifier s'il reste des valeurs manquantes ou infinies
            nan_after = df.isna().sum().sum()
            inf_after = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()

            # Journalisation du nettoyage
            if nan_before > 0 or inf_before > 0:
                logger.debug(
                    "Nettoyage des valeurs manquantes: %d NaN et %d infinis remplacés",
                    nan_before,
                    inf_before,
                )

            if nan_after > 0 or inf_after > 0:
                logger.warning(
                    "Il reste %d valeurs manquantes et %d valeurs infinies après nettoyage",
                    nan_after,
                    inf_after,
                )

            # Vérifier la cohérence des données
            inf_cols = df.columns[np.isinf(df).any()].tolist()
            if inf_cols:
                logger.warning(
                    "Valeurs infinies détectées dans les colonnes: %s", inf_cols
                )

            nan_cols = df.columns[df.isna().any()].tolist()
            if nan_cols:
                logger.warning(
                    "Valeurs manquantes détectées dans les colonnes: %s", nan_cols
                )

        except Exception as e:
            logger.error(
                "Erreur lors du nettoyage final des données: %s", str(e), exc_info=True
            )
            # En cas d'erreur, essayer de récupérer en forçant le remplissage par zéro
            df = df.fillna(0)
            df = df.replace([np.inf, -np.inf], 0)
            logger.warning(
                "Récupération après erreur: toutes les valeurs manquantes et infinies ont été remplacées par 0"
            )

        # Vérifier la forme finale des données
        logger.debug(
            "Traitement temporel terminé. Forme des données: %s, colonnes: %s",
            df.shape,
            ", ".join(df.columns.tolist()[:5]) + ("..." if len(df.columns) > 5 else ""),
        )

        return df

    def _apply_data_augmentation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data augmentation to the input data based on config.

        Args:
            df: Input DataFrame with time series data

        Returns:
            DataFrame with augmented data
        """
        if not hasattr(self, "config") or not self.config.get(
            "data_augmentation", {}
        ).get("enabled", False):
            return df

        df = df.copy()
        config = self.config.get("data_augmentation", {})
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

        # 1. Bruit gaussien
        self.features = config.get(
            "features", ["open", "high", "low", "close", "volume"]
        )
        self.technical_indicators = config.get("technical_indicators", [])

        # Log de la configuration des caractéristiques
        logger.info(f"Configuration - Features: {self.features}")
        logger.info(
            f"Configuration - Technical indicators: {self.technical_indicators}"
        )
        logger.info(
            f"Nombre total de caractéristiques: {len(self.features + self.technical_indicators)}"
        )
        logger.info(f"Observation shape: {self.observation_shape}")

        if config.get("gaussian_noise", {}).get("enabled", True):
            noise_std = config.get("gaussian_noise", {}).get("std", 0.01)
            for col in numeric_cols:
                if col in config.get("gaussian_noise", {}).get("exclude_columns", []):
                    continue
                noise = np.random.normal(0, noise_std * df[col].std(), size=len(df))
                df[col] = df[col] + noise

        # 2. Time warping (warping temporel)
        if config.get("time_warping", {}).get("enabled", False):
            window = config["time_warping"].get("window", 10)
            sigma = config["time_warping"].get("sigma", 0.2)

            for i in range(0, len(df) - window, window):
                window_slice = slice(i, min(i + window, len(df)))
                warp_factor = np.random.normal(1.0, sigma)

                for col in numeric_cols:
                    if col in config.get("time_warping", {}).get("exclude_columns", []):
                        continue

                    # Appliquer un facteur d'échelle aléatoire à la fenêtre
                    df.iloc[window_slice][col] *= warp_factor

        # 3. Permutation de fenêtres
        if config.get("window_permutation", {}).get("enabled", False):
            window_size = config["window_permutation"].get("window_size", 5)
            n_permutations = config["window_permutation"].get("n_permutations", 1)

            for _ in range(n_permutations):
                if len(df) > 2 * window_size:
                    start = np.random.randint(0, len(df) - 2 * window_size)
                    window1 = slice(start, start + window_size)
                    window2 = slice(start + window_size, start + 2 * window_size)

                    for col in numeric_cols:
                        if col in config.get("window_permutation", {}).get(
                            "exclude_columns", []
                        ):
                            continue

                        # Échanger les deux fenêtres
                        temp = df[col].iloc[window1].copy()
                        df[col].iloc[window1] = df[col].iloc[window2].values
                        df[col].iloc[window2] = temp.values

        # 4. Scaling aléatoire
        if config.get("random_scaling", {}).get("enabled", False):
            scale_range = config["random_scaling"].get("scale_range", [0.9, 1.1])
            for col in numeric_cols:
                if col in config.get("random_scaling", {}).get("exclude_columns", []):
                    continue

                scale = np.random.uniform(scale_range[0], scale_range[1])
                df[col] = df[col] * scale

        # 5. Ajout de tendances
        if config.get("trend_augmentation", {}).get("enabled", False):
            max_trend = config["trend_augmentation"].get("max_trend", 0.01)
            for col in numeric_cols:
                if col in config.get("trend_augmentation", {}).get(
                    "exclude_columns", []
                ):
                    continue

                trend = np.linspace(
                    0, np.random.uniform(-max_trend, max_trend) * len(df), len(df)
                )
                df[col] = df[col] * (1 + trend)

        # 6. Mélange temporel partiel
        if config.get("partial_shuffle", {}).get("enabled", False):
            segment_size = config["partial_shuffle"].get("segment_size", 5)
            for col in numeric_cols:
                if col in config.get("partial_shuffle", {}).get("exclude_columns", []):
                    continue

                for i in range(0, len(df) - segment_size, segment_size):
                    segment = df[col].iloc[i : i + segment_size]
                    if (
                        len(segment) == segment_size and np.random.random() < 0.3
                    ):  # 30% de chance de mélanger
                        df[col].iloc[i : i + segment_size] = segment.sample(
                            frac=1
                        ).values

        # 7. Ajout d'impulsions aléatoires
        if config.get("random_impulses", {}).get("enabled", False):
            impulse_prob = config["random_impulses"].get("probability", 0.01)
            max_impulse = config["random_impulses"].get("max_impulse", 0.1)

            for col in numeric_cols:
                if col in config.get("random_impulses", {}).get("exclude_columns", []):
                    continue

                for i in range(len(df)):
                    if np.random.random() < impulse_prob:
                        impulse = (
                            np.random.uniform(-max_impulse, max_impulse) * df[col].std()
                        )
                        df[col].iloc[i] += impulse

        return df

    def transform_with_cached_scaler(
        self, timeframe: str, window_data: pd.DataFrame, requested_features: list
    ) -> np.ndarray:
        """
        Transforme les données en utilisant un scaler mis en cache ou en ajuste un nouveau si nécessaire.

        Args:
            timeframe: Le timeframe des données (ex: '5m', '1h', '4h')
            window_data: DataFrame contenant les données à transformer
            requested_features: Liste des noms de fonctionnalités dans l'ordre attendu

        Returns:
            Tableau numpy transformé avec le même nombre de lignes que window_data
            et un nombre de colonnes correspondant à requested_features
        """
        # Vérifier si les données d'entrée sont vides
        if window_data.empty:
            logger.warning(
                "Données vides reçues pour le timeframe %s, retour de zéros", timeframe
            )
            return np.zeros((0, len(requested_features)), dtype=np.float32)

        # Créer une copie pour éviter les avertissements de modification
        window = window_data.copy()

        # Vérifier que le timeframe est valide
        if timeframe not in self.expected_features:
            logger.error(
                "Timeframe %s non reconnu dans les fonctionnalités attendues", timeframe
            )
            return np.zeros((len(window), len(requested_features)), dtype=np.float32)

        # Récupérer les fonctionnalités attendues pour ce timeframe
        expected_features = self.expected_features[timeframe]

        # Vérifier que les fonctionnalités demandées correspondent aux fonctionnalités attendues
        if set(requested_features) != set(expected_features):
            logger.warning(
                "Les fonctionnalités demandées ne correspondent pas aux fonctionnalités attendues pour %s. "
                "Attendu: %s, Reçu: %s. Utilisation des fonctionnalités attendues.",
                timeframe,
                expected_features,
                requested_features,
            )
            requested_features = expected_features

        # Signature canonique pour le cache (ordre des fonctionnalités important)
        feat_sig = tuple(requested_features)
        key = (timeframe, feat_sig)

        logger.debug(
            "Requête de transformation - Timeframe: %s, Fonctionnalités: %s",
            timeframe,
            feat_sig,
        )

        # Essayer de trouver un scaler mis en cache correspondant
        if key in self.scaler_cache:
            scaler = self.scaler_cache[key]
            cached_features = self.scaler_feature_order[key]
            logger.debug(
                "Utilisation du scaler en cache pour le timeframe %s avec la signature %s",
                timeframe,
                feat_sig,
            )
        else:
            # Aucun scaler en cache trouvé - en ajuster un nouveau
            logger.info(
                "Ajustement d'un nouveau scaler pour le timeframe %s avec %d fonctionnalités",
                timeframe,
                len(requested_features),
            )

            from sklearn.preprocessing import RobustScaler

            scaler = RobustScaler()

            # S'assurer que toutes les fonctionnalités attendues existent
            missing_features = [
                f for f in requested_features if f not in window.columns
            ]
            if missing_features:
                logger.debug(
                    "Ajout des fonctionnalités manquantes pour %s: %s avec des zéros",
                    timeframe,
                    missing_features,
                )
                for f in missing_features:
                    window.loc[:, f] = 0.0

            try:
                # Ajuster le scaler sur les fonctionnalités demandées
                scaler.fit(window[requested_features].values)

                # Mettre en cache le scaler
                self.scaler_cache[key] = scaler
                self.scaler_feature_order[key] = requested_features.copy()
                cached_features = requested_features

                logger.info(
                    "Nouveau scaler ajusté pour le timeframe %s avec %d fonctionnalités",
                    timeframe,
                    len(requested_features),
                )

            except Exception as e:
                logger.error(
                    "Échec de l'ajustement du nouveau scaler pour %s: %s",
                    timeframe,
                    str(e),
                    exc_info=True,
                )
                # Retourner des zéros comme solution de repli
                return np.zeros(
                    (len(window), len(requested_features)), dtype=np.float32
                )

        # S'assurer que toutes les fonctionnalités mises en cache existent dans la fenêtre
        missing_cached_features = [
            f for f in cached_features if f not in window.columns
        ]
        if missing_cached_features:
            logger.debug(
                "Ajout des fonctionnalités manquantes du cache pour %s: %s avec des zéros",
                timeframe,
                missing_cached_features,
            )
            for f in missing_cached_features:
                window.loc[:, f] = 0.0

        # Réorganiser les colonnes selon l'ordre des fonctionnalités mises en cache
        try:
            X = window[list(cached_features)].values.astype(np.float32)

            # Appliquer la transformation
            Xs = scaler.transform(X)

            # Vérifier la forme de sortie
            if Xs.shape[1] != len(cached_features):
                logger.error(
                    "Le nombre de fonctionnalités en sortie (%d) ne correspond pas à celui attendu (%d) pour %s",
                    Xs.shape[1],
                    len(cached_features),
                    timeframe,
                )
                return np.zeros((len(window), len(cached_features)), dtype=np.float32)

            return Xs

        except Exception as e:
            logger.error(
                "Échec de la transformation avec le scaler pour %s: %s",
                timeframe,
                str(e),
                exc_info=True,
            )
            # Solution de repli : retourner des zéros avec la forme correcte
            return np.zeros((len(window), len(cached_features)), dtype=np.float32)

    def _build_asset_timeframe_state(self, asset, timeframe, df: pd.DataFrame):
        """
        Construit l'état pour un actif et un timeframe donnés.
        Utilise automatiquement toutes les colonnes disponibles dans les données.
        """
        # Créer une copie pour éviter les avertissements
        df = df.copy()

        # Utiliser TOUTES les colonnes disponibles dans le DataFrame
        # (excluant seulement timestamp si elle existe)
        available_columns = [col for col in df.columns if col.lower() != "timestamp"]

        # Mettre à jour dynamiquement la configuration pour ce timeframe
        if timeframe not in self.features_config or len(
            self.features_config[timeframe]
        ) != len(available_columns):
            logger.info(
                f"Mise à jour dynamique des features pour {timeframe}: {len(available_columns)} colonnes - {available_columns}"
            )
            self.features_config[timeframe] = available_columns
            self.nb_features_per_tf[timeframe] = len(available_columns)

        # Utiliser les colonnes disponibles directement
        processed_required = available_columns

        # S'assurer que toutes les colonnes sont valides
        missing = [f for f in processed_required if f not in df.columns]
        if missing:
            logger.warning(
                f"Colonnes manquantes dans les données pour {timeframe}: {missing}"
            )
            # Ajouter les colonnes manquantes avec des valeurs nulles
            for col in missing:
                df[col] = 0.0

        # Sélectionner toutes les colonnes disponibles dans l'ordre
        try:
            # Vérifier si toutes les colonnes sont présentes
            missing_cols = [col for col in processed_required if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Colonnes manquantes après traitement: {missing_cols}"
                )

            arr = df[processed_required].to_numpy()
            logger.debug(f"État construit pour {asset} {timeframe}: shape {arr.shape}")
            return arr

        except Exception as e:
            logger.error(
                f"Erreur lors de la construction de l'état pour {asset} {timeframe}: {str(e)}"
            )
            logger.error(f"Colonnes disponibles: {df.columns.tolist()}")
            logger.error(f"Colonnes requises: {processed_required}")
            raise

    def align_timeframe_dims(
        self, obs_by_tf: Optional[Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """
        Assure que toutes les observations de timeframe ont le même nombre de fonctionnalités
        en ajoutant des zéros ou en tronquant si nécessaire.

        Cette méthode gère :
        - Les entrées manquantes ou invalides
        - Les dimensions de fonctionnalités variables entre les timeframes
        - Les formes de tableaux incohérentes
        - La conversion de type en float32
        - La gestion complète des erreurs et la journalisation

        Args:
            obs_by_tf: Dictionnaire associant les timeframes à leurs tableaux d'observations.
                      Peut être None ou vide.

        Returns:
            Tableau numpy 3D de forme (n_timeframes, window_size, max_features)
            Retourne des zéros si l'entrée est invalide ou vide.
        """
        # Suivi si nous utilisons un scénario de repli
        fallback_used = False

        # Valeur de retour vide par défaut avec les bonnes dimensions
        empty_return = np.zeros((0, self.window_size, 0), dtype=np.float32)

        # Gestion des entrées None ou vides
        if not obs_by_tf or not isinstance(obs_by_tf, dict):
            logger.warning(
                "Aucune observation de timeframe valide fournie ou l'entrée n'est pas un dictionnaire"
            )
            return empty_return

        # Journalisation des statistiques d'entrée
        logger.debug("Traitement de %d observations de timeframe", len(obs_by_tf))
        for tf, arr in obs_by_tf.items():
            if arr is not None and hasattr(arr, "shape"):
                logger.debug(
                    "  %s: shape=%s, dtype=%s, min=%s, max=%s",
                    tf,
                    arr.shape,
                    getattr(arr, "dtype", "inconnu"),
                    np.min(arr) if arr.size > 0 else "N/A",
                    np.max(arr) if arr.size > 0 else "N/A",
                )

        # Filtrer les tableaux None ou invalides, convertir au format cohérent
        valid_obs = {}
        for tf, arr in obs_by_tf.items():
            if arr is None or not isinstance(arr, np.ndarray) or arr.size == 0:
                # Utiliser les fonctionnalités attendues pour ce timeframe si disponible
                expected_features = self.expected_features.get(tf, ["close"])
                logger.warning(
                    "Tableau d'observation invalide pour %s - utilisation de zéros avec %d fonctionnalités",
                    tf,
                    len(expected_features),
                )
                valid_obs[tf] = np.zeros(
                    (self.window_size, len(expected_features)), dtype=np.float32
                )
                fallback_used = True
            else:
                # S'assurer d'avoir une copie pour éviter de modifier l'entrée
                valid_obs[tf] = np.asarray(arr, dtype=np.float32).copy()

                # Vérifier les valeurs NaN ou infinies
                if np.any(~np.isfinite(valid_obs[tf])):
                    logger.warning(
                        "Valeurs non finies trouvées dans les observations de %s - remplacement par des zéros",
                        tf,
                    )
                    valid_obs[tf][~np.isfinite(valid_obs[tf])] = 0.0
                    fallback_used = True

        if not valid_obs:
            logger.warning("Aucune observation valide après filtrage")
            return empty_return

        # S'assurer que tous les tableaux sont en 2D (window_size, n_features)
        for tf in list(valid_obs.keys()):
            arr = valid_obs[tf]
            try:
                # Gérer les tableaux 1D en les transformant en 2D
                if arr.ndim == 1:
                    logger.debug("Redimensionnement du tableau 1D pour %s en 2D", tf)
                    valid_obs[tf] = arr.reshape(-1, 1)
                # Aplatir les tableaux de dimension supérieure à 2
                elif arr.ndim > 2:
                    logger.warning(
                        "Aplatissement de l'observation %s de %dD à 2D", tf, arr.ndim
                    )
                    valid_obs[tf] = arr.reshape(arr.shape[0], -1)

                # S'assurer que la taille de la fenêtre correspond
                if valid_obs[tf].shape[0] != self.window_size:
                    logger.warning(
                        "Incompatibilité de taille de fenêtre pour %s: attendu %d, reçu %d",
                        tf,
                        self.window_size,
                        valid_obs[tf].shape[0],
                    )
                    # Tronquer ou compléter pour correspondre à window_size
                    if valid_obs[tf].shape[0] > self.window_size:
                        valid_obs[tf] = valid_obs[tf][-self.window_size :]
                    else:
                        pad = ((self.window_size - valid_obs[tf].shape[0], 0), (0, 0))
                        valid_obs[tf] = np.pad(
                            valid_obs[tf], pad, "constant", constant_values=0.0
                        )
                    fallback_used = True

            except Exception as e:
                logger.error("Erreur lors du traitement du tableau %s: %s", tf, str(e))
                # Utiliser les fonctionnalités attendues pour ce timeframe si disponible
                expected_features = self.expected_features.get(tf, ["close"])
                valid_obs[tf] = np.zeros(
                    (self.window_size, len(expected_features)), dtype=np.float32
                )
                fallback_used = True

        # Trouver le nombre maximum de fonctionnalités parmi tous les timeframes
        try:
            max_f = max(arr.shape[1] for arr in valid_obs.values())
            if max_f == 0:
                logger.warning("Aucune fonctionnalité trouvée dans aucun timeframe")
                return empty_return

            logger.debug(
                "Nombre maximum de fonctionnalités à travers les timeframes: %d", max_f
            )
            aligned = []
            timeframes_processed = []

            # Traiter chaque timeframe dans un ordre cohérent
            for tf in sorted(valid_obs.keys()):
                try:
                    arr = valid_obs[tf]
                    f = arr.shape[1]

                    # Obtenir les fonctionnalités attendues pour ce timeframe
                    expected_features = self.expected_features.get(tf, [])
                    expected_f = len(expected_features) if expected_features else max_f

                    # Utiliser le maximum entre le nombre de fonctionnalités attendues et le maximum observé
                    target_f = max(max_f, expected_f) if expected_features else max_f

                    # Créer un nouveau tableau avec exactement target_features colonnes
                    new_arr = np.zeros((arr.shape[0], target_f), dtype=np.float32)

                    # Copier les données existantes
                    copy_cols = min(f, target_f)
                    new_arr[:, :copy_cols] = arr[:, :copy_cols]

                    if f != target_f:
                        logger.warning(
                            "Ajustement de %s de %d à %d fonctionnalités",
                            tf,
                            f,
                            target_f,
                        )
                        fallback_used = True

                    aligned.append(new_arr)

                    timeframes_processed.append(tf)

                except Exception as e:
                    logger.error(
                        "Erreur lors du traitement de %s: %s", tf, str(e), exc_info=True
                    )
                    # Solution de repli : utiliser des zéros avec les dimensions correctes
                    fallback = np.zeros((self.window_size, max_f), dtype=np.float32)
                    aligned.append(fallback)
                    timeframes_processed.append(f"{tf} (erreur)")
                    fallback_used = True

            if not aligned:
                logger.warning("Aucune observation valide après alignement")
                return empty_return

            # Empiler tous les timeframes le long de la première dimension
            try:
                # Log des dimensions avant empilement
                for i, arr in enumerate(aligned):
                    logger.debug(
                        f"Avant empilement - Observation {i}: shape={arr.shape}, dtype={arr.dtype}"
                    )

                result = np.stack(aligned, axis=0)
                logger.debug(
                    f"Après empilement - Résultat: shape={result.shape}, dtype={result.dtype}"
                )

                # Validation finale de la forme de sortie
                if (
                    result.shape[0] != len(valid_obs)
                    or result.shape[1] != self.window_size
                ):
                    logger.error(
                        "Forme de sortie inattendue: %s, attendu (%d, %d, %d)",
                        result.shape,
                        len(valid_obs),
                        self.window_size,
                        max_f,
                    )
                    return empty_return

                # Journalisation finale
                if fallback_used:
                    logger.warning(
                        "Solution de repli utilisée pendant l'alignement. "
                        "Timeframes traités: %s",
                        ", ".join(timeframes_processed),
                    )
                else:
                    logger.debug(
                        "Alignement réussi de %d timeframes vers la forme %s",
                        len(timeframes_processed),
                        result.shape,
                    )

                return result

            except Exception as e:
                logger.error(
                    "Erreur lors de l'empilement des observations: %s",
                    str(e),
                    exc_info=True,
                )
                return empty_return

        except Exception as e:
            logger.error(
                f"Critical error in align_timeframe_dims: {str(e)}", exc_info=True
            )
            return empty_return

    def _wrap_observation(self, obs_candidate, portfolio_candidate=None):
        """
        Normalize the build_observation output to:
          {'observation': ndarray (float32), 'portfolio_state': ndarray shape (17,) (float32)}
        Works for obs_candidate being ndarray, list, dict, None.
        Includes extreme value protection.
        """
        # If a dict is already provided, try to standardize it
        if isinstance(obs_candidate, dict):
            obs = obs_candidate.get("observation", obs_candidate)
            ps = obs_candidate.get("portfolio_state", portfolio_candidate)
        else:
            obs = obs_candidate
            ps = portfolio_candidate

        # Normalize observation array with extreme value protection
        try:
            logger.debug(f"Avant conversion - Type: {type(obs)}")

            # Apply extreme value clipping before normalization
            if isinstance(obs, np.ndarray) and obs.size > 0:
                max_abs_val = np.abs(obs).max()
                if max_abs_val > 10000:  # Detect extreme values like 802654
                    logger.warning(
                        f"[EXTREME VALUES] Detected max={max_abs_val:.1f}, applying aggressive clipping"
                    )
                    obs = np.clip(obs, -1000.0, 1000.0)  # First stage clipping
                    # Then normalize to [-3, 3] range
                    obs_std = np.std(obs)
                    if obs_std > 0:
                        obs = obs / (obs_std / 3.0)
                    obs = np.clip(obs, -3.0, 3.0)  # Final clipping
                    logger.info(
                        f"[EXTREME VALUES] Fixed, new max: {np.abs(obs).max():.4f}"
                    )
            if hasattr(obs, "shape"):
                logger.debug(f"  Shape: {obs.shape}")
            obs_arr = np.asarray(obs, dtype=np.float32)
            logger.debug(
                f"Après conversion - Type: {type(obs_arr)}, Shape: {obs_arr.shape}"
            )
        except Exception:
            logger.warning(
                "_wrap_observation: could not convert observation to ndarray, using zeros"
            )
            # Fallback shape guess: use attributes if available
            default_shape = getattr(self, "observation_shape", None)
            if default_shape is None:
                # best effort: (n_timeframes, window_size, n_features)
                n_tf = len(getattr(self, "timeframes", [0]))
                ws = getattr(self, "window_size", 20)
                default_shape = (n_tf, ws, 1)
            obs_arr = np.zeros(default_shape, dtype=np.float32)

        # Normalize portfolio state to length 17
        try:
            ps_arr = np.asarray(ps, dtype=np.float32).flatten()
            if ps_arr.size < 17:
                ps_arr = np.concatenate(
                    [ps_arr, np.zeros(17 - ps_arr.size, dtype=np.float32)]
                )
            elif ps_arr.size > 17:
                ps_arr = ps_arr[:17]
        except Exception:
            ps_arr = np.zeros(17, dtype=np.float32)

        result = {"observation": obs_arr, "portfolio_state": ps_arr}
        # debug log
        logger.debug(
            "_wrap_observation -> obs type=%s shape=%s dtype=%s; ps shape=%s",
            type(result["observation"]),
            result["observation"].shape,
            result["observation"].dtype,
            result["portfolio_state"].shape,
        )
        return result

    def _expand_composite_indicators(self, indicators: List[str]) -> List[str]:
        """
        Étend les indicateurs composites en leurs composants individuels.

        Par exemple, 'STOCH_14_3' devient ['STOCHk_14_3', 'STOCHd_14_3']

        Args:
            indicators: Liste des indicateurs à étendre

        Returns:
            Liste des indicateurs avec les composites développés
        """
        expanded = []
        for indicator in indicators:
            if indicator in self.composite_indicators:
                expanded.extend(self.composite_indicators[indicator])
            else:
                expanded.append(indicator)
        return expanded

    def _process_timeframe_data(
        self,
        data: Dict[str, Dict[str, pd.DataFrame]],
        tf: str,
        current_idx: int,
        max_features: int,
        verbose_log: bool,
    ) -> Tuple[str, np.ndarray]:
        """
        Traite les données pour un timeframe spécifique et retourne un tableau numpy
        avec la forme (window_size, n_features=15).

        Le timestamp est préservé pour le suivi temporel mais n'est pas inclus dans les features.
        Seules les colonnes spécifiées dans self.expected_features sont conservées.

        Args:
            data: Dictionnaire contenant les données de marché pour tous les actifs et timeframes
            tf: Timeframe à traiter (ex: '5m', '1h', '4h')
            current_idx: Index actuel dans les données
            max_features: Nombre maximum de caractéristiques à travers tous les timeframes (non utilisé ici)
            verbose_log: Booléen pour activer/désactiver les logs détaillés.

        Returns:
            Tuple de (timeframe, processed_data) où processed_data a la forme (window_size, 15)
        """
        if verbose_log:
            logger.info(f"\n=== Traitement du timeframe {tf} ===")
            logger.info(
                f"Index courant: {current_idx}, Taille max des caractéristiques: {max_features}"
            )

        # Obtenir les caractéristiques attendues pour ce timeframe
        expected_features = self.expected_features.get(tf, [])

        # Développer les indicateurs composites
        expected_features = self._expand_composite_indicators(expected_features)

        if verbose_log:
            logger.info(
                f"Caractéristiques attendues pour {tf} ({len(expected_features)}): {expected_features}"
            )

        def create_default_observation():
            """Crée une observation par défaut avec la forme attendue."""
            shape = (self.window_size, len(expected_features))
            logger.warning(
                f"Création d'une observation par défaut avec la forme: {shape}"
            )
            return np.zeros(shape, dtype=np.float32)

        # Valider la configuration du timeframe
        if tf not in self.timeframes or not expected_features:
            logger.error(
                "Configuration invalide pour le timeframe %s. Vérifiez votre configuration.",
                tf,
            )
            return tf, create_default_observation()

        try:
            # 1. Obtenir les données pour tous les actifs pour ce timeframe
            asset_dfs = []
            for asset in data:
                if tf in data[asset] and not data[asset][tf].empty:
                    # Copier les données sans modifier la casse des colonnes
                    df = data[asset][tf].copy()
                    asset_dfs.append(df)

            if not asset_dfs:
                logger.warning(
                    "Aucune donnée trouvée pour le timeframe %s, utilisation de zéros",
                    tf,
                )
                return tf, create_default_observation()

            # 2. Concaténer les données de tous les actifs
            try:
                df = pd.concat(asset_dfs, axis=0)

                # Journalisation des informations de débogage
                logger.debug(
                    "Traitement des données %s. Forme: %s, colonnes: %s",
                    tf,
                    df.shape,
                    df.columns.tolist(),
                )

                # 3. Identifier les colonnes manquantes par rapport aux fonctionnalités attendues
                # Comparaison insensible à la casse
                df_columns_lower = {str(col).lower(): str(col) for col in df.columns}
                expected_features_lower = {
                    feat.lower(): feat for feat in expected_features
                }

                # Identifier les colonnes disponibles et manquantes
                available_lower = set(expected_features_lower.keys()) & set(
                    df_columns_lower.keys()
                )
                missing_lower = set(expected_features_lower.keys()) - set(
                    df_columns_lower.keys()
                )

                # Convertir en noms originaux pour les logs
                available_cols = {
                    expected_features_lower[col_lower] for col_lower in available_lower
                }
                missing_cols = {
                    expected_features_lower[col_lower] for col_lower in missing_lower
                }

                # Calculer les indicateurs techniques manquants au lieu de les remplir par des zéros
                if missing_cols:
                    logger.info(
                        "Colonnes manquantes dans les données %s: %s. Calcul des indicateurs techniques...",
                        tf,
                        missing_cols,
                    )
                    logger.debug("Colonnes disponibles: %s", available_cols)

                    # Calculer les indicateurs techniques manquants
                    df = _calculate_technical_indicators(df)

                    # Mettre à jour les colonnes disponibles après calcul
                    df_columns_upper = [str(col).upper() for col in df.columns]
                    df_columns_set = set(df_columns_upper)
                    missing_cols = set(expected_features) - df_columns_set
                    available_cols = set(expected_features) & df_columns_set

                    if missing_cols:
                        logger.warning(
                            "Colonnes encore manquantes après calcul des indicateurs %s: %s. Remplissage par des zéros.",
                            tf,
                            missing_cols,
                        )
                    else:
                        logger.info(
                            "Tous les indicateurs techniques calculés avec succès pour %s",
                            tf,
                        )

                # 4. Exclure le timestamp des features (gestion de la casse)
                timestamp_col = next(
                    (col for col in df.columns if col.lower() == "timestamp"), None
                )
                if timestamp_col is not None:
                    timestamps = df[timestamp_col].copy()
                    df = df.drop(columns=[timestamp_col])
                    logger.debug(
                        "Timestamp extrait et retiré des features (colonne: %s)",
                        timestamp_col,
                    )
                else:
                    if verbose_log:
                        logger.debug(
                            "Timestamp utilisé depuis l'index DatetimeIndex (pas de colonne timestamp séparée)"
                        )

                # 5. Créer un nouveau DataFrame avec les colonnes dans l'ordre attendu
                # et remplir avec des zéros les colonnes manquantes
                if verbose_log:
                    logger.info(
                        "Création du DataFrame traité avec les colonnes attendues"
                    )
                    logger.info("Colonnes attendues: %s", expected_features)
                    logger.info(
                        "Colonnes disponibles dans les données: %s", df.columns.tolist()
                    )

                processed_data = pd.DataFrame(index=df.index)

                # Liste pour suivre les colonnes ajoutées
                added_columns = []

                for col in expected_features:
                    # Vérifier si la colonne existe en utilisant le mapping insensible à la casse
                    col_lower = col.lower()

                    if col_lower in df_columns_lower:
                        # Utiliser le nom de colonne original du DataFrame
                        original_col = df_columns_lower[col_lower]
                        # Copier les données avec conversion en float32
                        processed_data[col] = df[original_col].astype(np.float32)
                        added_columns.append(col)

                        # Vérifier les valeurs manquantes ou infinies
                        if (
                            processed_data[col].isna().any()
                            or np.isinf(processed_data[col]).any()
                        ):
                            logger.warning(
                                "Valeurs manquantes ou infinies détectées dans la colonne %s. Remplacement par des zéros.",
                                col,
                            )
                            processed_data[col] = (
                                processed_data[col]
                                .fillna(0.0)
                                .replace([np.inf, -np.inf], 0.0)
                            )
                    else:
                        # Créer une colonne de zéros
                        processed_data[col] = 0.0
                        added_columns.append(f"{col} (ajoutée)")
                        logger.debug(
                            "Colonne %s manquante, remplacée par des zéros", col
                        )

                # 5. Vérification finale des dimensions
                if verbose_log:
                    logger.info(
                        "Colonnes ajoutées au DataFrame traité: %s", added_columns
                    )
                    logger.info(
                        "Nombre de colonnes dans le DataFrame traité: %d",
                        len(processed_data.columns),
                    )
                    logger.info(
                        "Colonnes actuelles: %s", processed_data.columns.tolist()
                    )

                # --- DEBUG LOGGING ---
                logger.info(
                    f"[STATE_BUILDER_DEBUG] _process_timeframe_data - Before slicing for {tf}: processed_data length: {len(processed_data)}"
                )
                # --- END DEBUG LOGGING ---

                # 6. S'assurer que nous avons suffisamment de données
                if len(processed_data) < self.window_size:
                    logger.warning(
                        "Pas assez de points de données pour %s: %d < %d. Remplissage avec des zéros.",
                        tf,
                        len(processed_data),
                        self.window_size,
                    )

                    if len(processed_data) == 0:
                        logger.warning(
                            "Aucune donnée disponible pour %s, création d'une observation par défaut",
                            tf,
                        )
                        return tf, create_default_observation()

                    # Répéter les données disponibles pour remplir la fenêtre
                    repeats = (self.window_size // len(processed_data)) + 1
                    processed_data = pd.concat(
                        [processed_data] * repeats, axis=0
                    ).reset_index(drop=True)
                    processed_data = processed_data.head(self.window_size)

                # 7. Sélectionner la fenêtre de données
                window_data = processed_data.iloc[-self.window_size :].copy()
                result = window_data.values.astype(np.float32)

                # --- DEBUG LOGGING ---
                logger.info(
                    f"[STATE_BUILDER_DEBUG] _process_timeframe_data - After slicing for {tf}: window_data length: {len(window_data)}"
                )
                # --- END DEBUG LOGGING ---

                # 8. Validation finale de la forme du résultat
                expected_shape = (self.window_size, len(expected_features))
                if result.shape != expected_shape:
                    logger.warning(
                        "Forme inattendue pour %s: %s. Redimensionnement à %s.",
                        tf,
                        result.shape,
                        expected_shape,
                    )

                    # Journalisation pour le débogage
                    logger.debug(
                        "Forme actuelle: %s, Forme attendue: %s",
                        result.shape,
                        expected_shape,
                    )

                    # Créer un nouveau tableau avec la forme correcte
                    new_result = np.zeros(expected_shape, dtype=np.float32)

                    # Copier autant de données que possible
                    min_rows = min(result.shape[0], expected_shape[0])
                    min_cols = min(result.shape[1], expected_shape[1])

                    new_result[:min_rows, :min_cols] = result[:min_rows, :min_cols]
                    result = new_result

                if verbose_log:
                    logger.info(
                        "Observation traitée pour %s: shape=%s, type=%s",
                        tf,
                        result.shape,
                        type(result).__name__,
                    )
                return tf, result

            except Exception as e:
                logger.error(
                    "Error processing data for %s: %s. Using default values.",
                    tf,
                    str(e),
                )
                logger.error(traceback.format_exc())
                return tf, create_default_observation()

        except Exception as e:
            logger.error(
                "Unexpected error processing timeframe %s: %s. Using default values.",
                tf,
                str(e),
            )
            logger.error(traceback.format_exc())
            return tf, create_default_observation()

    def build_observation(
        self, current_idx: int, data: Dict[str, Dict[str, pd.DataFrame]]
    ) -> Dict[str, np.ndarray]:
        """
        Construit et retourne l'observation finale sous forme de dictionnaire,
        conforme à ce qui est attendu par l'environnement.

        Args:
            current_idx: Index actuel dans les données
            data: Dictionnaire des données par actif et par timeframe

        Returns:
            Dictionnaire contenant:
            - 'observation': Tableau numpy des observations (n_timeframes, window_size, n_features)
            - 'portfolio_state': État du portefeuille (17,)
        """
        asset_name = next(iter(data), "UNKNOWN_ASSET")
        if not self._verbose_logging_done:
            logger.info("\n=== Construction de l'observation (premier appel) ===")
            logger.info("Index courant: %d", current_idx)
            logger.info(
                "Timeframes disponibles: %s", list(data.get(asset_name, {}).keys())
            )
        else:
            logger.info(f"Construction de l'observation pour l'actif {asset_name} - OK")

        try:
            # Initialiser le dictionnaire des observations
            observations = {}

            # Déterminer le nombre maximal de fonctionnalités à travers tous les timeframes
            max_features = 0
            if self.features_config:
                max_features = max(
                    len(feats) for feats in self.features_config.values()
                )
                if not self._verbose_logging_done:
                    logger.info(
                        "Nombre maximum de fonctionnalités configurées: %d",
                        max_features,
                    )

                    # Afficher les fonctionnalités configurées pour chaque timeframe
                    logger.info("Configuration des fonctionnalités par timeframe:")
                    for tf, feats in self.features_config.items():
                        logger.info(
                            "  - %s: %d fonctionnalités - %s", tf, len(feats), feats
                        )
            else:
                logger.warning(
                    "Aucune configuration de fonctionnalités trouvée, utilisation de 0 comme valeur par défaut"
                )

            # Traiter chaque timeframe
            for tf in self.timeframes:
                try:
                    if not self._verbose_logging_done:
                        logger.info("\n--- Traitement du timeframe: %s ---", tf)

                    tf_result = self._process_timeframe_data(
                        data,
                        tf,
                        current_idx,
                        max_features,
                        verbose_log=(not self._verbose_logging_done),
                    )

                    if tf_result is not None:  # Vérification explicite de None
                        tf_key, obs = tf_result
                        observations[tf_key] = obs
                        if not self._verbose_logging_done:
                            logger.info(
                                "Observation traitée pour %s: shape=%s, type=%s",
                                tf_key,
                                obs.shape if hasattr(obs, "shape") else "invalide",
                                type(obs).__name__,
                            )
                    else:
                        logger.warning("Résultat nul pour le timeframe %s, ignoré", tf)
                except Exception as e:
                    logger.error(
                        "Erreur lors du traitement du timeframe %s: %s",
                        tf,
                        str(e),
                        exc_info=True,
                    )

            if not self._verbose_logging_done:
                logger.info("\n=== Alignement des observations ===")
                logger.info("Timeframes à aligner: %s", list(observations.keys()))

            # Aligner les dimensions des observations
            aligned_obs = self.align_timeframe_dims(observations)

            # Vérifier que aligned_obs est valide
            if aligned_obs is None or not isinstance(aligned_obs, np.ndarray):
                logger.warning(
                    "Observation alignée invalide, utilisation d'un tableau de zéros"
                )
                aligned_obs = np.zeros(self.observation_shape, dtype=np.float32)

            if not self._verbose_logging_done:
                # Vérifier la forme de l'observation alignée
                logger.info("Forme de l'observation alignée: %s", aligned_obs.shape)
                logger.info("Forme attendue: %s", self.observation_shape)

            if aligned_obs.shape != self.observation_shape:
                logger.error(
                    "ERREUR: La forme de l'observation alignée (%s) ne correspond pas à la forme attendue (%s)",
                    aligned_obs.shape,
                    self.observation_shape,
                )
                logger.info(
                    "Redimensionnement de l'observation pour correspondre à la forme attendue"
                )

                # Créer un nouveau tableau avec la forme attendue
                new_obs = np.zeros(self.observation_shape, dtype=np.float32)

                # Copier les données disponibles
                min_timeframes = min(aligned_obs.shape[0], self.observation_shape[0])
                min_steps = min(aligned_obs.shape[1], self.observation_shape[1])
                min_features = min(aligned_obs.shape[2], self.observation_shape[2])

                new_obs[:min_timeframes, :min_steps, :min_features] = aligned_obs[
                    :min_timeframes, :min_steps, :min_features
                ]

                aligned_obs = new_obs
                logger.info(
                    "Nouvelle forme de l'observation après redimensionnement: %s",
                    aligned_obs.shape,
                )

            # Créer l'état du portefeuille par défaut
            portfolio_state = np.zeros(17, dtype=np.float32)
            if not isinstance(portfolio_state, np.ndarray):
                logger.warning(
                    "État du portefeuille invalide, utilisation d'un tableau de zéros"
                )
                portfolio_state = np.zeros(17, dtype=np.float32)

            # Créer l'observation finale avec vérification de type
            final_observation = {
                "observation": aligned_obs.astype(np.float32),
                "portfolio_state": portfolio_state.astype(np.float32),
            }

            # Vérification finale de la forme de l'observation
            obs_shape = final_observation["observation"].shape
            if not self._verbose_logging_done:
                logger.info("\n=== Vérification finale de l'observation ===")
                logger.info("Forme de l'observation: %s", obs_shape)
                logger.info(
                    "Type de l'observation: %s", type(final_observation["observation"])
                )
                logger.info(
                    "Forme de l'état du portefeuille: %s",
                    final_observation["portfolio_state"].shape,
                )

            if len(obs_shape) != 3:
                logger.error(
                    "ERREUR: L'observation doit avoir 3 dimensions, mais a %d",
                    len(obs_shape),
                )

            expected_features = (
                self.observation_shape[2] if len(self.observation_shape) > 2 else 0
            )
            if obs_shape[2] != expected_features:
                logger.error(
                    "ERREUR: Nombre inattendu de caractéristiques: %d (attendu: %d)",
                    obs_shape[2],
                    expected_features,
                )

            # --- DEBUG LOGGING ---
            obs_shape = final_observation["observation"].shape
            logger.info(
                f"[STATE_BUILDER_DEBUG] build_observation - Final observation shape: {obs_shape}, size: {final_observation['observation'].size}, NaNs: {np.isnan(final_observation['observation']).sum()}"
            )
            # --- END DEBUG LOGGING ---

            # Journaliser la forme de l'observation finale
            obs_shape = final_observation["observation"].shape
            logger.debug(
                "Observation finale construite: shape=%s, type=%s",
                obs_shape,
                type(final_observation["observation"]),
            )

            # Set the flag to True after the first successful run
            self._verbose_logging_done = True
            return final_observation

        except Exception as e:
            logger.error(
                "Erreur critique dans build_observation: %s", str(e), exc_info=True
            )
            # En cas d'erreur, retourner une observation par défaut valide pour éviter un crash
            default_obs = {
                "observation": np.zeros(self.observation_shape, dtype=np.float32),
                "portfolio_state": np.zeros(17, dtype=np.float32),
            }
            logger.warning(
                "Utilisation de l'observation par défaut en raison d'une erreur"
            )
            return default_obs

    def process_dataframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Traite un DataFrame avec des techniques avancées : GARCH pour la volatilité
        et filtre de Kalman pour lisser les indicateurs.

        Args:
            df: DataFrame d'entrée avec les données OHLCV et indicateurs
            timeframe: Timeframe des données ('5m', '1h', '4h')

        Returns:
            DataFrame avec des colonnes supplémentaires:
            - GARCH_VOL: Volatilité conditionnelle GARCH
            - RSI_14_SMOOTH: RSI lissé par filtre de Kalman
            - MACD_HIST_SMOOTH: MACD histogramme lissé
        """
        if df.empty or len(df) < 50:  # Minimum de données requis
            logger.warning(
                f"Données insuffisantes pour le traitement avancé ({len(df)} lignes)"
            )
            return df

        df_processed = df.copy()

        try:
            # 1. GARCH pour volatilité conditionnelle
            if ARCH_AVAILABLE and "close" in df.columns:
                logger.info(f"Application du modèle GARCH pour {timeframe}")

                # Calculer les rendements
                close_col = "close" if "close" in df.columns else "CLOSE"
                returns = df_processed[close_col].pct_change().dropna()

                if len(returns) >= 30:  # Minimum pour GARCH
                    try:
                        # Multiplier par 100 pour éviter les problèmes de convergence
                        returns_scaled = returns * 100

                        # Modèle GARCH(1,1)
                        model = arch_model(
                            returns_scaled, vol="Garch", p=1, q=1, rescale=False
                        )
                        res = model.fit(disp="off", show_warning=False)

                        # Extraire la volatilité conditionnelle
                        conditional_vol = res.conditional_volatility / 100  # Rescaler

                        # Aligner avec l'index original
                        vol_series = pd.Series(0.0, index=df_processed.index)
                        vol_series.iloc[1 : len(conditional_vol) + 1] = (
                            conditional_vol.values
                        )

                        df_processed["GARCH_VOL"] = vol_series.astype(np.float32)

                        logger.info(
                            f"GARCH volatilité calculée - Min: {conditional_vol.min():.6f}, "
                            f"Max: {conditional_vol.max():.6f}, Moyenne: {conditional_vol.mean():.6f}"
                        )

                    except Exception as e:
                        logger.warning(f"Erreur GARCH pour {timeframe}: {str(e)}")
                        # Utiliser l'ATR comme substitut
                        if "high" in df.columns and "low" in df.columns:
                            atr = self._calculate_atr(df_processed)
                            df_processed["GARCH_VOL"] = (
                                atr / df_processed[close_col]
                            ).astype(np.float32)
                        else:
                            df_processed["GARCH_VOL"] = (
                                returns.rolling(14)
                                .std()
                                .fillna(0.01)
                                .astype(np.float32)
                            )
                else:
                    logger.warning(
                        f"Pas assez de données pour GARCH ({len(returns)} rendements)"
                    )
                    df_processed["GARCH_VOL"] = pd.Series(
                        0.01, index=df_processed.index, dtype=np.float32
                    )
            else:
                logger.info(
                    "GARCH non disponible, utilisation de la volatilité des rendements"
                )
                if "close" in df.columns or "CLOSE" in df.columns:
                    close_col = "close" if "close" in df.columns else "CLOSE"
                    returns = df_processed[close_col].pct_change()
                    df_processed["GARCH_VOL"] = (
                        returns.rolling(14).std().fillna(0.01).astype(np.float32)
                    )
                else:
                    df_processed["GARCH_VOL"] = pd.Series(
                        0.01, index=df_processed.index, dtype=np.float32
                    )

            # 2. Filtre de Kalman pour lisser les indicateurs
            if KALMAN_AVAILABLE:
                logger.info(f"Application du filtre de Kalman pour {timeframe}")

                # Lisser RSI_14
                rsi_cols = [col for col in df_processed.columns if "RSI" in col.upper()]
                if rsi_cols:
                    rsi_col = rsi_cols[0]  # Prendre le premier RSI trouvé
                    try:
                        smoothed_rsi = self._apply_kalman_filter(
                            df_processed[rsi_col].values, "RSI"
                        )
                        df_processed["RSI_14_SMOOTH"] = pd.Series(
                            smoothed_rsi, index=df_processed.index, dtype=np.float32
                        )
                        logger.debug(
                            f"RSI lissé - Original std: {df_processed[rsi_col].std():.3f}, "
                            f"Lissé std: {df_processed['RSI_14_SMOOTH'].std():.3f}"
                        )
                    except Exception as e:
                        logger.warning(f"Erreur lissage RSI: {str(e)}")
                        df_processed["RSI_14_SMOOTH"] = df_processed[rsi_col].astype(
                            np.float32
                        )
                else:
                    logger.warning("Aucune colonne RSI trouvée pour le lissage")
                    df_processed["RSI_14_SMOOTH"] = pd.Series(
                        50.0, index=df_processed.index, dtype=np.float32
                    )

                # Lisser MACD Histogramme
                macd_hist_cols = [
                    col for col in df_processed.columns if "MACD_HIST" in col.upper()
                ]
                if macd_hist_cols:
                    macd_col = macd_hist_cols[0]
                    try:
                        smoothed_macd = self._apply_kalman_filter(
                            df_processed[macd_col].values, "MACD_HIST"
                        )
                        df_processed["MACD_HIST_SMOOTH"] = pd.Series(
                            smoothed_macd, index=df_processed.index, dtype=np.float32
                        )
                        logger.debug(
                            f"MACD lissé - Original std: {df_processed[macd_col].std():.6f}, "
                            f"Lissé std: {df_processed['MACD_HIST_SMOOTH'].std():.6f}"
                        )
                    except Exception as e:
                        logger.warning(f"Erreur lissage MACD: {str(e)}")
                        df_processed["MACD_HIST_SMOOTH"] = df_processed[
                            macd_col
                        ].astype(np.float32)
                else:
                    logger.warning("Aucune colonne MACD_HIST trouvée pour le lissage")
                    df_processed["MACD_HIST_SMOOTH"] = pd.Series(
                        0.0, index=df_processed.index, dtype=np.float32
                    )

            else:
                logger.info(
                    "Kalman non disponible, utilisation des moyennes mobiles exponentielles"
                )
                # Substituts avec EMA
                for col in ["RSI_14", "MACD_HIST"]:
                    if col in df_processed.columns:
                        smoothed = df_processed[col].ewm(alpha=0.3).mean()
                        df_processed[f"{col}_SMOOTH"] = smoothed.astype(np.float32)
                    else:
                        default_val = 50.0 if "RSI" in col else 0.0
                        df_processed[f"{col}_SMOOTH"] = pd.Series(
                            default_val, index=df_processed.index, dtype=np.float32
                        )

            # 3. Calcul de l'exposant de Hurst pour la détection de tendance
            if "close" in df.columns or "CLOSE" in df.columns:
                close_col = "close" if "close" in df.columns else "CLOSE"
                hurst_exp = self.calculate_hurst_exponent(
                    df_processed[close_col].values
                )
                df_processed["HURST_EXP"] = pd.Series(
                    hurst_exp, index=df_processed.index, dtype=np.float32
                )

                logger.info(
                    f"Exposant de Hurst calculé pour {timeframe}: {hurst_exp:.4f}"
                )
                if hurst_exp > 0.5:
                    trend_type = "persistante"
                elif hurst_exp < 0.5:
                    trend_type = "anti-persistante (mean-reverting)"
                else:
                    trend_type = "aléatoire"
                logger.info(f"Type de tendance détecté: {trend_type}")

            return df_processed

        except Exception as e:
            logger.error(f"Erreur dans process_dataframe pour {timeframe}: {str(e)}")
            return df

    def _apply_kalman_filter(self, data: np.ndarray, indicator_name: str) -> np.ndarray:
        """
        Applique un filtre de Kalman pour lisser les données d'un indicateur.

        Args:
            data: Données à lisser
            indicator_name: Nom de l'indicateur (pour les logs)

        Returns:
            Données lissées
        """
        if not KALMAN_AVAILABLE or len(data) < 10:
            logger.warning(
                f"Kalman non disponible ou données insuffisantes pour {indicator_name}"
            )
            return data

        try:
            # Configuration du filtre de Kalman
            # Modèle simple : état = [valeur, tendance]
            transition_matrices = np.array([[1, 1], [0, 1]])  # Position et vitesse
            observation_matrices = np.array(
                [[1, 0]]
            )  # On observe seulement la position

            # Ajuster le bruit selon le type d'indicateur
            if "RSI" in indicator_name:
                process_noise = 0.1  # RSI bouge relativement lentement
                observation_noise = 2.0  # Mais peut avoir des pics
            else:  # MACD_HIST
                process_noise = 0.01  # MACD peut changer rapidement
                observation_noise = 0.05  # Moins de bruit d'observation

            kf = KalmanFilter(
                transition_matrices=transition_matrices,
                observation_matrices=observation_matrices,
                initial_state_mean=[data[0], 0],  # Commencer avec la première valeur
                n_dim_state=2,
                n_dim_obs=1,
            )

            # Ajuster les matrices de covariance
            kf = kf.em(data.reshape(-1, 1), n_iter=5)

            # Appliquer le lissage
            state_means, _ = kf.smooth(data.reshape(-1, 1))
            smoothed = state_means[:, 0]  # Extraire seulement la position

            return smoothed.astype(np.float32)

        except Exception as e:
            logger.error(
                f"Erreur dans le filtre de Kalman pour {indicator_name}: {str(e)}"
            )
            # Substitut : moyenne mobile exponentielle
            return pd.Series(data).ewm(alpha=0.3).mean().values.astype(np.float32)

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calcule l'Average True Range comme substitut à GARCH si nécessaire.

        Args:
            df: DataFrame avec colonnes high, low, close
            period: Période pour la moyenne mobile

        Returns:
            Série ATR
        """
        high = df["high"] if "high" in df.columns else df["HIGH"]
        low = df["low"] if "low" in df.columns else df["LOW"]
        close = df["close"] if "close" in df.columns else df["CLOSE"]

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        atr = tr.rolling(period).mean()

        return atr.fillna(method="bfill").fillna(0.01)

    def calculate_hurst_exponent(self, prices: np.ndarray, max_lags: int = 20) -> float:
        """
        Calcule l'exposant de Hurst pour détecter la persistance/anti-persistance des séries.

        L'exposant de Hurst (H) indique :
        - H > 0.5 : Série persistante (tendance à continuer dans la même direction)
        - H < 0.5 : Série anti-persistante (tendance à revenir à la moyenne)
        - H ≈ 0.5 : Série aléatoire (marche aléatoire)

        Args:
            prices: Array des prix (séries temporelles)
            max_lags: Nombre maximum de décalages pour le calcul R/S

        Returns:
            Exposant de Hurst (entre 0 et 1)
        """
        if len(prices) < max_lags * 3:
            logger.warning(
                f"Données insuffisantes pour Hurst ({len(prices)} < {max_lags * 3})"
            )
            return 0.5  # Valeur neutre

        try:
            # Conversion en log-returns
            returns = np.diff(np.log(prices))
            if len(returns) < max_lags * 2:
                return 0.5

            # Calcul R/S pour différentes fenêtres
            lags = np.arange(2, min(max_lags + 1, len(returns) // 2))
            rs_values = []

            for lag in lags:
                # Diviser en sous-séries de taille 'lag'
                rs_per_window = []

                for i in range(0, len(returns) - lag + 1, lag):
                    window = returns[i : i + lag]
                    if len(window) < lag:
                        continue

                    # Calcul de la moyenne et des écarts cumulés
                    mean_return = np.mean(window)
                    cumulative_deviations = np.cumsum(window - mean_return)

                    # R = étendue des écarts cumulés
                    R = np.max(cumulative_deviations) - np.min(cumulative_deviations)

                    # S = écart-type de la fenêtre
                    S = np.std(window)

                    # Éviter la division par zéro
                    if S > 0 and R > 0:
                        rs_per_window.append(R / S)

                if rs_per_window:
                    rs_values.append(np.mean(rs_per_window))
                else:
                    rs_values.append(1.0)  # Valeur par défaut

            if len(rs_values) < 3:
                logger.warning("Pas assez de valeurs R/S calculées")
                return 0.5

            # Régression linéaire : log(R/S) = H * log(n) + c
            log_lags = np.log(lags[: len(rs_values)])
            log_rs = np.log(np.array(rs_values))

            # Filtrer les valeurs invalides
            valid_mask = np.isfinite(log_rs) & np.isfinite(log_lags)
            if np.sum(valid_mask) < 3:
                logger.warning(
                    "Pas assez de valeurs valides pour la régression de Hurst"
                )
                return 0.5

            log_lags = log_lags[valid_mask]
            log_rs = log_rs[valid_mask]

            # Régression linéaire simple
            slope, _ = np.polyfit(log_lags, log_rs, 1)

            # L'exposant de Hurst est la pente
            hurst = float(slope)

            # S'assurer que H est dans l'intervalle [0, 1]
            hurst = max(0.0, min(1.0, hurst))

            return hurst

        except Exception as e:
            logger.error(f"Erreur dans le calcul de l'exposant de Hurst: {str(e)}")
            return 0.5  # Valeur neutre par défaut

    def build_per_timeframe_observation(
        self, current_idx: int, data: Dict[str, Dict[str, pd.DataFrame]]
    ) -> Dict[str, np.ndarray]:
        """
        Builds a dictionary of observations, one for each timeframe, with correct shapes.
        This is the new standard method for creating observations for the multi-CNN model.
        """
        observations = {}
        window_sizes = getattr(self, "window_sizes", {"5m": 20, "1h": 10, "4h": 5})

        # Assume single asset data is passed in the `data` dict for simplicity
        asset_name = next(iter(data))
        asset_data = data[asset_name]

        for tf in self.timeframes:
            try:
                df = asset_data.get(tf)
                if df is None or df.empty:
                    raise ValueError(f"No data for timeframe {tf}")

                features = self.get_feature_names(tf)
                window_size = window_sizes.get(tf, 20)

                # Ensure all feature columns exist, fill with 0 if not
                df_features = pd.DataFrame(columns=features, index=df.index)
                for col in features:
                    if col in df.columns:
                        df_features[col] = df[col]
                    else:
                        df_features[col] = 0.0

                # Get the window of data
                start_idx = max(0, current_idx - window_size + 1)
                end_idx = current_idx + 1

                if end_idx > len(df_features):
                    window_slice = df_features.iloc[-window_size:]
                else:
                    window_slice = df_features.iloc[start_idx:end_idx]

                obs_array = window_slice.values.astype(np.float32)

                if obs_array.shape[0] < window_size:
                    pad_width = ((window_size - obs_array.shape[0], 0), (0, 0))
                    obs_array = np.pad(
                        obs_array, pad_width, mode="constant", constant_values=0.0
                    )

                if self.normalize and self.scalers.get(tf):
                    from sklearn.utils.validation import check_is_fitted
                    from sklearn.exceptions import NotFittedError

                    try:
                        check_is_fitted(self.scalers[tf])
                    except NotFittedError:
                        logger.warning(
                            f"Scaler for timeframe {tf} is not fitted. Fitting on the full available data for this timeframe."
                        )
                        self.scalers[tf].fit(df_features.values)

                    obs_array = self.scalers[tf].transform(obs_array)

                observations[tf] = obs_array

            except Exception as e:
                logger.error(
                    f"Error building observation for timeframe {tf}: {e}", exc_info=True
                )
                window_size = window_sizes.get(tf, 20)
                n_features = len(self.get_feature_names(tf))
                observations[tf] = np.zeros((window_size, n_features), dtype=np.float32)

        return observations
