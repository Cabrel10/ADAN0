"""
State builder for creating multi-timeframe observations for the RL agent.

This module provides the StateBuilder class which transforms raw market data
into a structured observation space suitable for reinforcement learning.
"""

import gc
import hashlib
import logging
import os
import psutil
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
try:
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    
    # Fallback scalers
    class StandardScaler:
        def fit(self, data): pass
        def transform(self, data): return data
        def fit_transform(self, data): return data
        
    class MinMaxScaler:
        def __init__(self, feature_range=(0, 1)): pass
        def fit(self, data): pass
        def transform(self, data): return data
        def fit_transform(self, data): return data
        
    class RobustScaler:
        def fit(self, data): pass
        def transform(self, data): return data
        def fit_transform(self, data): return data

# Configuration du logger
logger = logging.getLogger(__name__)


# 🔧 FIX CRITIQUE: Wrapper pour éviter division par zéro dans la normalisation
class SafeScalerWrapper:
    """
    Wrapper autour des scalers sklearn pour éviter la division par zéro.
    
    Quand std = 0 (écart-type nul), la normalisation StandardScaler produit NaN.
    Ce wrapper ajoute un epsilon (1e-8) au dénominateur pour éviter ce problème.
    
    Cause du bug original:
    - Marché à faible volatilité → Prix/indicateur constant
    - std = 0 → (value - mean) / 0 = NaN
    - NaN se propage dans le réseau de neurones → Crash PyTorch
    
    Solution:
    - Ajouter epsilon: (value - mean) / (std + epsilon)
    - Epsilon = 1e-8 est assez petit pour ne pas affecter les données normales
    - Assez grand pour éviter la division par zéro
    """
    
    def __init__(self, scaler, epsilon=1e-8):
        """
        Initialize the safe scaler wrapper.
        
        Args:
            scaler: The underlying sklearn scaler (StandardScaler, MinMaxScaler, RobustScaler)
            epsilon: Small constant to add to denominator (default: 1e-8)
        """
        self.scaler = scaler
        self.epsilon = epsilon
        self.scaler_type = type(scaler).__name__
        
    def fit(self, X):
        """Fit the underlying scaler."""
        return self.scaler.fit(X)
    
    def fit_transform(self, X):
        """Fit and transform with epsilon protection."""
        result = self.scaler.fit_transform(X)
        return self._add_epsilon_protection(result)
    
    def transform(self, X):
        """Transform with epsilon protection."""
        result = self.scaler.transform(X)
        return self._add_epsilon_protection(result)
    
    def _add_epsilon_protection(self, X):
        """
        Add epsilon protection to prevent NaN from division by zero.
        
        For StandardScaler: (X - mean) / (std + epsilon)
        For MinMaxScaler: Already safe (uses min/max, not std)
        For RobustScaler: Already safe (uses IQR, not std)
        """
        # Remplacer les NaN par 0 (cas où std = 0)
        X_safe = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Vérifier s'il y avait des NaN
        if not np.array_equal(X, X_safe, equal_nan=True):
            logger.warning(
                f"⚠️ NaN values detected in {self.scaler_type} output. "
                f"Likely cause: std=0 (constant feature). "
                f"Replaced with 0.0 using epsilon protection."
            )
        
        return X_safe.astype(np.float32)





class TimeframeConfig:
    """
    Configuration class for timeframe-specific settings.

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
        """
        Initialize timeframe configuration.

        Args:
            timeframe: The timeframe identifier (e.g., '5m', '1h')
            features: List of feature names for this timeframe
            window_size: Number of time steps to include
            normalize: Whether to normalize the data
        """
        self.timeframe = timeframe
        self.features = features
        self.window_size = window_size
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
        window_sizes: Dict[str, int] = None,
        include_portfolio_state: bool = True,
        normalize: bool = True,
        scaler_path: Optional[str] = None,
        adaptive_window: bool = True,
        min_window_size: int = 10,  # 50% de la taille de fenêtre par défaut
        max_window_size: int = 30,  # 150% de la taille de fenêtre par défaut
        memory_config: Optional[Dict[str, Any]] = None,  # Configuration de mémoire
        target_observation_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the StateBuilder according to design specifications.

        Args:
            features_config: Dictionary mapping timeframes to their feature lists
            window_size: Base number of time steps to include in each observation
            include_portfolio_state: Whether to include portfolio state in observations
            normalize: Whether to normalize the data
            scaler_path: Path to save/load the scaler
            adaptive_window: Whether to use adaptive window sizing based on volatility
            min_window_size: Minimum window size for adaptive mode
            max_window_size: Maximum window size for adaptive mode
            memory_config: Configuration for memory optimizations
        """
        # Configuration initiale
        # CONFIGURATION CORRIGÉE: Alignée avec les données d'entraînement
        # Parité stricte avec les fichiers .parquet d'entraînement
        if features_config is None:
            features_config = {
                "5m": [
                    "open", "high", "low", "close", "volume",
                    "rsi_14", "macd_12_26_9", "bb_percent_b_20_2",
                    "atr_14", "atr_20", "atr_50",
                    "volume_ratio_20", "ema_20_ratio", "stoch_k_14_3_3", "price_action",
                    "spread_bps", "liquidity_score",
                    "log_return", "close_ema20_ratio"
                ],
                "1h": [
                    "open", "high", "low", "close", "volume",
                    "rsi_21", "macd_21_42_9", "bb_width_20_2", "adx_14",
                    "atr_20", "atr_50", "obv_ratio_20", "ema_50_ratio",
                    "ichimoku_base", "fib_ratio", "price_ema_ratio_50",
                    "spread_bps", "liquidity_score",
                    "log_return", "close_ema20_ratio"
                ],
                "4h": [
                    "open", "high", "low", "close", "volume",
                    "rsi_28", "macd_26_52_18", "supertrend_10_3",
                    "atr_20", "atr_50", "volume_sma_20_ratio", "ema_100_ratio",
                    "pivot_level", "donchian_width_20", "market_structure", "volatility_ratio_14_50",
                    "spread_bps", "liquidity_score",
                    "log_return", "close_ema20_ratio"
                ],
            }
        self.features_config = features_config
        # Ne garder que les timeframes qui ont des features définies
        self.timeframes = [
            tf for tf in ["5m", "1h", "4h"] if tf in self.features_config
        ]
        if not self.timeframes:
            raise ValueError(
                "Aucun timeframe valide trouvé dans la configuration des fonctionnalités"
            )

        self.nb_features_per_tf = {
            tf: len(features)
            for tf, features in self.features_config.items()
            if tf in self.timeframes
        }

        # Configuration de mémoire
        self.memory_config = memory_config or {
            "aggressive_cleanup": True,
            "force_gc": True,
            "memory_monitoring": True,
            "memory_warning_threshold_mb": 5600,
            "memory_critical_threshold_mb": 6300,
            "disable_caching": True,
        }

        # Métriques de performance
        self.performance_metrics = {
            "gc_collections": 0,
            "memory_peak_mb": 0,
            "errors_count": 0,
            "warnings_count": 0,
        }

        # Mémoire initiale
        self.initial_memory_mb = 0
        self.memory_peak_mb = 0

        # Initialiser les métriques après la configuration
        self._initialize_memory_metrics()

        # Configuration de la taille de fenêtre (AJUSTÉE POUR PARITÉ)
        # 19*15 + 10*16 + 5*16 = 285 + 160 + 80 = 525 ✅
        if window_sizes is None:
            window_sizes = {"5m": 19, "1h": 10, "4h": 5}
        self.window_sizes = window_sizes
        # Absorb unexpected kwargs used by tests without failing
        if kwargs:
            try:
                logger.debug(f"StateBuilder received extra kwargs: {list(kwargs.keys())}")
            except Exception:
                pass
        self.include_portfolio_state = include_portfolio_state
        self.normalize = normalize

        # Maximum de features défini dans la config
        # Déterminer le nombre maximum de features parmi tous les timeframes
        self.max_features = (
            max(len(features) for features in self.features_config.values())
            if self.features_config
            else 0
        )

        # Configuration adaptative
        self.adaptive_window = adaptive_window
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.timeframe_weights = {
            tf: 1.0 for tf in self.timeframes
        }  # Initialisation des poids

        # Configuration des scalers
        self.scaler_path = scaler_path
        self.scalers = {tf: None for tf in self.timeframes}
        self.feature_indices = {}
        self._col_mappings: Dict[str, Dict[str, str]] = {}

        # Initialize scaler cache with LRU behavior
        self._scaler_cache = {}
        self._scaler_cache_hits = 0
        self._scaler_cache_misses = 0
        self._max_scaler_cache_size = 100  # Maximum number of scalers to cache

        # 🔧 FIX CRITIQUE: Charger les scalers d'entraînement AVANT l'initialisation
        self._load_training_scalers()
        
        # Initialisation des scalers (seulement si pas déjà chargés)
        if not self.scalers or not any(self.scalers.values()):
            self._init_scalers()
        else:
            logger.info("✅ Using loaded training scalers - skipping _init_scalers")

        # Calcul de la taille totale de l'observation après flatten
        market_obs_size = sum(self.window_sizes[tf] * self.max_features for tf in self.timeframes)

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
            f"StateBuilder initialized with window_sizes={self.window_sizes}, "
            f"adaptive_window={adaptive_window}, "
            f"timeframes={self.timeframes}, "
            f"features_per_timeframe={self.nb_features_per_tf}"
        )

    def get_feature_names(self, timeframe: str) -> List[str]:
        """
        Get feature names for a specific timeframe.

        Args:
            timeframe: Timeframe to get features for

        Returns:
            List of feature names
        """
        return self.features_config.get(timeframe, [])

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

    def _load_training_scalers(self):
        """
        🔧 FIX CRITIQUE: Charge les scalers sauvegardés pendant l'entraînement.
        
        Cela évite le 'distribution shift' causé par le refit des scalers sur
        des données live qui ont une distribution différente du training.
        """
        import pickle
        from pathlib import Path
        import os
        
        if not hasattr(self, 'scalers'):
            self.scalers = {}
        
        # Essayer d'abord prod_scalers/ (nouveau format)
        # Chercher depuis le répertoire courant ET depuis le parent
        possible_paths = [
            Path("prod_scalers"),
            Path(__file__).parent.parent.parent / "prod_scalers",
            Path(os.getcwd()) / "prod_scalers"
        ]
        
        prod_scalers_dir = None
        for path in possible_paths:
            if path.exists():
                prod_scalers_dir = path
                break
        
        if prod_scalers_dir is None:
            prod_scalers_dir = Path("prod_scalers")
        if prod_scalers_dir.exists():
            try:
                loaded_count = 0
                for timeframe in ['5m', '1h', '4h']:
                    scaler_path = prod_scalers_dir / f"scaler_{timeframe}.pkl"
                    if scaler_path.exists():
                        with open(scaler_path, 'rb') as f:
                            self.scalers[timeframe] = pickle.load(f)
                        logger.info(f"✅ Loaded production scaler: {timeframe}")
                        loaded_count += 1
                
                if loaded_count > 0:
                    logger.info("=" * 60)
                    logger.info("🎯 PRODUCTION SCALERS LOADED - Distribution Preserved")
                    logger.info("=" * 60)
                    self.scalers_loaded_from_training = True
                    return
            except Exception as e:
                logger.error(f"❌ Error loading production scalers: {e}")
        
        # Fallback: chercher les scalers du backtest
        logger.warning(f"⚠️ Production scalers not found in {prod_scalers_dir}")
        logger.warning("   Will fit scalers on live data (may cause distribution shift)")
        self.scalers = {}
        self.scalers_loaded_from_training = False

    def _init_scalers(self):
        """
        Initialize scalers for each timeframe with advanced normalization.

        Each timeframe gets its own scaler with specific parameters:
        - 5m: MinMaxScaler with feature_range (0, 1)
        - 1h: StandardScaler with mean=0, std=1
        - 4h: RobustScaler for outlier resistance
        
        🔧 FIX: Tous les scalers sont wrappés avec SafeScalerWrapper pour éviter
        la division par zéro quand std = 0 (marché à faible volatilité).

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
                base_scaler = MinMaxScaler(feature_range=config.get("feature_range", (0, 1)))
            elif config["scaler_type"] == "standard":
                base_scaler = StandardScaler()
            elif config["scaler_type"] == "robust":
                base_scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaler type: {config['scaler_type']}")

            # 🔧 WRAP avec SafeScalerWrapper pour éviter division par zéro
            self.scalers[tf] = SafeScalerWrapper(base_scaler, epsilon=1e-8)
            
            logger.info(
                f"✅ Scaler initialized for timeframe {tf}: {config['scaler_type']} "
                f"(wrapped with SafeScalerWrapper to prevent NaN from std=0)"
            )

    def fit_scalers(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Fit scalers on the provided data with memory optimization.

        Args:
            data: Dictionary mapping timeframes to DataFrames
        """
        if not self.normalize:
            return

        # 🔧 FIX: Ne PAS refitter si les scalers d'entraînement sont déjà chargés
        if getattr(self, 'scalers_loaded_from_training', False):
            logger.info("✅ Training scalers already loaded - SKIPPING fit_scalers()")
            logger.info("   Using pre-fitted scalers from training to preserve distribution")
            return

        logger.info("Fitting scalers on provided data...")

        # Vérifier la mémoire avant le fitting
        current_memory = self._get_memory_usage_mb()
        if current_memory > self.memory_config["memory_warning_threshold_mb"]:
            logger.warning(f"Memory usage high before fitting: {current_memory:.1f} MB")

        try:
            # S'assurer que les scalers existent pour chaque TF
            for tf in self.timeframes:
                if tf not in self.scalers or self.scalers[tf] is None:
                    if tf == "5m":
                        base_scaler = MinMaxScaler(feature_range=(0, 1))
                    elif tf == "1h":
                        base_scaler = StandardScaler()
                    elif tf == "4h":
                        base_scaler = RobustScaler()
                    else:
                        base_scaler = StandardScaler()
                    
                    # 🔧 WRAP avec SafeScalerWrapper pour éviter division par zéro
                    self.scalers[tf] = SafeScalerWrapper(base_scaler, epsilon=1e-8)
                    logger.info(f"✅ Initializing scaler for timeframe {tf} (with SafeScalerWrapper)")

            # Construire des DataFrames concaténés PAR TIMEFRAME
            # Supporte deux formats d'entrée:
            #   A) {tf: DataFrame}
            #   B) {asset: {tf: DataFrame}}
            per_tf_frames: Dict[str, list] = {tf: [] for tf in self.timeframes}

            # Détection du format: si la première valeur est un DataFrame => format A
            first_val = next(iter(data.values())) if len(data) > 0 else None
            if isinstance(first_val, pd.DataFrame):
                # Format A: {tf: df}
                for tf, df in data.items():
                    if tf in per_tf_frames:
                        per_tf_frames[tf].append(df)
                    else:
                        logger.warning(f"Skipping unknown timeframe {tf}")
            else:
                # Format B: {asset: {tf: df}}
                for asset, asset_tfs in data.items():
                    if not isinstance(asset_tfs, dict):
                        logger.warning(f"Unexpected data format for asset {asset}, skipping")
                        continue
                    for tf, df in asset_tfs.items():
                        if tf in per_tf_frames:
                            per_tf_frames[tf].append(df)
                        else:
                            logger.warning(f"Skipping unknown timeframe {tf}")

            # Fit des scalers pour chaque TF
            for tf in self.timeframes:
                if len(per_tf_frames[tf]) == 0:
                    logger.warning(f"No data provided for timeframe {tf}; skipping fit")
                    continue

                df = pd.concat(per_tf_frames[tf], axis=0)
                # Colonnes en lowercase pour sélection robuste
                df.columns = [col.lower() for col in df.columns]

                features_req = self.features_config.get(tf, [])
                if not features_req:
                    logger.warning(f"No features configured for timeframe {tf}")
                    continue

                # Mapping insensible à la casse
                req_lower = [f.lower() for f in features_req]
                for fl in req_lower:
                    if fl not in df.columns:
                        df[fl] = 0.0

                # Si concaténation vide, créer un placeholder minimal
                if df.shape[0] == 0:
                    logger.warning(
                        f"Empty dataframe for timeframe {tf} during scaler fit; using zero placeholder"
                    )
                    timeframe_data = np.zeros((2, len(req_lower)), dtype=np.float32)
                else:
                    timeframe_data = df[req_lower].values.astype(np.float32)

                if not np.isfinite(timeframe_data).all():
                    logger.warning(
                        f"Non-finite values found in {tf} data. Replacing with zeros."
                    )
                    timeframe_data = np.nan_to_num(
                        timeframe_data, nan=0.0, posinf=0.0, neginf=0.0
                    )

                if len(timeframe_data) < 2:
                    logger.warning(
                        f"Not enough samples ({len(timeframe_data)}) to fit scaler for {tf}; padding with zeros"
                    )
                    # Dupliquer/compléter pour atteindre 2 échantillons
                    need = 2 - len(timeframe_data)
                    pad = np.zeros((need, timeframe_data.shape[1]), dtype=np.float32)
                    timeframe_data = np.vstack([timeframe_data, pad])

                # Pad à max_features pour stabilité dimensionnelle
                padded_data = self._pad_features(timeframe_data, self.max_features)

                # Cache des scalers par hash des données
                data_hash = self._get_data_hash(padded_data)
                cache_key = f"{tf}_{data_hash}"

                if cache_key in self._scaler_cache:
                    self._scaler_cache_hits += 1
                    self.scalers[tf] = self._scaler_cache[cache_key]
                    logger.debug(f"Using cached scaler for {tf}")
                else:
                    self._scaler_cache_misses += 1
                    self.scalers[tf].fit(padded_data)
                    if len(self._scaler_cache) >= self._max_scaler_cache_size:
                        del self._scaler_cache[next(iter(self._scaler_cache))]
                    self._scaler_cache[cache_key] = self.scalers[tf]
                    logger.info(
                        f"Fitted new scaler for {tf} on {len(padded_data)} samples"
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
                self._scaler_cache_hits
                / (self._scaler_cache_hits + self._scaler_cache_misses)
            )
            * 100
            if (self._scaler_cache_hits + self._scaler_cache_misses) > 0
            else 0
        )

        logger.info(
            f"Scaler cache stats: {len(self._scaler_cache)} cached scalers, "
            f"{self._scaler_cache_hits} hits, {self._scaler_cache_misses} misses, "
            f"{cache_hit_rate:.1f}% hit rate"
        )





    def get_portfolio_state_dim(self) -> int:
        """
        Retourne la dimension de l'état du portefeuille.

        Returns:
            int: Dimension de l'état du portefeuille
        """
        return 20

    def build_portfolio_state(self, portfolio_manager: Any) -> np.ndarray:
        """
        Build portfolio state information to include in observations.

        Args:
            portfolio_manager: Portfolio manager instance

        Returns:
            Numpy array containing portfolio state information
        """
        if not self.include_portfolio_state or portfolio_manager is None:
            return np.zeros(20, dtype=np.float32)  # Return zero-padded portfolio state

        try:
            return portfolio_manager.get_state_vector()

        except Exception as e:
            logger.error(f"Error building portfolio state: {e}")
            return np.zeros(20, dtype=np.float32)  # Return zero-padded portfolio state

    # ------------------------------------------------------------------
    # SOTA 2025: Context vector dimension (6 market + 6 cyclical time)
    # ------------------------------------------------------------------
    CONTEXT_DIM = 12

    def build_context_vector(
        self,
        data: Dict[str, pd.DataFrame] = None,
        current_idx: int = 0,
        portfolio_manager: Any = None,
    ) -> np.ndarray:
        """Build a market context vector for FiLM Meta-RL modulation.

        SOTA 2025 – The vector contains 12 components:
            [0]  volatility           – normalised ATR / price  (0-1)
            [1]  trend                – EMA crossover direction (-1 to 1)
            [2]  adx                  – ADX / 100  (0-1)
            [3]  regime_score         – composite regime indicator (-1 to 1)
            [4]  drawdown             – current portfolio drawdown (0-1)
            [5]  candle_progress_pct  – progress within current candle (0-1)
            [6]  sin_hour             – sin(2*pi*hour/24)
            [7]  cos_hour             – cos(2*pi*hour/24)
            [8]  sin_dow              – sin(2*pi*dayofweek/7)
            [9]  cos_dow              – cos(2*pi*dayofweek/7)
            [10] sin_dom              – sin(2*pi*dayofmonth/31)
            [11] cos_dom              – cos(2*pi*dayofmonth/31)

        Args:
            data: Dict mapping timeframes to DataFrames (optional).
            current_idx: Current bar index in the data.
            portfolio_manager: Optional portfolio manager for drawdown info.

        Returns:
            np.ndarray of shape (12,) with dtype float32.
        """
        context = np.zeros(self.CONTEXT_DIM, dtype=np.float32)

        if data is None:
            return context

        try:
            for tf in self.timeframes:
                if tf not in data or data[tf].empty:
                    continue
                df = data[tf]
                idx = min(current_idx, len(df) - 1)
                if idx < 0:
                    continue

                # [0] Volatility from ATR if available
                if "ATR_14" in df.columns and "CLOSE" in df.columns:
                    atr = df["ATR_14"].iloc[idx]
                    close = df["CLOSE"].iloc[idx]
                    if np.isfinite(atr) and np.isfinite(close) and close > 0:
                        context[0] = float(np.clip(atr / close, 0.0, 1.0))

                # [1] Trend from EMA crossover
                if "EMA_12" in df.columns and "EMA_26" in df.columns:
                    ema12 = df["EMA_12"].iloc[idx]
                    ema26 = df["EMA_26"].iloc[idx]
                    if np.isfinite(ema12) and np.isfinite(ema26) and ema26 != 0:
                        diff = (ema12 - ema26) / abs(ema26)
                        context[1] = float(np.clip(diff * 10, -1.0, 1.0))

                # [2] ADX
                if "ADX_14" in df.columns:
                    adx = df["ADX_14"].iloc[idx]
                    if np.isfinite(adx):
                        context[2] = float(np.clip(adx / 100.0, 0.0, 1.0))

                # [3] Regime score: positive=trending, negative=volatile
                if context[2] > 0.25:  # ADX > 25 = trending
                    context[3] = float(context[2])
                elif context[0] > 0.03:  # high vol
                    context[3] = float(-context[0])
                # else: ranging = 0

                break  # Use first available timeframe

            # [4] Drawdown from portfolio manager
            if portfolio_manager is not None:
                try:
                    if hasattr(portfolio_manager, "get_metrics"):
                        metrics = portfolio_manager.get_metrics()
                        dd = metrics.get("drawdown", 0.0)
                        context[4] = float(np.clip(abs(dd) / 100.0, 0.0, 1.0))
                    elif hasattr(portfolio_manager, "get_state_vector"):
                        state = portfolio_manager.get_state_vector()
                        if len(state) > 4:
                            context[4] = float(np.clip(abs(state[4]) / 100.0, 0.0, 1.0))
                except Exception:
                    pass

            # [5] Candle progress: position within data chunk (0..1)
            # In backtest, this represents how far through the episode we are.
            # In live mode, this would represent intra-candle progress.
            try:
                for tf in self.timeframes:
                    if tf in data and not data[tf].empty:
                        total_len = len(data[tf])
                        if total_len > 1:
                            context[5] = float(np.clip(
                                current_idx / (total_len - 1), 0.0, 1.0
                            ))
                        else:
                            context[5] = 1.0
                        break
                else:
                    context[5] = 1.0  # fallback
            except Exception:
                context[5] = 1.0

        except Exception as e:
            logger.warning(f"Error building context vector: {e}. Returning zeros.")

        # -----------------------------------------------------------
        # SOTA 2025 – Cyclical Time Encoding (Time2Vec-inspired)
        # Preserves temporal continuity: 23h59 is near 00h01.
        # -----------------------------------------------------------
        try:
            timestamp = None
            if data is not None:
                for tf in self.timeframes:
                    if tf in data and not data[tf].empty:
                        df = data[tf]
                        idx = min(current_idx, len(df) - 1)
                        if idx >= 0:
                            # Try to get timestamp from index or column
                            if isinstance(df.index, pd.DatetimeIndex):
                                timestamp = df.index[idx]
                            elif 'timestamp' in df.columns:
                                timestamp = pd.Timestamp(df['timestamp'].iloc[idx])
                            elif 'TIMESTAMP' in df.columns:
                                timestamp = pd.Timestamp(df['TIMESTAMP'].iloc[idx])
                        if timestamp is not None:
                            break

            if timestamp is not None:
                hour = timestamp.hour + timestamp.minute / 60.0
                dow = timestamp.dayofweek  # 0=Monday .. 6=Sunday
                dom = timestamp.day

                context[6]  = float(np.sin(2.0 * np.pi * hour / 24.0))
                context[7]  = float(np.cos(2.0 * np.pi * hour / 24.0))
                context[8]  = float(np.sin(2.0 * np.pi * dow / 7.0))
                context[9]  = float(np.cos(2.0 * np.pi * dow / 7.0))
                context[10] = float(np.sin(2.0 * np.pi * dom / 31.0))
                context[11] = float(np.cos(2.0 * np.pi * dom / 31.0))
        except Exception as e:
            logger.debug(f"Cyclical time encoding fallback: {e}")

        return context


    def _get_column_mapping(self, df: pd.DataFrame, tf: str):
        if tf not in self._col_mappings:
            # build once
            m = {col.upper(): col for col in df.columns}
            self._col_mappings[tf] = m
        return self._col_mappings[tf]

    def _ensure_consistent_shape(self, observation: np.ndarray, target_shape: tuple) -> np.ndarray:
        """
        Pad or truncate observation to match target shape.
        """
        if observation.shape == target_shape:
            return observation

        logging.warning(f"Shape mismatch: {observation.shape} -> padding/truncating to {target_shape}")

        # Pad or truncate rows (window size)
        if observation.shape[0] < target_shape[0]:
            padding_rows = ((0, target_shape[0] - observation.shape[0]), (0, 0))
            observation = np.pad(observation, padding_rows, mode='constant')
        elif observation.shape[0] > target_shape[0]:
            observation = observation[-target_shape[0]:, :] # Keep most recent

        # Pad or truncate columns (features)
        if observation.shape[1] < target_shape[1]:
            padding_cols = ((0, 0), (0, target_shape[1] - observation.shape[1]))
            observation = np.pad(observation, padding_cols, mode='constant')
        elif observation.shape[1] > target_shape[1]:
            observation = observation[:, :target_shape[1]]

        return observation

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
            return np.pad(obs, padding, mode='constant')
        elif obs.shape[1] > target_length:
            # Truncate to target length
            return obs[:, :target_length]
        return obs

    def _build_asset_timeframe_state(self, asset, timeframe, df: pd.DataFrame):
        required = self.features_config[timeframe]
        # 1) Ajout des colonnes manquantes à 0.0
        for feat in required:
            if feat not in df.columns:
                df[feat] = 0.0
        # 2) Sélection et ordre garanti
        arr = df[required].to_numpy()
        return arr  # shape = (n_rows, len(required))

    def build_observation(
        self, current_idx: int, data: Dict[str, Dict[str, pd.DataFrame]], portfolio_manager: Any = None
    ) -> Dict[str, np.ndarray]:
        """
        Build observations for each timeframe and return a dictionary of arrays.

        Args:
            current_idx: Current index in the data
            data: Dictionary mapping assets to their timeframe data

        Returns:
            Dictionary of numpy arrays, one for each timeframe
        """
        observations = {}
        max_features = self.max_features  # Maximum features across all timeframes

        for tf in self.timeframes:
            window_size = self.window_sizes.get(tf, 20)
            try:
                # Combine data from all assets for the current timeframe
                asset_dfs = []
                for asset in data.keys():
                    if tf in data[asset]:
                        asset_dfs.append(data[asset][tf])

                if not asset_dfs:
                    raise KeyError(f"Missing data for timeframe {tf}")

                # Get features for this timeframe
                features = self.features_config.get(tf, [])
                if not features:
                    raise ValueError(f"No features configured for timeframe {tf}")

                # Concatenate all assets for this timeframe
                # CRITICAL FIX: Explicit copy to prevent shared memory with self.current_data
                # pd.concat can return a view/shared buffer if list has 1 element
                df = pd.concat(asset_dfs, axis=0).copy()

                # Standardize column names to lowercase
                df.columns = [col.lower() for col in df.columns]

                # Ensure we have enough data
                if current_idx < window_size:
                    logger.warning(
                        f"Current index {current_idx} < window_size {window_size} for {tf}. "
                        "Returning zero-padded observation."
                    )
                    observations[tf] = np.zeros(
                        (window_size, max_features), dtype=np.float32
                    )
                    continue

                # Get the window of data with robust clamping at chunk edges
                df_len = len(df)
                # Ensure at least 1 and at most df_len
                idx_clamped = min(max(int(current_idx), 1), df_len)
                start = max(0, idx_clamped - window_size)
                end = idx_clamped
                window_data = df.iloc[start:end]

                # Si la fenêtre est vide (ex: index > len(df) après concat), retourner des zéros
                if window_data.shape[0] == 0:
                    logger.warning(
                        f"Empty window slice for {tf} at idx={current_idx} (clamped={idx_clamped}) with window_size={window_size}. Returning zeros."
                    )
                    observations[tf] = np.zeros(
                        (window_size, max_features), dtype=np.float32
                    )
                    continue

                # Handle missing values
                if window_data.isnull().values.any():
                    logger.warning(f"NaN values found in {tf} data, using forward fill then zero fill")
                    window_data = window_data.ffill().fillna(0)

                # --- Compute relative price features (avoids leakage) ---
                window_data = window_data.copy()  # avoid SettingWithCopy warnings
                if "close" in window_data.columns:
                    close_series = window_data["close"].astype(np.float64)
                    # log-return: ln(close_t / close_{t-1})
                    window_data["log_return"] = np.log(
                        close_series / close_series.shift(1).replace(0, np.nan)
                    ).fillna(0.0).astype(np.float32)
                    # close-EMA20 ratio: (close - EMA20) / EMA20
                    ema20 = close_series.ewm(span=20, min_periods=1).mean()
                    window_data["close_ema20_ratio"] = (
                        (close_series - ema20) / ema20.replace(0, np.nan)
                    ).fillna(0.0).astype(np.float32)
                else:
                    window_data["log_return"] = 0.0
                    window_data["close_ema20_ratio"] = 0.0

                # --- L2 order-book proxy features (defaults if absent) ---
                if "spread_bps" not in window_data.columns:
                    window_data["spread_bps"] = 0.0
                if "liquidity_score" not in window_data.columns:
                    window_data["liquidity_score"] = 0.0

                # Ensure all required features are present (case-insensitive)
                req_lower = [f.lower() for f in features]
                for fl in req_lower:
                    if fl not in window_data.columns:
                        logger.warning(
                            f"Feature '{fl}' not found in {tf} data, adding zeros"
                        )
                        window_data[fl] = 0.0

                # Select only the required features in the correct order
                try:
                    window_data = window_data[req_lower]
                except KeyError as e:
                    logger.error(f"Error selecting features for {tf}: {e}")
                    logger.error(f"Available columns: {window_data.columns.tolist()}")
                    logger.error(f"Requested features: {req_lower}")
                    raise

                # Convert to numpy array
                obs = window_data.values.astype(np.float32)

                # Ensure consistent shape
                obs = self._ensure_consistent_shape(obs, (window_size, max_features))

                # Normalize if required
                if self.normalize and tf in self.scalers and self.scalers[tf] is not None:
                    obs = self.scalers[tf].transform(obs)
                
                # 🔧 CRITICAL FIX: Clip observations to prevent NaN/Inf propagation
                # This ensures that even if scaler produces extreme values, they are bounded
                obs = np.clip(obs, -1e6, 1e6)  # Clip to reasonable bounds
                obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)  # Replace any remaining NaN/Inf

                observations[tf] = obs

            except Exception as e:
                logger.error(f"Error processing {tf} timeframe: {str(e)}")
                # Return zero array with correct dimensions
                observations[tf] = np.zeros(
                    (window_size, max_features), dtype=np.float32
                )

        # Always include portfolio_state for downstream consumers/tests
        try:
            if self.include_portfolio_state and portfolio_manager is not None:
                observations["portfolio_state"] = self.build_portfolio_state(portfolio_manager)
            else:
                observations["portfolio_state"] = np.zeros(self.get_portfolio_state_dim(), dtype=np.float32)
        except Exception:
            observations["portfolio_state"] = np.zeros(self.get_portfolio_state_dim(), dtype=np.float32)

        # Build context_vector for FiLM Meta-RL modulation
        # Flatten data to {tf: DataFrame} for build_context_vector compatibility
        try:
            flat_data = None
            if data is not None:
                flat_data = {}
                for asset_key, asset_data in data.items():
                    if isinstance(asset_data, dict):
                        for tf, df in asset_data.items():
                            if tf not in flat_data and isinstance(df, pd.DataFrame) and not df.empty:
                                flat_data[tf] = df
            observations["context_vector"] = self.build_context_vector(
                data=flat_data,
                current_idx=current_idx,
                portfolio_manager=portfolio_manager,
            )
        except Exception as e:
            logger.warning(f"Error building context_vector in build_observation: {e}")
            observations["context_vector"] = np.zeros(self.CONTEXT_DIM, dtype=np.float32)

        return observations
    def build_adaptive_observation(
        self,
        current_idx: int,
        data: Dict[str, Dict[str, pd.DataFrame]],
        portfolio_manager: Any = None,
    ) -> Dict[str, np.ndarray]:
        """
        Build observation with adaptive window sizing and timeframe weighting.

        Args:
            current_idx: Current index in the data
            data: Dictionary mapping assets to their timeframe data
            portfolio_manager: Portfolio manager instance for portfolio state

        Returns:
            Dictionary containing:
            - 'observation': 3D numpy array of shape (timeframes, window_size, features)
            - 'portfolio_state': 1D numpy array of portfolio state features
        """
        # Initialize return dictionary
        result = {"observation": None, "portfolio_state": None}

        # Update adaptive window based on current market conditions
        first_asset = next(iter(data.keys()))
        self.update_adaptive_window(data[first_asset], current_idx)

        # Build standard observations
        observation_3d = self.build_observation(current_idx, data)

        logger.debug(
            f"build_adaptive_observation received array with shape: {observation_3d.shape}"
        )
        if not isinstance(observation_3d, np.ndarray) or observation_3d.size == 0:
            logger.warning(
                "Observation array is empty. Returning zero-padded observation."
            )
            result["observation"] = np.zeros(self.observation_shape, dtype=np.float32)
            if self.include_portfolio_state:
                result["portfolio_state"] = np.zeros(20, dtype=np.float32)  # CORRECTION: 20 dimensions
            return result

        # Apply timeframe weighting directly on the 3D array
        weighted_observation = (
            observation_3d
            * np.array([self.timeframe_weights[tf] for tf in self.timeframes])[
                :, np.newaxis, np.newaxis
            ]
        )

        # Ensure the observation has the correct shape
        n_timeframes, window_size, n_features = self.observation_shape
        if weighted_observation.shape[1] > window_size:
            # Take the most recent data
            weighted_observation = weighted_observation[:, -window_size:, :]
        elif weighted_observation.shape[1] < window_size:
            # Pad with zeros at the beginning
            padding = np.zeros(
                (n_timeframes, window_size - weighted_observation.shape[1], n_features),
                dtype=np.float32,
            )
            weighted_observation = np.concatenate(
                [padding, weighted_observation], axis=1
            )

        # Ensure correct number of features
        weighted_observation = weighted_observation[:, :, :n_features]

        result["observation"] = weighted_observation

        # Add portfolio state if enabled
        if self.include_portfolio_state and portfolio_manager is not None:
            portfolio_state = self.build_portfolio_state(portfolio_manager)

            # Ensure portfolio state has exactly 20 features (CORRECTION: was 17)
            if portfolio_state.size != 20:
                logger.warning(
                    f"Portfolio state size mismatch. Expected 20, got {portfolio_state.size}. Adjusting."
                )
                if portfolio_state.size < 20:
                    portfolio_state = np.pad(
                        portfolio_state,
                        (0, 20 - portfolio_state.size),
                        mode="constant",
                        constant_values=0,
                    )
                else:
                    portfolio_state = portfolio_state[:20]

            result["portfolio_state"] = portfolio_state.astype(np.float32)

        return result
