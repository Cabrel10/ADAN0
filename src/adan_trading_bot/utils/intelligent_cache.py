"""
Système de cache intelligent pour optimiser les calculs répétitifs dans ADAN Trading Bot.
Implémente la tâche 9.1.2 - Cache intelligent des états.
"""

import hashlib
import pickle
import time
import os
from typing import Any, Dict, Optional, Callable, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
from functools import wraps
import threading
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class LRUCache:
    """Cache LRU (Least Recently Used) thread-safe pour les calculs"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return value
            else:
                self.misses += 1
                return None

    def put(self, key: str, value: Any) -> None:
        """Ajoute une valeur au cache"""
        with self.lock:
            if key in self.cache:
                # Update existing key
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)

            self.cache[key] = value

    def clear(self) -> None:
        """Vide le cache"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0

            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'memory_usage_mb': self._estimate_memory_usage()
            }

    def _estimate_memory_usage(self) -> float:
        """Estime l'usage mémoire du cache en MB"""
        try:
            total_size = 0
            for key, value in self.cache.items():
                total_size += len(pickle.dumps(key)) + len(pickle.dumps(value))
            return total_size / (1024 * 1024)  # Convert to MB
        except:
            return 0.0


class PersistentCache:
    """Cache persistant sur disque pour les calculs coûteux"""

    def __init__(self, cache_dir: str = "cache", max_age_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_seconds = max_age_hours * 3600
        self.lock = threading.RLock()

    def _get_cache_path(self, key: str) -> Path:
        """Génère le chemin du fichier de cache"""
        # Use hash to avoid filesystem issues with long keys
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache persistant"""
        cache_path = self._get_cache_path(key)

        with self.lock:
            if not cache_path.exists():
                return None

            # Check if cache is expired
            file_age = time.time() - cache_path.stat().st_mtime
            if file_age > self.max_age_seconds:
                try:
                    cache_path.unlink()
                except:
                    pass
                return None

            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_path}: {e}")
                try:
                    cache_path.unlink()
                except:
                    pass
                return None

    def put(self, key: str, value: Any) -> None:
        """Sauvegarde une valeur dans le cache persistant"""
        cache_path = self._get_cache_path(key)

        with self.lock:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
            except Exception as e:
                logger.warning(f"Failed to save cache file {cache_path}: {e}")

    def clear(self) -> None:
        """Vide le cache persistant"""
        with self.lock:
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    cache_file.unlink()
                except:
                    pass

    def cleanup_expired(self) -> int:
        """Nettoie les fichiers de cache expirés"""
        cleaned_count = 0
        current_time = time.time()

        with self.lock:
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    file_age = current_time - cache_file.stat().st_mtime
                    if file_age > self.max_age_seconds:
                        cache_file.unlink()
                        cleaned_count += 1
                except:
                    pass

        return cleaned_count


class IntelligentCache:
    """Cache intelligent combinant mémoire et disque avec stratégies adaptatives"""

    def __init__(self,
                 memory_cache_size: int = 1000,
                 disk_cache_dir: str = "cache",
                 disk_cache_max_age_hours: int = 24,
                 auto_cleanup_interval: int = 3600):  # 1 hour

        self.memory_cache = LRUCache(memory_cache_size)
        self.disk_cache = PersistentCache(disk_cache_dir, disk_cache_max_age_hours)
        self.auto_cleanup_interval = auto_cleanup_interval
        self.last_cleanup = time.time()

        # Statistics
        self.memory_hits = 0
        self.disk_hits = 0
        self.total_misses = 0

        logger.info(f"IntelligentCache initialized: memory_size={memory_cache_size}, "
                   f"disk_dir={disk_cache_dir}, max_age={disk_cache_max_age_hours}h")

    def _generate_key(self, func_name: str, args: Tuple, kwargs: Dict) -> str:
        """Génère une clé unique pour la fonction et ses arguments"""
        # Convert numpy arrays and pandas objects to hashable representations
        hashable_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                hashable_args.append(('numpy_array', arg.shape, arg.dtype.str, hash(arg.tobytes())))
            elif isinstance(arg, pd.DataFrame):
                hashable_args.append(('dataframe', arg.shape, hash(tuple(arg.columns)), hash(arg.values.tobytes())))
            elif isinstance(arg, pd.Series):
                hashable_args.append(('series', len(arg), hash(arg.values.tobytes())))
            else:
                hashable_args.append(arg)

        hashable_kwargs = []
        for k, v in sorted(kwargs.items()):
            if isinstance(v, np.ndarray):
                hashable_kwargs.append((k, ('numpy_array', v.shape, v.dtype.str, hash(v.tobytes()))))
            elif isinstance(v, (pd.DataFrame, pd.Series)):
                hashable_kwargs.append((k, ('pandas_object', str(type(v)), hash(str(v)))))
            else:
                hashable_kwargs.append((k, v))

        key_data = (func_name, tuple(hashable_args), tuple(hashable_kwargs))
        key_str = str(key_data)

        # Use hash for shorter keys
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, func_name: str, args: Tuple, kwargs: Dict) -> Optional[Any]:
        """Récupère une valeur depuis le cache"""
        key = self._generate_key(func_name, args, kwargs)
        logger.debug(f"\nGET - Clé générée: {key}")
        logger.debug(f"GET - func_name: {func_name}")
        logger.debug(f"GET - args: {args}")
        logger.debug(f"GET - kwargs: {kwargs}")

        # Essayer d'abord le cache mémoire
        result = self.memory_cache.get(key)
        if result is not None:
            self.memory_hits += 1
            logger.debug(f"Trouvé dans le cache mémoire - memory_hits: {self.memory_hits}")
            return result

        # Ensuite essayer le cache disque
        result = self.disk_cache.get(key)
        if result is not None:
            self.disk_hits += 1
            logger.debug(f"Trouvé dans le cache disque - disk_hits: {self.disk_hits}")
            # Remonter en mémoire pour les accès futurs
            self.memory_cache.put(key, result)
            return result

        # Si non trouvé
        self.total_misses += 1
        logger.debug(f"Non trouvé - total_misses: {self.total_misses}")
        # Afficher les statistiques après chaque opération
        stats = self.get_comprehensive_stats()
        logger.debug(f"Statistiques après GET manqué: {stats}")
        return None

    def put(self, func_name: str, args: Tuple, kwargs: Dict, value: Any,
            persist_to_disk: bool = True) -> None:
        """Sauvegarde une valeur dans le cache"""
        key = self._generate_key(func_name, args, kwargs)
        logger.debug(f"PUT - Clé générée: {key}")
        logger.debug(f"PUT - func_name: {func_name}")
        logger.debug(f"PUT - args: {args}")
        logger.debug(f"PUT - kwargs: {kwargs}")

        # Always put in memory cache
        self.memory_cache.put(key, value)

        # Optionally put in disk cache for expensive computations
        if persist_to_disk:
            self.disk_cache.put(key, value)

        # Auto cleanup if needed
        self._auto_cleanup()

        # Afficher les statistiques après chaque opération
        stats = self.get_comprehensive_stats()
        logger.debug(f"Statistiques après PUT: {stats}")

    def _auto_cleanup(self) -> None:
        """Nettoyage automatique périodique"""
        current_time = time.time()
        if current_time - self.last_cleanup > self.auto_cleanup_interval:
            cleaned = self.disk_cache.cleanup_expired()
            if cleaned > 0:
                logger.info(f"Auto-cleanup removed {cleaned} expired cache files")
            self.last_cleanup = current_time

    def clear(self) -> None:
        """Vide tous les caches"""
        self.memory_cache.clear()
        self.disk_cache.clear()
        self.memory_hits = 0
        self.disk_hits = 0
        self.total_misses = 0

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Retourne des statistiques complètes du cache"""
        memory_stats = self.memory_cache.get_stats()

        total_requests = self.memory_hits + self.disk_hits + self.total_misses
        overall_hit_rate = (self.memory_hits + self.disk_hits) / total_requests if total_requests > 0 else 0

        return {
            'memory_cache': memory_stats,
            'disk_cache': {
                'directory': str(self.disk_cache.cache_dir),
                'file_count': len(list(self.disk_cache.cache_dir.glob("*.cache")))
            },
            'overall_stats': {
                'memory_hits': self.memory_hits,
                'disk_hits': self.disk_hits,
                'total_misses': self.total_misses,
                'total_requests': total_requests,
                'overall_hit_rate': overall_hit_rate,
                'memory_hit_rate': self.memory_hits / total_requests if total_requests > 0 else 0,
                'disk_hit_rate': self.disk_hits / total_requests if total_requests > 0 else 0
            }
        }


# Global cache instance
_global_cache = IntelligentCache()


def cached(persist_to_disk: bool = True, cache_instance: Optional[IntelligentCache] = None):
    """
    Décorateur pour mettre en cache les résultats de fonction.

    Args:
        persist_to_disk: Si True, sauvegarde aussi sur disque
        cache_instance: Instance de cache à utiliser (par défaut: cache global)

    Usage:
        @cached(persist_to_disk=True)
        def expensive_calculation(data):
            # Calcul coûteux
            return result
    """
    def decorator(func: Callable) -> Callable:
        cache = cache_instance or _global_cache

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get from cache
            cached_result = cache.get(func.__name__, args, kwargs)
            if cached_result is not None:
                return cached_result

            # Compute and cache result
            start_time = time.time()
            result = func(*args, **kwargs)
            computation_time = time.time() - start_time

            # Cache the result
            cache.put(func.__name__, args, kwargs, result, persist_to_disk)

            # Log for expensive computations
            if computation_time > 1.0:
                logger.info(f"Cached expensive computation {func.__name__}: {computation_time:.2f}s")

            return result

        # Add cache management methods to the function
        wrapper.clear_cache = lambda: cache.clear()
        wrapper.get_cache_stats = lambda: cache.get_comprehensive_stats()

        return wrapper

    return decorator


def get_global_cache_stats() -> Dict[str, Any]:
    """Retourne les statistiques du cache global"""
    return _global_cache.get_comprehensive_stats()


def clear_global_cache() -> None:
    """Vide le cache global"""
    _global_cache.clear()


# ==================== CACHES SPÉCIALISÉS ====================

class ObservationCache:
    """Cache spécialisé pour les observations d'état"""

    def __init__(self, max_size: int = 500):
        self.cache = IntelligentCache(memory_cache_size=max_size,
                                    disk_cache_max_age_hours=1)  # Short TTL for observations

    @cached(persist_to_disk=False)  # Observations change frequently
    def get_cached_observation(self, data_hash: str, current_idx: int,
                             window_size: int, timeframes: Tuple[str, ...]) -> Optional[np.ndarray]:
        """Cache pour les observations construites"""
        # This will be called by StateBuilder
        return None  # Placeholder - actual implementation in StateBuilder

    def invalidate_after_index(self, index: int) -> None:
        """Invalide le cache après un certain index (pour les données mises à jour)"""
        # Implementation would depend on how we store index information in keys
        self.cache.clear()  # Simple approach - clear all


class IndicatorCache:
    """Cache spécialisé pour les indicateurs techniques"""

    def __init__(self, max_size: int = 1000):
        self.cache = IntelligentCache(memory_cache_size=max_size,
                                    disk_cache_max_age_hours=24)  # Longer TTL for indicators

    def get_cached_indicator(self, indicator_name: str, data: np.ndarray,
                           **params) -> Optional[np.ndarray]:
        """Récupère un indicateur du cache"""
        return self.cache.get(f"indicator_{indicator_name}", (data,), params)

    def put_cached_indicator(self, indicator_name: str, data: np.ndarray,
                           result: np.ndarray, **params) -> None:
        """Sauvegarde un indicateur dans le cache"""
        self.cache.put(f"indicator_{indicator_name}", (data,), params, result, True)


# Global specialized caches
observation_cache = ObservationCache()
indicator_cache = IndicatorCache()
