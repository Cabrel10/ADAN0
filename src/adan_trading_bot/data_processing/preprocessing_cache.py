"""
Preprocessing Cache Module

This module provides a caching system for preprocessing operations to improve performance
by avoiding redundant computations of expensive transformations.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from filelock import FileLock

logger = logging.getLogger(__name__)


class PreprocessingCache:
    """
    A caching system for preprocessing operations that supports both in-memory and disk caching.

    This class provides a simple interface to cache the results of expensive preprocessing
    operations to avoid redundant computations. It supports both in-memory and disk-based
    caching with automatic invalidation based on function arguments and data content.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        memory_cache_size: int = 100,
        compress: bool = True,
        verbose: int = 0,
    ):
        """
        Initialize the PreprocessingCache.

        Args:
            cache_dir: Directory to store disk cache. If None, only in-memory caching is used.
            memory_cache_size: Maximum number of items to keep in memory (LRU cache).
            compress: Whether to compress cached data on disk.
            verbose: Verbosity level (0: silent, 1: info, 2: debug).
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.memory_cache = {}
        self.memory_cache_order = []
        self.memory_cache_size = memory_cache_size
        self.compress = compress
        self.verbose = verbose

        # Create cache directory if it doesn't exist
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        data_hash: Optional[str] = None
    ) -> str:
        """
        Generate a unique cache key for the function call.

        Args:
            func: The function being cached.
            args: Positional arguments passed to the function.
            kwargs: Keyword arguments passed to the function.
            data_hash: Optional precomputed hash of input data.

        Returns:
            A string representing the cache key.
        """
        # Get function name and module
        func_name = f"{func.__module__}.{func.__name__}"

        # Convert args and kwargs to a stable string representation
        args_repr = repr(args)
        kwargs_repr = json.dumps(kwargs, sort_keys=True)

        # Combine all components
        key_parts = [func_name, args_repr, kwargs_repr]
        if data_hash:
            key_parts.append(data_hash)

        # Create a hash of the key parts
        key_str = "::".join(str(part) for part in key_parts)
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()

    def _get_data_hash(self, data: Any) -> str:
        """
        Generate a hash of the input data for cache invalidation.

        Args:
            data: Input data to hash.

        Returns:
            A string hash of the data.
        """
        if isinstance(data, (pd.DataFrame, pd.Series)):
            # For DataFrames/Series, hash the values and index
            return hashlib.md5(
                pd.util.hash_pandas_object(data, index=True).values.tobytes()
            ).hexdigest()
        elif isinstance(data, np.ndarray):
            # For numpy arrays, hash the array data
            return hashlib.md5(data.tobytes()).hexdigest()
        elif hasattr(data, 'tobytes'):
            # For any object with tobytes() method
            return hashlib.md5(data.tobytes()).hexdigest()
        else:
            # Fallback to repr() for other types
            return hashlib.md5(repr(data).encode('utf-8')).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """
        Get the filesystem path for a cache key.

        Args:
            key: The cache key.

        Returns:
            Path object for the cache file.
        """
        if not self.cache_dir:
            raise ValueError("Cache directory not configured")

        # Create subdirectories based on key prefix to avoid too many files in one directory
        subdir = self.cache_dir / key[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key}.pkl"

    def _load_from_disk(self, key: str) -> Any:
        """
        Load data from disk cache.

        Args:
            key: The cache key.

        Returns:
            The cached data, or None if not found.
        """
        if not self.cache_dir:
            return None

        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None

        try:
            with FileLock(f"{cache_path}.lock"):
                return joblib.load(cache_path)
        except Exception as e:
            logger.warning(f"Error loading from cache {cache_path}: {e}")
            # Remove corrupted cache file
            try:
                cache_path.unlink()
            except Exception:
                pass
            return None

    def _save_to_disk(self, key: str, value: Any) -> None:
        """
        Save data to disk cache.

        Args:
            key: The cache key.
            value: The data to cache.
        """
        if not self.cache_dir:
            return

        cache_path = self._get_cache_path(key)
        temp_path = cache_path.with_suffix('.tmp')

        try:
            # Save to temporary file first
            joblib.dump(
                value,
                temp_path,
                compress=('zlib', 3) if self.compress else 0,
                protocol=4  # Compatible with Python 3.6+
            )

            # Atomic rename
            temp_path.replace(cache_path)

            if self.verbose > 1:
                logger.debug(f"Cached result to {cache_path}")

        except Exception as e:
            logger.error(f"Error saving to cache {cache_path}: {e}")
            # Clean up temporary file on error
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def get(
        self,
        func: Callable,
        *args,
        data: Any = None,
        data_hash: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Get a cached result or compute it if not in cache.

        Args:
            func: The function to cache.
            *args: Positional arguments to pass to the function.
            data: Optional input data (will be hashed for cache invalidation).
            data_hash: Optional precomputed hash of the input data.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of func(*args, **kwargs), either from cache or newly computed.
        """
        # Generate cache key
        data_hash = data_hash or (self._get_data_hash(data) if data is not None else None)
        key = self._get_cache_key(func, args, kwargs, data_hash)

        # Check in-memory cache first
        if key in self.memory_cache:
            if self.verbose > 1:
                logger.debug(f"Cache hit (memory): {func.__name__}")
            return self.memory_cache[key]

        # Check disk cache if enabled
        if self.cache_dir:
            cached = self._load_from_disk(key)
            if cached is not None:
                if self.verbose:
                    logger.debug(f"Cache hit (disk): {func.__name__}")
                # Update memory cache
                self._update_memory_cache(key, cached)
                return cached

        # Cache miss - compute the result
        if self.verbose:
            logger.debug(f"Cache miss: {func.__name__}")

        result = func(*args, **kwargs)

        # Cache the result
        self._update_memory_cache(key, result)
        if self.cache_dir:
            self._save_to_disk(key, result)

        return result

    def _update_memory_cache(self, key: str, value: Any) -> None:
        """
        Update the in-memory LRU cache.

        Args:
            key: The cache key.
            value: The value to cache.
        """
        if key in self.memory_cache:
            # Move to end of LRU
            self.memory_cache_order.remove(key)
        else:
            # Evict oldest item if cache is full
            if len(self.memory_cache) >= self.memory_cache_size and self.memory_cache_order:
                oldest_key = self.memory_cache_order.pop(0)
                del self.memory_cache[oldest_key]

        self.memory_cache[key] = value
        self.memory_cache_order.append(key)

    def clear(self, memory: bool = True, disk: bool = True) -> None:
        """
        Clear the cache.

        Args:
            memory: Whether to clear the in-memory cache.
            disk: Whether to clear the disk cache.
        """
        if memory:
            self.memory_cache.clear()
            self.memory_cache_order = []

        if disk and self.cache_dir:
            for item in self.cache_dir.glob('**/*'):
                if item.is_file() and item.suffix in ('.pkl', '.lock'):
                    try:
                        item.unlink()
                    except Exception as e:
                        logger.warning(f"Error deleting cache file {item}: {e}")


# Global cache instance
_global_cache = None

def get_global_cache(cache_dir: Optional[Union[str, Path]] = None) -> PreprocessingCache:
    """
    Get or create the global preprocessing cache instance.

    Args:
        cache_dir: Directory to store the cache. If None, uses the default location.

    Returns:
        The global PreprocessingCache instance.
    """
    global _global_cache
    if _global_cache is None:
        default_dir = Path.home() / '.adan_trading_bot' / 'preprocessing_cache'
        _global_cache = PreprocessingCache(
            cache_dir=cache_dir or default_dir,
            memory_cache_size=100,
            compress=True,
            verbose=1
        )
    return _global_cache


def cached_function(
    cache: Optional[PreprocessingCache] = None,
    key_func: Optional[Callable] = None,
    use_disk: bool = True
) -> Callable:
    """
    Decorator to cache function results.

    Args:
        cache: Optional PreprocessingCache instance. If None, uses the global cache.
        key_func: Optional function to generate cache keys. If None, uses default key generation.
        use_disk: Whether to use disk caching (if cache is None and using global cache).

    Returns:
        A decorator function.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get cache instance
            cache_inst = cache or get_global_cache()

            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key = cache_inst._get_cache_key(func, args, kwargs)

            # Check cache
            result = cache_inst.get(func, *args, **kwargs)
            return result

        return wrapper
    return decorator
