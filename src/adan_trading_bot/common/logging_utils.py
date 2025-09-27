#!/usr/bin/env python3
"""
Système de logging intelligent pour éviter les duplications entre workers
tout en préservant les informations importantes de diagnostic.

Ce module fournit une classe SmartLogger qui gère intelligemment
les logs entre workers parallèles en évitant les duplications
tout en conservant les informations critiques.

Usage:
    from adan_trading_bot.common.logging_utils import SmartLogger

    # Initialisation
    smart_logger = SmartLogger(logger, worker_id=0, total_workers=4)

    # Logs critiques (tous workers peuvent loguer)
    smart_logger.error("Erreur critique détectée")
    smart_logger.warning("Avertissement important")

    # Logs informationnels (rotation entre workers)
    smart_logger.info("Information générale", rotate=True)

    # Logs de debug (sampling)
    smart_logger.debug("Information de debug", sample_rate=0.1)
"""

import time
import threading
from typing import Dict, Optional, Any
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class SmartLogger:
    """
    Système de logging intelligent qui évite les duplications entre workers
    tout en préservant les informations importantes.

    Fonctionnalités:
    - Logs critiques: Tous workers peuvent loguer
    - Logs informationnels: Rotation entre workers
    - Logs de debug: Sampling configurable
    - Cache temporel pour éviter les doublons
    - Thread-safe
    """

    # Cache global partagé entre instances pour éviter les doublons
    _global_cache: Dict[str, float] = {}
    _cache_lock = threading.Lock()
    _rotation_counter = 0
    _rotation_lock = threading.Lock()

    def __init__(self, base_logger: logging.Logger, worker_id: int = 0, total_workers: int = 4):
        """
        Initialise le SmartLogger.

        Args:
            base_logger: Logger de base à utiliser
            worker_id: ID du worker (0, 1, 2, ...)
            total_workers: Nombre total de workers
        """
        self.base_logger = base_logger
        self.worker_id = worker_id
        self.total_workers = total_workers
        self._local_cache: Dict[str, float] = {}

        # Configuration par défaut
        self.cache_duration = 5.0  # 5 secondes
        self.default_sample_rate = 0.2  # 20% des messages de debug

        # Compteurs pour statistiques
        self.stats = {
            'critical': 0,
            'error': 0,
            'warning': 0,
            'info': 0,
            'debug': 0,
            'filtered': 0
        }

    def _should_log_with_cache(self, message: str, level: str, cache_duration: Optional[float] = None) -> bool:
        """
        Détermine si un message doit être loggé en utilisant le cache temporel.

        Args:
            message: Message à logger
            level: Niveau de log
            cache_duration: Durée du cache en secondes

        Returns:
            True si le message doit être loggé
        """
        if cache_duration is None:
            cache_duration = self.cache_duration

        current_time = time.time()
        cache_key = f"{level}:{message[:100]}"  # Limiter la clé à 100 chars

        with self._cache_lock:
            last_time = self._global_cache.get(cache_key, 0)
            if current_time - last_time >= cache_duration:
                self._global_cache[cache_key] = current_time
                return True
            else:
                self.stats['filtered'] += 1
                return False

    def _get_rotation_worker(self) -> int:
        """
        Obtient le worker qui doit logger pour cette rotation.

        Returns:
            ID du worker qui doit logger
        """
        with self._rotation_lock:
            SmartLogger._rotation_counter += 1
            return SmartLogger._rotation_counter % self.total_workers

    def _format_message(self, message: str, **kwargs) -> str:
        """
        Formate le message avec les informations du worker.

        Args:
            message: Message de base
            **kwargs: Paramètres additionnels

        Returns:
            Message formaté
        """
        prefix = f"[Worker {self.worker_id}]"

        # Ajouter des informations contextuelles si fournies
        if kwargs:
            context_parts = []
            for key, value in kwargs.items():
                context_parts.append(f"{key}={value}")
            context = " | ".join(context_parts)
            return f"{prefix} {message} | {context}"

        return f"{prefix} {message}"

    def critical(self, message: str, **kwargs):
        """
        Log un message critique. Tous les workers peuvent loguer.

        Args:
            message: Message à logger
            **kwargs: Contexte additionnel
        """
        formatted_msg = self._format_message(message, **kwargs)
        self.base_logger.critical(formatted_msg)
        self.stats['critical'] += 1

    def error(self, message: str, dedupe: bool = True, **kwargs):
        """
        Log une erreur. Tous les workers peuvent loguer avec déduplication optionnelle.

        Args:
            message: Message à logger
            dedupe: Si True, évite les doublons via cache
            **kwargs: Contexte additionnel
        """
        if dedupe and not self._should_log_with_cache(message, 'error'):
            return

        formatted_msg = self._format_message(message, **kwargs)
        self.base_logger.error(formatted_msg)
        self.stats['error'] += 1

    def warning(self, message: str, dedupe: bool = True, **kwargs):
        """
        Log un avertissement. Tous les workers peuvent loguer avec déduplication optionnelle.

        Args:
            message: Message à logger
            dedupe: Si True, évite les doublons via cache
            **kwargs: Contexte additionnel
        """
        if dedupe and not self._should_log_with_cache(message, 'warning'):
            return

        formatted_msg = self._format_message(message, **kwargs)
        self.base_logger.warning(formatted_msg)
        self.stats['warning'] += 1

    def info(self, message: str, rotate: bool = False, dedupe: bool = True, **kwargs):
        """
        Log une information.

        Args:
            message: Message à logger
            rotate: Si True, utilise la rotation entre workers
            dedupe: Si True, évite les doublons via cache
            **kwargs: Contexte additionnel
        """
        if rotate:
            # Seul le worker désigné par rotation peut loguer
            if self.worker_id != self._get_rotation_worker():
                self.stats['filtered'] += 1
                return

        if dedupe and not self._should_log_with_cache(message, 'info'):
            return

        formatted_msg = self._format_message(message, **kwargs)
        self.base_logger.info(formatted_msg)
        self.stats['info'] += 1

    def debug(self, message: str, sample_rate: Optional[float] = None, **kwargs):
        """
        Log un message de debug avec sampling.

        Args:
            message: Message à logger
            sample_rate: Taux d'échantillonnage (0.0 à 1.0)
            **kwargs: Contexte additionnel
        """
        import random

        if sample_rate is None:
            sample_rate = self.default_sample_rate

        # Sampling basé sur worker_id et message pour être déterministe
        seed = hash(f"{self.worker_id}:{message}") % 1000
        if (seed / 1000.0) > sample_rate:
            self.stats['filtered'] += 1
            return

        formatted_msg = self._format_message(message, **kwargs)
        self.base_logger.debug(formatted_msg)
        self.stats['debug'] += 1

    def info_once_per_worker(self, message: str, **kwargs):
        """
        Log une information une seule fois par worker (cache local).

        Args:
            message: Message à logger
            **kwargs: Contexte additionnel
        """
        if message in self._local_cache:
            self.stats['filtered'] += 1
            return

        self._local_cache[message] = time.time()
        formatted_msg = self._format_message(message, **kwargs)
        self.base_logger.info(formatted_msg)
        self.stats['info'] += 1

    def log_with_frequency(self, level: str, message: str, frequency: float = 10.0, **kwargs):
        """
        Log un message avec une fréquence maximale.

        Args:
            level: Niveau de log ('info', 'warning', 'error', etc.)
            message: Message à logger
            frequency: Fréquence maximale en secondes
            **kwargs: Contexte additionnel
        """
        if not self._should_log_with_cache(message, level, frequency):
            return

        formatted_msg = self._format_message(message, **kwargs)
        getattr(self.base_logger, level.lower())(formatted_msg)
        self.stats[level.lower()] += 1

    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de logging pour ce worker.

        Returns:
            Dictionnaire avec les statistiques
        """
        total_logged = sum(self.stats[k] for k in ['critical', 'error', 'warning', 'info', 'debug'])
        total_attempted = total_logged + self.stats['filtered']

        return {
            'worker_id': self.worker_id,
            'total_logged': total_logged,
            'total_filtered': self.stats['filtered'],
            'total_attempted': total_attempted,
            'filter_rate': self.stats['filtered'] / max(total_attempted, 1),
            'by_level': dict(self.stats)
        }

    def cleanup_cache(self, max_age: float = 300.0):
        """
        Nettoie les caches anciens.

        Args:
            max_age: Age maximum des entrées en secondes (défaut: 5 minutes)
        """
        current_time = time.time()

        with self._cache_lock:
            expired_keys = [
                key for key, timestamp in self._global_cache.items()
                if current_time - timestamp > max_age
            ]
            for key in expired_keys:
                del self._global_cache[key]

        # Nettoyer le cache local aussi
        expired_local = [
            key for key, timestamp in self._local_cache.items()
            if current_time - timestamp > max_age
        ]
        for key in expired_local:
            del self._local_cache[key]


def create_smart_logger(base_logger: logging.Logger, worker_id: int = 0, total_workers: int = 4) -> SmartLogger:
    """
    Factory function pour créer un SmartLogger.

    Args:
        base_logger: Logger de base
        worker_id: ID du worker
        total_workers: Nombre total de workers

    Returns:
        Instance de SmartLogger configurée
    """
    return SmartLogger(base_logger, worker_id, total_workers)


def smart_log_method(method_name: str = 'info', rotate: bool = False, dedupe: bool = True):
    """
    Décorateur pour ajouter le logging intelligent aux méthodes.

    Args:
        method_name: Niveau de log à utiliser
        rotate: Si True, utilise la rotation entre workers
        dedupe: Si True, évite les doublons

    Usage:
        @smart_log_method('info', rotate=True)
        def my_method(self):
            return "Message à logger"
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)

            # Essayer d'obtenir le smart_logger de l'instance
            if hasattr(self, 'smart_logger') and isinstance(self.smart_logger, SmartLogger):
                log_method = getattr(self.smart_logger, method_name)
                if method_name == 'info' and rotate:
                    log_method(str(result), rotate=True, dedupe=dedupe)
                elif method_name in ['error', 'warning'] and dedupe:
                    log_method(str(result), dedupe=dedupe)
                else:
                    log_method(str(result))

            return result
        return wrapper
    return decorator


# Configuration par défaut pour différents types d'usage
DEFAULT_CONFIGS = {
    'training': {
        'cache_duration': 5.0,
        'default_sample_rate': 0.1,  # 10% pour l'entraînement
    },
    'validation': {
        'cache_duration': 2.0,
        'default_sample_rate': 0.3,  # 30% pour la validation
    },
    'testing': {
        'cache_duration': 1.0,
        'default_sample_rate': 0.5,  # 50% pour les tests
    },
    'debug': {
        'cache_duration': 0.5,
        'default_sample_rate': 1.0,  # 100% en mode debug
    }
}


def configure_smart_logger(smart_logger: SmartLogger, config_name: str = 'training'):
    """
    Configure un SmartLogger avec des paramètres prédéfinis.

    Args:
        smart_logger: Instance à configurer
        config_name: Nom de la configuration ('training', 'validation', etc.)
    """
    if config_name in DEFAULT_CONFIGS:
        config = DEFAULT_CONFIGS[config_name]
        smart_logger.cache_duration = config['cache_duration']
        smart_logger.default_sample_rate = config['default_sample_rate']
