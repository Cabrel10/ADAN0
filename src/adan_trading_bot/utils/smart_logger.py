"""
Smart Logger System - Permet à tous les workers de loguer intelligemment
====================================================================

Ce module implémente un système de logging intelligent qui :
- Permet à tous les workers de loguer les erreurs critiques
- Évite la duplication excessive avec déduplication
- Utilise la rotation entre workers pour les logs informationnels
- Applique un sampling pour les logs de debug

Auteur: Trading Bot Team
Date: 2024
"""

import logging
import time
import hashlib
import threading
from typing import Dict, Optional, Any
from collections import defaultdict

class SmartLogger:
    """
    Système de logging intelligent pour environnements multi-workers.

    Stratégies par niveau :
    - CRITICAL/ERROR: Tous workers loggent sans restriction
    - WARNING: Déduplication intelligente avec fenêtre temporelle
    - INFO: Rotation entre workers ou sampling réduit
    - DEBUG: Sampling très réduit basé sur hash du message
    """

    # Cache global partagé entre toutes les instances (thread-safe)
    _global_cache = defaultdict(dict)
    _cache_lock = threading.Lock()

    def __init__(self, worker_id: int = 0, total_workers: int = 4, logger_name: Optional[str] = None):
        """
        Initialise le SmartLogger.

        Args:
            worker_id: ID du worker (0, 1, 2, 3...)
            total_workers: Nombre total de workers
            logger_name: Nom du logger (pour identification)
        """
        self.worker_id = worker_id
        self.total_workers = max(1, total_workers)
        self.logger_name = logger_name or f"worker_{worker_id}"

        # Cache local pour performance
        self._local_cache = {}
        self._rotation_counter = 0

        # Configuration des seuils
        self.dedup_window = 5.0  # secondes pour déduplication WARNING
        self.info_rotation_interval = 10  # steps entre rotations INFO
        self.debug_sample_rate = 0.1  # 10% des messages DEBUG

    def should_log(self, level: str, message: str, step: Optional[int] = None) -> bool:
        """
        Détermine si un message doit être loggé selon la stratégie intelligente.

        Args:
            level: Niveau du log (CRITICAL, ERROR, WARNING, INFO, DEBUG)
            message: Message à logger
            step: Step courant (optionnel)

        Returns:
            True si le message doit être loggé
        """
        level = level.upper()
        current_time = time.time()

        # CRITICAL et ERROR : Toujours loguer depuis tous les workers
        if level in ('CRITICAL', 'ERROR'):
            return True

        # WARNING : Déduplication intelligente
        if level == 'WARNING':
            return self._should_log_warning(message, current_time)

        # INFO : Rotation entre workers avec exception pour messages uniques
        if level == 'INFO':
            return self._should_log_info(message, current_time, step)

        # DEBUG : Sampling très réduit
        if level == 'DEBUG':
            return self._should_log_debug(message)

        # Par défaut, permettre le log
        return True

    def _should_log_warning(self, message: str, current_time: float) -> bool:
        """Gère la déduplication des WARNING."""
        cache_key = f"warning_{message}"

        with SmartLogger._cache_lock:
            last_time = SmartLogger._global_cache['warnings'].get(cache_key, 0)

            # Si assez de temps s'est écoulé OU si c'est un nouveau message
            if current_time - last_time > self.dedup_window:
                SmartLogger._global_cache['warnings'][cache_key] = current_time
                return True

        return False

    def _should_log_info(self, message: str, current_time: float, step: Optional[int]) -> bool:
        """Gère la rotation des INFO entre workers."""
        # Messages critiques : tous les workers peuvent loguer
        critical_keywords = [
            'portfolio', 'position', 'trade', 'profit', 'loss',
            'error', 'failed', 'success', 'completed', 'started'
        ]

        message_lower = message.lower()
        if any(keyword in message_lower for keyword in critical_keywords):
            return True

        # Rotation standard basée sur le step ou le temps
        if step is not None:
            rotation_cycle = step // self.info_rotation_interval
        else:
            # Utiliser le temps si pas de step
            rotation_cycle = int(current_time // 10)  # Cycle de 10 secondes

        return (rotation_cycle % self.total_workers) == self.worker_id

    def _should_log_debug(self, message: str) -> bool:
        """Gère le sampling des DEBUG."""
        # Utiliser hash du message pour sampling déterministe
        message_hash = hashlib.md5(message.encode()).hexdigest()
        hash_int = int(message_hash[:8], 16)

        # Sampling basé sur le hash
        return (hash_int % 100) < (self.debug_sample_rate * 100)

    def smart_log(self, logger: logging.Logger, level: str, message: str,
                  step: Optional[int] = None, *args, **kwargs):
        """
        Interface principale pour logger intelligemment.

        Args:
            logger: Instance du logger Python
            level: Niveau du log
            message: Message à logger
            step: Step courant (optionnel)
            *args, **kwargs: Arguments supplémentaires pour le logger
        """
        if not self.should_log(level, message, step):
            return

        # Préfixer avec l'ID du worker pour traçabilité
        prefixed_message = f"[Worker {self.worker_id}] {message}"

        # Logger selon le niveau
        level_upper = level.upper()
        if level_upper == 'CRITICAL':
            logger.critical(prefixed_message, *args, **kwargs)
        elif level_upper == 'ERROR':
            logger.error(prefixed_message, *args, **kwargs)
        elif level_upper == 'WARNING':
            logger.warning(prefixed_message, *args, **kwargs)
        elif level_upper == 'INFO':
            logger.info(prefixed_message, *args, **kwargs)
        elif level_upper == 'DEBUG':
            logger.debug(prefixed_message, *args, **kwargs)
        else:
            logger.info(prefixed_message, *args, **kwargs)

    def smart_error(self, logger: logging.Logger, message: str, *args, **kwargs):
        """Raccourci pour logger une erreur."""
        self.smart_log(logger, 'ERROR', message, *args, **kwargs)

    def smart_warning(self, logger: logging.Logger, message: str, *args, **kwargs):
        """Raccourci pour logger un warning."""
        self.smart_log(logger, 'WARNING', message, *args, **kwargs)

    def smart_info(self, logger: logging.Logger, message: str, step: Optional[int] = None, *args, **kwargs):
        """Raccourci pour logger une info."""
        self.smart_log(logger, 'INFO', message, step, *args, **kwargs)

    def smart_debug(self, logger: logging.Logger, message: str, *args, **kwargs):
        """Raccourci pour logger un debug."""
        self.smart_log(logger, 'DEBUG', message, *args, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Retourne des statistiques sur le logging."""
        with SmartLogger._cache_lock:
            total_warnings = len(SmartLogger._global_cache['warnings'])

        return {
            'worker_id': self.worker_id,
            'total_workers': self.total_workers,
            'logger_name': self.logger_name,
            'cached_warnings': total_warnings,
            'dedup_window': self.dedup_window,
            'debug_sample_rate': self.debug_sample_rate
        }

    def clear_cache(self):
        """Nettoie le cache (utile pour les tests)."""
        with SmartLogger._cache_lock:
            SmartLogger._global_cache.clear()
        self._local_cache.clear()

    @classmethod
    def create_for_worker(cls, worker_id: int, total_workers: int = 4,
                         logger_name: Optional[str] = None) -> 'SmartLogger':
        """
        Factory method pour créer un SmartLogger pour un worker spécifique.

        Args:
            worker_id: ID du worker
            total_workers: Nombre total de workers
            logger_name: Nom du logger

        Returns:
            Instance de SmartLogger configurée
        """
        return cls(worker_id=worker_id, total_workers=total_workers, logger_name=logger_name)

# Fonction utilitaire pour intégration facile
def create_smart_logger(worker_id: int, total_workers: int = 4,
                       logger_name: Optional[str] = None) -> SmartLogger:
    """
    Fonction utilitaire pour créer rapidement un SmartLogger.

    Usage:
        smart_logger = create_smart_logger(worker_id=1, total_workers=4)
        smart_logger.smart_info(logger, "Message important", step=100)
    """
    return SmartLogger.create_for_worker(worker_id, total_workers, logger_name)

# Exemple d'utilisation
if __name__ == "__main__":
    import logging

    # Configuration du logging de test
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_logger = logging.getLogger("test_smart_logger")

    # Test avec plusieurs workers
    print("=== TEST SMART LOGGER ===")

    workers = [create_smart_logger(i, 4, f"test_worker_{i}") for i in range(4)]

    # Test ERROR (tous doivent loguer)
    print("\n1. Test ERROR (tous workers):")
    for i, smart_logger in enumerate(workers):
        smart_logger.smart_error(test_logger, f"Erreur critique du worker {i}")

    # Test WARNING (déduplication)
    print("\n2. Test WARNING (déduplication):")
    for i, smart_logger in enumerate(workers):
        smart_logger.smart_warning(test_logger, "Message dupliqué")  # Seul le 1er passe

    # Test INFO (rotation)
    print("\n3. Test INFO (rotation):")
    for step in range(10):
        for i, smart_logger in enumerate(workers):
            smart_logger.smart_info(test_logger, f"Info step {step}", step)

    # Test DEBUG (sampling)
    print("\n4. Test DEBUG (sampling):")
    for i in range(20):
        workers[0].smart_debug(test_logger, f"Debug message {i}")

    # Statistiques
    print("\n5. Statistiques:")
    for smart_logger in workers:
        stats = smart_logger.get_stats()
        print(f"Worker {stats['worker_id']}: {stats}")
