"""
Shared Experience Buffer pour l'apprentissage distribué en trading spot.

Ce module implémente un buffer d'expérience partagé thread-safe qui supporte
le Prioritized Experience Replay (PER) pour l'apprentissage distribué.
"""

import numpy as np
import random
import heapq
import time
from typing import Dict, List, Tuple, Any, Optional, Union
import torch
import pickle
import os
import multiprocessing as mp
from multiprocessing import RLock, Manager
from pathlib import Path


class SharedExperienceBuffer:
    """
    Buffer d'expérience partagé thread-safe avec support pour le Prioritized Experience Replay.

    Ce buffer peut être partagé entre plusieurs processus et utilise des verrous
    pour assurer la cohérence des données.
    """

    def __init__(
        self,
        buffer_size: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6
    ) -> None:
        """
        Initialise le buffer d'expérience partagé.

        Args:
            buffer_size: Taille maximale du buffer
            alpha: Contrôle l'importance des priorités (0 = uniforme, 1 = pleinement prioritaire)
            beta: Contrôle l'importance de l'échantillonnage par importance
            beta_increment: Incrément progressif de beta vers 1.0
            epsilon: Petite constante pour éviter des priorités nulles
        """
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        # Utiliser un gestionnaire de mémoire partagée pour les données partagées
        self.manager = Manager()
        self.buffer = self.manager.list()
        self.priorities = self.manager.list()
        self._lock = RLock()

        # Variables d'état partagées
        self._shared_state = self.manager.dict()
        self._shared_state['pos'] = 0
        self._shared_state['max_priority'] = 1.0
        self._shared_state['num_additions'] = 0
        self._shared_state['num_samples'] = 0
        self._shared_state['last_add_time'] = time.time()
        self._shared_state['last_sample_time'] = time.time()
        self._shared_state['total_added'] = 0
        self._shared_state['total_sampled'] = 0

    def __len__(self) -> int:
        """Retourne le nombre d'expériences actuellement dans le buffer."""
        with self._lock:
            return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Vérifie si le buffer contient suffisamment d'expériences pour un batch."""
        return len(self) >= batch_size

    def add(self, experience: Dict[str, Any], priority: Optional[float] = None) -> None:
        """
        Ajoute une expérience au buffer de manière thread-safe.

        Args:
            experience: Dictionnaire contenant les données d'expérience
            priority: Priorité de l'expérience (optionnel)

        Returns:
            int: Taille actuelle du buffer après ajout
        """
        with self._lock:
            if priority is None:
                priority = self._shared_state['max_priority']
            else:
                # Appliquer la même transformation que dans update_priorities
                priority = (abs(priority) + self.epsilon) ** self.alpha

            # S'assurer que l'expérience est sérialisable
            exp_copy = {k: self._make_serializable(v) for k, v in experience.items()}

            if len(self.buffer) < self.buffer_size:
                idx = len(self.buffer)
                self.buffer.append(exp_copy)
                self.priorities.append(priority)  # Initialisé à 0, mis à jour ci-dessous
            else:
                idx = self._shared_state['pos']
                self.buffer[idx] = exp_copy
                self.priorities[idx] = priority

                # Mettre à jour la position pour le prochain ajout
                self._shared_state['pos'] = (idx + 1) % self.buffer_size
            self._shared_state['max_priority'] = max(self._shared_state['max_priority'], priority)

            # Mettre à jour les compteurs et horodatages
            self._shared_state['num_additions'] += 1
            self._shared_state['total_added'] += 1
            self._shared_state['last_add_time'] = time.time()

            # Notifier l'orchestrateur du nouvel ajout
            if hasattr(self, 'orchestrator'):
                self.orchestrator.metrics['buffer_additions'] += 1

    def sample(self, batch_size: int) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """
        Échantillonne un lot d'expériences du buffer de manière thread-safe.

        Args:
            batch_size: Taille du lot à échantillonner

        Returns:
            Tuple contenant:
                - Dictionnaire des expériences échantillonnées, regroupées par clé
                  (ex: {'state': np.ndarray, 'action': np.ndarray, ...})
                - Tableau des indices des expériences
                - Tableau des poids d'importance
        """
        with self._lock:
            if len(self.buffer) < batch_size:
                raise ValueError(
                    f"Tentative d'échantillonnage de {batch_size} expériences "
                    f"alors que le buffer n'en contient que {len(self.buffer)}"
                )

            # Convertir en listes Python pour éviter les problèmes de synchronisation
            priorities = list(self.priorities[:len(self.buffer)])

            # Calculer les probabilités d'échantillonnage
            probs = np.array(priorities) ** self.alpha
            probs_sum = probs.sum()
            if probs_sum > 0:
                probs /= probs_sum
            else:
                probs = np.ones(len(priorities)) / len(priorities)

            # Échantillonner les indices
            indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

            # Calculer les poids d'importance
            weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
            if len(weights) > 0:
                weights /= weights.max()  # Normaliser

            # Extraire le batch (liste de dictionnaires)
            batch_list: List[Dict[str, Any]] = []
            for idx in indices:
                batch_list.append(self.buffer[idx])

            # Regrouper par clé pour retourner un dictionnaire de tableaux
            batch: Dict[str, Any] = {}
            if len(batch_list) > 0:
                keys = set().union(*[exp.keys() for exp in batch_list])
                for k in keys:
                    values = [exp.get(k) for exp in batch_list]
                    # Convertir en numpy array lorsque c'est possible
                    try:
                        batch[k] = np.asarray(values)
                    except Exception:
                        batch[k] = values

            # Mettre à jour beta
            self.beta = min(1.0, self.beta + self.beta_increment)

        # Mettre à jour les statistiques d'échantillonnage
        with self._lock:
            self._shared_state['num_samples'] += batch_size
            self._shared_state['total_sampled'] += batch_size
            self._shared_state['last_sample_time'] = time.time()

            # Notifier l'orchestrateur des échantillons utilisés
            if hasattr(self, 'orchestrator'):
                self.orchestrator.metrics['buffer_samples_used'] += batch_size

        return batch, indices, weights

    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """
        Met à jour les priorités des expériences spécifiées de manière thread-safe.

        Args:
            indices: Liste des indices des expériences à mettre à jour
            priorities: Nouvelles priorités (doivent être de même longueur que les indices)

        Raises:
            ValueError: Si les longueurs des indices et des priorités ne correspondent pas
        """
        with self._lock:
            if len(indices) != len(priorities):
                raise ValueError(
                    f"Le nombre d'indices ({len(indices)}) doit correspondre "
                    f"au nombre de priorités ({len(priorities)})"
                )

            for idx, priority in zip(indices, priorities):
                # S'assurer que l'indice est valide
                if 0 <= idx < len(self.priorities):
                    # Mettre à jour la priorité avec la formule standard du PER
                    priority = (abs(priority) + self.epsilon) ** self.alpha
                    self.priorities[idx] = priority
                    self._shared_state['max_priority'] = max(self._shared_state['max_priority'], priority)

    def _update_priority(self, idx: int, priority: float) -> None:
        """
        Met à jour la priorité d'une expérience.

        Args:
            idx: Index de l'expérience à mettre à jour
            priority: Nouvelle priorité
        """
        with self._lock:
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
                # Mettre à jour la priorité maximale si nécessaire
                self._shared_state['max_priority'] = max(self._shared_state['max_priority'], priority)

    def _get_priority_weights(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Récupère les priorités et les poids normalisés pour les indices donnés.

        Cette méthode est principalement utilisée pour les tests.

        Args:
            indices: Liste des indices pour lesquels récupérer les poids

        Returns:
            Un tuple (priorities, weights) contenant les priorités brutes et les poids normalisés
        """
        with self._lock:
            # Récupérer les priorités pour les indices demandés
            priorities = np.array([self.priorities[i] for i in indices])

            # Calculer les poids d'importance (comme dans la méthode sample)
            if len(priorities) > 0:
                # Calculer les probabilités d'échantillonnage
                probs = priorities ** self.alpha
                probs_sum = probs.sum()
                if probs_sum > 0:
                    probs /= probs_sum
                else:
                    probs = np.ones_like(priorities) / len(priorities)

                # Calculer les poids d'importance
                weights = (len(self.buffer) * probs) ** (-self.beta)
                if len(weights) > 0:
                    weights /= weights.max()  # Normaliser

                return priorities, weights
            else:
                weights = np.array([])

            return priorities, weights

    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne des statistiques détaillées sur le buffer.

        Returns:
            Dictionnaire contenant les statistiques détaillées
        """
        with self._lock:
            current_time = time.time()
            time_since_add = current_time - self._shared_state.get('last_add_time', current_time)
            time_since_sample = current_time - self._shared_state.get('last_sample_time', current_time)

            add_rate = (
                self._shared_state.get('total_added', 0) /
                max(1, (current_time - self._shared_state.get('creation_time', current_time)))
            )

            sample_rate = (
                self._shared_state.get('total_sampled', 0) /
                max(1, (current_time - self._shared_state.get('creation_time', current_time)))
            )

            return {
                'size': len(self.buffer),
                'max_size': self.buffer_size,
                'num_additions': self._shared_state.get('num_additions', 0),
                'num_samples': self._shared_state.get('num_samples', 0),
                'total_added': self._shared_state.get('total_added', 0),
                'total_sampled': self._shared_state.get('total_sampled', 0),
                'seconds_since_last_add': time_since_add,
                'seconds_since_last_sample': time_since_sample,
                'add_rate_per_second': add_rate,
                'sample_rate_per_second': sample_rate,
                'utilization_percent': (len(self.buffer) / self.buffer_size) * 100 if self.buffer_size > 0 else 0,
                'priority_max': self._shared_state.get('max_priority', 0.0),
                'beta': self.beta
            }

    def _make_serializable(self, obj: Any) -> Any:
        """
        Convertit un objet en une forme sérialisable pour le partage entre processus.

        Args:
            obj: L'objet à sérialiser

        Returns:
            Une version sérialisable de l'objet
        """
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return [self._make_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        # Essayer de convertir en type natif Python
        try:
            return float(obj)
        except (TypeError, ValueError):
            return str(obj)

    def save(self, path: str) -> None:
        """Sauvegarde le buffer sur le disque."""
        with self._lock:
            # Créer le répertoire parent s'il n'existe pas
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Préparer les données à sauvegarder
            data = {
                'buffer': list(self.buffer),
                'priorities': list(self.priorities),
                'pos': self._shared_state['pos'],
                'alpha': self.alpha,
                'beta': self.beta,
                'beta_increment': self.beta_increment,
                'epsilon': self.epsilon,
                'stats': dict(self._shared_state)
            }

            # Sauvegarder avec pickle
            with open(path, 'wb') as f:
                pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'SharedExperienceBuffer':
        """Charge un buffer à partir du disque."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        # Créer une nouvelle instance
        buffer = cls(
            buffer_size=len(data['buffer']),  # Taille initiale basée sur les données chargées
            alpha=data.get('alpha', 0.6),
            beta=data.get('beta', 0.4),
            beta_increment=data.get('beta_increment', 0.001),
            epsilon=data.get('epsilon', 1e-6)
        )

        # Restaurer les données
        with buffer._lock:
            buffer.buffer = buffer.manager.list(data['buffer'])
            buffer.priorities = buffer.manager.list(data['priorities'])
            buffer._shared_state.update(data.get('stats', {}))

        return buffer
