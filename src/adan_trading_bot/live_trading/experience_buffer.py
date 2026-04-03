"""
Prioritized Experience Replay Buffer pour l'apprentissage par renforcement.

Ce module implémente un buffer d'expérience avec échantillonnage prioritaire (PER)
pour accélérer l'apprentissage en se concentrant sur les expériences les plus instructives.
"""

import numpy as np
import random
import heapq
from typing import Dict, List, Tuple, Any, Optional
import torch
import pickle
import os
from pathlib import Path

class PrioritizedExperienceReplayBuffer:
    """
    Buffer d'expérience avec échantillonnage prioritaire (Prioritized Experience Replay).
    
    Les expériences sont stockées avec une priorité basée sur l'erreur de prédiction (TD-error).
    Les expériences avec une erreur plus élevée ont une plus grande probabilité d'être échantillonnées.
    """
    
    def __init__(self, buffer_size: int = 10000, alpha: float = 0.6, beta: float = 0.4, 
                 beta_increment: float = 0.001, epsilon: float = 1e-6):
        """
        Initialise le buffer d'expérience prioritaire.
        
        Args:
            buffer_size: Taille maximale du buffer
            alpha: Contrôle l'importance des priorités (0 = uniforme, 1 = pleinement prioritaire)
            beta: Contrôle l'importance de l'échantillonnage par importance (0 = non corrigé, 1 = pleinement corrigé)
            beta_increment: Incrément progressif de beta vers 1.0
            epsilon: Petite constante pour éviter des priorités nulles
        """
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.buffer = []
        self.priorities = []
        self.pos = 0
        self.max_priority = 1.0
        
        # Pour l'échantillonnage efficace
        self._it_sum = []
        self._it_min = []
        self._max_priority = 1.0
        
        # Statistiques
        self.stats = {
            'num_additions': 0,
            'num_samples': 0,
            'avg_priority': 0.0,
            'max_priority': 0.0,
            'min_priority': float('inf'),
            'buffer_size': 0
        }
    
    def __len__(self) -> int:
        """Retourne le nombre d'expériences actuellement dans le buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Vérifie si le buffer contient suffisamment d'expériences pour un batch."""
        return len(self.buffer) >= batch_size
    
    def add(self, experience: Dict[str, Any], priority: Optional[float] = None) -> None:
        """
        Ajoute une expérience au buffer avec une priorité donnée.
        
        Si aucune priorité n'est fournie, utilise la priorité maximale actuelle.
        """
        if priority is None:
            priority = self._max_priority
        
        # Créer une copie profonde de l'expérience pour éviter les modifications ultérieures
        experience = {k: v.copy() if isinstance(v, (dict, list, np.ndarray)) else v 
                     for k, v in experience.items()}
        
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
            self.priorities.append(0.0)  # Initialisé à 0, sera mis à jour ci-dessous
        else:
            self.buffer[self.pos] = experience
            self.priorities[self.pos] = 0.0
        
        # Mettre à jour les priorités
        self._update_priority(self.pos, priority)
        
        # Mettre à jour la position pour le prochain ajout
        self.pos = (self.pos + 1) % self.buffer_size
        
        # Mettre à jour les statistiques
        self.stats['num_additions'] += 1
        self.stats['buffer_size'] = len(self.buffer)
    
    def sample(self, batch_size: int) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """
        Échantillonne un batch d'expériences de manière prioritaire.
        
        Returns:
            Un tuple contenant:
                - batch: Dictionnaire des expériences échantillonnées
                - indices: Indices des expériences échantillonnées
                - weights: Poids d'importance pour la correction de biais
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Tentative d'échantillonnage de {batch_size} expériences "
                          f"alors que le buffer n'en contient que {len(self.buffer)}")
        
        # Calculer les probabilités d'échantillonnage
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Échantillonner les indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Calculer les poids d'importance
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normaliser
        
        # Extraire le batch
        batch = {}
        for key in self.buffer[0].keys():
            batch[key] = np.array([self.buffer[idx][key] for idx in indices])
        
        # Mettre à jour beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Mettre à jour les statistiques
        self.stats['num_samples'] += batch_size
        self.stats['avg_priority'] = float(np.mean(priorities))
        self.stats['max_priority'] = float(np.max(priorities))
        self.stats['min_priority'] = float(np.min(priorities))
        
        return batch, indices, weights
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """
        Met à jour les priorités des expériences spécifiées.
        
        Args:
            indices: Liste des indices des expériences à mettre à jour
            priorities: Nouvelles priorités (doivent être de même longueur que les indices)
        """
        if len(indices) != len(priorities):
            raise ValueError(f"Le nombre d'indices ({len(indices)}) ne correspond pas "
                          f"au nombre de priorités ({len(priorities)})")
        
        # Ajouter un petit epsilon pour éviter les priorités nulles
        priorities = np.abs(priorities) + self.epsilon
        
        for idx, priority in zip(indices, priorities):
            if idx < 0 or idx >= len(self.buffer):
                continue
            
            # Mettre à jour la priorité
            self._update_priority(idx, priority)
    
    def _update_priority(self, idx: int, priority: float) -> None:
        """Met à jour la priorité d'une expérience et met à jour la priorité maximale."""
        self.priorities[idx] = priority
        self._max_priority = max(self._max_priority, priority)
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques actuelles du buffer."""
        return self.stats.copy()
    
    def save(self, path: str) -> None:
        """Sauvegarde le buffer sur le disque."""
        # Créer le répertoire parent s'il n'existe pas
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Préparer les données à sauvegarder
        data = {
            'buffer': self.buffer,
            'priorities': self.priorities,
            'pos': self.pos,
            'alpha': self.alpha,
            'beta': self.beta,
            'beta_increment': self.beta_increment,
            'epsilon': self.epsilon,
            'stats': self.stats
        }
        
        # Sauvegarder avec pickle
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'PrioritizedExperienceReplayBuffer':
        """Charge un buffer à partir du disque."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Créer une nouvelle instance
        buffer = cls(
            buffer_size=len(data['buffer']),  # Taille basée sur les données chargées
            alpha=data['alpha'],
            beta=data['beta'],
            beta_increment=data['beta_increment'],
            epsilon=data['epsilon']
        )
        
        # Restaurer l'état
        buffer.buffer = data['buffer']
        buffer.priorities = data['priorities']
        buffer.pos = data['pos']
        buffer.stats = data['stats']
        
        # Mettre à jour la priorité maximale
        if buffer.priorities:
            buffer._max_priority = max(buffer.priorities)
        
        return buffer
