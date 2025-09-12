"""
Module pour la fusion des poids des modèles dans un environnement d'apprentissage distribué.

Ce module fournit des fonctions pour fusionner les poids de plusieurs modèles PPO
et les redistribuer pour la synchronisation entre les workers.
"""

import torch
import numpy as np
from typing import Dict, List, Any
from stable_baselines3 import PPO


def average_weights(models: List[PPO]) -> Dict[str, torch.Tensor]:
    """
    Calcule la moyenne des poids de plusieurs modèles PPO.

    Args:
        models: Liste des modèles PPO dont les poids doivent être moyennés.

    Returns:
        Un dictionnaire contenant la moyenne des poids pour chaque paramètre.

    Raises:
        ValueError: Si la liste des modèles est vide.
    """
    if not models:
        raise ValueError("La liste des modèles ne peut pas être vide")

    # Récupère les noms des paramètres du premier modèle
    param_keys = models[0].policy.state_dict().keys()
    
    # Initialise un dictionnaire pour stocker les poids moyens
    avg_weights = {}
    
    # Pour chaque paramètre du modèle
    for key in param_keys:
        # Récupère les poids de chaque modèle pour ce paramètre
        weights = [model.policy.state_dict()[key].float() for model in models]
        
        # Calcule la moyenne des poids
        if len(weights) > 0:
            if isinstance(weights[0], torch.Tensor):
                # Pour les tenseurs PyTorch
                avg_weights[key] = torch.stack(weights, dim=0).mean(dim=0)
            else:
                # Pour les tableaux NumPy
                avg_weights[key] = np.mean(weights, axis=0)
    
    return avg_weights


def set_weights(model: PPO, weights: Dict[str, torch.Tensor]) -> None:
    """
    Applique un dictionnaire de poids à un modèle PPO.

    Args:
        model: Le modèle PPO à mettre à jour.
        weights: Dictionnaire des poids à appliquer.
    """
    # Charge l'état actuel du modèle
    model_state = model.policy.state_dict()
    
    # Met à jour les poids
    for key, value in weights.items():
        if key in model_state:
            # Vérifie que les dimensions correspondent
            if model_state[key].shape == value.shape:
                model_state[key].copy_(value)
            else:
                raise ValueError(
                    f"Dimension incompatible pour le paramètre {key}: "
                    f"attendu {model_state[key].shape}, reçu {value.shape}"
                )
    
    # Charge les poids mis à jour dans le modèle
    model.policy.load_state_dict(model_state)


def sync_models(models: List[PPO]) -> None:
    """
    Synchronise les poids d'une liste de modèles en calculant la moyenne des poids.
    
    Args:
        models: Liste des modèles à synchroniser.
    """
    if len(models) < 2:
        return  # Pas besoin de synchroniser un seul modèle
    
    # Calcule les poids moyens
    avg_weights = average_weights(models)
    
    # Applique les poids moyens à tous les modèles
    for model in models:
        set_weights(model, avg_weights)
