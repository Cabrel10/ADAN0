"""
ModelEnsemble - Système d'ensemble de modèles pour le trading algorithmique.

Ce module implémente un système d'ensemble de modèles qui combine les prédictions
de plusieurs modèles individuels pour améliorer la robustesse et les performances.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path

# Import local
from .custom_cnn import CustomCNN  # noqa: F401


@dataclass
class ModelPerformance:
    """Classe pour suivre les performances d'un modèle individuel dans l'ensemble."""
    model_name: str
    weights: float = 1.0
    total_predictions: int = 0
    correct_predictions: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        """Calcule la précision actuelle du modèle."""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions

    def update_performance(self, prediction_correct: bool, **metadata) -> None:
        """Met à jour les statistiques de performance du modèle."""
        self.total_predictions += 1
        if prediction_correct:
            self.correct_predictions += 1
        self.last_updated = datetime.utcnow()
        self.metadata.update(metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit les performances en dictionnaire pour la sérialisation."""
        return {
            'model_name': self.model_name,
            'weights': self.weights,
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'last_updated': self.last_updated.isoformat(),
            'accuracy': self.accuracy,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelPerformance':
        """Crée une instance à partir d'un dictionnaire."""
        # Crée une copie pour éviter de modifier le dictionnaire d'origine
        data = data.copy()

        # Convertit la chaîne de date en objet datetime si nécessaire
        if 'last_updated' in data and isinstance(data['last_updated'], str):
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])

        # Supprime les clés qui ne sont pas des paramètres du constructeur
        data.pop('accuracy', None)

        return cls(**data)


class VotingMechanism:
    """Mécanisme de vote pour combiner les prédictions de plusieurs modèles."""

    def __init__(self, method: str = 'weighted'):
        """
        Initialise le mécanisme de vote.

        Args:
            method: Méthode de vote ('majority', 'weighted', 'average')

        Raises:
            ValueError: Si la méthode spécifiée n'est pas valide
        """
        if method not in ['majority', 'weighted', 'average']:
            raise ValueError(
                f"Méthode de vote non supportée: {method}. "
                "Les méthodes valides sont: 'majority', 'weighted', 'average'"
            )
        self.method = method

    def combine_predictions(self, predictions: List[torch.Tensor],
                          weights: Optional[List[float]] = None) -> torch.Tensor:
        """
        Combine les prédictions de plusieurs modèles.

        Args:
            predictions: Liste des tenseurs de prédictions des modèles
            weights: Poids de chaque modèle (optionnel)

        Returns:
            Prédiction combinée

        Raises:
            ValueError: Si la méthode de vote n'est pas supportée
        """
        if not predictions:
            raise ValueError("Aucune prédiction fournie")

        if weights is None:
            weights = [1.0] * len(predictions)

        if len(predictions) != len(weights):
            raise ValueError("Le nombre de prédictions doit correspondre au nombre de poids")

        # Vérifie que la méthode est valide
        if self.method not in ['majority', 'weighted', 'average']:
            raise ValueError(f"Méthode de vote non supportée: {self.method}. "
                           "Les méthodes valides sont: 'majority', 'weighted', 'average'")

        if self.method == 'majority':
            # Vote majoritaire (pour la classification)
            stacked = torch.stack(predictions, dim=0)
            return torch.mode(stacked, dim=0).values

        elif self.method == 'weighted':
            # Moyenne pondérée (pour la régression)
            weights_tensor = torch.tensor(weights,
                                       device=predictions[0].device,
                                       dtype=predictions[0].dtype)
            weights_tensor = weights_tensor / weights_tensor.sum()  # Normalisation

            weighted_sum = torch.zeros_like(predictions[0])
            for pred, weight in zip(predictions, weights_tensor):
                weighted_sum += pred * weight

            return weighted_sum

        elif self.method == 'average':
            # Moyenne simple
            return torch.mean(torch.stack(predictions, dim=0), dim=0)


class ModelEnsemble:
    """Classe principale pour gérer un ensemble de modèles de trading."""

    def __init__(self, voting_method: str = 'weighted',
                 performance_file: Optional[str] = None):
        """
        Initialise l'ensemble de modèles.

        Args:
            voting_method: Méthode de vote ('majority', 'weighted', 'average')
            performance_file: Chemin vers le fichier de sauvegarde des performances
        """
        self.models: Dict[str, nn.Module] = {}
        self.performance: Dict[str, ModelPerformance] = {}
        self.voting_mechanism = VotingMechanism(method=voting_method)
        self.performance_file = performance_file

        if performance_file and os.path.exists(performance_file):
            self._load_performance()

    def add_model(self, model: nn.Module, model_name: str,
                 initial_weight: float = 1.0, **metadata) -> None:
        """
        Ajoute un modèle à l'ensemble.

        Args:
            model: Modèle PyTorch à ajouter
            model_name: Nom unique du modèle
            initial_weight: Poids initial pour le modèle
            **metadata: Métadonnées supplémentaires
        """
        if model_name in self.models:
            raise ValueError(f"Un modèle avec le nom '{model_name}' existe déjà")

        self.models[model_name] = model

        # Si des performances existent déjà pour ce modèle (chargées depuis un fichier), on les conserve
        if model_name not in self.performance:
            self.performance[model_name] = ModelPerformance(
                model_name=model_name,
                weights=initial_weight,
                metadata=metadata
            )

    def remove_model(self, model_name: str) -> None:
        """Supprime un modèle de l'ensemble."""
        if model_name in self.models:
            del self.models[model_name]
            del self.performance[model_name]

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Effectue une prédiction en utilisant l'ensemble des modèles.

        Args:
            x: Tenseur d'entrée

        Returns:
            Prédiction combinée de l'ensemble
        """
        if not self.models:
            raise ValueError("Aucun modèle n'a été ajouté à l'ensemble")

        predictions = []
        weights = []

        with torch.no_grad():
            for name, model in self.models.items():
                pred = model(x)
                predictions.append(pred)
                weights.append(self.performance[name].weights)

        # Combinaison des prédictions selon la méthode de vote
        return self.voting_mechanism.combine_predictions(predictions, weights)

    def update_weights(self, model_name: str, new_weight: float) -> None:
        """Met à jour le poids d'un modèle."""
        if model_name not in self.performance:
            raise ValueError(f"Modèle inconnu: {model_name}")

        self.performance[model_name].weights = max(0.0, new_weight)

    def update_performance(self, model_name: str, prediction_correct: bool, **metadata) -> None:
        """Met à jour les performances d'un modèle."""
        if model_name not in self.performance:
            raise ValueError(f"Modèle inconnu: {model_name}")

        self.performance[model_name].update_performance(prediction_correct, **metadata)

    def get_best_model(self) -> Tuple[Optional[str], float]:
        """Retourne le nom et la précision du meilleur modèle."""
        if not self.performance:
            return None, 0.0

        best_name = max(
            self.performance.items(),
            key=lambda x: x[1].accuracy
        )[0]
        return best_name, self.performance[best_name].accuracy

    def save_performance(self, filepath: Optional[str] = None) -> None:
        """Sauvegarde les performances des modèles dans un fichier JSON."""
        filepath = filepath or self.performance_file
        if not filepath:
            raise ValueError("Aucun fichier de performance spécifié")

        data = {
            'version': '1.0',
            'last_updated': datetime.utcnow().isoformat(),
            'models': {}
        }

        for name, perf in self.performance.items():
            data['models'][name] = perf.to_dict()

        # Création du répertoire parent si nécessaire
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_performance(self) -> None:
        """Charge les performances depuis un fichier JSON.

        Les performances sont chargées pour tous les modèles présents dans le fichier,
        même s'ils ne sont pas encore dans l'ensemble. Cela permet de conserver l'historique
        même si les modèles sont ajoutés ou supprimés ultérieurement.
        """
        if not self.performance_file or not os.path.exists(self.performance_file):
            return

        try:
            with open(self.performance_file, 'r') as f:
                data = json.load(f)

            # Vérifie la version du format
            if not isinstance(data, dict) or 'models' not in data:
                print("Format de fichier de performances invalide")
                return

            # Charge les performances pour chaque modèle
            for name, perf_data in data.get('models', {}).items():
                try:
                    # Crée ou met à jour l'entrée de performance
                    if name in self.performance:
                        # Si le modèle existe déjà, on met à jour ses performances
                        self.performance[name] = ModelPerformance.from_dict({
                            'model_name': name,
                            'weights': perf_data.get('weights', 1.0),
                            'total_predictions': perf_data.get('total_predictions', 0),
                            'correct_predictions': perf_data.get('correct_predictions', 0),
                            'last_updated': perf_data.get('last_updated', datetime.utcnow().isoformat()),
                            'metadata': perf_data.get('metadata', {})
                        })
                    else:
                        # Sinon, on crée une nouvelle entrée
                        self.performance[name] = ModelPerformance.from_dict({
                            'model_name': name,
                            'weights': perf_data.get('weights', 1.0),
                            'total_predictions': perf_data.get('total_predictions', 0),
                            'correct_predictions': perf_data.get('correct_predictions', 0),
                            'last_updated': perf_data.get('last_updated', datetime.utcnow().isoformat()),
                            'metadata': perf_data.get('metadata', {})
                        })
                except Exception as e:
                    print(f"Erreur lors du chargement des performances pour {name}: {e}")

        except (json.JSONDecodeError, IOError) as e:
            print(f"Erreur lors du chargement du fichier de performances: {e}")
        except Exception as e:
            print(f"Erreur inattendue lors du chargement des performances: {e}")
            raise  # Propager l'erreur pour faciliter le débogage

    def __len__(self) -> int:
        """Retourne le nombre de modèles dans l'ensemble."""
        return len(self.models)

    def __contains__(self, model_name: str) -> bool:
        """Vérifie si un modèle fait partie de l'ensemble."""
        return model_name in self.models
