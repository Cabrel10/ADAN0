"""
HyperparameterOptimizer - Optimisation automatisée des hyperparamètres avec Optuna.

Ce module implémente une interface pour optimiser les hyperparamètres des modèles
de trading en utilisant Optuna, avec support pour l'arrêt anticipé et l'élagage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import optuna
from optuna.pruners import BasePruner, MedianPruner
from optuna.samplers import BaseSampler, TPESampler
from optuna.study import Study
from optuna.trial import Trial

# Configuration du logger
logger = logging.getLogger(__name__)


@dataclass
class HyperparameterOptimizer:
    """
    Classe pour l'optimisation des hyperparamètres utilisant Optuna.

    Attributes:
        study_name: Nom de l'étude Optuna
        storage_url: URL de stockage pour les études (par défaut: sqlite:///optuna_studies.db)
        n_trials: Nombre d'essais d'optimisation à effectuer
        timeout: Délai maximum en secondes pour l'optimisation (None pour illimité)
        direction: Direction d'optimisation ('minimize' ou 'maximize')
        sampler: Échantillonneur Optuna (par défaut: TPESampler)
        pruner: Élagueur Optuna (par défaut: MedianPruner)
        n_jobs: Nombre de jobs parallèles (-1 pour utiliser tous les cœurs)
    """

    study_name: str = "adan_hyperparameter_study"
    storage_url: str = "sqlite:///optuna_studies.db"
    n_trials: int = 100
    timeout: Optional[int] = 3600  # 1 heure par défaut
    direction: str = "maximize"
    sampler: Optional[BaseSampler] = None
    pruner: Optional[BasePruner] = None
    n_jobs: int = -1

    # Paramètres par défaut
    _study: Optional[Study] = field(init=False, default=None)

    def __post_init__(self):
        """Initialisation des composants Optuna avec des valeurs par défaut si non spécifiés."""
        if self.sampler is None:
            self.sampler = TPESampler(n_startup_trials=10, n_ei_candidates=24)
        if self.pruner is None:
            self.pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    def create_study(self) -> Study:
        """Crée ou charge une étude Optuna."""
        try:
            storage = optuna.storages.get_storage(self.storage_url)
            study = optuna.create_study(
                study_name=self.study_name,
                storage=storage,
                sampler=self.sampler,
                pruner=self.pruner,
                direction=self.direction,
                load_if_exists=True
            )
            logger.info(f"Étude '{self.study_name}' chargée/créée avec succès")
            return study
        except Exception as e:
            logger.error(f"Erreur lors de la création/chargement de l'étude: {e}")
            raise

    def optimize(
        self,
        objective: Callable[[Trial], float],
        param_distributions: Dict[str, Any],
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Study:
        """
        Exécute l'optimisation des hyperparamètres.

        Args:
            objective: Fonction objectif à optimiser
            param_distributions: Dictionnaire des distributions de paramètres
            n_trials: Nombre d'essais (remplace la valeur de l'instance si spécifié)
            timeout: Délai maximum en secondes (remplace la valeur de l'instance si spécifié)
            **kwargs: Arguments additionnels pour la fonction objectif

        Returns:
            L'étude Optuna complétée
        """
        n_trials = n_trials or self.n_trials
        timeout = timeout or self.timeout

        # Enveloppe la fonction objectif pour passer les arguments supplémentaires
        def wrapped_objective(trial):
            return objective(trial, **kwargs)

        try:
            study = self.create_study()
            study.optimize(
                wrapped_objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=self.n_jobs
            )
            return study
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation: {e}")
            raise

    @staticmethod
    def suggest_hyperparameters(trial: Trial, param_distributions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggère des valeurs d'hyperparamètres pour un essai donné.

        Args:
            trial: Essai Optuna en cours
            param_distributions: Dictionnaire des distributions de paramètres

        Returns:
            Dictionnaire des valeurs d'hyperparamètres suggérées
        """
        params = {}

        for name, distribution in param_distributions.items():
            if not isinstance(distribution, dict):
                params[name] = distribution
                continue

            dist_type = distribution.get('type')

            if dist_type == 'categorical':
                params[name] = trial.suggest_categorical(name, distribution['choices'])
            elif dist_type == 'int':
                params[name] = trial.suggest_int(
                    name,
                    low=distribution['low'],
                    high=distribution['high'],
                    step=distribution.get('step', 1),
                    log=distribution.get('log', False)
                )
            elif dist_type == 'float':
                params[name] = trial.suggest_float(
                    name,
                    low=distribution['low'],
                    high=distribution['high'],
                    step=distribution.get('step'),
                    log=distribution.get('log', False)
                )
            elif dist_type == 'loguniform':
                params[name] = trial.suggest_float(
                    name,
                    low=distribution['low'],
                    high=distribution['high'],
                    log=True
                )
            else:
                raise ValueError(f"Type de distribution non supporté: {dist_type}")

        return params

    def get_best_params(self, study: Optional[Study] = None) -> Dict[str, Any]:
        """
        Récupère les meilleurs paramètres d'une étude.

        Args:
            study: Étude Optuna (si None, charge l'étude actuelle)

        Returns:
            Dictionnaire des meilleurs paramètres
        """
        study = study or self.create_study()
        return study.best_params

    def get_best_trial(self, study: Optional[Study] = None) -> optuna.trial.FrozenTrial:
        """
        Récupère le meilleur essai d'une étude.

        Args:
            study: Étude Optuna (si None, charge l'étude actuelle)

        Returns:
            Meilleur essai
        """
        study = study or self.create_study()
        return study.best_trial
