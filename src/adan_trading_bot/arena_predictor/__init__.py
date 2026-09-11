"""V18 — Arena Bayésien Prédictif.

Microservice isolé (comme future_arena) : remplace les constantes de
calibration arbitraires par un modèle prédictif entraîné hors-ligne sur les
statistiques ex-post du Future Arena.

Pipeline :
    Collector   (present_state -> params optimaux ex-post, écrit en JSONL)
        |  entraînement hors-ligne (scripts/train_arena_predictor.py)
        v
    Predictor   (MLP hétéroscédastique : state -> distribution des params)
        |  inférence live-safe
        v
    Estimator   (estimate_break_even / tp / sl / duration / confidence)

GARANTIE : le futur ne sert QU'À fabriquer les cibles hors-ligne. En live,
seul l'état présent est fourni au modèle.
"""
from __future__ import annotations

from .state_schema import (
    PresentState,
    TrainingSample,
    STATE_FEATURES,
    STATE_DIM,
    STATE_SCHEMA_VERSION,
    TARGET_PARAMS,
    TARGET_DIM,
)
from .collector import ArenaCollector
from .estimator import ArenaEstimator

__all__ = [
    "PresentState",
    "TrainingSample",
    "STATE_FEATURES",
    "STATE_DIM",
    "STATE_SCHEMA_VERSION",
    "TARGET_PARAMS",
    "TARGET_DIM",
    "ArenaCollector",
    "ArenaEstimator",
    # Predictor / TargetScaler importés paresseusement (dépendent de torch) :
    "get_predictor_classes",
]

__schema_version__ = STATE_SCHEMA_VERSION


def get_predictor_classes():
    """Import paresseux des classes qui dépendent de torch.

    Évite d'imposer torch aux consommateurs qui n'ont besoin que du Collector
    ou de l'Estimator en repli.
    """
    from .predictor import (
        ArenaPredictor,
        TargetScaler,
        gaussian_nll,
        save_predictor,
        load_predictor,
    )
    return {
        "ArenaPredictor": ArenaPredictor,
        "TargetScaler": TargetScaler,
        "gaussian_nll": gaussian_nll,
        "save_predictor": save_predictor,
        "load_predictor": load_predictor,
    }
