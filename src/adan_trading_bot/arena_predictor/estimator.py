"""V18 — Arena Bayésien Prédictif : Estimator (inférence LIVE-safe).

L'Estimator est le point d'entrée que l'environnement de trading utilise EN
LIGNE pour remplacer les constantes arbitraires (ADAN_BARRIER_MULT, profils
TP/SL fixes, MAX_DURATION, max_future_contrib) par des estimations dérivées
d'un modèle appris.

    estimate(present_state) -> {break_even, tp, sl, duration, confidence,
                                *_std}   (avec incertitude)

Comportement bayésien concret :
  - Chaque param vient avec un écart-type (std). Une std élevée signifie
    « régime inconnu » -> l'appelant peut être PLUS PRUDENT (barrière plus
    haute, TP plus conservateur, taille réduite).
  - `estimate_break_even()` renvoie une barrière ADAPTATIVE : mean + k*std,
    plancher aux frais A/R. Remplace la barrière statique 1.5*fees.

Chargement paresseux et tolérant : si aucun modèle n'est disponible (début du
tout premier run, avant que le predictor n'ait été entraîné), l'Estimator
tombe en repli sur des heuristiques sûres basées sur les frais A/R -> jamais
de crash, dégradation gracieuse.

RÈGLE D'OR : l'entrée est le PRÉSENT uniquement. Aucun futur.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

from .state_schema import PresentState, TARGET_PARAMS

_DEFAULT_MODEL_PATH = "models/arena_predictor/arena_predictor.pt"


class ArenaEstimator:
    """Wrapper d'inférence live-safe autour d'ArenaPredictor."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        round_trip_fees: float = 0.004,
        uncertainty_k: float = 0.5,
        mc_samples: int = 1,
        enabled: Optional[bool] = None,
    ):
        # Activé par ADAN_ARENA_PREDICT=1 (ou explicitement). Désactivé -> repli.
        if enabled is None:
            enabled = os.environ.get("ADAN_ARENA_PREDICT", "0") == "1"
        self.enabled = bool(enabled)
        self.model_path = model_path or os.environ.get(
            "ADAN_ARENA_MODEL_PATH", _DEFAULT_MODEL_PATH
        )
        self.round_trip_fees = float(round_trip_fees)
        self.uncertainty_k = float(uncertainty_k)
        self.mc_samples = int(os.environ.get("ADAN_ARENA_MC", str(mc_samples)))
        self._model = None
        self._scaler = None
        self._torch = None
        self._loaded = False
        self._load_failed = False

    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        if self._load_failed or not self.enabled:
            return False
        try:
            import torch  # lazy
            from .predictor import load_predictor

            if not os.path.exists(self.model_path):
                self._load_failed = True
                return False
            self._model, self._scaler = load_predictor(self.model_path)
            self._torch = torch
            self._loaded = True
            return True
        except Exception:
            self._load_failed = True
            return False

    # ------------------------------------------------------------------ #
    def _fallback(self) -> Dict[str, float]:
        """Heuristiques sûres basées sur les frais A/R (aucun modèle)."""
        rtf = self.round_trip_fees
        out = {
            "break_even": rtf,
            "tp": rtf * 3.0,
            "sl": rtf * 2.0,
            "duration": 20.0,
            "confidence": 0.5,
        }
        for p in TARGET_PARAMS:
            out[f"{p}_std"] = 0.0  # repli = certitude nulle assumée
        out["_source"] = "fallback"
        return out

    # ------------------------------------------------------------------ #
    def estimate(self, state: PresentState) -> Dict[str, float]:
        """Estime les 5 params + leur std à partir de l'état présent."""
        if not self._ensure_loaded():
            return self._fallback()
        try:
            torch = self._torch
            x = torch.tensor([state.to_vector()], dtype=torch.float32)
            mean_z, std_z = self._model.predict_dist(x, mc_samples=self.mc_samples)
            mean = self._scaler.inverse_mean(mean_z)[0]
            std = self._scaler.inverse_std(std_z)[0]
            out: Dict[str, float] = {}
            for i, p in enumerate(TARGET_PARAMS):
                out[p] = float(mean[i].item())
                out[f"{p}_std"] = float(abs(std[i].item()))
            # Contraintes de sécurité (les cibles sont des fractions positives).
            rtf = self.round_trip_fees
            out["break_even"] = max(rtf, out["break_even"])
            out["tp"] = max(rtf * 1.2, out["tp"])
            out["sl"] = max(rtf, min(0.05, out["sl"]))
            out["duration"] = max(1.0, out["duration"])
            out["confidence"] = max(0.0, min(1.0, out["confidence"]))
            out["_source"] = "model"
            return out
        except Exception:
            return self._fallback()

    # ------------------------------------------------------------------ #
    # Accès directs utilisés par l'environnement (remplacent les constantes)
    # ------------------------------------------------------------------ #
    def estimate_break_even(self, state: PresentState) -> float:
        """Barrière de rentabilité ADAPTATIVE = mean + k*std, plancher = frais.

        Remplace la barrière statique 1.5*fees et le multiplicateur ATR ad hoc.
        Plus l'incertitude (std) est grande, plus la barrière est prudente.
        """
        est = self.estimate(state)
        be = est.get("break_even", self.round_trip_fees)
        std = est.get("break_even_std", 0.0)
        adaptive = be + self.uncertainty_k * std
        return float(max(self.round_trip_fees, min(0.02, adaptive)))

    def estimate_tp_sl(self, state: PresentState) -> Dict[str, float]:
        est = self.estimate(state)
        return {"tp": est["tp"], "sl": est["sl"],
                "tp_std": est.get("tp_std", 0.0), "sl_std": est.get("sl_std", 0.0)}

    def estimate_duration(self, state: PresentState) -> float:
        return float(self.estimate(state).get("duration", 20.0))

    def estimate_confidence(self, state: PresentState) -> float:
        return float(self.estimate(state).get("confidence", 0.5))

    @property
    def is_active(self) -> bool:
        """True si un modèle est réellement chargé (pas en repli)."""
        return self._ensure_loaded()
