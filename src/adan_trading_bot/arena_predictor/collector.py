"""V18 — Arena Bayésien Prédictif : Collector.

Pendant l'entraînement, pour chaque trade clôturé, le Future Arena connaît
EX-POST le MFE / MAE / durée réels. Le Collector transforme ces observations
en CIBLES optimales et les associe à l'état PRÉSENT à l'ouverture :

    (present_state) -> (break_even*, tp*, sl*, duration*, confidence*)

Ces tuples sont écrits en JSONL (append-only, thread-safe best-effort). Un run
de 500k pas produit des dizaines de milliers d'échantillons, à partir desquels
le Predictor (predictor.py) apprend hors-ligne les distributions.

GARANTIE LIVE-SAFE : le futur ne sert QU'À fabriquer les cibles ci-dessous.
Il n'entre jamais dans `present_state`. En live, le Predictor n'a besoin que
de l'état présent.
"""
from __future__ import annotations

import io
import json
import os
import threading
from typing import Dict, Optional

from .state_schema import PresentState, TrainingSample


class ArenaCollector:
    """Écrit des échantillons (état présent -> params optimaux) en JSONL."""

    def __init__(self, out_path: Optional[str] = None, enabled: Optional[bool] = None):
        # Activé par env-var ADAN_ARENA_COLLECT=1 (ou explicitement).
        if enabled is None:
            enabled = os.environ.get("ADAN_ARENA_COLLECT", "0") == "1"
        self.enabled = bool(enabled)
        if out_path is None:
            out_path = os.environ.get(
                "ADAN_ARENA_COLLECT_PATH",
                "logs/arena/arena_samples.jsonl",
            )
        self.out_path = out_path
        self._lock = threading.Lock()
        self._count = 0
        if self.enabled:
            try:
                os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
            except Exception:
                self.enabled = False

    @staticmethod
    def optimal_params_from_future(
        *,
        entry_price: float,
        mfe: float,
        mae: float,
        steps_held: int,
        mfe_residual: Optional[float],
        round_trip_fees: float,
        pnl_net: float,
    ) -> Dict[str, float]:
        """Dérive les paramètres OPTIMAUX à partir des observations ex-post.

        - break_even* : plancher = frais A/R (impossible d'être rentable en-deçà).
          Si MFE est faible (marché plat), break-even reste au plancher.
        - tp*  = MFE réel (le take-profit idéal = l'excursion favorable atteinte).
        - sl*  = MAE réel borné (le stop idéal = pire excursion tolérée).
        - duration* = steps_held observé (durée réelle du cycle profitable),
          + bonus si mfe_residual élevé (aurait fallu tenir plus longtemps).
        - confidence* = 1 si le trade a été net-profitable, sinon proba douce.
        Toutes valeurs en fraction du prix (sauf duration en steps).
        """
        mfe = max(0.0, float(mfe or 0.0))
        mae = abs(float(mae or 0.0))
        rtf = float(round_trip_fees or 0.004)
        resid = float(mfe_residual or 0.0)

        break_even = max(rtf, rtf)  # plancher = frais; jamais en-dessous
        tp = max(rtf * 1.5, mfe) if mfe > 0 else rtf * 2.0
        sl = max(rtf, min(mae, 0.05)) if mae > 0 else rtf * 2.0
        # Si le marché a continué de monter après la sortie (resid grand),
        # la durée optimale aurait été plus longue.
        dur = float(max(1, steps_held))
        if resid > mfe and mfe > 0:
            dur *= min(2.0, 1.0 + (resid - mfe) / max(mfe, 1e-6))
        confidence = 1.0 if pnl_net > 0 else max(0.0, min(1.0, 0.5 + pnl_net / max(rtf, 1e-6)))
        return {
            "break_even": float(break_even),
            "tp": float(tp),
            "sl": float(sl),
            "duration": float(dur),
            "confidence": float(confidence),
        }

    def record(self, state: PresentState, params: Dict[str, float],
               meta: Optional[Dict] = None) -> None:
        if not self.enabled:
            return
        try:
            sample = TrainingSample(
                state=state.to_vector(),
                break_even=float(params.get("break_even", 0.004)),
                tp=float(params.get("tp", 0.008)),
                sl=float(params.get("sl", 0.008)),
                duration=float(params.get("duration", 20.0)),
                confidence=float(params.get("confidence", 0.5)),
                meta=meta or {},
            )
            line = json.dumps(sample.to_dict(), separators=(",", ":"))
            with self._lock:
                with io.open(self.out_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                self._count += 1
        except Exception:
            # Best-effort: la collecte ne doit JAMAIS casser l'entraînement.
            pass

    @property
    def count(self) -> int:
        return self._count
