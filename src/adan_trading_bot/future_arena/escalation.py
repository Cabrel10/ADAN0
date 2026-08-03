"""escalation — moteur de pénalité STATEFUL anti-motif (Lot C6).

Implémente la décision utilisateur §13.5 du cahier des charges :

  « Rien ne doit être prévisible/linéaire/statique pour le modèle. Une action
    n'est PAS pénalisée à la première occurrence. Si un motif se répète N fois
    sans apporter de rentabilité/qualité (ou mène à un crash), on le transforme
    PROGRESSIVEMENT en pénalité (escalation, pas un mur fixe). »

Pourquoi c'est crucial (cf. généalogie des exploits, cahier §2) : toute pénalité
CONSTANTE et PRÉVISIBLE finit contournée — le PPO mémorise le seuil et danse
juste en dessous. Une pénalité qui :
  - dépend de l'HISTORIQUE récent du motif (stateful),
  - croît NON-linéairement avec la répétition stérile,
  - a un seuil/pente légèrement STOCHASTIQUES,
… est beaucoup plus difficile à modéliser et donc à exploiter.

Module PUR (logique) mais STATEFUL par conception (c'est tout l'intérêt) :
l'état est ENCAPSULÉ dans une instance `EscalationTracker`, pas global. Aucune
I/O, aucun import du reste du bot. Reproductible via `seed`.

Garde-fou anti-prévisibilité : le bruit stochastique est borné et n'inverse
JAMAIS le signe de la pénalité (sinon on créerait un nouveau canal d'exploit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class EscalationConfig:
    """Paramètres du moteur d'escalation (par motif surveillé)."""

    # Nb de répétitions stériles tolérées avant que la pénalité démarre.
    grace: int = 3
    # Pénalité de base (magnitude) une fois la grâce dépassée. Négatif appliqué.
    base_penalty: float = 0.05
    # Exposant de croissance non-linéaire (>1 = accélère avec la répétition).
    growth_exponent: float = 1.6
    # Plafond absolu de la pénalité (sécurité : pas d'explosion du gradient).
    max_penalty: float = 3.0
    # Amplitude du bruit stochastique RELATIF (±frac) sur seuil/pente.
    noise_frac: float = 0.15
    # Décroissance du compteur quand le motif est INTERROMPU par du rentable.
    # 1.0 = reset complet ; 0.5 = on garde la moitié de la "dette".
    decay_on_good: float = 1.0
    # Décroissance passive par step (oubli lent même sans bon trade).
    passive_decay: float = 0.02


@dataclass
class _MotifState:
    """État interne d'un motif surveillé (compteur de répétition stérile)."""

    repetitions: float = 0.0
    last_penalty: float = 0.0
    total_penalty: float = 0.0
    triggers: int = 0


class EscalationTracker:
    """Suit plusieurs motifs et calcule leur pénalité d'escalation.

    Un « motif » est identifié par une clé libre (str), ex :
      - "micro_close"      : AGENT_CLOSE répété pour micro-gain
      - "hold_in_green"    : HOLD alors qu'on est en zone 🟢
      - "max_duration"     : position laissée mourir en MaxDuration
      - "oversize_red"     : grosse taille en zone 🔴

    Utilisation typique (par step) :
        pen = tracker.update("micro_close", sterile=True, severity=1.0)
        # pen <= 0 : pénalité à AJOUTER au reward (0 pendant la grâce)

    `sterile=True`  → le motif s'est répété SANS apporter de valeur → on escalade.
    `sterile=False` → le motif a produit quelque chose de bon → on décroît la dette.
    """

    def __init__(
        self,
        config: Optional[EscalationConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or EscalationConfig()
        self._rng = np.random.default_rng(seed)
        self._motifs: dict[str, _MotifState] = {}

    # ── API principale ──────────────────────────────────────────────────────
    def update(
        self,
        motif: str,
        sterile: bool,
        severity: float = 1.0,
    ) -> float:
        """Met à jour un motif et renvoie la pénalité (≤ 0) à appliquer ce step.

        - sterile=True : incrémente le compteur de répétition stérile.
        - sterile=False : le motif a été "racheté" par un bon résultat → decay.
        - severity : pondère l'intensité (ex. taille de la position en 🔴).

        Renvoie 0.0 tant que la grâce n'est pas dépassée. Au-delà, une pénalité
        négative croissante non linéaire, bornée, légèrement bruitée.
        """
        st = self._motifs.setdefault(motif, _MotifState())
        cfg = self.config

        if not sterile:
            # Le motif a produit de la valeur : on rembourse une partie de la dette.
            # decay_on_good=1.0 -> reset complet ; 0.5 -> on garde la moitié.
            st.repetitions = max(0.0, st.repetitions * (1.0 - cfg.decay_on_good))
            st.last_penalty = 0.0
            return 0.0

        # Motif stérile : oubli passif léger puis incrément.
        st.repetitions = max(0.0, st.repetitions - cfg.passive_decay) + 1.0

        # Seuil de grâce bruité (casse la prévisibilité du "mur").
        grace_noise = 1.0 + self._bounded_noise(cfg.noise_frac)
        effective_grace = cfg.grace * grace_noise

        if st.repetitions <= effective_grace:
            st.last_penalty = 0.0
            return 0.0

        # Au-delà de la grâce : escalation non linéaire.
        over = st.repetitions - effective_grace
        slope_noise = 1.0 + self._bounded_noise(cfg.noise_frac)
        magnitude = cfg.base_penalty * (over ** cfg.growth_exponent) * slope_noise * max(0.0, severity)
        magnitude = min(magnitude, cfg.max_penalty)

        penalty = -float(magnitude)
        st.last_penalty = penalty
        st.total_penalty += penalty
        st.triggers += 1
        return penalty

    def passive_step(self, motif: Optional[str] = None) -> None:
        """Oubli passif d'un step pour un motif (ou tous) sans événement.

        À appeler quand le motif n'est pas observé ce step, pour que la dette
        s'efface lentement (le modèle peut "se racheter" par l'inaction du motif).
        """
        targets = [motif] if motif else list(self._motifs.keys())
        for key in targets:
            st = self._motifs.get(key)
            if st is not None:
                st.repetitions = max(0.0, st.repetitions - self.config.passive_decay)

    # ── Introspection (tests / logs) ─────────────────────────────────────────
    def repetitions(self, motif: str) -> float:
        st = self._motifs.get(motif)
        return st.repetitions if st else 0.0

    def triggers(self, motif: str) -> int:
        st = self._motifs.get(motif)
        return st.triggers if st else 0

    def total_penalty(self, motif: str) -> float:
        st = self._motifs.get(motif)
        return st.total_penalty if st else 0.0

    def reset(self, motif: Optional[str] = None) -> None:
        if motif is None:
            self._motifs.clear()
        else:
            self._motifs.pop(motif, None)

    def snapshot(self) -> dict[str, dict[str, float]]:
        """État sérialisable (pour persistance entre sessions, §cahier modules)."""
        return {
            k: {
                "repetitions": v.repetitions,
                "last_penalty": v.last_penalty,
                "total_penalty": v.total_penalty,
                "triggers": float(v.triggers),
            }
            for k, v in self._motifs.items()
        }

    # ── interne ──────────────────────────────────────────────────────────────
    def _bounded_noise(self, frac: float) -> float:
        """Bruit uniforme dans [-frac, +frac]. Borné, ne change jamais le signe."""
        if frac <= 0.0:
            return 0.0
        return float(self._rng.uniform(-frac, frac))
