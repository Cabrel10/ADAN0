"""ActionSaturationGuard — réveil d'exploration anti-saturation par tête d'action.

CONTEXTE (audit A7 / SIZE, séance juin 2026)
--------------------------------------------
Le checkpoint 500k_FIXED présentait :
  * tête ``size`` GELÉE en sortie (100 % à -1.0, std=0.0) malgré ``ent_coef`` et gSDE ;
  * tête ``tp`` SATURÉE au plafond (99 % dans [+0.80,+1.0]).
Les poids ``action_net`` de ``size`` sont pourtant les plus GROS (‖w‖≈3.54) : ce ne
sont pas des poids morts mais une **saturation de la sortie** (pré-activation très
négative → ``tanh`` collé à -1). Le ``PpoStdSafetyCallback`` existant ne clampe que la
borne HAUTE de ``log_std`` (anti-explosion) ; il ne fait RIEN contre l'effondrement.

Ce callback est COMPLÉMENTAIRE :
  1. mesure, par dimension d'action, la fraction d'actions saturées (≈ ±1) dans le
     dernier rollout (``rollout_buffer.actions``) ;
  2. si une dimension dépasse ``sat_threshold`` pendant ``patience`` rollouts
     consécutifs, RELÈVE le plancher de ``log_std`` (réinjecte de l'exploration) ;
  3. journalise tout dans TensorBoard et les logs.

Conçu pour gSDE (``log_std`` shape ``(latent, A)``) ET DiagGaussian (shape ``(A,)``).
Désactivé par défaut côté appelant : on l'ajoute explicitement aux callbacks.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)

DEFAULT_ACTION_NAMES = ["direction", "size", "tf", "sl", "tp"]


class ActionSaturationGuard(BaseCallback):
    """Détecte la saturation de sortie par tête et réveille l'exploration.

    Args:
        sat_edge: seuil |action| au-delà duquel l'action est dite "saturée" (0.98).
        sat_threshold: fraction de saturation déclenchant une alerte (0.95).
        patience: nb de rollouts saturés consécutifs avant intervention (2).
        bump_log_std: incrément ajouté au log_std de la/les dim(s) saturée(s) (+0.5).
        max_log_std: plafond de sécurité du log_std après bump (+2.0).
        action_names: noms lisibles des dimensions (par défaut ADAN 5 têtes).
        intervene: si False, OBSERVE et journalise sans modifier le réseau (dry-run).
    """

    def __init__(
        self,
        sat_edge: float = 0.98,
        sat_threshold: float = 0.95,
        patience: int = 2,
        bump_log_std: float = 0.5,
        max_log_std: float = 2.0,
        action_names: Optional[List[str]] = None,
        intervene: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.sat_edge = float(sat_edge)
        self.sat_threshold = float(sat_threshold)
        self.patience = int(patience)
        self.bump_log_std = float(bump_log_std)
        self.max_log_std = float(max_log_std)
        self.action_names = action_names or DEFAULT_ACTION_NAMES
        self.intervene = bool(intervene)
        self._streak: Optional[np.ndarray] = None  # rollouts saturés consécutifs / dim
        self._n_bumps = 0

    # ── helpers ────────────────────────────────────────────────────────────────
    def _action_name(self, j: int) -> str:
        return self.action_names[j] if j < len(self.action_names) else f"a{j}"

    def _bump_log_std(self, dims: List[int]) -> None:
        """Relève le log_std des dimensions saturées (réveil d'exploration)."""
        policy = getattr(self.model, "policy", None)
        if policy is None or not hasattr(policy, "log_std"):
            return
        try:
            with torch.no_grad():
                ls = policy.log_std  # (A,) DiagGaussian OU (latent, A) gSDE
                if ls.dim() == 1:
                    for j in dims:
                        if 0 <= j < ls.shape[0]:
                            ls.data[j] = torch.clamp(
                                ls.data[j] + self.bump_log_std, max=self.max_log_std)
                elif ls.dim() == 2:
                    # gSDE : colonne j = dim d'action j (latent features en lignes)
                    A = ls.shape[1]
                    for j in dims:
                        if 0 <= j < A:
                            ls.data[:, j] = torch.clamp(
                                ls.data[:, j] + self.bump_log_std, max=self.max_log_std)
            self._n_bumps += 1
            logger.warning(
                "[SaturationGuard] BUMP log_std (+%.2f) sur dims=%s (bump #%d)",
                self.bump_log_std, [self._action_name(j) for j in dims], self._n_bumps,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[SaturationGuard] bump échoué: %s", e)

    # ── cycle SB3 ───────────────────────────────────────────────────────────────
    def _on_training_start(self) -> None:
        try:
            A = int(self.model.action_space.shape[0])
        except Exception:
            A = len(self.action_names)
        self._streak = np.zeros(A, dtype=int)

    def _on_rollout_end(self) -> None:
        buf = getattr(self.model, "rollout_buffer", None)
        if buf is None or getattr(buf, "actions", None) is None:
            return
        acts = np.asarray(buf.actions)  # (n_steps, n_envs, A) ou (n_steps, A)
        if acts.ndim == 3:
            acts = acts.reshape(-1, acts.shape[-1])
        elif acts.ndim != 2:
            return
        A = acts.shape[1]
        if self._streak is None or self._streak.shape[0] != A:
            self._streak = np.zeros(A, dtype=int)

        sat_frac = np.mean(np.abs(acts) >= self.sat_edge, axis=0)  # par dim
        action_std = acts.std(axis=0)
        saturated_now = sat_frac >= self.sat_threshold

        # logs TensorBoard + console
        for j in range(A):
            nm = self._action_name(j)
            if self.logger is not None:
                self.logger.record(f"saturation/{nm}_sat_frac", float(sat_frac[j]))
                self.logger.record(f"saturation/{nm}_action_std", float(action_std[j]))
            if self.verbose > 0 and saturated_now[j]:
                logger.info(
                    "[SaturationGuard] '%s' saturé : %.1f%% à |a|>=%.2f (std=%.4f)",
                    nm, sat_frac[j] * 100, self.sat_edge, action_std[j],
                )

        # mise à jour des séries consécutives
        self._streak = np.where(saturated_now, self._streak + 1, 0)
        to_fix = [j for j in range(A) if self._streak[j] >= self.patience]
        if to_fix:
            if self.intervene:
                self._bump_log_std(to_fix)
            else:
                logger.warning(
                    "[SaturationGuard] (dry-run) dims saturées à corriger: %s",
                    [self._action_name(j) for j in to_fix],
                )
            for j in to_fix:
                self._streak[j] = 0  # reset après action

    def _on_step(self) -> bool:
        return True
