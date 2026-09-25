"""
system_one_core.py — ADAN-System-One · Étape 4 (Moteur de jugement)
====================================================================

Le « JEV personnel » d'ADAN. Z (l'embedding d'état issu de la perception,
étape 3) n'est JAMAIS transformé directement en ordre : il est **interrogé
par des questions structurées**, dont les réponses sont typées, calibrables
et composables. C'est la rupture avec le « Z → action » d'ADAN0/PPO.

Trois types de réponses (référence paradigm System One / TypeSafe) :

  NOUL   : P(vérité) ∈ [0,1]  — une affirmation (« ce sweep atteint 3.5R ? »)
  CHOICE : distribution catégorielle — une alternative (« régime BULL/BEAR/… »)
  SCORE  : distribution ORDINALE monotone sur 0..10 — « qualité du setup »
           P(score > k) décroissante en k → distribution reconstruite,
           E[score], incertitude. (Corrige la somme naïve de sigmoïdes du
           prototype : ici les seuils cumulatifs sont structurellement ordonnés.)

Architecture — moteur GÉNÉRAL de jugement (pas 7 têtes isolées) :
  Chaque question = un embedding appris (Query-Slot) + la notion partagée
  (décodeur Noul/Choice/Score commun) + un ADAPTATEUR spécifique à la
  question. Le système apprend ainsi que « la même notion probabiliste
  s'applique à plusieurs affirmations, chacune avec sa sémantique ».

Calibration : les logits bruts ne sont jamais des probabilités. La méthode
`calibrate(temperatures)` applique un temperature scaling appris sur le split
VALIDATION (étape 12) ; `predict_calibrated` ne sert que des probabilités
dont P=0.70 signifie réellement 70 % de réussite.

Causalité : ce module ne voit que Z (produit à partir du LivingStateSnapshot
causal). Aucune donnée future ne peut y entrer.

Référence : dev.md §ADAN-SYSTEM-ONE, workflow étape 4 + §III (Query-Slots).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Registre des questions (Query-Slots) ─────────────────────────────────────
# Chaque entrée : (nom, type). L'embedding du slot encode « la question ».
NOUL_QUESTIONS = [
    "mfe_atteint_tp",        # le sweep atteindra-t-il la cible 3.5R avant invalidation ?
    "risque_adverse_faible", # MAE restera-t-il sous le seuil d'invalidation ?
    "expansion_imminente",   # expansion du contenant probable sous l'horizon ?
    "anomalie_donnees",      # l'état est-il corrompu / non fiable ? (→ abstention)
    "sweep_confirme",        # le balayage est-il un vrai piège à liquidité ?
    "reintegration_valide",  # la réintégration du range est-elle confirmée ?
]
CHOICE_QUESTIONS = {
    "regime": ["BULL", "BEAR", "RANGE", "TRAP"],           # régime de marché
    "direction": ["LONG", "SHORT", "AUCUNE"],              # direction de l'edge
}
SCORE_QUESTIONS = [
    "qualite_setup",     # qualité globale du setup (0..10, ordinal)
    "conviction",        # force de la conviction (0..10, ordinal)
]
N_SCORE_LEVELS = 11  # 0..10


# ─── Réponse typée ────────────────────────────────────────────────────────────
@dataclass
class Judgment:
    """Réponses d'un état Z à toutes les questions (batch ou unitaire)."""
    noul: Dict[str, float]                  # question -> P(vérité) calibrée
    choice: Dict[str, Dict[str, float]]     # question -> {catégorie: proba}
    score: Dict[str, Dict[str, float]]      # question -> {esperance, incertitude, ...}
    calibrated: bool = False


# ─── Décodeurs partagés (la « notion ») + adaptateurs (la « sémantique ») ────
class NoulDecoder(nn.Module):
    """Notion probabiliste partagée : (Z ⊕ question_adaptée) → logit binaire."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, zq: torch.Tensor) -> torch.Tensor:
        return self.net(zq).squeeze(-1)  # (...,) logit


class ChoiceDecoder(nn.Module):
    """Notion catégorielle partagée : logits par classe."""

    def __init__(self, dim: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, zq: torch.Tensor) -> torch.Tensor:
        return self.net(zq)  # (..., n_classes)


class OrdinalScoreDecoder(nn.Module):
    """SCORE ordinal monotone : prédit les logits cumulatifs P(score > k)
    pour k = 0..9, structurellement décroissants (tri via -cumsum de softplus).

    Correction du prototype : au lieu de sommer 10 sigmoïdes indépendantes
    (aucune monotonie garantie), on impose P(>k) ≥ P(>k+1) par construction,
    ce qui autorise une vraie distribution ordinale + incertitude.
    """

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(dim, hidden), nn.GELU())
        self.head_first = nn.Linear(hidden, 1)               # logit du seuil 0
        self.head_steps = nn.Linear(hidden, N_SCORE_LEVELS - 2)  # Δ entre seuils

    def forward(self, zq: torch.Tensor) -> torch.Tensor:
        h = self.body(zq)
        first = self.head_first(h)                            # (..., 1)
        steps = F.softplus(self.head_steps(h))               # (..., 9) ≥ 0
        # seuils cumulatifs DÉCROISSANTS : logit_k = first − cumsum(steps)
        cumulative = torch.cat([first, first - torch.cumsum(steps, dim=-1)], dim=-1)
        return cumulative                                    # (..., 10) logits de P(score>k)


# ─── Cœur System One ─────────────────────────────────────────────────────────
class SystemOneCore(nn.Module):
    """Moteur de jugement : Z + Query-Slots → réponses NOUL/CHOICE/SCORE.

    Paramètres :
      dim_z     : dimension de l'embedding d'état Z (sortie de la perception)
      dim_slot  : dimension des embeddings de question (Query-Slots)
      dim_hidden: largeur des décodeurs partagés
    """

    def __init__(self, dim_z: int, dim_slot: int = 64, dim_hidden: int = 128):
        super().__init__()
        self.dim_z = dim_z
        self.dim_slot = dim_slot

        # Query-Slots : un embedding appris par question (la « question » elle-même)
        self.slot_noul = nn.Parameter(torch.randn(len(NOUL_QUESTIONS), dim_slot) * 0.02)
        self.slot_score = nn.Parameter(torch.randn(len(SCORE_QUESTIONS), dim_slot) * 0.02)
        self.slot_choice = nn.ParameterDict({
            q: nn.Parameter(torch.randn(dim_slot) * 0.02) for q in CHOICE_QUESTIONS
        })

        # Adaptateurs spécifiques : projettent (Z, slot) → espace de jugement
        self.adapter = nn.Linear(dim_z + dim_slot, dim_hidden)

        # Décodeurs partagés (la notion) — un par TYPE de réponse
        self.dec_noul = NoulDecoder(dim_hidden, dim_hidden)
        self.dec_choice = nn.ModuleDict({
            q: ChoiceDecoder(dim_hidden, len(cats), dim_hidden)
            for q, cats in CHOICE_QUESTIONS.items()
        })
        self.dec_score = OrdinalScoreDecoder(dim_hidden, dim_hidden)

        # Températures de calibration (apprises sur VALIDATION, étape 12).
        # buffer → non entraîné par SGD, mais sauvegardé avec le modèle.
        self.register_buffer("temp_noul", torch.ones(len(NOUL_QUESTIONS)))
        self.register_buffer("temp_score", torch.ones(len(SCORE_QUESTIONS)))
        self.register_buffer("temp_choice", torch.ones(len(CHOICE_QUESTIONS)))
        self._calibrated = False

    # ── utilitaire interne : fusion Z + slot → espace de jugement ────────────
    def _fuse(self, z: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        """z : (..., dim_z) ; slot : (dim_slot,) ou (..., dim_slot)."""
        if slot.dim() < z.dim():
            slot = slot.expand(*z.shape[:-1], -1)
        return self.adapter(torch.cat([z, slot], dim=-1))

    # ── logits bruts (pour l'ENTRAÎNEMENT, Brier/NLL) ────────────────────────
    def forward(self, z: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """Retourne les logits bruts par question. z : (batch, dim_z) ou (dim_z,)."""
        single = z.dim() == 1
        if single:
            z = z.unsqueeze(0)

        out: Dict[str, Dict[str, torch.Tensor]] = {"noul": {}, "choice": {}, "score": {}}

        # NOUL — une logit par question, via slot + adaptateur + décodeur partagé
        for qi, q in enumerate(NOUL_QUESTIONS):
            zq = self._fuse(z, self.slot_noul[qi])
            out["noul"][q] = self.dec_noul(zq)               # (batch,)

        # CHOICE — distribution catégorielle par question
        for q, cats in CHOICE_QUESTIONS.items():
            zq = self._fuse(z, self.slot_choice[q])
            out["choice"][q] = self.dec_choice[q](zq)        # (batch, n_classes)

        # SCORE — logits cumulatifs ordinaux (batch, 10) par question
        for qi, q in enumerate(SCORE_QUESTIONS):
            zq = self._fuse(z, self.slot_score[qi])
            out["score"][q] = self.dec_score(zq)

        if single:
            for kind in out:
                for q in out[kind]:
                    out[kind][q] = out[kind][q].squeeze(0)
        return out

    # ── calibration (températures apprises sur VALIDATION — étape 12) ────────
    @torch.no_grad()
    def calibrate(self, temp_noul=None, temp_choice=None, temp_score=None):
        """Fixe les températures (temperature scaling). Appelé par
        offline/train_calibrated_judgments.py après ajustement sur VAL."""
        if temp_noul is not None:
            self.temp_noul.copy_(torch.as_tensor(temp_noul, dtype=self.temp_noul.dtype))
        if temp_choice is not None:
            self.temp_choice.copy_(torch.as_tensor(temp_choice, dtype=self.temp_choice.dtype))
        if temp_score is not None:
            self.temp_score.copy_(torch.as_tensor(temp_score, dtype=self.temp_score.dtype))
        self._calibrated = True

    # ── probabilités calibrées (pour la POLICY — jamais les logits bruts) ────
    @torch.no_grad()
    def predict_calibrated(self, z: torch.Tensor) -> Judgment:
        """Réponses probabilistes calibrées. P=0.70 ⇒ 70 % de réussite réelle."""
        self.eval()
        logits = self.forward(z)
        single = z.dim() == 1

        noul: Dict[str, float] = {}
        for qi, q in enumerate(NOUL_QUESTIONS):
            p = torch.sigmoid(logits["noul"][q] / self.temp_noul[qi])
            noul[q] = float(p.item() if single else p[0].item())

        choice: Dict[str, Dict[str, float]] = {}
        for ci, (q, cats) in enumerate(CHOICE_QUESTIONS.items()):
            probs = F.softmax(logits["choice"][q] / self.temp_choice[ci], dim=-1)
            probs = probs if not single else probs.unsqueeze(0)
            choice[q] = {c: float(probs[0, k].item()) for k, c in enumerate(cats)}

        score: Dict[str, Dict[str, float]] = {}
        for qi, q in enumerate(SCORE_QUESTIONS):
            cum = torch.sigmoid(logits["score"][q] / self.temp_score[qi])  # (b,10) P(>k)
            cum = cum if not single else cum.unsqueeze(0)
            # distribution ordinale : P(=k) = P(>k-1) − P(>k) pour k = 0..10
            #   P(>-1) = 1 (borne gauche), P(>10) = 0 (borne droite) → (b, 11)
            p_gt_prev = torch.cat([torch.ones_like(cum[:, :1]), cum], dim=1)   # P(>k-1)
            p_gt_next = torch.cat([cum, torch.zeros_like(cum[:, :1])], dim=1)  # P(>k)
            dist = (p_gt_prev - p_gt_next).clamp_min(0)        # (b, 11) sur 0..10
            levels = torch.arange(N_SCORE_LEVELS, device=z.device, dtype=dist.dtype)
            esperance = float((dist[0] * levels).sum().item())
            variance = float((dist[0] * (levels - esperance) ** 2).sum().item())
            score[q] = {
                "esperance": round(esperance, 3),
                "incertitude": round(variance ** 0.5, 3),
                "distribution": [round(float(x), 4) for x in dist[0].tolist()],
            }

        return Judgment(noul=noul, choice=choice, score=score,
                        calibrated=self._calibrated)


# ─── Construction depuis le snapshot (pont étape 1 → étape 4) ────────────────
def snapshot_to_z_features(snap) -> np.ndarray:  # type: ignore[name-defined]
    """Aplatit un LivingStateSnapshot en vecteur de features (entrée de la
    perception). Sans PyTorch ici : la perception (CNN+attention+FiLM) est
    l'étape 3 ; cette fonction fournit le vecteur brut que l'encodeur consomme.
    Ordre fixe et documenté — toute modification casse la compatibilité des
    poids entraînés."""
    import numpy as np

    parts = [
        snap.bar_5m,                                  # 8
        snap.seq_5m[-6:].flatten(),                   # 6 dernières 5m (48)
        snap.running_1h,                              # 4
        [snap.phase_1h, snap.pos_in_1h,
         float(snap.sweep_high_1h), float(snap.sweep_low_1h)],   # 4
        np.nan_to_num(snap.prev_1h, nan=0.0),         # 5
        snap.running_4h,                              # 4
        [snap.phase_4h, snap.pos_in_4h,
         float(snap.sweep_high_4h), float(snap.sweep_low_4h)],   # 4
        np.nan_to_num(snap.prev_4h, nan=0.0),         # 5
        snap.portfolio,                               # 5
        [float(snap.integrity_ok)],                   # 1
    ]
    return np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts])


# Dimension du vecteur de features (référence pour la perception / l'encodeur)
Z_FEATURES_DIM = 8 + 48 + 4 + 4 + 5 + 4 + 4 + 5 + 5 + 1  # = 88
