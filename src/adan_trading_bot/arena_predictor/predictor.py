"""V18 — Arena Bayésien Prédictif : Predictor (MLP hétéroscédastique).

Ce module apprend, HORS-LIGNE, la fonction :

    present_state (13-dim)  ->  distribution des params optimaux (5 x Gaussien)

Chaque paramètre-cible est prédit comme un couple (mean, log_std). Le log_std
capture l'INCERTITUDE aléatoire (hétéroscédastique) : dans un régime de marché
familier, std faible -> confiance ; dans un régime rare/ambigu, std large ->
prudence. C'est l'esprit « bayésien » demandé au §V18 : on ne renvoie pas une
valeur ponctuelle mais une distribution, et l'incertitude module la décision.

Approximation bayésienne :
  - Hétéroscédasticité native : chaque cible a son log_std prédit.
  - MC-Dropout optionnel (dropout actif à l'inférence) pour approximer
    l'incertitude épistémique (variance entre passes). Voir `predict_dist`.

Entraînement : maximum de vraisemblance gaussien (negative log-likelihood).
    NLL = 0.5 * ( (y-mu)^2 / sigma^2 + log sigma^2 )

Live-safe : l'entrée est UNIQUEMENT le present_state. Aucun futur.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .state_schema import STATE_DIM, TARGET_DIM, TARGET_PARAMS, STATE_SCHEMA_VERSION

# Bornes de sécurité sur log_std pour éviter NaN / explosions.
_LOG_STD_MIN = -6.0
_LOG_STD_MAX = 2.0


class ArenaPredictor(nn.Module):
    """MLP hétéroscédastique : state -> (mean[5], log_std[5]).

    Sortie : tensor (B, 2*TARGET_DIM) = [means (5) | log_stds (5)].
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        target_dim: int = TARGET_DIM,
        hidden: Tuple[int, ...] = (64, 64),
        dropout: float = 0.10,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.target_dim = int(target_dim)
        self.dropout_p = float(dropout)

        layers: List[nn.Module] = []
        prev = self.state_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(self.dropout_p))
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head_mean = nn.Linear(prev, self.target_dim)
        self.head_logstd = nn.Linear(prev, self.target_dim)

        # Init raisonnable : log_std ~ 0 (std ~ 1) au départ.
        nn.init.zeros_(self.head_logstd.bias)
        nn.init.normal_(self.head_logstd.weight, std=1e-3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mean = self.head_mean(h)
        log_std = torch.clamp(self.head_logstd(h), _LOG_STD_MIN, _LOG_STD_MAX)
        return mean, log_std

    # ------------------------------------------------------------------ #
    # Inférence avec incertitude (MC-Dropout épistémique + aléatoire)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_dist(
        self, x: torch.Tensor, mc_samples: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Renvoie (mean, std) combinant incertitude aléatoire + épistémique.

        Si mc_samples > 1 : active le dropout (train mode) et échantillonne
        plusieurs passes ; std_total^2 = E[sigma^2] + Var[mu] (loi de la
        variance totale).
        """
        if mc_samples <= 1:
            self.eval()
            mean, log_std = self.forward(x)
            return mean, torch.exp(log_std)

        was_training = self.training
        self.train()  # active le dropout
        means = []
        var_aleatoric = []
        for _ in range(int(mc_samples)):
            m, ls = self.forward(x)
            means.append(m)
            var_aleatoric.append(torch.exp(2.0 * ls))
        if not was_training:
            self.eval()
        M = torch.stack(means, dim=0)          # (S, B, D)
        Va = torch.stack(var_aleatoric, dim=0)  # (S, B, D)
        mean = M.mean(dim=0)
        var_total = Va.mean(dim=0) + M.var(dim=0, unbiased=False)
        return mean, torch.sqrt(torch.clamp(var_total, min=1e-12))


def gaussian_nll(mean: torch.Tensor, log_std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood gaussienne hétéroscédastique (moyenne batch)."""
    var = torch.exp(2.0 * log_std)
    nll = 0.5 * (((target - mean) ** 2) / (var + 1e-12) + 2.0 * log_std + math.log(2.0 * math.pi))
    return nll.mean()


# ---------------------------------------------------------------------- #
# Normalisation des cibles : les 5 params ont des échelles très différentes
# (break_even ~0.004, duration ~20). On standardise pour stabiliser le NLL.
# ---------------------------------------------------------------------- #
class TargetScaler:
    """Standardisation z-score par cible, persistable en JSON."""

    def __init__(self, mean: Optional[List[float]] = None, std: Optional[List[float]] = None):
        self.mean = mean or [0.0] * TARGET_DIM
        self.std = std or [1.0] * TARGET_DIM

    def fit(self, targets: torch.Tensor) -> "TargetScaler":
        m = targets.mean(dim=0)
        s = targets.std(dim=0)
        s = torch.where(s < 1e-6, torch.ones_like(s), s)
        self.mean = m.tolist()
        self.std = s.tolist()
        return self

    def transform(self, targets: torch.Tensor) -> torch.Tensor:
        m = torch.tensor(self.mean, dtype=targets.dtype, device=targets.device)
        s = torch.tensor(self.std, dtype=targets.dtype, device=targets.device)
        return (targets - m) / s

    def inverse_mean(self, mean_z: torch.Tensor) -> torch.Tensor:
        m = torch.tensor(self.mean, dtype=mean_z.dtype, device=mean_z.device)
        s = torch.tensor(self.std, dtype=mean_z.dtype, device=mean_z.device)
        return mean_z * s + m

    def inverse_std(self, std_z: torch.Tensor) -> torch.Tensor:
        s = torch.tensor(self.std, dtype=std_z.dtype, device=std_z.device)
        return std_z * s

    def to_dict(self) -> Dict:
        return {"mean": self.mean, "std": self.std}

    @staticmethod
    def from_dict(d: Dict) -> "TargetScaler":
        return TargetScaler(mean=d.get("mean"), std=d.get("std"))


def save_predictor(model: ArenaPredictor, scaler: TargetScaler, path: str) -> None:
    """Sauvegarde poids + scaler + métadonnées de schéma."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "state_dim": model.state_dim,
        "target_dim": model.target_dim,
        "target_params": TARGET_PARAMS,
        "dropout": model.dropout_p,
        "state_dict": model.state_dict(),
        "scaler": scaler.to_dict(),
    }
    torch.save(payload, path)


def load_predictor(path: str, map_location: str = "cpu") -> Tuple[ArenaPredictor, TargetScaler]:
    """Charge poids + scaler. Reconstruit l'architecture depuis les métadonnées."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    # Déduit hidden depuis les shapes du state_dict (Linear backbone).
    hidden = []
    for k, v in payload["state_dict"].items():
        if k.startswith("backbone.") and k.endswith(".weight") and v.dim() == 2:
            hidden.append(v.shape[0])
    # Ne garde que les couches Linear (LayerNorm a aussi des .weight 1D, exclus).
    model = ArenaPredictor(
        state_dim=int(payload.get("state_dim", STATE_DIM)),
        target_dim=int(payload.get("target_dim", TARGET_DIM)),
        hidden=tuple(hidden) if hidden else (64, 64),
        dropout=float(payload.get("dropout", 0.10)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    scaler = TargetScaler.from_dict(payload.get("scaler", {}))
    return model, scaler
