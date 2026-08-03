"""wick_stats — distributions empiriques de mèches → percentiles SL/TP cibles.

Implémente le Lot B2 du cahier des charges ADAN0 v2.

Idée (cahier §3.4 b/c) :
  * SL cible ≈ percentile haut des mèches ADVERSES futures (assez large pour ne
    pas être stoppé par le bruit, assez serré pour limiter le risque).
  * TP cible ≈ percentile des excursions FAVORABLES futures (assez ambitieux
    pour ne pas laisser d'argent sur la table, atteignable statistiquement).

Le contexte de calcul est OBLIGATOIREMENT segmenté (cahier §10.10) :
  (timeframe, actif, régime). JAMAIS de mélange bull/bear/range global.

Découverte empirique majeure (BTCUSDT, étude 2026-06-24), utile à la
réconciliation des frais (Lot A2) :
  - mèche 5m p95 ≈ 0.13 % ; range 5m médian ≈ 0.12 %
  - or les bandes SL scalper du code = 2-3 % → 15× la mèche p95 !
  - => les bandes actuelles sont calibrées pour 0.80 % de frais, totalement
    disproportionnées pour le bruit réel du 5m. Confirme : problème incitatif.

Module PUR : pas d'état global, pas d'I/O caché, aucun import du reste du bot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .future_zones import wick_ratios


# ───────────────────────────────────────────────────────────────────────────
#  Types
# ───────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class WickDistribution:
    """Distribution empirique des mèches d'un contexte donné (en % du prix)."""

    timeframe: str
    regime: Optional[int]
    n_samples: int
    # percentiles (clé = percentile entier, valeur = ratio de prix)
    up_pct: dict[int, float] = field(default_factory=dict)    # mèche haute
    down_pct: dict[int, float] = field(default_factory=dict)  # mèche basse
    range_pct: dict[int, float] = field(default_factory=dict) # amplitude bougie

    def percentile_up(self, p: int) -> float:
        return self.up_pct.get(p, 0.0)

    def percentile_down(self, p: int) -> float:
        return self.down_pct.get(p, 0.0)


@dataclass(frozen=True)
class SLTPTargets:
    """Cibles SL/TP dérivées de la distribution, pour un sens de trade donné.

    Pour un LONG :
      - adverse = mèches BASSES (le prix plonge sous l'entrée) -> SL
      - favorable = excursion HAUTE -> TP
    Pour un SHORT : symétrique.
    """

    timeframe: str
    regime: Optional[int]
    direction: str                  # "long" ou "short"
    sl_target_pct: float            # SL recommandé (ratio de prix)
    tp_target_pct: float            # TP recommandé (ratio de prix)
    sl_percentile: int
    tp_percentile: int
    noise_floor_pct: float          # mèche médiane (bruit) : SL ne doit pas être dessous


# ───────────────────────────────────────────────────────────────────────────
#  Calcul des distributions
# ───────────────────────────────────────────────────────────────────────────
_DEFAULT_PERCENTILES = (10, 25, 50, 75, 90, 95, 99)


def compute_wick_distribution(
    df: pd.DataFrame,
    timeframe: str,
    regime: Optional[int] = None,
    percentiles: tuple[int, ...] = _DEFAULT_PERCENTILES,
) -> WickDistribution:
    """Calcule la distribution des mèches (% du prix) d'un contexte.

    `df` doit contenir open/high/low/close. `regime` est seulement étiqueté
    (le filtrage par régime se fait par l'appelant qui passe le sous-DataFrame).
    """
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"wick_stats: colonne '{col}' manquante")
    if len(df) == 0:
        return WickDistribution(timeframe, regime, 0)

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    safe_c = np.where(c > 1e-12, c, np.nan)
    up_pct = (h - np.maximum(o, c)) / safe_c
    down_pct = (np.minimum(o, c) - l) / safe_c
    range_pct = (h - l) / safe_c

    def pcts(arr: np.ndarray) -> dict[int, float]:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {p: 0.0 for p in percentiles}
        return {p: float(np.percentile(arr, p)) for p in percentiles}

    return WickDistribution(
        timeframe=timeframe,
        regime=regime,
        n_samples=int(np.isfinite(range_pct).sum()),
        up_pct=pcts(up_pct),
        down_pct=pcts(down_pct),
        range_pct=pcts(range_pct),
    )


def compute_distributions_by_regime(
    df: pd.DataFrame,
    timeframe: str,
    regime_col: Optional[str] = None,
    percentiles: tuple[int, ...] = _DEFAULT_PERCENTILES,
) -> dict[Optional[int], WickDistribution]:
    """Une distribution par régime (si `regime_col` fourni et présent), sinon une
    seule distribution globale (clé None).

    Respecte §10.10 : on NE mélange PAS les régimes dans une stat unique.
    """
    if regime_col and regime_col in df.columns:
        out: dict[Optional[int], WickDistribution] = {}
        for reg, sub in df.groupby(regime_col):
            out[int(reg)] = compute_wick_distribution(sub, timeframe, int(reg), percentiles)
        return out
    return {None: compute_wick_distribution(df, timeframe, None, percentiles)}


# ───────────────────────────────────────────────────────────────────────────
#  Dérivation des cibles SL/TP
# ───────────────────────────────────────────────────────────────────────────
def derive_sltp_targets(
    dist: WickDistribution,
    direction: str = "long",
    sl_percentile: int = 90,
    tp_percentile: int = 75,
    sl_safety: float = 1.2,
) -> SLTPTargets:
    """Dérive un SL/TP cible depuis une distribution de mèches.

    Logique :
      - SL : on prend le percentile haut (ex p90) des mèches ADVERSES, multiplié
        par une marge de sécurité (sl_safety) pour ne pas être stoppé par le
        bruit p90. Plancher : la mèche médiane (noise floor) — un SL plus serré
        que le bruit médian serait stoppé en permanence.
      - TP : percentile des mèches FAVORABLES (ex p75) — atteignable souvent,
        pas trop gourmand. (À combiner ensuite avec la MFE par pivot, §3.4c.)

    direction == "long"  : adverse = mèches basses, favorable = mèches hautes.
    direction == "short" : adverse = mèches hautes, favorable = mèches basses.
    """
    if direction not in ("long", "short"):
        raise ValueError("direction doit être 'long' ou 'short'")

    if direction == "long":
        adverse = dist.down_pct
        favorable = dist.up_pct
    else:
        adverse = dist.up_pct
        favorable = dist.down_pct

    noise_floor = adverse.get(50, 0.0)
    sl_raw = adverse.get(sl_percentile, noise_floor) * sl_safety
    sl_target = max(sl_raw, noise_floor)  # jamais plus serré que le bruit médian
    tp_target = favorable.get(tp_percentile, 0.0)

    return SLTPTargets(
        timeframe=dist.timeframe,
        regime=dist.regime,
        direction=direction,
        sl_target_pct=float(sl_target),
        tp_target_pct=float(tp_target),
        sl_percentile=sl_percentile,
        tp_percentile=tp_percentile,
        noise_floor_pct=float(noise_floor),
    )


def sl_quality(sl_chosen: float, targets: SLTPTargets, tol: float = 0.5) -> float:
    """Score [0,1] de cohérence d'un SL choisi vs la cible (cahier §3.4b).

    1.0 si SL ≈ cible. Pénalise SL trop large (laisse courir la perte) ET trop
    serré (stoppé par le bruit). tol = tolérance relative avant pénalité forte.
    """
    target = targets.sl_target_pct
    if target <= 1e-9:
        return 0.0
    ratio = sl_chosen / target
    # gaussienne centrée sur 1.0, largeur tol
    return float(np.exp(-((ratio - 1.0) ** 2) / (2 * tol ** 2)))


def tp_quality(tp_chosen: float, mfe_future: float, tol: float = 0.5) -> float:
    """Score [0,1] de cohérence d'un TP choisi vs la MFE future (cahier §3.4c).

    Pénalise un TP ridicule (<< MFE : laisse de l'argent) et un TP irréaliste
    (>> MFE : jamais atteint). Optimal ≈ une fraction réaliste de la MFE.
    """
    if mfe_future <= 1e-9:
        return 0.0
    ratio = tp_chosen / mfe_future
    return float(np.exp(-((ratio - 1.0) ** 2) / (2 * tol ** 2)))
