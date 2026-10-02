"""future_zones — détection de pivots + MFE/MAE + zones 🟢🟡🔴 par chunk.

Implémente les Lots B1 et B5 du cahier des charges ADAN0 v2.

Principes (rappel cahier §10.10) :
  * Calcul EX-POST : on regarde le futur du chunk. Autorisé UNIQUEMENT pour
    façonner la récompense d'entraînement (jamais dans l'observation acteur).
  * Par CHUNK (≈ 1 journée), pas sur le dataset entier : on cible 10-15 points
    critiques par jour (pour 15-21 trades/jour).
  * Statistiques contextuelles : (timeframe, actif, régime, chunk). Le régime
    est un paramètre OPTIONNEL fourni par l'appelant (jamais calculé ici).
  * Module PUR : pas d'état global, pas d'I/O caché, pas d'import du reste du bot.

Conventions de colonnes attendues (minuscule) : open, high, low, close.
volume est optionnel (utilisé pour la confiance si présent).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ───────────────────────────────────────────────────────────────────────────
#  Types
# ───────────────────────────────────────────────────────────────────────────
class Zone(str, Enum):
    """Pertinence d'un point d'entrée jugée ex-post sur le futur du chunk."""

    GREEN = "green"    # 🟢 structurelle : forte extension favorable, faible DD
    ORANGE = "orange"  # 🟡 neutre : RR moyen, TP atteignable mais oscillant
    RED = "red"        # 🔴 toxique : retracement immédiat / faible potentiel


class PivotDirection(str, Enum):
    """Sens du pivot détecté."""

    LOW = "low"    # creux local -> opportunité d'achat (long)
    HIGH = "high"  # sommet local -> opportunité de vente (short)


@dataclass(frozen=True)
class Pivot:
    """Un point critique détecté dans le chunk (index positionnel local)."""

    idx: int                 # position dans le chunk (0-based)
    direction: PivotDirection
    price: float             # close au pivot
    timestamp: Optional[pd.Timestamp] = None


@dataclass(frozen=True)
class CriticalPoint:
    """Pivot enrichi de son analyse forward (MFE/MAE/zone/confiance)."""

    idx: int
    direction: PivotDirection
    price: float
    timestamp: Optional[pd.Timestamp]
    mfe: float               # Maximum Favorable Excursion (ratio, ex 0.04 = +4%)
    mae: float               # Maximum Adverse Excursion (ratio, >=0, ex 0.01)
    zone: Zone
    quality_score: float     # [0,1] : qualité statistique de l'entrée
    confidence: float        # [0,1] : confiance dans la détection du pivot
    regime: Optional[int] = None
    horizon: int = 0         # nb bougies de lookahead utilisées
    rr_cap: float = 10.0     # plafond appliqué au ratio (cohérent avec classify)

    @property
    def rr(self) -> float:
        """Ratio reward/risk ex-post = MFE / MAE, borné à rr_cap.

        MAE a déjà reçu son plancher (mae_floor) au calcul ; on plafonne ici
        pour rester cohérent avec classify_zone et éviter les valeurs absurdes.
        """
        if self.mae <= 1e-9:
            return self.rr_cap if self.mfe > 0 else 0.0
        return min(self.mfe / self.mae, self.rr_cap)


@dataclass(frozen=True)
class ZoneConfig:
    """Paramètres de détection/classification (par profil de trading).

    Valeurs par défaut = profil 'scalper' sur 5m (lookahead 3h = 36 bougies 5m).
    Toutes les bornes sont documentées et testables ; rien n'est magique.
    """

    # Détection de pivots (fractal de Williams : k bougies de chaque côté)
    fractal_k: int = 2
    # Filtre ZigZag : amplitude minimale (ratio) pour retenir un pivot
    min_swing_pct: float = 0.003          # 0.3 % mini pour qu'un pivot compte
    # Lookahead forward pour MFE/MAE (en nombre de bougies)
    horizon: int = 36                     # 3h en 5m ; 12h swing -> 144
    # Classification de zone (sur le RR ex-post = MFE/MAE)
    green_min_rr: float = 1.5             # 🟢 si RR >= 1.5 ET MFE significatif
    red_max_rr: float = 0.8               # 🔴 si RR <= 0.8 (DD domine)
    green_min_mfe: float = 0.006          # MFE plancher pour 🟢 (sinon 🟡)
    # Plancher de MAE : un creux pile sur le pivot a un drawdown forward ~0,
    # ce qui ferait exploser RR=MFE/MAE vers l'infini (RR=19M observé sur données
    # réelles). On impose un MAE minimal réaliste = max(mae_floor_abs, frais A/R).
    # Sans ça, le quality_score et la sélection sont faussés (étude empirique 5m).
    mae_floor: float = 0.0015             # 0.15 % (≈ slippage + frais A/R)
    rr_cap: float = 10.0                  # RR borné pour rester interprétable
    # Sélection finale : nb de points critiques gardés par chunk
    max_points: int = 15
    min_points: int = 10
    # Relâchement adaptatif : si trop peu de pivots, on assouplit min_swing_pct
    # par paliers (×0.6) jusqu'à atteindre min_points (anti journée calme).
    adaptive_swing: bool = True
    # Distance de proximité au pivot (en bougies) pour was_near_critical_point
    proximity_window: int = 3

    def for_swing(self) -> "ZoneConfig":
        """Variante swing (lookahead plus long, swings plus amples)."""
        return ZoneConfig(
            fractal_k=3,
            min_swing_pct=0.01,
            horizon=144,
            green_min_rr=self.green_min_rr,
            red_max_rr=self.red_max_rr,
            green_min_mfe=0.02,
            mae_floor=self.mae_floor,
            rr_cap=self.rr_cap,
            max_points=self.max_points,
            min_points=self.min_points,
            adaptive_swing=self.adaptive_swing,
            proximity_window=self.proximity_window,
        )


# ───────────────────────────────────────────────────────────────────────────
#  Mèches futures (Future Wick Ratios) — cahier §3.2
# ───────────────────────────────────────────────────────────────────────────
def wick_ratios(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mèches haute et basse normalisées par l'amplitude de la bougie.

    W_up   = (High − max(Open, Close)) / (High − Low)
    W_down = (min(Open, Close) − Low)  / (High − Low)

    Renvoie (w_up, w_down), chacun dans [0, 1]. Si High == Low (bougie plate),
    la mèche vaut 0 (pas de division par zéro).
    """
    open_ = np.asarray(open_, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)

    rng = high - low
    safe = np.where(rng > 1e-12, rng, np.nan)
    body_top = np.maximum(open_, close)
    body_bot = np.minimum(open_, close)

    w_up = np.where(rng > 1e-12, (high - body_top) / safe, 0.0)
    w_down = np.where(rng > 1e-12, (body_bot - low) / safe, 0.0)
    return np.nan_to_num(w_up, nan=0.0), np.nan_to_num(w_down, nan=0.0)


# ───────────────────────────────────────────────────────────────────────────
#  Détection de pivots (fractal de Williams + filtre ZigZag) — Lot B5
# ───────────────────────────────────────────────────────────────────────────
def detect_pivots(df: pd.DataFrame, config: ZoneConfig) -> list[Pivot]:
    """Détecte les pivots (creux/sommets locaux) d'un chunk.

    Méthode : fractal de Williams (un point est un sommet s'il est le plus haut
    sur ±k bougies ; un creux s'il est le plus bas). Puis filtre ZigZag :
    on ne garde un pivot que si l'amplitude depuis le pivot opposé précédent
    dépasse `min_swing_pct` (anti-bruit).

    Le chunk est passé tel quel (index positionnel = position dans le chunk).
    NE regarde QUE l'intérieur du chunk : aucune fuite hors des bornes fournies.
    """
    _require_ohlc(df)
    n = len(df)
    k = config.fractal_k
    if n < 2 * k + 1:
        return []

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    ts = df.index if isinstance(df.index, pd.DatetimeIndex) else None

    raw: list[Pivot] = []
    for i in range(k, n - k):
        window_hi = high[i - k : i + k + 1]
        window_lo = low[i - k : i + k + 1]
        if high[i] == window_hi.max() and np.argmax(window_hi) == k:
            raw.append(Pivot(i, PivotDirection.HIGH, float(close[i]),
                             ts[i] if ts is not None else None))
        elif low[i] == window_lo.min() and np.argmin(window_lo) == k:
            raw.append(Pivot(i, PivotDirection.LOW, float(close[i]),
                             ts[i] if ts is not None else None))

    return _zigzag_filter(raw, config.min_swing_pct)


def _detect_pivots_adaptive(df: pd.DataFrame, config: ZoneConfig) -> list[Pivot]:
    """detect_pivots avec relâchement adaptatif du filtre ZigZag.

    Sur une journée calme, `min_swing_pct` peut éliminer trop de pivots
    (< min_points). On ré-applique le filtre brut avec un seuil de plus en plus
    permissif (×0.6 par palier) jusqu'à atteindre min_points ou un plancher.
    Sans cela on tombait à 6-8 points/jour au lieu des 10-15 ciblés (étude 5m).
    """
    _require_ohlc(df)
    n = len(df)
    k = config.fractal_k
    if n < 2 * k + 1:
        return []

    # Détection fractale brute, calculée une seule fois
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    ts = df.index if isinstance(df.index, pd.DatetimeIndex) else None
    raw: list[Pivot] = []
    for i in range(k, n - k):
        wh = high[i - k : i + k + 1]
        wl = low[i - k : i + k + 1]
        if high[i] == wh.max() and np.argmax(wh) == k:
            raw.append(Pivot(i, PivotDirection.HIGH, float(close[i]),
                             ts[i] if ts is not None else None))
        elif low[i] == wl.min() and np.argmin(wl) == k:
            raw.append(Pivot(i, PivotDirection.LOW, float(close[i]),
                             ts[i] if ts is not None else None))

    swing = config.min_swing_pct
    pivots = _zigzag_filter(raw, swing)
    if not config.adaptive_swing:
        return pivots

    # Relâche jusqu'à min_points (plancher de seuil pour éviter le bruit pur)
    floor = config.min_swing_pct * 0.1
    while len(pivots) < config.min_points and swing > floor:
        swing *= 0.6
        pivots = _zigzag_filter(raw, swing)
    return pivots


def _zigzag_filter(pivots: Sequence[Pivot], min_swing_pct: float) -> list[Pivot]:
    """Garde les pivots alternés (haut/bas) dont l'amplitude > min_swing_pct."""
    if not pivots:
        return []
    kept: list[Pivot] = [pivots[0]]
    for p in pivots[1:]:
        last = kept[-1]
        if p.direction == last.direction:
            # même sens : on garde l'extrême (plus haut des hauts / plus bas des bas)
            better = (
                (p.direction == PivotDirection.HIGH and p.price > last.price)
                or (p.direction == PivotDirection.LOW and p.price < last.price)
            )
            if better:
                kept[-1] = p
            continue
        swing = abs(p.price - last.price) / last.price if last.price else 0.0
        if swing >= min_swing_pct:
            kept.append(p)
    return kept


# ───────────────────────────────────────────────────────────────────────────
#  MFE / MAE forward — Lot B1
# ───────────────────────────────────────────────────────────────────────────
def compute_mfe_mae(
    df: pd.DataFrame,
    idx: int,
    direction: PivotDirection,
    horizon: int,
    mae_floor: float = 0.0,
) -> tuple[float, float]:
    """MFE/MAE sur `horizon` bougies APRÈS l'index `idx` (ex-post).

    Pour un LOW (long) : favorable = hausse, adverse = baisse.
    Pour un HIGH (short) : favorable = baisse, adverse = hausse.

    `mae_floor` : drawdown minimal réaliste (frais + slippage). Évite qu'un
    pivot pile sur un creux donne MAE≈0 -> RR infini (cf. ZoneConfig.mae_floor).

    Renvoie (mfe, mae) en ratios positifs (ex : 0.04 = 4 %).
    Lookahead borné par la fin du chunk (jamais hors des bornes fournies).
    """
    _require_ohlc(df)
    n = len(df)
    entry = float(df["close"].iloc[idx])
    end = min(n, idx + 1 + horizon)
    if end <= idx + 1 or entry <= 0:
        return 0.0, 0.0

    fut_high = df["high"].iloc[idx + 1 : end].to_numpy(dtype=float)
    fut_low = df["low"].iloc[idx + 1 : end].to_numpy(dtype=float)

    if direction == PivotDirection.LOW:  # long
        mfe = (fut_high.max() - entry) / entry
        mae = (entry - fut_low.min()) / entry
    else:  # short
        mfe = (entry - fut_low.min()) / entry
        mae = (fut_high.max() - entry) / entry

    return max(0.0, float(mfe)), max(float(mae_floor), float(mae))


# ───────────────────────────────────────────────────────────────────────────
#  Classification de zone — Lot B1
# ───────────────────────────────────────────────────────────────────────────
def classify_zone(mfe: float, mae: float, config: ZoneConfig) -> tuple[Zone, float]:
    """Classe un point en 🟢/🟡/🔴 et renvoie (zone, quality_score ∈ [0,1]).

    Logique (cahier §3.3) :
      🟢 : RR = MFE/MAE >= green_min_rr ET MFE >= green_min_mfe
           (forte extension favorable, faible drawdown)
      🔴 : RR <= red_max_rr (le drawdown domine, stop quasi garanti)
      🟡 : entre les deux (RR moyen)

    quality_score : sigmoïde lissée de l'edge, saturée [0,1]. Sert à pondérer
    le bonus 🟢 (jamais une table indexée par le temps : c'est une qualité).
    """
    # mae a déjà reçu son plancher dans compute_mfe_mae ; on borne aussi RR ici
    # pour rester interprétable (un RR "infini" n'a pas de sens statistique).
    raw_rr = mfe / mae if mae > 1e-9 else (config.rr_cap if mfe > 0 else 0.0)
    rr = min(raw_rr, config.rr_cap)

    if rr >= config.green_min_rr and mfe >= config.green_min_mfe:
        zone = Zone.GREEN
    elif rr <= config.red_max_rr:
        zone = Zone.RED
    else:
        zone = Zone.ORANGE

    # quality_score : combine l'edge (rr borné) et l'amplitude (mfe), borné [0,1].
    rr_clip = min(rr, config.rr_cap) / config.rr_cap
    mfe_clip = min(mfe / 0.05, 1.0)              # 0..1 (mfe=5% -> 1.0)
    quality = float(np.clip(0.5 * rr_clip + 0.5 * mfe_clip, 0.0, 1.0))
    return zone, quality


# ───────────────────────────────────────────────────────────────────────────
#  Pipeline complet : chunk -> 10-15 points critiques — Lot B5 + B1
# ───────────────────────────────────────────────────────────────────────────
def build_critical_points(
    df: pd.DataFrame,
    config: Optional[ZoneConfig] = None,
    regime: Optional[int] = None,
) -> list[CriticalPoint]:
    """Pipeline : pivots -> MFE/MAE -> zones -> sélection des 10-15 meilleurs.

    `regime` : étiquette de régime HMM du chunk (optionnelle), simplement
    propagée dans chaque CriticalPoint (jamais calculée ici).

    Renvoie une liste de CriticalPoint triés par index croissant.
    Sélection : on garde au plus `config.max_points`, priorité aux zones
    extrêmes (🟢 et 🔴) et à la plus forte |quality - 0.5| (= signal le plus net).
    """
    config = config or ZoneConfig()
    _require_ohlc(df)

    pivots = _detect_pivots_adaptive(df, config)
    points: list[CriticalPoint] = []
    for p in pivots:
        mfe, mae = compute_mfe_mae(
            df, p.idx, p.direction, config.horizon, mae_floor=config.mae_floor
        )
        if mfe == 0.0 and mae == 0.0:
            continue  # pas de futur exploitable (fin de chunk)
        zone, quality = classify_zone(mfe, mae, config)
        confidence = _pivot_confidence(df, p, config)
        points.append(
            CriticalPoint(
                idx=p.idx,
                direction=p.direction,
                price=p.price,
                timestamp=p.timestamp,
                mfe=mfe,
                mae=mae,
                zone=zone,
                quality_score=quality,
                confidence=confidence,
                regime=regime,
                horizon=config.horizon,
                rr_cap=config.rr_cap,
            )
        )

    return _select_top(points, config)


def _select_top(points: list[CriticalPoint], config: ZoneConfig) -> list[CriticalPoint]:
    """Garde les points les plus informatifs (max_points), triés par idx."""
    if len(points) <= config.max_points:
        return sorted(points, key=lambda c: c.idx)

    def informativeness(c: CriticalPoint) -> float:
        # signal net = zone extrême + confiance + écart de qualité au neutre
        extreme = 1.0 if c.zone in (Zone.GREEN, Zone.RED) else 0.4
        return extreme * (0.5 + 0.5 * c.confidence) * (0.5 + abs(c.quality_score - 0.5))

    top = sorted(points, key=informativeness, reverse=True)[: config.max_points]
    return sorted(top, key=lambda c: c.idx)


def _pivot_confidence(df: pd.DataFrame, pivot: Pivot, config: ZoneConfig) -> float:
    """Confiance [0,1] dans le pivot : amplitude locale + volume relatif.

    Pur, borné ; volume optionnel. Sert uniquement à classer la sélection.
    """
    n = len(df)
    k = config.fractal_k
    lo = max(0, pivot.idx - k)
    hi = min(n, pivot.idx + k + 1)
    local_high = float(df["high"].iloc[lo:hi].max())
    local_low = float(df["low"].iloc[lo:hi].min())
    amplitude = (local_high - local_low) / pivot.price if pivot.price else 0.0
    amp_score = min(amplitude / 0.01, 1.0)  # 1 % d'amplitude locale -> 1.0

    vol_score = 0.5
    if "volume" in df.columns:
        vols = df["volume"].iloc[lo:hi].to_numpy(dtype=float)
        med = np.median(df["volume"].to_numpy(dtype=float)) or 1.0
        vol_score = float(np.clip(vols.mean() / med / 2.0, 0.0, 1.0))

    return float(np.clip(0.6 * amp_score + 0.4 * vol_score, 0.0, 1.0))


# ───────────────────────────────────────────────────────────────────────────
#  Helpers
# ───────────────────────────────────────────────────────────────────────────
def _require_ohlc(df: pd.DataFrame) -> None:
    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(
            f"future_zones: colonnes OHLC manquantes {missing}. "
            f"Colonnes reçues : {list(df.columns)[:12]}..."
        )
