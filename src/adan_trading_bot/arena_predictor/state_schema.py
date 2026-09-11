"""V18 — Arena Bayésien Prédictif : schéma d'état PRÉSENT-SEULEMENT.

Le vecteur d'état décrit UNIQUEMENT le présent du marché à l'ouverture d'une
position. Aucune information future n'y figure -> compatible live.

Ce schéma est le contrat partagé entre :
  - le Collector (qui enregistre state -> params optimaux ex-post),
  - le Predictor (qui apprend state -> distribution des params),
  - l'Estimator (qui, en live, prédit les params depuis le state présent).

RÈGLE D'OR (cahier §10.10) : le futur ne sert qu'à fabriquer les CIBLES
(labels) hors-ligne. Il n'entre JAMAIS dans le vecteur d'état ci-dessous.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import math

# Ordre canonique des features d'état présent. NE PAS réordonner (le modèle
# entraîné dépend de cet ordre). Ajouter de nouvelles features UNIQUEMENT à la
# fin, et bumper STATE_SCHEMA_VERSION.
STATE_FEATURES: List[str] = [
    "atr_pct",          # volatilité ATR / close (présent)
    "rsi",              # RSI normalisé /100
    "adx",              # ADX normalisé /100
    "bb_percent_b",     # position dans les bandes de Bollinger
    "volatility_ratio", # ratio de volatilité court/long terme
    "volume_ratio",     # volume relatif vs moyenne 20
    "ema_ratio",        # prix / EMA (tendance présente)
    "macdh",            # histogramme MACD (momentum présent)
    "di_delta",         # DI+ - DI- (direction présente)
    "regime",           # régime HMM /  encodé [0..1] (optionnel, 0.5 si absent)
    "tf_onehot_5m",     # timeframe courant one-hot
    "tf_onehot_1h",
    "tf_onehot_4h",
]

STATE_DIM = len(STATE_FEATURES)
STATE_SCHEMA_VERSION = "1.0.0"

# Paramètres cibles que le prédicteur apprend (distributions gaussiennes).
# Chacun est prédit comme (mean, log_std) -> incertitude native.
TARGET_PARAMS: List[str] = [
    "break_even",   # seuil de rentabilité minimal (fraction, ex 0.004)
    "tp",           # take-profit optimal (fraction du prix, ex 0.018)
    "sl",           # stop-loss optimal (fraction du prix, ex 0.008)
    "duration",     # durée de détention optimale (steps, normalisée /500)
    "confidence",   # proba que le trade soit profitable [0..1]
]
TARGET_DIM = len(TARGET_PARAMS)


def _safe(x, default=0.0):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float(default)
        return v
    except Exception:
        return float(default)


@dataclass
class PresentState:
    """Snapshot présent-seulement du marché à l'ouverture d'un trade."""
    atr_pct: float = 0.0
    rsi: float = 0.5
    adx: float = 0.0
    bb_percent_b: float = 0.5
    volatility_ratio: float = 1.0
    volume_ratio: float = 1.0
    ema_ratio: float = 1.0
    macdh: float = 0.0
    di_delta: float = 0.0
    regime: float = 0.5
    timeframe: str = "5m"

    def to_vector(self) -> List[float]:
        tf = str(self.timeframe or "5m").lower()
        return [
            _safe(self.atr_pct),
            _safe(self.rsi, 0.5),
            _safe(self.adx),
            _safe(self.bb_percent_b, 0.5),
            _safe(self.volatility_ratio, 1.0),
            _safe(self.volume_ratio, 1.0),
            _safe(self.ema_ratio, 1.0),
            _safe(self.macdh),
            _safe(self.di_delta),
            _safe(self.regime, 0.5),
            1.0 if tf == "5m" else 0.0,
            1.0 if tf == "1h" else 0.0,
            1.0 if tf == "4h" else 0.0,
        ]

    @staticmethod
    def from_market_row(row: Dict[str, float], timeframe: str = "5m",
                        regime: Optional[float] = None) -> "PresentState":
        """Construit un PresentState depuis une ligne de features de marché.

        `row` peut contenir des noms de colonnes variables selon le TF
        (rsi_14 / rsi_21 / rsi_28, ema_20_ratio / ema_50_ratio, ...). On
        cherche par préfixe pour rester robuste au timeframe.
        """
        def pick(prefixes, default=0.0):
            for k, v in row.items():
                kl = str(k).lower()
                for p in prefixes:
                    if kl.startswith(p):
                        return _safe(v, default)
            return float(default)

        rsi_raw = pick(["rsi"], 50.0)
        adx_raw = pick(["adx"], 0.0)
        return PresentState(
            atr_pct=pick(["atr_pct", "atrr"], 0.0),
            rsi=rsi_raw / 100.0 if rsi_raw > 1.5 else rsi_raw,
            adx=adx_raw / 100.0 if adx_raw > 1.5 else adx_raw,
            bb_percent_b=pick(["bb_percent_b", "bbp"], 0.5),
            volatility_ratio=pick(["volatility_ratio"], 1.0),
            volume_ratio=pick(["volume_ratio"], 1.0),
            ema_ratio=pick(["ema_"], 1.0),
            macdh=pick(["macdh"], 0.0),
            di_delta=pick(["di_delta"], 0.0),
            regime=0.5 if regime is None else _safe(regime, 0.5),
            timeframe=timeframe,
        )


@dataclass
class TrainingSample:
    """Un échantillon (état présent -> params optimaux ex-post)."""
    state: List[float]
    break_even: float
    tp: float
    sl: float
    duration: float
    confidence: float
    meta: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)
