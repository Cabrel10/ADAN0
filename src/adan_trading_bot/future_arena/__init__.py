"""future_arena — Arène Guidée par le Futur (ADAN0 v2).

Module microservice ISOLABLE (cf. cahier des charges §5, §10).

Responsabilité unique : à partir d'un chunk OHLCV (≈ 1 journée), calculer
EX-POST (lookahead autorisé UNIQUEMENT à l'entraînement) les informations
privilégiées qui serviront à façonner la récompense :

  - pivots (points critiques) du chunk            -> Lot B5
  - MFE / MAE forward par pivot                    -> Lot B1
  - classification de zone 🟢 / 🟡 / 🔴            -> Lot B1
  - mèches futures (Future Wick Ratios)            -> §3.2

GARANTIES DE DÉCOUPLAGE (anti-enlisement) :
  * AUCUN import du reste de adan_trading_bot.
  * AUCUN état global, AUCun I/O caché : fonctions pures (DataFrame -> objets).
  * Le régime HMM est un PARAMÈTRE optionnel, jamais calculé ici.
  * Le futur n'est JAMAIS injecté dans l'observation de l'acteur : ces objets
    servent uniquement au calcul de récompense / au critique privilégié.

Ces objets ne doivent JAMAIS être indexés par le temps dans le reward
(danger « oracle parfait », cahier §10.10) : ils décrivent la QUALITÉ
statistique d'une décision, pas « la bonne réponse à l'instant t ».
"""

from .future_zones import (
    Zone,
    Pivot,
    PivotDirection,
    ZoneConfig,
    wick_ratios,
    detect_pivots,
    compute_mfe_mae,
    classify_zone,
    build_critical_points,
)
from .wick_stats import (
    WickDistribution,
    SLTPTargets,
    compute_wick_distribution,
    compute_distributions_by_regime,
    derive_sltp_targets,
    sl_quality,
    tp_quality,
)
from .escalation import (
    EscalationConfig,
    EscalationTracker,
)
from .reward_service import (
    RewardMode,
    RewardConfig,
    RewardService,
    TradeOutcome,
    RewardBreakdown,
    ROUND_TRIP_FEES_DEFAULT,
    profile_tf_targets,
    net_pnl,
    entry_quality_score,
    sl_quality as reward_sl_quality,
    tp_quality as reward_tp_quality,
    sizing_quality,
    agent_close_barrier,
    temporal_efficiency,
    lost_potential_penalty,
    symlog,
)

__all__ = [
    "Zone",
    "Pivot",
    "PivotDirection",
    "ZoneConfig",
    "wick_ratios",
    "detect_pivots",
    "compute_mfe_mae",
    "classify_zone",
    "build_critical_points",
    "WickDistribution",
    "SLTPTargets",
    "compute_wick_distribution",
    "compute_distributions_by_regime",
    "derive_sltp_targets",
    "sl_quality",
    "tp_quality",
    "EscalationConfig",
    "EscalationTracker",
    "RewardMode",
    "RewardConfig",
    "RewardService",
    "TradeOutcome",
    "RewardBreakdown",
    "ROUND_TRIP_FEES_DEFAULT",
    "profile_tf_targets",
    "net_pnl",
    "entry_quality_score",
    "reward_sl_quality",
    "reward_tp_quality",
    "sizing_quality",
    "agent_close_barrier",
    "temporal_efficiency",
    "lost_potential_penalty",
    "symlog",
]

__schema_version__ = "1.0.0"
