"""
geometry_engine.py — canonical ATR/SL admission; full geometry NOT locked

The historical discussion below is exploratory. SL bounds are requested
constraints, TP3.5R is a baseline, and TP_MAX/fills/portfolio economics remain
unresolved. No optimality or live-trading authorization is implied.
====================================================================

Grave dans le code les conclusions ÉCONOMIQUES de la Phase 0b (commit deb24df,
dev.md §Phase 0b) — leçon apprise sur 946 633 bougies BTCUSDT :

  L'EV BRUTE du signal sweep est positive sur toute la grille testée
  (+0.065R … +0.147R), mais l'EV NETTE n'émerge que si les frais ramenés
  en unités de risque restent absorbables :

      frais_R = frais_RT / sl_distance_pct

  · SL « micro-mèche » ~0.15 % → frais_R ≈ 2.67 R  → trade mort d'avance
  · SL structurel 1.2 %      → frais_R ≈ 0.33 R (taker) / 0.07 R (maker)

  D'où les invariants NON NÉGOCIABLES gravés ici :
    1. SL_distance = max(1.2 %, 1.0 × ATR_1h %)   — plancher structurel,
       interdiction formelle du micro-SL qui gonfle les frais en R.
    2. TP = 3.5 × SL_distance                     — expansion du contenant 4h.
    3. Frais_R ≤ 0.30 R                           — refus géométrique sinon.
    4. EV nette = P(win)·TP_R − (1−P(win))·1 − frais_R  > 0 exigé.

L'IA ne choisit jamais une sortie arbitraire : le stop est une INVALIDATION
structurelle (mèche/ATR/niveau), la cible une EXPANSION du contenant.

Référence : workflow étape 7 + dev.md §Phase 0b (conséquences gravées).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
from adan_trading_bot.features.feature_availability_contract import FeatureAvailabilityError

# Requested SL constraints, measured on TRAIN; not permanently optimal geometry.
SL_FLOOR_PCT = 0.012          # plancher structurel : 1.2 %
SL_ATR_MULT = 1.0             # alternative : 1.0 × ATR (du contenant 1h)
SL_MIN_BOUND = 0.012          # SL_MIN = max(1.20%, 1.0 × ATR_1h)
SL_MAX_BOUND = 0.030          # SL_MAX = min(3.00%, 2.5 × ATR_1h)
SL_ATR_MAX_MULT = 2.5
TP_R_RATIO = 3.5              # exploratory Phase0b baseline, not an optimality claim
TP_MIN_R = 3.5                # baseline
from adan_trading_bot.policy.market_contract import FEES_R_MAX, load_market_contract, MarketContractError
FEES_RT_TAKER = 0.0040        # 0.40 % aller-retour (stress-test)
FEES_RT_MAKER = 0.0008        # 0.08 % aller-retour (ordres limit post-only)


@dataclass(frozen=True)
class PlanCandidate:
    """Spécification d'un plan de trading candidat."""
    direction: str                     # LONG entry only under SPOT
    sl_pct: float                      # ex: 0.012 (1.2%)
    tp_r: float                        # ex: 3.5 (3.5R)
    horizon: int = 288                 # 288 barres 5m = 24h
    execution_mode: str = "MAKER_POST_ONLY"
    portfolio_state: Optional[object] = None
    atr_observation: Optional[object] = None
    sl_min_bound: Optional[float] = None
    sl_max_bound: Optional[float] = None
    tp_status: str = "BASELINE_ONLY_TP_MAX_UNRESOLVED"
    market: str = "SPOT"

    def admissible(self, market_contract=None):
        market = market_contract or load_market_contract()
        return (self.market == market.market and self.direction in market.entry_directions and self.horizon > 0
                and self.atr_observation is not None and self.atr_observation.available
                and self.sl_min_bound is not None and self.sl_max_bound is not None
                and math.isfinite(self.sl_pct) and math.isfinite(self.tp_r)
                and self.sl_min_bound <= self.sl_pct <= self.sl_max_bound
                and self.tp_r >= TP_MIN_R)


@dataclass(frozen=True)
class TradeGeometry:
    """Géométrie validée d'un trade candidat (ou rejet motivé)."""
    viable: bool
    direction: str              # 'LONG' | 'SHORT' | 'NONE'
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    sl_distance_pct: float = 0.0
    tp_distance_pct: float = 0.0
    risk_r: float = 1.0
    frais_r: float = 0.0
    ev_brute_r: float = 0.0
    ev_nette_r: float = 0.0
    motif_refus: str = ""
    candidate_plan: Optional[PlanCandidate] = None


def _atr_pct_from_snapshot(snap, availability_contract=None) -> float:
    """Canonical fraction only, admitted by feature contract; never a proxy."""
    if availability_contract is None:
        raise FeatureAvailabilityError("Canonical ATR requires feature availability contract")
    values = availability_contract.materialize(snap, ["c1h.atr_1h", "c1h.atr_1h_pct"])
    return values["c1h.atr_1h_pct"]


def compute_geometry(
    direction: str,
    entry: float,
    invalidation_level: float,
    atr_1h_pct: Optional[float] = None,
    p_win: float = 0.55,
    fees_rt: Optional[float] = None,
    *, snapshot=None, availability_contract=None,
) -> TradeGeometry:
    """Construit et valide la géométrie d'un trade candidat.

    direction          : 'LONG' ou 'SHORT'
    entry              : prix d'entrée (open de la 5m suivante, typiquement)
    invalidation_level : niveau structurel d'invalidation (mèche/niveau) —
                         le SL technique, AVANT application du plancher.
    atr_1h_pct         : optional fraction cross-check, NEVER the authoritative source;
                         snapshot + availability_contract are mandatory
    p_win              : probabilité calibrée de succès (issue du JEV, étape 4)
    fees_rt            : régime de frais aller-retour (maker par défaut)

    Retourne un TradeGeometry — viable=False avec motif si un invariant tombe.
    """
    from adan_trading_bot.policy.market_contract import load_market_contract, MarketContractError
    if fees_rt is None:
        fees_rt = load_market_contract().cost_rt      # configured spot per-side costs x2
    try:
        load_market_contract().require_direction(direction)
    except MarketContractError as error:
        return TradeGeometry(False, "NONE", motif_refus=f"direction invalide: {error}")
    if entry <= 0:
        return TradeGeometry(False, "NONE", motif_refus="entry invalide")

    try:
        fraction = _atr_pct_from_snapshot(snapshot, availability_contract)
        floor_pct, ceiling_pct = compute_sl_bounds(fraction)
        floor_pct = max(floor_pct, fees_rt / FEES_R_MAX)
    except (FeatureAvailabilityError, ValueError, AttributeError) as error:
        return TradeGeometry(False, "NONE", motif_refus=f"ATR unavailable: {error}")
    if atr_1h_pct is not None and not math.isclose(atr_1h_pct, fraction, rel_tol=1e-12, abs_tol=1e-12):
        return TradeGeometry(False, "NONE", motif_refus="ATR scalar differs from canonical snapshot fraction")
    if not math.isfinite(entry) or not math.isfinite(invalidation_level) or not (0 <= p_win <= 1) or not math.isfinite(fees_rt) or fees_rt < 0:
        return TradeGeometry(False, "NONE", motif_refus="invalid economics or price")
    if floor_pct > ceiling_pct:
        return TradeGeometry(False, "NONE", motif_refus="SL_MIN > SL_MAX")
    structural_pct = abs(entry - invalidation_level) / entry if invalidation_level > 0 else 0.0
    sl_pct = max(structural_pct, floor_pct)
    if sl_pct > ceiling_pct:
        return TradeGeometry(False, "NONE", motif_refus="structural SL above canonical SL_MAX")

    # ── Invariant 2 : cible = expansion du contenant (3.5 R) ─────────────────
    tp_pct = TP_R_RATIO * sl_pct

    if direction == "SHORT":
        stop = entry * (1 + sl_pct)
        target = entry * (1 - tp_pct)
    else:
        stop = entry * (1 - sl_pct)
        target = entry * (1 + tp_pct)

    # ── Invariant 3 : viabilité économique (leçon Phase 0b) ──────────────────
    frais_r = fees_rt / sl_pct
    if frais_r > FEES_R_MAX:
        return TradeGeometry(
            False, "NONE", entry=entry, sl_distance_pct=sl_pct,
            frais_r=frais_r,
            motif_refus=(f"géométrie non viable : frais {frais_r:.2f}R > "
                         f"{FEES_R_MAX}R (SL {sl_pct*100:.2f}% trop serré)"),
        )

    # ── Invariant 4 : EV nette positive exigée ───────────────────────────────
    ev_brute = p_win * TP_R_RATIO - (1.0 - p_win) * 1.0
    ev_nette = ev_brute - frais_r
    if ev_nette <= 0:
        return TradeGeometry(
            False, "NONE", entry=entry, sl_distance_pct=sl_pct,
            tp_distance_pct=tp_pct, frais_r=frais_r,
            ev_brute_r=ev_brute, ev_nette_r=ev_nette,
            motif_refus=f"EV nette {ev_nette:+.3f}R ≤ 0 (P(win)={p_win:.2f})",
        )

    return TradeGeometry(
        True, direction, entry=entry, stop_loss=stop, take_profit=target,
        sl_distance_pct=sl_pct, tp_distance_pct=tp_pct, risk_r=1.0,
        frais_r=frais_r, ev_brute_r=ev_brute, ev_nette_r=ev_nette,
        candidate_plan=PlanCandidate(direction, sl_pct, TP_MIN_R,
                                     atr_observation=snapshot.atr_1h,
                                     sl_min_bound=floor_pct, sl_max_bound=ceiling_pct,
                                     portfolio_state=snapshot.portfolio.copy()),
    )


def compute_sl_bounds(atr_1h_pct: float) -> Tuple[float, float]:
    """
    Calcule les bornes d'admissibilité du Stop Loss (ORDRE 4A) :
      SL_MIN = max(1.20%, 1.0 × ATR_1h)
      SL_MAX = min(3.00%, 2.5 × ATR_1h)
    """
    if not math.isfinite(atr_1h_pct) or atr_1h_pct < 0:
        raise ValueError("ATR must be an available finite fraction, not percent-points")
    sl_min = max(SL_MIN_BOUND, SL_ATR_MULT * atr_1h_pct)
    sl_max = min(SL_MAX_BOUND, SL_ATR_MAX_MULT * atr_1h_pct)
    return sl_min, sl_max


def generate_candidate_grid(snapshot, direction: str, *, availability_contract,
                            horizon: int = 288, market_contract=None) -> List[PlanCandidate]:
    """Provenance-bearing SL candidates; TP3.5R is baseline ONLY.

    No invented phase/regime TP_MAX. Conditional TP research is a later gate.
    Warmup/gap or inadmissible SL interval means no candidate, not a zero ATR.
    """
    from adan_trading_bot.policy.market_contract import load_market_contract
    market = market_contract or load_market_contract()
    market.require_direction(direction)          # SHORT under SPOT raises, never silently dropped
    if horizon <= 0:
        return []
    try:
        fraction = _atr_pct_from_snapshot(snapshot, availability_contract)
        sl_min, sl_max = compute_sl_bounds(fraction)
    except (FeatureAvailabilityError, ValueError, AttributeError):
        return []
    # Cost feasibility: round-trip costs must stay <= FEES_R_MAX in R units.
    sl_min = max(sl_min, market.min_sl_for_costs)
    if sl_min > sl_max:
        return []
    sls = sorted(set([sl_min, sl_max] + [s for s in (0.012, 0.015, 0.018, 0.020, 0.025, 0.030)
                                     if sl_min <= s <= sl_max]))
    plans = [PlanCandidate(direction, sl, TP_MIN_R, horizon=horizon, market=market.market,
                           portfolio_state=snapshot.portfolio.copy(),
                           atr_observation=snapshot.atr_1h,
                           sl_min_bound=sl_min, sl_max_bound=sl_max) for sl in sls]
    if not all(plan.admissible(market) for plan in plans):
        raise ValueError("Generated inadmissible candidate")
    return plans


def select_best_plan(
    snapshot,
    candidate_evaluations: List[Tuple[PlanCandidate, float]], # List of (plan, p_win)
    fees_rt: Optional[float] = None,
    *, availability_contract=None,
) -> Tuple[Optional[PlanCandidate], Optional[TradeGeometry]]:
    """
    Sélectionne le meilleur plan admissible (ORDRE 4B) :
      - Rejette si frais_R > 0.30 R
      - Rejette si EV_nette ≤ 0
      - Recherche le plan maximisant la probabilité de succès P(TP avant SL)
        sous contrainte d'EV nette positive.
    """
    best_plan = None
    best_geom = None
    best_ev = float("-inf")
    entry = snapshot.price
    if fees_rt is None:
        from adan_trading_bot.policy.market_contract import load_market_contract
        fees_rt = load_market_contract().cost_rt
    try:
        fraction = _atr_pct_from_snapshot(snapshot, availability_contract)
        sl_min, sl_max = compute_sl_bounds(fraction)
        sl_min = max(sl_min, fees_rt / FEES_R_MAX)
    except (FeatureAvailabilityError, ValueError, AttributeError):
        return None, None
    if sl_min > sl_max or not math.isfinite(fees_rt) or fees_rt < 0:
        return None, None

    market = load_market_contract()
    for plan, p_win in candidate_evaluations:
        market.require_direction(plan.direction)
        if (not plan.admissible(market) or not sl_min <= plan.sl_pct <= sl_max
                or plan.sl_min_bound != sl_min or plan.sl_max_bound != sl_max
                or plan.atr_observation != snapshot.atr_1h
                or not math.isfinite(p_win) or not 0 <= p_win <= 1
                or plan.tp_r != TP_MIN_R or plan.tp_status != "BASELINE_ONLY_TP_MAX_UNRESOLVED"):
            continue
        frais_r = fees_rt / plan.sl_pct
        if frais_r > FEES_R_MAX:
            continue

        ev_brute = p_win * plan.tp_r - (1.0 - p_win) * 1.0
        ev_nette = ev_brute - frais_r
        if ev_nette <= 0:
            continue

        # Calcul des prix cibles
        if plan.direction == "SHORT":
            stop = entry * (1.0 + plan.sl_pct)
            target = entry * (1.0 - plan.tp_r * plan.sl_pct)
        else:
            stop = entry * (1.0 - plan.sl_pct)
            target = entry * (1.0 + plan.tp_r * plan.sl_pct)

        geom = TradeGeometry(
            viable=True,
            direction=plan.direction,
            entry=entry,
            stop_loss=stop,
            take_profit=target,
            sl_distance_pct=plan.sl_pct,
            tp_distance_pct=plan.tp_r * plan.sl_pct,
            risk_r=1.0,
            frais_r=frais_r,
            ev_brute_r=ev_brute,
            ev_nette_r=ev_nette,
            candidate_plan=plan
        )

        if ev_nette > best_ev:
            best_ev = ev_nette
            best_plan = plan
            best_geom = geom

    return best_plan, best_geom
