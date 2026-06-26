"""reward_service — service de récompense PUR, guidé par le futur (Lot C1/C2/C3).

C'est le module qui CONSOMME ``future_zones`` / ``wick_stats`` / ``escalation``
pour transformer le "pipeline mort" en une vraie fonction objectif (cahier §10.1,
remarque utilisateur : « sans ça, tu as un pipeline mort »).

DÉCISIONS UTILISATEUR INTÉGRÉES (séance frais/anti-prévisibilité) :
  * Trading SPOT, frais fixés à 0.80 % aller-retour (``ROUND_TRIP_FEES_DEFAULT``).
  * 4 profils (scalper / intraday / swing / position) × 3 timeframes (5m / 1h / 4h).
    → les cibles SL/TP sont définies PAR (profil × timeframe), pas seulement par
      profil (cf. ``PROFILE_TF_BANDS``).
  * Anti-prévisibilité : RIEN de linéaire/statique. Les pénalités de motif stérile
    passent par ``escalation.EscalationTracker`` (croissance non-linéaire bornée +
    bruit + stateful). La pénalité n'apparaît pas à la 1ʳᵉ occurrence.

PHILOSOPHIE OBJECTIF (revue utilisateur, à NE PAS trahir) :
    Qualité du point d'entrée  +  Qualité de gestion de position  +  Profit RÉEL net
  et JAMAIS seulement « suivre les zones vertes du passé » (danger oracle, §10.10).
  Les zones n'entrent JAMAIS dans l'observation de l'acteur ; elles ne servent qu'à
  façonner la récompense ex-post via une QUALITÉ STATISTIQUE, jamais un index temps.

GARANTIES DE DÉCOUPLAGE :
  * Fonctions pures + une classe stateful encapsulée (RewardService) — pas d'état
    global, pas d'I/O, pas d'import du reste du bot. Rejouable hors-ligne (§5.2).
  * ``reward_mode ∈ {"classic", "future_guided", "stochastic_hybrid"}`` (décision
    C5 option (c)) : le futur peut être branché/débranché pour benchmark A/B.

Toutes les grandeurs PnL/SL/TP/MFE/MAE sont des RATIOS (0.04 = +4 %).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .future_zones import CriticalPoint, PivotDirection, Zone
from .escalation import EscalationConfig, EscalationTracker


# ───────────────────────────────────────────────────────────────────────────
# Frais (SPOT) — décision utilisateur : 0.80 % aller-retour.
# ───────────────────────────────────────────────────────────────────────────
ROUND_TRIP_FEES_DEFAULT: float = 0.008  # 0.40 % entrée + 0.40 % sortie (SPOT)


# ───────────────────────────────────────────────────────────────────────────
# Cibles SL/TP par (profil × timeframe).  RATIOS de prix.
#
# ⚠️ STATUT (revue utilisateur 2026-06-26) : ces cibles ne sont plus le signal
# PRINCIPAL de sl_quality/tp_quality. Le juge principal est désormais le MAE/MFE
# RÉELLEMENT observé (capture-ratio) — voir tp_quality/sl_quality. Ces tables ne
# servent QUE de FALLBACK FAIBLE quand le futur est indisponible (mode classic).
# Elles ne plafonnent JAMAIS la récompense : le marché n'a pas de TP maximum.
#
# ANTI-DETTE : toutes ces constantes sont SURCHARGEABLES depuis config.yaml
# (bloc reward_shaping.future_reward.targets / tf_scale) via configure_targets().
# Le module ne contient donc plus de constante métier "dure" non configurable.
# ───────────────────────────────────────────────────────────────────────────
# facteur d'échelle temporelle (FALLBACK) : 5m = base, 1h et 4h élargissent.
# NB : ce n'est qu'un fallback ; le potentiel réel est porté par le MFE observé,
# qui encode déjà nativement la durée de détention et le régime de marché.
_TF_SCALE_DEFAULT = {"5m": 1.0, "1h": 1.8, "4h": 3.0}
_TF_SCALE = dict(_TF_SCALE_DEFAULT)

# (sl_target, tp_target) de BASE en 5m, par profil — FALLBACK uniquement.
_PROFILE_BASE_5M_DEFAULT = {
    "scalper":  (0.012, 0.030),   # serré : on capture l'impulsion courte
    "intraday": (0.025, 0.060),
    "swing":    (0.045, 0.110),
    "position": (0.090, 0.230),
}
_PROFILE_BASE_5M = {k: tuple(v) for k, v in _PROFILE_BASE_5M_DEFAULT.items()}


def configure_targets(targets: Optional[dict] = None,
                      tf_scale: Optional[dict] = None) -> None:
    """Surcharge les cibles FALLBACK depuis config.yaml (anti-dette technique).

    Args:
        targets: {profil: [sl_target, tp_target]} en ratios, base 5m.
        tf_scale: {timeframe: facteur} d'élargissement temporel.

    Idempotent et tolérant : toute clé manquante garde la valeur par défaut.
    N'impacte QUE le fallback (pas de futur) — le chemin MFE/MAE reste prioritaire.
    """
    global _PROFILE_BASE_5M, _TF_SCALE
    if isinstance(targets, dict):
        for prof, pair in targets.items():
            try:
                sl_t, tp_t = float(pair[0]), float(pair[1])
                _PROFILE_BASE_5M[str(prof).lower()] = (sl_t, tp_t)
            except Exception:
                continue
    if isinstance(tf_scale, dict):
        for tf, fac in tf_scale.items():
            try:
                _TF_SCALE[str(tf).lower()] = float(fac)
            except Exception:
                continue


def profile_tf_targets(profile: str, timeframe: str) -> tuple[float, float]:
    """Renvoie (sl_target, tp_target) en RATIO pour (profil × timeframe).

    FALLBACK uniquement : utilisé par sl_quality/tp_quality SEULEMENT si le
    MAE/MFE observé est indisponible. Le modèle ne voit jamais ces cibles.
    """
    profile = (profile or "intraday").lower()
    timeframe = (timeframe or "5m").lower()
    base_sl, base_tp = _PROFILE_BASE_5M.get(profile, _PROFILE_BASE_5M["intraday"])
    scale = _TF_SCALE.get(timeframe, 1.0)
    return base_sl * scale, base_tp * scale


class RewardMode(str, Enum):
    """Mode de calcul (décision C5 option (c))."""

    CLASSIC = "classic"                    # PnL net + gestion, sans zones
    FUTURE_GUIDED = "future_guided"        # + EQS/zone/SL/TP qualité
    STOCHASTIC_HYBRID = "stochastic_hybrid"  # future_guided + calibrateur si TP collapse


@dataclass(frozen=True)
class RewardConfig:
    """Pondérations et garde-fous du reward-service.

    Aucune pondération n'est "magique" : chacune répond à un comportement ciblé
    (cf. grille REWARD_ANTIHACK §3.6). Les zones NE DOIVENT PAS dominer le PnL
    réel (``w_pnl`` reste le poids structurant) — garde-fou anti-oracle.
    """

    mode: RewardMode = RewardMode.FUTURE_GUIDED

    # Frais SPOT aller-retour (décision : 0.80 %).
    round_trip_fees: float = ROUND_TRIP_FEES_DEFAULT

    # Poids du PnL NET réel — reste le terme STRUCTURANT (jamais dominé par zones).
    w_pnl: float = 1.0
    # Poids qualité d'entrée (EQS) : récompense « pris au bon endroit » AVANT PnL.
    w_eqs: float = 0.35
    # Poids qualité SL / TP / sizing (gestion de position).
    w_sl: float = 0.20
    w_tp: float = 0.20
    w_sizing: float = 0.25
    # Poids de la pénalité « zone verte ratée » (HOLD/mauvais sens près d'un 🟢).
    w_missed_green: float = 0.30
    # k_wrong > k_miss : être à CONTRESENS d'un 🟢 est pire que de le rater passif.
    k_miss: float = 1.0
    k_wrong: float = 1.8

    # Barrière AGENT_CLOSE dynamique (§3.5) : seuil = barrier_mult × frais A/R.
    barrier_mult: float = 1.5
    # Constante de temps (steps) de l'efficacité temporelle par sortie gagnante.
    tau_steps: float = 12.0
    # Poids de la pénalité « destruction de potentiel futur » sur AGENT_CLOSE.
    # (remarque utilisateur : la VRAIE faille AGENT_CLOSE n'est pas le micro-gain
    #  seul, mais COUPER un trade qui allait devenir une zone verte majeure.)
    w_lost_potential: float = 0.40
    # Fraction du MFE résiduel considérée comme « raisonnablement capturable »
    # (on ne punit pas de ne pas avoir pris le pic absolu, seulement le gros gâchis).
    capturable_frac: float = 0.6

    # Le futur ne doit JAMAIS dominer : on plafonne sa contribution nette par step.
    max_future_contrib: float = 0.60

    # Config de l'escalation (anti-prévisibilité). Partagée par les motifs.
    escalation: EscalationConfig = field(default_factory=EscalationConfig)

    # Sécurité numérique pour le symlog final.
    use_symlog: bool = True


@dataclass(frozen=True)
class TradeOutcome:
    """Décrit l'événement de step à récompenser (contrat I/O versionné §5.2).

    Tout est optionnel sauf le strict nécessaire : le service tolère un step
    "sans trade" (HOLD) pour calculer pénalités de zone manquée / passivité.
    """

    # Identité du contexte (profil/timeframe) → cibles SL/TP.
    profile: str = "intraday"
    timeframe: str = "5m"

    # Action décodée [direction, size, tf, sl, tp] (sémantique RÉELLE, pas ghost).
    direction: float = 0.0       # >0 long, <0 short, ~0 hold
    size: float = 0.0            # [0,1] fraction du sizing max autorisé
    sl_chosen: float = 0.0       # ratio
    tp_chosen: float = 0.0       # ratio

    # Résultat réalisé du trade quand il y a clôture.
    closed: bool = False
    pnl_gross: float = 0.0       # ratio AVANT frais
    steps_held: int = 0
    close_reason: str = ""       # "TP" | "SL" | "AGENT_CLOSE" | "MaxDuration" | ""

    # Information privilégiée forward (futur) — None si indisponible (mode classic).
    mfe: Optional[float] = None
    mae: Optional[float] = None
    # MFE RÉSIDUEL après le point de sortie : potentiel favorable laissé sur la
    # table une fois le trade fermé (ratio, >=0). Mesure la « destruction de
    # potentiel futur » d'un AGENT_CLOSE prématuré. None si indisponible.
    mfe_residual: Optional[float] = None
    # Pivot 🟢 le plus proche dans la fenêtre de proximité (None si aucun).
    nearest_green: Optional[CriticalPoint] = None
    near_green: bool = False     # dans la fenêtre de proximité d'un 🟢 ?
    # Diagnostic d'effondrement de la tête TP (pour stochastic_hybrid).
    tp_head_entropy: Optional[float] = None


@dataclass
class RewardBreakdown:
    """Décomposition transparente (debug/logs/replay offline)."""

    pnl_net: float = 0.0
    eqs: float = 0.0
    sl_q: float = 0.0
    tp_q: float = 0.0
    sizing_q: float = 0.0
    missed_green: float = 0.0
    agent_close: float = 0.0
    lost_potential: float = 0.0     # destruction de potentiel futur (AGENT_CLOSE)
    temporal: float = 0.0
    escalation: float = 0.0
    future_contrib: float = 0.0     # somme des termes "futur" (plafonnée)
    raw: float = 0.0
    final: float = 0.0
    notes: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "pnl_net": self.pnl_net,
            "eqs": self.eqs,
            "sl_q": self.sl_q,
            "tp_q": self.tp_q,
            "sizing_q": self.sizing_q,
            "missed_green": self.missed_green,
            "agent_close": self.agent_close,
            "lost_potential": self.lost_potential,
            "temporal": self.temporal,
            "escalation": self.escalation,
            "future_contrib": self.future_contrib,
            "raw": self.raw,
            "final": self.final,
            **{f"note_{k}": v for k, v in self.notes.items()},
        }


# ───────────────────────────────────────────────────────────────────────────
# Fonctions pures de qualité (testables isolément)
# ───────────────────────────────────────────────────────────────────────────
def net_pnl(pnl_gross: float, round_trip_fees: float) -> float:
    """PnL NET = PnL brut − frais aller-retour. SPOT, donc frais déduits une fois."""
    return float(pnl_gross) - float(round_trip_fees)


def _gauss(x: float, target: float, tol: float) -> float:
    """Cloche gaussienne centrée sur ``target`` (1.0 au pic, →0 loin). tol>0."""
    if tol <= 0:
        return 1.0 if abs(x - target) < 1e-9 else 0.0
    z = (x - target) / (target * tol if target > 1e-9 else tol)
    return math.exp(-0.5 * z * z)


def entry_quality_score(mfe: Optional[float], mae: Optional[float],
                        zone: Optional[Zone]) -> float:
    """EQS ∈ [-1, +1] : qualité du POINT d'entrée, indépendante du résultat réel.

    🟢 → bonus fort, 🟡 → ~neutre, 🔴 → pénalité forte. Le score s'appuie sur le
    RR ex-post (MFE/MAE) borné, pas sur un index — donc pas d'oracle parfait.
    """
    if mfe is None or mae is None:
        return 0.0
    mae_eff = max(mae, 1e-6)
    rr = min(mfe / mae_eff, 10.0)
    # mappe rr∈[0,10] → edge∈[-1,1] de façon non linéaire (tanh), centré ~RR=1.5.
    edge = math.tanh((rr - 1.5) / 2.0)
    if zone == Zone.GREEN:
        return max(0.0, edge)           # on ne récompense que l'edge réel
    if zone == Zone.RED:
        return -abs(edge) - 0.2         # toxique : pénalité plancher même si edge~0
    return 0.5 * edge                   # orange : atténué


def sl_quality(sl_chosen: float, sl_target: float, mae: Optional[float],
               tol: float = 0.5) -> float:
    """Qualité du SL ∈ [-1, +1].

    PHILOSOPHIE (revue utilisateur) : le marché n'a pas de SL "cible" fixe. Le
    SEUL juge légitime est le MAE RÉELLEMENT observé après l'entrée (le bruit que
    le marché a effectivement infligé). La cible profil×TF (``sl_target``) n'est
    qu'un FALLBACK FAIBLE quand le MAE est indisponible (mode classic / pas de
    futur). On NE veut PAS enfermer le SL dans un plafond pédagogique arbitraire.

    Critère principal (MAE-relatif), SL idéal ≈ MAE × marge :
      - SL < MAE            → trop serré, stoppé par le bruit (pénalité forte).
      - MAE ≤ SL ≤ 2.0×MAE  → zone optimale (couvre le bruit sans gaspiller).
      - SL > 2.5×MAE        → trop large, capital exposé inutilement.
    """
    if sl_chosen <= 0:
        return -0.5
    # ── Chemin PRINCIPAL : relatif au MAE observé (pas de cible fixe) ──────────
    if mae is not None and mae > 1e-6:
        ratio = sl_chosen / mae
        if ratio < 1.0:
            # trop serré : on perd sur le bruit. Pénalité ∝ déficit, plancher -1.
            return max(-1.0, -1.0 * (1.0 - ratio))
        if ratio <= 2.0:
            # zone optimale : pic à ~1.3×MAE, décroît doucement vers 2×.
            # mappe ratio∈[1,2] → score∈[+1, +0.4] (toujours positif = bon SL).
            return float(max(0.4, 1.0 - 0.6 * (ratio - 1.0)))
        # trop large : pénalité croissante mais bornée.
        return float(max(-1.0, 0.4 - 0.5 * (ratio - 2.0)))
    # ── FALLBACK FAIBLE : pas de MAE → proximité à la cible profil×TF ──────────
    prox = _gauss(sl_chosen, sl_target, tol)
    return max(-1.0, min(1.0, 2.0 * prox - 1.0))


def tp_quality(tp_chosen: float, tp_target: float, mfe: Optional[float],
               tol: float = 0.5) -> float:
    """Qualité du TP ∈ [-1, +1].

    PHILOSOPHIE (revue utilisateur) : le marché n'a PAS de TP maximum. Pendant un
    bullrun le MFE peut être +35 % ; pendant un marché plat +1.5 %. Le Future
    Arena doit juger le TP choisi contre le POTENTIEL RÉELLEMENT OBSERVÉ (MFE),
    PAS contre une cible codée en dur (sinon « bravo, ton TP de 4 % était parfait »
    alors que le marché a fait +35 %). On mesure donc un CAPTURE-RATIO :

        capture = tp_chosen / mfe   (part du potentiel réel capturée)

      - capture très faible (< 0.25)  → énormément laissé sur la table (pénalité).
      - capture ∈ [0.4, 0.9]          → excellent : on prend gros sans être gourmand.
      - capture > 1.0 (TP > MFE)      → TP inatteignable, jamais touché (pénalité).

    La cible profil×TF (``tp_target``) n'est qu'un FALLBACK FAIBLE quand le MFE
    est indisponible. Ainsi un TP de 1.6 % est EXCELLENT si MFE=1.8 %, mais MÉDIOCRE
    si MFE=18 % — exactement le signal pédagogique voulu, sans plafond arbitraire.
    """
    if tp_chosen <= 0:
        return -0.5
    # ── Chemin PRINCIPAL : capture-ratio relatif au MFE observé ────────────────
    if mfe is not None and mfe > 1e-6:
        capture = tp_chosen / mfe
        if capture > 1.0:
            # TP au-delà du potentiel réel → ordre jamais touché. Pénalité ∝ excès.
            return float(max(-1.0, -0.6 * min(1.0, capture - 1.0) - 0.1))
        if capture >= 0.4:
            # zone optimale [0.4, 1.0] : pic vers ~0.7 (gros gain, marge anti-mèche).
            # mappe capture∈[0.4,1.0] → score∈[+0.5, +1.0, +0.6] (cloche douce).
            # +1.0 au pic capture≈0.7, légèrement < à 1.0 (ne pas viser le sommet exact).
            peak = 0.7
            spread = 0.45
            z = (capture - peak) / spread
            return float(max(0.5, 1.0 - 0.5 * z * z))
        # capture < 0.4 : argent laissé sur la table, pénalité ∝ (0.4 - capture).
        return float(max(-1.0, -1.0 * (0.4 - capture) / 0.4))
    # ── FALLBACK FAIBLE : pas de MFE → proximité à la cible profil×TF ──────────
    prox = _gauss(tp_chosen, tp_target, tol)
    return max(-1.0, min(1.0, 2.0 * prox - 1.0))


def sizing_quality(size: float, mfe: Optional[float], mae: Optional[float],
                   zone: Optional[Zone]) -> float:
    """Qualité de la taille ∈ [-1, +1] (la plus importante, §3.4(d)).

    edge fort + petite taille → sous-exploitation (pénalité douce).
    edge faible/toxique + grosse taille → sur-exposition (pénalité forte).
    """
    size = max(0.0, min(1.0, size))
    if mfe is None or mae is None:
        return 0.0
    mae_eff = max(mae, 1e-6)
    rr = min(mfe / mae_eff, 10.0)
    # taille idéale ∝ edge normalisé.
    ideal = max(0.0, min(1.0, (rr - 0.8) / 4.0))   # rr<=0.8 → 0 ; rr>=4.8 → 1
    diff = size - ideal
    if zone == Zone.RED and size > 0.2:
        return -1.0 * size                          # grosse taille en 🔴 = grave
    # pénalité asymétrique : sur-exposition plus punie que sous-exploitation.
    if diff > 0:
        return max(-1.0, -1.5 * diff)               # trop gros
    return max(-1.0, 0.8 * diff)                    # trop petit (moins puni)


def agent_close_barrier(pnl_gross: float, round_trip_fees: float,
                        barrier_mult: float) -> tuple[bool, float]:
    """Barrière de rentabilité dynamique pour AGENT_CLOSE (§3.5 volet 1).

    Renvoie (bloque, penalite_gradient). seuil = barrier_mult × frais A/R.
    Sous le seuil : l'action devrait être convertie en HOLD côté env ET on émet
    une pénalité gradient (≠ no-op actuel). Au-dessus : pas de blocage.
    """
    threshold = barrier_mult * round_trip_fees
    if pnl_gross < threshold:
        # pénalité proportionnelle au manque de rentabilité (gradient explicite).
        deficit = (threshold - pnl_gross) / max(threshold, 1e-9)
        return True, -0.15 * min(1.0, deficit)
    return False, 0.0


def temporal_efficiency(pnl_net: float, steps_held: int, tau: float) -> float:
    """Facteur d'efficacité temporelle (§3.5 volet 2).

    Bonus = PnL_net × (1 − exp(−steps_held/τ)). Couper à 2 steps → ~0 ;
    tenir un vrai cycle → ~PnL_net. N'applique le facteur que sur du PnL positif
    (on ne "récompense" pas une perte tenue longtemps).
    """
    if pnl_net <= 0 or steps_held <= 0 or tau <= 0:
        return 0.0
    return float(pnl_net) * (1.0 - math.exp(-steps_held / tau))


def lost_potential_penalty(pnl_realized: float, mfe_residual: Optional[float],
                           round_trip_fees: float, capturable_frac: float) -> float:
    """Pénalité « destruction de potentiel futur » sur sortie volontaire (≤ 0).

    Remarque utilisateur : la vraie faille AGENT_CLOSE n'est PAS le micro-gain
    seul, c'est COUPER un trade qui allait devenir une zone verte majeure. Seule
    la vision future locale (MFE résiduel après la sortie) peut le mesurer.

    Logique :
      - potentiel raisonnablement capturable = capturable_frac × MFE_résiduel,
        net des frais (on ne pénalise pas l'inatteignable, ni < frais A/R).
      - si ce potentiel dépasse ce que l'agent a réellement encaissé, on punit
        l'écart de façon NON LINÉAIRE (sqrt borné) → gradient clair mais pas
        explosif, et asymétrique (couper tôt un gros mouvement coûte cher).
      - aucune pénalité si le potentiel résiduel est négligeable (sortie OK).
    """
    if mfe_residual is None or mfe_residual <= 0:
        return 0.0
    capturable = capturable_frac * mfe_residual - round_trip_fees
    if capturable <= 0:
        return 0.0
    # gâchis = potentiel capturable non réalisé (relatif au réalisé).
    waste = max(0.0, capturable - max(0.0, pnl_realized))
    if waste <= 0:
        return 0.0
    # non-linéaire concave bornée : sqrt amplifie les petits gâchis sans exploser.
    return -min(1.0, math.sqrt(waste / max(round_trip_fees, 1e-6)) * 0.1)


def symlog(x: float) -> float:
    """Compression symétrique log (cohérente avec l'env actuel)."""
    return math.copysign(math.log1p(abs(x)), x)


# ───────────────────────────────────────────────────────────────────────────
# Service stateful (escalation encapsulée). Rejouable via seed.
# ───────────────────────────────────────────────────────────────────────────
class RewardService:
    """Calcule la récompense d'un step en consommant le futur (privilégié).

    STATEFUL par conception (escalation anti-motif) mais l'état est ENCAPSULÉ —
    pas de global, pas d'I/O. Deux services avec le même seed et la même séquence
    d'événements produisent exactement la même trajectoire (reproductibilité).
    """

    def __init__(self, config: Optional[RewardConfig] = None,
                 seed: Optional[int] = None) -> None:
        self.config = config or RewardConfig()
        self._esc = EscalationTracker(self.config.escalation, seed=seed)

    # ── API principale ───────────────────────────────────────────────────────
    def compute(self, ev: TradeOutcome) -> RewardBreakdown:
        cfg = self.config
        bd = RewardBreakdown()

        # Mode effectif (stochastic_hybrid bascule si la tête TP est collapsée).
        mode = cfg.mode
        if mode == RewardMode.STOCHASTIC_HYBRID and ev.tp_head_entropy is not None \
                and ev.tp_head_entropy < 0.1:
            bd.notes["tp_collapse_detected"] = 1.0

        use_future = mode in (RewardMode.FUTURE_GUIDED, RewardMode.STOCHASTIC_HYBRID)

        # ── 1) PnL NET réel (terme structurant, jamais dominé) ────────────────
        pnl_n = 0.0
        if ev.closed:
            pnl_n = net_pnl(ev.pnl_gross, cfg.round_trip_fees)
        bd.pnl_net = cfg.w_pnl * pnl_n

        # ── 2) Barrière AGENT_CLOSE + efficacité temporelle (§3.5) ────────────
        if ev.closed and ev.close_reason.upper() == "AGENT_CLOSE":
            blocked, pen = agent_close_barrier(
                ev.pnl_gross, cfg.round_trip_fees, cfg.barrier_mult)
            bd.agent_close = pen
            if blocked:
                bd.notes["agent_close_blocked"] = 1.0
                # motif stérile : micro-close répété → escalation.
                sterile = True
            else:
                sterile = False
            # efficacité temporelle sur la part nette positive.
            bd.temporal = temporal_efficiency(pnl_n, ev.steps_held, cfg.tau_steps)
            esc = self._esc.update("micro_close", sterile=sterile,
                                   severity=max(0.2, ev.size))
            bd.escalation += esc
        else:
            # pas de micro-close ce step → oubli passif léger.
            self._esc.passive_step("micro_close")

        # ── 3) Termes guidés par le futur (plafonnés, jamais dominants) ───────
        future_sum = 0.0
        if use_future:
            zone = _zone_from_mfe_mae(ev.mfe, ev.mae, ev.nearest_green)
            sl_t, tp_t = profile_tf_targets(ev.profile, ev.timeframe)

            is_trade = abs(ev.direction) > 1e-6 or ev.size > 1e-6
            if is_trade:
                bd.eqs = cfg.w_eqs * entry_quality_score(ev.mfe, ev.mae, zone)
                bd.sl_q = cfg.w_sl * sl_quality(ev.sl_chosen, sl_t, ev.mae)
                bd.tp_q = cfg.w_tp * tp_quality(ev.tp_chosen, tp_t, ev.mfe)
                bd.sizing_q = cfg.w_sizing * sizing_quality(
                    ev.size, ev.mfe, ev.mae, zone)
                future_sum += bd.eqs + bd.sl_q + bd.tp_q + bd.sizing_q

            # zone verte ratée / contresens (§10.11) — stateful escalation.
            bd.missed_green = self._missed_green_term(ev, zone)
            future_sum += bd.missed_green

            # oversize en zone rouge → motif stérile escaladé.
            if zone == Zone.RED and ev.size > 0.2 and is_trade:
                esc_red = self._esc.update("oversize_red", sterile=True,
                                           severity=ev.size)
                bd.escalation += esc_red
            else:
                self._esc.passive_step("oversize_red")

            # destruction de potentiel futur : ne s'applique qu'aux sorties
            # VOLONTAIRES (AGENT_CLOSE), jamais sur un SL légitime (le marché
            # a tranché) ni un TP atteint (objectif rempli).
            if ev.closed and ev.close_reason.upper() == "AGENT_CLOSE":
                pnl_realized_net = net_pnl(ev.pnl_gross, cfg.round_trip_fees)
                bd.lost_potential = cfg.w_lost_potential * lost_potential_penalty(
                    pnl_realized_net, ev.mfe_residual,
                    cfg.round_trip_fees, cfg.capturable_frac)
                future_sum += bd.lost_potential

            # plafond anti-oracle : le futur ne peut pas dominer le PnL réel.
            future_sum = max(-cfg.max_future_contrib,
                             min(cfg.max_future_contrib, future_sum))
        bd.future_contrib = future_sum

        # ── 4) Composition + symlog ───────────────────────────────────────────
        raw = (bd.pnl_net + bd.agent_close + bd.temporal
               + bd.escalation + future_sum)
        bd.raw = raw
        bd.final = symlog(raw) if cfg.use_symlog else raw
        return bd

    # ── helpers internes ──────────────────────────────────────────────────────
    def _missed_green_term(self, ev: TradeOutcome, zone: Optional[Zone]) -> float:
        """Pénalité « zone verte ratée / contresens » (§10.11), via escalation.

        - HOLD près d'un 🟢 : motif stérile → escalation (pénalité progressive).
        - Position à CONTRESENS d'un 🟢 : pénalité plus forte (k_wrong > k_miss).
        - Trade ALIGNÉ sur un 🟢 : on rembourse la dette (sterile=False).
        """
        cfg = self.config
        gp = ev.nearest_green
        if not ev.near_green or gp is None:
            self._esc.passive_step("hold_in_green")
            return 0.0

        mfe = gp.mfe if gp.mfe is not None else (ev.mfe or 0.0)
        sl_t, _ = profile_tf_targets(ev.profile, ev.timeframe)
        magnitude = (mfe / max(sl_t, 1e-6))
        is_trade = abs(ev.direction) > 1e-6

        # sens attendu : LOW pivot → long (dir>0) ; HIGH pivot → short (dir<0).
        want_long = gp.direction == PivotDirection.LOW
        aligned = (is_trade and ((want_long and ev.direction > 0)
                                 or (not want_long and ev.direction < 0)))
        wrong_way = (is_trade and not aligned)

        if aligned:
            # bon comportement : on rembourse la dette du motif.
            self._esc.update("hold_in_green", sterile=False)
            return 0.0

        severity = magnitude * (cfg.k_wrong if wrong_way else cfg.k_miss)
        if wrong_way:
            severity *= max(0.2, ev.size)   # contresens pondéré par la taille réelle
        esc = self._esc.update("hold_in_green", sterile=True, severity=severity)
        return cfg.w_missed_green * esc

    # ── persistance (mémoire entre sessions, §modules avancés) ────────────────
    def snapshot(self) -> dict:
        return {"escalation": self._esc.snapshot()}


def _zone_from_mfe_mae(mfe: Optional[float], mae: Optional[float],
                       nearest_green: Optional[CriticalPoint]) -> Optional[Zone]:
    """Déduit une zone à partir du MFE/MAE du step, ou du pivot 🟢 proche.

    Si on a un pivot 🟢 proche, sa zone prime (déjà classée par future_zones).
    Sinon on reconstruit une zone grossière depuis le RR ex-post du step.
    """
    if nearest_green is not None:
        return nearest_green.zone
    if mfe is None or mae is None:
        return None
    mae_eff = max(mae, 1e-6)
    rr = mfe / mae_eff
    if rr >= 1.5 and mfe >= 0.006:
        return Zone.GREEN
    if rr <= 0.8:
        return Zone.RED
    return Zone.ORANGE
