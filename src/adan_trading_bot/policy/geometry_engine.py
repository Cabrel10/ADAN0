"""
geometry_engine.py — ADAN-System-One · Étape 7 (Géométrie du trade)
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
from typing import Optional

# ─── Constantes prouvées par la Phase 0b (jamais modifiées sans nouvel audit) ─
SL_FLOOR_PCT = 0.012          # plancher structurel : 1.2 %
SL_ATR_MULT = 1.0             # alternative : 1.0 × ATR (du contenant 1h)
TP_R_RATIO = 3.5              # cible d'expansion : 3.5 R (cellule gagnante 0b)
FEES_R_MAX = 0.30             # refus géométrique si frais > 0.30 R
FEES_RT_TAKER = 0.0040        # 0.40 % aller-retour (stress-test)
FEES_RT_MAKER = 0.0008        # 0.08 % aller-retour (ordres limit post-only)


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


def _atr_pct_from_snapshot(snap) -> float:
    """Deprecated misleading name: running range / phase is NOT an ATR.

    LivingStateSnapshot currently carries no completed-hour ATR. Fail closed
    instead of substituting a phase-extrapolated range or zero during warmup.
    """
    raise ValueError("Canonical ATR_1h unavailable in snapshot: require 14 complete hours, TR, lag1; running range/phase is prohibited")


def compute_geometry(
    direction: str,
    entry: float,
    invalidation_level: float,
    atr_1h_pct: Optional[float] = None,
    p_win: float = 0.55,
    fees_rt: float = FEES_RT_MAKER,
) -> TradeGeometry:
    """Construit et valide la géométrie d'un trade candidat.

    direction          : 'LONG' ou 'SHORT'
    entry              : prix d'entrée (open de la 5m suivante, typiquement)
    invalidation_level : niveau structurel d'invalidation (mèche/niveau) —
                         le SL technique, AVANT application du plancher.
    atr_1h_pct         : ATR du contenant 1h en % du prix (si None → plancher seul)
    p_win              : probabilité calibrée de succès (issue du JEV, étape 4)
    fees_rt            : régime de frais aller-retour (maker par défaut)

    Retourne un TradeGeometry — viable=False avec motif si un invariant tombe.
    """
    if direction not in ("LONG", "SHORT"):
        return TradeGeometry(False, "NONE", motif_refus="direction invalide")
    if entry <= 0:
        return TradeGeometry(False, "NONE", motif_refus="entry invalide")

    # ── Invariant 1 : plancher structurel du SL ──────────────────────────────
    structural_pct = abs(entry - invalidation_level) / entry if invalidation_level > 0 else 0.0
    floor_pct = SL_FLOOR_PCT
    if atr_1h_pct and atr_1h_pct > 0:
        floor_pct = max(SL_FLOOR_PCT, SL_ATR_MULT * atr_1h_pct)
    sl_pct = max(structural_pct, floor_pct)   # JAMAIS sous le plancher

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
    )
