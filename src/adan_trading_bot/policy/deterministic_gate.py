"""
deterministic_gate.py — ADAN-System-One · Étapes 5 & 6 (Abstention & Direction)
================================================================================

Le gardien de l'abstention — le droit fondamental qui manquait à PPO :
**0 à 5 trades par jour MAXIMUM**, jamais de sur-trading, jamais de gradient
de punition pour le silence. Une opportunité n'est validée que si AUCUNE
raison de ne pas agir n'existe.

Chaîne de refus (workflow étape 5, dans l'ordre — premier refus gagne) :
  1. integrity_ok == False          → HOLD (données corrompues)
  2. quota journalier atteint (≥ 5) → HOLD (jamais de sur-trading)
  3. cooldown actif                 → HOLD
  4. anomalie détectée (JEV)        → HOLD
  5. incertitude élevée (JEV)       → HOLD
  6. EV nette ≤ 0 (géométrie, étape 7 déjà calculée)
  7. régime incompatible (TRAP)     → HOLD

Direction (étape 6) — SOUS CONTRAT SPOT STRICT (policy/market_contract.py) :
  sweep_low_1h  (piège en-dessous) → BUY (entrée LONG)
  sweep_high_1h (piège au-dessus)  → HOLD si flat (aucun SHORT en spot) ;
                                     la sortie d'un long ouvert relève du
                                     lifecycle manager (SELL_EXIT), pas du gate
  ni l'un ni l'autre / les deux    → HOLD

La probabilité P(win) vient du jugement calibré (system_one_core) ; la
géométrie est validée par geometry_engine (étape 7) AVANT la décision finale.

Référence : workflow étapes 5-6 + dev.md §ADAN-SYSTEM-ONE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .geometry_engine import TradeGeometry

# ─── Constantes du gardien (workflow) ─────────────────────────────────────────
MAX_TRADES_PER_DAY = 5          # quota journalier strict (jamais dépassé)
P_ANOMALY_MAX = 0.35            # P(anomalie) au-delà → HOLD
P_SWEEP_MIN = 0.50              # P(sweep confirmé) en dessous → HOLD
REGIME_BLOCKED = "TRAP"         # régime TRAP → abstention structurelle
P_REGIME_TRAP_MAX = 0.40        # si P(TRAP) > 0.40 → HOLD


@dataclass(frozen=True)
class GateDecision:
    """Verdict du gardien : HOLD motivé ou GO avec direction."""
    go: bool
    direction: str                      # entry direction admitted by market contract, or 'NONE'
    motif: str
    checks: List[str] = field(default_factory=list)   # journal des filtres passés
    action: str = "HOLD"                # BUY | HOLD (SELL_EXIT handled by lifecycle manager)


def evaluate(snapshot, judgment, geometry: TradeGeometry,
             trades_today: int = 0, cooldown_active: bool = False) -> GateDecision:
    """Évalue si l'état S_t + le jugement calibré + la géométrie autorisent
    un trade. Retourne GateDecision(go, direction, motif, checks)."""
    checks: List[str] = []

    # 1. Intégrité des données (étape 2) — veto absolu
    if not snapshot.integrity_ok:
        return GateDecision(False, "NONE", "intégrité des données non prouvée", checks)
    checks.append("integrity_ok")

    # 2. Quota journalier — jamais de sur-trading
    if trades_today >= MAX_TRADES_PER_DAY:
        return GateDecision(False, "NONE",
                            f"quota journalier atteint ({trades_today}/{MAX_TRADES_PER_DAY})", checks)
    checks.append(f"quota {trades_today}/{MAX_TRADES_PER_DAY}")

    # 3. Cooldown
    if cooldown_active:
        return GateDecision(False, "NONE", "cooldown actif", checks)
    checks.append("cooldown écoulé")

    # 4. Anomalie détectée par le JEV
    p_anomaly = judgment.noul.get("anomalie_donnees", 0.0)
    if p_anomaly > P_ANOMALY_MAX:
        return GateDecision(False, "NONE",
                            f"anomalie probable (P={p_anomaly:.2f} > {P_ANOMALY_MAX})", checks)
    checks.append(f"anomalie P={p_anomaly:.2f}")

    # 5. Régime TRAP (marché piège) — abstention structurelle
    p_trap = judgment.choice.get("regime", {}).get(REGIME_BLOCKED, 0.0)
    if p_trap > P_REGIME_TRAP_MAX:
        return GateDecision(False, "NONE",
                            f"régime TRAP dominant (P={p_trap:.2f})", checks)
    checks.append(f"régime TRAP P={p_trap:.2f}")

    # 6. Géométrie économiquement viable (déjà validée à l'étape 7)
    if not geometry.viable:
        return GateDecision(False, "NONE",
                            f"géométrie refusée : {geometry.motif_refus}", checks)
    checks.append(f"EV nette {geometry.ev_nette_r:+.3f}R")

    # ── Étape 6 : DIRECTION ──────────────────────────────────────────────────
    p_sweep_ok = judgment.noul.get("sweep_confirme", 0.0)
    from .market_contract import load_market_contract
    market = load_market_contract()
    direction = "NONE"
    if snapshot.sweep_high_1h and not snapshot.sweep_low_1h:
        candidate = "SHORT"          # piège au-dessus
        if candidate not in market.entry_directions:
            return GateDecision(False, "NONE", f"sweep haut : SHORT interdit en {market.market} → HOLD", checks)
        direction = candidate
    elif snapshot.sweep_low_1h and not snapshot.sweep_high_1h:
        direction = "LONG"           # piège en-dessous → on achète la réintégration

    if direction == "NONE":
        return GateDecision(False, "NONE",
                            "aucun sweep directionnel exploitable", checks)
    if p_sweep_ok < P_SWEEP_MIN:
        return GateDecision(False, "NONE",
                            f"sweep non confirmé par le JEV (P={p_sweep_ok:.2f} < {P_SWEEP_MIN})",
                            checks)
    checks.append(f"sweep confirmé P={p_sweep_ok:.2f}")

    # Cohérence direction demandée / direction détectée
    if geometry.direction != direction:
        return GateDecision(False, "NONE",
                            f"incohérence direction géométrie ({geometry.direction}) "
                            f"vs signal ({direction})", checks)

    return GateDecision(True, direction,
                        f"opportunité validée (BUY {direction}, EV {geometry.ev_nette_r:+.3f}R)",
                        checks, action="BUY")
