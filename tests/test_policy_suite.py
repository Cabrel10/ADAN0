#!/usr/bin/env python3
"""
test_policy_suite.py — Tests du bloc policy (ADAN-System-One, étapes 5-8)
==========================================================================

Vérifie les invariants gravés par la Phase 0b et le workflow :

  G1. Rejet d'un stop < 1.2 % (plancher structurel)
  G2. Plancher ATR : SL = max(1.2 %, 1.0 × ATR_1h)
  G3. Refus géométrique si frais_R > 0.30 R
  G4. EV nette avec frais maker 0.08 % correctement déduite
  G5. TP = 3.5 R exactement

  D1. integrity_ok == False → abstention immédiate
  D2. Blocage strict au 6ᵉ trade de la journée (quota 5/j)
  D3. Cooldown actif → HOLD
  D4. Anomalie JEV élevée → HOLD
  D5. Régime TRAP dominant → HOLD
  D6. Sweep haut → SHORT, sweep bas → LONG, aucun → HOLD
  D7. Sweep non confirmé par le JEV → HOLD
  D8. Chemin complet GO (tous filtres passés)

  R1. Kelly ≤ 0 → refus
  R2. Kelly fractionnaire plafonné à 20 % du capital
  R3. Taille < 15 $ → refus (minimum Binance)
  R4. Circuit breaker journalier (pertes > 5 %)
  R5. Exposition totale plafonnée à 40 %

Exécution :
  /home/ubuntu/webapp/MORNINGSTAR/miniconda3/envs/trading_env/bin/python3
      tests/test_policy_suite.py
"""

import sys
import os
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from adan_trading_bot.data.nested_state_builder import (
    NestedStateBuilder, LivingStateSnapshot,
)
from adan_trading_bot.models.system_one_core import Judgment
from adan_trading_bot.policy.geometry_engine import (
    compute_geometry as production_compute_geometry, TradeGeometry, SL_FLOOR_PCT, TP_R_RATIO,
    FEES_R_MAX, FEES_RT_MAKER,
)
from adan_trading_bot.policy.deterministic_gate import (
    evaluate, MAX_TRADES_PER_DAY,
)
from adan_trading_bot.policy.risk_engine import (
    size_position, MAX_EXPOSURE_PER_TRADE, MIN_NOTIONAL_USD,
    DAILY_LOSS_CIRCUIT,
)

PASS, FAIL = "✅", "❌"
failures = []


def check(name, cond, detail=""):
    print(f"  {PASS if cond else FAIL} {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def compute_geometry(*args, atr_1h_pct=None, **kwargs):
    """Feed production with real canonical historical ATR, not an invented scalar.

    Default fixture ATR=.64%: SL interval [1.2%,1.6%] admits original tests.
    Explicit 2% ATR case uses raw completed-hour TR=2 at price=100.
    """
    from adan_trading_bot.features.feature_registry import get_feature_registry
    from adan_trading_bot.features.feature_availability_contract import FeatureAvailabilityContract
    fraction = 0.0064 if atr_1h_pct is None else atr_1h_pct
    spread = fraction * 100 / 2
    history = pd.DataFrame({'open': 100., 'high': 100. + spread, 'low': 100. - spread,
                            'close': 100., 'volume': 1.},
                           index=pd.date_range('2020-01-01', periods=600, freq='5min'))
    snap = NestedStateBuilder(history).snapshot(300)
    return production_compute_geometry(*args, atr_1h_pct=atr_1h_pct, snapshot=snap,
                                        availability_contract=FeatureAvailabilityContract(get_feature_registry()), **kwargs)


def make_snapshot(sweep_high=1, sweep_low=0, integrity=True, price=100.0):
    """Snapshot synthétique minimal valide."""
    return LivingStateSnapshot(
        timestamp=pd.Timestamp("2024-06-01 12:00"),
        price=price,
        bar_5m=np.zeros(8), seq_5m=np.zeros((36, 8)),
        phase_1h=11 / 12, running_1h=np.array([99.0, 101.0, 98.0, 50.0]),
        pos_in_1h=0.66, sweep_high_1h=sweep_high, sweep_low_1h=sweep_low,
        prev_1h=np.array([99.0, 100.5, 98.5, 99.5, 600.0]),
        phase_4h=44 / 48, running_4h=np.array([98.0, 101.5, 97.0, 200.0]),
        pos_in_4h=0.55, sweep_high_4h=0, sweep_low_4h=0,
        prev_4h=np.array([98.0, 100.0, 97.5, 99.0, 2500.0]),
        portfolio=np.zeros(5), integrity_ok=integrity,
    )


def make_judgment(p_sweep=0.7, p_anomaly=0.1, p_trap=0.1, p_win=0.6):
    return Judgment(
        noul={"mfe_atteint_tp": p_win, "sweep_confirme": p_sweep,
              "anomalie_donnees": p_anomaly, "risque_adverse_faible": 0.7,
              "expansion_imminente": 0.65, "reintegration_valide": 0.7},
        choice={"regime": {"BULL": 0.3, "BEAR": 0.4, "RANGE": 0.2, "TRAP": p_trap},
                "direction": {"LONG": 0.2, "SHORT": 0.6, "AUCUNE": 0.2}},
        score={"qualite_setup": {"esperance": 6.5, "incertitude": 1.2, "distribution": []},
               "conviction": {"esperance": 7.0, "incertitude": 1.0, "distribution": []}},
        calibrated=True,
    )


print("=" * 72)
print("TEST policy/ — géométrie, gate d'abstention, risk engine")
print("=" * 72)

# ═══ GÉOMÉTRIE (étape 7) ═══
print("\n── G. Géométrie (invariants Phase 0b) ──")

# SPOT STRICT : entrées LONG uniquement ; coûts = contrat (2×(0.20 %+0.05 %) = 0.50 % RT).
from adan_trading_bot.policy.market_contract import load_market_contract
MARKET = load_market_contract()
COST = MARKET.cost_rt

# G0 : SHORT refusé par le contrat spot
g0 = compute_geometry("SHORT", entry=100.0, invalidation_level=101.5, p_win=0.60)
check("G0 SHORT refusé en spot", not g0.viable and "direction invalide" in g0.motif_refus)

# G1 : ATR .64% -> SL_MAX=1.6% < coût/.30=1.667% : abstention obligatoire.
g = compute_geometry("LONG", entry=100.0, invalidation_level=99.90, p_win=0.60)
check("G1 intervalle SL incompatible avec frais spot → refus", not g.viable and "SL_MIN > SL_MAX" in g.motif_refus,
      g.motif_refus)

# G2 : ATR 1h 2.0 % → SL = ATR
g2 = compute_geometry("LONG", entry=100.0, invalidation_level=99.90, atr_1h_pct=0.020, p_win=0.60)
check("G2 ATR 2.0% > plancher → SL = 2.0%", abs(g2.sl_distance_pct - 0.020) < 1e-9)

# G3 : ATR .70% -> SL_MAX=1.75%, plancher coût 1.667% admissible.
g3 = compute_geometry("LONG", entry=100.0, invalidation_level=99.90, atr_1h_pct=0.007, p_win=0.60)
check("G3 coût spot → SL plancher coût 1.667%, frais ≤ 0.30R", g3.viable and g3.frais_r <= FEES_R_MAX + 1e-12,
      f"frais_r={g3.frais_r:.3f}R")

# G4 : EV nette avec coût spot — SL 2.0 % (ATR 2 %) : frais = 0.005/0.02 = 0.25R
g4 = compute_geometry("LONG", entry=100.0, invalidation_level=98.0, atr_1h_pct=0.020, p_win=0.60)
ev_brute_attendu = 0.60 * TP_R_RATIO - 0.40
ev_nette_attendu = ev_brute_attendu - (COST / g4.sl_distance_pct)
check("G4 EV nette spot exacte", g4.viable and abs(g4.ev_nette_r - ev_nette_attendu) < 1e-6,
      f"EV={g4.ev_nette_r:+.3f}R attendu {ev_nette_attendu:+.3f}R")

check("G5 TP = 3.5×SL", abs(g4.tp_distance_pct - TP_R_RATIO * g4.sl_distance_pct) < 1e-12)

g6 = compute_geometry("LONG", entry=100.0, invalidation_level=98.0, atr_1h_pct=0.020, p_win=0.15)
check("G6 P(win)=0.15 → EV brute < 0 → refus", not g6.viable and g6.ev_brute_r < 0,
      f"EV_brute={g6.ev_brute_r:+.3f}R")

# ═══ GATE D'ABSTENTION (étapes 5-6) ═══
print("\n── D. Gate d'abstention & direction (spot) ──")

geom_ok = compute_geometry("LONG", entry=100.0, invalidation_level=98.0, atr_1h_pct=0.020, p_win=0.60)

# D1 : intégrité False → HOLD immédiat
d = evaluate(make_snapshot(integrity=False), make_judgment(), geom_ok)
check("D1 integrity_ok=False → HOLD", not d.go and d.checks == [])

# D2 : 6ᵉ trade du jour → HOLD (quota 5)
d = evaluate(make_snapshot(), make_judgment(), geom_ok, trades_today=MAX_TRADES_PER_DAY)
check("D2 6ᵉ trade (quota 5) → HOLD", not d.go and "quota" in d.motif)

# D3 : cooldown actif → HOLD
d = evaluate(make_snapshot(), make_judgment(), geom_ok, cooldown_active=True)
check("D3 cooldown actif → HOLD", not d.go)

# D4 : anomalie JEV élevée → HOLD
d = evaluate(make_snapshot(), make_judgment(p_anomaly=0.8), geom_ok)
check("D4 anomalie P=0.80 → HOLD", not d.go)

# D5 : régime TRAP dominant → HOLD
d = evaluate(make_snapshot(), make_judgment(p_trap=0.9), geom_ok)
check("D5 régime TRAP P=0.90 → HOLD", not d.go)

# D6 : spot — sweep haut → HOLD (aucun SHORT) ; sweep bas → BUY LONG ; aucun → HOLD
d = evaluate(make_snapshot(sweep_high=1, sweep_low=0), make_judgment(), geom_ok)
check("D6a sweep haut → HOLD (SHORT interdit en spot)", not d.go and "SHORT interdit" in d.motif)
d = evaluate(make_snapshot(sweep_high=0, sweep_low=1), make_judgment(), geom_ok)
check("D6b sweep bas → BUY LONG", d.go and d.direction == "LONG" and d.action == "BUY")
d = evaluate(make_snapshot(sweep_high=0, sweep_low=0), make_judgment(), geom_ok)
check("D6c aucun sweep → HOLD", not d.go)

# D7 : sweep non confirmé par le JEV → HOLD
d = evaluate(make_snapshot(sweep_high=0, sweep_low=1), make_judgment(p_sweep=0.30), geom_ok)
check("D7 sweep non confirmé (P=0.30) → HOLD", not d.go and "non confirmé" in d.motif)

# D8 : chemin complet GO
d = evaluate(make_snapshot(sweep_high=0, sweep_low=1), make_judgment(), geom_ok, trades_today=2)
check("D8 chemin complet → GO BUY LONG", d.go and d.direction == "LONG" and d.action == "BUY"
      and len(d.checks) >= 6, f"{len(d.checks)} filtres passés")

# ═══ RISK ENGINE (étape 8) ═══
print("\n── R. Risk engine (Kelly fractionnaire + plafonds) ──")

# R1 : Kelly ≤ 0 → refus (p trop faible pour b=3.5 : seuil p = 1/(1+3.5) ≈ 0.222)
r = size_position(capital=10000, p_win=0.10, tp_r=3.5, sl_distance_pct=0.012)
check("R1 Kelly ≤ 0 (p=0.10) → refus", not r.ok and r.f_kelly <= 0)

# R2 : p élevé → Kelly plafonné à 20 % du capital
r = size_position(capital=10000, p_win=0.95, tp_r=3.5, sl_distance_pct=0.012)
check("R2 Kelly plafonné à 20 % (2000$)", r.ok and abs(r.size_usd - 10000 * MAX_EXPOSURE_PER_TRADE) < 1e-6,
      f"size={r.size_usd:.0f}$")

# R3 : taille < 15 $ → refus (petit capital + p modéré)
r = size_position(capital=60, p_win=0.30, tp_r=3.5, sl_distance_pct=0.012)
check("R3 taille < 15$ → refus minimum Binance", not r.ok and "Binance" in r.motif_refus)

# R4 : circuit breaker journalier
r = size_position(capital=10000, p_win=0.60, tp_r=3.5, sl_distance_pct=0.012,
                  daily_pnl=-(DAILY_LOSS_CIRCUIT + 0.01) * 10000)
check("R4 pertes jour > 5 % → arrêt", not r.ok and "circuit breaker" in r.motif_refus)

# R5 : exposition totale plafonnée à 40 %
r = size_position(capital=10000, p_win=0.95, tp_r=3.5, sl_distance_pct=0.012,
                  exposure_current=3900)   # 39 % déjà exposé
check("R5 exposition 39%+20% → réduite au reliquat (100$)",
      r.ok and abs(r.size_usd - (0.40 * 10000 - 3900)) < 1e-6, f"size={r.size_usd:.0f}$")
r = size_position(capital=10000, p_win=0.95, tp_r=3.5, sl_distance_pct=0.012,
                  exposure_current=4100)   # 41 % déjà exposé
check("R5b exposition 41 % → refus (au plafond)", not r.ok)

# ─── Verdict ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
if failures:
    print(f"❌ {len(failures)} test(s) en échec : {failures}")
    sys.exit(1)
print(f"✅ TOUS LES TESTS POLICY PASSENT (géométrie + gate + risk).")
