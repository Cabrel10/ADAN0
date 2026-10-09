#!/usr/bin/env python3
"""
test_lifecycle_manager.py — Tests de la Vigie Active (ADAN-System-One, étape 10)
=================================================================================

Couvre la logique de gestion de position à chaque clôture 5m :

  L0. SPOT strict : position SHORT refusée
  L1. TP atteint (LONG) → CLOSE (SELL_EXIT) au prix cible
  L2. SL touché → CLOSE au prix stop
  L3. Break-even : MFE ≥ +1.5R → SL remonté à entry ± frais (MOVE_SL)
  L4. Trailing : MFE ≥ +2.0R → SL suit le prix à 1.0 × ATR
  L5. Invalidation de thèse : après ≥ 6 barres, clôture contre la position
      au-delà de l'extrême des 3 dernières barres avec R négatif → CLOSE
      anticipée (perte < 1R, ex. −0.3R)
  L6. Anomalie critique (JEV P>0.70) → CLOSE immédiate au marché
  L7. Time-stop après 288 barres → CLOSE
  L8. HOLD nominal : thèse intacte, aucune règle déclenchée
  L9. R/MFE/MAE suivis correctement barre par barre

Exécution :
  /home/ubuntu/webapp/MORNINGSTAR/miniconda3/envs/trading_env/bin/python3
      tests/test_lifecycle_manager.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adan_trading_bot.position.lifecycle_manager import (
    LifecycleManager, OpenPosition,
    BREAKEVEN_TRIGGER_R, TRAILING_TRIGGER_R, MAX_HOLD_BARS,
)
from adan_trading_bot.models.system_one_core import Judgment

PASS, FAIL = "✅", "❌"
failures = []


def check(name, cond, detail=""):
    print(f"  {PASS if cond else FAIL} {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def make_position(direction="LONG", entry=100.0, sl_pct=0.012, tp_r=3.5):
    """Position SHORT à 100, SL 1.2% (101.20), TP 3.5R (95.80) par défaut."""
    risk = entry * sl_pct
    if direction == "SHORT":
        return OpenPosition(direction="SHORT", entry=entry,
                            stop_loss=entry + risk, take_profit=entry - tp_r * risk,
                            size_usd=1000, entry_index=0, fees_rt=0.0008,
                            atr_1h_pct=0.012)
    return OpenPosition(direction="LONG", entry=entry,
                        stop_loss=entry - risk, take_profit=entry + tp_r * risk,
                        size_usd=1000, entry_index=0, fees_rt=0.0008,
                        atr_1h_pct=0.012)


def make_judgment(p_anomaly=0.1):
    return Judgment(
        noul={"anomalie_donnees": p_anomaly},
        choice={}, score={}, calibrated=True,
    )


print("=" * 72)
print("TEST lifecycle_manager — vigie active de position (étape 10)")
print("=" * 72)

# SPOT STRICT : seules des positions LONG peuvent exister ; la sortie est SELL_EXIT.
# Les anciens cas SHORT sont remplacés par leurs miroirs LONG exacts (prix reflétés
# autour de 100), et une position SHORT doit être refusée à la construction.
print("\nL0 — Contrat SPOT : position SHORT refusée")
from adan_trading_bot.policy.market_contract import MarketContractError, load_market_contract
try:
    LifecycleManager(OpenPosition(direction="SHORT", entry=100.0, stop_loss=101.2, take_profit=95.8,
                                  size_usd=16.4, entry_index=0, fees_rt=0.005, atr_1h_pct=0.012))
    check("L0 SHORT refusé en spot", False)
except MarketContractError:
    check("L0 SHORT refusé en spot", True)

print("\nL1 — Take-profit atteint")
pos = make_position("LONG")   # entry 100, TP 104.20
m = LifecycleManager(pos)
a = m.on_bar(o=101.0, h=104.30, l=100.5, c=104.0)
check("L1 LONG : high > TP → CLOSE (SELL_EXIT) au TP", a.action == "CLOSE"
      and "take-profit" in a.reason and abs(a.exit_price - 104.20) < 1e-9)

print("\nL2 — Stop-loss touché")
pos = make_position("LONG")   # SL 98.80
m = LifecycleManager(pos)
a = m.on_bar(o=99.5, h=99.8, l=98.70, c=98.9)
check("L2 LONG : low ≤ SL → CLOSE −1R", a.action == "CLOSE"
      and "stop-loss" in a.reason and abs(a.r_courant + 1.0) < 1e-9)

print("\nL3 — Break-even (MFE ≥ +1.5R)")
pos = make_position("LONG")   # entry 100, SL 98.80, risk 1.20
m = LifecycleManager(pos)
a = m.on_bar(o=100.5, h=101.80, l=100.2, c=101.40)   # MFE = 1.80/1.20 = +1.5R
check("L3 MFE=+1.5R → MOVE_SL break-even", a.action == "MOVE_SL"
      and "break-even" in a.reason, f"new_stop={a.new_stop}")
check("L3 SL break-even = entry + frais (LONG)",
      abs(pos.stop_loss - (100.0 + 100.0 * load_market_contract().cost_rt)) < 1e-9, f"SL={pos.stop_loss:.4f}")
check("L3 SL au-dessus de l'entry (hors gaps/slippage)", pos.stop_loss > pos.entry)

print("\nL4 — Trailing ATR (MFE ≥ +2.0R)")
pos = make_position("LONG")
m = LifecycleManager(pos)
m.on_bar(o=100.5, h=101.80, l=100.2, c=101.40)
a = m.on_bar(o=101.4, h=102.40, l=101.3, c=102.20)   # MFE = 2.40/1.2 = +2.0R
check("L4 MFE=+2.0R → trailing MOVE_SL", a.action in ("MOVE_SL", "HOLD")
      and pos.trailing_done, f"action={a.action}, SL={pos.stop_loss:.3f}")
best_price = pos.entry + pos.mfe_r * m.risk
check("L4 SL trailing suit le prix à 1×ATR", abs(pos.stop_loss - (best_price - 1.20)) < 1e-6,
      f"SL={pos.stop_loss:.3f}")

print("\nL5 — Invalidation de thèse (coupe anticipée)")
pos = make_position("LONG")
m = LifecycleManager(pos)
for k in range(6):
    base = 100.0 - 0.10 * (k + 1)
    a = m.on_bar(o=base + 0.05, h=base + 0.08, l=base - 0.06, c=base)
check("L5 invalidation après 6 barres → CLOSE anticipée",
      a.action == "CLOSE" and "invalidation" in a.reason.lower(), f"r={a.r_courant:+.2f}R")
check("L5 perte contenue (< 1R)", -1.0 < a.r_courant < 0, f"perte {a.r_courant:+.2f}R")

print("\nL6 — Anomalie critique (JEV)")
pos = make_position("LONG")
m = LifecycleManager(pos)
a = m.on_bar(o=100.2, h=100.6, l=100.0, c=100.4, judgment=make_judgment(0.85))
check("L6 P(anomalie)=0.85 → CLOSE immédiate au marché", a.action == "CLOSE"
      and "anomalie" in a.reason and abs(a.exit_price - 100.4) < 1e-9)

print("\nL7 — Time-stop (288 barres)")
pos = make_position("LONG")
m = LifecycleManager(pos)
a = None
for k in range(MAX_HOLD_BARS):
    a = m.on_bar(o=100.1, h=100.2, l=99.95, c=100.05)
    if a.action == "CLOSE":
        break
check("L7 time-stop après 288 barres → CLOSE", a.action == "CLOSE" and "time-stop" in a.reason)

print("\nL8 — HOLD nominal (thèse intacte)")
pos = make_position("LONG")
m = LifecycleManager(pos)
a = m.on_bar(o=100.1, h=100.4, l=99.9, c=100.3)
check("L8 première barre en léger profit → HOLD", a.action == "HOLD" and a.r_courant > 0,
      f"r={a.r_courant:+.2f}R")

print("\nL9 — Suivi MFE/MAE barre par barre")
pos = make_position("LONG")
m = LifecycleManager(pos)
m.on_bar(o=100, h=100.8, l=99.6, c=100.5)
m.on_bar(o=100.5, h=101.2, l=100.1, c=101.0)
check("L9 MFE suivi (+1.00R max)", abs(pos.mfe_r - 1.0) < 1e-6, f"MFE={pos.mfe_r:+.2f}R")
check("L9 MAE suivi (−0.33R min)", abs(pos.mae_r - (-1 / 3)) < 0.01, f"MAE={pos.mae_r:+.2f}R")

print("\nL10 — Ambiguïté SL/TP et gap : convention de l'oracle")
pos = make_position("LONG")
a = LifecycleManager(pos).on_bar(o=100., h=105., l=98., c=101.)
check("L10 même barre TP+SL → SL_FIRST", a.action == "CLOSE" and "stop-loss" in a.reason)
pos = make_position("LONG")
a = LifecycleManager(pos).on_bar(o=97., h=98., l=96., c=97.)
check("L10 gap SL → pire open", a.action == "CLOSE" and a.exit_price == 97.)

# ─── Verdict ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
if failures:
    print(f"❌ {len(failures)} test(s) en échec : {failures}")
    sys.exit(1)
print("✅ TOUS LES TESTS LIFECYCLE PASSENT (vigie active opérationnelle).")
print("   TP/SL · break-even +1.5R · trailing +2.0R · invalidation anticipée")
print("   anomalie critique · time-stop · suivi MFE/MAE")
