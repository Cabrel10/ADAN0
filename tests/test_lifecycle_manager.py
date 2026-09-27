#!/usr/bin/env python3
"""
test_lifecycle_manager.py — Tests de la Vigie Active (ADAN-System-One, étape 10)
=================================================================================

Couvre la logique de gestion de position à chaque clôture 5m :

  L1. TP atteint (SHORT et LONG) → CLOSE au prix cible
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


def make_position(direction="SHORT", entry=100.0, sl_pct=0.012, tp_r=3.5):
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

# ─── L1 : TP atteint ──────────────────────────────────────────────────────────
print("\nL1 — Take-profit atteint")
pos = make_position("SHORT")  # entry 100, TP 95.80
m = LifecycleManager(pos)
a = m.on_bar(o=99.0, h=99.5, l=95.70, c=95.90)  # low 95.70 < TP 95.80
check("L1a SHORT : low < TP → CLOSE au TP", a.action == "CLOSE"
      and "take-profit" in a.reason and abs(a.exit_price - 95.80) < 1e-9,
      f"exit={a.exit_price}, r={a.r_courant:+.2f}R")
pos = make_position("LONG")   # entry 100, TP 104.20
m = LifecycleManager(pos)
a = m.on_bar(o=101.0, h=104.30, l=100.5, c=104.0)  # high 104.30 > TP 104.20
check("L1b LONG : high > TP → CLOSE au TP", a.action == "CLOSE"
      and abs(a.exit_price - 104.20) < 1e-9)

# ─── L2 : SL touché ──────────────────────────────────────────────────────────
print("\nL2 — Stop-loss touché")
pos = make_position("SHORT")  # SL 101.20
m = LifecycleManager(pos)
a = m.on_bar(o=100.5, h=101.30, l=100.2, c=101.1)  # high 101.30 ≥ SL 101.20
check("L2 SHORT : high ≥ SL → CLOSE −1R", a.action == "CLOSE"
      and "stop-loss" in a.reason and abs(a.r_courant + 1.0) < 1e-9)

# ─── L3 : Break-even à +1.5R ─────────────────────────────────────────────────
print("\nL3 — Break-even (MFE ≥ +1.5R)")
pos = make_position("SHORT")  # entry 100, SL 101.20, risk 1.20
m = LifecycleManager(pos)
# barre qui descend à 98.20 → MFE = (100−98.20)/1.20 = +1.5R pile
a = m.on_bar(o=99.5, h=99.8, l=98.20, c=98.60)
check("L3 MFE=+1.5R → MOVE_SL break-even", a.action == "MOVE_SL"
      and "break-even" in a.reason, f"new_stop={a.new_stop}")
# nouveau SL doit être entry − frais (SHORT) = 100 − 0.08 = 99.92
check("L3 SL break-even = entry − frais (SHORT)",
      abs(pos.stop_loss - (100.0 - 100.0 * 0.0008)) < 1e-9,
      f"SL={pos.stop_loss:.4f}")
# le trade est désormais garanti sans perte : SL < entry pour un SHORT
check("L3 trade garanti sans perte (SL < entry)", pos.stop_loss < pos.entry)

# ─── L4 : Trailing à +2.0R ───────────────────────────────────────────────────
print("\nL4 — Trailing ATR (MFE ≥ +2.0R)")
pos = make_position("SHORT")  # entry 100, atr_1h_pct=0.012 → trail_dist=1.20
m = LifecycleManager(pos)
# d'abord atteindre +1.5R (break-even), puis +2.0R
m.on_bar(o=99.5, h=99.8, l=98.20, c=98.60)          # MFE +1.5R → BE
a = m.on_bar(o=98.6, h=98.7, l=97.60, c=97.80)      # MFE = (100−97.60)/1.2 = +2.0R
check("L4 MFE=+2.0R → trailing MOVE_SL", a.action in ("MOVE_SL", "HOLD")
      and pos.trailing_done, f"action={a.action}, SL={pos.stop_loss:.3f}")
# trailing SL = meilleur prix (97.60) + trail_dist (1.20) = 98.80, < BE SL
best_price = pos.entry - pos.mfe_r * m.risk
expected_trail = best_price + 1.20
check("L4 SL trailing suit le prix à 1×ATR", abs(pos.stop_loss - expected_trail) < 1e-6,
      f"SL={pos.stop_loss:.3f} attendu≈{expected_trail:.3f}")

# ─── L5 : Invalidation de thèse (sortie anticipée < 1R) ──────────────────────
print("\nL5 — Invalidation de thèse (coupe anticipée)")
pos = make_position("SHORT")  # entry 100, SL 101.20
m = LifecycleManager(pos)
# 6 barres qui montent doucement contre le SHORT (sans toucher le SL 101.20)
for k in range(6):
    base = 100.0 + 0.10 * (k + 1)  # 100.1 … 100.6
    a = m.on_bar(o=base - 0.05, h=base + 0.06, l=base - 0.08, c=base)
# à la 6e barre : close 100.6 > plus haut des 3 dernières (≈100.46) et R<0
check("L5 invalidation après 6 barres → CLOSE anticipée",
      a.action == "CLOSE" and "invalidation" in a.reason.lower(),
      f"r={a.r_courant:+.2f}R")
check("L5 perte contenue (< 1R, ≈ −0.5R)", -1.0 < a.r_courant < 0,
      f"perte {a.r_courant:+.2f}R au lieu de −1R")

# ─── L6 : Anomalie critique → sortie immédiate ───────────────────────────────
print("\nL6 — Anomalie critique (JEV)")
pos = make_position("SHORT")
m = LifecycleManager(pos)
a = m.on_bar(o=99.8, h=100.0, l=99.4, c=99.6, judgment=make_judgment(0.85))
check("L6 P(anomalie)=0.85 → CLOSE immédiate au marché", a.action == "CLOSE"
      and "anomalie" in a.reason and abs(a.exit_price - 99.6) < 1e-9)

# ─── L7 : Time-stop ──────────────────────────────────────────────────────────
print("\nL7 — Time-stop (288 barres)")
pos = make_position("SHORT")
m = LifecycleManager(pos)
a = None
for k in range(MAX_HOLD_BARS):
    a = m.on_bar(o=99.9, h=100.05, l=99.80, c=99.95)  # marché plat, rien ne se passe
    if a.action == "CLOSE":
        break
check("L7 time-stop après 288 barres → CLOSE", a.action == "CLOSE"
      and "time-stop" in a.reason)

# ─── L8 : HOLD nominal ───────────────────────────────────────────────────────
print("\nL8 — HOLD nominal (thèse intacte)")
pos = make_position("SHORT")
m = LifecycleManager(pos)
a = m.on_bar(o=99.9, h=100.1, l=99.6, c=99.7)  # léger profit, rien déclenché
check("L8 première barre en léger profit → HOLD", a.action == "HOLD"
      and a.r_courant > 0, f"r={a.r_courant:+.2f}R")

# ─── L9 : suivi MFE/MAE ──────────────────────────────────────────────────────
print("\nL9 — Suivi MFE/MAE barre par barre")
pos = make_position("SHORT")
m = LifecycleManager(pos)
m.on_bar(o=100, h=100.4, l=99.2, c=99.5)   # MFE: (100−99.2)/1.2=+0.67R ; MAE: (100−100.4)/1.2=−0.33R
m.on_bar(o=99.5, h=99.9, l=98.8, c=99.0)   # MFE: (100−98.8)/1.2=+1.00R
check("L9 MFE suivi (+1.00R max)", abs(pos.mfe_r - 1.0) < 1e-6, f"MFE={pos.mfe_r:+.2f}R")
check("L9 MAE suivi (−0.33R min)", abs(pos.mae_r - (-1 / 3)) < 0.01, f"MAE={pos.mae_r:+.2f}R")

# ─── Verdict ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
if failures:
    print(f"❌ {len(failures)} test(s) en échec : {failures}")
    sys.exit(1)
print("✅ TOUS LES TESTS LIFECYCLE PASSENT (vigie active opérationnelle).")
print("   TP/SL · break-even +1.5R · trailing +2.0R · invalidation anticipée")
print("   anomalie critique · time-stop · suivi MFE/MAE")
