#!/usr/bin/env python3
"""Audit causal v37 — POLICY RAW -> ROUTING -> GATES -> EXECUTION.

Sources (run d'entraînement, pas de trace JSONL active) :
  [TARGET_WEIGHT] : echantillon 1/50 du brut policy (Raw=a0, Thr, Action post-routing)
  [ACTION_DIFF]   : Requested (post-routing) vs Executed + compteurs cumulatifs

Decomposition par echantillon (Raw, Thr, Action, Requested, Executed) :
  raw_buy  = Raw >  +Thr
  raw_sell = Raw <  -Thr
  raw_wait = |Raw| <= Thr            (policy a choisi la neutralite)
  Action=BUY/SELL                    (routing a valide l'intention)
  Action=HOLD avec intention forte  (routing a transforme en HOLD)
"""
import re
import sys
import ast
from collections import Counter

LOG = sys.argv[1] if len(sys.argv) > 1 else "logs/v37_500k/btc_500k.log"

TW_RE = re.compile(
    r"\[TARGET_WEIGHT\] Step (\d+) \| \S+ \| Action=(\w+) \| "
    r"Raw=(-?[\d.]+) \| Thr=([\d.]+) \| SizeRaw=(-?[\d.]+) \| Capital=([\d.]+)"
)
AD_RE = re.compile(
    r"\[ACTION_DIFF\] Step (\d+) \| Requested=(\w+) Executed=(\w+) \| "
    r"budget=([\d.]+)/([\d.]+) \| inv_penalty=(-?[\d.]+) \| "
    r"rejections=(\{[^}]*\}) \| pipeline=(\{.*\})\s*$"
)
CLOSE_RE = re.compile(r"\[POSITION FERM[ÉE]E\].*?PnL: \$(-?[\d.]+).*?Raison: (\S+)")

# --- echantillons TARGET_WEIGHT + jointure ACTION_DIFF (meme step) ---
tw = {}   # step -> (raw, thr, routed_action)
ad = {}   # step -> (requested, executed, budget)
pipeline_hist = []  # (step, pipeline_dict)
rejections_last = None

with open(LOG, encoding="utf-8", errors="replace") as f:
    for line in f:
        m = TW_RE.search(line)
        if m:
            tw[int(m.group(1))] = (float(m.group(3)), float(m.group(4)), m.group(2))
            continue
        m = AD_RE.search(line)
        if m:
            step = int(m.group(1))
            ad[step] = (m.group(2), m.group(3), float(m.group(4)))
            try:
                rejections_last = ast.literal_eval(m.group(6))
                pipeline_hist.append((step, ast.literal_eval(m.group(7))))
            except Exception:
                pass

# --- decomposition causale sur les steps joints ---
CATS = [
    "raw_wait",                       # policy neutre (|a0|<=thr)
    "raw_buy_routed_BUY",             # intention BUY validee par routing
    "raw_sell_routed_SELL",           # intention SELL validee par routing
    "raw_buy_to_HOLD_routing",        # routing a neutralise un BUY (buy_while_long/slot)
    "raw_sell_to_HOLD_routing",       # routing a neutralise un SELL (sell_while_flat)
    "raw_intent_unclear",             # HOLD route avec intention faible inattendue
]
joint = Counter()
gate_transforms = Counter()   # Requested != Executed
executed_after_gates = Counter()

for step, (raw, thr, routed) in sorted(tw.items()):
    raw_buy, raw_sell = raw > thr, raw < -thr
    if not raw_buy and not raw_sell:
        cat = "raw_wait"
    elif raw_buy and routed == "BUY":
        cat = "raw_buy_routed_BUY"
    elif raw_sell and routed == "SELL":
        cat = "raw_sell_routed_SELL"
    elif raw_buy and routed == "HOLD":
        cat = "raw_buy_to_HOLD_routing"
    elif raw_sell and routed == "HOLD":
        cat = "raw_sell_to_HOLD_routing"
    else:
        cat = "raw_intent_unclear"
    joint[cat] += 1

    if step in ad:
        req, exe, _budget = ad[step]
        if req != exe:
            gate_transforms[f"{req}->{exe}"] += 1
        else:
            executed_after_gates[exe] += 1

# --- deltas de compteurs pipeline (fenetres de 50 steps) ---
def delta(key):
    if len(pipeline_hist) < 2:
        return 0
    return pipeline_hist[-1][1].get(key, 0) - pipeline_hist[0][1].get(key, 0)

total = sum(joint.values())
steps_covered = (pipeline_hist[-1][0] - pipeline_hist[0][0]) if len(pipeline_hist) > 1 else 0

print("=" * 74)
print(f"AUDIT CAUSAL v37 — {LOG}")
print(f"echantillons TARGET_WEIGHT joints : {total}  | fenetre steps : {steps_covered}")
print("=" * 74)
print("\n[1] DISTRIBUTION BRUTE DE LA POLICY (echantillon 1/50)")
for c in CATS:
    n = joint.get(c, 0)
    pct = 100.0 * n / total if total else 0.0
    print(f"  {c:28s} {n:6d}  {pct:5.1f}%")

print("\n[2] TRANSFORMATIONS PAR LES GATES (Requested != Executed, echantillon)")
for k, v in sorted(gate_transforms.items(), key=lambda kv: -kv[1]):
    print(f"  {k:16s} {v:6d}")
if not gate_transforms:
    print("  (aucune)")

print("\n[3] EXECUTIONS CONFORMES (Requested == Executed, echantillon)")
for k, v in sorted(executed_after_gates.items(), key=lambda kv: -kv[1]):
    print(f"  {k:16s} {v:6d}")

print("\n[4] COMPTEURS PIPELINE (cumul depuis le debut + deltas)")
if pipeline_hist:
    first_step, first = pipeline_hist[0]
    last_step, last = pipeline_hist[-1]
    print(f"  cumul au step {last_step} (policy={last.get('policy',0)}) :")
    for k, v in last.items():
        print(f"    {k:24s} {v}")
    print(f"\n  DELTAS sur la fenetre [{first_step}..{last_step}] :")
    dpol = max(1, delta("policy"))
    for k in ("deadband_reject", "routing_reject", "budget_insufficient",
              "close_gap_active", "daily_close_quota", "below_break_even",
              "hold_min_active", "portfolio_reject", "trade_executed"):
        d = delta(k)
        print(f"    {k:24s} {d:6d}  ({100.0*d/dpol:5.1f}% des decisions de la fenetre)")

print("\n[5] REJETS CUMULES (dernier point — sature si plus aucun rejet recent)")
if rejections_last:
    for k, v in rejections_last.items():
        if v:
            print(f"  {k:28s} {v}")

# --- raisons de fermeture + PnL ---
reasons = Counter()
pnl_by_reason = {}
with open(LOG, encoding="utf-8", errors="replace") as f:
    for line in f:
        m = CLOSE_RE.search(line)
        if m:
            r = m.group(2)
            reasons[r] += 1
            pnl_by_reason.setdefault(r, []).append(float(m.group(1)))
print("\n[6] FERMETURES DE POSITION (log complet)")
for r, n in reasons.most_common():
    pnls = pnl_by_reason[r]
    print(f"  {r:28s} n={n:5d}  PnL_total=${sum(pnls):+.2f}  PnL_moy=${sum(pnls)/len(pnls):+.4f}")

print("\n[7] VERDICT CAUSAL")
rw = joint.get("raw_wait", 0)
b_hold = joint.get("raw_buy_to_HOLD_routing", 0)
s_hold = joint.get("raw_sell_to_HOLD_routing", 0)
b_ok = joint.get("raw_buy_routed_BUY", 0)
s_ok = joint.get("raw_sell_routed_SELL", 0)
intent = b_ok + s_ok + b_hold + s_hold
print(f"  intentions fortes BUY/SELL : {intent} ({100*intent/max(1,total):.1f}% des decisions)")
print(f"  policy neutre (|a0|<=thr)  : {rw} ({100*rw/max(1,total):.1f}%)")
if intent > 0:
    blocked = b_hold + s_hold
    print(f"  intentions neutralisees par routing : {blocked}/{intent} "
          f"({100*blocked/intent:.1f}%)")
if intent / max(1, total) < 0.05:
    print("  => COLLAPSE HOLD CONFIRME cote POLICY (la policy ne demande presque rien)")
elif (b_hold + s_hold) > 0.5 * max(1, intent):
    print("  => TRANSFORMATION ENVIRONNEMENT DOMINANTE (routing neutralise les intentions)")
else:
    print("  => PIPELINE SAIN : les intentions de la policy atteignent l'execution")
