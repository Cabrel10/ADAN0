#!/usr/bin/env python3
"""
Autopsie economique ADAN — trace la chaine capital sur donnees existantes.
Usage: economic_autopsy.py <rewards_jsonl> [label]

Repond aux questions:
  - equity: max jamais atteint, % steps >= 21, distribution
  - trades: close reason (AGENT_CLOSE / SL / TP), duration, PnL brut/net, commission
  - sizing: distribution size_pct action, notional reellement engage, exposition %
  - Q7: la taille de position est-elle apprise/variee ou clampee ?
  - premier point de rupture dans la chaine (5 etapes)
"""
import json, sys, statistics as st

f = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else "RUN"

n = 0
eqs = []
max_eq = -1e9
over21 = 0
# trades reconstructed via realized-PnL delta (a close event)
trades = []          # each: pnl, dur, reason, notional_at_open, size_pct_at_open, direction
open_notional = None
open_size_pct = None
open_dir = None
open_entry = None
prev_real = 0.0
size_pcts = []       # action size_pct EVERY step where an open occurred
exposures = []       # notional/equity at open
close_reasons = {}
durations = []
by_ep_start = {}     # episode -> first equity
by_ep_end = {}       # episode -> last equity
commissions = []

with open(f) as fh:
    for line in fh:
        try: d = json.loads(line)
        except: continue
        n += 1
        pf = d["portfolio"]; eq = pf["equity"]
        eqs.append(eq); max_eq = max(max_eq, eq)
        if eq >= 21.0: over21 += 1
        ep = d.get("episode", 0)
        if ep not in by_ep_start: by_ep_start[ep] = eq
        by_ep_end[ep] = eq
        pos = d.get("positions", {})
        od = pos.get("open_details") or []
        # capture an OPEN this step
        if od:
            o = od[0]
            sz = o.get("size", 0.0); entry = o.get("entry", 0.0)
            notional = abs(sz*entry)
            open_notional = notional; open_entry = entry
            open_size_pct = d["action"]["semantics"].get("size")
            open_dir = d["action"]["semantics"].get("direction")
            size_pcts.append(open_size_pct)
            if eq>0: exposures.append(100.0*notional/eq)
        real = d["pnl"].get("realized", 0.0)
        comm = d["pnl"].get("total_commission", 0.0)
        trig = d.get("triggers", {})
        reason = trig.get("reason", "") or ("SL" if trig.get("sl_triggered") else ("TP" if trig.get("tp_triggered") else ""))
        # a CLOSE: realized became non-zero this step
        if abs(real) > 1e-12 and abs(prev_real) < 1e-12:
            dur = trig.get("duration_seconds", 0.0)
            trades.append({
                "pnl": real, "dur": dur, "reason": reason or "UNKNOWN",
                "notional": open_notional, "size_pct": open_size_pct,
                "dir": open_dir, "comm": comm,
            })
            close_reasons[reason or "UNKNOWN"] = close_reasons.get(reason or "UNKNOWN",0)+1
            durations.append(dur)
            commissions.append(comm)
            open_notional = None; open_size_pct=None; open_dir=None
        prev_real = real

def pct(x, xs):
    xs=sorted(xs)
    if not xs: return 0
    i=int(x/100*(len(xs)-1)); return xs[i]

wins=[t for t in trades if t["pnl"]>0]
losses=[t for t in trades if t["pnl"]<0]
flat=[t for t in trades if t["pnl"]==0]
gross_win=sum(t["pnl"] for t in wins)
gross_loss=sum(t["pnl"] for t in losses)

print("="*68)
print(f"AUTOPSIE ECONOMIQUE — {label}")
print("="*68)
print(f"steps                     : {n:,}")
print(f"episodes                  : {len(by_ep_start)}")
print("-"*68)
print("[EQUITY]")
print(f"  equity initiale         : {eqs[0]:.4f}")
print(f"  equity MAX jamais       : {max_eq:.4f}   (gain max absolu = {max_eq-20.5:+.4f} $)")
print(f"  equity MIN              : {min(eqs):.4f}")
print(f"  equity moyenne          : {st.mean(eqs):.4f}")
print(f"  steps equity >= 21.0    : {over21}  ({100*over21/n:.4f}%)")
print(f"  franchit durablement 21 : {'OUI' if over21>0 else 'NON — JAMAIS'}")
print("-"*68)
print("[TRADES]  (reconstruits via delta realized-PnL)")
print(f"  nb trades fermes        : {len(trades)}")
print(f"  gagnants / perdants     : {len(wins)} / {len(losses)}  (flat={len(flat)})")
if trades:
    wr=100*len(wins)/len(trades)
    print(f"  win rate                : {wr:.1f}%")
    print(f"  PnL brut cumule         : {sum(t['pnl'] for t in trades):+.3f}")
    print(f"  gross win / gross loss  : {gross_win:+.3f} / {gross_loss:+.3f}")
    print(f"  profit factor           : {(gross_win/abs(gross_loss)) if gross_loss else float('inf'):.3f}")
    if wins:   print(f"  gain moyen (gagnant)    : {st.mean([t['pnl'] for t in wins]):+.4f}   max={max(t['pnl'] for t in wins):+.4f}")
    if losses: print(f"  perte moyenne (perdant) : {st.mean([t['pnl'] for t in losses]):+.4f}   min={min(t['pnl'] for t in losses):+.4f}")
    print(f"  expectancy / trade      : {st.mean([t['pnl'] for t in trades]):+.4f}")
    print(f"  commission moyenne/trade: {st.mean(commissions):.5f}")
print("-"*68)
print("[CLOSE REASON]")
tot=len(trades) or 1
for r,c in sorted(close_reasons.items(), key=lambda x:-x[1]):
    print(f"  {r:14s}: {c:5d}  ({100*c/tot:.1f}%)")
print("-"*68)
print("[DUREE DE DETENTION]  (secondes)")
if durations:
    print(f"  mediane                 : {pct(50,durations):.0f}s  ({pct(50,durations)/60:.1f} min)")
    print(f"  p10 / p90               : {pct(10,durations):.0f}s / {pct(90,durations):.0f}s")
    print(f"  moyenne                 : {st.mean(durations):.0f}s")
    # 5m bars: 300s = 1 bar
    print(f"  mediane en barres 5m    : {pct(50,durations)/300:.1f} barres")
print("-"*68)
print("[SIZING / EXPOSITION]  (Q2, Q5, Q7)")
if size_pcts:
    print(f"  nb ouvertures mesurees  : {len(size_pcts)}")
    print(f"  size_pct action  min/med/max : {min(size_pcts):+.3f} / {pct(50,size_pcts):+.3f} / {max(size_pcts):+.3f}")
    print(f"  size_pct  ecart-type    : {st.pstdev(size_pcts):.3f}   (Q7: ~0 => clampe/fige, >0 => varie)")
if exposures:
    print(f"  exposition notional/eq %  min/med/max : {min(exposures):.1f}% / {pct(50,exposures):.1f}% / {max(exposures):.1f}%")
    print(f"  exposition moyenne      : {st.mean(exposures):.1f}%")
print("="*68)
