#!/usr/bin/env python3
"""
Etude de MATURITE contrefactuelle (SANS entrainement).

Pour chaque entree ADAN (prix d'entree connu, position LONG), on retrouve la
barre 5m correspondante dans l'OHLCV et on mesure, apres l'entree, a des horizons
h = 5/10/20/40/80 barres :
    MFE(h) = excursion favorable max   = max(high[t+1..t+h])/entry - 1   (long)
    MAE(h) = excursion defavorable max = min(low[t+1..t+h])/entry - 1
    R(h)   = close[t+h]/entry - 1      (rendement si on sortait a h)

But : repondre a "combien des trades tues a MaxDuration=20 auraient ete
profitables avec 40 ou 80 barres ?" et classer chaque entree :
    ENTRY_BAD   : MFE(80) < seuil_gain            (jamais assez profitable)
    ENTRY_LATE  : devient bon (MFE>=TP) seulement APRES 20 barres
    ENTRY_GOOD  : atteint MFE>=TP DANS les 20 barres
    ENTRY_GREAT : gros MFE (>=2*TP) rapidement (<=10 barres)

Usage: maturity_study.py <entries_json> <ohlcv_parquet> [tp_pct] [sl_pct]
"""
import json, sys, statistics as st
import pandas as pd

entries_path = sys.argv[1]
parq = sys.argv[2]
TP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.04   # 4% default TP
SL = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02   # 2% default SL
HORIZONS = [5, 10, 20, 40, 80]

entries = json.load(open(entries_path))
df = pd.read_parquet(parq)
df = df.reset_index()
high = df["high"].values
low = df["low"].values
close = df["close"].values
openp = df["open"].values
N = len(df)

# map entry price -> nearest bar index (by open, then close). tolerance small.
def find_idx(price):
    # match on open first (env enters at open of a bar)
    best_i, best_d = -1, 1e18
    for i in range(N):
        d = abs(openp[i] - price)
        if d < best_d:
            best_d, best_i = d, i
    return best_i, best_d

# Build a price->index lookup once (open prices are fairly unique)
open_lookup = {}
for i in range(N):
    open_lookup.setdefault(round(openp[i], 2), i)

def idx_for(price):
    r = round(price, 2)
    if r in open_lookup:
        return open_lookup[r], 0.0
    return find_idx(price)

rows = []
unmatched = 0
for e in entries:
    i, dd = idx_for(e["entry"])
    if i < 0 or dd > max(1.0, e["entry"] * 5e-5):  # >5bps mismatch => skip
        unmatched += 1
        continue
    rec = {"entry": e["entry"], "idx": i, "pnl": e["pnl"], "reason": e["reason"],
           "dur_bars": round(e["dur"] / 300)}
    for h in HORIZONS:
        j = min(i + h, N - 1)
        seg_hi = high[i + 1:j + 1]
        seg_lo = low[i + 1:j + 1]
        if len(seg_hi) == 0:
            rec[f"mfe{h}"] = 0.0; rec[f"mae{h}"] = 0.0; rec[f"r{h}"] = 0.0
            continue
        rec[f"mfe{h}"] = float(seg_hi.max() / e["entry"] - 1.0)      # long favorable = up
        rec[f"mae{h}"] = float(seg_lo.min() / e["entry"] - 1.0)      # long adverse = down
        rec[f"r{h}"] = float(close[j] / e["entry"] - 1.0)
    rows.append(rec)

n = len(rows)
print("=" * 72)
print(f"ETUDE DE MATURITE — {n} entrees appariees  (unmatched={unmatched})")
print(f"TP={TP*100:.1f}%  SL={SL*100:.1f}%  (long)")
print("=" * 72)

def pctl(p, xs):
    xs = sorted(xs); return xs[int(p / 100 * (len(xs) - 1))] if xs else 0.0

# --- MFE / MAE / R par horizon ---
print("\n[COURBE DE MATURITE — mediane sur toutes les entrees]")
print(f"{'h':>4} | {'MFE med':>9} {'MFE p75':>9} | {'MAE med':>9} {'MAE p25':>9} | {'R(h) med':>9}")
for h in HORIZONS:
    mfe = [r[f"mfe{h}"] for r in rows]
    mae = [r[f"mae{h}"] for r in rows]
    rr = [r[f"r{h}"] for r in rows]
    print(f"{h:>4} | {pctl(50,mfe)*100:>8.2f}% {pctl(75,mfe)*100:>8.2f}% | "
          f"{pctl(50,mae)*100:>8.2f}% {pctl(25,mae)*100:>8.2f}% | {pctl(50,rr)*100:>8.2f}%")

# --- % d'entrees dont MFE >= TP a chaque horizon ---
print("\n[% entrees atteignant MFE >= TP  (le TP EST atteignable si on tient)]")
for h in HORIZONS:
    hit = sum(1 for r in rows if r[f"mfe{h}"] >= TP)
    print(f"  h={h:>3} barres : {100*hit/n:5.1f}%   ({hit}/{n})")

# --- % entrees touchant SL (MAE <= -SL) ---
print("\n[% entrees touchant SL (MAE <= -SL)]")
for h in HORIZONS:
    hit = sum(1 for r in rows if r[f"mae{h}"] <= -SL)
    print(f"  h={h:>3} barres : {100*hit/n:5.1f}%   ({hit}/{n})")

# --- LA question : trades tues a MaxDuration, sauves par 40/80 barres ? ---
maxdur = [r for r in rows if r["reason"] == "MaxDuration"]
print(f"\n[TRADES FERMES PAR MaxDuration : {len(maxdur)} / {n} = {100*len(maxdur)/n:.1f}%]")
if maxdur:
    for h in HORIZONS:
        would = sum(1 for r in maxdur if r[f"mfe{h}"] >= TP)
        print(f"  auraient touche TP en {h:>3} barres : {100*would/len(maxdur):5.1f}%  ({would}/{len(maxdur)})")
    # edge destroyed: R20 vs best achievable by 40/80
    r20 = [r["r20"] for r in maxdur]
    best80 = [max(r["mfe20"], r["mfe40"], r["mfe80"]) for r in maxdur]
    print(f"  R median a 20 barres (sortie forcee)   : {pctl(50,r20)*100:+.2f}%")
    print(f"  MFE median atteint d'ici 80 barres     : {pctl(50,best80)*100:+.2f}%")

# --- Classification des entrees ---
def classify(r):
    mfe10, mfe20, mfe80 = r["mfe10"], r["mfe20"], r["mfe80"]
    if mfe10 >= 2 * TP:
        return "ENTRY_GREAT"
    if mfe20 >= TP:
        return "ENTRY_GOOD"
    if mfe80 >= TP:   # only becomes good after 20
        return "ENTRY_LATE"
    return "ENTRY_BAD"

cls = {}
for r in rows:
    c = classify(r); cls[c] = cls.get(c, 0) + 1
print("\n[CLASSIFICATION DES ENTREES]")
for c in ("ENTRY_GREAT", "ENTRY_GOOD", "ENTRY_LATE", "ENTRY_BAD"):
    v = cls.get(c, 0)
    print(f"  {c:12s}: {v:5d}  ({100*v/n:5.1f}%)")

# --- VERDICT: entree vs gestion ---
good_or_better = cls.get("ENTRY_GREAT",0)+cls.get("ENTRY_GOOD",0)+cls.get("ENTRY_LATE",0)
print("\n[VERDICT ENTREE vs GESTION]")
print(f"  entrees exploitables (MFE>=TP a un horizon <=80) : {100*good_or_better/n:.1f}%")
print(f"  dont LATE (edge existe mais >20 barres)          : {100*cls.get('ENTRY_LATE',0)/n:.1f}%")
print(f"  entrees vraiment mauvaises (BAD)                 : {100*cls.get('ENTRY_BAD',0)/n:.1f}%")
print("  => si LATE eleve : l'horizon 20 DETRUIT de l'edge (probleme GESTION).")
print("  => si BAD eleve  : l'entree est le probleme (probleme DETECTION).")

json.dump(rows, open("/tmp/maturity_rows.json", "w"))
print("\nsaved /tmp/maturity_rows.json")
