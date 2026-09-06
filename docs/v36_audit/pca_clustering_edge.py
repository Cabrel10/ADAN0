#!/usr/bin/env python3
"""
ACP (PCA) + clustering pour DECOUVRIR un edge d'entree conditionnel.  ZERO training.

Idee (validee avec l'utilisateur) : au lieu de juger les entrees d'ADAN (qui sont
~aleatoires), on cherche s'il EXISTE des types de situations de marche ou l'edge
forward est reel. Si oui -> la detection est reparable (un meilleur cerveau peut
apprendre a n'entrer que la). Si non -> le dataset/TF n'offre pas d'edge.

Protocole:
  1. features = indicateurs presents au moment t (AUCUN futur).
  2. standardisation -> PCA (variance expliquee) -> KMeans (k clusters).
  3. pour chaque cluster: forward MFE20/MAE20/R20 (long ET short), win-rate a TP
     calibre volatilite, et surtout EDGE vs moyenne globale.
  4. validation: un cluster n'est "reel" que si son outcome differe nettement du
     tirage global (pas juste une difference geometrique).

Usage: pca_clustering_edge.py <ohlcv_parquet> [k] [horizon] [tp_atr_mult]
"""
import sys
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

parq = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
H = int(sys.argv[3]) if len(sys.argv) > 3 else 20
TP_ATR = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0  # TP = 3xATR

df = pd.read_parquet(parq).reset_index()
N = len(df)
close = df["close"].values; high = df["high"].values; low = df["low"].values
atr = df["atr_pct"].values if "atr_pct" in df else np.full(N, 0.0015)

# feature columns = indicators only (exclude raw OHLCV + look-ahead-y stuff)
feat_cols = [c for c in df.columns if c not in
             ("timestamp", "open", "high", "low", "close", "volume")]
X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values

# forward outcomes (long) — valid only where i+H exists
valid = np.arange(0, N - H - 1)
def fwd(i):
    j = i + H
    mfe = high[i + 1:j + 1].max() / close[i] - 1.0
    mae = low[i + 1:j + 1].min() / close[i] - 1.0
    r = close[j] / close[i] - 1.0
    return mfe, mae, r

Xv = X[valid]
scaler = StandardScaler()
Xs = scaler.fit_transform(Xv)

pca = PCA(n_components=0.90, random_state=0)  # keep 90% variance
Xp = pca.fit_transform(Xs)
print("=" * 72)
print(f"ACP + CLUSTERING — {parq.split('/')[-2]}  n={len(valid)}  H={H} barres")
print("=" * 72)
print(f"features utilisees ({len(feat_cols)}): {feat_cols}")
print(f"PCA: {Xp.shape[1]} composantes pour 90% variance  "
      f"(explained: {np.cumsum(pca.explained_variance_ratio_)[:5].round(3).tolist()}...)")

km = KMeans(n_clusters=K, n_init=10, random_state=0)
lab = km.fit_predict(Xp)

# global baseline
mfes = np.array([fwd(i)[0] for i in valid])
maes = np.array([fwd(i)[1] for i in valid])
r20s = np.array([fwd(i)[2] for i in valid])
atrv = atr[valid]
g_mfe, g_r = np.median(mfes), np.median(r20s)
print(f"\nBASELINE globale: MFE{H} med={g_mfe*100:+.3f}%  R{H} med={g_r*100:+.3f}%  "
      f"ATR med={np.median(atrv)*100:.3f}%")
print(f"TP teste = {TP_ATR}xATR (median TP={TP_ATR*np.median(atrv)*100:.3f}%)")

print("\n" + "-" * 72)
print(f"{'clu':>3} {'n':>6} {'%':>5} | {'MFE med':>9} {'R med':>9} | "
      f"{'winTP-L':>8} {'winTP-S':>8} | {'edge_R':>8}")
print("-" * 72)
rows = []
for c in range(K):
    m = lab == c
    nc = int(m.sum())
    mfe_c = np.median(mfes[m]); r_c = np.median(r20s[m])
    # win at vol-calibrated TP, first-touch proxy vs SL=TP
    tp = TP_ATR * atrv[m]
    win_long = np.mean(mfes[m] >= tp) * 100          # long reaches TP up
    win_short = np.mean(-maes[m] >= tp) * 100         # short reaches TP down
    edge_r = (r_c - g_r) * 100
    rows.append((c, nc, mfe_c, r_c, win_long, win_short, edge_r))
    print(f"{c:>3} {nc:>6} {100*nc/len(valid):>4.1f}% | {mfe_c*100:>+8.3f}% {r_c*100:>+8.3f}% | "
          f"{win_long:>7.1f}% {win_short:>7.1f}% | {edge_r:>+7.3f}")

# highlight best/worst clusters by |edge|
rows_sorted = sorted(rows, key=lambda x: x[3])
print("\n[CLUSTERS LES PLUS DIRECTIONNELS (R median)]")
worst = rows_sorted[0]; best = rows_sorted[-1]
print(f"  plus BAISSIER  : cluster {worst[0]} R{H}med={worst[3]*100:+.3f}%  "
      f"(short winTP={worst[5]:.1f}%, n={worst[1]})")
print(f"  plus HAUSSIER  : cluster {best[0]}  R{H}med={best[3]*100:+.3f}%  "
      f"(long  winTP={best[4]:.1f}%, n={best[1]})")

# VERDICT: does ANY cluster show a real, sized edge?
max_absedge = max(abs(x[6]) for x in rows)
best_dir_win = max(max(x[4] for x in rows), max(x[5] for x in rows))
print("\n[VERDICT EDGE CONDITIONNEL]")
print(f"  |edge_R| max entre clusters : {max_absedge:.3f} pts  "
      f"(vs bruit ~ATR {np.median(atrv)*100:.3f}%)")
print(f"  meilleur win-rate TP dir.   : {best_dir_win:.1f}%")
if max_absedge > 0.5 * np.median(atrv) * 100 and best_dir_win > 55:
    print("  => EDGE CONDITIONNEL PLAUSIBLE : certains clusters battent le hasard.")
    print("     -> la DETECTION est reparable (entrainer un detecteur selectif).")
else:
    print("  => PAS d'edge conditionnel net sur ces features/TF.")
    print("     -> soit features insuffisantes, soit ce TF/univers n'offre pas d'edge.")
    print("     -> changer features/timeframe/actifs AVANT de retoucher ADAN.")
