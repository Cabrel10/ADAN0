#!/usr/bin/env python3
"""
Offline market audit (ZERO training) — vectorized.

Replays, for ONE asset's featured 5m parquet, the full market-intrinsic study:
  - ATR distribution
  - forward MFE(h)/MAE(h)/R(h) at h = 5/10/20/40/80 (vectorized)
  - TP reachability at fixed % thresholds (0.5/1/2%) AND vol-calibrated (k*ATR)
  - SL reachability
  - random-entry baseline (all bars = the random baseline itself)
  - PCA + KMeans clustering -> forward outcome per cluster -> conditional edge
  - optimal horizon H* (max median R over the profitable direction)

Outputs a JSON summary (for the BTC-vs-DOGE table) + prints a human report.

Usage:
  offline_audit.py <featured_5m_parquet> <label> [out_json] [k] [max_rows]

max_rows (optional): subsample the most-RECENT max_rows bars for the clustering
step only (MFE/MAE stats always use the full set). Default: use all.
"""
import sys, json
import numpy as np
import pandas as pd

parq = sys.argv[1]
LABEL = sys.argv[2] if len(sys.argv) > 2 else "ASSET"
OUT = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/audit_{LABEL}.json"
K = int(sys.argv[4]) if len(sys.argv) > 4 else 8
MAX_ROWS = int(sys.argv[5]) if len(sys.argv) > 5 else 0

HORIZONS = [5, 10, 20, 40, 80]
TP_FIXED = [0.005, 0.01, 0.02]      # 0.5% / 1% / 2%
TP_ATR_MULTS = [2.0, 3.0]           # vol-calibrated TP
SL_ATR_MULT = 2.0

df = pd.read_parquet(parq)
df = df.reset_index()
N = len(df)
high = df["high"].to_numpy(np.float64)
low = df["low"].to_numpy(np.float64)
close = df["close"].to_numpy(np.float64)
atr = df["atr_pct"].to_numpy(np.float64) if "atr_pct" in df else np.full(N, 0.0015)

print("=" * 74)
print(f"OFFLINE AUDIT — {LABEL}   n={N}   ({df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]})")
print("=" * 74)

atr_med = float(np.nanmedian(atr))
print(f"ATR median 5m       : {atr_med*100:.4f}%")
print(f"ATR p25/p75         : {np.nanpercentile(atr,25)*100:.4f}% / {np.nanpercentile(atr,75)*100:.4f}%")

# ---- Vectorized rolling forward MFE/MAE/R via sliding max/min ----
def rolling_fwd_max(arr, h):
    # max of arr[i+1 .. i+h] for each i; edges -> nan
    out = np.full(N, np.nan)
    for off in range(1, h + 1):
        shifted = np.empty(N); shifted[:] = np.nan
        shifted[:N - off] = arr[off:]
        if off == 1:
            out[:N - off] = shifted[:N - off]
        else:
            out = np.fmax(out, shifted)
    return out

def rolling_fwd_min(arr, h):
    out = np.full(N, np.nan)
    for off in range(1, h + 1):
        shifted = np.empty(N); shifted[:] = np.nan
        shifted[:N - off] = arr[off:]
        if off == 1:
            out[:N - off] = shifted[:N - off]
        else:
            out = np.fmin(out, shifted)
    return out

summary = {"label": LABEL, "n_bars": N,
           "range": [str(df['timestamp'].iloc[0]), str(df['timestamp'].iloc[-1])],
           "atr_median_pct": atr_med * 100,
           "horizons": {}, "reachability": {}, "clustering": {}}

print("\n[COURBE DE MATURITE — random baseline = TOUTES les barres]")
print(f"{'h':>4} | {'MFE med':>9} {'MFE p75':>9} | {'MAE med':>9} {'MAE p25':>9} | {'R med':>9} {'R mean':>9}")
mfe_by_h = {}
mae_by_h = {}
r_by_h = {}
for h in HORIZONS:
    fmax = rolling_fwd_max(high, h)
    fmin = rolling_fwd_min(low, h)
    fclose = np.roll(close, -h); fclose[N - h:] = np.nan
    mfe = fmax / close - 1.0
    mae = fmin / close - 1.0
    r = fclose / close - 1.0
    mfe_by_h[h] = mfe; mae_by_h[h] = mae; r_by_h[h] = r
    print(f"{h:>4} | {np.nanmedian(mfe)*100:>8.3f}% {np.nanpercentile(mfe,75)*100:>8.3f}% | "
          f"{np.nanmedian(mae)*100:>8.3f}% {np.nanpercentile(mae,25)*100:>8.3f}% | "
          f"{np.nanmedian(r)*100:>8.4f}% {np.nanmean(r)*100:>8.4f}%")
    summary["horizons"][h] = {
        "mfe_med_pct": float(np.nanmedian(mfe) * 100),
        "mfe_p75_pct": float(np.nanpercentile(mfe, 75) * 100),
        "mae_med_pct": float(np.nanmedian(mae) * 100),
        "mae_p25_pct": float(np.nanpercentile(mae, 25) * 100),
        "r_med_pct": float(np.nanmedian(r) * 100),
        "r_mean_pct": float(np.nanmean(r) * 100),
    }

# ---- TP / SL reachability (long) at each horizon ----
print("\n[REACHABILITY — % barres dont MFE>=TP  (long)]")
hdr = "  h |" + "".join(f" TP{int(t*1000)/10:g}% " for t in TP_FIXED) + \
      "".join(f" TP{m:g}xATR " for m in TP_ATR_MULTS) + f"| SL{SL_ATR_MULT:g}xATR"
print(hdr)
for h in HORIZONS:
    mfe = mfe_by_h[h]; mae = mae_by_h[h]
    valid = ~np.isnan(mfe)
    row = f"{h:>3} |"
    rk = {}
    for t in TP_FIXED:
        p = np.mean(mfe[valid] >= t) * 100
        rk[f"tp_{t}"] = float(p)
        row += f"  {p:5.1f}% "
    for m in TP_ATR_MULTS:
        tp = m * atr[valid]
        p = np.mean(mfe[valid] >= tp) * 100
        rk[f"tp_{m}xatr"] = float(p)
        row += f"  {p:5.1f}%  "
    slp = np.mean(mae[valid] <= -SL_ATR_MULT * atr[valid]) * 100
    rk["sl_hit"] = float(slp)
    row += f"|  {slp:5.1f}%"
    print(row)
    summary["reachability"][h] = rk

# ---- Optimal horizon H* (median R, both directions) ----
h_star_long = max(HORIZONS, key=lambda h: np.nanmedian(r_by_h[h]))
h_star_short = max(HORIZONS, key=lambda h: -np.nanmedian(r_by_h[h]))
summary["h_star_long"] = h_star_long
summary["h_star_short"] = h_star_short
print(f"\n[H* — meilleur horizon rendement median]  long H*={h_star_long}  short H*={h_star_short}")

# ---- PCA + KMeans clustering on indicators (conditional edge) ----
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

feat_cols = [c for c in df.columns if c not in
             ("timestamp", "index", "open", "high", "low", "close", "volume")]
Hc = 20  # clustering horizon
mfe20 = mfe_by_h[20]; mae20 = mae_by_h[20]; r20 = r_by_h[20]
valid = ~np.isnan(r20)

idx_all = np.where(valid)[0]
if MAX_ROWS and len(idx_all) > MAX_ROWS:
    idx_use = idx_all[-MAX_ROWS:]   # most recent window for clustering
else:
    idx_use = idx_all

X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()[idx_use]
Xs = StandardScaler().fit_transform(X)
pca = PCA(n_components=0.90, random_state=0)
Xp = pca.fit_transform(Xs)
km = KMeans(n_clusters=K, n_init=10, random_state=0)
lab = km.fit_predict(Xp)

r20u = r20[idx_use]; mfe20u = mfe20[idx_use]; mae20u = mae20[idx_use]; atru = atr[idx_use]
rsi = df["rsi_14"].to_numpy()[idx_use] if "rsi_14" in df else np.full(len(idx_use), np.nan)
adx = df["adx_14"].to_numpy()[idx_use] if "adx_14" in df else np.full(len(idx_use), np.nan)
g_r = float(np.median(r20u))

print(f"\n[ACP+CLUSTERING H={Hc}] PCA {Xp.shape[1]} comps (90% var) | n_clustered={len(idx_use)} | baseline R20 med={g_r*100:+.4f}%")
print(f"{'clu':>3} {'n':>7} {'%':>5} | {'R20 med':>9} {'edge':>8} | {'winTP3xL':>9} {'winTP3xS':>9} | {'RSI':>5} {'ADX':>5}")
clusters = []
for c in range(K):
    m = lab == c
    nc = int(m.sum())
    r_c = float(np.median(r20u[m]))
    tp = 3.0 * atru[m]
    win_l = float(np.mean(mfe20u[m] >= tp) * 100)
    win_s = float(np.mean(-mae20u[m] >= tp) * 100)
    edge = (r_c - g_r) * 100
    rsi_c = float(np.nanmedian(rsi[m])); adx_c = float(np.nanmedian(adx[m]))
    clusters.append({"cluster": c, "n": nc, "pct": 100*nc/len(idx_use),
                     "r20_med_pct": r_c*100, "edge_pct": edge,
                     "win_tp3x_long": win_l, "win_tp3x_short": win_s,
                     "rsi_med": rsi_c, "adx_med": adx_c})
    print(f"{c:>3} {nc:>7} {100*nc/len(idx_use):>4.1f}% | {r_c*100:>+8.4f}% {edge:>+7.3f} | "
          f"{win_l:>8.1f}% {win_s:>8.1f}% | {rsi_c:>5.1f} {adx_c:>5.1f}")

best = max(clusters, key=lambda x: x["r20_med_pct"])
worst = min(clusters, key=lambda x: x["r20_med_pct"])
max_absedge = max(abs(x["edge_pct"]) for x in clusters)
best_win = max(max(x["win_tp3x_long"] for x in clusters),
               max(x["win_tp3x_short"] for x in clusters))
edge_real = bool(max_absedge > 0.5 * atr_med * 100 and best_win > 55)
summary["clustering"] = {
    "k": K, "horizon": Hc, "baseline_r20_med_pct": g_r*100,
    "best_cluster": best, "worst_cluster": worst,
    "max_abs_edge_pct": max_absedge, "best_dir_winTP_pct": best_win,
    "edge_conditional_plausible": edge_real,
    "clusters": clusters,
}
print(f"\n[VERDICT] best cluster {best['cluster']}: R20 {best['r20_med_pct']:+.4f}% "
      f"(RSI~{best['rsi_med']:.0f} ADX~{best['adx_med']:.0f}, winTP3xL={best['win_tp3x_long']:.1f}%)")
print(f"  |edge_R| max = {max_absedge:.3f} pts vs ATR {atr_med*100:.3f}%  | best dir winTP = {best_win:.1f}%")
print(f"  => EDGE CONDITIONNEL {'PLAUSIBLE' if edge_real else 'NON PROBANT'}")

json.dump(summary, open(OUT, "w"), indent=2, default=str)
print(f"\nsaved {OUT}")
