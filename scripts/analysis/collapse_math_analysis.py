#!/usr/bin/env python3
"""
Analyse mathématique rigoureuse du collapse ADAN0 : V12 (baseline, future_contrib actif)
vs V13 (ablation future_contrib=0).

Méthodes appliquées (demande utilisateur) :
  1. Moyenne arithmétique  — tendance centrale a0_mean, pct_buy par segment
  2. Moyenne harmonique    — tendance des ratios pct_buy/pct_sell, flat/open
  3. Moyenne géométrique   — croissance composée du drift a0_mean
  4. Triangulation         — matrice corr + distance euclidienne (a0_mean,pct_buy,steps_open)
  5. ACP (PCA)             — dimensions expliquant la variance du collapse
  6. ADL (LDA)             — séparation sain (<10k) vs collapsé (>30k)
  7. SVD                   — rang effectif de la matrice de transition état->action
  + Test t de Student      — les pentes de collapse V12 vs V13 diffèrent-elles ?
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import hmean, gmean, ttest_ind, pearsonr, linregress

np.set_printoptions(precision=4, suppress=True)

V12 = "logs/training/diag_v12_500k.csv"
V13 = "logs/training/diag_v13_ablation_nofuture.csv"

def load(path):
    df = pd.read_csv(path).sort_values("timesteps").reset_index(drop=True)
    return df

v12 = load(V12)
v13 = load(V13)

# align on common step range (V13 only reaches 10k)
maxstep = int(v13["timesteps"].max())
v12c = v12[v12["timesteps"] <= maxstep].reset_index(drop=True)

print("="*72)
print("DONNÉES — comparaison sur plage commune [2000 .. %d]" % maxstep)
print("="*72)
print(f"V12 baseline: {len(v12c)} points | V13 ablation: {len(v13)} points")
print("\nV12 baseline (future_contrib ACTIF):")
print(v12c[["timesteps","a0_mean","a0_pct_buy","a0_pct_sell","steps_open_pct","illegal_ratio","policy_entropy"]].to_string(index=False))
print("\nV13 ablation (future_contrib=0):")
print(v13[["timesteps","a0_mean","a0_pct_buy","a0_pct_sell","steps_open_pct","illegal_ratio","policy_entropy"]].to_string(index=False))

# ----------------------------------------------------------------------
# 1. MOYENNE ARITHMÉTIQUE (par segment)
# ----------------------------------------------------------------------
print("\n" + "="*72)
print("1. MOYENNE ARITHMÉTIQUE — tendance centrale")
print("="*72)
for name, d in [("V12", v12c), ("V13", v13)]:
    print(f"{name}: a0_mean={np.mean(d['a0_mean']):+.4f} | pct_buy={np.mean(d['a0_pct_buy']):.4f} | "
          f"pct_sell={np.mean(d['a0_pct_sell']):.4f} | steps_open={np.mean(d['steps_open_pct']):.4f}")

# ----------------------------------------------------------------------
# 2. MOYENNE HARMONIQUE (ratios) — sensible aux petites valeurs
# ----------------------------------------------------------------------
print("\n" + "="*72)
print("2. MOYENNE HARMONIQUE — ratios (pénalise les déséquilibres)")
print("="*72)
for name, d in [("V12", v12c), ("V13", v13)]:
    # ratio buy/sell : harmonique reflète l'équilibre; si sell->0, hmean s'effondre
    buy = np.clip(d["a0_pct_buy"].values, 1e-6, None)
    sell = np.clip(d["a0_pct_sell"].values, 1e-6, None)
    hm_bs = hmean(np.vstack([buy, sell]), axis=0)  # per-point harmonic of (buy,sell)
    flat = np.clip(d["steps_flat_pct"].values, 1e-6, None)
    openp = np.clip(d["steps_open_pct"].values, 1e-6, None)
    hm_fo = hmean(np.vstack([flat, openp]), axis=0)
    print(f"{name}: H(buy,sell) start={hm_bs[0]:.4f} end={hm_bs[-1]:.4f} (chute={hm_bs[0]-hm_bs[-1]:+.4f}) | "
          f"H(flat,open) start={hm_fo[0]:.4f} end={hm_fo[-1]:.4f}")
print("  NB: H(buy,sell) élevé = équilibré; chute = déséquilibre (collapse directionnel)")

# ----------------------------------------------------------------------
# 3. MOYENNE GÉOMÉTRIQUE — croissance composée du drift
# ----------------------------------------------------------------------
print("\n" + "="*72)
print("3. MOYENNE GÉOMÉTRIQUE — croissance composée du drift a0_mean & pct_buy")
print("="*72)
for name, d in [("V12", v12c), ("V13", v13)]:
    # taux de croissance composé de pct_buy (toujours >0) : gmean des ratios successifs
    pb = d["a0_pct_buy"].values
    ratios = pb[1:] / pb[:-1]
    g = gmean(ratios)
    # drift a0_mean : gmean(1+diff) pour capturer croissance additive composée
    da0 = np.diff(d["a0_mean"].values)
    g_a0 = gmean(1.0 + np.clip(da0, -0.99, None))
    print(f"{name}: gmean(pct_buy ratios)={g:.4f}/segment (>1 => croissance BUY) | "
          f"gmean(1+Δa0_mean)={g_a0:.5f} (>1 => drift a0 composé positif)")

# ----------------------------------------------------------------------
# 4. TRIANGULATION — corrélation + distance euclidienne dans (a0_mean,pct_buy,steps_open)
# ----------------------------------------------------------------------
print("\n" + "="*72)
print("4. TRIANGULATION — convergence des 3 signaux du collapse")
print("="*72)
for name, d in [("V12", v12c), ("V13", v13)]:
    M = d[["a0_mean","a0_pct_buy","steps_open_pct"]].values
    C = np.corrcoef(M.T)
    # distance euclidienne du point start au point end dans l'espace normalisé
    Mn = (M - M.mean(0)) / (M.std(0) + 1e-9)
    dist = np.linalg.norm(Mn[-1] - Mn[0])
    print(f"{name}: corr(a0,pct_buy)={C[0,1]:+.3f} corr(a0,open)={C[0,2]:+.3f} corr(buy,open)={C[1,2]:+.3f} "
          f"| trajet euclidien(norm)={dist:.3f}")
print("  NB: 3 corr proches de +1 => les 3 signaux dérivent ENSEMBLE (collapse cohérent)")

# ----------------------------------------------------------------------
# 5. ACP / PCA — quelles métriques portent la variance du collapse
# ----------------------------------------------------------------------
print("\n" + "="*72)
print("5. ACP (PCA) — axes principaux de la variance du collapse")
print("="*72)
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    feats = ["a0_mean","a0_std","a0_pct_buy","a0_pct_sell","steps_open_pct","illegal_ratio","policy_entropy"]
    for name, d in [("V12", v12c), ("V13", v13)]:
        X = StandardScaler().fit_transform(d[feats].values)
        pca = PCA().fit(X)
        evr = pca.explained_variance_ratio_
        print(f"\n{name}: variance expliquée PC1={evr[0]:.3f} PC2={evr[1]:.3f} (PC1+PC2={evr[0]+evr[1]:.3f})")
        load = pca.components_[0]
        order = np.argsort(-np.abs(load))
        print(f"  PC1 loadings (dominant): " + ", ".join(f"{feats[i]}={load[i]:+.2f}" for i in order[:4]))
except Exception as e:
    print("PCA skipped:", e)

# ----------------------------------------------------------------------
# 6. ADL / LDA — séparation sain vs collapsé (utilise V12 full car il va à 40k)
# ----------------------------------------------------------------------
print("\n" + "="*72)
print("6. ADL (LDA) — séparation sain (<10k) vs collapsé (>30k) [V12 full]")
print("="*72)
try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    feats = ["a0_mean","a0_std","a0_pct_buy","a0_pct_sell","steps_open_pct","illegal_ratio","policy_entropy"]
    dfull = v12.copy()
    dfull["label"] = np.where(dfull["timesteps"] < 10000, "sain",
                       np.where(dfull["timesteps"] > 30000, "collapse", "transition"))
    lab = dfull[dfull["label"].isin(["sain","collapse"])]
    X = lab[feats].values; y = (lab["label"]=="collapse").astype(int).values
    lda = LinearDiscriminantAnalysis().fit(X, y)
    coef = lda.coef_[0]
    order = np.argsort(-np.abs(coef))
    acc = lda.score(X, y)
    print(f"  Séparation sain/collapse : accuracy={acc:.3f} (n_sain={sum(y==0)}, n_collapse={sum(y==1)})")
    print(f"  Discriminants dominants : " + ", ".join(f"{feats[i]}={coef[i]:+.2f}" for i in order[:4]))
except Exception as e:
    print("LDA skipped:", e)

# ----------------------------------------------------------------------
# 7. SVD — rang effectif de la matrice de transition état->action
# ----------------------------------------------------------------------
print("\n" + "="*72)
print("7. SVD — rang effectif matrice de transition (proxy via req_*_pct)")
print("="*72)
for name, d in [("V12", v12c), ("V13", v13)]:
    # matrice [temps x (req_HOLD,req_BUY,req_SELL)] : distribution demandée par step
    T = d[["req_HOLD_pct","req_BUY_pct","req_SELL_pct"]].values
    U, S, Vt = np.linalg.svd(T, full_matrices=False)
    # rang effectif (participation ratio des valeurs singulières)
    s2 = S**2
    eff_rank = (s2.sum()**2) / (s2**2).sum()
    print(f"{name}: singular values={S} | rang effectif={eff_rank:.3f} "
          f"(1.0 => 1 seul mode domine = comportement dégénéré)")

# ----------------------------------------------------------------------
# TEST DÉCISIF — pentes de collapse V12 vs V13 (test t de Student)
# ----------------------------------------------------------------------
print("\n" + "="*72)
print("TEST DÉCISIF — l'ablation a-t-elle changé la PENTE de collapse ?")
print("="*72)
for metric in ["a0_mean","a0_pct_buy"]:
    r12 = linregress(v12c["timesteps"], v12c[metric])
    r13 = linregress(v13["timesteps"], v13[metric])
    # test t sur différence de pentes : t = (b1-b2)/sqrt(se1^2+se2^2)
    t = (r12.slope - r13.slope) / np.sqrt(r12.stderr**2 + r13.stderr**2)
    print(f"\n{metric}:")
    print(f"  V12 pente={r12.slope:+.3e}/step (se={r12.stderr:.1e}, R²={r12.rvalue**2:.3f})")
    print(f"  V13 pente={r13.slope:+.3e}/step (se={r13.stderr:.1e}, R²={r13.rvalue**2:.3f})")
    print(f"  Δpente(V12-V13)={r12.slope-r13.slope:+.3e} | t={t:+.2f} "
          f"(|t|<2 => pentes NON significativement différentes)")

print("\n" + "="*72)
print("VERDICT")
print("="*72)
sl12 = linregress(v12c["timesteps"], v12c["a0_pct_buy"]).slope
sl13 = linregress(v13["timesteps"], v13["a0_pct_buy"]).slope
if abs(sl13) >= 0.5*abs(sl12):
    print("ABLATION future_contrib=0 A ÉCHOUÉ : la dérive pct_buy persiste (pente V13 >= 50% V12).")
    print("=> future_contrib N'EST PAS le moteur principal. Chercher ailleurs (pnl_base/closure/latent).")
else:
    print("ABLATION future_contrib=0 A RÉDUIT la dérive (pente V13 < 50% V12).")
    print("=> future_contrib EST un moteur significatif. Fix = rééquilibrer ce terme.")
