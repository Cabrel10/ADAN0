# Résultats de l'ablation causale V13 — future_contrib=0

MAJ 2026-07-04. Méthode : run court 10k steps, `ADAN_ABLATE_FUTURE_CONTRIB=1`,
toutes choses égales par ailleurs (profil intraday, mêmes seeds/hyperparams).
Comparaison vs baseline V12 sur plage commune [2000..10000].

## Données brutes (5 points chacun)

| step | V12 a0_mean | V12 pct_buy | V13 a0_mean | V13 pct_buy |
|------|-------------|-------------|-------------|-------------|
| 2000 | -0.005 | 0.454 | +0.010 | 0.496 |
| 4000 | +0.012 | 0.511 | +0.039 | 0.586 |
| 6000 | +0.030 | 0.548 | +0.051 | 0.616 |
| 8000 | +0.059 | 0.639 | +0.056 | 0.626 |
| 10000 | +0.061 | 0.649 | +0.061 | 0.653 |

## Analyse mathématique (7 méthodes)

1. **Moyenne arithmétique** : V13 (a0_mean +0.043, pct_buy 0.595) ≈ V12 (+0.031, 0.560).
   Pas d'amélioration, légèrement PIRE.
2. **Moyenne harmonique H(buy,sell)** : chute V13=+0.0623 ≈ V12=+0.0620. Déséquilibre identique.
3. **Moyenne géométrique** gmean(pct_buy ratios) : V13=1.071/seg ≈ V12=1.094/seg. Même drift composé.
4. **Triangulation** : corr(a0_mean,pct_buy)=+0.998 (V12) / +0.997 (V13). Les 3 signaux
   (a0_mean, pct_buy, steps_open) dérivent ENSEMBLE de façon identique.
5. **ACP** : PC1 = 77% (V12) / 82% (V13) de la variance, dominé par (a0_mean, pct_buy,
   pct_sell). Même axe principal de collapse dans les deux runs.
6. **ADL/LDA** : séparation sain(<10k)/collapse(>30k) à 100% ; discriminants dominants =
   **a0_std, policy_entropy, illegal_ratio** — PAS future_contrib.
7. **SVD** : rang effectif de la matrice de transition état→action = **1.001** dans les
   DEUX cas → matrice dégénérée (1 seul mode = toujours-BUY).

## Test décisif (test t de Student sur les pentes)

| Métrique | Pente V12 | Pente V13 | Δpente | t | Verdict |
|----------|-----------|-----------|--------|---|---------|
| a0_mean | +8.9e-6 (R²=.96) | +6.0e-6 (R²=.87) | +2.9e-6 | **+1.71** | NON sig (|t|<2) |
| pct_buy | +2.6e-5 (R²=.96) | +1.8e-5 (R²=.86) | +8.2e-6 | **+1.56** | NON sig (|t|<2) |

## VERDICT

**L'ablation future_contrib=0 A ÉCHOUÉ.** Les pentes de collapse V12 et V13 sont
statistiquement indistinguables (|t| < 2). `future_contrib` **N'EST PAS le moteur
principal** du BUY runaway. Il contribuait au reward mais sa suppression ne change pas
la dynamique de dérive.

## Implication pour le prochain fix

Le moteur est un terme structurellement pro-position présent MÊME sans future_contrib.
Candidats restants à investiguer (par ordre de suspicion) :
1. **`latent_pnl`** — asymétrique (lambda_loss 0.15 > lambda_gain 0.10) et n'existe QUE
   quand une position est ouverte → récompense structurellement le fait d'être en position.
2. **`closure_bonus`** — si asymétrique gagnant/perdant, biaise vers/contre SELL.
3. **`pnl_base`** — la mécanique même de récompense du PnL réalisé (fermeture).
4. Absence de coût de détention (holding) → aucune pression pour sortir de LONG.
