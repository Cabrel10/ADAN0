# ADAN — MTF v2 (round 4) : verdict final apres methodologie durcie (2026-09-10)

## Protocole applique (cahier des charges integral)
Confluence continue c(t)=tanh(Σ w_k z_k), poids par information mutuelle (train only) ;
Edge Ratio Sweeney (separation MAE winners/losers, SL* par Youden) ; TP* = quantile MFE
conditionne a la survie a SL* ; EV normalisee par temps de detention ; IC Wilson ;
puissance pre-enregistree n>=200/cellule ; block bootstrap (blocs 5 trades, B=2000) ;
stabilite spatiale plateau 3x3 ; FDR Benjamini-Hochberg ; validation croisee BTC<->DOGE ;
walk-forward 3 tiers chronologiques.

## Bug corrige en cours de route (negatif -> pourquoi -> corrige)
atr_pct est deja une FRACTION (mediane 0.14%) ; la division /100 ecrasait SL/TP
(tous les trades stoppes a h=1, ht.mean()==1, ev_per_bar==ev_net). Fix commite (b04ce4d).
Resultats ci-dessous = probe corrige (run2, JSON mtf_confluence_v2_20260910_093254.json).

## Resultat : NEGATIF
- Walk-forward OOS (DOGE, 2 fenetres) : EV_net = -0.495% / -0.551% (IC95 entierement < 0).
- Cross-asset gele (BTC->DOGE test, n=52) : EV_net = -0.517%, P(TP) Wilson 9.6% [4.2, 20.6].
- Aucune cellule ne passe CI95_low > 0 avec n>=200 hors train.
- EV brute ~ 0 : les pertes nettes correspondent aux seuls couts (0.50%).

## Limites honnetes mesurees (non resolus)
- Datasets courts : BTC train = 7991 barres 5m (~28 j) -> deciles sous-dimensionnes,
  monotonie non testable (b1 nan), puissance insuffisante sur BTC.
- Poids MI du 4h ~ 0 : discretisation par quantiles sur ~80 trades -> information nulle ;
  ne prouve pas l'inutilite du 4h, seulement l'absence de signal mesurable a ce n.

## CONCLUSION (confirmee par 2 methodologies independantes R3 + R4)
La confluence MTF est un biais predictif REEL (R1/R2 : filtre > aveugle, reproductible)
mais d'amplitude << frontiere economique. A 0.40% de frais round-trip sur du 5m :
NO-GO pour une strategie autonome. Conformement au garde-fou : AUCUNE modification
du reward PPO. Voies recommandees par ordre de levier economique :
  1. Reduction des frais (maker 0.02-0.05% via BNB/limits) — change la frontiere ;
  2. Horizon multi-jour (le signal 4h n'a de sens qu'a cette echelle) ;
  3. Integration comme PRIOR de policy (masque d'actions / shaping MFE-MAE),
     jamais comme strategie autonome a ces frais.
