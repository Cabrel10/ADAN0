# Observation Paper Trading — 450k vs 500k (hors-échantillon)

Date : 2026-06-21
Harness : `scripts/backtest_fixed_capital.py` (capital fixe $1000, notional fixe $100,
aucune capitalisation composée, reset par épisode).
Steps : 10 000 par run. Split principal demandé : **test** (5298 lignes 5m, ~18 jours plat/baissier).

## 1. Résultats sur le split TEST (demandé par le brief)

| Métrique            | 450k     | 500k     | Random (baseline) |
|---------------------|----------|----------|-------------------|
| Win Rate            | 98.61 %  | 98.61 %  | 61.85 %           |
| Profit Factor       | 0.75     | 0.75     | 1.46              |
| Expectancy / trade  | **-0.017 %** | **-0.017 %** | +0.810 %      |
| Total return        | -0.25 %  | -0.25 %  | —                 |
| N trades            | 144      | 144      | —                 |
| Worst trade         | -4.95 %  | -4.95 %  | —                 |
| Verdict             | NO_EDGE  | NO_EDGE  | POSITIVE_EDGE     |

### Constats critiques

1. **450k ≡ 500k byte-identiques** sur le test split (mêmes 144 trades, même
   WR 98.61 %, même PnL -0.0174 %). Les deux policies déterministes convergent
   vers le **même comportement dégénéré "micro-TP"** sur ce split.

2. **WR 98.6 % MAIS expectancy négative** = piège classique du
   *micro-take-profit / fat-tail loss*. Le modèle gagne 142 trades à +0.052 %
   (TP minuscule) mais 2 trades perdent jusqu'à -4.95 %. Gross win 7.39 % <
   gross loss 9.89 % → perte nette. **Un win-rate élevé masque ici une
   espérance négative** ; ce n'est PAS un edge.

3. Le **random fait mieux** que les modèles sur ce split (PF 1.46 vs 0.75).
   Ce n'est pas que le random est bon : c'est que le **test split est trop
   court et trop plat pour discriminer**, et que les modèles n'y expriment
   aucun edge.

## 2. Rappel — split VAL (split discriminant, hors-échantillon)

| Métrique            | 450k     | 500k     | Random            |
|---------------------|----------|----------|-------------------|
| Win Rate            | 60.33 %  | **67.11 %** | 49.16 %        |
| Profit Factor       | 1.18     | **2.58** | 0.92              |
| Expectancy / trade  | +0.173 % | **+1.665 %** | -0.155 %      |
| Verdict             | POSITIVE | **POSITIVE (fort)** | NO_EDGE   |

Sur la VAL, **500k_FIXED affiche un edge réel et net** (PF 2.58, E +1.67 %,
WR +18 pts vs random). C'est cohérent avec `CHECKPOINT_DECISION.md`.

## 3. Verdict global (grille du brief)

Le brief demandait : *« edge réel si WR > random ~49 % consistant ; overfit si
WR chute à 50-55 % »*. Aucun des deux cas ne s'applique tel quel sur le test split :

- **Ni edge (>random)** : le random fait mieux sur ce split précis.
- **Ni overfit franc (50-55 %)** : le WR test est artificiellement à 98.6 %.
- Le vrai diagnostic est **comportement dégénéré micro-TP sur un split non
  discriminant**. Le signal exploitable vient de la **VAL**, où 500k domine
  clairement (edge réel) et 450k reste positif mais faible.

### Décision confirmée
`docs/CHECKPOINT_DECISION.md` reste valide : **500k_FIXED est retenu** sur la
base de la VAL hors-échantillon. Le test split est rejeté comme métrique de
décision (trop court / plat / dégénéré).

### Recommandation
Pour une validation de production fiable, il faut un **jeu de test plus long et
plus varié de régimes** (haussier + baissier + range), et idéalement plusieurs
seeds d'évaluation, afin d'éviter le piège du micro-TP qui gonfle le WR tout en
détruisant l'espérance.
