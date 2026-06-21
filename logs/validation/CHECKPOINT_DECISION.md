# ADAN0 — Décision de checkpoint (backtest honnête à capital fixe)

> Généré le 2026-06-21. Source : `scripts/backtest_fixed_capital.py`
> Données : split `test` (2026-05-12 → 05-30, marché baissier ~-9%, ~5300 lignes 5m).
> Convention : capital fixe 1000 $, notional fixe 100 $/trade, **AUCUN compounding**,
> reset propre par épisode, `pnl_pct` lu directement du `trade_log` du portefeuille.

## 1. Pourquoi ce backtest remplace `deterministic_backtest.py`

`deterministic_backtest.py` calculait :

```python
total_return_pct = (last_portfolio_value - 20.50) / 20.50 * 100
```

où `last_portfolio_value` est l'**équité composée** de l'environnement RL, jamais
remise à zéro entre épisodes. Résultat : le capital s'accumule mécaniquement
(position sizing sur capital croissant) et produit des chiffres impossibles
(+46.89 % / +1485 % sur un marché à -9 %). Le `pnl_sum=0` et
`trades_detected_in_loop=0` confirmaient que la boucle ne « voyait » aucun trade
— tout venait de `env_info`.

`backtest_fixed_capital.py` mesure la **stratégie**, pas la trajectoire d'équité :
chaque trade clôturé contribue son `pnl_pct` (frais inclus) appliqué à un notional
fixe. 90 trades gagnants ne peuvent plus faire boule de neige.

## 2. Résultats bruts (5000 steps, split test)

| Métrique                  | RANDOM   | 450k     | 500k     |
|---------------------------|----------|----------|----------|
| n_trades                  | 579      | 116      | 103      |
| **win_rate**              | **49.4 %** | **83.6 %** | **89.3 %** |
| avg pnl % / trade         | +2.169 % | +6.356 % | +2.076 % |
| median pnl %              | -4.02 %  | +12.0 %  | +0.82 %  |
| profit_factor             | 1.88     | 11.81    | 9.49     |
| sharpe_like               | 0.295    | 0.985    | 0.566    |
| max consécutif perdants   | 10       | 3        | 7        |
| total_return_pct (*)      | 125.6 %  | 73.7 %   | 21.4 %   |

(*) `total_return_pct = somme(pnl_pct) × (notional/capital)`. **Cette métrique
biaise en faveur du sur-trading** : le random fait 5.6× plus de trades (579 vs
103), donc sa somme de % gonfle artificiellement. Elle n'est PAS le critère de
décision — la qualité par trade l'est.

## 3. Lecture critique

- **Le random a un win-rate ≈ 49.4 %** : exactement le coin-flip attendu d'un
  agent aléatoire avec SL/TP symétriques. C'est le **plancher de bruit** correct
  → le backtest n'est pas truqué en faveur du modèle.
- **Edge de win-rate réel** :
  - 450k : 83.6 % → **+34.2 points** au-dessus du random
  - 500k : 89.3 % → **+39.9 points** au-dessus du random
- **Profit factor** : random 1.88 vs modèles 9–12 → les modèles sont beaucoup
  plus sélectifs (peu de trades, mais de qualité).
- **Le `total_return` du random (125 %) > modèles** est un **artefact de
  fréquence** : 579 paris à PF 1.88 cumulent plus de % brut que 103 paris à PF 9,
  mais avec une variance énorme (10 pertes consécutives, médiane NÉGATIVE -4 %).
  En capital réel composé, le random serait ruiné par les séries de pertes ; les
  modèles non.

## 4. Règle de décision (fournie par l'utilisateur)

> Si `WR_500k ≈ WR_random` → pas d'alpha → ré-entraîner.
> Si `WR_500k > WR_random + 15 pts` → fine-tune.

Application :

```
WR_random + 15 pts = 49.4 + 15 = 64.4 %
450k = 83.6 %  > 64.4 %  → ALPHA CONFIRMÉ
500k = 89.3 %  > 64.4 %  → ALPHA CONFIRMÉ
```

**Les deux checkpoints battent largement le seuil d'alpha (+34 et +40 pts).**

## 5. Quel checkpoint ?

| Critère                      | 450k       | 500k       | Gagnant |
|------------------------------|------------|------------|---------|
| Win rate                     | 83.6 %     | **89.3 %** | 500k    |
| PnL moyen / trade            | **+6.36 %**| +2.08 %    | 450k    |
| Profit factor                | **11.81**  | 9.49       | 450k    |
| Sharpe-like                  | **0.985**  | 0.566      | 450k    |
| Max pertes consécutives      | **3**      | 7          | 450k    |
| Médiane pnl %                | **+12 %**  | +0.82 %    | 450k    |

**DÉCISION : retenir le checkpoint 450k comme candidat principal.**

Justification :
- Le 500k a un win-rate plus élevé (89 %) mais des gains/trade beaucoup plus
  petits (médiane +0.82 % vs +12 %) et un drawdown de série plus profond (7 vs 3).
  Il « gratte » beaucoup de petits gains → fragile aux frais/slippage réels.
- Le 450k combine PnL/trade élevé, meilleur profit factor, meilleur Sharpe et
  la plus courte série de pertes → **profil risque/rendement nettement supérieur**.
- Le sur-entraînement entre 450k→500k semble avoir poussé vers une politique
  « haut WR / petit gain » (signe de léger overfit sur le bruit).

## 6. Mises en garde avant tout déploiement (paper trading)

⚠️ Ce backtest reste sur un échantillon **court et baissier** (18 jours). Avant
paper trading :

1. Vérifier que l'env paper = env backtest (mêmes slippage 0.02 %, fees 0.1 %,
   fréquence max 1 trade/4h, formules SL/TP identiques).
2. **Désactiver le compounding** dans l'env RL (lock `initial_equity = 20.50`).
3. Rejouer 450k ET 500k sur le split `val` pour confirmer la généralisation.
4. Ne lancer le paper trading que si les deux restent favorables hors échantillon.

## 7. Livrables

- `logs/validation/backtest_CORRECTED_450k_test.json`
- `logs/validation/backtest_CORRECTED_500k_test.json`
- `logs/validation/backtest_RANDOM_test.json`
- `scripts/backtest_fixed_capital.py` (script de référence)
