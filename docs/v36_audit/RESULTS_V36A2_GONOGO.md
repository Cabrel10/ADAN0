# V36-A2 — GO/NO-GO final (dernier test de pondération simpliste)

**Date** : 2026-08-23
**Run** : V36-A2 = V36-A avec `drawdown_penalty_factor` ÷ 4 par tier.
**Terminé** : 50 176 steps / 97 updates PPO, 6347 s.
**Terminal** : equity 20.50, realized_pnl **-27.33**, 526 trades.

## Métriques comparées

| | V36-A | V36-A2 | seuil GO |
|---|---|---|---|
| PnL réalisé cumulé | -37.18 | **-27.33** | > 0 ❌ |
| profit factor | 0.208 | **0.257** | > 1 ❌ |
| win rate | 17.3 % | 18.8 % | — |
| reward moy gagnants | -0.012 | **+0.027** | > perdants |
| reward moy perdants | — | -0.328 | ✅ gagnants > perdants |
| BUY / SELL / HOLD (moy5) | 52 / 4928 / 140 | **0 / 5104 / 16** | équilibre ❌ |
| drawdown share | 58.6 % | 26.6 % | — |
| pnl_reward share | 22.6 % | 39.6 % | — |
| EV (moy5) | -0.487 | -1.596 | — |
| expectancy / trade | -0.060 | -0.052 | > 0 ❌ |

## Ce que ÷4 a changé — et ce qu'il n'a PAS changé

**Changé (mécanique)** : drawdown_penalty n'écrase plus le signal (58.6 → 26.6 %),
pnl_reward remonte (22.6 → 39.6 %), critère « reward gagnants > perdants » ENFIN
vérifié. PnL brut -37 → -27.

**PAS changé (le cœur)** : ratios de clôture **identiques** à A —
MaxDuration 89.4 % (A : 89.9 %), take_profit **0.2 % (1 trade/526)**,
exposition médiane 80.5 %, equity max jamais **20.53** → jamais ≥ 21.
Régression : **BUY collapse à 0** (politique 100 % SELL).

## Décision : **NO-GO 500k** + **STOP variations de reward**

Re-pondérer ne touche pas la brique cassée = **le contrat de SORTIE** (voir
`AUTOPSIE_ECONOMIQUE.md`). Prochaine étape = **étude sans entraînement** de la
géométrie temporelle (courbe de maturité MFE/MAE, ACP+clustering, horizon H*),
puis refonte du contrat {horizon, sortie}. **Pas de A3/A4/A5.**
