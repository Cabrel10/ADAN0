# Rapport d'évolution par checkpoint — Run 500k

Outil : `scripts/analysis/checkpoint_evolution.py` (bin = 50k timesteps).
Source : `logs/training/fa_500k.log`.

## Tableau d'évolution

| Ckpt (~steps) | ep_rew | TP/SL | entropy | expl_var | value_loss | tp_sat% | sl_sat% | fa_share% |
|---|---|---|---|---|---|---|---|---|
| 50k | 1360 | **2.62** | -2.68 | 0.002 | 0.43 | 25 | 6 | 16.1 |
| 100k | 2238 | **6.44** ⭐ | -3.77 | -0.04 | 0.56 | 37 | 12 | 6.4 |
| 150k | 2607 | 5.25 | -4.87 | -0.15 | 0.61 | 34 | 27 | 6.1 |
| 200k | 2984 | 4.61 | -6.01 | -0.13 | 0.61 | 39 | 42 | 6.0 |
| 250k | 3080 | 3.64 | -6.44 | -0.10 | 0.68 | 43 | 47 | 6.0 |

## Lecture

### 🚩 Signal #1 — DIVERGENCE reward ↑ vs qualité ↓ (alerte automatique)
Le **reward monte continûment** (1360→3080) MAIS le **TP/SL pique à 100k (6.44)
puis DÉCROÎT** (→3.64 à 250k). Le reward ne mesure donc PLUS la qualité de trading
après 100k. C'est précisément le cas "reward augmente mais profit factor diminue"
à détecter. **Le reward seul est trompeur ici.**

### 🚩 Signal #2 — Saturation SL en explosion (6%→47%)
Le modèle colle de plus en plus à la **borne SL** (sl_sat 6→47%). Couplé à la
chute du TP/SL, cela suggère qu'après 100k le modèle élargit ses SL (vers la borne)
pour "survivre" plus longtemps et grappiller du reward de durée/exposition, au
détriment de la qualité des sorties. Début de dérive vers une stratégie sous-optimale.

### 🚩 Signal #3 — Entropy collapse + explained_var négatif
- entropy -2.68→-6.44 : la policy se **fige** (exploration quasi nulle).
- explained_var reste **≤0** : le critic n'explique jamais les retours → avantages
  bruités → le PPO optimise sur un signal de valeur peu fiable. C'est cohérent avec
  un modèle qui "trouve un truc qui marche" sans vraie compréhension de la valeur.

### Verdict transitions
- 50k→100k : **PROGRESSE FORT** (reward +878, TP/SL 2.6→6.4).
- 100k→150k→200k : reward monte mais **qualité trading se dégrade** (TP/SL ↓).
- 200k→250k : **PLATEAU reward** (+96) + qualité encore en baisse → début de plateau/dérive.

## CONCLUSION

**Le modèle a atteint son meilleur compromis qualité/robustesse autour de 100k steps**,
PAS au dernier checkpoint. Après 100k il "sur-optimise le reward" en dégradant la
qualité réelle des trades (TP/SL) et en saturant les bornes SL.

### Candidat retenu pour validation out-of-sample
➡️ **`ppo_adan0_sandbox_checkpoint_100000_steps.zip`**
- TP/SL le plus élevé (6.44), saturation encore modérée (tp 37% / sl 12%),
  entropy pas encore effondrée, fa_share sain (6.4%).

### Candidat de secours
➡️ `ppo_adan0_sandbox_checkpoint_150000_steps.zip` (TP/SL 5.25, compromis).

### Recommandation sur le run
Continuer le run n'améliore probablement PAS la qualité de trading (elle se dégrade).
Mais le laisser finir donne plus de checkpoints à comparer. **La décision finale
d'actif/paper se fera APRÈS le replay out-of-sample net de frais du checkpoint 100k.**
