# Rapport d'analyse — Run 500k (capture-ratio, DiagGaussian)

Log : `logs/training/fa_500k.log` (6.4M lignes parsées, ~200k steps au moment de l'analyse, run encore actif).
Outil : `scripts/analysis/analyze_run.py` (lecture seule, streaming).

---

## A. Évolution RL (santé de l'apprentissage)

| Métrique | first → last | tendance | lecture |
|---|---|---|---|
| **ep_rew_mean** | 1578 → 3039 | **UP +85%** | le reward épisodique monte fortement |
| **entropy_loss** | -2.59 → -6.22 | **DOWN -81%** | la politique se **concentre** (exploration ↓) |
| **value_loss** | 0.42 → 0.62 | UP +14% | le critic peine à suivre (cible non-stationnaire) |
| **explained_variance** | 0.011 → -0.12 | STABLE ≈0 / parfois négatif | ⚠️ **le critic n'explique PAS les retours** |
| **approx_kl** | 0.17 → 0.18 | STABLE (mais max 1.27) | pas de divergence brutale moyenne, pics ponctuels |
| **clip_fraction** | 0.52 → 0.53 | STABLE ~0.55 | ⚠️ **très élevé** (sain < 0.2) : updates trop agressifs |
| **policy_gradient_loss** | -0.018 → -0.022 | STABLE | ok |

**Lecture RL** : le reward monte (+85%) et l'entropie chute → le modèle **apprend et se spécialise**, il ne stagne pas. MAIS deux drapeaux :
1. **explained_variance ≈ 0 / négatif** : le critic ne modélise pas bien la valeur. L'avantage estimé est bruité → apprentissage policy peu fiable.
2. **clip_fraction ~0.55** : plus de la moitié des updates sont clippés → le learning rate / n_epochs (20) est trop agressif pour la non-stationnarité. Risque d'instabilité.

---

## B. Évolution trading (comportement de sortie)

| | TP_HIT | SL_HIT | AGENT_CLOSE | TP/SL |
|---|---|---|---|---|
| **Global** | 45 384 | 10 254 | 60 | **4.43** |

Évolution par tiers du run :

| Tiers | TP% | SL% | TP/SL |
|---|---|---|---|
| early | 77.2 | 22.4 | 3.44 |
| mid | 86.1 | 13.9 | **6.18** |
| late | 81.1 | 18.9 | 4.30 |

**Lecture trading** : **inversion totale** vs l'ancien modèle "lâche" (qui faisait 84% SL_HIT, expectancy −$0.58). Ici **81% TP_HIT** en fin de run. AGENT_CLOSE quasi nul (60) → le modèle ne micro-scalpe plus pour fuir, il **laisse courir jusqu'au TP**. Le ratio TP/SL monte (3.4→6.2) puis redescend (4.3) : pic au milieu, léger recul ensuite = le modèle a trouvé une zone exploitable et commence à osciller autour.

⚠️ **Nuance importante** : un TP/SL de 4.4 avec TP 1.5% / SL ~1% est *mathématiquement très favorable*, mais à confirmer **net de frais** (0.50% A/R) sur un replay hors-échantillon. Un TP de 1.53% net de 0.50% frais = **~1.03% net** : encore positif, mais la marge se réduit. **C'est le test décisif avant tout paper.**

---

## C. Comportement (Future Arena + actions)

| Métrique | first → last | lecture |
|---|---|---|
| **FA future_share** | 29.6% → **6.0%** | ✅ la Future Arena ne domine PAS (cible <40%). Le PnL pilote. Pas d'exploitation du shaping. |
| FA mean_abs_pnl | 0.038 → 0.652 | le signal PnL grandit (trades plus gros / plus fréquents) |
| **tp_pct_mean** | 1.28% → 1.53% | TP choisi stable, cohérent bande scalper [0.6%, 2.0%] |
| **tp_sat** | 0% → **43%** | ⚠️ le modèle pousse vers la **borne haute TP** 43% du temps |
| **sl_sat** | 1% → **47%** | ⚠️ le modèle pousse vers la **borne SL** 47% du temps |
| tp_raw_mean | -0.02 → +0.33 | dérive vers TP plus large |
| sl_raw_mean | -0.01 → +0.57 | dérive vers SL plus large |

**Lecture comportementale** : le modèle **converge vers les bornes** (saturation 43-47%). Interprétation : avec un marché BTC où la MFE typique dépasse souvent 1.5%, le modèle apprend (rationnellement) qu'**il voudrait un TP encore plus haut que la borne scalper (2%)**. La borne le bride. C'est cohérent avec ta philosophie : *le marché n'a pas de TP max*. **Le profil scalper est peut-être trop serré pour ce que le modèle veut faire.**

---

## VERDICT

**Le modèle ÉVOLUE et cherche activement une voie exploitable** — il ne stagne pas, ne s'effondre pas (collapse). Preuves :
- ep_rew_mean +85%, entropy ↓ (spécialisation), TP/SL 4.4, 81% TP_HIT.
- future_share 6% → la performance vient du **vrai PnL**, pas d'un hack de reward.

**Réserves (à lever avant paper/live)** :
1. **explained_variance ≈ 0** → critic faible. À surveiller : si ça reste ≤0 jusqu'à 500k, le modèle "réussit" surtout par chance de régime, pas par valeur apprise.
2. **clip_fraction ~0.55** → updates trop agressifs (baisser LR ou n_epochs).
3. **Saturation des bornes 43-47%** → le scalper bride le modèle. Envisager d'élargir la borne TP scalper OU router vers un profil intraday.
4. **Performance NETTE de frais NON confirmée hors-échantillon** → backtest replay obligatoire avant tout paper.

**Recommandation paper trading** : ne PAS lancer en paper tant que (4) n'est pas validé sur un replay out-of-sample net de frais. Le modèle est prometteur mais entraîné sur BTCUSDT ; un lancement paper sur SOL/NEAR/BCH/DOGE = inférence hors-distribution → risque élevé. Décision après replay.
