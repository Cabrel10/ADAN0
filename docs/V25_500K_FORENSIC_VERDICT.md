# Verdict forensic du run V25 500k

Date d'analyse : 2026-08-03  
Périmètre : ADAN0 uniquement  
Run : `adan_500k_v25_20260802T141709Z`  
Checkpoint : `ppo_adan0_FA_500k_v25_20260802T141709Z.zip`

## 1. Verdict exécutif

Le run V25 s'est terminé normalement au sens informatique, mais le modèle produit est catastrophique au sens trading et ne doit être utilisé ni en paper trading ni en live.

Constats principaux :

- Stable-Baselines3 a terminé à 500 224 timesteps pour 500 000 demandés. L'écart de 224 est l'overshoot normal d'un rollout PPO de 512 steps.
- La dernière télémétrie policy est à `global_step=500220` et le checkpoint final existe.
- La trace contient 1 814 OPEN mais seulement 1 805 CLOSE.
- Les 1 805 cycles complets donnent 258 gains, 1 547 pertes et un PnL net tracé de `-106.100896`.
- Après le dernier CLOSE à 164 418, le modèle reste flat et émet presque exclusivement SELL. Il ne trade plus pendant 335 802 steps observés.
- À partir de 200k, aucun BUY brut n'est observé : chaque step audité est un SELL lorsque flat, routé vers HOLD conformément au routeur canonique.
- Le reward terminal devient presque entièrement nul. PPO continue donc ses updates dans une vaste région sans signal financier utile.
- Le watchdog Future Arena rapporte une part absolue de 59,5 %, au-dessus de la cible de 40 %, statut `CRITICAL`.
- Les métriques PPO terminales sont dégradées : `approx_kl=0.0632`, `clip_fraction=0.63`, `explained_variance=-0.0503`.

Décision :

1. V25 est interdit de paper/live.
2. L'artefact lifecycle historique reste rouge et ne doit pas être réécrit.
3. Aucun changement PPO, reward ou routing n'est justifié par les seules preuves V25.
4. Le premier correctif autorisé est la cohérence lifecycle `close_position -> receipt financier -> CLOSE JSONL`, puis le comptage run-level.

## 2. Invariants préservés

Les seize features canoniques restent inchangées :

```python
FEATURES = [
    "ema_ratio",
    "macdh",
    "rsi",
    "adx_14",
    "di_delta",
    "atr_pct",
    "bb_percent_b_20_2",
    "obv_slope",
    "volume_ratio_20",
    "volatility_ratio_14_50",
    "fib_ratio",
    "price_action",
    "vwap_ratio",
    "market_structure",
    "bb_width_20_2",
    "log_return",
]
```

Mapping physique conservé :

```text
ema_20_ratio  -> ema_ratio
macdh_12_26_9 -> macdh
rsi_14        -> rsi
```

Le présent verdict ne change ni PPO, ni reward, ni hyperparamètres, ni routeur.

## 3. Artefacts et intégrité du run

Artefacts analysés en lecture seule :

- log d'entraînement V25 ;
- trace action pipeline JSONL ;
- CSV ActionDim ;
- CSV diagnostic ;
- manifeste runtime ;
- checkpoint final.

Preuves de terminaison normale :

```text
Training done: +500000 steps (cum=500224)
```

Aucun traceback terminal n'est présent. Le checkpoint final mesure 7 382 827 octets.

Le manifeste runtime est toutefois resté à `STARTING`. Il s'agit d'une incohérence de statut runtime, pas d'une preuve d'échec de l'entraînement.

## 4. Télémétrie action pipeline complète

La trace contient 998 407 événements, soit 339 808 720 octets.

| Stage | Nombre |
|---|---:|
| policy | 500 220 |
| routing_reject | 438 099 |
| barrier_reject | 24 157 |
| deadband_reject | 19 193 |
| budget_reject | 13 119 |
| trade_executed | 3 619 |

`trade_executed=3 619` signifie ici `1 814 OPEN + 1 805 CLOSE`. Ce nombre n'est pas un nombre de cycles complets.

Raisons principales :

| Raison | Nombre |
|---|---:|
| sell_while_flat | 428 958 |
| negative_ev_fee_gate | 24 157 |
| inside_action_deadband | 19 193 |
| decision_budget_or_quota | 13 119 |
| buy_while_long | 9 141 |
| position_opened | 1 814 |
| MaxDuration | 1 193 |
| stop_loss | 416 |
| agent_close | 147 |
| take_profit | 49 |

## 5. Première égalité lifecycle rompue

L'anomalie prouvée est :

```text
close_position() réussi
!=
CLOSE JSONL émis
```

Le bilan exact est :

```text
4 DRAWDOWN_KILL
+ 5 CHUNK_END_FORCE_CLOSE
= 9 CLOSE manquants
```

Les OPEN orphelins commencent aux global steps :

```text
4787, 10976, 16395, 23873, 31808,
63555, 71507, 87386, 103256
```

Le portefeuille financier ferme bien ces positions. L'erreur se situe après la fermeture financière : le receipt n'est pas publié dans la trace. La correction doit impérativement respecter :

```text
close_position()
-> receipt financier complet
-> cash/equity/PnL déjà mis à jour
-> émission d'un CLOSE réel et unique
```

Il est interdit de tracer avant la fermeture financière. La jointure reste exclusivement fondée sur `position_id`.

Défauts latents similaires identifiés :

- `BANKRUPT_FORCE_CLOSE` ignore son receipt et avale les exceptions ;
- le remplacement d'une position par `FORCE_TRADE` ferme financièrement sans autorité de publication unique ;
- le résumé sandbox lit le snapshot du dernier épisode au lieu du cumul du run.

## 6. Finance canonique et réconciliation

Formules vérifiées :

```python
pnl_gross = (exit_price - entry_price) * size
fees = (entry_price + exit_price) * size * fee_pct
pnl_net = pnl_gross - fees
```

Pour V25 :

```text
fee_pct par côté = 0.002
round trip nominal = 0.004
slippage = 2 bps par côté
size = requested_notional * 1.0002 / entry_fill_price
```

Cette dernière relation explique que le sizing part du prix avant slippage, puis que le fill BUY est augmenté de 2 bps. Les quinze cycles ci-dessous sont réconciliés avec un résidu nul ou de l'ordre de `1e-17`.

## 7. Quinze cycles représentatifs

Le capital avant/après est reconstruit à partir du capital initial d'épisode de 20,50 et de la somme ordonnée des `pnl_net` CLOSE du même épisode. Il s'agit du capital réalisé après clôture, pas d'un mark-to-market intrastep.

| # | OPEN->CLOSE | Durée steps | Action OPEN `[dir,size,tf,sl,tp]` | Sortie | Capital avant->après | Gross | Frais | Net |
|---:|---:|---:|---|---|---:|---:|---:|---:|
| 1 | 4->9 | 5 | `[+.3115,+.0148,-1,-.2422,+.5588]` | agent_close | 20.500000->20.469365 | +0.032309 | 0.062944 | -0.030635 |
| 2 | 5180->5201 | 21 | `[+.1129,-.1870,-.8741,-1,+.5244]` | MaxDuration | 19.605951->19.607401 | +0.061710 | 0.060260 | +0.001450 |
| 3 | 10683->10696 | 13 | `[+.4216,+.7347,-.0450,-.1134,+.4547]` | stop_loss | 12.707071->12.586311 | -0.076905 | 0.043855 | -0.120760 |
| 4 | 15636->15657 | 21 | `[+.8256,+.5010,-.0033,-.3435,+.5278]` | MaxDuration | 12.765522->12.855907 | +0.134663 | 0.044278 | +0.090385 |
| 5 | 21235->21251 | 16 | `[+.3557,+.0744,+.4039,-.3465,+.2716]` | MaxDuration | 15.191791->15.165883 | +0.020731 | 0.046639 | -0.025908 |
| 6 | 27055->27071 | 16 | `[+.0846,-.3283,+.3651,-.4270,+.7065]` | MaxDuration | 16.563911->16.513075 | -0.000029 | 0.050806 | -0.050835 |
| 7 | 35459->35480 | 21 | `[+.5190,-.2861,-.1049,-.4036,+.3050]` | MaxDuration | 16.634995->16.614634 | +0.030725 | 0.051086 | -0.020361 |
| 8 | 44722->44743 | 21 | `[+.0895,-.0763,-.2096,-.1296,+.6039]` | MaxDuration | 15.412835->15.409896 | +0.044426 | 0.047364 | -0.002939 |
| 9 | 53672->53693 | 21 | `[+.4140,+.7395,+.1882,-.5931,+.3897]` | MaxDuration | 14.634678->14.585269 | -0.004529 | 0.044880 | -0.049409 |
| 10 | 63231->63252 | 21 | `[+.4761,+.0094,+.2916,-.8417,+.3084]` | MaxDuration | 14.366760->14.325840 | +0.003153 | 0.044073 | -0.040920 |
| 11 | 71929->71931 | 2 | `[+.9564,+.2207,-.0222,+.0064,+1]` | agent_close | 20.163372->20.087982 | -0.013571 | 0.061820 | -0.075391 |
| 12 | 81403->81408 | 5 | `[+.2314,+.3021,-.1158,-.2899,+.4677]` | stop_loss | 18.403466->18.259762 | -0.087431 | 0.056274 | -0.143704 |
| 13 | 91418->91436 | 18 | `[+.1167,-.1712,-.1629,-.1803,+.3655]` | stop_loss | 17.701755->17.556849 | -0.090792 | 0.054115 | -0.144907 |
| 14 | 106666->106687 | 21 | `[+.1203,+.3596,-.2582,+.0876,+.6709]` | MaxDuration | 18.775781->18.678941 | -0.039328 | 0.057512 | -0.096840 |
| 15 | 164415->164418 | 3 | `[+.0683,+.5943,-1,-.1887,+.5952]` | agent_close | 20.500000->20.403760 | -0.033427 | 0.062812 | -0.096240 |

### Causalité de fermeture

Pour `agent_close`, un SELL accepté est la cause de la fermeture.

Pour `MaxDuration`, `stop_loss` et `take_profit`, l'action policy enregistrée au même step n'est pas nécessairement causale. Le lifecycle marché est évalué avant le routing de l'action du step. Exemple : au step 5201, la policy émet `direction=+0.1873`, mais le CLOSE `MaxDuration` vient de la position préexistante.

### Gates pendant détention

Exemple cycle 2, steps 5180-5201 :

```text
OPEN                         1
buy_while_long              10
decision_budget_or_quota     7
inside_action_deadband       3
MaxDuration CLOSE            1
```

Exemple cycle 14 :

```text
OPEN                         1
decision_budget_or_quota    15
buy_while_long               3
inside_action_deadband       1
MaxDuration CLOSE            1
```

Les SELL demandés pendant détention ont donc souvent été neutralisés par le budget/quota. Cette observation ne suffit pas à prouver que ce mécanisme a initié la dérive directionnelle.

## 8. Chaîne d'action auditée

Chaîne effective :

```text
Observation
-> policy Box(5)
-> action reçue par l'environnement
-> routing state-conditioned
-> risk/EV filters
-> deadband
-> decision budget/quota
-> PortfolioManager
-> lifecycle financier
-> receipt
-> trace CLOSE (défaillante sur certains force-close)
-> reward
```

Routeur canonique vérifié :

```text
FLAT + direction > seuil       -> BUY
FLAT + direction <= seuil      -> HOLD
LONG + direction < -seuil sell -> SELL
LONG sinon                     -> HOLD
```

Ainsi, `flat + direction négative -> HOLD`. Un SELL flat ne doit pas créer de CLOSE. Le comportement terminal `sell_while_flat` est conforme au routeur ; ce n'est pas un bug du PortfolioManager.

Exemple terminal vérifié :

```text
164415: direction +0.068286 -> OPEN
164418: direction -1.0      -> AGENT_CLOSE, pnl_net=-0.096239563
164419: direction -1.0      -> sell_while_flat
164420: direction -1.0      -> sell_while_flat
```

## 9. Dérive directionnelle

Fenêtres de 5 000 steps :

| Fenêtre | Moyenne direction | SELL bruts | BUY bruts | OPEN | EV rejects | Acceptation BUY observable |
|---|---:|---:|---:|---:|---:|---:|
| 97 500-102 499 | -0.23469 | 3 226 | 911 | 41 | 798 | 4,9 % |
| 127 500-132 499 | -0.52148 | 4 416 | 213 | 13 | 241 | 5,1 % |
| 147 500-152 499 | -0.70959 | 4 842 | 42 | 6 | 52 | 10,3 % |
| 161 500-166 499 | -0.89568 | 4 988 | 1 | 1 | 2 | non robuste |
| 197 500-202 499 | -0.99183 | 5 000 | 0 | 0 | 0 | sans objet |

Le gate EV filtre fortement les BUY encore proposés vers 100k-130k. Mais la quantité absolue de BUY s'effondre plus vite que leur taux d'acceptation. Après 200k, le gate ne reçoit plus aucun BUY et ne peut donc pas expliquer directement le maintien terminal du collapse.

## 10. Reward et signal d'apprentissage

`ADAN_REWARD_TELEM` n'était pas activé. Il n'existe donc pas de CSV exhaustif des composantes par step.

Le log `TIER_REWARD`, échantillonné toutes les 50 étapes locales, permet néanmoins de mapper exactement 9 954 échantillons sur 64 épisodes et sur un `global_step`.

Moyennes observées :

| Fenêtre | Reward moyen | Part de zéros |
|---|---:|---:|
| 100k-109999 | -0.87990 | non calculée |
| 130k-139999 | -0.25230 | non calculée |
| 150k-159999 | -0.02368 | non calculée |
| 160k-169999 | -0.00200 | 87,0 % |
| 180k-189999 | -0.00079 | 95,5 % |

Autour de 100k, le capital est proche de 17,08-17,42, le drawdown penalty est approximativement compris entre -2,25 et -2,78 et le reward final entre -1,19 et -1,33.

Autour de 164k, avant le dernier trade, le capital revient à 20,50 et le reward est le plus souvent exactement nul. Autour de 200k, capital, PnL instantané et drawdown sont stables et le reward est presque toujours nul.

### Composantes reconstructibles

Pour chaque CLOSE, la base PnL est calculable exactement :

```python
pnl_base_reward = pnl_net * 100 / 20.50 * 0.5
```

La finance, les frais et cette base sont donc récupérables.

Le `closure_bonus` n'est récupérable que lorsqu'il apparaît dans un échantillon `TIER_REWARD` correspondant. Les autres composantes ne sont pas séparables de façon exhaustive pour chaque cycle sans `ADAN_REWARD_TELEM` : Future Arena, latent PnL, symmetry, action entropy, saturation, close intention et éventuelles interactions de clipping/symlog. Elles doivent rester marquées comme inconnues plutôt que déduites artificiellement.

Mécanismes opt-in absents du launcher V25 : anchor lambda, time decay, smart flat, MTM reward et holding cost. Le helper `_sterile_penalty_v5()` est défini mais aucune invocation active n'a été trouvée. Les rejets EV utilisent `_inv_pen_weight=0.0` et sont donc neutres via `_step_invalid_penalty`.

## 11. Future Arena

Le watchdog terminal rapporte :

```text
future_share=59.5%
target<40%
status=CRITICAL
mean_abs_future=0.0010
mean_abs_pnl=0.0007
```

Preuve : sur les contributions absolues accumulées, Future Arena dépasse la limite de dominance prévue.

Limite de preuve : ce ratio ne démontre pas à lui seul que Future Arena a initié la dérive négative de la tête direction.

## 12. Verdict causal hiérarchisé

### Prouvé

1. V25 finit dans un régime flat-SELL sans trades.
2. Le routeur transforme correctement les SELL flat en HOLD.
3. Le PortfolioManager traite correctement les SELL acceptés lorsque LONG.
4. Les pertes et le drawdown créent un signal négatif dense avant 130k.
5. Le nombre de BUY proposés par la policy s'effondre jusqu'à zéro.
6. Le reward terminal devient presque entièrement nul.
7. PPO continue ses updates dans cette région plate.
8. Future Arena dépasse sa limite de dominance absolue.
9. Neuf fermetures financières ne sont pas tracées à cause d'une rupture receipt->trace.

### Fortement corrélé

1. La dégradation financière et le drawdown précèdent le collapse directionnel complet.
2. La faible exécutabilité des BUY, notamment via le gate EV, réduit fortement les trajectoires d'apprentissage associées aux BUY.
3. Les métriques PPO terminales dégradées coïncident avec la région plate et saturée.

### Plausible mais non isolé

1. Une interaction entre reward financier négatif, gate EV, composants de shaping actifs et partage actor/critic peut avoir poussé la policy vers l'attracteur SELL-flat neutre.
2. La dominance Future Arena peut perturber le crédit d'action, mais son rôle causal exact n'est pas isolé.
3. Le budget/quota peut retarder des sorties voulues pendant détention, sans preuve qu'il initie la dérive.

### Non démontrable avec les artefacts V25

1. Le signe exact des avantages PPO par type d'action.
2. Le composant précis qui initie le drift.
3. La part causale individuelle de Future Arena, du gate EV, du critic, du bootstrap ou de l'architecture partagée.
4. Le breakdown reward complet de chacun des 500 220 steps.

Conclusion causale : les preuves autorisent les corrections lifecycle, comptage et télémétrie. Elles n'autorisent pas encore une modification PPO, reward ou routing.

## 13. Gates avant toute nouvelle exécution longue

Avant un nouveau 500k :

1. tests lifecycle ciblés verts ;
2. smoke 2048 avec `opens == closes` ;
3. `unclosed_positions == 0` ;
4. validateur lifecycle `ok=true` ;
5. réconciliation entre résumé, trace et somme des `pnl_net` ;
6. backtests des checkpoints disponibles avec liquidation terminale ;
7. paper/live toujours bloqués pour V25 ;
8. aucune modification PPO/reward/routing sans nouvelle preuve instrumentée.
