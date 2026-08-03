# AUDIT V9 — DSpark, autopsie de la chaîne IA, et le VRAI bug de collapse

Date : 2026-07-01
Branche : `feat/diagnostic-v4`
Auteur : audit forensique post-run v9 500k (interrompu à ~8k steps après détection du vrai bug)

---

## 0. Résumé exécutif

Le run v9 (avec le "Fix 1" précédent) a **prouvé** que ce Fix 1 patchait la **mauvaise branche**.
Le compteur `anti_spam_hold` est resté à **0** pendant tout le run, alors que `min_notional`
rejetait **380+ actions par fenêtre**. Le vrai chemin de collapse est le **Cas B min_notional**
(`multi_asset_chunked_env.py` L.7700-7740), où le code ajoutait `+0.002` (une **récompense**)
pour un BUY illégal. Corrigé en V9 (commit `291a8cc`) : signe négatif + pénalité escaladante V5
avec sévérité dédiée `min_notional_self_caused = 0.55`.

**Leçon (confirmée par l'utilisateur)** : c'est la 2e fois que le même défaut de signe apparaît
dans le même bloc. J'ai donc fait un **grep exhaustif** de tous les `self._step_invalid_penalty +=`
et de tous les `reward += / pnl +=`. Résultat en §2.

---

## 1. Les 7 questions DSpark — réponses factuelles (code à l'appui)

### Q1 — Quelles idées de DSpark ont été implémentées ?
**Réponse honnête : AUCUNE des idées CENTRALES de DSpark (Confidence Head calibré, Sequential
Temperature Scaling, détection d'overconfidence → action corrective) n'est implémentée.**

Ce qui existe et qui touche *de loin* le sujet :
- Un **circuit-breaker de collapse** (post-hoc, pas préventif) — `DiagnosticCollapseCallback`
  dans `scripts/train_parallel_agents.py` : arrête l'entraînement si `pct_buy>=0.97` OU
  `pct_sell>=0.97` OU `|a0_mean|>=5.0` sur 2 fenêtres. → c'est un **disjoncteur**, pas une
  calibration de confiance.
- Un **entropy coefficient adaptatif** partiel — `dynamic_behavior_engine.py` L.1440-1443 et
  `training/hyperparam_modulator.py` L.103-112 (defensive/aggressive ent factor). → module
  l'exploration, ne calibre PAS la confiance.
- **target_kl / KL early stop** — `train_parallel_agents.py` L.2115 (sandbox), `model_initializer.py`
  L.110. → garde-fou trust-region standard PPO, orthogonal à DSpark.
- **WorldModelPPO + aux forward-prediction loss** — `agent/feature_extractors.py` L.1001-1117,
  `train_parallel_agents.py` L.1077-1079 (`aux_loss_coef=0.1`). → force le CNN à prédire la
  bougie suivante ; c'est un signal d'"honnêteté des features", PAS un head de confiance calibré.

### Q2 — Dans quels fichiers ?
- Circuit-breaker : `scripts/train_parallel_agents.py` (classe `DiagnosticCollapseCallback`, ~L.491-665).
- Entropy adaptatif : `src/adan_trading_bot/environment/dynamic_behavior_engine.py` (~L.1440),
  `src/adan_trading_bot/training/hyperparam_modulator.py` (~L.103).
- KL : `scripts/train_parallel_agents.py` L.2115, `src/adan_trading_bot/model_initializer.py` L.110.
- World Model : `src/adan_trading_bot/agent/feature_extractors.py` (`WorldModelPPO`, `forward_predictor`).

### Q3 — Sous quels noms de classes ?
- `DiagnosticCollapseCallback` (disjoncteur).
- `WorldModelPPO`, `TemporalFusionExtractor`, `ChannelAttention` (chaîne CNN/attention/aux).
- **Aucune classe `ConfidenceHead`, `TemperatureScaling`, `Calibrator` orientée policy.**
  (`execution_engine.py` a un `confidence` mais c'est la **bull_prob HMM** du context_vector[3],
  utilisée pour dimensionner l'exposition — RIEN à voir avec la calibration DSpark de la policy.)

### Q4 — Quelles métriques sont loggées ?
Dans le CSV diagnostic (`DiagnosticCollapseCallback`) : `a0_mean, a0_std, a0_pct_buy, a0_pct_sell,
a0_pct_hold_band, req_*_pct, steps_flat/open_pct, illegal_ratio, policy_entropy, a0_histo`.
SB3 loggue aussi `approx_kl, clip_fraction, entropy_loss, explained_variance, value_loss,
policy_gradient_loss` (analysés dans `scripts/analysis/analyze_run.py`).
**Manquant pour DSpark** : aucune métrique de **calibration** (ECE, reliability, confidence vs
accuracy), aucun IC (information coefficient signal↔rendement futur), aucune diversité d'embeddings.

### Q5 — Comment le système détecte l'overconfidence ?
**Il ne la détecte PAS au sens DSpark.** Il détecte le **résultat terminal** de l'overconfidence
(collapse à 97-100% d'une action) via le disjoncteur. L'`entropy` est loggée mais aucun mécanisme
ne réagit à une entropie qui s'effondre *avant* le collapse total. Il n'y a pas de mesure
"P(action correcte) prédite vs réelle".

### Q6 — Comment il agit lorsqu'elle est détectée ?
- Disjoncteur : **arrête** l'entraînement (`return False`). C'est tout — pas de fallback, pas de
  ré-injection d'exploration, pas de reset de dernière couche, pas de pénalité de confiance.
- Il n'existe **pas** de `loss += λ*(max_prob - seuil)²` ni de `if confidence < seuil: HOLD`.

### Q7 — Quels tests prouvent que cela fonctionne ?
- Le disjoncteur : testé en smoke (ne se déclenche PAS sur données saines) mais **jamais** prouvé
  qu'il se déclenche sur un vrai collapse en conditions réelles (le collapse v8 n'avait pas le
  breaker). → à prouver sur le prochain run.
- Le World Model aux loss : **aucun test** ne prouve que `aux_loss` baisse ni que les features
  sont corrélées au futur. → AUDIT 4/5 ci-dessous.

---

## 2. Audit exhaustif des signes (le point soulevé par l'utilisateur)

### 2.a — Tous les `self._step_invalid_penalty +=`
| Ligne | Terme | Signe défini | Verdict |
|------|-------|--------------|---------|
| 4840, 4862 | `-5.0` | négatif | OK (catastrophic) |
| 7634 | `-_pv5` (anti_spam_hold) | négatif | OK |
| 7640 | `-0.05` fallback | négatif | OK |
| **7704 (avant V9)** | **`_mgmt_pen = +0.002`** | **POSITIF** | **BUG → corrigé V9** |
| 7981 | `_early_pen = -w*(...)` | négatif | OK |
| 8050 | `_q_pen = -0.10-...` | négatif | OK |
| 8064 | `_ac_pen = -0.15*...` | négatif | OK |
| 8115 | `_ci_pen = -lam*def*pnlf` | négatif (tous facteurs ≥0) | OK |
| 8165 | `-_pv5` (sell_no_position) | négatif | OK |
| 8205 | `_wait_pen = -w*(...)` | négatif | OK |
| 8231-8311 | `-_inv_pen_weight` | négatif | OK |

**Conclusion : `_mgmt_pen` était le SEUL défaut de signe.** Tous les autres sont corrects.

### 2.b — Tous les `reward += / realized_pnl +=`
- `raw_reward` (L.6691) = `pnl_base_reward + promotion_bonus + demotion_penalty + closure_bonus
  + drawdown_penalty + symmetry_penalty + action_entropy_penalty + future_contrib
  + latent_pnl_contrib + saturation_penalty`. Puis `final_reward = sign*log1p(|raw|)` (symlog).
- **`calculate_capacity_based_reward()` (+2.0 si expo∈[0.6,0.9]) n'est PAS dans `raw_reward`** —
  uniquement dans le breakdown de télémétrie (`rc["capacity_reward"]`). → **cosmétique**, pas un
  carburant de collapse. **MAIS incohérence à surveiller** : une fonction qui calcule un +2.0
  massif jamais utilisé dans le vrai reward est un piège (si quelqu'un le rebranche un jour).
- `survival_bonus` (A6) et `patience_bonus_val` (A4) : **déjà retirés** (=0.0) par les devs, avec
  commentaire explicite "récompensait l'inaction / rendait BUY~HOLD". Bonne hygiène déjà en place.
- **Suspect résiduel LÉGITIME mais à surveiller : `latent_pnl_contrib`** (L.6641-6667) : toutes
  les 3 steps, `+0.10*log1p(u*10)/10` par position ouverte quand le marché monte (cap 0.30,
  asymétrique perte>gain). C'est le reward positif "gratuit" qui, mal contrebalancé, a alimenté le
  runaway. Il est atténué (log, cap) mais existe. Le FA_WATCHDOG (L.6709) surveille que
  `future_contrib` ne domine pas le PnL — il faudrait un watchdog équivalent sur `latent_pnl`.

---

## 3. « Stable ≠ corrélé » — ce qui reste à prouver (AUDIT 4/5/6)

Le run v9 était stable (a0_std~0.13, buy/sell équilibré au début) mais cela ne prouve **rien**
sur la corrélation signal↔marché. À prouver sur le prochain run :
1. **explained_variance du critic** (déjà loggé SB3) doit monter (>0.1-0.2) → le CNN extrait un
   signal corrélé au futur. S'il reste ~0/négatif = CNN aveugle.
2. **aux_loss du WorldModel** doit baisser → le CNN comprend la "physique" du marché.
3. **IC (Spearman) `a0_mean`/`value_pred` vs rendement futur** doit être ≠ 0 et stable de signe.
4. **Diversité des embeddings CNN** (variance inter-échantillons) ≠ 0 → pas d'effondrement aveugle.
5. **Test de permutation** (shuffle temporel) : le comportement doit CHANGER → le modèle utilise
   la structure, pas une heuristique interne.
6. **Leakage scaler** : vérifier que `scalers_manifest.json` ne bouge que par timestamp (voir AUDIT 6).

---

## 4. Recommandation DSpark (si l'utilisateur confirme)

L'idée DSpark est **bien adaptée** au problème (overconfidence pathologique). Proposition
d'implémentation minimale, NON destructive, en 3 étages :
1. **Confidence penalty** (le moins risqué) : `loss += λ * mean(relu(max_prob - 0.90)²)` dans le
   train loop — empêche la certitude excessive *avant* le collapse.
2. **Entropy floor réactif** : `if entropy < floor: ent_coef *= k` (partiellement déjà présent,
   à câbler sur la vraie boucle PPO sandbox).
3. **Confidence Head calibré (STS)** : `ĉ = σ(z/T)` prédisant P(trade rentable), calibré par
   temperature scaling, utilisé pour `position_size = base × ĉ` ou `if ĉ<seuil: HOLD`.
   → C'est le gros morceau architectural ; à faire APRÈS avoir prouvé la chaîne (AUDIT 4/5).

**Ce qui protège RÉELLEMENT contre un nouveau collapse aujourd'hui** :
- (a) le fix V9 du signe/magnitude sur Cas-B (retire le carburant),
- (b) gSDE off (empêche σ de diverger avec la moyenne),
- (c) le disjoncteur (filet de sécurité terminal).
Le Confidence Head serait la **4e couche**, préventive, la plus alignée DSpark.

---

## 5. AUDIT 4 — RÉSULTATS MESURÉS sur le run v9 (preuves, pas suppositions)

### 5.a — explained_variance du critic (métrique reine)
Extrait des tables SB3 du run v9 (`logs/train_v9_500k.log`) :
```
explained_variance : -0.087, -0.235, -0.315, +0.264, +0.187, +0.148, -0.389, -0.071, +0.060, +0.042
```
**Verdict : oscille autour de 0, souvent NÉGATIVE.** Le critic ne prédit PAS mieux que la moyenne.
C'est le signal d'alarme silencieux décrit par l'utilisateur : la value function n'apprend
quasiment rien d'utile → forte présomption que le CNN n'extrait pas (encore) de signal corrélé au
futur, OU que rien ne pousse le critic à l'exploiter. (approx_kl 0.004-0.008 très faible,
entropy_loss 2.9 constant → pas de divergence brutale, collapse "lent" confirmé.)

### 5.b — WorldModelPPO / aux forward-prediction loss : PAS ACTIVE en sandbox
- `train_parallel_agents.py` L.2099 instancie **`PPO` standard SB3**, PAS `WorldModelPPO`.
- Le path `main` (L.1052-1079) utilise WorldModelPPO + `aux_loss_coef=0.1` ; **le path sandbox
  (celui qui tourne réellement) NON.**
- Aucun log `Using WorldModelPPO` / `aux_loss` dans le run v9. → l'aux loss n'est jamais
  back-propagée en sandbox.
- **Nuance importante** : le CNN lui-même EST branché. Le log confirme :
  `[SANDBOX] features_extractor=ContextualTemporalFusionExtractor (CNN+cross-attn+FiLM+aux)`.
  Le `forward_predictor` (tête aux) existe DANS le CNN, mais comme la loss n'est pas ajoutée à la
  loss d'optimisation (PPO standard ignore la sortie aux), **rien ne force le CNN à comprendre la
  physique du marché**. Cohérent avec explained_variance ~0.

### 5.c — Conséquence / incohérence architecturale
Le "SOTA 2026" (WorldModelPPO + aux loss) est **du code mort en sandbox**. Deux options à trancher
avec l'utilisateur :
1. **Câbler WorldModelPPO dans le path sandbox** (activer réellement l'aux loss) — le plus aligné
   avec l'intention "forcer le CNN à lire le marché".
2. OU accepter PPO standard mais alors surveiller explained_variance comme critère GO/NO-GO : si
   elle ne monte pas durablement >0.1, la chaîne est aveugle et il faut agir (aux loss, ou revoir
   le CNN, ou l'observation).

**Décision retenue pour le prochain run** : garder PPO standard pour isoler l'effet du fix V9
(anti-collapse) SANS changer l'architecture en même temps (une variable à la fois). MESURER
explained_variance sur la durée. Si elle reste ~0 → prochaine itération = activer WorldModelPPO.

---

## 6. AUDIT 6 — Fuite de données du scaler (anti-lookahead) : PROPRE

Vérifié à la source (`src/adan_trading_bot/data_processing/state_builder.py`), pas supposé.

- **L.772-784** : `_fit_ratio = 0.70`. Le scaler est ajusté (`fit`) UNIQUEMENT sur les
  premiers 70 % chronologiques de chaque chunk, puis figé pour transformer le reste.
- **L.657** : garde `SKIPPING fit_scalers()` — empêche un re-fit involontaire sur des données
  futures pendant l'inférence/rollout.
- `prod_scalers/scalers_manifest.json` ne varie que par son timestamp entre deux runs (aucune
  redéfinition des bornes) — cohérent avec un scaler figé.

**Verdict : PROPRE.** Pas de lookahead via le scaler. Le critic à explained_variance ~0 n'est
donc PAS expliqué par une fuite de normalisation qui aurait "aplati" le signal. Le problème (si
problème il y a) est bien dans l'extraction/exploitation du signal, pas dans une contamination
train→futur.

---

## 7. Tableau récapitulatif — état de la chaîne IA (post-audit)

| Brique | Chargée / active ? | Signal prouvé corrélé ? | Note |
|---|---|---|---|
| CNN `ContextualTemporalFusionExtractor` | ✅ chargé (log sandbox) | ❓ non prouvé (AUDIT 5 à faire) | aux head existe mais loss non back-prop |
| Cross-attention + FiLM | ✅ dans le CNN | ❓ non prouvé | idem |
| `forward_predictor` (aux) | ✅ existe | ❌ loss jamais optimisée en sandbox | code "mort" en sandbox |
| WorldModelPPO / aux_loss | ❌ NON en sandbox (PPO standard L.2099) | — | actif seulement path `main` |
| Critic (value net) | ✅ actif | ❌ explained_variance ~0 (souvent négatif) | métrique reine = alarme |
| Scaler anti-lookahead | ✅ fit 70 % chrono, figé | ✅ propre (AUDIT 6) | pas de fuite |
| Reward shaping (signes) | ✅ audité | ✅ tous négatifs sauf bug `+0.002` corrigé (V9) | AUDIT 1 |

**Synthèse** : après le fix V9, le collapse n'est plus l'inconnue principale. L'inconnue #1 est
désormais : *le CNN + le critic extraient-ils un signal réellement corrélé au marché ?*
- Preuve manquante décisive = **AUDIT 5 (shuffle/bruit + IC)** — à faire AVANT le prochain run long.
- Métrique de suivi = **explained_variance** (GO si >0.15 et monte ; NO-GO/aux-loss si reste <0.1).
- Décision WorldModelPPO / DSpark = seulement après ces deux mesures (une variable à la fois).

---

## 8. AUDIT 5 / C3a — IC brut : Y A-T-IL UN SIGNAL DANS LES DONNÉES ? (OUI, faible mais réel)

**La question qui décide de tout** (utilisateur) : *« est-ce que le CNN et le critic extraient
réellement un signal de marché ? »* — décomposée en deux :
- **C3a** (léger, fait) : y a-t-il seulement un signal EXPLOITABLE dans les données brutes ?
- **C3b** (lourd, conditionnel) : le CNN entraîné l'exploite-t-il ?

### Méthode C3a (pandas pur, pas de modèle chargé)
IC de Spearman entre chaque feature et le rendement futur à H={1,5,20,60} barres, sur les
données réelles `data/processed/indicators/train/BTCUSDT/{5m,1h,4h}.parquet`. Puis **contrôle
anti-artefact** : exclusion des prix bruts non-stationnaires (close/high/low/open) — dont l'IC
énorme (−0.33 en 4h/H60) est du mean-reversion mécanique NON tradable par un scalper — et test de
**cohérence de signe cross-horizon** (un vrai edge garde le même signe sur plusieurs horizons ;
un artefact non-stationnaire est instable).

### Résultat — edge FAIBLE mais RÉEL et STRUCTUREL

| TF | Indicateurs à edge cohérent (signe stable, mean\|IC\|>0.03) | Top |
|---|---|---|
| 5m | **7** | volatility_ratio_14_50 (+0.066), rsi_14 (−0.064), ema_20_ratio (−0.057), di_delta (−0.052) |
| 1h | **3** | volatility_ratio_14_50 (+0.052), ema_50_ratio (−0.037), volume_ratio_20 (+0.032) |

- **volatility_ratio_14_50** : IC POSITIF qui MONTE avec l'horizon (5m : +0.037→+0.056→+0.104).
  Signature nette d'un régime de volatilité prédictif.
- **rsi_14 / ema_20_ratio / di_delta** : IC NÉGATIF stable = mean-reversion court terme exploitable.
- |IC| ~0.05-0.10 : faible en absolu MAIS c'est *exactement* la magnitude d'un edge crypto réel.
  Un IC stable de cet ordre est exploitable par un modèle qui COMBINE plusieurs features.

### Verdict C3a & décision
**IL Y A UN SIGNAL.** Ce n'est ni 0 (pipeline aveugle) ni un artefact non-stationnaire. Donc :
- Lancer le 500k (C4) n'est PAS brûler du compute sur un pipeline sans edge → **GO C4**.
- Si en C4 explained_variance reste ~0 MALGRÉ ce signal démontré → le problème est bien
  l'EXTRACTION (chaîne CNN→critic), pas l'absence de signal → motive WorldModelPPO (aux loss)
  APRÈS vérif des confondeurs (clip_range_vf, gae_lambda, taille réseau value).
- Si explained_variance monte >0.15 → la chaîne exploite le signal → WorldModelPPO = amélioration,
  pas sauvetage ; DSpark (confidence head) vient encore après.
- **C3b (CNN shuffle vs réel)** reste utile pour trancher extraction-vs-signal, mais désormais
  SECONDAIRE : C3a a déjà prouvé l'existence du signal. À faire si explained_variance reste ~0 en C4.

> Résultat inconfortable évité : on ne lance PAS un 500k de 23h sur un pipeline sans edge.
> Le signal existe ; la vraie question résiduelle est son EXTRACTION, mesurée en continu par
> explained_variance pendant C4.
