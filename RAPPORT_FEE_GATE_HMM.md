# Rapport — le bloqueur réel de Gate C : `fee_gate` alimenté par un HMM quasi one-hot

**Statut : RETRACTATION PARTIELLE de `c2ca902`.**
Ce document est versionné volontairement (et non déposé sous `logs/`, qui est
gitignoré en `.gitignore:29` et a déjà été purgé en masse une fois) parce qu'il
corrige une conclusion fausse déjà présente dans l'historique git.

---

## 0. Ce que `c2ca902` affirmait, et pourquoi c'est faux

> « L'EV fee gate est fermé par construction : `sl_hi=0.0235 > tp_lo=0.0135`
> donne `p_min` 0.60–0.74 > `p_hmm≈0.5`. »

Raisonnement mené sur les **bornes hautes uniquement**. En relisant
`multi_asset_chunked_env.py` L9320-9345, l'agent ne reçoit pas une paire fixe
mais une **boîte** :

```
sl_pct ∈ [sl_lo = 0.003 , sl_hi = 0.0235]
tp_pct ∈ [tp_lo = 0.0135, tp_hi = 0.0222]
```

En bas de bande, `sl=0.003 / tp=0.0222` ⇒ `p_min = 0.0070/0.0252 = 0.278`,
largement franchissable. **Le gate n'est donc PAS fermé par construction.**

---

## 1. Chaîne causale mesurée (et non déduite)

```
HMM quasi one-hot  (bear p50 = 1.000, bull p50 = 0.000)
      ↓
p_hmm = clip(context_vector[3], 0.01, 0.99)   → planché à 0.01 dans 95.7% des cas
      ↓
EV fee gate : bloque si p_hmm <= p_min_required
      ↓
201 / 222 = 90.5 % de TOUTE l'intention BUY détruite
      ↓
HOLD exécuté 96.0 %   vs   HOLD demandé 54.8 %
      ↓
Gate C FAIL (seuil <= 0.80)   +   Gate B action_diff 0.354 FAIL (seuil < 0.05)
      ↓
overall_verdict = NO_GO , launch_authorized = false
```

## 2. Mesures — `scripts/diag_fee_gate_measured.py`

Sonde qui enveloppe `resolve_ev_fee_gate` pour enregistrer ses **entrées
réelles**. 500 steps, seed 330500, env réel, policy uniform-random.
210 invocations du gate : 201 bloquées / 9 passées.

| Terme | p10 | p50 | p90 | Lecture |
|---|---|---|---|---|
| `p_min_required` | 0.368 | **0.470** | 0.551 | sain, cohérent avec la boîte SL/TP — **pas la cause** |
| `p_hmm` | 0.010 | **0.010** | 0.333 | planché au minimum — **la cause** |
| marge `p_hmm − p_min` | −0.527 | −0.448 | −0.156 | déficit structurel de signal |

- part `p_hmm < 0.5` : **0.957**
- part `p_min > 0.5` : 0.286
- `dominant_term` : `H-A_signal_p_hmm_too_low`

**L'économie n'a jamais été la contrainte active. C'est le signal.**

## 3. Ventilation — `scripts/diag_routing_ventilation.py`

500 steps, comptabilité exacte 500/500 :

| Compteur | Valeur |
|---|---|
| demandé BUY / SELL / HOLD | 222 / 225 / 53 |
| **`fee_gate`** | **201** (40.2 % de tous les steps) |
| `routing_reject_sell_while_flat` | 217 |
| `routing_reject_deadband` | 39 |
| `routing_reject_buy_while_long` | 17 |
| `trade_executed` | 18 |

Les 217 SELL-while-FLAT sont un artefact d'état de position propre à une sonde
aléatoire ; `fee_gate` est ce qui supprime la seule action déployant du capital.

## 4. Origine de `p_hmm` — `scripts/diag_context_vector_semantics.py`

300 observations. Schéma confirmé dans `state_builder.build_context_vector` :
`[3]=bull`, `[4]=sideways`, `[5]=bear` (défaut uniforme 0.33/0.33/0.34).

| index | mean | p50 | max | valeurs distinctes |
|---|---|---|---|---|
| `ctx[3]` bull | 0.068 | **0.000** | 1.0 | 22 |
| `ctx[4]` sideways | 0.032 | 0.000 | 0.333 | 2 |
| `ctx[5]` bear | 0.900 | **1.000** | 1.0 | 22 |

**Σ des moyennes = 1.0000** ⇒ ce sont de vraies postérieures, correctement
indexées. L'index est bon, le vecteur est bien formé. Le gate se comporte
**correctement** compte tenu de son entrée.

## 5. Réserve explicite — dégénérescence vs fenêtre non représentative

Un HMM à forte persistance de régime produit *normalement* des postérieures
proches de one-hot. Ce n'est pas en soi une panne. Or les fenêtres du probe
sont courtes (1 chunk : ~7991 lignes 5m, 912 lignes 1h, 521 lignes 4h) et
peuvent être réellement à 90 % dans un seul régime.

**Deux hypothèses restent donc ouvertes, et le rapport ne les tranche pas :**

- **H-α** le HMM est mal entraîné / mal calibré → postérieures non informatives
- **H-β** le HMM décrit correctement une fenêtre localement baissière, non
  représentative → c'est le protocole de mesure qu'il faut élargir

Discriminant requis avant toute action sur le HMM : même distribution
`ctx[3,5]` sur une fenêtre nettement plus longue (multi-chunks), et sur DOGE.

## 6. Corollaire — pourquoi DOGE plafonnait aussi

DOGE montrait 94.8 % d'intention brute BUY-ish sans jamais faire croître son
capital. `fee_gate` dépend des **données de marché** (via le HMM), pas de la
policy : le mécanisme est donc *partagé entre actifs*. Une forte volonté
d'achat côté DOGE serait annulée par le même gate. À tester en rejouant
`diag_fee_gate_measured.py` sur les données DOGE.

## 7. Ce qu'il ne faut PAS faire

`ADAN_DISABLE_EV_FEE_GATE=1` **n'est pas le correctif**. Selon
`resolve_ev_fee_gate()`, désactiver ne fait que rétrograder le blocage en
télémétrie `disabled_advisory` : on laisserait alors passer des trades à EV
négative. Gate B/C remonteraient sans qu'aucune décision économique ne se soit
améliorée — exactement la métrique artificiellement embellie à éviter.

## 8. Portée du diagnostic — limite assumée

Ce rapport démontre : *HMM quasi one-hot → `p_hmm` planché → `fee_gate` bloque
le BUY → Gate B/C FAIL*. Il ne démontre **pas** encore que le HMM est la cause
de l'échec d'apprentissage PPO. Cette dernière chaîne exige le Step Causal
Recorder + le radar PPO :

```
observation → PPO → BUY demandé → fee_gate → HOLD exécuté
  → transition économique → reward → advantage/return → gradient → policy t+1
```

De même, le correctif `terminated`/`truncated` (`f38b8c2`) corrige une
sémantique **démontrablement** fausse, mais la causalité complète avec
l'`explained_variance` négative reste à mesurer **après** correction.

## 9. Décision

**500k reste bloqué**, sur base mesurée. Le bloc « GO confirmé / Gate B 3.5 % /
Gate C 60.1 % » est périmé : le log canonique sur disque
(`logs/validation/gate_c_run_20260904_225928.log`) dit `NO_GO`,
`launch_authorized=false`, B 0.354 FAIL, C 0.960 FAIL, A INCONCLUSIVE.

Ordre : discriminer H-α/H-β → corriger en amont du gate → remesurer `p_hmm`
avant/après → rejouer Gate C canonique → exiger B PASS **et** C PASS →
constellation causale → smoke PPO → décision 500k.

---

## Note de méthode

Deux fois dans cette session, une conclusion tirée de la lecture des bornes du
code a été renversée par la mesure des valeurs à l'exécution. **Mesurer coûte
moins cher que discuter** — et une conclusion fausse poussée dans git est plus
coûteuse que pas de conclusion du tout.

---

# ADDENDUM — H-β tranché, et la cause racine se déplace vers les DONNÉES

## A. Discriminant H-α / H-β : verdict **H-β**

`scripts/diag_hmm_regime_discriminant.py` — postérieures HMM confrontées au
mouvement réalisé de chaque fenêtre (200 steps/fenêtre, seed 330500) :

| split | retour réalisé | part bear > 0.9 | `p_hmm` p50 | part `p_hmm` au plancher |
|---|---|---|---|---|
| train | **−17.14 %** | 0.795 | 0.010 | 0.780 |
| val | **+1.87 %** | 0.435 | **0.333** | 0.435 |
| test | +0.49 % | 0.620 | 0.010 | 0.605 |

`bear_share_on_falling = 0.795` vs `bear_share_on_rising = 0.5275`.

**Le HMM n'est pas dégénéré : il suit la réalité.** Sur la fenêtre haussière
(val) il relâche `p_hmm` à 0.333, sur la fenêtre à −17 % il dit bear. H-α est
**infirmée**.

### Conséquence — renversement d'interprétation

`fee_gate` refuse d'acheter un marché qui baisse de 17 %. C'est **correct**.
Les 96 % de HOLD exécuté ne sont donc pas un bug de routage : c'est la réponse
**rationnelle** à la fenêtre de données fournie. Gate C mesurait un
comportement sain sur un univers malade.

## B. Cause racine réelle : l'univers d'entraînement exposé

`scripts/diag_train_universe_bias.py`, lu directement sur les parquets :

| dataset | lignes 5m | jours | retour total | déciles ↑/↓ | couverture d'1 chunk |
|---|---|---|---|---|---|
| `BTCUSDT` ← **utilisé** | 11 417 | 39.6 | **−15.21 %** | 4/5 | **0.700** |
| `BTCUSDT_binance` | **946 633** | **3 286.9** | **+1 708.64 %** | 7/3 | 0.008 |
| `DOGEUSDT` ← **utilisé** | 25 000 | 86.8 | −7.34 % | 3/6 | 0.320 |
| `DOGEUSDT_binance` | **749 774** | **2 603.4** | **+2 012.18 %** | 4/6 | 0.011 |

`config/config.yaml:252` déclare `assets: [BTCUSDT]` — **jamais**
`BTCUSDT_binance`. L'env rapporte `current_chunk: 1/1`.

**L'agent est entraîné sur 39,6 jours de BTC en baisse de 15 %, alors que
3 287 jours en hausse de 1 708 % sont présents sur le disque.** Un seul chunk
couvre 70 % de ce mini-dataset : il n'y a quasiment aucune diversité de régime
à apprendre.

Cela réconcilie enfin toutes les observations ouvertes :

- capital qui ne dépasse jamais durablement son point de départ (BTC **et**
  DOGE) → les deux mini-datasets sont baissiers (−15.2 %, −7.3 %)
- plancher identique 12,30 / retour exact à 20,50 sur deux actifs → même
  géométrie de dataset court et monotone
- `explained_variance` négative dès le premier update → une seule fenêtre
  quasi déterministe, valeur d'état non généralisable
- 94,8 % d'intention BUY côté DOGE sans croissance → `fee_gate` annule
  correctement l'achat dans un marché baissier
- V29/V30/V33 passant les gates synthétiques puis échouant → les gates
  sondaient la mécanique, jamais l'univers de données

## C. Ce que cela invalide

Aucun correctif sur le reward, le reset, PPO, le cooldown, `tp_lo`, la gSDE ou
le learning rate ne peut produire de croissance de capital sur un univers
baissier de 39 jours. Les six « fix » cumulés (gSDE v30, lr v31,
capacity_reward v33, inaction_penalty v33, cooldown v34, tp_lo v34) opéraient
tous en aval d'une contrainte de données jamais mesurée.

**Un 500k sur `BTCUSDT` brûlerait le budget sur 39,6 jours de baisse.**

## D. Prochaine action unique

Basculer la config sur les datasets complets (`BTCUSDT_binance` /
`DOGEUSDT_binance`) et vérifier le chunking (`1/1` → multi-chunks), puis
**remesurer `p_hmm` et rejouer Gate C canonique**. Tout le reste attend.

---

# TROISIÈME AUTO-CORRECTION — QUALIFICATION DE `a7f517f`

`a7f517f` a été poussé avec une mesure faite sur les **mauvais fichiers**, et
sa formulation (« l'agent s'entraîne sur 39 jours de BTC baissier ») confondait
*ce que mes sondes ont vu* avec *ce que les runs réels ont vu*. Les deux ne sont
pas la même chose. Correction ci-dessous, mesurée.

## E. L'erreur de chemin

`scripts/diag_train_universe_bias.py` v1 lisait :

    data/processed/<asset>/<asset>_5m_featured.parquet

Or `src/adan_trading_bot/data_processing/data_loader.py` L256-273 résout :

    config.data_dirs[<split>] / <ASSET_VARIANT> / <tf>.parquet
    → data/processed/indicators/<split>/<ASSET>/5m.parquet

Le script mesurait donc des parquets que le pipeline ne charge jamais.
`data/processed/indicators/train/` contient quatre variantes :
`BTCUSDT`, `BTCUSDT_BINANCE`, `DOGEUSDT`, `DOGEUSDT_BINANCE`.

## F. Mesure sur les chemins réels du loader

Source : `logs/validation/train_universe_bias_20260904_235734.json`
(script repointé, marqueur `ADAN0_LOADER_PATHS`).

| split/asset | rows 5m | jours | retour % | déciles ↑ | déciles ↓ | 1 chunk = |
|---|---|---|---|---|---|---|
| train/BTCUSDT | 7 991 | 27,7 | **−17,14** | 5 | 5 | **100,00 %** |
| train/BTCUSDT_BINANCE | 662 643 | 2 300,8 | **+928,90** | 7 | 3 | 1,21 % |
| train/DOGEUSDT | 17 500 | 60,8 | −29,17 | 2 | 8 | 45,66 % |
| train/DOGEUSDT_BINANCE | 524 841 | 1 822,4 | **+2 774,48** | 6 | 4 | 1,52 % |
| val/BTCUSDT | 1 143 | 4,0 | +1,87 | 5 | 4 | 699,13 % |
| val/BTCUSDT_BINANCE | 141 994 | 493,0 | +92,54 | 4 | 6 | 5,63 % |
| test/BTCUSDT | 2 283 | 7,9 | +0,49 | 4 | 5 | 350,02 % |
| test/BTCUSDT_BINANCE | 141 996 | 493,0 | −8,66 | 4 | 6 | 5,63 % |
| test/DOGEUSDT_BINANCE | 112 467 | 390,5 | −61,96 | 2 | 8 | 7,11 % |

Verdict machine :
`H1_parquet_is_large_and_two_sided_but_env_exposes_one_chunk`.

## G. Le fait qui change l'interprétation

`scripts/launch_asset_run.py` L57 :

```python
ap.add_argument("--asset", required=True,
                choices=["BTCUSDT_BINANCE", "DOGEUSDT_BINANCE"])
```

et `derive_config()` L35-43 réécrit `cfg["data"]["assets"]`,
`cfg["environment"]["assets"]` **et** `wcfg["assets"]` de chaque worker.
Un run réel **ne peut pas** s'exécuter sur `BTCUSDT` : le choix est contraint
au niveau de l'argparse. Le commentaire `_SLTP` cite « the actual 662,603-row
BTC TRAIN parquet », cohérent avec les 662 643 lignes mesurées ici.

Or `logs/validation/gate_c_run_20260904_225928.log` ligne 8 enregistre
`"asset": "BTCUSDT"`, et mes sondes codaient en dur `assets=["BTCUSDT"]`.

**Conclusion corrigée** : ce n'est pas l'entraînement qui était enfermé dans
27,7 jours — ce sont **Gate C canonique et toutes mes sondes de cette session**.
`a7f517f` attribuait au run une pathologie qui est en réalité celle de
l'instrumentation.

## H. Ce que cela requalifie, chiffre par chiffre

Tous les nombres suivants ont été obtenus sur `train/BTCUSDT`
(7 991 lignes, 27,7 jours, −17,14 %, un unique chunk) — donc **hors de
l'univers des runs réels** :

| mesure | valeur | statut après correction |
|---|---|---|
| `p_hmm` p50 | 0,010 (plancher) | non transposable |
| `fee_gate` rejets | 201 / 222 = 90,5 % | non transposable |
| HOLD exécutés | 96 % | non transposable |
| Gate B / Gate C | 0,354 FAIL / 0,960 FAIL | **invalide comme verdict de run** |
| verdict H-bêta | HMM suit le réel | méthode valide, échantillon à refaire |
| `current_chunk: 1/1` | — | **expliqué** : 7 991 lignes = exactement 1 chunk |

Le `1/1` n'était pas un bug de chunking : `train/BTCUSDT` tient dans un seul
chunk (`share_of_history_in_one_chunk = 1.0000`, mesuré). Sur
`train/BTCUSDT_BINANCE` un chunk ne couvre que 1,21 % de l'historique, soit
~82 chunks. Il n'y a donc **aucun** correctif de chunking à écrire.

## I. Statut des hypothèses

- **CONFIRMÉ** — le loader lit `indicators/<split>/<ASSET>/`, pas
  `processed/<asset>/*_featured.parquet`.
- **CONFIRMÉ** — `launch_asset_run.py` interdit `BTCUSDT` par argparse et
  réécrit les trois clés d'actifs.
- **CONFIRMÉ** — Gate C canonique a tourné sur `BTCUSDT` (log ligne 8).
- **CONFIRMÉ** — les gros splits `_BINANCE` sont bilatéraux (7↑/3↓ en train
  BTC), donc l'argument « univers monotone » tombe pour les runs réels.
- **INFIRMÉ** — « l'agent s'entraîne sur 39 jours de BTC baissier »
  (formulation de `a7f517f`). Les runs ne peuvent pas charger ce split.
- **INFIRMÉ** — « le chunking est cassé (1/1) ». Arithmétique de dataset.
- **NON RÉSOLU** — la distribution de `p_hmm` sur `BTCUSDT_BINANCE`.
  Mesure en cours ; c'est elle qui décidera si `fee_gate` est un vrai blocage
  économique ou un artefact de fenêtre.
- **NON RÉSOLU** — Gate B / Gate C sur l'univers réel. Le NO_GO reste en
  vigueur : il n'est pas *réfuté*, il est *non mesuré*.

## J. Ce que cela ne change pas

Le correctif `terminated`/`truncated` (`f38b8c2`) reste valide et indépendant :
il porte sur la sémantique Gymnasium, pas sur les données. Sa causalité sur
`explained_variance` reste toutefois **à mesurer après correction**, sur
`BTCUSDT_BINANCE`.

**500k reste bloqué.** Motif mis à jour : ce n'est plus « univers baissier »,
c'est « aucun gate n'a jamais été mesuré sur l'univers que le run charge ».

---

# CAUSE RACINE DU SIGNAL HMM — MESURÉE, CLASSÉE

Mesures sur `BTCUSDT_BINANCE` / `DOGEUSDT_BINANCE` (l'univers que
`launch_asset_run.py` charge réellement), 300 pas par fenêtre.

## K. La réserve de l'utilisateur était fondée

Sur l'univers réel, la postérieure **n'est pas dégénérée** :

| asset/split | ONE_HOT | NEAR_ONE_HOT | INTERMEDIATE | valeurs `p_hmm` distinctes | Σ p50 |
|---|---|---|---|---|---|
| BTC train | 80,7 % | 6,7 % | 12,7 % | 28 | 1,0 |
| BTC val | 76,7 % | 8,0 % | 15,3 % | 42 | 1,0 |
| DOGE train | 66,0 % | 16,7 % | 17,3 % | 65 | 1,0 |
| DOGE val | 67,7 % | 14,7 % | 17,7 % | 61 | 1,0 |

C'est le profil **normal** d'un HMM à forte persistance : postérieures souvent
saturées, mais graduées (jusqu'à 65 valeurs distinctes) et sommant à 1. Le
« HMM dégénéré » observé plus tôt était un artefact de la fenêtre de 27,7 jours.
`0.333333` apparaît **exactement 29 fois dans chaque fenêtre** — c'est le
warm-up (`_hmm_min_obs = 60`), pas un fallback permanent.

Moteur interrogé sur ses propres compteurs
(`logs/validation/hmm_engine_health_20260905_003032.json`) :
`_hmm_fit_count = 5`, `_hmm_fit_failures = 0`,
`_hmm_last_fallback_reason = None`, features vivantes
(`log_return` 300 valeurs distinctes, `rsi_norm` ∈ [0,26 ; 0,90]).
Classification : **NI entraînement, NI calibration, NI données, NI mapping.**

## L. Mais le compteur a révélé une anomalie dure

`get_regime_probabilities` est appelée **599 fois pour 300 pas**, et
`observation_id` est `None` dans **49,92 %** des appels, avec `log_return`
présent seulement 300 fois sur 599.

Il existe deux appelants :

- **A** `multi_asset_chunked_env.py` L6315 → `_get_current_market_data_for_hmm()`
  → vraies features + `observation_id`.
- **B** `dynamic_behavior_engine.py` L915 (`detect_market_regime`), atteint
  depuis `update_risk_parameters` L986 et depuis env L1357 avec
  `market_conditions` — un dict qui **ne contient aucune** des 4 features HMM.

Sur le chemin B, `get_regime_probabilities` L631+ retombe sur ses défauts
`log_ret=0.0, atr_pct=0.0, rsi_norm=0.5, volume_ratio=1.0`, et
`observation_id=None` **désactive aussi le cache de déduplication** L636-639
(`if observation_id is not None and ...`). L'appel n'est donc jamais
court-circuité, et `_update_hmm` exécute inconditionnellement :

```python
self._hmm_obs_buffer.append([log_return, atr_pct, rsi_norm, volume_ratio])
self._hmm_total_obs += 1
```

## M. Contamination confirmée, chiffrée

`logs/validation/hmm_buffer_contamination_20260905_003847.json` :

| mesure | valeur |
|---|---|
| `buffer_len` | 500 |
| `buffer_synthetic_rows` | **250** |
| `buffer_synthetic_share` | **0,5000** |
| `buffer_distinct_points` | **251** (250 réels + 1 point faux répété) |
| `share_calls_synthetic` | 0,4992 |
| `share_posterior_computed_on_synthetic_last_row` | **0,4992** |
| `engine_total_obs` | 600 pour 300 pas |
| `final_probs` | [0,999906 ; 0,0 ; 0,000094] |

Deux conséquences distinctes, toutes deux mesurées :

1. **Le fit est empoisonné.** La fenêtre glissante de 500 points ne contient que
   251 points distincts : la moitié est un unique point constant répété 250
   fois. Un composant gaussien se verrouille dessus (variance → 0), ce qui
   explique mécaniquement la saturation et `sideways_max = 0.333333` sur trois
   fenêtres sur quatre — l'état « sideways » est capturé par le point mort.
2. **La postérieure retournée décrit parfois une observation fictive.**
   `_update_hmm` renvoie `predict_proba(X)[-1]`, soit la postérieure de la
   **dernière ligne ajoutée**. Dans 49,92 % des appels cette dernière ligne est
   le vecteur synthétique. Une fois sur deux, `p_hmm` ne parle pas du marché.

Classification finale, dans la taxonomie demandée :
**TRANSFORMATION / PLUMBING** — un second appelant sans features partage le
même buffer de fit et le même chemin de retour que le calcul de marché.
Ce n'est ni un problème d'entraînement, ni de calibration, ni de données, ni de
régime réellement observé, ni de mapping bull/bear, ni de normalisation.

## N. Conséquence économique, mesurée sur l'univers réel

Avec `p_min_required` p50 ≈ 0,466 (sain, et `p_min_min` ≈ 0,328) :

| asset/split | invocations | block_rate |
|---|---|---|
| BTC train | 92 | 0,7065 |
| BTC val | 115 | 0,8435 |
| DOGE train | 112 | 0,8125 |
| DOGE val | 84 | 0,6429 |

Le `fee_gate` bloque 64-84 % même sur l'univers réel. `p_min` n'est pas le
terme fautif : `p_hmm` l'est, et une moitié de ses valeurs est calculée sur une
observation synthétique. **Hypothèse de mécanisme partagé DOGE/BTC : CONFIRMÉE**
(même signature sur les deux actifs), ce qui répond au point 2 de l'utilisateur.

## O. Ce que ce résultat impose

Aucun correctif de reward, PPO, cooldown ou SL/TP n'est pertinent tant que
`p_hmm` est calculé une fois sur deux sur un point mort. C'est un défaut de
plomberie à corriger avant toute remesure de Gate B / Gate C, et avant tout
jugement sur `explained_variance`.

**500k reste bloqué.**
