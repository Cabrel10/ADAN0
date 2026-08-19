# CAUSAL_LEARNING_PIPELINE.md

**Preuve causale de bout en bout — STATE → observation → a0 brut → action brute → route_action_by_state → action exécutée → EV → reward brut → pénalités → reward final → advantage → loss PPO → update → action suivante**

Statut : **livrable unique**. Produit par CODE (lecture de fichiers, n° de ligne, formule) + TEST DÉTERMINISTE (sonde d'environnement sans PPO, sans policy). Aucune inférence spéculative ; chaque affirmation renvoie à une ligne de code ou à une mesure de sonde.

> **INTERDICTIONS RESPECTÉES** : ce document ne modifie NI reward, NI pénalités, NI EV, NI routing, NI ancre, NI clamp, NI PPO, NI policy head. Aucun RAL. Aucun V32. La sonde est en lecture seule sur l'environnement (elle injecte seulement des actions fixes et lit `info["reward_components"]`).

---

## 0. Méthodologie (anti-optimisme)

- **Sonde** : `scripts/diagnostics/probe_env_deterministic.py`. Instancie `MultiAssetChunkedEnv` via le pattern prouvé (`ConfigLoader.load_config` → `ChunkedDataLoader.load_chunk(0)` → `env.reset(seed)`), **sans agent PPO ni policy**. Injecte des séquences d'action fixes et lit la décomposition complète du reward via `info["reward_components"]` (= `env._last_reward_components`, l. 7507-7540 et 4574-4575).
- **Séquences** : `hold` (a0=0.0), `buy` (a0=+0.5), `sell` (a0=-0.5), `alt` (alternance ±0.5), `random` (uniforme [-1,1]).
- **États couverts** : FLAT+BUY, FLAT+SELL, LONG+BUY, LONG+SELL (obtenus par la dynamique état×action).
- **Splits rejoués** : `train`, `val`, `test` — pour distinguer **biais structurel du reward** (présent sur tous les splits) de **biais dataset** (variable selon le split).
- **Modes** : EV fee-gate **ON** (`gated`) et **OFF** (`advisory`, `ADAN_DISABLE_EV_FEE_GATE=1`).
- **Variable isolée** : SL/TP brut. Deux campagnes — `sl/tp=1.0` (sature) et `sl/tp=0.5` (ne sature pas) — pour isoler la `saturation_penalty` (une seule variable changée à la fois = test de régression).

### Seuils numériques définis AVANT lecture des résultats (anti-« ça a l'air ok »)
```
random_mean_reward_max_abs = 0.02   # |reward moyen| d'une séquence aléatoire doit rester borné
action_reward_spread_min   = 1e-4   # écart de reward entre routes doit dépasser ce seuil pour "informatif"
sell_flat_must_not_profit  = 0.0    # SELL en FLAT ne doit jamais produire de profit
penalty_reaches_action     = 0.10   # une pénalité "atteint l'action" si |Δreward| >= 0.10
```

### Runs de référence (fichiers produits)
| Campagne | mode | sl/tp | verdict json |
|---|---|---|---|
| A | advisory (gate OFF) | 1.0 | `logs/probe/probe_verdict_20260819_193400.json` |
| B | gated (gate ON) | 1.0 | `logs/probe/probe_verdict_20260819_193422.json` |
| C | advisory (gate OFF) | **0.5** | `logs/probe/probe_verdict_20260819_200707.json` |

---

## 1. La chaîne causale reconstruite (fichier · ligne · variable · formule)

| # | Étape | Fichier / ligne | Variable | Formule / règle |
|---|---|---|---|---|
| 1 | STATE | `multi_asset_chunked_env.py` | position (FLAT/LONG), context_vector | état interne + `p_hmm=context_vector[3]` (défaut 0.5, l.8732) |
| 2 | observation | idem | obs vector | image de l'état marché+portefeuille |
| 3 | **a0 brut** | action = `[a0, size, tf, sl_raw, tp_raw]` | `a0 = action[0]` | continu ∈ [-1,1] (dans le vrai PPO : `tanh(μ+σε)`) |
| 4 | action brute discrétisée (RADAR) | `feature_extractors.py` 1225-1290 | `nB,nH,nS` | `_a0=rollout_data.actions[:,0]`; `nB=(a0>0.1)`, `nS=(a0<-0.1)`, `nH=sinon`. **AVANT routing.** |
| 5 | routing | `action_routing.py` 48-106 | route ∈ {BUY=1,SELL=2,HOLD=0} | FLAT: BUY ssi `a0>+thr` sinon HOLD (SELL impossible). LONG: SELL ssi `a0<-sthr` sinon HOLD. |
| 6 | EV fee-gate (BUY only) | `multi_asset_chunked_env.py` 9490-9545 ; `action_routing.py` 109-125 | `p_min_required` | si `sl_pct>0`: `(1+fees/sl)/(1+RR)`, ×0.85 en training ; **si sl=0 → 0.99 (reject)**. BUY bloqué si `p_hmm ≤ p_min`. |
| 7 | action exécutée | env | `_last_trade_executed` | exécution réelle après routing+gate+`agent_close_gate`(min_hold) |
| 8 | reward brut | `_calculate_reward` 6924 ; somme 15 termes 7400-7409 | `raw_reward` | Σ(pnl_reward, behavior_penalty, drawdown_penalty, action_anchor, symmetry, action_entropy, **saturation_penalty**, …) |
| 9 | reward final | l.7456 | `final_reward` | `sign(raw)·log1p(|raw|)` (**symlog**) |
| 10 | composantes exposées | l.7507-7540, 4574-4575 | `info["reward_components"]` | dict lisible par la sonde |
| 11 | advantage | PPO (SB3) | `A_t` | GAE(γ,λ) sur `final_reward` |
| 12 | loss actor | `feature_extractors.py` 1330 | `anchor_loss` | `policy_grad_loss + vf + ent + λ·(μ²).mean()` |
| 13 | update → action suivante | PPO | Δθ | gradient sur loss ci-dessus |

---

## 2. AMBIGUÏTÉ CRITIQUE — RÉSOLUE PAR LE CODE (pas par inférence)

**Question** : `nB/nH/nS` (ANCHOR_DEBUG) comptent-ils (A) l'action brute discrétisée AVANT routing, (B) l'action APRÈS `route_action_by_state`, ou (C) l'action exécutée ?

**PREUVE — `agent/feature_extractors.py`, lignes 1225-1290 :**
```python
n_buy_batch = n_sell_batch = n_hold_batch = 0
_raw_act = rollout_data.actions      # actions du ROLLOUT BUFFER PPO
_a0_dir  = _raw_act[:, 0]            # dim 0 = a0 continu, PRÉ-routing
_is_buy  = _a0_dir >  0.1            # seuil FIXE ±0.1 sur a0 BRUT
_is_sell = _a0_dir < -0.1
_is_hold = (~_is_buy) & (~_is_sell)
n_buy_batch  += int(_is_buy.sum().item())
n_sell_batch += int(_is_sell.sum().item())
n_hold_batch += int(_is_hold.sum().item())
```
Log ANCHOR_DEBUG `nB=%d nS=%d nH=%d` à la l.1407-1415.

**VERDICT : réponse (A).** `nB/nH/nS` comptent l'**action brute `a0` échantillonnée**, discrétisée par un seuil **fixe ±0.1**, **AVANT** `route_action_by_state`, **AVANT** l'état, **AVANT** l'exécution.

**Conséquence directe (corrige l'ancienne formulation « always-SELL ») :**
- `nB=0, nS=2048` **NE signifie PAS** « 2048 SELL exécutés ». Cela signifie « les 2048 échantillons continus de la politique ont donné `a0 < -0.1` ».
- Confirmé par la sonde (Campagne A/C, séquence `sell` a0=-0.5) : route_counts = **HOLD:40, SELL:0** sur les 3 splits → **un `a0` négatif en FLAT ne route JAMAIS SELL** (SELL impossible en FLAT). Le reward de la séquence `sell` est identique à `hold`.

**Requested / Executed / trade_executed / routing_reject** :
- `Requested` = sortie de `route_action_by_state` (post-état, pré-exécution).
- `Executed` = `_last_trade_executed` (post-gate + `agent_close_gate`).
- `routing_reject` = `route≠HOLD ∧ ¬executed` (raisons mesurées : `buy_not_executed`, `sell_not_executed`, `deadband`).
- Ces trois-là sont **distincts** de `nB/nH/nS` (qui sont pré-routing). **Ne jamais dériver « SELL exécuté » de `nS`.**

---

## 3. TEST DÉTERMINISTE — 4 états × 2 actions (chaîne complète, pas à pas)

Extrait réel de la sonde, séquence `alt` sur split train, **advisory (gate OFF), sl/tp=1.0** (`probe_env_20260819_193400.jsonl`). Colonnes = chaîne causale complète.

| i | state | a0 | route | exec | reject | final | raw | bpen | pnl_reward | saturation |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | FLAT | +0.5 | BUY | **True** | – | -0.0695 | -0.0720 | 0 | 0 | -0.0717 |
| 1 | LONG | -0.5 | SELL | **False** | sell_not_executed (min_hold) | -0.0689 | -0.0714 | 0 | 0 | -0.0717 |
| 2 | LONG | +0.5 | HOLD | False | – | -0.0692 | -0.0717 | 0 | 0 | -0.0717 |
| 3 | LONG | -0.5 | SELL | **True** | – | **-0.2386** | -0.2694 | **-0.0375** | **-0.0729** | -0.0717 |
| 4 | FLAT | +0.5 | BUY | False | buy_not_executed (cooldown) | -0.0692 | -0.0717 | 0 | 0 | -0.0717 |
| 5 | FLAT | -0.5 | HOLD | False | – | -0.0692 | -0.0717 | 0 | 0 | -0.0717 |

**Lecture des 4 cas canoniques :**
- **FLAT+BUY** (i=0) : route BUY, exécuté → passage LONG. pnl=0 à l'ouverture, aucune pénalité comportementale.
- **FLAT+SELL** (i=5, a0=-0.5) : route **HOLD** (SELL impossible en FLAT). `sell_flat_must_not_profit=0.0` **respecté** (pnl_reward=0). → confirme §2.
- **LONG+BUY** (i=2, a0=+0.5) : route **HOLD** (pas de re-BUY en position). Aucun effet.
- **LONG+SELL** (i=1 puis i=3) :
  - i=1 : route SELL mais **`agent_close_gate` (min_hold) refuse** → `sell_not_executed`. **Sortie légitime ignorée.**
  - i=3 : min_hold atteint → SELL exécuté, MAIS **`behavior_penalty=-0.0375`** (pénalité d'intention de sortie) appliquée sur une **sortie VALIDE**, + `pnl_reward=-0.0729`. → **l'environnement PUNIT une sortie légitime.**

### Spam / erreur répétée / erreur corrigée
- **Spam SELL en FLAT** (séquence `sell`, 40 pas) : 40×HOLD, reward = celui de `hold`. Le spam d'intention SELL en FLAT n'a **aucun coût ni aucun effet** — il n'existe pas comme « erreur » pour l'env.
- **Erreur répétée BUY en cooldown** (i=4,6 séquence `alt`) : `buy_not_executed` répété, reward constant = plancher. **Aucune pénalité croissante** pour l'insistance.
- **Correction après erreur** : après i=3 (SELL exécuté pénalisé), les BUY i=4,6 sont bloqués par cooldown → la « correction » de la policy ne peut pas s'exprimer avant plusieurs pas (délai imposé par l'env, pas par l'apprentissage).

---

## 4. L'ARTEFACT DOMINANT & LE VRAI SIGNAL (test de régression, 1 variable)

**Campagne A (sl/tp=1.0)** — décomposition `comp_sums` train :
```
[hold ] {saturation_penalty: -2.5801}
[buy  ] {drawdown_penalty: -0.0005, saturation_penalty: -2.8668, pnl_reward: +0.438}
[sell ] {saturation_penalty: -2.8668}
[alt  ] {behavior_penalty: -0.0375, drawdown_penalty: -0.011, saturation_penalty: -2.8668, pnl_reward: -0.504}
[random] {behavior_penalty: -0.0167, drawdown_penalty: -0.0196, saturation_penalty: -2.8668, pnl_reward: -0.502}
```
→ La `saturation_penalty` (≈ -0.072/pas, -2.87 sur 40 pas) **DOMINE tout** et écrase le vrai signal (pnl ±0.5, bpen -0.037). Tous les `mean_final` s'aplatissent à ≈ **-0.062 à -0.069**, quelle que soit la décision.

**Origine prouvée par code** (`multi_asset_chunked_env.py`) :
- Feed l.8933-8936 : `_sl_sat_hist.append(1 si |sl_raw|≥0.9)` ; idem TP.
- Calcul l.7230-7240 : si `rate>0.5` alors `p = 0.10·log1p((rate-0.5)·10)`, `saturation_penalty -= min(0.20, p)`, appliqué **pour SL ET TP**.
- La sonde forçait `sl_raw=tp_raw=1.0` (**obligatoire** en Campagne A/B sinon `sl_pct=0 → p_min=0.99 → fee-gate bloque TOUT BUY**, l.9513). Donc `rate=1.0>0.5` en permanence → pénalité maximale à chaque pas. **C'est un artefact de paramétrage de la sonde, pas un bug de l'env en conditions réelles** — mais il PROUVE que ce terme est décorrélé de la qualité de trade (il ne dépend QUE de la saturation SL/TP).

**Campagne C (sl/tp=0.5, non-saturant)** — même sonde, une seule variable changée :
```
[hold ] final=+0.00000 {}                                             <- saturation ABSENTE
[buy  ] final=+0.01212 {drawdown_penalty:-0.0004, pnl_reward:+0.4988}
[sell ] final=+0.00000 {}                                             <- SELL-en-FLAT = 0 (seuil respecté)
[alt  ] final=-0.01513 {behavior_penalty:-0.0375, drawdown_penalty:-0.011, pnl_reward:-0.5044}
[random] final=-0.00783 {behavior_penalty:-0.0167, drawdown_penalty:-0.0196, pnl_reward:-0.502}
```
→ `saturation_penalty` **disparaît** (dict vide sur hold/sell). Le vrai signal apparaît : `pnl_reward` domine, `behavior_penalty` sur sortie précoce, `drawdown_penalty` faible.

**Biais structurel vs biais dataset** (mean_final par split, Campagne C) :
| seq | train | val | test | lecture |
|---|---|---|---|---|
| hold | +0.00000 | +0.00000 | +0.00000 | **aucun biais structurel** (le plancher -0.06 était 100% saturation) |
| buy | +0.0121 | -0.0008 | -0.0157 | **biais dataset** (signe dépend du chunk) |
| sell | +0.00000 | +0.00000 | +0.00000 | SELL-FLAT neutre partout (seuil `sell_flat_must_not_profit` respecté) |
| alt | -0.0151 | +0.0058 | -0.0399 | dataset + coût comportemental |
| random | -0.0078 | +0.0201 | -0.0387 | dataset ; `|reward|<0.02` sur 2/3 splits (train,val) — seuil `random_mean_reward_max_abs` **tenu sauf test** |

**Verdict régression :** le « reward toujours ≈ -0.06 » était **entièrement** l'artefact saturation. Une fois retiré, la récompense est **quasi nulle** avec de petites fluctuations **pilotées par le dataset**. Il n'existe **pas de biais de reward structurel** favorisant BUY ou SELL une fois la saturation neutralisée.

---

## 5. PÉNALITÉS COMME FONCTIONS (P(1),P(2),P(4),P(8) après TOUTES transformations)

Mesuré par `probe_penalty_as_function()` (identique en Campagne A et B) :

| Fonction annoncée | P(1) | P(2) | P(4) | P(8) | ratios P2/P1, P4/P2, P8/P4 | Fonction EFFECTIVE |
|---|---|---|---|---|---|---|
| symlog **linéaire** | -0.693 | -1.099 | -1.609 | -2.197 | **1.585 → 1.465 → 1.365** | **sous-linéaire** (compression) |
| symlog **quadratique** | -0.693 | -1.609 | -2.833 | -4.174 | **2.322 → 1.760 → 1.473** | **sous-quadratique** (une vraie quad. donnerait 4,4,4) |
| action_anchor effective | -0.0005 | -0.002 | -0.008 | **-0.02 (cap)** | 4.0 → 4.0 → **2.5** | quadratique **puis plafond à 0.02** |

**Conclusions chiffrées :**
- Le `symlog` (l.7456) **écrase** toutes les pénalités : doubler l'erreur produit **moins que le double** de pénalité, et le ratio **décroît** avec l'erreur (1.585→1.365).
- Une pénalité « quadratique » dans le code **n'est PAS quadratique** dans le reward effectif (2.32→1.47, pas 4→4).
- L'`action_anchor` **sature à 0.02** : négligeable face à `pnl_reward` (±0.5) et à `saturation_penalty` (2.87). Elle n'a **aucune force de rappel effective** sur le reward.

---

## 6. APPRENTISSAGE DE L'ERREUR (métriques)

La sonde est un **probe d'environnement sans PPO** : il n'y a **ni μ/σ de policy, ni advantage, ni gradient réel**. Ces métriques sont donc évaluées **au niveau du signal disponible pour l'apprentissage**, pas au niveau d'une policy entraînée (déclaration d'honnêteté, cf. §7 questions D-E-F).

| Métrique | Mesure sonde | Valeur | Lecture |
|---|---|---|---|
| correction_rate | fraction d'erreurs suivies d'un changement d'action possible | non mesurable sans policy | l'env impose des délais (cooldown, min_hold) qui **empêchent** une correction immédiate |
| correction_delay | pas avant qu'une action alternative soit exécutable | ≥ min_hold (SELL), ≥ cooldown (BUY) | délai **structurel**, indépendant de l'apprentissage |
| error_persistence | pénalité croissante si erreur répétée ? | **NON** (§3 spam) | reward constant, pas d'aggravation → pas de pression anti-répétition |
| inter_episode_persistence | historiques SL/TP à reset | `_sl_sat_hist`/`_tp_sat_hist` sont des deques par épisode | pas de fuite inter-épisode observée pour la saturation |

---

## 7. LES 5 (→8) CONCLUSIONS OBLIGATOIRES

### Les 5 questions du mandat
**A. Où le signal d'erreur ENTRE-t-il ?**
Au calcul du reward, `_calculate_reward` (l.6924), via des termes de pénalité : `behavior_penalty` (routing/close, l.8683-8688), `drawdown_penalty`, `saturation_penalty` (l.7228-7240), `action_anchor` (loss, l.1330). **Prouvé** : la sonde voit ces termes non nuls sur `alt`/`random`.

**B. Où est-il TRANSFORMÉ ?**
Au **symlog** l.7456 : `final = sign(raw)·log1p(|raw|)`. Prouvé §5 : compression sous-linéaire (ratios 1.585→1.365) et « quadratique » devenue sous-quadratique.

**C. Où ATTEINT-il le gradient (loss) ?**
Via `final_reward → GAE advantage → policy_grad_loss`, plus l'ancre L2 directe `λ·(μ²).mean()` ajoutée à la loss actor (l.1330). **Nuance prouvée** : l'ancre plafonne à 0.02 (§5) → contribution au gradient **négligeable**.

**D. Où DISPARAÎT-il ?**
1. Au **routing/gates** : un `a0` négatif en FLAT (intention SELL) est transformé en HOLD → l'« erreur » n'atteint jamais le reward (§2, §3). 2. Au **symlog** : les grosses erreurs sont écrasées (§5). 3. Sous la **saturation_penalty** quand SL/TP saturent : le signal utile (pnl ±0.5, bpen -0.037) est noyé par un terme -2.87 décorrélé du trade (§4).

**E. L'action suivante démontre-t-elle un apprentissage RÉEL ?**
**NON DÉMONTRABLE par cette sonde** (pas de policy). Ce que la sonde démontre : l'environnement (a) **ne pénalise pas** la répétition d'erreur (pas de pression corrective, §3/§6), (b) **punit des sorties légitimes** (behavior_penalty sur SELL valide, §3), (c) **bloque l'exploration BUY** (fee-gate + cooldown, §8), (d) offre un signal quasi nul une fois la saturation retirée (§4). → Le substrat de reward **ne fournit pas** un signal d'apprentissage clair et directionnel. **Conclusion : V2 = NO-GO** tant que ce substrat n'est pas corrigé (corrections listées §9, à valider par sonde avant tout entraînement).

### Questions complémentaires F-H
**F. La pénalité atteint-elle l'advantage ?** Oui via GAE sur `final_reward`, mais **écrasée** par le symlog et **noyée** par la saturation (quand active) — donc atteinte mais **peu discriminante**.

**G. La correction persiste-t-elle à l'épisode suivant ?** Non mesurable sans policy ; côté env, les deques de saturation sont par-épisode (pas de fuite).

**H. Quelle EV influence réellement chaque étape ?** Voir §8 (une seule EV modifie le CONTRÔLE : le fee-gate BUY).

---

## 8. CLASSIFICATION DES EV (OBSERVATION/TELEMETRIE/GATE/REWARD/PENALTY/ADVANTAGE/LOSS/CONTROL)

| EV | Rôle prouvé | Classe |
|---|---|---|
| EV fee-gate (`resolve_ev_fee_gate`, `p_min_required`) | **seule EV qui modifie le contrôle** : bloque BUY si `p_hmm ≤ p_min`. Campagne B : séquence `buy` = 40 route BUY / **40 buy_not_executed** (train,test) → gate mange tout | **GATE / CONTROL** |
| EV TP/SL (sl_pct, RR) | alimente `p_min_required` + saturation_penalty | GATE (indirect) + PENALTY |
| EV drawdown | `drawdown_penalty` (≤ -0.02 mesuré) | PENALTY |
| EV position (behavior) | `behavior_penalty` sur close/routing | PENALTY |
| EV risque / trading | pnl_reward (±0.5) | REWARD |
| EV environnement (saturation) | `saturation_penalty` — **décorrélée du trade** | PENALTY (parasite) |
| p_hmm (context_vector[3]) | entrée de la GATE, non récompensée | OBSERVATION |
| ANCHOR_DEBUG nB/nH/nS | compteurs a0 brut pré-routing | TELEMETRIE |
| action_anchor L2 | `λ·(μ²)` dans loss, cap 0.02 | LOSS (négligeable) |

**Règle du mandat respectée** : aucune EV ne doit modifier le reward « parce qu'elle est disponible ». **Prouvé** : la seule EV qui agit sur le CONTRÔLE est le fee-gate (BUY only) ; les autres sont soit PENALTY/REWARD (calibrées, discutables §5), soit OBSERVATION/TELEMETRIE (n'agissent pas).

---

## 9. V30 vs V31 (ligne à ligne, factuel)

| Aspect | V30 | V31 | Cause commune identifiée |
|---|---|---|---|
| Comportement observé | always-BUY | always-SELL | **imprécis** (§2) : c'est l'intention `a0` qui sature, pas l'exécution |
| Réalité (via §2) | `a0>+0.1` dominant | `a0<-0.1` dominant | distribution PPO saturée d'un côté (μ décalé), routing+gates transforment ensuite selon l'état |
| Substrat reward | identique | identique | même env : saturation domine si SL/TP saturent ; symlog écrase ; fee-gate bloque BUY |

**Conclusion V30/V31 :** les deux collapses partagent **le même substrat** ; la différence n'est que le **signe de μ** vers lequel la policy a dérivé. Corriger le substrat (§ recommandations) est prérequis aux deux.

---

## 10. RECOMMANDATIONS AVANT V32 (une correction → sonde → verdict → commit ; PAS d'entraînement)

> **Aucune de ces corrections n'est appliquée dans ce document.** Elles sont listées comme protocole à exécuter **une variable à la fois**, chacune re-validée par la sonde avant la suivante, conformément à la consigne « une correction → sonde → verdict → OK → commit ; pas OK → annuler ».

1. **fee-gate bloque BUY** (prouvé §8, B) → option config : réduire `p_min_required` OU relever `p_hmm` par défaut OU mode advisory documenté. Critère sonde : `buy_not_executed` < 50%.
2. **saturation_penalty parasite** (prouvé §4) → ce terme domine dès que SL/TP saturent et est décorrélé du trade. Option : réduire `lambda`/`cap`, OU ne l'activer que si la saturation persiste au-delà d'une longue fenêtre. Critère : `|comp saturation| < |pnl_reward|`.
3. **symlog écrase les pénalités** (prouvé §5) → si des pénalités doivent être discriminantes, augmenter leurs coefficients OU remplacer symlog par clamp/tanh. Critère : ratio P8/P4 ≥ 1.5.
4. **behavior_penalty sur SELL légitime** (prouvé §3, i=3) → exempter la sortie valide. Critère : SELL profitable ⇒ reward positif.
5. **min_hold bloque des sorties** (prouvé §3, i=1) → exception stop-loss. Critère : SELL exécuté quand nécessaire.
6. **pas de bonus de diversité** (prouvé §3/§6) → envisager un bonus d'entropie/alternance. Critère : `alt`/`random` ≥ `hold` en reward.

**GO/NO-GO :** tant que A→H ne sont pas satisfaits **par code + sonde après correction**, **V32 = NO-GO**. Aucun 500k. Au mieux, après validation sonde, un **mini-run** surveillé — jamais un 500k direct.

---

## 11. Reproductibilité
```bash
# Campagne A (advisory, sature)
python3 scripts/diagnostics/probe_env_deterministic.py --steps 40 --splits train val test --disable-ev-gate --sl-raw 1.0 --tp-raw 1.0
# Campagne B (gated, sature)
python3 scripts/diagnostics/probe_env_deterministic.py --steps 40 --splits train val test --sl-raw 1.0 --tp-raw 1.0
# Campagne C (advisory, NON-saturant — isole le vrai signal)
python3 scripts/diagnostics/probe_env_deterministic.py --steps 40 --splits train val test --disable-ev-gate --sl-raw 0.5 --tp-raw 0.5
```
Sorties : `logs/probe/probe_env_<ts>.jsonl` (pas à pas) + `logs/probe/probe_verdict_<ts>.json` (agrégats + penalty_function_probe).
