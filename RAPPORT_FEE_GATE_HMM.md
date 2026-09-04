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
