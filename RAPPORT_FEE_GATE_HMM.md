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
