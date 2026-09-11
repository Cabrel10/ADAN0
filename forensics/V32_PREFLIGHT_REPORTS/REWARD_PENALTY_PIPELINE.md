# RAPPORT 6 — REWARD_PENALTY_PIPELINE

**Objet** : tracer le pipeline COMPLET
`action → routage → erreur → pénalité → reward → gradient → correction`, et
identifier chaque **rupture de boucle de rétroaction** (là où le signal se perd).

**Source** : code env / action_routing / feature_extractors, +
`ABSORPTION_QUANTIFIED.md`, `L2_DEEP.md`, `RADAR.md`. Chiffres réels V31.

---

## 1. Pipeline nominal (ce qui DEVRAIT boucler)

```
(1) actor: mu,sigma  ─►  a0 = tanh(mu + sigma·ε)          feature_extractors
(2) a0  ─►  route_action_by_state ─► intent∈{BUY,HOLD,SELL} action_routing 48-106
(3) intent=BUY ─► resolve_ev_fee_gate ─► accept/reject      action_routing 109-125
(4) exécution OU rejet ─► erreur détectée (routing_reject)  env 8646-8694
(5) erreur ─► behavior_penalty (_step_invalid_penalty)      env 8683-8688
(6) _calculate_reward: raw_reward = pnl + pénalités + ...    env 6924-7409
(7) final_reward = symlog(raw_reward)                        env 7456
(8) buffer PPO ─► advantage(BUY/HOLD/SELL) ─► loss           SB3 rollout
(9) loss + anchor_loss(λ·mu²) ─► gradient ─► MAJ mu,sigma    feature_extractors 1330
(10) retour en (1) avec mu corrigé
```

Pour que la boucle **corrige** une erreur, il faut que l'étape (8) dispose d'un
**contre-exemple** : au moins un échantillon de chaque action pour comparer les
advantages. C'est ici que tout casse.

---

## 2. Les RUPTURES de boucle (mesurées)

### Rupture R1 — le contre-exemple disparaît du buffer (RUPTURE MAJEURE)

Point : entre (2) et (8). Quand μ dérive, `P(BUY)/échantillon` s'effondre
(ABSORPTION_QUANTIFIED) :

| μ | P(BUY)/sample | E[nB] / buffer 2048 | régime |
|------|--------------|---------------------|--------|
| -0.50 | 5.89e-02 | 120.7 | diversité OK |
| -0.80 | 8.84e-03 | 18.1 | raréfaction |
| **-1.145** | **4.79e-04** | **1.0** | QUASI-ABSORBANT |
| -1.50 | 1.03e-05 | 0.0 | ABSORBANT |
| -8.20 | 0 | 0 | ABSORBANT terminal |

→ dès μ≈-1.145, `E[nB]≈1` puis 0 → **nB=nH=0** dans le buffer →
`adv_BUY=adv_HOLD=NaN` (L2_DEEP) → l'étape (8) n'a plus de contre-exemple →
la correction (9→10) ne peut plus s'exprimer. **La boucle est physiquement
ouverte alors que le code tourne encore.**

### Rupture R2 — l'exécution s'effondre (routage stérile)

Point : étape (4). RADAR.md / L2_DEEP : à l'état absorbant,
**95.5 % routing_reject, 0.32 % exec** (fenêtre 9 : 406k policy → 32 exec =
0.01 %). L'agent « décide » massivement mais **rien ne s'exécute** → aucune
transition d'état s→s′ réelle → le PnL réel (terme structurant de raw_reward)
ne varie plus → (6) ne reçoit plus de signal PnL informatif, seulement des
pénalités/coûts de fond.

### Rupture R3 — le shaping ev_norm n'est pas branché (RAPPORT 2 §5)

Point : étape (6). `ev_norm` (β=0.1) est calculé dans `RewardCalculator`
(reward_calculator.py 273) qui n'est PAS le chemin effectif ; dans
`_calculate_reward` (env) il n'entre que comme télémétrie (env 4416→4505).
→ un signal de valeur est produit mais **jamais injecté dans le gradient**.

### Rupture R4 — le symlog écrase les pénalités (RAPPORT 3 §6)

Point : étape (7). `final=sign·log1p(|raw|)` : -0.28 → -0.247 ; -2.0 → -1.10.
Au-delà de |raw|~1, augmenter une pénalité n'apporte quasi aucun signal. →
la boucle « pénalité plus forte = correction plus forte » est **saturée** par
construction.

### Rupture R5 — la seule force sur μ est l'ancre L2, et elle est mal calibrée

Point : étape (9). Seul `anchor_loss = λ·(mu²).mean()` (feature_extractors 1330)
agit **directement sur μ**. En V31, λ=0.05 a stabilisé μ dans **[-9.2, -7.7]**
(L2_DEEP) — un équilibre STABLE mais LOIN de 0. L'ancre a « tenu » μ… au mauvais
endroit. Pire, `ABSORPTION_QUANTIFIED` prouve qu'un **clamp μ=±3 est inutile**
(E[nB]=0 déjà à μ=-3) : il faut borner **|μ| ≲ 0.7-0.8** pour garder E[nB]≫1.

---

## 3. Fenêtre d'intervention (fait critique)

`ABSORPTION_QUANTIFIED` : la raréfaction commence à **upd≈292** (nB=28),
l'absorption se scelle à **upd≈336** (nB=nH=0 durable), collapse déclaré durable
à **upd=368** (RADAR.md). Sur 865 updates totaux, **il y avait ~40 updates de
fenêtre d'intervention** (292→336). Un radar live L4 (RAPPORT 5) échantillonnant
par update aurait déclenché AVANT upd=336.

---

## 4. Diagramme des ruptures

```
(1)actor ──► (2)route ──► (3)fee_gate ──► (4)exec/reject ──► (5)penalty ──► (6)reward ──► (7)symlog ──► (8)buffer ──► (9)loss+anchor ──► (10)MAJ μ,σ
                 │                            │                                │ R4 écrase   │ R1 nB=nH=0   │ R5 μ borné
                 │                            │ R2 95.5% reject                │             │ adv=NaN      │ à -8, pas 0
                 │                            └────────────────────────────────┘             │              │
                 └─────────── R3 ev_norm calculé mais non branché (chemin mort) ──────────────┘              │
                                                                                    BOUCLE OUVERTE ◄──────────┘
```

---

## 5. Hiérarchie des ruptures (par gravité causale)

| Rupture | Gravité | Réversible en cours de run ? | Levier correct |
|---------|---------|------------------------------|----------------|
| **R1** nB=nH=0 / adv=NaN | **FATALE** | NON une fois scellée | borner |μ|≲0.7-0.8 + plancher exploration (PAS reward) |
| R5 ancre μ mal calibrée | fatale (cause de R1) | oui si détectée tôt | recalibrer λ / cible μ≈0 / clamp σ (gSDE off) |
| R2 exec 0.32 % | grave (symptôme de R1) | suit R1 | résolue si R1 résolue |
| R4 symlog écrase | structurelle | non (design) | ne PAS compter sur les pénalités pour corriger μ |
| R3 ev_norm non branché | modérée | oui | brancher ev_norm dans _calculate_reward OU l'assumer télémétrie |

---

## 6. Synthèse — verdict pour le RAL

1. **La rupture fatale (R1) est une rupture de DIVERSITÉ D'ÉCHANTILLONNAGE, pas
   de reward.** Elle se produit à μ≈-1.145, bien avant la saturation tanh.
2. **Un RAL qui module les coefficients de reward agit sur les étapes (6)-(7),
   EN AVAL de R4 (symlog) et SANS toucher (9) où vit R1/R5.** → Confirmation
   formelle du risque utilisateur : **moduler le reward ne rouvre PAS la boucle**.
3. **Le seul levier qui rouvre la boucle est en (9)** : borner μ dans |μ|≲0.7-0.8
   (ancre recalibrée) et/ou garantir un plancher de σ/exploration pour maintenir
   `min(P) > 1/N_batch`. C'est un levier **loss actor**, pas reward.
4. Donc le RAL n'est LÉGITIME que s'il : (a) lit `s4` live (RAPPORT 5), (b) tant
   que `s4` sain, module le reward pour la qualité (PF/diversité) ; (c) dès que
   `s4` menace, déclenche le levier (9) (contrainte μ/σ) — jamais une pénalité
   reward supplémentaire.
