# Verdict forensique — ADAN V27, checkpoint 320k

Date d'analyse : 2026-08-10
Périmètre : ADAN0 uniquement. MapNet gelé.
Run : `adan_500k_v27_hmm_semantic_20260809T123040Z` (arrêté par COLLAPSE-BREAKER à 473 601 steps)
Checkpoint autopsié : `checkpoints/ppo_adan0_v27_hmm_semantic_20260809T123040Z_checkpoint_320000_steps.zip` (référence immuable)

Question centrale posée :

> Quand ADAN vend, est-ce que cette décision est réellement rentable, ou est-ce que l'environnement lui apprend simplement que SELL est l'action la moins dangereuse ?

## 1. Verdict exécutif

**SELL n'est pas rentable. SELL est l'action la moins dangereuse, et le système de reward V27 la rend structurellement attractive dans les deux états du portefeuille.**

Preuves directes :

1. Les trades réellement exécutés ont un **edge net négatif** : 9 137 cycles complets, PnL net cumulé **−542,99** (trace) / −539,86 (compteur run). Win rate global ≈ 14 %.
2. Au checkpoint 320k, backtest déterministe sur le split **val** (données non vues) : 2 000 steps, **0 trade, 0 tentative**, action saturée (`action_max = 1.0` sur 100 % des steps), equity figée à 20,50, reward constant **−0,306272** par step. La politique est déjà dans l'attracteur SELL-flat hors-échantillon.
3. La matrice `position_state × action` (label brut de la politique, confirmé par l'existence de 2 182 SELL+flat dans le CSV) montre les deux gradients qui fabriquent le collapse :

| État | Action | raw_reward moyen | n |
|---|---|---:|---:|
| flat | BUY | **−0,0154** | 544 |
| flat | HOLD | −0,0086 | 375 |
| flat | SELL | **−0,0063** | 2 182 |
| long | BUY | −0,0138 | 464 |
| long | HOLD | +0,0035 | 237 |
| long | SELL | **+0,0090** | 901 |

En FLAT, **BUY-ouvrant est l'action la plus punie** ; en LONG, **SELL-clôturant est la seule action récompensée positivement**. Les deux gradients pointent vers SELL. La dérive monotone de `a0_mean` (0,00 → −0,98) est la conséquence attendue de cette topologie de reward, pas une anomalie du réseau.

## 2. Réconciliation financière (point resté ouvert depuis V25 — résolu)

Écart apparent : `terminal_realized_pnl = −539,86` vs equity 20,50 → 19,51.

- La somme des `pnl` des 9 137 événements CLOSE de la trace JSONL = **−542,9953**.
- Différence trace − compteur run = **exactement +3,1318**, soit le montant exact des **42 `DRAWDOWN_KILL_FORCE_CLOSE`**. Le compteur run-level exclut les clôtures drawdown-kill (défaut de comptabilité, à corriger).
- L'equity 20,50 → 19,51 n'est que le **dernier des 146 épisodes** (chunk_size = 1 000). PnL cumulé multi-épisodes et equity terminale du dernier épisode ne sont pas la même quantité : il n'y a pas d'incohérence financière, seulement deux métriques de portée différente.
- `behavior_penalty` est bien ajouté au `raw_reward` (ligne 7397) et **pas** à `realized_pnl` : le commentaire ligne 9304 (« realized_pnl += _step_invalid_penalty ») est obsolète. Pas de contamination de la métrique financière.

## 3. PPO — timing des early-stops KL (point ouvert — résolu)

Compte précis : **594** occurrences « Early stopping at … due to reaching max kl » sur 925 rollouts = **64 %** (et non 533 — le chiffre 533 venait d'un motif de grep plus étroit).

Distribution par phase :

| Phase | a0_mean | % SELL | KL moyen | % rollouts KL > 0,05 | EV critic |
|---|---:|---:|---:|---:|---:|
| 0–100k | +0,015 | 47 % | 0,0029 | 0 % | +0,058 |
| 100–200k | −0,043 | 53 % | 0,0367 | 30 % | +0,072 |
| 200–300k | −0,316 | 76 % | 0,0782 | **74 %** | +0,102 |
| 300–380k | −0,616 | 91 % | 0,1146 | **87 %** | +0,083 |
| 380–474k | −0,878 | 97 % | 0,1201 | 86 % | **−0,031** |

Jalons : `a0_mean < −0,3` à t = 226 816 ; `< −0,7` à t = 351 744.

**Le KL devient chronique (74 %) dès 200–300k, c'est-à-dire avant et pendant le franchissement de −0,3.** Les early-stops précèdent et accompagnent la dérive ; ils ne la suivent pas. Conséquence : plus de la moitié des mises à jour PPO sont tronquées exactement au moment où la politique dérive — le mécanisme censé contenir la dérive (max_kl) la fige au lieu de la corriger, car chaque update partiellement appliqué pousse un peu plus vers SELL sans jamais permettre une correction complète.

## 4. Câblage réel du reward (vérifié dans le code)

- `_inv_pen_weight = 0.0` (ligne 8313, commentaire « was 0.005 — C6 fix (all gate rejections = 0 reward) ») : **toutes les pénalités de gate** (cooldown_wait, risk_gate, omega4e, daily_limit, fee_gate, hold_min) sont neutralisées.
- Branche SELL-sans-position explicitement morte (lignes 9309–9323) : routée en HOLD, zéro pénalité, commentaire « The old sterile penalty (V5) is REMOVED… it is simply routed to HOLD upstream ». La pénalité stérile V5 est donc bien du **code mort par conception V12**, pas par oubli.
- Seules survivent dans `behavior_penalty` : la pénalité clamp SL/TP (−0,01) et les pénalités catastrophiques (−5,0).
- La « couche B » (pénalité des actions invalides) discutée dans les propositions de correction **n'existe pas dans V27** — supprimée délibérément (C6 + V12) pour tuer la « pollution de gradient ». Le collapse V27 démontre que cette suppression a recréé le point fixe sans douleur : SELL-flat coûte −0,0063 (uniquement via saturation/drawdown/future), pas via une pénalité d'invalidité.

## 5. Décomposition des clôtures par raison (trade par trade)

| Raison | n | PnL somme | PnL moyen | win % |
|---|---:|---:|---:|---:|
| MaxDuration | 7 019 | −348,81 | −0,0497 | 17,6 % |
| stop_loss | 1 149 | −230,24 | −0,2004 | 0,0 % |
| agent_close | 518 | −38,86 | −0,0750 | 6,4 % |
| take_profit | 402 | +78,39 | +0,1950 | 100 % |
| DRAWDOWN_KILL | 42 | −3,13 | −0,0746 | 0 % |
| CHUNK_END | 7 | −0,34 | −0,0490 | 0 % |

Lecture :

- **77 % des clôtures sont des MaxDuration** — l'environnement ferme à la place de l'agent, à perte moyenne. L'agent ne sait pas sortir ; il attend le timeout.
- Les seuls gains systématiques viennent des take_profit (4,4 % des clôtures).
- Les `agent_close` (vrais SELL décidés par la politique en position) ne sont que 518 sur 9 137 (5,7 %) et perdent en moyenne −0,075. **Même quand l'agent décide lui-même de vendre, il perd.**

## 6. Verdict causal hiérarchisé

### Prouvé

1. Edge net des trades exécutés : **négatif** (−542,99 sur 9 137 cycles).
2. SELL-flat n'est pas puni par une pénalité d'invalidité (V12, code mort assumé) ; son coût résiduel (−0,0063) est le plus faible des trois actions en FLAT.
3. BUY-flat est l'action la plus punie en FLAT (−0,0154) alors que c'est l'ouverture légitime.
4. SELL-long est la seule cellule de la matrice récompensée positivement (+0,0090) — alors même que les clôtures décidées par l'agent perdent en moyenne. Le reward de clôture décorele le signal du PnL réel.
5. Le collapse est progressif et monotone (0,00 → −0,98 sur 473k steps), déjà complet hors-échantillon au checkpoint 320k.
6. 64 % des rollouts tronqués par max_kl, à partir de la phase où la dérive s'installe.
7. Réconciliation financière résolue (écart = DRAWDOWN_KILL exclus du compteur run).

### Fortement corrélé

1. La topologie du reward (matrice §1) précède la dérive directionnelle : c'est un problème de **reward shaping**, pas de représentation CNN/FiLM/attention. Le basculement SELL intervient **après** que le reward a rendu BUY plus coûteux — branche « reward shaping » de l'arbre.
2. La saturation de la tête direction (action_max = 1,0 en backtest) coïncide avec la région de reward quasi nul terminal.

### Plausible mais non isolé

1. La part exacte de `saturation_penalty`, `future_contrib` et `closure_bonus` dans la punition de BUY-flat (le CSV montre leurs moyennes par cellule, mais leur calibration relative reste à modéliser).
2. Le rôle du critic (EV −0,031 en fin de run) dans l'amplification finale.

### Non démontrable avec les artefacts actuels

1. Le contrefactuel propre : ce que HOLD/BUY auraient donné **au même step** sur les trades exécutés (nécessite la simulation A–F de l'arbre — voir §8).
2. Les ablations CNN/FiLM/attention (étape 5 de l'arbre) — **non prioritaires** : la cause reward est prouvée en amont.

## 7. Décision selon l'arbre

```text
Trades exécutés ?            OUI (9 137)
Edge net positif ?           NON (−542,99)
Comparaison par état :       SELL < BUY en FLAT ; SELL seul > 0 en LONG
→ branche « REWARD À CORRIGER » (prouvée, pas seulement suspectée)
PPO :                        KL chronique AVANT la dérive → à corriger aussi,
                             mais en SECOND, après le reward
500k :                       🛑 REFUSÉ — portes 1, 2, 3 toutes rouges
```

## 8. Prochaine étape autorisée (et seule)

Conformément à l'arbre, **aucune modification de code n'est encore autorisée**. La prochaine action est la **simulation contrefactuelle A–F sur le checkpoint 320k** (politique réelle / HOLD forcé / BUY forcé / SELL forcé / sans frais / pénalité SELL modifiée) pour quantifier précisément quelle composante du reward fabrique la punition de BUY-flat (−0,0154) et la prime de SELL-long (+0,0090).

Hypothèse falsifiable associée (à tester, pas à appliquer) :

> « Si l'on neutralise la pénalité implicite sur BUY-flat et la prime implicite sur SELL-long dans une simulation courte, la distribution BUY/SELL/HOLD se rééquilibre sans toucher à PPO ni à l'architecture. »

Toute modification ultérieure devra : (1) cibler uniquement le mécanisme prouvé fautif, (2) être paramétrée dans `config/config.yaml` (constantes existantes : `invalid_trade_penalty_weight`, `capital_tier_rewards`, etc.), (3) passer un smoke 2048 avec matrice `état × action` assainie, (4) passer les 4 portes avant tout run 500k.

**V27-320k reste la référence immuable.**

---

## Annexe — Simulation contrefactuelle A–F (exécutée le 2026-08-10)

Batterie `scripts/backtest/counterfactual_320k.py`, 6 bras × 3 000 steps, split val,
checkpoint 320k gelé, résultats dans `logs/validation/counterfactual_320k/`.

| Bras | Equity | Return | Trades | Reward |
|---|---:|---:|---:|---:|
| A — politique réelle | 20.50 | 0.00% | 0 | −917.5913 |
| B — HOLD forcé | 20.50 | 0.00% | 0 | −917.5913 |
| C — BUY forcé | 16.51 | −19.47% | 59 | −937.1939 |
| D — SELL forcé | 20.50 | 0.00% | 0 | −917.5913 |
| E — politique sans frais | 20.50 | 0.00% | 0 | −917.5913 |
| F — A + pénalité −0.05/SELL stérile | 20.50 | 0.00% | 0 | −1 067.5913 |

Faits tranchés :

1. A ≡ B ≡ D ≡ E au centième (reward −917.5912693738937 identique) : la politique
   320k ne produit hors-échantillon QUE des SELL stériles — 3 000/3 000 steps
   (bras F). Le collapse est total et déterministe ; SELL = HOLD déguisé.
2. SELL ne produit aucun edge (equity figée, identique au HOLD forcé).
   Réponse définitive : action la moins dangereuse, pas rentable.
3. BUY forcé perd −19.47 % sur cette fenêtre (59 trades, 0 gagnant) — avec les
   size/SL/TP du modèle collapsé ; attribution marché vs politique non isolable.
4. Bras F : la pénalité couche-B proposée (−0.05/SELL stérile) aurait coûté
   −150 de reward sur ce comportement — première quantification directe de
   l'effet dissuasif attendu de la correction.

Décision inchangée : branche « collapse structurel + reward shaping prouvé ».
Correction ciblée autorisée à l'étude : pénalité SELL-flat + dé-pénalisation
BUY-flat, constantes dans config/config.yaml, puis smoke 2048 from scratch.
