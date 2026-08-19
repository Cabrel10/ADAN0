# RAPPORT 2 — EV_DEPENDENCY_GRAPH

**Objet** : cartographier les DÉPENDANCES entre les EV inventoriées au RAPPORT 1
— quelle EV alimente quelle autre, et laquelle atteint réellement le contrôle
(l'action) vs le gradient (via reward) vs rien (télémétrie pure).

**Méthode** : lecture ciblée des chemins d'appel dans
`src/adan_trading_bot/environment/` au commit courant de `genspark_ai_developer`.
Chaque arête est justifiée par une paire fichier:ligne. Aucune supposition.

---

## 1. Les trois plans (rappel RAPPORT 1)

| Plan | Effet | Entre dans le gradient PPO ? |
|------|-------|------------------------------|
| **CONTRÔLE** | modifie l'action émise (route BUY/HOLD/SELL) | Oui — indirectement (change les transitions donc les rewards observés) |
| **REWARD (shaping)** | modifie `raw_reward` puis `final_reward` | Oui — directement (dans la loss critic + advantage) |
| **TÉLÉMÉTRIE** | JSONL / logs | Non |

---

## 2. Graphe de dépendances (texte)

```
                       ┌─────────────────────────────┐
                       │  policy actor (feature_ext.) │
                       │  mu, sigma -> a0 = tanh(...)  │  a0 ∈ [-1,1]
                       └───────────────┬─────────────┘
                                       │ a0 (dim 0 du vecteur d'action)
                                       ▼
             ┌───────────────────────────────────────────────┐
             │  _route_action_by_state(a0, in_position, ...)   │
             │  action_routing.route_action_by_state 48-106    │
             │  FLAT : BUY ssi a0>thr sinon HOLD               │
             │  LONG : SELL ssi a0<-sthr sinon HOLD            │
             └───────┬───────────────────────────────┬────────┘
                     │ intent=BUY                     │ intent=SELL/HOLD
                     ▼                                ▼
   ┌──────────────────────────────┐      (SELL/HOLD ne passent PAS
   │ resolve_ev_fee_gate(          │       par le gate d'EV — voir §4)
   │   p_hmm, p_min_required,       │
   │   disabled)                    │  action_routing 109-125
   │ p_hmm <- env 8723-8732         │
   └──────────┬────────────────────┘
              │ blocked==True -> BUY refusé (routing_reject)
              ▼
        action effective (BUY exécuté | HOLD | SELL)
              │
              ▼
   ┌──────────────────────────────────────────────────────────┐
   │ _calculate_reward(action, realized_pnl)  env 6924          │
   │  raw_reward = pnl_base + behavior_penalty + action_anchor  │
   │    + holding_cost + drawdown + symmetry + action_entropy   │
   │    + future_contrib + latent_pnl + saturation   (env 7400) │
   │  final_reward = symlog(raw_reward)              (env 7456)  │
   └──────────┬───────────────────────────────────┬────────────┘
              │ ev_norm (télémétrie)               │ final_reward
              ▼                                     ▼
   ┌────────────────────────────┐        ┌──────────────────────┐
   │ RewardCollector.log_step    │        │ buffer PPO -> advantage│
   │ reward_collector.py 66,191  │        │ -> loss actor/critic   │
   │ (JSONL, AUCUN gradient)     │        │ + anchor_loss L2       │
   └────────────────────────────┘        │ feature_extractors 1330│
                                          └──────────────────────┘
```

---

## 3. Arêtes CONFIRMÉES (fichier:ligne)

| # | De | Vers | Preuve | Type |
|---|----|------|--------|------|
| E1 | actor `mu,sigma` | `a0` | `feature_extractors.py` (DiagGaussian/tanh), anchor `(mu**2).mean()` 1330 | contrôle |
| E2 | `a0` | `route_action_by_state` | `action_routing.py` 48-106 ; appelé env 8641 | contrôle |
| E3 | `route_action_by_state`=BUY | `resolve_ev_fee_gate` | gate importé env l.32 ; `p_hmm` calc env 8723-8732 | contrôle |
| E4 | `resolve_ev_fee_gate`=blocked | `routing_reject` + `behavior_penalty` | env 8655-8694 (`self._step_invalid_penalty += _bp_val`) | contrôle→reward |
| E5 | action effective | `_calculate_reward` | env 6924 | reward |
| E6 | composantes | `raw_reward` | env 7400 | reward |
| E7 | `raw_reward` | `final_reward=symlog` | env 7456 | reward |
| E8 | `reward_breakdown['ev_norm']` | `RewardCollector.log_step` | env 4416→4505 ; collector 66,191 | télémétrie |
| E9 | `final_reward` | buffer PPO → advantage → loss | standard SB3/PPO (rollout) | gradient |
| E10 | actor `mu` | `anchor_loss` | `feature_extractors.py` 1330-1331 (`loss += anchor_lambda*(mu**2).mean()`) | gradient (régularisation) |

---

## 4. Fait structurel MAJEUR : le gate d'EV ne s'applique QU'À BUY

`resolve_ev_fee_gate` (E3/E4) n'est consulté que sur la branche **intent=BUY**
du routage. Conséquences prouvées par les autopsies :

- **SELL ne franchit aucun gate d'EV** → aucune EV ne peut *freiner* un SELL.
  C'est cohérent avec le collapse V31 « always-SELL » (μ→-8.58) : rien dans le
  plan EV/contrôle ne s'oppose à une dérive vers SELL. Réf :
  `forensics/v31_500k_collapse_20260818/RAPPORT_COLLAPSE_V31_500K.md`.
- **BUY subit un frein asymétrique** (fee gate + behavior_penalty côté V28) →
  cohérent avec le collapse V30 « always-BUY » n'ayant PAS été freiné par l'EV
  (l'asymétrie de pénalité poussait a0→+1, autopsie V30 RC1 révisé).

**L'asymétrie du graphe (BUY gaté, SELL non gaté) est donc un facteur causal
direct des DEUX collapses**, dans des directions opposées selon quelle force
domine (pénalité BUY V30 vs σ non bornée V31).

---

## 5. Fait : deux chemins de reward coexistent, un seul est effectif

- `RewardCalculator.calculate()` (reward_calculator.py 157, ev_norm β=0.1 l.273)
  = chemin « standalone ».
- `MultiAssetChunkedEnv._calculate_reward()` (env 6924) = chemin **effectif**
  prouvé par régression 40 000 échantillons (autopsie V30, addendum).

→ L'EV de reward `ev_norm` documentée au RAPPORT 1 §B transite par le chemin
**non effectif**. Dans les runs V30/V31, `ev_norm` est donc **quasi inerte sur
le gradient** ; il n'agit qu'en télémétrie (E8). C'est une **rupture de chaîne**
(un signal de valeur calculé mais non branché) — à documenter aussi au RAPPORT 6.

---

## 6. Nœuds sans arête entrante « live » (EV fantômes)

Les EV par état×action qui EXPLIQUENT le collapse (EV(FLAT+SELL), EV(OPEN+BUY),
P(BUY)/échantillon, PF/WR par fenêtre) n'ont **aucune arête entrante dans le
graphe d'exécution** : elles ne sont produites que par les scripts forensics
*a posteriori* (RAPPORT 1 §D). Le futur radar live (RAPPORT 5) doit créer ces
nœuds et les brancher en lecture — **sans** créer de nouvelle boucle de contrôle
non maîtrisée (risque identifié : reproduire l'échec de l'ancre L2).

---

## 7. Synthèse des dépendances

1. **Chaîne de contrôle** : `mu,sigma → a0 → route → (fee_gate si BUY) → action`.
   Une seule EV agit ici (`resolve_ev_fee_gate`), et **seulement sur BUY**.
2. **Chaîne de reward** : `action → _calculate_reward → raw → symlog → final →
   advantage → loss`. `ev_norm` (β=0.1) est censé y contribuer mais passe par le
   mauvais objet (`RewardCalculator`) ⇒ inerte.
3. **Chaîne de télémétrie** : `ev_norm/PF/WR → RewardCollector → JSONL`. Aucun
   retour vers le contrôle ni le gradient.
4. **Asymétrie BUY-gaté / SELL-non-gaté** = arête causale commune aux deux
   collapses (RAPPORT 7 approfondira la comparaison V30↔V31).
