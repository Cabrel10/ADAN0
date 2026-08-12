# ADAN — PHASE 2 : Traçage complet action → rejet (lecture seule, GEL respecté)

**Date** : 2026-08-12
**Statut** : PHASE 1 ✓ (contrat capital_tiers) → **ce document = PHASE 2** → PHASE 3 (invariants automatisés) à suivre.
**Méthode** : lecture du code + logs réels du run `v28_rewardfix_20260811T140655Z`. Aucune modification de code, aucun entraînement.

---

## 1. Le graphe complet (runtime réel, `MultiAssetChunkedEnv._execute_trades`)

```
RÉSEAU (policy PPO)
  │  action = 25 dims = 5 assets × [a0, size_raw, tf_raw, sl_raw, tp_raw]
  ▼
TIER LOCKING (env l.8537)  ← _locked_tier figé au reset d'épisode
  │  CAPITAL TIER SUPREMACY (l.8525) :
  │  min_exp, max_exp, max_risk_pct lus EXCLUSIVEMENT du tier (/100)
  │  jamais du worker_config.
  ▼
SLOT DISPONIBLE (l.8627) : _slot_available = n_open < max_concurrent_positions (tier)
  ▼
ROUTING D'ÉTAT (action_routing.py::route_action_by_state, appelé l.8641)
  │  FLAT : a0 > +thr  → BUY   | sinon (même a0=-1.0) → HOLD neutre
  │  LONG : a0 < -sthr → SELL  | sinon (même a0=+1.0) → HOLD neutre
  │  FLAT + slot indisponible → HOLD forcé (portfolio_reject: tier_slot_unavailable)
  │  ── classification si HOLD (l.8648-8656) :
  │     |a0| ≤ thr            → deadband_reject   (télémétrie pipeline)
  │     intent ≠ état         → routing_reject    (sell_while_flat / buy_while_long)
  │  ── V28 : behavior_penalties ADDITIVES sur routing_reject
  │     (sell_while_flat / buy_while_open, config reward_shaping.behavior_penalties)
  │     avec CAUSAL GUARD : pas de pénalité si SL/TP/MaxDuration a fermé ce step.
  ▼
ANTI-SPAM HOLD (l.8754) : BUY-while-open & |Δexposure| < 10% → HOLD neutre
  │  (branche MORTE sous routing v12 : BUY n'est jamais routé en position)
  ▼
SIZING — ⚠️ DOUBLE CHEMIN CONCURRENT :
  │  l.8719 : target_exposure = min_exp + normalized(size_raw) × (max_exp-min_exp)
  │           ← utilise l'action de l'agent (dim size_raw)
  │  l.8745 : target_exposure = exp_min + (exp_max-exp_min) × bull_prob_HMM
  │           ← ÉCRASE la valeur précédente. **Le sizing de l'agent est ignoré.**
  │  LINEAR_EXPO garantit mathématiquement exposure ∈ [0.70, 0.90] (Micro).
  ▼
SIZE_GATE (l.8778-8854)
  │  notional = max(min_order 11$, capital_cash × exposure)
  │  notional > cash → notional = cash   (clamp cash, opérationnel)
  │  notional < 11$ → PROB_SIZER Bernoulli (arrondi à 11$ avec p=notional/11)
  │                   sinon HOLD neutre (min_notional Cas A/B, 0 pénalité v12)
  ▼
SL/TP dynamique borné par profil + hard_constraints (SL∈[0.3%,6%], TP∈[0.5%,12%])
  ▼
┌─ SELL (discrete==2, position ouverte) ─────────────────────────────┐
│ HOLD_MIN (l.9073) : steps_held < hold_min_steps[tf] (5m=6)          │
│   → cooldown_hold_min                                              │
│ HYSTERESIS #1 (l.9109) : exposure < 5% → HOLD (« trop petit pour   │
│   valoir les frais ») → hysteresis                                 │
│ AGENT_CLOSE BARRIER (l.9125+) : seuil break-even dynamique =       │
│   1.5 × frais aller-retour. unrealized_pnl < seuil                 │
│   → _resolve_agent_close_gate → hysteresis                         │
│     (reasons: below_break_even_barrier / decision_budget_or_quota) │
└────────────────────────────────────────────────────────────────────┘
┌─ BUY (discrete==1, pas de position) ───────────────────────────────┐
│ WAIT_BLOCK (l.9429) : steps_since_sell < wait_steps (5m=6)          │
│   → cooldown_wait                                                  │
│ RISK_GATE (l.9457) : n_open ≥ max_concurrent_positions (tier)      │
│   → risk_gate          ✅ seul gate DIRECTEMENT adossé au tier     │
│ OMEGA-4E cooldown (l.9479) : délai min entre 2 BUY (profil)        │
│   → cooldown_omega4e                                               │
│ DAILY_LIMIT (l.9486) → daily_limit                                 │
│ EV_GATE (l.9504) : p_min = (1 + fees/SL)/(1 + RR), adouci ×0.85    │
│   en training ; p_hmm ≤ p_min → fee_gate                           │
│ CASH & SURVIVAL (l.9556) : notional < 11$ ou cash insuffisant      │
│   → min_notional                                                   │
└────────────────────────────────────────────────────────────────────┘
  ▼
TRADE_OPEN (open_position, fill = open[t+1] + slippage 2bps)
```

---

## 2. Matrice exhaustive (format demandé)

| Étape | Entrée | Transformation | Peut modifier le domaine tier ? | Peut rejeter ? | Type |
|---|---|---|---|---|---|
| Agent (réseau) | a0, size_raw… | — | non | non | décision |
| Tier locking | capital cash | sélection tier | non (figé épisode) | non | contrat |
| Routing d'état | a0, in_position, slot | a0 → HOLD/BUY/SELL | non | neutralise (HOLD) | routing_état |
| Behavior penalties (V28) | intent vs état | pénalité additive | non | non (reward seul) | reward |
| Anti-spam HOLD | BUY-while-open, Δexp | override HOLD | non | neutralise | redondance |
| **Sizing agent (l.8719)** | size_raw | → exposure | **oui (dans [70,90])** | non | **écrasé** |
| **LINEAR_EXPO (l.8745)** | bull_prob_HMM | → exposure ∈ [70,90] | **non (borné tier)** | non | sizing final |
| SIZE_GATE | cash, min_order | notional clamp cash | non (cash ≠ tier) | HOLD (prob/neutre) | opérationnel |
| HOLD_MIN | steps_held | blocage temporel | non | cooldown_hold_min | opérationnel |
| HYSTERESIS #1 | exposure < 5% | HOLD | non | hysteresis | économique (SELL) |
| AGENT_CLOSE barrier | PnL < 1.5×frais | HOLD + trace | non | hysteresis | économique (SELL) |
| WAIT_BLOCK | steps_since_sell | blocage temporel | non | cooldown_wait | opérationnel |
| RISK_GATE | n_open vs tier | hard gate | non | risk_gate | **contrat tier** |
| OMEGA-4E | délai entre BUY | blocage temporel | non | cooldown_omega4e | opérationnel |
| DAILY_LIMIT | quota journalier | blocage | non | daily_limit | opérationnel |
| EV_GATE | p_hmm vs p_min | rejet économique | non | fee_gate | économique |
| CASH & SURVIVAL | cash, notional | rejet | non | min_notional | opérationnel |

**Conclusion matrice** : aucun module aval ne modifie le domaine `capital_tiers`. Le seul gate directement adossé au tier est `RISK_GATE` (max_concurrent). L'exposition est bornée par LINEAR_EXPO **dans** la plage tier — jamais en dehors. Le contrat est structurellement respecté dans ce chemin.

---

## 3. La distinction qui manquait au diagnostic

`illegal_ratio` (télémétrie) = **somme de toutes les `rejection_reasons`** / steps (train_parallel_agents.py l.706,774). Ce compteur **confond 4 catégories sémantiquement différentes** :

```
VIOLATION DU CONTRAT CAPITAL_TIER   → quasi INEXISTANTE (risk_gate ≈ 0 dans les logs)
REJET ÉCONOMIQUE                    → hysteresis + fee_gate  ≈ 76 % des rejets
REJET OPÉRATIONNEL (temps/quota)    → cooldown_* + daily     ≈ 12 %
ROUTING D'ÉTAT / HOLD NEUTRE        → compté dans pipeline, pas toujours dans reasons
```

### Preuve par les logs réels V28

`[EPISODE_REJECTIONS]` (fin d'épisode, run 20260811) :

```
Worker 0 | {'fee_gate': 42, 'risk_gate': 0, 'cooldown_wait': 1,
  'cooldown_hold_min': 57, 'cooldown_omega4e': 0, 'min_notional': 0,
  'hysteresis': 327, 'anti_spam_hold': 0, 'daily_limit': 0,
  'pm_rejected': 0, 'sell_no_position': 0} | trade_attempts=79 invalid=100
```

`[ACTION_DIFF]` (step 1500) — décomposition du pipeline :

```
policy=1500 → deadband_reject=269 (18%) | routing_reject=622 (41%)
  | budget_reject=395 (26%) | trade_executed=129 (8.6%)
```

### Lecture

| Raison | Mécanisme | Nature | Compte dans illegal_ratio ? |
|---|---|---|---|
| **hysteresis (~330-440/ép)** | SELL bloqué : exposure < 5% OU PnL < 1.5×frais (AGENT_CLOSE barrier) | économique SELL | **oui — dominant** |
| **fee_gate (~30-42/ép)** | EV<0 : p_hmm ≤ p_min | économique BUY | oui |
| **cooldown_hold_min (~52/ép)** | SELL < 6 steps après BUY (5m) | opérationnel | oui |
| **routing_reject (41% des steps)** | intent ≠ état (sell_while_flat…) | routing neutre | partiellement (via reasons) |
| **deadband_reject (18%)** | \|a0\| ≤ seuil | HOLD normal | non (pipeline seul) |
| **risk_gate (0)** | dépassement slots tier | contrat tier | oui mais ≈ 0 |

---

## 4. Ce que PHASE 2 établit (et réfute)

1. **RÉFUTÉ** : « les rejets viennent d'un rétrécissement du domaine tier par hard_constraints ». Les 2 clés incriminées (`max_position_size_pct: 0.5`, `max_risk_per_trade_pct: 0.02`) sont **du dead config** — aucun consommateur dans `src/`.
2. **CONFIRMÉ** : `illegal_ratio ≈ 94 %` mesure majoritairement des **frictions économiques SELL** (`hysteresis`) — l'agent demande à sortir, la barrière break-even (1.5× frais) ou le plancher d'exposition 5% le bloque.
3. **ANOMALIE ARCHITECTURALE MAJEURE** : le sizing issu de `size_raw` (l.8719) est calculé puis **écrasé** par LINEAR_EXPO×HMM (l.8745). L'agent ne contrôle pas son exposition — elle est dictée par `bull_prob_HMM`. Conséquence directe : **la dimension size de la politique n'a aucun gradient utile**, et `size_sat_frac` / `size_post_mean` de la télémétrie actiondim décrivent un canal mort.
4. **SIGNAL FANTÔME** : à 20.5 $ avec exposure [70%,90%] → notional 14-18 $ ; le plancher `min_order 11$` (= 54 % du capital) n'est jamais atteint en pratique (exposition tier ≫ plancher) — le PROB_SIZER ne sert qu'en fin de drawdown.
5. **SELL structuralement piégé** : pour clôturer, il faut (a) routing LONG + a0 < -thr, (b) HOLD_MIN passé, (c) exposure ≥ 5%, (d) PnL ≥ 1.5× frais. La condition (d) transforme AGENT_CLOSE en quasi-TP obligatoire → cohérent avec le diagnostic antérieur « 0% TP atteint, 100% AGENT_CLOSE » **inversé** en V28 : désormais la barrière bloque les micro-closes, d'où `hysteresis` dominant.

---

## 5. Hypothèses révisées (fin PHASE 2)

| Hypothèse | Statut |
|---|---|
| A — hard_constraints cause les rejets | ❌ **réfutée** (dead config) |
| B — clamps/transformation violent le contrat tier | ❌ non dans le chemin runtime (clamps présents uniquement dans `portfolio_manager.calculate_final_trade_parameters`, **jamais appelé** par l'env) |
| C — plusieurs chemins de sizing concurrents | ✅ **confirmée** (l.8719 vs l.8745 ; l'agent perd) |
| D — bug d'unité `×90` dans action_translator | ✅ confirmée dans le code mais **code mort** (non importé) |
| E — illegal_ratio = violations du contrat tier | ❌ **réfutée** : illegal_ratio = frictions économiques + opérationnelles. Le contrat tier n'est presque jamais violé (risk_gate ≈ 0) |

### Nouvelle lecture causale de V28

```
log_std non borné (σ ×14)
  → a0 saturé à ±1 (direction_sat_frac 1.00)
  → intents extrêmes : FLAT → toujours BUY (a0=+1 > thr) ; LONG → toujours SELL (a0=-1)
  → SELL demandé en boucle mais bloqué par AGENT_CLOSE barrier (PnL < 1.5×frais)
     → hysteresis monte (dominant)
  → trades exécutés rares (8.6 %) et perdants (SELL winrate 0.94 % sur fenêtre 100→6100)
  → reward décorrélé (corr 0.30) car dominé par saturation_penalty (-0.125 mean)
```

---

## 6. Ce que PHASE 3 devra prouver (invariants automatisés)

Tests à valeurs contrôlées, sans entraînement :

```
T1  exposure ∈ {0.69, 0.70, 0.75, 0.90, 0.91}
    → LINEAR_EXPO produit TOUJOURS ∈ [0.70, 0.90] (par construction) — vérifier
      qu'aucun chemin (fallback l.8869, PROB_SIZER l.8790) ne sort de la plage.
T2  RISK_GATE : n_open = max_concurrent → BUY rejeté, n_open < max → passe.
T3  AGENT_CLOSE barrier : PnL < 1.5×frais → SELL bloqué (hysteresis) ;
    PnL ≥ seuil → SELL autorisé.
T4  routing : FLAT + a0 ∈ {-1, 0, +1} → {HOLD, HOLD, BUY} ;
              LONG + a0 ∈ {-1, 0, +1} → {SELL, HOLD, HOLD}.
T5  size_raw ∈ {-1, 0, +1} → prouver que l'exposition finale NE DÉPEND PAS de
    size_raw (anomalie C) — test d'identité entre deux runs à size_raw opposés.
T6  illegal_ratio : vérifier que la télémétrie distingue violation contrat /
    rejet économique / rejet opérationnel (actuellement : NON → fix télémétrie V29).
```

> **Règle inchangée** : un seul invariant contractuel FAIL → TRAINING ABORTED. Aucune modification de code n'a été faite en PHASE 1–2 (GEL respecté). PHASE 4 (correction minimale) ne commencera qu'après la matrice de conformité PHASE 3 validée.

---

## 7. Statut d'avancement

- [x] PHASE 0 GEL
- [x] PHASE 1 audit contrat capital_tiers (section du rapport 7-gates, commit `0cf5ea6`)
- [x] PHASE 2 traçage action → rejet (ce document)
- [ ] PHASE 3 tests d'invariants automatisés (T1–T6)
- [ ] PHASE 4 correction minimale (conditionnée aux preuves)
- [ ] PHASE 5 re-tests
- [ ] PHASE 6 training de validation (smoke 2048, gates)
- [ ] PHASE 7 500k
- [ ] PHASE 8 rapport + décision
