# ADAN V28 — Diagnostic Quant 7 Gates (run 500k terminé)

**Run analysé** : `v28_rewardfix_20260811T140655Z` — **500,224 steps COMPLETED**, 977 rollouts diag.
**Run de contrôle** : `v28_rewardfix_20260810T200536Z` (mort à ~351 rollouts).
**Date du rapport** : 2026-08-12.
**Sources** : `logs/training/diag_500k_*20260811T140655Z.csv` (977 rows), `reward_components_*20260811T140655Z.csv` (4951 rows, steps 100→6100), `actiondim_*20260811T140655Z.csv` (977 rows, complet).

> ⚠️ **Note télémétrie** : `reward_components.csv` ne couvre que les steps **100 → 6100**. Le portfolio et la corrélation reward/PnL ne sont donc mesurables que sur cette fenêtre. C'est un **trou de télémétrie** à corriger en V29 (cf. §Prescription).

---

## Verdict global

**V28 = ÉCHEC STRUCTUREL CONFIRMÉ.** Le run a atteint 500k steps, mais l'agent est entré en rupture de variance à **~90k steps** et n'a jamais récupéré. Les ~410k steps suivants ont produit un agent cliniquement instable : 94% d'actions illégales, saturation complète de l'espace d'action, reward décorrélé du PnL.

V28 reste le run **le plus informatif** à ce jour : il a guéri la paralysie SELL de V25–V27 et révélé la maladie suivante (absence de contrôle de variance + reward mal aligné).

---

## GATE 1 — Intégrité du checkpoint : ✅ PASS

- Checkpoint zip intègre : 6 entries, 7.44 MB, aucune corruption détectée.
- Chargement OK.

## GATE 2 — Performance économique : 🔴 ÉCHEC

Fenêtre mesurable : steps 100→6100 (reward_components).

| Métrique | Valeur |
|----------|--------|
| Portfolio initial → final | 20.50 → 19.40 (**-5.4%**) |
| Portfolio minimum | **12.32** (-40% drawdown) |
| pnl_base mean | **-0.00538** |

**PnL par action** (steps 100→6100) :

| Action | n | mean_pnl | winrate | mean_reward |
|--------|---|----------|---------|-------------|
| BUY | 2523 | -0.00048 | 0.0% | -0.01468 |
| HOLD | 82 | 0.00000 | 0.0% | -0.01283 |
| SELL | 2346 | **-0.01084** | **0.94%** | -0.00749 |

→ SELL est l'action la plus perdante (winrate < 1%), mais reçoit le reward le moins négatif. Le signal est inversé.

## GATE 3 — (réservé)

## GATE 4 — (réservé)

## GATE 5 — Corrélation reward ↔ PnL : 🔴 DÉCORRÉLÉ

```
corr(final_reward, pnl_base) = 0.3028
```

**C'est le chiffre le plus important du run.** L'agent peut recevoir une récompense positive pendant qu'il perd de l'argent, et une pénalité pendant qu'il en gagne. Aucun algorithme d'apprentissage ne peut converger vers une stratégie rentable dans ces conditions.

Magnitude des pénalités (steps 100→6100) :

| Composante | mean | min |
|-----------|------|-----|
| saturation_penalty | **-0.1249** | -0.3584 |
| behavior_invalid_penalty | -0.0087 | -0.28 |
| drawdown_penalty | -0.0008 | -0.1415 |
| closure_bonus | -0.0050 | -0.2 |

→ `saturation_penalty` (mean -0.125) est **14× plus gros** que `behavior_invalid_penalty` (mean -0.009). Le reward shaping pousse l'agent à minimiser la saturation, pas à faire du profit.

## GATE 6 — Stabilité PPO / distribution d'actions : 🔴 INSTABLE

### Rupture a0_std (reproduite dans les 2 runs V28)

| Run | Rupture a0_std > 1.0 | Rollouts |
|-----|----------------------|----------|
| Mort (20260810) | step **79,872** | 351 |
| Complet (20260811) | step **90,112** | 977 |

→ Pathologie **structurelle**, pas un artefact.

### Trajectoire de la mort (run complet)

| Phase | a0_std médian | illegal% | clip% | Verdict |
|-------|---------------|----------|-------|---------|
| 0–100k | **0.62** | 71% | 37% | ✅ Sain |
| 100–200k | **1.95** | 94% | — | ⚠️ Rupture |
| 200–300k | **5.65** | 94% | — | 🔴 Chaos |
| 300–400k | **7.59** | 94% | — | 🔴 Chaos |
| 400–500k | **8.64** | 94% | 70% | 🔴 Chaos |

**Multiplicateur σ : 0.62 → 8.64 = ×14.**

### Métriques PPO fin de run (400–500k) vs début (0–100k)

| Métrique | 0–100k | 400–500k | Cible saine |
|----------|--------|----------|-------------|
| approx_kl | 0.051 | **0.508** | 0.01–0.03 |
| clip_fraction | 0.367 | **0.700** | < 0.3 |
| policy_entropy | 0.426 | 0.523 | stable |
| critic_explained_variance | ~0.15–0.20 | **0.139** | > 0.5 |
| illegal_ratio | 0.709 | **0.942** | < 0.3 |

### Saturation de l'espace d'action (actiondim, run complet)

| Métrique | début (50 prem.) | fin (50 dern.) |
|----------|------------------|----------------|
| direction_sat_frac | 0.009 | **1.000** (100% saturation) |
| direction_post_mean | 0.06 | **-1.773** |
| tp_post_mean | 0.016 | **3.041** (take-profit explose) |

## GATE 7 — Infrastructure : ✅ PASS

- 34 GB libres sur disque.
- Aucun run actif au moment du diagnostic.

---

## Synthèse causale

1. Le fix V28 (behavior_penalties SELL-flat = -0.28) a **brisé la prison SELL** de V27 : l'agent prend des positions, explore, vit dans le marché. C'est un vrai progrès.
2. Mais guérir la paralysie a révélé l'**absence de contrôle de variance** : `log_std` non borné → σ ×14 → l'agent hurle des ordres à ±5σ.
3. **94–96% des actions sont rejetées** par le pipeline de validation → aucun trade réel → le critic n'apprend pas (EV ~0.14) → l'acteur monte σ pour explorer → spirale.
4. Le **reward est décorrélé du PnL** (corr 0.30), dominé par `saturation_penalty` → le modèle optimise un proxy, pas le profit.

---

## 🔴 SECTION CAPITAL_TIERS — le contrat normatif (référentiel absolu)

> Ajouté après la synthèse utilisateur du 2026-08-12. Cette section **prime** sur toute interprétation antérieure du diagnostic.

### Le principe

`capital_tiers` n'est **pas** un plafond de sécurité parmi d'autres. C'est **le référentiel normatif de l'espace d'action légal**. Pour un tier donné, il définit l'ensemble des plages de comportement autorisées, et :

> **Aucun module aval (DBE, volatility_guard, hard_constraints) n'a le droit de réduire ni d'augmenter ces bornes. Tout dépassement — par le haut OU par le bas — est une VIOLATION, qui doit être REJETÉE, jamais corrigée silencieusement.**

Pour Micro (capital 11–30 $) :

```
exposure_range        : [70, 90]   → l'exposition NORMALE est l'intervalle [70%, 90%]
max_position_size_pct : 90
risk_per_trade_pct    : 4.0
max_concurrent_positions : 1
drawdown_limit        : 40% (training)
leverage              : 1
```

Donc :
- `exposure = 60%` → **VIOLATION** (< min 70%)
- `exposure = 95%` → **VIOLATION** (> max 90%)
- `risk = 2.5%` ou `risk = 5%` → **VIOLATION** (≠ 4%)
- Un module qui fait `exposure × 0.6 = 42–54%` → **VIOLATION** (sort du domaine légal)

### Architecture attendue

```
              CAPITAL_TIER (autorité normative)
                      │
             ESPACE D'ACTION LÉGAL
        ┌────────────────────────────┐
        │ exposure ∈ [70%, 90%]      │
        │ risk      = 4%             │
        │ position  ≤ 90%            │
        │ concurrent ≤ 1             │
        └────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
       HMM           DBE      Volatility Guard
        │             │             │
        └─────────────┼─────────────┘
                      ▼
              VALIDATION TIER
                      │
              ┌───────┴───────┐
              ▼               ▼
            LÉGAL          VIOLATION
              │               │
              ▼               ▼
           EXECUTE          REJECT
```

**Anti-pattern à détecter dans le code** : tout `min()` / `max()` appliqué par un module secondaire qui **ramène silencieusement** une valeur dans la plage du tier. Cela **cache la violation** au lieu de la rejeter :

```python
# BUG POTENTIEL (cache la violation) :
exposure = max(tier.min_exposure, exposure)   # transforme 64% → 70% silencieusement

# CORRECT (rejette) :
if exposure < tier.min_exposure or exposure > tier.max_exposure:
    reject("TIER_VIOLATION")
```

### ⚠️ Caveat unités — exigence préalable

> **0.5 ne peut pas être affirmé comme 50 % ou 0,5 % sans vérifier l'unité utilisée par le code. Même chose pour 0.02. Le premier travail de l'audit est de retrouver les conversions exactes dans le code, pas de partir de la valeur YAML seule.**

### Violations de configuration SUSPECTÉES (à confirmer par le code)

| # | Tier Micro | Config concurrente | Violation suspectée |
|---|-----------|--------------------|---------------------|
| 1 | `max_position_size_pct: 90` | `hard_constraints.max_position_size_pct: 0.5` | Si 0.5 = 50% → plafond réduit 90→50. Si 0.5% → 0.10 $/position, impossible avec min_order 11 $. **Unité à déterminer.** |
| 2 | `risk_per_trade_pct: 4.0` | `hard_constraints.max_risk_per_trade_pct: 0.02` + `position_sizing.max_risk_per_trade_pct: 0.01` | Réduction en cascade 4% → 2% → 1%. |
| 3 | `min_capital: 11.0` | `hard_constraints.min_order_value_usdt: 11.0` | 11 $ = **54% du capital 20.5 $** → force une exposition minimale de 54% par trade, vide le tier de sa substance. |

### Incohérences structurelles (non-tier mais dangereuses)

| # | Problème | Détail |
|---|----------|--------|
| 4 | `features_config` dupliqué | `data.features_config` (5m: 16 indicateurs) vs `environment.features_config` (7 indicateurs). Si le code lit le mauvais bloc, l'espace d'observation est tronqué. |
| 5 | Deux seuils d'action | `environment.action_thresholds.5m: 0.10` vs `timeframe_trading_config.5m.action_threshold: 0.01` (facteur 10). |
| 6 | `invalid_action: -1.0` | vs mécanisme stérile V28 (-0.28). Si actif, pénalité 3.5× trop forte → paralysie ou frénésie. |

### Lien causal avec les symptômes V28

> « cela expliquerait pourquoi les positions de l'agent sont très majoritairement rejetées car en 50k steps le modèle serait déjà capable de poser des positions dans l'intervalle et éviter des pénalités bêtes. »

**Hypothèse crédible** : si le pipeline rétrécit l'espace légal `capital_tiers` **après** la décision du modèle (hard_constraints 0.5 vs tier 90, volatility_guard ×0.6, etc.), alors le modèle apprend correctement à viser [70%, 90%] mais se fait rejeter par un validateur qui applique une autre politique. Résultat : illegal_ratio 94–96% **malgré** un apprentissage correct.

**À PROUVER par l'audit du code (PHASE 1).**

---

## Stratégie imposée — 8 phases

```
PHASE 0 — GEL
   aucun entraînement, aucune modification du modèle,
   aucun changement arbitraire de config
        ↓
PHASE 1 — AUDIT DU CONTRAT CAPITAL_TIERS
   trouver loader tier, calcul exposure (LINEAR_EXPO, bull_prob),
   position sizing, DBE, volatility_guard, validation/rejet.
   DÉTERMINER LES UNITÉS EXACTES DANS LE CODE (0.5 = ratio ou % ?)
        ↓
PHASE 2 — TRAÇAGE COMPLET DE L'ACTION
   capital_tiers → paramètres effectifs → agent action → DBE →
   volatility_guard → hard_constraints → validator → TRADE/REJECT
        ↓
PHASE 3 — TESTS D'INVARIANTS
        ↓
PHASE 4 — CORRECTION MINIMALE
        ↓
PHASE 5 — RE-RUN DES TESTS
        ↓
PHASE 6 — TRAINING DE VALIDATION
        ↓
PHASE 7 — 500k STEPS
        ↓
PHASE 8 — RAPPORT + DÉCISION
```

### Invariants obligatoires (par tier)

```
min_exposure        <= exposure_final      <= max_exposure
risk_final          <= risk_per_trade_pct
position_size_final <= max_position_size_pct
concurrent_positions <= max_concurrent_positions
```

**Et** : aucun module secondaire ne doit silencieusement transformer les bornes du tier.

### Matrice de conformité à produire (PHASE 3)

| Tier | Paramètre | Config tier | Valeur finale | Module modificateur | Conforme |
|------|-----------|-------------|---------------|---------------------|----------|
| Micro | exposure min | 70% | ? | ? | ✅/❌ |
| Micro | exposure max | 90% | ? | ? | ✅/❌ |
| Micro | risk/trade | 4% | ? | ? | ✅/❌ |
| Micro | position max | 90% | ? | ? | ✅/❌ |
| Small | exposure min | 35% | ? | ? | ✅/❌ |
| ... | ... | ... | ... | ... | ... |

### Checklist pré-training (toutes PASS requises)

```
CONFIG CONTRACT:         PASS/FAIL
ACTION SPACE:            PASS/FAIL
FEATURE SCHEMA:          PASS/FAIL
POSITION SIZING:         PASS/FAIL
RISK SIZING:             PASS/FAIL
VALIDATION PIPELINE:     PASS/FAIL
REWARD/PENALTY:          PASS/FAIL
TRAIN/VAL/TEST SCHEMA:   PASS/FAIL
```

> **Si un seul est FAIL → TRAINING ABORTED.** Pas de « on lance quand même 50k pour voir ». Faire tourner des steps avec un environnement incohérent ne produit pas de connaissance : ça produit un modèle qui apprend les défauts du simulateur.

### Méthodologie

> **Audite → prouve → corrige minimalement → revalide → entraîne uniquement si tous les invariants passent.**

---

## Prescription V29 (à appliquer UNIQUEMENT après PHASE 3 PASS)

Correctifs **minimaux**, conditionnés à la preuve de l'audit :

| # | Correctif | Cible | Condition |
|---|-----------|-------|-----------|
| 1 | **Borner log_std ∈ [-2, 0]** (σ ∈ [0.135, 1.0]) | architecture PPO (clip dur, PAS reward) | toujours |
| 2 | **Réduire saturation_penalty** -0.12 → ~-0.02 | reward_calculator | toujours |
| 3 | **Ajouter exit quality** (SELL récompensé si pnl_sell > pnl_hold) | reward_calculator | toujours |
| 4 | **Réparer le pipeline de validation** pour respecter capital_tiers | validation/sizing | si PHASE 1–3 confirment la violation |
| 5 | **Aligner reward sur PnL** — condition absolue `corr(reward, pnl) > 0.50` avant tout 500k | reward | toujours |
| 6 | **Télémétrie pnl_base/portfolio sur TOUT le run** (pas 100→6100) + distinguer illegal_ratio (action_out_of_band vs order_rejected_by_env) | télémétrie | toujours |

### Gates avant le 500k

| Gate | Test | Décision |
|------|------|----------|
| G1 | checkpoint/env chargeables | KO → pas de run |
| G2 | corr(reward, PnL) > 0.50 (mesuré AVANT le moindre smoke, sur 1000 steps déterministes) | faible → pas de 500k |
| G3 | illegal_ratio maîtrisé (< 50% au smoke 2048) | non → pas de 500k |
| G4 | action distribution non saturée (< 80%) | saturation → pas de 500k |
| G5 | PPO stable sur 50k (a0_std < 1.0, KL/clip maîtrisés) | instable → pas de 500k |

### Smoke test 2048 — gates automatiques

```
a0_std < 1.0  ET  illegal_ratio < 50%  ET  saturation < 80%
   → sinon STOP, corriger avant tout run long
```

---

## Ce qu'il NE faut PAS faire

- ❌ Relancer V28 tel quel.
- ❌ Ajouter de nouvelles features (HMM, attention, etc.).
- ❌ Modifier l'architecture CNN/FiLM.
- ❌ Changer les hyperparamètres PPO (lr, gamma…) sans comprendre pourquoi.
- ❌ Entraîner avant que tous les invariants capital_tiers passent.

Le problème n'est pas architectural : c'est un paramètre non borné (log_std) + un reward mal calibré (saturation_penalty) + **potentiellement** un pipeline de validation qui viole le contrat capital_tiers.

---

## Statut d'avancement (2026-08-12)

- [x] GATE 1–7 diagnostic complet (ce rapport)
- [x] PHASE 0 GEL en vigueur (aucun training lancé)
- [ ] PHASE 1 audit capital_tiers dans le code (loader tier, sizing, unités)
- [ ] PHASE 2 traçage action complet
- [ ] PHASE 3 tests d'invariants + matrice de conformité
- [ ] PHASES 4–8 (correction → re-tests → validation → 500k → rapport)
