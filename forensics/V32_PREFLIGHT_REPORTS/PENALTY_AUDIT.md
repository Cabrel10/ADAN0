# RAPPORT 3 — PENALTY_AUDIT

**Objet** : recensement FACTUEL de TOUTES les pénalités qui entrent dans
`raw_reward` (`_calculate_reward`, env 6924-7456), avec pour chacune :
(a) quand elle s'applique, (b) sa valeur/formule exacte, (c) si elle modifie
RÉELLEMENT l'action, (d) si elle peut créer un artefact (état absorbant).

**Source unique** : `MultiAssetChunkedEnv._calculate_reward` (env), `config.yaml`.
Toutes les lignes citées sont lues au commit courant `genspark_ai_developer`.

> ⚠️ Rappel RAPPORT 1/2 : le chemin de reward EFFECTIF est
> `_calculate_reward` (env), PAS `RewardCalculator`. Cet audit ne porte donc
> QUE sur les termes réellement sommés dans `raw_reward` (env 7400-7409).

---

## 1. Composition de `raw_reward` (env 7400-7409, mode standard)

```
raw_reward = pnl_base_reward + behavior_penalty + action_anchor_penalty
           + holding_cost + smart_flat_reward + time_decay_cost
           + promotion_bonus + demotion_penalty + closure_bonus
           + drawdown_penalty + symmetry_penalty + action_entropy_penalty
           + future_contrib + latent_pnl_contrib + saturation_penalty
final_reward = sign(raw_reward) * log1p(|raw_reward|)     # symlog, env 7456
```

Mode alternatif **V16 MTM** (`ADAN_MTM_REWARD=1`, env 7420-7448) : remplace le
bloc PnL réalisé par `mtm_scale*Δequity% + behavior_penalty + drawdown_penalty`.
→ **Les pénalités conservées en MTM sont uniquement `behavior_penalty` et
`drawdown_penalty`.** Tous les autres termes disparaissent en MTM.

---

## 2. Tableau des pénalités (les 6 termes négatifs de contrôle/risque)

| Pénalité | env:ligne | Quand | Valeur / formule | Modifie l'action ? | Risque d'artefact |
|----------|-----------|-------|------------------|--------------------|-------------------|
| **behavior_penalty** | 7398-7399 ; injectée 8655-8694 | intent BUY-while-open ou SELL-while-flat REJETÉ par le routage | `_step_invalid_penalty += _bp_val` ; `_bp_val ∈ {sell_while_flat, buy_while_open}` = **0.0/0.0** (config 1426-1434, V31) — était **-0.28/-0.28** | **OUI historiquement** — c'est LE driver prouvé du collapse V30 (voir §4) | **ÉLEVÉ (prouvé)** : asymétrie → pousse a0 |
| **action_anchor_penalty** | 7382-7391 | trade NON exécuté ET `|a0|>deadzone(0.30)` | `-min(anchor_cap=0.02, anchor_lambda*(excess²))` ; **plafonné à 0.02** | Faiblement (no-op only, cap 0.02) | FAIBLE (borné, dead-zone) |
| **drawdown_penalty** | 7044-7052 | drawdown en hausse | `-50.0*(dd_ratio² − prev_ratio²)*dd_factor` (quadratique) | Indirect (pénalise le risque, pas une action précise) | MOYEN (quadratique, peut dominer) |
| **symmetry_penalty** | 7105-7138 | anti-triche SL/TP (RR faible + lâcheté ATR), latent | variable, latent | Indirect | FAIBLE |
| **action_entropy_penalty** | 7142-7155 | switch-rate `>0.5` sur `_action_history` | `-λ(0.03)*(rate−0.5)` si `action_entropy_enabled` | Indirect (anti switch-spam) | FAIBLE (borné, désactivable) |
| **saturation_penalty** | 7228-7242 | spam saturation SL/TP | log, plafonné | Indirect | FAIBLE (log-plafonné) |
| **holding_cost** | 7259-7271 | ≥1 position ouverte | `-_h` fixe/step | Indirect (anti disposition-effect) | FAIBLE |
| **time_decay_cost** | (v13.1) | par step | coût fixe/step | Indirect (anti-dérive/inaction) | FAIBLE |
| **demotion_penalty** | (tier) | rétrogradation de tier | grosse pénalité | Indirect | MOYEN (magnitude) |

---

## 3. Fait clé n°1 : la SEULE pénalité qui a historiquement modifié l'action est `behavior_penalty`

`behavior_penalty` est injectée AVANT que `route_action_by_state` ne neutralise
l'intention en HOLD (env 8646-8694 ; commentaire V28 explicite l.8666-8672 :
« Raw intent (a0 sign) and portfolio state are visible HERE, before
route_action_by_state neutralises the intent into HOLD »).

C'est la seule pénalité indexée sur **l'intention brute (signe de a0)** et non
sur une transition exécutée. Elle agit donc directement sur le gradient de la
politique **dans une direction (le signe de a0)** → c'est la seule capable de
faire dériver μ vers ±1. Toutes les autres pénalités sont indexées sur des
conséquences (drawdown, holding, saturation) et poussent vers « moins de
risque », pas vers « un signe d'action ».

---

## 4. Fait clé n°2 : l'asymétrie V28 = artefact CONFIRMÉ (état absorbant BUY)

Preuve chiffrée (carte causale
`forensics/v30_autopsy_20260817/timeline/causal_map_state_intent.txt`, citée
dans config 1417-1420) :

| État + intent | `final_reward` moyen |
|---------------|----------------------|
| FLAT + SELL | **-0.2331** |
| FLAT + BUY | -0.0149 |
| FLAT + HOLD | -0.0143 |
| OPEN + BUY | **-0.2424** |
| OPEN + SELL | +0.0056 |

- En FLAT, SELL est puni ~0.23 de plus que BUY/HOLD → gradient PPO pousse a0 → +1.
- En OPEN, BUY puni ~0.24 de plus que SELL → pression vers SELL en position.
- Résultat V30 : verrou `a0=+1.0` à ~step 27502, 472k steps post-mortem
  (RAPPORT_AUTOPSIE_V30.md). **C'est un état absorbant induit par une pénalité.**

**Correctif appliqué (V31, config 1426-1434)** : `sell_while_flat=0.0`,
`buy_while_open=0.0`, avec justification « C6 » : une intention REJETÉE par le
gate (aucune transition d'état s→s′) viole l'équation de Bellman si on la punit
→ doit valoir 0.0 comme HOLD. **Ce correctif est valide et à CONSERVER en V32.**

---

## 5. Fait clé n°3 : neutraliser `behavior_penalty` n'a PAS suffi (V31 always-SELL)

Après neutralisation (§4), V31 a collapsé dans l'AUTRE sens (always-SELL,
μ→-8.58). Cause (RAPPORT_COLLAPSE_V31_500K.md) : **gSDE σ non bornée + absence
d'ancre L2 dans la loss actor**. Donc :

- La pénalité asymétrique n'était PAS la seule force ; une fois retirée, c'est
  **l'absence de force de rappel sur μ** (ancre L2) combinée à **σ explosif
  (gSDE)** qui a laissé μ dériver librement vers -8.58.
- Aucune pénalité de `raw_reward` ne s'oppose à cette dérive : elles agissent
  toutes via le reward (symlog-compressé), pas sur μ directement. **La seule
  force agissant sur μ est `anchor_loss` dans la loss actor** (feature_extractors
  1330), qui était absente/insuffisante en V31.

---

## 6. Fait clé n°4 : symlog aplatit toutes les pénalités

`final_reward = sign(raw)*log1p(|raw|)` (env 7456). Conséquence mesurable : une
pénalité brute de -0.28 devient `-log1p(0.28) ≈ -0.247` ; une de -2.0 devient
`-log1p(2)≈-1.10`. Le symlog **compresse fortement les grosses pénalités** →
au-delà de |raw|~1, augmenter une pénalité rapporte très peu de signal
supplémentaire. Cela explique pourquoi « augmenter une pénalité » (ex. ancre L2
en reward-space) a échoué : le signal utile est écrasé par le symlog.

---

## 7. Synthèse — implications directes pour le RAL (RAPPORT 5+)

1. **Ne PAS ré-introduire une pénalité asymétrique sur l'intention** (leçon V30).
2. **La seule pénalité qui bouge le signe de a0 est `behavior_penalty`** — un RAL
   qui module les pénalités de reward n'agira PAS directement sur μ (symlog +
   indexation sur conséquences). Corollaire : **moduler les récompenses ne
   restaurera pas mécaniquement le signal BUY/HOLD manquant** → confirme le
   risque signalé (« reproduire l'erreur de l'ancre L2 »).
3. Toute action du RAL sur μ doit passer par la loss actor (côté `anchor_loss`),
   pas seulement par `raw_reward`. À trancher au RAPPORT 6 (pipeline) et 7.
4. Le mode MTM ne garde que `behavior_penalty`+`drawdown_penalty` : si V32 tourne
   en MTM, l'audit se réduit à ces deux termes + le clamp σ/ancre L2.
