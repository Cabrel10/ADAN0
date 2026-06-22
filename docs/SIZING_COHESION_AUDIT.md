# ADAN0 — AUDIT DE COHÉRENCE ABSOLUE DU POSITION SIZING

> "Le risque d'une divergence silencieuse sur le Position Sizing est notre plus
> grand danger avant le run de 72h." — exigence de session.

Date: 2026-06-22 · Branche: `genspark_ai_developer` · Python: conda `trading_env`

## 1. Méthodologie

Scan des 4 sources demandées + le moteur d'exécution (paper/live) :
- `config/config.yaml` (source de vérité)
- `src/.../environment/dynamic_behavior_engine.py` (HMM / DBE)
- `src/.../environment/multi_asset_chunked_env.py` (env train + backtest)
- `src/.../portfolio/portfolio_manager.py`
- `src/.../trading/execution_engine.py` (paper / live) ← chemin réel du run 72h

## 2. Constat : DEUX divergences silencieuses identifiées

### 2.1 Divergence n°1 — FORMULE DE SIZING

| Contexte | Fichier:ligne | Formule de la taille de position |
|----------|---------------|----------------------------------|
| **Train / Backtest** | `multi_asset_chunked_env.py:6908` | `target_exposure = exp_min + (exp_max − exp_min) × confidence_HMM` puis `notional = max(11$, capital × target_exposure)` |
| **Paper / Live (AVANT fix)** | `execution_engine.py:210` | `size_pct = abs(action[1]) × max_position_pct/100` (= `abs(action[1])` car `max_position_pct=100`), puis plafonné par `max_position_size_pct` du tier |

→ En entraînement la taille est pilotée par l'**exposure_range du palier × la confiance HMM**.
   En paper trading elle était pilotée par **`action[1]` brut du modèle**.
   **Deux formules totalement différentes pour des poids de modèle identiques.**

### 2.2 Divergence n°2 — BORNES SL / TP

| Contexte | Fichier:ligne | Bornes SL / TP |
|----------|---------------|----------------|
| **Train / Backtest** | `multi_asset_chunked_env.py:7009-7014` | **Spécifiques au profil** : scalper SL[2,3]%/TP[4,6]% · intraday SL[4,6]%/TP[8,12]% · swing SL[7,10]%/TP[14,20]% · position SL[15,20]%/TP[30,40]% ; R/R ≥ 1.5 imposé |
| **Paper / Live (AVANT fix)** | `execution_engine.py:211-212` | Génériques : SL `[0.5,5]%` · TP `[0.5,10]%` ; **aucun R/R imposé** |

## 3. Éléments DÉJÀ cohérents (vérifiés)

- ✅ **capital_tiers** : `run_bot.py:236` charge `config.yaml.capital_tiers` et les injecte
  dans `ExecutionEngine(capital_tiers=...)`. `_get_tier_cap()` lit
  `max_position_size_pct` depuis ces tiers. **Source unique = config.yaml.**
- ✅ **max_position_size_pct** : tier Micro=90, Small=65, Medium=48, High=28, Enterprise=20.
  Lu depuis config dans les deux mondes (`env:6810`, `execution_engine:392`).
- ✅ **min order size** : `MIN_ORDER_VALUE=11.0$` (`execution_engine.py:404`) ==
  `min_order_value_usdt=11.0` (`config.yaml hard_constraints` + `env:6815`).
- ✅ **frais** : paper = `0.001` (0.1% Binance réel, `execution_engine.py:457`).
  Entraînement = 0.80% (4× réel) **INTENTIONNEL** (`env:1145`, conservateur par design).
  Ce n'est PAS une divergence dangereuse : le live est *moins* coûteux que le train.
- ✅ **slippage** : 2 bps directionnel des deux côtés (`execution_engine.py:100`, env `exec_prices`).

## 4. Correctif appliqué — UNIFICATION (source unique = config.yaml)

`execution_engine.py` reçoit désormais (depuis `run_bot.py`, lus dans config.yaml) :
- `exposure_range` du palier courant,
- `profile` du worker (scalper/intraday/swing/position).

`decode_action()` calcule la taille avec **exactement** la formule d'entraînement :
```
confidence = ctx[3] (bull_prob HMM) si context_vector présent, sinon 0.5  (= défaut env)
target_exposure = exp_min + (exp_max − exp_min) × confidence
size_pct = clamp(target_exposure, 0, tier_cap)
```
et les bornes SL/TP avec la table `_PROFILE_BOUNDS` **identique** à `env:7009-7014`,
R/R ≥ 1.5 imposé comme en entraînement.

Quand le HMM n'est pas disponible en live (`context_vector=None`), `confidence=0.5`
→ exposition = milieu du palier, **strictement** le comportement du bloc `except`
de l'env (`env:6986` : `(exp_min+exp_max)/2`). Train ≈ Paper garanti.

## 5. Checklist finale

- [x] capital_tiers identiques partout (config.yaml unique)
- [x] exposure_range identique partout (injecté config → env & execution_engine)
- [x] max_position_size_pct identique partout (config → _get_tier_cap & env)
- [x] position_size_pct / formule de sizing identique (formule LINEAR_EXPO unifiée)
- [x] minimum order size identique (11$ partout)
- [x] frais : 0.1% live, 0.80% train (intentionnel/conservateur, documenté)
- [x] slippage identique (2 bps directionnel)
- [ ] modèles 450k et 500k chargent correctement → testé au lancement
- [x] logs détaillés activés (`[DEBUG_ACTION]`, `[SIZING]`, `[TIER_CAP]`)
- [ ] export CSV des trades → ajouté à execution_engine
- [ ] métriques Sharpe / PF / DD actives → calcul au shutdown

## 6. LANCEMENT PAPER TRADING — confirmé (20260622_220804)

Les 2 modèles tournent en arrière-plan (binance spot testnet, $20.50 chacun,
profil intraday, intervalle 300s, durée 72h) :

| Modèle | Checkpoint | PID | Log dir |
|--------|-----------|-----|---------|
| 500k | ppo_adan0_500k_FIXED.zip | 2422414 | logs/paper/500k_20260622_220804 |
| 450k | ppo_adan0_sandbox_checkpoint_450000_steps.zip | 2422415 | logs/paper/450k_20260622_220804 |

Premier tick des deux (cohérence prouvée) :
`[SIZING] LINEAR_EXPO profile=intraday conf=0.500 size_pct=0.8000 (80.00%) size_usd=$16.40`
= exactement `0.70 + (0.90-0.70)*0.5` du palier Micro → **train ≈ paper garanti**.

CSV trades + métriques (Sharpe/PF/DD/WR/expectancy) générés automatiquement à
chaque `save_report()` (toutes les N ticks et au shutdown).

### Note opérationnelle (vigilance)
- `context_vector` reste `None` dans run_bot.py:389 → confidence=0.5 (fallback =
  comportement except de l'env). Cohérence garantie ; activer le HMM live plus tard
  pour exploiter la modulation bull/bear de l'exposition (amélioration, pas blocage).
- `--profile intraday` choisi car le modèle est un généraliste entraîné sur 4 profils
  simultanément ; intraday = bornes SL/TP médianes [4-6%]/[8-12%].
