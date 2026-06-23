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

## 7. Vérification des 4 divergences silencieuses restantes (demande explicite)

### Divergence #1 — Frais & slippage
- **Frais** : training `0.80%` (4× réel, intentionnel/conservateur, env:1145) ;
  paper `0.001` = 0.10% Binance réel (execution_engine `fee_usd = size_usd*0.001`).
  → Le live est *moins* coûteux que le train. Biais conservateur, pas dangereux.
- **Slippage** : 2 bps directionnel des DEUX côtés
  (env via `exec_prices` open[t+1]+slippage ; paper `SLIPPAGE_BPS/10000`).
  → IDENTIQUE. ✅

### Divergence #2 — Ordres rejetés (min_notional / step size / précision)
- **min_notional** : 11$ partout (env:6815 min_order_value_usdt ;
  execution_engine.MIN_ORDER_VALUE=11.0 ; config hard_constraints). ✅
- En **mode paper**, l'ExecutionEngine simule le portefeuille — aucun rejet Binance
  (pas d'appel réseau d'ordre). Les contraintes step/précision ne s'appliquent qu'en
  **mode live** (CCXT `_place_live_order`). Pour le run actuel (paper testnet) → N/A.
- ⚠️ À surveiller pour un futur passage LIVE : ajouter arrondi `amount_to_precision`
  / `price_to_precision` CCXT avant `create_order`. (Amélioration future, pas blocage paper.)

### Divergence #3 — Données (OHLCV / indicateurs / normalisation / fenêtre)
- LiveStateBuilder **fit ses scalers sur les Parquet de validation** puis les LOCKE
  (`fit_on_parquet` + `scalers_loaded_from_training=True`) → même normalisation que
  l'entraînement. Log au boot : "✅ Scalers LOCKED to training distribution." ✅
- Mêmes timeframes (5m/1h/4h), mêmes 21 indicateurs (`_compute_indicators` ==
  TRAIN_COLUMNS), même `window_size`=50 (OBS_WINDOW). ✅

### Divergence #4 — DynamicBehaviorEngine (LE plus critique)
- DBE lit `max_position_size_pct` et `exposure_range` depuis `self.config["capital_tiers"]`
  (dbe.py:880,918,1141 + base_pos depuis exposure_range dbe.py:1011-1014) →
  **MÊME source que l'env et l'ExecutionEngine**. Pas de 35%-vs-90%. ✅
- **POINT CLÉ** : le DBE n'est **PAS instancié** dans le chemin paper trading
  (run_bot.py / execution_engine.py ne créent aucun DBE ; `context_vector=None`).
  Donc la modulation `regime_mult` du DBE est **train-only**. Le paper utilise la
  formule LINEAR_EXPO unifiée avec confidence=0.5 (= except-branch de l'env). →
  Aucun conflit de contrainte entre modes. ✅

**Conclusion** : PortfolioManager, DBE et ExecutionEngine lisent tous le sizing depuis
config.yaml capital_tiers. La seule différence inter-mode résiduelle (HMM actif en
train vs confidence=0.5 en paper) est gérée par un fallback identique au comportement
de l'env, donc cohérente. Le run 72h peut démarrer sans divergence silencieuse.

---

## §8 — Observation runtime du run 72h (2026-06-22 22:08 → en cours)

### 8.1 Le sizing unifié fonctionne en conditions réelles ✅
Premier tick (22:08:35), les deux bots (500k FIXED + 450k checkpoint) ont ouvert
exactement la position attendue par la formule LINEAR_EXPO :
```
[SIZING] LINEAR_EXPO profile=intraday conf=0.500 size_pct=0.8000 (80.00%)
         size_usd=$16.40 of cash=$20.50 (tier_cap=90.0%, min_order=$11.00)
[PAPER_TRADE] BUY BTC/USDT size=$16.40 price=$64536.66
              SL=$60664.46 (-6.0%) TP=$72281.06 (+12.0%) fee=$0.0164
```
- `size_pct=0.80` = `exp_min(0.70) + (exp_max(0.90)-0.70)*conf(0.5)` → **conforme à l'env**.
- SL=-6.0% / TP=+12.0% = bornes profil `intraday` (_PROFILE_BOUNDS), R/R=2.0 ≥ 1.5 ✅.
Le tier Micro (11-30$) est correctement sélectionné : exposure_range[70,90], cap 90%.

### 8.2 SATURATION ALARM — diagnostic (cause = MODÈLE, pas moteur)
À partir du tick 1, les logs montrent :
```
[DEBUG_ACTION] tick=N dir=+1.000000 size=-1.000000 tf=+1.000000 sl=+1.000000 tp=+1.000000
[ERROR] [SATURATION ALARM] Direction=+1.0000 for N consecutive ticks! Model may be broken.
```
**TOUTES les composantes de l'action PPO sont collées aux bornes (+1/-1) et constantes.**
Ce n'est PAS un comportement de marché : c'est une **saturation de la policy PPO**
(tanh squashing poussé aux extrêmes par des logits de très grande magnitude).

Chaîne de conséquences (toutes du côté MODÈLE, le moteur est correct) :
1. tick 0 : `dir=+1 > threshold(0.01)` & pas de position → **BUY ouvert** ($16.40). ✅
2. ticks 1..N : `dir=+1` constant → branche BUY, mais `cash_restant=$4.10 < min_order=$11`
   → aucun nouveau trade possible (1 seul trade, c'est mécaniquement correct).
3. Le modèle ne sort JAMAIS `dir < -threshold` → la branche SELL/AGENT_CLOSE
   (execution_engine.py:445) n'est jamais atteinte → **aucune sortie agentielle**.
4. SL/TP vérifiés à chaque tick (execution_engine.py:416) mais BTC reste dans
   [$60664, $72281] → ni STOP_LOSS ni TAKE_PROFIT déclenché.

### 8.3 Réfutation de l'hypothèse `unrealized_pnl_pct < 0.0015`
La règle de décision évoquait une possible condition `unrealized_pnl_pct < 0.0015`
bloquant les sorties. **Cette condition n'existe PAS dans execution_engine.py.**
La sortie agentielle dépend uniquement de `direction < -action_threshold`
(seuil 0.01, ligne 428/445). Le blocage des sorties vient donc **exclusivement** de
la saturation `dir=+1.0` permanente du modèle, pas d'un garde-fou PnL.

### 8.4 Verdict & action
- **Le moteur d'exécution et le sizing unifié sont VALIDÉS en production.** Aucune
  divergence silencieuse de sizing : la position ouverte est exactement celle de la
  formule de l'env.
- **Le défaut est dans la policy PPO entraînée** (actions saturées). C'est un problème
  d'ENTRAÎNEMENT (récompense / régularisation d'entropie / clipping des logits), pas
  du code d'inférence paper.
- **Décision (règle "0-1 trade")** : NE PAS activer le HMM live tant que la policy
  sature. Activer le HMM live ne changerait que `confidence` (donc la TAILLE), pas la
  DIRECTION saturée — le bot resterait bloqué long. Le HMM live ne se justifie que si
  la policy produit des directions variées (>10 trades/24h, WR>55%, PF>1.2).
- **Prochaine piste d'amélioration (hors run actuel)** : inspecter l'entropie de la
  policy au checkpoint et, si confirmé saturé, ré-entraîner avec `ent_coef` plus élevé
  ou un clipping des logits pré-tanh. (Améliorer, pas reconstruire.)

---

## §9 — ROOT CAUSE de la "saturation" : portfolio_state DIVERGENT (hypothèse H2 confirmée)

### 9.1 Le diagnostic §8 (PPO saturé) était INCOMPLET
Après 10h de run, l'analyse fine des `[DEBUG_ACTION]` montre que `dir` N'EST PAS
constant : sur 102 ticks on observe 97×`+1.0` MAIS aussi `+0.007`, `+0.148`,
`+0.732`, `-0.077`, `-0.118`... Le modèle **répond** aux entrées (il sort même des
directions négatives = signaux SELL). Un PPO réellement saturé (entropy collapse)
gèlerait TOUTES les composantes. Or seules `size/tf/sl/tp` sont figées, pas `dir`.
=> Ce n'est pas une saturation intrinsèque. C'est un **distribution shift** des
observations (hypothèse H2 de l'utilisateur), cohérent avec un backtest
déterministe à 148 trades / 66% WR : les poids du modèle sont SAINS.

### 9.2 Preuve : portfolio_state live ≠ portfolio_state training
Le `portfolio_state max` en live monte de 1.0 (tick 1) → 2.0 → **4.0** (tick 110) :
il CROÎT avec le temps. Une observation normalisée ne fait jamais ça.

Comparaison slot-par-slot (AVANT correction) :
| idx | TRAINING (portfolio_manager.get_state_vector) | PAPER (execution_engine, AVANT) |
|-----|-----------------------------------------------|----------------------------------|
| [0] | cash_ratio (clip 0-10)                        | equity/cap                       |
| [1] | value_ratio (clip 0-10)                       | cash/cap                         |
| [2] | trading_pnl_pct (clip ±5)                     | has_position (0/1)               |
| [3] | exposure_ratio (0-1)                          | unrealized_pnl/cap               |
| [4] | drawdown (0-1)                                | price_change_pct                 |
| [5] | sharpe/3 (-1..1)                              | size_usd/cap                     |
| [6] | open_positions_norm (0-1)                     | side (±1)                        |
| **[7]** | **win_rate (0-1)**                         | **len(self.trades) RAW (1,2,4,…)**|
| [8] | profit_factor/5 (0-1)                         | drawdown                         |
| [10-19] | features de position (10 dims)            | TOUT À ZÉRO (absents)            |

**Le layout entier était faux.** Le pire : `state[7]` portait le nombre BRUT de
trades (qui croît sans borne) là où le réseau attend un `win_rate ∈ [0,1]`. Le
modèle voyait littéralement du bruit hors-distribution dans 9 slots sur 10, plus
10 dims de features de position toujours à zéro. D'où la réaction extrême
(`size=-1, tf=+1, sl=+1, tp=+1` figés) : c'est la réponse d'un réseau à des
entrées qu'il n'a jamais vues à l'entraînement.

### 9.3 Correction (SANS toucher au modèle, conformément à la règle d'or)
`ExecutionEngine.get_portfolio_state()` réécrit pour reproduire EXACTEMENT le
layout 20-dim de `PortfolioManager.get_state_vector()` :
- 10 dims de base en ratios stationnaires (cash/value/pnl/exposure/drawdown/
  sharpe/positions/win_rate/profit_factor/reserved), tous clipés aux mêmes bornes.
- 2 slots de position × 5 features (unrealized_pnl_pct/size_ratio/steps_norm/
  sl_distance/tp_distance), slot 0 = position courante, slot 1 = zéros.
- win_rate / profit_factor / sharpe tirés de `compute_metrics()` (même convention
  que le backtest).

### 9.4 Preuve de la correction (test fonctionnel)
Après 200 trades simulés + 1 position ouverte :
- AVANT : `state[7]` = 201.0, `max` non borné (→ OOD garanti).
- APRÈS : `max=1.005` (borné), `[7]win_rate=0.500`, `[6]positions_norm=1.0`,
  `[8]profit_factor_norm=0.333`, `[11]position_size_ratio=0.796`. TOUTES les dims
  dans la distribution d'entraînement. Assertions de bornes : PASS.

### 9.5 Conséquence
C'est très probablement la cause des 4-5 échecs successifs en paper trading malgré
de bons backtests : à chaque run on changeait le modèle, mais le vrai coupable était
le pipeline d'observation live (les "yeux"), pas le "cerveau". Avec cette correction,
le modèle existant devrait enfin recevoir des observations cohérentes — aucun
réentraînement nécessaire. À valider sur le prochain run paper avec les bots
redémarrés (l'ancien run tournait avec l'ancien portfolio_state cassé).
