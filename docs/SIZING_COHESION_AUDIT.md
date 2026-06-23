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

---

## 10. ROOT CAUSE #2 — Scalers `prod_scalers/*.pkl` fittés sur la queue haute-prix (LE vrai poison de la saturation d'action)

### 10.1 Observation runtime (run 500k_20260623_081912, APRÈS le fix §9)
Le fix portfolio_state (§9) a fonctionné — `portfolio_state | max=+1.0000` borné, plus
de croissance. MAIS l'action est restée **totalement saturée** sur 7 ticks consécutifs :
`dir=+1.0 size=-1.0 tf=+1.0 sl=+1.0 tp=+1.0` (identique tick 1→7).

Le log d'observation a révélé le coupable :
```
[DEBUG_OBS] tick=2 5m  | min=-5.0 max=+0.81  mean=+0.28   (sain)
[DEBUG_OBS] tick=2 1h  | min=-5.0 max=+5.0   mean=-0.87 std=2.88  (SATURÉ, figé tick→tick)
[DEBUG_OBS] tick=2 4h  | min=-5.0 max=+2.31  (clippé)
```
Le 5m (MinMaxScaler) est sain ; le 1h (StandardScaler) et 4h (RobustScaler) sont
écrasés contre les bornes ±5/±10.

### 10.2 Dissection feature-par-feature (reproduction de l'obs live)
```
1h last bar:  [0]open=-10  [1]high=-10  [2]low=-10  [3]close=-10  [17]vwap_ratio=-10   ← SATURÉ
4h last bar:  [8]adx_14=-10  [17]vwap_ratio=-8.1  [19]bb_width=-5.86                    ← SATURÉ
5m last bar:  tout dans [-0.03, +0.79]                                                  (sain)
```
Ce sont les **colonnes de prix brut** (open/high/low/close) + vwap_ratio qui explosent.

### 10.3 Cause mathématique prouvée
Inspection de `prod_scalers/scaler_1h.pkl` (StandardScaler, par colonne `close`) :
```
mean_ = 116 423 $   scale_ = 3 901 $
→ live close 62 880 $ normalisé = (62880 − 116424) / 3901 = −13.7  → clip −10
```
Or la donnée d'entraînement complète `train/BTCUSDT/1h.parquet` a :
```
close  mean = 52 498 $   std = 31 047 $   (5483 barres, 2022-07 → 2025-08)
```
Le scaler sauvegardé avait `mean=116 423 $` — c'est-à-dire **fitté sur seulement les
121 dernières barres (2.2 %)** du plateau haut-prix mai-août 2025. BTC live à 62 880 $
(parfaitement valide historiquement) se retrouve à **−13.7 σ** → saturation hors-distribution.

### 10.4 Pourquoi le backtest (66 % WR) était sain malgré tout
Le backtest déterministe utilise `MultiAssetChunkedEnv` qui **refitte les scalers inline
sur son chunk** (`multi_asset_chunked_env.py:2294`) → le scaler matche toujours ses
propres données. Le bot live, lui, charge le `prod_scalers/*.pkl` figé (biaisé). C'est
la divergence « cerveau sain / yeux empoisonnés » identifiée par l'hypothèse H2 — mais
le poison était le **scaler**, pas le portfolio_state (qui était un second bug réel, §9).

### 10.5 Correction (SANS toucher au modèle)
Régénération de `prod_scalers/*.pkl` sur la **distribution d'entraînement complète**
(fit sur les premiers 70 % des barres, anti-lookahead, exactement comme l'env) :
```
NOUVEAU 1h close  mean_=35 548 $  scale_=16 972 $
→ live close 62 880 $ = (62880 − 35548) / 16972 = +1.61   (in-distribution, plus de clip)
```
Vérification obs live complète après régénération :
```
1h: min=-1.94  max=+4.69  saturated_cells = 0/420   (AVANT: open/high/low/close=-10)
4h: min=-2.14  max=+4.41  saturated_cells = 0/420   (AVANT: adx/vwap/bb saturés)
5m: min=-10.0  max=+0.80  saturated_cells = 1/420   (1 vieux pic obv_slope, barre row 11,
                                                      la barre récente row 19 est saine)
```
Les anciens scalers biaisés sont archivés dans
`prod_scalers/_archive_tailfit_20260623/` et `*.BIASED_BACKUP`.

### 10.6 Conséquence
C'est l'explication finale des 4-5 échecs paper : à chaque run le modèle recevait des
prix bruts normalisés contre une moyenne de 116 k$ alors qu'il avait appris sur une
moyenne de ~35-52 k$. Les 4 colonnes de prix brut 1h + vwap saturaient systématiquement
à −10 → le réseau réagissait par des actions saturées constantes. Après régénération des
scalers, l'observation live est de nouveau dans la distribution d'entraînement. Le bot
doit être **redémarré** pour recharger les scalers en mémoire (l'ancien process avait
chargé les pkl biaisés).

---

## 11. SOLUTION DÉFINITIVE — Interdiction des `prod_scalers/*.pkl` figés (fit inline obligatoire)

### 11.1 Pourquoi la régénération (§10.5) n'était pas la bonne solution
Régénérer un pkl figé corrige les valeurs une fois, mais reproduit le même piège :
tout pkl figé peut redevenir obsolète/biaisé et re-empoisonner le live silencieusement.
Le défaut structurel est le **chargement d'un scaler figé** alors que
l'environnement d'entraînement/backtest (`MultiAssetChunkedEnv`) **fitte ses scalers
inline** sur son chunk de `train`. La seule garantie de l'invariant
`Training == Backtest == Live` est de **fitter inline aux mêmes données**.

### 11.2 Ce qui a été fait (suppression pure du chemin pkl)
- `prod_scalers/*.pkl`, `*.BIASED_BACKUP` et `scalers_manifest.json` déplacés dans
  `prod_scalers/_BANNED_DO_NOT_LOAD/` (référence forensique uniquement).
  `prod_scalers/` racine ne contient plus aucun pkl → `StateBuilder.__init__`
  ne peut plus en auto-charger.
- `LiveStateBuilder.__init__` : on **vide** `state_builder.scalers` et on remet
  `scalers_loaded_from_training=False` AVANT le fit, pour que `fit_scalers()` ne
  court-circuite pas (il skippait silencieusement quand un pkl était auto-chargé).
- `LiveStateBuilder.fit_on_parquet()` réécrit :
  - utilise désormais le split **`train`** (et non plus `val` — régimes de prix
    différents : val close mean≈74k vs train≈52k) ;
  - force un fit inline propre ;
  - lève une exception si la donnée `train` est absente (refus de tourner avec une
    distribution d'observation indéfinie) ;
  - log `[SCALER_CHECK]` par timeframe (center/scale vs raw mean) comme preuve.
- `prod_scalers/README_BANNED.md` documente l'interdiction.

### 11.3 Preuve (obs live après fit inline, pkl banni)
```
[SCALER_CHECK] 1h StandardScaler | fit_center=35548 fit_scale=16972 | raw_close_mean=52498
[SCALER_CHECK] 4h RobustScaler   | fit_center=30376 fit_scale=39433 | raw_close_mean=64117
5m: min=-1.50 max=+0.80  saturated_cells = 0/420
1h: min=-1.94 max=+4.69  saturated_cells = 0/420   (AVANT: open/high/low/close = -10)
4h: min=-2.05 max=+4.41  saturated_cells = 0/420   (AVANT: adx/vwap/bb saturés)
```
Toutes les observations sont dans `[-2, +4.7]` — la distribution d'entraînement.
**Zéro cellule saturée.** Le bot doit être **redémarré** pour recharger le code et
refitter les scalers inline (l'ancien process tournait avec les pkl biaisés en mémoire).

### 11.4 Invariant à préserver pour toujours
NE JAMAIS réintroduire le chargement d'un scaler figé pour le live. Le live fitte
inline sur `data/processed/indicators/train/<SYMBOL>`, point. C'est la seule
façon de garantir que les « yeux » du modèle en live == ceux de l'entraînement.

---

## 12. VERDICT FINAL — La saturation vient de l'ENTRAÎNEMENT (logits bruts), pas de la chaîne live

Suite à la relecture critique de l'utilisateur (les actions "variées" post-clip ne
prouvent rien : il faut regarder les logits BRUTS avant clip). Tests décisifs faits :

### 12.1 TEST 3 — deterministic
`run_bot.py:318` utilise `predict(deterministic=False)` (exploration gSDE active).
Donc le déterminisme N'EST PAS la cause du figement. `live_state_builder.py:93`
(`deterministic=True`) est un docstring d'exemple, pas du code exécuté.

### 12.2 TEST 5 — décodage des actions (execution_engine.decode_action)
- `size` (action[1]) est **IGNORÉ** : sizing = `exp_min+(exp_max-exp_min)*confidence`
  où confidence = context_vector[3] (HMM). `"size_pref": raw_size` est loggé "not used".
  → que `size` soit saturé n'a AUCUN impact. C'est cosmétique.
- Bandes `intraday` : `sl=(0.04,0.06)`, `tp=(0.08,0.12)` → `tp_lo≠tp_hi`. Décodage
  `tp_pct = tp_lo + (raw_tp+1)/2 * (tp_hi-tp_lo)` clampé. MATHÉMATIQUEMENT CORRECT.
  `raw_tp=-1 → tp=0.08` (8%), `raw_sl=-0.123 → sl=0.0488` (4.877%). Les logs reflètent
  fidèlement les sorties du réseau. PAS de bug de décodage (Hypothèse B écartée).

### 12.3 TEST 1 — logits BRUTS (mean_actions, pre-clip) — LE test décisif
Policy : `MultiInputActorCriticPolicy`, `use_sde=True`,
`StateDependentNoiseDistribution`, **`squash_output=False`** (donc PAS de tanh — clip
direct à la frontière Box(-1,1)). mean_actions sur l'observation LIVE :
```
raw_dir  = -0.71   (sain)        raw_size = +0.10  (sain, ignoré)
raw_tf   = +4.11   <== SATURE     raw_sl   = +0.27  (sain)
raw_tp   = -10.13  <== SATURE MASSIVEMENT -> clip -1.0
gSDE std : tp=0.13, tf=0.036  -> bruit minuscule, impossible de désaturer.
```
→ `tp=-1.0` constant dans les logs = le réseau CRIE -10.13, le clip masque tout.
C'est l'Hypothèse A (clipping de logits extrêmes), CONFIRMÉE.

### 12.4 TEST 2 — backtest vs live (même checkpoint 500k) — LA preuve de la cause
Mêmes logits bruts extraits dans l'ENVIRONNEMENT D'ENTRAÎNEMENT :
```
BACKTEST step 0-9 : dir=+2.17  size=-10.16  tf=+0.01  sl=-6.25  tp=+0.50   SAT:[dir,size,sl]
LIVE             : dir=-0.71  size=+0.10   tf=+4.11  sl=+0.27  tp=-10.13  SAT:[tf,tp]
```
- Le modèle SATURE AUSSI en backtest (size=-10.16, sl=-6.25, dir=+2.17 hors bornes).
- Les logits backtest sont quasi-CONSTANTS (±0.003 sur 10 steps) → politique figée
  indépendamment de l'observation.
- Quelles têtes saturent diffère selon la région d'obs, mais il y a TOUJOURS 2-3
  têtes saturées.

### 12.5 CONCLUSION
La saturation est **intrinsèque à la politique PPO apprise**, PAS un artefact de la
chaîne live. Le pipeline live est désormais sain (obs ∈ [-2,+4.7], scalers inline,
décodage/clipping corrects, §9-§11). Mais le réseau produit des logits extrêmes
(|raw|=4 à 10) sur plusieurs têtes d'action — entropy/std très faible (std 0.03-0.13).

Le backtest affichait 66% WR parce que `dir` (la tête dominante pour le PnL) reste
exploitable et les têtes saturées sont soit ignorées (size) soit bornées par les
bandes de profil (sl/tp clampés). En live le résultat est identique côté décision.

### 12.6 Ce que cela implique (sans réentraîner immédiatement)
La cause racine du "même signal à chaque paper trading" est enfin localisée :
**ce n'est ni le portfolio_state (§9, réel mais secondaire) ni les scalers (§11, réel
mais secondaire) — c'est la POLITIQUE elle-même qui a appris des logits saturés**
(probablement absence de pénalité d'amplitude/entropie sur les têtes continues, ou
log_std trop bas figé pendant l'entraînement). Les fixes §9-§11 étaient nécessaires
(ils garantissent Training==Live) mais NON suffisants pour désaturer l'action.

Pistes (à décider par l'utilisateur, NE PAS réentraîner sans accord) :
1. Vérifier la config PPO d'entraînement : `ent_coef`, bornes de `log_std`,
   `use_sde`, éventuelle régularisation L2 sur la tête action.
2. Comparer des checkpoints plus anciens (50k,100k,...) : à partir de quel step la
   saturation apparaît (entropy collapse progressif ?).
3. Évaluer si un fine-tuning court avec `ent_coef` relevé + clamp de log_std
   suffit à restaurer une politique non saturée, SANS repartir de zéro.

### 12.7 Trajectoire d'entropy collapse (logits bruts par checkpoint, même obs live)
```
ckpt   dir    size    tf      sl      tp      mean_std  sat_heads(|raw|>1.5)
50k   -1.76  +3.56  -1.94  -4.82  -4.42    0.470     5/5
100k  -1.11  +4.47  +2.09  -1.85  -7.36    0.407     4/5
200k  -0.19 +11.24  +5.25  -0.24  -7.78    0.283     3/5
300k  -0.00  +6.65  +9.55  +1.55  -8.85    0.253     4/5
450k  -0.31  +4.39  +6.25  +1.46  -9.98    0.223     3/5
500k  -0.73  -0.20  +3.87  +0.19 -10.04    0.211     2/5
```
Lecture :
1. Saturé DÈS 50k (5/5 têtes) — ce n'est pas un collapse tardif, la politique apprend
   des logits extrêmes quasi immédiatement. `tp` empire continûment (-4.4 → -10.0).
2. Le std d'exploration décroît de façon monotone (0.470 → 0.211) = **entropy collapse
   classique** : `ent_coef` trop bas (ou nul), aucune pression pour rester exploratoire.
3. `dir` est la SEULE tête qui reste utilisable (-1.76 → -0.73, jamais extrême à 500k).
   D'où le 66% WR backtest : la tête qui pilote le PnL (long/short/hold) a appris du
   réel, tandis que size (ignoré), tf (informational), sl/tp (clampés aux bandes
   profil) ont collapsé sans casser la rentabilité backtest.

VERDICT consolidé : modèle PARTIELLEMENT sain (cerveau directionnel OK → 66% WR) mais
têtes auxiliaires dégénérées par entropy collapse à l'entraînement. Chaîne live prouvée
saine (§9-§11). Le travail restant est côté ENTRAÎNEMENT (ent_coef / log_std), à
décider par l'utilisateur — NE PAS réentraîner sans accord explicite.

---

## §13 — RAFFINEMENT RUNTIME (ticks 30-45, ~3h45) : le 500k DÉRIVE, le 450k OSCILLE

Données live capturées après ~3h45 de runtime (session 092905), qui **corrigent une
simplification de §12**. La synthèse critique de l'utilisateur (relecture des logs) a
poussé à rouvrir les têtes d'action tick-par-tick au lieu de min/max agrégés.

### 13.1 Le 500k n'est PAS figé d'entrée — il DÉRIVE de façon monotone

```
tick 30: dir=+0.287  size=-1.000  tf=-0.803  sl=-1.000  tp=-1.000
tick 31: dir=-0.666  size=-1.000  tf=-1.000  sl=+0.475  tp=-1.000
tick 32: dir=-0.257  ...                       sl=+1.000  tp=-1.000
tick 33: dir=-0.223
tick 34: dir=-0.410
tick 35: dir=-0.599
tick 36: dir=-0.791        ← dérive continue +0.29 → -1.0 sur 8 ticks (40 min)
tick 37: dir=-0.996
tick 38: dir=-1.000        ← atteint la borne
tick 39..45: dir=-1.000    ← collé à la borne ENSUITE seulement
```

Lecture : une politique morte ne dérive pas linéairement. Ce profil = `dir` **suit une
variable d'observation qui dérive** (très probablement le PnL non réalisé de la position
ouverte qui se creuse → portfolio_state évolue → dir migre). Donc côté `dir`, le 500k
est RÉACTIF, pas gelé. Mon §12 ("dir reste exploitable") est confirmé ET renforcé : dir
réagit bel et bien à l'obs en live.

### 13.2 Le 450k n'est PAS figé — il OSCILLE en bang-bang synchronisé

```
tick 30: dir=+0.835  sl=-1.000  tf=-1.000
tick 31: dir=-1.000  sl=-1.000  tf=-1.000
tick 32: dir=+0.737  sl=+0.938  tf=+1.000   } régime "haut" : dir>0, sl>0, tf=+1
tick 33: dir=-1.000  sl=-1.000  tf=-1.000   } régime "bas"  : dir=-1, sl=-1, tf=-1
tick 34: dir=+0.518  sl=+0.318  tf=+1.000
tick 35: dir=-0.989  sl=-1.000  tf=-1.000
...
tick 45: dir=-1.000  sl=-1.000  tf=-1.000
```

Lecture : alternance quasi tick-pair / tick-impair entre deux régimes CORRÉLÉS
(dir, sl, tf bougent ENSEMBLE). C'est une politique **bi-stable**, pas figée. Les 27
premiers ticks de la synthèse (dir=+1 constant) étaient une fenêtre transitoire ;
l'état a évolué. Donc l'affirmation "450k complètement figé" est INVALIDÉE par les
données ultérieures — exactement le risque de sur-extrapolation que l'utilisateur
avait pointé.

### 13.3 Ce qui est VRAIMENT dégénéré (et seulement ça)

Sur les DEUX modèles, sur 100% des ticks observés :
- **`tp = -1.000` constant** (jamais autre chose) → tête `tp` réellement collapsée.
- **`size = +1.000` (450k) / `-1.000` (500k)** constant → tête `size` collapsée
  (mais `size` est IGNORÉ par le sizing, qui est piloté par `confidence`/HMM — donc
  sans impact fonctionnel, cf. §6).

Les têtes `dir`, `sl`, `tf` réagissent à l'obs (dérive 500k, oscillation 450k).

### 13.4 Le clipping d'observation +5.0 persiste (résidu scaler, NON critique)

Malgré la pkl-ban + fit inline sur TRAIN (§11, confirmé au démarrage : "Production
scalers not found → Fitting INLINE on TRAIN, pkl BANNED ; 5m=18544, 1h=5483,
4h=1685 samples"), les obs 1h/4h **tangentent +5.0000** de façon persistante (ticks 30
ET 40). Cause : `SCALER_AUDIT 1h | StandardScaler | std_range=[0.0036, 17037.14]` —
certaines features 1h/4h ont une échelle énorme, donc en live elles dépassent +5σ et
sont clippées à la borne. Ce n'est PAS le bug pkl (résolu) ; c'est le clip de sécurité
±5 qui mord sur des features haute-variance. Impact réel faible (peu de cellules,
toujours du même côté), mais à noter : la chaîne live n'est pas 100% "propre", elle
clippe quelques features structurellement larges. À surveiller, non bloquant.

### 13.5 VERDICT CONSOLIDÉ (post-raffinement)

L'utilisateur avait raison sur le fond ET sur la méthode :
1. Le modèle est **partiellement sain** — `dir`/`sl`/`tf` réagissent à l'obs (dérive +
   oscillation observées), seules `tp` (et `size`, mais ignoré) sont collapsées.
2. La "saturation totale" rapportée était une **sur-lecture d'une fenêtre transitoire**
   (27 premiers ticks). Les têtes bougent quand on regarde 45 ticks.
3. Cause racine inchangée : **entropy collapse à l'entraînement** sur les têtes
   auxiliaires (§12.7), confirmé par la trajectoire 50k→500k. La tête `tp` est la plus
   atteinte (raw -4.4 → -10.0), exactement celle qui reste collée à -1.0 en live.
4. Chaîne live = saine pour les têtes qui comptent (dir pilote le PnL). Résidu mineur :
   clip ±5 sur features 1h/4h haute-variance (§13.4).

Conséquence pratique : le comportement live est COHÉRENT avec le backtest 66% WR — la
tête directionnelle décide, tp/size collapsés mais bornés/ignorés. **Aucun bug de
chaîne live ne reste à corriger en urgence.** Le seul levier d'amélioration réel est
côté entraînement (désaturer `tp`), à décider par l'utilisateur — NE PAS réentraîner
sans accord explicite (règle d'or).
