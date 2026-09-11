# BENCHMARK CLUSTERING — Référence non-neuronale (pré-ADAN)

**Phase du protocole figé :** `DONNÉES 5–9 ANS ✅ → AUDIT BTC/DOGE ✅ → BENCHMARK CLUSTERING ✅ → FROZEN CONFIG → SMOKE → 250k → 500k → OOS → COMPARAISON`

Ce document établit **la barre que le NN d'ADAN devra franchir**. Il répond à la question :

> « Est-ce qu'ADAN apporte quelque chose qu'un clustering ne sait pas faire ? »

Aucun entraînement. Aucun réseau. Uniquement des règles fixes sur les vraies données Binance 5m,
avec un contrat de trade identique pour toutes les stratégies afin que PF / Sharpe / DD soient
directement comparables.

---

## 1. Protocole commun (identique à toutes les stratégies d'entrée)

| Paramètre | Valeur | Source |
|---|---|---|
| Exécution | LONG only, une position à la fois, entrée à l'**open de la bougie suivante** | `simulate_entries()` |
| Take-Profit | `TP = 3.0 × ATR%` | `BM_TP_ATR=3.0` |
| Stop-Loss | `SL = 2.0 × ATR%` | `BM_SL_ATR=2.0` |
| Horizon max | `H = 40` bougies (3h20 en 5m) | `BM_HORIZON=40` |
| Frais | `6 bps / côté` (12 bps aller-retour) | `BM_FEE=0.0006` |
| TP/SL intrabar | déclenchés via `high`/`low` de chaque bougie | — |
| Graine aléatoire | `0` (déterministe) | `BM_SEED=0` |

Datasets : splits **TRAIN** et **VALIDATION** Binance (70/15/15, val_test, TEST jamais touché ici).

- BTC : train 662 643 bougies, val 141 994 (2017-08 → 2026-08)
- DOGE : train 524 841 bougies, val 112 466 (2019-07 → 2026-08)

Stratégies :
1. **buy_hold** — entrée bougie 0, jamais de sortie (= courbe de prix).
2. **random_entry** — chaque bougie flat entre LONG avec p=0.02 ; sortie TP/SL/horizon.
3. **random_exit** — toujours en marché quand flat (entrée à chaque bougie).
4. **momentum** — `ema_20_ratio>1 & macdh>0 & rsi∈(50,70)`.
5. **mean_reversion** — `rsi<35 & bb_percent_b<0.1`.
6. **cluster_oversold_adx** — `rsi<40 & adx>25` (**la signature découverte par l'audit**).

---

## 2. Résultats BTC

### TRAIN (662 643 bougies, 2017-08 → 2026)

| stratégie | trades | win% | PF | avgR% | totRet% | Sharpe | maxDD% |
|---|---:|---:|---:|---:|---:|---:|---:|
| buy_hold | 1 | – | – | – | **+928.8** | – | −84.0 |
| random_entry | 9 884 | 38.3 | 0.654 | −0.138 | −100.0 | −13.05 | −100.0 |
| random_exit | 38 692 | 39.1 | 0.698 | −0.126 | −100.0 | −11.15 | −100.0 |
| momentum | 23 288 | 38.9 | 0.680 | −0.124 | −100.0 | −11.79 | −100.0 |
| mean_reversion | 11 158 | 40.5 | 0.750 | −0.116 | −100.0 | −9.18 | −100.0 |
| **cluster_oversold_adx** | 11 124 | **40.9** | **0.78** | **−0.103** | −100.0 | **−7.48** | −100.0 |

### VALIDATION (141 994 bougies)

| stratégie | trades | win% | PF | avgR% | totRet% | Sharpe | maxDD% |
|---|---:|---:|---:|---:|---:|---:|---:|
| buy_hold | 1 | – | – | – | **+92.5** | – | −33.0 |
| random_entry | 2 167 | 39.4 | 0.594 | −0.119 | −92.7 | −19.26 | −93.0 |
| random_exit | 9 745 | 40.0 | 0.607 | −0.118 | −100.0 | −19.27 | −100.0 |
| momentum | 5 540 | 40.1 | 0.597 | −0.116 | −99.9 | −19.27 | −99.9 |
| mean_reversion | 2 679 | 41.1 | 0.663 | −0.110 | −95.0 | −16.62 | −95.1 |
| **cluster_oversold_adx** | 2 538 | 40.8 | **0.673** | −0.111 | −94.4 | **−15.10** | −94.5 |

---

## 3. Résultats DOGE

### TRAIN (524 841 bougies, 2019-07 → 2026)

| stratégie | trades | win% | PF | avgR% | totRet% | Sharpe | maxDD% |
|---|---:|---:|---:|---:|---:|---:|---:|
| buy_hold | 1 | – | – | – | **+2684.7** | – | −93.3 |
| random_entry | 7 912 | 39.7 | 0.784 | −0.118 | −100.0 | −6.43 | −100.0 |
| random_exit | 32 713 | 40.2 | 0.792 | −0.119 | −100.0 | −6.43 | −100.0 |
| momentum | 20 038 | 37.8 | 0.733 | −0.148 | −100.0 | −8.53 | −100.0 |
| mean_reversion | 8 703 | 43.3 | 0.900 | −0.059 | −99.8 | −2.99 | −99.9 |
| **cluster_oversold_adx** | 8 465 | **43.9** | **0.93** | **−0.042** | −99.3 | **−1.95** | −99.7 |

### VALIDATION (112 466 bougies)

| stratégie | trades | win% | PF | avgR% | totRet% | Sharpe | maxDD% |
|---|---:|---:|---:|---:|---:|---:|---:|
| buy_hold | 1 | – | – | – | **+92.5** | – | −72.8 |
| random_entry | 1 673 | 39.6 | 0.741 | −0.144 | −92.0 | −10.70 | −92.2 |
| random_exit | 7 277 | 40.8 | 0.789 | −0.118 | −100.0 | −8.80 | −100.0 |
| **momentum** | 4 171 | 42.0 | **0.833** | −0.087 | −97.9 | −6.58 | −98.1 |
| mean_reversion | 2 170 | 40.0 | 0.777 | −0.132 | −95.2 | −10.10 | −95.5 |
| cluster_oversold_adx | 2 029 | 39.8 | 0.775 | −0.139 | −95.0 | −9.48 | −95.0 |

---

## 4. Lecture des résultats — la barre pour ADAN

### 4.1 Aucune stratégie non-neuronale d'ENTRÉE n'est profitable
Avec ce contrat fixe (TP=3×ATR, SL=2×ATR, H=40) et 12 bps de frais aller-retour,
**toutes** les stratégies à entrées répétées ont **PF < 1** et convergent vers −100 %
en capital composé. La cause est structurelle :
- win rate 38–44 % avec un ratio TP:SL de 3:2 (≈1.5) donne une espérance négative avant frais dès que
  le win rate tombe sous ~40 %,
- les frais (12 bps/trade) sur des milliers de trades érodent le reste.

C'est le **plancher naïf** : brasser du signal fixe sur ce marché fait perdre de l'argent.

### 4.2 `buy_hold` est le vrai adversaire fort
Sur 5–9 ans, BTC fait **+929 %** (train) et DOGE **+2685 %** (train). Une politique
« acheter et ne rien faire » écrase toutes les règles actives. **ADAN doit au minimum
battre le couple risque/rendement de buy_hold** (buy_hold paie ce rendement au prix d'un
DD de −84 % à −93 %). Le vrai objectif d'ADAN n'est donc pas « gagner de l'argent »
(buy_hold le fait) mais **gagner avec un drawdown nettement inférieur** — c'est là que
la gestion active a un sens.

### 4.3 Le signal `cluster_oversold_adx` est le **moins mauvais** des signaux — cohérent avec l'audit
- Sur **les deux** actifs et **en train**, `cluster_oversold_adx` a le **meilleur PF, le meilleur
  win rate et le meilleur Sharpe** parmi les stratégies d'entrée.
- DOGE cluster : **PF 0.93 / win 43.9 %** — le plus proche du break-even de tout le tableau.
- Cela confirme quantitativement l'audit : DOGE a un edge conditionnel plus riche
  (0.450 pt) que BTC (0.203 pt), même signature `rsi<40 & adx>25`.

**Mais** : même le meilleur signal reste PF<1. Autrement dit, **un clustering seul, avec un
horizon et une géométrie SL/TP fixes, ne suffit pas**. La question devient précise :

> ADAN doit démontrer qu'un réseau qui **module dynamiquement** (a) quand entrer,
> (b) la géométrie SL/TP (en unités ATR, pas % fixe), et (c) l'horizon H*(régime),
> peut transformer ce PF≈0.8–0.93 en **PF>1 avec un DD contrôlé**.
> S'il n'y parvient pas, il « n'apporte rien qu'un clustering ne sait pas faire ».

### 4.4 Généralisation train → val
- BTC : le classement des stratégies est **stable** train→val (cluster/mean_reversion en tête).
- DOGE : léger changement — en val, `momentum` (PF 0.833) dépasse `cluster` (0.775).
  Signe d'une **dépendance au régime** ; à surveiller quand ADAN sera testé OOS.

---

## 5. Conséquences pour la suite (sans rien modifier de gelé)

Ces benchmarks **ne déclenchent aucune modification** de la config figée
(reward, DBE, Future Arena, MaxDuration, action space, mapping SL/TP, PPO). Ils fixent
seulement les **cibles de comparaison** de la phase COMPARAISON finale :

| Cible | Barre à battre (référence) |
|---|---|
| vs bruit | PF > 0.78 (BTC) / 0.93 (DOGE) — battre le meilleur signal fixe |
| vs buy_hold | rendement comparable **avec DD << 84–93 %** |
| edge du NN | PF > 1.0 net de frais sur VAL, puis OOS TEST |

**Prochaine étape du protocole : FROZEN CONFIG** (documenter/geler l'architecture, reward,
DBE, Future Arena, MaxDuration, action space, mapping SL/TP, PPO, hyperparamètres, et
pointer ENV/dataset vers les données `*_binance`). **Aucun 500k avant cela.**

---

## 6. Reproductibilité

```bash
PY=/home/ubuntu/webapp/MORNINGSTAR/miniconda3/envs/trading_env/bin/python
A=/home/ubuntu/webapp/adan_v36_audit
cd /home/ubuntu/webapp/MORNINGSTAR/ADAN0
# BTC
$PY $A/benchmark_nonneural.py data/processed/indicators/train/BTCUSDT_binance/5m.parquet BTC_train_binance $A/bench_BTC_train.json
$PY $A/benchmark_nonneural.py data/processed/indicators/val/BTCUSDT_binance/5m.parquet   BTC_val_binance   $A/bench_BTC_val.json
# DOGE
$PY $A/benchmark_nonneural.py data/processed/indicators/train/DOGEUSDT_binance/5m.parquet DOGE_train_binance $A/bench_DOGE_train.json
$PY $A/benchmark_nonneural.py data/processed/indicators/val/DOGEUSDT_binance/5m.parquet   DOGE_val_binance   $A/bench_DOGE_val.json
```

Contrat modifiable via env : `BM_TP_ATR`, `BM_SL_ATR`, `BM_HORIZON`, `BM_FEE`, `BM_SEED`.
Artefacts JSON : `bench_BTC_train.json`, `bench_BTC_val.json`, `bench_DOGE_train.json`, `bench_DOGE_val.json`.
