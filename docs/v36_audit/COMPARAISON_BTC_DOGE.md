# Comparaison offline BTC vs DOGE — datasets historiques Binance (5 ans+)

> **Protocole (verbatim utilisateur) :** « On repart sur une vraie base historique …
> dataset historique propre → benchmark statistique → deux agents indépendants →
> progression 10k→500k sans intervention. » Aucun entraînement à ce stade — **c'est
> l'analyse qui doit parler**.

## 0. Source des données — data.binance.vision (archives publiques)

Reconstruction complète depuis les archives mensuelles Binance + archives
quotidiennes pour le mois courant, **checksums SHA256 vérifiés**, dédup, détection
de gaps et validation OHLC. Aucune dépendance à l'API de trading (pas de rate limit).

| Actif | TF | Barres | Période | Dupes | Bougies aberrantes |
|---|---|---:|---|---:|---:|
| BTCUSDT | 5m | **946 633** | 2017-08-17 → 2026-08-22 (**9 ans**) | 0 | 0 |
| BTCUSDT | 1h | 78 901 | 2017-08-17 → 2026-08-22 | 0 | 0 |
| BTCUSDT | 4h | 19 740 | 2017-08-17 → 2026-08-22 | 0 | 0 |
| DOGEUSDT | 5m | **749 774** | 2019-07-05 → 2026-08-22 (**7 ans**) | 0 | 0 |
| DOGEUSDT | 1h | 62 489 | 2019-07-05 → 2026-08-22 | 0 | 0 |
| DOGEUSDT | 4h | 15 631 | 2019-07-05 → 2026-08-22 | 0 | 0 |

- Premier mois réellement disponible : **BTC 2017-08** (listing BTCUSDT),
  **DOGE 2019-07** (listing DOGEUSDT). Pas de fausse symétrie temporelle imposée.
- Le seul « échec » d'archive par run = l'archive quotidienne du **jour courant**
  (2026-08-23) pas encore publiée → HTTP 404 attendu, sans impact.
- Rappel : l'ancien dataset BTC (≈3 mois CCXT, 7 991 barres train) est conservé
  comme **baseline expérimentale**, pas comme dataset scientifique.

Pipeline identique appliqué aux deux actifs : mêmes **21 features**, split
chronologique **70 / 15 / 15** (TRAIN → VAL → **TEST = segment final jamais vu**),
scaler à fitter **uniquement sur TRAIN** (transform sur val/test).

Tailles de split 5m : BTC train 662 643 / val 141 994 / test 141 996 ;
DOGE train 524 841 / val 112 466 / test 112 467.

---

## 1. LA table de décision (calculée sur TRAIN 5m)

| Métrique (TRAIN 5m) | **BTC** (9 ans) | **DOGE** (7 ans) | ancien BTC 3 mois |
|---|---:|---:|---:|
| Barres train | 662 643 | 524 841 | 7 991 |
| Période train | 2017-08 → 2023-12 | 2019-07 → 2024-07 | 2026-05 → 2026-07 |
| **ATR médian 5m** | **0,235 %** | **0,306 %** | 0,143 % |
| ATR p25 / p75 | 0,153 % / 0,360 % | 0,206 % / 0,480 % | — |
| MFE médian @20 | 0,401 % | 0,558 % | — |
| MFE médian @40 | 0,581 % | 0,804 % | — |
| R médian @20 | +0,0121 % | +0,0000 % | — |
| **TP 0,5 % atteignable @20** | 42,2 % | **53,9 %** | — |
| **TP 1 % atteignable @20** | 19,8 % | **29,2 %** | — |
| **TP 2 % atteignable @20** | 6,2 % | **11,8 %** | — |
| SL 2×ATR touché @20 | 46,0 % | 47,6 % | — |
| **edge conditionnel max (|ΔR20|)** | **0,203 pt** | **0,450 pt** | 0,138 pt |
| meilleur cluster (signature) | RSI~27 / ADX~35 | RSI~34 / ADX~36 | RSI~37 / ADX~38 |
| R20 médian du meilleur cluster | +0,206 % | **+0,463 %** | +0,138 % |
| H* long (rendement médian) | 80 barres | 5 barres | ~40 barres |

---

## 2. Lecture — quel « cas » ?

Le protocole prévoyait trois cas :

- **Cas A** : BTC edge faible, DOGE edge fort → le terrain BTC fait partie du problème.
- **Cas B** : edge dans les deux → l'info existe, ADAN ne l'extrait pas (architecture).
- **Cas C** : faible dans les deux → features insuffisantes.

**Verdict des données : entre A et B, penchant net vers A.**

1. **DOGE offre un terrain plus riche que BTC.**
   - Volatilité +30 % (ATR 0,306 % vs 0,235 %).
   - TP atteignables ~1,5–2× plus souvent (TP 1 %@20 : 29 % vs 20 % ; TP 2 %@20 : 12 % vs 6 %).
   - **Edge conditionnel plus de 2× plus fort** (|ΔR20| 0,450 vs 0,203 pt).
   - Un cluster (0,9 % des barres, RSI~34 ADX~36) atteint **R20 +0,46 %**, ~2× le
     meilleur cluster BTC.

2. **La même signature apparaît des deux côtés : survente (RSI bas) + tendance forte
   (ADX élevé).** C'est le régime que le filtre d'entrée proposé (verrou #1) vise.
   Le signal existe donc dans les deux marchés → composante Cas B réelle.

3. **Mais l'edge reste modeste en valeur absolue** (0,2–0,45 pt de R20 médian contre
   un bruit ATR de 0,24–0,31 %), et le win-rate first-touch à TP=3×ATR reste < 40 %
   même sur le meilleur cluster. Aucun cluster ne « bat le hasard » au sens fort
   (winTP > 55 %). Autrement dit : **le régime exploitable existe, il est plus net
   sur DOGE, mais il est fin.**

### Conséquence directe pour le contrat ADAN

- Le TP effectif ≈ 6,3 % (bande FREE_SLTP 0,6 %–12 %) est **absurde** sur les deux
  marchés : TP 2 %@20 n'est atteint que 6 % (BTC) / 12 % (DOGE) du temps ; 6 % l'est
  encore bien moins. La calibration TP/SL doit passer en **unités d'ATR** (verrou #2).
- H* diffère fortement (BTC 80 / DOGE 5 en rendement médian brut, cluster ~20–40) :
  un **MaxDuration fixe unique est inadapté aux deux** ; c'est un paramètre par actif
  (verrou #3), à figer par marché avant le run.

---

## 3. Ce que cette table décide (et ne décide pas)

**Décidé :**
- On a désormais un vrai dataset scientifique (5–9 ans, propre, vérifié) pour BTC et DOGE.
- DOGE est bien le bon 2ᵉ actif : contraste réel, edge conditionnel plus fort.
- Le benchmark de marché (random = toutes les barres) est établi : R médian ≈ 0,
  edge conditionnel faible-à-modéré, plus fort sur DOGE.

**Pas encore décidé (nécessite le benchmark rule-based + le diagnostic 10k–50k) :**
- Est-ce qu'un cerveau profond bat une simple règle de clustering ? → à mesurer
  (benchmark cluster-strategy vs ADAN, jalons 50k).
- Est-ce que BTC+DOGE (multi-actif) aide la représentation ? → 3ᵉ test, après les
  deux cerveaux indépendants.

**Prochaine étape strictement offline :** benchmark sans réseau
(BUY&HOLD / random entry / random exit / momentum / mean-reversion / cluster-strategy)
sur BTC et DOGE, PF/Sharpe/DD, AVANT tout entraînement — conformément à l'ordre
d'exécution figé (étapes 10–11).

> Rien n'est lancé côté ADAN. Aucun reward, config ou mapping n'a été modifié.
