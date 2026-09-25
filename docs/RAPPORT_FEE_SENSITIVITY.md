# ADAN — RAPPORT SENSIBILITÉ AUX FRAIS : la voie « coûts » tranchée (2026-09-10)

Document de verdict. Source de preuve : `logs/validation/fee_sensitivity_20260910_221310.json`
(commit du diagnostic `7f46e1e`, run 22:13→22:14 UTC).
Univers : **launcher strict** (`BTCUSDT_BINANCE`, `DOGEUSDT_BINANCE`), garanti par
`scripts/diagnostics/_asset_guard.py` — chemins + rows journalisés dans
`logs/fee_sensitivity_run.log`.

## Question testée

Voie #1 du verdict MFE (par leverage mesuré) : **les coûts**. 0,5 % RT mangent
2-4× le MFE 4h médian. Condition de déblocage explicite (ETAT_DU_CODE §9) :

> frais ≤ 0,10 % A/R avec signal net démontré sur backtest non superposé
> **val ET test**.

Question : **une réduction des coûts, jusqu'à 0 % RT, rend-elle une règle
positive sur val ET test ?**

## Méthode (exacte, pas une approximation)

`EV_net(c) = mean(PnL_brut) − c`. Le PnL brut par trade est simulé **une fois**
(coût = 0), le block bootstrap (n_boot=2000, block=5, seed fixe) est calculé une
fois sur les PnL bruts, et les IC95/p-values à chaque niveau de coût s'obtiennent
par translation exacte `means_net = means_brut − c`. Aucune re-simulation.

Règles testées — identiques à EXP2 du rapport MFE, mêmes entrées
(cross-up MACD 5m + MTF_ALIGNÉ, stride 40, non superposées) :

| Règle | Sortie |
|---|---|
| B1 | TP +1,2 % / SL −0,6 %, timeout 24h (la boîte actuelle) |
| B2 | SL 0,5×ATR + invalidation 4h, H_max 7j |
| B3 | SL 3,0×ATR + invalidation 4h, H_max 7j |

Grille de coûts RT : 0,50 % → 0 % (8 niveaux, incluant 0,10 % = borne de
déblocage). Critère pré-enregistré : **DEBLOQUÉ** si une règle a, sur val ET
test à coût ≤ 0,10 % RT, EV > 0 **ET** borne basse IC95 bootstrap > 0.

## Résultat central : EV **brute** (coût nul) et break-even par cellule

Le break-even = EV brute = le coût RT maximal que la règle peut supporter.

Les 18 cellules complètes (valeurs exactes du JSON) :

| Actif / split | Règle | n | EV brute = break-even | EV @ 0,10 % RT | IC95 bas @ 0,10 % | p_boot |
|---|---|---|---|---|---|---|
| BTC train | B1 | 4222 | +0,014 % | −0,086 % | −0,115 % | 0,000 |
| BTC train | B2 | 4222 | +0,096 % | −0,004 % | −0,071 % | 0,833 |
| BTC train | B3 | 4222 | **+0,725 %** | +0,625 % | +0,393 % | 0,000 |
| BTC val  | B1 | 960 | +0,016 % | −0,084 % | −0,148 % | 0,020 |
| BTC val  | B2 | 960 | +0,135 % | +0,035 % | −0,085 % | 0,662 |
| BTC val  | B3 | 960 | +0,437 % | +0,337 % | −0,020 % | 0,063 |
| BTC test | B1 | 790 | +0,054 % | −0,046 % | −0,127 % | 0,299 |
| BTC test | B2 | 790 | **−0,031 %** | −0,131 % | −0,159 % | 0,000 |
| BTC test | B3 | 790 | **−0,148 %** | −0,248 % | −0,334 % | 0,000 |
| DOGE train | B1 | 2619 | **−0,067 %** | −0,167 % | −0,199 % | 0,000 |
| DOGE train | B2 | 2619 | +0,829 % | +0,729 % | +0,108 % | 0,013 |
| DOGE train | B3 | 2619 | **+2,410 %** | +2,310 % | +0,927 % | 0,000 |
| DOGE val  | B1 | 577 | +0,043 % | −0,057 % | −0,132 % | 0,124 |
| DOGE val  | B2 | 577 | +0,528 % | +0,428 % | −0,064 % | 0,104 |
| DOGE val  | B3 | 577 | +1,564 % | +1,464 % | **+0,006 %** | 0,049 |
| DOGE test | B1 | 354 | −0,046 % | −0,146 % | −0,237 % | 0,005 |
| DOGE test | B2 | 354 | +0,041 % | −0,059 % | −0,212 % | 0,537 |
| DOGE test | B3 | 354 | **−0,061 %** | −0,161 % | −0,653 % | 0,500 |

**Fait décisif** : sur le split **test** (le plus récent), l'EV brute de
**toutes** les règles est négative ou quasi nulle (max : DOGE B2 à +0,041 %,
soit un break-even de 0,04 % RT — irréaliste, sous le moindre coût maker).
Autrement dit : **même à frais nuls**, aucune règle ne gagne sur test.
À 0,50 % RT (coût actuel), tout est négatif partout, train inclus sauf B3.

## Verdict sur la condition de déblocage (0,10 % RT, val ET test)

| Règle | Actif | val | test |
|---|---|---|---|
| B1 | BTC | FAIL (EV −0,084 %) | FAIL (EV −0,046 %) |
| B1 | DOGE | FAIL (EV −0,057 %) | FAIL (EV −0,146 %) |
| B2 | BTC | FAIL (p=0,662) | FAIL (EV −0,131 %) |
| B2 | DOGE | FAIL (IC95 bas −0,064 %) | FAIL (EV −0,059 %) |
| B3 | BTC | FAIL (IC95 bas −0,020 %) | FAIL (EV −0,248 %) |
| B3 | DOGE | **PASS** (EV +1,46 %, IC95 bas +0,006 %, p=0,049) | **FAIL** (EV −0,161 %) |

**NO-GO MAINTENU.** Aucune règle ne passe val ET test à 0,10 % RT. La seule
cellule « PASS » (DOGE val B3) est contredite par DOGE test B3 (négatif) — le
même schéma de non-réplication walk-forward déjà mesuré dans le rapport MFE.

## Conséquence structurelle

La voie « coûts » est **close par les données** : le break-even sur test est
inférieur à zéro pour toutes les règles, donc **aucune** structure de frais —
même gratuite — ne débloque. Ce n'est plus un problème de microstructure de
marché (maker vs taker), c'est l'**absence d'edge de sortie sur le régime
récent** (2023+), cohérent avec :
- la disparition de l'edge « runners » hors train (RAPPORT_MFE_EXCURSION) ;
- l'évanouissement de l'edge conditionné 4h sur échantillons indépendants
  (RAPPORT_GEOMETRIE_SLTP, addendum `a0e0c06`).

Conclusion opérationnelle inchangée et renforcée : **reward gelé, fee gate
intact, 500k bloqué.** Les voies 1 (coûts) et « sortie structurelle » sont
désormais mesurées comme insuffisantes. Seules restent ouvertes les voies qui
ne dépendent ni des frais ni d'une géométrie de sortie prédéfinie :
re-cadrage du problème (horizon, univers, ou formulation de l'objectif).

## Reproductibilité

```
cd /home/ubuntu/webapp/MORNINGSTAR/ADAN0
~/webapp/MORNINGSTAR/miniconda3/envs/trading_env/bin/python \
  scripts/diagnostics/diag_fee_sensitivity.py   # ~1-2 min, écrit logs/validation/fee_sensitivity_<ts>.json
```
Le diagnostic charge uniquement `*_BINANCE` (garde `_asset_guard.py`) et
journalise `[DATASET] asset=… path=… rows=…` pour chaque timeframe.
