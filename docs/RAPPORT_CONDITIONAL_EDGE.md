# RAPPORT — ETAPE 1 : test du QUADRANT CACHE (conditional edge)

- Script : `scripts/diagnostics/diag_conditional_edge.py`
- Log : `logs/conditional_edge_run.log`
- JSON : `logs/validation/conditional_edge_20260916_222549.json`
- Trades persistes : `logs/validation/conditional_edge_trades/*.jsonl` (6 fichiers, 21 385 trades)
- Date : 2026-09-16 · Univers : `_BINANCE` (garde-fou `_asset_guard`)

---

## 1. Question posee

Toutes les sondes precedentes mesurent la moyenne INCONDITIONNELLE d'une regle
fixe (9840 allers-retours, EV +0,01 %/trade). Un quadrant favorable concentre
(400 trades a +0,60 % noyes dans 9440 trades a -0,02 %) produirait exactement
cette moyenne sans etre detectable. Question : **l'EV est-elle concentree dans
un sous-ensemble identifiable par les features a l'entree ?**

## 2. Protocole pre-enregistre

- Machine a etats spot C1 de spot_mirror (achat cross_up+MTF_ALIGNE -> vente
  cross_down/invalidation 4h/timeout 7j), 32 features a l'entree par trade.
- Analyse par deciles (train = decouverte exploratoire).
- Modele joint HistGradientBoosting (prof. 3, reg forte) gele sur train,
  evalue val/test. 4 tests = 2 actifs x {val,test}, alpha = 0,0125.
- GO si EV top-decile > 0,15 % ET IC95bas > 0,04 % (cout maker RT reel) ET
  n>=200 ET val ET test ET coherence inter-actifs.

## 3. Resultats

### Deciles (train, decouverte)
Des spreads existent IN-SAMPLE : DOGE montre des deciles a +0,95 %
(ema_100_ratio_4h), +0,89 % (rsi_28_4h), IC95bas > 0. BTC plus faible
(meilleur decile +0,13 %, di_delta_4h).

### Modele joint (decision, gele)

| Actif | split | EV top-decile | IC95 bas | p | n_top | verdict |
|---|---|---|---|---|---|---|
| BTC | val | +0,011 % | -0,066 % | 0,70 | 225 | fail |
| BTC | test | +0,090 % | -0,001 % | 0,054 | 183 | fail |
| DOGE | val | +0,187 % | -0,049 % | 0,126 | 135 | fail |
| DOGE | test | +0,176 % | -0,142 % | 0,292 | 90 | fail |

**4/4 FAIL.** Le quadrant qui semblait exister sur train ne se generalise
pas : c'est du bruit de regime, pas une structure recuperable.

## 4. Verdict pre-enregistre

> **NO-GO.** L'EV des trades n'est pas recuperable de facon concentree a
> partir de ces features OHLCV/MTF a cet horizon. Le GO 500k n'est PAS
> justifie ; l'etape 5 reste gelee. C'est precisement le garde-fou qui
> empeche de reproduire le run v30 (2324 trades, 1,5 % de gagnants).

## 5. Ce que cette mesure ajoute a la chaine

R3 (pas de cellule coherente) -> MFE (runners existent, non detectes) ->
LOW-MAE (N3 echoue) -> fee_sensitivity (pas d'edge brut) -> spot_mirror
(pas d'edge dans l'autre direction) -> **conditional_edge (l'edge n'est pas
non plus concentré/recuperable par les features disponibles)**.

La branche OHLCV/5m est desormais fermee sous ses trois formes :
inconditionnelle longue, inconditionnelle bidirectionnelle, et conditionnelle
(selectionnee par features). Restent : microstructure (OI, funding, taker
imbalance) ou horizon multi-jour — ou arret de la branche.

## 6. Travail parallele decide (taches peripheriques)

Conformement a la directive : poursuite des taches d'amelioration de la
perception d'ADAN — interface de visualisation approfondie (modeles, actions,
trades) a finaliser/verifier/tester, surveillance detaillee des composants.
