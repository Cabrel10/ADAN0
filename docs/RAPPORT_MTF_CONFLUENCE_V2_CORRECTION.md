# ADAN — RAPPORT DE CORRECTION R3/R4 : univers de données (2026-09-10)

Document de rétraction/correction méthodologique. Les rapports d'origine
(`RAPPORT_MTF_ROBUSTNESS_R3.md`, `RAPPORT_MTF_CONFLUENCE_V2.md`) sont **conservés intacts** ;
le présent document les corrige sans les remplacer.

## CAUSE

Audit d'identité du dataset (directive : « Prouver asset + chemin + rows pour chaque étape ») :

- L'univers réellement chargé par le launcher ADAN est **BTCUSDT_BINANCE** (662 643 barres 5m train,
  2017→2023) et **DOGEUSDT_BINANCE** (524 841 barres 5m train).
- Les actifs `BTCUSDT` (7 991 barres, ~28 jours) et `DOGEUSDT` (17 500 barres) sont des **contrôles
  de petite taille**, hors univers de trading.
- **R3** (`diag_mtf_robustness.py`) mélangeait les 4 actifs comme pairs :
  `ASSETS = ["BTCUSDT", "BTCUSDT_BINANCE", "DOGEUSDT", "DOGEUSDT_BINANCE"]`.
- **R4** (`diag_mtf_confluence_v2.py`) entraînait les poids MI sur les petits contrôles :
  `ASSET_CORE = ["BTCUSDT", "DOGEUSDT"]`.

Le verdict NO-GO de R3/R4 était donc mesuré (au moins partiellement) **sur un univers que le
launcher ne charge pas** — verdict SUSPECT, conformément au diagnostic qui a motivé la directive.

## IMPACT (mesuré sur les JSON d'origine)

R4 ancien (`logs/validation/mtf_confluence_v2_20260910_093254.json`) — contamination avérée :

| Mesure | BTCUSDT (7 991 barres) | DOGEUSDT (17 500 barres) |
|---|---|---|
| Poids MI | {5m: 0.6528, 1h: -0.3472, 4h: -0.0} | {5m: 0.6203, 1h: -0.3797, 4h: -0.0} |
| Monotonie | b1 = nan, p_perm = nan (non testable) | b1 = nan, p_perm = nan |
| Cellules FDR | 5 cellules seulement | 16 cellules |
| Walk-forward | fold0 « fit trop petit », fold1 « oos trop petit » (n_oos = 29) | n_oos = 58 / 60 |
| Cross-asset | 3 des 4 transferts « n ou geometrie insuffisante » (n = 11 à 24) ; seul BTC→DOGE/test exécuté (n = 52) | |

Conséquences : puissance statistique insuffisante, poids 4h ~ 0 par sous-échantillonnage
(discretisation par quantiles sur ~80 trades), monotonie non testable, walk-forward et
validation croisée largement non exécutables. Le NO-GO R4 reposait en pratique sur
**2 fenêtres OOS de n ≈ 58-60 trades** — très en dessous du seuil de puissance pré-enregistré
(n >= 200/cellule).

R3 ancien (`logs/validation/mtf_robustness_20260910_092159.json`) : 12 combinaisons
(4 actifs × 3 règles) dont la moitié sur les petits contrôles ; toutes « aucune cellule
cohérente ». Le verdict global était correct en direction mais l'audit ne permettait pas de
garantir qu'il tenait sur l'univers du launcher.

## ACTION

Une seule variable changée : **l'univers de données**. Protocole identique (vérifié :
`params` des JSON ancien/nouveau strictement égaux — fees_rt = 0.004, slippage_rt = 0.001,
H_list = [40, 80, 160, 288], K_SL = [0.5, 0.75, 1.0, 1.5, 2.0], K_TP = [1.0, 1.5, 2.0, 3.0, 4.0],
n_boot = 2000, block = 5, n_min_power = 200, fdr_q = 0.05).

1. **Garde-fou partagé** `scripts/diagnostics/_asset_guard.py` (commit 3a03762) :
   - `LAUNCHER_ASSETS = ("BTCUSDT_BINANCE", "DOGEUSDT_BINANCE")` — source de vérité unique ;
   - `assert_launcher_asset()` lève `RuntimeError` sur tout actif étranger (empêche structurellement
     la récidive, ne dépend pas de la mémoire de session) ;
   - `assert_dataset_identity()` journalise `[DATASET] asset=… tf=… split=… path=<absolu> rows=…`
     à chaque chargement parquet (preuve d'audit systématique).
2. **Câblage** : R1 `ASSETS = list(get_launcher_assets())` (répercussion automatique sur R2/R3 qui
   l'importent) ; R4 `ASSET_CORE = list(get_launcher_assets())`.
3. **Re-run R3** : `logs/validation/mtf_robustness_20260910_122302.json`
   (log `logs/mtf_robustness_run_BINANCE.log` — 18 lignes `[DATASET]` prouvent l'univers).
4. **Re-run R4** : `logs/validation/mtf_confluence_v2_20260910_122356.json`
   (log `logs/mtf_confluence_v2_run_BINANCE.log`).

## RESULTAT AVANT (univers contaminé)

- **R3** : 4 actifs mélangés (2 contrôles de 8-17k barres présentés comme pairs) ;
  12 combinaisons « aucune cellule cohérente » ; verdict NO-GO non attribuable à l'univers réel.
- **R4** : poids MI entraînés sur 7 991 barres ; walk-forward BTC in exécutable (« trop petit »),
  DOGE sur n_oos = 58/60 ; monotonie NaN ; cross-asset 3/4 skips ; 5 à 16 cellules FDR ;
  EV_net OOS ≈ -0.495 % / -0.551 % sur échantillons trop petits pour conclure.

## RESULTAT APRÈS (univers launcher, protocole identique)

- **R3** (`mtf_robustness_20260910_122302.json`) : BTCUSDT_BINANCE (662 643 barres train) et
  DOGEUSDT_BINANCE (524 841 barres train) uniquement ; 6 combinaisons (2 actifs × 3 règles),
  toutes « aucune cellule cohérente » (EV_net > 0 sur train ET val ET test, n >= 30, bootstrap
  significatif). Verdict : **NÉGATIF confirmé sur l'univers réel**.
- **R4** (`mtf_confluence_v2_20260910_122356.json`) :
  - BTCUSDT_BINANCE : poids MI {5m: 0.196, 1h: -0.390, 4h: -0.414} ; monotonie b1 = -0.00387,
    p_perm = 0.0005 ; 44/44 cellules FDR rejetées ; WF fold0 n_oos = 2 298,
    EV_net = -0.5257 % IC95 [-0.5433, -0.5087] ; fold1 n_oos = 2 285, EV_net = -0.5041 %
    IC95 [-0.5166, -0.4910].
  - DOGEUSDT_BINANCE : poids MI {5m: 0.245, 1h: 0.385, 4h: 0.371} ; b1 = -0.00586,
    p_perm = 0.011 ; 44/44 cellules FDR rejetées ; WF fold0 n_oos = 1 830, EV_net = -0.5354 %
    IC95 [-0.5749, -0.4949] ; fold1 n_oos = 1 817, EV_net = -0.4832 % IC95 [-0.5078, -0.4575].
  - Cross-asset gelé (H=80) : 4/4 transferts exécutés (n = 1 156 à 1 467, vs 11-52 avant),
    EV_net de -0.492 % à -0.507 %, IC95 entièrement < 0 partout.
  - Verdict : **aucune cellule ne passe (CI95_low > 0 ET n >= 200 hors train)**.

| Comparaison clé | Avant (contaminé) | Après (univers launcher) |
|---|---|---|
| Actifs mesurés | BTCUSDT + DOGEUSDT (petits contrôles) | BTCUSDT_BINANCE + DOGEUSDT_BINANCE |
| Barres train 5m | 7 991 / 17 500 | 662 643 / 524 841 |
| WF n_oos | 29-60 (ou skip) | 1 817-2 298 |
| Monotonie | NaN (non testable) | mesurée (p = 0.0005 / 0.011) |
| Cellules FDR | 5-16 | 44 |
| Cross-asset | 3/4 skips (n = 11-52) | 4/4 exécutés (n = 1 156-1 467) |
| EV_net OOS | ≈ -0.50 % à -0.55 % | ≈ -0.48 % à -0.54 % |

## CONCLUSION

**Le NO-GO TIENT.** Refait à protocole strictement identique en ne changeant que l'univers de
données, R3 et R4 restent tous deux négatifs — cette fois avec la puissance statistique requise
(WF n_oos ≈ 2 000 vs ≈ 60, seuil n >= 200 respecté, FDR 44 cellules, cross-asset complet).

Faits établis par le re-run corrigé :

1. Le biais prédictif MTF est réel et **significatif** (monotonie p = 0.0005 BTC / 0.011 DOGE)
   mais il est **inverse** (b1 < 0 : les déciles de haute confluence ont des rendements forward
   *inférieurs*) et/ou d'amplitude très inférieure à la frontière de coûts : EV brute ≈ 0,
   les pertes nettes ≈ les seuls coûts (0.50 % round-trip).
2. Conformément à la directive (« Tant que 8 n'est pas franchi, on ne touche pas au reward PPO ») :
   l'étape 8 est désormais franchie — **AUCUNE modification du reward PPO n'a été faite** entre
   temps, et la décision est : NO-GO maintenu pour une stratégie autonome à 0.40 % de frais.
3. Étape suivante autorisée par la directive (étape 9) : **fee frontier réelle** — re-mesurer
   l'EV nette à 0.40 % (config ADAN), 0.20 % (spot standard) et 0.15 % (BNB) de frais round-trip,
   toujours sur l'univers launcher et sous le garde-fou `_asset_guard.py`. L'étape 10
   (TP utile / SL utile → reward PPO) reste **gelée** tant que la chaîne n'est pas démontrée
   à au moins un niveau de frais réaliste.
4. Le garde-fou partagé rend la contamination précédente **structurellement impossible** dans les
   probes câblés (RuntimeError sur actif étranger + journalisation [DATASET] systématique).
