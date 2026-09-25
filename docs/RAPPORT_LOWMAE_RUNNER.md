# ADAN — RAPPORT LOW-MAE / RUNNER PREDICTOR (2026-09-11)

Document de verdict. Source de preuve : `logs/validation/lowmae_runner_20260911_123605.json`
(généré 12:36 UTC). Script : `scripts/diagnostics/diag_lowmae_runner.py`.
Univers : **launcher strict** (`BTCUSDT_BINANCE`, `DOGEUSDT_BINANCE`) via `_asset_guard.py`.

## Question testée

Le rapport MFE/excursion (2026-09-10) avait mesuré : runners réels (p99 MFE 7j = +38 %
BTC, +202 % DOGE), classe C_runner avec MAE médian ~ −0,25 % vs −9,9 % pour les faux
signaux (ratio 40×), et délai d'entrée 0→40 barres sans coût. La chaîne causale à tester :

```
N0 queue MFE existe          → PASS (mesuré, MFE rapport)
N1 classes C/D séparées      → PASS (mesuré, ratio 40×)
N2 MAE_early observable      → trivial (calculable en temps réel)
N3 MAE_early PRÉDIT final_7j → TESTÉ ICI
N4 délai admissible          → PASS (EXP3, EV plate)
N5 règle filtrée EV_net > 0  → TESTÉ ICI
N6 robustesse splits/actifs  → TESTÉ ICI
```

## Corrections de protocole appliquées (revue pré-exécution)

1. **Superposition** : la classification A-E initiale était à stride=40 sur fenêtres 7j
   (chevauchement > 99 %). Ici : entrées à stride=192 (16 h), bootstrap en blocs couvrant
   l'horizon complet (11 entrées/bloc ≈ 2016 barres), et **référence stride=2016**
   (fenêtres 7j strictement non chevauchantes) pour compter les épisodes de runners
   *réellement distincts*.
2. **Biais de régime** : train = +929 % buy-and-hold BTC. Tout seuil est gelé sur train
   puis appliqué tel quel à val/test. Aucun ré-ajustement.
3. **Seuils pré-enregistrés** (écrits dans le script avant tout calcul) :
   - N3 PASS si AUC ≥ 0,60 avec IC95_low ≥ 0,55 **OU** spread déciles ≥ 20 pts
   - N5 PASS si EV_net > 0 **ET** IC95_low > 0 (coûts RT 0,5 % déduits)
   - N6 PASS si signe identique sur train ET val ET test
   - Bonferroni α = 0,05/12 = 0,00417 ; block bootstrap B=2000 ; n < 200 →
     statut « PUISSANCE INSUFFISANTE » (3e statut, ni PASS ni FAIL)
   - Échec N3 = mort de l'hypothèse MAE_early **uniquement** (l'étage B teste les
     features d'entrée séparément).
4. **Causalité stricte** : MAE_6/12/24/48/96/192 n'utilisent que les barres *après*
   l'entrée mais *avant* la décision ; le label (final_7j) est postérieur. Aucune
   information future dans les prédicteurs.

## N3 — MAE_early prédit-il l'issue ? **ÉCHEC**

AUC(MAE_24 → final_7j > 0), entrées stride=192 :

| Cellule | n | n_eff | AUC | IC95 | spread déciles | Verdict |
|---|---|---|---|---|---|---|
| BTC train | 3171 | ~288 | 0,524 | [0,499 ; 0,547] | 8,6 pts | FAIL |
| BTC val | 673 | ~61 | 0,499 | [0,453 ; 0,549] | 11,8 pts | FAIL (= hasard) |
| BTC test | 671 | ~61 | 0,578 | [0,522 ; 0,626] | 25,9 pts | FAIL (AUC < 0,60) |
| DOGE train | 2516 | ~229 | 0,543 | [0,517 ; 0,570] | 13,5 pts | FAIL |
| DOGE val | 531 | ~48 | 0,553 | [0,496 ; 0,603] | 17,9 pts | FAIL |
| DOGE test | 532 | ~48 | 0,563 | [0,503 ; 0,608] | 22,2 pts | FAIL |

Lecture : aucun AUC n'atteint 0,60 avec IC95_low ≥ 0,55 ; les spreads train (8,6 et
13,5 pts) sont loin des 20 pts requis. Les spreads val/test plus élevés sur des n_eff
faibles (~48-61) ont des IC larges à cheval sur le hasard — conformément à la règle
pré-enregistrée, **0,578 n'est pas un « presque », c'est un échec**.

Variantes testées (toutes dans le JSON) : MAE/MFE/ret à 6/12/24/48/96/192 barres vers
trois cibles (win, runner, bad). Aucune combinaison ne franchit les seuils.

**Conséquence pré-enregistrée : l'hypothèse « low-MAE précoce → continuation » est
abandonnée.** Le chemin des 2 premières heures ne porte pas l'information de l'issue
à 7 jours. N1 (séparation finale) ne se matérialise pas causalement tôt — exactement
le scénario « C : −0,2 % → +15 % ; D : −0,2 % → −10 %, identiques à t+2h » anticipé
par la revue.

## Étage B — Features disponibles à l'entrée → runner ? **ÉCHEC**

Meilleures AUC_runner par cellule (atr_pct, bb_width, volatility_ratio, adx, rsi,
trend_40h, ret_24b, ema_20_ratio, bb_percent_b) :

| Cellule | meilleure feature | AUC_runner | AUC_win |
|---|---|---|---|
| BTC train | trend_40h | 0,525 | 0,495 |
| BTC val | rsi_14 / bb_percent_b | 0,533 | 0,516 |
| BTC test | bb_percent_b | 0,514 | 0,505 |
| DOGE train | adx_14 | 0,516 | 0,508 |
| DOGE val | rsi_14 | 0,552 | 0,521 |
| DOGE test | rsi_14 | 0,518 | 0,461 |

Aucune feature ne dépasse 0,56, **aucune ne garde le même signe d'avantage d'un split
à l'autre** (rsi_14 : +0,052 val DOGE, −0,039 sur AUC_win test DOGE). C'est le profil
exact de l'artefact de régime déjà rencontré (`ema>1`) : un « signal » qui n'est que
la reformulation locale du buy-and-hold de la période.

## N5 — Règle filtrée EV_net > 0 ? **ÉCHEC**

Entrée différée à t+24 barres (légitime : EXP3 a montré que le délai ne coûte rien),
filtre `MAE_24 > −X %`, hold 7j, coûts 0,5 % déduits :

| Cellule | baseline EV (toutes entrées) | filtre MAE24 > −0,5 % | filtre MAE24 > −0,25 % |
|---|---|---|---|
| BTC train | +0,75 % [−0,23 ; +1,72] | +0,65 % [−0,26 ; +1,57] | +0,57 % [−0,33 ; +1,46] |
| BTC val | +0,71 % [−0,64 ; +2,04] | +0,32 % [−1,06 ; +1,68] | +0,38 % [−1,03 ; +1,80] |
| BTC test | −0,65 % [−1,88 ; +0,19] | −0,44 % [−1,70 ; +0,36] | −0,08 % [−1,15 ; +0,70] |
| DOGE train | +3,44 % [+0,31 ; +7,14] | +2,61 % [−0,06 ; +6,12] | +2,33 % [−0,38 ; +5,88] |
| DOGE val | +2,34 % [−1,38 ; +7,01] | +2,57 % [−1,61 ; +6,96] | +2,24 % [−1,40 ; +6,57] |
| DOGE test | **−1,90 % [−3,93 ; −0,27]** | −1,47 % [−3,46 ; −0,11] | −0,84 % [−2,81 ; +0,88] |

Deux faits :
1. **Le filtre ne bat jamais la baseline** — souvent il la dégrade. Filtrer sur
   MAE_24 détruit de l'EV au lieu d'en créer : les trajectoires écartées (gros MAE
   précoce) contiennent les rebonds que le filtre prétend éviter.
2. Seule DOGE train a un IC95_low > 0 sur la baseline (+0,31 %) — et c'est la cellule
   à +2774 % de buy-and-hold. Voir N6.

## N6 — Robustesse inter-splits ? **ÉCHEC — et c'est le résultat central**

| Cellule | buy-and-hold | baseline EV net (7j, cross-up MACD) |
|---|---|---|
| BTC train | **+929 %** | +0,75 % |
| BTC val | +93 % | +0,71 % |
| BTC test | **−9 %** | **−0,65 %** |
| DOGE train | **+2774 %** | +3,44 % |
| DOGE val | +93 % | +2,34 % |
| DOGE test | **−62 %** | **−1,90 % (IC95_low < 0)** |

L'EV des entrées cross-up MACD à horizon 7j **suit le signe du buy-and-hold de la
période**. Quand le marché monte, « entrer sur croisement et tenir 7 jours » gagne ;
quand il baisse, ça perd — et le filtre MAE ne change rien à cette dépendance.
**Ce que B3 mesurait comme un « edge runners » sur train était du beta de marché
long, pas de l'alpha de sélection.** C'est la confirmation causale du doute exprimé
lors de la revue du rapport MFE : la classe runner du train était gonflée par le
régime haussier.

## Superposition — réponse à la critique

Référence stride=2016 (fenêtres 7j strictement non chevauchantes, train) :

| Actif | entrées indépendantes | runners | épisodes distincts | MFE médian runner | MAE-avant-pic médian |
|---|---|---|---|---|---|
| BTC train | 326 | 61 (18,7 %) | **61** | +10,3 % | −0,53 % |
| DOGE train | 258 | 39 (15,1 %) | **38** | +15,0 % | −0,47 % |

**Les runners ne sont pas un artefact de rééchantillonnage** : 61 épisodes temporellement
distincts sur BTC train, 38 sur DOGE. L'objection « 3-4 épisodes macro re-mesurés des
milliers de fois » est réfutée. Ce qui est réfuté en revanche, c'est leur
**prévisibilité** à partir des observables testés (MAE précoce + 9 features d'entrée).

Note : la proportion de runners passe de 5,5 % (stride=40, fenêtres chevauchantes) à
18,7 % (stride=2016) — sens inverse attendu, cohérent avec le fait qu'à stride 40 les
fenêtres s'arrêtaient souvent avant le pic d'un épisode tardif. Les deux chiffres
mesurent des choses différentes ; seul le strict-2016 est interprétable comme
« probabilité d'épisode runner ».

## Verdict

| Sous-hypothèse | Statut |
|---|---|
| Les runners existent comme épisodes distincts (pas un artefact) | **CONFIRMÉ** (61 + 38 épisodes, strict-2016) |
| MAE_early prédit l'issue à 7j (N3) | **INFIRMÉ** (AUC ≤ 0,578, IC à cheval sur 0,5) |
| Features d'entrée prédisent les runners (étage B) | **INFIRMÉ** (≤ 0,552, signe instable inter-splits) |
| Filtre MAE_24 → EV nette positive (N5) | **INFIRMÉ** (filtre ≤ baseline ; test négatif) |
| Signal indépendant du régime (N6) | **INFIRMÉ** (EV = f(buy-and-hold) : beta, pas alpha) |

**NO-GO maintenu et renforcé.** Le reward PPO reste gelé — et ce rapport ajoute une
raison structurelle au gel : la seule source d'EV détectable à 7j sur l'univers
launcher est le **beta directionnel**, que le RL ne peut pas transformer en alpha par
le reward. Conséquences inchangées : fee gate intact, 500k bloqué.

## Ce qui reste debout après ce NO-GO

1. **EXP3 (délai 0→40 ≈ EV plate)** reste le seul résultat « actionnable » : un prior
   sélectif n'a pas de coût d'opportunité de délai. Mais ce probe montre qu'aucun des
   priors testés (MAE précoce, volatilité, momentum, régime 40h) n'a de pouvoir.
2. **Le problème est maintenant précisément localisé** : il faut soit (a) des données
   que ces parquets n'ont pas (microstructure : ΔOI, funding, taker imbalance,
   liquidations — la voie MICROSTRUCTURE-RUNNER reste ouverte mais exige une
   acquisition de données, pas un probe), soit (b) une formulation où le beta est
   assumé (stratégie régime-conditionnée : long uniquement quand un détecteur de
   régime haussier fiable existe — or le détecteur de régime est lui-même non démontré
   sur val/test, cf. sonde ATR du 2026-09-11).
3. **Méthodologie verrouillée pour la suite** : tout futur probe devra rapporter
   (i) le nombre d'épisodes distincts, (ii) les IC block-bootstrap couvrant l'horizon,
   (iii) le buy-and-hold du split à côté de chaque EV. Un EV positif sans ces trois
   éléments est désormais considéré non interprétable.

## Reproductibilité

```bash
cd /home/ubuntu/webapp/MORNINGSTAR/ADAN0
.venv-diag/bin/python3 scripts/diagnostics/diag_lowmae_runner.py
# ~10 min, écrit logs/validation/lowmae_runner_<ts>.json
```
