# Passation SPOT strict — 2026-10-09

## Autorité SPOT stricte — 2026-10-09 : 500K BLOQUÉ / NO-GO diagnostique

**Marché = spot pur ; actions = BUY / SELL-sortie / HOLD** (`SELL_EXIT` interne).
LONG est la direction d'une entrée BUY ; SELL clôture uniquement un inventaire LONG
possédé. Aucun SHORT, emprunt, marge ni levier >1. Un flag futures ne suffit pas :
aucun chemin margin/short validé n'existe, il est refusé par le contrat central.

Cette section prévaut sur les anciens prototypes et verdicts exploratoires ci-dessous.
L'ancien GATE 4 reste une preuve arithmétique historique, PAS une autorisation de
production SPOT. Les anciens TRAIN/VAL avec SHORT sont isolés dans
`data/plan_dataset/{train,val}_v1_INVALID_SHORT_spot` ; ne pas les recycler par filtrage.
Les anciens checkpoints entraînés dessus ne constituent pas des checkpoints SPOT validés.

### Contrat transversal et tests

`policy/market_contract.py` est partagé par candidate factory, outcomes, dataset,
encodeur/perception, plan_judgment_core, trainer, risk engine, gate et exécution.
Le trainer vérifie TOUS les plans avant de former les entrées ; SHORT, coût incohérent,
NaN/inf, hash de contrat/registre invalide, doublon et outcome dans STATE sont refusés.
Le réseau reçoit seulement des groupes STATE(t), PLAN(t) et PORTFOLIO(t).
Le test de contamination injecte un SHORT en fin de dataset SPOT et exige une erreur.
La vigie applique SL_FIRST pour la même barre TP/SL et le pire open en cas de gap.
L'exécution ouvre seulement BUY, et SELL ne peut dépasser l'inventaire possédé.

Suite finale : spot 10/10, dataset 4/4, plan labels 9/9, ATR 10/10, perception 8/8,
micro-risk 3/3 ; suites policy et lifecycle PASS. Aucun entraînement nouveau lancé.
Les 9 colonnes restent distinctes : Y_TP_FIRST, Y_SL_FIRST, TIMEOUT, Y_WIN,
NET_RETURN, MFE, MAE, TIME_TO_TP, TIME_TO_SL. Y_WIN = (NET_RETURN > 0), jamais TP_FIRST.
L'API géométrie nomme P(TP_FIRST) explicitement ; sa formule binaire reste un proxy
legacy, pas l'EV réelle des timeouts/gaps. L'audit ci-dessous utilise NET_RETURN des plans.

### Frais : blocage réel, sans substitution

Lecture signée READ-ONLY de `/api/v3/account/commission` implémentée pour Binance
BTCUSDT SPOT ; commissions taker entrée/sortie + buyer/seller + tax/special, sans
réduction BNB supposée. Snapshot frais daté, SHA256, limite de fraîcheur 24h.
Tentative réelle du lecteur : **NO-GO, clés API indisponibles dans ce runtime**.
Aucun snapshot authentifié fabriqué, aucune clé affichée, aucun ordre envoyé.

Les coûts CONFIGURÉS sont donc seulement DIAGNOSTIQUES : commission 0.20% par côté,
slippage 0.05% par côté ; `fees_rt` (coût total modélisé) = 0.50% aller-retour.
FEES_R_MAX reste 0.30R, sans relâchement ; SL_MIN effectif =
max(1.2%, ATR_1h, fees_rt/0.30), SL_MAX = min(3%, 2.5×ATR_1h).
Avec cette config, le plancher frais est 1.6667% ; un intervalle vide impose HOLD.
La génération de production `train_v1`/`val_v1` est refusée sans frais compte vérifiés,
même si `--diagnostic-unverified-fees` est passé. Les diagnostics sont ailleurs.

### Cardinalité exacte et diagnostics TRAIN puis VAL

Comptage exhaustif avec les mêmes gates, stride=12, horizons 48/144/288,
TP baseline 3.5R, LONG uniquement, frais CONFIGURÉS :
**36 755 décisions examinées, 26 987 états admis, 330 975 plans admis**.
9 768 décisions refusées pour intervalle SL vide ; aucun veto intégrité/ATR/risk.
Ce comptage ne calcule pas les outcomes ; ce n'est PAS un dataset de production.
Les ~849 864 plans de l'ancien TRAIN avec SHORT ne représentent pas 500K SPOT.
La cardinalité devra être recomptée après frais réels et correction de couverture.
Ne pas fabriquer/dupliquer des exemples pour atteindre le quota.

Échantillons sauvegardés, 500 décisions sélectionnées uniformément sur chaque split :

| Mesure | TRAIN | VAL |
|---|---:|---:|
| États admis | 364 | 253 |
| Plans LONG | 4 431 | 3 000 |
| TP_FIRST | 4.1074% | 3.1667% |
| SL_FIRST | 36.4703% | 26.8000% |
| TIMEOUT | 59.4223% | 70.0333% |
| Y_WIN | 35.9061% | 35.5000% |
| EV nette moyenne (R) | **−0.179229** | **−0.162727** |
| Timeouts rentables (plans) | 1 409 | 970 |

Oracle scalaire indépendant sur les fichiers sauvegardés : **0 divergence sur les
9 outcomes, 0 sur les 18 contrôles**, TRAIN et VAL. Auto-test : une corruption par
champ, chacune détectée exactement une fois. Noms UNKNOWN rejetés ; pas de zéro
injecté ; pas de déduplication silencieuse ; frontières/gaps/clock vérifiés.
Les taux restent descriptifs : plans alternatifs chevauchants, pas trades exécutés.
Les anciens ~4–5% TP_FIRST et ~40% Y_WIN sont conservés comme observations sous
l'ancien action-space/coût ; leur écart n'est ni caché ni utilisé pour autoriser SPOT.

### TP_MAX et couverture : pas de verrouillage

Même grille prédéfinie 3.5/4/4.5/5/5.5/6R et mêmes buckets SL sur TRAIN et VAL,
avec reachability MFE, TP_FIRST, SL_FIRST, TIMEOUT, Y_WIN, NET_RETURN par cellule.
La tranche 1.2–1.5% est vide avec les coûts configurés. Les 6 cellules non vides
à TP=3.5R ont toutes une EV négative sur les deux échantillons. Le seuil TRAIN
MIN_SUPPORT=200 n'est pas atteint pour une proposition sur ces petits échantillons.
L'ancien audit TRAIN reste une proposition historique ; **TP_MAX non modifié**.

Défaut mesuré : stride=12 verrouille les phases horaires par segment ; VAL n'a
que la phase 1 (253 états). Les phases 11/12 de la thèse sweep ne sont pas représentées.
Ces échantillons ne valident donc PAS la Phase 0b spécifique au signal ni la couverture
complète. Contrôle avec un stride copremier à 12 proposé, **pas lancé dans cette passe**.
Le script Phase 0b révisé rapporte une EV de plans LONG et aucun verdict d'alpha sur TEST.

### Arrêt et reprise ordonnée

**Si EV LONG-only nette ≤0 après frais compte vérifiés sur TRAIN OU VAL : NO-GO,
aucun 500K.** Ici : NO-GO diagnostique aux frais configurés et frais compte manquants.
Le script écrit son rapport puis renvoie 2 en cas de NO-GO. Le trainer bloque également
les runs atteignant 500K présentations ou >10K updates tant que les gates sont ouverts.

1. Fournir au runtime un accès READ-ONLY au compte via variables d'environnement ;
   ne jamais coller les secrets dans dev.md/PR/logs. Lire le palier réel, revalider coûts,
   FEES_R_MAX et intervalle SL, et mesurer couverture de toutes les phases/cellules.
2. Recompter puis reconstruire les vrais `data/plan_dataset/train_v1` et `val_v1`
   avec manifest, code/registry/action-space/source/table SHA256, ressources/PID et code propre.
3. Oracle sauvegardé TRAIN temporel multi-plans puis VAL ; audit économie LONG-only et
   comparaison TP_MAX TRAIN/VAL, sans TEST et sans choix opportuniste de cellule VAL.
4. Si économie et datasets valides : smoke trainer réel, forward/backward, loss finie,
   checkpoint/resume déterministe, normalisation TRAIN-only, calibration VAL et TOUTES
   les baselines obligatoires (le seul prior TRAIN déjà implémenté ne suffit pas).
5. GPU 1K → 5K → 10K ensuite seulement ; aucune autorisation actuelle.
6. Le futur **500K = 500 000 exemples State×Plan distincts admissibles**, pas timestamps,
   pas 500 000 optimizer updates. `--steps` actuel compte des updates et n'est pas ce quota.
   Conserver seed, sélection exacte/IDs, manifests/hashes/code, hyperparamètres,
   checkpoints, métriques TRAIN et VAL ; tous ces gates restent bloqués.

Preuves locales : `data/plan_dataset/spot_diagnostics/`.
Passation versionnée : `docs/passation/spot_2026_10_09/` ; PR #11.

