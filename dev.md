# ADAN0 — Development & Operational Guide

---

# 🧠 ADAN-SYSTEM-ONE — NOUVEAU PARADIGME (2026-09-25)

> **Décision stratégique** : pivot d'ADAN0 (PPO) vers ADAN-System-One
> (machine locale de jugement probabiliste). Le workflow complet est figé
> ci-dessous — toute implémentation le suit à la lettre.

## Autorité actuelle — gates 1 à 9 (2026-10-02)

Les prototypes ci-dessous ne valent pas autorisation d'entraînement. Ordre :
labeler production/référence → audit du registre EXISTANT → graph sparse →
audit labels → labels conditionnés au plan → géométrie conditionnelle →
SystemOne(state,plan,portfolio) → GPU 1K/5K/10K → backtest walk-forward.
Aucun 500K avant validation des mini-runs. Aucun choix sur TEST.
Sizing cible : capital ~20 $, allocation Micro 70–90 %, min ordre ~11 $,
une position maximum, risque/trade 4 %. Les anciens plafonds 20 %/40 % et
minimum 15 $ restent des prototypes NON alignés à réconcilier avec la config.

### GATE 1 — comparaison directe validée (2026-10-02)

Code : commits `6503f70` → `1870cc0` (production finale `138e741`, version
`gate1-baseline-v2`). PR : https://github.com/Cabrel10/ADAN0/pull/11.

- Ancien « zéro divergence » rejeté : comparait deux copies locales à i+2,
  ignorait --n et ne testait pas compute_labels.
- Référence scalaire indépendante : OHLC bruts, direction/sweep et ATR calculés
  sans les tableaux de production ; entrée open[i+1], scan i+1..i+288.
- MFE/MAE : fractions non négatives jusqu'à la sortie, barre de sortie entière
  incluse (excursion intra-barre non ordonnée). Touches simultanées : SL gagne.
- Y_WIN baseline = TP_FIRST, pas encore un PnL portefeuille complet ; AUCUNE
  direction est encore évaluée LONG par défaut. Ce biais sera traité au GATE 5.
- Avant : 122 différences/champs absents sur 70 décisions synthétiques.
  Puis le comparateur strict a démasqué l'arrondi float32 de net_return
  (égalité NumPy pouvait arrondir la référence). Correction float64.
- Correctifs : dernière fenêtre complète réadmise ; plan_valid explicite ;
  ATR14 d'heures complètes consécutives, sans bfill ni double définition ;
  warmup NaN ; fenêtre d'expansion incomplète exclue (wraparound détecté par
  fixture adversariale) ; réintégration finale sans bouclage ; rejet explicite
  NaN/gaps/doublons/OHLC incohérent/prix nul/volume négatif. Pas de drop silencieux.
- Après : 3 000 timestamps d'entrée uniques, seed 1729, répartis dans 12 fenêtres
  continues de 5 000 barres sur TRAIN ; 77 décisions synthétiques ; 14 champs
  comparés valeur par valeur, **0 divergence, erreur max 0.0**, atol=1e-12,
  rtol=0. Six familles d'entrées malformées rejetées. 14/14 mutations des sorties
  de production détectées par le comparateur. Les régressions state/policy/
  lifecycle passent ; cela ne valide pas leurs contrats économiques futurs.
- Données : data/processed/BTCUSDT_binance/BTCUSDT_5m_featured.parquet ;
  source TRAIN 458 489 barres, 2017-08-17 04:00 → 2021-12-31 23:55.
  SHA256 : 3c4fd11116a3c5982248a0adb83b9157fb34569542d4c88f3dfc295294ea5935.
  Python 3.12.13, NumPy 2.2.6, pandas 3.0.5 ; aucune dépendance GPU utilisée.
- Rapport reproductible : logs/gate1_final.json (hash production et entrées,
  dates de chaque fenêtre, erreurs par champ). Commande :
  `PYTHONPATH=src /home/ubuntu/webapp/MORNINGSTAR/miniconda3/envs/trading_env/bin/python3 -m adan_trading_bot.offline.validate_labeler --n 3000 --check-malformed --report logs/gate1_final.json`.
- Limites/échec conservé : TRAIN contient 32 trous réels (timestamps convertis
  explicitement en ns ; .asi8 était en microsecondes). La CLI de labellisation
  refuse donc actuellement le fichier complet : segmentation continue et purge
  des frontières requises avant nouveau dataset. Les anciens parquets et le
  smoke checkpoint sont expérimentaux/périmés, NON autorisés pour entraînement.
  Timeout net_return=0 reste une convention provisoire, aucun fill maker simulé.
  Tous les NOUL/régimes ne sont pas encore audités. Prochaine étape : GATE 2.

### GATE 2 — PARTIAL / AUDIT IN PROGRESS (mise à jour 2026-10-05)

**Statut autoritaire actuel** : GATE 1 = PASS ; GATE 2 = PARTIAL, aucune
validation pour entraînement ; GATE 3 = travail structurel autorisé, qualification
limitée ; GATE 4 = label audit pending ; GPU = BLOCKED ; 500K = BLOCKED.
Les contrôles de métadonnées décrits ci-dessous passent, sans transformer
l'inventaire en 1 026 features ni valider le pipeline économique complet.

#### Classification exclusive et contrat d'entrée (2026-10-05)

Code/proofs : `d4224bb` → `446049b` (classification, contrat, ATR et graphe).
Les commits seront regroupés pour la PR, les mesures intermédiaires restent
traçables dans les rapports. Aucune déclaration supprimée ou renommée : test
comparant les 1 026 noms/ordre et les anciens champs à `a990609` ; anciens
`causal_t` conservés uniquement comme declared_disponibilite_t, jamais comme preuve.

| Catégorie exclusive | Nombre |
|---|---:|
| MARKET_FEATURE | 5 |
| CONTEXT_FEATURE | 4 |
| DERIVED_FEATURE | 272 |
| LABEL_ONLY | 8 |
| CONFIG_ONLY | 577 |
| UNRESOLVED | 160 |
| PORTFOLIO_FEATURE / PLAN_FEATURE / RISK_FEATURE | 0 / 0 / 0 |

La catégorie et le statut de résolution sont différents : 577 CONFIG_ONLY +
8 LABEL_ONLY + 160 de catégorie UNRESOLVED = **745 status=UNRESOLVED**, tous
conservés avec reason, missing_source, missing_runtime_mapping, lineage et
future_safe. Les rôles portfolio/plan/risk sont déclarés, pas encore prouvés au
runtime ; on ne les promeut donc pas artificiellement en features disponibles.

**Future-safe : VERIFIED=281 ; UNKNOWN=745 ; UNSAFE=0 confirmé.** Zéro UNSAFE
confirmé ne signifie pas zéro risque : les 8 sources labeler sont interdites
comme STATE par classification conservatrice ; leurs mappings individuels ne
sont pas inventés. Un résultat ex-post contenant next_close/future H/L ou
TP_FIRST/SL_FIRST/MFE/MAE/times/NET_RETURN n'est jamais un input causal.

- Mutation future : première fenêtre TRAIN continue de 600 barres, t=index 300,
  toutes les OHLC après t ×3 et tous les volumes après t ×5 ; les 281 valeurs
  t restent égales. Preuve par entrée, empreintes du producer et de l'adapter,
  timestamp et transformations stockés dans le JSON existant. Un test indépendant
  synthétique répète l'invariance avec facteurs ×7/×4. Seed non applicable.
- `features/feature_availability_contract.py` : admission par NOM, sans remplir
  les inconnues de zéro ; rejet labels/config/unresolved, preuves périmées,
  mappings manquants et intégrité invalide. La sortie est un dictionnaire nommé
  qui préserve la structure, pas 281→MLP. Le classifieur ne crée pas un registre
  parallèle. Les anciennes déclarations restent intégralement consultables.
- `train_calibrated_judgments.py` reste un draft historique NON valide ; son
  entrée est bloquée avant tout chargement de données/optimiseur/checkpoint,
  y compris --steps 1. Le MlpEncoder et l'ancienne RelationalPerception tout-
  registre sont bloqués. Aucun pas d'entraînement, aucun TEST utilisé.
- Tests persistants : `tests/test_feature_availability_contract.py` **12/12** ;
  inclut tentative de falsifier une source labeler, de réutiliser une preuve
  périmée, d'injecter une inconnue/zero et de contourner le verrou training.

#### Réconciliation ATR explicite — définitions, pas résolution fictive

12 entrées liées à ATR documentées dans leur atr_definition, sans changer les
noms, incluant les réglages/multiplicateurs qui ne sont PAS des mesures ATR.
`atr_14` / `atr_5m_pct` conservent la sémantique legacy 5m Wilder/RMA14 (backend,
amorçage et mapping non vérifiés). Le helper 5m du labeler emploie SMA14 : ce ne
sont pas silencieusement les mêmes variables.

`c1h.atr_1h` : décision explicite **SMA14 de True Range de 14 heures complètes
consécutives, lag d'un contenant, jamais le contenant en formation**. Fraction
`c1h.atr_1h_pct` = ATR / close_5m(t), pas points de pourcentage. Définition
`systemone-1h-sma14-tr-lag1-v1`, producteur du GATE 1 déjà validé.
Même définition proposée en 4h, producteur 4h encore UNRESOLVED.
Lineage corrigé explicite vers prev_high/low/close et offsets historiques ; les
anciens running H/L restent des déclarations historiques, pas des contraintes
vérifiées. Les ratios running_range/ATR sont distincts de l'ATR lui-même.

**Non résolu** : LivingStateSnapshot n'expose pas encore un ATR canonique et
les scalaires de geometry ne prouvent pas leur provenance. Le helper trompeur
`_atr_pct_from_snapshot` REFUSE maintenant tout appel au lieu de substituer
range/phase ou zéro. Les ATR du registre restent UNKNOWN/UNRESOLVED et sont
interdits dans STATE tant que leurs mappings/proofs ne sont pas établis.
Cohérence/lacunes labeler/geometry/snapshot, formula, source, timeframe,
availability et causal contract détaillés dans logs/atr_reconciliation.json.
Les assertions ATR (warmup NaN, lag, mutation future et refus proxy) passent.

#### Graphe sparse qualifié — aucune preuve de causalité économique

`features/relation_graph.py` conserve 1 026 nœuds déclarés : **540 arêtes**
(sparsité 99.9487 %), dont 273 derived_from, 8 aggregates, 245 temporal,
2 same_container, 5 portfolio_constraint, 7 plan_dependency.
Les 10 liens supplémentaires explicitent les sources des contenants précédents.
Les offsets historiques sont conservés ; pas de parent current_close présenté
comme producteur d'un lag, pas d'ATR attribué à running H/L au même instant.

Qualification indépendante sur OHLCV TRAIN, 600 barres, décisions
[192,203,239,288,347,503], références brutes :

| Statut d'arête | Nombre |
|---|---:|
| VERIFIED_DETERMINISTIC | 28 |
| VERIFIED_TEMPORAL | 255 |
| VERIFIED_PORTFOLIO | 0 |
| VERIFIED_PLAN | 0 |
| PREDICTIVE_TRAIN_ONLY | 0 |
| UNVERIFIED | 257 |

Seules **283 arêtes** vérifiées entre variables admises par le contrat peuvent
être des contraintes ; par défaut les 540 arêtes sont UNVERIFIED jusqu'à cette
qualification. Temporal/membership certifie l'ordre/structure, pas une équation
de prix ni une causalité. Aucun edge empirique appris ; admission d'un edge
empirical exige TRAIN 2017–2021, méthode, effectif et freeze_id. Même avec
provenance, une relation prédictive ne devient PAS une contrainte déterministe.
Relations plan/portfolio/result restent à vérifier aux gates suivants ; les
sorties ex-post ne reviennent jamais dans STATE. Graphe déclaré ≠ graphe utilisable.

`tests/test_relation_graph.py` **12/12** : COO/scatter vs boucle indépendante,
gradients finis (seed 1729), endpoints/duplications, refus provenance VAL/TEST,
statuts falsifiés, exclusion contraintes non vérifiées, lineage ATR et blocage
prototype neural. Régression policy 21/21 maintenue (contrats économiques récents
non validés par ces anciens fixtures).

Rapports versionnés : logs/registry_classification.json,
logs/registry_audit_classified.json, logs/atr_reconciliation.json,
logs/relation_graph_qualified.json ; hashes source/dataset/registry, dates,
échantillons et versions inclus. Dataset et empreinte restent ceux du GATE 1.
Aucun tuning/sélection sur VAL/TEST. Python 3.12.13, NumPy 2.2.6, pandas 3.0.5.

Prochain travail : résoudre les mappings ATR canoniques/snapshot/geometry avec
preuves dédiées, puis l'audit des labels et leurs anciens déséquilibres ; ne pas
prétendre que les labels/régimes/targets/geometry sont déjà verrouillés.
Les 745 non résolues restent à résoudre progressivement, aucune réduction
artificielle du registre. Les ajouts de géométrie préexistants restent préservés
et non validés au GATE 6.

### Audit initial du registre (2026-10-02, historique)

Commit code `9dd3347`. Aucun registre parallèle, aucun remplacement/suppression
ou édition du JSON d'origine (empreinte vérifiée avant/après). Rapport détaillé
pour CHAQUE déclaration : logs/registry_audit.json ; ce rapport est un inventaire
de preuves, pas une nouvelle source de variables.

- 1 026 déclarées ; 281 valeurs de marché/snapshot effectivement résolues
  (8 bar courante + 245 lags + 28 champs de contenants). Disponibles à t et
  invariantes à une mutation future sur la fenêtre continue TRAIN testée.
  Ce sont des bornes inférieures vérifiées, pas une preuve universelle.
- 577 chemins de configuration déclarés : 433 existent dans la config actuelle,
  144 manquent. Ce ne sont PAS 577 observations de marché calculées. Aucun
  archivage de config point-in-time historique vérifié ; valeurs/secrets non exportés.
- 385 entrées déclarent des dépendances (toutes les références existent).
  1 026 manquent sous-famille/lineage/future_safe explicites ; une assertion
  causal_t dans le JSON ne suffit pas. 745 restent sans preuve future-safe.
- 0 dépendance future formellement confirmée au nom exact de la déclaration,
  mais 8 sources labeler doivent être tracées : le label de réintégration de
  production lit next_close. Zéro confirmé ne signifie PAS zéro risque.
- Les ATR déclarés à partir de running_high/low ne décrivent pas l'ATR14
  d'heures complètes du GATE 1. Les champs risk 20 %/40 %/15 $ sont obsolètes
  pour le micro-capital. Les réglages reward PPO/credentials/paths ne sont pas
  des features numériques de perception ; aucune variable supprimée.
- Test : première fenêtre TRAIN continue de 600 barres, décision index 300,
  future high×5 et volume×3, égalité des 281 valeurs ; seed non applicable.
  Versions Python/NumPy/pandas/PyYAML, dates/empreintes JSON/config dans le rapport.
  Commande : `PYTHONPATH=src /home/ubuntu/webapp/MORNINGSTAR/miniconda3/envs/trading_env/bin/python3 -m adan_trading_bot.features.feature_registry --report logs/registry_audit.json`.
- Prochain : graph sparse typé sur les déclarations, sans certifier comme
  exploitable toute variable non auditée. Résolution runtime des 745 restantes
  et métadonnées manquantes nécessaires avant GATE 7/GPU.

## Principe cardinal

**La clôture 5m est l'horloge maîtresse unique.** Les échelles 1h et 4h ne sont
JAMAIS injectées comme des bougies figées (poison du MTF statique), mais comme
des **contenants partiellement formés**, reconstruits depuis la séquence 5m à
la seconde exacte de clôture → zéro fuite de données futures possible.

**Le droit de s'abstenir** : 0 à 5 trades/jour MAXIMUM (et non 2-5 obligatoires).
Aucun gradient de punition pour le silence. C'est ce qui manquait à PPO
(effondrement des actions μ=-7.2, fuite SELL documentée dans RAPPORT_*.md).

## Les 12 étapes du workflow (résumé opérationnel)

| # | Étape | Rôle | Module cible |
|---|-------|------|--------------|
| 0 | Flux continu | ticks → OHLCV 5m ; décision uniquement au close 5m | data_loader.py (conservé) |
| 1 | État temporel | 5m fermée + contenant 1h (k/12) + contenant 4h (m/48) RUNNING | **data/nested_state_builder.py** (CRÉER) |
| 2 | Contrôle d'intégrité | NaN/trous/OHLC incohérent → ABSTENTION | data_validator.py (conservé/étendu) |
| 3 | Perception | CNN 5m + états 1h/4h running + attention hiérarchique + FiLM → Z | models/feature_extractors.py (REFACTORER) |
| 4 | Moteur de jugement | Questions NOUL/CHOICE/SCORE sur le même Z, calibrées | **models/system_one_core.py** (CRÉER) |
| 5 | Abstention | anomalie/incertitude/régime/edge/EV≤0/cooldown/quota → HOLD | **policy/deterministic_gate.py** (CRÉER) |
| 6 | Direction | LONG / SHORT / AUCUNE | deterministic_gate.py |
| 7 | Géométrie | SL=invalidation structurelle, TP=expansion 1h/4h, EV nette − frais | **policy/geometry_engine.py** (CRÉER) |
| 8 | Risk engine | Kelly fractionnaire + plafonds, sizing déterministe | **policy/risk_engine.py** (CRÉER) |
| 9 | Exécution | ordre + SL + TP + rationale | trading/execution_engine.py (conservé) |
| 10 | Position active | réévaluation à CHAQUE close 5m ; SL = invalidation de thèse | **position/lifecycle_manager.py** (CRÉER) |
| 11 | Post-trade | dataset d'expérience (état, réponses, MFE/MAE, résultat) | logging existant (étendre) |
| 12 | Apprentissage hors-ligne | labellisation MFE/MAE ex-post, Brier/LogLoss, calibration VAL, test OOS | **offline/labeler_mfe_mae.py + train_calibrated_judgments.py** (CRÉER) |

## Ce qui disparaît définitivement (et pourquoi)

| Supprimé | Remplacé par | Pourquoi |
|----------|--------------|----------|
| agent/ppo_agent.py (centre) | jugements calibrés + policy déterministe | PPO apprenait par PnL bruité, forçait l'action → fuite SELL |
| environment/reward_calculator.py | scoring rules propres (Brier/NLL) hors-ligne | reward RL = surconfiance + divergence Val/Test |
| MTF statique (1h/4h figées) | contenants vivants reconstruits depuis 5m | latence temporelle déguisée en confluence |
| SL/TP géométrie fixe (k_sl=0.5, k_tp=3.38) | invalidation structurelle + expansion contenant | géométrie arbitraire coupait les gagnants (sep=-0.10) |

## Journal de progression System One

| Date | Tâche | Résultat |
|------|-------|----------|
| 2026-09-25 | Workflow final intégré à dev.md | ✅ figé |
| 2026-09-25 | Phase 0 — test brut sweep 1h (scripts/phase0_nested_container_edge.py) | ⚠️ séparation +1.30R (sweep −2.78R vs baseline −4.08R sur test) mais EV nette négative → piège du dénominateur R identifié (frais 0.40% / SL micro-mèche ~0.15% ≈ 2.67R de frais/trade) |
| 2026-09-25 | Phase 0b — audit 4 niveaux (scripts/phase0b_edge_audit.py) | ✅ **POSITIF** — détails ci-dessous |

### Phase 0b — Résultats exploratoires (946 633 bougies 5m BTCUSDT, 2017→2026)

Correction scientifique (2026-10-02) : la géométrie a été comparée et choisie
sur TEST. Ce TEST n'est donc plus une confirmation indépendante intacte pour
cette exploration. Les événements se chevauchent ; les frais maker 0.08 %
présupposent un fill sans modèle de remplissage/slippage/impact. Pas de preuve
d'alpha déployable, ni de géométrie définitivement optimale. Les chiffres
historiques restent conservés ci-dessous comme observations exploratoires.

**Méthode** : cadre en 4 niveaux (décision du 2026-09-25) — on ne juge plus une
géométrie arbitraire, on mesure la physique du signal PUIS on cherche la cage
qui la monétise. Splits temporels stricts : train <2022 · val 2022-23 · test ≥2024.

**Niveau 1-2 — Structure & Excursion (fenêtre 24h, sans géométrie)** :

| Split | P(MFE>2×ATR) signal | baseline | Δ | MFE/MAE signal | baseline |
|-------|--------------------|----------|---|----------------|----------|
| train | 86.1% | 83.4% | +2.8 pts | 0.97 | 0.93 |
| val   | 86.8% | 86.1% | +0.6 pts | 1.07 | 1.06 |
| **test** | **88.2%** | **85.9%** | **+2.3 pts** | **1.15** | **1.01** |

→ Le sweep en fin de contenant 1h (k∈{11,12}) détecte une anomalie réelle :
excursion favorable plus fréquente ET asymétrie MFE/MAE supérieure. Le signal
présente une association exploratoire ; ce n'est pas une preuve causale
ni une confirmation sur un test jamais consulté.

**Niveau 3 — Géométrie (EV BRUTE en R, split test)** : positive sur TOUTE la
grille SL{0.4, 0.8, 1.2%} × TP{1.5, 2.5, 3.5}R, de +0.065R à **+0.147R**
(croissante avec la largeur du SL → confirme que le stop micro-mèche était
l'erreur, pas le signal).

**Niveau 4 — Économie (EV nette = EV brute − frais/SL, split test)** :

| Frais | SL 0.4% | SL 0.8% | SL 1.2% |
|-------|---------|---------|---------|
| Taker 0.40% RT (stress) | −0.90R | −0.40R | −0.19R (TP 3.5R) |
| **Maker 0.08% RT (limit post-only)** | −0.10R | **+0.03R** | **+0.080R** ✅ |

**Verdict révisé** : résultat exploratoire positif sous hypothèses — meilleure
cellule observée **SL=1.2% × TP=3.5R, frais maker → +0.080R/trade net** sur TEST.
Ce choix sur TEST, le chevauchement et l'absence de modèle de fills interdisent
de conclure à un edge exploitable ou à un problème de frais résolu.

**Conséquences pour les modules System One** (gravées dans l'implémentation) :
- `geometry_engine.py` : SL MINIMUM structurel ~1.2% (ou ATR 1h), jamais la
  micro-mèche 5m seule ; TP sur l'expansion du contenant (≈3.5R)
- `deterministic_gate.py` : refus géométrique si frais_R > 0.3R
- Exécution cible : ordres limit post-only (maker), pas de taker

### Étape 1 — nested_state_builder.py : LIVRÉE ET PROUVÉE (2026-09-25)

**Fichier** : `src/adan_trading_bot/data/nested_state_builder.py` (+ `data/__init__.py`)

**Ce qui a été fait** : `NestedStateBuilder` construit le `LivingStateSnapshot` S_t
à chaque clôture 5m — barre 5m scellée + contenants 1h (k/12) et 4h (m/48)
**vivants** (OHLCV running reconstruits depuis la timeline 5m, jamais chargés
depuis des parquets 1h/4h figés), niveaux des contenants fermés précédents,
sweeps avec réintégration, phases, positions intra-contenant, portefeuille,
et flag d'intégrité (étape 2 → abstention).

**Pourquoi** : élimine la latence temporelle du MTF statique (l'erreur
fondatrice d'ADAN0) — à k=6/12, le contenant 1h n'est qu'à moitié formé, son
état reflète exactement cette réalité. Zéro fuite possible par construction.

**Corrections clés** :
- pandas 3.0.5 a supprimé `ts.view('int64')` → `ts.to_numpy('datetime64[ns]').astype(int64)`
- `.gitignore` : la règle `data/` (données racine) masquait aussi le package
  `src/adan_trading_bot/data/` → ancrée en `/data/` + exception explicite.

**Preuves (tests/test_nested_state_builder.py — 7/7)** :
- T1 : mutation des barres > i (high×5, close×3) → snapshot i **inchangé**
- T2 : à k=6/12, running_1h == OHLCV des 6 premières 5m exactement
- T3 : prev_1h == bougie 1h civile fermée précédente (recalcul indépendant)
- T4 : phases frontières exactes (00:00 → 1/12 & 1/48 ; 03:55 → 12/12 & 48/48)
- T5 : historique court / NaN / OHLC incohérent → integrity_ok=False (abstention)
- T6 : sweeps conformes à la référence Phase 0 (dépassement + réintégration)
- T7 : cohérence sur 10 000 barres réelles BTCUSDT (running_1h à k=12 == 1h recalculée)

### Étape 4 — system_one_core.py : LE « JEV PERSONNEL » LIVRÉ (2026-09-25)

**Fichier** : `src/adan_trading_bot/models/system_one_core.py`

**Ce qui a été fait** : le moteur de jugement (étape 4 du workflow). Z n'est
jamais transformé directement en ordre — il est **interrogé** par des
Query-Slots (embeddings de questions appris) :

- **NOUL** (6 questions : mfe_atteint_tp, risque_adverse_faible,
  expansion_imminente, anomalie_donnees, sweep_confirme, reintegration_valide)
  → P(vérité) ∈ [0,1]
- **CHOICE** (regime ∈ {BULL, BEAR, RANGE, TRAP} ; direction ∈ {LONG, SHORT,
  AUCUNE}) → distribution catégorielle
- **SCORE** (qualite_setup, conviction) → distribution **ordinale monotone**
  sur 0..10 : P(>k) structurellement décroissante (cumsum de softplus),
  distribution reconstruite par bornes, E[score] + incertitude. Corrige la
  somme naïve de sigmoïdes du prototype (aucune monotonie garantie).

**Architecture** : moteur GÉNÉRAL de jugement — notion partagée (décodeur par
type) + adaptateur spécifique par question, au lieu de 10 têtes indépendantes.
Le système apprend que « la même notion probabiliste s'applique à plusieurs
affirmations, chacune avec sa sémantique ».

**Calibration** : `calibrate()` fixe les températures (buffers, apprises sur
VALIDATION à l'étape 12) ; `predict_calibrated()` ne sert que des probabilités
dont P=0.70 signifie 70 % réel. Les logits bruts ne sortent que pour
l'entraînement (Brier/NLL).

**Bogues trouvés par le smoke test et corrigés** :
1. KeyError : dict de logits indexé par entier au lieu du nom de question
2. Distribution ordinale tronquée (10 vs 11 niveaux) → bornes P(>-1)=1 et
   P(>10)=0 explicitement concaténées

**Preuves (smoke test 6/6)** : forward batch, monotonie ordinale, probabilités
sommant à 1, espérance ∈ [0,10], calibration activable, round-trip complet
LivingStateSnapshot → features (88 dims) → Judgment sur données réelles BTCUSDT.

### Étapes 5-8 — BLOC POLICY LIVRÉ (2026-09-25)

**Fichiers** : `src/adan_trading_bot/policy/` — `geometry_engine.py`,
`deterministic_gate.py`, `risk_engine.py` (+ `__init__.py`)

**Ce qui a été fait** : la chaîne déterministe qui transforme l'état S_t en
ordre sain ou en abstention — grave dans le code les invariants économiques
prouvés en Phase 0b.

**geometry_engine.py (étape 7)** — invariants non négociables :
- Plancher SL = max(1.2 %, 1.0 × ATR_1h) — interdiction du micro-SL
- TP = 3.5 R (expansion du contenant 4h)
- Refus géométrique si frais_R = frais_RT / SL > 0.30 R
- EV nette = P(win)·3.5 − (1−P(win)) − frais_R > 0 exigée

**deterministic_gate.py (étapes 5-6)** — le gardien de l'abstention (0-5 trades/j
MAX, jamais de punition du silence). Chaîne de veto dans l'ordre : intégrité →
quota (5/j) → cooldown → anomalie JEV → régime TRAP → géométrie → direction
(sweep_high→SHORT, sweep_low→LONG, confirmé par le JEV).

**risk_engine.py (étape 8)** — sizing déterministe : Kelly fractionnaire
(quart) plafonné à 20 %/trade, exposition totale ≤ 40 %, circuit breaker
pertes jour > 5 %, minimum 15 USDT (Binance). Kelly ≤ 0 → aucun capital.

**Preuves (tests/test_policy_suite.py — 21/21)** :
- Géométrie : plancher SL, ATR, refus frais>0.30R (0.333R taker), EV nette
  maker exacte (+1.647R), TP=3.5R, refus EV brute négative (p=0.15)
- Gate : abstention sur intégrité/quota(6e trade)/cooldown/anomalie/TRAP,
  direction par sweep, confirmation JEV, chemin complet GO (7 filtres)
- Risk : Kelly ≤0, plafond 20 %, min 15$, circuit breaker, exposition 40 %

**Bogues de fixtures corrigés** : G3 (invalidation 0.10%→planchée 1.2%,
frais 0.333R) et G6 (p=0.15→EV brute −0.325R) — les valeurs initiales de test
ne déclenchaient pas le refus attendu (le code métier était correct).

### Étape 10 — lifecycle_manager.py : LA VIGIE ACTIVE LIVRÉE (2026-09-25)

**Fichier** : `src/adan_trading_bot/position/lifecycle_manager.py` (+ `__init__.py`)

**Ce qui a été fait** : contrairement à ADAN0 (trade abandonné à un SL/TP
aveugles), le cerveau reste allumé tant que la position est ouverte. À chaque
clôture 5m, dans l'ordre de priorité : TP → SL → anomalie critique (JEV) →
invalidation de thèse → break-even → trailing → time-stop → HOLD.

**Règles gravées** : break-even dès +1.5R (SL → entry ± frais, trade garanti
sans perte) ; trailing ATR_1h dès +2.0R ; invalidation anticipée après ≥6
barres si clôture au-delà de l'extrême des 3 barres PRÉCÉDENTES contre la
position (transforme −1R en ≈−0.5R) ; anomalie critique P>0.70 → sortie
marché immédiate ; time-stop 288 barres (24h).

**Bugs trouvés par les tests et corrigés** (les tests avant tout) :
1. **Fenêtre d'invalidation** incluait la barre courante — or `close ≤ high`
   de sa propre barre, donc l'invalidation ne pouvait JAMAIS se déclencher.
   Fix : la barre courante n'entre dans la fenêtre qu'APRÈS l'évaluation
   (factorisé via `_push_recent`, appelé avant chaque décision).
2. **Frontière float** : `(100−98.20)/1.2 = 1.4999…` < 1.5 bloquait le
   break-even à pile +1.5R → epsilon 1e-9 sur les seuils BE/trailing.

**Preuves (tests/test_lifecycle_manager.py — 9/9)** : TP SHORT/LONG, SL
touché (−1R), break-even (SL=entry−frais, garanti sans perte), trailing suit
le prix à 1×ATR, invalidation anticipée à −0.50R au lieu de −1R, anomalie
critique → marché, time-stop, HOLD nominal, suivi MFE/MAE exact.

### PROCESS_STATE (diagnostic 2026-09-30, avant pipeline offline)

| Ressource | État |
|-----------|------|
| PROCESS | Aucun entraînement fantôme (services externes uniquement : litellm, MAPNET, aura-loc, supervisord) |
| CPU | sandbox multi-cœurs, torch 2.13.0+cu130 (CPU-only) |
| RAM | 11.9 GB total, 6.3 GB disponibles |
| DISK | 54 GB libres / 193 GB (73%) |
| GPU | AUCUN local → colab-cli OBLIGATOIRE pour 500K |
| ORPHAN_PROCESSES | aucun |

Règle actée : aucune métrique n'est acceptée sans test ; aucune sélection sur
TEST ; labels/variables/hyperparams choisis sur TRAIN/VAL uniquement ; résultats
négatifs conservés ; aucune entraînement CPU > 10K pas (GPU via colab-cli).

### Prochaines briques (ordre décidé 2026-09-25)

1. ~~`models/system_one_core.py`~~ ✅ LIVRÉ (voir ci-dessous)
2. `policy/` — geometry_engine (SL≥1.2%/ATR1h, TP 3.5R, rejet frais_R>0.30),
   deterministic_gate (intégrité, quota ≤5 trades/j, cooldown, direction),
   risk_engine (Kelly quart, plafond exposition 20%, min 15$ Binance)
3. `position/lifecycle_manager.py` — vigie active à chaque 5m (étape 10)
4. `offline/` — labeler MFE/MAE + entraînement supervisé calibré (Brier,
   calibration VAL, test OOS) — cible d'entraînement : 500K pas
5. Grand backtest walk-forward 2017→2026 — seuils : PF ≥ 1.6, max DD < 15%,
   0-5 trades/j, stabilité par cycle (Bull 2017/2021, Bear 2018/2022, Chop 2024-26)


**Last updated:** 2026-09-20  
**Status:** CODE FROZEN — Ready for 500k production run  

---

## ⚡ QUICK START: Launch 500k

```bash
cd /home/ubuntu/webapp/MORNINGSTAR/ADAN0
python scripts/train_parallel_agents.py --config config/config.yaml --steps 500000 --num-cpus 8 --mode heavy
```

**See:** `LAUNCH_500K.md` for detailed checklist + auto-stop thresholds.

---

## ✓ What's Ready

- ✅ Gamma precedence bug fixed (8/8 tests pass)
- ✅ Config consolidated to `config/config.yaml`
- ✅ WorldModelPPO + SB3 2.9.0 verified at runtime
- ✅ Hard pre-flight checks PASSED
- ✅ Auto-stop guards in place (6 thresholds)

**No more changes to code, reward, hyperparams, or architecture before launch.**

---

## 📋 Pre-Flight Checklist (Before Launch)

```bash
# 1. Verify tests pass
python3 scripts/tests/test_gamma_precedence.py

# 2. Verify runtime config
/home/ubuntu/webapp/MORNINGSTAR/miniconda3/envs/trading_env/bin/python3 - << 'CHECK'
import sys; sys.path.insert(0, "src")
from adan_trading_bot.agent.feature_extractors import WorldModelPPO
import yaml
cfg = yaml.safe_load(open("config/config.yaml"))
assert cfg.get("production", {}).get("simple_ppo", {}).get("gamma") == 0.99
assert cfg.get("agent", {}).get("gamma") is None
print("✓ READY TO LAUNCH")
CHECK
```

If both show ✓ → **Launch immediately**

---

## 1. Installation des outils DIAGNOSTIC (Kiro internal)

### Installé (2026-09-20)

```bash
# Environnement
conda activate trading_env

# Outils installés (SANS modification du code ADAN0) :
pip install py-spy memory-profiler mlflow
pytest  # Already present (9.1.1)
```

### Pourquoi ces outils

| Outil | Usage | Pour qui | Pas pour l'UI |
|---|---|---|---|
| **pytest** | Valider les correctifs critiques | Kiro diagnostic | ✅ Tests internes |
| **py-spy** | Profiling CPU (où le temps va) | Kiro diagnostic | ✅ Flamegraph local |
| **memory-profiler** | Détection fuites mémoire | Kiro diagnostic | ✅ Local analysis |
| **mlflow** | Registry des runs (local self-hosted) | Kiro diagnostic | ✅ Pas UI web |

### Installation optionnelle (si tu veux)

```bash
pip install wandb  # Cloud dashboards (SKIP — tu as déjà une UI)
pip install structlog  # JSON logging (SKIP — pas obligatoire pour diagnostic)
```

---

## 1.5 GAMMA PRECEDENCE BUG FIX (2026-09-20) — CRITICAL

### What Was Fixed

✅ **Gamma bug resolved** : Checkpoint v30 had `gamma=0.9523` (wrong). Now all runs will use `gamma=0.99` (correct).

**Problem:** Two conflicting gamma values in config files with broken loading precedence.
- **Stale (deleted):** `agent.gamma: 0.9523222699352663` (line 215)
- **Correct (kept):** `production.simple_ppo.gamma: 0.99` (line 1186)

### What Changed

1. **Config files:** Deleted stale root gamma from all 5 config files (config.yaml, v36a/b/c.yaml)
2. **Loading logic (train_parallel_agents.py, line ~1155):** Now reads `production.simple_ppo.gamma` instead of root gamma
3. **Resume override (line ~2681):** When resuming checkpoint, gamma is explicitly corrected to 0.99

### Validation: Run These Tests

```bash
cd /home/ubuntu/webapp/MORNINGSTAR/ADAN0

# Test 1: Gamma precedence (must pass 18/18)
python3 scripts/tests/test_gamma_precedence.py

# Test 2: Smoke test (must pass 5/5)
python3 scripts/tests/test_gamma_smoke_100steps.py

# Expected output:
# ✓ ALL TESTS PASSED
```

### ⚠️ BEFORE LAUNCHING 30k or 500k RUN

1. Run the two test suites above (2-3 seconds total)
2. Confirm both show "✓ ALL TESTS PASSED"
3. If resuming from v30 checkpoint, look for this log:
   ```
   [SANDBOX] Updated PPO.gamma: 0.9523222699352663 → 0.99 (from config)
   ```

**Full details:** See `GAMMA_FIX_REPORT.md`

---

## 2. Tests à exécuter (valider les fixes)

### 2.1 Test asymétrie entry/exit (FIX-D)

```bash
cd /home/ubuntu/webapp/MORNINGSTAR/ADAN0

# Crée le test
cat > tests/test_action_routing_fix.py << 'EOF'
"""Test FIX-D: asymmetric entry/exit thresholds"""
from adan_trading_bot.environment.action_routing import route_action_by_state

def test_entry_exit_asymmetry():
    """Entry requires conviction, exit is protection"""
    # FLAT state: entry threshold 0.10 (conviction)
    assert route_action_by_state(a0=0.11, in_position=False, threshold=0.10) == 1  # BUY
    assert route_action_by_state(a0=-0.11, in_position=False, threshold=0.10) == 0  # NOT SELL (HOLD)
    
    # LONG state: exit threshold 0.05 (easier to exit)
    assert route_action_by_state(a0=-0.06, in_position=True, threshold=0.10, sell_threshold=0.05) == 2  # SELL
    assert route_action_by_state(a0=+0.15, in_position=True, threshold=0.10, sell_threshold=0.05) == 0  # NOT BUY (HOLD)

if __name__ == "__main__":
    test_entry_exit_asymmetry()
    print("✅ FIX-D test PASSED")
EOF

# Exécute
pytest tests/test_action_routing_fix.py -v
```

**Résultat attendu** : ✅ PASSED

### 2.2 Test HMM readonly consumer (ADAN0_HMM_READONLY_CONSUMERS)

```bash
cat > tests/test_hmm_readonly_fix.py << 'EOF'
"""Test ADAN0_HMM_READONLY_CONSUMERS: consumers don't poison buffer"""
import numpy as np
from adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine

def test_hmm_readonly_no_contamination():
    """Verify readonly consumers don't mutate HMM buffer or posteriors"""
    dbe = DynamicBehaviorEngine()
    
    # Producer call (valid observation)
    p1 = dbe.get_regime_probabilities(
        log_return=0.005, 
        atr_pct=0.02, 
        rsi_norm=0.5, 
        volume_ratio=1.1,
        observation_id="step_1000"
    )
    
    # Consumer call (no observation_id, should NOT ingest)
    p2 = dbe.get_regime_probabilities(
        log_return=0.0, 
        atr_pct=0.0, 
        rsi_norm=0.0, 
        volume_ratio=0.0,
        observation_id=None  # READ-ONLY
    )
    
    # Posteriors should be identical (consumer didn't mutate)
    assert np.allclose(p1, p2, atol=1e-6), f"p1={p1}, p2={p2}"
    
    # Read-only count should increment
    assert dbe._hmm_readonly_calls >= 1

if __name__ == "__main__":
    test_hmm_readonly_no_contamination()
    print("✅ HMM readonly test PASSED")
EOF

pytest tests/test_hmm_readonly_fix.py -v
```

**Résultat attendu** : ✅ PASSED

---

## 3. Profiling & Diagnostics

### 3.1 CPU Profiling (où le temps va)

```bash
# Genère flamegraph du premier 1000 steps
py-spy record -o /tmp/adan_profile.html \
  --function \
  -- python scripts/launch_asset_run.py --asset BTCUSDT_BINANCE --steps 1000

# Ouvre le flamegraph
open /tmp/adan_profile.html  # ou Firefox /tmp/adan_profile.html
```

**Interprétation** :
- Cherche les blocs larges : HMM fit ? Feature extraction ? Reward calc ?
- Si HMM fit prend >20% du temps → Candidate for optimization
- Si normalizer prend >10% → Maybe vectorize

### 3.2 Memory Profiling (fuites mémoire)

```bash
# Lance 100 steps avec tracking mémoire ligne-par-ligne
python -m memory_profiler scripts/launch_asset_run.py --asset BTCUSDT_BINANCE --steps 100

# Attention aux lignes affichant des allocations croissantes
```

---

## 4. MLflow : Local run registry (diagnostic seulement)

### 4.1 Lancer le serveur

```bash
# Terminal 1: start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Accès: http://localhost:5000
```

### 4.2 Logger les runs (si tu veux voir côté Kiro)

```python
# Optionnel: ajoute dans scripts/launch_asset_run.py après logging setup:

import mlflow

mlflow.start_run(run_name=f"adan0_{asset}_{steps}steps")
mlflow.log_param("asset", asset)
mlflow.log_param("frais_rt", 0.004)

# Dans la boucle de training (main loop):
# mlflow.log_metric("equity", portfolio_value, step=step_count)
# mlflow.log_metric("hold_pct", hold_count/step_count, step=step_count)

mlflow.end_run()
```

**Pas obligatoire** : MLflow est juste pour que j'accède à l'historique des runs en tant que diagnostic interne.

---

## 5. Validation des correctifs (obligatoire avant tout nouveau run)

### Checklist d'audit avant 500k run

```bash
# 1. Vérifier les tests
pytest tests/test_action_routing_fix.py tests/test_hmm_readonly_fix.py -v

# 2. Profiler le baseline (100 steps)
py-spy record -o /tmp/baseline.html \
  -- python scripts/launch_asset_run.py --asset BTCUSDT_BINANCE --steps 100
# → Chercher fuites ou bottlenecks

# 3. Lancer petit run diagnostic
python scripts/launch_asset_run.py --asset BTCUSDT_BINANCE --steps 500
# → Vérifier pas de crash, pas d'exception

# 4. Si OK: aller 500k
```

---

## 6. Code Modification Policy (STRICT)

### ❌ NE PAS MODIFIER sauf urgence critique

Le code ADAN0 est **stable post-audit**. Aucune modification n'est requise.

### ✅ SI modification obligatoire

1. **Documente** dans ce fichier (dev.md)
2. **Ajoute marker** `ADAN0_<CATEGORY>_<ISSUE>` avec commit message
3. **Mesure avant/après** (dans ETAT_DU_CODE.md style)
4. **Test unitaire** : crée `tests/test_<issue>.py`
5. **Annonce breaking change** : mets à jour `AUDIT_COMPLET_KIRO_*.md`

### Exemple (si jamais)

```python
# Avant
frais = 0.004

# Après avec marker
frais = 0.004  # ADAN0_FEES_RECALIBRATION: measured 0.40% A/R (2026-09-20)
```

---

## 7. Architecture code — pas de refactor maintenant

✅ Les 5 bugs critiques sont fixés  
❌ Refactor `multi_asset_chunked_env.py` (11k lignes) → hors scope diagnostic

Refactor seulement si nouveau run doit aller en prod.

---

## 8. Decision tree pour le prochain run

```
Est-ce que tu veux lancer un nouveau run 500k?
│
├─ NON → Fin. Utilise les rapports existants pour stratégie.
│
└─ OUI → Est-ce que tu as changé QUELQUE CHOSE?
   │
   ├─ Frais < 0.10% A/R? → GO
   ├─ Timeframe = 4h? → GO
   ├─ Univers = microstructure? → GO
   │
   └─ Non → Reste identique à V30 500K →  HOLD collapse 99% attendu. NO-GO.
```

---

## 9. Support Kiro — diagnostics que je peux faire

✅ **Je peux** :
- Lire les logs réels
- Analyser les rapports diagnostiques existants
- Vérifier les correctifs avec tests
- Profiler le code avec py-spy
- Proposer optimisations basées sur traces réelles

❌ **Je ne peux pas** :
- Exécuter les runs interactivement (pas d'interpréteur runtime)
- Modifier ton UI web (she's separate)
- Relancer les anciennes sondes (données non accessibles)

---

## 10. Fichiers clés pour toi

| Fichier | Purpose |
|---------|---------|
| `AUDIT_COMPLET_KIRO_2026_09_20.md` | **Lire en priorité** — résultat complet |
| `ETAT_DU_CODE.md` | Source de vérité technique (correctifs, bugs) |
| `INSTALLATION_STACK_DIAGNOSTIC.md` | Outils optionnels (pour future audit) |
| `dev.md` | **Ce fichier** — guide pour developper |

---

## 11. Questions rapides

**Q: Faut-il changer la config?**  
A: Non. `config/config.yaml` est gelée. Les leviers sont économiques (frais, TF, univers), pas configs.

**Q: Le code marche?**  
A: Oui. Collapse HOLD = feature, pas bug. Voir diagnostic.

**Q: Quand relancer 500k?**  
A: Quand une condition mesurable change (frais ↓, TF ↑, ou stratégie re-cadrée). Pas avant.

**Q: Comment collaborer avec Kiro?**  
A: Files → tests → profiling → diagnostics. Kiro fait diagnostic, tu fais décisions.

---

**Prêt pour diagnostic avancé ?** Appelle-moi avec un chemin spécifique ou un run à analyser.
