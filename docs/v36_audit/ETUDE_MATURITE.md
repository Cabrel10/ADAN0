# Étude de maturité contrefactuelle ADAN — MaxDuration est-il vraiment l'ennemi ?

**Date** : 2026-08-23
**Méthode** : **zéro entraînement.** On reconstruit chaque entrée ADAN (V36-A2,
526 trades), on l'apparie à la barre 5m OHLCV, et on mesure MFE/MAE/R après
h = 5/10/20/40/80 barres. 492/526 entrées appariées (34 non trouvées).
**Scripts** : `adan_v36_audit/maturity_study.py`, `economic_autopsy.py`.
**Données** : `data/processed/indicators/train/BTCUSDT/5m.parquet` (7991 barres,
2026-05-01 → 2026-07-12).

---

## 0. TL;DR — la chaîne a DEUX ruptures, et on peut enfin les classer

> **La question posée** : « combien des trades tués à MaxDuration=20 auraient été
> gagnants avec 40 ou 80 barres ? »
> **Réponse** : presque aucun **au TP actuel (4 %)** — mais pas parce que 20
> barres suffisent, **parce que l'entrée n'a aucun edge** et que **le TP=4 % est
> ~28× l'ATR, physiquement inatteignable**.

Deux verrous, classés par ordre de causalité :

1. **VERROU RACINE — DÉTECTION (l'entrée n'a aucun edge).**
   Test décisif ADAN vs entrées aléatoires, 20 barres, long :
   | | meanR20 | meanMFE20 |
   |---|---|---|
   | entrées ADAN (492) | **-0.102 %** | +0.412 % |
   | entrées ALÉATOIRES (5000) | -0.068 % | +0.424 % |
   | **edge ADAN − aléatoire** | **-0.035 pt** | **-0.012 pt** |

   **Les entrées d'ADAN sont statistiquement indiscernables de points au hasard
   — et même légèrement PIRES.** Aucune politique de sortie, aucun horizon,
   aucun réglage de TP ne peut rentabiliser des entrées sans information.

2. **VERROU AMPLIFICATEUR — CONTRAT (TP dé-calibré à la volatilité).**
   ATR% médian = **0.143 %**. MFE médian à 20 barres = **0.30 % ≈ 2.1× ATR**.
   TP config = **4 % = ~28× ATR**. → le TP n'est atteint que **2.2 %** du temps
   même à 80 barres. D'où : 90.7 % des trades meurent au timer, et le résultat
   net est une marche quasi-aléatoire rognée par les frais.

MaxDuration=20 **tronque** bien de l'edge *si* le TP était atteignable (cf. §2),
mais il ne peut pas être « l'ennemi principal » tant que l'entrée ne porte
aucun signal exploitable. **MaxDuration est le 2e ou 3e maillon, pas le 1er.**

---

## 1. Courbe de maturité (médiane sur 492 entrées)

| h (barres) | MFE méd | MFE p75 | MAE méd | MAE p25 | R(h) méd |
|---|---|---|---|---|---|
| 5  | 0.13 % | 0.30 % | -0.15 % | -0.28 % | -0.01 % |
| 10 | 0.21 % | 0.40 % | -0.23 % | -0.42 % |  0.00 % |
| 20 | 0.30 % | 0.57 % | -0.32 % | -0.59 % | +0.02 % |
| 40 | 0.42 % | 0.83 % | -0.51 % | -0.90 % | -0.01 % |
| 80 | 0.65 % | 1.31 % | -0.73 % | -1.47 % | +0.01 % |

Lecture : le marché, vu depuis les entrées d'ADAN, est **symétrique** (MFE ≈ |MAE|
à chaque horizon) et **R(h) médian ≈ 0** partout. C'est la signature d'une marche
sans dérive : **il n'y a pas de direction gagnante à exploiter à l'entrée.**

---

## 2. Le TP est le vrai « chiffre magique » cassé (pas seulement l'horizon)

% d'entrées atteignant `MFE ≥ TP`, par TP et horizon :

| TP | h5 | h10 | h20 | h40 | h80 |
|---|---|---|---|---|---|
| 0.2 % | 38.2 | 50.8 | **62.0** | 74.0 | 84.1 |
| 0.3 % | 24.8 | 35.6 | **50.0** | 64.4 | 77.6 |
| 0.5 % | 10.4 | 20.9 | **30.3** | 43.7 | 61.2 |
| 1.0 % |  2.4 |  4.9 | 10.6 | 20.7 | 34.6 |
| 2.0 % |  0.2 |  1.0 |  1.4 |  5.3 |  9.3 |
| **4.0 %** | 0.0 | 0.0 | **0.2** | 0.6 | **2.2** |

- **Au TP config (4 %) : 0.2 % atteint à 20 barres, 2.2 % à 80.** Inatteignable.
- À un TP **calibré volatilité (0.3-0.5 %)** : 30-50 % atteignent à 20 barres,
  61-78 % à 80 barres. **L'horizon DOUBLE la reachability (h20→h80)** — donc
  MaxDuration=20 *tronque bien* de l'edge — MAIS seulement une fois le TP rendu
  atteignable.

**Conséquence : réparer l'horizon sans réparer le TP ne sert à rien, et réparer
le TP sans réparer l'entrée ne sert à rien non plus.**

---

## 3. Test « first-touch » (TP=SL=0.5 %, symétrique, long)

| H max | TP-first | SL-first | timeout/ambigu |
|---|---|---|---|
| 5  | 10.2 % |  7.3 % | 82.5 % |
| 10 | 18.5 % | 17.9 % | 63.6 % |
| 20 | 24.4 % | **26.8 %** | 48.8 % |
| 40 | 30.5 % | **37.4 %** | 32.1 % |
| 80 | 29.3 % | **34.6 %** | 36.2 % |

Même à un TP atteignable, **SL-first ≥ TP-first** dès 20 barres : côté long, plus
on attend, plus on touche le SL en premier. C'est cohérent avec **edge ≈ 0** :
allonger l'horizon sans signal directionnel **augmente le risque, pas le gain.**

---

## 4. Classification des entrées (TP=4 %)

| classe | n | % |
|---|---|---|
| ENTRY_GREAT (MFE≥8 % en ≤10 barres) | 0 | 0.0 % |
| ENTRY_GOOD (MFE≥TP en ≤20 barres) | 1 | 0.2 % |
| ENTRY_LATE (MFE≥TP mais >20 barres) | 10 | 2.0 % |
| **ENTRY_BAD (jamais MFE≥TP même à 80)** | **481** | **97.8 %** |

Au TP=4 %, tout est « BAD » — mais c'est un artefact du TP absurde. Le message
robuste vient du **test d'edge vs aléatoire (§0)** : indépendamment du TP,
**l'entrée ne bat pas le hasard.**

---

## 5. Ce que ça implique pour la refonte (contrat + cerveau)

L'ordre des chantiers est maintenant **imposé par les données** :

1. **DÉTECTION D'ABORD (le cerveau).** Tant que `edge(entrée) ≈ edge(aléatoire)`,
   rien en aval ne peut créer de l'expectancy. Il faut que l'entrée porte un
   signal : représentation (CNN/attention), features, et surtout un **critère
   d'entrée sélectif** (n'entrer que sur les situations où MFE attendu > coûts).
   → C'est ici que **ACP + clustering** servent : découvrir s'il existe des
   *clusters de situations* où MFE20 bat nettement l'aléatoire (edge conditionnel).
   Si de tels clusters existent → détection réparable. S'ils n'existent pas sur
   ce dataset → le dataset/timeframe n'offre pas d'edge et il faut changer
   d'univers (plus de volatilité, autre TF, autres actifs).

2. **CALIBRER TP/SL À LA VOLATILITÉ (contrat).** Exprimer TP et SL **en unités
   d'ATR**, pas en % fixe. Un TP ~2-3× ATR (≈0.3-0.5 % ici) est atteignable ;
   4 % (28× ATR) ne l'est jamais.

3. **PATIENCE ADAPTATIVE (contrat).** Remplacer `max_duration=20` constant par
   `H* = f(Q, régime DBE, volatilité ATR, budget cadence)`, borné par le modèle
   statistique de maturité (H_min ≤ H_agent ≤ H_max). Future Arena sert à
   **construire le professeur hors-ligne** (P(TP|state,h), P(SL|state,h),
   H* = argmax_h EV(h)), **jamais** comme observation runtime (anti look-ahead).

4. **CADENCE 7-15 trades/j = budget, pas récompense.** Un budget d'opportunités
   dépensé sur la qualité, jamais un bonus « fais un trade » (→ spam).

---

## 6. ACP + clustering — l'edge conditionnel EXISTE (résultat clé)

Étude complémentaire (`pca_clustering_edge.py`, zéro entraînement) : on
standardise les 16 indicateurs présents à t, ACP (9 composantes = 90 % variance),
KMeans k=8 sur les 7970 barres, puis outcome forward par cluster.

Baseline globale : R20 médian **-0.005 %** (aucune dérive), ATR médian 0.143 %.

**Cluster 4 (n=905, 11.4 % des barres) — edge réel et cohérent :**
| horizon | R médian | win > 0 |
|---|---|---|
| R10 | +0.066 % | 58.1 % |
| R20 | +0.137 % | **62.7 %** |
| R40 | +0.186 % | **65.4 %** |
| R80 | +0.130 % | 55.9 % |

Signature (z-score) : **RSI bas (-1.05), ADX haut (+1.02), di_delta bas (-1.12),
market_structure bas (-0.91)** = setup **survente dans une tendance forte**
(rebond long). Economiquement interprétable, pas un artefact.

Cluster 1 (n=632) montre aussi un edge long croissant (R80 +0.255 %, win 56 %).
Les 2 gros clusters « morts » (2 & 5, 62 % des barres) sont ≈ 0.

**→ Un edge d'entrée conditionnel existe sur ~11-19 % des barres, avec un
win-rate directionnel 62-65 % à 20-40 barres.** ADAN ne l'exploite pas (ses
entrées = aléatoires, §0), mais **un détecteur sélectif le pourrait.** Verdict :
**la DÉTECTION est réparable.**

Point capital pour le contrat : dans le cluster 4, **l'edge culmine à ~40 barres**
(R20 +0.137 → R40 +0.186), donc **MaxDuration=20 tronque justement ce cluster.**
L'horizon optimal est une **propriété du cluster**, pas une constante.

---

## 7. Verdict final (corrigé après clustering)

Ordre causal des 3 verrous, imposé par les données :

1. **DÉTECTION (racine).** Les entrées actuelles d'ADAN ≈ aléatoires (§0). MAIS
   un edge conditionnel réel existe (cluster survente+ADX fort, win 62-65 %,
   §6). → réparable par un **critère d'entrée sélectif** ; ne PAS entrer partout.
2. **CALIBRATION TP/SL EN ATR (contrat).** TP=4 % = 28× ATR = inatteignable ;
   viser ~2-3× ATR (≈0.3-0.5 %).
3. **HORIZON ADAPTATIF (contrat).** H* dépend du cluster/régime (~40 barres pour
   le cluster 4). `max_duration=20` constant tronque l'edge → remplacer par
   `H* = f(Q, régime DBE, ATR, budget cadence)`, borné par le modèle de maturité.

**Aucun nouveau reward, aucun 500k** tant que ces 3 verrous ne sont pas traités
dans cet ordre. A2 reste le dernier témoin de pondération.
