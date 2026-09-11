# Rapport — géométrie SL/TP : la contrainte liante est les FRAIS, pas la boîte

> Mesuré sur les parquets réels `data/processed/indicators/train/BTCUSDT_BINANCE/`
> (5m : 662 643 barres, 1h, 4h). Chaque chiffre est reproductible par les sondes
> citées. Lecture seule : aucun fichier de production modifié.

## 0. Question posée

Le diagnostic du run `v30_500k` (collapse hold 99 %, win rate 1.5 %, RR 0.318)
désignait la géométrie SL/TP comme « prochain levier » : `sl_hi=0.0235` /
`tp_lo=0.0135` impose RR < 1 et exige 60-74 % de win rate. La question était :
**quelle boîte SL/TP rend le trade net +EV ?**

## 1. Méthode — premier passage

Pour chaque paire `(sl, tp)` d'une grille et chaque barre d'entrée
échantillonnée du parquet réel :
- TP touché en premier (`high >= entry*(1+tp)`) → gain `+tp`
- SL touché en premier (`low <= entry*(1-sl)`) → perte `-sl`
- ni l'un ni l'autre avant H barres → MaxDuration, clôture au marché
- si high ET low touchent la même barre → SL (pessimiste, ordre intra-barre inconnu)

`EV_net = P(TP)·tp − P(SL)·sl + P(MD)·E[ret|MD] − round_trip_fees`
avec `round_trip_fees = 0.004` (0.40 %, mesuré sur le run réel : 0.40 %
implicite, config `commission 0.002`/côté).

Sondes : `diag_sltp_first_passage.py`, `diag_sltp_conditioned.py`,
`diag_fee_frontier.py`. Rapports JSON dans `logs/validation/`.

## 2. Résultat 1 — inconditionnel : jeu à somme nulle avant frais

Grille 7×6 (sl 0.3-2.35 %, tp 0.6-3.0 %), H=40 et H=100 barres 5m :

| Mesure | Valeur |
|---|---|
| Paires +EV net | **0 / 42** |
| EV net de CHAQUE paire | ≈ **−0.0039** (= −frais) |
| Paire actuelle (0.0235 / 0.0135) | EV = −0.0039 |
| Dispersion entre meilleure et pire paire | < 0.0002 |

L'EV de toute la grille est **identique aux frais près**. Le 5m BTC est un jeu
à somme nulle avant frais : déplacer SL/TP ne fait que redistribuer entre
P(TP), P(SL) et P(MD), sans créer d'espérance. **Aucune boîte SL/TP statique
ne peut être profitable à ce niveau de frais.**

## 3. Résultat 2 — conditionné : aucun filtre simple ne crée d'edge net

11 filtres d'entrée testés (momentum 6b/24b, mean-reversion, RSI<30, RSI>70,
ATR bas/haut, tendance EMA20 up/down), chacun × 42 paires × 2 horizons :

| Filtre | Meilleure paire | EV net | Edge brut |
|---|---|---|---|
| rsi>70, H=100 | (0.012, 0.030) | **−0.0029** | **+0.107 %** |
| trend_down, H=100 | (0.0235, 0.010) | −0.0037 | +0.030 % |
| meanrev_24b, H=100 | (0.0235, 0.010) | −0.0037 | +0.030 % |
| momentum_6b, H=100 | (0.0235, 0.030) | −0.0039 | +0.010 % |
| ALL (inconditionnel) | (0.0235, 0.030) | −0.0038 | +0.020 % |

Le meilleur filtre simple (RSI>70, contre-tendance) dégage un edge brut de
**+0.107 %** — encore **3.7× sous les frais de 0.40 %**. Le point mort de
frais pour cette config est 0.107 % ; il faudrait des frais ≤ 0.10 % pour la
rendre nettement positive.

## 4. Résultat 3 — la frontière de frais ne se franchit pas en montant de TF

Meilleur edge brut par timeframe (grille élargie jusqu'à sl 3 %, tp 6 %) :

| Timeframe | H | Meilleure paire | Edge brut (= frontière frais) | Viable à 0.40 % ? |
|---|---|---|---|---|
| 5m | 40 | (0.030, 0.060) | +0.010 % | NON |
| 5m | 100 | (0.030, 0.060) | +0.027 % | NON |
| 1h | 40 | (0.020, 0.060) | +0.116 % | NON |
| 4h | 30 | (0.020, 0.060) | **+0.148 %** | **NON** |

L'edge croît avec le timeframe (×14 de 5m à 4h) car les mouvements visés sont
plus amples, mais **même en 4h la frontière reste 2.7× sous les frais actuels**.

**Test de robustesse split-half (4h, H=30)** : la même paire (0.020, 0.060)
gagne sur les deux moitiés du dataset (+0.175 % / +0.120 %), et l'évaluation
out-of-sample donne +0.120 %. L'edge 4h est réel, pas un sur-ajustement — il
est simplement trop petit face à 0.40 %.

## 5. Conclusion — CONFIRMÉ

- **CONFIRMÉ** : la géométrie SL/TP n'est PAS le levier. Aucune boîte statique,
  aucun filtre d'entrée simple, aucun timeframe parmi {5m, 1h, 4h} n'atteint
  un EV net positif à 0.40 % de frais round-trip sur BTC.
- **CONFIRMÉ** : la contrainte liante est le **niveau de frais** (0.40 % A/R).
  L'edge maximal mesuré (+0.148 % en 4h) exigerait des frais ≤ 0.148 %, soit
  une réduction de **63 %**.
- **INFIRMÉ** : « réparer la géométrie SL/TP suffit à débloquer un run 500k ».
  C'est faux à frais constants — mesuré, pas déduit.
- **PROBABLE** : un edge exploitable nécessite soit des frais réels plus bas
  (maker fees, autre venue), soit un signal prédictif plus fort que les filtres
  simples testés (c'est ce que le RL pourrait, en principe, apprendre — mais
  les 3 runs 500k montrent qu'il ne l'a pas fait : collapse en hold).

## 6. Implications pour la décision 500k

1. **Ne pas lancer de 500k en changeant uniquement la boîte SL/TP** : mesuré
   sans effet (EV ≈ −frais quelle que soit la paire).
2. Le collapse en hold de la policy (99 %) est, vu ces mesures, une **réponse
   rationnelle** à un univers où tout trade a une espérance négative de
   −0.40 %. La policy a appris à ne pas trader parce que trader détruit de
   l'equity. Ce n'est pas une panne d'apprentissage, c'est la bonne réponse
   au problème posé.
3. Pour rendre un run économiquement viable, il faut agir sur l'un de :
   - **les frais** (descendre sous ~0.15 % A/R : maker-only, venue moins
     chère, ou actif à spread plus faible) — levier le plus direct ;
   - **le timeframe** (le 4h a la plus haute frontière, +0.148 %) — nécessaire
     mais pas suffisant seul ;
   - **le signal** (au-delà des filtres simples) — non démontré par le RL à ce
     jour.

## 7. Note de méthode

Trois sondes indépendantes, trois méthodes (grille inconditionnelle, filtres
conditionnels, frontière par timeframe) convergent vers la même borne :
l'edge brut maximal disponible sur ces données est de l'ordre de **+0.10 à
+0.15 %**, contre des frais de **0.40 %**. La marge n'existe pas à ce niveau
de frais. Mesurer coûte moins cher que discuter — ici, mesurer a renversé
l'hypothèse « la géométrie est le levier ».

---
*Rapport généré le 2026-09-09. Sondes : `scripts/diagnostics/diag_sltp_first_passage.py`,
`diag_sltp_conditioned.py`, `diag_fee_frontier.py`. Données :
`data/processed/indicators/train/BTCUSDT_BINANCE/{5m,1h,4h}.parquet`.*

---

# ADDENDUM — la piste « edge conditionné 4h » est INFIRMÉE (artefact de superposition)

> Ajout du 2026-09-09, même session. Une extension naturelle des sections 2-4
> a failli produire une conclusion fausse. Documentée ici pour que personne
> ne la ressorte comme piste valide.

## A. Ce qui a été observé (in-sample, entrées superposées)

Le scan conditionné étendu aux TF 1h/4h montrait des edges bruts
apparemment exploitables en 4h : `rsi>70` +0.714 %, `ema>1` +0.522 %,
`atr<p25` +0.518 % (paire 0.030/0.060, H=30). Le test OOS sur entrées
superposées semblait confirmer `ema>1` : +0.522 % train / +0.414 % val /
+0.234 % test — signe consistant sur les 3 splits, y compris sur le test
baissier (−9.2 % buy&hold).

## B. Le biais

Entrées à stride=1 avec horizon H=30 → chaque trade recouvre ~97 % du
suivant. L'échantillon effectif n'est pas n=1 412 mais ~50 trades
indépendants. Les percentiles d'entrée étaient aussi superposés : le
« signal » mesurait la même poignée de mouvements 60 fois.

## C. Le test honnête — entrées NON superposées (stride = H)

| split | filtre | n indép | PnL moyen | t-stat | IC95% |
|---|---|---|---|---|---|
| train | ALL | 460 | +0.205 % | +1.11 | ±0.362 % |
| train | ema>1 | 237 | **+0.670 %** | **+2.52** | ±0.521 % |
| val | ema>1 | 51 | +0.009 % | +0.02 | ±1.065 % |
| test | ema>1 | 50 | **−0.183 %** | −0.38 | ±0.934 % |

Delta `ema>1 − ALL` : train **+0.465 %**, val **+0.023 %**, test **+0.016 %**.

## D. Verdict — INFIRMÉ

L'edge `ema>1` n'existe que sur le train (période +904.9 % buy&hold : la
tendance longue y est gratuite) et s'évanouit intégralement sur val et test
dès que les trades sont indépendants. `atr<p25` changeait déjà de signe
entre val et test sur échantillons superposés ; `rsi>70` n'a jamais eu
d'effectif suffisant (n<100). **Aucun filtre simple testé ne produit un
edge net reproductible hors échantillon, à aucun des niveaux de frais
simulés (0.04 % à 0.40 % A/R), sur aucun des trois timeframes.**

## E. Conséquence pour la décision 500k — NO-GO économique justifié

La chaîne complète est maintenant mesurée de bout en bout :
1. toute géométrie SL/TP statique : EV ≈ −frais (§2) ;
2. tout filtre d'entrée simple : pas d'edge net après frais (§3) ;
3. monter de timeframe ne franchit pas la frontière (§4) ;
4. le seul edge conditionné apparent était un artefact statistique (§A-D).

Un run 500k sur cette config économique (0.40 % A/R) ne peut pas produire
de croissance de capital par construction : chaque trade a une espérance
de −0.40 % et aucune source de alpha mesurable n'a été démontrée. Le
collapse hold 99 % de la policy est la **bonne** réponse à ce problème.

Conditions mesurables de déblocage (au moins l'une) :
- frais réels ramenés ≤ 0.10 % A/R **et** démonstration préalable, sur
  backtest non superposé, d'un signal net > frais sur val ET test ;
- ou re-cadrage du problème (market-making, horizon multi-jours avec
  funding, cross-asset) — hors périmètre de la config actuelle.

Tant qu'aucune de ces conditions n'est **mesurée** (pas supposée), tout
lancement 500k reste NO-GO. C'est le « NO-GO justifié par des raisons
précises » que prévoit le protocole.
