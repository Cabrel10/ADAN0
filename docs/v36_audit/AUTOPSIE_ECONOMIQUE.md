# Autopsie économique ADAN — pourquoi l'equity ne franchit jamais 21 $

**Date** : 2026-08-23
**Méthode** : analyse des jsonl déjà écrits (V36-A et V36-A2), **aucun entraînement supplémentaire**.
**Source** : `logs/rewards/worker_0_rewards_20260823_141325.jsonl` (A) et
`logs/rewards/worker_0_rewards_20260823_160703.jsonl` (A2), 50 176 steps chacun.
**Script** : `adan_v36_audit/economic_autopsy.py`.

---

## 0. La question centrale, répondue empiriquement

> « Pourquoi il n'arrive jamais à 21 $ en solde total ? »

Ce n'est pas une impression : c'est **mesuré sur chaque step**.

| | V36-A | V36-A2 (drawdown ÷4) |
|---|---|---|
| equity initiale | 20.50 | 20.50 |
| **equity MAX jamais atteinte** | **20.5408** | **20.5296** |
| gain max absolu jamais réalisé | **+0.041 $** | **+0.030 $** |
| equity min | 12.41 | 12.71 |
| equity moyenne | 16.68 | 17.73 |
| **steps avec equity ≥ 21 $** | **0 (0.0000 %)** | **0 (0.0000 %)** |

**Réponse directe : l'equity ne franchit jamais 21 $ parce qu'elle ne dépasse
même jamais 20.55 $.** Le meilleur point haut de toute l'histoire (100k steps
cumulés sur les deux bras) est **+0.04 $** au-dessus du capital de départ. Il ne
manque donc pas « 0.50 $ pour franchir 21 » — il manque toute capacité à
**construire** du capital. La trajectoire est une érosion lente 20.5 → 12–13,
puis ça remonte, sans jamais capitaliser au-dessus du départ.

Note : `episodes = 1`. **Il n'y a pas de reset par épisode** sur ce run (un seul
long épisode de 50k steps). L'hypothèse « reset efface les gains » est donc
**INFIRMÉE** pour ce run : le capital *est* continu, il ne monte simplement jamais.

---

## 1. La chaîne des 5 ruptures possibles — où casse-t-elle ?

Rappel du diagnostic demandé :

```
1. il ne détecte pas le bon trade
2. il le détecte mais choisit une trop petite taille
3. il choisit une bonne taille mais mauvais SL/TP
4. il choisit correctement mais ferme trop tôt / mal
5. il gagne mais le capital n'est pas réinjecté
```

### Étape 2 (taille trop petite) — **INFIRMÉE**

Le sizing **est appris et varie pleinement** :

| mesure (A2, 10 068 ouvertures) | valeur |
|---|---|
| `size_pct` action min / med / max | -1.000 / -0.082 / +1.000 |
| **écart-type `size_pct`** | **0.378** (≠ 0 → NON clampé) |
| exposition notional/equity min / med / max | 68.7 % / **80.5 %** / 90.4 % |
| exposition moyenne | **80.7 %** |

**→ Réponse à la Q7 : la taille de position n'est PAS clampée ni figée.** Le
réseau engage en médiane **80 % du capital** (soit ~16 $ sur 20.5 $), exactement
la « grosse position » que tu imaginais. Donc le problème n'est **pas** un
sous-dimensionnement. Ton intuition « prendre 15-18 $ » — il le fait déjà.

### Étape 5 (capital non réinjecté / compounding) — **INFIRMÉE pour ce run**

`episodes = 1`, cash et equity sont continus d'un step à l'autre (`cash_after`
du step N = `cash_before` du step N+1). Le capital gagné *serait* réutilisable.
Le problème n'est pas la plomberie du compounding : c'est qu'il n'y a **rien à
réinjecter** (expectancy négative).

### Étape 4 (mauvaise gestion de sortie) — **CONFIRMÉE : c'est ICI que casse la chaîne**

Distribution des raisons de clôture (identique sur les deux bras) :

| raison de clôture | V36-A | V36-A2 |
|---|---|---|
| **MaxDuration** (timer force la sortie) | **89.9 %** | **89.4 %** |
| AGENT_CLOSE (l'agent décide) | 7.4 % | 6.8 % |
| stop_loss | 2.4 % | 3.6 % |
| **take_profit** | **0.2 % (1 trade / 623)** | **0.2 % (1 trade / 526)** |

- Durée de détention : **médiane 6000 s = 20 barres 5m** = exactement le plafond
  `max_duration_steps: 20` du config. **90 % des trades meurent au chronomètre.**
- **Le take-profit ne se déclenche presque JAMAIS (1 fois sur 500-600).**

**→ Le trade n'est ni piloté par une décision de sortie, ni par le TP : il est
tué par un timer.** Le réseau ouvre une position à ~80 % d'exposition, puis
**subit** le marché pendant 20 barres et sort à ce que le prix veut bien lui
donner à l'instant T+20. C'est une sortie **aveugle**.

### Étape 1 & 3 (détection + géométrie SL/TP) — **contributifs, en amont de l'étape 4**

SL/TP **sont** pilotés par le réseau (`ADAN_FREE_SLTP=1`, enveloppe SL [0.3 %,6 %],
TP [0.3 %,12 %]) et varient (std SL 0.391, std TP 0.375). Mais :

- TP médian placé ~à peine positif, et surtout **jamais atteint** dans la
  fenêtre de 20 barres → sur des barres 5m BTC, une excursion favorable
  suffisante pour toucher le TP en < 100 min est rare.
- Résultat par trade (A2) : gain moyen gagnant **+0.096 $**, perte moyenne
  perdante **-0.086 $**, **win rate 18.8 %**, **expectancy -0.052 $/trade**.

Avec 18.8 % de gagnants à +0.096 et 81.2 % de perdants à -0.086 :
`0.188×0.096 − 0.812×0.086 = +0.018 − 0.070 = −0.052 $/trade`. Négatif.

---

## 2. Distinction A / B / C (opportunités)

- **C = % des positions ADAN rentables** : **≈ 18-19 %** (mesuré, ci-dessus).
- **A = % d'opportunités marché rentables** : ton observation ≈ **25 %**
  (non recalculable ici car le jsonl ne journalise pas le prix hors position ;
  à mesurer séparément sur le dataset OHLCV, cf. §4).
- **B = % d'opportunités réellement captées** : non mesurable directement ici,
  mais le signe fort est que **la sortie est un timer, pas une décision** — donc
  même quand ADAN entre sur une bonne opportunité (A), il ne la **tient pas
  jusqu'à la récompense** : il la coupe à 20 barres. B est donc probablement
  petit **non par manque d'entrées, mais par destruction systématique des
  sorties**.

**Conclusion A/B/C : le goulot n'est ni A (opportunités existent) ni le sizing
(80 % engagé). Le goulot est la transformation opportunité → PnL réalisé, tuée à
l'étape SORTIE.**

---

## 3. Ce que cela prouve sur le contrat de reward (et pourquoi A2 est le dernier test simpliste)

A → A2 a divisé `drawdown_penalty` par 4. Effet mécanique réel :
`drawdown_penalty` 58.6 % → 26.6 % de l'amplitude, `pnl_reward` 22.6 % → 39.6 %,
et pour la première fois **reward moyen gagnants (+0.027) > reward moyen perdants
(-0.328)**. PnL brut -37.2 → -27.3. **Mais** : BUY = 0 (politique 100 % SELL),
PF 0.257, expectancy toujours négative, et surtout **les ratios de clôture n'ont
PAS bougé** (MaxDuration 90 %, TP 0.2 % dans les deux). 

**→ Re-pondérer le reward ne touche pas la vraie brique cassée** : la géométrie
temporelle de sortie. C'est exactement pourquoi la règle « A2 = dernier test de
pondération simpliste » est juste. **Décision : NO-GO 500k, et STOP aux
micro-corrections A3/A4/A5.**

---

## 4. Refonte proposée (contrat, pas curseur) — à valider AVANT tout V37

La brique qui « ne parle pas la même langue » est identifiée : **le couple
{horizon de détention, condition de sortie}**. Le réseau décide l'entrée, la
taille, le SL et le TP, mais **la sortie effective est dictée à 90 % par un timer
de 20 barres** que la politique ne contrôle pas et qui tombe presque toujours
avant le TP.

Trois vérifications/chantiers, dans l'ordre, **sans toucher au reward** :

1. **Mesurer A proprement** (opportunité marché) : sur le dataset OHLCV, pour
   chaque barre, calculer MFE/MAE sur 20 barres et `profit_at_1x/2x/5x/10x`.
   → confirme si un TP atteignable existe dans la fenêtre, ou si 20 barres est
   simplement trop court pour l'enveloppe TP choisie.
2. **Aligner horizon et TP** : soit allonger `max_duration_steps`, soit resserrer
   l'enveloppe TP pour qu'un TP soit *physiquement atteignable* en 20 barres.
   C'est un paramètre de **contrat** (géométrie), pas un poids de reward.
3. **Rendre la sortie une vraie décision** : donner à la politique un signal/So
   reward sur *tenir vs couper* une position gagnante, au lieu de la faire mourir
   au chronomètre. Objectif unique : `maximiser E[log(equity_{t+1}/equity_t)]`,
   risque et drawdown en **contraintes**, sortie/hold en **décision**.

---

## 5. Verdict

- **Équité ne franchit jamais 21 $** parce que le plus haut jamais atteint est
  **20.55 $** : ADAN n'érode pas un capital qu'il fait grossir, il ne le fait
  **jamais grossir**.
- **Premier point de rupture de la chaîne = étape 4 (sortie)** : 90 % des trades
  fermés par un **timer de 20 barres**, TP touché **0.2 %** du temps. Le sizing
  (80 % d'exposition) et le compounding (épisode continu) **fonctionnent** ; la
  détection/géométrie contribuent mais en amont de la sortie.
- **A2 = NO-GO** et **dernier test de pondération** : la re-pondération n'a pas
  touché les ratios de sortie → prochaine étape = **refonte du contrat
  {horizon, sortie}**, pas un A3.
