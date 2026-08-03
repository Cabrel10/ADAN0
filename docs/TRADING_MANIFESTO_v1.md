# TRADING MANIFESTO v1 — Définition formelle du trader & test de validation

> RÈGLE : on ne valide rien sans test clairement défini ni sans vision claire,
> établie et exploitable. Ce document est écrit AVANT toute modification de reward.
> Il définit (1) le trader hypothétique de référence, (2) la métrique de déviation,
> (3) le mécanisme correctif, (4) les critères de succès CHIFFRÉS, (5) les
> garde-fous anti-excès-inverse.

---

## 0. CONSTAT MESURÉ (socle factuel, pas d'hypothèse)

Comparaison des 11 derniers runs au comportement sain (source : `diag_*.csv`) :

| RUN     | last | a0_mean_fin | slope/step | buy_end | sell_end | collapse@ |
|---------|------|-------------|------------|---------|----------|-----------|
| v9      | 8k   | +0.091      | +1.1e-05   | 0.726   | 0.230    | jamais    |
| v10     | 70k  | +0.285      | +3.5e-06   | 0.975   | 0.021    | jamais*   |
| v11     | 78k  | +0.273      | +3.2e-06   | 0.970   | 0.025    | jamais*   |
| v12     | 40k  | +0.301      | +7.8e-06   | 0.979   | 0.015    | jamais*   |
| v13     | 88k  | +4.315      | +5.0e-05   | 1.000   | 0.000    | 17k       |
| hc012   | 110k | +1.325      | +1.3e-05   | 1.000   | 0.000    | 24k       |
| hc016   | 28k  | +0.277      | +1.0e-05   | 0.972   | 0.015    | jamais*   |
| hc020   | 15k  | −0.238      | −1.7e-05   | 0.052   | 0.938    | SELL-run! |
| td_iso  | 10k  | +0.191      | +1.8e-05   | 0.902   | 0.074    | jamais*   |
| archfix | 24k  | +0.183      | +7.3e-06   | 0.902   | 0.080    | jamais*   |
| selfix  | 366k | +6.524      | +1.9e-05   | 1.000   | 0.000    | 16k       |

*« jamais » = pas atteint AVANT la fin du run, mais slope>0 => en route vers collapse.

**Verdict global : AUCUN run n'a jamais atteint le comportement sain.**
Tous dérivent vers BUY-runaway (a0→+∞), sauf hc020 qui a fait l'excès INVERSE
(SELL-runaway). C'est la preuve que le problème n'est ni le routage, ni un seuil,
ni holding_cost : **il manque la définition de ce qu'est un bon trade / une position
qu'on doit couper.**

---

## 1. LE TRADER HYPOTHÉTIQUE DE RÉFÉRENCE (oracle)

Sur une fenêtre de N steps (profil 5m), un trader sain :
- passe **~92-97 % du temps en HOLD** (flat en attente OU en position en gestion) ;
- a **count(BUY) ≈ count(SELL) ± 1** (invariant comptable : toute position ouverte
  finit par se fermer) ;
- réalise **1 à 20 trades complets** sur 1000 min (200 steps) selon l'agressivité ;
- **coupe ses positions perdantes** au lieu de les laisser dériver jusqu'au SL/Max
  Duration.

Le signal d'alarme n'est PAS « BUY existe » ni « HOLD domine » (c'est sain), c'est
quand **BUY et SELL divergent durablement** (buy→1, sell→0) OU l'inverse.

---

## 2. LA MALADIE, CHIFFRÉE (reward_components_selfix, n=30 + simulation)

Asymétrie mesurée dans le reward :

| État LONG × action        | raw_reward | interprétation             |
|---------------------------|------------|----------------------------|
| + BUY (a0>0, routé no-op) | −0.0038    | GRATUIT (aucune réalisation)|
| + SELL (a0<0, réalise)    | −0.3041    | PUNI 80× (PnL réalisé)     |

Le `latent_pnl` EXISTE déjà (L.6699-6727) mais est **calibré invisible** :
- compression `log1p(u*10)/10` → une perte de −2 % ne donne que **−0.0027/pas** ;
- appliqué **1 pas sur 3** seulement ;
- **cumul sur un trade tenu −2 % / 20 pas = −0.0164** contre **−0.30 pour vendre**.
- ⇒ tenir la perte est **18× moins cher** que la couper. PPO l'apprend → a0→+∞.

C'est le **disposition effect** (aversion à réaliser la perte) câblé dans une
mauvaise calibration.

---

## 3. LE CORRECTIF (une seule variable changée : le Delta Latent PnL)

**Principe (« battement de cœur »)** : tant qu'on est en position, CHAQUE pas doit
refléter la variation du PnL non réalisé. Tenir une position qui saigne DOIT faire
mal, proportionnellement au saignement — sinon « ne rien faire » reste gratuit.

**Formule cible** (linéaire, chaque pas, asymétrique, plafonnée) :
```
u = (current_price - entry_price) / entry_price      # PnL latent fractionnaire (SPOT long)
si u >= 0 :  contrib = +min(cap, lambda_gain * u)     # gain latent doucement récompensé
si u <  0 :  contrib = -min(cap, lambda_loss * |u|)   # perte latente clairement punie
```
avec `lambda_loss > lambda_gain` (asymétrie : couper une perte doit être plus urgent
que savourer un gain). Chaque pas (pas 1/3). Plafond `cap` pour éviter l'explosion.

**Effet attendu** : sur un trade tenu −2 % pendant 20 pas, le cumul devient
comparable ou pire que −0.30 → **VENDRE redevient l'action rationnelle**. Le no-op
BUY-while-long n'est PLUS gratuit car le latent négatif s'applique quel que soit
`discrete_action` tant que la position est ouverte.

---

## 4. GARDE-FOU ANTI-EXCÈS-INVERSE (leçon de hc020)

Risque identifié par l'utilisateur : rendre la sortie trop facile/douloureuse peut
provoquer l'excès inverse (SELL-runaway, over-trading érodant le capital en frais
0.5 %). Trois protections **mesurées**, pas supposées :

1. **`cap` sur la contribution latente** : borne la douleur par pas (pas de panique).
2. **Le gain latent reste récompensé** : tenir une position GAGNANTE reste rentable
   → l'agent n'a pas intérêt à tout vendre instantanément (sinon il perd le gain
   latent futur). Ça crée l'équilibre : couper les perdantes, laisser courir les
   gagnantes.
3. **Métrique de churn surveillée en continu** : `AGENT_CLOSE` et durée moyenne de
   détention. Si durée moyenne < 3 pas → over-trading → ÉCHEC du run.

---

## 5. TEST DE VALIDATION — critères CHIFFRÉS (exploitable, binaire)

Le run est un **SUCCÈS** si et seulement si, sur un horizon **> 100k steps**
(au-delà de l'horizon historique de collapse ~16-24k) :

| # | Critère                              | Seuil de SUCCÈS            | Échec si |
|---|--------------------------------------|----------------------------|----------|
| S1| a0_mean ne diverge pas               | \|slope\| < 5e-06/step ET \|a0_mean\| < 0.5 @100k | slope>1e-05 soutenu |
| S2| pct_buy ne sature pas                | pct_buy < 0.90 @100k       | pct_buy ≥ 0.99 |
| S3| SELL survit                          | pct_sell > 0.02 @100k      | pct_sell = 0 sur 20 fenêtres |
| S4| BUY≈SELL appariés                    | 0.5 < count(BUY)/count(SELL) < 2.0 | ratio > 5 ou < 0.2 |
| S5| Pas d'over-trading (anti-hc020)      | durée détention moy > 3 pas ET AGENT_CLOSE/SLTP < 5 | durée < 3 pas |
| S6| Capital préservé                     | portfolio_final > 0.9 × 20.5 | < 0.7 × capital |

**Un seul critère parmi S1-S3 en échec = collapse non résolu = run ÉCHEC.**
**S4-S6 en échec = excès inverse / over-trading = run ÉCHEC.**
Le run n'est déclaré gagnant que si **les 6 critères** tiennent après 100k.

---

## 6. CE QU'ON NE TOUCHE PAS (contraintes absolues)

- FRAIS à 0.5 % (`commission: 0.0025`, `round_trip_fees: 0.005`) — INTACT.
- action dims 1-4 (Size/TF/SL/TP = Future Arena/Oracle) — INTACT.
- PAS de VecNormalize, PAS de sb3-contrib/MaskablePPO.
- obs_schema 28 dims, capital 20.5, exposure [70,90], min_order 11.0 — INTACT.
- On change **UNE seule variable** : la calibration du `latent_pnl` (delta latent).
  Tout le reste du reward (future_contrib, closure_bonus, symmetry) est CONSERVÉ.

---

## 7. POURQUOI CE FIX ET PAS UN AUTRE (traçabilité)

- holding_cost : DISPROUVÉ (ne fait que retarder, hc012@110k, hc016@28k).
- FIX D routing : DISPROUVÉ (contourné par migration de distribution, selfix@16k).
- Delta latent PnL : c'est le SEUL mécanisme qui rend le no-op non gratuit SANS
  toucher au routage ni aux frais. Il attaque la cause racine MESURÉE (asymétrie 80×)
  directement là où elle est (le reward par pas en position), pas un symptôme.

Prochaine étape (si ce fix passe S1-S6) : brancher le tuteur mort (RewardCalculator)
et la déviation vs oracle sur fenêtre glissante — mais UNE variable à la fois.
