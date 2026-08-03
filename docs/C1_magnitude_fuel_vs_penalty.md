# C1 — Magnitude noir-sur-blanc : le fix V9 PRÉVIENT-il ou RETARDE-t-il le collapse ?

**Question de l'utilisateur (concern #1)** : *« à quel step la pénalité `min_notional_self_caused`
atteint-elle une magnitude qui dépasse le `latent_pnl_contrib` cumulé par step (le carburant) ?
Sinon le collapse peut simplement se reproduire mais démarrer un peu plus tard. »*

Réponse calculée à partir des **formules réelles du code** (pas d'estimation) :
`multi_asset_chunked_env.py` L.6639-6667 (carburant) et L.7353-7378 (frein), constantes de
`config.yaml`.

---

## 1. Le CARBURANT — `latent_pnl_contrib` par step

Formule (gain) : `c = lambda_gain * log1p(u*10)/10`, plafonné à `cap=0.30`, appliqué **1 fois
toutes les 3 steps** (`every_n_steps=3`), **par position ouverte**. En SPOT leverage=1 il n'y a
qu'**une** position BTC → une seule contribution.

Constantes : `lambda_gain=0.10`, `every_n=3`, `cap=0.30`.

| PnL latent `u` | c (par application) | moyenne par step (÷3) |
|---|---|---|
| +1 % (courant) | 0.000953 | **0.000318** |
| +10 % (tendance forte) | 0.006931 | **0.002310** |
| plafond dur (cap) | 0.300000 | **0.100000** |

Le carburant réaliste par step est **très petit** (~3e-4). Le carburant MAX théorique (0.10/step)
n'est atteint que si le PnL latent sature le plafond log — cas extrême.

## 2. Le FREIN — pénalité V5 `min_notional_self_caused`

`acc_t = 0.97·acc_{t-1} + 0.55` → point fixe `acc* = 0.55/0.03 = 18.33`.
`pen = 0.02 · min(15, 1+0.45·acc) · mult · ramp`, plafonné à 0.30.
À `acc*` : `term = 1+0.45·18.33 = 9.25` (n'atteint PAS le plafond `CAP/BASE=15`).
`ramp = min(1, step/50000)` (warmup).

**Pénalité à régime établi** (spam continu, acc saturé à 9.25) :
- `pen = 0.02·9.25·mult·ramp = 0.185·mult·ramp` (plafonnée à 0.30).

## 3. Croisement frein vs carburant — LE VERDICT

| Carburant | Frein le dépasse à (mult=1) | (mult=3, collapse détecté) |
|---|---|---|
| réaliste +1 % (3e-4/step) | **step ~100** | step ~50 |
| fort +10 % (2.3e-3/step) | **step ~1000** | step ~500 |
| **MAX cap (0.10/step)** | **step ~50 000** | **step ~10 000** |

## 4. Interprétation — nuancée mais honnête

**BONNE nouvelle (cas nominal)** : contre le carburant *réaliste* (+1 %) et même *fort* (+10 %),
le frein gagne **avant 1000 steps**, même pendant le warmup, même à mult=1. Dans le régime de
marché normal, le fix V9 **prévient** le collapse — il ne se contente pas de le retarder.

**MAUVAISE nouvelle (le point exact soulevé par l'utilisateur)** : contre le carburant **MAX**
(PnL latent proche du plafond, tendance haussière très forte tenue plusieurs steps), le frein ne
dépasse le carburant qu'à **~50 000 steps (mult=1)** ou **~10 000 steps (mult=3)** à cause du
**warmup ramp de 50 000**. Or c'est **exactement la fenêtre 0–8 000** où le biais BUY de v9 était
monté à 0.72. → **Dans un marché en forte tendance, le warmup laisse une fenêtre où le carburant
peut dominer le frein et amorcer le biais.**

**Conclusion** : le fix V9 est correct sur le *signe* (plus de +0.002) et suffisant en régime
normal, MAIS le **warmup de 50 000 est un risque résiduel** précisément dans le scénario
défavorable. Le collapse pourrait être **retardé plutôt que prévenu** si le marché de la fenêtre
d'entraînement est fortement haussier sur les premiers dizaines de milliers de steps.

## 5. Décision (avant de brûler le compute)

Deux leviers, **sans toucher aux frais ni au reward** :

**(a) Réduire `sterile_warmup_steps` 50000 → 15000** pour `min_notional_self_caused`. À 15k, la
rampe est pleine bien avant 128k et le frein dépasse le carburant MAX dès ~15k (mult=1) / ~5k
(mult=3). Le warmup reste assez long pour laisser PPO explorer les actions *légales* sans être
puni trop tôt (le warmup avait été mis pour ça), mais coupe la fenêtre de vulnérabilité.

**(b) Garder 50000 mais surveiller empiriquement** `a0_pct_buy` sur les 0–15k premiers steps
(comme demandé par l'utilisateur : « vérifier empiriquement sur les premiers milliers de steps »).
Si le biais BUY repart > 0.65 avant 15k → tuer et appliquer (a).

**Recommandation retenue** : appliquer **(a)** — réduire le warmup à 15000 **uniquement pour la
famille auto-infligée** (garder 50000 pour les fautes non contrôlables type `min_notional` Cas A).
C'est ciblé, dérivé du calcul (pas une analogie), et supprime le risque résiduel identifié sans
changer le reward ni les frais. Puis surveillance empirique 0–15k au prochain run comme filet.

> Le warmup ne doit protéger l'exploration QUE des fautes non contrôlables. Le Cas B
> (auto-infligé, sur-exposition = le chemin de collapse prouvé) ne mérite pas 50k steps de
> quasi-impunité.
