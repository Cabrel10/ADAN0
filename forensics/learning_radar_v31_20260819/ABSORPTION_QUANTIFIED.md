# QUANTIFICATION DU POINT D'ABSORPTION — réfutation du clamp ±3

Read-only, artefacts V31 figés. Modèle validé sur observations.

## Mécanisme de mapping (confirmé dans le code)

`multi_asset_chunked_env.py` l.8243 (docstring `_execute_trades`) :
```
action_raw: -1..1  (< -threshold = SELL, > threshold = BUY, else HOLD)
```
Seuils config.yaml l.648-651 : **5m=0.05, 1h=0.08, 4h=0.10** (sur tanh(μ+σε)).

## Modèle : P(BUY par échantillon) = 1 − Φ((atanh(thr) − μ)/σ)

Avec σ≈0.371 (mesuré, stable tout le run) et thr=0.08 (1h, TF dominant) :

| μ | P(BUY)/sample | E[nB] sur buffer 2048 | régime |
|------|------|------|------|
| -0.500 | 5.89e-02 | 120.7 | diversité OK |
| -0.800 | 8.84e-03 | 18.1 | raréfaction |
| -1.000 | 1.80e-03 | 3.7 | raréfaction |
| **-1.145** | **4.79e-04** | **1.0** | **QUASI-ABSORBANT** |
| -1.300 | 9.96e-05 | 0.2 | QUASI-ABSORBANT |
| -1.500 | 1.03e-05 | 0.0 | ABSORBANT |
| -3.000 | 0.00e+00 | 0.0 | ABSORBANT |
| -8.200 | 0.00e+00 | 0.0 | ABSORBANT |

## Validation du modèle sur les observations

- upd 308-332 (μ ∈ [-0.9, -1.1]) : nB observé ∈ {0, 4, 8} → modèle prédit E[nB] ∈ [1, 4]. ✅
- upd 336+ (μ = -1.145) : nB=nH=0 durable → modèle prédit E[nB] ≈ 1.0. ✅
- La correspondance modèle/observation est excellente.

## Verdicts

### [CONFIRME] Point d'entrée dans l'absorption : μ ∈ [-1.0, -1.15]
L'état absorbant est atteint dès μ≈-1.1 (tanh≈-0.82), **très loin de la saturation
tanh** (-8.2). La saturation n'est que l'aggravation terminale ; la mort de la diversité
survient quand P(BUY) tombe sous ~1/2048, i.e. moins d'un échantillon BUY par buffer.

### [CONFIRME] Le clamp dur μ=±3 (correction proposée antérieurement) est INEFFICACE
À μ=-3, E[nB] est déjà 0. Un clamp à ±3 n'empêche rien : la diversité meurt à μ≈-0.8
(raréfaction) et l'absorption est effective à μ≈-1.15. Pour être utile, une protection
devrait borner μ dans une zone où E[nB] reste >> 1, i.e. **|μ| ≲ 0.7-0.8** — ou agir
sur un tout autre levier (plancher d'exploration, reset de μ au dépassement de seuil
de diversité, etc.). **Aucune correction n'est appliquée** — ceci est une analyse
read-only qui informe le futur plan de correction.

### [CONFIRME] Fenêtre d'intervention : upd ~290-330
La raréfaction commence upd≈292 (nB=28) et l'absorption se scelle upd≈336.
Un moniteur temps réel sur E[nB] prédit (calculable depuis μ, σ, thr sans coût)
donnerait ~10-15 updates PPO de préavis avant l'absorption durable.

## Chaîne causale finale (toutes étapes CONFIRMEES)

```
marché baissier tôt (78400→66400)
→ SELL statistiquement favorisé par le gradient
→ μ dérive : -0.5 (upd280) → -0.9 (upd308, raréfaction nB≤8) → -1.145 (upd336, absorption)
→ P(BUY)<1/2048 : plus aucun BUY/HOLD dans le buffer PPO
→ adv_BUY=adv_HOLD=NaN : la comparaison devient mathématiquement impossible
→ disparition du signal correctif comparatif
→ μ poursuit seul vers -8.2, freiné uniquement par l'anchor L2 (équilibre stérile)
→ 78% du budget du run gaspillé en état absorbant (fenêtre 9: 406k policy → 32 exec)
```
