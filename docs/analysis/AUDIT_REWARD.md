# Audit du reward — cause racine de la dérive (SL saturation 6%→47%)

Source : `multi_asset_chunked_env.py::_calculate_reward` (ligne 6273) + `config.yaml`.

## Composition réelle du reward (vérifiée dans le code)

```
raw_reward = pnl_base_reward          # = pnl_pct × 0.5  (~0.65/step observé)
           + promotion_bonus          # ⚠️ one-shot au franchissement de palier
           + demotion_penalty         # symétrique (négatif)
           + closure_bonus            # +0.5 agent_close profitable, -0.2 maxduration
           + drawdown_penalty         # -50 × dd² (quadratique)
           + symmetry_penalty         # anti-triche SL/TP, LATENT/step, cap 0.15
           + action_entropy_penalty   # anti switch-spam
           + future_contrib           # MFE/MAE (cap ±0.60)
final_reward = sign × log1p(|raw_reward|)   # compression symlog DÉJÀ présente
```

## Ce qui est DÉJÀ sain (invalide certaines hypothèses)
- ✅ `survival_bonus`, `patience_bonus`, `duration_bonus` = **déjà supprimés** (§A4/A6).
  → la dérive n'est PAS due à un bonus de survie/durée non-borné.
- ✅ compression `log1p` déjà en place (log1p(3000)=8.0).
- ✅ il existe déjà une **pénalité anti-lâcheté SL** (symmetry point b : SL > 2×ATR → pénalité).
- ✅ `future_contrib` plafonné et watchdog à 6% (ne domine pas).

## 🚩 CAUSE RACINE : promotion_bonus ×10 >> pénalité anti-SL

`config.yaml` capital_tier_rewards :
| Palier | promotion_bonus | (ancien) |
|---|---|---|
| Small | **5.0** | 0.5 |
| Medium | **10.0** | 1.0 |
| High | **20.0** | 2.0 |
| Enterprise | **40.0** | 4.0 |

Le run est passé de 20.5 à ~2323 d'equity → il traverse TOUS les paliers →
**+5+10+20+40 = +75 de bonus cumulé** par épisode qui progresse.

Comparaison des échelles par step :
- PnL : ~0.65/step
- symmetry anti-lâcheté SL : λ=0.02, cap 0.15/step → en pratique ~0.01-0.05/step
- promotion : **+5 à +40 d'un coup**

→ Le bonus de promotion est **~100-750× plus fort** que la pénalité censée empêcher
le SL large. Le modèle apprend la stratégie : *élargir le SL pour survivre →
accumuler de petits gains → franchir les paliers → encaisser +75*. La qualité des
sorties (TP/SL) devient secondaire. **C'est le reward hacking observé.**

Cela explique TOUS les symptômes :
- SL saturation ↑ (6→47%) : élargir le SL = survivre = franchir les paliers.
- reward ↑ alors que TP/SL ↓ : le reward est dominé par les bonus de palier, pas la qualité.
- explained_variance ≤0 : le critic doit prédire des sauts +75 sporadiques (très haute
  variance, non-stationnaire) → il n'y arrive pas.

## Corrections retenues (validées avec l'utilisateur)

1. **SL borné STRICTEMENT** (≠ TP qui reste libre). Le SL a déjà une borne dure
   (_BOUNDS) ; on AJOUTE une pénalité anti-saturation SL **croissante log** au-delà
   d'un seuil (ex. ratio de saturation > 0.80), sévère mais plafonnée (log, pas exp).
2. **promotion_bonus one-shot proportionnel au palier** (déjà one-shot) mais
   **ramener l'échelle** pour qu'il ne domine plus le PnL, et le rendre **décroissant
   log** avec le nombre de promotions (pas exponentiel doublant).
3. **Pénalité comportement frénétique** : plus l'erreur (ex. SL saturé / switch-spam)
   se répète, plus la pénalité est sévère que la précédente, mais **croissance
   logarithmique avec un seuil** (pas exponentielle).
4. **Reward relatif au marché** : `pnl_relative = agent_return − buy_hold_return`
   pour qu'un +10% en bullrun à +20% soit jugé médiocre.
5. **Compression** : log1p déjà là ; envisager tanh(raw/scale) si l'échelle dérive encore.
6. **Hyperparams critic** (run v2) : n_epochs 20→10, lr 3e-4→2e-4, vf_coef 0.5→0.7,
   max_grad_norm 0.5→0.3, gae_lambda 0.95→0.97.

## Principe directeur (utilisateur)
- Le **SL doit être borné obligatoirement** (contrairement au TP, libre).
- Les comportements frénétiques répétés doivent être **identifiés et punis de plus en
  plus sévèrement**, mais avec un **seuil** : croissance **logarithmique, pas exponentielle**.
- Le bonus de promotion est donné **une seule fois**, **proportionnel au palier franchi**.
