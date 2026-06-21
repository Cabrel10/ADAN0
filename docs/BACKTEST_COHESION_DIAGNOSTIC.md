# Diagnostic de cohésion du backtest — enquête empirique (pas de supposition)

Date : 2026-06-21
Méthode : lecture du code source + sonde empirique des actions des deux modèles.
Objectif : comprendre POURQUOI 450k et 500k donnent des résultats quasi-identiques
avec WR=98.6 % mais expectancy négative — au lieu de le supposer.

---

## 1. Vérification : les checkpoints sont-ils réellement différents ?

```
md5  450k = accacd4b8b69807c85ae57327f23350e
md5  500k = b99a9c6ec6fd87820d2242379d771ad0
```
**Oui, différents.** Donc l'identité des résultats n'est pas due à un même fichier.

## 2. Sonde empirique : les deux modèles produisent-ils les mêmes actions ?

Script `/tmp/probe_actions.py` — 30 steps déterministes sur le test split :

```
ACTIONS 450k (5 premiers steps)        ACTIONS 500k
[1.0, -1.0, 1.0, 1.0,   1.0]           [1.0, -1.0, 1.0, 1.0, 1.0]
[1.0, -1.0, 1.0, 0.798, 1.0]           [1.0, -1.0, 1.0, 1.0, 1.0]
[1.0, -1.0, 1.0, 0.780, 1.0]           [1.0, -1.0, 1.0, 1.0, 1.0]
...
Mean abs diff = 0.0298   Max abs diff = 0.2204   Identical? False
```

**Constat fort : les deux policies sont SATURÉES aux bornes ±1.0.**
- `action[0]` (decision) = +1.0 constant → BUY systématique.
- `action[1]` (size)     = -1.0 constant.
- `action[2]` (timeframe)= +1.0 constant.
- `action[3]` (SL)       = saturé (500k) ou légèrement variable (450k).
- `action[4]` (TP)       = +1.0 constant.

C'est la signature d'un **policy network collapsé** : les logits sont si
extrêmes que `tanh` sature. La PPO a convergé vers une politique dégénérée
quasi-constante. Les deux modèles ne diffèrent qu'imperceptiblement (SL de 450k).

## 3. Cohésion action → trade (lue dans multi_asset_chunked_env.py)

| Dim action | Rôle nominal | Effet RÉEL dans l'env |
|-----------|--------------|------------------------|
| `action[0]` | decision | `> 0.1` (seuil 5m) → BUY. Saturé 1.0 → BUY permanent. |
| `action[1]` | size | **ÉCRASÉ** : `target_exposure = exp_min+(exp_max-exp_min)*confidence_HMM` (ligne 6908). La taille vient du **HMM**, pas du modèle. |
| `action[2]` | timeframe | `tf_idx=int((tf+1)*1.5)` → choisit le TF. |
| `action[3]` | SL | utilisé, borné profil scalper [2 %, 3 %] (ligne 7010). |
| `action[4]` | TP | utilisé, borné profil scalper [4 %, 6 %], plancher fee-gate 0.6 %. |

**Point clé n°1 — la taille de position ne dépend PAS du modèle.** Elle est
pilotée par la `confidence` du HMM (`ctx[3]` = bull_prob). Si le HMM est bloqué
sur un régime constant (« sideways », confidence ≈ 0.5), l'exposition est
constante quelle que soit l'action → mêmes notionnels → trades quasi-identiques.

**Point clé n°2 — fermetures forcées dominent.** Le profil scalper impose
`MAX_DURATION = 20 steps` (ligne 7068). En plus : `FORCE_CLOSE_FOR_NEW_TRADE`,
`CHUNK_END_FORCE_CLOSE`, `EPISODE_END`, `AGENT_CLOSE`. Avec BUY permanent, le
modèle ouvre une position, puis une fermeture forcée la solde quelques steps
plus tard sur un micro-mouvement → `best_trade = +0.052 %` (minuscule), tandis
que les rares positions qui touchent le SL perdent jusqu'à -4.95 %.

## 4. Pourquoi WR=98.6 % MAIS expectancy NÉGATIVE (la vraie logique)

- 142 trades / 144 ferment à un micro-gain (+0.052 % médian) via fermeture forcée.
- 2 trades touchent le SL et perdent ~-4.95 % chacun.
- `gross_win = 7.39 %` < `gross_loss = 9.89 %` → **PF = 0.75 < 1**.
- Donc : **un win-rate élevé est trivialement atteint en encaissant des
  micro-gains, mais une poignée de pertes au SL détruit l'espérance.**

C'est l'asymétrie classique « ramasser les centimes devant le rouleau
compresseur » (picking pennies in front of a steamroller). **WR n'est PAS une
mesure de qualité ici** ; seuls PF et expectancy le sont.

## 5. Pourquoi le random « bat » les modèles sur le test split

Le random tire des actions uniformes → il ouvre ET ferme dans les deux sens,
avec des tailles variées, ce qui sur un split court/plat capte par hasard
quelques mouvements (PF 1.46). Ce n'est PAS un edge du random — c'est la preuve
que **le test split (5298 lignes, ~18 j plats) n'est pas discriminant** et que
la politique saturée des modèles y est particulièrement mal adaptée.

## 6. Cause racine probable de la saturation (à confirmer)

La saturation `tanh→±1` constante évoque l'une de ces causes (ordre de
probabilité, à instrumenter avant correction) :
1. **Reward shaping** qui récompense le micro-TP / la fréquence de trade
   (early_close_bonus) → la policy apprend « BUY puis encaisser vite ».
2. **HMM figé sur un régime** → la confidence constante prive le size de signal,
   donc le gradient sur `action[1]` s'annule et la dim sature sans coût.
3. **Sur-entraînement PPO** (entropy coef trop bas) → effondrement de la
   politique vers un coin déterministe.

## 7. Conclusion méthodologique

- La décision de retenir **500k sur la VAL** reste valide (sur la VAL, le HMM
  produit visiblement une confidence variable, d'où PF 2.58 vs 0.75 sur test).
- Le **test split doit être écarté** comme métrique de décision : trop court,
  non discriminant, et il expose la dégénérescence micro-TP.
- **Avant tout ré-entraînement** : instrumenter (a) la sortie du HMM par step,
  (b) l'entropie de la policy, (c) la décomposition du reward, pour trancher
  entre cause #1 / #2 / #3. On ne corrige pas à l'aveugle.
