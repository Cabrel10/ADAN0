# DIAGNOSTIC CAUSAL V35 — fondé sur données, pas hypothèses

Date : 2026-08-23
Source : `logs/rewards/worker_0_rewards_20260822_162820.jsonl` (1.87 GB, run V35 500k complet)
Méthode : audit streaming de 500,224 steps / 3,016 clôtures, réconciliation exacte du raw_reward.
Checkpoint figé de référence : `checkpoints/ppo_adan0_v35_500k.zip`

---

## 0. Statut V35
- Run **TERMINÉ proprement** : 500,224 steps, `run_open_positions=0`, aucun process actif.
- Résultat financier : PnL cumulé ≈ **-187.9 $** sur le run (≈ -3/épisode), break-even terminal (reset épisode).
- Trades : 3,027 ; SL_HIT ≈ 496 vs TP_HIT ≈ 80 → ratio SL/TP ≈ **6:1** ; winrate sorties ≈ 14%.

---

## 1. RÉCONCILIATION DU REWARD (vérifiée, diff=0.0000)

`raw_reward` (ce qui atteint PPO) = somme EXACTE de 12 termes.
`final_reward = sign(raw)·log1p(|raw|)` (symlog global APRÈS sommation).

**IMPORTANT — pièges télémétriques écartés :**
`capacity_reward` (±2.0/-1.5), `pos_limit_penalty`, `duration_penalty`, `frequency_daily`
sont dans le breakdown JSON `rc[]` MAIS **n'entrent PAS dans raw_reward** → n'atteignent
jamais le gradient PPO. Un premier audit naïf les avait comptés (capacity=99% de l'amplitude) :
**INFIRMÉ** après réconciliation.

---

## 2. HIÉRARCHIE RÉELLE DES TERMES PPO (mean_abs, run complet)

| Rang | Terme | Part amplitude | Signe observé | Nature |
|------|-------|---------------:|---------------|--------|
| 1 | `symmetry_penalty`  | **32.8%** | négatif PUR (max=0.000) | forme SL/TP (RR + ATR) |
| 2 | `future_contrib`    | **22.4%** | ± (oracle MFE/MAE)      | pédagogique |
| 3 | `drawdown_penalty`  | **20.5%** | négatif dominant        | risque |
| 4 | `pnl_reward`        | **13.1%** | ±                       | **SIGNAL FINANCIER** |
| 5 | `closure_bonus`     |  9.9% | ±                       | comportemental |
| 6 | `saturation_penalty`|  1.1% | négatif                 | anti-spam |
| 7-12 | latent_pnl, behavior, anchor, entropy, promo, demote | <0.1% | — | quasi inactifs |

**Pénalités quasi-exclusivement négatives : symmetry + drawdown + saturation = 54.4%.**

---

## 3. CONCLUSIONS (classées)

### CONFIRMÉ
1. **Le signal financier est structurellement minoritaire.** `pnl_reward` = 13% du reward.
   Le critic doit surtout prédire des pénalités de forme/risque, pas la rentabilité.
2. **Le trio de pénalités de forme domine (54%)** et est presque toujours négatif.
   `symmetry_penalty` seul (32.8%) est **2.5× plus gros que pnl_reward** et **jamais positif** :
   c'est une taxe permanente, pas un signal directionnel.
3. **Même les trades GAGNANTS reçoivent un reward moyen NÉGATIF** (-0.137).
   L'agent n'a jamais reçu de signal net positif pour avoir gagné de l'argent.
   Déséquilibre clôtures : 2,546 pertes / 470 gains = **5.4:1**.
4. **SL saturent la borne min** (`sl_raw=-1.000` → SL=0.30%) → stops immédiats →
   SL_HIT massif. Mécanisme concret du ratio SL/TP 6:1.
5. **Contradiction de design** : run lancé `free_sltp=1` (SL/TP libres) MAIS
   `symmetry_penalty` (anti-asymétrie RR, cible RR=1.5 ±0.5) reste actif et pénalise
   en continu tout RR>2 — or les trades ont RR≈2. On libère puis on taxe l'usage.

### INFIRMÉ
6. **"Future Arena pousse vers les pertes / hindsight non apparié dominant"** : INFIRMÉ.
   `corr(future_contrib, realized_pnl) = +0.41` (aligné, pas anti-corrélé).
   Incohérences oracle (perte récompensée) = 2.3% seulement.
   FA reste un problème de MAGNITUDE (22%, 2e terme, dilue pnl) mais pas de DIRECTION.
7. **"Reward cassé / décorrélé du PnL"** : INFIRMÉ.
   `corr(reward.total, realized_pnl) = +0.81`, `corr(raw_pre_symlog, realized_pnl) = +0.85`.
   Le reward SUIT le PnL — il est juste noyé sous des termes de forme plus gros.
8. **"Fuite de lookahead dans la normalisation"** : INFIRMÉ.
   `LOOKAHEAD RISK` = **0 occurrence** sur tout le run V35 (fix "fit scaler 70%" a tenu).
9. **"capacity_reward domine le reward"** : INFIRMÉ (télémétrique, hors PPO).

### CONFIRMÉ (architecture)
10. **La tête auxiliaire N'est PAS un professeur financier propre.**
    `feature_extractors.py:1515` : `target = batch.returns` (retour GAE du reward composite),
    puis symlog. Elle apprend le MÊME cocktail que le critic → inutilisable telle quelle
    comme branche "vérité financière" en V36-C sans recâblage sur une cible indépendante.
11. **L'action_space est 5D/actif** : `[Action, Size, Timeframe, StopLoss, TakeProfit]`
    (`multi_asset_chunked_env.py:2266`). Le RÉSEAU pilote directement SL/TP (dims 4-5),
    ce n'est pas un moteur DBE externe. D'où l'importance directe de symmetry_penalty.

---

## 4. LE VRAI PROBLÈME (une phrase)

> V35 n'a pas un réseau trop petit ni un reward cassé. Il a un **contrat d'apprentissage
> déséquilibré** : le signal financier (13%) est dominé par un trio de pénalités de forme
> et de risque (54%) presque toujours négatives, de sorte que même gagner de l'argent
> rapporte un reward négatif. L'agent optimise donc la minimisation de pénalités de forme
> (SL serrés qui saturent, RR≈2 taxé), pas la rentabilité.

---

## 5. IMPLICATION POUR V36

La correction n'est PAS "réduire Future Arena à 40%" (FA n'est que le 2e terme et est aligné).
La correction de fond : **re-hiérarchiser le reward pour que le signal financier domine**,
et **résoudre la contradiction free_sltp vs symmetry_penalty**.

→ Voir `SPEC_V36.md`.
