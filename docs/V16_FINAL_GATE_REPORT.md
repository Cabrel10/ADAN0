# RAPPORT V16 FINAL GATE — ADAN0 v2 PPO (BTC/USDT SPOT)

Date d'analyse : 2026-07-09
Run analysé : `v16_500k` (Mark-to-Market reward + L2 anchor lambda=0.05, aux_loss_coef=0.0)
Branche : `feat/diagnostic-v4` @ `c4cd243`
Analyste : campagne d'expérimentation scientifique (protocole "V16 Final Gate")

---

## 0. TL;DR (verdict en une phrase)

**V16 a VALIDÉ son hypothèse** (le reward Mark-to-Market supprime l'asymétrie anti-SELL : la
politique reste saine et équilibrée ~50/50 jusqu'à ~320k steps, un record vs le collapse
historique à 15k). **MAIS** le run a révélé la couche suivante du problème : le **Critic est
incapable (explained_variance ≈ 0)**, ce qui produit des avantages bruités, fait exploser le
KL (0.67) et saturer le clip (0.94), et finit par **déborder l'ancre L2 après ~320-330k** →
réapparition du collapse directionnel (a0_mean 0.8, pct_buy 0.69) en fin de run.

**Candidat de production = checkpoint 320k** (dernier point sain), PAS le 500k.
**Prochaine hypothèse V17 = qualité du Critic** (une seule variable à changer).

---

## PHASE 1 — VÉRIFICATION DE SANTÉ

### 1.A — Environnement : ✅ SAIN
| Point | Résultat |
|---|---|
| Observations finies | ✅ (bornes Box(-inf,inf) = déclaration d'espace, pas de vrais Inf en donnée) |
| NaN dans reward/gradient | ✅ AUCUN. Les 25 occurrences "nan" sont : (a) `[CAUSAL_SHIFT] 1 leading NaN row dropped` = protection anti-fuite temporelle FONCTIONNELLE ; (b) `adv_HOLD=nan` quand nH=0 = mean nan-safe d'ensemble vide, sans effet |
| Reward MTM cohérent | ✅ `[V16_MTM] ON`, `Portfolio value != realized` (positions latentes valorisées) |
| Fuite de futur | ✅ shift(1) causal appliqué sur 1h/4h/5m |
| Traceback / desync / leak | ✅ AUCUN |
| illegal_ratio | ⚠️ 0.192 → 0.391 (élevé, à surveiller — sandbox) |

### 1.B — PPO : ❌ DÉGRADÉ (cause racine)
| Métrique | Début | Fin | Cible | Verdict |
|---|---|---|---|---|
| **explained_variance** | ~0 | **-0.18 / +0.19** | > 0.3 | ❌ **Critic aveugle** |
| **approx_kl** | 0.18 | **0.67** | ~0.035 | ❌ explose |
| **clip_fraction** | 0.62 | **0.94** | < 0.3 | ❌ saturé |
| **value_loss** | 0.018 | 0.060 | ↓ | ⚠️ remonte |
| **entropy** | -0.581 | -0.571 | active | ⚠️ quasi gelée |

### 1.C — Politique : ✅ SAINE jusqu'à 320k, ❌ COLLAPSE ensuite
Courbe `a0_mean / a0_std / pct_buy` :
```
step      a0_mean   a0_std   pct_buy    état
  1000    -0.068     0.76     0.467     ✅
 81000    +0.048     0.82     0.540     ✅
161000    +0.065     0.79     0.527     ✅
241000    +0.133     0.88     0.525     ✅
300000    +0.046     0.90     0.482     ✅
320000    +0.072     0.865    0.511     ✅ ← DERNIER SAIN (a0_std<1.0)
─────────────────────────────────────────────
325000    +0.023     1.006    0.513     ⚠️ a0_std franchit 1.0
335000    +0.324     1.318    0.580     ❌
365000    +0.827     1.622    0.712     ❌ collapse
462000    +0.795     1.72     0.689     ❌ (état actuel du run)
```

---

## CHAÎNE CAUSALE ÉTABLIE

```
Critic incapable (EV ≈ 0)
        ↓ avantages A = R + γV(s') − V(s) = bruit
Politique optimise sur un signal aléatoire
        ↓ KL explose (0.67) + clip sature (0.94)
a0_std explose (0.87 → 1.9), a0_mean dérive (0.07 → 0.8)
        ↓ après ~320-330k
Ancre L2 (lambda=0.05) débordée
        ↓
Collapse directionnel (pct_buy 0.69)
```

Preuve corrélée par les probes Critic :
- upd=314 (zone saine) : adv_BUY=-0.14, adv_SELL=+0.22 → **SELL avantageux (succès V16)**
- upd=742 (zone collapse) : adv_BUY=+0.15, adv_SELL=-0.43 → asymétrie anti-SELL **de retour**,
  simultanément à a0_std=1.67.

---

## PHASE 2 — CHOIX DU MEILLEUR MODÈLE

**On ne garde PAS le 500k.** Checkpoints sains disponibles (sauvés tous les 10k) :
- **320k = CANDIDAT PRINCIPAL** (a0_mean=+0.072, a0_std=0.865, pct_buy=0.511)
- 310k (a0_mean=+0.001, a0_std=0.921, pct_buy=0.466) = candidat de repli le plus neutre
- 300k (a0_mean=+0.046, a0_std=0.904, pct_buy=0.482)

Fichiers : `checkpoints/ppo_adan0_sandbox_checkpoint_{300000,310000,320000}_steps.zip`

---

## PHASE 3-4-5 — PLAN DE VALIDATION TRADING (à exécuter APRÈS fin du run)

> Ces phases nécessitent le CPU libre (le run 500k sature actuellement les workers).
> Ne PAS lancer en parallèle du training (fausserait débit + mesures).

Outillage disponible dans le repo (réutilisable, aucun code neuf requis) :
- `scripts/backtest/deterministic_backtest.py` → PHASE 4 (stochastic vs deterministic)
- `scripts/backtest/backtest_fixed_capital.py` → PHASE 3 (backtest à capital fixe 20.5)
- `scripts/backtest/forensic_trades.py` → Autopsie Niveau 4 (pourquoi BUY/SELL/HOLD)
- `scripts/backtest/offline_reward_replay.py` → rejouer le reward hors-ligne
- `src/adan_trading_bot/evaluation/decision_quality_analyzer.py` → qualité décision

Métriques à calculer par checkpoint {300k,310k,320k} : Profit net, Profit Factor, Win Rate,
Expectancy, Sharpe, Sortino, Calmar, Max Drawdown, Ulcer Index, MAR, Avg Trade, Trade
duration, Exposure, Recovery Factor.

Baselines obligatoires : Always Long, Always Flat, Always Sell, Random, Momentum,
Mean Reversion, Buy & Hold.

Critère GO production : PF > 1 ET Sharpe > 0 ET le PPO bat au moins Buy&Hold + Random,
en mode DÉTERMINISTE (robustesse hors bruit d'exploration).

---

## PHASE 6 — NOTE QUALITÉ

| Dimension | Note | Justification |
|---|---|---|
| Environment | 98/100 | Sain, anti-fuite OK, MTM cohérent ; -2 illegal_ratio élevé |
| Reward (V16) | 92/100 | Hypothèse validée (SELL avantageux zone saine) ; robuste 320k steps |
| Policy | 70/100 | Saine 0→320k puis collapse — variance non maîtrisée fin de run |
| Critic | 25/100 | **explained_variance ≈ 0 = cause racine.** Aveugle. |
| Robustesse | 45/100 | Tient 320k (record) mais dégénère ; KL/clip explosent |
| Trading edge | N/A | Non mesuré (backtest à faire post-run) |
| **TOTAL (hors trading)** | **~66/100** | Progrès majeur vs runs précédents, bloqué par le Critic |

---

## SI LE MODÈLE EST "MORT" AU BACKTEST → AUTOPSIE → V17

Le diagnostic pointe déjà **une seule hypothèse** à modifier (démarche expérimentale
reproductible) :

### HYPOTHÈSE V17 : réparer le Critic (explained_variance ≈ 0)
Options (à départager, UNE SEULE à la fois) :
1. **vf_coef ↑** (le Critic n'apprend pas assez vite) et/ou lr Critic séparé.
2. **Normalisation des returns / value targets** (VecNormalize sur reward, ou reward scaling
   MTM : le delta-equity_pct peut être trop petit → gradient value ténu).
3. **n_epochs / batch** : KL=0.67 + clip=0.94 suggèrent des updates trop agressifs →
   baisser lr ou target_kl, ou augmenter n_steps pour des avantages moins bruités.
4. **Architecture Critic** (tête value trop faible face au CNN multi-échelle).
5. **Renforcer/annealer l'ancre** au-delà de 320k (palliatif, ne règle pas la cause).

Recommandation : commencer par (2)+(3) car EV≈0 + KL explosif = signature d'un signal de
value mal calibré ET d'updates trop grands. Garder MTM (validé) et l'ancre (validée).

---

## PHASE 7 — REPRODUCTIBILITÉ (voir tag git v16-final)

Archivé : code exact (c4cd243), hyperparamètres sandbox (n_steps=512, batch=64,
target_kl≈0.035, use_sde), env-vars (ADAN_MTM_REWARD=1, ADAN_L2_ANCHOR_LAMBDA=0.05,
ADAN_AUX_LOSS_COEF=0.0), logs, diagnostics CSV, checkpoints retenus {300k,310k,320k}, ce rapport.
