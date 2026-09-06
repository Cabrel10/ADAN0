# RAPPORT 5 — LEARNING_RADAR_V2_SPEC (radar LIVE)

**Objet** : spécifier un radar d'apprentissage **en ligne** (calculé PENDANT
l'entraînement), par opposition au radar V31 qui n'existe que dans les scripts
forensics *a posteriori* (RAPPORT 1 §D). Il doit détecter l'absorption AVANT
qu'elle ne devienne irréversible (avant `adv=NaN`).

**Base factuelle** : les 5 niveaux et leurs formules validées sont ceux du radar
post-mortem V31 (`RADAR.md`, `ABSORPTION_QUANTIFIED.md`, `L2_DEEP.md`). La source
de données live existe déjà : `RewardCollector.log_step` (reward_collector.py
40-80) logue **50+ métriques par step**, dont `action_raw`, `ev_norm`,
`profit_factor`, `win_rate`, `sharpe_ratio`, `invalid_trade_attempts`,
`action_buy_score/sell_score/hold_score`. Le radar V2 CONSOMME ces champs — il
ne réinvente rien.

---

## 1. Les 5 niveaux — signaux, fréquence, granularité

| Niveau | Ce qu'il mesure | Signal(s) live | Source champ | Fréquence | Score V31 (réf) |
|--------|-----------------|----------------|--------------|-----------|-----------------|
| **L1 conséquences** | le flux PnL/fees/hold atteint-il l'agent | PnL réalisé, fees, hold_dur par close | `realized_pnl`, `total_commission`, `trade_duration_seconds` | par close | 15.8 |
| **L2 erreurs** | réaction locale + persistance | `invalid_trade_attempts`, transition action_t vs t-1, P(rép après rejet) | `invalid_trade_attempts`, `action_taken`, `action_raw` | par step (fenêtre glissante 500) | 40.9 |
| **L3 environnement** | adaptation vol→fréquence/durée | corr(vol, freq_trades) EN PRÉ-COLLAPSE | `volatility`, `trades_count_*` | par 1k steps | 83.1 (artefact) |
| **L4 cohérence/diversité** | **DÉTECTION D'ABSORPTION** | `P(BUY),P(HOLD),P(SELL)/échantillon`, `nB/nH/nS buffer`, `%advBUY_nan`, `a0_mean/std` | `action_raw` (μ,σ), rollout buffer | **par update PPO** | **2.1** |
| **L5 performance** | WR/PF/Sharpe trend | `win_rate`, `profit_factor`, `sharpe_ratio` | idem collector | par 1k steps | 33.3 |

---

## 2. L4 — le niveau CRITIQUE (détection d'absorption live)

L4 est le seul niveau qui aurait pu STOPPER V31 à temps (collapse durable à
`upd=368` sur 865 updates ; il restait 57 % du run à sauver).

**Signaux L4 (par update PPO)** :

```
mu_mean   = mean(action_raw[:,0] avant tanh)        # dérive du signe
sigma_mean= mean(sigma tête policy)                 # explosion gSDE ?
P_BUY  = 1 - Phi((atanh(thr) - mu)/sigma)           # ABSORPTION_QUANTIFIED
P_SELL = Phi((-atanh(sthr) - mu)/sigma)
P_HOLD = 1 - P_BUY - P_SELL
nB,nH,nS = comptes d'actions routées dans le buffer PPO
adv_nan_frac = fraction d'advantages NaN pour BUY/HOLD
share_dominant = max(nB,nH,nS)/(nB+nH+nS)
```

**Seuil de danger (dérivé de ABSORPTION_QUANTIFIED)** :

```
s4_danger  <=>  min(P_BUY, P_HOLD, P_SELL) < 1 / N_batch
absorption <=>  s4_danger persistant sur K updates consécutifs (K=3 proposé)
```

Réf. chiffrée V31 : `share_SELL=1.0`, `a0_mean=-8.065`, `advBUY_nan=100 %` dès
`upd=368` — tous détectables par ces signaux ~360 updates avant la fin.

---

## 3. Différences radar POST-MORTEM (V31) vs radar LIVE (V2)

| Aspect | Post-mortem (V31) | Live (V2) |
|--------|-------------------|-----------|
| Moment | après le run, sur artefacts figés | pendant, par update/step |
| L3 corr(vol,freq) | 0.831 sur run complet = **artefact du collapse** (REFUTÉ) | calculée **fenêtre glissante PRÉ-collapse** (0.408 réel) → jamais contaminée |
| L4 | reconstruit depuis logs | lu direct du rollout buffer PPO (nB/nH/nS/adv) |
| Action possible | aucune (déjà fini) | **early-stop / alerte / hook RAL** |
| Coût | nul (offline) | doit être O(1) par update (pas de I/O lourd) |

**Piège L3 documenté** : ne JAMAIS calculer une corrélation d'adaptation sur une
fenêtre qui inclut le collapse — elle devient un artefact (leçon RADAR.md :
0.831 global vs 0.408 pré-collapse). Le radar live doit fenêtrer.

---

## 4. Granularité & fréquence (contrat de performance)

| Niveau | Fréquence de calcul | Fenêtre | Justification |
|--------|---------------------|---------|---------------|
| L1 | par close | — | événement rare |
| L2 | par step, agrégé /500 steps | 500 | +42pts mesuré sur n=320 → fenêtre ≥300 |
| L3 | par 1000 steps | glissante, **exclut collapse** | éviter l'artefact 0.831 |
| **L4** | **par update PPO** | 1 update + mémoire K=3 | fenêtre de réaction avant NaN |
| L5 | par 1000 steps | glissante | trend WR/PF/Sharpe |

L4 par update est non-négociable : c'est la seule granularité assez fine pour
agir avant `adv=NaN` (irréversible).

---

## 5. Vecteur de santé S = (s1..s5) exposé par le radar

Chaque niveau produit un score normalisé [0,1] (0=danger, 1=sain) :

```
s1 = f(|PnL réalisé exploité| / |PnL potentiel|)          # L1
s2 = f(taux de correction post-erreur, 1 - persistance)   # L2
s3 = f(corr(vol,freq) pré-collapse fenêtrée)              # L3
s4 = min(P_BUY,P_HOLD,P_SELL) * N_batch, clip [0,1]       # L4  <-- clé anti-absorption
s5 = f(PF, WR, Sharpe trend)                              # L5
```

C'est EXACTEMENT le vecteur que le futur `RadarRewardAdapter` (RAPPORT 6/RAL)
consommera. **Contrainte de conception** (issue RAPPORT 3/4) : le RAL ne doit
agir sur les coefficients de reward QUE tant que `s4` est sain ; si `s4` chute,
la réponse correcte n'est PAS de moduler le reward (inefficace, symlog + μ) mais
de **contraindre μ/σ dans la loss actor** (ancre L2 / clamp σ) — la seule force
qui restaure la diversité d'échantillonnage (RAPPORT 4 §7).

---

## 6. Points d'instrumentation dans le code (pour l'implémentation)

| Signal | Point d'accès | Fichier:ligne |
|--------|---------------|---------------|
| `action_raw`, scores BUY/SELL/HOLD | déjà loggé | reward_collector.py 62,74 |
| μ, σ tête policy | tête actor | feature_extractors.py (DiagGaussian) |
| nB/nH/nS, adv | rollout buffer | callback PPO (SB3 `on_rollout_end`) |
| ev_norm, PF, WR | déjà loggé | reward_collector.py 66, 40-80 |
| frontière épisode (hook) | `reset()` | env 2646 |
| injection reward | `_calculate_reward` | env 6924 / final 7456 |

---

## 7. Synthèse

1. Le radar V2 ne crée pas de nouvelles mesures : il **branche en ligne** des
   champs déjà produits par `RewardCollector` + le rollout buffer PPO.
2. **L4 (diversité/absorption) est le cœur** : `s4 = min(P)·N_batch`, calculé
   par update, avec early-stop/alerte si `s4<θ` sur K updates.
3. Le radar V2 est la **précondition** du RAL : sans `s4` live, le RAL est
   aveugle et risque de reproduire l'échec de l'ancre L2 (moduler un reward qui
   n'atteint pas μ).
4. Le fenêtrage pré-collapse de L3 est obligatoire (leçon artefact 0.831).
