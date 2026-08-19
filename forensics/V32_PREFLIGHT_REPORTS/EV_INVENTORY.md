# RAPPORT 1 — EV_INVENTORY

**Objet** : inventaire FACTUEL de toutes les « EV » (Expected Value / signaux de
valeur attendue) présentes dans le code ADAN0. Aucune supposition — chaque entrée
est référencée par fichier:ligne, lue au commit courant de `genspark_ai_developer`.

**Méthode** : `grep -rn "ev_norm|expected_value|resolve_ev_fee_gate|p_hmm|advantage|profit_factor|winrate|sharpe|expectancy"` sur `src/adan_trading_bot/`, puis lecture ciblée.

> ⚠️ Distinction essentielle établie par ce rapport : dans ADAN, « EV » recouvre
> DEUX familles différentes qu'il ne faut pas confondre —
> (A) l'**EV décisionnelle** qui influence le contrôle (gate d'entrée), et
> (B) les **EV télémétriques / de reward** qui n'entrent PAS dans le gradient de
> contrôle mais servent au shaping ou au logging.

---

## A. EV décisionnelle (influence le CONTRÔLE / la sélection d'action)

| Nom | Fichier | Ligne | Description | Utilisé où |
|-----|---------|-------|-------------|-----------|
| `resolve_ev_fee_gate(p_hmm, p_min_required, disabled)` | `environment/action_routing.py` | 109-125 | Gate EV basé frais : bloque un BUY si `p_hmm <= p_min_required`, SAUF si `disabled` (mode advisory). Retourne `(blocked, reason)` avec `reason ∈ {accepted, disabled_advisory, negative_ev_fee_gate}`. | Importé env l.32 ; `p_hmm` calculé env l.8732 ; compteur `fee_gate` env l.707 |
| `p_hmm` (confiance régime HMM) | `environment/multi_asset_chunked_env.py` | 8723-8732 | Probabilité de régime issue du DBE/HMM, lecture SEULE pour l'EV gate. Défaut 0.5. | Alimente `resolve_ev_fee_gate` |
| `resolve_agent_close_gate(exit_authority, budget_blocked, below_break_even)` | `environment/action_routing.py` | 128-148 | Gate de clôture (pas une EV numérique mais une décision de valeur : autorise/bloque une sortie policy selon budget + barrière break-even). | Importé env l.31 |

**Fait clé** : `resolve_ev_fee_gate` est la SEULE EV qui peut modifier une action
(bloquer un BUY). Le commit `28dad74 "restore actionable EV-gated HMM pipeline"`
confirme que ce pipeline est actif. C'est le point de contact EV → contrôle.

---

## B. EV de reward (shaping — entre dans `raw_reward`, donc dans le gradient PPO via reward)

| Nom | Fichier | Ligne | Description | Utilisé où |
|-----|---------|-------|-------------|-----------|
| `ev_norm` (bonus EV) | `environment/reward_calculator.py` | 182-184, 270-273 | Bonus de reward `beta * clip(ev_norm, -1, 1)`, `beta=0.1` (réduit de 1.0, commit `900bf7c`). Si `ev_norm==0` → proxy `+0.5 si pnl>0 / -0.5 si pnl<0`. | Dans `RewardCalculator.calculate()` |
| `_beta` (multiplicateur EV bonus) | `environment/reward_calculator.py` | 126 | Constante = 0.1 (« REDUCED from 1.0 to prevent hacking »). | idem |

**Fait clé** : `RewardCalculator.calculate()` (l.157) est le chemin reward
« True Quant standalone ». MAIS l'autopsie V30 (RAPPORT_AUTOPSIE_V30.md, addendum)
a prouvé par régression sur 40 000 échantillons réels que le reward EFFECTIF du
run 500k est produit par `MultiAssetChunkedEnv._calculate_reward()` (env l.6924),
et que `RewardCalculator` n'est PAS le driver du gradient V30/V31. Donc `ev_norm`
via `RewardCalculator` est, dans les runs réels, en grande partie **inactif**.

---

## C. EV télémétrique (loggée, N'ENTRE PAS dans le gradient)

| Nom | Fichier | Ligne | Description | Utilisé où |
|-----|---------|-------|-------------|-----------|
| `ev_norm` (télémétrie) | `environment/multi_asset_chunked_env.py` | 4416, 4505 | Lue depuis `reward_breakdown.get('ev_norm', 0.0)` puis passée au `RewardCollector.log_step(ev_norm=...)`. | JSONL `logs/rewards/` |
| `ev_norm` (collector) | `utils/reward_collector.py` | 66, 191 | Champ `excellence.ev_norm` du JSONL. Pur logging. | Analyse post-mortem |
| `profit_factor`, `win_rate`, `sharpe_ratio`, `sortino_ratio`, `calmar_ratio` | `utils/reward_collector.py` | 53, 145-147 | Métriques de risque/perf loggées par step. Pur logging. | JSONL |

---

## D. « EV » au sens performance attendue par action×état (mesurées A POSTERIORI par le radar)

Ces EV n'existent PAS comme variables dans le code d'entraînement : elles sont
CALCULÉES par les scripts forensics à partir des logs. Elles sont l'inventaire
cible que le futur radar live devra produire (cf. RAPPORT 5).

| Nom (radar) | Source | Description | Statut |
|-------------|--------|-------------|--------|
| EV par état×intention (FLAT+SELL, OPEN+BUY, …) | `forensics/v30_autopsy_20260817/timeline/causal_map_state_intent.txt` | `final_reward` moyen par (état,action) — a prouvé l'asymétrie V28 | Post-mortem uniquement |
| P(BUY)/échantillon = 1−Φ((atanh(thr)−μ)/σ) | `forensics/learning_radar_v31_20260819/ABSORPTION_QUANTIFIED.md` | Probabilité d'échantillonner BUY selon (μ,σ,thr) | Post-mortem (modèle validé) |
| PF, WR par fenêtre décile | `forensics/learning_radar_v31_20260819/RADAR.md` | Profit factor / winrate par tranche temporelle | Post-mortem |
| Matrices de transition d'action | `forensics/learning_radar_v31_20260819/L2_DEEP.md` | P(action_t | action_{t−1}) | Post-mortem |

---

## Synthèse

1. **Une seule EV agit sur le contrôle** : `resolve_ev_fee_gate` (entrée BUY).
2. **L'EV de reward (`ev_norm`, β=0.1)** existe mais transite par `RewardCalculator`
   qui n'est PAS le chemin de reward effectif des runs 500k (prouvé V30).
3. **Le vrai reward** est composé dans `env._calculate_reward` (l.6924-7456) —
   voir RAPPORT 3 (PENALTY_AUDIT) et RAPPORT 6 (PIPELINE).
4. **Les EV par état×action** (celles qui expliquent le collapse) n'existent que
   dans les scripts forensics post-mortem → à porter EN LIGNE (RAPPORT 5).

**Conclusion factuelle** : le système « EV » actuel est fragmenté et en grande
partie télémétrique. Le RAL de la spec initiale supposait des EV live par point
de radar qui **n'existent pas encore dans le code d'entraînement**. Il faut
d'abord les construire (RAPPORT 5) avant toute modulation de reward.
