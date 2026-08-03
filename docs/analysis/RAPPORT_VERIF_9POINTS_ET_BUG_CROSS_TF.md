# RAPPORT D'EMBRANCHEMENT — Vérification 9 points + BUG CRITIQUE cross-timeframe

Date : 2026-06-27
Branche : feat/future-arena-v2
Méthode : inspection directe du code réellement exécuté (step(), _calculate_reward(),
update_market_price(), close_position()) — PAS la conception/docs.

---

## 0. ALERTE GÉNÉRALE — le run 500k est INVALIDE

Le run 500k a été **GELÉ (tué)**. L'equity 20.5 → 2323 USDT ne vaut **rien** :
le modèle n'a **pas appris à trader**, il a exploité une **incohérence
d'indexation cross-timeframe** de l'environnement (preuve §B). Tous les
checkpoints (50k/100k/150k/200k/250k) issus de ce run sont **à jeter**.

---

## A. CHECKLIST 9 POINTS (preuve par ligne de code)

| # | Question | Réponse | Preuve (fichier:ligne) |
|---|----------|---------|------------------------|
| 1 | max_agent_close (7/jour) appliqué ? | **OUI (partiel)** | `trade_frequency_controller.can_close_trade()` L112-131 = toujours `(True,None)` (NON), MAIS `multi_asset_chunked_env` L7669-7675 : `_budget_blocked=(... or _ac_today>=_ac_max_day=7)` → `discrete_action=0` (forcé HOLD). Reset quotidien L5728 (`num_timesteps//288`). Le quota EST imposé via decision_budget. |
| 2 | max_SL appliqué à l'exécution ? | **OUI** | Décodage action `env` L7451 `_BOUNDS` + `np.clip(sl_lo+..., sl_lo, sl_hi)` L7464. SL hard-borné par profil (scalper 0.012, intraday 0.020, swing 0.035, position 0.060). `portfolio_manager.calculate_final_trade_parameters` L479 re-clamp `max(min(sl,0.06),0.003)`. |
| 3 | récompense qualité entrée connectée ? | **OUI (faible)** | `reward_service.entry_quality_score` L267, agrégé dans `bd.future_contrib` L512, remonté via `_future_contrib_from_receipts` L6254-6268 → `future_contrib` dans `raw_reward` L6519. MAIS empaqueté + plafonné cap=0.60. |
| 4 | récompense qualité TP/SL connectée ? | **OUI (faible)** | idem #3 : tp_quality/sl_quality dans `bd.future_contrib`. Bridge **ACTIVE** (log : `mode=future_guided cap=0.60`). **MAIS** `mean_abs_future=0.0444` vs `mean_abs_pnl=0.7203` → terme futur **16× plus faible que le PnL** (`future_share=5.8%`). Trop faible pour guider. |
| 5 | pénalité perte latente appliquée ? | **NON** | `_calculate_reward` (L6273-6526) ne contient **AUCUN** terme `unrealized_pnl`. Seul `symmetry_penalty` est "latent" mais pénalise RR/SL-vs-ATR, PAS le PnL courant. → **À AJOUTER (demande utilisateur).** |
| 6 | pénalité fermeture précoce connectée ? | **OUI** | `close_intention_penalty` config L1564 `enabled:true min_hold:6`. Code env L7733-7749 : `_ci_pen = -lambda*dur_deficit*pnl_factor` ajouté à `_step_invalid_penalty` → reward. Mais en pratique 0 AGENT_CLOSE récents (le modèle laisse jouer TP). |
| 7 | pénalité saturation SL/TP appliquée ? | **NON** | env L7469-7472 : ACTION-SATURATION TRACKER **LOGGE seulement** `tp_pct_mean`, aucune pénalité. Aucune `sat_pen` dans le reward. → **À AJOUTER.** |
| 8 | reward utilise les vraies valeurs exécutées ? | **OUI** | `close_position` receipt L1004-1027 : `exit_price`, `entry_price`, `pnl` (net frais réels), `stop_loss_pct`/`take_profit_pct` lus depuis `position`. Pas de valeur DBE d'affichage. ⚠️ MAIS ces valeurs réelles sont calculées sur des prix CORROMPUS (bug §B). |
| 9 | variables DBE et trade réelles identiques ? | **OUI** | Le reward lit `position.stop_loss_pct`/`take_profit_pct` (valeurs réellement posées, post-clip/ATR). Le DBE ne fait qu'ajuster ±15% AVANT le clip ; la valeur finale stockée = exécutée. |

### `_step_invalid_penalty` → reward : chaîne PROUVÉE
- Toutes les pénalités (budget L7683, barrier L7697, close_intent L7743, wait L7802,
  risk_gate L7828) s'accumulent dans `self._step_invalid_penalty`.
- `_execute_trades` L8043 : `realized_pnl += self._step_invalid_penalty`.
- `step()` L3271 : `realized_pnl, ... = self._execute_trades(...)`.
- `step()` L3675 : `reward = self._calculate_reward(action, realized_pnl)`.
- `_calculate_reward` L6334-6336 : `pnl_net=realized_pnl` → `pnl_pct` → `pnl_base_reward=pnl_pct*0.5` L6435.
→ **Les pénalités SONT bien dans le reward.**

---

## B. BUG CRITIQUE — incohérence cross-timeframe (PROUVÉ)

### B.1 Symptôme observé (log 500k, 400k dernières lignes)
- TP_HIT=3430, SL_HIT=536, AGENT_CLOSE=0
- hold_steps : median=1, mean=1.59
- Exemple : `entry_price=87972 | tp_price=89731 (+2%) | high_price=115891 (+31%) | hold_steps=1`
- Échantillon 200 TP_HIT : TP choisi +1.93% ; **high bougie +23.7% en moyenne ; 100% high>+10%**

### B.2 La donnée est SAINE (ce n'est PAS un problème de données brutes)
`scripts/diagnostics/prove_cross_tf_bug.py` PREUVE 1 :
- 5m : (high-low)/close mean=0.157% p99=0.664% **max=3.75%**
- Une bougie 5m à +20/30% est **impossible**.

### B.3 Cause racine (code)
1. `self.timeframes = ["5m","1h","4h"]`.
2. `env` L7279-7281 : `tf_idx=int((tf_raw+1)*1.5)` ; `current_timeframe_for_trade=self.timeframes[tf_idx]`.
   → le TF est **décodé depuis l'action du modèle, change à CHAQUE step**.
3. Ordre dans `step()` :
   - L3106 `_get_current_prices()` (close) au TF de l'action N-1
   - L3123-3124 `_get_price_for_asset('low'/'high')` au **même TF N-1**
   - L3130 `update_market_price()` → **check TP/SL EN PREMIER**
   - L3271 `_execute_trades()` → décode l'action N → **change le TF** (L7281), pose le BUY avec `entry_price` au TF N.
4. `_get_price_for_asset` L5106-5109 : index = `base_step/(tf_min/5)` clampé `min(.., len-1)`.
   → 4h (1685 lignes) idx=354 vs 5m (18544 lignes) idx=17000 : **moments calendaires différents**.
5. Résultat : `position.entry_price` (figé au TF du BUY) comparé à `high_price`/`low_price`
   relus à un AUTRE TF/index → TP/SL déclenchés sur des **mouvements fictifs**.

### B.4 Preuve empirique chiffrée (`prove_cross_tf_bug.py`)
```
step_in_chunk=17000 | close@4h(idx354)=27256 | high@5m(idx17000)=74507 | div=+173% | TP+2% OUI
divergence max |close4h vs high5m| = 493.07%
TP +2% touché trivialement : 11/11 cas (100%)
```
**VERDICT : bug cross-timeframe PROUVÉ (div 493% >> 5%).**
Affecte TP **et** SL (votre point #3 : SL_HIT à -18.9% impossible avec SL capé 1.2%).

---

## C. PLAN DE CORRECTION (ordre)

### FIX 1 — cross-timeframe (CRITIQUE, prérequis à tout)
- Le check TP/SL DOIT lire high/low au **TF d'entrée de la position** (`position.timeframe`),
  jamais `current_timeframe_for_trade`.
- `_get_current_prices` et les highs/lows pour `update_market_price` doivent utiliser un TF
  **stable** = TF de la position ouverte (ou 5m fixe comme TF de mark-to-market).
- Recommandation forte : **mark-to-market TOUJOURS en 5m** (le TF le plus fin, contigu),
  `current_timeframe_for_trade` ne sert qu'à l'observation/décision, jamais au prix d'exécution.

### FIX 2 — instrumentation + garde-fou (assertions)
- Logger `entry_tf, check_tf, entry_idx, check_idx, entry_price, high, low` à open/check/close.
- En mode debug : `assert entry_tf==check_tf`, `assert (high-low)/close < 5%`.
- Watchdog runtime : si div(close,high) > 5% → log ERROR + compteur.

### FIX 3 — PnL latent (demande utilisateur)
- Toutes les N steps (N=3) tant qu'une position est ouverte :
  `reward += lambda_latent * unrealized_pnl_pct` si >0,
  `reward -= lambda_latent * |unrealized_pnl_pct|` si <0.
- Borné, log (pas exponentiel). Donne au modèle la "ligne imaginaire" gain/perte.

### FIX 4 — pénalité saturation SL/TP (VERIF#7 NON)
- Si SL/TP saturent les bornes (clip actif) trop souvent sur fenêtre glissante :
  pénalité **logarithmique** croissante mais plafonnée (pas exponentielle).

### FIX 5 — promotion_bonus (root cause reward hacking, secondaire)
- Rescaler proportionnel/log, one-shot par tier, ne doit jamais dominer le PnL.

### Nettoyage
- Supprimer logs pollués (fa_500k.log 1.9GB), checkpoints invalides du run buggé.
- Relancer un run propre APRÈS FIX 1+2 validés (re-run le script de preuve = 0 div).
