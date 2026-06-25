# Système de Récompense & Pénalités V3 — après corrections (2026-06-25)

> Document de conception. Fait suite aux 3 découvertes critiques de la session
> d'audit d'exécution. À lire AVANT toute reprise d'entraînement 500k.

---

## 0. Contexte — pourquoi V3

L'audit du run 500k DiagGaussian (stable, 3h, 85k steps) a révélé que le run
était **stable mais entraîné sur un portefeuille physiquement impossible** :

| Bug | Symptôme | Statut |
|-----|----------|--------|
| **#3a Levier fantôme 5x** | `max_position_value = cash * 5.0` (spot, leverage=1) → position 229$ sur compte 14-20$ ; equity 20$→3792$ (185x) | ✅ CORRIGÉ (`portfolio_manager` : notional ≤ `leverage*cash` = cash en spot) |
| **#3b Cap de palier ignoré** | `enable_position_cap=False` → exposure_range 70-90% appliqué seulement à h==4 | ✅ CORRIGÉ (`config: enable_position_cap: true`) |
| **#3c AGENT_CLOSE sans limite** | rien ne bornait les interruptions manuelles → micro-scalping + blocage des BUY | ✅ CORRIGÉ (quota `max_per_day:7`, `max_consecutive:3`) |
| **#3d Exploit TP-large / SL-large** | l'agent met TP et SL au maximum de la bande, encaisse 72% de petits TP, subit de rares gros SL | ⚠️ À TRAITER PAR LE REWARD V3 |

**Conséquence centrale :** tant que le portefeuille était cassé (#3a), les
récompenses PnL étaient fausses → `approx_kl=0.96`, `clip_fraction=0.70`,
`explained_variance<0`, dérive `size_μ → -1.3` n'étaient que des **symptômes**.
Avec #3a/#3b/#3c corrigés, le reward doit maintenant traiter #3d et guider le
sizing.

---

## 1. Principes directeurs du reward V3

1. **Le reward récompense le PnL NET RÉEL** (après frais A/R), jamais le PnL brut.
2. **Risk-adjusted, pas absolu** : un gain de 1% à risque 6% vaut moins qu'un
   gain de 1% à risque 2%. On pénalise l'asymétrie « petit TP / gros SL ».
3. **Le sizing doit être tiré par le résultat, pas figé** : sur-pénaliser une
   sous-exploitation manifeste (micro-position sur un mouvement favorable) et
   sur-pénaliser une sur-exposition qui finit en gros SL.
4. **Borné et lisse** : chaque composante est clampée ; pas de terme qui peut
   dominer (sinon il « noie » le signal — leçon de l'Arène V2).
5. **Les contraintes dures (levier, cap, quota AGENT_CLOSE) sont gérées par
   l'environnement** (rejet/conversion en HOLD), PAS par le reward. Le reward
   ajoute seulement un *gradient* doux pour décourager d'y revenir.

---

## 2. Formule globale (par step)

```
R_t =  w_pnl   * R_pnl_net        # cœur : PnL net réalisé (clos ce step)
     + w_unreal* R_unrealized     # mark-to-market borné (continuité)
     - w_dd    * P_drawdown       # pénalité drawdown (quadratique douce)
     - w_asym  * P_asymmetry      # NOUVEAU : asymétrie TP/SL (anti-exploit #3d)
     - w_size  * P_size_misuse    # NOUVEAU : mauvais sizing (sous/sur-exposition)
     - w_freq  * P_frequency      # fréquence trades hors-cible
     - w_quota * P_agentclose     # gradient quota AGENT_CLOSE (complète l'env)
     + B_closure                  # bonus clôture profitable / pénalité MAX_DURATION
     + B_tier                     # bonus promotion / pénalité démotion palier
```

Tous les `R_*`, `P_*`, `B_*` sont **clampés dans [-1, +1]** avant pondération.

---

## 3. Détail des composantes

### 3.1 R_pnl_net (cœur)
```
R_pnl_net = clip( pnl_net_realized / (capital_avant_trade * pnl_ref), -1, +1 )
pnl_ref = 0.02   # 1 unité de reward = +2% net du capital
```
- `pnl_net_realized` = somme des `receipt["pnl"]` (déjà net de frais) clos ce step.
- Normalisé par le capital AVANT le trade → invariant d'échelle entre paliers.

### 3.2 R_unrealized (continuité, borné)
```
R_unrealized = clip( delta_unrealized_pct / 0.05, -0.3, +0.3 )
```
- Petit signal de continuité pour les positions ouvertes (évite reward sparse).
- Plafonné à ±0.3 pour ne pas encourager le « paper gain » non réalisé.

### 3.3 P_drawdown (douce, quadratique)
```
dd = max(0, (peak_equity - equity) / peak_equity)
P_drawdown = clip( (dd / dd_ref)^2, 0, 1 ),   dd_ref = max_drawdown_pct du palier
```
- Quadratique → tolère le bruit, punit fort l'approche du seuil de kill.
- **Important** : `dd_ref` = `max_drawdown_pct` du palier courant (Micro=40%).

### 3.4 P_asymmetry — ANTI-EXPLOIT #3d (NOUVEAU)
L'exploit : TP et SL tous deux poussés au max de bande → R/R réel défavorable
au regard de la *probabilité* (beaucoup de petits gains, rares pertes énormes).

On pénalise l'**espérance déséquilibrée** au moment de l'OUVERTURE :
```
rr = tp_pct / sl_pct                       # ratio reward/risk effectif
# cible : rr ∈ [1.5, 3.0]. En dessous = SL trop large vs TP.
if rr < 1.5:
    P_asymmetry = clip( (1.5 - rr) / 1.5, 0, 1 )      # SL trop large
elif rr > 4.0:
    P_asymmetry = clip( (rr - 4.0) / 4.0, 0, 1 )      # TP irréaliste (jamais atteint)
else:
    P_asymmetry = 0
```
Plus, à la CLÔTURE, pénalité si le pattern « gros SL » se matérialise :
```
if reason == "SL_HIT" and loss_pct > 1.5 * median_win_pct_recent:
    P_asymmetry += clip( loss_pct / (3 * sl_ref), 0, 0.5 )
```
→ force l'agent vers des SL plus serrés et un R/R sain, cassant le
« ramasseur de miettes ».

### 3.5 P_size_misuse — SIZING TIRÉ PAR LE RÉSULTAT (NOUVEAU)
Remplace l'ancienne dérive non guidée de `size_μ`. Deux volets, évalués **à la
clôture** (quand on connaît le mouvement réalisé) :

**a) Sous-exploitation** (micro-position sur trade gagnant) :
```
if pnl_net > 0 and position_size_pct < 0.5 * tier_max_pos_pct:
    opportunity = (mouvement_favorable_pct)               # ex: +8% atteint
    P_size_misuse += clip( opportunity * (1 - size_ratio) / 0.10, 0, 0.5 )
    # size_ratio = position_size_pct / tier_max_pos_pct
```
→ « tu avais raison mais trop petit » : douleur proportionnelle au manque à gagner.

**b) Sur-exposition** (grosse position sur trade perdant) :
```
if pnl_net < 0 and position_size_pct > 0.7 * tier_max_pos_pct:
    P_size_misuse += clip( abs(loss_pct) * size_ratio / 0.06, 0, 0.5 )
```
→ « tu as misé gros et perdu » : douleur proportionnelle à la taille ET à la perte.

**Effet net** : `size_μ` n'est plus laissé dériver ; il est attiré vers la taille
qui maximise le PnL risk-adjusted. C'est exactement le remède que l'utilisateur
réclamait (« obliger, par la douleur mathématique, à ajuster size_μ »).

### 3.6 P_frequency (cible de trades)
```
# cible : total_daily_min .. total_daily_max (config = 1..50, cible douce ~8-15)
trades_today vs [low, high]
if trades_today < low:  P_frequency = clip((low - trades_today)/low, 0, 1) * 0.3
if trades_today > high: P_frequency = clip((trades_today - high)/high, 0, 1) * 0.3
```

### 3.7 P_agentclose (gradient complétant l'env)
L'environnement convertit déjà l'AGENT_CLOSE hors-quota en HOLD (#3c). Le reward
ajoute un **gradient doux** pour que PPO apprenne à NE PAS tenter :
```
if agent_close_consecutive >= max_consecutive - 1:   # on approche du mur
    P_agentclose = 0.1
if agent_close_count_today >= max_per_day - 1:
    P_agentclose += 0.1
# + la pénalité fixe -0.10 déjà émise par l'env quand le quota bloque réellement
```

### 3.8 B_closure / B_tier (existants, conservés)
- `+0.5` AGENT_CLOSE profitable, `-0.x` MAX_DURATION (déjà en place ligne 6094).
- Bonus promotion / pénalité démotion de palier (déjà en place ligne 6066).

---

## 4. Poids recommandés (point de départ)

| Poids | Valeur | Justification |
|-------|--------|---------------|
| `w_pnl` | **1.0** | le PnL net est le signal maître |
| `w_unreal` | 0.15 | continuité, doit rester minoritaire |
| `w_dd` | 0.30 | survie, mais ne doit pas paralyser le sizing |
| `w_asym` | **0.40** | fort : c'est l'anti-exploit principal |
| `w_size` | **0.35** | fort : guide le sizing (cœur du problème) |
| `w_freq` | 0.05 | léger (l'env gère déjà les cooldowns) |
| `w_quota` | 0.10 | léger (l'env gère déjà le hard-block) |

> Règle d'or : `w_asym + w_size + w_dd ≤ w_pnl + 0.5` pour éviter que les
> pénalités « noient » le PnL (leçon V2). Ici : 0.40+0.35+0.30 = 1.05 ≤ 1.5 ✅.

---

## 5. Métriques de validation (run de contrôle 50k après V3)

| Métrique | Cible | Anti-objectif (exploit) |
|----------|-------|-------------------------|
| equity (Micro, cap 30$) | croissance réaliste, jamais > leverage*cash | explosion 185x |
| notional / cash | ≤ 1.0 (spot) | > 1.0 |
| size_μ (post-tanh) | converge ∈ [-0.5, +0.5], **stable** | dérive monotone → -1.3/-7 |
| R/R réalisé (tp/sl) | médiane ∈ [1.5, 3.0] | tp_max & sl_max systématiques |
| PnL médian TP vs SL | \|SL médian\| ≤ 2× TP médian | TP=+1$ / SL=-66$ |
| AGENT_CLOSE / jour | ≤ 7 | spam |
| approx_kl | < 0.05 | 0.96 |
| clip_fraction | < 0.3 | 0.70 |
| explained_variance | > 0.3 | < 0 |

Si `approx_kl` reste > 0.1 APRÈS correction portefeuille, alors réduire
`learning_rate` (ex. 3e-4 → 1e-4) et/ou `target_kl=0.05` dans PPO — mais
seulement APRÈS avoir confirmé que le portefeuille est sain.

---

## 6. Ordre d'implémentation conseillé

1. ✅ (FAIT) Corriger le levier fantôme + cap palier + quota AGENT_CLOSE.
2. Implémenter `P_asymmetry` (§3.4) — plus haut ROI, casse l'exploit #3d.
3. Implémenter `P_size_misuse` (§3.5) — guide le sizing.
4. Brancher les poids §4 derrière des clés config `reward_v3.weights.*`
   (overridables par env, comme `ADAN_*`), avec un flag `reward_v3.enabled`.
5. Run de contrôle 50k (DiagGaussian, `ADAN_USE_SDE=0 ADAN_LOG_STD_INIT=-1.0`),
   valider la table §5.
6. Seulement ensuite : 500k.

---

## 7. Ce qui NE change PAS (et pourquoi)

- **DiagGaussian (`ADAN_USE_SDE=0`)** reste le mode d'exploration : prouvé stable
  (σ plat 0.367). gSDE reste désactivé tant que le reste n'est pas validé.
- **L'architecture** (ContextualTemporalFusionExtractor) : intacte, prouvée
  fonctionnelle par `audit_execution.py`.
- **Les frais** : 0.4% par côté (0.8% A/R) — réalistes, conservés.
- **Le `agent_close_barrier`** (seuil 1.5× frais) : conservé, complémentaire au
  quota.

---

## 8. ADDENDUM — Decision Budget & Symmetry Enforcement (IMPLÉMENTÉ 2026-06-25)

Suite à l'analyse « décisions gratuites / coût nul des intentions », les murs
rigides (`if/else`, quota 7/jour) ont été remplacés / complétés par une
**friction structurelle continue** de niveau HFT institutionnel. Tout est
configurable sous `trading_rules:` dans `config/config.yaml`.

### 8.1 Decision Budget (jauge d'énergie [0,1])
| Événement | Effet sur la jauge |
|---|---|
| Départ / nouvel épisode | `budget = max = 1.0` |
| HOLD (aucun trade exécuté) | `+recharge_hold` (0.02), plafonné à `max` |
| BUY exécuté | `−cost_buy` (0.15) |
| AGENT_CLOSE exécuté | `−cost_close` (0.30) |
| `budget < cost_close` **ou** gap < `min_gap_steps` **ou** quota/jour atteint | AGENT_CLOSE → **HOLD forcé** + pénalité gradient `−0.10 − 0.10·deficit` |

**Effet mécanique prouvé** (`tests/test_decision_budget_v3.py`) : sur 40
tentatives de scalping consécutives → **4 closes seulement**, 36 bloquées,
cooldown naturel de **12 steps** entre chaque close. Micro-scalping
mathématiquement étouffé.

### 8.2 Symmetry & Volatility Enforcement (pénalité LATENTE, par step ouvert)
- **Asymétrie RR** : `−rr_lambda · (|RR − target_rr| − rr_tolerance)` si déviation
  > tolérance. `RR = take_profit_pct / stop_loss_pct`. Cible 1.5, tolérance 0.5.
- **Lâcheté SL/ATR** : `−sl_atr_lambda · ((SL − sl_atr_mult_max·ATR%) / ATR%)`
  si le SL dépasse `2.0 × ATR%`. ATR% lu via `_get_atr_pct_for_asset`.
- Plafonnée à `max_step_penalty` (0.15) par position et par step.

### 8.3 Close Intention Penalty (au site AGENT_CLOSE)
`−lambda_duration · ((min_hold − durée)/min_hold) · pnl_factor` si durée <
`min_hold_steps` (6). `pnl_factor = 1.5` si PnL ≤ 0 (close panique non rentable).

### 8.4 Action Entropy Constraint (fenêtre glissante)
`−lambda_switch · (taux_switch − switch_threshold)` si la fréquence de
changement d'action sur `window` (20) steps dépasse `switch_threshold` (0.5).
Alimentée par `_action_history` (actions EFFECTIVES).

### 8.5 Intégration reward
`symmetry_penalty` et `action_entropy_penalty` sont ajoutés à `raw_reward` ;
`close_intention` et le gate budget passent par `_step_invalid_penalty`
(propagé dans `realized_pnl`). Le tout reste comprimé par le `symlog` final.
