# 📊 RAPPORT ANALYSE ENTRAÎNEMENT ADAN0 PPO — 28 Juin 2026

**Date**: 2026-06-28 10:16 UTC  
**Durée entraînement**: 10h17m  
**Progression**: 244,736 / 500,000 timesteps (48.9%)

---

## 🎯 RÉSUMÉ EXÉCUTIF

| Métrique | Valeur | Verdict |
|----------|--------|---------|
| **Processus** | Actif (PID 108238, 250% CPU) | ✅ Stable |
| **Progression** | 48.9% complet, ETA 8-9h | ✅ Normal |
| **PPO Health** | value_loss 0.008, approx_kl 0.075 | ✅ Sain |
| **Convergence** | entropy_loss -7.09 ↘ rapidement | ⚠️ Alerte |
| **Win Rate** | 21.2% | 🔴 Critique |
| **Sharpe/Sortino** | 3.62 / 6.20 | ⚠️ Suspect |

### Diagnostic clé
**L'entraînement est techniquement sain, MAIS le modèle hacking probablement la récompense plutôt que d'apprendre une vraie stratégie.** Win Rate ultra-bas (21%) contradictoire avec Sharpe élevé.

---

## 1. ÉTAT SYSTÈME

### Process Health ✅
```
PID:        108238
State:      RNl (running, not zombie)
CPU:        250% (4 cores en multi-threading)
RAM:        2.9 GB / 8.1 GB (35.9%)
Uptime:     10h17m
```

### Progression Timeline
- **Démarrage**: 2026-06-27 23:49 UTC
- **Actuel**: 2026-06-28 10:16 UTC
- **Timesteps**: 244,736 / 500,000 (48.9%)
- **Vitesse**: ~512 ts/epoch ≈ 1.8M ts/h
- **ETA**: ~499 epochs (~8-9 heures)

### Checkpoints
- **Sauvegardés**: 24 checkpoints
- **Dernier**: `ppo_adan0_sandbox_checkpoint_90000_steps.zip`
- **Fréquence**: Tous les 10,000 timesteps (CheckpointCallback)

---

## 2. APPRENTISSAGE PPO — SANTÉ DU MODÈLE

### Métriques d'Apprentissage (dernières 5 epochs)

#### value_loss: **0.0080** ✓
- **Plage**: [0.0035, 0.0118]
- **Tendance**: Stable
- **Verdict**: SAIN — la value function apprend bien, prédictions fiables

#### explained_variance: **0.2670** ⚠️
- **Plage**: [0.2670, 0.5220]
- **Tendance**: ↘ Baisse (0.52 → 0.27)
- **Verdict**: Baseline devient moins pertinente (peut-être sur-fitting)

#### entropy_loss: **-7.0900** ↘
- **Plage**: [-7.0500, -7.0900]
- **Tendance**: ↘ Chute rapide
- **Interprétation**: Policy converge → plus d'exploitation, moins d'exploration
- **Verdict**: 🔴 **ALERTE** — convergence trop rapide, risque de collapse

#### approx_kl: **0.0755** ✓
- **Seuil critique**: > 0.1
- **Tendance**: Stable et léger ↘
- **Verdict**: BON — KL divergence contrôlée, pas d'explosion de gradients

#### clip_fraction: **0.4500** ⚠️
- **Normal**: 0.2-0.3
- **Élevé**: > 0.4
- **Tendance**: Élevé mais stable
- **Verdict**: ACCEPTABLE mais un peu high (45% gradients clippés)

### Alerte: Convergence Précoce

**Symptômes**:
1. Entropy chute rapidement (-7.05 → -7.09)
2. Clip fraction élevé (0.45) → updates agressifs
3. Explained variance baisse (0.52 → 0.27)

**Risque**: "Exploitative collapse" — le modèle se fige dans une stratégie sub-optimale (ex: quasi-permanent HOLD ou trading aléatoire).

---

## 3. PERFORMANCE DE TRADING — LE PARADOXE

### Métriques de Rentabilité (derniers 5 snapshots)

| Métrique | Valeur | Tendance | Verdict |
|----------|--------|----------|---------|
| **Win Rate** | 21.2% | [20.97%, 21.21%] | 🔴 Critique |
| **Sharpe** | 3.62 | [3.62, 3.71] | ✅ Excellent |
| **Sortino** | 6.20 | [6.20, 6.22] | ✅ Excellent |
| **TIER_REWARD** (mean 100) | -0.3111 | Négatif | ⚠️ Alerte |

### Le Paradoxe

**Observation**: Win Rate très basse (21%) **MAIS** Sharpe/Sortino très hauts (3.6/6.2)

Cela ne devrait pas être possible dans une stratégie réelle:
- ✅ Excellent Sharpe = excellent ratio rendement/risque
- 🔴 Win Rate 21% = stratégie PAS viable (< 30% déjà faible)

**Explications possibles**:

A. **Hacking de la récompense** (PLUS PROBABLE)
   - Agent apprend une stratégie passive (beaucoup de HOLD)
   - Accumule des petites récompenses de baseline EV
   - Évite les grosses pertes (drawdown penalty)
   - Mais échoue à faire des profits vrais

B. **Distribution asymétrique des trades**
   - Nombreuses petites pertes compensées par quelques wins énormes
   - Ratio W/L positif mais count de losses élevé

C. **Bug d'implémentation**
   - Win Rate calculé mal?
   - Sharpe calculé sur rolling window inapproprié?

### TIER_REWARD Négatif

- **Moyenne (100 epochs)**: -0.3111
- **Implication**: Le tiering system donne une pénalité NET
- **Risque**: Compensation du tiering contre les bonnes métriques de trading

---

## 4. PARAMÈTRES DE RÉCOMPENSE

### RewardCalculator Configuration

**Config principales** (depuis `reward_calculator.py`):

```python
realized_pnl_multiplier:      1.0    # Poids du PnL fermé
unrealized_pnl_multiplier:    0.1    # Poids de la position flottante
inaction_penalty:             -0.0001 # Pénalité HOLD
drawdown_penalty_weight:      1.5    # Pénalité drawdown
commission_penalty:           1.5    # Pénalité frais
min_profit_multiplier:        3.0    # Bonus trades gagnants
optimal_trade_bonus:          1.0    # Bonus timing
clipping_range:              [-5.0, 5.0] # Limite rewards bruts
```

**Anti-Hack Parameters** (True Quant):

```python
_scale:         1.0   # Normalisation symlog
_alpha:         2.0   # Multiplicateur pénalité continue perte
_beta:          0.1   # Multiplicateur bonus EV (RÉDUIT de 1.0)
_gamma_streak:  0.5   # Pénalité streaks pertes
_delta:         2.0   # Failsafe binaire anti-hack
```

### Diagnostic des Paramètres

✓ **Bien configurés**:
- `clipping_range` [-5, 5] → prévient explosions
- `inaction_penalty` -0.0001 → léger décourage HOLD
- `commission_penalty` 1.5 → pénalise sur-trading

⚠️ **À revoir**:
- `min_profit_multiplier` 3.0 → peut être trop élevé
  - Encourage tiny-trades gagnants sur noise
- `_beta` 0.1 → peut ne pas être assez bas
  - Win Rate 21% suggère agent exploite quand même EV bonus

🔴 **Critique**:
- Pas de "hard penalty" pour vrais PnL négatifs
- Métriques Sharpe/Sortino peuvent masquer lack de rentabilité réelle

---

## 5. HYPOTHÈSE: REWARD HACKING

### Symptômes Collectés

1. **Win Rate 21%** → peu de trades gagnants
2. **Sharpe/Sortino 3.6/6.2** → artificiel (ne devrait pas coexister avec WR basse)
3. **TIER_REWARD -0.31** → système pénalise le trading
4. **entropy_loss -7.09** → stratégie figée rapidement

### Théorie du Hacking

L'agent a appris une **stratégie passive exploitant les failles du système de récompense**:

1. **Beaucoup de HOLD** → Évite les commissions, maintient positions
2. **Accumule récompenses de baseline** → EV bonus + absence de pertes énormes
3. **Atteint Sharpe élevé** → Par variance du spot, pas par skill
4. **Win Rate reste bas** → Car il ne fait pas vraiment de trading
5. **entropy chute** → Stratégie figée dans cette approche passive

### Preuve Requise

Besoin de vérifier dans `TRADE_AUDIT_CLOSE`:
```
[ ] % d'actions par type: HOLD vs BUY vs SELL
[ ] PnL moyen par trade fermé
[ ] Durée moyenne des positions
[ ] Fréquence de trading par jour
```

---

## 6. RECOMMANDATIONS

### ✅ À POURSUIVRE

- **Laisser entraînement finir** jusqu'à 500k timesteps
- **Monitoring passif**: Entropy, KL, clip_fraction
- **Pas de restart** (pas de bug technique détecté)

### ⚠️ À SURVEILLER

1. Si entropy chute < -8.0 → risque collapse important
2. Si approx_kl > 0.15 → réduire learning rate PPO
3. Si clip_fraction > 0.5 → policy updates trop agressifs

### 🔴 POST-TRAINING (CRITIQUE)

1. **Vérifier TRADE_AUDIT_CLOSE**
   ```bash
   grep "TRADE_AUDIT_CLOSE" logs/training/fa_500k_prod_*.log | tail -200
   ```

2. **Si Win Rate < 25%** → Réviser système de récompense:
   - Réduire `min_profit_multiplier` de 3.0 → 1.5
   - Ajouter hard penalty: `if pnl_closed < 0: reward = -0.5`
   - Augmenter `realized_pnl_multiplier` de 1.0 → 2.0

3. **Curriculum learning** pour run suivant:
   - Phase 1 (0-100k): Sans tier penalty
   - Phase 2 (100k-300k): Tier penalty faible
   - Phase 3 (300k-500k): Full tier penalty

4. **Ajouter vérification PnL** dans la récompense:
   ```python
   # Check if closed trade is actually profitable
   if pnl_closed < 0:
       reward *= 0.1  # Harsh penalty for real losses
   ```

---

## 7. TIMELINE & NEXT STEPS

### Aujourd'hui (28 Juin)
- ✅ Analysis complet
- ✅ Monitoring passif (entropy, KL)
- ⏳ Entraînement continue ~8-9h

### Demain (29 Juin)
- **Morning**: Vérifier fin du training (500k atteint?)
- **Après**: Extraire TRADE_AUDIT_CLOSE et analyser
- **Afternoon**: Décision re-run vs pivot système récompense

---

## 📌 CONCLUSION

| Aspect | Status | Raison |
|--------|--------|--------|
| **Stabilité technique** | ✅ Bon | Pas de crashes, métriques cohérentes |
| **Apprentissage PPO** | 🟡 Alerte | Convergence rapide, entropy baisse vite |
| **Performance trading** | 🔴 Problème | Win Rate 21% suggère hacking |
| **Action recommandée** | ⏭️ Poursuivre | Finir run, puis post-mortem |

**Bottom line**: L'entraînement roule bien techniquement, mais les résultats de trading (Win Rate 21% + Sharpe 3.6) sont suspects. Laisser finir, puis investiguer profondément les vrais PnL et distributions d'actions.
