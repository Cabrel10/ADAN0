# 🎯 ACTION PLAN — POST-TRAINING INVESTIGATION

**Généré**: 2026-06-28 10:16 UTC  
**Training Status**: 48.9% complet (244.7k / 500k timesteps)  
**Concern Level**: 🔴 **CRITIQUE** — Reward hacking suspected

---

## 📋 PRIORITÉS

| Priorité | Tâche | Raison | Effort |
|----------|-------|--------|--------|
| **🔴 P0** | Extraire + Analyser TRADE_AUDIT_CLOSE | Vérifier si vraie stratégie | 1h |
| **🔴 P0** | Calculer distribution réelle des actions | % HOLD vs BUY vs SELL | 30m |
| **🟠 P1** | Revoir paramètres `min_profit_multiplier` | Trop haut (3.0) | 15m |
| **🟠 P1** | Ajouter hard PnL penalty | Sanction vraies pertes | 30m |
| **🟡 P2** | Implémenter curriculum learning | Phase graduelle | 2h |

---

## 🔍 PHASE 1: INVESTIGATION (Jour 1, ~2h)

### 1.1 Extraire TRADE_AUDIT_CLOSE

**Objectif**: Voir la distribution réelle des trades et actions

```bash
# Command à exécuter demain matin
cd ~/webapp/MORNINGSTAR/ADAN0
LOG="logs/training/fa_500k_prod_20260627_234932.log"

# Extract derniers 500 trades
grep "TRADE_AUDIT_CLOSE" "$LOG" | tail -500 > /tmp/trades_analysis.txt

# Parse pour statistiques
python3 << 'EOF'
import re
trades = {}
with open('/tmp/trades_analysis.txt', 'r') as f:
    for line in f:
        # Parse chaque ligne TRADE_AUDIT_CLOSE
        # Extraire: action (BUY/SELL/HOLD), PnL, duration
        pass  # À implémenter spécifiquement
EOF
```

### 1.2 Analyser Distribution d'Actions

**Questions clés à répondre**:

```
1. % d'actions par type:
   [ ] HOLD: ___% (si > 70% → PROBLÈME)
   [ ] BUY:  ___% (devrait être 10-20%)
   [ ] SELL: ___% (devrait être 10-20%)

2. PnL statistics:
   [ ] PnL moyen par trade: $_____
   [ ] Win count: _____
   [ ] Loss count: _____
   [ ] Ratio W/L: _____
   
3. Timing:
   [ ] Durée moyenne positions: ___h
   [ ] Trades par jour: _____
   [ ] Max drawdown intra-trade: ___% 
```

**Résultat attendu**:
- ✅ HOLD: 45-55% → normal
- ✅ BUY/SELL: 20-30% chacun → normal
- 🔴 HOLD: > 75% → PROBLÈME (reward hacking)

### 1.3 Extraire Métriques d'Apprentissage Finales

Après que le training finisse (demain matin):

```bash
# Dernières 10 epochs et stats globales
tail -200 "$LOG" | grep -E "total_timesteps|value_loss|entropy_loss|approx_kl"
```

---

## 🔧 PHASE 2: CORRECTIONS MINEURE (1h, si nécessaire)

### 2.1 Réduire `min_profit_multiplier`

**Problème**: Actuellement 3.0 → encourage tiny-trades gagnants sur noise

**Solution**:

```python
# File: src/adan_trading_bot/environment/reward_calculator.py
# Dans __init__, ligne ~76

# BEFORE:
self.min_profit_multiplier = self.config.get("min_profit_multiplier", 3.0)

# AFTER:
self.min_profit_multiplier = self.config.get("min_profit_multiplier", 1.5)
```

**Et dans config/environment.yaml**:

```yaml
reward_shaping:
  min_profit_multiplier: 1.5  # Reduced from 3.0
```

### 2.2 Augmenter `realized_pnl_multiplier`

**Problème**: Actuellement 1.0 → pas assez de poids sur PnL réel fermé

**Solution**:

```python
# BEFORE:
self.pnl_multiplier = self.config.get("realized_pnl_multiplier", 1.0)

# AFTER:
self.pnl_multiplier = self.config.get("realized_pnl_multiplier", 2.0)
```

### 2.3 Ajouter Hard Penalty pour PnL Négatif

**Problème**: Pas de pénalité aigüe pour vraies pertes

**Solution** (ajouter dans `calculate` method):

```python
def calculate(self, ..., trade_pnl, ...):
    # ... existing code ...
    
    # NEW: Hard penalty for real losses
    if trade_pnl < 0:
        loss_penalty = -0.5  # Harsh
        reward = max(reward, loss_penalty)
    
    # ... clipping, etc. ...
```

---

## 🎓 PHASE 3: CURRICULUM LEARNING (2h, pour next run)

### 3.1 Stratégie Multi-Phase

**Concept**: Entraîner progressivement sans tier penalty d'abord

```
Phase 1 (0-100k timesteps):
  - NO tier penalty
  - Focus pure PnL learning
  - Reward: simply PnL + commission cost
  
Phase 2 (100k-300k timesteps):
  - Light tier penalty (0.5x normal)
  - Start introducing capital constraints
  - Reward: PnL + 0.5x tier penalty
  
Phase 3 (300k-500k timesteps):
  - Full tier penalty
  - Strict capital tier enforcement
  - Reward: full formula (PnL + tier + drawdown)
```

### 3.2 Implémentation

```python
# File: src/adan_trading_bot/environment/reward_calculator.py

def __init__(self, env_config, current_timestep=0):
    # ... existing code ...
    
    # NEW: curriculum factor based on timestep
    self.total_training_steps = env_config.get("total_training_steps", 500000)
    self.current_timestep = current_timestep
    self._phase = self._compute_phase()
    self._tier_penalty_multiplier = self._compute_tier_multiplier()

def _compute_phase(self):
    """Compute training phase based on timesteps."""
    pct = self.current_timestep / self.total_training_steps
    if pct < 0.2:
        return 1  # Phase 1: 0-20%
    elif pct < 0.6:
        return 2  # Phase 2: 20-60%
    else:
        return 3  # Phase 3: 60-100%

def _compute_tier_multiplier(self):
    """Return tier penalty multiplier based on phase."""
    phase_map = {
        1: 0.0,    # No tier penalty
        2: 0.5,    # Half penalty
        3: 1.0,    # Full penalty
    }
    return phase_map.get(self._phase, 1.0)

def calculate(self, ...):
    # ... existing code ...
    
    # Apply curriculum-based tier penalty
    tier_reward = tier_reward * self._tier_penalty_multiplier
    
    # ... rest of calculation ...
```

---

## 📊 PHASE 4: RE-VALIDATION (3h, pour v2 training)

### 4.1 Metrics to Track

```
BEFORE training:
  - Note baseline: Win Rate, Sharpe, Sortino
  - Take checkpoint at 100k timesteps

AFTER training (500k):
  - Compare: Win Rate (should be > 35% ideally)
  - Compare: Sharpe/Sortino (should stay healthy)
  - Check: TIER_REWARD (should be close to 0, not -0.31)
  - Verify: % HOLD (should be 45-55%, not > 75%)
```

### 4.2 Decision Tree

```
IF Win Rate > 35% AND HOLD% < 60% THEN:
    ✅ ACCEPT: Model learned real strategy
    → Deploy to paper trading
    
ELSE IF Win Rate > 25% AND HOLD% < 65% THEN:
    🟡 ACCEPTABLE: Some improvement
    → Iterate Phase 2 tuning, re-run
    
ELSE (Win Rate < 25% OR HOLD% > 75%) THEN:
    🔴 REJECT: Still hacking
    → Major redesign needed (see section below)
```

---

## 🔨 IF PROBLEM PERSISTS: MAJOR REDESIGN

**If Phase 3 still shows WinRate < 25%**:

### Option A: Completely Remove Tier System

```python
# Simplify reward to pure PnL
reward = pnl * 10  # Scale to reasonable [-1, 1] range
# No tier penalty at all
```

**Rationale**: Tier system might be causing hacking behavior

### Option B: Separate Tier as Constraint, Not Reward

```python
# Use tier as CONSTRAINT (observation input), not reward signal
# Let agent learn tier management implicitly

obs = [price, position, account_pct, tier_level]  # tier as info
reward = pnl_simple  # No tier in reward
```

### Option C: Use Real Backtest Metrics

```python
# Instead of Sharpe/Sortino, reward on:
# - Positive PnL trades (count)
# - Total return % (realized)
# - Drawdown avoided (measured as peak-to-trough)

reward = (realized_pnl / initial_capital) - (drawdown * 0.5)
```

---

## 📅 TIMELINE

```
2026-06-28 (Today):
  08:16 → 18:00: Training finishes (~9h remaining)

2026-06-29 (Tomorrow):
  08:00: Check if training completed
  08:15-09:00: Extract TRADE_AUDIT_CLOSE
  09:00-10:00: Data analysis + decision
  10:00-11:00: If OK → apply Phase 2 fixes
  
2026-06-30 (Day After):
  Start v2 training with curriculum learning
  
2026-07-01:
  Analyze v2 results, decide on major redesign if needed
```

---

## ✅ CHECKLIST

- [ ] Training completes at 500k timesteps
- [ ] Extract TRADE_AUDIT_CLOSE from final log
- [ ] Calculate % HOLD vs BUY vs SELL
- [ ] Analyze PnL distribution
- [ ] Document findings in results file
- [ ] If Win Rate < 25%: apply Phase 2 fixes
- [ ] If Win Rate 25-35%: iterate Phase 2
- [ ] If Win Rate > 35%: accept and deploy
- [ ] Implement curriculum learning for v2
- [ ] Start v2 training with new config
- [ ] Monitor v2 for same hacking patterns

---

## 📝 NOTES

**Key insight**: The paradox of high Sharpe/Sortino + low Win Rate is the smoking gun. This only happens when:
1. Agent takes very few trades (mostly HOLD)
2. Small wins + small losses average out on Sharpe math
3. But counting only closed trades shows 21% win rate

This is textbook reward hacking. Fix the reward function to align with real trading objectives.

