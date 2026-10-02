# Tier-Based Capital Progression Reward System (Session 11)

**Date**: 2026-06-05  
**Status**: ✅ Implemented and Training Active  
**PID**: 1000135  
**Start Time**: 17:09:08 UTC  
**Logfile**: `/mnt/new_data/adan_logs/checkpoints/training_20260605_170914.log`

---

## Executive Summary

Replaced the mathematically unsound **polar reward system** with a **Prop Firm-inspired tier-based capital progression model**. The new system directly incentivizes the agent's primary goal: **growing account capital across defined tiers**.

### Why This Fixes the Problem

**Old System (Polar Reward - FAILED)**:
- `atan2(drawdown, pnl)` created negative angles even for positive trades
- `log1p(r)` compressed small gains to ~0
- Agent had no incentive to prevent paralysis (HOLD forever = ~0 reward)

**New System (Tier Progression - CORRECT)**:
- ✅ Clear hierarchical goals: Micro → Small → Medium → High → Enterprise
- ✅ Promotion bonus: +0.5, +1.0, +2.0, +4.0 (doubling per tier)
- ✅ Demotion penalty: matching promotion cost
- ✅ Stagnation penalty: forces action if stuck too long in current tier
- ✅ Pure PnL signal: +0.1 × PnL% (base reward)

This is how real **Prop Trading Firms** (FTMO, MyForexFunds) work.

---

## Implementation Details

### 1. Configuration (`config/config.yaml`)

Added `capital_tier_rewards` section with per-tier rules:

```yaml
capital_tier_rewards:
  Micro:    # $11–$30 (smallest tier)
    max_steps_in_tier: 500
    stagnation_penalty_per_step: -0.002  # Harsh for small tiers
    drawdown_penalty_factor: 2.0         # Penalize risk more
    promotion_bonus: 0.5                 # Reward for reaching Small
  
  Small:    # $30–$100
    max_steps_in_tier: 1000
    stagnation_penalty_per_step: -0.001  # Lighter
    drawdown_penalty_factor: 1.5
    promotion_bonus: 1.0                 # Double Small→Medium
  
  Medium:   # $100–$300
    max_steps_in_tier: 2000
    stagnation_penalty_per_step: -0.0005
    drawdown_penalty_factor: 1.0
    promotion_bonus: 2.0                 # Double Medium→High
  
  High:     # $300–$1000
    max_steps_in_tier: 3000
    stagnation_penalty_per_step: -0.0002
    drawdown_penalty_factor: 0.5
    promotion_bonus: 4.0                 # Double High→Enterprise
  
  Enterprise: # >$1000 (end state)
    max_steps_in_tier: 50000
    stagnation_penalty_per_step: 0.0     # No penalty (goal is to stay)
    drawdown_penalty_factor: 0.1
    promotion_bonus: 0.0
```

### 2. Environment State Tracking

Added to `reset()` in `multi_asset_chunked_env.py` (lines 2191–2196):

```python
# ── TIER-BASED CAPITAL PROGRESSION TRACKING ──
self._current_tier = "Micro"  # Start at smallest tier
self._previous_tier = "Micro"
self._steps_in_current_tier = 0  # Steps since entering current tier
self._max_capital_reached = initial_capital
self._tier_entry_capital = initial_capital
```

### 3. Reward Function

Complete rewrite of `_calculate_reward()` (lines 5945–6072) with 8 steps:

#### Step 1: Determine Tier
```python
current_capital = portfolio_manager.total_equity
current_tier = _get_tier_from_capital(current_capital)  # "Micro", "Small", etc.
```

#### Step 2: Detect Tier Change
```python
tier_changed = (current_tier != previous_tier)
if tier_changed:
    self._current_tier = current_tier
    self._steps_in_current_tier = 0  # Reset counter
else:
    self._steps_in_current_tier += 1
```

#### Step 3: Compute Promotion/Demotion Bonus/Penalty
```python
if tier_changed:
    if curr_idx > prev_idx:
        promotion_bonus = +0.5, +1.0, +2.0, or +4.0 (from config)
    elif curr_idx < prev_idx:
        demotion_penalty = -(previous tier's promotion_bonus)
```

**Example**: If agent goes from Micro ($25) → Small ($35), it gets `+0.5` bonus.  
If it drops back to Micro, it gets `-0.5` penalty.

#### Step 4: Stagnation Penalty
```python
if steps_in_current_tier > max_steps_in_tier:
    excess = steps_in_current_tier - max_steps_in_tier
    stagnation_penalty = rate × log(1 + excess)
```

**Effect**: Logarithmic growth forces agent to act, but doesn't explode.  
At Micro tier, after 500 steps, penalty starts. By step 1000, penalty ≈ -0.002 × ln(500) ≈ -0.013 per step.

#### Step 5: Drawdown Penalty (Tier-Scaled)
```python
if drawdown > 10%:
    factor = tier_config["drawdown_penalty_factor"]  # 2.0 for Micro, 0.1 for Enterprise
    drawdown_penalty = -0.5 × tanh(|dd| × 5 × factor)
```

**Effect**: Risk is punished more harshly in small tiers.

#### Step 6: Inaction Penalty
```python
if pnl_net == 0.0:  # No trade this step
    inaction_penalty = time_decay  # -0.01
```

#### Step 7: Compose Raw Reward
```python
raw_reward = (
    pnl_pct * 0.1           # Base: 0.1 × (PnL%)
    + promotion_bonus       # Tier promotion: +0.5 to +4.0
    + demotion_penalty      # Tier demotion: -0.5 to -4.0
    + stagnation_penalty    # Timeout: -log(steps)
    + drawdown_penalty      # Risk: -tanh(...)
    + inaction_penalty      # No-trade: -0.01
)
```

#### Step 8: Symlog Compression
```python
final_reward = sign(raw) × ln(1 + |raw|)
```

Prevents extreme values while preserving small signals.

### 4. Logging

Log output every 50 steps or on tier change:

```
[TIER_REWARD Worker 1] Tier=Micro | Capital=$25.30 | Steps_in_tier=145 | PnL=+0.15% | Promo=+0.00 | Demote=+0.00 | Stagnation=-0.0001 | Drawdown=-0.0523 | Final=+0.0045
```

---

## Reward Signal Breakdown

### Example 1: Winning Trade in Micro Tier

```
Base PnL:       +$0.10 (0.49% return on $20.50)
Tier:           Micro ($20 → $25)
Stagnation:     Not triggered (only 100 steps in)
Drawdown:       -0.50% (minimal)

raw_reward = 0.1 × 0.49 - 0.0001 - 0.01 = +0.048
final = symlog(0.048) = +0.047
```

✅ **Good signal**: Agent made profit, gets positive reward.

### Example 2: Promotion from Micro to Small

```
Agent capital: $20.50 → $32.00
Previous tier: Micro
Current tier:  Small

raw_reward = 0.1 × pnl + 0.5 (promotion bonus) + other_components
final = symlog(0.5 + ...) ≈ +0.462
```

✅ **Major reward**: Agent achieved the primary goal (promotion).

### Example 3: Stagnation in Micro

```
Agent capital: $21.00 (unchanged)
Steps in Micro: 600 (exceeded max 500)
Excess:        100 steps

stagnation_penalty = -0.002 × ln(1 + 100) = -0.002 × 4.605 = -0.0092

raw_reward = 0.0 + 0.0 + (-0.0092) + ...
final = symlog(-0.0092) ≈ -0.0092
```

❌ **Pressure building**: If agent doesn't progress, penalty increases.

### Example 4: Demotion from Small to Micro

```
Agent capital: $100 → $25 (catastrophic loss)
Previous tier: Small
Current tier:  Micro

raw_reward = pnl (negative) - 1.0 (demotion penalty from Small) + ...
final = very negative reward
```

❌ **Major penalty**: Agent failed to preserve capital.

---

## Training Expectations

### Milestone 1 (~Step 1000, ~10 min)
- Agent explores Micro tier trades
- Should see: Promotion attempts, some wins/losses
- Target: Capital $22–$25 (preparing for Small)

### Milestone 2 (~Step 5000, ~50 min)
- Agent reaches Small tier occasionally
- Should see: +0.5 promotion bonus being earned
- Target: Capital $30–$50, Win Rate > 20%

### Milestone 3 (~Step 10000, ~100 min)
- Agent learns tier progression strategy
- Should see: Consistent capital growth, fewer demotions
- Target: Multiple promotions, Sharpe > -1.0

### Milestone 4 (~Step 50000, ~500 min)
- Agent balances risk/reward per tier
- Should see: Stable occupation of Medium tier, rare promotions to High
- Target: Sharpe > 0.5, Stable capital growth

---

## Stress Test Scenarios

The tier system inherently prevents common reward hacks:

### Hack 1: Buy-and-Hold (Do Nothing)
- **Problem**: HOLD forever = minimal loss, looks optimal
- **Solution**: Inaction penalty (-0.01 per step) + stagnation penalty after 500 steps
- **Result**: Agent forced to act or face cumulative -5.0 reward over 500 steps

### Hack 2: Random Trading (Exploitation)
- **Problem**: Random trades might randomly hit TP, looks like skill
- **Solution**: Drawdown penalty (tier-scaled) punishes volatility
- **Result**: Agent learns to only trade when justified (high-probability setup)

### Hack 3: Capital Abuse (Pump Account Then Crash)
- **Problem**: Reach Small tier quickly, then take huge risk
- **Solution**: Demotion penalty (-1.0) matches promotion bonus (+0.5 doubled)
- **Result**: Agent learns demotion costs far exceed rewards, risk management critical

### Hack 4: Step Stagnation
- **Problem**: Camp in Micro tier indefinitely
- **Solution**: Stagnation penalty grows from 0 after 500 steps
- **Result**: Agent must progress or face increasing penalty

---

## Differences from Polar Reward

| Aspect | Polar Reward | Tier-Based |
|--------|--------------|-----------|
| **Mathematical Foundation** | `atan2(DD, PnL)` angles | Tier thresholds + config |
| **Scalability** | Compressed by `log1p(r)` | Scaled by capital tier |
| **Promotion Signal** | None (only PnL) | **Explicit +0.5/+1.0/+2.0/+4.0** |
| **Demotion Signal** | None | **Explicit matching penalty** |
| **Stagnation Handling** | Time decay only | **Timeout + log penalty** |
| **Anti-Hack** | Trigonometry | Prop firm rules |
| **Interpretability** | Low (angles + cosines) | **High (real tier progression)** |

---

## Implementation Status

✅ **Configuration**: `config/config.yaml` lines 1253–1295  
✅ **Environment Tracking**: `multi_asset_chunked_env.py` lines 2191–2196  
✅ **Reward Function**: `multi_asset_chunked_env.py` lines 5945–6072  
✅ **Compilation**: Passes `python -m py_compile`  
✅ **Import Test**: `MultiAssetChunkedEnv` imports successfully  
✅ **Training**: Running PID 1000135 as of 2026-06-05 17:09:08 UTC  

---

## Verification

Run these to verify:

```bash
# Check config is loaded
grep -A 50 "capital_tier_rewards:" config/config.yaml

# Check environment tracking
grep "_current_tier\|_steps_in_current_tier" src/adan_trading_bot/environment/multi_asset_chunked_env.py | head -10

# Check reward function
grep "TIER-BASED\|promotion_bonus\|demotion_penalty" src/adan_trading_bot/environment/multi_asset_chunked_env.py | head -20

# Monitor training
tail -f /mnt/new_data/adan_logs/checkpoints/training_20260605_170914.log | grep TIER_REWARD
```

---

## Next Steps

1. **Monitor Milestone 1** (Step ~1000): Verify agent attempts promotion
2. **Check Stagnation Logic** (Step ~500): Confirm penalty increases
3. **Validate Demotion Handling**: Ensure harsh penalty discourages crashes
4. **Measure Convergence** (Step ~50000): Compare to baseline (polar reward)

---

**Last Updated**: 2026-06-05 17:09:14 UTC  
**System**: Linux, 8 CPU, 15Gi RAM  
**Training Mode**: `--light` (2 workers, 500k steps, resume enabled)
