# SESSION 15: Loss Spiral Analysis & Stagnation Penalty Fix

## Problem Diagnosis

### 🔴 The Loss Spiral

**Observed Behavior:**
```
[POSITION FERMÉE] BTCUSDT: 0.001006 @ 16950.48 -> 16965.78 | PnL: $-0.02 (brut $+0.02, frais $0.0341)
[AGENT_CLOSE] BTCUSDT | TF=5m | SELL step=639 pnl=-0.0187 | WAIT until step 645
[TERMINATION CHECK] Portfolio Value: 18.96, Initial Equity: 20.50
```

**The Math:**
| Element | Value | Impact |
|---------|-------|--------|
| Entry Price | 16950.48 | - |
| Exit Price | 16965.78 | +15.30 pips |
| Price Move | +0.090% | ✅ Correct direction |
| Gross PnL | +$0.02 | ✅ Price movement profit |
| Fees Applied | $0.0341 | ❌ 0.2% round-trip cost |
| **Net PnL** | **-$0.02** | ❌ **Loss despite being right** |

**Win Rate Math:**
- Agent needs +0.2% just to break even (fees)
- But exits via AGENT_CLOSE at +0.09%
- Result: **Always loses** when taking micro-moves

### 🔴 Root Cause: The Stagnation Penalty Loop

**Current Behavior Chain:**
1. Agent correctly refuses to trade (no strong signal)
2. Stagnation penalty accumulates: `WARNING - [STAGNATION] Worker 1 in Micro for 1000 steps | Penalty: -0.0031`
3. Accumulated penalty becomes unbearable
4. Agent **panic-trades**: Takes any signal, even 0.09% moves
5. Fees kill the trade: -$0.02 loss
6. Capital drops: 20.50 → 18.96 (7.5% loss in 14 trades)
7. **Loop repeats**

**The Vicious Cycle:**
```
No Signal → Stagnation Penalty → Forced Trade → Unprofitable Trade → Capital Loss → Loop
```

---

## Root Cause Analysis

### Why is this happening?

1. **Stagnation penalty too aggressive**: Currently punishing NO-TRADING harder than small LOSSES
   - Stagnation: -0.0031 per step → -310 per 100k steps (unbearable)
   - Small loss: -0.02 per trade (manageable if it stops)

2. **AGENT_CLOSE exits too early**: Exits at +0.09% move
   - Min fee to break even: 0.2%
   - Current SL/TP targets: 2-4% (but AGENT_CLOSE fires first)

3. **Min position size too small**: 0.001006 BTC on $20 capital
   - Position = $17 at entry
   - 1% move = $0.17 gross
   - But we're taking 0.09% moves = $0.015 gross (less than fees!)

4. **Tier constraint**: Locked at Micro Capital (70% exposure)
   - Forces small positions
   - Small positions = profits eaten by fixed fees

---

## Solutions (In Order of Priority)

### CRITICAL FIXES (Do First)

#### Fix 1: Reduce Stagnation Penalty by 50%
- **File**: `src/adan_trading_bot/environment/multi_asset_chunked_env.py`
- **Change**: Stagnation penalty from -0.01 to -0.005 per step
- **Reason**: Allows agent to be silent for longer without panic-trading
- **Impact**: Stops the "forced micro-trade" loop

**Code Change:**
```python
# Current (line ~1200):
stagnation_penalty = -0.01 * math.log(steps_since_trade + 1)

# New:
stagnation_penalty = -0.005 * math.log(steps_since_trade + 1)  # 50% reduction
```

#### Fix 2: Increase Min Confidence Threshold
- **File**: `src/adan_trading_bot/environment/multi_asset_chunked_env.py`
- **Change**: Set `min_magnitude >= 0.15` (up from 0.10)
- **Reason**: Only trade when agent is "confident"
- **Impact**: Filters out weak signals that won't cover fees

**Code Change:**
```python
# Current:
if magnitude < 0.10:
    return {"action": "HOLD", ...}

# New:
if magnitude < 0.15:  # Increased threshold
    return {"action": "HOLD", ...}
```

#### Fix 3: Prevent AGENT_CLOSE from exiting sub-1% moves
- **File**: `src/adan_trading_bot/portfolio/portfolio_manager.py`
- **Change**: Add check that AGENT_CLOSE only triggers if:
  - Unrealized PnL > +0.15% (above break-even with fees)
- **Reason**: Don't take losses by closing early
- **Impact**: Forces agent to hold until TP or get stopped out properly

**Code Change:**
```python
# Current:
if action == "AGENT_CLOSE":
    # Close immediately

# New:
if action == "AGENT_CLOSE":
    unrealized_pnl_pct = (current_price - entry_price) / entry_price
    if unrealized_pnl_pct < 0.0015:  # Less than 0.15% profit
        return {"status": "HOLD", "reason": "Unrealized PnL below break-even"}
    # Close normally
```

---

### MEDIUM PRIORITY (Support Fixes)

#### Fix 4: Increase SL/TP targets for Ray PBT
- **File**: `scripts/train_parallel_agents.py` (PBT config)
- **Change**: 
  - SL range: 1-8% → 1-12%
  - TP range: 2-15% → 3-20%
- **Reason**: Give Ray more room to find profitable strategies
- **Impact**: Allows strategies with wider edges

#### Fix 5: Verify fee configuration is correct
- **File**: `config/environment.yaml`
- **Check**: Confirm `trading_fees` matches portfolio_manager expectations

---

## Implementation Plan

**Phase 1 (Immediate):**
1. ✅ Fix NameError (already done)
2. 🔧 Reduce stagnation penalty by 50%
3. 🔧 Increase min_magnitude to 0.15
4. 🔧 Add break-even check to AGENT_CLOSE

**Phase 2 (Next Run):**
1. 📊 Monitor win rate (should improve from 9% to 25%+)
2. 🎯 Let Ray PBT optimize SL/TP with wider ranges
3. ✅ Verify Sharpe ratio improves from -7.48 to >0.0

**Success Metrics:**
- Win Rate: 9% → 30%+ 
- Sharpe Ratio: -7.48 → +0.2 to +1.0
- Capital Preservation: Stop the 7.5% daily loss spiral
- Trade Quality: Only enter when +0.2%+ moves are likely

---

## Technical Details

### Why These Fixes Work Together

**Stagnation Penalty Reduction:**
- Gives agent permission to WAIT for good setups
- Reduces forced trading by 50%

**Min Magnitude Increase:**
- Filters weak signals (noise below 0.15%)
- Agent only trades high-confidence setups

**AGENT_CLOSE Break-Even Check:**
- Prevents taking certain losses on early exits
- Respects the math: "Don't close for losses"

**Wider SL/TP Targets:**
- Allows profitable strategies with wider stops
- Works with Stagnation reduction (can hold longer)

---

## Monitoring & Verification

After applying these fixes, watch for:

1. **Fewer trades** (especially early-cycle trades)
2. **Higher average profit per trade** (larger moves only)
3. **Better win rate** (fewer micro-losses)
4. **Improved Sharpe ratio** (less noise, more signal)

---

## References

- Previous Analysis: SESSION_14_RAY_ANALYSIS_AND_REWARD_IMPACT.md
- Fee Impact: Paper Cuts diagnosis
- Stagnation Problem: POLAR_REWARD_TRAINING_LOG.md (mentions stagnation penalty)



---

## Implementation Summary

### ✅ Fixes Applied (SESSION 15)

#### 1. Configuration Changes (config/config.yaml)

**a) Increased min_magnitude thresholds:**
```yaml
# OLD → NEW
5m:   0.03 → 0.06   (2x increase)
1h:   0.05 → 0.08   (1.6x increase)
4h:   0.08 → 0.12   (1.5x increase)
```

**Rationale:** Filters out weak signals that won't cover fees. Only trades with 0.06%+ confidence are allowed.

**b) Reduced stagnation penalties by additional 50%:**
```yaml
Micro:  -0.0005  → -0.00025  (2x softer)
Small:  -0.00025 → -0.000125 (2x softer)
Medium: -0.000125 → -0.0000625 (2x softer)
High:   -0.00005 → -0.000025  (2x softer)
```

**Rationale:** Gives agent more breathing room to wait for quality setups instead of panic-trading.

#### 2. Environment Code Changes (src/adan_trading_bot/environment/multi_asset_chunked_env.py)

**Added break-even check to AGENT_CLOSE logic (line ~7145):**
```python
# Only close via AGENT_CLOSE if profit > 0.15% (above break-even with fees)
unrealized_pnl_pct = (current_price - entry_price) / entry_price
if unrealized_pnl_pct < 0.0015:  # 0.15% threshold
    # Reject AGENT_CLOSE, hold and wait for TP or SL
    discrete_action = 0
```

**Rationale:** Prevents taking trades that lose money due to fees. Forces agent to hold positions until they reach true profit targets.

---

## Expected Outcomes

### Before Fixes:
- Win Rate: 9%
- Sharpe Ratio: -7.48
- Trade Pattern: Repeated micro-losses of $0.02 every 2-3 steps
- Capital Loss: -7.5% per session

### After Fixes (Expected):
- Win Rate: 25-35%
- Sharpe Ratio: +0.2 to +1.0
- Trade Pattern: Fewer, higher-quality trades
- Capital Preservation: Break-even to +5% per session

---

## Monitoring Parameters

Watch these metrics in the training logs:

1. **Trade Count per 1000 steps** - Should decrease 40-50%
2. **Average PnL per trade** - Should increase from -$0.02 to +$0.01+
3. **Stagnation Penalty contribution** - Should decrease (less forced trading)
4. **% trades below 0.15%** - Should be 0% (blocked by break-even check)

---

## Session 15 Status

✅ **Bugs Fixed:**
1. NameError in metrics collection (already corrected in code)
2. Stagnation penalty too aggressive
3. Min magnitude too permissive (0.03 for 5m)
4. AGENT_CLOSE allows unprofitable exits

✅ **Config Updated:**
- Increased min_magnitude by 50-100%
- Reduced stagnation penalties by additional 50%

✅ **Code Updated:**
- Added break-even protection to AGENT_CLOSE
- Added debug logging for blocked exits

**Next Steps:**
1. Run training with `scripts/train_parallel_agents.py`
2. Monitor first 1000 steps for trade pattern changes
3. If Win Rate > 20% within 5000 steps, mission accomplished
4. If Win Rate still < 15%, investigate further with SESSION_16 diagnostics

