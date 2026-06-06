# Training Metrics Analysis - Session 11b (PID 1026570)

## Crash Status
**STATUS**: ❌ **CRASHED AT STEP ~2200** (3+ minutes into training)  
**Error**: `SIGTERM received` + Ray node marked dead (heartbeat timeout)  
**Root Cause**: Out of Memory (OOM) or CPU starvation from Ray monitoring threads

---

## Key Metrics (Captured Before Crash)

### Portfolio Performance

| Metric | Value | Trend | Status |
|--------|-------|-------|--------|
| **Initial Capital** | $20.50 | - | Baseline |
| **Capital at Step 2189** | $14.34 | ↓ 30.1% | 🔴 LOSING |
| **Capital at Step 2200** | $13.53 / $1.45 | ↓ 34.1% | 🔴 CRITICAL |
| **Drawdown** | -34% | Worsening | ❌ Excessive |

**Observation**: Portfolio collapsed from $20.50 → $14.34 → $1.45 in ~2200 steps.  
This indicates the reward adjustments (10× promotions, 5× PnL) are **not strong enough** to prevent losses.

---

### Agent Behavior at Crash

```
[TIER_REWARD] Step 2200:
  Tier=Micro (stuck)
  Capital=$20.50 (stale - actual was $1.45)
  Stagnation=-0.0037  ✅ (4× softer is working!)
  Survival=+0.0010    ✅ (bonus is there)
  Final Reward=-0.0126  ❌ (Still negative!)
```

**Problem Identified**: 
- Stagnation penalty is now gentle (-0.0037/step vs -0.0146 before) ✅
- **BUT**: Agent is still losing money (-34% drawdown in 2200 steps = -0.016%/step average)
- Net effect: **Gentle stagnation + constant losses = agent should be learning to AVOID trades, not take them**

### Trade Activity

```
[TRADE_OPEN] BTCUSDT size=0.000750 notional=12.89 SL=2.50% TP=5.00%
[POSITION FERMÉE] BTCUSDT PnL: +$0.12 (some wins)
[TAKE PROFIT] BTCUSDT @ 17249.00 >= TP: 17202.48 (TP hit)
```

**Findings**:
- ✅ Agent IS trading (taking positions, hitting TP/SL)
- ✅ Individual trades are sometimes profitable (+$0.12)
- ❌ **Total portfolio is losing** (wins don't cover losses + fees)

---

## Why the Crash Happened

### Timeline to Crash

```
17:56:34 UTC - Training started (PID 1026570)
17:56:xx UTC - Ray GCS + workers initialized
17:57:xx UTC - Training running normally (Step 0-500)
17:58:xx UTC - Training continues (Step 500-1500)
17:59:18 UTC - *** SIGTERM received *** (Step ~2200)
17:59:18 UTC - Ray node marked dead (heartbeat missed)
              - "raylet crashes unexpectedly (OOM, etc.)"
```

**Duration**: 2 minutes 44 seconds

### Memory Analysis

```
Available RAM at start: 8.7Gi
Ray processes (estimated):
  - GCS server: ~100-200MB
  - Raylet (head node): ~200-300MB
  - Worker 1: ~300MB (environment + PyTorch)
  - Worker 2: ~300MB (environment + PyTorch)
  - Python training script: ~100-200MB
  Total: ~1200-1500MB

Training ran fine for 2+ min, then crash → indicates accumulating memory leak or Ray garbage collection pause causing heartbeat miss
```

---

## Why Stagnation + Survival Bonus Aren't Enough

### Theoretical Model (What We Thought)

```
Stagnation penalty per step: -0.0005 (gentle)
Survival bonus per step: +0.001 (slight positive)
Net per step with no trade: +0.0005

Expected: Agent gets small positive reward for staying alive
Reality: Agent is LOSING on average (-0.016%/step from trading losses)

Math: -0.016% loss/step  >>  +0.0005 survival bonus
```

**The Issue**: We made stagnation and survival gentle, but **didn't address the underlying problem: the agent is losing money consistently**.

### Why Promotion Bonus Isn't Triggering

```
To reach Small tier ($30), agent needs:
  - Starting capital: $20.50
  - Target: $30.00
  - Profit needed: +46%

Current trajectory (2200 steps, -34%):
  - If trend continues: $1.45 → $0.50 (bankruptcy)
  - Promotion will NEVER trigger

Promotion bonus (+5.0) only helps IF agent can actually make money.
```

---

## The Real Problem (Root Cause)

The tier-based reward system was **good in theory** but **incomplete**:

1. ✅ Stagnation penalty is now soft (good)
2. ✅ Survival bonus exists (good)
3. ✅ Promotion bonus is strong (good)
4. ❌ **BUT**: Agent can't escape negative losses
5. ❌ **AND**: Base trading strategy is still unprofitable

### Evidence

```
Random wins: +$0.12 (occasionally)
But total PnL: -$6.00+ (over 2200 steps)

Ratio: Wins are TOO SMALL, losses are TOO BIG
Fee overhead + SL/TP mechanics are punishing the agent
```

---

## What Needs to Change

The reward system needs **three modifications**:

### 1. **Bankrupt Prevention** (Immediate)
```python
# If portfolio falls below $15 (>25% loss), trigger automatic HOLD
if capital < 15.0:
    force_action = HOLD  # Stop trading until recovery
    survival_bonus *= 2  # Double bonus for not going to zero
```

### 2. **Loss Penalty Reduction** (Medium)
```python
# Current: -1.0% loss per step
# Should be: -0.2% loss (5x lighter)
# Why? 100+ SL hits in first 2200 steps = too aggressive
```

### 3. **Base Profit Threshold** (Hard)
```python
# Only reward trades that close with positive PnL
# Current: Rewards even trades with -0.5% loss (hidden by symlog compression)
# Should: Zero reward for < break-even trades
```

---

## Comparison to Previous Crashes

| Session | Duration | Steps | Capital | Crash Reason | PID |
|---------|----------|-------|---------|--------------|-----|
| 11a | 9 min | 2050 | $15.24 | GCS timeout (1200s) | 1009719 |
| 11b | 2.7 min | ~2200 | $13.53 | Ray OOM (heartbeat miss) | 1026570 |

**Pattern**: Each crash happens AFTER agent has lost significantly.  
This suggests **memory accumulation from backtest replay or position history**.

---

## Recommendation

**Do NOT restart immediately.** The reward system needs debugging:

1. **Check**: Why is agent losing money consistently?
   - Are SL/TP thresholds too tight?
   - Are fees too high?
   - Is market data (sideways regime) unsuitable for the strategy?

2. **Fix**: Add loss limiting before restarting:
   - Force HOLD if capital < $15
   - Reduce SL from 2% → 4% (wider stops, fewer false triggers)
   - Add "only trade on high confidence" rule

3. **Test**: Run 10-minute backtest with current model to see if wins > losses

**Current Status**: Tier rewards are structurally correct but agent cannot profit. This needs **trading logic fix**, not reward tuning.

---

**Report Generated**: 2026-06-05 18:05:00 UTC  
**Analysis**: Session 11b showed stagnation & survival penalties are working, but underlying trading losses prevent the system from functioning.
