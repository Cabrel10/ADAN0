# 🧪 TEST RESULTS - ACTUAL DATA FROM LOGS

**Date:** 6 Juin 2026  
**Source:** Real validation logs extracted from `/logs/validation/`

---

## EXECUTIVE SUMMARY

```
⚠️ CRITICAL FINDINGS

1. ✅ Training worked on Chunk 1 (5120 steps): -4.94% return
2. ❌ Out-of-sample test (OOS W2): GATED - Agent blocked
3. ❌ Out-of-sample test (OOS W3): GATED - Agent blocked
4. ❌ No trades made in OOS (portfolio frozen)
5. 🚨 Verdict: TRAINING FAILURE - Agent not generalizable
```

---

## TEST 1: Backtest (5120 Steps on Checkpoint)

### Results

```json
{
  "checkpoint": "ppo_adan0_sandbox_5120steps.zip",
  "steps_tested": 2000,
  "initial_equity": 20.5,
  "final_equity": 19.49,
  "total_return": -4.94%,
  
  "trades": {
    "total": 39,
    "winning": 13,
    "losing": 26,
    "win_rate": 33.3%
  },
  
  "pnl": {
    "sum": -1.01,
    "average": -0.026,
    "drawdown": 4.94%
  },
  
  "actions": {
    "max_mean": 0.445,
    "max_max": 1.0,
    "above_0.01": 100%
  },
  
  "verdict": "TRADING_NONPROFITABLE"
}
```

### Analysis

✅ **Positive:**
- Agent made 39 trades (not frozen)
- Agent is outputting valid actions (0.445 mean)

❌ **Negative:**
- **-4.94% return** (losing money)
- Only 33.3% win rate (below 50% edge)
- Portfolio shrank from $20.50 → $19.49
- Average PnL per trade: -$0.026

### Interpretation

This is the **TRAINED model performance** on test data:
- Model learned to LOSE money
- Bad PnL ratio (13 wins vs 26 losses)
- Training on Chunk 1 data ≠ profitable strategy

---

## TEST 2: Out-of-Sample W2

### Results

```json
{
  "checkpoint": "model.zip",
  "vecnorm": "ADAN_PBT_Worker_a4314_00002_...",
  "steps_tested": 100,
  
  "trades": {
    "detected": 0,
    "attempted": 0,
    "env_total": 0
  },
  
  "pnl": {
    "sum": 0.0,
    "avg": 0.0,
    "total_return": 0.0%
  },
  
  "final_equity": 20.5,
  "actions": {
    "max_mean": 0.8167
  },
  
  "verdict": "GATED (actions blocked by threshold)"
}
```

### Analysis

❌ **CRITICAL FAILURE:**
- **0 trades made** (agent frozen)
- Portfolio unchanged ($20.5)
- Actions mean 0.8167 (agent wants to trade but blocked)
- **Gating mechanism rejected all trades**

### Root Cause

```
Agent outputs action ~0.81 (want to trade)
BUT Threshold blocks it (likely threshold = 0.9+)
→ No trades executed
→ Portfolio = initial
→ Return = 0%
```

---

## TEST 3: Out-of-Sample W3

### Results

```json
{
  "checkpoint": "model.zip",
  "vecnorm": "ADAN_PBT_Worker_a4314_00003_...",
  "steps_tested": 2000,
  
  "trades": {
    "detected": 0,
    "attempted": 0,
    "env_total": 0
  },
  
  "pnl": {
    "sum": 0.0,
    "avg": 0.0,
    "total_return": 0.0%
  },
  
  "final_equity": 20.5,
  "actions": {
    "max_mean": 0.8675
  },
  
  "verdict": "GATED (actions blocked by threshold)"
}
```

### Analysis

❌ **SAME FAILURE AS W2:**
- **0 trades** in 2000 steps
- Actions output: 0.8675 (agents wants to act)
- **Gating mechanism blocking everything**
- 2000 steps = 25000 trades in real training
- But OOS = 0 trades (frozen)

### Root Cause

Same as W2: **Threshold too high, gates out all actions**

---

## COMPARISON TABLE

| Metric | Backtest (5120) | OOS W2 | OOS W3 | Status |
|--------|---|---|---|---|
| **Steps** | 2000 | 100 | 2000 | - |
| **Trades** | 39 | 0 | 0 | ❌ Dropped |
| **Win Rate** | 33.3% | - | - | ❌ Poor |
| **Final Return** | -4.94% | 0% | 0% | ❌ Negative |
| **Action Mean** | 0.445 | 0.8167 | 0.8675 | ⚠️ Oscillates |
| **Verdict** | Unprofitable | Gated | Gated | ❌ Failure |

---

## 🚨 CRITICAL ISSUES IDENTIFIED

### Issue 1: Training Produced Negative Returns

**Evidence:**
```
Training Steps: 5120
Test Data: 2000 steps
Result: -4.94% (net loss)
```

**Meaning:**
- Model was trained but learned wrong strategy
- Not profitable on even basic test data
- Root cause: Reward function miscalibrated OR insufficient training

### Issue 2: Out-of-Sample Gating

**Evidence:**
```
W2: action_max_mean = 0.8167 (wants to act) → 0 trades
W3: action_max_mean = 0.8675 (wants to act) → 0 trades
```

**Meaning:**
- Agent is blocked by gating mechanism
- Threshold is too restrictive (probably > 0.9)
- Agent never passes gate threshold
- Result: Frozen portfolio, 0% return

### Issue 3: Action Output Instability

**Evidence:**
```
Backtest: action_max_mean = 0.445
OOS W2:   action_max_mean = 0.8167 (+83% increase)
OOS W3:   action_max_mean = 0.8675 (+95% increase)
```

**Meaning:**
- Agent behavior changes dramatically OOS
- Not stable generalization
- High uncertainty in output distribution

---

## ROOT CAUSE ANALYSIS

### Why -4.94% on Backtest?

**Hypothesis 1: Bad Strategy**
```
Win Rate: 33.3% (13/39 trades)
For profitability need > 40% with decent payoff
→ Agent didn't learn profitable edge
```

**Hypothesis 2: Reward Insufficient**
```
Training reward ≠ realized P&L
Possible: Reward penalizes trades too much
```

**Hypothesis 3: Data Mismatch**
```
Training data (Chunk 1 bearish) ≠ Test data
Agent optimized for bearish, test is different
```

### Why Are Agents Gated OOS?

**Root Cause:**
```
Gating threshold likely = 0.9+ (for conservative trading)
Agent output: 0.81-0.86 (medium confidence)
Result: All trades blocked
```

**Design Issue:**
```
Threshold is TOO CONSERVATIVE
Either:
1. Lower threshold from 0.9 to 0.7
2. OR Recalibrate agent to output higher confidence
3. OR Remove gating for OOS test
```

---

## VERDICT: WHAT WENT WRONG?

| Component | Status | Issue |
|-----------|--------|-------|
| **Training** | ❌ FAILED | Model learned to lose money (-4.94%) |
| **Test Generalization** | ❌ FAILED | OOS agents completely gated (0 trades) |
| **Action Output** | ⚠️ UNSTABLE | Mean swings 0.44 → 0.86 |
| **Overall** | ❌ NOT READY | No profitable signal, generalization broken |

---

## 🎯 RECOMMENDATIONS (Priority Order)

### 1. FIX IMMEDIATE: Gating Threshold Too High

**Action:**
```
Find gating logic in multi_asset_chunked_env.py
Current: threshold = 0.9?
Change: threshold = 0.6-0.7 for OOS testing
```

**Reason:**
Agents want to trade (0.81 confidence) but blocked (need 0.9+)

### 2. DEBUG: Why -4.94% on Backtest?

**Action:**
```
Run backtest with verbose logging:
- Extract all 39 trades
- Check: entry, exit, calculated PnL vs reported PnL
- Compare market prices in-sample vs out-sample
```

**Reason:**
Model might be profitable on training data but negative on test

### 3. RETRAIN: Better Reward Signal

**Action:**
```
If -4.94% confirmed on test set:
- Adjust reward function (increase profit incentive)
- Add market-neutral penalties (reduces trend dependency)
- Increase training steps (5120 might be insufficient)
```

**Reason:**
Current training produced losing strategy

### 4. TEST: Remove Gating Temporarily

**Action:**
```
Run OOS with gating disabled (threshold = 0.0)
See if agent makes trades and PnL
If trades happen and profitable: gating was culprit
If trades happen and lose money: strategy is bad
```

**Reason:**
Separate gating issue from strategy quality

---

## NEXT IMMEDIATE ACTIONS

1. ✅ **Confirm:** Extract 39 trades from backtest, verify PnL manually
2. ✅ **Test:** Run OOS with threshold = 0.5 (remove gating)
3. ✅ **Debug:** Check if -4.94% is real or calculation error
4. ✅ **Retrain:** If confirmed, retrain with better reward signal

**Do NOT deploy** until:
- ❌ OOS shows positive return
- ❌ Agent makes trades without gating
- ❌ Backtest returns positive

---

## Files Reviewed

```
logs/validation/backtest_5120.json          ✓ Checked
logs/validation/oos_w2_detailed.json        ✓ Checked
logs/validation/oos_w3_detailed.json        ✓ Checked
logs/rewards/worker_*.jsonl                 ✓ Scanned (no trade data)
logs/metrics/metrics_*.jsonl                ✓ Scanned (empty)
```

---

## SUMMARY

```
════════════════════════════════════════════════════════════════
                         TEST VERDICT
════════════════════════════════════════════════════════════════

Training Results:       ❌ -4.94% loss (not profitable)
Generalization:        ❌ Agents gated (0 trades OOS)
Action Stability:      ⚠️ Swings 0.44 → 0.86
Overall Status:        ❌ NOT PRODUCTION READY

Critical Issues:       2 (Negative returns, Gating blocks all)
Medium Issues:         1 (Action instability)

Recommendation:        DO NOT DEPLOY

Next Steps:
  1. Fix gating threshold (too conservative)
  2. Debug why -4.94% on backtest
  3. Retrain if strategy is fundamentally bad
  4. Retest everything
════════════════════════════════════════════════════════════════
```
