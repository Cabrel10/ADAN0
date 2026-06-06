# SESSION 15: Quick Fix Checklist ✅

## The Problem
Agent losing money despite being right on trades:
- Trade: +0.09% price move (correct direction)
- Gross: +$0.02
- Fees: -$0.0341
- **Net: -$0.02** ❌

## Root Causes
1. **Stagnation penalty** was forcing agent into unprofitable micro-trades
2. **Min magnitude** (0.03) allowed trades too small to cover fees
3. **AGENT_CLOSE** had no minimum profit threshold

## Fixes Applied

### ✅ Fix #1: Reduce Stagnation Penalty (config/config.yaml)
```yaml
# Stagnation penalties reduced by 50% (beyond previous 4x reduction)
Micro:  -0.0005 → -0.00025
Small:  -0.00025 → -0.000125
Medium: -0.000125 → -0.0000625
High:   -0.00005 → -0.000025
```

**Impact:** Less pressure to panic-trade. Agent can WAIT for good setups.

### ✅ Fix #2: Increase Min Magnitude (config/config.yaml)
```yaml
# Only trade signals with sufficient confidence
5m:   0.03 → 0.06  (2x more confident)
1h:   0.05 → 0.08  (1.6x more confident)
4h:   0.08 → 0.12  (1.5x more confident)
```

**Impact:** Filters out weak noise. Only 0.06%+ moves are tradeable at 5m.

### ✅ Fix #3: Add Break-Even Protection (src/adan_trading_bot/environment/multi_asset_chunked_env.py)
```python
# Line ~7145: AGENT_CLOSE break-even check
unrealized_pnl_pct = (current_price - entry_price) / entry_price
if unrealized_pnl_pct < 0.0015:  # 0.15% minimum
    # Reject AGENT_CLOSE, keep position
    discrete_action = 0
```

**Impact:** AGENT_CLOSE won't exit with tiny profits that get eaten by fees.

---

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| Win Rate | 9% | 25-35% |
| Sharpe Ratio | -7.48 | +0.2 to +1.0 |
| Avg Trade | -$0.02 | +$0.01 to +$0.05 |
| Capital Change | -7.5% | +0% to +5% |
| Trades/1000 steps | ~200-300 | ~100-150 |

---

## How to Verify

After starting training, check logs for:

1. **Fewer trades with bigger moves:**
   ```
   [AGENT_CLOSE] BTCUSDT: ... pnl=+0.0532 | WAIT  # Good ✅
   [AGENT_CLOSE] BTCUSDT: ... pnl=-0.0015 | WAIT  # Blocked ✅
   ```

2. **Better win rate at step 5000:**
   ```
   Worker 0 | Step 5000 | WinRate 28% | Sharpe +0.35
   ```

3. **Reduced panic-trading:**
   - Fewer [STAGNATION] warnings
   - More HOLD actions than before

---

## Files Changed

1. **config/config.yaml** - Updated min_magnitude and stagnation penalties
2. **src/adan_trading_bot/environment/multi_asset_chunked_env.py** - Added break-even check to AGENT_CLOSE
3. **SESSION_15_LOSS_ANALYSIS_AND_FIX.md** - Full analysis

---

## Ready to Train?

✅ All fixes applied and syntax-checked.
✅ Configuration updated.
✅ Code protection logic added.

**Command to start:**
```bash
python scripts/train_parallel_agents.py --config config/config.yaml
```

**Watch for improvement within first 2000 steps!**

