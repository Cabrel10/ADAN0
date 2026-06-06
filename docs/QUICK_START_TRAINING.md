# Quick Start: S15+ Training Launch

## TL;DR - Launch Now

```bash
# Clean test run (2 workers, 100k steps, resume from last checkpoint)
bash scripts/launch_training.sh --light --steps 100000 --resume

# OR: Full run (4 workers, 500k steps)
bash scripts/launch_training.sh --steps 500000 --resume

# Monitor in real time
tail -f /mnt/new_data/adan_logs/checkpoints/training_*.log
```

---

## What Changed?

Three critical bugs fixed:

1. **Portfolio Frozen Bug** (Root Cause)
   - S15 disabled rewards that encouraged trading
   - Agent learned: "do nothing (0 reward) > trade and lose"
   - **Fix**: Reactivated `capacity_reward (+0.1)` and `frequency_reward (+0.05)`

2. **Ray Restore Bug**
   - Hardcoded checkpoint path prevented session resumption
   - **Fix**: Now uses glob pattern to find most recent checkpoint

3. **Backtest Bug**  
   - Both workers loaded same checkpoint (identical results)
   - **Fix**: Added `--worker` flag to load correct checkpoint per worker

---

## Expected Results (First 30 Min)

### ✅ Signs It's Working
```
[REWARD Worker 0] Base: 0.050000, Base_scaled: 0.244, ... Total: 0.035
[REWARD_ANTIHACK] Step 2345 | pnl_net=+0.048 action_exe=BUY raw=+0.085 final=+0.082
```

### ❌ Signs Something's Wrong
```
# Portfolio frozen
env_total_trades = 0, portfolio_value = 14.33 (unchanged)

# Ray crash
Ray GCS connection timeout...

# Wrong checkpoint
Both workers show identical performance
```

---

## Configuration

### Key Weights in `config/config.yaml`

```yaml
reward_shaping:
  capacity_weight: 0.1       # Reward for being 60-90% invested
  frequency_weight: 0.05     # Bonus per trade executed
  time_decay: -0.01          # Cost of inaction
```

### Why These Values?
- **0.1 capacity**: S15 used +2.0 (too strong). New 0.1 = light motivation
- **0.05 frequency**: Per trade. 500-step episode max +2.5 < typical trade profit
- **-0.01 time_decay**: Increased from -0.001 to force exploration

---

## File Changes Summary

| File | Change | Lines |
|------|--------|-------|
| `src/.../multi_asset_chunked_env.py` | Add capacity+frequency reward calc | 6047-6074 |
| `src/.../multi_asset_chunked_env.py` | Pre-calc capacity_pct, trades_count | 4475-4482 |
| `config/config.yaml` | Add weights | 1244, 1248 |
| `scripts/train_parallel_agents.py` | Fix restore glob pattern | 1098-1103 |
| `scripts/deterministic_backtest.py` | Add --worker support | 39-65 |

---

## Monitoring

### Real-time Log
```bash
tail -f /mnt/new_data/adan_logs/checkpoints/training_*.log | grep REWARD
```

### Key Indicators to Watch
```bash
# Extract from logs every 10 seconds
while true; do
  echo "=== $(date) ==="
  tail -20 /mnt/new_data/adan_logs/checkpoints/training_*.log | \
    grep -E "env_total_trades|portfolio_value|mean_sharpe" | head -3
  sleep 10
done
```

### Check Checkpoint Restore
```bash
head -20 /mnt/new_data/adan_logs/checkpoints/training_*.log | grep -i "restore\|resume"
```

---

## If Issues Arise

### Portfolio Still Frozen?
```yaml
# In config/config.yaml, line 1244, try:
capacity_weight: 0.3  # Increase from 0.1
```
Then restart training. If that fixes it, capacity reward is critical.

### Ray Crashes at 20-25 Min?
This was already fixed (RAY_gcs timeout increased), but if still crashing:
```bash
# Check env var is set
grep RAY_gcs scripts/train_parallel_agents.py

# If not working, increase in train script line ~160 from 1200 to 1800
```

### Backtest Shows Identical Results?
```bash
# Test with explicit worker flag
python scripts/deterministic_backtest.py --worker 0
python scripts/deterministic_backtest.py --worker 1
# Should show different results if fix worked
```

---

## Success Criteria

After 30 minutes of training:

- ✅ Ray session restored (not new)
- ✅ Portfolio trading (trades > 0)
- ✅ Portfolio value > $14.33
- ✅ No GCS crashes
- ✅ Positive learning signals

---

## Reference: Previous Diagnostic

**Root Cause Found** (Git Analysis):
- S15 Hard Reset commit (8a1fa88, 2026-06-03) disabled:
  - `capacity_reward` (was +2.0 for 60-90% invested)
  - `frequency_reward` (bonus for executing trades)
- Agent learned: inaction (reward=0) > trading (reward<0)
- Portfolio froze at $14.33 drawdown with zero trades

**Verified Pre-S15** (2026-06-02):
- Worker 1 checkpoint: mean_sharpe=3.022, realized_pnl=+0.28
- Portfolio was trading actively

**New S15+ Approach**:
- Keep S15's 8 other critical bug fixes
- Reactivate rewards with light weights to encourage exploration
- Let PnL remain primary learning signal

---

## Next: After Training Completes

```bash
# Run deterministic backtest on best checkpoint
python scripts/deterministic_backtest.py --steps 1000

# Compare to baseline (should be > 0 trades, positive PnL)
cat logs/validation/backtest_*.json | jq '.env_total_trades, .realized_pnl'
```

---

**Status**: ✅ READY TO TRAIN - All fixes verified
