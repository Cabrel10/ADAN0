# Training Readiness Checklist - S15+ Reward Reactivation

## ✅ Completion Status: READY FOR TRAINING

Generated: 2026-06-05 15:50 UTC

---

## Critical Fixes Implemented

### 1. ✅ Capacity Reward Reactivation
- **File**: `src/adan_trading_bot/environment/multi_asset_chunked_env.py`
- **Lines**: 6047-6060 (reward calculation)
- **Config**: `config/config.yaml` line 1244
- **Status**: COMPLETE
- **Details**:
  - Lightweight +0.1 bonus when portfolio is 60-90% invested
  - Prevents reward starvation that froze trading at $14.33 drawdown
  - Not strong enough to cause reward hacking (vs original +2.0)
  - Calculation: `capacity_pct = _current_capacity_pct` (set in step() at line 4475)

### 2. ✅ Frequency Reward Reactivation  
- **File**: `src/adan_trading_bot/environment/multi_asset_chunked_env.py`
- **Lines**: 6062-6074 (reward calculation)
- **Config**: `config/config.yaml` line 1248
- **Status**: COMPLETE
- **Details**:
  - Light +0.05 bonus per executed trade this step
  - Encourages exploration without overwhelming PnL signal
  - Calculated as: `frequency_reward = _freq_weight * trades_executed_this_step`
  - Trade count set in step() at line 4482

### 3. ✅ Ray Checkpoint Restore Fix
- **File**: `scripts/train_parallel_agents.py`
- **Lines**: 1098-1103
- **Status**: COMPLETE
- **Details**:
  - Fixed hardcoded path: `experiment_state-2026-06-04_16-06-16.json`
  - Now uses glob pattern to find most recent: `experiment_state-*.json`
  - Allows session resumption without crashes
  - Supports both sandbox and PBT directory structures

### 4. ✅ PBT Checkpoint Loading in Backtest
- **File**: `scripts/deterministic_backtest.py`
- **Lines**: 39-65
- **Status**: COMPLETE
- **Details**:
  - Added `--ckpt-dir` parameter for PBT structure support
  - Added `--worker-idx` parameter to load specific worker checkpoints
  - Extracts `worker_idx` from checkpoint path: `worker_idx=(\d+)`
  - Loads correct profile per worker (w1, w2, etc.)

### 5. ✅ Ray Environment Configuration
- **File**: `scripts/train_parallel_agents.py`
- **Lines**: 150-170 (env vars)
- **Status**: COMPLETE
- **Details**:
  - `RAY_gcs_rpc_server_reconnect_timeout_s`: 600s → 1200s (prevents GCS crashes)
  - `RAY_memory` limit configured
  - `RAY_task_retry_delay_ms` set to prevent cascading failures

---

## Configuration Values

### Reward Shaping Weights
```yaml
reward_shaping:
  capacity_weight: 0.1      # S15+ lightweight reactivation
  frequency_weight: 0.05    # S15+ lightweight reactivation  
  time_decay: -0.01         # Increased from -0.001 (forces exploration)
```

### Why These Values?
- **capacity_weight = 0.1**: 
  - S15 used +2.0 (too strong, biases toward holding)
  - Proposed: 0.1 (light motivation for being invested)
  - Alternative tested: 0.3 (still light but more motivating)
  
- **frequency_weight = 0.05**:
  - Per trade executed (not per timeframe)
  - Over 500-step episode: max +2.5 if trading every step
  - Symlog(+2.5) = 1.44 (still << realized trade profit signal)
  - Prevents dominant effect on behavior

---

## Files Modified

1. ✅ `src/adan_trading_bot/environment/multi_asset_chunked_env.py`
   - Lines 4475-4482: Pre-calculate capacity_pct and trades_executed_this_step
   - Lines 6047-6074: Add capacity + frequency reward logic
   
2. ✅ `config/config.yaml`
   - Lines 1244, 1248: Add capacity_weight and frequency_weight

3. ✅ `scripts/train_parallel_agents.py`
   - Lines 1098-1103: Fix checkpoint restore with glob pattern
   - Lines 150-170: Configure Ray environment variables

4. ✅ `scripts/deterministic_backtest.py`
   - Lines 39-65: Add PBT checkpoint discovery with worker_idx support

5. ✅ `scripts/launch_training.sh`
   - Already configured for light mode (2 workers) and heavy mode (4 workers)

---

## Verification Tests

### Code Quality
- ✅ Python syntax: `python -m py_compile` passes
- ✅ Imports: All modules load without errors
- ✅ Config parsing: capacity_weight=0.1, frequency_weight=0.05 loaded

### Logic Verification
- ✅ Capacity calculation: `_current_capacity_pct` set in step()
- ✅ Frequency tracking: `_trades_executed_this_step` set in step()
- ✅ Reward composition: Both rewards added to raw signal before symlog
- ✅ Fallback handling: Both use `.getattr()` with defaults to prevent crashes

### Training Infrastructure
- ✅ Checkpoint restore: Uses glob pattern, supports both directory structures
- ✅ Ray configuration: Timeouts increased, memory limits set
- ✅ PBT backtest: Worker-specific checkpoint loading implemented

---

## Expected Training Behavior

### Session Indicators (First 30 Minutes)
1. **Ray Session**: Should use existing session (log: "Restoring from...") not create new
2. **Portfolio**: Should NOT be frozen at $14.33, should show trading activity
3. **Capacity Reward**: Will print "[REWARD Worker X] ... capacity_reward=+0.1" ~60-90% of steps
4. **Frequency Bonus**: Will print "+0.05" bonus when trades executed
5. **No GCS Crashes**: System should run > 30 min without Ray failures

### Key Metrics to Monitor
- `env_total_trades` > 0 (not frozen)
- `mean_sharpe` increasing (model learning)
- `realized_pnl` positive trend (profitable trades)
- Portfolio value > $14.33 (above drawdown)

---

## Next Steps (User Action)

### To Launch Training
```bash
# Light mode (2 workers, reliable for testing)
bash scripts/launch_training.sh --light --steps 500000 --resume

# Monitor logs
tail -f /mnt/new_data/adan_logs/checkpoints/training_*.log

# Expected output:
# [REWARD Worker 0] Base: 0.050000, ... Total: 0.035421
# [REWARD_ANTIHACK] Step 1234 | pnl_net=+0.050 action_req=BUY action_exe=BUY raw=0.085 final=+0.0823
```

### To Test Specific Worker Checkpoint
```bash
python scripts/deterministic_backtest.py --ckpt-dir /path/to/ray_results --worker 0
```

---

## Rollback Plan (If Issues Arise)

### If portfolio frozen again:
1. Check if capacity_reward being applied: grep "capacity_reward" in logs
2. Increase capacity_weight to 0.3 in config.yaml line 1244
3. Verify _current_capacity_pct being calculated: check step() line 4475

### If Ray crashes at 20-25 min:
1. Already fixed: RAY_gcs_rpc_server_reconnect_timeout_s: 1200s
2. Try increasing to 1800s if still crashing
3. Check: `RAY_memory` env var in train script

### If backtest loads wrong checkpoint:
1. Verify checkpoint path contains `worker_idx=N` 
2. Test with explicit --worker flag: `--worker 0`
3. Check glob pattern in deterministic_backtest.py line 50

---

## Session History Reference

- **S15 Hard Reset (2026-06-03 10:38)**: Disabled capacity + frequency rewards (too strong)
- **S15 also fixed**: 8 critical bugs (VecNormalize gamma, grid_search, gSDE, etc.)
- **S15+ Solution**: Reactivate rewards with controlled light weights (0.1, 0.05)
- **Expected outcome**: Balanced exploration (rewards) + exploitation (PnL signal)

---

## Validation Checklist

Run before launching production training:

```bash
# 1. Syntax check
python -m py_compile src/adan_trading_bot/environment/multi_asset_chunked_env.py

# 2. Config validation
python -c "from adan_trading_bot.common.config_loader import ConfigLoader; cfg = ConfigLoader.load_config('config/config.yaml'); print(cfg['reward_shaping'])"

# 3. Import check
python -c "from scripts.train_parallel_agents import main; print('✅ Training script ready')"

# 4. Checkpoint restore pattern
grep "experiment_state-\*.json" scripts/train_parallel_agents.py

# 5. PBT backtest support
grep "worker_idx" scripts/deterministic_backtest.py
```

All checks should show ✅ PASS before launching training.
