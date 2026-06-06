# SESSION 15: Ray PBT Configuration Verification

## Current Ray Configuration (from scripts/train_parallel_agents.py)

### 🎯 PBT Scheduler Settings

```python
PopulationBasedTraining(
    time_attr="training_iteration",
    perturbation_interval=2,          # Mutate every 2 iterations
    metric="mean_reward",              # Optimize for reward
    mode="max",                        # Maximize reward
    hyperparam_mutations={...}
)
```

**What this means:**
- Every 2 training iterations, Ray evaluates workers
- Top performers get copied to worst performers
- Hyperparameters are mutated using the ranges below

---

### 📊 Hyperparameter Ranges for PBT Evolution

#### PPO Hyperparameters (Ray optimizes these):
```yaml
learning_rate: loguniform(1e-5, 1e-3)      # 0.00001 to 0.001
ent_coef:      uniform(0.0, 0.05)          # 0.0 to 0.05
gamma:         uniform(0.95, 0.999)        # 0.95 to 0.999
```

#### Trading Hyperparameters (Ray optimizes these):
```yaml
sl_pct: uniform(0.01, 0.08)      # Stop-Loss: 1% to 8%
tp_pct: uniform(0.02, 0.15)      # Take-Profit: 2% to 15%
```

**SESSION 15 NOTE:** These ranges are good! They give Ray room to discover profitable strategies with:
- **Wider stops** (8% SL) compatible with reduced stagnation penalty
- **Flexible TP** (2-15%) to adapt to market conditions

---

### 🔄 Trial Management

```python
num_samples=1           # Set to 1 because grid_search defines trial count
max_concurrent_trials=2 # Only 2 workers concurrent (reduced for stability)
reuse_actors=True       # Reuse actors between trials (prevents GCS crashes)
```

**What this means:**
- Only 2 workers run at once (avoids resource exhaustion)
- Actors are reused instead of destroyed (more stable)
- Workers are selected via `grid_search` (deterministic per-profile assignment)

---

### 📋 Trial Resources

```python
# On Colab with GPU:
cpu_per_trial = (available_cpus - 1) / num_samples
gpu_per_trial = 1.0 / num_samples
```

**For your setup with 8 CPUs, 2 concurrent trials:**
- CPU per trial: (8-1)/2 = 3.5 CPUs
- GPU per trial: 0.5 GPU each

---

### 💾 Checkpointing Strategy

```python
CheckpointConfig(
    num_to_keep=3,                        # Keep only 3 checkpoints
    checkpoint_score_attribute="timesteps_total",
    checkpoint_score_order="max",         # Keep the latest ones
)
```

**What this means:**
- Every checkpoint is scored by total timesteps
- Keep the 3 most advanced checkpoints (not best reward, just latest)
- Saves space while allowing recovery

---

### ⏸️ Resume Logic

```python
if resume:
    # Find most recent experiment_state-*.json
    exp_states = sorted(glob(storage_path / "experiment_state-*.json"))
    exp_state_file = exp_states[-1]  # Use newest
    tuner = tune.Tuner.restore(storage_path, trainable=ADAN_PBT_Worker)
```

**What this means:**
- Automatically finds and resumes from latest experiment state
- Handles multiple resume attempts gracefully
- Falls back to fresh start if no valid state found

---

### 🛑 Failure Handling

```python
failure_config=tune.FailureConfig(max_failures=3)  # Retry 3 times before giving up
```

**What this means:**
- If a trial crashes, it's restarted up to 3 times
- After 3 failures, it's abandoned
- Other trials continue running

---

## Performance Analysis: Is This Configuration Optimal?

### ✅ Good Decisions

1. **`reuse_actors=True`**: Prevents Ray GCS (Global Control Store) crashes
   - Critical for long-running training
   - Reduces memory churn

2. **`max_concurrent_trials=2`**: Conservative and stable
   - Reduces resource contention
   - Prevents OOM (Out of Memory) errors
   - Allows meaningful PBT evolution with fewer workers

3. **`perturbation_interval=2`**: Good balance
   - Not too aggressive (every iteration)
   - Not too conservative (every 10 iterations)
   - Allows workers to stabilize before copying

4. **Trading hyperparameter ranges** are compatible with SESSION 15 fixes:
   - `sl_pct` goes up to 8% (works with reduced stagnation penalty)
   - `tp_pct` up to 15% (room for swing trades)

### ⚠️ Areas to Monitor

1. **`num_cpus=8`**: Check if actual available CPUs ≥ 8
   - If less, Ray will oversubscribe
   - Look for "CPU allocation exceeded" warnings

2. **`checkpoint_score_attribute="timesteps_total"`**: Keeps latest, not best
   - Good for recovery, but might abandon good early checkpoints
   - Consider adding a secondary `best_checkpoint_dir` logic

3. **`metric="mean_reward"`**: Is this metric being properly logged?
   - SESSION 15 bug fix ensured metrics collection works
   - If workers crash during collection, PBT can't evolve properly

---

## What Ray Does Every 2 Iterations

### The PBT Evolution Cycle:

**Step 1: Evaluate**
```
Worker 0: reward=+0.05, sl=2%, tp=4%
Worker 1: reward=-0.10, sl=3%, tp=5%
```

**Step 2: Copy & Mutate** (Worker 1 copies from Worker 0 + perturbation)
```
Worker 1: reward=-0.10 → +0.05 (copy)
Worker 1: sl=2% → 2.2%, tp=4% → 4.3% (mutate by ±10%)
```

**Step 3: Continue Training**
```
Both workers train for another 2 iterations with new hyperparams
```

---

## Verification Checklist ✓

- [x] PBT scheduler configured
- [x] Perturbation interval set to 2 (reasonable)
- [x] Metric set to "mean_reward" (must be logged)
- [x] Trading hyperparams in reasonable ranges (1-8% SL, 2-15% TP)
- [x] Trial resources balanced
- [x] Checkpoint strategy defined
- [x] Resume logic in place
- [x] Failure handling configured
- [x] reuse_actors=True (critical fix)

---

## Recommendation for SESSION 15

**No changes needed to Ray configuration.** It's well-configured for:

1. Stability (2 concurrent trials, reuse_actors=True)
2. Evolution (perturbation_interval=2, proper hyperparameter ranges)
3. Recovery (3 checkpoints kept, resume logic)

**What matters now:**
1. ✅ Metrics collection working (just fixed)
2. ✅ Min magnitude filtering working (config updated)
3. ✅ AGENT_CLOSE break-even check working (code updated)
4. ✅ Stagnation penalty reduced (config updated)

Ray will automatically evolve SL/TP and PPO hyperparams as training progresses. Just let it run and monitor the wins!

---

## Monitoring Commands

Watch these metrics as training progresses:

```bash
# Watch mean_reward increase over time
tail -f logs/central/adan_20260606.log | grep "mean_reward"

# Watch best_trial metrics
tail -f logs/central/adan_20260606.log | grep "best_trial"

# Check for PBT copying events
tail -f logs/central/adan_20260606.log | grep "Copying"

# Monitor perturbation mutations
tail -f logs/central/adan_20260606.log | grep "perturb"
```

