# ADAN0 — Research Synthesis & Documented Decisions

**Date**: 2026-06-16  
**Status**: Research complete, decisions documented BEFORE implementation

---

## 1. Problem Statement (Observed Facts Only)

1. **124 consecutive ticks** with `Dir=-1.000` (always SELL at maximum confidence)
2. **Size=20.0%** on every tick (from old default `--max-position-pct 20.0`)
3. Checkpoint used: `ppo_adan0_sandbox_500224steps.zip` (the worst saturated one)
4. **No VecNormalize .pkl** exists in any checkpoint directory
5. **No `--action-threshold` or `--max-position-pct`** was passed explicitly
6. Equity unchanged ($20.50) because Dir=-1.000 means SELL but no position open

## 2. Research Sources Consulted

### Source 1: SB3 Official Documentation — VecNormalize
**URL**: `stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html`  
**Key findings**:
- VecNormalize uses **RunningMeanStd** (moving average) to normalize observations
- Default `clip_obs=10.0` — clips normalized output to [-10, 10]
- **Must be saved separately** from model: `vec_env.save(stats_path)`
- **Must be loaded at inference**: `VecNormalize.load(stats_path, venv)`
- At inference: set `training=False` (freeze stats) and `norm_reward=False`
- For Dict observation spaces: `norm_obs_keys` controls which keys are normalized
- `normalize_obs()` formula: `clip((obs - mean) / sqrt(var + eps), -clip_obs, clip_obs)`

### Source 2: SB3 VecNormalize Source Code
**URL**: `github.com/DLR-RM/stable-baselines3/blob/master/.../vec_normalize.py`  
**Key findings**:
- `__getstate__` explicitly **excludes `venv`** from pickle — the wrapper is serialized independently
- `save()` uses `pickle.dump(self, ...)` — saves running mean/var stats
- `load()` uses `pickle.load()` then `set_venv()` — restores stats on new env
- For Dict obs spaces, it maintains **separate RunningMeanStd per key**
- The `_normalize_obs()` method: `clip((obs - obs_rms.mean) / sqrt(obs_rms.var + epsilon), -clip_obs, clip_obs)`

### Source 3: SB3 Examples — PyBullet VecNormalize Pattern
**URL**: `stable-baselines3.readthedocs.io/en/master/guide/examples.html`  
**Canonical save/load pattern**:
```python
# TRAINING:
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
model = PPO("MlpPolicy", vec_env)
model.learn(total_timesteps=2000)
model.save("ppo_halfcheetah")
stats_path = "vec_normalize.pkl"
vec_env.save(stats_path)  # <-- SEPARATE SAVE

# INFERENCE:
vec_env = make_vec_env("...", n_envs=1)
vec_env = VecNormalize.load(stats_path, vec_env)  # <-- LOAD STATS
vec_env.training = False    # <-- FREEZE STATS
vec_env.norm_reward = False  # <-- NO REWARD NORM AT INFERENCE
model = PPO.load("ppo_halfcheetah", env=vec_env)
```

### Source 4: SB3 RL Tips & Tricks
**URL**: `stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html`  
**Key findings**:
- **"always normalize your observation space if you can"** — explicitly recommended
- **"normalize your action space and make it symmetric"** — [-1, 1] recommended
- **"Most RL algorithms rely on a Gaussian distribution (initially centered at 0 with std 1)"** — explains why unnormalized obs cause saturation
- For continuous actions: PPO relies on Gaussian → tanh squashing maps to [-1,1]
- **"Normalization is critical for [continuous action] algorithms"** — their bold emphasis

### Source 5: SB3 GitHub Issue #1018 — Action Saturation
**URL**: `github.com/hill-a/stable-baselines/issues/1018`  
**Key quote**: 
> "Telling the agent that the actions are in [-1, 1], without using Tanh, simply not works because this leads to saturating the actions (-1 or 1) from the first step evaluation"

**Interpretation**: Tanh is used BY DESIGN to bound actions, but if pre-activation values are extreme (due to unnormalized inputs), tanh saturates to ±1 permanently.

### Source 6: SB3 GitHub Issue #1592 — Training vs Inference Unscaling
**URL**: `github.com/DLR-RM/stable-baselines3/issues/1592`  
**Key finding**: With `squash_output=True` (used for gSDE), `unscale_action` behavior differs between `collect_rollouts()` (training) and `predict()` (inference). This means even identical weights can produce different actions train vs deploy.

### Source 7: SB3 GitHub Issue #2101 — VecNormalize.load() Inconsistency
**URL**: `github.com/DLR-RM/stable-baselines3/issues/2101`  
**Key finding**: `VecNormalize.load()` can create nested wrappers where inner/outer `.training` attributes mismatch. Must explicitly set `vec_env.training = False` AFTER load.

### Source 8: SB3 GitHub Issue #698 — VecNormalize Usage Questions
**URL**: `github.com/hill-a/stable-baselines/issues/698`  
**Key finding**: When VecNormalize is used, observation_space bounds should reflect normalized range (e.g., [-10,10]), not raw data range. The scaler converts raw → normalized, then clips.

---

## 3. Root Cause Analysis (Evidence-Based)

### Problem 1: "Why Dir=-1.000 on every tick?"

**Root cause: Observation distribution mismatch → tanh saturation**

Evidence chain:
1. ADAN0 training used **StateBuilder with per-timeframe scalers** (MinMax for 5m, Standard for 1h, Robust for 4h) — confirmed in `state_builder.py`
2. **No VecNormalize was used during training** — confirmed by: (a) no `.pkl` files in checkpoints dir, (b) bot log says "No VecNormalize found"
3. Live inference calls `fit_on_parquet()` which re-fits scalers on validation data — **but validation data covers different price range** than training data
4. Result: obs values enter the network at magnitudes the weights never saw
5. Diagnostic script confirmed: all 11 checkpoints saturate on real-magnitude inputs (min 6/20 at ckpt 100k, max 17/20 at ckpt 500k)
6. **This is textbook covariate shift**, exactly matching SB3's documentation warning

### Problem 2: "Is the sizing too aggressive by config, or does it prove the model is broken?"

**Answer: Both, but they are independent issues**

**Sizing by config** (not a model bug):
- Default `--max-position-pct` was 20.0 → `Size=20.0%` is exactly the configured default
- No CLI override was passed: `cat /proc/3155791/cmdline` shows no `--max-position-pct` argument
- `config.yaml` has contradictory values: Micro tier says 90%, hard_constraints says 0.5%
- The running bot used the CLI default (20%), not the config.yaml values
- **Verdict**: The 20% sizing is "aggressive by config" — it's the default, not a model malfunction

**Model broken** (separate issue):
- Dir=-1.000 on 124 consecutive ticks = model always outputs max-SELL
- With BTC oscillating between $66,388 and $66,658 (0.4% range), a working model should show varied signals
- The saturation diagnostic proves the model weights are permanently stuck
- **Verdict**: The model IS broken for deployment — but the sizing percentage is a configuration issue

### Problem 3: "Can checkpoint 100k be saved?"

**Evidence**:
- Random input test: 6/20 saturated (best of all checkpoints)
- Scaled random input (*0.1): 0/20 saturated, std=0.32 (actual variance!)
- BUT with Parquet-fitted scalers + real data: still outputs Dir=-1.000
- The scaler mismatch causes obs magnitudes that push even the 100k weights past tanh saturation

**Conclusion from research**: 
- SB3 docs are unambiguous: **normalization stats MUST match between train and inference**
- Since no VecNormalize pkl was saved during training, we cannot reconstruct the exact normalization
- The custom StateBuilder scalers are a substitute for VecNormalize, but they were **never serialized alongside checkpoints**
- **Verdict**: Checkpoint 100k shows life but cannot be deployed without reconstructing exact training-time normalization

---

## 4. Documented Decisions

### Decision 1: RETRAIN (do NOT attempt deployment with existing checkpoints)

**Rationale** (from documentation research):
1. SB3 official docs: "Don't forget to save the VecNormalize statistics when saving the agent" — we didn't
2. SB3 RL tips: "Normalization is critical for [continuous action] algorithms" — confirmed by our diagnostic
3. Without saved normalization statistics, deploying any checkpoint means guaranteed covariate shift
4. Even checkpoint 100k, which shows life at small magnitudes, fails on real data
5. Attempting to "guess" the right normalization would be fragile and unverifiable

**For next training run, MUST implement**:
1. Either use VecNormalize (with `clip_obs=10.0`) and save `.pkl` with every checkpoint
2. OR serialize StateBuilder scalers alongside every checkpoint `.zip`
3. Add a callback that saves both model + normalization stats atomically
4. Validate deployment with a held-out period BEFORE paper trading

### Decision 2: KEEP safety guards even after retraining

**Rationale**:
1. The config.yaml inconsistency (90% tier vs 0.5% hard constraint) exists independently of model quality
2. Mechanical safety cap (10% in execution_engine.py) prevents catastrophic losses from any future model issue
3. Observation clipping [-5,5] is consistent with SB3's VecNormalize `clip_obs=10.0` approach (we use a tighter bound)
4. Saturation alarm provides early warning if a retrained model also starts saturating

### Decision 3: Config.yaml needs harmonization BEFORE retraining

**The contradiction**:
```yaml
capital_tiers[0].max_position_size_pct: 90      # Says 90%
environment.hard_constraints.max_position_size_pct: 0.5  # Says 0.5%
CLI default --max-position-pct: 5.0 (after our fix, was 20.0)
execution_engine MECHANICAL_MAX: 10%
```

**Resolution**: 
- For Micro Capital ($11-30): max_position_size_pct should be 5-10% (not 90%)
- hard_constraints.max_position_size_pct 0.5% is likely a typo (should be 50% or removed)
- The mechanical 10% cap in execution_engine is the real safety net
- These must be resolved in config before retraining

---

## 5. Session Evidence Captured

### Final session snapshot (saved to `docs/FINAL_SESSION_SNAPSHOT.txt`):
- **Process**: PID 3155791, checkpoint 500k, no CLI overrides
- **3 session JSONs**: 
  - Session 1: 24 ticks, 0 trades, $20.50 equity, 0.0% return
  - Session 2: 1 trade (BUY $4.10 at $66,369), -0.024% return, $20.4951 equity
  - Session 3: 0 trades, $20.50 equity
- **All ticks**: Dir=-1.000, Size=20.0%, consistent saturation
- **Process killed**: SIGKILL after SIGTERM (graceful shutdown didn't work within 3s)

---

## 6. Action Items (Prioritized)

| Priority | Action | Depends On |
|----------|--------|-----------|
| P0 | Harmonize config.yaml position sizes | Nothing |
| P0 | Add scaler serialization to training pipeline | Nothing |
| P1 | Retrain with VecNormalize OR saved StateBuilder scalers | P0 |
| P1 | Add checkpoint validation callback (save normalization + test inference) | P0 |
| P2 | Paper-trade retrained model on held-out period first | P1 |
| P2 | Add obs distribution monitoring to live bot | P1 |
| KEEP | Safety guards (clip [-5,5], max 5%, mechanical 10%) | Already done |
| KEEP | Saturation alarm in predict_action_async | Already done |
