#!/usr/bin/env python3
"""
ADAN0 Diagnostic Script — Dir=+1.000 Saturation Analysis

This script:
1. Loads the PPO model checkpoint
2. Fetches ONE tick of live data
3. Builds observations through the EXACT same pipeline as run_bot.py
4. Dumps observation min/max/mean for every key
5. Runs ONE inference and prints raw action outputs
6. Tests if the model is fundamentally saturated by feeding random noise observations
"""
import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.chdir(str(Path(__file__).resolve().parent.parent))

print("=" * 70)
print("  ADAN0 SATURATION DIAGNOSTIC")
print("=" * 70)

# ── Step 1: Load model ──
print("\n[1/6] Loading model...")
from stable_baselines3 import PPO

CHECKPOINT = "checkpoints/ppo_adan0_sandbox_500224steps.zip"
if not os.path.isfile(CHECKPOINT):
    # Try other checkpoints
    for cp in sorted(Path("checkpoints").glob("*.zip")):
        CHECKPOINT = str(cp)
    print(f"  Using: {CHECKPOINT}")

model = PPO.load(CHECKPOINT, device="cpu")
print(f"  ✓ Model loaded")
print(f"  obs_space keys: {list(model.observation_space.spaces.keys())}")
for k, v in model.observation_space.spaces.items():
    print(f"    {k}: shape={v.shape}, dtype={v.dtype}, low={v.low.min():.2f}, high={v.high.max():.2f}")
print(f"  action_space: shape={model.action_space.shape}, low={model.action_space.low}, high={model.action_space.high}")

# Check if model uses SDE
policy = model.policy
print(f"  use_sde: {getattr(policy, 'use_sde', 'N/A')}")
print(f"  log_std_init: {getattr(policy, 'log_std_init', 'N/A')}")

# ── Step 2: Check VecNormalize ──
print("\n[2/6] Checking VecNormalize...")
vecnorm_path = CHECKPOINT.replace(".zip", "_vecnorm.pkl")
if os.path.isfile(vecnorm_path):
    print(f"  ✓ VecNormalize file found: {vecnorm_path}")
else:
    print(f"  ✗ NO VecNormalize file at: {vecnorm_path}")
    print(f"  ⚠ This means raw observations go directly to model")
    print(f"  ⚠ If model was trained WITH VecNormalize, this WILL cause saturation")

# ── Step 3: Build live observations ──
print("\n[3/6] Building live observations...")
try:
    from adan_trading_bot.trading.live_state_builder import LiveStateBuilder
    builder = LiveStateBuilder(exchange_id="binance", symbol="BTC/USDT")
    
    # Check if scalers were fitted
    sb = builder.state_builder
    has_scalers = hasattr(sb, 'scalers') and sb.scalers
    fitted = getattr(sb, 'scalers_loaded_from_training', False)
    print(f"  StateBuilder has scalers: {has_scalers}")
    print(f"  Scalers locked to training distribution: {fitted}")
    
    # Build observation
    portfolio_state = np.zeros(20, dtype=np.float32)
    portfolio_state[0] = 1.0  # equity ratio = 1 (full capital)
    portfolio_state[1] = 1.0  # cash ratio = 1
    
    obs = builder.build_observation(
        portfolio_state=portfolio_state,
        context_vector=None,
    )
    
    print("\n  ── OBSERVATION DUMP ──")
    for key, val in obs.items():
        arr = np.array(val)
        print(f"  {key:20s} | shape={str(arr.shape):12s} | "
              f"min={arr.min():12.4f} | max={arr.max():12.4f} | "
              f"mean={arr.mean():12.4f} | std={arr.std():12.4f} | "
              f"NaN={np.isnan(arr).sum()}")
        
        # Flag dangerous values
        if arr.max() > 100 or arr.min() < -100:
            print(f"    🚨 DANGER: Values outside [-100, 100] will saturate tanh!")
        if arr.max() > 10 or arr.min() < -10:
            print(f"    ⚠️ WARNING: Values outside [-10, 10] — likely needs normalization")
    
    live_obs_ok = True
except Exception as e:
    print(f"  ✗ Failed to build live observations: {e}")
    import traceback
    traceback.print_exc()
    live_obs_ok = False
    obs = None

# ── Step 4: Run inference on live obs ──
print("\n[4/6] Running inference on LIVE observations...")
if obs is not None:
    try:
        action, _states = model.predict(obs, deterministic=True)
        action = np.array(action).flatten()
        print(f"  Raw action:     {action}")
        print(f"  direction:      {action[0]:+.6f}")
        print(f"  size_pct raw:   {action[1]:+.6f}")
        print(f"  tf_pref:        {action[2]:+.6f}")
        print(f"  sl_pct raw:     {action[3]:+.6f}")
        print(f"  tp_pct raw:     {action[4]:+.6f}")
        
        if abs(action[0]) > 0.999:
            print(f"  🚨 CONFIRMED: Direction is saturated at {action[0]:+.6f}")
            print(f"     tanh(x) = {action[0]:.6f} means the pre-activation is > |7|")
            print(f"     This IS the saturation bug.")
        
        # Also test stochastic inference
        action_stoch, _ = model.predict(obs, deterministic=False)
        action_stoch = np.array(action_stoch).flatten()
        print(f"\n  Stochastic action: {action_stoch}")
        print(f"  direction (stoch): {action_stoch[0]:+.6f}")
        
        if abs(action_stoch[0]) > 0.999:
            print(f"  🚨 Even stochastic inference is saturated → network weights are dead")
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()

# ── Step 5: Test with RANDOM observations ──
print("\n[5/6] Testing inference on RANDOM normalized observations...")
try:
    # Create random obs that mimics what training would produce (mean≈0, std≈1)
    random_obs = {}
    for k, space in model.observation_space.spaces.items():
        # Random values in [-2, 2] like properly normalized data
        random_obs[k] = np.random.uniform(-2, 2, size=space.shape).astype(np.float32)
    
    actions_rand = []
    for i in range(10):
        random_obs_i = {}
        for k, space in model.observation_space.spaces.items():
            random_obs_i[k] = np.random.uniform(-2, 2, size=space.shape).astype(np.float32)
        a, _ = model.predict(random_obs_i, deterministic=True)
        a = np.array(a).flatten()
        actions_rand.append(a[0])  # direction
    
    print(f"  Directions on random obs: {[f'{d:+.4f}' for d in actions_rand]}")
    
    all_saturated = all(abs(d) > 0.99 for d in actions_rand)
    mean_dir = np.mean(actions_rand)
    std_dir = np.std(actions_rand)
    print(f"  Mean direction: {mean_dir:+.4f}, Std: {std_dir:.4f}")
    
    if all_saturated:
        print(f"  🚨 CRITICAL: Model is saturated even on RANDOM inputs")
        print(f"     → The model weights are fundamentally broken")
        print(f"     → Retraining is likely necessary")
    elif std_dir < 0.1:
        print(f"  ⚠️ Low variance on random inputs → model has weak discrimination")
    else:
        print(f"  ✓ Model shows variance on random inputs → saturation is caused by INPUT data")
        print(f"     → Fix: normalize the observations before feeding to model")
except Exception as e:
    print(f"  ✗ Random obs test failed: {e}")
    import traceback
    traceback.print_exc()

# ── Step 6: Compare with properly scaled observations ──
print("\n[6/6] Testing with manually normalized live observations...")
if obs is not None:
    try:
        # Force-normalize observations to [-2, 2] range
        normed_obs = {}
        for k, val in obs.items():
            arr = np.array(val)
            # Min-max normalize to [-1, 1]
            vmin, vmax = arr.min(), arr.max()
            if vmax - vmin > 1e-10:
                normed = 2.0 * (arr - vmin) / (vmax - vmin) - 1.0
            else:
                normed = np.zeros_like(arr)
            normed_obs[k] = normed.astype(np.float32)
        
        action_normed, _ = model.predict(normed_obs, deterministic=True)
        action_normed = np.array(action_normed).flatten()
        print(f"  Direction on force-normalized obs: {action_normed[0]:+.6f}")
        
        action_normed_s, _ = model.predict(normed_obs, deterministic=False)
        action_normed_s = np.array(action_normed_s).flatten()
        print(f"  Direction stochastic:             {action_normed_s[0]:+.6f}")
        
        if abs(action_normed[0]) < 0.99:
            print(f"  ✓ CONFIRMED: Normalization FIXES the saturation!")
            print(f"     → The problem is 100% observation scaling, NOT model weights")
        else:
            print(f"  ⚠️ Still saturated even with force-normalization")
            print(f"     → Model weights may need retraining")
    except Exception as e:
        print(f"  ✗ Normalized test failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("  DIAGNOSTIC COMPLETE")
print("=" * 70)
