#!/usr/bin/env python3
"""
Extract the best model from PBT results based on mean_reward or balance.
Run after training completes.
"""
import glob
import shutil
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAY_DIR = PROJECT_ROOT / "logs" / "ray_results" / "adan_pbt_training"
OUT_DIR = PROJECT_ROOT / "models" / "rl_agents" / "best"
OUT_DIR.mkdir(parents=True, exist_ok=True)

best = {"reward": -1e9, "balance": 0, "path": None, "worker": None, "iter": 0}

for csv_path in RAY_DIR.glob("**/progress.csv"):
    trial_dir = csv_path.parent
    try:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        # Take best row by reward
        for row in rows:
            reward = float(row.get("mean_reward", 0) or 0)
            balance = float(row.get("mean_balance", 0) or 0)
            iteration = int(row.get("training_iteration", 0) or 0)
            if reward > best["reward"]:
                best.update({"reward": reward, "balance": balance,
                             "path": trial_dir, "iter": iteration,
                             "worker": trial_dir.name})
    except Exception as e:
        print(f"Skip {csv_path}: {e}")

if best["path"] is None:
    print("No results found.")
    exit(1)

print(f"\nBest trial: {best['worker']}")
print(f"  reward={best['reward']:.2f}, balance={best['balance']:.3f}, iter={best['iter']}")

# Find latest checkpoint in best trial
checkpoints = sorted(best["path"].glob("checkpoint_*"))
if not checkpoints:
    print("No checkpoints found in best trial.")
    exit(1)

latest_ckpt = checkpoints[-1]
print(f"  checkpoint: {latest_ckpt.name}")

# Copy model.zip + vecnormalize.pkl
for fname in ["model.zip", "vecnormalize.pkl", "worker_state.json"]:
    src = latest_ckpt / fname
    if src.exists():
        dst = OUT_DIR / fname
        shutil.copy2(src, dst)
        print(f"  Copied {fname} -> {dst}")

# Save metadata
meta = OUT_DIR / "training_info.txt"
with open(meta, "w") as f:
    f.write(f"Best trial: {best['worker']}\n")
    f.write(f"Mean reward: {best['reward']:.4f}\n")
    f.write(f"Mean balance: {best['balance']:.4f}\n")
    f.write(f"Training iteration: {best['iter']}\n")
    f.write(f"Checkpoint: {latest_ckpt}\n")
    f.write(f"\nUsage:\n")
    f.write(f"  from stable_baselines3 import PPO\n")
    f.write(f"  from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv\n")
    f.write(f"  model = PPO.load('models/rl_agents/best/model.zip')\n")
    f.write(f"  # For inference with normalization:\n")
    f.write(f"  env = DummyVecEnv([make_env])\n")
    f.write(f"  env = VecNormalize.load('models/rl_agents/best/vecnormalize.pkl', env)\n")
    f.write(f"  env.training = False\n")
    f.write(f"  env.norm_reward = False\n")

print(f"\nDone. Best model saved to {OUT_DIR}")
print(f"Load with: PPO.load('{OUT_DIR}/model.zip')")
