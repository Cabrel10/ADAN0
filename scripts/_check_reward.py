#!/usr/bin/env python3
import sys
sys.path.insert(0, "src")
import inspect
import adan_trading_bot.environment.multi_asset_chunked_env as m

src = inspect.getsource(m.MultiAssetChunkedEnv._calculate_reward)
lines = src.split("\n")
print(f"_calculate_reward: {len(lines)} lines")

dead = ["frequency_reward = self", "np.tanh(pnl", "np.clip(total", "multi_bonus_threshold"]
for k in dead:
    found = any(k in l for l in lines)
    print(f"  {'DEAD CODE STILL PRESENT' if found else 'CLEAN'}: {k}")

new = ["symlog", "pnl_net", "drawdown_penalty", "trade_cost"]
for k in new:
    found = any(k in l for l in lines)
    print(f"  {'OK' if found else 'MISSING'}: {k}")

# Quick functional test
import copy, numpy as np
from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from stable_baselines3.common.vec_env import DummyVecEnv

cfg = ConfigLoader.load_config("config/config.yaml")
wc = copy.deepcopy(cfg["workers"]["w1"])
wc["worker_id"] = 0
loader = ChunkedDataLoader(config=cfg, worker_config=wc, worker_id=0)
data = loader.load_chunk(0)

def make():
    return MultiAssetChunkedEnv(data=data, config=cfg, worker_config=wc, worker_id=0, live_mode=False)

env = DummyVecEnv([make])
obs = env.reset()

print("\n=== REWARD ALIGNMENT TEST ===")
rewards_seen = []
for _ in range(200):
    action = env.action_space.sample()
    obs, reward, done, info = env.step([action])
    r = float(reward[0])
    rewards_seen.append(r)
    if done[0]:
        obs = env.reset()

print(f"  Rewards over 200 steps: min={min(rewards_seen):.4f} max={max(rewards_seen):.4f} mean={sum(rewards_seen)/len(rewards_seen):.4f}")
print(f"  Max reward: {max(rewards_seen):.4f} (should be << 100, not 342)")
print(f"  Reward hacking: {'ELIMINATED' if max(rewards_seen) < 5 else 'STILL PRESENT'}")
env.close()
