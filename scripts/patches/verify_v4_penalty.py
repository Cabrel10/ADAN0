#!/usr/bin/env python3
"""
DIAGNOSTIC-V4 verification: prove the symmetric sterile penalty FIRES.

The 300-step smoke test only logs ACTION_DIFF every 50 steps and an untrained
gSDE-off policy mostly requests HOLD, so it rarely samples a BUY-illegal step.
This script instead drives the env DIRECTLY:

  1. Build the real MultiAssetChunkedEnv (w1 scalper, test split).
  2. Force BUY (action0 = +1) on EVERY step for ~400 steps.
  3. After each step, read env._step_invalid_penalty and the rejection counters.
  4. Assert that when min_notional / anti_spam_hold rejections occur, the
     accumulated penalty is NON-ZERO (was 0.0 before V4).

Fact-based: we measure the actual attribute, not a sampled log line.
"""
import os, sys
import numpy as np

os.environ.setdefault("ADAN_USE_SDE", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader

cfg = ConfigLoader.load_config(os.path.join(REPO, "config", "config.yaml"))
workers = cfg.get("workers", {})
w1 = dict(workers.get("w1", {}))
w1.setdefault("data_split_override", "test")
w1.setdefault("timeframes", cfg.get("data", {}).get("timeframes", ["5m", "1h", "4h"]))

# Load data exactly like the sandbox harness.
loader = ChunkedDataLoader(config=cfg, worker_config=w1, worker_id=0)
data = loader.load_chunk(0)

env = MultiAssetChunkedEnv(
    data=data,
    config=cfg,
    worker_config=w1,
    worker_id=0,
    live_mode=False,
)
obs, _ = env.reset()

# Action space: continuous Box. action0 = direction (+1 BUY / -1 SELL).
act_dim = env.action_space.shape[0]
buy_action = np.zeros(act_dim, dtype=np.float32)
buy_action[0] = 1.0   # force BUY every step
if act_dim > 1:
    buy_action[1] = 1.0  # max size

penalty_fired = 0
penalty_total = 0.0
min_notional_seen = 0
anti_spam_seen = 0
prev_rej = dict(env.rejection_reasons)

N = 400
for i in range(N):
    obs, reward, term, trunc, info = env.step(buy_action)
    pen = float(getattr(env, "_step_invalid_penalty", 0.0))
    rej = env.rejection_reasons
    mn = rej.get("min_notional", 0)
    asp = rej.get("anti_spam_hold", 0)
    # detect a NEW BUY-illegal rejection this step
    new_mn = mn - prev_rej.get("min_notional", 0)
    new_asp = asp - prev_rej.get("anti_spam_hold", 0)
    if (new_mn > 0 or new_asp > 0):
        if pen < 0:
            penalty_fired += 1
            penalty_total += pen
    min_notional_seen = mn
    anti_spam_seen = asp
    prev_rej = dict(rej)
    if term or trunc:
        obs, _ = env.reset()

print("=" * 60)
print("DIAGNOSTIC-V4 PENALTY VERIFICATION")
print("=" * 60)
print(f"steps driven (forced BUY)   : {N}")
print(f"min_notional rejections     : {min_notional_seen}")
print(f"anti_spam_hold rejections   : {anti_spam_seen}")
print(f"steps where BUY-illegal had NON-ZERO penalty : {penalty_fired}")
print(f"total penalty accumulated   : {penalty_total:.5f}")
print("-" * 60)
if penalty_fired > 0:
    print("RESULT: PASS  — BUY-illegal now costs reward (gradient can see it).")
    sys.exit(0)
else:
    if min_notional_seen == 0 and anti_spam_seen == 0:
        print("RESULT: INCONCLUSIVE — no BUY-illegal rejection occurred in window.")
        sys.exit(2)
    print("RESULT: FAIL  — BUY-illegal rejections occurred but penalty stayed 0.")
    sys.exit(1)
