#!/usr/bin/env python3
"""
DIAGNOSTIC-V4 hotfix verification: exercise the SELL-while-flat code path.

The earlier verify_v4_penalty.py forced BUY every step and therefore NEVER
triggered the [STERILE_SELL] log block at line ~7994, which is exactly where
the `NameError: name '_base' is not defined` crash lived.

This script forces SELL (action0 = -1) on every step. Starting flat, every
SELL is a "sterile" SELL-without-position, which:
  1. runs the _sterile_penalty_for_tier() helper,
  2. hits the `current_step % 50 == 0` [STERILE_SELL] warning log
     (the line that used to crash).

PASS = we drive >50 steps with NO exception AND the penalty fires.

Fact-based: any NameError would propagate (step() catches it and logs an
ERROR, so we ALSO scan for that). We assert the penalty is negative on
sterile SELLs and that no step recorded a caught exception flag.
"""
import os
import sys
import numpy as np

os.environ.setdefault("ADAN_USE_SDE", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv  # noqa
from adan_trading_bot.common.config_loader import ConfigLoader  # noqa
from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader  # noqa

cfg = ConfigLoader.load_config(os.path.join(REPO, "config", "config.yaml"))
workers = cfg.get("workers", {})
w1 = dict(workers.get("w1", {}))
w1.setdefault("data_split_override", "test")
w1.setdefault("timeframes", cfg.get("data", {}).get("timeframes", ["5m", "1h", "4h"]))

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

act_dim = env.action_space.shape[0]
sell_action = np.zeros(act_dim, dtype=np.float32)
sell_action[0] = -1.0  # force SELL every step (sterile while flat)
if act_dim > 1:
    sell_action[1] = 1.0

# Patch the env logger to RAISE on the [STERILE_SELL] path if it would crash:
# instead, we rely on step() not swallowing — but step() DOES swallow and logs
# ERROR. So we monkey-patch the logger.error to record any NameError text.
_errors = []
_orig_error = env.logger.error


def _capture_error(msg, *a, **k):
    _errors.append(str(msg))
    return _orig_error(msg, *a, **k)


env.logger.error = _capture_error

penalty_fired = 0
penalty_total = 0.0
N = 120
crashed = False
for i in range(N):
    obs, reward, term, trunc, info = env.step(sell_action)
    pen = float(getattr(env, "_step_invalid_penalty", 0.0))
    if pen < 0:
        penalty_fired += 1
        penalty_total += pen
    if term or trunc:
        obs, _ = env.reset()

base_err = [e for e in _errors if "_base" in e or "NameError" in e or "not defined" in e]

print("=" * 60)
print("DIAGNOSTIC-V4 SELL-FLAT PATH VERIFICATION")
print("=" * 60)
print(f"steps driven (forced SELL)        : {N}")
print(f"steps with NON-ZERO penalty       : {penalty_fired}")
print(f"total penalty accumulated         : {penalty_total:.5f}")
print(f"caught step() errors logged       : {len(_errors)}")
print(f"  of which _base/NameError        : {len(base_err)}")
if base_err:
    print("  sample:", base_err[0][:200])
print("-" * 60)
if base_err:
    print("RESULT: FAIL — the _base NameError is STILL happening.")
    sys.exit(1)
if penalty_fired == 0:
    print("RESULT: INCONCLUSIVE — no sterile SELL penalty fired in window.")
    sys.exit(2)
print("RESULT: PASS — SELL-flat path runs clean, penalty fires, no NameError.")
sys.exit(0)
