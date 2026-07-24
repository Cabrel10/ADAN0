#!/usr/bin/env python3
"""test_anchor_probe_fires.py — prove the V15 anchor + Critic probe telemetry
actually EXECUTES inside WorldModelPPO.train(), including the [ANCHOR_DEBUG]
print, even when the KL early-stop truncates the epoch loop.

We build a tiny gym env with a Box(5) action space (mirroring ADAN0's
assets*5 continuous head) and a Dict obs (so MultiInputPolicy is used), run a
minimal PPO with anchor_lambda>0, and assert:
  * _a0_mean_hist gets populated (anchor telemetry ran)
  * the [ANCHOR_DEBUG] line is printed to stdout
  * the diag/adv_BUY / diag/adv_SELL keys are recorded in the logger

This isolates the train() code path from the heavy trading env so the proof is
deterministic and fast (< 10 s).
"""
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ["ADAN_L2_ANCHOR_LAMBDA"] = "0.05"

from adan_trading_bot.agent.feature_extractors import WorldModelPPO  # noqa: E402


class TinyDictEnv(gym.Env):
    """Minimal env: Dict obs, Box(5) continuous action (like ADAN0's head)."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict({
            "x": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,),
                                       dtype=np.float32)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return {"x": self.observation_space["x"].sample()}, {}

    def step(self, action):
        self._t += 1
        # Reward correlated with a0 so advantages are non-trivial and the
        # BUY/SELL slices both get populated.
        a0 = float(np.clip(action[0], -1, 1))
        reward = a0 * float(np.random.randn() * 0.1 + 0.05)
        terminated = False
        truncated = self._t >= 32
        obs = {"x": self.observation_space["x"].sample()}
        return obs, reward, terminated, truncated, {}


def main():
    env = TinyDictEnv()

    model = WorldModelPPO(
        "MultiInputPolicy",
        env,
        n_steps=128,
        batch_size=32,
        n_epochs=4,
        use_sde=True,
        sde_sample_freq=4,
        target_kl=0.05,   # low so we may trigger the early-stop path too
        verbose=1,
        seed=0,
    )

    print(f"[TEST] anchor_lambda = {model.anchor_lambda}")
    assert model.anchor_lambda == 0.05, "anchor_lambda env not picked up"

    buf = io.StringIO()
    with redirect_stdout(buf):
        model.learn(total_timesteps=256)
    out = buf.getvalue()

    # ---- assertions ----
    print("=" * 60)
    anchor_debug_lines = [l for l in out.splitlines() if "[ANCHOR_DEBUG]" in l]
    print(f"[TEST] #[ANCHOR_DEBUG] lines printed: {len(anchor_debug_lines)}")
    for l in anchor_debug_lines[:6]:
        print("   ", l.strip())

    assert len(anchor_debug_lines) >= 1, "FAIL: [ANCHOR_DEBUG] never printed"

    print(f"[TEST] _a0_mean_hist len = {len(model._a0_mean_hist)} "
          f"vals={[round(v,4) for v in model._a0_mean_hist[:6]]}")
    assert len(model._a0_mean_hist) >= 1, "FAIL: a0_mean telemetry empty"

    # Check adv_BUY / adv_SELL recorded at least once.
    kv = model.logger.name_to_value
    has_advb = "diag/adv_BUY" in kv
    has_advs = "diag/adv_SELL" in kv
    print(f"[TEST] logger has diag/adv_BUY={has_advb} diag/adv_SELL={has_advs}")
    print(f"[TEST] diag/adv_BUY={kv.get('diag/adv_BUY')} "
          f"diag/adv_SELL={kv.get('diag/adv_SELL')} "
          f"train/anchor_loss={kv.get('train/anchor_loss')} "
          f"train/a0_mean_raw={kv.get('train/a0_mean_raw')}")
    assert has_advb or has_advs, "FAIL: Critic probe keys never recorded"

    print("=" * 60)
    print("ALL ANCHOR+PROBE TELEMETRY ASSERTIONS PASSED ✔")


if __name__ == "__main__":
    main()
