"""FINDING #4 — Future Arena RewardBridge end-to-end wiring test.

Proves:
  1. The bridge builds from `reward_shaping.future_reward` and is ACTIVE.
  2. compute_mfe_mae returns sane ratios on a synthetic chunk.
  3. A REALISTIC TP (near the future MFE) is rewarded MORE than a UTOPIAN TP
     (10%, never touched) — i.e. the model is taught to choose capturable TP/SL.
  4. The reward breakdown exposes eqs / sl_q / tp_q.

Run: python tests/test_future_arena_bridge.py
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adan_trading_bot.future_arena.reward_bridge import RewardBridge
from adan_trading_bot.future_arena.future_zones import (
    compute_mfe_mae,
    PivotDirection,
    ZoneConfig,
)


def _make_chunk(n=80, mfe_pct=0.025, mae_pct=0.008, entry=100.0):
    """Synthetic 5m chunk: from entry the price rises ~mfe_pct then dips ~mae_pct."""
    highs, lows, closes, opens = [], [], [], []
    price = entry
    for i in range(n):
        # gentle drift up then a small dip, bounded by mfe/mae
        drift = mfe_pct * np.sin(i / n * np.pi)        # peaks mid-chunk
        dip = mae_pct * np.sin(i / n * np.pi * 0.5)
        c = entry * (1 + drift)
        h = c * (1 + 0.0008)
        lo = entry * (1 - dip) * (1 - 0.0008)
        opens.append(price)
        highs.append(h)
        lows.append(lo)
        closes.append(c)
        price = c
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})
    return df


def main():
    ok = True

    # ── 1. Bridge builds + active ─────────────────────────────────────────────
    cfg = {
        "reward_shaping": {
            "future_reward": {
                "enabled": True,
                "mode": "future_guided",
                "round_trip_fees": 0.005,
                "max_future_contrib": 0.60,
            }
        }
    }
    bridge = RewardBridge.from_config(cfg, seed=0)
    assert not bridge.is_noop, "bridge should be ACTIVE"
    print(f"[1] Bridge ACTIVE: mode={bridge.config.mode.value} rtf={bridge.config.round_trip_fees}")

    # ── 2. MFE/MAE sane ───────────────────────────────────────────────────────
    df = _make_chunk()
    zcfg = ZoneConfig()
    mfe, mae = compute_mfe_mae(df, 0, PivotDirection.LOW, zcfg.horizon, mae_floor=zcfg.mae_floor)
    print(f"[2] MFE={mfe:.4f} ({mfe*100:.2f}%)  MAE={mae:.4f} ({mae*100:.2f}%)")
    assert 0.005 < mfe < 0.05, f"MFE out of expected range: {mfe}"
    assert mae >= zcfg.mae_floor, "MAE floor not applied"

    # ── 3. realistic TP rewarded MORE than utopian TP ─────────────────────────
    common = dict(
        profile="intraday", timeframe="5m", closed=True, pnl_gross=0.5,
        steps_held=10, close_reason="TP", direction=1.0, size=0.3,
        sl_chosen=0.008, mfe=mfe, mae=mae,
    )
    # realistic TP ~ at the achievable MFE
    realistic = bridge.contribution(tp_chosen=round(mfe, 4), **common)
    bd_real = bridge.last_breakdown().as_dict()
    # utopian TP = 10% (never touched)
    utopian = bridge.contribution(tp_chosen=0.10, **common)
    bd_uto = bridge.last_breakdown().as_dict()

    print(f"[3] realistic TP={mfe*100:.2f}% -> contrib={realistic:+.4f}  tp_q={bd_real['tp_q']:.3f}")
    print(f"    utopian   TP=10.00% -> contrib={utopian:+.4f}  tp_q={bd_uto['tp_q']:.3f}")
    assert realistic > utopian, "realistic TP must score higher than utopian TP"
    assert bd_real["tp_q"] > bd_uto["tp_q"], "tp_q must be higher for realistic TP"

    # ── 4. breakdown exposes eqs/sl_q/tp_q ────────────────────────────────────
    for k in ("eqs", "sl_q", "tp_q", "future_contrib"):
        assert k in bd_real, f"missing breakdown key {k}"
    print(f"[4] breakdown keys present: eqs={bd_real['eqs']:.3f} sl_q={bd_real['sl_q']:.3f} "
          f"tp_q={bd_real['tp_q']:.3f} future_contrib={bd_real['future_contrib']:+.4f}")

    # ── 5. contribution is capped ─────────────────────────────────────────────
    assert abs(realistic) <= 0.60 + 1e-6, "contribution exceeds cap"
    print(f"[5] contribution within cap: |{realistic:.4f}| <= 0.60")
    print(f"    bridge stats: n_calls={bridge.n_calls} n_active={bridge.n_active}")
    assert bridge.n_active > 0, "bridge never activated"

    print("\nALL FUTURE-ARENA BRIDGE CHECKS PASSED ✅")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
