#!/usr/bin/env python3
"""ACM Capability Vector calibration sampler.

Question this answers (the real residual risk after the 20->28 migration):
  Are the 8 ACM features [20-27] real SIGNAL or just NOISE for the PPO?

It builds the SAME env as training/backtest, steps it, and records the RAW
(pre-VecNormalize) portfolio_state[20:28] at every step. It then prints a
calibration table (mean/std/min/max/P1/P99/zero_ratio/nan_count) per feature,
PLUS a per-regime split (FLAT vs IN-POSITION) to expose non-stationarity.

Verdict heuristics per feature:
  DEAD       zero_ratio > 0.95                      -> almost no information
  SATURATED  std < 0.02 (near-constant)             -> network can't learn
  EXPLOSIVE  std > 5.0 or |max| > 10                 -> dominates the loss
  NON-STAT   |mean_flat - mean_open| > 0.5 AND both regimes non-trivial
  OK         otherwise

Usage:
  python scripts/diagnostics/acm_calibration.py --steps 3000 [--split val] \
      [--policy checkpoints/ppo_adan0_sandbox_512steps.zip]
"""
from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Silence the very verbose per-step env/portfolio INFO logs: they slow the
# sampler to a crawl (~10 lines/step) and add nothing to the calibration.
# We only want the final calibration table, so force WARNING globally and
# pin the chatty modules to ERROR. Set ADAN_CAL_VERBOSE=1 to opt back in.
if os.environ.get("ADAN_CAL_VERBOSE", "0") != "1":
    logging.disable(logging.INFO)
    for _name in (
        "adan_trading_bot.environment.multi_asset_chunked_env",
        "adan_trading_bot.environment.dynamic_behavior_engine",
        "adan_trading_bot.portfolio.portfolio_manager",
        "adan_trading_bot.environment.reward_calculator",
    ):
        logging.getLogger(_name).setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

ACM_NAMES = [
    "can_open", "can_close", "free_slots_ratio", "cash_ratio_for_trade",
    "risk_budget_remaining", "max_size_remaining", "cooldown_active",
    "capital_self_caused",
]


def _verdict(mean, std, zero_ratio, mx, mean_flat, mean_open, regimes_ok):
    if zero_ratio > 0.95:
        return "DEAD"
    if std < 0.02:
        return "SATURATED"
    if std > 5.0 or abs(mx) > 10.0:
        return "EXPLOSIVE"
    if regimes_ok and abs(mean_flat - mean_open) > 0.5:
        return "NON-STAT"
    return "OK"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--policy", type=str, default=None,
                    help="Optional 28d checkpoint to drive actions; else random.")
    ap.add_argument("--csv", type=str, default=None,
                    help="Optional path to dump the raw per-step ACM samples.")
    args = ap.parse_args()

    from adan_trading_bot.common.config_loader import ConfigLoader
    from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
    from adan_trading_bot.environment.multi_asset_chunked_env import (
        MultiAssetChunkedEnv,
    )

    cfg = ConfigLoader.load_config(str(REPO_ROOT / "config" / "config.yaml"))
    cfg.setdefault("environment", {})["rich_display_interval"] = 999999

    wc = copy.deepcopy(cfg.get("workers", {}).get("w1", {}))
    wc.update({
        "worker_id": 0,
        "data_split_override": args.split,
        "timeframes": ["5m", "1h", "4h"],
        "assets": ["BTCUSDT"],
    })

    data = ChunkedDataLoader(config=cfg, worker_config=wc, worker_id=0).load_chunk(0)
    env = MultiAssetChunkedEnv(
        data=data, config=cfg, worker_config=wc, worker_id=0, live_mode=False
    )

    # Optional policy to drive realistic actions (else random exploration).
    model = None
    if args.policy and os.path.isfile(args.policy):
        from stable_baselines3 import PPO
        model = PPO.load(args.policy, device="cpu")
        print(f"[ACM-CAL] Driving actions with policy {args.policy}")
    else:
        print("[ACM-CAL] Driving actions RANDOMLY (no policy)")

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    samples = []          # raw ACM rows
    regime_flat = []      # rows while no position open (open_positions_norm==0)
    regime_open = []      # rows while a position is open

    def _acm(o):
        ps = np.asarray(o["portfolio_state"], dtype=np.float64)
        return ps[20:28], ps  # acm block, full vector

    for step in range(args.steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=False)
        else:
            action = env.action_space.sample()
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, _r, term, trunc, _info = step_out
            done = bool(term) or bool(trunc)
        else:
            obs, _r, done, _info = step_out
        acm, full = _acm(obs)
        samples.append(acm)
        # regime by open_positions_norm at slot [6]
        if float(full[6]) <= 1e-9:
            regime_flat.append(acm)
        else:
            regime_open.append(acm)
        if done:
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    arr = np.array(samples)  # (N, 8)
    flat = np.array(regime_flat) if regime_flat else np.zeros((0, 8))
    openp = np.array(regime_open) if regime_open else np.zeros((0, 8))
    n = len(arr)
    print(f"\n[ACM-CAL] collected {n} steps "
          f"(flat={len(flat)}, in_position={len(openp)})\n")

    header = (f"{'idx':>3} {'feature':<22} {'mean':>8} {'std':>8} "
              f"{'min':>7} {'max':>7} {'P1':>7} {'P99':>7} {'zero%':>6} "
              f"{'nan':>4} {'m_flat':>8} {'m_open':>8} {'verdict':>10}")
    print(header)
    print("-" * len(header))

    table = []
    for i, name in enumerate(ACM_NAMES):
        col = arr[:, i]
        nan_count = int(np.isnan(col).sum())
        colc = col[~np.isnan(col)] if nan_count else col
        mean = float(np.mean(colc)) if len(colc) else float("nan")
        std = float(np.std(colc)) if len(colc) else float("nan")
        mn = float(np.min(colc)) if len(colc) else float("nan")
        mx = float(np.max(colc)) if len(colc) else float("nan")
        p1 = float(np.percentile(colc, 1)) if len(colc) else float("nan")
        p99 = float(np.percentile(colc, 99)) if len(colc) else float("nan")
        zero_ratio = float(np.mean(np.abs(colc) < 1e-9)) if len(colc) else 1.0
        m_flat = float(np.mean(flat[:, i])) if len(flat) else float("nan")
        m_open = float(np.mean(openp[:, i])) if len(openp) else float("nan")
        regimes_ok = len(flat) > 20 and len(openp) > 20
        v = _verdict(mean, std, zero_ratio, mx, m_flat, m_open, regimes_ok)
        print(f"{20+i:>3} {name:<22} {mean:>8.4f} {std:>8.4f} {mn:>7.3f} "
              f"{mx:>7.3f} {p1:>7.3f} {p99:>7.3f} {zero_ratio*100:>5.1f}% "
              f"{nan_count:>4} {m_flat:>8.4f} {m_open:>8.4f} {v:>10}")
        table.append({"idx": 20 + i, "feature": name, "mean": mean, "std": std,
                      "min": mn, "max": mx, "p1": p1, "p99": p99,
                      "zero_ratio": zero_ratio, "nan": nan_count,
                      "mean_flat": m_flat, "mean_open": m_open, "verdict": v})

    print("\n[ACM-CAL] verdict summary:")
    from collections import Counter
    c = Counter(t["verdict"] for t in table)
    for k, val in c.items():
        print(f"  {k}: {val}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
        print(f"\n[ACM-CAL] table written to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
