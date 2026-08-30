#!/usr/bin/env python3
"""Empirical 500-step financial-stability check on the real BTC environment.

The policy is uniformly random over the environment's Box(5) action space. A
positive trade reward is the positive final environment reward emitted on the
same step as a receipt-backed profitable close. The capacity component is
telemetry-only in the current optimized reward and is reported as such.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("ADAN_TRAINING_SILENT", "1")
os.environ.setdefault("ADAN_RICH_STEP_EVERY", "999999")
logging.disable(logging.WARNING)

DEFAULT_OUTPUT = REPO_ROOT / "logs" / "validation" / "financial_stability_check.json"
EXACT_STEPS = 500


def classify_ratio(ratio: float | None) -> str:
    """Apply the requested decision thresholds without manufacturing a PASS."""
    if ratio is None:
        return "INCONCLUSIVE_NO_POSITIVE_TRADES"
    if ratio < 0.1:
        return "PASS_LAUNCH_500K"
    if ratio > 0.3:
        return "FAIL_REDUCE_CAPACITY_COEFFICIENT"
    return "INTERMEDIATE_REDUCE_AND_REVALIDATE"


def calculate_ratio(
    capacity_rewards: list[float], positive_trade_rewards: list[float]
) -> tuple[float, float | None, float | None]:
    """Return |capacity| mean, positive close-reward mean, and their ratio."""
    capacity_mean_abs = float(np.mean(np.abs(capacity_rewards)))
    if not positive_trade_rewards:
        return capacity_mean_abs, None, None
    trade_mean_positive = float(np.mean(positive_trade_rewards))
    if trade_mean_positive <= 0.0:
        return capacity_mean_abs, trade_mean_positive, None
    return capacity_mean_abs, trade_mean_positive, capacity_mean_abs / trade_mean_positive


def _receipt_pnl(receipt: dict[str, Any]) -> float:
    for key in ("pnl_net", "pnl", "realized_pnl", "net_pnl"):
        value = receipt.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
    return 0.0


def _open_count(env: Any) -> int:
    positions = getattr(getattr(env, "portfolio_manager", None), "positions", {})
    return sum(bool(getattr(position, "is_open", False)) for position in positions.values())


def build_environment(split: str, seed: int):
    from adan_trading_bot.common.config_loader import ConfigLoader
    from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
    from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

    cfg = ConfigLoader.load_config(str(REPO_ROOT / "config" / "config.yaml"))
    cfg.setdefault("environment", {})["rich_display_interval"] = 999999
    wc = copy.deepcopy(cfg.get("workers", {}).get("w1", {}))
    wc.update(
        {
            "worker_id": 0,
            "data_split": split,
            "data_split_override": split,
            "timeframes": ["5m", "1h", "4h"],
            "assets": ["BTCUSDT"],
        }
    )
    data = ChunkedDataLoader(config=cfg, worker_config=wc, worker_id=0).load_chunk(0)
    env = MultiAssetChunkedEnv(
        data=data,
        config=cfg,
        worker_config=wc,
        worker_id=0,
        live_mode=False,
    )
    env.action_space.seed(seed)
    env.reset(seed=seed)
    return env


def run_check(*, steps: int, split: str, seed: int) -> dict[str, Any]:
    if steps != EXACT_STEPS:
        raise ValueError(f"financial stability protocol requires exactly {EXACT_STEPS} steps")

    np.random.seed(seed)
    env = build_environment(split, seed)
    capacity_rewards: list[float] = []
    positive_trade_rewards: list[float] = []
    profitable_closes_with_nonpositive_reward = 0
    close_events: list[dict[str, Any]] = []
    openings = 0
    closings = 0
    winning_closings = 0
    losing_closings = 0
    episodes = 0
    previous_open_count = _open_count(env)

    for step_index in range(1, steps + 1):
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        reward = float(reward)
        components = dict(getattr(env, "_last_reward_components", {}) or {})
        capacity_rewards.append(float(components.get("capacity_reward", 0.0) or 0.0))

        current_open_count = _open_count(env)
        receipts = [
            item
            for item in list(getattr(env, "_step_closed_receipts", []) or [])
            if isinstance(item, dict)
        ]
        closings += len(receipts)
        openings += max(0, current_open_count - previous_open_count + len(receipts))
        previous_open_count = current_open_count

        for receipt in receipts:
            pnl = _receipt_pnl(receipt)
            is_win = pnl > 0.0
            if is_win:
                winning_closings += 1
                if reward > 0.0:
                    positive_trade_rewards.append(reward)
                else:
                    profitable_closes_with_nonpositive_reward += 1
            else:
                losing_closings += 1
            close_events.append(
                {
                    "step": step_index,
                    "pnl": pnl,
                    "reward": reward,
                    "positive_trade_reward_eligible": bool(is_win and reward > 0.0),
                    "reason": receipt.get("reason", receipt.get("close_reason")),
                    "reward_components": {
                        key: float(components.get(key, 0.0) or 0.0)
                        for key in (
                            "pnl_reward",
                            "closure_bonus",
                            "drawdown_penalty",
                            "behavior_penalty",
                            "future_contrib",
                            "symmetry_penalty",
                            "action_entropy_penalty",
                            "saturation_penalty",
                            "raw",
                            "final_reward",
                            "capacity_reward",
                        )
                    },
                }
            )

        if terminated or truncated:
            episodes += 1
            env.reset(seed=seed + episodes)
            env.action_space.seed(seed + episodes)
            previous_open_count = _open_count(env)

    capacity_mean_abs, trade_mean_positive, ratio = calculate_ratio(
        capacity_rewards, positive_trade_rewards
    )
    verdict = classify_ratio(ratio)
    return {
        "protocol": "real_environment_uniform_random_policy",
        "asset": "BTCUSDT",
        "split": split,
        "seed": seed,
        "steps": steps,
        "capacity_coefficient": 0.1,
        "capacity_reward_in_optimized_raw_reward": False,
        "metric_definition": {
            "capacity_reward_mean_abs": "mean(abs(step telemetry capacity_reward))",
            "trade_reward_mean_positive": (
                "mean(final step reward > 0 on receipt-backed close with pnl > 0)"
            ),
        },
        "counts": {
            "openings": openings,
            "closings": closings,
            "winning_closings": winning_closings,
            "losing_closings": losing_closings,
            "positive_trade_rewards": len(positive_trade_rewards),
            "profitable_closes_with_nonpositive_reward": (
                profitable_closes_with_nonpositive_reward
            ),
            "capacity_positive_steps": sum(value > 0.0 for value in capacity_rewards),
            "capacity_negative_steps": sum(value < 0.0 for value in capacity_rewards),
            "capacity_zero_steps": sum(value == 0.0 for value in capacity_rewards),
            "episodes_or_resets": episodes,
        },
        "capacity_reward_mean_abs": capacity_mean_abs,
        "trade_reward_mean_positive": trade_mean_positive,
        "ratio": ratio,
        "thresholds": {"pass_below": 0.1, "fail_above": 0.3},
        "verdict": verdict,
        "close_events": close_events,
    }


def write_report_atomic(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=EXACT_STEPS)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--seed", type=int, default=330500)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = run_check(steps=args.steps, split=args.split, seed=args.seed)
    output = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    write_report_atomic(report, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verdict"] == "PASS_LAUNCH_500K" else 2


if __name__ == "__main__":
    raise SystemExit(main())
