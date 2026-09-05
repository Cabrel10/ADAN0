#!/usr/bin/env python3
"""Strict pre-training financial gate on the real BTC environment.

A/C/D are measured on the first exactly 500 uniformly random policy steps.
B is measured on exactly 1000 steps from the same uninterrupted trajectory.
E is derived from the active configuration and BTC launcher's frozen TP domain.
No missing observation or inconclusive metric is converted into a PASS.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("ADAN_TRAINING_SILENT", "1")
os.environ.setdefault("ADAN_RICH_STEP_EVERY", "999999")
logging.disable(logging.WARNING)

DEFAULT_OUTPUT = REPO_ROOT / "logs" / "validation" / "financial_stability_check.json"
FINANCIAL_STEPS = 500
ACTION_DIFF_STEPS = 1000
CAPACITY_RATIO_LIMIT = 0.10
ACTION_DIFF_LIMIT = 0.05
HOLD_LIMIT = 0.80
REWARD_STD_MIN = 0.01
FEES_TO_TP_LIMIT = 0.30

# ADAN0_GATE_ASSET_SOURCE_FIX: the assets scripts/launch_asset_run.py L57 allows.
# A gate that measures anything else is not measuring the training universe.
LAUNCHER_ASSETS = ("BTCUSDT_BINANCE", "DOGEUSDT_BINANCE")
DEFAULT_ASSET = "BTCUSDT_BINANCE"


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _receipt_pnl(receipt: dict[str, Any]) -> float:
    for key in ("pnl_net", "pnl", "realized_pnl", "net_pnl"):
        value = receipt.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
    return 0.0


def _open_count(env: Any) -> int:
    positions = getattr(getattr(env, "portfolio_manager", None), "positions", {})
    return sum(bool(getattr(position, "is_open", False)) for position in positions.values())


def _load_btc_financial_contract() -> dict[str, float]:
    """Read fees from config and TP bounds from the actual BTC launcher."""
    config = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text()) or {}
    future_cfg = ((config.get("reward_shaping", {}) or {}).get("future_reward", {}) or {})
    trading_cfg = config.get("trading_rules", {}) or {}
    round_trip_fees = _finite_float(future_cfg.get("round_trip_fees"), -1.0)
    commission_per_side = _finite_float(trading_cfg.get("commission_pct"), -1.0)

    launcher_tree = ast.parse((REPO_ROOT / "scripts" / "launch_asset_run.py").read_text())
    sltp: dict[str, Any] | None = None
    for node in ast.walk(launcher_tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "_SLTP" for target in node.targets):
            candidate = ast.literal_eval(node.value)
            if isinstance(candidate, dict):
                sltp = candidate
                break
    if not sltp or "BTCUSDT" not in sltp:
        raise RuntimeError("BTCUSDT SL/TP contract not found in launch_asset_run.py")

    btc = sltp["BTCUSDT"]
    tp_low = _finite_float(btc.get("tp_lo"), -1.0)
    tp_high = _finite_float(btc.get("tp_hi"), -1.0)
    sl_high = _finite_float(btc.get("sl_hi"), -1.0)
    if min(round_trip_fees, commission_per_side, tp_low, tp_high, sl_high) < 0.0:
        raise RuntimeError("invalid negative or missing financial contract value")
    if tp_high < tp_low:
        raise RuntimeError("BTC TP upper bound is below lower bound")
    return {
        "commission_per_side": commission_per_side,
        "round_trip_fees": round_trip_fees,
        "tp_low": tp_low,
        "tp_high": tp_high,
        "sl_high": sl_high,
        "mean_tp": (tp_low + tp_high) / 2.0,
    }


def _apply_btc_launcher_runtime(contract: dict[str, float]) -> dict[str, str]:
    """Mirror the launcher's execution-relevant BTC invariants for the gate."""
    requested = {
        "ADAN_FREE_SLTP": "1",
        "ADAN_TP_LO": str(contract["tp_low"]),
        "ADAN_TP_HI": str(contract["tp_high"]),
        "ADAN_SL_HI": str(contract["sl_high"]),
    }
    for key, value in requested.items():
        os.environ.setdefault(key, value)
    return {key: os.environ[key] for key in requested}


def build_environment(split: str, seed: int, asset: str = DEFAULT_ASSET):
    from adan_trading_bot.common.config_loader import ConfigLoader
    from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
    from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

    # ADAN0_GATE_ASSET_SOURCE_FIX: this function used to hardcode
    # assets=["BTCUSDT"]. data_loader.py L256-273 resolves
    # data_dirs[split]/<ASSET>/<tf>.parquet, and data/processed/indicators/train
    # holds BOTH "BTCUSDT" (7,991 rows / 27.7 d / -17.14%, one single chunk) and
    # "BTCUSDT_BINANCE" (662,643 rows / 2,300.8 d / +928.90%). Meanwhile
    # launch_asset_run.py L57 restricts --asset to the _BINANCE variants, so
    # every real training run loads the big split while this canonical gate
    # silently measured the small one -- which is how the current NO_GO verdict
    # (gate_c_run_20260904_225928.log, line 8: "asset": "BTCUSDT") was produced
    # on 27.7 unrepresentative days. The asset is now an explicit parameter
    # defaulting to the universe the runs actually use, and it is echoed into
    # the report so no verdict can ever again hide which universe it measured.
    if asset not in LAUNCHER_ASSETS:
        raise RuntimeError(
            f"asset {asset!r} is not one of the launcher's assets "
            f"{sorted(LAUNCHER_ASSETS)}; a gate must measure the universe the "
            f"run actually loads"
        )
    cfg = ConfigLoader.load_config(str(REPO_ROOT / "config" / "config.yaml"))
    cfg.setdefault("environment", {})["rich_display_interval"] = 999999
    # Mirror launch_asset_run.py::derive_config, which rewrites all three keys.
    cfg.setdefault("data", {})["assets"] = [asset]
    cfg.setdefault("environment", {})["assets"] = [asset]
    worker_config = copy.deepcopy(cfg.get("workers", {}).get("w1", {}))
    worker_config.update(
        {
            "worker_id": 0,
            "data_split": split,
            "data_split_override": split,
            "timeframes": ["5m", "1h", "4h"],
            "assets": [asset],
        }
    )
    data = ChunkedDataLoader(
        config=cfg, worker_config=worker_config, worker_id=0
    ).load_chunk(0)
    env = MultiAssetChunkedEnv(
        data=data,
        config=cfg,
        worker_config=worker_config,
        worker_id=0,
        live_mode=False,
    )
    env.action_space.seed(seed)
    env.reset(seed=seed)
    return env


def _gate(verdict: str, **details: Any) -> dict[str, Any]:
    return {"verdict": verdict, **details}


def _overall_verdict(gates: dict[str, dict[str, Any]]) -> str:
    verdicts = [gate["verdict"] for gate in gates.values()]
    if any(verdict == "FAIL" for verdict in verdicts):
        return "NO_GO"
    if any(verdict != "PASS" for verdict in verdicts):
        return "INCONCLUSIVE"
    return "GO"


def run_check(
    *, steps: int, split: str, seed: int, asset: str = DEFAULT_ASSET
) -> dict[str, Any]:
    if steps != FINANCIAL_STEPS:
        raise ValueError(
            f"A/C/D protocol requires exactly {FINANCIAL_STEPS} steps; B always uses "
            f"{ACTION_DIFF_STEPS} steps"
        )

    np.random.seed(seed)
    contract = _load_btc_financial_contract()
    launcher_runtime = _apply_btc_launcher_runtime(contract)
    env = build_environment(split, seed, asset)
    capacity_rewards: list[float] = []
    financial_rewards: list[float] = []
    winning_trade_rewards: list[float] = []
    close_events: list[dict[str, Any]] = []
    requested_actions: list[int] = []
    executed_actions: list[int] = []
    openings = 0
    closings = 0
    winning_closings = 0
    losing_closings = 0
    profitable_closes_with_nonpositive_reward = 0
    episodes = 0
    previous_open_count = _open_count(env)

    try:
        for step_index in range(1, ACTION_DIFF_STEPS + 1):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            reward = _finite_float(reward)
            components = dict(getattr(env, "_last_reward_components", {}) or {})
            requested_actions.append(int(getattr(env, "_last_discrete_action_requested", 0)))
            executed_actions.append(int(getattr(env, "_last_discrete_action", 0)))

            if step_index <= FINANCIAL_STEPS:
                financial_rewards.append(reward)
                capacity_rewards.append(
                    _finite_float(components.get("capacity_reward", 0.0))
                )

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
                            winning_trade_rewards.append(reward)
                        else:
                            profitable_closes_with_nonpositive_reward += 1
                    else:
                        losing_closings += 1
                    close_events.append(
                        {
                            "step": step_index,
                            "pnl": pnl,
                            "final_step_reward": reward,
                            "positive_winning_trade_reward": bool(is_win and reward > 0.0),
                            "reason": receipt.get("reason", receipt.get("close_reason")),
                            "reward_components": {
                                key: _finite_float(components.get(key, 0.0))
                                for key in (
                                    "pnl_reward",
                                    "closure_bonus",
                                    "drawdown_penalty",
                                    "behavior_penalty",
                                    "future_contrib",
                                    "raw",
                                    "final_reward",
                                    "capacity_reward",
                                    "inaction_penalty",
                                )
                            },
                        }
                    )
            elif terminated or truncated:
                previous_open_count = 0

            if terminated or truncated:
                episodes += 1
                env.reset(seed=seed + episodes)
                env.action_space.seed(seed + episodes)
                previous_open_count = _open_count(env)
    finally:
        env.close()

    capacity_max_abs = float(np.max(np.abs(capacity_rewards)))
    capacity_mean = float(np.mean(capacity_rewards))
    capacity_std = float(np.std(capacity_rewards))
    reward_std = float(np.std(financial_rewards))
    winning_trade_reward_mean = (
        float(np.mean(winning_trade_rewards)) if winning_trade_rewards else None
    )
    telemetry_ratio = (
        capacity_max_abs / winning_trade_reward_mean
        if winning_trade_reward_mean is not None and winning_trade_reward_mean > 0.0
        else None
    )
    # Source audit proves capacity_reward is appended only to telemetry after raw_reward.
    effective_capacity_contribution_max_abs = 0.0
    effective_ratio = (
        effective_capacity_contribution_max_abs / winning_trade_reward_mean
        if winning_trade_reward_mean is not None and winning_trade_reward_mean > 0.0
        else None
    )

    if telemetry_ratio is None:
        gate_a = _gate(
            "INCONCLUSIVE",
            reason="no receipt-backed winning close with positive final step reward",
        )
    else:
        gate_a = _gate("PASS" if telemetry_ratio < CAPACITY_RATIO_LIMIT else "FAIL")
    gate_a.update(
        {
            "capacity_reward_max_abs_telemetry": capacity_max_abs,
            "capacity_reward_mean_telemetry": capacity_mean,
            "capacity_reward_std_telemetry": capacity_std,
            "winning_trade_reward_mean": winning_trade_reward_mean,
            "telemetry_ratio": telemetry_ratio,
            "effective_ppo_contribution_max_abs": effective_capacity_contribution_max_abs,
            "effective_ppo_ratio": effective_ratio,
            "capacity_reward_in_optimized_raw_reward": False,
            "threshold": {"operator": "<", "value": CAPACITY_RATIO_LIMIT},
            "decision_basis": "telemetry_ratio_required_by_gate_A",
        }
    )

    divergence_count = sum(
        requested != executed
        for requested, executed in zip(requested_actions, executed_actions)
    )
    divergence_rate = divergence_count / ACTION_DIFF_STEPS
    gate_b = _gate(
        "PASS" if divergence_rate < ACTION_DIFF_LIMIT else "FAIL",
        steps=ACTION_DIFF_STEPS,
        divergence_count=divergence_count,
        divergence_rate=divergence_rate,
        threshold={"operator": "<", "value": ACTION_DIFF_LIMIT},
        source="direct requested/executed environment attributes",
    )

    random_window_executed = executed_actions[:FINANCIAL_STEPS]
    random_window_requested = requested_actions[:FINANCIAL_STEPS]
    executed_hold_count = sum(action == 0 for action in random_window_executed)
    requested_hold_count = sum(action == 0 for action in random_window_requested)
    executed_hold_rate = executed_hold_count / FINANCIAL_STEPS
    requested_hold_rate = requested_hold_count / FINANCIAL_STEPS
    gate_c = _gate(
        "PASS" if executed_hold_rate <= HOLD_LIMIT else "FAIL",
        steps=FINANCIAL_STEPS,
        executed_hold_count=executed_hold_count,
        executed_hold_rate=executed_hold_rate,
        requested_hold_count=requested_hold_count,
        requested_hold_rate=requested_hold_rate,
        threshold={"operator": "<=", "value": HOLD_LIMIT},
        decision_basis="executed HOLD rate",
    )

    gate_d = _gate(
        "PASS" if reward_std > REWARD_STD_MIN else "FAIL",
        steps=FINANCIAL_STEPS,
        reward_std=reward_std,
        reward_mean=float(np.mean(financial_rewards)),
        reward_min=float(np.min(financial_rewards)),
        reward_max=float(np.max(financial_rewards)),
        threshold={"operator": ">", "value": REWARD_STD_MIN},
    )

    fees_to_mean_tp = (
        contract["round_trip_fees"] / contract["mean_tp"]
        if contract["mean_tp"] > 0.0
        else None
    )
    if fees_to_mean_tp is None:
        gate_e = _gate("INCONCLUSIVE", reason="mean TP is not positive")
    else:
        gate_e = _gate("PASS" if fees_to_mean_tp < FEES_TO_TP_LIMIT else "FAIL")
    gate_e.update(
        {
            **contract,
            "fees_to_mean_tp_ratio": fees_to_mean_tp,
            "threshold": {"operator": "<", "value": FEES_TO_TP_LIMIT},
        }
    )

    gates = {
        "A_capacity_vs_winning_trade": gate_a,
        "B_action_diff": gate_b,
        "C_random_hold": gate_c,
        "D_reward_std": gate_d,
        "E_fees_vs_mean_tp": gate_e,
    }
    return {
        "protocol": "real_environment_uniform_random_policy_strict_A_to_E",
        "asset": asset,
        "split": split,
        "seed": seed,
        "measurement_windows": {
            "financial_steps_A_C_D": FINANCIAL_STEPS,
            "action_diff_steps_B": ACTION_DIFF_STEPS,
            "trajectory": "single uninterrupted trajectory except natural episode resets",
        },
        "launcher_runtime_invariants": launcher_runtime,
        "reward_influence_audit": {
            "capacity_reward": "telemetry_only_not_in_raw_reward",
            "inaction_penalty_function": "constant_0.0",
            "inaction_penalty_runtime_call_sites": 0,
        },
        "counts_first_500": {
            "openings": openings,
            "closings": closings,
            "winning_closings": winning_closings,
            "losing_closings": losing_closings,
            "positive_winning_trade_rewards": len(winning_trade_rewards),
            "profitable_closes_with_nonpositive_reward": (
                profitable_closes_with_nonpositive_reward
            ),
            "capacity_positive_steps": sum(value > 0.0 for value in capacity_rewards),
            "capacity_negative_steps": sum(value < 0.0 for value in capacity_rewards),
            "capacity_zero_steps": sum(value == 0.0 for value in capacity_rewards),
        },
        "episodes_or_resets_1000": episodes,
        "gates": gates,
        "overall_verdict": _overall_verdict(gates),
        "launch_authorized": _overall_verdict(gates) == "GO",
        "close_events_first_500": close_events,
    }


def write_report_atomic(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=FINANCIAL_STEPS)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--seed", type=int, default=330500)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    # ADAN0_GATE_ASSET_SOURCE_FIX: explicit, and restricted to the launcher's
    # assets so the gate can no longer silently fall back to the small split.
    parser.add_argument(
        "--asset", choices=LAUNCHER_ASSETS, default=DEFAULT_ASSET
    )
    args = parser.parse_args()

    report = run_check(
        steps=args.steps, split=args.split, seed=args.seed, asset=args.asset
    )
    output = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    write_report_atomic(report, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["overall_verdict"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
