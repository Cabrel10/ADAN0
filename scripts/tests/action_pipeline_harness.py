#!/usr/bin/env python3
"""Deterministic action-pipeline crash test using the real ADAN environment.

This diagnostic intentionally does not load PPO. It injects controlled continuous
Box(5) actions into the same data loader, environment, routing, gates, portfolio,
fees, SL/TP logic, reward, and telemetry used by training.

Example:
    PYTHONPATH=src:. python scripts/tests/action_pipeline_harness.py \
        --split val --steps 1000 \
        --out logs/validation/action_pipeline_harness_v26.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("ADAN_TRAINING_SILENT", "1")
os.environ.setdefault("ADAN_RICH_STEP_EVERY", "999999")
logging.disable(logging.WARNING)

SCENARIOS = (
    "sell_while_flat",
    "constant_buy",
    "buy_then_hold",
    "buy_then_sell",
    "loss_cut",
)


def scripted_direction(
    scenario: str,
    *,
    in_position: bool,
    has_opened: bool,
    steps_held: int,
    unrealized_pnl_pct: float,
    min_hold_steps: int,
) -> float:
    """Return only the controlled direction head for one scenario."""
    if scenario == "sell_while_flat":
        return -1.0
    if scenario == "constant_buy":
        return 1.0
    if scenario == "buy_then_hold":
        return 0.0 if has_opened else 1.0
    if scenario == "buy_then_sell":
        return -1.0 if in_position else 1.0
    if scenario == "loss_cut":
        if not in_position:
            return 1.0
        if steps_held >= min_hold_steps and unrealized_pnl_pct < 0.0:
            return -1.0
        return 0.0
    raise ValueError(f"Unknown scenario: {scenario}")


def controlled_action(
    direction: float,
    *,
    size_raw: float,
    sl_raw: float,
    tp_raw: float,
) -> np.ndarray:
    """Build [direction, size, timeframe, SL, TP] for BTCUSDT on 5m."""
    return np.asarray([direction, size_raw, -1.0, sl_raw, tp_raw], dtype=np.float32)


def _open_position(env: Any) -> Any | None:
    positions = getattr(getattr(env, "portfolio_manager", None), "positions", {})
    if not isinstance(positions, dict):
        return None
    for position in positions.values():
        if position is not None and bool(getattr(position, "is_open", False)):
            return position
    return None


def _position_state(env: Any) -> tuple[bool, int, float]:
    position = _open_position(env)
    if position is None:
        return False, 0, 0.0
    current_step = int(getattr(env, "current_step", 0))
    opened_step = int(getattr(position, "open_step", current_step) or current_step)
    entry = float(getattr(position, "entry_price", 0.0) or 0.0)
    current = float(getattr(position, "current_price", entry) or entry)
    pnl_pct = (current - entry) / entry if entry > 0.0 else 0.0
    return True, max(0, current_step - opened_step), float(pnl_pct)


def _counter_delta(current: dict[str, int], previous: dict[str, int]) -> Counter:
    delta: Counter = Counter()
    for key, value in current.items():
        change = int(value) - int(previous.get(key, 0))
        if change > 0:
            delta[key] += change
    return delta


def _gate_category(stage: str, reason: str) -> str:
    """Map a pipeline rejection to one mutually exclusive execution gate."""
    haystack = f"{stage}:{reason}".lower()
    if "fee" in haystack:
        return "fee_gate"
    if "min_notional" in haystack or "min_order" in haystack:
        return "min_notional"
    if "daily" in haystack:
        return "daily_limit"
    if "cash" in haystack or "budget" in haystack:
        return "cash_budget"
    if "portfolio" in haystack or "pm_reject" in haystack:
        return "portfolio"
    if "barrier" in haystack:
        return "barrier"
    return "other"


def _read_trace(path: Path) -> dict[str, Any]:
    stages: Counter = Counter()
    reasons: Counter = Counter()
    stage_reasons: Counter = Counter()
    rejection_gates: Counter = Counter()
    sizings: list[dict[str, Any]] = []
    opens: list[dict[str, Any]] = []
    closes: list[dict[str, Any]] = []
    if not path.exists():
        return {
            "stages": {},
            "reasons": {},
            "stage_reasons": {},
            "rejection_gates": {},
            "sizings": [],
            "opens": [],
            "closes": [],
        }

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        stage = str(event.get("stage", "unknown"))
        reason = str(event.get("reason", "unknown"))
        stages[stage] += 1
        reasons[reason] += 1
        stage_reasons[f"{stage}:{reason}"] += 1
        if stage == "sizing_decoded":
            sizings.append(event)
        elif stage.endswith("reject") or "reject" in reason.lower() or reason in {
            "min_notional",
            "daily_limit",
        }:
            rejection_gates[_gate_category(stage, reason)] += 1
        if stage == "trade_executed" and event.get("lifecycle_event") == "open":
            opens.append(event)
        elif stage == "trade_executed" and event.get("lifecycle_event") == "close":
            closes.append(event)

    return {
        "stages": dict(stages),
        "reasons": dict(reasons),
        "stage_reasons": dict(stage_reasons),
        "rejection_gates": dict(rejection_gates),
        "sizings": sizings,
        "opens": opens,
        "closes": closes,
    }


def _numeric_component_sums(target: defaultdict[str, float], components: Any) -> None:
    if not isinstance(components, dict):
        return
    for key, value in components.items():
        if isinstance(value, (int, float)) and np.isfinite(value):
            target[str(key)] += float(value)


def _close_summary(closes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    closes = list(closes)
    by_reason: dict[str, list[float]] = defaultdict(list)
    for event in closes:
        by_reason[str(event.get("reason", "unknown"))].append(
            float(event.get("pnl_net", 0.0) or 0.0)
        )

    def stats(values: list[float]) -> dict[str, Any]:
        return {
            "count": len(values),
            "wins": sum(value > 0.0 for value in values),
            "win_rate": sum(value > 0.0 for value in values) / len(values) if values else 0.0,
            "pnl_sum": float(sum(values)),
            "pnl_mean": float(np.mean(values)) if values else 0.0,
        }

    all_values = [value for values in by_reason.values() for value in values]
    return {
        "all": stats(all_values),
        "by_reason": {reason: stats(values) for reason, values in sorted(by_reason.items())},
    }


def _open_summary(opens: Iterable[dict[str, Any]]) -> dict[str, Any]:
    opens = list(opens)

    def range_for(key: str) -> dict[str, float | None]:
        values = [float(event[key]) for event in opens if event.get(key) is not None]
        return {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": float(np.mean(values)) if values else None,
        }

    return {
        "count": len(opens),
        "sl_pct": range_for("sl_pct"),
        "tp_pct": range_for("tp_pct"),
        "notional_usd": range_for("notional_usd"),
        "entry_atr_pct": range_for("entry_atr_pct"),
    }


def _sizing_summary(
    sizings: Iterable[dict[str, Any]],
    opens: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize policy sizing separately from cash-clamped/executed sizing."""
    sizings = list(sizings)
    opens = list(opens)

    def stats(events: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
        values = [float(event[key]) for event in events if event.get(key) is not None]
        return {
            "count": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": float(np.mean(values)) if values else None,
        }

    return {
        "decoded_events": len(sizings),
        "size_raw": stats(sizings, "size_raw"),
        "normalized_size": stats(sizings, "normalized_size"),
        "target_exposure_pct": stats(sizings, "target_exposure_pct"),
        "requested_notional_usd": stats(sizings, "requested_notional_usd"),
        "cash_capped_notional_usd": stats(sizings, "notional_usd"),
        "executed_notional_usd": stats(opens, "notional_usd"),
    }


def _scenario_verdict(result: dict[str, Any]) -> str:
    scenario = result["scenario"]
    stages = result["pipeline"]["stages"]
    opens = int(stages.get("trade_executed", 0)) - result["closes"]["all"]["count"]
    closes = result["closes"]["all"]["count"]
    if scenario == "sell_while_flat":
        if opens == 0 and stages.get("routing_reject", 0) > 0:
            return "PASS_NEUTRAL_ROUTING_HOLD"
        return "FAIL_UNEXPECTED_FLAT_SELL_BEHAVIOR"
    if opens <= 0:
        blocking_reasons = {
            reason: count
            for reason, count in result["pipeline"]["stage_reasons"].items()
            if not reason.startswith("policy:")
        }
        dominant = max(
            blocking_reasons.items(),
            key=lambda item: item[1],
            default=("unknown", 0),
        )[0]
        return f"BLOCKED_BEFORE_OPEN:{dominant}"
    if scenario in {"buy_then_sell", "loss_cut"} and closes <= 0:
        return "OPENED_BUT_POLICY_EXIT_NOT_EXECUTED"
    if result["closes"]["all"]["win_rate"] <= 0.0:
        return "TRADES_EXECUTE_BUT_ALL_CLOSED_AT_LOSS"
    return "TRADING_LIFECYCLE_FUNCTIONAL"


def run_scenario(
    scenario: str,
    *,
    steps: int,
    split: str,
    seed: int,
    size_raw: float,
    sl_raw: float,
    tp_raw: float,
    trace_path: Path,
) -> dict[str, Any]:
    from adan_trading_bot.common.config_loader import ConfigLoader
    from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
    from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

    np.random.seed(seed)
    if trace_path.exists():
        trace_path.unlink()
    os.environ["ADAN_PIPELINE_TRACE_PATH"] = str(trace_path)

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
    env.reset(seed=seed)

    min_hold_steps = int(
        cfg.get("trading_rules", {})
        .get("cooldown", {})
        .get("hold_min_steps", {})
        .get("5m", 6)
    )
    initial_equity = float(env.portfolio_manager.get_equity())
    requested: Counter = Counter()
    requested_by_state: Counter = Counter()
    rejection_totals: Counter = Counter()
    reward_components: defaultdict[str, float] = defaultdict(float)
    total_reward = 0.0
    episodes = 0
    has_opened = False
    previous_rejections = dict(getattr(env, "rejection_reasons", {}))
    previous_attempts = int(getattr(env, "trade_attempts", 0))
    trade_attempts = 0

    for _ in range(steps):
        in_position, steps_held, unrealized_pnl_pct = _position_state(env)
        if in_position:
            has_opened = True
        direction = scripted_direction(
            scenario,
            in_position=in_position,
            has_opened=has_opened,
            steps_held=steps_held,
            unrealized_pnl_pct=unrealized_pnl_pct,
            min_hold_steps=min_hold_steps,
        )
        label = "BUY" if direction > 0.0 else "SELL" if direction < 0.0 else "HOLD"
        requested[label] += 1
        requested_by_state[f"{'LONG' if in_position else 'FLAT'}:{label}"] += 1
        action = controlled_action(
            direction,
            size_raw=size_raw,
            sl_raw=sl_raw,
            tp_raw=tp_raw,
        )
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        _numeric_component_sums(reward_components, info.get("reward_components"))

        current_rejections = dict(getattr(env, "rejection_reasons", {}))
        rejection_totals.update(_counter_delta(current_rejections, previous_rejections))
        previous_rejections = current_rejections
        current_attempts = int(getattr(env, "trade_attempts", 0))
        if current_attempts > previous_attempts:
            trade_attempts += current_attempts - previous_attempts
        previous_attempts = current_attempts

        if terminated or truncated:
            episodes += 1
            env.reset(seed=seed + episodes)
            has_opened = False
            previous_rejections = dict(getattr(env, "rejection_reasons", {}))
            previous_attempts = int(getattr(env, "trade_attempts", 0))

    env.finalize_open_positions(reason="HARNESS_END")
    final_equity = float(env.portfolio_manager.get_equity())
    trace = _read_trace(trace_path)
    policy_events = max(1, int(trace["stages"].get("policy", 0)))
    pipeline_rates = {
        stage: count / policy_events for stage, count in trace["stages"].items()
    }
    result = {
        "scenario": scenario,
        "split": split,
        "seed": seed,
        "steps": steps,
        "action": {
            "size_raw": size_raw,
            "timeframe_raw": -1.0,
            "sl_raw": sl_raw,
            "tp_raw": tp_raw,
        },
        "requested": dict(requested),
        "requested_by_state": dict(requested_by_state),
        "trade_attempts": trade_attempts,
        "rejection_reasons": dict(rejection_totals),
        "pipeline": {
            "stages": trace["stages"],
            "stage_rates_per_policy": pipeline_rates,
            "stage_reasons": trace["stage_reasons"],
            "rejection_gates": trace["rejection_gates"],
        },
        "sizing": _sizing_summary(trace["sizings"], trace["opens"]),
        "opens": _open_summary(trace["opens"]),
        "closes": _close_summary(trace["closes"]),
        "reward": {
            "total": total_reward,
            "component_sums": dict(reward_components),
        },
        "equity": {
            "initial": initial_equity,
            "final": final_equity,
            "return_pct": (final_equity - initial_equity) / initial_equity * 100.0,
        },
        "episodes_completed": episodes,
        "trace_path": str(trace_path.relative_to(REPO_ROOT)),
    }
    result["verdict"] = _scenario_verdict(result)
    return result


def run_harness(
    *,
    scenarios: Iterable[str],
    steps: int,
    split: str,
    seed: int,
    size_raw_values: Iterable[float],
    sl_raw: float,
    tp_raw: float,
    trace_dir: Path,
) -> dict[str, Any]:
    results = []
    scenarios = tuple(scenarios)
    size_raw_values = tuple(size_raw_values)
    for size_index, size_raw in enumerate(size_raw_values):
        size_tag = f"{size_raw:+.2f}".replace("+", "pos").replace("-", "neg").replace(".", "p")
        for scenario_index, scenario in enumerate(scenarios):
            print(
                f"[harness] scenario={scenario} size_raw={size_raw:+.2f} "
                f"steps={steps} split={split}",
                file=sys.stderr,
            )
            trace_path = trace_dir / f"{scenario}_size_{size_tag}.jsonl"
            results.append(
                run_scenario(
                    scenario,
                    steps=steps,
                    split=split,
                    seed=seed + size_index * len(SCENARIOS) + scenario_index,
                    size_raw=size_raw,
                    sl_raw=sl_raw,
                    tp_raw=tp_raw,
                    trace_path=trace_path,
                )
            )
    return {
        "diagnostic": "controlled_action_pipeline_without_ppo",
        "config": "config/config.yaml",
        "split": split,
        "steps_per_scenario": steps,
        "seed": seed,
        "size_raw_values": list(size_raw_values),
        "scenarios": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=26027)
    parser.add_argument(
        "--size-raw",
        type=float,
        action="append",
        help="Repeat to build a sizing matrix; default: 0.0",
    )
    parser.add_argument("--sl-raw", type=float, default=1.0)
    parser.add_argument("--tp-raw", type=float, default=1.0)
    parser.add_argument("--scenario", action="append", choices=SCENARIOS)
    parser.add_argument(
        "--trace-dir",
        default="logs/validation/action_pipeline_harness_traces",
    )
    parser.add_argument(
        "--out",
        default="logs/validation/action_pipeline_harness_v26.json",
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    size_raw_values = args.size_raw if args.size_raw is not None else [0.0]
    if any(not -1.0 <= value <= 1.0 for value in size_raw_values):
        parser.error("--size-raw must be in [-1, 1]")
    if not -1.0 <= args.sl_raw <= 1.0 or not -1.0 <= args.tp_raw <= 1.0:
        parser.error("--sl-raw and --tp-raw must be in [-1, 1]")

    trace_dir = Path(args.trace_dir)
    if not trace_dir.is_absolute():
        trace_dir = REPO_ROOT / trace_dir
    trace_dir.mkdir(parents=True, exist_ok=True)
    scenarios = tuple(args.scenario) if args.scenario else SCENARIOS
    report = run_harness(
        scenarios=scenarios,
        steps=args.steps,
        split=args.split,
        seed=args.seed,
        size_raw_values=size_raw_values,
        sl_raw=args.sl_raw,
        tp_raw=args.tp_raw,
        trace_dir=trace_dir,
    )
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[harness] report={out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
