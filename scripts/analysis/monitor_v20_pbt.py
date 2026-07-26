#!/usr/bin/env python3
"""Continuous, trial-aware supervisor for the V20 Ray/PBT run.

This monitor never edits rewards, hyperparameters, capital tiers, fees,
features, or checkpoints. Alerts use persistence and robust rolling statistics
(median/MAD) rather than a single noisy threshold. An optional fail-stop can
terminate only the parent training PID after repeated critical polls.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import signal
import subprocess
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PPO_METRICS = (
    "mean_reward",
    "explained_variance",
    "approx_kl",
    "clip_fraction",
    "entropy_loss",
    "policy_gradient_loss",
    "value_loss",
    "std",
)


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def robust_signal(
    values: list[float], x_values: list[float] | None = None, ewma_alpha: float = 0.30
) -> dict[str, float | None]:
    """Return scale-aware raw trend diagnostics without hiding observations.

    Median and MAD use their standard definitions (including averaging the two
    central values for even samples). The regression slope is normalized per
    10k training steps when timestamps are available. CUSUM is expressed in
    robust-sigma units with allowance k=0.5; it is diagnostic, not by itself a
    stop condition.
    """
    clean = [float(value) for value in values if math.isfinite(value)]
    if not clean:
        return {
            "median": None, "mad": None, "robust_scale": None,
            "last_robust_z": None, "slope_per_10k_steps": None,
            "ewma": None, "ewma_delta": None,
            "cusum_positive": None, "cusum_negative": None,
        }
    middle = float(median(clean))
    mad = float(median([abs(value - middle) for value in clean]))
    scale = 1.4826 * mad
    last_z = (clean[-1] - middle) / scale if scale > 0 else 0.0

    xs = list(x_values) if x_values and len(x_values) == len(clean) else list(range(len(clean)))
    slope = 0.0
    if len(clean) >= 2:
        x_mean = sum(xs) / len(xs)
        y_mean = sum(clean) / len(clean)
        denominator = sum((value - x_mean) ** 2 for value in xs)
        raw_slope = (
            sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, clean)) / denominator
            if denominator else 0.0
        )
        slope = raw_slope * (10_000.0 if x_values else 1.0)

    ewma = clean[0]
    previous_ewma = ewma
    for value in clean[1:]:
        previous_ewma = ewma
        ewma = ewma_alpha * value + (1.0 - ewma_alpha) * ewma

    cusum_positive = 0.0
    cusum_negative = 0.0
    if scale > 0:
        for value in clean:
            standardized = (value - middle) / scale
            cusum_positive = max(0.0, cusum_positive + standardized - 0.5)
            cusum_negative = min(0.0, cusum_negative + standardized + 0.5)
    return {
        "median": middle,
        "mad": mad,
        "robust_scale": scale,
        "last_robust_z": last_z,
        "slope_per_10k_steps": slope,
        "ewma": ewma,
        "ewma_delta": ewma - previous_ewma,
        "cusum_positive": cusum_positive,
        "cusum_negative": cusum_negative,
    }


def latest_progress(storage: Path) -> dict[str, dict[str, Any]]:
    """Load per-trial metrics from CSV, with Ray JSONL as a fallback."""
    reports: dict[str, dict[str, Any]] = {}
    for path_string in glob.glob(str(storage / "**/progress.csv"), recursive=True):
        path = Path(path_string)
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, csv.Error):
            continue
        if rows:
            reports[path.parent.name] = {"path": str(path), "rows": rows}

    for path_string in glob.glob(str(storage / "**/result.json"), recursive=True):
        path = Path(path_string)
        trial = path.parent.name
        if trial in reports:
            continue
        rows: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        rows.append(item)
        except OSError:
            continue
        if rows:
            reports[trial] = {"path": str(path), "rows": rows}
    return reports


def _stage_rates(counts: Counter[str]) -> dict[str, float | None]:
    policy = counts.get("policy", 0)
    return {
        key: value / policy if policy else None
        for key, value in sorted(counts.items())
        if key != "policy"
    }


def pipeline_counts(pattern: str, decision_window: int = 10_000) -> dict[str, Any]:
    """Aggregate the action funnel globally and independently per worker.

    The recent window is measured in policy decisions, not JSONL lines: one
    decision can emit multiple stages, so a line-based denominator is biased.
    """
    counts: Counter[str] = Counter()
    by_worker: defaultdict[str, Counter[str]] = defaultdict(Counter)
    recent_decisions: defaultdict[str, deque[set[str]]] = defaultdict(
        lambda: deque(maxlen=decision_window)
    )
    invalid = 0
    missing = 0
    files = sorted(glob.glob(pattern))
    required = {"step", "worker_id", "asset", "stage", "action_in", "action_out", "reason"}
    for file_name in files:
        try:
            with open(file_name, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        invalid += 1
                        continue
                    if not required.issubset(item):
                        missing += 1
                    stage = str(item.get("stage", "unknown"))
                    worker = str(item.get("worker_id", "unknown"))
                    counts[stage] += 1
                    by_worker[worker][stage] += 1
                    if stage == "policy":
                        recent_decisions[worker].append({"policy"})
                    elif recent_decisions[worker]:
                        recent_decisions[worker][-1].add(stage)
        except OSError:
            continue

    worker_reports: dict[str, Any] = {}
    for worker, worker_counts in sorted(by_worker.items()):
        recent = Counter(stage for decision in recent_decisions[worker] for stage in decision)
        worker_reports[worker] = {
            "events": int(sum(worker_counts.values())),
            "counts": dict(sorted(worker_counts.items())),
            "rates_per_policy": _stage_rates(worker_counts),
            "recent_window_decisions": len(recent_decisions[worker]),
            "recent_counts": dict(sorted(recent.items())),
            "recent_rates_per_policy": _stage_rates(recent),
        }
    return {
        "files": files,
        "events": int(sum(counts.values())),
        "counts": dict(sorted(counts.items())),
        "invalid_json": invalid,
        "missing_required_fields": missing,
        "rates_per_policy": _stage_rates(counts),
        "by_worker": worker_reports,
    }


def arena_health(path: Path, recent_window: int = 10_000) -> dict[str, Any]:
    if not path.exists():
        return {"records": 0, "recent_records": 0}
    records = 0
    wins = 0
    reasons: Counter[str] = Counter()
    recent_outcomes: deque[tuple[str, bool]] = deque(maxlen=recent_window)
    malformed = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                records += 1
                meta = item.get("meta", {})
                profitable = finite(meta.get("pnl_net")) is not None and float(meta.get("pnl_net")) > 0
                reason = str(meta.get("reason", "unknown"))
                wins += profitable
                reasons[reason] += 1
                recent_outcomes.append((reason, profitable))
    except OSError:
        pass
    recent_reasons = Counter(reason for reason, _ in recent_outcomes)
    recent_wins = sum(profitable for _, profitable in recent_outcomes)
    recent_count = len(recent_outcomes)
    recent_closings = sum(recent_reasons.values())
    return {
        "records": records,
        "profitable_outcome_rate": wins / records if records else None,
        "reasons": dict(sorted(reasons.items())),
        "recent_records": recent_count,
        "recent_profitable_outcome_rate": recent_wins / recent_count if recent_count else None,
        "recent_reasons": dict(sorted(recent_reasons.items())),
        "recent_max_duration_rate": (
            recent_reasons.get("MaxDuration", 0) / recent_closings if recent_closings else None
        ),
        "malformed": malformed,
    }


def process_health(pid: int | None) -> dict[str, Any]:
    virtual = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_usage(str(PROJECT_ROOT))
    process_data: dict[str, Any] = {"alive": False, "pid": pid}
    if pid:
        try:
            process = psutil.Process(pid)
            children = process.children(recursive=True)
            process_data.update(
                {
                    "alive": process.is_running(),
                    "rss_bytes_tree": process.memory_info().rss
                    + sum(child.memory_info().rss for child in children if child.is_running()),
                    "cpu_percent_tree": process.cpu_percent(None)
                    + sum(child.cpu_percent(None) for child in children if child.is_running()),
                    "children": len(children),
                    "status": process.status(),
                }
            )
        except (psutil.Error, OSError):
            pass
    return {
        "process": process_data,
        "ram_percent": virtual.percent,
        "ram_available_bytes": virtual.available,
        "swap_percent": swap.percent,
        "disk_percent": disk.percent,
        "disk_free_bytes": disk.free,
        "load_average": os.getloadavg(),
    }


def persistent_alerts(
    trial: str,
    rows: list[dict[str, str]],
    history_size: int,
) -> tuple[dict[str, Any], list[str]]:
    """Evaluate every PPO metric alone, then coherent metric groups.

    CRITICAL alerts are restricted to persistent numerical/pathological states.
    WARN alerts preserve suspicious but scale-dependent evidence for review.
    """
    recent = rows[-history_size:]
    metrics: dict[str, Any] = {}
    alerts: list[str] = []
    steps = [finite(row.get("timesteps_total")) for row in recent]

    for key in PPO_METRICS:
        pairs = [
            (step, value)
            for row, step in zip(recent, steps)
            if step is not None and (value := finite(row.get(key))) is not None
        ]
        values = [value for _, value in pairs]
        x_values = [step for step, _ in pairs]
        missing = len(recent) - len(values)
        metrics[key] = {
            "last": values[-1] if values else None,
            "raw_window": values,
            "window": len(values),
            "missing": missing,
            **robust_signal(values, x_values),
        }
        if len(values) >= 5:
            z = metrics[key].get("last_robust_z")
            if z is not None and abs(z) >= 6.0:
                alerts.append(f"WARN {trial}:{key}: robust outlier z={z:.2f}")
        if len(recent) >= 3 and all(finite(row.get(key)) is None for row in recent[-3:]):
            alerts.append(f"CRITICAL {trial}:{key}: telemetry absent/non-finite for 3 polls")

    def last_n(key: str, n: int) -> list[float]:
        return [value for row in recent[-n:] if (value := finite(row.get(key))) is not None]

    # Isolated invariants and persistent numerical circuit breakers.
    kl = last_n("approx_kl", 3)
    clip = last_n("clip_fraction", 3)
    ev = last_n("explained_variance", 4)
    entropy = last_n("entropy_loss", 3)
    value_loss = last_n("value_loss", 3)
    std = last_n("std", 3)
    if kl and any(value < -1e-6 for value in kl):
        alerts.append(f"WARN {trial}: approx_kl below zero; inspect estimator/noise")
    if len(kl) == 3 and all(value > 0.20 for value in kl):
        alerts.append(f"CRITICAL {trial}: persistent KL trust-region breach")
    if clip and any(value < 0.0 or value > 1.0 for value in clip):
        alerts.append(f"CRITICAL {trial}: clip_fraction outside [0,1]")
    if len(clip) == 3 and all(value > 0.50 for value in clip):
        alerts.append(f"CRITICAL {trial}: persistent PPO clipping saturation")
    if ev and any(value > 1.000001 for value in ev):
        alerts.append(f"CRITICAL {trial}: explained_variance above mathematical maximum 1")
    if len(ev) == 4 and all(value < -0.50 for value in ev):
        alerts.append(f"CRITICAL {trial}: persistent critic divergence")
    if entropy and any(value > 1e-8 for value in entropy):
        alerts.append(f"WARN {trial}: entropy_loss positive (SB3 convention expects -entropy)")
    if value_loss and any(value < 0.0 for value in value_loss):
        alerts.append(f"CRITICAL {trial}: value_loss is negative")
    if std and any(value <= 0.0 for value in std):
        alerts.append(f"CRITICAL {trial}: non-positive exploration std")
    if len(std) == 3 and (
        all(value < 1e-3 for value in std) or all(value > 5.0 for value in std)
    ):
        alerts.append(f"CRITICAL {trial}: persistent exploration-scale collapse/explosion")

    # Grouped diagnoses require agreement between independent signals.
    groups = {
        "policy_trust_region": "healthy",
        "critic": "healthy",
        "exploration": "healthy",
        "performance": "observed",
    }
    if len(kl) == 3 and len(clip) == 3 and all(v > 0.20 for v in kl) and all(v > 0.50 for v in clip):
        groups["policy_trust_region"] = "critical"
    elif (kl and kl[-1] > 0.10) or (clip and clip[-1] > 0.35):
        groups["policy_trust_region"] = "watch"
    value_z = metrics["value_loss"].get("last_robust_z")
    if len(ev) == 4 and all(v < -0.50 for v in ev):
        groups["critic"] = "critical"
    elif (ev and ev[-1] < 0.0) or (value_z is not None and value_z > 4.0):
        groups["critic"] = "watch"
    if len(std) == 3 and (all(v < 1e-3 for v in std) or all(v > 5.0 for v in std)):
        groups["exploration"] = "critical"
    elif std and (std[-1] < 0.05 or std[-1] > 3.0):
        groups["exploration"] = "watch"

    reward = last_n("mean_reward", 4)
    if len(reward) == 4 and all(value < -2.0 for value in reward):
        slope = metrics["mean_reward"].get("slope_per_10k_steps")
        if slope is not None and slope < 0:
            groups["performance"] = "critical"
            alerts.append(f"CRITICAL {trial}: reward below -2 and persistently decreasing")

    last = rows[-1]
    metrics["groups"] = groups
    metrics["timesteps_total"] = finite(last.get("timesteps_total"))
    metrics["training_iteration"] = finite(last.get("training_iteration"))
    metrics["profile"] = last.get("config/worker_config/profile")
    metrics["worker_idx"] = finite(last.get("config/worker_config/worker_idx"))
    return metrics, alerts


def run_benchmark(
    arena: Path,
    run_log: Path,
    output_dir: Path,
    milestone: int,
) -> dict[str, Any]:
    if not arena.exists() or arena.stat().st_size == 0:
        return {"ok": False, "error": "Arena sample file unavailable"}
    output_json = output_dir / f"adan_benchmark_v20_{milestone:06d}.json"
    output_markdown = output_dir / f"adan_benchmark_v20_{milestone:06d}.md"
    command = [
        str(Path(__import__("sys").executable)),
        str(PROJECT_ROOT / "scripts/analysis/adan_benchmark.py"),
        "--asset", "BTCUSDT",
        "--timeframe", "5m",
        "--arena", str(arena),
        "--training-log", str(run_log),
        "--output-json", str(output_json),
        "--output-markdown", str(output_markdown),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=600,
    )
    if completed.returncode:
        return {"ok": False, "error": completed.stderr[-2000:]}
    try:
        report = json.loads(output_json.read_text(encoding="utf-8"))
        score = report["G_global_score"]["global_score_0_to_20"]
    except (OSError, ValueError, KeyError) as error:
        return {"ok": False, "error": str(error)}
    return {"ok": True, "score": score, "json": str(output_json)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--pipeline-glob", required=True)
    parser.add_argument("--arena", type=Path, required=True)
    parser.add_argument("--snapshot-jsonl", type=Path, required=True)
    parser.add_argument("--alerts-log", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--stale-minutes", type=float, default=45.0)
    parser.add_argument("--history", type=int, default=20)
    parser.add_argument("--pipeline-window", type=int, default=10_000)
    parser.add_argument("--expected-trials", type=int, default=4)
    parser.add_argument(
        "--terminate-after-critical-polls",
        type=int,
        default=0,
        help="Send SIGTERM to --pid after N consecutive CRITICAL polls (0 disables).",
    )
    parser.add_argument(
        "--benchmark-milestones",
        default="50000,100000,200000,300000,400000,500000",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.snapshot_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.alerts_log.parent.mkdir(parents=True, exist_ok=True)
    args.benchmark_dir.mkdir(parents=True, exist_ok=True)
    milestones = sorted(
        {int(value) for value in args.benchmark_milestones.split(",") if value.strip()}
    )
    completed_milestones: set[int] = set()
    last_steps: dict[str, float] = {}
    last_progress_time: dict[str, float] = defaultdict(time.time)
    monitor_started = time.time()
    consecutive_critical_polls = 0
    termination_sent = False

    while True:
        now = datetime.now(timezone.utc).isoformat()
        resources = process_health(args.pid)
        reports = latest_progress(args.storage)
        trial_metrics: dict[str, Any] = {}
        alerts: list[str] = []
        for trial, data in sorted(reports.items()):
            metrics, trial_alerts = persistent_alerts(trial, data["rows"], args.history)
            trial_metrics[trial] = metrics
            alerts.extend(trial_alerts)
            steps = finite(metrics.get("timesteps_total")) or 0.0
            if steps > last_steps.get(trial, -1.0):
                last_steps[trial] = steps
                last_progress_time[trial] = time.time()
            elif time.time() - last_progress_time[trial] > args.stale_minutes * 60:
                alerts.append(
                    f"CRITICAL {trial}: no timestep progress for "
                    f"{args.stale_minutes:.0f} minutes"
                )

        pipeline = pipeline_counts(args.pipeline_glob, args.pipeline_window)
        arena = arena_health(args.arena, args.pipeline_window)
        if pipeline["invalid_json"] or pipeline["missing_required_fields"]:
            alerts.append("CRITICAL pipeline JSON integrity failure")
        for trial, metrics in trial_metrics.items():
            worker_idx = metrics.get("worker_idx")
            if worker_idx is None:
                continue
            worker = pipeline["by_worker"].get(str(int(worker_idx)), {})
            recent_count = worker.get("recent_window_decisions", 0)
            trade_rate = worker.get("recent_rates_per_policy", {}).get("trade_executed")
            metrics["pipeline_worker"] = worker
            if recent_count >= args.pipeline_window and (trade_rate is None or trade_rate == 0.0):
                alerts.append(
                    f"CRITICAL {trial}: zero trade_executed over {recent_count} policy decisions"
                )
        max_duration_rate = arena.get("recent_max_duration_rate")
        if arena.get("recent_records", 0) >= 1000 and max_duration_rate is not None and max_duration_rate > 0.60:
            alerts.append(
                f"CRITICAL arena: MaxDuration={max_duration_rate:.1%} over recent closings"
            )
        if (
            time.time() - monitor_started > 15 * 60
            and len(trial_metrics) < args.expected_trials
        ):
            alerts.append(
                f"CRITICAL population: only {len(trial_metrics)}/{args.expected_trials} trials visible"
            )
        if resources["ram_percent"] >= 90:
            alerts.append(f"CRITICAL host RAM pressure {resources['ram_percent']:.1f}%")
        if resources["disk_free_bytes"] < 20 * 1024**3:
            alerts.append("CRITICAL host disk free space below 20 GiB")
        if args.pid and not resources["process"]["alive"]:
            alerts.append("CRITICAL training process is not alive")

        critical_now = any(alert.startswith("CRITICAL ") for alert in alerts)
        consecutive_critical_polls = consecutive_critical_polls + 1 if critical_now else 0
        if (
            args.pid
            and args.terminate_after_critical_polls > 0
            and consecutive_critical_polls >= args.terminate_after_critical_polls
            and resources["process"]["alive"]
            and not termination_sent
        ):
            try:
                os.kill(args.pid, signal.SIGTERM)
                termination_sent = True
                alerts.append(
                    "CRITICAL supervisor: SIGTERM sent after "
                    f"{consecutive_critical_polls} consecutive critical polls"
                )
            except OSError as error:
                alerts.append(f"WARN supervisor: SIGTERM failed: {error}")

        benchmark_events: dict[str, Any] = {}
        max_steps = max(
            (finite(metrics.get("timesteps_total")) or 0.0 for metrics in trial_metrics.values()),
            default=0.0,
        )
        for milestone in milestones:
            if max_steps >= milestone and milestone not in completed_milestones:
                benchmark_events[str(milestone)] = run_benchmark(
                    args.arena, args.run_log, args.benchmark_dir, milestone
                )
                completed_milestones.add(milestone)

        snapshot = {
            "timestamp_utc": now,
            "resources": resources,
            "trials": trial_metrics,
            "pipeline": pipeline,
            "arena": arena,
            "alerts": alerts,
            "critical_streak": consecutive_critical_polls,
            "termination_sent": termination_sent,
            "benchmarks": benchmark_events,
        }
        with args.snapshot_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        if alerts or benchmark_events:
            with args.alerts_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "timestamp_utc": now,
                    "alerts": alerts,
                    "benchmarks": benchmark_events,
                }, ensure_ascii=False) + "\n")
        print(json.dumps({
            "timestamp": now,
            "alive": resources["process"]["alive"],
            "ram_pct": resources["ram_percent"],
            "trials": {
                trial: {
                    "steps": metrics.get("timesteps_total"),
                    "reward": metrics.get("mean_reward", {}).get("last"),
                    "ev": metrics.get("explained_variance", {}).get("last"),
                    "kl": metrics.get("approx_kl", {}).get("last"),
                }
                for trial, metrics in trial_metrics.items()
            },
            "pipeline": pipeline["counts"],
            "arena_records": arena["records"],
            "critical_streak": consecutive_critical_polls,
            "alerts": alerts,
        }, ensure_ascii=False), flush=True)
        if args.pid and not resources["process"]["alive"]:
            return 1 if alerts else 0
        time.sleep(max(5, args.poll))


if __name__ == "__main__":
    raise SystemExit(main())
