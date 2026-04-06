#!/usr/bin/env python3
"""
ADAN Training Results Analyzer
================================
Run this on your LOCAL machine where the training logs reside.

Usage:
    python scripts/analyze_training_results.py

    # Or specify a custom path:
    python scripts/analyze_training_results.py --path /mnt/new_data/t10_training
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────
DEFAULT_TRAIN_DIR = "/mnt/new_data/t10_training"


def find_progress_csvs(base_dir: str) -> list:
    """Recursively find all progress.csv files."""
    pattern = os.path.join(base_dir, "**", "progress.csv")
    return sorted(glob.glob(pattern, recursive=True))


def find_result_jsons(base_dir: str) -> list:
    """Recursively find all result.json files."""
    pattern = os.path.join(base_dir, "**", "result.json")
    return sorted(glob.glob(pattern, recursive=True))


def parse_progress_csv(filepath: str) -> list:
    """Parse a Ray Tune / SB3 progress CSV safely (handles multi-JSON lines)."""
    rows = []
    try:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"  [WARN] CSV parse error for {filepath}: {e}")
    return rows


def parse_result_json(filepath: str) -> list:
    """Parse result.json — handles both single JSON and JSONL (extra data)."""
    results = []
    try:
        with open(filepath, "r") as f:
            content = f.read().strip()
            if not content:
                return results
            # Try single JSON first
            try:
                results.append(json.loads(content))
                return results
            except json.JSONDecodeError:
                pass
            # Try JSONL (one JSON per line)
            for line in content.splitlines():
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"  [WARN] JSON parse error for {filepath}: {e}")
    return results


def safe_float(val, default=0.0):
    """Safely convert to float."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def analyze_worker_csv(filepath: str) -> dict:
    """Analyze a single worker's progress.csv and extract key metrics."""
    rows = parse_progress_csv(filepath)
    if not rows:
        return {"error": "empty or unparseable", "path": filepath}

    # Common Ray Tune / SB3 column names
    reward_cols = [
        "episode_reward_mean", "mean_reward", "reward",
        "ep_rew_mean", "rollout/ep_rew_mean"
    ]
    step_cols = [
        "timesteps_total", "total_timesteps", "step",
        "time/total_timesteps", "training_iteration"
    ]
    loss_cols = [
        "loss", "train/loss", "train/policy_gradient_loss",
        "rollout/ep_len_mean"
    ]

    def find_col(row, candidates):
        for c in candidates:
            if c in row and row[c]:
                return c
        return None

    first_row = rows[0]
    reward_col = find_col(first_row, reward_cols)
    step_col = find_col(first_row, step_cols)
    loss_col = find_col(first_row, loss_cols)

    rewards = []
    steps = []
    losses = []

    for r in rows:
        if reward_col and r.get(reward_col):
            rewards.append(safe_float(r[reward_col]))
        if step_col and r.get(step_col):
            steps.append(safe_float(r[step_col]))
        if loss_col and r.get(loss_col):
            losses.append(safe_float(r[loss_col]))

    result = {
        "path": filepath,
        "total_rows": len(rows),
        "columns": list(first_row.keys()),
        "reward_col": reward_col,
        "step_col": step_col,
    }

    if rewards:
        result["reward_first"] = rewards[0]
        result["reward_last"] = rewards[-1]
        result["reward_best"] = max(rewards)
        result["reward_worst"] = min(rewards)
        result["reward_mean"] = sum(rewards) / len(rewards)
        # Trend: compare last 10% vs first 10%
        n = max(1, len(rewards) // 10)
        first_slice = rewards[:n]
        last_slice = rewards[-n:]
        result["reward_trend_first10pct"] = sum(first_slice) / len(first_slice)
        result["reward_trend_last10pct"] = sum(last_slice) / len(last_slice)
        result["reward_improving"] = result["reward_trend_last10pct"] > result["reward_trend_first10pct"]

    if steps:
        result["step_first"] = steps[0]
        result["step_last"] = steps[-1]
        result["total_steps"] = steps[-1]

    if losses:
        result["loss_first"] = losses[0]
        result["loss_last"] = losses[-1]
        result["loss_mean"] = sum(losses) / len(losses)

    # Extract extra metrics if available
    extra_keys = [
        "mean_sharpe", "mean_balance", "entropy_loss",
        "learning_rate", "ent_coef", "gamma",
        "ep_len_mean", "rollout/ep_len_mean",
        "explained_variance", "approx_kl",
    ]
    last_row = rows[-1]
    for k in extra_keys:
        if k in last_row and last_row[k]:
            result[f"final_{k}"] = safe_float(last_row[k])

    return result


def analyze_log_file(log_path: str, max_lines: int = 50000) -> dict:
    """Scan a log file for key events and errors."""
    counters = defaultdict(int)
    errors = []
    last_lines = []
    total_lines = 0

    try:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                total_lines += 1
                if total_lines > max_lines:
                    break

                # Count key events
                for pattern in [
                    "TRADE_OPEN", "AGENT_CLOSE", "EPISODE_END",
                    "HOLD_MIN", "WAIT_BLOCK", "REWARD_ANTIHACK",
                    "ACTION_DIFF", "EPISODE_REJECTIONS",
                    "NameError", "AttributeError", "KeyError",
                    "RuntimeError", "OOM", "CUDA", "ray.tune",
                    "CHECKPOINT", "BEST_MODEL", "training_iteration",
                ]:
                    if pattern in line:
                        counters[pattern] += 1

                if "Error" in line or "ERROR" in line or "Traceback" in line:
                    errors.append(line.strip()[:300])

                # Keep last 20 lines
                last_lines.append(line.strip())
                if len(last_lines) > 20:
                    last_lines.pop(0)

    except Exception as e:
        return {"error": str(e), "path": log_path}

    return {
        "path": log_path,
        "total_lines_scanned": total_lines,
        "event_counts": dict(counters),
        "error_count": len(errors),
        "sample_errors": errors[:10],
        "last_20_lines": last_lines,
    }


def find_checkpoints(base_dir: str) -> list:
    """Find saved model checkpoints."""
    patterns = ["**/*.zip", "**/best_model*", "**/checkpoint*"]
    found = []
    for p in patterns:
        found.extend(glob.glob(os.path.join(base_dir, p), recursive=True))
    return sorted(set(found))[:50]  # limit


def main():
    parser = argparse.ArgumentParser(description="ADAN Training Results Analyzer")
    parser.add_argument("--path", default=DEFAULT_TRAIN_DIR,
                        help=f"Training output directory (default: {DEFAULT_TRAIN_DIR})")
    parser.add_argument("--log", default=None,
                        help="Specific log file to analyze (e.g., FINAL_1M.log)")
    parser.add_argument("--max-log-lines", type=int, default=100000,
                        help="Max lines to scan per log file")
    args = parser.parse_args()

    base = args.path
    print("=" * 80)
    print(f"  ADAN Training Results Analyzer")
    print(f"  Base directory: {base}")
    print("=" * 80)

    if not os.path.exists(base):
        print(f"\n[ERROR] Directory does not exist: {base}")
        print("  Specify correct path with: --path /your/training/dir")
        sys.exit(1)

    # ── 1. Find and analyze progress.csv files ──
    print("\n" + "─" * 60)
    print("  1. PROGRESS CSV FILES (Training Curves)")
    print("─" * 60)

    csvs = find_progress_csvs(base)
    print(f"  Found {len(csvs)} progress.csv file(s)")

    worker_results = []
    for csv_path in csvs:
        print(f"\n  Analyzing: {csv_path}")
        analysis = analyze_worker_csv(csv_path)
        worker_results.append(analysis)

        if "error" in analysis:
            print(f"    [SKIP] {analysis['error']}")
            continue

        print(f"    Rows: {analysis['total_rows']}")
        if "total_steps" in analysis:
            print(f"    Total Steps: {analysis['total_steps']:,.0f}")
        if "reward_mean" in analysis:
            print(f"    Reward: mean={analysis['reward_mean']:.4f}, "
                  f"best={analysis['reward_best']:.4f}, "
                  f"last={analysis['reward_last']:.4f}")
        if "reward_improving" in analysis:
            trend = "IMPROVING" if analysis["reward_improving"] else "DECLINING/FLAT"
            print(f"    Trend: {trend} "
                  f"(first10%={analysis['reward_trend_first10pct']:.4f}, "
                  f"last10%={analysis['reward_trend_last10pct']:.4f})")
        if "loss_last" in analysis:
            print(f"    Loss (last): {analysis['loss_last']:.6f}")
        for k, v in analysis.items():
            if k.startswith("final_"):
                print(f"    {k}: {v:.6f}")

    # ── 2. Result JSON files ──
    print("\n" + "─" * 60)
    print("  2. RESULT JSON FILES (Ray Tune Trials)")
    print("─" * 60)

    jsons = find_result_jsons(base)
    print(f"  Found {len(jsons)} result.json file(s)")

    for jpath in jsons[:10]:
        print(f"\n  File: {jpath}")
        results = parse_result_json(jpath)
        if not results:
            print(f"    [EMPTY/UNPARSEABLE]")
            continue
        last = results[-1]
        # Print key metrics
        for k in ["training_iteration", "timesteps_total",
                   "episode_reward_mean", "mean_reward",
                   "mean_sharpe", "mean_balance",
                   "done", "status"]:
            if k in last:
                print(f"    {k}: {last[k]}")

    # ── 3. Log file analysis ──
    print("\n" + "─" * 60)
    print("  3. LOG FILE ANALYSIS")
    print("─" * 60)

    if args.log:
        log_files = [args.log]
    else:
        log_files = glob.glob(os.path.join(base, "**", "*.log"), recursive=True)
        # Also check for FINAL_1M.log specifically
        final_log = os.path.join(base, "logs", "FINAL_1M.log")
        if os.path.exists(final_log) and final_log not in log_files:
            log_files.insert(0, final_log)

    print(f"  Found {len(log_files)} log file(s)")

    for lf in log_files[:10]:
        size_mb = os.path.getsize(lf) / (1024 * 1024)
        print(f"\n  Log: {lf} ({size_mb:.1f} MB)")
        analysis = analyze_log_file(lf, max_lines=args.max_log_lines)
        if "error" in analysis:
            print(f"    [ERROR] {analysis['error']}")
            continue
        print(f"    Lines scanned: {analysis['total_lines_scanned']:,}")
        if analysis["event_counts"]:
            print(f"    Event counts:")
            for evt, cnt in sorted(analysis["event_counts"].items(),
                                   key=lambda x: -x[1]):
                print(f"      {evt}: {cnt:,}")
        if analysis["error_count"] > 0:
            print(f"    Errors found: {analysis['error_count']}")
            for e in analysis["sample_errors"][:5]:
                print(f"      >> {e[:200]}")
        print(f"    Last lines:")
        for line in analysis["last_20_lines"][-5:]:
            print(f"      {line[:200]}")

    # ── 4. Checkpoints ──
    print("\n" + "─" * 60)
    print("  4. MODEL CHECKPOINTS")
    print("─" * 60)

    ckpts = find_checkpoints(base)
    print(f"  Found {len(ckpts)} checkpoint file(s)")
    for c in ckpts[:20]:
        size_mb = os.path.getsize(c) / (1024 * 1024)
        print(f"    {c} ({size_mb:.1f} MB)")

    # ── 5. PBT Summary ──
    print("\n" + "─" * 60)
    print("  5. PBT SUMMARY (if available)")
    print("─" * 60)

    summary_files = glob.glob(os.path.join(base, "**", "pbt_summary.json"),
                              recursive=True)
    if summary_files:
        for sf in summary_files:
            print(f"\n  File: {sf}")
            try:
                with open(sf) as f:
                    summary = json.load(f)
                print(json.dumps(summary, indent=2))
            except Exception as e:
                print(f"    [ERROR] {e}")
    else:
        print("  No pbt_summary.json found (training may not have completed)")

    # ── 6. Overall Verdict ──
    print("\n" + "=" * 80)
    print("  VERDICT")
    print("=" * 80)

    if not csvs and not jsons:
        print("  [INCOMPLETE] No progress.csv or result.json found.")
        print("  The training may have crashed before producing metrics.")
    elif worker_results:
        completed = [w for w in worker_results
                     if "total_steps" in w and w["total_steps"] > 0]
        if completed:
            best = max(completed, key=lambda w: w.get("reward_best", -999))
            print(f"  Workers with data: {len(completed)}/{len(worker_results)}")
            print(f"  Best worker reward: {best.get('reward_best', 'N/A')}")
            print(f"  Best worker total steps: {best.get('total_steps', 'N/A')}")
            max_steps = max(w.get("total_steps", 0) for w in completed)
            if max_steps < 10000:
                print(f"  [WARNING] Max steps ({max_steps:,.0f}) is very low — training may have terminated early.")
            improving = [w for w in completed if w.get("reward_improving")]
            print(f"  Workers showing improvement: {len(improving)}/{len(completed)}")
        else:
            print("  [INCOMPLETE] Progress files exist but no steps recorded.")
    print("=" * 80)


if __name__ == "__main__":
    main()
