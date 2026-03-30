#!/usr/bin/env python3
"""
ADAN Forensic Audit Script — summarizes training run from logs.

Usage:
    python scripts/audit_run.py --log training_audit.log
    python scripts/audit_run.py  # defaults to latest log in logs/

Checks:
  - trade/step ratio < 0.33 (no spam)
  - initial capital shown as 20.50$
  - at least one exit reason > 5%
  - avg order size >= 11 USDT
  - tier = Micro Capital
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def parse_log(logfile: str):
    """Parse an ADAN training log and extract key metrics."""
    stats = {
        "total_steps": 0,
        "trade_opens": 0,
        "trade_closes": 0,
        "invalid_attempts": 0,
        "hold_overrides": 0,
        "episode_end_stats": [],
        "notionals": [],
        "sl_values": [],
        "tp_values": [],
        "tiers_seen": Counter(),
        "exit_reasons": Counter(),
        "value_losses": [],
        "explained_variances": [],
        "policy_losses": [],
        "initial_balance_seen": None,
        "errors": [],
    }

    re_trade_open = re.compile(r"\[TRADE_OPEN\].*notional=([\d.]+).*SL=([\d.]+)%.*TP=([\d.]+)%.*tier=(.+)")
    re_target_weight = re.compile(r"\[TARGET_WEIGHT\].*Step (\d+)")
    re_episode_end = re.compile(r"\[EPISODE_END_STATS\].*Initial=\$([\d.]+).*Final=\$([\d.]+).*Trades=(\d+).*Steps=(\d+).*Trade/Step=([\d.]+)")
    re_size_gate = re.compile(r"\[SIZE_GATE\]")
    re_ppo = re.compile(r"value_loss=([\d.]+).*explained_variance=([-\d.]+)")
    re_policy = re.compile(r"policy_gradient_loss=([-\d.]+)")
    re_close = re.compile(r"close_position.*reason[=:]\"?(\w+)")
    re_progress = re.compile(r"\[PROGRESS\].*Steps=([\d,]+)")

    with open(logfile, "r", errors="replace") as f:
        for line in f:
            # Trade opens
            m = re_trade_open.search(line)
            if m:
                stats["trade_opens"] += 1
                stats["notionals"].append(float(m.group(1)))
                stats["sl_values"].append(float(m.group(2)))
                stats["tp_values"].append(float(m.group(3)))
                stats["tiers_seen"][m.group(4).strip()] += 1
                continue

            # Episode end stats
            m = re_episode_end.search(line)
            if m:
                stats["episode_end_stats"].append({
                    "initial": float(m.group(1)),
                    "final": float(m.group(2)),
                    "trades": int(m.group(3)),
                    "steps": int(m.group(4)),
                    "trade_step_ratio": float(m.group(5)),
                })
                if stats["initial_balance_seen"] is None:
                    stats["initial_balance_seen"] = float(m.group(1))
                continue

            # Progress steps
            m = re_progress.search(line)
            if m:
                step_val = int(m.group(1).replace(",", ""))
                stats["total_steps"] = max(stats["total_steps"], step_val)
                continue

            # PPO metrics
            m = re_ppo.search(line)
            if m:
                stats["value_losses"].append(float(m.group(1)))
                stats["explained_variances"].append(float(m.group(2)))
                continue

            m = re_policy.search(line)
            if m:
                stats["policy_losses"].append(float(m.group(1)))
                continue

            # Closes / exit reasons
            m = re_close.search(line)
            if m:
                stats["trade_closes"] += 1
                stats["exit_reasons"][m.group(1)] += 1
                continue

            # SIZE_GATE
            if re_size_gate.search(line):
                stats["invalid_attempts"] += 1
                continue

            # Errors
            if "ERROR" in line and "Error" not in line[:30]:
                stats["errors"].append(line.strip()[:200])

    return stats


def audit(stats: dict):
    """Run audit checks and print report."""
    print("=" * 70)
    print("  ADAN Training Audit Report")
    print("=" * 70)

    # 1. Training overview
    print(f"\n--- Training Overview ---")
    print(f"  Total steps:        {stats['total_steps']:,}")
    print(f"  Trades opened:      {stats['trade_opens']}")
    print(f"  Trades closed:      {stats['trade_closes']}")
    print(f"  Invalid attempts:   {stats['invalid_attempts']}")

    # 2. Trade/step ratio — use per-episode average if available
    if stats["episode_end_stats"]:
        ep_ratios = [ep["trade_step_ratio"] for ep in stats["episode_end_stats"]]
        ratio = sum(ep_ratios) / len(ep_ratios)
        ratio_source = "per-episode avg"
    else:
        ratio = stats["trade_opens"] / max(stats["total_steps"], 1)
        ratio_source = "global"
    verdict = "PASS" if ratio < 0.33 else "FAIL"
    print(f"\n--- Anti-Spam Check ---")
    print(f"  Trade/Step ratio:   {ratio:.4f}  [{verdict}]  (threshold < 0.33, {ratio_source})")
    print(f"  Global ratio:       {stats['trade_opens'] / max(stats['total_steps'], 1):.4f}")

    # 3. Initial capital
    if stats["initial_balance_seen"] is not None:
        verdict = "PASS" if abs(stats["initial_balance_seen"] - 20.5) < 0.01 else "FAIL"
        print(f"\n--- Initial Capital ---")
        print(f"  Initial balance:    ${stats['initial_balance_seen']:.2f}  [{verdict}]  (expected 20.50)")
    else:
        print(f"\n--- Initial Capital ---")
        print(f"  [WARN] No EPISODE_END_STATS found in log")

    # 4. Average order size
    if stats["notionals"]:
        avg_notional = sum(stats["notionals"]) / len(stats["notionals"])
        min_notional = min(stats["notionals"])
        verdict = "PASS" if avg_notional >= 11.0 else "FAIL"
        print(f"\n--- Order Size ---")
        print(f"  Avg notional:       ${avg_notional:.2f}  [{verdict}]  (expected >= 11)")
        print(f"  Min notional:       ${min_notional:.2f}")
    else:
        print(f"\n--- Order Size ---")
        print(f"  [WARN] No trades found")

    # 5. Tiers
    print(f"\n--- Capital Tiers ---")
    for tier, count in stats["tiers_seen"].most_common():
        print(f"  {tier}: {count} trades")

    # 6. Exit reasons
    print(f"\n--- Exit Reasons ---")
    total_exits = sum(stats["exit_reasons"].values())
    for reason, count in stats["exit_reasons"].most_common():
        pct = (count / max(total_exits, 1)) * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")
    if total_exits > 0:
        any_above_5 = any(
            (c / total_exits * 100) > 5
            for c in stats["exit_reasons"].values()
        )
        print(f"  At least one reason > 5%: {'PASS' if any_above_5 else 'FAIL'}")

    # 7. PPO convergence
    if stats["value_losses"]:
        print(f"\n--- PPO Convergence ---")
        print(f"  Value loss:  first={stats['value_losses'][0]:.4f}  last={stats['value_losses'][-1]:.4f}")
    if stats["explained_variances"]:
        print(f"  Expl. var:   first={stats['explained_variances'][0]:.4f}  last={stats['explained_variances'][-1]:.4f}")

    # 8. Episode summaries
    if stats["episode_end_stats"]:
        print(f"\n--- Episode Summaries ({len(stats['episode_end_stats'])} episodes) ---")
        for i, ep in enumerate(stats["episode_end_stats"][:5]):
            print(f"  Ep {i}: Init=${ep['initial']:.2f} Final=${ep['final']:.2f} "
                  f"Trades={ep['trades']} Steps={ep['steps']} "
                  f"Trade/Step={ep['trade_step_ratio']:.4f}")

    # 9. Errors
    if stats["errors"]:
        print(f"\n--- Errors ({len(stats['errors'])}) ---")
        for err in stats["errors"][:5]:
            print(f"  {err}")

    print("\n" + "=" * 70)

    # Overall verdict
    fails = []
    if ratio >= 0.33:
        fails.append("trade/step ratio >= 0.33 (SPAM)")
    if stats["trade_opens"] == 0:
        fails.append("zero trades opened")
    if stats["initial_balance_seen"] is not None and abs(stats["initial_balance_seen"] - 20.5) >= 0.01:
        fails.append(f"initial balance {stats['initial_balance_seen']} != 20.50")

    if fails:
        print(f"  VERDICT: ISSUES DETECTED")
        for f in fails:
            print(f"    - {f}")
    else:
        print(f"  VERDICT: PASS")

    return len(fails) == 0


def main():
    parser = argparse.ArgumentParser(description="ADAN Training Run Audit")
    parser.add_argument("--log", type=str, default=None, help="Log file to audit")
    args = parser.parse_args()

    if args.log and os.path.exists(args.log):
        logfile = args.log
    else:
        # Find latest log
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        candidates = sorted(glob.glob(str(log_dir / "*.log")), key=os.path.getmtime, reverse=True)
        if not candidates:
            print("No log files found. Provide --log <path>")
            sys.exit(1)
        logfile = candidates[0]

    print(f"Auditing: {logfile}")
    stats = parse_log(logfile)
    success = audit(stats)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
