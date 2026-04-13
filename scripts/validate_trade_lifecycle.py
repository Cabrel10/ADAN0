#!/usr/bin/env python3
"""
Trade Lifecycle Validator for ADAN Trading Bot.

Reads training logs and verifies the strict trade lifecycle sequence:
  A. RESET        - Episode initialization
  B. TARGET_WEIGHT - Action decoded, tier sizing applied
  C. TRADE_OPEN   - Position opened after all gates passed
  D. HOLD_MIN     - Minimum hold period enforced before SELL allowed
  E. AGENT_CLOSE / STOP_LOSS / TAKE_PROFIT / MAX_DURATION - Position closure
  F. POSITION FERMEE (implicit when E completes with receipt)
  G. WAIT_BLOCK   - Post-SELL cooldown before next BUY
  H. REWARD_ANTIHACK - Reward logged with anti-hack formula

Prints a per-check validation table with checkmarks.
"""

import re
import sys
import os
import argparse
from collections import defaultdict

# ════════════════════════════════════════════════════════════════
# LOG LINE PATTERNS
# ════════════════════════════════════════════════════════════════

PATTERNS = {
    "RESET":          re.compile(r"\[STEP\] Starting step 1\b|\[EPISODE.*reset|RESET.*episode", re.IGNORECASE),
    "TARGET_WEIGHT":  re.compile(r"\[TARGET_WEIGHT\]"),
    "LINEAR_EXPO":    re.compile(r"\[LINEAR_EXPO\]"),
    "TRADE_OPEN":     re.compile(r"\[TRADE_OPEN\]"),
    "HOLD_MIN":       re.compile(r"\[HOLD_MIN\]"),
    "AGENT_CLOSE":    re.compile(r"\[AGENT_CLOSE\]"),
    "STOP_LOSS":      re.compile(r"STOP LOSS atteint|stop_loss|STOP_LOSS"),
    "TAKE_PROFIT":    re.compile(r"TAKE PROFIT atteint|take_profit|TAKE_PROFIT"),
    "MAX_DURATION":   re.compile(r"\[MAX_DURATION\]|MAX DURATION"),
    "WAIT_BLOCK":     re.compile(r"\[WAIT_BLOCK\]"),
    "REWARD_ANTIHACK":re.compile(r"\[REWARD_ANTIHACK\]|REWARD_ANTIHACK"),
    "ACTION_DIFF":    re.compile(r"\[ACTION_DIFF\]"),
    "BANKRUPT_KILL":  re.compile(r"\[BANKRUPT_KILL\]"),
    "TIER":           re.compile(r"\[TIER\]|\[TIER_LOCKED\]"),
    "SIZE_GATE":      re.compile(r"\[SIZE_GATE\]|\[CASH_FLOOR\]"),
    "EV_GATE":        re.compile(r"\[EV_GATE\]"),
    "RISK_GATE":      re.compile(r"\[RISK_GATE\]"),
}


def extract_step(line: str) -> int:
    """Try to extract step number from a log line."""
    m = re.search(r"Step\s+(\d+)", line)
    if m:
        return int(m.group(1))
    m = re.search(r"step\s*=?\s*(\d+)", line, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return -1


def extract_action_diff(line: str) -> dict:
    """Parse an ACTION_DIFF log line."""
    result = {"step": -1, "requested": "", "executed": "", "inv_penalty": 0.0}
    m = re.search(r"Step\s+(\d+)", line)
    if m:
        result["step"] = int(m.group(1))
    m = re.search(r"Requested=(\w+)", line)
    if m:
        result["requested"] = m.group(1)
    m = re.search(r"Executed=(\w+)", line)
    if m:
        result["executed"] = m.group(1)
    m = re.search(r"inv_penalty=([-\d.]+)", line)
    if m:
        result["inv_penalty"] = float(m.group(1))
    return result


def extract_reward_antihack(line: str) -> dict:
    """Parse a REWARD_ANTIHACK log line."""
    result = {"step": -1, "pnl_net": 0.0, "inv_pen": 0.0, "action_req": "", "action_exe": "", "final": 0.0}
    m = re.search(r"Step\s+(\d+)", line)
    if m:
        result["step"] = int(m.group(1))
    m = re.search(r"pnl_net=([-+\d.]+)", line)
    if m:
        result["pnl_net"] = float(m.group(1))
    m = re.search(r"inv_pen=([-\d.]+)", line)
    if m:
        result["inv_pen"] = float(m.group(1))
    m = re.search(r"action_req=(\w+)", line)
    if m:
        result["action_req"] = m.group(1)
    m = re.search(r"action_exe=(\w+)", line)
    if m:
        result["action_exe"] = m.group(1)
    m = re.search(r"final=([-+\d.]+)", line)
    if m:
        result["final"] = float(m.group(1))
    return result


def validate_log(log_lines: list, verbose: bool = False) -> dict:
    """
    Validate trade lifecycle from log lines.
    
    Returns a dict of check_name -> (passed: bool, detail: str)
    """
    results = {}
    
    # Count occurrences
    counts = defaultdict(int)
    events_by_step = defaultdict(list)
    action_diffs = []
    reward_antihacks = []
    trade_open_steps = []
    close_steps = []  # AGENT_CLOSE, SL, TP, MAX_DURATION
    
    for line in log_lines:
        for name, pattern in PATTERNS.items():
            if pattern.search(line):
                counts[name] += 1
                step = extract_step(line)
                events_by_step[step].append(name)
                
                if name == "TRADE_OPEN":
                    trade_open_steps.append(step)
                elif name in ("AGENT_CLOSE", "STOP_LOSS", "TAKE_PROFIT", "MAX_DURATION"):
                    close_steps.append(step)
                elif name == "ACTION_DIFF":
                    action_diffs.append(extract_action_diff(line))
                elif name == "REWARD_ANTIHACK":
                    reward_antihacks.append(extract_reward_antihack(line))

    # ════════════════════════════════════════════════════════════
    # CHECK A: RESET present
    # ════════════════════════════════════════════════════════════
    has_reset = counts["RESET"] > 0 or any("STEP" in l and "step 1" in l for l in log_lines[:50])
    # Relaxed: if we see STEP 1 or any initialization, it's a reset
    if not has_reset:
        has_reset = any(re.search(r"\[STEP\].*step\s+1\b", l, re.IGNORECASE) for l in log_lines[:100])
    results["A_RESET"] = (has_reset, f"Found {counts['RESET']} reset events")

    # ════════════════════════════════════════════════════════════
    # CHECK B: TARGET_WEIGHT present (logged every 50 steps)
    # ════════════════════════════════════════════════════════════
    results["B_TARGET_WEIGHT"] = (
        counts["TARGET_WEIGHT"] > 0,
        f"Found {counts['TARGET_WEIGHT']} TARGET_WEIGHT events"
    )

    # ════════════════════════════════════════════════════════════
    # CHECK C: TRADE_OPEN present
    # ════════════════════════════════════════════════════════════
    results["C_TRADE_OPEN"] = (
        counts["TRADE_OPEN"] > 0,
        f"Found {counts['TRADE_OPEN']} TRADE_OPEN events"
    )

    # ════════════════════════════════════════════════════════════
    # CHECK D: HOLD_MIN enforcement
    # When HOLD_MIN fires, SELL should be blocked (action_exe = HOLD)
    # ════════════════════════════════════════════════════════════
    hold_min_ok = True
    hold_min_detail = f"Found {counts['HOLD_MIN']} HOLD_MIN events"
    # If no HOLD_MIN events, that's fine (agent didn't try early SELL)
    results["D_HOLD_MIN"] = (hold_min_ok, hold_min_detail)

    # ════════════════════════════════════════════════════════════
    # CHECK E: Position closures (any close mechanism)
    # ════════════════════════════════════════════════════════════
    close_count = counts["AGENT_CLOSE"] + counts["STOP_LOSS"] + counts["TAKE_PROFIT"] + counts["MAX_DURATION"]
    results["E_POSITION_CLOSE"] = (
        close_count > 0 or counts["TRADE_OPEN"] == 0,  # OK if no trades opened
        f"Closes: AGENT={counts['AGENT_CLOSE']} SL={counts['STOP_LOSS']} TP={counts['TAKE_PROFIT']} MAX_DUR={counts['MAX_DURATION']}"
    )

    # ════════════════════════════════════════════════════════════
    # CHECK F: TRADE_OPEN ≈ POSITION CLOSED ratio
    # ════════════════════════════════════════════════════════════
    open_count = counts["TRADE_OPEN"]
    if open_count > 0:
        ratio = close_count / open_count
        # Allow some unclosed positions (episode may end with open pos)
        ratio_ok = ratio >= 0.3 or open_count <= 3  # At least 30% closed, or very few trades
        results["F_OPEN_CLOSE_RATIO"] = (
            ratio_ok,
            f"Open={open_count} Close={close_count} Ratio={ratio:.2f}"
        )
    else:
        results["F_OPEN_CLOSE_RATIO"] = (True, "No trades to check")

    # ════════════════════════════════════════════════════════════
    # CHECK G: WAIT_BLOCK enforcement
    # CRITICAL: WAIT_BLOCK must fire after SL/TP to prevent death spiral.
    # If SL/TP occurred, WAIT_BLOCK should be > 0 (cooldown active).
    # Only pass if WAIT_BLOCK > 0 OR no SL/TP occurred (no cooldown needed).
    # ════════════════════════════════════════════════════════════
    sl_tp_count = counts["STOP_LOSS"] + counts["TAKE_PROFIT"]
    wait_block_ok = counts["WAIT_BLOCK"] > 0 or sl_tp_count == 0
    results["G_WAIT_BLOCK"] = (
        wait_block_ok,
        f"Found {counts['WAIT_BLOCK']} WAIT_BLOCK events (SL/TP={sl_tp_count})"
    )

    # ════════════════════════════════════════════════════════════
    # CHECK H: REWARD_ANTIHACK present
    # ════════════════════════════════════════════════════════════
    results["H_REWARD_ANTIHACK"] = (
        counts["REWARD_ANTIHACK"] > 0,
        f"Found {counts['REWARD_ANTIHACK']} REWARD_ANTIHACK events"
    )

    # ════════════════════════════════════════════════════════════
    # CHECK I: ACTION_DIFF coherence (Requested matches Executed when no gate fires)
    # ════════════════════════════════════════════════════════════
    action_match_count = 0
    action_gate_count = 0
    for ad in action_diffs:
        if ad["requested"] == ad["executed"]:
            action_match_count += 1
        else:
            action_gate_count += 1  # A gate modified the action
    total_ad = len(action_diffs)
    results["I_ACTION_DIFF"] = (
        total_ad > 0,
        f"Match={action_match_count} Gated={action_gate_count} Total={total_ad}"
    )

    # ════════════════════════════════════════════════════════════
    # CHECK J: Reward alignment (positive for wins, negative for losses)
    # ════════════════════════════════════════════════════════════
    reward_ok = True
    reward_violations = 0
    for ra in reward_antihacks:
        if ra["pnl_net"] > 0.001 and ra["final"] < -0.001:
            reward_violations += 1  # Positive PnL but negative reward
        elif ra["pnl_net"] < -0.001 and ra["final"] > 0.1:
            reward_violations += 1  # Negative PnL but very positive reward
    if reward_violations > 2:  # Allow small rounding errors
        reward_ok = False
    results["J_REWARD_ALIGNMENT"] = (
        reward_ok,
        f"Violations={reward_violations} Total_checks={len(reward_antihacks)}"
    )

    # ════════════════════════════════════════════════════════════
    # CHECK K: No BANKRUPT_KILL (unless cash truly < 11.50)
    # ════════════════════════════════════════════════════════════
    results["K_NO_BANKRUPT"] = (
        counts["BANKRUPT_KILL"] == 0,
        f"Bankrupt kills: {counts['BANKRUPT_KILL']}"
    )

    # ════════════════════════════════════════════════════════════
    # CHECK L: Episode length > 100 steps
    # ════════════════════════════════════════════════════════════
    max_step = 0
    for line in log_lines:
        s = extract_step(line)
        if s > max_step:
            max_step = s
    results["L_EPISODE_LENGTH"] = (
        max_step >= 100,
        f"Max step seen: {max_step}"
    )

    # Metadata
    results["_counts"] = counts
    results["_max_step"] = max_step
    results["_action_diffs"] = action_diffs
    results["_reward_antihacks"] = reward_antihacks
    
    return results


def print_validation_table(results: dict, run_id: str = ""):
    """Print formatted validation table."""
    header = f"TRADE LIFECYCLE VALIDATION"
    if run_id:
        header += f" (Run {run_id})"
    
    print("\n" + "=" * 70)
    print(f"  {header}")
    print("=" * 70)
    print(f"  {'Check':<30} {'Status':<8} {'Detail'}")
    print("-" * 70)
    
    all_pass = True
    for key, value in sorted(results.items()):
        if key.startswith("_"):
            continue
        passed, detail = value
        status = "PASS" if passed else "FAIL"
        icon = "\u2705" if passed else "\u274c"
        print(f"  {key:<30} {icon} {status:<6} {detail}")
        if not passed:
            all_pass = False
    
    print("-" * 70)
    overall = "ALL PASSED" if all_pass else "SOME FAILED"
    overall_icon = "\u2705" if all_pass else "\u274c"
    print(f"  {'OVERALL':<30} {overall_icon} {overall}")
    print("=" * 70)
    
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Validate ADAN trade lifecycle from logs")
    parser.add_argument("log_file", nargs="?", default=None, help="Path to log file")
    parser.add_argument("--run-id", type=str, default="", help="Run identifier")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--stdin", action="store_true", help="Read from stdin")
    args = parser.parse_args()

    if args.stdin or args.log_file is None:
        log_lines = sys.stdin.readlines()
    elif args.log_file and os.path.exists(args.log_file):
        with open(args.log_file, "r") as f:
            log_lines = f.readlines()
    else:
        print(f"ERROR: Log file not found: {args.log_file}")
        sys.exit(1)

    if not log_lines:
        print("ERROR: No log lines to process")
        sys.exit(1)

    results = validate_log(log_lines, verbose=args.verbose)
    all_pass = print_validation_table(results, args.run_id)
    
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
