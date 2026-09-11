#!/usr/bin/env python3
"""
ÉTAPE 5 + ÉTAPE 7 monitor.

Reads the REAL SB3 PPO counter `total_timesteps` (NOT the env "Starting step"
chunk counter which caused the "5-6k steps" confusion in V30) from a live
training log and evaluates the V31 gate thresholds over the 5k-20k window.

GATES (mandat ÉTAPE 7):
  clip_fraction < 0.30
  approx_kl     < 0.15
  std           in [0.35, 0.40]
  entropy_loss  > -3.0
HARD STOP: clip_fraction > 0.5 in the window -> print STOP and exit 2.

Usage:
  python3 monitor_v31.py <logfile>                 # one-shot snapshot
  python3 monitor_v31.py <logfile> --window 5000 20000
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_ppo_tables import parse  # reuse the vetted parser

HARD_STOP_CLIP = 0.5


def check_row(u):
    """Return (ok, list of failed gate strings) for one PPO update."""
    fails = []
    cf = u.get("clip_fraction")
    kl = u.get("approx_kl")
    std = u.get("std")
    ent = u.get("entropy_loss")
    if cf is not None and cf >= 0.30:
        fails.append(f"clip_fraction={cf:.3f}>=0.30")
    if kl is not None and kl >= 0.15:
        fails.append(f"approx_kl={kl:.4f}>=0.15")
    if std is not None and not (0.35 <= std <= 0.40):
        fails.append(f"std={std:.3f} outside[0.35,0.40]")
    if ent is not None and ent <= -3.0:
        fails.append(f"entropy_loss={ent:.2f}<=-3.0")
    return (len(fails) == 0), fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("--window", nargs=2, type=float, default=[5000, 20000],
                    metavar=("LO", "HI"))
    args = ap.parse_args()

    if not os.path.exists(args.logfile):
        print(f"[monitor] log not found yet: {args.logfile}")
        sys.exit(0)

    ups = parse(args.logfile)
    lo, hi = args.window
    if not ups:
        print(f"[monitor] no PPO tables yet in {args.logfile} "
              f"(training may still be in the first rollout)")
        sys.exit(0)

    latest = ups[-1]
    print(f"[monitor] latest total_timesteps = {latest.get('total_timesteps', '?'):.0f} "
          f"| {len(ups)} PPO updates parsed")
    print(f"[monitor] latest: clip={latest.get('clip_fraction', float('nan')):.3f} "
          f"kl={latest.get('approx_kl', float('nan')):.4f} "
          f"std={latest.get('std', float('nan')):.3f} "
          f"ent={latest.get('entropy_loss', float('nan')):.2f} "
          f"lr={latest.get('learning_rate', float('nan')):.2e} "
          f"ev={latest.get('explained_variance', float('nan')):.3f}")

    win = [u for u in ups if lo <= u.get("total_timesteps", -1) <= hi]
    print(f"\n[monitor] ETAPE 7 window [{lo:.0f}, {hi:.0f}] -> {len(win)} updates")

    hard_stop = False
    any_fail = False
    for u in win:
        cf = u.get("clip_fraction")
        if cf is not None and cf > HARD_STOP_CLIP:
            hard_stop = True
        ok, fails = check_row(u)
        if not ok:
            any_fail = True
            print(f"  ts={u.get('total_timesteps', 0):.0f} FAIL: {', '.join(fails)}")

    if hard_stop:
        print("\n*** HARD STOP: clip_fraction > 0.5 in window. "
              "Do NOT stack a 2nd change (target_kl/n_epochs) before closing "
              "whether LR alone was insufficient. ***")
        sys.exit(2)
    if win and not any_fail:
        print("  ALL GATES PASS in window.")
    elif not win:
        print("  (window not yet reached)")
    sys.exit(0)


if __name__ == "__main__":
    main()
