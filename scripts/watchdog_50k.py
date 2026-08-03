#!/usr/bin/env python3
"""watchdog_50k.py — V15 diagnostic-run supervisor.

Strategy pivot (Session 9): the behavioral collapse (pct_buy > 0.85 @ ~15k)
PRECEDES the numeric explosion (a0_mean > 5 @ ~233k). The drama is fully
resolved inside the first ~50k steps, so a 50k run is sufficient to (a) prove
the L2 loss anchor keeps a0_mean near 0 and (b) read the Critic's mind via the
adv_BUY / adv_SELL probes injected into WorldModelPPO.train().

This watchdog:
  1. Launches the PPO training subprocess (50k steps) with the anchor + diag
     env-vars set (unless --no-launch, in which case it only monitors).
  2. Every POLL seconds, reads the DiagnosticCollapseCallback CSV and prints a
     one-line statistical summary: step, a0_mean, a0_std, pct_buy, req_SELL.
  3. Tails the SB3 stdout log for the [ANCHOR_DEBUG] line (adv_BUY vs adv_SELL).
  4. ALERTS (does NOT kill by default) if a0_mean > 1.0 OR (pct_buy > 0.85 while
     step < 50k) — the early-collapse signature. --kill-on-collapse turns the
     alert into a hard SIGTERM.
  5. Stops ONLY when the target step is reached or the process dies.
  6. On exit, prints the mean adv_BUY vs adv_SELL parsed from the log so the
     root-cause verdict (reward/critic biased?) is immediate.

Usage:
    python scripts/watchdog_50k.py                 # launch + monitor 50k
    python scripts/watchdog_50k.py --steps 500000  # full run
    python scripts/watchdog_50k.py --no-launch     # monitor an existing run
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = "/home/ubuntu/webapp/MORNINGSTAR/miniconda3/envs/trading_env/bin/python"

ANCHOR_RE = re.compile(
    r"\[ANCHOR_DEBUG\].*a0_mean=(?P<a0m>[-\d.]+).*a0_std=(?P<a0s>[-\d.]+).*"
    r"anchor=(?P<anc>[-\dna.]+).*adv_BUY=(?P<advb>[-\dna.]+).*"
    r"adv_SELL=(?P<advs>[-\dna.]+).*adv_HOLD=(?P<advh>[-\dna.]+)"
)


def _read_last_csv_row(path: Path):
    """Return the last data row of the diag CSV as a dict, or None."""
    if not path.exists():
        return None
    try:
        with path.open("r", newline="") as fh:
            rows = list(csv.DictReader(fh))
        return rows[-1] if rows else None
    except Exception:
        return None


def _f(row, key, default=float("nan")):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _tail_anchor(log_path: Path, max_lines: int = 4000):
    """Scan the tail of the log for the most recent [ANCHOR_DEBUG] matches.

    Returns (last_match_dict_or_None, [all adv_BUY floats], [all adv_SELL floats]).
    """
    if not log_path.exists():
        return None, [], []
    try:
        with log_path.open("r", errors="replace") as fh:
            lines = fh.readlines()[-max_lines:]
    except Exception:
        return None, [], []
    last = None
    advb, advs = [], []
    for ln in lines:
        m = ANCHOR_RE.search(ln)
        if not m:
            continue
        d = m.groupdict()
        last = d
        try:
            b = float(d["advb"]); advb.append(b)
        except ValueError:
            pass
        try:
            s = float(d["advs"]); advs.append(s)
        except ValueError:
            pass
    return last, advb, advs


def main():
    ap = argparse.ArgumentParser(description="V15 diagnostic watchdog")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--poll", type=int, default=15, help="poll interval (s)")
    ap.add_argument("--lambda-anchor", type=float, default=0.05)
    ap.add_argument("--diag-every", type=int, default=1000)
    ap.add_argument("--tag", type=str, default="v15_50k")
    ap.add_argument("--no-launch", action="store_true",
                    help="monitor only, do not spawn the trainer")
    ap.add_argument("--kill-on-collapse", action="store_true",
                    help="SIGTERM the run when the collapse signature trips")
    args = ap.parse_args()

    log_dir = PROJECT_ROOT / "logs" / "training"
    log_dir.mkdir(parents=True, exist_ok=True)
    diag_csv = log_dir / f"diag_{args.tag}.csv"
    run_log = log_dir / f"{args.tag}.log"

    proc = None
    if not args.no_launch:
        # Start fresh: remove a stale diag CSV so we read only this run's rows.
        if diag_csv.exists():
            diag_csv.unlink()
        env = dict(os.environ)
        env["ADAN_L2_ANCHOR_LAMBDA"] = str(args.lambda_anchor)
        # Isolate the anchor: no forward-prediction MSE confound during the
        # diagnostic (aux_loss_coef=0 -> WorldModelPPO skips the aux step).
        env.setdefault("ADAN_AUX_LOSS_COEF", "0.0")
        env["ADAN_DIAG_COLLAPSE"] = "1"
        env["ADAN_DIAG_EVERY"] = str(args.diag_every)
        env["ADAN_DIAG_CSV"] = str(diag_csv)
        cmd = [
            PY, str(PROJECT_ROOT / "scripts" / "train_parallel_agents.py"),
            "--mode", "sandbox", "--steps", str(args.steps), "--no-subproc",
        ]
        print(f"[WATCHDOG] launching: {' '.join(cmd)}", flush=True)
        print(f"[WATCHDOG] lambda={args.lambda_anchor} diag_csv={diag_csv}",
              flush=True)
        lf = run_log.open("w")
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env,
                                stdout=lf, stderr=subprocess.STDOUT)
        print(f"[WATCHDOG] PID={proc.pid} log={run_log}", flush=True)

    t0 = time.time()
    last_step = -1
    alert_count = 0
    while True:
        time.sleep(args.poll)
        elapsed = time.time() - t0

        # ---- process liveness ----
        proc_dead = False
        if proc is not None:
            rc = proc.poll()
            if rc is not None:
                proc_dead = True

        # ---- read diag CSV ----
        row = _read_last_csv_row(diag_csv)
        anchor, advb_all, advs_all = _tail_anchor(run_log)

        if row is not None:
            step = int(_f(row, "timesteps", -1))
            a0m = _f(row, "a0_mean")
            a0s = _f(row, "a0_std")
            pbuy = _f(row, "a0_pct_buy")
            rsell = _f(row, "req_SELL_pct")
            ent = _f(row, "policy_entropy")
            histo = row.get("a0_histo", "")

            anc_str = ""
            if anchor is not None:
                anc_str = (f" | adv_BUY={anchor.get('advb')} "
                           f"adv_SELL={anchor.get('advs')} "
                           f"anchor={anchor.get('anc')}")

            marker = ""
            collapse_sig = (a0m > 1.0) or (pbuy > 0.85 and step < args.steps)
            if collapse_sig:
                marker = "  <<< ALERT collapse-signature"
                alert_count += 1

            print(
                f"[WATCHDOG t={elapsed:6.0f}s] step={step:>7} "
                f"a0_mean={a0m:+.4f} a0_std={a0s:.4f} pct_buy={pbuy:.3f} "
                f"req_SELL={rsell:.3f} ent={ent:.3f} histo={histo}"
                f"{anc_str}{marker}",
                flush=True,
            )

            if collapse_sig and args.kill_on_collapse and proc is not None:
                print(f"[WATCHDOG] collapse signature + --kill-on-collapse "
                      f"-> SIGTERM PID {proc.pid}", flush=True)
                proc.send_signal(signal.SIGTERM)

            if step >= args.steps and step != last_step:
                print(f"[WATCHDOG] target {args.steps} reached (step={step}).",
                      flush=True)
                break
            last_step = step
        else:
            print(f"[WATCHDOG t={elapsed:6.0f}s] waiting for diag CSV "
                  f"({diag_csv.name}) ... (proc_dead={proc_dead})", flush=True)

        if proc_dead:
            print(f"[WATCHDOG] training process exited rc={proc.poll()}.",
                  flush=True)
            break

    # ---- final verdict digest ----
    _, advb_all, advs_all = _tail_anchor(run_log, max_lines=200000)
    print("\n" + "=" * 66, flush=True)
    print("[WATCHDOG] RUN FINISHED — Critic probe digest (Q1/Q2)", flush=True)
    if advb_all:
        mb = sum(advb_all) / len(advb_all)
        print(f"  mean(adv_BUY)  over {len(advb_all):4d} updates = {mb:+.5f}",
              flush=True)
    if advs_all:
        ms = sum(advs_all) / len(advs_all)
        print(f"  mean(adv_SELL) over {len(advs_all):4d} updates = {ms:+.5f}",
              flush=True)
    if advb_all and advs_all:
        mb = sum(advb_all) / len(advb_all)
        ms = sum(advs_all) / len(advs_all)
        print(f"  DELTA adv_BUY - adv_SELL = {mb - ms:+.5f}", flush=True)
        if mb - ms > 0.05:
            print("  VERDICT HINT: adv_BUY >> adv_SELL -> Reward/Critic bias "
                  "toward BUY (root cause is BEFORE the actor).", flush=True)
        else:
            print("  VERDICT HINT: adv_BUY ~ adv_SELL -> Critic is NOT the "
                  "driver; look at actor / std / geometry.", flush=True)
    print(f"  total alerts fired: {alert_count}", flush=True)
    print("=" * 66, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WATCHDOG] interrupted by user.", flush=True)
        sys.exit(1)
