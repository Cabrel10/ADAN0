#!/usr/bin/env python3
"""
monitor_runs.py — passive health snapshot for the two independent 500k ADAN runs.

Does NOT stop anything. It reports, per asset:
  - alive (process present) / DONE exit code
  - last cumulative_timesteps seen in the log
  - last PPO metrics (approx_kl, entropy_loss, explained_variance, fps)
  - last portfolio value / realized pnl
  - integrity flags: NaN, Traceback, collapse-breaker, critic-breaker
A run should only be STOPPED on a real technical/scientific failure
(NaN / corrupted checkpoint / data leak / crash / broken invariant),
never on an intermediate bad PnL.

Usage:
  python scripts/monitor_runs.py
"""
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs" / "exp5y"

RUNS = {
    "BTC": {"log": LOGS / "run500k_BTC.log", "json": LOGS / "run500k_BTC.json",
            "asset": "BTCUSDT_BINANCE", "ckpt": ROOT / "checkpoints" / "adan_BTC_500k"},
    "DOGE": {"log": LOGS / "run500k_DOGE.log", "json": LOGS / "run500k_DOGE.json",
             "asset": "DOGEUSDT_BINANCE", "ckpt": ROOT / "checkpoints" / "adan_DOGE_500k"},
}

# STOP-worthy patterns (real technical/scientific failures only).
# NaN detection targets ACTUAL runtime failures, not benign INFO logs that
# merely mention the word "NaN" (e.g. "prevent NaN from std=0",
# "leading NaN rows dropped", "SafeScalerWrapper"). We require the NaN mention
# to co-occur with an error/loss/reward context.
FATAL = [
    (re.compile(r"Traceback \(most recent call last\)"), "crash_traceback"),
    (re.compile(r"(loss|reward|action|gradient|value)\s*(is|=|:)?\s*nan", re.I), "nan_in_training"),
    (re.compile(r"nan.*(loss|reward|gradient|encountered in|not finite)", re.I), "nan_in_training"),
    (re.compile(r"COLLAPSE[_ ]?BREAKER.*TRIGGER", re.I), "collapse_breaker"),
    (re.compile(r"CRITIC[_ ]?BREAKER.*TRIGGER", re.I), "critic_breaker"),
    (re.compile(r"checkpoint.*corrupt", re.I), "checkpoint_corruption"),
]

def procs_alive():
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
    except Exception:
        return {}
    alive = {}
    for line in out.splitlines():
        for tag, cfg in RUNS.items():
            if "launch_asset_run.py" in line and f"--asset {cfg['asset']}" in line and "--steps 500000" in line:
                pid = line.strip().split()[0]
                alive[tag] = pid
    return alive

def tail(path, n=4000):
    if not path.exists():
        return ""
    data = path.read_text(errors="replace")
    return data[-n * 200:]  # cheap tail by chars

def last_match(text, pat):
    m = None
    for m in pat.finditer(text):
        pass
    return m

def snapshot(tag, cfg, alive):
    log_txt = tail(cfg["log"])
    rep = {"asset": cfg["asset"], "alive": tag in alive, "pid": alive.get(tag)}

    dm = re.search(r"DONE exit=(\d+)", log_txt)
    rep["done_exit"] = int(dm.group(1)) if dm else None

    # cumulative timesteps
    ts = re.findall(r"total_timesteps\s*\|\s*(\d+)", log_txt)
    rep["timesteps"] = int(ts[-1]) if ts else None
    # ppo metrics
    for key, rx in [("approx_kl", r"approx_kl\s*\|\s*([\d.eE+-]+)"),
                    ("entropy_loss", r"entropy_loss\s*\|\s*([\d.eE+-]+)"),
                    ("explained_var", r"explained_variance\s*\|\s*([\d.eE+-]+)"),
                    ("fps", r"fps\s*\|\s*([\d.eE+-]+)")]:
        vals = re.findall(rx, log_txt)
        rep[key] = vals[-1] if vals else None
    # trading
    pv = re.findall(r"Portfolio value:\s*([\d.eE+-]+)", log_txt)
    rep["portfolio_value"] = float(pv[-1]) if pv else None

    # integrity
    flags = []
    for pat, name in FATAL:
        m = last_match(log_txt, pat)
        if m:
            flags.append(name)
    rep["fatal_flags"] = flags
    rep["verdict"] = "STOP" if flags else ("DONE" if rep["done_exit"] == 0 else "RUNNING")
    # checkpoints present
    rep["checkpoints"] = sorted(p.name for p in cfg["ckpt"].glob("*.zip")) if cfg["ckpt"].exists() else []
    return rep

def main():
    alive = procs_alive()
    print("=" * 70)
    print("ADAN 500k — RUN MONITOR (report only; no auto-stop)")
    print("=" * 70)
    for tag, cfg in RUNS.items():
        r = snapshot(tag, cfg, alive)
        print(f"\n[{tag}] {r['asset']}  verdict={r['verdict']}")
        print(f"  alive={r['alive']} pid={r['pid']} done_exit={r['done_exit']}")
        print(f"  timesteps={r['timesteps']} fps={r['fps']} approx_kl={r['approx_kl']} "
              f"expl_var={r['explained_var']} ent={r['entropy_loss']}")
        print(f"  portfolio_value={r['portfolio_value']} checkpoints={r['checkpoints']}")
        if r["fatal_flags"]:
            print(f"  *** FATAL FLAGS: {r['fatal_flags']} -> STOP JUSTIFIED ***")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
