"""Training control: launch / stop runs, edit safe hyperparameters, pick worker.

HARD SAFETY RULES:
- The fee keys (commission, round_trip_fees) are NEVER writable. Any attempt is
  rejected. Fees stay at 0.5% (commission 0.0025, round_trip_fees 0.005).
- Hyperparameters are passed to the training process via environment variables
  (the sandbox harness already reads ADAN_ENT_COEF, ADAN_CKPT_FREQ, etc.),
  NOT by rewriting config.yaml. This avoids corrupting the canonical config.
- Launch uses the exact same command shape as the proven 500k V4 run.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil

from .. import settings

CONDA_PY = ("/home/ubuntu/webapp/MORNINGSTAR/miniconda3/envs/"
            "trading_env/bin/python")
TRAIN_SCRIPT = "scripts/train_parallel_agents.py"

# Hyperparameters the UI may set, mapped to the env vars the harness reads.
# Only a safe whitelist; fees are intentionally absent.
HYPERPARAM_ENV = {
    "ent_coef": "ADAN_ENT_COEF",
    "ckpt_freq": "ADAN_CKPT_FREQ",
    "diag_every": "ADAN_DIAG_EVERY",
}

FORBIDDEN_KEYS = {"commission", "round_trip_fees", "fee", "fees"}

WORKERS = ["scalper", "w1", "w2", "w3"]


def _train_proc() -> psutil.Process | None:
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if settings.TRAIN_PROCESS_MATCH in cmd and "grep" not in cmd:
            return p
    return None


def status() -> dict[str, Any]:
    p = _train_proc()
    if not p:
        return {"running": False}
    try:
        return {
            "running": True,
            "pid": p.pid,
            "cmdline": " ".join(p.cmdline()),
            "cpu_percent": p.cpu_percent(interval=0.0),
            "memory_percent": round(p.memory_percent(), 1),
        }
    except psutil.NoSuchProcess:
        return {"running": False}


def stop() -> dict[str, Any]:
    p = _train_proc()
    if not p:
        return {"ok": True, "message": "no training process running"}
    pid = p.pid
    try:
        p.send_signal(signal.SIGTERM)
        time.sleep(3)
        if p.is_running():
            p.send_signal(signal.SIGKILL)
        return {"ok": True, "message": f"stopped PID {pid}"}
    except psutil.NoSuchProcess:
        return {"ok": True, "message": f"PID {pid} already gone"}


def validate_hyperparams(params: dict[str, Any]) -> tuple[bool, str]:
    for k in params:
        kl = str(k).lower()
        if kl in FORBIDDEN_KEYS or "fee" in kl or "commission" in kl:
            return False, f"Refused: '{k}' touches fees (LOCKED at 0.5%)."
        if k not in HYPERPARAM_ENV:
            return False, f"Refused: '{k}' not in safe whitelist {list(HYPERPARAM_ENV)}."
    return True, "ok"


def launch(steps: int, worker: str, hyperparams: dict[str, Any],
           diag: bool = True) -> dict[str, Any]:
    if _train_proc():
        return {"ok": False, "message": "a training run is already active; stop it first."}

    if worker not in WORKERS:
        return {"ok": False, "message": f"unknown worker '{worker}', allowed {WORKERS}"}

    ok, msg = validate_hyperparams(hyperparams)
    if not ok:
        return {"ok": False, "message": msg}

    steps = int(max(100, min(steps, 2_000_000)))
    profile = "scalper" if worker == "scalper" else worker

    env = dict(os.environ)
    env.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    })
    for k, v in hyperparams.items():
        env[HYPERPARAM_ENV[k]] = str(v)
    if diag:
        env["ADAN_DIAG_COLLAPSE"] = "1"
        env.setdefault("ADAN_DIAG_EVERY", "5000")
        # Write to a fresh timestamped diag file so a web-launched run does NOT
        # clobber the currently-displayed telemetry; the dashboard auto-tracks
        # the newest file via settings.resolve_telemetry_csv().
        env["ADAN_DIAG_CSV"] = str(
            settings.LOGS_DIR / f"diag_web_{worker}_{int(time.time())}_500k.csv"
        )
    env.setdefault("ADAN_CKPT_FREQ", "10000")

    log_path = settings.LOGS_DIR / f"train_web_{worker}_{int(time.time())}.log"
    cmd = ["nice", "-n", "19", "taskset", "-c", "1-3", CONDA_PY, TRAIN_SCRIPT,
           "--mode", "sandbox", "--steps", str(steps),
           "--profiles", profile, "--config", "config/config.yaml"]

    logf = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, cwd=str(settings.REPO_ROOT), env=env,
        stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {
        "ok": True,
        "pid": proc.pid,
        "steps": steps,
        "worker": worker,
        "hyperparams": hyperparams,
        "log": str(log_path.relative_to(settings.REPO_ROOT)),
        "message": f"launched {worker} for {steps} steps (fees LOCKED).",
    }
