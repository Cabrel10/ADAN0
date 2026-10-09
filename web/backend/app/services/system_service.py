"""System + training-process status (CPU/RAM/swap/PID)."""
from __future__ import annotations

import time
from typing import Any

import psutil

from .. import settings

_LAST = {"ts": 0.0, "timestep": None}


def training_process() -> dict[str, Any]:
    """Find the running sandbox training process, if any."""
    match = settings.TRAIN_PROCESS_MATCH
    for proc in psutil.process_iter(["pid", "name", "cmdline", "cpu_percent",
                                     "memory_percent", "create_time"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if match in cmd and "grep" not in cmd:
            ct = proc.info.get("create_time") or time.time()
            return {
                "running": True,
                "pid": proc.info["pid"],
                "cpu_percent": proc.info.get("cpu_percent"),
                "memory_percent": round(proc.info.get("memory_percent") or 0, 1),
                "elapsed_sec": int(time.time() - ct),
            }
    return {"running": False, "pid": None}


def system_stats() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.0),
        "cpu_count": psutil.cpu_count(),
        "mem_total_gb": round(vm.total / 1e9, 2),
        "mem_used_gb": round(vm.used / 1e9, 2),
        "mem_available_gb": round(vm.available / 1e9, 2),
        "mem_percent": vm.percent,
        "swap_total_gb": round(sw.total / 1e9, 2),
        "swap_used_gb": round(sw.used / 1e9, 2),
        "swap_percent": sw.percent,
    }
