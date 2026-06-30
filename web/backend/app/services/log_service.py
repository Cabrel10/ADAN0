"""Parses the SB3 training stdout log (train_v4_500k.log)."""
from __future__ import annotations

import re
from collections import deque
from typing import Any

from .. import settings

_TS_RE = re.compile(r"total_timesteps\s*\|\s*(\d+)")
_ERR_RE = re.compile(r"Error in step|NameError|Traceback|not defined", re.IGNORECASE)
_CKPT_RE = re.compile(r"checkpoint.*?(\d+)_steps", re.IGNORECASE)


def tail_lines(n: int = 200) -> list[str]:
    path = settings.TRAIN_LOG
    if not path.exists():
        return []
    dq: deque[str] = deque(maxlen=n)
    try:
        with path.open("r", errors="replace") as f:
            for line in f:
                dq.append(line.rstrip("\n"))
    except Exception:
        return []
    return list(dq)


# How many bytes from the end of the log to scan. Training logs can reach
# 80+ MB; reading the whole file on every poll took ~25 s. The latest
# total_timesteps and any recent error live near the tail, so we seek there.
_TAIL_BYTES = 2 * 1024 * 1024  # 2 MiB


def _read_tail_bytes(path, n: int) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > n:
                f.seek(size - n)
                # drop the partial first line
                f.readline()
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def parse_progress() -> dict[str, Any]:
    """Latest total_timesteps + error state, scanning only the log TAIL.

    Bounded to the last few MiB so it stays O(1) regardless of total log size
    (logs can be 80+ MB). The latest progress line and recent errors are always
    at the end of the file.
    """
    path = settings.TRAIN_LOG
    info: dict[str, Any] = {
        "last_timestep": None,
        "has_errors": False,
        "error_count": 0,
        "last_error": None,
        "log_exists": path.exists(),
    }
    if not path.exists():
        return info
    last_ts = None
    err_count = 0
    last_err = None
    tail = _read_tail_bytes(path, _TAIL_BYTES)
    for line in tail.splitlines():
        m = _TS_RE.search(line)
        if m:
            last_ts = int(m.group(1))
        if _ERR_RE.search(line):
            err_count += 1
            last_err = line.strip()[:300]
    info["last_timestep"] = last_ts
    info["error_count"] = err_count
    info["has_errors"] = err_count > 0
    info["last_error"] = last_err
    info["scan"] = "tail"
    return info
