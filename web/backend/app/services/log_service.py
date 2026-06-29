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


def parse_progress() -> dict[str, Any]:
    """Scan the log for the latest total_timesteps and error state."""
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
    try:
        with path.open("r", errors="replace") as f:
            for line in f:
                m = _TS_RE.search(line)
                if m:
                    last_ts = int(m.group(1))
                if _ERR_RE.search(line):
                    err_count += 1
                    last_err = line.strip()[:300]
    except Exception:
        pass
    info["last_timestep"] = last_ts
    info["error_count"] = err_count
    info["has_errors"] = err_count > 0
    info["last_error"] = last_err
    return info
