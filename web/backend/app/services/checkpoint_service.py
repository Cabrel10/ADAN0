"""Lists training checkpoints from checkpoints/*.zip."""
from __future__ import annotations

import re
from typing import Any

from .. import settings

_STEP_RE = re.compile(r"checkpoint_(\d+)_steps")
_STEP_RE2 = re.compile(r"_(\d+)steps")


def _extract_step(name: str) -> int | None:
    m = _STEP_RE.search(name)
    if m:
        return int(m.group(1))
    m = _STEP_RE2.search(name)
    if m:
        return int(m.group(1))
    return None


def list_checkpoints() -> list[dict[str, Any]]:
    d = settings.CHECKPOINTS_DIR
    if not d.exists():
        return []
    out: list[dict[str, Any]] = []
    for p in d.glob("*.zip"):
        try:
            st = p.stat()
        except OSError:
            continue
        out.append({
            "name": p.name,
            "step": _extract_step(p.name),
            "size_bytes": st.st_size,
            "size_mb": round(st.st_size / 1e6, 2),
            "mtime": st.st_mtime,
        })
    # newest first
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out
