"""Central paths/config for the ADAN0 Terminal backend.

Source of truth = the repo files produced by the training harness.
All paths are resolved relative to the ADAN0 repo root so the backend works
regardless of the CWD uvicorn is launched from.
"""
from __future__ import annotations

import os
from pathlib import Path

# .../ADAN0/web/backend/app/settings.py -> repo root is parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]

LOGS_DIR = REPO_ROOT / "logs" / "training"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def _latest(glob_pattern: str, fallback: str, exclude_substrings: tuple[str, ...] = ()) -> str:
    """Return the newest (by mtime) file matching glob_pattern in LOGS_DIR,
    skipping any path containing one of exclude_substrings. Falls back to
    `fallback` if nothing matches. This makes the dashboard track the LIVE /
    most-recent run instead of a hardcoded stale V4 file (the reason the site
    kept showing metrics with nothing running)."""
    try:
        cands = [
            p for p in LOGS_DIR.glob(glob_pattern)
            if p.is_file() and not any(x in p.name for x in exclude_substrings)
        ]
        if not cands:
            return fallback
        newest = max(cands, key=lambda p: p.stat().st_mtime)
        return str(newest)
    except Exception:
        return fallback


# Active run artifacts. Default = newest matching file (auto-tracks the live
# run). Explicitly overridable via env var. We exclude tiny *smoke* / *val300*
# validation files and *ARCHIVE*/*ECHEC* so the dashboard locks onto a real
# long run; if only those exist they still win via the fallback list ordering.
TRAIN_LOG = Path(os.environ.get(
    "ADAN_WEB_TRAIN_LOG",
    _latest("train_*_500k.log", str(LOGS_DIR / "train_v4_500k.log"),
            exclude_substrings=("web_scalper",)),
))
TELEMETRY_CSV = Path(os.environ.get(
    "ADAN_WEB_TELEMETRY_CSV",
    _latest("diagnostic_collapse_*.csv", str(LOGS_DIR / "diagnostic_collapse_v4.csv"),
            exclude_substrings=("ARCHIVE", "ECHEC", "_smoke", "_val300")),
))

# obs_schema version the live training uses (exposed to the frontend so the
# dashboard can show a Schema/Version panel and flag v1(20)/v2(28) runs).
OBS_SCHEMA_VERSION = os.environ.get("ADAN_WEB_OBS_SCHEMA", "obs_schema_v2")
OBS_PORTFOLIO_DIM = int(os.environ.get("ADAN_WEB_OBS_PORTFOLIO_DIM", "28"))

# How the training process is identified in `ps`.
TRAIN_PROCESS_MATCH = "train_parallel_agents.py"

# Collapse decision thresholds (FACTS-based, from docs/web_interface.md).
COLLAPSE_A0_STD_WARN = 3.0      # a0_std above this = warning
COLLAPSE_A0_STD_CRIT = 6.0      # a0_std above this = critical (V3 hit 13.48)
COLLAPSE_HOLD_PCT_WARN = 0.05   # req_HOLD_pct below this = warning
COLLAPSE_ILLEGAL_WARN = 0.95    # illegal_ratio above this = warning

TARGET_TOTAL_STEPS = int(os.environ.get("ADAN_WEB_TARGET_STEPS", "500000"))
