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


def _latest(glob_patterns, fallback: str, exclude_substrings: tuple[str, ...] = ()) -> str:
    """Return the newest (by mtime) file matching ANY of glob_patterns in
    LOGS_DIR, skipping any path containing one of exclude_substrings. Falls
    back to `fallback` if nothing matches. This makes the dashboard track the
    LIVE / most-recent run instead of a hardcoded stale file (the reason the
    site kept showing metrics from an old run that was no longer training).

    `glob_patterns` may be a single pattern (str) or a list of patterns; all
    matches across patterns are pooled and the globally newest wins. This is
    required because the training harness naming convention drifted from
    ``diagnostic_collapse_vN_500k.csv`` / ``train_vN_500k.log`` to
    ``diag_<tag>_500k*.csv`` / ``train_<tag>_500k_<timestamp>.log`` and the
    old single-glob only matched the stale files.
    """
    if isinstance(glob_patterns, (str, bytes)):
        glob_patterns = [glob_patterns]
    try:
        cands = []
        seen = set()
        for pat in glob_patterns:
            for p in LOGS_DIR.glob(pat):
                if not p.is_file():
                    continue
                if any(x in p.name for x in exclude_substrings):
                    continue
                if p in seen:
                    continue
                seen.add(p)
                cands.append(p)
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
# Glob patterns + excludes for run-artifact selection. Defined here (above the
# import-time constants) so both the constants and the dynamic resolvers below
# share ONE source of truth. New convention:
#   train_<tag>_500k_<timestamp>.log  /  diag_<tag>_500k.csv
# Legacy convention:
#   train_vN_500k.log  /  diagnostic_collapse_*.csv
# Excludes drop archived / failed / smoke / validation / *_COLLAPSED_* frozen
# copies so the dashboard locks onto the primary live run file.
_TRAIN_LOG_GLOBS = ["train_*_500k_*.log", "train_*_500k.log"]
_TRAIN_LOG_EXCLUDE = ("web_scalper",)
_TELEMETRY_GLOBS = ["diag_*_500k.csv", "diag_*.csv", "diagnostic_collapse_*.csv"]
_TELEMETRY_EXCLUDE = (
    "ARCHIVE", "ECHEC", "_smoke", "_val300",
    "COLLAPSED", "FIXAC", "FINAL", "_iso",
)

TRAIN_LOG = Path(os.environ.get(
    "ADAN_WEB_TRAIN_LOG",
    _latest(_TRAIN_LOG_GLOBS, str(LOGS_DIR / "train_v4_500k.log"), _TRAIN_LOG_EXCLUDE),
))
TELEMETRY_CSV = Path(os.environ.get(
    "ADAN_WEB_TELEMETRY_CSV",
    _latest(_TELEMETRY_GLOBS, str(LOGS_DIR / "diagnostic_collapse_v4.csv"), _TELEMETRY_EXCLUDE),
))

def resolve_telemetry_csv() -> Path:
    """Re-resolve the newest telemetry CSV at call time.

    Honours the ADAN_WEB_TELEMETRY_CSV override; otherwise re-scans LOGS_DIR so
    a long-running backend automatically follows a freshly-started run (e.g. the
    next 500k launch) without needing a restart. Services should call this
    instead of reading the frozen import-time TELEMETRY_CSV constant."""
    override = os.environ.get("ADAN_WEB_TELEMETRY_CSV")
    if override:
        return Path(override)
    return Path(_latest(_TELEMETRY_GLOBS, str(TELEMETRY_CSV), _TELEMETRY_EXCLUDE))


def resolve_train_log() -> Path:
    """Re-resolve the newest training log at call time (see resolve_telemetry_csv)."""
    override = os.environ.get("ADAN_WEB_TRAIN_LOG")
    if override:
        return Path(override)
    return Path(_latest(_TRAIN_LOG_GLOBS, str(TRAIN_LOG), _TRAIN_LOG_EXCLUDE))


# obs_schema version the live training uses (exposed to the frontend so the
# dashboard can show a Schema/Version panel and flag v1(20)/v2(28) runs).
OBS_SCHEMA_VERSION = os.environ.get("ADAN_WEB_OBS_SCHEMA", "obs_schema_v2")
OBS_PORTFOLIO_DIM = int(os.environ.get("ADAN_WEB_OBS_PORTFOLIO_DIM", "28"))

# How the training process is identified in `ps`.
TRAIN_PROCESS_MATCH = "train_parallel_agents.py"

# Collapse decision thresholds (FACTS-based, from docs/web_interface.md).
# ---- Mode 1: BIMODAL saturation (old V3 collapse: a0_std explodes to 13.48) ----
COLLAPSE_A0_STD_WARN = 3.0      # a0_std above this = warning
COLLAPSE_A0_STD_CRIT = 6.0      # a0_std above this = critical (V3 hit 13.48)
COLLAPSE_HOLD_PCT_WARN = 0.05   # req_HOLD_pct below this = warning
COLLAPSE_ILLEGAL_WARN = 0.95    # illegal_ratio above this = warning
# ---- Mode 2: DIRECTIONAL runaway (new selfix/manifesto collapse: a0_mean drifts
# unbounded with LOW a0_std and pct_buy->1.0 / pct_sell->1.0). The old verdict
# missed this entirely and wrongly reported "healthy" on a fully-collapsed BUY
# runaway run (a0_mean=+2.39, pct_buy=1.0). These thresholds fix that. ----
COLLAPSE_A0_MEAN_WARN = 1.0     # |a0_mean| above this = directional drift warning
COLLAPSE_A0_MEAN_CRIT = 1.8     # |a0_mean| above this = directional collapse
COLLAPSE_PCT_SIDE_WARN = 0.85   # pct_buy OR pct_sell above this = one-sided warning
COLLAPSE_PCT_SIDE_CRIT = 0.97   # pct_buy OR pct_sell above this = one-sided collapse

TARGET_TOTAL_STEPS = int(os.environ.get("ADAN_WEB_TARGET_STEPS", "500000"))
