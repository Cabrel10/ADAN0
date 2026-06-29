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

# The active 500k V4 run artifacts (defaults; overridable via env).
TRAIN_LOG = Path(os.environ.get("ADAN_WEB_TRAIN_LOG", str(LOGS_DIR / "train_v4_500k.log")))
TELEMETRY_CSV = Path(os.environ.get("ADAN_WEB_TELEMETRY_CSV", str(LOGS_DIR / "diagnostic_collapse_v4.csv")))

# How the training process is identified in `ps`.
TRAIN_PROCESS_MATCH = "train_parallel_agents.py"

# Collapse decision thresholds (FACTS-based, from docs/web_interface.md).
COLLAPSE_A0_STD_WARN = 3.0      # a0_std above this = warning
COLLAPSE_A0_STD_CRIT = 6.0      # a0_std above this = critical (V3 hit 13.48)
COLLAPSE_HOLD_PCT_WARN = 0.05   # req_HOLD_pct below this = warning
COLLAPSE_ILLEGAL_WARN = 0.95    # illegal_ratio above this = warning

TARGET_TOTAL_STEPS = int(os.environ.get("ADAN_WEB_TARGET_STEPS", "500000"))
