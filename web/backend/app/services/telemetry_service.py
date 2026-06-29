"""Reads the diagnostic collapse telemetry CSV produced by
DiagnosticCollapseCallback (real columns, see docs/web_interface.md).

CSV columns:
  timesteps, a0_mean, a0_std, a0_pct_buy, a0_pct_sell, a0_pct_hold_band,
  req_HOLD_pct, req_BUY_pct, req_SELL_pct, steps_flat_pct, steps_open_pct,
  illegal_ratio, policy_entropy, a0_histo
"""
from __future__ import annotations

import csv
from typing import Any

from .. import settings

NUMERIC_COLS = [
    "timesteps", "a0_mean", "a0_std", "a0_pct_buy", "a0_pct_sell",
    "a0_pct_hold_band", "req_HOLD_pct", "req_BUY_pct", "req_SELL_pct",
    "steps_flat_pct", "steps_open_pct", "illegal_ratio", "policy_entropy",
]


def _parse_row(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in NUMERIC_COLS:
        val = row.get(col, "")
        try:
            out[col] = float(val) if val not in ("", None) else None
        except (ValueError, TypeError):
            out[col] = None
    if out.get("timesteps") is not None:
        out["timesteps"] = int(out["timesteps"])
    histo = row.get("a0_histo", "") or ""
    try:
        out["a0_histo"] = [int(x) for x in histo.split("|") if x != ""]
    except ValueError:
        out["a0_histo"] = []
    return out


def read_telemetry(since: int = 0, limit: int = 5000) -> list[dict[str, Any]]:
    """Return parsed telemetry rows with timesteps > `since`."""
    path = settings.TELEMETRY_CSV
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                parsed = _parse_row(raw)
                ts = parsed.get("timesteps")
                if ts is None or ts <= since:
                    continue
                rows.append(parsed)
    except Exception:
        return rows
    return rows[-limit:]


def latest() -> dict[str, Any] | None:
    rows = read_telemetry(since=-1)
    return rows[-1] if rows else None


def collapse_verdict() -> dict[str, Any]:
    """FACTS-based collapse assessment from the latest + trend of a0_std."""
    rows = read_telemetry(since=-1)
    if not rows:
        return {
            "status": "unknown",
            "level": "unknown",
            "reasons": ["no telemetry yet"],
            "latest": None,
        }
    last = rows[-1]
    reasons: list[str] = []
    level = "ok"

    a0_std = last.get("a0_std")
    hold = last.get("req_HOLD_pct")
    illegal = last.get("illegal_ratio")

    if a0_std is not None and a0_std >= settings.COLLAPSE_A0_STD_CRIT:
        level = "critical"
        reasons.append(f"a0_std={a0_std:.2f} >= {settings.COLLAPSE_A0_STD_CRIT} (bimodal saturation)")
    elif a0_std is not None and a0_std >= settings.COLLAPSE_A0_STD_WARN:
        level = "warning" if level == "ok" else level
        reasons.append(f"a0_std={a0_std:.2f} >= {settings.COLLAPSE_A0_STD_WARN}")

    if hold is not None and hold <= settings.COLLAPSE_HOLD_PCT_WARN:
        level = "warning" if level == "ok" else level
        reasons.append(f"req_HOLD_pct={hold:.3f} <= {settings.COLLAPSE_HOLD_PCT_WARN}")

    if illegal is not None and illegal >= settings.COLLAPSE_ILLEGAL_WARN:
        level = "warning" if level == "ok" else level
        reasons.append(f"illegal_ratio={illegal:.3f} >= {settings.COLLAPSE_ILLEGAL_WARN}")

    # Trend: is a0_std rising over the last samples?
    trend = None
    if len(rows) >= 3:
        first_std = rows[0].get("a0_std")
        if first_std is not None and a0_std is not None:
            trend = a0_std - first_std

    if not reasons:
        reasons.append("distribution healthy (a0_std moderate, HOLD present)")

    return {
        "status": "collapsing" if level == "critical" else ("at_risk" if level == "warning" else "healthy"),
        "level": level,
        "reasons": reasons,
        "a0_std_trend": trend,
        "latest": last,
        "samples": len(rows),
    }
