#!/usr/bin/env python3
"""Validate append-order position lifecycle invariants in ADAN JSONL traces."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def validate_pipeline_lifecycle(pattern: str, *, require_closed: bool = True) -> dict[str, Any]:
    """Pair OPEN/CLOSE events by position_id without sorting resettable steps."""
    environments: dict[str, dict[str, Any]] = {}
    position_owner: dict[str, str] = {}
    violations: list[dict[str, Any]] = []
    invalid_json = 0
    lifecycle_events = 0
    unchanged_price_closes = 0

    for file_name in sorted(glob.glob(pattern)):
        with Path(file_name).open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    invalid_json += 1
                    violations.append({"type": "invalid_json", "file": file_name, "line": line_number})
                    continue
                lifecycle = event.get("lifecycle_event")
                if event.get("stage") != "trade_executed" or lifecycle not in {"open", "close"}:
                    continue
                lifecycle_events += 1
                env_id = str(event.get("env_instance_id") or "missing")
                worker_id = event.get("worker_id")
                env_key = f"{worker_id}:{env_id}"
                state = environments.setdefault(env_key, {
                    "worker_id": worker_id,
                    "env_instance_id": env_id,
                    "last_sequence": 0,
                    "open": {},
                    "closed": set(),
                    "max_open_positions": 0,
                    "opens": 0,
                    "closes": 0,
                })
                sequence = event.get("event_sequence")
                if not isinstance(sequence, int) or sequence <= state["last_sequence"]:
                    violations.append({"type": "non_monotonic_sequence", "env": env_key, "sequence": sequence, "line": line_number})
                if isinstance(sequence, int):
                    state["last_sequence"] = max(state["last_sequence"], sequence)

                position_id = event.get("position_id")
                if not position_id:
                    violations.append({"type": "missing_position_id", "env": env_key, "lifecycle_event": lifecycle, "line": line_number})
                    continue
                position_id = str(position_id)
                owner = position_owner.setdefault(position_id, env_key)
                if owner != env_key:
                    violations.append({"type": "position_id_reused_across_environments", "position_id": position_id, "owners": [owner, env_key]})

                if lifecycle == "open":
                    state["opens"] += 1
                    if position_id in state["open"] or position_id in state["closed"]:
                        violations.append({"type": "duplicate_open", "env": env_key, "position_id": position_id})
                        continue
                    state["open"][position_id] = event
                    state["max_open_positions"] = max(state["max_open_positions"], len(state["open"]))
                    limit = event.get("max_positions", 1)
                    if not isinstance(limit, int) or limit < 1 or len(state["open"]) > limit:
                        violations.append({"type": "position_limit_exceeded", "env": env_key, "open_count": len(state["open"]), "limit": limit})
                else:
                    state["closes"] += 1
                    if position_id in state["closed"]:
                        violations.append({"type": "duplicate_close", "env": env_key, "position_id": position_id})
                        continue
                    opened = state["open"].pop(position_id, None)
                    if opened is None:
                        violations.append({"type": "orphan_close", "env": env_key, "position_id": position_id})
                        continue
                    state["closed"].add(position_id)
                    if event.get("asset") != opened.get("asset"):
                        violations.append({"type": "asset_mismatch", "env": env_key, "position_id": position_id})
                    open_global = opened.get("global_step")
                    close_global = event.get("global_step")
                    if not isinstance(open_global, int) or not isinstance(close_global, int) or close_global <= open_global:
                        violations.append({"type": "nonpositive_trade_duration", "env": env_key, "position_id": position_id, "open_global_step": open_global, "close_global_step": close_global})
                    try:
                        unchanged_price_closes += float(event["exit_price"]) == float(opened["entry_price"])
                    except (KeyError, TypeError, ValueError):
                        violations.append({"type": "missing_or_invalid_trade_price", "env": env_key, "position_id": position_id})

    unclosed = []
    summaries: dict[str, Any] = {}
    for env_key, state in sorted(environments.items()):
        remaining = sorted(state["open"])
        unclosed.extend({"env": env_key, "position_id": position_id} for position_id in remaining)
        summaries[env_key] = {
            "worker_id": state["worker_id"],
            "env_instance_id": state["env_instance_id"],
            "opens": state["opens"],
            "closes": state["closes"],
            "max_open_positions": state["max_open_positions"],
            "remaining_open": remaining,
            "last_sequence": state["last_sequence"],
        }
    if require_closed:
        violations.extend({"type": "unclosed_position", **item} for item in unclosed)
    return {
        "ok": not violations,
        "files": sorted(glob.glob(pattern)),
        "lifecycle_events": lifecycle_events,
        "invalid_json": invalid_json,
        "unchanged_price_closes": unchanged_price_closes,
        "unclosed_positions": unclosed,
        "violations": violations,
        "environments": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline_glob")
    parser.add_argument("--allow-open-at-eof", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_pipeline_lifecycle(args.pipeline_glob, require_closed=not args.allow_open_at_eof)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
