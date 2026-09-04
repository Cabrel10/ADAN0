#!/usr/bin/env python3
"""Ventilate the requested->executed HOLD gap by exact rejection reason.

Why this script
---------------
The canonical Gate C run (logs/validation/gate_c_run_20260904_225928.log)
returned NO_GO with:
    requested_hold_rate = 0.548
    executed_hold_rate  = 0.960     (Gate C threshold <= 0.80  -> FAIL)
    action divergence   = 0.354     (Gate B threshold <  0.05  -> FAIL)

So the router converts ~41 points of non-HOLD intent into executed HOLD.
The aggregate `routing_reject` counter cannot say *why*.  This script replays a
uniform-random policy on the real env and dumps the ventilated counters added
in commit 25ce66c plus `rejection_reasons`, so the dominant gate is named
instead of guessed.

Output: logs/validation/routing_ventilation_<ts>.json
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("ADAN_TRAINING_SILENT", "1")
os.environ.setdefault("ADAN_RICH_STEP_EVERY", "999999")

STEPS = int(os.environ.get("DIAG_STEPS", "500"))
SEED = int(os.environ.get("DIAG_SEED", "330500"))


def build_env():
    from adan_trading_bot.common.config_loader import ConfigLoader
    from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
    from adan_trading_bot.environment.multi_asset_chunked_env import (
        MultiAssetChunkedEnv,
    )

    cfg = ConfigLoader.load_config(str(REPO_ROOT / "config" / "config.yaml"))
    cfg.setdefault("environment", {})["rich_display_interval"] = 999999
    wc = copy.deepcopy(cfg.get("workers", {}).get("w1", {}))
    wc.update({
        "worker_id": 0, "data_split": "train", "data_split_override": "train",
        "timeframes": ["5m", "1h", "4h"], "assets": ["BTCUSDT"],
    })
    data = ChunkedDataLoader(config=cfg, worker_config=wc,
                             worker_id=0).load_chunk(0)
    env = MultiAssetChunkedEnv(data=data, config=cfg, worker_config=wc,
                               worker_id=0, live_mode=False)
    env.reset(seed=SEED)
    return env


def main() -> None:
    rng = np.random.default_rng(SEED)
    env = build_env()

    requested = Counter()
    executed = Counter()
    kinds = Counter()
    boundaries = 0
    invariant_seen = 0
    invariant_ok = 0

    def bucket(direction: float) -> str:
        if direction > 0.10:
            return "BUY"
        if direction < -0.10:
            return "SELL"
        return "HOLD"

    for i in range(STEPS):
        action = rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
        requested[bucket(float(action[0]))] += 1
        _, _, term, trunc, info = env.step(action)

        # executed intent, read from the env's own pipeline attributes
        exe = getattr(env, "_last_executed_action_kind", None)
        if exe is None:
            counts = getattr(env, "action_pipeline_counts", {}) or {}
            exe = "?"
        executed[str(exe)] += 1

        comps = getattr(env, "_last_reward_components", None) or {}
        if "invariant_ok" in comps:
            invariant_seen += 1
            invariant_ok += int(bool(comps["invariant_ok"]))

        kinds[info.get("termination_kind", "none")] += 1
        if term or trunc:
            boundaries += 1
            env.reset()

    pipeline = dict(getattr(env, "action_pipeline_counts", {}) or {})
    reasons = dict(getattr(env, "rejection_reasons", {}) or {})

    total_rej = sum(v for k, v in pipeline.items()
                    if k.startswith("routing_reject"))
    ventilated = {k: v for k, v in pipeline.items()
                  if k.startswith("routing_reject_")}
    ranked = sorted(reasons.items(), key=lambda kv: -kv[1])[:12]

    report = {
        "steps": STEPS,
        "seed": SEED,
        "requested_buckets": dict(requested),
        "executed_buckets": dict(executed),
        "action_pipeline_counts": pipeline,
        "routing_reject_total": total_rej,
        "routing_reject_ventilated": ventilated,
        "rejection_reasons_top": dict(ranked),
        "dominant_reason": ranked[0][0] if ranked else None,
        "termination_kinds": dict(kinds),
        "boundaries_hit": boundaries,
        "invariant_steps_instrumented": invariant_seen,
        "invariant_steps_ok": invariant_ok,
        "invariant_coverage": (invariant_seen / STEPS) if STEPS else 0.0,
    }

    out = REPO_ROOT / "logs" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"routing_ventilation_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\n[WROTE] {path}")


if __name__ == "__main__":
    main()
