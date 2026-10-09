#!/usr/bin/env python3
"""Identify what context_vector[3] actually contains (the p_hmm source).

Chain so far
------------
1. Canonical Gate C: NO_GO (executed HOLD 96% vs requested 55%).
2. Ventilation: fee_gate kills 201/222 = 90.5% of BUY intent.
3. Measured gate inputs (fee_gate_measured_*.json):
       p_min_required  p50 = 0.470   (reasonable, NOT the problem)
       p_hmm           p50 = 0.010   min 0.01, mean 0.080
       share p_hmm<0.5 = 0.957
   => dominant term is H-A: the SIGNAL, not the economics. This REFUTES the
      "closed by construction" conclusion of commit c2ca902, which was derived
      from the SL/TP upper bounds instead of measured runtime values.

p_hmm is read at env L9106-9114 as:
       bull_prob = float(ctx[3]);  p_hmm = clip(bull_prob, 0.01, 0.99)
A p50 of exactly 0.01 means ctx[3] is <= 0.01 more than half the time, i.e. it
is being clamped at the floor. A genuine bull probability from a 3-state HMM
should hover around 1/3 and vary. So either
  H-1 ctx[3] is not the bull probability (wrong index / wrong semantics), or
  H-2 ctx[3] IS a probability but the HMM is degenerate/untrained, or
  H-3 ctx[3] is a standardized feature (z-score), hence often negative, and
      clipping to [0.01,0.99] silently converts "below average" into
      "1% chance of going up".

This script dumps the full context_vector distribution per index so the answer
is read off the data, not argued. If some other index behaves like a
probability (bounded [0,1], mean ~1/3, non-degenerate) while index 3 behaves
like a z-score (signed, mean ~0), H-1/H-3 are confirmed together and the fix
is an index/semantics correction, not a threshold tweak.
"""
from __future__ import annotations

import copy
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("ADAN_TRAINING_SILENT", "1")
os.environ.setdefault("ADAN_RICH_STEP_EVERY", "999999")

STEPS = int(os.environ.get("DIAG_STEPS", "300"))
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


def summarize(vals: list[float]) -> dict:
    s = sorted(vals)
    if not s:
        return {"n": 0}
    def q(p):
        return round(s[min(len(s) - 1, int(p * len(s)))], 6)
    uniq = len({round(v, 6) for v in s})
    return {
        "n": len(s), "unique_values": uniq,
        "min": round(s[0], 6), "p10": q(0.10), "p50": q(0.50),
        "p90": q(0.90), "max": round(s[-1], 6),
        "mean": round(st.fmean(s), 6),
        "std": round(st.pstdev(s), 6) if len(s) > 1 else 0.0,
        "share_negative": round(sum(1 for v in s if v < 0) / len(s), 4),
        "share_in_unit_interval":
            round(sum(1 for v in s if 0.0 <= v <= 1.0) / len(s), 4),
        "constant": uniq == 1,
    }


def classify(stats: dict) -> str:
    if stats.get("n", 0) == 0:
        return "empty"
    if stats["constant"]:
        return "CONSTANT_dead_feature"
    if stats["share_in_unit_interval"] > 0.98:
        if 0.15 <= stats["mean"] <= 0.85:
            return "probability_like"
        return "unit_bounded_but_skewed"
    if stats["share_negative"] > 0.15:
        return "SIGNED_zscore_like_NOT_a_probability"
    return "unbounded_or_other"


def main() -> None:
    rng = np.random.default_rng(SEED)
    env = build_env()

    per_index: dict[int, list[float]] = {}
    captured = 0
    ctx_len = None

    for _ in range(STEPS):
        action = rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
        _, _, term, trunc, _ = env.step(action)
        obs = getattr(env, "_last_observation", None)
        if isinstance(obs, dict):
            ctx = obs.get("context_vector")
            if ctx is not None and hasattr(ctx, "__len__"):
                ctx_len = len(ctx)
                captured += 1
                for i in range(len(ctx)):
                    try:
                        per_index.setdefault(i, []).append(float(ctx[i]))
                    except Exception:
                        pass
        if term or trunc:
            env.reset()

    stats = {i: summarize(v) for i, v in sorted(per_index.items())}
    kinds = {i: classify(s) for i, s in stats.items()}
    prob_like = [i for i, k in kinds.items() if k == "probability_like"]

    idx3 = stats.get(3, {"n": 0})
    idx3_kind = kinds.get(3, "missing")

    if idx3_kind == "SIGNED_zscore_like_NOT_a_probability":
        verdict = "CONFIRMED_H1_H3_index3_is_not_a_probability"
    elif idx3_kind == "CONSTANT_dead_feature":
        verdict = "CONFIRMED_index3_is_dead"
    elif idx3_kind == "probability_like":
        verdict = "REFUTED_index3_looks_like_a_probability_investigate_HMM"
    else:
        verdict = f"UNRESOLVED_index3_kind={idx3_kind}"

    report = {
        "context": (
            "p_hmm is read as clip(context_vector[3], 0.01, 0.99) at env "
            "L9106-9114 and feeds the EV fee gate that rejects 90.5% of BUY "
            "intent. Measured p_hmm p50 = 0.01 (floor). This run identifies "
            "what index 3 really is."
        ),
        "steps": STEPS,
        "seed": SEED,
        "observations_captured": captured,
        "context_vector_len": ctx_len,
        "index_3_stats": idx3,
        "index_3_classification": idx3_kind,
        "verdict": verdict,
        "indices_that_look_like_probabilities": prob_like,
        "classification_per_index": kinds,
        "stats_per_index": stats,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out = REPO_ROOT / "logs" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"context_vector_semantics_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\n[WROTE] {path}")


if __name__ == "__main__":
    main()
