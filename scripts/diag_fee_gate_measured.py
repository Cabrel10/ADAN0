#!/usr/bin/env python3
"""Measure the EV fee gate instead of deducing it — self-correction.

Why this exists
---------------
The previous commit (c2ca902) claimed the fee gate was "closed by
construction". That claim was based on the *upper* bounds only
(sl_hi=0.0235, tp_lo=0.0135 -> p_min 0.60-0.74 > p_hmm~0.5).

Re-reading the env at L9320-9345 shows the agent does not get a fixed pair: it
gets a BOX.
    sl_pct in [sl_lo=0.003, sl_hi=0.0235]
    tp_pct in [tp_lo=0.0135, tp_hi=0.0222]
At the bottom of the SL band the gate is easily satisfiable:
    sl=0.003, tp=0.0222 -> p_min = 0.007/0.0252 = 0.278 << 0.5
So "closed by construction" is WRONG as stated. Under a uniform-random policy
roughly a third of the box should clear p_min at p_hmm=0.5, yet the measured
rejection was 201/222 = 90.5%. Something else is doing the work.

This script stops guessing: it monkeypatches `resolve_ev_fee_gate` to record
the ACTUAL (p_hmm, p_min_required) pairs seen at runtime, so the dominant term
is measured, not inferred. No production code is modified.

Distinguishes the two candidate explanations:
  H-A  p_hmm is systematically well below 0.5 (signal, not economics)
  H-B  p_min is systematically high because the realized sl/tp pairs cluster
       at high SL / low TP (economics, e.g. RR enforcement or ATR floor
       pushing SL up)
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

STEPS = int(os.environ.get("DIAG_STEPS", "500"))
SEED = int(os.environ.get("DIAG_SEED", "330500"))

SAMPLES: list[dict] = []


def install_probe() -> None:
    """Wrap resolve_ev_fee_gate at the env's import site to log its inputs."""
    from adan_trading_bot.environment import multi_asset_chunked_env as menv

    original = menv._resolve_ev_fee_gate

    def probed(*, p_hmm, p_min_required, disabled):
        blocked, reason = original(
            p_hmm=p_hmm, p_min_required=p_min_required, disabled=disabled
        )
        SAMPLES.append({
            "p_hmm": float(p_hmm),
            "p_min_required": float(p_min_required),
            "margin": float(p_hmm) - float(p_min_required),
            "blocked": bool(blocked),
            "reason": reason,
        })
        return blocked, reason

    menv._resolve_ev_fee_gate = probed


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


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    s = sorted(values)
    def q(p: float) -> float:
        return round(s[min(len(s) - 1, int(p * len(s)))], 6)
    return {
        "n": len(s),
        "min": round(s[0], 6), "p10": q(0.10), "p50": q(0.50),
        "p90": q(0.90), "max": round(s[-1], 6),
        "mean": round(st.fmean(s), 6),
    }


def main() -> None:
    install_probe()
    rng = np.random.default_rng(SEED)
    env = build_env()

    for _ in range(STEPS):
        action = rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            env.reset()

    blocked = [s for s in SAMPLES if s["blocked"]]
    passed = [s for s in SAMPLES if not s["blocked"]]

    p_hmm_all = [s["p_hmm"] for s in SAMPLES]
    p_min_all = [s["p_min_required"] for s in SAMPLES]

    # Which side dominates? Compare each distribution against the 0.5 anchor.
    n = len(SAMPLES) or 1
    p_hmm_below_half = sum(1 for v in p_hmm_all if v < 0.5) / n
    p_min_above_half = sum(1 for v in p_min_all if v > 0.5) / n
    both = sum(1 for s in SAMPLES
               if s["p_hmm"] < 0.5 and s["p_min_required"] > 0.5) / n

    if p_hmm_below_half > 0.8 and p_min_above_half < 0.5:
        dominant = "H-A_signal_p_hmm_too_low"
    elif p_min_above_half > 0.8 and p_hmm_below_half < 0.5:
        dominant = "H-B_economics_p_min_too_high"
    elif p_hmm_below_half > 0.6 and p_min_above_half > 0.6:
        dominant = "BOTH_signal_and_economics"
    else:
        dominant = "INCONCLUSIVE_see_distributions"

    report = {
        "corrects_previous_claim": (
            "commit c2ca902 stated the gate is 'closed by construction'. That "
            "was derived from the upper bounds only. sl_lo=0.003 with "
            "tp_hi=0.0222 gives p_min=0.278, so the gate IS satisfiable inside "
            "the action box. This run measures which term actually blocks."
        ),
        "steps": STEPS,
        "seed": SEED,
        "gate_invocations": len(SAMPLES),
        "blocked": len(blocked),
        "passed": len(passed),
        "block_rate": round(len(blocked) / n, 4),
        "p_hmm_distribution": summarize(p_hmm_all),
        "p_min_required_distribution": summarize(p_min_all),
        "margin_distribution_p_hmm_minus_p_min": summarize(
            [s["margin"] for s in SAMPLES]),
        "share_p_hmm_below_0.5": round(p_hmm_below_half, 4),
        "share_p_min_above_0.5": round(p_min_above_half, 4),
        "share_both_adverse": round(both, 4),
        "dominant_term": dominant,
        "reason_counts": {
            r: sum(1 for s in SAMPLES if s["reason"] == r)
            for r in {s["reason"] for s in SAMPLES}
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out = REPO_ROOT / "logs" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"fee_gate_measured_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\n[WROTE] {path}")


if __name__ == "__main__":
    main()
