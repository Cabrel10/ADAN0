#!/usr/bin/env python3
"""Classify WHY hmm_probs collapses — joint mode analysis, not per-index medians.

Why this exists
---------------
diag_hmm_regime_discriminant.py reported, on the real (_BINANCE) universe:

    BTCUSDT_BINANCE train   bull_p50 0.333333  sideways_p50 0.0  bear_p50 0.333333
    BTCUSDT_BINANCE val     bull_p50 0.0       sideways_p50 0.0  bear_p50 1.0

Those are PER-INDEX medians, so they need not sum to 1 and cannot identify the
underlying vectors. 0.333333 is exactly state_builder.py's uniform fallback
(context[3]=0.33, [4]=0.33, [5]=0.34), which raises a specific suspicion:

    the pipeline may be emitting the DEFAULT vector rather than a posterior

This script records the JOINT triple (ctx[3], ctx[4], ctx[5]) per step and
classifies each observation into one bucket, so the cause can be named from the
user's taxonomy instead of guessed:

    DEFAULT_UNIFORM  ~= (0.33, 0.33, 0.34)  -> no HMM output; fallback used
    ONE_HOT          one component >= 0.99  -> posterior, saturated
    INTERMEDIATE     genuinely graded posterior (the healthy case)
    INVALID_SUM      does not sum to 1      -> transformation / mapping bug

The economic consequence is attached: p_hmm = clip(ctx[3], 0.01, 0.99) is
compared against the p_min_required the fee gate actually computes. If p_hmm
only ever takes values in {0.01, 0.33, 0.99} while p_min sits near 0.47, then
the gate can only open on the 0.99 mode — a structural fact about the SIGNAL's
support, independent of the data window.

Read-only: no production code is modified.
"""
from __future__ import annotations

import copy
import json
import os
import statistics as st
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("ADAN_TRAINING_SILENT", "1")
os.environ.setdefault("ADAN_RICH_STEP_EVERY", "999999")

STEPS = int(os.environ.get("DIAG_STEPS", "300"))
SEED = int(os.environ.get("DIAG_SEED", "330500"))
ASSETS = [a for a in os.environ.get(
    "DIAG_ASSETS", "BTCUSDT_BINANCE,DOGEUSDT_BINANCE").split(",") if a]
SPLITS = [s for s in os.environ.get("DIAG_SPLITS", "train").split(",") if s]

# state_builder.py build_context_vector defaults.
DEFAULT_TRIPLE = (0.33, 0.33, 0.34)
TOL = 5e-3

GATE_SAMPLES: list[dict] = []


def install_gate_probe() -> None:
    """Record the (p_hmm, p_min_required) pairs the fee gate really sees."""
    from adan_trading_bot.environment import multi_asset_chunked_env as menv
    original = menv._resolve_ev_fee_gate

    def probed(*, p_hmm, p_min_required, disabled):
        blocked, reason = original(
            p_hmm=p_hmm, p_min_required=p_min_required, disabled=disabled
        )
        GATE_SAMPLES.append({
            "p_hmm": float(p_hmm),
            "p_min_required": float(p_min_required),
            "blocked": bool(blocked),
            "reason": reason,
        })
        return blocked, reason

    menv._resolve_ev_fee_gate = probed


def classify(triple: tuple[float, float, float]) -> str:
    b, s, r = triple
    total = b + s + r
    if all(abs(v - d) <= TOL for v, d in zip(triple, DEFAULT_TRIPLE)):
        return "DEFAULT_UNIFORM"
    if abs(total - 1.0) > 1e-2:
        return "INVALID_SUM"
    if max(triple) >= 0.99:
        return "ONE_HOT"
    if max(triple) >= 0.90:
        return "NEAR_ONE_HOT"
    return "INTERMEDIATE"


def build_env(cfg, asset: str, split: str):
    from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
    from adan_trading_bot.environment.multi_asset_chunked_env import (
        MultiAssetChunkedEnv,
    )
    cfg = copy.deepcopy(cfg)
    wc = copy.deepcopy(cfg.get("workers", {}).get("w1", {}))
    wc.update({
        "worker_id": 0, "data_split": split, "data_split_override": split,
        "timeframes": ["5m", "1h", "4h"], "assets": [asset],
    })
    cfg.setdefault("data", {})["assets"] = [asset]
    cfg.setdefault("environment", {})["assets"] = [asset]
    loader = ChunkedDataLoader(config=cfg, worker_config=wc, worker_id=0)
    data = loader.load_chunk(0)
    env = MultiAssetChunkedEnv(data=data, config=cfg, worker_config=wc,
                               worker_id=0, live_mode=False)
    env.reset(seed=SEED)
    return env


def probe(env, steps: int) -> dict:
    rng = np.random.default_rng(SEED)
    buckets = Counter()
    triples: list[tuple[float, float, float]] = []
    sums: list[float] = []
    for _ in range(steps):
        act = rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
        _, _, term, trunc, _ = env.step(act)
        obs = getattr(env, "_last_observation", None)
        if isinstance(obs, dict):
            ctx = obs.get("context_vector")
            if ctx is not None and len(ctx) >= 6:
                t = (float(ctx[3]), float(ctx[4]), float(ctx[5]))
                triples.append(t)
                sums.append(round(sum(t), 6))
                buckets[classify(t)] += 1
        if term or trunc:
            env.reset()

    n = len(triples) or 1
    p_hmm = [max(0.01, min(0.99, t[0])) for t in triples]
    support = Counter(round(p, 3) for p in p_hmm)
    return {
        "n_obs": len(triples),
        "mode_shares": {k: round(v / n, 4) for k, v in buckets.most_common()},
        "mode_counts": dict(buckets),
        "distinct_p_hmm_values": len(support),
        "p_hmm_support_top": support.most_common(8),
        "p_hmm_p50": round(st.median(p_hmm), 6) if p_hmm else None,
        "sum_p50": round(st.median(sums), 6) if sums else None,
        "sideways_ever_above_0.05": any(t[1] > 0.05 for t in triples),
        "sideways_max": round(max((t[1] for t in triples), default=0.0), 6),
    }


def main() -> None:
    from adan_trading_bot.common.config_loader import ConfigLoader
    install_gate_probe()
    cfg = ConfigLoader.load_config(str(REPO_ROOT / "config" / "config.yaml"))
    cfg.setdefault("environment", {})["rich_display_interval"] = 999999

    windows = []
    for asset in ASSETS:
        for split in SPLITS:
            GATE_SAMPLES.clear()
            try:
                env = build_env(cfg, asset, split)
            except Exception as exc:
                windows.append({"asset": asset, "split": split,
                                "error": f"{type(exc).__name__}: {exc}"})
                continue
            stats = probe(env, STEPS)
            gs = list(GATE_SAMPLES)
            pmins = [g["p_min_required"] for g in gs]
            stats["fee_gate"] = {
                "invocations": len(gs),
                "blocked": sum(1 for g in gs if g["blocked"]),
                "block_rate": (round(sum(1 for g in gs if g["blocked"])
                                     / len(gs), 4) if gs else None),
                "p_min_p50": (round(st.median(pmins), 6) if pmins else None),
                "p_min_min": (round(min(pmins), 6) if pmins else None),
                "share_pmin_below_0.34": (
                    round(sum(1 for p in pmins if p < 0.34) / len(pmins), 4)
                    if pmins else None),
            }
            windows.append({"asset": asset, "split": split,
                            "measured": stats})
            del env

    ok = [w for w in windows if "measured" in w]
    default_share = [w["measured"]["mode_shares"].get("DEFAULT_UNIFORM", 0.0)
                     for w in ok]
    inter_share = [w["measured"]["mode_shares"].get("INTERMEDIATE", 0.0)
                   for w in ok]
    distinct = [w["measured"]["distinct_p_hmm_values"] for w in ok]

    if not ok:
        cause = "NO_WINDOW_MEASURED"
    elif default_share and st.fmean(default_share) > 0.4:
        cause = ("FALLBACK_DOMINANT_no_posterior_produced -> the uniform "
                 "default is emitted for most steps; classify as MISSING/"
                 "NOT-WIRED HMM output, not as a mis-trained HMM")
    elif inter_share and st.fmean(inter_share) < 0.05 and \
            distinct and st.fmean(distinct) <= 6:
        cause = ("DISCRETE_SUPPORT_no_graded_posterior -> p_hmm takes only a "
                 "handful of values (saturation + fallback); classify as "
                 "TRANSFORMATION/THRESHOLD-NORMALISATION, not data")
    else:
        cause = "GRADED_POSTERIOR_present_see_table"

    report = {
        "question": (
            "Why does hmm_probs become bull~0 / sideways~0 / bear~1, and is "
            "0.333333 a posterior or state_builder's uniform fallback?"
        ),
        "method": (
            "Record the JOINT (ctx[3],ctx[4],ctx[5]) triple per step on the "
            "universe launch_asset_run.py actually loads, bucket each "
            "observation, and attach the fee gate's real p_min_required."
        ),
        "default_triple_from_state_builder": DEFAULT_TRIPLE,
        "steps_per_window": STEPS,
        "seed": SEED,
        "assets": ASSETS,
        "splits": SPLITS,
        "windows": windows,
        "cause_classification": cause,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = REPO_ROOT / "logs" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"hmm_posterior_modes_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\n[WROTE] {path}")


if __name__ == "__main__":
    main()
