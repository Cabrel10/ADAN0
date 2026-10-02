#!/usr/bin/env python3
"""Prove/refute that a second caller poisons the HMM fit buffer.

Mechanism under test
--------------------
diag_hmm_engine_health.py measured, on BTCUSDT_BINANCE/train, 300 env steps:

    calls to get_regime_probabilities        = 599   (~2 per step)
    observation_id None share                = 0.4992
    log_return present                       = 300 / 599
    engine healthy: fit_count 5, failures 0, fallback_reason None
    final posterior                          = [0.999906, 0.0, 0.000094]

There are exactly two production callers:

  A. multi_asset_chunked_env.py L6315  self.dbe.get_regime_probabilities(md)
     md = _get_current_market_data_for_hmm() -> real features + observation_id

  B. dynamic_behavior_engine.py L915   self.get_regime_probabilities(md)
     reached from detect_market_regime, itself called at L986 from
     update_risk_parameters and from env L1357 with `market_conditions`,
     a dict that does NOT carry log_return / atr_pct / rsi_norm /
     volume_ratio_20 / observation_id.

Path B therefore hits get_regime_probabilities L631+ with:
    log_ret = 0.0, atr_pct = 0.0, rsi_norm = 0.5, volume_ratio = 1.0
and observation_id None, which ALSO defeats the dedup cache at L636-639
(`if observation_id is not None and observation_id == last`), so the call is
never short-circuited.

_update_hmm then unconditionally does:
    self._hmm_obs_buffer.append([log_return, atr_pct, rsi_norm, volume_ratio])
    self._hmm_total_obs += 1

Two consequences, both testable:
  C1 the 500-row fit window is ~50% one single REPEATED constant point
     -> a degenerate Gaussian component locks onto it
  C2 the returned posterior is predict_proba(X)[-1], i.e. the posterior of
     whichever row was appended LAST. When path B ran last, p_hmm describes a
     SYNTHETIC observation, not the market.

This script counts the synthetic rows inside the live buffer and measures how
often the last row is synthetic. No production code is modified.
"""
from __future__ import annotations

import copy
import json
import os
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
ASSET = os.environ.get("DIAG_ASSET", "BTCUSDT_BINANCE")
SPLIT = os.environ.get("DIAG_SPLIT", "train")

# The exact vector path B produces (defaults from get_regime_probabilities).
SYNTHETIC = (0.0, 0.0, 0.5, 1.0)
TOL = 1e-12

TRACE: list[dict] = []


def _as_list(arr) -> list | None:
    """np arrays are not safely truthy; convert explicitly."""
    if arr is None:
        return None
    return [round(float(x), 6) for x in np.asarray(arr).ravel()]


def is_synthetic(row) -> bool:
    if len(row) < 4:
        return False
    return all(abs(float(a) - b) <= TOL for a, b in zip(row[:4], SYNTHETIC))


def instrument(env) -> None:
    """Record, per call, whether the incoming features are the synthetic ones."""
    dbe = env.dbe
    original = dbe.get_regime_probabilities

    def wrapped(market_data):
        has_obs_id = market_data.get("observation_id") is not None
        lr = market_data.get("log_return", None)
        synthetic_input = (not has_obs_id) and (lr in (None, 0.0))
        out = original(market_data)
        buf = getattr(dbe, "_hmm_obs_buffer", []) or []
        TRACE.append({
            "has_observation_id": has_obs_id,
            "synthetic_input": bool(synthetic_input),
            "last_row_synthetic": bool(buf and is_synthetic(buf[-1])),
            "probs": _as_list(out),
        })
        return out

    dbe.get_regime_probabilities = wrapped


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
        "worker_id": 0, "data_split": SPLIT, "data_split_override": SPLIT,
        "timeframes": ["5m", "1h", "4h"], "assets": [ASSET],
    })
    cfg.setdefault("data", {})["assets"] = [ASSET]
    cfg.setdefault("environment", {})["assets"] = [ASSET]
    data = ChunkedDataLoader(config=cfg, worker_config=wc,
                             worker_id=0).load_chunk(0)
    env = MultiAssetChunkedEnv(data=data, config=cfg, worker_config=wc,
                               worker_id=0, live_mode=False)
    env.reset(seed=SEED)
    return env


def main() -> None:
    env = build_env()
    instrument(env)
    rng = np.random.default_rng(SEED)
    for _ in range(STEPS):
        act = rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
        _, _, term, trunc, _ = env.step(act)
        if term or trunc:
            env.reset()

    dbe = env.dbe
    buf = list(getattr(dbe, "_hmm_obs_buffer", []) or [])
    n_syn = sum(1 for r in buf if is_synthetic(r))
    n = len(buf) or 1

    # How many DISTINCT points does the fit actually see?
    uniq = {tuple(round(float(v), 10) for v in r[:4]) for r in buf}

    calls = len(TRACE) or 1
    syn_calls = sum(1 for t in TRACE if t["synthetic_input"])
    last_syn = sum(1 for t in TRACE if t["last_row_synthetic"])

    contaminated = n_syn / n > 0.10
    if contaminated:
        verdict = (
            "CONFIRMED_buffer_contaminated_by_featureless_second_caller -> "
            "cause class = TRANSFORMATION/PLUMBING, not model training and "
            "not the data window. detect_market_regime() feeds a constant "
            "synthetic observation into the same rolling fit buffer used for "
            "the market posterior."
        )
    elif syn_calls / calls > 0.10:
        verdict = ("PARTIAL_second_caller_present_but_buffer_clean -> check "
                   "whether _update_hmm is reached on that path")
    else:
        verdict = "REFUTED_no_synthetic_contamination_measured"

    report = {
        "question": ("Does the featureless second caller "
                     "(detect_market_regime) poison the HMM fit buffer and "
                     "the returned posterior?"),
        "synthetic_vector_tested": SYNTHETIC,
        "asset": ASSET, "split": SPLIT, "steps": STEPS, "seed": SEED,
        "calls_total": len(TRACE),
        "calls_with_observation_id": sum(1 for t in TRACE
                                         if t["has_observation_id"]),
        "calls_with_synthetic_input": syn_calls,
        "share_calls_synthetic": round(syn_calls / calls, 4),
        "share_posterior_computed_on_synthetic_last_row":
            round(last_syn / calls, 4),
        "buffer_len": len(buf),
        "buffer_synthetic_rows": n_syn,
        "buffer_synthetic_share": round(n_syn / n, 4),
        "buffer_distinct_points": len(uniq),
        "engine_fit_count": getattr(dbe, "_hmm_fit_count", None),
        "engine_fit_failures": getattr(dbe, "_hmm_fit_failures", None),
        "engine_total_obs": getattr(dbe, "_hmm_total_obs", None),
        "final_probs": _as_list(getattr(dbe, "_hmm_probs", None)),
        "verdict": verdict,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = REPO_ROOT / "logs" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"hmm_buffer_contamination_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"\n[WROTE] {path}")


if __name__ == "__main__":
    main()
