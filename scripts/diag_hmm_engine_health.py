#!/usr/bin/env python3
"""Ask the HMM engine itself why it is not producing a posterior.

Why this exists
---------------
diag_hmm_posterior_modes.py measures the OUTPUT triple. That still leaves the
cause ambiguous. But the engine already keeps its own diagnostics:

    dynamic_behavior_engine.py
      L354  self._hmm_last_fallback_reason = "warming_up"
      L382  self._hmm_probs = np.ones(3)/3          # == 0.333333 exactly
      L352  self._hmm_fit_count
      L353  self._hmm_fit_failures
      L365  self._hmm_total_obs

and it logs "[HMM_FIT] Converged (...)" on every successful fit.

Measured fact that motivates this script: a 300-step run on
BTCUSDT_BINANCE produced 316 KB of logs containing ZERO lines matching "HMM".
No convergence line, and no "[HMM_FIT] ALL inits failed" warning either. That is
only consistent with the fit block never being reached, or with the whole call
being swallowed by the bare

    multi_asset_chunked_env.py L6315-6316
        hmm_probs = self.dbe.get_regime_probabilities(market_data)
      except Exception:
        pass   # <- silently degrades to the uniform prior

This script does three things, all read-only w.r.t. production code:

  1. wraps get_regime_probabilities to CAPTURE any exception the env swallows
     (re-raising nothing: it records the traceback then returns the real value)
  2. reads the engine's own counters after N steps
  3. reports the exact fallback reason string

Taxonomy mapping (user's list):
  reason == "warming_up"            -> NOT_ENOUGH_OBS (a wiring/config issue,
                                       since _hmm_min_obs vs steps must agree)
  reason == "hmmlearn_unavailable"  -> DEPENDENCY missing
  reason == "all_fit_strategies_failed" -> TRAINING/CALIBRATION
  reason startswith "update_error:" -> TRANSFORMATION bug
  captured exception at call site   -> DATA/PLUMBING (never reaches the HMM)
  fit_count > 0 and graded output   -> HMM healthy; look elsewhere
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
import traceback
from collections import Counter
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

CALL_ERRORS: Counter = Counter()
CALL_TRACEBACKS: list[str] = []
CALLS = {"n": 0, "returned_none": 0}
FEATURES: list[dict] = []


def instrument(env) -> None:
    """Capture what the env's bare `except Exception: pass` would hide."""
    dbe = getattr(env, "dbe", None)
    if dbe is None:
        return
    original = dbe.get_regime_probabilities

    def wrapped(market_data):
        CALLS["n"] += 1
        FEATURES.append({
            k: market_data.get(k)
            for k in ("log_return", "atr_pct", "rsi_norm", "volume_ratio_20",
                      "observation_id")
        })
        try:
            out = original(market_data)
        except Exception as exc:
            CALL_ERRORS[f"{type(exc).__name__}: {exc}"] += 1
            if len(CALL_TRACEBACKS) < 3:
                CALL_TRACEBACKS.append(traceback.format_exc()[-1500:])
            raise
        if out is None:
            CALLS["returned_none"] += 1
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


def num(v):
    try:
        f = float(v)
        return None if not np.isfinite(f) else round(f, 8)
    except Exception:
        return None


def main() -> None:
    env = build_env()
    instrument(env)
    rng = np.random.default_rng(SEED)
    for _ in range(STEPS):
        act = rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
        _, _, term, trunc, _ = env.step(act)
        if term or trunc:
            env.reset()

    dbe = getattr(env, "dbe", None)
    engine = {
        "dbe_present": dbe is not None,
        "dbe_is_dynamic_behavior_engine": (
            dbe is getattr(env, "dynamic_behavior_engine", None)),
    }
    for attr in ("_hmm_fitted", "_hmm_fit_count", "_hmm_fit_failures",
                 "_hmm_last_fallback_reason", "_hmm_total_obs",
                 "_hmm_last_refit_obs", "_hmm_min_obs", "_hmm_window",
                 "_hmm_state_order", "_hmm_n_init"):
        engine[attr] = getattr(dbe, attr, "<absent>")
    buf = getattr(dbe, "_hmm_obs_buffer", None)
    engine["_hmm_obs_buffer_len"] = len(buf) if buf is not None else "<absent>"
    model = getattr(dbe, "_hmm_model", "<absent>")
    engine["_hmm_model_is_none"] = model is None
    probs = getattr(dbe, "_hmm_probs", None)
    engine["_hmm_probs_final"] = ([round(float(x), 6) for x in probs]
                                  if probs is not None else None)

    # Are the features the engine receives even alive?
    feat_stats = {}
    for key in ("log_return", "atr_pct", "rsi_norm", "volume_ratio_20"):
        vals = [num(f.get(key)) for f in FEATURES]
        vals = [v for v in vals if v is not None]
        if vals:
            feat_stats[key] = {
                "n": len(vals), "min": min(vals), "max": max(vals),
                "distinct": len(set(vals)),
                "all_zero": all(v == 0.0 for v in vals),
            }
        else:
            feat_stats[key] = {"n": 0}
    obs_ids = [f.get("observation_id") for f in FEATURES]
    feat_stats["observation_id"] = {
        "n": len(obs_ids),
        "distinct": len(set(obs_ids)),
        "sample": obs_ids[:3],
        "none_share": (round(sum(1 for o in obs_ids if o is None)
                             / max(len(obs_ids), 1), 4)),
    }

    reason = engine.get("_hmm_last_fallback_reason")
    fit_count = engine.get("_hmm_fit_count")
    if CALL_ERRORS:
        cause = ("DATA_PLUMBING_call_raises_and_env_swallows_it -> the bare "
                 "'except Exception: pass' at multi_asset_chunked_env L6315 "
                 "hides a real error; HMM never runs")
    elif CALLS["n"] == 0:
        cause = ("NOT_WIRED_get_regime_probabilities_never_called -> context "
                 "[3:6] can only be the state_builder default")
    elif engine.get("_hmm_model_is_none") is True:
        cause = "DEPENDENCY_hmmlearn_unavailable"
    elif reason == "warming_up":
        cause = ("NOT_ENOUGH_OBS_never_leaves_warmup -> _hmm_min_obs is not "
                 "reached within an episode; posterior is the uniform prior "
                 "by construction, independently of the data window")
    elif reason == "all_fit_strategies_failed":
        cause = "TRAINING_CALIBRATION_all_fit_strategies_failed"
    elif isinstance(reason, str) and reason.startswith("update_error:"):
        cause = f"TRANSFORMATION_bug_{reason}"
    elif fit_count and int(fit_count) > 0:
        cause = "HMM_FITTED_look_at_posterior_saturation_not_at_the_engine"
    else:
        cause = "UNRESOLVED_see_engine_dump"

    report = {
        "question": ("Why is context[3:6] not a graded posterior? Ask the "
                     "engine's own counters instead of inferring."),
        "asset": ASSET, "split": SPLIT, "steps": STEPS, "seed": SEED,
        "calls_to_get_regime_probabilities": CALLS["n"],
        "calls_returning_none": CALLS["returned_none"],
        "swallowed_exceptions": dict(CALL_ERRORS),
        "swallowed_tracebacks": CALL_TRACEBACKS,
        "engine_state": engine,
        "features_seen_by_engine": feat_stats,
        "cause_classification": cause,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = REPO_ROOT / "logs" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"hmm_engine_health_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"\n[WROTE] {path}")


if __name__ == "__main__":
    main()
