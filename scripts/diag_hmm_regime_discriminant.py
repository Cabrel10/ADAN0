#!/usr/bin/env python3
"""Discriminate H-alpha (HMM broken) from H-beta (window unrepresentative).

Open question left by RAPPORT_FEE_GATE_HMM.md
---------------------------------------------
Measured on ONE chunk of BTC train data, the HMM posteriors are near one-hot on
bear (ctx[5] p50 = 1.000, ctx[3] bull p50 = 0.000, sum of means = 1.0000).
That floors p_hmm at 0.01 and makes fee_gate reject 90.5% of BUY intent.

But a regime HMM with strong persistence NORMALLY emits near one-hot
posteriors. So one chunk cannot distinguish:

  H-alpha  the HMM is mis-trained / mis-calibrated / degenerate
           -> it would say "bear" on ANY window, including a rising one
  H-beta   the HMM is fine and the probe window is genuinely ~90% bear
           -> the posteriors track the actual price path

This script does NOT compute the HMM again. It reads the posteriors the
pipeline actually produces, over MULTIPLE chunks and BOTH assets, and lines
each window's regime call up against that window's REALIZED price direction.
That comparison is the discriminant:

  * if bear-share stays ~90% even on windows whose close rises  -> H-alpha
  * if bear-share tracks the sign of the realized move          -> H-beta
  * if bear-share is high everywhere AND every window happens to fall
    -> UNRESOLVED, the data split itself is one-sided (report it, do not guess)

Also records, per window, the p_hmm that fee_gate would see, so the economic
consequence is attached to each regime call rather than argued in the
abstract.
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

STEPS_PER_WINDOW = int(os.environ.get("DIAG_STEPS", "200"))
SEED = int(os.environ.get("DIAG_SEED", "330500"))
ASSETS = [a for a in os.environ.get("DIAG_ASSETS", "BTCUSDT").split(",") if a]
SPLITS = [s for s in os.environ.get("DIAG_SPLITS", "train,val,test").split(",")
          if s]


def load_cfg():
    from adan_trading_bot.common.config_loader import ConfigLoader
    cfg = ConfigLoader.load_config(str(REPO_ROOT / "config" / "config.yaml"))
    cfg.setdefault("environment", {})["rich_display_interval"] = 999999
    return cfg


def build_env(cfg, asset: str, split: str, chunk: int):
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
    # ADAN0_ASSET_PARAM: launch_asset_run.py::derive_config rewrites all three
    # asset keys. Mirror it so the probe sees the same universe as a real run.
    cfg.setdefault("data", {})["assets"] = [asset]
    cfg.setdefault("environment", {})["assets"] = [asset]
    loader = ChunkedDataLoader(config=cfg, worker_config=wc, worker_id=0)
    data = loader.load_chunk(chunk)
    env = MultiAssetChunkedEnv(data=data, config=cfg, worker_config=wc,
                               worker_id=0, live_mode=False)
    env.reset(seed=SEED)
    return env, data


def realized_direction(data, asset: str) -> dict:
    """Sign and size of the 5m close move across the loaded window."""
    try:
        df = None
        if isinstance(data, dict):
            inner = data.get(asset, data)
            if isinstance(inner, dict):
                df = inner.get("5m")
            else:
                df = inner
        if df is None or not hasattr(df, "columns"):
            return {"available": False}
        col = next((c for c in df.columns
                    if str(c).lower() in {"close", "close_5m"}), None)
        if col is None:
            return {"available": False}
        s = df[col].dropna()
        if len(s) < 10:
            return {"available": False}
        first, last = float(s.iloc[0]), float(s.iloc[-1])
        ret = (last - first) / first if first else 0.0
        return {
            "available": True, "rows": int(len(s)),
            "first_close": round(first, 4), "last_close": round(last, 4),
            "window_return_pct": round(ret * 100.0, 4),
            "direction": "UP" if ret > 0.002 else
                         ("DOWN" if ret < -0.002 else "FLAT"),
        }
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def probe_window(env, steps: int) -> dict:
    rng = np.random.default_rng(SEED)
    bull, side, bear = [], [], []
    for _ in range(steps):
        act = rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
        _, _, term, trunc, _ = env.step(act)
        obs = getattr(env, "_last_observation", None)
        if isinstance(obs, dict):
            ctx = obs.get("context_vector")
            if ctx is not None and len(ctx) >= 6:
                bull.append(float(ctx[3]))
                side.append(float(ctx[4]))
                bear.append(float(ctx[5]))
        if term or trunc:
            env.reset()

    def med(v):
        return round(st.median(v), 6) if v else None

    def mean(v):
        return round(st.fmean(v), 6) if v else None

    n = len(bear) or 1
    p_hmm = [max(0.01, min(0.99, b)) for b in bull]
    return {
        "n_obs": len(bear),
        "bull_mean": mean(bull), "bull_p50": med(bull),
        "sideways_mean": mean(side), "sideways_p50": med(side),
        "bear_mean": mean(bear), "bear_p50": med(bear),
        "share_bear_above_0.9": round(sum(1 for b in bear if b > 0.9) / n, 4),
        "share_bull_above_0.5": round(sum(1 for b in bull if b > 0.5) / n, 4),
        "p_hmm_p50": med(p_hmm),
        "p_hmm_share_at_floor_0.01":
            round(sum(1 for p in p_hmm if p <= 0.0101) / n, 4),
    }


def main() -> None:
    cfg = load_cfg()
    windows = []

    for asset in ASSETS:
        for split in SPLITS:
            try:
                env, data = build_env(cfg, asset, split, 0)
            except Exception as exc:
                windows.append({"asset": asset, "split": split,
                                "error": f"{type(exc).__name__}: {exc}"})
                continue
            stats = probe_window(env, STEPS_PER_WINDOW)
            price = realized_direction(data, asset)
            windows.append({"asset": asset, "split": split,
                            "posteriors": stats, "realized": price})
            del env

    ok = [w for w in windows if "posteriors" in w]
    bear_shares = [w["posteriors"]["share_bear_above_0.9"] for w in ok]
    ups = [w for w in ok
           if w.get("realized", {}).get("direction") == "UP"]
    downs = [w for w in ok
             if w.get("realized", {}).get("direction") == "DOWN"]

    bear_on_up = [w["posteriors"]["share_bear_above_0.9"] for w in ups]
    bear_on_down = [w["posteriors"]["share_bear_above_0.9"] for w in downs]

    if not ok:
        verdict = "NO_WINDOW_MEASURED"
    elif not ups:
        verdict = ("UNRESOLVED_no_rising_window_in_sample_"
                   "cannot_separate_H_alpha_from_H_beta")
    elif bear_on_up and st.fmean(bear_on_up) > 0.8:
        verdict = "H_ALPHA_hmm_says_bear_even_on_rising_windows"
    elif (bear_on_up and bear_on_down
          and st.fmean(bear_on_up) + 0.2 < st.fmean(bear_on_down)):
        verdict = "H_BETA_regime_call_tracks_realized_direction"
    else:
        verdict = "UNRESOLVED_see_per_window_table"

    report = {
        "question": (
            "Is the near one-hot bear posterior a broken HMM (H-alpha) or a "
            "genuinely bearish, unrepresentative probe window (H-beta)?"
        ),
        "method": (
            "Read the posteriors the pipeline produces over several "
            "asset/split windows and compare each window's regime call with "
            "that window's realized 5m close move."
        ),
        "steps_per_window": STEPS_PER_WINDOW,
        "seed": SEED,
        "assets": ASSETS,
        "splits": SPLITS,
        "windows": windows,
        "aggregate": {
            "windows_measured": len(ok),
            "bear_share_mean": (round(st.fmean(bear_shares), 4)
                                if bear_shares else None),
            "rising_windows": len(ups),
            "falling_windows": len(downs),
            "bear_share_on_rising": (round(st.fmean(bear_on_up), 4)
                                     if bear_on_up else None),
            "bear_share_on_falling": (round(st.fmean(bear_on_down), 4)
                                      if bear_on_down else None),
        },
        "verdict": verdict,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out = REPO_ROOT / "logs" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"hmm_discriminant_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\n[WROTE] {path}")


if __name__ == "__main__":
    main()
