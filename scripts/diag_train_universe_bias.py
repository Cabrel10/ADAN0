#!/usr/bin/env python3
"""Measure the training universe actually reachable by the agent.

Why
---
The HMM discriminant (logs/validation/hmm_discriminant_*.json) returned
H_BETA: the regime call TRACKS the realized move.
    train split : window return -17.14%  -> bear share 0.795, p_hmm p50 0.01
    val   split : window return  +1.87%  -> bear share 0.435, p_hmm p50 0.333
    test  split : window return  +0.49%  -> bear share 0.620, p_hmm p50 0.01
So the HMM is NOT degenerate. It says "bear" because the train window really
falls 17%. That reframes everything: fee_gate correctly refuses to buy a
falling market, and the agent's 96% executed HOLD is a RATIONAL response to
the data it is given, not a routing bug.

The remaining question is therefore about the DATA, not the model: the env
reported `current_chunk: 1/1`, i.e. a SINGLE chunk per split. If the whole
reachable training universe is one monotonically falling window, then:
  * BUY is EV-negative almost everywhere in it,
  * the only capital-preserving policy IS to hold,
  * and no reward/PPO/reset fix can produce capital growth on it.

This script quantifies that directly from the parquet files on disk, without
the env: how much history exists, what the env's chunk actually covers, and
how one-sided each region is. It separates:
  H-1  the parquet is large and diverse, the LOADER only exposes a tiny slice
       -> fix the data pipeline / chunking, not the model
  H-2  the parquet itself is short and one-sided
       -> the dataset must be extended before any 500k is meaningful
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# ADAN0_LOADER_PATHS: the first version of this script measured
# data/processed/<asset>/<asset>_5m_featured.parquet. That is NOT what the
# pipeline reads. data_loader.py L256-273 resolves
#   config.data_dirs[split] / <ASSET_VARIANT> / <tf>.parquet
# i.e. data/processed/indicators/<split>/<ASSET>/5m.parquet. Measuring the
# wrong files is what made commit a7f517f mis-state the exposure.
_IND = REPO_ROOT / "data/processed/indicators"
TARGETS = {
    f"{split}/{asset}": _IND / split / asset / "5m.parquet"
    for split in ("train", "val", "test")
    for asset in ("BTCUSDT", "BTCUSDT_BINANCE", "DOGEUSDT", "DOGEUSDT_BINANCE")
}

# The env probe loaded 7991 rows of 5m for the train split.
ENV_CHUNK_ROWS_OBSERVED = 7991


def close_col(df: pd.DataFrame):
    for c in df.columns:
        if str(c).lower() in {"close", "close_5m"}:
            return c
    return None


def segment_stats(s: pd.Series, n_parts: int = 10) -> list[dict]:
    out = []
    size = max(1, len(s) // n_parts)
    for k in range(n_parts):
        seg = s.iloc[k * size:(k + 1) * size]
        if len(seg) < 2:
            continue
        a, b = float(seg.iloc[0]), float(seg.iloc[-1])
        r = (b - a) / a if a else 0.0
        out.append({
            "part": k, "rows": int(len(seg)),
            "return_pct": round(r * 100.0, 3),
            "direction": "UP" if r > 0.002 else ("DOWN" if r < -0.002 else "FLAT"),
        })
    return out


def main() -> None:
    assets = {}
    for name, path in TARGETS.items():
        if not path.exists():
            assets[name] = {"exists": False}
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            assets[name] = {"exists": True,
                            "error": f"{type(exc).__name__}: {exc}"}
            continue
        col = close_col(df)
        if col is None:
            assets[name] = {"exists": True, "rows": int(len(df)),
                            "error": "no close column"}
            continue
        s = df[col].dropna()
        first, last = float(s.iloc[0]), float(s.iloc[-1])
        full_ret = (last - first) / first if first else 0.0
        segs = segment_stats(s)
        ups = sum(1 for x in segs if x["direction"] == "UP")
        downs = sum(1 for x in segs if x["direction"] == "DOWN")

        # What fraction of history does one env chunk actually cover?
        coverage = (ENV_CHUNK_ROWS_OBSERVED / len(s)) if len(s) else None

        assets[name] = {
            "exists": True,
            "rows_5m": int(len(s)),
            "approx_days_5m": round(len(s) * 5 / 60 / 24, 1),
            "first_close": round(first, 6),
            "last_close": round(last, 6),
            "full_history_return_pct": round(full_ret * 100.0, 3),
            "decile_segments": segs,
            "rising_deciles": ups,
            "falling_deciles": downs,
            "one_sided": bool(ups == 0 or downs == 0),
            "env_chunk_rows_observed": ENV_CHUNK_ROWS_OBSERVED,
            "share_of_history_in_one_chunk":
                round(coverage, 4) if coverage else None,
        }

    measured = {k: v for k, v in assets.items()
                if v.get("exists") and "rows_5m" in v}
    big_and_diverse = [k for k, v in measured.items()
                       if v["rows_5m"] > 4 * ENV_CHUNK_ROWS_OBSERVED
                       and not v["one_sided"]]

    if not measured:
        verdict = "NO_PARQUET_MEASURED"
    elif big_and_diverse:
        verdict = ("H1_parquet_is_large_and_two_sided_but_env_exposes_one_chunk"
                   " -> data exposure problem, not a model problem")
    else:
        verdict = ("H2_parquet_itself_is_short_or_one_sided"
                   " -> dataset must be extended before 500k")

    report = {
        "question": (
            "Given H_BETA (the HMM tracks reality), is the agent's 96% HOLD a "
            "rational response to a one-sided training universe? And is that "
            "universe small because of the loader or because of the data?"
        ),
        "upstream_evidence": {
            "hmm_discriminant_verdict": "H_BETA_regime_call_tracks_realized_direction",
            "train_window_return_pct": -17.1409,
            "train_bear_share": 0.795,
            "train_p_hmm_p50": 0.01,
            "val_window_return_pct": 1.8739,
            "val_bear_share": 0.435,
            "val_p_hmm_p50": 0.333333,
            "env_reported_chunks": "1/1",
        },
        "assets": assets,
        "verdict": verdict,
        "candidates_large_and_two_sided": big_and_diverse,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out = REPO_ROOT / "logs" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"train_universe_bias_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\n[WROTE] {path}")


if __name__ == "__main__":
    main()
