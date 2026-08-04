#!/usr/bin/env python3
"""Audit persisted StateBuilder scalers without fitting or mutating them."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def _unwrap(scaler):
    return getattr(scaler, "scaler", scaler)


def _representative_sample(scaler, feature_count: int) -> np.ndarray:
    base = _unwrap(scaler)
    for attr in ("mean_", "center_", "data_min_"):
        values = getattr(base, attr, None)
        if values is not None:
            return np.asarray(values, dtype=np.float64).reshape(1, -1)
    return np.zeros((1, feature_count), dtype=np.float64)


def audit_scaler(path: Path) -> dict:
    with path.open("rb") as handle:
        scaler = pickle.load(handle)
    base = _unwrap(scaler)
    feature_count = int(getattr(base, "n_features_in_", 0) or 0)
    if feature_count <= 0:
        raise ValueError(f"{path}: missing valid n_features_in_")
    sample = _representative_sample(scaler, feature_count)
    transformed = np.asarray(scaler.transform(sample), dtype=np.float64)
    finite = transformed[np.isfinite(transformed)]
    return {
        "path": str(path),
        "wrapper": type(scaler).__name__,
        "scaler": type(base).__name__,
        "n_features": feature_count,
        "shape": list(transformed.shape),
        "all_finite": bool(np.isfinite(transformed).all()),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "abs_max": float(np.abs(finite).max()) if finite.size else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaler-dir", default="prod_scalers")
    parser.add_argument("--max-abs", type=float, default=10.0)
    args = parser.parse_args()
    paths = sorted(Path(args.scaler_dir).glob("scaler_*.pkl"))
    if not paths:
        raise SystemExit(f"No scaler_*.pkl found in {args.scaler_dir}")
    reports = [audit_scaler(path) for path in paths]
    ok = all(report["all_finite"] and report["abs_max"] <= args.max_abs for report in reports)
    print(json.dumps({"ok": ok, "scalers": reports}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
