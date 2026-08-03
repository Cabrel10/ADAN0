#!/usr/bin/env python3
"""Reproducible dataset/model intelligence bulletin for ADAN.

The script separates three evidence levels:

* market-opportunity analysis, reconstructed unambiguously from Parquet bars;
* trade-conditioned Future Arena labels joined by exact market timestamp/row id;
* policy-head sensitivity, measured by feature permutation on real observations.

Lifecycle telemetry is never joined by price. A trade is admitted only when its OPEN
contains all 16 immutable decision features and an exact entry market timestamp that
exists in the featured 5m bars.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter
from itertools import combinations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

FEATURES = [
    "ema_ratio",
    "macdh",
    "rsi",
    "adx_14",
    "di_delta",
    "atr_pct",
    "bb_percent_b_20_2",
    "obv_slope",
    "volume_ratio_20",
    "volatility_ratio_14_50",
    "fib_ratio",
    "price_action",
    "vwap_ratio",
    "market_structure",
    "bb_width_20_2",
    "log_return",
]
TF_ALIASES = {
    "5m": {"ema_ratio": "ema_20_ratio", "macdh": "macdh_12_26_9", "rsi": "rsi_14"},
    "1h": {"ema_ratio": "ema_50_ratio", "macdh": "macdh_21_42_9", "rsi": "rsi_21"},
    "4h": {"ema_ratio": "ema_100_ratio", "macdh": "macdh_26_52_18", "rsi": "rsi_28"},
}
ACTION_HEADS = ["direction", "size", "timeframe", "sl", "tp"]
QUANTILES = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]


def native(value: Any) -> Any:
    """Convert numpy/pandas values to strict JSON-compatible Python values."""
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def finite_frame(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.loc[:, list(columns)].replace([np.inf, -np.inf], np.nan).dropna()
    return out.astype(float)


def feature_column(tf: str, canonical: str) -> str:
    return TF_ALIASES.get(tf, {}).get(canonical, canonical)


def load_data(data_root: Path) -> dict[str, dict[str, pd.DataFrame]]:
    data: dict[str, dict[str, pd.DataFrame]] = {}
    for split in ("train", "val", "test"):
        data[split] = {}
        for tf in ("5m", "1h", "4h"):
            path = data_root / split / "BTCUSDT" / f"{tf}.parquet"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_parquet(path).sort_index()
            missing = [feature_column(tf, f) for f in FEATURES if feature_column(tf, f) not in frame]
            if missing:
                raise ValueError(f"{path}: missing required columns {missing}")
            required = {"open", "high", "low", "close"}
            if not required.issubset(frame.columns):
                raise ValueError(f"{path}: missing OHLC columns {sorted(required - set(frame.columns))}")
            data[split][tf] = frame
    return data


def distribution_section(data: dict[str, dict[str, pd.DataFrame]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split, by_tf in data.items():
        result[split] = {}
        for tf, frame in by_tf.items():
            stats: dict[str, Any] = {}
            for feature in FEATURES:
                values = pd.to_numeric(frame[feature_column(tf, feature)], errors="coerce")
                finite = values[np.isfinite(values)]
                q = finite.quantile(QUANTILES)
                stats[feature] = {
                    "source_column": feature_column(tf, feature),
                    "count": int(finite.size),
                    "missing_or_non_finite": int(values.size - finite.size),
                    "mean": finite.mean(),
                    "std": finite.std(ddof=1),
                    "skew": finite.skew(),
                    "quantiles": {f"q{int(p * 100):02d}": q.loc[p] for p in QUANTILES},
                }
            result[split][tf] = stats
    return result


def add_future_targets(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Build long-side future excursion labels without using future in inputs."""
    close = frame["close"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    n = len(frame) - horizon
    if n <= 0:
        raise ValueError(f"horizon {horizon} exceeds frame length {len(frame)}")

    out = pd.DataFrame(index=frame.index[:n])
    for feature in FEATURES:
        out[feature] = frame[feature_column("5m", feature)].iloc[:n].to_numpy(dtype=float)

    entry = close[:n]
    future_close = close[horizon : horizon + n]
    highs = np.lib.stride_tricks.sliding_window_view(high[1:], horizon)[:n]
    lows = np.lib.stride_tricks.sliding_window_view(low[1:], horizon)[:n]
    high_arg = highs.argmax(axis=1)
    low_arg = lows.argmin(axis=1)
    future_high = highs[np.arange(n), high_arg]
    future_low = lows[np.arange(n), low_arg]
    atr_pct = np.maximum(out["atr_pct"].to_numpy(dtype=float), 1e-12)
    atr_raw = entry * atr_pct

    out["entry_price"] = entry
    out["atr_raw"] = atr_raw
    out["future_return"] = future_close / entry - 1.0
    out["mfe_pct"] = future_high / entry - 1.0
    out["mae_pct"] = np.maximum(entry / np.maximum(future_low, 1e-12) - 1.0, 0.0)
    out["mfe_atr"] = out["mfe_pct"] / atr_pct
    out["mae_atr"] = out["mae_pct"] / atr_pct
    out["time_to_mfe"] = high_arg + 1
    out["time_to_mae"] = low_arg + 1
    out["direction_up"] = (out["future_return"] > 0.0).astype(int)
    # A non-tautological long-entry outcome after an explicit 20 bps round trip.
    out["good_long"] = (out["future_return"] > 0.002).astype(int)
    out["bad_long"] = (out["future_return"] < -0.002).astype(int)
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def resolve_telemetry_paths(patterns: list[str]) -> list[Path]:
    """Resolve one or more shell-style patterns without relying on shell expansion."""
    resolved: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        absolute_pattern = str(candidate if candidate.is_absolute() else ROOT / candidate)
        resolved.extend(Path(path).resolve() for path in glob.glob(absolute_pattern))
    unique = sorted(set(resolved))
    if not unique:
        raise FileNotFoundError(f"No telemetry files match: {patterns}")
    return unique


def load_trade_arena(
    paths: list[Path],
    bars: pd.DataFrame,
    horizon: int,
    tp_min: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join lifecycle trades to bars using exact IDs/timestamps, never prices."""
    opens: dict[str, dict[str, Any]] = {}
    closes: dict[str, dict[str, Any]] = {}
    invalid_json = 0
    duplicate_events: list[str] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue
            position_id = event.get("position_id")
            lifecycle = event.get("lifecycle_event")
            if not position_id or lifecycle not in {"open", "close"}:
                continue
            target = opens if lifecycle == "open" else closes
            if position_id in target:
                duplicate_events.append(f"{path.name}:{line_number}:{lifecycle}:{position_id}")
            else:
                target[position_id] = event

    canonical_snapshot = {feature_column("5m", feature): feature for feature in FEATURES}
    bars = bars.sort_index()
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for position_id, opened in opens.items():
        closed = closes.get(position_id)
        if closed is None:
            rejected["missing_close"] += 1
            continue
        snapshot = opened.get("entry_feature_snapshot")
        if not isinstance(snapshot, dict) or any(source not in snapshot for source in canonical_snapshot):
            rejected["incomplete_feature_snapshot"] += 1
            continue
        timestamp = pd.Timestamp(opened.get("entry_market_timestamp"))
        if timestamp not in bars.index:
            rejected["entry_timestamp_not_in_bars"] += 1
            continue
        location = bars.index.get_loc(timestamp)
        if not isinstance(location, (int, np.integer)):
            rejected["non_unique_entry_timestamp"] += 1
            continue
        stop = min(int(location) + horizon, len(bars))
        future = bars.iloc[int(location):stop]
        if len(future) < horizon:
            rejected["insufficient_future_bars"] += 1
            continue
        entry_price = float(opened["entry_price"])
        if not np.isfinite(entry_price) or entry_price <= 0:
            rejected["invalid_entry_price"] += 1
            continue
        future_high = float(future["high"].max())
        future_low = float(future["low"].min())
        mfe_pct = future_high / entry_price - 1.0
        mae_pct = max(entry_price / max(future_low, 1e-12) - 1.0, 0.0)
        atr_pct = float(opened.get("entry_atr_pct") or snapshot[feature_column("5m", "atr_pct")])
        row: dict[str, Any] = {
            "position_id": position_id,
            "worker_id": int(opened.get("worker_id", -1)),
            "entry_market_timestamp": timestamp,
            "entry_row_id": opened.get("entry_row_id"),
            "entry_price": entry_price,
            "exit_price": float(closed["exit_price"]),
            "pnl_net": float(closed["pnl_net"]),
            "close_reason": str(closed.get("reason", "UNKNOWN")),
            "tp_pct": float(opened.get("tp_pct", 0.0)),
            "sl_pct": float(opened.get("sl_pct", 0.0)),
            "mfe_pct": mfe_pct,
            "mae_pct": mae_pct,
            "mfe_atr": mfe_pct / max(atr_pct, 1e-12),
            "mae_atr": mae_pct / max(atr_pct, 1e-12),
            "mfe_gt_tp_min": int(mfe_pct > tp_min),
            "chosen_tp_attainable": int(mfe_pct >= float(opened.get("tp_pct", 0.0))),
            "profitable_trade": int(float(closed["pnl_net"]) > 0.0),
        }
        row.update({canonical: float(snapshot[source]) for source, canonical in canonical_snapshot.items()})
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("entry_market_timestamp").reset_index(drop=True) if rows else pd.DataFrame()
    audit = {
        "files": [str(path.relative_to(ROOT)) for path in paths],
        "invalid_json": invalid_json,
        "open_count": len(opens),
        "close_count": len(closes),
        "joined_count": len(frame),
        "join_rate": len(frame) / len(opens) if opens else 0.0,
        "duplicate_events": duplicate_events,
        "rejected": dict(rejected),
        "join_keys": ["position_id", "entry_market_timestamp", "entry_row_id"],
        "price_join_forbidden": True,
        "feature_mapping": canonical_snapshot,
    }
    return frame, audit


def rank_scores(names: list[str], values: np.ndarray) -> list[dict[str, Any]]:
    order = np.argsort(np.nan_to_num(values, nan=-np.inf))[::-1]
    return [{"feature": names[i], "score": values[i]} for i in order]


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    return roc_auc_score(y_true, score) if np.unique(y_true).size == 2 else None


def calibration_payload(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    observed, predicted = calibration_curve(y_true, scores, n_bins=10, strategy="quantile")
    return {
        "brier_score": brier_score_loss(y_true, scores),
        "predicted_probability": predicted,
        "observed_frequency": observed,
    }


def diagnostic_classification(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    target: str,
    seed: int,
) -> dict[str, Any]:
    """Strict temporal classifier diagnostics with all-feature ablations and PDP."""
    x_train = finite_frame(train, FEATURES)
    x_test = finite_frame(holdout, FEATURES)
    y_train = train.loc[x_train.index, target].to_numpy(dtype=int)
    y_test = holdout.loc[x_test.index, target].to_numpy(dtype=int)
    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        return {"error": "both temporal partitions must contain two classes"}

    tree = ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    ).fit(x_train, y_train)
    boost = HistGradientBoostingClassifier(
        max_iter=200,
        max_leaf_nodes=15,
        class_weight="balanced",
        random_state=seed,
    ).fit(x_train, y_train)
    tree_score = tree.predict_proba(x_test)[:, 1]
    boost_score = boost.predict_proba(x_test)[:, 1]
    tree_pred = tree.predict(x_test)
    boost_pred = boost.predict(x_test)
    permutation = permutation_importance(
        tree,
        x_test,
        y_test,
        scoring="balanced_accuracy",
        n_repeats=8,
        random_state=seed,
        n_jobs=-1,
    )
    mi = mutual_info_classif(x_train, y_train, random_state=seed)

    baseline_auc = safe_auc(y_test, tree_score)
    baseline_balanced = balanced_accuracy_score(y_test, tree_pred)
    ablations = []
    for removed in FEATURES:
        kept = [feature for feature in FEATURES if feature != removed]
        ablated = ExtraTreesClassifier(
            n_estimators=180,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ).fit(x_train[kept], y_train)
        score = ablated.predict_proba(x_test[kept])[:, 1]
        prediction = ablated.predict(x_test[kept])
        auc = safe_auc(y_test, score)
        balanced = balanced_accuracy_score(y_test, prediction)
        ablations.append(
            {
                "removed_feature": removed,
                "roc_auc": auc,
                "balanced_accuracy": balanced,
                "auc_delta_vs_all": (auc - baseline_auc) if auc is not None and baseline_auc is not None else None,
                "balanced_accuracy_delta_vs_all": balanced - baseline_balanced,
                "interpretation": "diagnostic_only_all_16_features_remain_mandatory",
            }
        )

    rule_model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=max(8, len(x_train) // 50),
        class_weight="balanced",
        random_state=seed,
    ).fit(x_train, y_train)
    rule_score = rule_model.predict_proba(x_test)[:, 1]

    one_dimensional: dict[str, Any] = {}
    for feature in FEATURES:
        pdp = partial_dependence(boost, x_train, [feature], grid_resolution=12)
        one_dimensional[feature] = {
            "grid": pdp["grid_values"][0],
            "average_probability": pdp["average"][0],
        }
    top_features = [row["feature"] for row in rank_scores(FEATURES, permutation.importances_mean)[:6]]
    two_dimensional: dict[str, Any] = {}
    for first, second in combinations(top_features, 2):
        feature_indices = (
            int(x_train.columns.get_loc(first)),
            int(x_train.columns.get_loc(second)),
        )
        pdp = partial_dependence(boost, x_train, [feature_indices], grid_resolution=8)
        two_dimensional[f"{first}__{second}"] = {
            "feature_pair": [first, second],
            "first_grid": pdp["grid_values"][0],
            "second_grid": pdp["grid_values"][1],
            "average_probability": pdp["average"][0],
        }

    return {
        "target": target,
        "train_rows": len(x_train),
        "holdout_rows": len(x_test),
        "prevalence_train": y_train.mean(),
        "prevalence_holdout": y_test.mean(),
        "extra_trees": {
            "balanced_accuracy": baseline_balanced,
            "roc_auc": baseline_auc,
            "confusion_matrix": confusion_matrix(y_test, tree_pred, labels=[0, 1]),
            "calibration": calibration_payload(y_test, tree_score),
        },
        "hist_gradient_boosting": {
            "balanced_accuracy": balanced_accuracy_score(y_test, boost_pred),
            "roc_auc": safe_auc(y_test, boost_score),
            "confusion_matrix": confusion_matrix(y_test, boost_pred, labels=[0, 1]),
            "calibration": calibration_payload(y_test, boost_score),
        },
        "mutual_information": rank_scores(FEATURES, mi),
        "permutation_importance": rank_scores(FEATURES, permutation.importances_mean),
        "ablations": ablations,
        "explicit_rules": {
            "rules": export_text(rule_model, feature_names=FEATURES),
            "balanced_accuracy": balanced_accuracy_score(y_test, rule_model.predict(x_test)),
            "roc_auc": safe_auc(y_test, rule_score),
        },
        "partial_dependence": {
            "one_dimensional": one_dimensional,
            "two_dimensional": two_dimensional,
        },
    }


def trade_arena_section(frame: pd.DataFrame, seed: int) -> dict[str, Any]:
    if frame.empty:
        return {"available": False, "reason": "no exactly joined trades"}
    split = max(1, min(len(frame) - 1, int(len(frame) * 0.70)))
    train = frame.iloc[:split].copy()
    holdout = frame.iloc[split:].copy()
    targets = {}
    for target in ("mfe_gt_tp_min", "chosen_tp_attainable", "profitable_trade"):
        targets[target] = diagnostic_classification(train, holdout, target, seed)
    return {
        "available": True,
        "temporal_protocol": "first 70% of entry timestamps -> final 30%; no random row split",
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "close_reasons": dict(Counter(frame["close_reason"])),
        "pnl_net_sum": frame["pnl_net"].sum(),
        "mfe_atr_quantiles": frame["mfe_atr"].quantile([0.1, 0.5, 0.9]).to_dict(),
        "mae_atr_quantiles": frame["mae_atr"].quantile([0.1, 0.5, 0.9]).to_dict(),
        "targets": targets,
    }


def supervised_section(train: pd.DataFrame, holdout: pd.DataFrame, seed: int) -> dict[str, Any]:
    x_train = finite_frame(train, FEATURES)
    x_test = finite_frame(holdout, FEATURES)
    train = train.loc[x_train.index]
    holdout = holdout.loc[x_test.index]
    result: dict[str, Any] = {"temporal_protocol": "train split -> test split; no random row split"}

    classification_targets = ["direction_up", "good_long"]
    regression_targets = ["future_return", "mfe_atr", "mae_atr", "time_to_mfe"]
    result["classification"] = {}
    for target in classification_targets:
        y_train = train[target].to_numpy(dtype=int)
        y_test = holdout[target].to_numpy(dtype=int)
        if np.unique(y_train).size < 2:
            result["classification"][target] = {"error": "single training class"}
            continue
        tree = ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=8, class_weight="balanced", n_jobs=-1, random_state=seed
        ).fit(x_train, y_train)
        boost = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, random_state=seed).fit(
            x_train, y_train
        )
        tree_score = tree.predict_proba(x_test)[:, 1]
        boost_score = boost.predict_proba(x_test)[:, 1]
        perm = permutation_importance(
            tree, x_test, y_test, scoring="balanced_accuracy", n_repeats=8, random_state=seed, n_jobs=-1
        )
        mi = mutual_info_classif(x_train, y_train, random_state=seed)
        result["classification"][target] = {
            "prevalence_train": y_train.mean(),
            "prevalence_test": y_test.mean(),
            "extra_trees": {
                "balanced_accuracy": balanced_accuracy_score(y_test, tree.predict(x_test)),
                "roc_auc": safe_auc(y_test, tree_score),
            },
            "hist_gradient_boosting": {
                "balanced_accuracy": balanced_accuracy_score(y_test, boost.predict(x_test)),
                "roc_auc": safe_auc(y_test, boost_score),
            },
            "mutual_information": rank_scores(FEATURES, mi),
            "permutation_importance": rank_scores(FEATURES, perm.importances_mean),
        }

    result["regression"] = {}
    for target in regression_targets:
        y_train = train[target].to_numpy(dtype=float)
        y_test = holdout[target].to_numpy(dtype=float)
        tree = ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=8, n_jobs=-1, random_state=seed
        ).fit(x_train, y_train)
        boost = HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15, random_state=seed).fit(
            x_train, y_train
        )
        pred_tree = tree.predict(x_test)
        pred_boost = boost.predict(x_test)
        perm = permutation_importance(
            tree, x_test, y_test, scoring="neg_mean_absolute_error", n_repeats=8, random_state=seed, n_jobs=-1
        )
        mi = mutual_info_regression(x_train, y_train, random_state=seed)
        spearman = np.array([spearmanr(x_train[f], y_train, nan_policy="omit").statistic for f in FEATURES])
        result["regression"][target] = {
            "baseline_mae": mean_absolute_error(y_test, np.full_like(y_test, np.median(y_train))),
            "extra_trees": {"mae": mean_absolute_error(y_test, pred_tree), "r2": r2_score(y_test, pred_tree)},
            "hist_gradient_boosting": {
                "mae": mean_absolute_error(y_test, pred_boost),
                "r2": r2_score(y_test, pred_boost),
            },
            "mutual_information": rank_scores(FEATURES, mi),
            "absolute_spearman": rank_scores(FEATURES, np.abs(spearman)),
            "permutation_importance": rank_scores(FEATURES, perm.importances_mean),
        }
    return result


def interaction_section(train: pd.DataFrame, holdout: pd.DataFrame, seed: int) -> dict[str, Any]:
    """Rank pairwise nonlinear gains on held-out future-return MAE."""
    target = "future_return"
    y_train = train[target].to_numpy(dtype=float)
    y_test = holdout[target].to_numpy(dtype=float)
    baseline = mean_absolute_error(y_test, np.full_like(y_test, np.median(y_train)))
    single_mae: dict[str, float] = {}
    for feature in FEATURES:
        model = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=7, random_state=seed)
        model.fit(train[[feature]], y_train)
        single_mae[feature] = mean_absolute_error(y_test, model.predict(holdout[[feature]]))
    candidates = list(FEATURES)
    pairs = []
    for i, first in enumerate(candidates):
        for second in candidates[i + 1 :]:
            model = HistGradientBoostingRegressor(max_iter=160, max_leaf_nodes=15, random_state=seed)
            model.fit(train[[first, second]], y_train)
            mae = mean_absolute_error(y_test, model.predict(holdout[[first, second]]))
            best_single = min(single_mae[first], single_mae[second])
            pairs.append(
                {
                    "features": [first, second],
                    "mae": mae,
                    "gain_vs_best_single": best_single - mae,
                    "gain_vs_constant": baseline - mae,
                }
            )
    pairs.sort(key=lambda row: row["gain_vs_best_single"], reverse=True)
    return {
        "method": "held-out two-feature gradient boosting gain over the better univariate model",
        "baseline_mae": baseline,
        "candidate_features": candidates,
        "pair_count": len(pairs),
        "all_pairs": pairs,
        "top_pairs": pairs[:15],
    }


def cluster_section(frame: pd.DataFrame, seed: int) -> dict[str, Any]:
    x = finite_frame(frame, FEATURES)
    scaler = StandardScaler()
    z = scaler.fit_transform(x)
    # Cap silhouette cost while preserving deterministic coverage.
    sample_size = min(3000, len(z))
    sample_idx = np.linspace(0, len(z) - 1, sample_size, dtype=int)
    scores: dict[int, float] = {}
    models: dict[int, KMeans] = {}
    for k in range(2, 7):
        model = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(z)
        scores[k] = silhouette_score(z[sample_idx], model.labels_[sample_idx])
        models[k] = model
    best_k = max(scores, key=scores.get)
    labels = models[best_k].labels_
    aligned = frame.loc[x.index].copy()
    aligned["cluster"] = labels
    summaries = []
    atr_high = aligned["atr_pct"].quantile(0.75)
    adx_mid = aligned["adx_14"].median()
    for cluster_id, group in aligned.groupby("cluster"):
        feature_z = pd.Series(models[best_k].cluster_centers_[int(cluster_id)], index=FEATURES)
        if group["atr_pct"].median() >= atr_high:
            regime_name = "forte_volatilite"
        elif group["adx_14"].median() >= adx_mid:
            regime_name = "tendance"
        else:
            regime_name = "range"
        summaries.append(
            {
                "cluster": int(cluster_id),
                "regime_name": regime_name,
                "count": len(group),
                "fraction": len(group) / len(aligned),
                "future_return_mean": group["future_return"].mean(),
                "future_return_std": group["future_return"].std(),
                "mfe_atr_median": group["mfe_atr"].median(),
                "mae_atr_median": group["mae_atr"].median(),
                "recommended_tp_atr": group["mfe_atr"].quantile(0.50),
                "recommended_sl_atr": group["mae_atr"].quantile(0.75),
                "good_long_rate": group["good_long"].mean(),
                "dominant_feature_z": feature_z.abs().sort_values(ascending=False).head(5).index.tolist(),
            }
        )
    return {
        "selection": "maximum silhouette among k=2..6",
        "silhouette": {str(k): score for k, score in scores.items()},
        "selected_k": best_k,
        "clusters": summaries,
    }


def barrier_geometry(frame: pd.DataFrame, horizon: int) -> dict[str, Any]:
    """Simulate ATR barriers; same-candle dual hits are explicitly ambiguous."""
    base = frame.iloc[: len(frame) - horizon]
    close = frame["close"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    atr_pct = np.maximum(base[feature_column("5m", "atr_pct")].to_numpy(dtype=float), 1e-12)
    entry = close[: len(base)]
    atr_raw = entry * atr_pct
    combos = []
    for tp_mult in (1.0, 2.0, 3.0, 4.0, 6.0):
        for sl_mult in (0.5, 1.0, 1.5, 2.0, 3.0):
            counts: Counter[str] = Counter()
            pnl_atr: list[float] = []
            durations: list[int] = []
            for row in range(len(base)):
                tp = entry[row] + tp_mult * atr_raw[row]
                sl = entry[row] - sl_mult * atr_raw[row]
                reason = "MAX_DURATION"
                outcome = (close[row + horizon] - entry[row]) / atr_raw[row]
                duration = horizon
                for step in range(1, horizon + 1):
                    tp_hit = high[row + step] >= tp
                    sl_hit = low[row + step] <= sl
                    if tp_hit and sl_hit:
                        reason, outcome, duration = "AMBIGUOUS_SAME_BAR", np.nan, step
                        break
                    if tp_hit:
                        reason, outcome, duration = "TAKE_PROFIT", tp_mult, step
                        break
                    if sl_hit:
                        reason, outcome, duration = "STOP_LOSS", -sl_mult, step
                        break
                counts[reason] += 1
                durations.append(duration)
                if np.isfinite(outcome):
                    pnl_atr.append(float(outcome))
            combos.append(
                {
                    "tp_atr": tp_mult,
                    "sl_atr": sl_mult,
                    "close_reasons": dict(counts),
                    "mean_duration_bars": np.mean(durations),
                    "expectancy_atr_excluding_ambiguous": np.mean(pnl_atr),
                }
            )
    target = add_future_targets(frame, horizon)
    return {
        "horizon_bars": horizon,
        "entry_count": len(target),
        "geometry_quantiles": {
            name: {f"q{int(q * 100):02d}": target[name].quantile(q) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)}
            for name in ("atr_raw", "atr_pct", "mfe_pct", "mae_pct", "mfe_atr", "mae_atr")
        },
        "counterfactual_barriers": combos,
        "ambiguity_policy": "If TP and SL occur in one OHLC bar, no intrabar ordering is invented.",
    }


def good_bad_section(frame: pd.DataFrame) -> dict[str, Any]:
    good = frame[frame["good_long"] == 1]
    bad = frame[frame["bad_long"] == 1]
    rows = []
    for feature in FEATURES:
        pooled = frame[feature].std(ddof=1)
        delta = good[feature].mean() - bad[feature].mean()
        rows.append(
            {
                "feature": feature,
                "good_mean": good[feature].mean(),
                "bad_mean": bad[feature].mean(),
                "standardized_mean_difference": delta / pooled if pooled > 0 else 0.0,
            }
        )
    rows.sort(key=lambda row: abs(row["standardized_mean_difference"]), reverse=True)
    return {
        "definition": "long entry: good if horizon return > +20 bps; bad if < -20 bps",
        "good_count": len(good),
        "bad_count": len(bad),
        "neutral_count": len(frame) - len(good) - len(bad),
        "feature_contrasts": rows,
    }


def find_checkpoint(checkpoint_root: Path) -> Path | None:
    candidates = list(checkpoint_root.glob("**/model.zip"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def aligned_observations(
    data: dict[str, dict[str, pd.DataFrame]], sample_count: int, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    """Build timestamp-aligned, normalized validation observations."""
    from adan_trading_bot.data_processing.state_builder import StateBuilder
    from adan_trading_bot.trading.live_state_builder import OBS_WINDOW, TRAIN_COLUMNS

    train = data["train"]
    val = data["val"]
    builder = StateBuilder(
        features_config=TRAIN_COLUMNS,
        window_sizes={tf: OBS_WINDOW for tf in ("5m", "1h", "4h")},
        include_portfolio_state=True,
        normalize=True,
    )
    builder.scalers = {}
    builder.scalers_loaded_from_training = False
    builder.fit_scalers({"BTCUSDT": train})
    builder.scalers_loaded_from_training = True

    eligible = val["5m"].index[OBS_WINDOW:]
    rng = np.random.default_rng(seed)
    timestamps = np.sort(rng.choice(eligible, size=min(sample_count, len(eligible)), replace=False))
    batches: dict[str, list[np.ndarray]] = {tf: [] for tf in ("5m", "1h", "4h")}
    contexts: list[np.ndarray] = []
    for timestamp in timestamps:
        flat: dict[str, pd.DataFrame] = {}
        for tf in ("5m", "1h", "4h"):
            source = pd.concat([train[tf], val[tf]]).sort_index()
            window = source.loc[:timestamp].tail(OBS_WINDOW).copy()
            if len(window) < OBS_WINDOW:
                break
            values = window[TRAIN_COLUMNS[tf]].to_numpy(dtype=np.float64)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            values = builder.scalers[tf].transform(values)
            batches[tf].append(np.clip(values, -10.0, 10.0).astype(np.float32))
            flat[tf] = window
        else:
            contexts.append(builder.build_context_vector(flat, current_idx=len(flat["5m"])))
            continue
        # Remove a partial sample if one timeframe lacked enough history.
        target_len = len(contexts)
        for tf in batches:
            batches[tf] = batches[tf][:target_len]

    count = len(contexts)
    observations = {tf: np.stack(batches[tf][:count]) for tf in batches}
    observations["context_vector"] = np.stack(contexts).astype(np.float32)
    observations["portfolio_state"] = np.zeros((count, 28), dtype=np.float32)
    indices = {
        tf: {feature: TRAIN_COLUMNS[tf].index(feature_column(tf, feature)) for feature in FEATURES}
        for tf in ("5m", "1h", "4h")
    }
    return observations, indices


def policy_attribution(
    data: dict[str, dict[str, pd.DataFrame]], checkpoint: Path | None, sample_count: int, seed: int
) -> dict[str, Any]:
    if checkpoint is None:
        return {"available": False, "reason": "no model.zip found"}
    try:
        from stable_baselines3 import PPO

        import torch

        model = PPO.load(checkpoint, device="cpu")
        observations, indices = aligned_observations(data, sample_count, seed)

        def policy_means(obs: dict[str, np.ndarray]) -> np.ndarray:
            """Return distribution means before Box clipping hides saturation."""
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                distribution = model.policy.get_distribution(obs_tensor)
                return distribution.distribution.mean.detach().cpu().numpy().astype(float)

        baseline_postclip, _ = model.predict(observations, deterministic=True)
        baseline_postclip = np.asarray(baseline_postclip, dtype=float)
        baseline = policy_means(observations)
        rng = np.random.default_rng(seed)
        per_tf: dict[str, Any] = {}
        combined: dict[str, Any] = {}
        for tf in ("5m", "1h", "4h"):
            per_tf[tf] = {}
            for feature in FEATURES:
                changed = {key: value.copy() for key, value in observations.items()}
                permutation = rng.permutation(len(baseline))
                column = indices[tf][feature]
                changed[tf][:, :, column] = changed[tf][permutation, :, column]
                action = policy_means(changed)
                delta = np.abs(action - baseline).mean(axis=0)
                per_tf[tf][feature] = {ACTION_HEADS[i]: delta[i] for i in range(len(ACTION_HEADS))}
        for feature in FEATURES:
            changed = {key: value.copy() for key, value in observations.items()}
            permutation = rng.permutation(len(baseline))
            for tf in ("5m", "1h", "4h"):
                column = indices[tf][feature]
                changed[tf][:, :, column] = changed[tf][permutation, :, column]
            action = policy_means(changed)
            delta = np.abs(action - baseline).mean(axis=0)
            combined[feature] = {ACTION_HEADS[i]: delta[i] for i in range(len(ACTION_HEADS))}
        rankings = {
            head: sorted(
                ({"feature": f, "mean_absolute_action_delta": combined[f][head]} for f in FEATURES),
                key=lambda row: row["mean_absolute_action_delta"],
                reverse=True,
            )
            for head in ACTION_HEADS
        }
        return {
            "available": True,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "model_class": type(model).__name__,
            "extractor_class": type(model.policy.features_extractor).__name__,
            "use_sde": bool(model.use_sde),
            "sample_count": len(baseline),
            "method": "timestamp-aligned validation observations; within-feature sample permutation; pre-clipping policy-distribution mean delta",
            "limitations": [
                "Sensitivity is associational, not a causal or SHAP attribution.",
                "Context and portfolio vectors are held fixed during indicator permutations.",
                "A smoke checkpoint measures the current smoke policy, not a converged 500k policy.",
                "Pre-clipping means are required because Box clipping makes saturated heads look exactly constant.",
            ],
            "saturated_postclip_heads": [
                ACTION_HEADS[i]
                for i in range(len(ACTION_HEADS))
                if np.mean(np.abs(baseline_postclip[:, i]) >= 0.999) >= 0.95
            ],
            "baseline_preclip_mean": {
                ACTION_HEADS[i]: {"mean": baseline[:, i].mean(), "std": baseline[:, i].std()}
                for i in range(len(ACTION_HEADS))
            },
            "baseline_postclip_action": {
                ACTION_HEADS[i]: {
                    "mean": baseline_postclip[:, i].mean(),
                    "std": baseline_postclip[:, i].std(),
                    "saturation_fraction": np.mean(np.abs(baseline_postclip[:, i]) >= 0.999),
                }
                for i in range(len(ACTION_HEADS))
            },
            "combined_timeframe_rankings": rankings,
            "combined_raw": combined,
            "per_timeframe_raw": per_tf,
        }
    except Exception as exc:  # analysis must still produce its dataset evidence
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def recommendations(report: dict[str, Any]) -> list[dict[str, str]]:
    reg = report["market_models"]["regression"]
    mfe_r2 = max(reg["mfe_atr"][name]["r2"] for name in ("extra_trees", "hist_gradient_boosting"))
    mae_r2 = max(reg["mae_atr"][name]["r2"] for name in ("extra_trees", "hist_gradient_boosting"))
    direction_auc = max(
        value or 0.5
        for value in (
            report["market_models"]["classification"]["direction_up"]["extra_trees"]["roc_auc"],
            report["market_models"]["classification"]["direction_up"]["hist_gradient_boosting"]["roc_auc"],
        )
    )
    actions = report["policy_attribution"]
    recs = [
        {
            "decision": "Represent SL/TP in ATR multiples",
            "status": "SUPPORTED_FOR_ABLATION",
            "evidence": "ATR units normalize volatility-scale drift and enable comparable barrier geometry, but the weak held-out risk predictability does not justify an untested production replacement.",
        },
        {
            "decision": "Add current SL/TP distances and remaining time to the observation while a position is open",
            "status": "SUPPORTED",
            "evidence": "These are Markov state variables required to evaluate risk geometry; they are not future leakage.",
        },
    ]
    if max(mfe_r2, mae_r2) >= 0.10:
        recs.append(
            {
                "decision": "Train a separate Risk Head on ex-post Future Arena labels",
                "status": "SUPPORTED_FOR_CONTROLLED_COMPARISON",
                "evidence": f"Held-out nonlinear predictability exists (best MFE/MAE R²={max(mfe_r2, mae_r2):.3f}); compare against PPO risk actions before replacement.",
            }
        )
    else:
        recs.append(
            {
                "decision": "Train a separate Risk Head on ex-post Future Arena labels",
                "status": "NOT_YET_SUPPORTED",
                "evidence": f"Held-out MFE/MAE predictability is weak (best R²={max(mfe_r2, mae_r2):.3f}); improve labels/data before architectural separation.",
            }
        )
    recs.append(
        {
            "decision": "Remove SL/TP from PPO action immediately",
            "status": "REJECTED_PENDING_ABLATION",
            "evidence": "Attribution and market predictability alone do not prove that a supervised Risk Head outperforms joint PPO control; run matched-seed ablations first.",
        }
    )
    recs.append(
        {
            "decision": "Change PPO/reward from this bulletin alone",
            "status": "REJECTED",
            "evidence": f"Direction predictability is reported objectively (best held-out AUC={direction_auc:.3f}); no intuitive reward edit is justified without an ablation.",
        }
    )
    if not actions.get("available"):
        recs.append(
            {
                "decision": "Draw conclusions about policy-head feature usage",
                "status": "BLOCKED",
                "evidence": actions.get("reason", "checkpoint attribution unavailable"),
            }
        )
    return recs


def arena_verdict(report: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "exactly_16_canonical_features": report["feature_count"] == 16 and report["features"] == FEATURES,
        "all_120_nonlinear_pairs": report["nonlinear_interactions"].get("pair_count") == 120,
        "trade_join_complete": report["trade_telemetry_audit"].get("join_rate") == 1.0,
        "trade_sample_sufficient": report["trade_telemetry_audit"].get("joined_count", 0) >= 100,
        "strict_json_telemetry": report["trade_telemetry_audit"].get("invalid_json") == 0,
        "policy_attribution_available": bool(report["policy_attribution"].get("available")),
    }
    market_target = report["market_models"]["classification"]["good_long"]
    best_auc = max(
        market_target["extra_trees"].get("roc_auc") or 0.0,
        market_target["hist_gradient_boosting"].get("roc_auc") or 0.0,
    )
    checks["market_signal_above_random"] = best_auc >= 0.52
    trade_target = report["trade_arena"].get("targets", {}).get("mfe_gt_tp_min", {})
    if trade_target.get("error"):
        checks["trade_target_evaluable"] = False
    else:
        checks["trade_target_evaluable"] = bool(trade_target)
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "GREEN" if not failed else "RED",
        "checks": checks,
        "failed_checks": failed,
        "authorization": "ARENA_GATE_PASSED" if not failed else "500K_FORBIDDEN",
        "note": "A red verdict preserves all 16 features; it requests better evidence, never feature deletion.",
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ADAN — Bulletin d’intelligence dataset/modèle",
        "",
        f"Généré: `{report['generated_at_utc']}`  ",
        f"Horizon principal: **{report['horizon_bars']} barres 5m**.",
        "",
        "## Verdict Arena",
        "",
        f"**{report['arena_verdict']['status']} — {report['arena_verdict']['authorization']}**",
        "",
        *[f"- {'PASS' if passed else 'FAIL'} `{name}`" for name, passed in report['arena_verdict']['checks'].items()],
        "",
        "## Portée et intégrité",
        "",
        "- Les labels futurs sont utilisés uniquement pour l’analyse ex-post, jamais comme observation acteur.",
        "- Aucun rapprochement ambigu des trades par prix n’est effectué.",
        "- Les raisons de fermeture ci-dessous sont des simulations contrefactuelles OHLC; un double hit dans une barre reste `AMBIGUOUS_SAME_BAR`.",
        "- SHAP n’est pas requis: information mutuelle, Extra Trees, gradient boosting et permutation held-out sont fournis.",
        "",
        "## Qualité prédictive hors échantillon",
        "",
        "| Cible | Extra Trees | Gradient Boosting | Baseline |",
        "|---|---:|---:|---:|",
    ]
    for target, values in report["market_models"]["regression"].items():
        lines.append(
            f"| {target} (R²) | {values['extra_trees']['r2']:.3f} | "
            f"{values['hist_gradient_boosting']['r2']:.3f} | MAE constante {values['baseline_mae']:.4g} |"
        )
    lines.extend(["", "| Cible classification | Extra Trees AUC | Boosting AUC |", "|---|---:|---:|"])
    for target, values in report["market_models"]["classification"].items():
        et = values["extra_trees"]["roc_auc"]
        gb = values["hist_gradient_boosting"]["roc_auc"]
        lines.append(f"| {target} | {et if et is not None else float('nan'):.3f} | {gb if gb is not None else float('nan'):.3f} |")

    lines.extend(["", "## Features dominantes par permutation held-out", ""])
    for target, values in report["market_models"]["regression"].items():
        top = ", ".join(row["feature"] for row in values["permutation_importance"][:5])
        lines.append(f"- **{target}**: {top}")

    trade = report["trade_arena"]
    lines.extend(["", "## Arena conditionnée par les trades v24", ""])
    lines.append(
        f"Jointure exacte: {report['trade_telemetry_audit']['joined_count']}/"
        f"{report['trade_telemetry_audit']['open_count']} trades."
    )
    if trade.get("available"):
        lines.append(f"PnL net des trades joints: {trade['pnl_net_sum']:.6f}.")
        for target, values in trade["targets"].items():
            if values.get("error"):
                lines.append(f"- **{target}**: non évaluable ({values['error']}).")
            else:
                lines.append(
                    f"- **{target}**: AUC Extra Trees {values['extra_trees']['roc_auc']:.3f}, "
                    f"AUC Gradient Boosting {values['hist_gradient_boosting']['roc_auc']:.3f}."
                )

    lines.extend(["", "## Bons trades vs mauvais trades (long contrefactuel)", ""])
    gb = report["good_vs_bad"]
    lines.append(
        f"Définition: {gb['definition']}. Bons={gb['good_count']}, mauvais={gb['bad_count']}, neutres={gb['neutral_count']}."
    )
    lines.append("")
    lines.append("Plus grands écarts standardisés: " + ", ".join(
        f"{row['feature']} ({row['standardized_mean_difference']:+.2f}σ)" for row in gb["feature_contrasts"][:8]
    ))

    lines.extend(["", "## Régimes", ""])
    clusters = report["regimes"]
    lines.append(f"K sélectionné par silhouette: **{clusters['selected_k']}**.")
    for cluster in clusters["clusters"]:
        lines.append(
            f"- Régime {cluster['cluster']} (**{cluster['regime_name']}**): {cluster['fraction']:.1%}, retour moyen "
            f"{cluster['future_return_mean']:.3%}, MFE/ATR médian {cluster['mfe_atr_median']:.2f}, "
            f"MAE/ATR médian {cluster['mae_atr_median']:.2f}, TP conseillé {cluster['recommended_tp_atr']:.2f} ATR, "
            f"SL conseillé {cluster['recommended_sl_atr']:.2f} ATR."
        )

    lines.extend(["", "## Sorties réseau", ""])
    attribution = report["policy_attribution"]
    if attribution.get("available"):
        lines.append(f"Checkpoint: `{attribution['checkpoint']}` ({attribution['extractor_class']}).")
        saturated = attribution["saturated_postclip_heads"]
        if saturated:
            lines.append(
                "Têtes saturées après clipping sur ≥95% des observations: **"
                + ", ".join(saturated)
                + "**. Les rangs ci-dessous utilisent donc les moyennes pré-clipping."
            )
        for head in ("direction", "size", "sl", "tp"):
            top = attribution["combined_timeframe_rankings"][head][:5]
            lines.append("- **" + head + "**: " + ", ".join(
                f"{row['feature']} ({row['mean_absolute_action_delta']:.4g})" for row in top
            ))
    else:
        lines.append(f"Attribution indisponible: {attribution.get('reason')}")

    lines.extend(["", "## Décisions", ""])
    for rec in report["recommendations"]:
        lines.append(f"- **{rec['status']} — {rec['decision']}**: {rec['evidence']}")

    lines.extend(
        [
            "",
            "## Limites bloquantes de télémétrie trade",
            "",
            *[f"- {item}" for item in report["telemetry_limitations"]],
            "",
            "Le JSON contient l’ensemble des distributions, importances, interactions, clusters et grilles TP/SL/ATR.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "data/processed/indicators", help="indicator split root"
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="optional exact model.zip")
    parser.add_argument(
        "--checkpoint-root", type=Path, default=ROOT / "checkpoints/v21_smoke_gsde", help="checkpoint search root"
    )
    parser.add_argument("--horizon", type=int, default=36, help="5m future horizon in bars")
    parser.add_argument("--tp-min", type=float, default=0.005, help="minimum TP target used by MFE > TP_min")
    parser.add_argument(
        "--input",
        "--telemetry",
        dest="telemetry",
        nargs="+",
        default=["logs/action_pipeline/v24_smoke_ray_w*.jsonl"],
        help="lifecycle JSONL files or glob patterns",
    )
    parser.add_argument("--policy-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument(
        "--json-output",
        "--output",
        dest="json_output",
        type=Path,
        default=ROOT / "reports/model_dataset_intelligence.json",
    )
    parser.add_argument("--markdown-output", type=Path, default=ROOT / "reports/model_dataset_intelligence.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.horizon < 2:
        raise ValueError("--horizon must be >= 2")
    os.chdir(ROOT)
    data = load_data(args.data_root.resolve())
    train_targets = add_future_targets(data["train"]["5m"], args.horizon)
    test_targets = add_future_targets(data["test"]["5m"], args.horizon)
    combined_targets = pd.concat([train_targets, test_targets]).sort_index()
    full_5m_bars = pd.concat(
        [data[split]["5m"] for split in ("train", "test", "val")]
    ).sort_index()
    if not full_5m_bars.index.is_unique:
        raise ValueError("5m split timestamps are not unique; exact Arena join is unsafe")
    telemetry_paths = resolve_telemetry_paths(args.telemetry)
    trade_frame, trade_audit = load_trade_arena(
        telemetry_paths,
        full_5m_bars,
        args.horizon,
        args.tp_min,
    )
    checkpoint = args.checkpoint.resolve() if args.checkpoint else find_checkpoint(args.checkpoint_root.resolve())

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "horizon_bars": args.horizon,
        "horizon_minutes": args.horizon * 5,
        "tp_min": args.tp_min,
        "dataset_rows": {
            split: {tf: len(frame) for tf, frame in by_tf.items()} for split, by_tf in data.items()
        },
        "distributions": distribution_section(data),
        "good_vs_bad": good_bad_section(test_targets),
        "market_models": supervised_section(train_targets, test_targets, args.seed),
        "trade_telemetry_audit": trade_audit,
        "trade_arena": trade_arena_section(trade_frame, args.seed),
        "nonlinear_interactions": interaction_section(train_targets, test_targets, args.seed),
        "regimes": cluster_section(combined_targets, args.seed),
        "risk_geometry": {
            str(horizon): barrier_geometry(data["test"]["5m"], horizon)
            for horizon in sorted({15, args.horizon})
            if horizon < len(data["test"]["5m"])
        },
        "policy_attribution": policy_attribution(data, checkpoint, args.policy_samples, args.seed),
        "telemetry_limitations": [
            "Trade MFE/MAE uses exact entry timestamp and a fixed future horizon; it does not invent intrabar ordering.",
            "The immutable feature snapshot is the decision-close row t while execution is open[t+1], preventing actor leakage.",
            "Policy attribution uses real validation observations but does not claim SHAP-equivalent causal credit.",
            "A 2048-step smoke policy is diagnostic evidence, not a converged production policy.",
        ],
    }
    report["recommendations"] = recommendations(report)
    report["arena_verdict"] = arena_verdict(report)
    clean = native(report)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(clean, indent=2, sort_keys=False, allow_nan=False) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(clean), encoding="utf-8")
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.markdown_output}")
    if not clean["policy_attribution"].get("available"):
        print("WARNING: policy attribution unavailable:", clean["policy_attribution"].get("reason"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
