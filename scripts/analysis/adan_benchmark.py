#!/usr/bin/env python3
"""Permanent, asset-agnostic ADAN benchmark.

The benchmark separates observable model skill from market opportunity and from
non-causal physical ceilings. It never loads future data into a policy and does
not alter training, rewards, capital tiers, fees, features, or architecture.

Opportunity classes are calibrated from the selected calibration split using
empirical quantiles of a fee-aware quality statistic. They are therefore not
hard-coded for BTC, crypto, equities, or any particular price/volatility scale.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC_ROOT))

from adan_trading_bot.arena_predictor.state_schema import PresentState  # noqa: E402

LABELS = ("toxic", "weak", "neutral", "good", "excellent")
PPO_KEYS = (
    "ep_rew_mean",
    "approx_kl",
    "clip_fraction",
    "entropy_loss",
    "policy_gradient_loss",
    "value_loss",
    "explained_variance",
    "std",
)
METRIC_RE = re.compile(r"\|\s+([a-zA-Z0-9_/]+)\s+\|\s+(-?[0-9.eE+]+)\s+\|")
ANCHOR_RE = re.compile(
    r"ANCHOR_DEBUG.*?a0_mean=([-+0-9.eE]+).*?a0_std=([-+0-9.eE]+)"
)


@dataclass(frozen=True)
class CostModel:
    commission_per_side: float
    slippage_per_side: float

    @property
    def round_trip_commission(self) -> float:
        return 2.0 * self.commission_per_side

    @property
    def round_trip_execution_cost(self) -> float:
        return 2.0 * (self.commission_per_side + self.slippage_per_side)

    def as_dict(self) -> dict[str, float]:
        return {
            "commission_per_side": self.commission_per_side,
            "slippage_per_side": self.slippage_per_side,
            "round_trip_commission": self.round_trip_commission,
            "round_trip_execution_cost": self.round_trip_execution_cost,
        }


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if denominator else default


def describe(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0}
    quantiles = np.quantile(array, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "max": float(array.max()),
    }


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def resolve_cost_model(config: Mapping[str, Any]) -> CostModel:
    rules = config.get("trading_rules", {})
    commission = finite_float(rules.get("commission_pct"), 0.0)
    slippage = finite_float(rules.get("slippage_pct"), 0.0)
    if commission < 0 or slippage < 0:
        raise ValueError("Commission and slippage must be non-negative")
    return CostModel(commission, slippage)


def resolve_horizon(config: Mapping[str, Any], requested: int | None) -> int:
    if requested is not None:
        horizon = requested
    else:
        future = config.get("reward_shaping", {}).get("future_reward", {})
        horizon = int(future.get("horizon", 36))
    if horizon <= 0:
        raise ValueError("Horizon must be positive")
    return horizon


def discover_split_paths(
    data_root: Path, asset: str | None, timeframe: str
) -> dict[str, Path]:
    candidates = sorted(data_root.glob(f"*/*/{timeframe}.parquet"))
    if asset:
        candidates = [path for path in candidates if path.parent.name.upper() == asset.upper()]
    if not candidates:
        raise FileNotFoundError(
            f"No {timeframe}.parquet under {data_root}/<split>/<asset>/"
        )
    assets = sorted({path.parent.name for path in candidates})
    selected_asset = asset or assets[0]
    if len(assets) > 1 and not asset:
        raise ValueError(f"Multiple assets found {assets}; pass --asset")
    return {
        path.parent.parent.name: path
        for path in candidates
        if path.parent.name.upper() == selected_asset.upper()
    }


def load_split(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).sort_index()
    required = {"open", "high", "low", "close"}
    missing = required.difference(str(column).lower() for column in frame.columns)
    if missing:
        raise ValueError(f"{path} lacks OHLC columns: {sorted(missing)}")
    frame.columns = [str(column).lower() for column in frame.columns]
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate timestamps in {path}")
    if frame[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError(f"NaN OHLC values in {path}")
    return frame


def future_excursions(frame: pd.DataFrame, horizon: int, cost: float) -> pd.DataFrame:
    """Compute long-only ex-post opportunity statistics from each decision bar.

    Entry is next-bar open. Future bars [t+1, t+horizon] are privileged labels,
    never causal features. Ratios make the result invariant to nominal price.
    """
    count = len(frame)
    rows: list[dict[str, Any]] = []
    opens = frame["open"].to_numpy(dtype=np.float64)
    highs = frame["high"].to_numpy(dtype=np.float64)
    lows = frame["low"].to_numpy(dtype=np.float64)
    for index in range(max(0, count - 1)):
        end = min(count, index + horizon + 1)
        if index + 1 >= end:
            continue
        entry = opens[index + 1]
        if not math.isfinite(entry) or entry <= 0:
            continue
        future_high = highs[index + 1 : end]
        future_low = lows[index + 1 : end]
        mfe_path = (future_high - entry) / entry
        mae_path = (entry - future_low) / entry
        mfe = max(0.0, float(np.nanmax(mfe_path)))
        mae = max(0.0, float(np.nanmax(mae_path)))
        time_to_mfe = int(np.nanargmax(mfe_path)) + 1
        net_mfe = mfe - cost
        fee_aware_quality = net_mfe / max(mae + cost, np.finfo(float).eps)
        rows.append(
            {
                "row_index": index,
                "timestamp": frame.index[index],
                "entry_timestamp": frame.index[index + 1],
                "entry_price": entry,
                "mfe": mfe,
                "mae": mae,
                "net_mfe": net_mfe,
                "rr": mfe / max(mae, np.finfo(float).eps),
                "fee_aware_quality": fee_aware_quality,
                "time_to_mfe": time_to_mfe,
                "economically_viable": bool(net_mfe > 0),
            }
        )
    return pd.DataFrame(rows).set_index("row_index", drop=False)


def adaptive_thresholds(calibration_quality: Sequence[float]) -> list[float]:
    values = np.asarray(calibration_quality, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < len(LABELS):
        raise ValueError("Not enough finite opportunities to calibrate five classes")
    return [float(value) for value in np.quantile(values, [0.2, 0.4, 0.6, 0.8])]


def classify_quality(values: Sequence[float], thresholds: Sequence[float]) -> np.ndarray:
    if len(thresholds) != len(LABELS) - 1:
        raise ValueError("Expected four adaptive thresholds")
    return np.asarray(LABELS, dtype=object)[
        np.digitize(np.asarray(values, dtype=np.float64), thresholds, right=True)
    ]


def dataset_report(
    frame: pd.DataFrame, opportunities: pd.DataFrame, cost: CostModel
) -> dict[str, Any]:
    returns = frame["close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if "atr_pct" in frame:
        atr = frame["atr_pct"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    else:
        true_range = (frame["high"] - frame["low"]) / frame["close"].replace(0, np.nan)
        atr = true_range.dropna()
    absolute_path = float(returns.abs().sum())
    net_path = abs(float(frame["close"].iloc[-1] / frame["close"].iloc[0] - 1.0))
    return {
        "methodology": "causal market description plus explicitly labelled ex-post excursions",
        "rows": int(len(frame)),
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_timestamps": int(frame.index.duplicated().sum()),
        "bar_returns": describe(returns),
        "atr_pct": describe(atr),
        "atr_to_round_trip_cost_median": safe_div(
            float(atr.median()) if len(atr) else 0.0,
            cost.round_trip_execution_cost,
        ),
        "path_efficiency": safe_div(net_path, absolute_path),
        "mfe": describe(opportunities["mfe"]),
        "mae": describe(opportunities["mae"]),
        "time_to_mfe_steps": describe(opportunities["time_to_mfe"]),
        "economically_viable_rate": float(opportunities["economically_viable"].mean()),
    }


def opportunity_report(
    opportunities: pd.DataFrame, thresholds: Sequence[float]
) -> dict[str, Any]:
    classes = classify_quality(opportunities["fee_aware_quality"], thresholds)
    counts = Counter(str(value) for value in classes)
    total = len(classes)
    return {
        "methodology": "ex-post non-causal physical opportunity map",
        "class_basis": "calibration-split empirical quintiles of (MFE-cost)/(MAE+cost)",
        "adaptive_thresholds": list(thresholds),
        "class_counts": {label: int(counts[label]) for label in LABELS},
        "class_rates": {label: safe_div(counts[label], total) for label in LABELS},
        "fee_aware_quality": describe(opportunities["fee_aware_quality"]),
        "viable_count": int(opportunities["economically_viable"].sum()),
        "viable_rate": float(opportunities["economically_viable"].mean()),
    }


def state_key(vector: Sequence[float], decimals: int = 9) -> tuple[float, ...]:
    return tuple(round(finite_float(value), decimals) for value in vector)


def build_state_index(
    frame: pd.DataFrame, timeframe: str
) -> tuple[dict[tuple[float, ...], list[int]], int]:
    mapping: dict[tuple[float, ...], list[int]] = defaultdict(list)
    for index, (_, row) in enumerate(frame.iterrows()):
        vector = PresentState.from_market_row(row.to_dict(), timeframe=timeframe).to_vector()
        mapping[state_key(vector)].append(index)
    ambiguous = sum(1 for indexes in mapping.values() if len(indexes) > 1)
    return dict(mapping), ambiguous


def load_arena(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def align_arena_entries(
    records: Sequence[Mapping[str, Any]],
    frame: pd.DataFrame,
    opportunities: pd.DataFrame,
    timeframe: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mapping, ambiguous_keys = build_state_index(frame, timeframe)
    aligned: list[dict[str, Any]] = []
    missing = 0
    ambiguous_records = 0
    for record in records:
        candidates = mapping.get(state_key(record.get("state", [])), [])
        if not candidates:
            missing += 1
            continue
        if len(candidates) > 1:
            ambiguous_records += 1
        row_index = candidates[0]
        if row_index not in opportunities.index:
            missing += 1
            continue
        item = dict(record)
        item["row_index"] = row_index
        item["opportunity"] = opportunities.loc[row_index].to_dict()
        aligned.append(item)
    return aligned, {
        "records": len(records),
        "matched": len(aligned),
        "missing": missing,
        "ambiguous_state_keys": ambiguous_keys,
        "records_on_ambiguous_keys": ambiguous_records,
        "unique_entry_rows": len({item["row_index"] for item in aligned}),
    }


def entry_report(
    aligned: Sequence[Mapping[str, Any]], opportunities: pd.DataFrame
) -> dict[str, Any]:
    viable_rows = set(
        int(value) for value in opportunities.loc[opportunities["economically_viable"], "row_index"]
    )
    entered_rows = {int(item["row_index"]) for item in aligned}
    event_true_positive = sum(
        bool(item["opportunity"]["economically_viable"]) for item in aligned
    )
    event_false_positive = len(aligned) - event_true_positive
    unique_true_positive = len(entered_rows & viable_rows)
    false_negative = len(viable_rows - entered_rows)
    nonviable_rows = set(int(value) for value in opportunities["row_index"]) - viable_rows
    avoided_nonviable = len(nonviable_rows - entered_rows)
    base_rate = safe_div(len(viable_rows), len(opportunities))
    precision = safe_div(event_true_positive, len(aligned))
    recall = safe_div(unique_true_positive, len(viable_rows))
    avoidance = safe_div(avoided_nonviable, len(nonviable_rows))
    return {
        "methodology": "entry decisions scored against ex-post labels; no PnL used",
        "event_precision": precision,
        "unique_opportunity_recall": recall,
        "f1": safe_div(2.0 * precision * recall, precision + recall),
        "false_positive_events": event_false_positive,
        "false_negative_unique_rows": false_negative,
        "good_opportunity_hit_rate": recall,
        "bad_opportunity_avoidance": avoidance,
        "market_viable_base_rate": base_rate,
        "precision_uplift_points_vs_market": precision - base_rate,
        "entered_events": len(aligned),
        "entered_unique_rows": len(entered_rows),
    }


def exit_report(aligned: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pnl = np.asarray(
        [finite_float(item.get("meta", {}).get("pnl_net")) for item in aligned],
        dtype=np.float64,
    )
    reasons = Counter(str(item.get("meta", {}).get("reason", "unknown")) for item in aligned)
    oracle_wins = np.asarray(
        [bool(item["opportunity"]["economically_viable"]) for item in aligned], dtype=bool
    )
    actual_wins = pnl > 0
    held = np.asarray(
        [finite_float(item.get("meta", {}).get("steps_held")) for item in aligned]
    )
    time_to_mfe = np.asarray(
        [finite_float(item["opportunity"]["time_to_mfe"]) for item in aligned]
    )
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = abs(float(pnl[pnl < 0].sum()))
    opportunity_count = int(oracle_wins.sum())
    converted = int(np.logical_and(oracle_wins, actual_wins).sum())
    regrets = int(np.logical_and(oracle_wins, ~actual_wins).sum())
    return {
        "methodology": "actual outcomes on identical entries versus ex-post exit ceiling",
        "trades": int(len(pnl)),
        "win_rate": float(actual_wins.mean()) if len(actual_wins) else 0.0,
        "profit_factor": safe_div(gross_profit, gross_loss),
        "expectancy_pnl_units": float(pnl.mean()) if len(pnl) else 0.0,
        "total_pnl_units": float(pnl.sum()),
        "oracle_same_entry_win_rate": float(oracle_wins.mean()) if len(oracle_wins) else 0.0,
        "profitable_opportunity_conversion": safe_div(converted, opportunity_count),
        "exit_regret_rate": safe_div(regrets, opportunity_count),
        "hold_steps": describe(held),
        "time_to_mfe_steps_same_entries": describe(time_to_mfe),
        "exit_delay_vs_mfe_steps": describe(held - time_to_mfe),
        "close_reasons": dict(sorted(reasons.items())),
        "max_duration_rate": safe_div(
            reasons.get("MaxDuration", 0) + reasons.get("MAX_DURATION", 0), len(aligned)
        ),
        "mfe_capture_ratio": None,
        "mfe_capture_ratio_unavailable_reason": (
            "Arena V19 stores PnL in currency units but omits entry notional/exit price; "
            "a dimensionally valid PnL-return/MFE ratio cannot be reconstructed."
        ),
    }


def risk_report(aligned: Sequence[Mapping[str, Any]], initial_capital: float) -> dict[str, Any]:
    pnl = np.asarray(
        [finite_float(item.get("meta", {}).get("pnl_net")) for item in aligned],
        dtype=np.float64,
    )
    if not len(pnl):
        return {"methodology": "actual sequential trade outcomes", "trades": 0}

    # Arena concatenates independent episodes. Replaying all 8,445 trades as one
    # account fabricates impossible negative equity and a >100% drawdown. A new
    # episode is identified only from the persisted monotonic global_step reset.
    episode_pnl: list[list[float]] = [[]]
    previous_step: float | None = None
    for item, value in zip(aligned, pnl):
        step = finite_float(item.get("meta", {}).get("global_step"), -1.0)
        if previous_step is not None and step < previous_step:
            episode_pnl.append([])
        episode_pnl[-1].append(float(value))
        previous_step = step
    episode_pnl = [episode for episode in episode_pnl if episode]

    all_returns: list[float] = []
    all_drawdowns: list[float] = []
    ending_equities: list[float] = []
    minimum_equities: list[float] = []
    for episode in episode_pnl:
        episode_array = np.asarray(episode, dtype=np.float64)
        equity = initial_capital + np.cumsum(episode_array)
        previous_equity = np.r_[initial_capital, equity[:-1]]
        peaks = np.maximum.accumulate(np.r_[initial_capital, equity])[:-1]
        all_returns.extend(
            (episode_array / np.maximum(previous_equity, np.finfo(float).eps)).tolist()
        )
        all_drawdowns.extend(
            ((equity - peaks) / np.maximum(peaks, np.finfo(float).eps)).tolist()
        )
        ending_equities.append(float(equity[-1]))
        minimum_equities.append(float(equity.min()))

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    win_rate = float((pnl > 0).mean())
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = abs(float(losses.mean())) if len(losses) else 0.0
    payoff = safe_div(avg_win, avg_loss)
    kelly = win_rate - safe_div(1.0 - win_rate, payoff) if payoff else 0.0
    var95 = float(np.quantile(pnl, 0.05))
    tail = pnl[pnl <= var95]
    per_trade_returns = np.asarray(all_returns, dtype=np.float64)
    downside = per_trade_returns[per_trade_returns < 0]
    return {
        "methodology": (
            "actual sequential Arena PnL, reset at persisted episode boundaries; "
            "no annualization without a stable clock"
        ),
        "initial_capital": initial_capital,
        "episodes": len(episode_pnl),
        "total_pnl_all_episodes": float(pnl.sum()),
        "mean_episode_ending_equity": float(np.mean(ending_equities)),
        "median_episode_ending_equity": float(np.median(ending_equities)),
        "minimum_episode_equity": float(np.min(minimum_equities)),
        "ending_equity": float(np.mean(ending_equities)),
        "max_drawdown": abs(float(np.min(all_drawdowns))),
        "var_95_pnl_units": var95,
        "expected_shortfall_95_pnl_units": float(tail.mean()) if len(tail) else var95,
        "sharpe_per_trade": safe_div(float(per_trade_returns.mean()), float(per_trade_returns.std())),
        "sortino_per_trade": safe_div(
            float(per_trade_returns.mean()), float(downside.std()) if len(downside) else 0.0
        ),
        "empirical_kelly_fraction": float(kelly),
        "position_sizing_audit": None,
        "position_sizing_unavailable_reason": "Arena V19 does not persist entry notional or risk-at-entry.",
    }


def parse_ppo_log(path: Path) -> dict[str, Any]:
    series: dict[str, list[float]] = defaultdict(list)
    anchor_mean: list[float] = []
    anchor_std: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "|" in line:
                match = METRIC_RE.search(line)
                if match:
                    key = match.group(1).split("/")[-1]
                    if key in PPO_KEYS:
                        series[key].append(finite_float(match.group(2)))
            if "ANCHOR_DEBUG" in line:
                match = ANCHOR_RE.search(line)
                if match:
                    anchor_mean.append(finite_float(match.group(1)))
                    anchor_std.append(finite_float(match.group(2)))
    report = {
        key: {**describe(values), "last": values[-1] if values else None}
        for key, values in sorted(series.items())
    }
    report["a0_mean"] = {**describe(anchor_mean), "last": anchor_mean[-1] if anchor_mean else None}
    report["a0_std"] = {**describe(anchor_std), "last": anchor_std[-1] if anchor_std else None}
    ev = np.asarray(series.get("explained_variance", []), dtype=np.float64)
    report["critic_negative_update_rate"] = float((ev < 0).mean()) if len(ev) else None
    report["methodology"] = "causal training telemetry emitted by PPO"
    return report


def load_action_diagnostics(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"available": False}
    frame = pd.read_csv(path)
    result: dict[str, Any] = {"available": True, "rows": int(len(frame))}
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            result[column] = describe(frame[column].dropna())
    return result


def bounded(value: float) -> float:
    return max(0.0, min(1.0, finite_float(value)))


def score_reports(
    entry: Mapping[str, Any],
    exit_data: Mapping[str, Any],
    risk: Mapping[str, Any],
    ppo: Mapping[str, Any],
) -> dict[str, Any]:
    """Transparent scale-free score; market reports A/B remain context only."""
    base = finite_float(entry.get("market_viable_base_rate"))
    precision = finite_float(entry.get("event_precision"))
    precision_skill = bounded(safe_div(precision - base, max(1.0 - base, 1e-12)) + 0.5)
    entry_components = {
        "precision_skill_vs_market": precision_skill,
        "opportunity_recall": bounded(entry.get("unique_opportunity_recall", 0.0)),
        "bad_opportunity_avoidance": bounded(entry.get("bad_opportunity_avoidance", 0.0)),
    }
    exit_components = {
        "profitable_opportunity_conversion": bounded(
            exit_data.get("profitable_opportunity_conversion", 0.0)
        ),
        "win_rate_relative_to_same_entry_oracle": bounded(
            safe_div(
                finite_float(exit_data.get("win_rate")),
                finite_float(exit_data.get("oracle_same_entry_win_rate")),
            )
        ),
        "non_maxduration_rate": bounded(1.0 - finite_float(exit_data.get("max_duration_rate"))),
    }
    drawdown = finite_float(risk.get("max_drawdown"), 1.0)
    ending = finite_float(risk.get("ending_equity"))
    initial = finite_float(risk.get("initial_capital"), 1.0)
    risk_components = {
        "capital_survival": bounded(safe_div(max(ending, 0.0), max(initial, 1e-12))),
        "drawdown_control": bounded(1.0 - drawdown),
        "positive_expectancy": 1.0 if finite_float(exit_data.get("expectancy_pnl_units")) > 0 else 0.0,
    }
    ev = finite_float(ppo.get("explained_variance", {}).get("mean"), -1.0)
    negative_rate = finite_float(ppo.get("critic_negative_update_rate"), 1.0)
    a0_std = finite_float(ppo.get("a0_mean", {}).get("std"), float("inf"))
    ppo_components = {
        "explained_variance": bounded((ev + 1.0) / 2.0),
        "nonnegative_critic_updates": bounded(1.0 - negative_rate),
        "finite_policy_direction": 1.0 if math.isfinite(a0_std) else 0.0,
    }
    domains = {
        "entry": entry_components,
        "exit": exit_components,
        "risk": risk_components,
        "ppo": ppo_components,
    }
    domain_scores = {
        name: 20.0 * float(np.mean(list(components.values())))
        for name, components in domains.items()
    }
    return {
        "methodology": (
            "scale-free transparent component mean; A/B describe difficulty and are not "
            "credited to the model"
        ),
        "components_0_to_1": domains,
        "domain_scores_0_to_20": domain_scores,
        "global_score_0_to_20": float(np.mean(list(domain_scores.values()))),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    scores = report["G_global_score"]
    lines = [
        "# ADAN Permanent Benchmark",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        f"Asset/timeframe: `{report['asset']}` / `{report['timeframe']}`",
        f"Horizon: `{report['horizon_steps']}` bars",
        "",
        "## Methodological contract",
        "",
        "- **Causal**: market descriptors and PPO telemetry available at decision/training time.",
        "- **Ex-post**: MFE/MAE labels used only to grade decisions after the fact.",
        "- **Physical ceiling**: same-entry and full-opportunity results are non-causal bounds.",
        "- Opportunity classes are empirical calibration quantiles, not asset-specific constants.",
        "- Configured fees and slippage are read without reduction or negotiation.",
        "",
        "## Global score",
        "",
        "| Domain | Score /20 |",
        "|---|---:|",
    ]
    for domain, value in scores["domain_scores_0_to_20"].items():
        lines.append(f"| {domain.title()} | {value:.3f} |")
    lines.extend(
        [
            f"| **Global** | **{scores['global_score_0_to_20']:.3f}** |",
            "",
            "## A — Dataset report",
            "",
            "```json",
            json.dumps(report["A_dataset_report"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## B — Opportunity report",
            "",
            "```json",
            json.dumps(report["B_opportunity_report"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## C — Entry report",
            "",
            "```json",
            json.dumps(report["C_entry_report"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## D — Exit report",
            "",
            "```json",
            json.dumps(report["D_exit_report"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## E — Risk report",
            "",
            "```json",
            json.dumps(report["E_risk_report"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## F — PPO report",
            "",
            "```json",
            json.dumps(report["F_ppo_report"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## G — Score details",
            "",
            "```json",
            json.dumps(scores, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/config.yaml")
    parser.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "data/processed/indicators"
    )
    parser.add_argument("--asset")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--arena", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--diagnostics-csv", type=Path)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    cost = resolve_cost_model(config)
    horizon = resolve_horizon(config, args.horizon)
    records = load_arena(args.arena)
    arena_asset = next(
        (
            str(record.get("meta", {}).get("asset"))
            for record in records
            if record.get("meta", {}).get("asset")
        ),
        None,
    )
    asset = args.asset or arena_asset
    paths = discover_split_paths(args.data_root, asset, args.timeframe)
    if args.calibration_split not in paths:
        raise KeyError(
            f"Calibration split {args.calibration_split!r} unavailable; found {sorted(paths)}"
        )
    frames = {split: load_split(path) for split, path in paths.items()}
    opportunities = {
        split: future_excursions(frame, horizon, cost.round_trip_execution_cost)
        for split, frame in frames.items()
    }
    thresholds = adaptive_thresholds(
        opportunities[args.calibration_split]["fee_aware_quality"]
    )
    aligned, alignment = align_arena_entries(
        records,
        frames[args.calibration_split],
        opportunities[args.calibration_split],
        args.timeframe,
    )
    if records and not aligned:
        raise RuntimeError("No Arena record aligned to calibration data")
    initial_capital = finite_float(
        config.get("portfolio", {}).get(
            "initial_balance", config.get("environment", {}).get("initial_balance", 20.5)
        ),
        20.5,
    )
    split_reports = {
        split: {
            "dataset": dataset_report(frames[split], opportunities[split], cost),
            "opportunity": opportunity_report(opportunities[split], thresholds),
        }
        for split in sorted(frames)
    }
    entry = entry_report(aligned, opportunities[args.calibration_split])
    exit_data = exit_report(aligned)
    risk = risk_report(aligned, initial_capital)
    ppo = parse_ppo_log(args.training_log)
    ppo["action_diagnostics"] = load_action_diagnostics(args.diagnostics_csv)
    score = score_reports(entry, exit_data, risk, ppo)
    report = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "asset": asset or next(iter(paths.values())).parent.name,
        "timeframe": args.timeframe,
        "horizon_steps": horizon,
        "calibration_split": args.calibration_split,
        "cost_model": cost.as_dict(),
        "sources": {
            "config": str(args.config),
            "splits": {split: str(path) for split, path in paths.items()},
            "arena": str(args.arena),
            "training_log": str(args.training_log),
            "diagnostics_csv": str(args.diagnostics_csv) if args.diagnostics_csv else None,
        },
        "anti_oracle_contract": {
            "actor_observation": "present-only; benchmark does not instantiate or modify actor",
            "future_usage": "ex-post labels and physical ceilings only",
            "arena_can_replace_action": False,
            "production_dependency": False,
        },
        "alignment": alignment,
        "A_dataset_report": {
            "methodology": "per-split market identity; model-independent",
            "splits": {split: data["dataset"] for split, data in split_reports.items()},
        },
        "B_opportunity_report": {
            "methodology": "adaptive fee-aware ex-post opportunity frontier",
            "calibration_split": args.calibration_split,
            "splits": {split: data["opportunity"] for split, data in split_reports.items()},
        },
        "C_entry_report": entry,
        "D_exit_report": exit_data,
        "E_risk_report": risk,
        "F_ppo_report": ppo,
        "G_global_score": score,
        "limitations": [
            "Long-only opportunity labels match the current ADAN routing contract.",
            "MFE/MAE and opportunity classes are non-causal labels, never live predictors.",
            "V19 Arena lacks entry notional and exit price, so MFE capture and sizing are not fabricated.",
            "Global score is diagnostic, not a profitability guarantee.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({
        "json": str(args.output_json),
        "markdown": str(args.output_markdown),
        "score": score["global_score_0_to_20"],
        "alignment": alignment,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
