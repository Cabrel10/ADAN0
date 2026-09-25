"""Regression tests for the permanent asset-agnostic ADAN benchmark."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/analysis/adan_benchmark.py"
SPEC = importlib.util.spec_from_file_location("adan_benchmark", MODULE_PATH)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def market_frame(scale: float) -> pd.DataFrame:
    base = np.asarray([100, 101, 99, 102, 100, 103, 101, 104], dtype=float) * scale
    return pd.DataFrame(
        {
            "open": base,
            "high": base * np.asarray([1.01, 1.02, 1.01, 1.03, 1.01, 1.02, 1.01, 1.02]),
            "low": base * np.asarray([0.99, 0.98, 0.99, 0.97, 0.99, 0.98, 0.99, 0.98]),
            "close": base,
        },
        index=pd.date_range("2025-01-01", periods=len(base), freq="5min"),
    )


def test_future_excursions_are_invariant_to_nominal_asset_price() -> None:
    penny = benchmark.future_excursions(market_frame(0.00001), horizon=3, cost=0.0044)
    expensive = benchmark.future_excursions(market_frame(10000), horizon=3, cost=0.0044)
    for column in ("mfe", "mae", "net_mfe", "rr", "fee_aware_quality"):
        np.testing.assert_allclose(penny[column], expensive[column], rtol=1e-12, atol=1e-12)


def test_opportunity_thresholds_are_empirical_not_asset_constants() -> None:
    quality = np.arange(100, dtype=float)
    thresholds = benchmark.adaptive_thresholds(quality)
    assert thresholds == list(np.quantile(quality, [0.2, 0.4, 0.6, 0.8]))
    classes = benchmark.classify_quality(quality, thresholds)
    counts = {label: int((classes == label).sum()) for label in benchmark.LABELS}
    assert counts == {label: 20 for label in benchmark.LABELS}


def test_cost_model_reads_full_configured_fees_and_slippage() -> None:
    cost = benchmark.resolve_cost_model(
        {"trading_rules": {"commission_pct": 0.002, "slippage_pct": 0.0002}}
    )
    assert cost.commission_per_side == 0.002
    assert cost.slippage_per_side == 0.0002
    assert cost.round_trip_execution_cost == 0.0044


def test_risk_report_resets_equity_when_global_step_resets() -> None:
    aligned = [
        {"meta": {"global_step": 10, "pnl_net": -1.0}},
        {"meta": {"global_step": 20, "pnl_net": -1.0}},
        {"meta": {"global_step": 5, "pnl_net": -1.0}},
        {"meta": {"global_step": 15, "pnl_net": -1.0}},
    ]
    report = benchmark.risk_report(aligned, initial_capital=10.0)
    assert report["episodes"] == 2
    assert report["mean_episode_ending_equity"] == 8.0
    assert report["minimum_episode_equity"] == 8.0
    assert report["max_drawdown"] == 0.2
