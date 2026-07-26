"""V20 regression tests for routing, close authority, and trace telemetry."""

from __future__ import annotations

import json

import pytest

from adan_trading_bot.environment.action_routing import (
    HOLD,
    SELL,
    resolve_agent_close_gate,
    route_action_by_state,
)
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager
from scripts.analysis.monitor_v20_pbt import (
    persistent_alerts,
    pipeline_counts,
    robust_signal,
)


def test_negative_action_while_flat_is_routing_hold_not_deadband() -> None:
    """Explains the V19 deterministic a0=-0.67 -> zero-trade observation."""
    assert abs(-0.67) > 0.05
    assert route_action_by_state(-0.67, in_position=False, threshold=0.05) == HOLD
    assert route_action_by_state(-0.67, in_position=True, threshold=0.05) == SELL


def test_v20_exit_authority_overrides_budget_and_profit_barrier() -> None:
    blocked, reason = resolve_agent_close_gate(
        exit_authority=True,
        budget_blocked=True,
        below_break_even=True,
    )
    assert blocked is False
    assert reason == "exit_authority"


def test_legacy_close_gates_remain_available_when_authority_disabled() -> None:
    assert resolve_agent_close_gate(
        exit_authority=False,
        budget_blocked=True,
        below_break_even=False,
    ) == (True, "decision_budget_or_quota")
    assert resolve_agent_close_gate(
        exit_authority=False,
        budget_blocked=False,
        below_break_even=True,
    ) == (True, "below_break_even_barrier")


def _duration_manager(profile: str) -> PortfolioManager:
    manager = PortfolioManager.__new__(PortfolioManager)
    manager.profile = profile
    manager.config = {
        "trading_rules": {
            "duration_tracking": {
                "5m": {"max_duration_steps": 50, "optimal_duration": 25},
                "1h": {"max_duration_steps": 20},
                "4h": {"max_duration_steps": 15},
                "profile_overrides": {
                    "scalper": {"5m": {"max_duration_steps": 20}},
                },
            }
        }
    }
    return manager


def test_single_duration_authority_preserves_effective_v19_limits() -> None:
    expected_5m = {
        "scalper": 20,
        "intraday": 50,
        "swing": 50,
        "position": 50,
    }
    for profile, expected in expected_5m.items():
        manager = _duration_manager(profile)
        assert manager._resolve_max_duration_steps("5m") == expected
        assert manager._resolve_max_duration_steps("1h") == 20
        assert manager._resolve_max_duration_steps("4h") == 15


def test_duration_resolution_does_not_infer_profile_from_worker_id() -> None:
    manager = _duration_manager("intraday")
    manager.worker_id = 0
    assert manager._resolve_max_duration_steps("5m") == 50


def test_pipeline_trace_writes_exact_transition(tmp_path) -> None:
    env = MultiAssetChunkedEnv.__new__(MultiAssetChunkedEnv)
    env.current_step = 42
    env.worker_id = 3
    env.action_pipeline_counts = {
        "policy": 0,
        "deadband_reject": 0,
        "routing_reject": 0,
        "budget_reject": 0,
        "barrier_reject": 0,
        "portfolio_reject": 0,
        "trade_executed": 0,
    }
    env._action_pipeline_trace_path = str(tmp_path / "pipeline.jsonl")

    env._trace_action_pipeline(
        "routing_reject",
        "BTCUSDT",
        -0.67,
        0,
        "sell_while_flat",
        threshold=0.05,
    )

    assert env.action_pipeline_counts["routing_reject"] == 1
    event = json.loads((tmp_path / "pipeline.jsonl").read_text().strip())
    assert event == {
        "step": 42,
        "worker_id": 3,
        "asset": "BTCUSDT",
        "stage": "routing_reject",
        "action_in": -0.67,
        "action_out": 0,
        "reason": "sell_while_flat",
        "threshold": 0.05,
    }


def test_monitor_robust_statistics_use_standard_even_sample_median() -> None:
    signal = robust_signal([1.0, 2.0, 10.0, 20.0], [0.0, 10_000.0, 20_000.0, 30_000.0])
    assert signal["median"] == 6.0
    assert signal["mad"] == 4.5
    assert signal["robust_scale"] == pytest.approx(6.6717)
    assert signal["slope_per_10k_steps"] == pytest.approx(6.5)


def test_monitor_pipeline_window_uses_policy_decisions_per_worker(tmp_path) -> None:
    path = tmp_path / "v20_0.jsonl"
    events = [
        {"step": 1, "worker_id": 0, "asset": "BTCUSDT", "stage": "policy", "action_in": 1, "action_out": 1, "reason": "policy_output"},
        {"step": 1, "worker_id": 0, "asset": "BTCUSDT", "stage": "trade_executed", "action_in": 1, "action_out": 1, "reason": "opened"},
        {"step": 2, "worker_id": 0, "asset": "BTCUSDT", "stage": "policy", "action_in": 0, "action_out": 0, "reason": "policy_output"},
        {"step": 2, "worker_id": 0, "asset": "BTCUSDT", "stage": "deadband_reject", "action_in": 0, "action_out": 0, "reason": "deadband"},
        {"step": 1, "worker_id": 1, "asset": "BTCUSDT", "stage": "policy", "action_in": 0, "action_out": 0, "reason": "policy_output"},
    ]
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    report = pipeline_counts(str(tmp_path / "v20_*.jsonl"), decision_window=2)
    assert report["by_worker"]["0"]["recent_window_decisions"] == 2
    assert report["by_worker"]["0"]["recent_rates_per_policy"]["trade_executed"] == 0.5
    assert report["by_worker"]["1"]["recent_rates_per_policy"].get("trade_executed") is None


def test_monitor_entropy_and_std_are_interpreted_with_sb3_sign_convention() -> None:
    rows = []
    for iteration in range(1, 5):
        rows.append({
            "training_iteration": str(iteration),
            "timesteps_total": str(iteration * 10_000),
            "mean_reward": "0.1",
            "explained_variance": "0.2",
            "approx_kl": "0.01",
            "clip_fraction": "0.1",
            "entropy_loss": str(-2.0 + iteration * 0.1),
            "policy_gradient_loss": "-0.01",
            "value_loss": "0.5",
            "std": "0.36",
            "config/worker_config/profile": "scalper",
            "config/worker_config/worker_idx": "0",
        })
    metrics, alerts = persistent_alerts("trial_scalper", rows, history_size=20)
    assert metrics["groups"]["exploration"] == "healthy"
    assert not [alert for alert in alerts if "entropy" in alert or "exploration" in alert]
