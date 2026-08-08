"""V20 regression tests for routing, close authority, and trace telemetry."""

from __future__ import annotations

import inspect
import json
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from adan_trading_bot.environment.action_routing import (
    HOLD,
    SELL,
    resolve_agent_close_gate,
    resolve_ev_fee_gate,
    route_action_by_state,
)
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.performance.metrics import PerformanceMetrics
from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager
from adan_trading_bot.utils.reward_collector import RewardCollector
from scripts import train_parallel_agents as training
from scripts.analysis.monitor_v20_pbt import (
    persistent_alerts,
    pipeline_counts,
    robust_signal,
)
from scripts.analysis.validate_action_pipeline import validate_pipeline_lifecycle


def test_negative_action_while_flat_is_routing_hold_not_deadband() -> None:
    """Explains the V19 deterministic a0=-0.67 -> zero-trade observation."""
    assert abs(-0.67) > 0.05
    assert route_action_by_state(-0.67, in_position=False, threshold=0.05) == HOLD
    assert route_action_by_state(-0.67, in_position=True, threshold=0.05) == SELL


def test_ev_fee_gate_blocks_negative_ev_by_default() -> None:
    assert resolve_ev_fee_gate(
        p_hmm=1.0 / 3.0,
        p_min_required=0.38,
        disabled=False,
    ) == (True, "negative_ev_fee_gate")


def test_ev_fee_gate_flag_bypasses_as_explicit_advisory() -> None:
    assert resolve_ev_fee_gate(
        p_hmm=1.0 / 3.0,
        p_min_required=0.38,
        disabled=True,
    ) == (False, "disabled_advisory")


def test_ev_fee_gate_accepts_positive_ev_without_bypass() -> None:
    assert resolve_ev_fee_gate(
        p_hmm=0.60,
        p_min_required=0.38,
        disabled=False,
    ) == (False, "accepted")


@pytest.mark.parametrize(
    ("five_minute", "expected_close"),
    [
        (pd.DataFrame({"close": [100.0, 101.0]}), 101.0),
        (pd.DataFrame(), 201.0),
    ],
)
def test_hmm_market_data_selects_dataframe_without_boolean_evaluation(
    five_minute: pd.DataFrame,
    expected_close: float,
) -> None:
    env = MultiAssetChunkedEnv.__new__(MultiAssetChunkedEnv)
    env.step_in_chunk = 1
    env.current_data = {
        "BTCUSDT": {
            "5m": five_minute,
            "1h": pd.DataFrame({"close": [200.0, 201.0]}),
            "4h": pd.DataFrame({"close": [300.0, 301.0]}),
        }
    }

    market_data = env._get_current_market_data_for_hmm()

    assert market_data["close"] == pytest.approx(expected_close)
    assert market_data["prev_close"] == pytest.approx(expected_close - 1.0)


def _hmm_market_frame(first_close: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [first_close, first_close + 1.0],
            "volume": [10.0, 12.0],
        }
    )


def _hmm_market_snapshot(timeframes: dict[str, object]) -> dict[str, float]:
    env = object.__new__(MultiAssetChunkedEnv)
    env.step_in_chunk = 1
    env.current_data = {"BTCUSDT": timeframes}
    return env._get_current_market_data_for_hmm()


def test_hmm_market_data_uses_first_non_empty_timeframe_without_dataframe_truthiness() -> None:
    empty = pd.DataFrame()
    frame_5m = _hmm_market_frame(500.0)
    frame_1h = _hmm_market_frame(100.0)
    frame_4h = _hmm_market_frame(40.0)

    assert _hmm_market_snapshot(
        {"5m": frame_5m, "1h": frame_1h, "4h": frame_4h}
    )["close"] == pytest.approx(501.0)
    assert _hmm_market_snapshot(
        {"5m": empty, "1h": frame_1h, "4h": frame_4h}
    )["close"] == pytest.approx(101.0)
    assert _hmm_market_snapshot(
        {"5m": None, "1h": empty, "4h": frame_4h}
    )["close"] == pytest.approx(41.0)


def test_hmm_market_data_returns_safe_defaults_without_usable_timeframe() -> None:
    snapshot = _hmm_market_snapshot(
        {"5m": None, "1h": pd.DataFrame(), "4h": pd.DataFrame()}
    )

    assert snapshot["close"] == 0.0
    assert snapshot["prev_close"] == 0.0
    assert snapshot["rsi_norm"] == 0.5
    assert snapshot["volume_ratio_20"] == 1.0


def test_v20_exit_authority_cannot_override_budget_quota_or_gap() -> None:
    """A policy exit must not silently bypass the anti-churn hard gate."""
    assert resolve_agent_close_gate(
        exit_authority=True,
        budget_blocked=True,
        below_break_even=True,
    ) == (True, "decision_budget_or_quota")


def test_v20_exit_authority_can_cut_risk_after_hard_budget_gate() -> None:
    """Once structurally eligible, a loss cut may bypass the profit barrier."""
    assert resolve_agent_close_gate(
        exit_authority=True,
        budget_blocked=False,
        below_break_even=True,
    ) == (False, "exit_authority")


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


def test_portfolio_ledger_open_close_and_reset_uses_real_cash() -> None:
    config = {
        "initial_capital": 100.0,
        "assets": ["BTCUSDT"],
        "environment": {"commission": 0.001},
        "risk_management": {"min_trade_value": 1.0},
        "trading_rules": {"leverage": 1.0},
    }
    manager = PortfolioManager(config=config, worker_id=0, max_positions=1)
    opened = manager.open_position(
        asset="BTCUSDT",
        price=100.0,
        size=0.5,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        timestamp=datetime(2026, 1, 1, 0, 0),
        current_prices={"BTCUSDT": 100.0},
    )
    assert opened is not None
    assert manager.cash == pytest.approx(49.95)
    assert manager.get_portfolio_value() == pytest.approx(99.95)

    closed = manager.close_position(
        asset="BTCUSDT",
        price=110.0,
        timestamp=datetime(2026, 1, 1, 1, 0),
        current_prices={"BTCUSDT": 110.0},
        reason="AGENT_CLOSE",
    )
    assert closed is not None
    assert closed["pnl_gross"] == pytest.approx(5.0)
    assert closed["fees"] == pytest.approx(0.105)
    assert closed["pnl"] == pytest.approx(4.895)
    assert manager.cash == pytest.approx(104.895)
    assert manager.get_portfolio_value() == pytest.approx(104.895)
    assert manager.total_realized_pnl == pytest.approx(4.895)
    assert manager.realized_equity == pytest.approx(104.895)

    manager.reset()
    assert manager.cash == pytest.approx(100.0)
    assert manager.get_portfolio_value() == pytest.approx(100.0)
    assert manager.total_realized_pnl == 0.0
    assert manager.realized_equity == pytest.approx(100.0)


def _portfolio_for_lifecycle_tests(*, max_duration_steps: int = 144) -> PortfolioManager:
    return PortfolioManager(
        config={
            "initial_capital": 100.0,
            "assets": ["BTCUSDT", "ETHUSDT"],
            "environment": {"commission": 0.001},
            "risk_management": {"min_trade_value": 1.0},
            "trading_rules": {
                "duration_tracking": {
                    "5m": {"max_duration_steps": max_duration_steps}
                }
            },
        },
        worker_id=0,
        max_positions=1,
    )


def test_single_position_limit_rejects_second_asset() -> None:
    manager = _portfolio_for_lifecycle_tests()
    opened = manager.open_position(
        asset="BTCUSDT",
        price=100.0,
        size=0.2,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        timestamp=datetime(2026, 1, 1, 0, 0),
        current_prices={"BTCUSDT": 100.0, "ETHUSDT": 50.0},
    )
    rejected = manager.open_position(
        asset="ETHUSDT",
        price=50.0,
        size=0.2,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        timestamp=datetime(2026, 1, 1, 0, 1),
        current_prices={"BTCUSDT": 100.0, "ETHUSDT": 50.0},
    )

    assert opened is not None
    assert rejected is None
    assert len(manager._get_open_positions()) == 1


@pytest.mark.parametrize(
    ("reason", "current_step", "low", "high", "price", "max_duration"),
    [
        ("stop_loss", 1, 97.0, 100.0, 100.0, 144),
        ("take_profit", 1, 100.0, 105.0, 100.0, 144),
        ("MaxDuration", 2, 100.0, 100.0, 100.0, 1),
    ],
)
def test_automatic_close_is_exactly_once_and_preserves_position_identity(
    reason: str,
    current_step: int,
    low: float,
    high: float,
    price: float,
    max_duration: int,
) -> None:
    manager = _portfolio_for_lifecycle_tests(max_duration_steps=max_duration)
    opened = manager.open_position(
        asset="BTCUSDT",
        price=100.0,
        size=0.2,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        timestamp=datetime(2026, 1, 1, 0, 0),
        current_prices={"BTCUSDT": 100.0},
        current_step=0,
    )
    manager.register_market_timestamp(datetime(2026, 1, 1, 0, 5))

    pnl, receipts = manager.update_market_price(
        {"BTCUSDT": price},
        current_step=current_step,
        current_lows={"BTCUSDT": low},
        current_highs={"BTCUSDT": high},
    )
    cash_after_close = manager.cash
    second_pnl, second_receipts = manager.update_market_price(
        {"BTCUSDT": price},
        current_step=current_step + 1,
        current_lows={"BTCUSDT": low},
        current_highs={"BTCUSDT": high},
    )

    assert opened is not None
    assert len(receipts) == 1
    assert receipts[0]["close_reason"] == reason
    assert receipts[0]["position_id"] == opened["position_id"]
    assert pnl == pytest.approx(receipts[0]["pnl"])
    assert manager.positions["BTCUSDT"].is_open is False
    assert second_receipts == []
    assert second_pnl == 0.0
    assert manager.cash == pytest.approx(cash_after_close)


def test_financial_pnl_counters_survive_episode_reset() -> None:
    env = MultiAssetChunkedEnv.__new__(MultiAssetChunkedEnv)

    env._initialize_financial_telemetry()
    env._record_step_realized_pnl(2.0)
    assert env._step_realized_pnl == pytest.approx(2.0)
    assert env._episode_realized_pnl == pytest.approx(2.0)
    assert env._run_realized_pnl == pytest.approx(2.0)

    env._finalize_episode_financial_telemetry(reset_close_pnl=0.0)
    assert env._last_episode_realized_pnl == pytest.approx(2.0)
    assert env._episode_realized_pnl == pytest.approx(0.0)
    assert env._run_realized_pnl == pytest.approx(2.0)

    env._record_step_realized_pnl(-0.5)
    assert env._step_realized_pnl == pytest.approx(-0.5)
    assert env._episode_realized_pnl == pytest.approx(-0.5)
    assert env._run_realized_pnl == pytest.approx(1.5)


def test_financial_metrics_publish_distinct_step_episode_run_and_equity() -> None:
    env = MultiAssetChunkedEnv.__new__(MultiAssetChunkedEnv)
    env.portfolio_manager = SimpleNamespace(
        initial_capital=100.0,
        get_metrics=lambda: {
            "total_value": 104.0,
            "total_realized_pnl": 3.0,
            "cash": 54.0,
            "equity": 104.0,
            "realized_equity": 103.0,
            "unrealized_pnl_total": 1.0,
            "sharpe_ratio": 0.5,
            "max_drawdown": 2.0,
            "win_rate": 0.6,
            "total_trades": 4,
            "open_positions_count": 1,
        },
    )
    env._initialize_financial_telemetry()
    env._record_step_realized_pnl(2.0)
    env._finalize_episode_financial_telemetry(reset_close_pnl=0.25)
    env._record_step_realized_pnl(-0.5)

    metrics = env.get_portfolio_metrics_dict()[0]
    assert metrics["realized_pnl_step"] == pytest.approx(-0.5)
    assert metrics["realized_pnl_episode"] == pytest.approx(2.25)
    assert metrics["realized_pnl_episode_current"] == pytest.approx(-0.5)
    assert metrics["realized_pnl_cumulative"] == pytest.approx(1.75)
    assert metrics["cash"] == pytest.approx(54.0)
    assert metrics["equity"] == pytest.approx(104.0)
    assert metrics["realized_equity"] == pytest.approx(103.0)


def test_metrics_monitor_keeps_non_ambiguous_financial_series() -> None:
    metrics = {
        "total_value": 104.0,
        "initial_capital": 100.0,
        "total_realized_pnl": 0.0,
        "realized_pnl_step": -0.5,
        "realized_pnl_episode": 2.25,
        "realized_pnl_episode_current": -0.5,
        "realized_pnl_cumulative": 1.75,
        "cash": 54.0,
        "equity": 104.0,
        "realized_equity": 103.0,
    }
    fake_env = SimpleNamespace(
        env_method=lambda _name: [[metrics]],
        get_attr=lambda _name: [{}],
    )
    monitor = training.MetricsMonitor(
        {"portfolio": {"initial_balance": 100.0}},
        num_workers=1,
        log_interval=1,
    )
    monitor.model = SimpleNamespace(
        get_env=lambda: fake_env,
        logger=MagicMock(),
    )
    monitor.step_count = 1

    monitor._collect_worker_metrics()

    worker_metrics = monitor.worker_metrics[0]
    assert worker_metrics["realized_pnl_steps"][-1] == pytest.approx(-0.5)
    assert worker_metrics["realized_pnl_episodes"][-1] == pytest.approx(2.25)
    assert worker_metrics["realized_pnl_episode_currents"][-1] == pytest.approx(-0.5)
    assert worker_metrics["realized_pnl_cumulatives"][-1] == pytest.approx(1.75)
    assert worker_metrics["cash_values"][-1] == pytest.approx(54.0)
    assert worker_metrics["equity_values"][-1] == pytest.approx(104.0)
    assert worker_metrics["realized_equity_values"][-1] == pytest.approx(103.0)
    assert worker_metrics["portfolio_values"][-1] == pytest.approx(104.0)


def _reward_env(*, drawdown_percent: float, behavior_penalty: float = 0.0):
    env = MultiAssetChunkedEnv.__new__(MultiAssetChunkedEnv)
    env.config = {"reward_shaping": {"capital_tier_rewards": {}}}
    env.portfolio_manager = SimpleNamespace(
        initial_capital=100.0,
        initial_equity=100.0,
        positions={},
        get_portfolio_value=lambda: 100.0 - drawdown_percent,
        get_metrics=lambda: {
            "drawdown": drawdown_percent,
            "peak_equity": 100.0,
        },
    )
    env.current_step = 1
    env.worker_id = 0
    env.logger = MagicMock()
    env._step_closed_receipts = []
    env._step_invalid_penalty = behavior_penalty
    env._last_trade_executed = False
    return env


def test_drawdown_penalty_consumes_positive_percent_metric_as_ratio() -> None:
    env = _reward_env(drawdown_percent=10.0)
    reward = env._calculate_reward(np.zeros(5, dtype=np.float32), realized_pnl=0.0)
    assert env._last_reward_components["drawdown_pct"] == pytest.approx(10.0)
    assert env._last_reward_components["drawdown_penalty"] == pytest.approx(-0.5)
    assert reward == pytest.approx(-math.log1p(0.5))


def test_drawdown_delta_is_zero_when_flat_and_telescopes_on_recovery() -> None:
    env = _reward_env(drawdown_percent=10.0)
    drawdown = {"percent": 10.0}
    env.portfolio_manager.get_metrics = lambda: {
        "drawdown": drawdown["percent"],
        "peak_equity": 100.0,
    }

    contributions = []
    for drawdown_percent in (10.0, 10.0, 5.0):
        drawdown["percent"] = drawdown_percent
        env._calculate_reward(np.zeros(5, dtype=np.float32), realized_pnl=0.0)
        contributions.append(env._last_reward_components["drawdown_penalty"])

    assert contributions == pytest.approx([-0.5, 0.0, 0.375])
    assert sum(contributions) == pytest.approx(-50.0 * 0.05**2)
    assert env._reward_previous_drawdown_ratio == pytest.approx(0.05)


def test_gym_reset_restarts_drawdown_reward_potential() -> None:
    source = inspect.getsource(MultiAssetChunkedEnv.reset)
    episode_reset_at = source.index("self.episode_reward = 0.0")
    potential_reset_at = source.index("self._reward_previous_drawdown_ratio = 0.0")
    step_reset_at = source.index("self.step_in_chunk = 0")

    assert episode_reset_at < potential_reset_at < step_reset_at


def test_losing_agent_close_keeps_behavior_penalty_out_of_financial_pnl() -> None:
    env = _reward_env(drawdown_percent=0.0, behavior_penalty=-0.05)
    env._step_closed_receipts = [{"reason": "AGENT_CLOSE", "pnl": -1.0}]
    reward = env._calculate_reward(np.zeros(5, dtype=np.float32), realized_pnl=-1.0)
    components = env._last_reward_components
    assert components["pnl"] == pytest.approx(-1.0)
    assert components["behavior_penalty"] == pytest.approx(-0.05)
    assert components["closure_bonus"] == 0.0
    assert components["raw"] == pytest.approx(-0.55)
    assert reward == pytest.approx(-math.log1p(0.55))


@pytest.mark.parametrize(
    ("preferred", "iteration", "expected"),
    [
        (512, 8192, 512),
        (2048, 8192, 2048),
        (8192, 8192, 8192),
        (16384, 8192, 8192),
        (8192, 10_000, 5000),
    ],
)
def test_profile_rollout_is_bounded_to_exact_ray_iteration(
    preferred: int,
    iteration: int,
    expected: int,
) -> None:
    assert training._resolve_exact_rollout_steps(preferred, iteration) == expected
    assert iteration % expected == 0
    assert expected <= iteration


def test_pbt_restore_preserves_mutations_identity_and_exact_checkpoint(tmp_path) -> None:
    """Ray exploit restore must import weights without reverting target mutations."""
    worker = object.__new__(training.ADAN_PBT_Worker)
    worker.adan_config = {"capital_tiers": {"Micro": {"max_positions": 1}}}
    worker.worker_idx = 3
    worker.profile = "position"
    worker.learning_rate = 3e-4
    worker.ent_coef = 0.01
    worker.gamma = 0.99
    worker.sl_pct = 0.02
    worker.tp_pct = 0.04
    worker.interval_timesteps = 8192
    worker._max_iterations = 61
    worker._total_timesteps = 0

    def stale_model():
        optimizer = SimpleNamespace(param_groups=[{"lr": 9e-4}])
        policy = SimpleNamespace(optimizer=optimizer)
        return SimpleNamespace(
            learning_rate=9e-4,
            lr_schedule=lambda _: 9e-4,
            ent_coef=0.09,
            gamma=0.91,
            rollout_buffer=SimpleNamespace(gamma=0.91),
            policy=policy,
        )

    worker.model = stale_model()
    worker.vec_env = SimpleNamespace(gamma=0.91)
    source_root = tmp_path / "source_checkpoints"
    exact_checkpoint = source_root / "checkpoint_00000100"
    newer_checkpoint = source_root / "checkpoint_00000200"
    exact_checkpoint.mkdir(parents=True)
    newer_checkpoint.mkdir()
    worker.checkpoint_dir = str(source_root)
    ray_checkpoint = tmp_path / "ray_checkpoint"
    ray_checkpoint.mkdir()
    (ray_checkpoint / "ray_metadata.json").write_text(
        json.dumps(
            {
                "total_timesteps": 100,
                "learning_rate": 9e-4,
                "ent_coef": 0.09,
                "gamma": 0.91,
                "worker_idx": 0,
                "profile": "scalper",
                "sl_pct": 0.08,
                "tp_pct": 0.15,
                "checkpoint_dir": str(source_root),
            }
        )
    )

    loaded = []

    def fake_load_checkpoint(path: str, *, restore_hyperparameters: bool = True):
        loaded.append((path, restore_hyperparameters))
        worker.model = stale_model()
        worker.vec_env = SimpleNamespace(gamma=0.91)
        if restore_hyperparameters:
            worker.learning_rate = 9e-4
            worker.ent_coef = 0.09
            worker.gamma = 0.91

    worker.load_checkpoint = fake_load_checkpoint
    mutated = {
        "adan_config": worker.adan_config,
        "worker_config": {"worker_idx": 0, "profile": "scalper"},
        "learning_rate": 1e-4,
        "ent_coef": 0.02,
        "gamma": 0.975,
        "sl_pct": 0.03,
        "tp_pct": 0.06,
        "interval_timesteps": 8192,
        "_max_iterations": 61,
    }

    assert worker.reset_config(mutated) is True
    worker._restore(str(ray_checkpoint))

    assert loaded == [(str(exact_checkpoint), False)]
    assert (worker.worker_idx, worker.profile) == (3, "position")
    assert mutated["worker_config"] == {"worker_idx": 3, "profile": "position"}
    assert worker.learning_rate == pytest.approx(1e-4)
    assert worker.ent_coef == pytest.approx(0.02)
    assert worker.gamma == pytest.approx(0.975)
    assert worker.sl_pct == pytest.approx(0.03)
    assert worker.tp_pct == pytest.approx(0.06)
    assert worker.model.learning_rate == pytest.approx(1e-4)
    assert worker.model.ent_coef == pytest.approx(0.02)
    assert worker.model.gamma == pytest.approx(0.975)
    assert worker.model.rollout_buffer.gamma == pytest.approx(0.975)
    assert worker.vec_env.gamma == pytest.approx(0.975)
    assert worker.model.policy.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


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
    env.global_step = 142
    env.episode_count = 2
    env.current_chunk_idx = 7
    env.env_instance_id = "env-test"
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
        "event_sequence": 1,
        "env_instance_id": "env-test",
        "worker_id": 3,
        "global_step": 142,
        "episode_count": 2,
        "chunk_index": 7,
        "step": 42,
        "asset": "BTCUSDT",
        "stage": "routing_reject",
        "action_in": -0.67,
        "action_out": 0,
        "reason": "sell_while_flat",
        "position_id": None,
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


def test_lifecycle_validator_pairs_by_position_id_in_append_order(tmp_path) -> None:
    path = tmp_path / "pipeline.jsonl"
    events = [
        {"event_sequence": 1, "env_instance_id": "env-a", "worker_id": 0, "global_step": 10, "stage": "trade_executed", "lifecycle_event": "open", "position_id": "p1", "asset": "BTCUSDT", "entry_price": 100.0, "max_positions": 1},
        {"event_sequence": 2, "env_instance_id": "env-a", "worker_id": 0, "global_step": 11, "stage": "policy", "asset": "BTCUSDT"},
        {"event_sequence": 3, "env_instance_id": "env-a", "worker_id": 0, "global_step": 12, "stage": "trade_executed", "lifecycle_event": "close", "position_id": "p1", "asset": "BTCUSDT", "entry_price": 100.0, "exit_price": 101.0},
    ]
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    report = validate_pipeline_lifecycle(str(path))
    assert report["ok"] is True
    assert report["environments"]["0:env-a"]["max_open_positions"] == 1
    assert report["unchanged_price_closes"] == 0


def test_lifecycle_validator_rejects_duplicate_close_and_overlap(tmp_path) -> None:
    path = tmp_path / "pipeline.jsonl"
    events = [
        {"event_sequence": 1, "env_instance_id": "env-a", "worker_id": 0, "global_step": 10, "stage": "trade_executed", "lifecycle_event": "open", "position_id": "p1", "asset": "BTCUSDT", "entry_price": 100.0, "max_positions": 1},
        {"event_sequence": 2, "env_instance_id": "env-a", "worker_id": 0, "global_step": 11, "stage": "trade_executed", "lifecycle_event": "open", "position_id": "p2", "asset": "ETHUSDT", "entry_price": 50.0, "max_positions": 1},
        {"event_sequence": 3, "env_instance_id": "env-a", "worker_id": 0, "global_step": 12, "stage": "trade_executed", "lifecycle_event": "close", "position_id": "p1", "asset": "BTCUSDT", "exit_price": 101.0},
        {"event_sequence": 4, "env_instance_id": "env-a", "worker_id": 0, "global_step": 13, "stage": "trade_executed", "lifecycle_event": "close", "position_id": "p1", "asset": "BTCUSDT", "exit_price": 102.0},
    ]
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    report = validate_pipeline_lifecycle(str(path))
    types = {violation["type"] for violation in report["violations"]}
    assert report["ok"] is False
    assert "position_limit_exceeded" in types
    assert "duplicate_close" in types
    assert "unclosed_position" in types


def _receipt_backed_close_env(tmp_path, reason: str):
    env = MultiAssetChunkedEnv.__new__(MultiAssetChunkedEnv)
    env.portfolio_manager = _portfolio_for_lifecycle_tests()
    env.current_step = 7
    env.global_step = 7
    env.episode_count = 0
    env.current_chunk_idx = 0
    env.env_instance_id = f"env-{reason.lower()}"
    env.worker_id = 0
    env.action_pipeline_counts = {"trade_executed": 0}
    env._action_pipeline_trace_path = str(tmp_path / f"{reason}.jsonl")
    env._initialize_financial_telemetry()
    opened = env.portfolio_manager.open_position(
        asset="BTCUSDT",
        price=100.0,
        size=0.2,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        timestamp=datetime(2026, 1, 1, 0, 0),
        current_prices={"BTCUSDT": 100.0},
        current_step=1,
    )
    assert opened is not None
    env._trace_action_pipeline(
        "trade_executed",
        "BTCUSDT",
        1,
        1,
        "position_opened",
        position_id=opened["position_id"],
        lifecycle_event="open",
        entry_price=opened["price"],
        max_positions=1,
    )
    return env, opened


@pytest.mark.parametrize(
    "reason",
    [
        "CHUNK_END_FORCE_CLOSE",
        "DRAWDOWN_KILL_FORCE_CLOSE",
        "BANKRUPT_FORCE_CLOSE",
    ],
)
def test_force_close_paths_publish_one_receipt_backed_close(
    tmp_path,
    reason: str,
) -> None:
    env, opened = _receipt_backed_close_env(tmp_path, reason)

    receipt = env._close_position_with_lifecycle(
        asset="BTCUSDT",
        price=101.0,
        timestamp=datetime(2026, 1, 1, 0, 5),
        current_prices={"BTCUSDT": 101.0},
        reason=reason,
    )
    duplicate = env._close_position_with_lifecycle(
        asset="BTCUSDT",
        price=102.0,
        timestamp=datetime(2026, 1, 1, 0, 10),
        current_prices={"BTCUSDT": 102.0},
        reason=reason,
    )

    assert receipt is not None
    assert duplicate is None
    assert receipt["position_id"] == opened["position_id"]
    assert receipt["close_reason"] == reason
    assert env.portfolio_manager._get_open_positions() == []
    events = [
        json.loads(line)
        for line in Path(env._action_pipeline_trace_path).read_text().splitlines()
    ]
    close_events = [event for event in events if event.get("lifecycle_event") == "close"]
    assert len(close_events) == 1
    assert close_events[0]["position_id"] == opened["position_id"]
    assert close_events[0]["exit_price"] == pytest.approx(receipt["exit_price"])
    assert close_events[0]["pnl_net"] == pytest.approx(receipt["pnl_net"])
    assert env._run_opens == 1
    assert env._run_closes == 1
    assert env._run_completed_cycles == 1
    assert env._run_open_position_ids - env._run_closed_position_ids == set()


def test_run_lifecycle_counters_survive_episode_financial_reset(tmp_path) -> None:
    env, _ = _receipt_backed_close_env(tmp_path, "COUNTER_RESET")
    receipt = env._close_position_with_lifecycle(
        asset="BTCUSDT",
        price=101.0,
        timestamp=datetime(2026, 1, 1, 0, 5),
        current_prices={"BTCUSDT": 101.0},
        reason="EPISODE_END",
    )
    assert receipt is not None

    env._finalize_episode_financial_telemetry(
        reset_close_pnl=float(receipt["pnl_net"])
    )

    assert env._run_opens == 1
    assert env._run_closes == 1
    assert env._run_completed_cycles == 1
    assert env._run_open_position_ids == {str(receipt["position_id"])}
    assert env._run_closed_position_ids == {str(receipt["position_id"])}
    metrics = env.get_portfolio_metrics_dict()[0]
    assert metrics["run_opens"] == 1
    assert metrics["run_closes"] == 1
    assert metrics["run_completed_cycles"] == 1
    assert metrics["run_open_positions"] == 0


def test_global_step_advances_before_terminal_lifecycle_checks() -> None:
    source = inspect.getsource(MultiAssetChunkedEnv.step)
    increment_at = source.index("self.global_step += 1")
    drawdown_at = source.index("if self._check_drawdown_termination():")
    execute_at = source.index("self._execute_trades(")

    assert source.count("self.global_step += 1") == 1
    assert increment_at < drawdown_at < execute_at


def test_sandbox_source_finalizes_before_save_and_summary() -> None:
    source = inspect.getsource(training.sandbox_train)
    learn_at = source.index("model.learn(")
    finalize_at = source.index("env.finalize_open_positions(reason=\"TRAINING_END\")")
    telemetry_at = source.index("env._finalize_episode_financial_telemetry(")
    save_at = source.index("model.save(ckpt_path)")
    summary_at = source.index("info = env.get_info()")

    assert learn_at < finalize_at < telemetry_at < save_at < summary_at
    assert 'info.get("run_completed_cycles"' in source
    assert 'info.get("run_open_positions"' in source
    assert '"terminal_cash"' in source
    assert '"terminal_equity"' in source
    assert '"terminal_realized_pnl"' in source
    assert 'financial_metrics.get("run_opens"' in source
    assert 'financial_metrics.get("run_closes"' in source


def test_diagnostic_numeric_stats_tracks_nonfinite_values() -> None:
    stats = training.DiagnosticCollapseCallback._numeric_stats(
        [1.0, 2.0, np.nan, np.inf]
    )

    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(2.0)
    assert stats["mean"] == pytest.approx(1.5)
    assert stats["p50"] == pytest.approx(1.5)
    assert stats["nonfinite_frac"] == pytest.approx(0.5)


def test_diagnostic_rollout_health_reports_critic_and_observation_clipping() -> None:
    callback = object.__new__(training.DiagnosticCollapseCallback)
    callback.model = SimpleNamespace(
        rollout_buffer=SimpleNamespace(
            rewards=np.array([[1.0], [2.0], [3.0], [4.0]]),
            returns=np.array([[1.0], [2.0], [3.0], [4.0]]),
            advantages=np.array([[0.0], [0.0], [1.0], [0.0]]),
            values=np.array([[1.0], [2.0], [2.0], [4.0]]),
            observations={
                "market": np.array([0.0, 10.0, -10.0, np.nan]),
                "portfolio": np.array([0.0, 0.5, 1.0, 2.0]),
            },
            episode_starts=np.array([[1.0], [0.0], [0.0], [1.0]]),
        )
    )

    health = callback._rollout_health()
    observations = json.loads(health["observation_stats"])

    assert health["critic_explained_variance"] == pytest.approx(0.85)
    assert health["episode_starts"] == 2
    assert json.loads(health["reward_stats"])["mean"] == pytest.approx(2.5)
    assert observations["market"]["abs_ge_10_frac"] == pytest.approx(2.0 / 3.0)
    assert observations["market"]["nonfinite_frac"] == pytest.approx(0.25)
    assert observations["portfolio"]["abs_ge_10_frac"] == 0.0


def test_diagnostic_flush_waits_for_completed_rollout() -> None:
    callback = object.__new__(training.DiagnosticCollapseCallback)
    callback.num_timesteps = 512
    callback._next_flush = 512
    callback.log_every = 512
    callback._flush_pending = True
    flushes = []
    callback._flush = lambda: flushes.append(callback.num_timesteps)

    callback._on_rollout_end()

    assert flushes == [512]
    assert callback._flush_pending is False
    assert callback._next_flush == 1024


def test_critic_breaker_stops_only_when_explicitly_armed() -> None:
    callback = object.__new__(training.DiagnosticCollapseCallback)
    callback.locals = {}
    callback.num_timesteps = 512
    callback._collapse_tripped = False
    callback._breaker_enabled = False
    callback._critic_breaker_reason = "non-finite value"
    callback._critic_breaker_enabled = False

    assert callback._on_step() is True

    callback._critic_breaker_enabled = True
    assert callback._on_step() is False


def test_finalize_open_positions_is_idempotent_and_traces_one_close(tmp_path) -> None:
    env = MultiAssetChunkedEnv.__new__(MultiAssetChunkedEnv)
    env.portfolio_manager = _portfolio_for_lifecycle_tests()
    env.current_step = 7
    env.global_step = 7
    env.episode_count = 0
    env.current_chunk_idx = 0
    env.env_instance_id = "env-finalize"
    env.worker_id = 0
    env.action_pipeline_counts = {"trade_executed": 0}
    env._action_pipeline_trace_path = str(tmp_path / "pipeline.jsonl")
    env._get_current_prices = lambda: {"BTCUSDT": 101.0}
    env._get_current_timestamp = lambda: datetime(2026, 1, 1, 0, 35)

    opened = env.portfolio_manager.open_position(
        asset="BTCUSDT",
        price=100.0,
        size=0.2,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        timestamp=datetime(2026, 1, 1, 0, 0),
        current_prices={"BTCUSDT": 100.0},
        current_step=1,
    )
    assert opened is not None

    first = env.finalize_open_positions(reason="TRAINING_END")
    cash_after_first = env.portfolio_manager.cash
    second = env.finalize_open_positions(reason="TRAINING_END")

    assert len(first) == 1
    assert second == []
    assert env.portfolio_manager.cash == pytest.approx(cash_after_first)
    assert env.portfolio_manager.positions["BTCUSDT"].is_open is False
    events = [json.loads(line) for line in Path(env._action_pipeline_trace_path).read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["lifecycle_event"] == "close"
    assert events[0]["position_id"] == opened["position_id"]
    assert events[0]["reason"] == "TRAINING_END"


def test_finalize_open_positions_survives_missing_timestamp_and_prices(tmp_path) -> None:
    env = MultiAssetChunkedEnv.__new__(MultiAssetChunkedEnv)
    env.portfolio_manager = _portfolio_for_lifecycle_tests()
    env.current_step = 7
    env.global_step = 7
    env.episode_count = 0
    env.current_chunk_idx = 0
    env.env_instance_id = "env-finalize-fallback"
    env.worker_id = 0
    env.action_pipeline_counts = {"trade_executed": 0}
    env._action_pipeline_trace_path = str(tmp_path / "pipeline.jsonl")

    opened = env.portfolio_manager.open_position(
        asset="BTCUSDT",
        price=100.0,
        size=0.2,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        timestamp=datetime(2026, 1, 1, 0, 0),
        current_prices={"BTCUSDT": 100.0},
        current_step=1,
    )
    assert opened is not None
    env._get_current_prices = MagicMock(side_effect=RuntimeError("prices unavailable"))
    env._get_current_timestamp = MagicMock(side_effect=RuntimeError("timestamp unavailable"))

    receipts = env.finalize_open_positions(reason="EPISODE_END")

    assert len(receipts) == 1
    assert receipts[0]["position_id"] == opened["position_id"]
    assert receipts[0]["close_reason"] == "EPISODE_END"
    assert env.portfolio_manager._get_open_positions() == []
    events = [
        json.loads(line)
        for line in Path(env._action_pipeline_trace_path).read_text().splitlines()
    ]
    assert len(events) == 1
    assert events[0]["lifecycle_event"] == "close"


def test_reset_refuses_to_hide_position_when_all_close_prices_are_invalid(tmp_path) -> None:
    env = MultiAssetChunkedEnv.__new__(MultiAssetChunkedEnv)
    env.portfolio_manager = _portfolio_for_lifecycle_tests()
    env._initialize_financial_telemetry()
    env._financial_episode_active = True
    env.current_step = 7
    env.global_step = 7
    env.episode_count = 0
    env.current_chunk_idx = 0
    env.env_instance_id = "env-reset-fail-closed"
    env.worker_id = 0
    env.action_pipeline_counts = {"trade_executed": 0}
    env._action_pipeline_trace_path = str(tmp_path / "pipeline.jsonl")

    opened = env.portfolio_manager.open_position(
        asset="BTCUSDT",
        price=100.0,
        size=0.2,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        timestamp=datetime(2026, 1, 1, 0, 0),
        current_prices={"BTCUSDT": 100.0},
        current_step=1,
    )
    assert opened is not None
    position = env.portfolio_manager.positions["BTCUSDT"]
    position.current_price = float("nan")
    position.entry_price = float("nan")
    env._get_current_prices = MagicMock(side_effect=RuntimeError("prices unavailable"))
    env._get_current_timestamp = MagicMock(side_effect=RuntimeError("timestamp unavailable"))

    with pytest.raises(RuntimeError, match="Refusing episode reset with unclosed positions"):
        env.reset()

    assert len(env.portfolio_manager._get_open_positions()) == 1
    assert not Path(env._action_pipeline_trace_path).exists()


def test_entry_market_telemetry_identifies_fill_and_decision_rows() -> None:
    env = MultiAssetChunkedEnv.__new__(MultiAssetChunkedEnv)
    index = pd.date_range("2026-01-01", periods=4, freq="5min")
    indicators = {
        "ema_20_ratio": 1.0,
        "macdh_12_26_9": 0.1,
        "rsi_14": 55.0,
        "adx_14": 20.0,
        "di_delta": 2.0,
        "atr_pct": 0.01,
        "bb_percent_b_20_2": 0.5,
        "obv_slope": 0.2,
        "volume_ratio_20": 1.1,
        "volatility_ratio_14_50": 0.9,
        "fib_ratio": 0.4,
        "price_action": 0.3,
        "vwap_ratio": 1.0,
        "market_structure": 1.0,
        "bb_width_20_2": 0.02,
        "log_return": 0.001,
    }
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            **{name: [value] * 4 for name, value in indicators.items()},
        },
        index=index,
    )
    env.current_data = {"BTCUSDT": {"5m": frame}}
    env.execution_timeframe = "5m"
    env.step_in_chunk = 1
    env.current_chunk_idx = 7

    telemetry = env._build_entry_market_telemetry("BTCUSDT", 102.0)

    assert telemetry["entry_telemetry_available"] is True
    assert telemetry["decision_row_id"] == "7:5m:1"
    assert telemetry["entry_row_id"] == "7:5m:2"
    assert telemetry["decision_market_timestamp"] == index[1].isoformat()
    assert telemetry["entry_market_timestamp"] == index[2].isoformat()
    assert telemetry["entry_price_source"] == "open[t+1]"
    assert telemetry["feature_snapshot_row_role"] == "decision_close_t"
    assert len(telemetry["entry_feature_snapshot"]) == 16
    assert telemetry["entry_atr_raw"] == pytest.approx(1.02)


def test_entry_market_telemetry_marks_last_bar_fill_fallback() -> None:
    env = MultiAssetChunkedEnv.__new__(MultiAssetChunkedEnv)
    frame = pd.DataFrame(
        {"open": [100.0], "close": [100.5], "atr_pct": [0.02]},
        index=pd.DatetimeIndex(["2026-01-01"]),
    )
    env.current_data = {"BTCUSDT": {"5m": frame}}
    env.execution_timeframe = "5m"
    env.step_in_chunk = 0
    env.current_chunk_idx = 1

    telemetry = env._build_entry_market_telemetry("BTCUSDT", 100.5)

    assert telemetry["entry_row_id"] == "1:5m:0"
    assert telemetry["entry_price_source"] == "close[t]_FALLBACK"
    assert telemetry["entry_atr_raw"] == pytest.approx(2.01)


def test_ray_cleanup_finalizes_before_closing_vector_env() -> None:
    worker = object.__new__(training.ADAN_PBT_Worker)
    worker.worker_idx = 2
    worker.vec_env = MagicMock()

    worker.cleanup()

    assert worker.vec_env.method_calls[0].args == (
        "finalize_open_positions",
    )
    assert worker.vec_env.method_calls[0].kwargs == {"reason": "TRAINING_END"}
    assert str(worker.vec_env.method_calls[1]) == "call.close()"


def test_reward_collector_path_survives_cwd_change(tmp_path, monkeypatch) -> None:
    original_cwd = tmp_path / "original"
    other_cwd = tmp_path / "other"
    original_cwd.mkdir()
    other_cwd.mkdir()
    monkeypatch.chdir(original_cwd)
    collector = RewardCollector("relative/rewards")
    expected_dir = original_cwd / "relative" / "rewards"

    monkeypatch.chdir(other_cwd)
    collector.log_episode_summary("0", 1, {"reward": 1.0})

    assert Path(collector.log_dir) == expected_dir
    files = list(expected_dir.glob("worker_0_rewards_*.jsonl"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["type"] == "episode_summary"


def test_performance_metrics_path_survives_cwd_and_pickle_restore(tmp_path, monkeypatch) -> None:
    import pickle

    original_cwd = tmp_path / "original"
    other_cwd = tmp_path / "other"
    original_cwd.mkdir()
    other_cwd.mkdir()
    monkeypatch.chdir(original_cwd)
    metrics = PerformanceMetrics(worker_id=0, metrics_dir="relative/metrics")
    expected_file = metrics.metrics_file
    restored = pickle.loads(pickle.dumps(metrics))

    monkeypatch.chdir(other_cwd)
    restored._log_metrics({"event": "after_cwd_change"})

    assert expected_file.is_absolute()
    events = [json.loads(line) for line in expected_file.read_text().splitlines()]
    assert events[-1]["event"] == "after_cwd_change"


def test_max_duration_alert_is_non_lethal_performance_warning() -> None:
    source = Path("scripts/analysis/monitor_v20_pbt.py").read_text()
    assert 'f"WARN arena: MaxDuration=' in source
    assert 'f"CRITICAL arena: MaxDuration=' not in source
