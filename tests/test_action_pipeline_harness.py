"""Unit contracts for the controlled action-pipeline harness."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.tests.action_pipeline_harness import (
    _gate_category,
    _scenario_verdict,
    _sizing_summary,
    controlled_action,
    scripted_direction,
)


@pytest.mark.parametrize(
    ("scenario", "in_position", "has_opened", "steps_held", "pnl", "expected"),
    [
        ("sell_while_flat", False, False, 0, 0.0, -1.0),
        ("constant_buy", False, False, 0, 0.0, 1.0),
        ("constant_buy", True, True, 8, -0.01, 1.0),
        ("buy_then_hold", False, False, 0, 0.0, 1.0),
        ("buy_then_hold", True, True, 4, 0.01, 0.0),
        ("buy_then_hold", False, True, 0, 0.0, 0.0),
        ("buy_then_sell", False, False, 0, 0.0, 1.0),
        ("buy_then_sell", True, True, 1, -0.01, -1.0),
        ("loss_cut", False, False, 0, 0.0, 1.0),
        ("loss_cut", True, True, 5, -0.01, 0.0),
        ("loss_cut", True, True, 6, 0.01, 0.0),
        ("loss_cut", True, True, 6, -0.01, -1.0),
    ],
)
def test_scripted_direction_contract(
    scenario: str,
    in_position: bool,
    has_opened: bool,
    steps_held: int,
    pnl: float,
    expected: float,
) -> None:
    assert scripted_direction(
        scenario,
        in_position=in_position,
        has_opened=has_opened,
        steps_held=steps_held,
        unrealized_pnl_pct=pnl,
        min_hold_steps=6,
    ) == expected


def test_unknown_scenario_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown scenario"):
        scripted_direction(
            "not-a-scenario",
            in_position=False,
            has_opened=False,
            steps_held=0,
            unrealized_pnl_pct=0.0,
            min_hold_steps=6,
        )


def test_controlled_action_uses_real_box5_layout() -> None:
    action = controlled_action(1.0, size_raw=-0.75, sl_raw=0.25, tp_raw=0.75)
    np.testing.assert_allclose(action, [1.0, -0.75, -1.0, 0.25, 0.75])
    assert action.dtype == np.float32
    assert action.shape == (5,)


@pytest.mark.parametrize(
    ("stage", "reason", "expected"),
    [
        ("barrier_reject", "negative_ev_fee_gate", "fee_gate"),
        ("budget_reject", "cash_insufficient", "cash_budget"),
        ("routing_reject", "min_notional", "min_notional"),
        ("portfolio_reject", "pm_rejected", "portfolio"),
        ("barrier_reject", "risk_barrier", "barrier"),
        ("routing_reject", "daily_limit", "daily_limit"),
        ("routing_reject", "sell_while_flat", "other"),
    ],
)
def test_gate_categories_are_mutually_exclusive(stage: str, reason: str, expected: str) -> None:
    assert _gate_category(stage, reason) == expected


def test_sizing_summary_separates_policy_cash_and_execution() -> None:
    summary = _sizing_summary(
        [
            {
                "size_raw": 0.8,
                "normalized_size": 0.9,
                "target_exposure_pct": 0.85,
                "requested_notional_usd": 85.0,
                "notional_usd": 20.0,
            }
        ],
        [{"notional_usd": 5.0}],
    )
    assert summary["size_raw"]["mean"] == pytest.approx(0.8)
    assert summary["requested_notional_usd"]["mean"] == pytest.approx(85.0)
    assert summary["cash_capped_notional_usd"]["mean"] == pytest.approx(20.0)
    assert summary["executed_notional_usd"]["mean"] == pytest.approx(5.0)


def test_verdict_reports_execution_gate_instead_of_policy_event() -> None:
    result = {
        "scenario": "buy_then_sell",
        "pipeline": {
            "stages": {"policy": 30, "barrier_reject": 30},
            "stage_reasons": {
                "policy:policy_output": 30,
                "barrier_reject:negative_ev_fee_gate": 30,
            },
        },
        "closes": {"all": {"count": 0, "win_rate": 0.0}},
    }

    assert _scenario_verdict(result) == (
        "BLOCKED_BEFORE_OPEN:barrier_reject:negative_ev_fee_gate"
    )
