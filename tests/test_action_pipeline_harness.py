"""Unit contracts for the controlled action-pipeline harness."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.tests.action_pipeline_harness import (
    _scenario_verdict,
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
    action = controlled_action(1.0, sl_raw=0.25, tp_raw=0.75)
    np.testing.assert_allclose(action, [1.0, 0.0, -1.0, 0.25, 0.75])
    assert action.dtype == np.float32
    assert action.shape == (5,)


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
