"""Unit tests for the 500-step financial stability decision logic."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "financial_stability_check",
    ROOT / "scripts" / "validation" / "financial_stability_check.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_ratio_uses_absolute_capacity_and_positive_trade_mean():
    capacity, trades, ratio = MODULE.calculate_ratio(
        [-0.03, 0.01, -0.02], [0.4, 0.2]
    )
    assert capacity == pytest.approx(0.02)
    assert trades == pytest.approx(0.3)
    assert ratio == pytest.approx(1.0 / 15.0)


def test_no_positive_trade_is_inconclusive():
    capacity, trades, ratio = MODULE.calculate_ratio([-0.03], [])
    assert capacity == pytest.approx(0.03)
    assert trades is None
    assert ratio is None
    assert MODULE.classify_ratio(ratio) == "INCONCLUSIVE_NO_POSITIVE_TRADES"


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (0.099, "PASS_LAUNCH_500K"),
        (0.1, "INTERMEDIATE_REDUCE_AND_REVALIDATE"),
        (0.3, "INTERMEDIATE_REDUCE_AND_REVALIDATE"),
        (0.301, "FAIL_REDUCE_CAPACITY_COEFFICIENT"),
    ],
)
def test_requested_thresholds(ratio, expected):
    assert MODULE.classify_ratio(ratio) == expected
