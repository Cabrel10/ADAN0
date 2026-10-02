"""Unit tests for the strict A-E financial stability gate."""

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


def _gates(*verdicts: str) -> dict[str, dict[str, str]]:
    return {
        chr(ord("A") + index): {"verdict": verdict}
        for index, verdict in enumerate(verdicts)
    }


def test_overall_verdict_requires_every_gate_to_pass() -> None:
    assert MODULE._overall_verdict(_gates("PASS", "PASS", "PASS", "PASS", "PASS")) == "GO"
    assert MODULE._overall_verdict(_gates("PASS", "INCONCLUSIVE", "PASS")) == "INCONCLUSIVE"
    assert MODULE._overall_verdict(_gates("PASS", "FAIL", "INCONCLUSIVE")) == "NO_GO"


def test_gate_keeps_verdict_and_details() -> None:
    assert MODULE._gate("PASS", ratio=0.09) == {"verdict": "PASS", "ratio": 0.09}


def test_btc_contract_uses_complete_round_trip_fees() -> None:
    contract = MODULE._load_btc_financial_contract()
    assert contract["round_trip_fees"] == pytest.approx(0.004)
    assert contract["round_trip_fees"] >= 2.0 * contract["commission_per_side"]
    assert contract["mean_tp"] == pytest.approx(
        (contract["tp_low"] + contract["tp_high"]) / 2.0
    )


def test_launcher_runtime_does_not_disable_economic_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ("ADAN_FREE_SLTP", "ADAN_TP_LO", "ADAN_TP_HI", "ADAN_SL_HI"):
        monkeypatch.delenv(key, raising=False)
    contract = {
        "tp_low": 0.006,
        "tp_high": 0.06,
        "sl_high": 0.06,
    }
    runtime = MODULE._apply_btc_launcher_runtime(contract)
    assert runtime["ADAN_FREE_SLTP"] == "1"
    assert "ADAN_DISABLE_EV_FEE_GATE" not in runtime


def test_protocol_rejects_noncanonical_step_count() -> None:
    with pytest.raises(ValueError, match="exactly 500 steps"):
        MODULE.run_check(steps=499, split="train", seed=1)
