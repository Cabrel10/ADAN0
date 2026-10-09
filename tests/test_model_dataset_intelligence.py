"""Regression tests for the supervised Future Arena bulletin."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "model_dataset_intelligence",
    ROOT / "scripts/analysis/model_dataset_intelligence.py",
)
assert SPEC and SPEC.loader
bulletin = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bulletin)


def _sample(state: list[float], *, tp: float, pnl: float) -> dict:
    return {
        "state": state,
        "break_even": 0.004,
        "tp": tp,
        "sl": 0.008,
        "duration": 20.0,
        "confidence": float(pnl > 0),
        "meta": {
            "pnl_net": pnl,
            "global_step": 10,
            "reason": "MaxDuration",
        },
    }


def test_arena_teacher_loader_is_explicit_about_censored_schema(tmp_path: Path) -> None:
    base = [0.001, 0.5, 0.2, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0]
    path = tmp_path / "arena.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(_sample(base, tp=0.006, pnl=-0.1)),
                json.dumps(_sample([0.002, *base[1:]], tp=0.012, pnl=0.1)),
                "{not-json}",
            ]
        ),
        encoding="utf-8",
    )

    frame, audit = bulletin.load_arena_teacher_samples(path, tp_min=0.005)

    assert len(frame) == 2
    assert audit["invalid_json"] == 1
    assert audit["state_feature_count"] == 13
    assert audit["raw_mfe_available"] is False
    assert audit["requested_mfe_gt_tp_min_exactly_available"] is False
    assert {
        "regime",
        "tf_onehot_5m",
        "tf_onehot_1h",
        "tf_onehot_4h",
    }.issubset(audit["constant_context_features"])
    assert frame["tp_above_collector_floor"].tolist() == [0, 1]
    assert frame["profitable_trade"].tolist() == [0, 1]


def test_group_holdout_never_leaks_repeated_present_states() -> None:
    rows = []
    for group in range(20):
        for repetition in range(3):
            rows.append({"_state_group": str(group), "value": group + repetition})
    frame = pd.DataFrame(rows)

    train, holdout = bulletin._group_holdout(frame, seed=21)

    assert set(train["_state_group"]).isdisjoint(set(holdout["_state_group"]))
    assert len(train) + len(holdout) == len(frame)


def test_policy_attribution_requires_all_persisted_training_scalers() -> None:
    class Builder:
        scalers = {"5m": object(), "1h": object(), "4h": object()}
        scalers_loaded_from_training = True

    bulletin.require_persisted_scalers(Builder(), ("5m", "1h", "4h"))

    Builder.scalers_loaded_from_training = False
    try:
        bulletin.require_persisted_scalers(Builder(), ("5m", "1h", "4h"))
    except RuntimeError as exc:
        assert "persisted training scalers" in str(exc)
    else:
        raise AssertionError("refitted or unpersisted scalers must be rejected")


def test_arena_verdict_does_not_claim_long_run_authorization() -> None:
    report = {
        "feature_count": 16,
        "features": bulletin.FEATURES,
        "nonlinear_interactions": {"pair_count": 120},
        "trade_telemetry_audit": {
            "join_rate": 1.0,
            "joined_count": 118,
            "invalid_json": 0,
        },
        "policy_attribution": {"available": True},
        "arena_teacher_8445": {"audit": {"valid_rows": 8445, "invalid_json": 0}},
        "market_models": {
            "classification": {
                "good_long": {
                    "extra_trees": {"roc_auc": 0.53},
                    "hist_gradient_boosting": {"roc_auc": 0.54},
                }
            }
        },
        "trade_arena": {"targets": {"mfe_gt_tp_min": {"target": "mfe_gt_tp_min"}}},
    }

    verdict = bulletin.arena_verdict(report)

    assert verdict["status"] == "GREEN"
    assert "not a long-run authorization" in verdict["note"]
    assert verdict["checks"]["collector_teacher_sample_sufficient"] is True
