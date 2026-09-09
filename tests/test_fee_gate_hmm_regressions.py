"""Regression tests guarding the fee_gate / HMM discoveries.

These tests freeze the two production bugs that were measured (not deduced)
during the 2026-09 diagnostics saga, so that any refactor reintroducing them
fails loudly instead of silently poisoning a training run:

1. HMM read-only consumers (dynamic_behavior_engine.py, marker
   ADAN0_HMM_READONLY_CONSUMERS, commit 50eeee4):
   a caller supplying neither observation_id nor a usable log_return must
   receive the cached posterior WITHOUT ingesting a row into the rolling fit
   buffer and WITHOUT overwriting _hmm_probs. Measured before the fix: 250 of
   500 buffer rows were one repeated synthetic point and 49.92% of posteriors
   described that fake row.

2. EV fee gate semantics (action_routing.resolve_ev_fee_gate): the gate must
   block BUY when p_hmm <= p_min_required, accept strictly above, and when
   disabled must degrade to advisory telemetry — never silently drop the
   measurement.
"""
import numpy as np
import pytest

from adan_trading_bot.environment.action_routing import resolve_ev_fee_gate
from adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine


def _make_dbe() -> DynamicBehaviorEngine:
    """Minimal DBE instance sufficient for the HMM plumbing tests.

    _hmm_obs_buffer is initialized lazily inside _init_hmm() (engine L357),
    which get_regime_probabilities calls first — so we trigger it once here
    before any assertion reads the buffer.
    """
    dbe = DynamicBehaviorEngine(config={"risk_parameters": {}}, worker_id=0)
    dbe._init_hmm()
    return dbe


class TestHMMReadOnlyConsumers:
    """Guards the ADAN0_HMM_READONLY_CONSUMERS producer/consumer split."""

    def test_consumer_does_not_ingest_into_fit_buffer(self):
        dbe = _make_dbe()
        buf_before = len(dbe._hmm_obs_buffer)
        # Consumer call: featureless market_conditions (no id, no log_return).
        dbe.get_regime_probabilities({"close": 100.0, "asset": "BTCUSDT"})
        assert len(dbe._hmm_obs_buffer) == buf_before, (
            "a featureless consumer call must not append to the HMM fit buffer"
        )

    def test_consumer_does_not_overwrite_probs_state(self):
        dbe = _make_dbe()
        sentinel = np.array([0.7, 0.2, 0.1])
        dbe._hmm_probs = sentinel.copy()
        out = dbe.get_regime_probabilities({"close": 100.0, "asset": "BTCUSDT"})
        np.testing.assert_array_equal(
            dbe._hmm_probs, sentinel,
            err_msg="consumer call must leave _hmm_probs untouched")
        np.testing.assert_array_equal(out, sentinel)

    def test_consumer_gets_a_copy_not_the_live_state(self):
        dbe = _make_dbe()
        out = dbe.get_regime_probabilities({"close": 100.0, "asset": "BTCUSDT"})
        assert out is not dbe._hmm_probs, (
            "consumer must receive a copy; mutating it must not corrupt state")

    def test_readonly_call_counter_increments(self):
        dbe = _make_dbe()
        n0 = getattr(dbe, "_hmm_readonly_calls", 0)
        dbe.get_regime_probabilities({"close": 100.0})
        dbe.get_regime_probabilities({"close": 101.0})
        assert dbe._hmm_readonly_calls == n0 + 2

    def test_producer_with_observation_id_ingests_once(self):
        dbe = _make_dbe()
        buf_before = len(dbe._hmm_obs_buffer)
        md = {"observation_id": 1, "log_return": 0.001,
              "atr_pct": 0.002, "rsi_norm": 0.5, "volume_ratio_20": 1.0}
        dbe.get_regime_probabilities(md)
        assert len(dbe._hmm_obs_buffer) == buf_before + 1, (
            "a producer call must append exactly one row")
        # Same observation_id again -> dedup cache, no second ingestion.
        dbe.get_regime_probabilities(dict(md))
        assert len(dbe._hmm_obs_buffer) == buf_before + 1, (
            "repeated observation_id must be served from the dedup cache")

    def test_producer_with_log_return_only_still_ingests(self):
        # log_ret != 0.0 alone qualifies as producer even without an id.
        dbe = _make_dbe()
        buf_before = len(dbe._hmm_obs_buffer)
        dbe.get_regime_probabilities({"log_return": 0.002})
        assert len(dbe._hmm_obs_buffer) == buf_before + 1

    def test_close_prev_close_fallback_counts_as_producer(self):
        # The fallback path (close/prev_close -> log_ret != 0) must produce.
        dbe = _make_dbe()
        buf_before = len(dbe._hmm_obs_buffer)
        dbe.get_regime_probabilities({"close": 101.0, "prev_close": 100.0})
        assert len(dbe._hmm_obs_buffer) == buf_before + 1

    def test_probs_contract_shape_and_sum(self):
        dbe = _make_dbe()
        out = dbe.get_regime_probabilities({"close": 100.0})
        assert out.shape == (3,)
        assert abs(float(out.sum()) - 1.0) < 1e-6
        assert np.all(out >= 0.0)


class TestEVFeeGateSemantics:
    """Guards the resolve_ev_fee_gate contract measured in RAPPORT_FEE_GATE_HMM."""

    def test_blocks_when_p_hmm_at_or_below_threshold(self):
        blocked, reason = resolve_ev_fee_gate(
            p_hmm=0.10, p_min_required=0.47, disabled=False)
        assert blocked and reason == "negative_ev_fee_gate"
        # Boundary: strictly greater passes, equal blocks.
        blocked, _ = resolve_ev_fee_gate(
            p_hmm=0.47, p_min_required=0.47, disabled=False)
        assert blocked

    def test_accepts_strictly_above_threshold(self):
        blocked, reason = resolve_ev_fee_gate(
            p_hmm=0.471, p_min_required=0.47, disabled=False)
        assert not blocked and reason == "accepted"

    def test_disabled_degrades_to_advisory_never_silent(self):
        # Gate condition fails but disabled -> pass through, and the
        # negative comparison MUST stay observable (not 'accepted').
        blocked, reason = resolve_ev_fee_gate(
            p_hmm=0.10, p_min_required=0.47, disabled=True)
        assert not blocked
        assert reason == "disabled_advisory", (
            "disabled gate must report 'disabled_advisory' so the blocked "
            "comparison remains in telemetry — see RAPPORT_FEE_GATE_HMM §7")

    def test_disabled_true_positive_still_reports_accepted(self):
        blocked, reason = resolve_ev_fee_gate(
            p_hmm=0.90, p_min_required=0.47, disabled=True)
        assert not blocked and reason == "accepted"
