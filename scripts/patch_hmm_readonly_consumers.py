#!/usr/bin/env python3
"""Stop featureless consumers from ingesting fake observations into the HMM.

Measured problem (logs/validation/hmm_buffer_contamination_20260905_003847.json)
------------------------------------------------------------------------------
    buffer_len                                        500
    buffer_synthetic_rows                             250   (50.0%)
    buffer_distinct_points                            251   (250 real + 1 fake)
    share_posterior_computed_on_synthetic_last_row  0.4992
    engine_total_obs                                  600   for 300 env steps

`get_regime_probabilities` has two callers:

  A. multi_asset_chunked_env.py L6315 -> _get_current_market_data_for_hmm()
     passes the 4 real features AND an observation_id. This is the PRODUCER.

  B. dynamic_behavior_engine.py L915, via detect_market_regime(), reached from
     update_risk_parameters() L986 and env L1357 with `market_conditions`
     (built at env L874: only {"close", "asset"} plus optional indicators,
     never prev_close, never observation_id). This is a CONSUMER: it only wants
     to READ the current regime label. It carries no HMM features, so the
     method falls back to (0.0, 0.0, 0.5, 1.0) and _update_hmm appends that
     constant row to the rolling fit buffer — 250 times out of 500.

Fix
---
Make the read path read-only. A call that supplies neither an observation_id
nor a usable log_return is a consumer, so return the cached posterior instead of
ingesting a synthetic observation. The producer path is untouched: it always
supplies both.

This is deliberately the smallest possible change:
  * no HMM hyperparameter is touched
  * no reward, PPO, SL/TP or gate logic is touched
  * the producer's semantics are byte-identical
  * a counter (_hmm_readonly_calls) is added so the effect is measurable

Idempotent: re-running is a no-op. Anchors must match byte-exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "src/adan_trading_bot/environment/dynamic_behavior_engine.py"
MARK = "ADAN0_HMM_READONLY_CONSUMERS"


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


ANCHOR_BODY = '''        self._init_hmm()

        observation_id = market_data.get("observation_id")
        if observation_id is not None and observation_id == self._hmm_last_observation_id:
            return self._hmm_probs.copy()
        self._hmm_last_observation_id = observation_id

        # Extract 4D feature vector
        log_ret = market_data.get("log_return", 0.0)
        atr_pct = market_data.get("atr_pct", 0.0)
        rsi_norm = market_data.get("rsi_norm", 0.5)
        volume_ratio = market_data.get("volume_ratio_20", 1.0)

        # Fallback: compute log_return from close/prev_close if not provided directly
        if log_ret == 0.0:
            close = market_data.get("close", 0.0)
            prev_close = market_data.get("prev_close", close)
            if prev_close > 0 and close > 0:
                log_ret = float(np.log(close / prev_close))

        return self._update_hmm(log_ret, atr_pct, rsi_norm, volume_ratio)
'''

REPLACEMENT_BODY = '''        self._init_hmm()

        observation_id = market_data.get("observation_id")
        if observation_id is not None and observation_id == self._hmm_last_observation_id:
            return self._hmm_probs.copy()

        # Extract 4D feature vector
        log_ret = market_data.get("log_return", 0.0)
        atr_pct = market_data.get("atr_pct", 0.0)
        rsi_norm = market_data.get("rsi_norm", 0.5)
        volume_ratio = market_data.get("volume_ratio_20", 1.0)

        # Fallback: compute log_return from close/prev_close if not provided directly
        if log_ret == 0.0:
            close = market_data.get("close", 0.0)
            prev_close = market_data.get("prev_close", close)
            if prev_close > 0 and close > 0:
                log_ret = float(np.log(close / prev_close))

        # ADAN0_HMM_READONLY_CONSUMERS: separate READ from WRITE.
        # Measured before this guard (hmm_buffer_contamination_20260905_003847):
        #   599 calls for 300 env steps; observation_id None in 49.92% of them;
        #   250 of the 500 rolling-fit rows were ONE repeated synthetic point
        #   (0.0, 0.0, 0.5, 1.0), leaving 251 distinct points out of 500; and
        #   49.92% of returned posteriors were predict_proba(X)[-1] of that
        #   fake row, i.e. p_hmm did not describe the market at all.
        # Cause: detect_market_regime() (L915) is a CONSUMER. It is reached with
        # `market_conditions` (env L874 = {"close", "asset"} + indicators, no
        # prev_close, no observation_id), so every feature fell back to its
        # default and got appended to the fit buffer.
        # A caller that supplies neither an observation_id nor a usable
        # log_return cannot be the producer, so serve it the cached posterior
        # and ingest nothing.
        _is_producer = (observation_id is not None) or (log_ret != 0.0)
        if not _is_producer:
            self._hmm_readonly_calls = getattr(
                self, "_hmm_readonly_calls", 0) + 1
            return self._hmm_probs.copy()

        self._hmm_last_observation_id = observation_id
        return self._update_hmm(log_ret, atr_pct, rsi_norm, volume_ratio)
'''

# Expose the new counter in the engine's own diagnostics init.
ANCHOR_INIT = '''            self._hmm_total_obs = 0       # total observations ever ingested
            self._hmm_last_refit_obs = 0  # value of _hmm_total_obs at last fit attempt
'''

REPLACEMENT_INIT = '''            self._hmm_total_obs = 0       # total observations ever ingested
            self._hmm_last_refit_obs = 0  # value of _hmm_total_obs at last fit attempt
            # ADAN0_HMM_READONLY_CONSUMERS: count consumer calls served from
            # cache (they used to poison the fit buffer). Auditable at runtime.
            self._hmm_readonly_calls = 0
'''


def main() -> None:
    if not TARGET.exists():
        fail(f"target missing: {TARGET}")
    src = TARGET.read_text()

    if MARK in src:
        print(f"[SKIP] already patched ({MARK} present) — idempotent no-op")
        return

    for name, anchor in (("body", ANCHOR_BODY), ("init", ANCHOR_INIT)):
        if src.count(anchor) != 1:
            fail(f"anchor '{name}' found {src.count(anchor)} times, expected 1")

    src = src.replace(ANCHOR_BODY, REPLACEMENT_BODY, 1)
    src = src.replace(ANCHOR_INIT, REPLACEMENT_INIT, 1)

    import ast
    try:
        ast.parse(src)
    except SyntaxError as exc:
        fail(f"patched source does not parse: {exc}")

    TARGET.write_text(src)
    print(f"[OK] patched {TARGET}")
    print(f"[OK] markers: {src.count(MARK)}")


if __name__ == "__main__":
    main()
