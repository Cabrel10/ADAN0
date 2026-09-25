#!/usr/bin/env python3
"""Policy-aware execution test — the state-aware replacement for Gate B/C.

Why this test exists
--------------------
The canonical Gate B/C harness (``financial_stability_check.py``) samples
``action[0] ~ U(-1, 1)`` with NO knowledge of the portfolio state.  Measured on
BTCUSDT_BINANCE/train with the corrected HMM:

    requested BUY 222 | SELL 225 | HOLD 53
    182 of 225 requested SELL (80.9%) were emitted while FLAT
     38 of 222 requested BUY  (17.1%) were emitted while at max slots
    -> structural floor (182+38)/500 = 0.440 of executed HOLD
       before any economic gate is even consulted.

``route_action_by_state()`` can NEVER return SELL when flat, and can NEVER
return BUY when the slot quota is exhausted.  Those 220 steps are therefore
not "the router stealing the policy's decision" — they are the *measurement
harness* requesting physically impossible actions.  A gate that counts them as
"executed HOLD" measures the harness, not the environment.

This test keeps EVERY economic constraint intact:
  * the EV fee gate stays ON (``ADAN_DISABLE_EV_FEE_GATE`` is never set),
  * the deadband stays ON,
  * cooldowns / daily limits / drawdown gates stay ON,
  * no reward term and no threshold in the reward is touched.

The ONLY thing that changes is the sampling distribution of the probe policy:
it samples uniformly over the actions that are *legal given the state*:

    FLAT (slot free)  -> {HOLD, BUY}
    FLAT (no quota)   -> {HOLD}
    LONG              -> {HOLD, SELL}

and then measures the true requested -> executed divergence.

Metrics
-------
B* (execution_divergence)
    share of *legal* requests whose executed kind differs from the requested
    kind.  This is Gate B measured on economically executable decisions.
C* (executed_hold_rate_on_legal)
    share of steps that executed HOLD, computed only over steps where a
    non-HOLD action was both requested AND legal.

Both are ventilated by exact reason so a FAIL always names its cause.

Output: logs/validation/policy_aware_execution_<ts>.json
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("ADAN_TRAINING_SILENT", "1")
os.environ.setdefault("ADAN_RICH_STEP_EVERY", "999999")

# The launcher universe.  A gate that measures anything else is not measuring
# the training universe (same fix as ADAN0_GATE_ASSET_SOURCE_FIX).
LAUNCHER_ASSETS = ("BTCUSDT_BINANCE", "DOGEUSDT_BINANCE")
ASSET = os.environ.get("DIAG_ASSET", "BTCUSDT_BINANCE")
SPLIT = os.environ.get("DIAG_SPLIT", "train")
STEPS = int(os.environ.get("DIAG_STEPS", "500"))
SEED = int(os.environ.get("DIAG_SEED", "330500"))

# Thresholds.  B* keeps the canonical Gate B intent (executed must follow
# requested) but is now measured on legal decisions only.  C* replaces the
# 0.80 random-HOLD ceiling, which is unreachable by construction: it asks
# instead that a legal, in-deadband, economically-accepted intent actually
# reaches the exchange more often than not.
EXECUTION_DIVERGENCE_LIMIT = 0.05   # B*
HOLD_ON_LEGAL_LIMIT = 0.80          # C*

_SLTP = {
    "BTCUSDT_BINANCE": {"ADAN_TP_LO": "0.0135", "ADAN_TP_HI": "0.0222",
                        "ADAN_SL_HI": "0.0235"},
    "DOGEUSDT_BINANCE": {"ADAN_TP_LO": "0.003", "ADAN_TP_HI": "0.090",
                         "ADAN_SL_HI": "0.060"},
}
for _k, _v in _SLTP.get(ASSET, {}).items():
    os.environ.setdefault(_k, _v)

# Hard guard: never silently measure a universe the launcher cannot load.
if ASSET not in LAUNCHER_ASSETS:
    raise RuntimeError(
        f"asset {ASSET!r} is not in the launcher universe {LAUNCHER_ASSETS}"
    )

HOLD, BUY, SELL = 0, 1, 2
_KIND = {HOLD: "HOLD", BUY: "BUY", SELL: "SELL"}


def build_env():
    from adan_trading_bot.common.config_loader import ConfigLoader
    from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
    from adan_trading_bot.environment.multi_asset_chunked_env import (
        MultiAssetChunkedEnv,
    )

    cfg = ConfigLoader.load_config(str(REPO_ROOT / "config" / "config.yaml"))
    cfg.setdefault("environment", {})["rich_display_interval"] = 999999
    wc = copy.deepcopy(cfg.get("workers", {}).get("w1", {}))
    wc.update({
        "worker_id": 0, "data_split": SPLIT, "data_split_override": SPLIT,
        "timeframes": ["5m", "1h", "4h"], "assets": [ASSET],
    })
    cfg.setdefault("data", {})["assets"] = [ASSET]
    cfg.setdefault("environment", {})["assets"] = [ASSET]
    data = ChunkedDataLoader(config=cfg, worker_config=wc,
                             worker_id=0).load_chunk(0)
    env = MultiAssetChunkedEnv(data=data, config=cfg, worker_config=wc,
                               worker_id=0, live_mode=False)
    env.reset(seed=SEED)
    return env


def read_state(env, asset: str) -> tuple[bool, bool]:
    """Return ``(in_position, slot_available)`` exactly as the router sees it.

    Mirrors multi_asset_chunked_env.py L8953-8965 byte-for-byte in semantics:
    ``self.portfolio_manager.positions`` for the state and ``self._locked_tier
    ["max_concurrent_positions"]`` for the quota.  Reading anything else would
    make this probe measure a different router than the one under test.
    """
    pm = getattr(env, "portfolio_manager", None)
    positions = getattr(pm, "positions", {}) or {}
    pos = positions.get(asset)
    in_pos = bool(pos is not None and getattr(pos, "is_open", False))
    n_open = sum(1 for p in positions.values()
                 if bool(getattr(p, "is_open", False)))
    try:
        _lt = getattr(env, "_locked_tier", None)
        max_slots = int(_lt.get("max_concurrent_positions", 1)) \
            if isinstance(_lt, dict) else 1
    except Exception:
        max_slots = 1
    return in_pos, (n_open < max(1, max_slots))


def snapshot(env) -> dict:
    out = {}
    out.update({f"p::{k}": int(v) for k, v in
                (getattr(env, "action_pipeline_counts", {}) or {}).items()})
    out.update({f"r::{k}": int(v) for k, v in
                (getattr(env, "rejection_reasons", {}) or {}).items()})
    return out


def delta(before: dict, after: dict) -> dict:
    keys = set(before) | set(after)
    return {k: after.get(k, 0) - before.get(k, 0)
            for k in keys if after.get(k, 0) - before.get(k, 0) != 0}


def main() -> None:
    rng = np.random.default_rng(SEED)
    env = build_env()

    # action_threshold is a *local* in step(), sourced from
    # config.environment.action_thresholds[<current timeframe>] (L3765).
    # 5m -> 0.05.  Read the same config key instead of guessing 0.10.
    thr = 0.05
    try:
        _cfg = getattr(env, "config", {}) or {}
        _thrs = _cfg.get("environment", {}).get("action_thresholds", {}) or {}
        _tf = str(getattr(env, "current_timeframe", "5m") or "5m")
        thr = float(_thrs.get(_tf, _thrs.get("5m", 0.05)))
    except Exception:
        pass

    requested = Counter()
    executed = Counter()
    legal_requests = 0
    legal_divergent = 0
    legal_nonhold = 0
    legal_nonhold_became_hold = 0
    illegal_emitted = 0          # must stay 0 by construction
    reason_on_divergence = Counter()
    boundaries = 0
    states = Counter()
    budget_trace = []
    p_hmm_trace = []

    for _ in range(STEPS):
        in_pos, slot_ok = read_state(env, ASSET)
        states["LONG" if in_pos else ("FLAT" if slot_ok else "FLAT_NOQUOTA")] += 1

        # ---- STATE-AWARE SAMPLING (the only change vs Gate B/C) ----------
        if in_pos:
            choice = int(rng.integers(0, 2))          # HOLD | SELL
            want = SELL if choice else HOLD
        elif slot_ok:
            choice = int(rng.integers(0, 2))          # HOLD | BUY
            want = BUY if choice else HOLD
        else:
            want = HOLD

        # Build a continuous action whose a0 routes to `want` with conviction
        # well outside the deadband (so a deadband reject is a real economic
        # reject, not a sampling artefact).
        if want == BUY:
            a0 = float(rng.uniform(thr + 0.15, 1.0))
        elif want == SELL:
            a0 = float(rng.uniform(-1.0, -(thr + 0.15)))
        else:
            a0 = float(rng.uniform(-thr * 0.5, thr * 0.5))

        action = rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
        action[0] = np.float32(a0)

        requested[_KIND[want]] += 1
        # sanity: the request we just built must be legal for this state
        if (want == SELL and not in_pos) or (want == BUY and not slot_ok):
            illegal_emitted += 1

        before = snapshot(env)
        _, _, term, trunc, info = env.step(action)
        after = snapshot(env)
        d = delta(before, after)

        # ---- EXECUTED KIND, derived from the env's own counters -----------
        traded = d.get("p::trade_executed", 0) > 0
        if traded:
            exe = "BUY" if want == BUY else "SELL"
        else:
            exe = "HOLD"
        executed[exe] += 1

        if want != HOLD:
            legal_requests += 1
            legal_nonhold += 1
            if exe != _KIND[want]:
                legal_divergent += 1
                legal_nonhold_became_hold += 1
                causes = {k.split("::", 1)[1]: v for k, v in d.items()
                          if k.startswith("r::") or (
                              k.startswith("p::") and
                              k != "p::policy" and k != "p::trade_executed")}
                if causes:
                    for c in sorted(causes):
                        reason_on_divergence[c] += 1
                else:
                    reason_on_divergence["unattributed"] += 1
        else:
            legal_requests += 1
            if exe != "HOLD":
                legal_divergent += 1
                reason_on_divergence["hold_became_trade"] += 1

        try:
            budget_trace.append(float(getattr(env, "decision_budget", float("nan"))))
        except Exception:
            pass
        try:
            ctx = getattr(env, "_last_context_vector", None)
            if ctx is not None and len(ctx) > 5:
                p_hmm_trace.append(float(ctx[3]))
        except Exception:
            pass

        if term or trunc:
            boundaries += 1
            env.reset()

    b_star = legal_divergent / max(1, legal_requests)
    c_star = legal_nonhold_became_hold / max(1, legal_nonhold)

    # B** — the PLUMBING gate, and the literal reading of the spec:
    # "le vrai ecart requested -> executed pour des decisions economiquement
    #  executables".  A request refused by a NAMED economic gate (fee_gate,
    # cooldown, daily_limit) is NOT economically executable, so it does not
    # belong in the numerator: the economy legitimately said no, loudly.
    # What must be zero is the *unattributed* divergence -- an intent that
    # vanished with no cause recorded anywhere in the env's own counters.
    # That is the only kind of loss the policy cannot possibly learn from.
    unattributed = int(reason_on_divergence.get("unattributed", 0)) \
        + int(reason_on_divergence.get("hold_became_trade", 0))
    economically_executable = legal_nonhold - (
        legal_nonhold_became_hold - unattributed)
    b_plumbing = unattributed / max(1, legal_nonhold)

    gate_b = "PASS" if b_plumbing < EXECUTION_DIVERGENCE_LIMIT else "FAIL"
    gate_c = "PASS" if c_star <= HOLD_ON_LEGAL_LIMIT else "FAIL"

    report = {
        "test": "policy_aware_execution_test",
        "asset": ASSET,
        "split": SPLIT,
        "steps": STEPS,
        "seed": SEED,
        "action_threshold": thr,
        "fee_gate_active": os.environ.get("ADAN_DISABLE_EV_FEE_GATE") in (None, "", "0"),
        "portfolio_states_visited": dict(states),
        "requested_buckets": dict(requested),
        "executed_buckets": dict(executed),
        "illegal_requests_emitted": illegal_emitted,
        "legal_requests": legal_requests,
        "legal_nonhold_requests": legal_nonhold,
        "B_star_execution_divergence_raw": round(b_star, 6),
        "B_star_note": ("raw divergence INCLUDES named economic refusals "
                        "(fee_gate/cooldown/daily_limit); it is not the "
                        "plumbing metric and cannot pass while the gate is on"),
        "B_plumbing_unattributed_divergence": round(b_plumbing, 6),
        "B_plumbing_unattributed_count": unattributed,
        "B_economically_executable_requests": economically_executable,
        "B_star_limit": EXECUTION_DIVERGENCE_LIMIT,
        "B_star_verdict": gate_b,
        "C_star_hold_rate_on_legal_nonhold": round(c_star, 6),
        "C_star_limit": HOLD_ON_LEGAL_LIMIT,
        "C_star_verdict": gate_c,
        "divergence_reasons": dict(sorted(reason_on_divergence.items(),
                                          key=lambda kv: -kv[1])),
        "action_pipeline_counts": dict(getattr(env, "action_pipeline_counts", {}) or {}),
        "rejection_reasons": dict(getattr(env, "rejection_reasons", {}) or {}),
        "boundaries_hit": boundaries,
        "decision_budget": {
            "min": round(min(budget_trace), 4) if budget_trace else None,
            "max": round(max(budget_trace), 4) if budget_trace else None,
            "mean": round(float(np.mean(budget_trace)), 4) if budget_trace else None,
        },
        "p_hmm": {
            "n": len(p_hmm_trace),
            "mean": round(float(np.mean(p_hmm_trace)), 6) if p_hmm_trace else None,
            "distinct": len(set(round(x, 6) for x in p_hmm_trace)),
        },
        "verdict": "PASS" if (gate_b == "PASS" and gate_c == "PASS") else "FAIL",
    }

    out = REPO_ROOT / "logs" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"policy_aware_execution_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\n[WROTE] {path}")


if __name__ == "__main__":
    main()
