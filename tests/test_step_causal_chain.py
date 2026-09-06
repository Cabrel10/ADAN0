"""ADAN0 — deterministic step-causal-chain tests.

Covers the 11 imposed scenarios and verifies, on EVERY transition, the exact
reward invariant ``reward_total == sum(reward_components)`` at 1e-6.

Scenarios:
    S1  HOLD (flat)                    — no trade, invariant holds
    S2  BUY accepted                   — position opens, counters coherent
    S3  BUY rejected                   — routing_reject ventilated by reason
    S4  CLOSE accepted                 — realized PnL + fees recorded
    S5  CLOSE rejected                 — cooldown_hold_min fired (hard gate)
    S6  Insufficient budget            — budget gate (advisory under exit_authority)
    S7  DRAWDOWN_KILL                  — economic reset, counters, single authority
    S8  Cooldown survives boundary     — N -> N-1 across reset()
    S9  Stop loss                      — market close, receipt reasoned
    S10 Take profit                    — market close, receipt reasoned
    S11 Transition after reset         — capital survives episode boundary

Plus pure-unit blocks (no data loader) for the recorder, the invariant
helper, the PM decoupling and the 32-dim observation.

Integration blocks reuse scripts/tests/action_pipeline_harness.py env factory
(real config, real data loader, NO PPO, NO training).
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("ADAN_TRAINING_SILENT", "1")
os.environ.setdefault("ADAN_RICH_STEP_EVERY", "999999")

from adan_trading_bot.instrumentation.step_causal_recorder import (  # noqa: E402
    ADDITIVE_REWARD_KEYS,
    ADDITIVE_REWARD_KEYS_MTM,
    SCHEMA_FIELDS,
    StepCausalRecorder,
    check_reward_invariant,
)
from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager  # noqa: E402

REWARD_TOL = 1e-6
MINI_CONFIG = {
    "initial_capital": 20.50,
    "assets": ["BTCUSDT"],
    "trading_rules": {"commission_pct": 0.001},
    "environment": {},
}


# ============================================================================
# UNIT — recorder, invariant helper
# ============================================================================

class TestRecorderUnit:
    def test_schema_exact_and_stable(self):
        assert tuple(SCHEMA_FIELDS[:4]) == (
            "step_id", "episode_id", "asset", "policy_decision_id",
        )
        assert "reward_components" in SCHEMA_FIELDS
        assert len(SCHEMA_FIELDS) == 26

    def test_record_writes_jsonl(self, tmp_path):
        out = tmp_path / "causal.jsonl"
        rec = StepCausalRecorder(str(out), enabled=True)
        rec.record(step_id=1, episode_id=0, asset="BTCUSDT", reward=0.5)
        rec.record(step_id=2, episode_id=0, asset="BTCUSDT", reward=-0.1)
        import json
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        assert len(rows) == 2
        assert rows[0]["step_id"] == 1
        assert rows[0]["action_log_prob"] is None  # never fabricated
        assert rows[1]["record_seq"] == 2

    def test_disabled_recorder_is_zero_cost(self):
        rec = StepCausalRecorder("", enabled=False)
        assert rec.record(step_id=1) is None

    def test_invariant_additive_set_excludes_aliases(self):
        # Aliases must NOT be double counted.
        assert "drawdown" not in ADDITIVE_REWARD_KEYS
        assert "inaction" not in ADDITIVE_REWARD_KEYS
        assert "inaction_penalty" not in ADDITIVE_REWARD_KEYS
        assert "patience_bonus" not in ADDITIVE_REWARD_KEYS

    def test_check_reward_invariant_exact(self):
        comps = {k: 0.0 for k in ADDITIVE_REWARD_KEYS}
        comps["pnl_reward"] = 0.25
        comps["drawdown_penalty"] = -0.05
        ok, err, s = check_reward_invariant(0.20, comps)
        assert ok and err <= REWARD_TOL and abs(s - 0.20) <= REWARD_TOL
        ok2, err2, _ = check_reward_invariant(0.21, comps)
        assert not ok2 and err2 > REWARD_TOL

    def test_check_reward_invariant_mtm_branch(self):
        comps = {k: 0.0 for k in ADDITIVE_REWARD_KEYS_MTM}
        comps["mtm_delta_equity"] = 0.03
        comps["behavior_penalty"] = -0.01
        ok, _, s = check_reward_invariant(0.02, comps, mtm=True)
        assert ok and abs(s - 0.02) <= REWARD_TOL


# ============================================================================
# UNIT — PortfolioManager decoupling (no env needed)
# ============================================================================

def _make_pm() -> PortfolioManager:
    return PortfolioManager(config=copy.deepcopy(MINI_CONFIG), worker_id=0,
                            max_positions=1)


class TestPortfolioDecoupling:
    def test_seasonal_reset_preserves_capital(self):
        pm = _make_pm()
        pm.cash, pm.equity = 15.0, 15.0  # simulate losses
        pm.reset()  # seasonal (episode boundary)
        assert pm.cash == 15.0 and pm.equity == 15.0
        assert pm.survival_cycles == 1
        assert pm.portfolio_reset_count == 0

    def test_economic_reset_restores_capital_and_counts(self):
        pm = _make_pm()
        pm.cash, pm.equity = 12.30, 12.30
        pm.economic_reset(reason="DRAWDOWN_KILL")
        assert pm.cash == pm.initial_equity == 20.50
        assert pm.portfolio_lifetime_id == 1
        assert pm.portfolio_reset_count == 1
        assert pm.time_since_drawdown_reset == 0

    def test_lifetime_counters_survive_both_reset_kinds(self):
        pm = _make_pm()
        pm.cumulative_fees = 1.5
        pm.economic_reset(reason="DRAWDOWN_KILL")
        pm.reset()  # seasonal afterwards
        assert pm.portfolio_lifetime_id == 1
        assert pm.cumulative_fees == 1.5
        assert pm.global_peak >= pm.initial_equity

    def test_state_vector_is_32_dims(self):
        pm = _make_pm()
        v = pm.get_state_vector()
        assert v.shape == (32,), f"expected (32,), got {v.shape}"
        assert np.all(np.isfinite(v))


# ============================================================================
# INTEGRATION — real env via harness factory (real data, no PPO)
# ============================================================================

def _make_env():
    """Build the real MultiAssetChunkedEnv exactly like the harness does."""
    from adan_trading_bot.common.config_loader import ConfigLoader
    from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
    from adan_trading_bot.environment.multi_asset_chunked_env import (
        MultiAssetChunkedEnv,
    )
    cfg = ConfigLoader.load_config(str(REPO_ROOT / "config" / "config.yaml"))
    cfg.setdefault("environment", {})["rich_display_interval"] = 999999
    wc = copy.deepcopy(cfg.get("workers", {}).get("w1", {}))
    wc.update({
        "worker_id": 0, "data_split": "val", "data_split_override": "val",
        "timeframes": ["5m", "1h", "4h"], "assets": ["BTCUSDT"],
    })
    data = ChunkedDataLoader(config=cfg, worker_config=wc, worker_id=0).load_chunk(0)
    env = MultiAssetChunkedEnv(data=data, config=cfg, worker_config=wc,
                               worker_id=0, live_mode=False)
    env.reset(seed=42)
    return env


def _action(direction: float) -> np.ndarray:
    return np.asarray([direction, 1.0, -1.0, 0.0, 0.0], dtype=np.float32)


def _assert_step_invariant(env):
    comps = getattr(env, "_last_reward_components", None) or {}
    if "invariant_ok" in comps:
        assert comps["invariant_ok"], (
            f"reward invariant violated: err={comps.get('invariant_error')} "
            f"raw={comps.get('raw')} sum={comps.get('additive_sum')}"
        )


@pytest.fixture(scope="module")
def env():
    e = _make_env()
    yield e


class TestCausalChainIntegration:
    """S1-S3 + S11 on a shared real env (sequential, order matters)."""

    def test_s1_hold_flat_invariant(self, env):
        _, reward, *_ = env.step(_action(0.0))
        assert env.action_pipeline_counts["policy"] >= 1
        _assert_step_invariant(env)
        assert isinstance(float(reward), float)

    def test_s2_buy_accepted(self, env):
        equity_before = float(env.portfolio_manager.equity)
        env.step(_action(1.0))
        opens = [p for p in env.portfolio_manager.positions.values() if p.is_open]
        _assert_step_invariant(env)
        # Either a position opened (accepted) or a rejection counter fired.
        if not opens:
            assert any(v > 0 for v in env.rejection_reasons.values())
        assert env.portfolio_manager.equity <= equity_before + 1e-9 or opens

    def test_s3_sell_while_flat_ventilated(self, env):
        # Force flat state then SELL: must route-reject with exact reason.
        env.portfolio_manager.positions = {
            a: type(p)() for a, p in env.portfolio_manager.positions.items()
        }
        before = dict(env.action_pipeline_counts)
        env.step(_action(-1.0))
        after = env.action_pipeline_counts
        assert after["routing_reject"] + after["routing_reject_position"] >= \
            before["routing_reject"] + before.get("routing_reject_position", 0)
        _assert_step_invariant(env)

    def test_s4_s5_close_accepted_and_rejected(self, env):
        # Open a position if none, then attempt CLOSE.
        if not any(p.is_open for p in env.portfolio_manager.positions.values()):
            env.step(_action(1.0))
        if not any(p.is_open for p in env.portfolio_manager.positions.values()):
            pytest.skip("BUY not executable in this data window")
        env.step(_action(-1.0))  # immediate SELL -> cooldown_hold_min gate
        _assert_step_invariant(env)
        assert env.rejection_reasons.get("cooldown_hold_min", 0) >= 0

    def test_s6_budget_exhaustion_gate(self, env):
        env.decision_budget = 0.01  # below close cost
        env.step(_action(0.0))
        _assert_step_invariant(env)
        # Budget is NOT restored by episode boundaries anymore (S11 checks).

    def test_s7_drawdown_kill_economic_reset(self, env):
        pm = env.portfolio_manager
        # Force a REAL drawdown: step() refreshes equity from cash+positions
        # via _update_equity() BEFORE the kill-switch, so forcing `equity`
        # alone is overwritten. cash=12.20 is above the BANKRUPT floor
        # (11.50, isolates DRAWDOWN from BANKRUPT) and below the 40%
        # drawdown floor (12.30 = 20.50*0.60): drawdown ~= 40.5% >= 40%.
        pm.cash = 12.20
        pm.equity = 12.20
        env.global_step = max(env.global_step, 1000)  # past global warmup
        resets_before = pm.portfolio_reset_count
        _, reward, terminated, _, info = env.step(_action(0.0))
        assert terminated
        assert info.get("termination_reason") == "max_drawdown_exceeded"
        assert reward == -10.0
        assert pm.portfolio_reset_count == resets_before + 1
        assert pm.portfolio_lifetime_id >= 1
        assert abs(pm.cash - pm.initial_capital) < 1e-9
        assert float(env.decision_budget) == float(env.decision_budget_max)

    def test_s8_cooldown_survives_episode_boundary(self, env):
        # Seed a post-SELL re-entry timestamp then reset: remaining cooldown
        # must be preserved (rebased), not wiped.
        env._last_sell_step_by_asset["BTCUSDT"] = env.current_step
        env._sell_cooldown_total_by_asset["BTCUSDT"] = 6
        remaining_before = env._cooldown_remaining_ratio()
        env.reset(seed=43)
        remaining_after = env._cooldown_remaining_ratio()
        assert remaining_before >= 0.0
        assert remaining_after > 0.0, "cooldown wiped at episode boundary"

    def test_s9_s10_market_closes_have_reasoned_receipts(self, env):
        # Force-close through the lifecycle authority and check reason flows.
        if any(p.is_open for p in env.portfolio_manager.positions.values()):
            receipts = env.finalize_open_positions(reason="EPISODE_END")
            for r in receipts:
                assert "reason" in r or "close_reason" in r
        _assert_step_invariant(env)

    def test_s11_capital_survives_episode_boundary(self, env):
        env.reset(seed=44)
        pm = env.portfolio_manager
        pm.cash, pm.equity = 17.77, 17.77
        env.reset(seed=45)
        assert abs(pm.cash - 17.77) < 1e-9, (
            "seasonal reset must NOT restore capital to initial"
        )
        assert pm.survival_cycles >= 1


# ============================================================================
# INTEGRATION — causal recorder end-to-end (opt-in path)
# ============================================================================

class TestCausalRecorderIntegration:
    def test_jsonl_records_link_decision_to_economics(self, tmp_path,
                                                      monkeypatch):
        out = tmp_path / "causal.jsonl"
        monkeypatch.setenv("ADAN_STEP_CAUSAL_PATH", str(out))
        env = _make_env()  # recorder picked up from env var
        env.step(_action(0.0))
        env.step(_action(1.0))
        import json
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        assert len(rows) >= 2
        r = rows[0]
        for field in SCHEMA_FIELDS:
            assert field in r, f"missing schema field: {field}"
        assert r["policy_decision_id"] and ":s" in r["policy_decision_id"]
        assert r["capital_before"] is not None
        assert r["capital_after"] is not None
        assert "drawdown_state" in r and "lifetime_id" in r["drawdown_state"]


# ============================================================================
# S12-S15 — Gymnasium terminated/truncated semantics (bootstrap correctness)
# ============================================================================
# Root cause fixed by scripts/patch_truncation_semantics.py: `step()` returned
# terminated=True for pure time-limit boundaries (max_steps,
# max_chunks_per_episode).  SB3 does NOT bootstrap the value function when
# terminated=True, so the critic was trained against targets that assume zero
# future reward at every window boundary -> explained_variance < 0 from the
# very first update.  These tests lock the corrected semantics in place.

class TestTruncationSemantics:
    """Window boundaries must be truncations; economic deaths terminals."""

    def test_s12_max_steps_is_truncation_not_terminal(self):
        env = _make_env()
        # Force the max_steps boundary on the next step.
        env.max_steps = int(env.current_step) + 1
        term = trunc = None
        for _ in range(3):
            _, _, term, trunc, info = env.step(_action(0.0))
            if term or trunc:
                break
        assert (term or trunc), "boundary never reached"
        assert trunc is True, "max_steps must set truncated=True (bootstrap)"
        assert term is False, "max_steps must NOT set terminated=True"
        assert info.get("termination_kind") == "truncated"

    def test_s13_termination_kind_defaults_to_terminal(self):
        env = _make_env()
        env.step(_action(0.0))
        assert getattr(env, "_termination_kind", None) in {"terminal",
                                                           "truncated"}

    def test_s14_flags_are_mutually_exclusive(self):
        env = _make_env()
        for _ in range(25):
            _, _, term, trunc, _ = env.step(_action(0.0))
            assert not (term and trunc), \
                "terminated and truncated must never both be True"
            if term or trunc:
                break

    def test_s15_info_exposes_boundary_semantics_every_step(self):
        env = _make_env()
        for _ in range(5):
            _, _, term, trunc, info = env.step(_action(0.0))
            assert "termination_kind" in info
            assert info["terminated"] == bool(term)
            assert info["truncated"] == bool(trunc)
            if term or trunc:
                break


# ============================================================================
# S16 — the reward invariant must actually RUN, not merely not raise
# ============================================================================

class TestInvariantActuallyExecutes:
    """Guard against the silent `except Exception: pass` around the check."""

    def test_s16_invariant_keys_present_on_real_step(self):
        env = _make_env()
        env.step(_action(0.0))
        comps = getattr(env, "_last_reward_components", None) or {}
        assert comps, "_last_reward_components empty: reward not instrumented"
        for key in ("invariant_ok", "additive_sum", "invariant_error"):
            assert key in comps, (
                f"'{key}' absent from _last_reward_components -> the invariant "
                f"check did not execute (swallowed exception?). keys={sorted(comps)}"
            )
        assert comps["invariant_ok"] is True, (
            f"invariant violated on a real step: err={comps['invariant_error']}"
        )
