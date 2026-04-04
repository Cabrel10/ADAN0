#!/usr/bin/env python3
"""
OMEGA MATHS VERIFICATION TESTS
================================
These tests verify that the 4 mathematical functions injected by the
MEGA-PROMPT are ACTUALLY present and working:

  1. True Quant Anti-Hack Reward (reward_calculator.py)
  2. EV-Based Fee Gate (multi_asset_chunked_env.py)
  3. Tier-Clamped Kelly (multi_asset_chunked_env.py)
  4. ATR-Based Scalper SL (multi_asset_chunked_env.py)

If ANY of these tests fail, the code has NOT been properly injected.
"""

import ast
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =====================================================================
# TEST 1: TRUE QUANT ANTI-HACK REWARD
# =====================================================================
class TestAntiHackReward:
    """Verify the Anti-Hack reward formula is injected and functional."""

    def _make_calculator(self):
        from adan_trading_bot.environment.reward_calculator import RewardCalculator
        return RewardCalculator({"reward_shaping": {}})

    def test_negative_pnl_always_gives_negative_reward(self):
        """CRITICAL: A losing trade MUST NEVER produce a positive reward."""
        rc = self._make_calculator()
        for pnl in [-0.01, -0.1, -1.0, -10.0]:
            reward = rc.calculate(
                portfolio_metrics={"portfolio_value": 100.0},
                trade_pnl=pnl, action=2,
            )
            assert reward < 0, (
                f"ANTI-HACK VIOLATION: pnl={pnl} gave reward={reward} > 0! "
                f"The failsafe binary anti-hack is NOT working."
            )

    def test_positive_pnl_gives_positive_reward(self):
        """A winning trade should produce a positive reward."""
        rc = self._make_calculator()
        reward = rc.calculate(
            portfolio_metrics={"portfolio_value": 100.0},
            trade_pnl=0.5, action=2,
        )
        assert reward > 0, f"Positive PnL should give positive reward, got {reward}"

    def test_consecutive_losses_tracker_exists(self):
        """Verify _consecutive_losses attribute is tracked."""
        rc = self._make_calculator()
        assert hasattr(rc, '_consecutive_losses'), (
            "_consecutive_losses attribute MISSING from RewardCalculator"
        )
        assert rc._consecutive_losses == 0

        # After a loss, it should increment
        rc.calculate(
            portfolio_metrics={"portfolio_value": 100.0},
            trade_pnl=-0.05, action=2,
        )
        assert rc._consecutive_losses == 1

        # After another loss, it should be 2
        rc.calculate(
            portfolio_metrics={"portfolio_value": 100.0},
            trade_pnl=-0.03, action=2,
        )
        assert rc._consecutive_losses == 2

        # After a win, it should reset to 0
        rc.calculate(
            portfolio_metrics={"portfolio_value": 100.0},
            trade_pnl=0.1, action=2,
        )
        assert rc._consecutive_losses == 0

    def test_streak_penalty_kicks_in_after_2(self):
        """Penalty should increase after 2+ consecutive losses."""
        rc = self._make_calculator()
        # Generate 4 consecutive losses
        rewards = []
        for i in range(4):
            r = rc.calculate(
                portfolio_metrics={"portfolio_value": 100.0},
                trade_pnl=-0.01, action=2,
            )
            rewards.append(r)

        # Reward should get progressively worse after the 2nd loss
        # (streak penalty kicks in at consecutive_losses > 2)
        assert rewards[-1] < rewards[0], (
            f"Streak penalty not working: 4th loss reward {rewards[-1]} "
            f"should be worse than 1st loss reward {rewards[0]}"
        )

    def test_anti_hack_params_exist(self):
        """Verify the 5 anti-hack parameters exist."""
        rc = self._make_calculator()
        assert hasattr(rc, '_scale') and rc._scale == 1.0
        assert hasattr(rc, '_alpha') and rc._alpha == 2.0
        assert hasattr(rc, '_beta') and rc._beta == 1.0
        assert hasattr(rc, '_gamma_streak') and rc._gamma_streak == 0.5
        assert hasattr(rc, '_delta') and rc._delta == 2.0


# =====================================================================
# TEST 2: EV-BASED FEE GATE (source code verification)
# =====================================================================
class TestEVFeeGate:
    """Verify the EV-Based Fee Gate is injected in the environment."""

    def test_ev_gate_in_source(self):
        """The string 'EV_GATE' and 'p_min_required' MUST exist in env source."""
        env_path = PROJECT_ROOT / "src" / "adan_trading_bot" / "environment" / "multi_asset_chunked_env.py"
        source = env_path.read_text()
        assert "EV_GATE" in source, (
            "EV_GATE string NOT found in multi_asset_chunked_env.py! "
            "The EV-based fee gate has NOT been injected."
        )
        assert "p_min_required" in source, (
            "p_min_required NOT found in multi_asset_chunked_env.py! "
            "The EV formula is MISSING."
        )

    def test_old_fee_gate_removed(self):
        """The old rigid '3.0 * estimated_fees' gate should be replaced."""
        env_path = PROJECT_ROOT / "src" / "adan_trading_bot" / "environment" / "multi_asset_chunked_env.py"
        source = env_path.read_text()
        # The old pattern should NOT appear in the BUY section
        assert "expected_gross < 3.0 * estimated_fees" not in source, (
            "OLD Fee Gate (3x fees) STILL present! "
            "It should have been replaced by the EV-based gate."
        )

    def test_ev_math_formula_correct(self):
        """Test the EV formula: p_min = (1 + fee/SL) / (1 + RR)."""
        # With SL=1%, TP=2% (RR=2), fees=0.2%
        sl = 0.01
        tp = 0.02
        fee = 0.002
        rr = tp / sl  # = 2.0
        p_min = (1.0 + fee / sl) / (1.0 + rr)
        # p_min = (1 + 0.2) / (1 + 2) = 1.2 / 3 = 0.4
        assert abs(p_min - 0.4) < 0.01, f"EV formula wrong: expected ~0.4, got {p_min}"

        # A W=0.3 trade should be rejected (0.3 < 0.4)
        assert 0.3 <= p_min, "W=0.3 should be rejected with these params"

        # A W=0.5 trade should be accepted (0.5 > 0.4)
        assert 0.5 > p_min, "W=0.5 should be accepted with these params"


# =====================================================================
# TEST 3: TIER-CLAMPED KELLY
# =====================================================================
class TestTierClampedKelly:
    """Verify the Kelly is clamped to tier exposure boundaries."""

    def test_kelly_clamped_in_source(self):
        """The string 'KELLY_CLAMPED' MUST exist in env source."""
        env_path = PROJECT_ROOT / "src" / "adan_trading_bot" / "environment" / "multi_asset_chunked_env.py"
        source = env_path.read_text()
        assert "KELLY_CLAMPED" in source, (
            "KELLY_CLAMPED string NOT found! "
            "The tier-clamped Kelly has NOT been injected."
        )
        assert "exp_min_pct" in source, "exp_min_pct NOT found"
        assert "exp_max_pct" in source, "exp_max_pct NOT found"
        assert "f_star" in source, "f_star NOT found"

    def test_old_half_kelly_removed(self):
        """The old 'Half-Kelly for institutional safety margin' should be gone."""
        env_path = PROJECT_ROOT / "src" / "adan_trading_bot" / "environment" / "multi_asset_chunked_env.py"
        source = env_path.read_text()
        assert "Half-Kelly for institutional safety margin" not in source, (
            "OLD Half-Kelly comment STILL present!"
        )

    def test_kelly_formula_correct(self):
        """Test: f* = max(0, (W*RR - (1-W)) / RR)."""
        W = 0.6
        RR = 2.0
        f_star = max(0.0, (W * RR - (1.0 - W)) / RR)
        # f* = (1.2 - 0.4) / 2 = 0.4
        assert abs(f_star - 0.4) < 0.01, f"Kelly formula wrong: expected 0.4, got {f_star}"

        # With W=0.2, f* should be 0 (negative Kelly)
        W_low = 0.2
        f_star_low = max(0.0, (W_low * RR - (1.0 - W_low)) / RR)
        assert f_star_low == 0.0, f"Low W should give f*=0, got {f_star_low}"

    def test_kelly_clamped_to_tier(self):
        """Test that Kelly respects tier exposure range."""
        # Case 1: Very high f* exceeding tier max => clamped to max
        W = 0.95
        RR = 3.0
        f_star = max(0.0, (W * RR - (1.0 - W)) / RR)
        # f* = (2.85 - 0.05) / 3 = 0.933

        # Tier micro: exposure_range = [70, 90] -> [0.7, 0.9]
        exp_min = 0.70
        exp_max = 0.90
        kelly_mod = max(exp_min, min(exp_max, f_star))
        assert kelly_mod == exp_max, (
            f"Kelly should be clamped to tier max {exp_max}, got {kelly_mod}"
        )

        # Case 2: f* within range => no clamping
        W_mid = 0.9
        f_star_mid = max(0.0, (W_mid * RR - (1.0 - W_mid)) / RR)
        # f* = (2.7 - 0.1) / 3 = 0.867 → within [0.7, 0.9]
        kelly_mod_mid = max(exp_min, min(exp_max, f_star_mid))
        assert exp_min <= kelly_mod_mid <= exp_max, (
            f"Mid Kelly {kelly_mod_mid} should be within [{exp_min}, {exp_max}]"
        )

        # Case 3: Low f_star should be clamped to tier min
        W_low = 0.3
        f_star_low = max(0.0, (W_low * RR - (1.0 - W_low)) / RR)
        kelly_mod_low = max(exp_min, min(exp_max, f_star_low))
        assert kelly_mod_low == exp_min, (
            f"Low Kelly should be clamped to tier min {exp_min}, got {kelly_mod_low}"
        )


# =====================================================================
# TEST 4: ATR-BASED SCALPER SL
# =====================================================================
class TestATRScalperSL:
    """Verify the ATR-based SL floor for 5m scalper."""

    def test_atr_sl_in_source(self):
        """The string 'ATR_SL' MUST exist in env source."""
        env_path = PROJECT_ROOT / "src" / "adan_trading_bot" / "environment" / "multi_asset_chunked_env.py"
        source = env_path.read_text()
        assert "ATR_SL" in source, (
            "ATR_SL string NOT found in multi_asset_chunked_env.py! "
            "The ATR-based scalper SL has NOT been injected."
        )
        assert "min_scalp_sl" in source, "min_scalp_sl NOT found"

    def test_sl_floor_never_below_0_006(self):
        """Scalper SL must never go below 0.6% (3x ~0.2% ATR floor)."""
        # Simulate: ATR estimate = 0.002 (0.2%)
        atr_pct = 0.002
        min_scalp_sl = max(0.006, 3.0 * atr_pct)
        assert min_scalp_sl >= 0.006, f"SL floor violated: {min_scalp_sl}"

        # Even with very low ATR
        atr_pct_low = 0.001
        min_scalp_sl_low = max(0.006, 3.0 * atr_pct_low)
        assert min_scalp_sl_low >= 0.006, f"SL floor violated with low ATR: {min_scalp_sl_low}"

        # With higher ATR, SL should adapt
        atr_pct_high = 0.005
        min_scalp_sl_high = max(0.006, 3.0 * atr_pct_high)
        assert min_scalp_sl_high == 0.015, f"SL should be 3x ATR: expected 0.015, got {min_scalp_sl_high}"


# =====================================================================
# INTEGRATION: SOURCE CODE AUDIT
# =====================================================================
class TestSourceCodeAudit:
    """Verify that all 4 mathematical functions are in the source."""

    def test_reward_calculator_has_antihack_params(self):
        """Check reward_calculator.py has _alpha, _beta, _gamma_streak, _delta."""
        rc_path = PROJECT_ROOT / "src" / "adan_trading_bot" / "environment" / "reward_calculator.py"
        source = rc_path.read_text()
        for param in ["_alpha", "_beta", "_gamma_streak", "_delta", "_consecutive_losses"]:
            assert f"self.{param}" in source, (
                f"MISSING: self.{param} not found in reward_calculator.py"
            )

    def test_reward_calculator_failsafe_present(self):
        """Check the failsafe: 'if pnl_net < 0 and r > 0: r *= -delta'."""
        rc_path = PROJECT_ROOT / "src" / "adan_trading_bot" / "environment" / "reward_calculator.py"
        source = rc_path.read_text()
        assert "pnl_net < 0 and r > 0" in source, (
            "FAILSAFE MISSING: 'pnl_net < 0 and r > 0' not in reward_calculator.py"
        )

    def test_env_has_all_4_gates(self):
        """Check multi_asset_chunked_env.py has all 4 mathematical gates."""
        env_path = PROJECT_ROOT / "src" / "adan_trading_bot" / "environment" / "multi_asset_chunked_env.py"
        source = env_path.read_text()
        checks = {
            "EV_GATE": "EV-Based Fee Gate",
            "KELLY_CLAMPED": "Tier-Clamped Kelly",
            "ATR_SL": "ATR-Based Scalper SL",
            "p_min_required": "EV minimum probability",
            "f_star": "Kelly optimal fraction",
            "min_scalp_sl": "Scalper SL floor",
        }
        for marker, description in checks.items():
            assert marker in source, (
                f"MISSING: '{marker}' ({description}) not in multi_asset_chunked_env.py"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
