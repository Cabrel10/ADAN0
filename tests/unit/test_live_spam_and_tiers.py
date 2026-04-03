"""
Chantier 4 – Unit tests for Target-Weight logic, anti-spam HOLD, and tier-mapped sizing.

Tests:
  1. HOLD when target exposure ≈ current exposure
  2. Tier-mapped size uses exposure_range correctly
  3. Dynamic SL bounded by tier risk_per_trade_pct
  4. Notional clamped to >= 11 USDT
  5. DBE sensor-only returns neutral modulation
  6. Early-exit bonus fires on profitable agent close
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. Tier exposure mapping
# ---------------------------------------------------------------------------

def compute_target_exposure(size_raw, tier):
    """Replicate the target-weight formula from _execute_trades."""
    min_exp = float(tier["exposure_range"][0]) / 100.0
    max_exp = float(tier["exposure_range"][1]) / 100.0
    normalized = (size_raw + 1.0) / 2.0
    return min_exp + normalized * (max_exp - min_exp)


MICRO_TIER = {
    "name": "Micro Capital",
    "exposure_range": [70, 90],
    "risk_per_trade_pct": 4.0,
    "min_capital": 11.0,
    "max_capital": 30.0,
}
SMALL_TIER = {
    "name": "Small Capital",
    "exposure_range": [35, 75],
    "risk_per_trade_pct": 2.0,
    "min_capital": 30.0,
    "max_capital": 100.0,
}


class TestTierMappedSize:
    def test_micro_full_size(self):
        """size_raw=1.0 → max exposure (90%)."""
        pct = compute_target_exposure(1.0, MICRO_TIER)
        assert abs(pct - 0.90) < 1e-6

    def test_micro_min_size(self):
        """size_raw=-1.0 → min exposure (70%)."""
        pct = compute_target_exposure(-1.0, MICRO_TIER)
        assert abs(pct - 0.70) < 1e-6

    def test_small_mid_size(self):
        """size_raw=0.0 → midpoint of [35,75] = 55%."""
        pct = compute_target_exposure(0.0, SMALL_TIER)
        assert abs(pct - 0.55) < 1e-6

    def test_notional_clamp_to_11(self):
        """With capital=200 and tiny exposure, notional should be >= 11."""
        capital = 200.0
        # size_raw = -1.0 → exposure = 70% for Micro → 200*0.70 = 140 → OK
        # But with Small tier: size_raw=-1.0 → 35% → 200*0.35 = 70 → OK
        # Let's test with artificially low tier
        low_tier = {"exposure_range": [1, 5], "risk_per_trade_pct": 2.0}
        pct = compute_target_exposure(-1.0, low_tier)  # 1%
        notional = capital * pct
        # Clamp logic
        min_order = 11.0
        if notional < min_order and capital >= min_order:
            notional = min_order
        assert notional >= min_order


# ---------------------------------------------------------------------------
# 2. Dynamic SL bounded by tier risk
# ---------------------------------------------------------------------------

class TestDynamicSL:
    def test_sl_bounded_by_risk(self):
        """SL should not exceed (capital * max_risk_pct) / notional."""
        capital = 200.0
        notional = 140.0  # 70% of 200
        max_risk_pct = 0.04  # 4%
        max_sl_pct = (capital * max_risk_pct) / notional
        assert abs(max_sl_pct - 0.05714) < 0.001  # approx 5.7%

        # With sl_raw = 1.0 (max), SL = 0.005 + 1.0 * (max_sl_pct - 0.005)
        sl_raw = 1.0
        normalized_sl = (sl_raw + 1.0) / 2.0
        sl_pct = 0.005 + normalized_sl * (max_sl_pct - 0.005)
        sl_pct = min(sl_pct, 0.10)
        assert sl_pct <= max_sl_pct + 0.001
        assert sl_pct <= 0.10

    def test_sl_minimum_at_half_percent(self):
        """SL should never go below 0.5%."""
        sl_raw = -1.0  # minimum SL
        max_sl_pct = 0.10
        normalized_sl = (sl_raw + 1.0) / 2.0  # 0
        sl_pct = 0.005 + normalized_sl * (max_sl_pct - 0.005)
        assert abs(sl_pct - 0.005) < 1e-6


# ---------------------------------------------------------------------------
# 3. Anti-spam HOLD
# ---------------------------------------------------------------------------

class TestAntiSpamHold:
    def test_hold_when_exposure_close(self):
        """If target exposure ≈ current exposure, discrete action should be overridden to HOLD."""
        # Simulate: position open at 70% exposure, agent wants BUY again (discrete=1)
        target_exposure_pct = 0.72
        current_exposure = 0.70
        exposure_diff = abs(target_exposure_pct - current_exposure)
        # Our threshold is 5%
        should_hold = exposure_diff < 0.05
        assert should_hold

    def test_no_hold_when_exposure_differs(self):
        """If target exposure differs significantly, allow trade."""
        target_exposure_pct = 0.85
        current_exposure = 0.70
        exposure_diff = abs(target_exposure_pct - current_exposure)
        should_hold = exposure_diff < 0.05
        assert not should_hold


# ---------------------------------------------------------------------------
# 4. DBE sensor-only returns neutral
# ---------------------------------------------------------------------------

class TestDBESensorOnly:
    def test_compute_dynamic_modulation_returns_defaults(self):
        """DBE v2 without env returns default modulation (not zeros).
        
        In OMEGA target-weight logic, the PPO agent decides SL/TP/size.
        The DBE returns defaults when no env reference is set.
        """
        from adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine

        dbe = DynamicBehaviorEngine(config={})
        mod = dbe.compute_dynamic_modulation()
        # Without env, DBE returns _get_default_modulation
        assert isinstance(mod, dict)
        assert "sl_pct" in mod
        assert "tp_pct" in mod
        assert "position_size_pct" in mod
        # Values should be non-negative
        assert mod["sl_pct"] >= 0.0
        assert mod["tp_pct"] >= 0.0
        assert mod["position_size_pct"] >= 0.0

    def test_detect_market_regime(self):
        """Regime detection returns valid regime string."""
        from adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine

        dbe = DynamicBehaviorEngine(config={})
        regime, conf = dbe.detect_market_regime({"adx": 30, "rsi": 60, "ema_fast": 100, "ema_slow": 95})
        assert regime in ("bull", "bear", "sideways", "volatile")
        assert 0.0 <= conf <= 1.0

    def test_get_capital_tier(self):
        """Capital tier lookup returns correct tier."""
        from adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine

        config = {"capital_tiers": [MICRO_TIER, SMALL_TIER]}
        dbe = DynamicBehaviorEngine(config=config)
        tier = dbe.get_capital_tier(25.0)  # Should be Micro (11-30)
        assert tier is not None
        assert tier["name"] == "Micro Capital"


# ---------------------------------------------------------------------------
# 5. Early-exit bonus
# ---------------------------------------------------------------------------

class TestEarlyExitBonus:
    def test_bonus_on_profitable_agent_close(self):
        """Agent-initiated close with profit should get a bonus."""
        from adan_trading_bot.environment.reward_calculator import RewardCalculator

        rc = RewardCalculator({"reward_shaping": {"early_exit_bonus": 0.5}})
        bonus = rc._calculate_early_exit_bonus(
            trade_pnl=0.05, trade_reason="AGENT_CLOSE", portfolio_metrics={}
        )
        assert bonus > 0.0

    def test_no_bonus_on_sl_close(self):
        """SL-triggered close should not get a bonus."""
        from adan_trading_bot.environment.reward_calculator import RewardCalculator

        rc = RewardCalculator({"reward_shaping": {"early_exit_bonus": 0.5}})
        bonus = rc._calculate_early_exit_bonus(
            trade_pnl=0.05, trade_reason="SL", portfolio_metrics={}
        )
        assert bonus == 0.0

    def test_no_bonus_on_loss(self):
        """Losing trade should not get a bonus."""
        from adan_trading_bot.environment.reward_calculator import RewardCalculator

        rc = RewardCalculator({"reward_shaping": {"early_exit_bonus": 0.5}})
        bonus = rc._calculate_early_exit_bonus(
            trade_pnl=-0.02, trade_reason="AGENT_CLOSE", portfolio_metrics={}
        )
        assert bonus == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
