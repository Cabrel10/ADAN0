"""
OMEGA-4C – DBE -> FiLM sensor-only integration tests.

Asserts:
  1. Sensor-only DBE returns neutral multipliers (sl=0, tp=0, pos_size=0).
  2. get_sensor_snapshot returns all expected keys.
  3. Sensor snapshot dimensions are correct for FiLM input.
  4. Regime detection is pure (no side-effects on sizing).
"""

import numpy as np
import pytest


class TestDBESensorOnlyIntegration:
    """Verify DBE is purely a sensor and does not override agent decisions."""

    def _make_dbe(self, **kwargs):
        from adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine
        config = kwargs.pop("config", {})
        return DynamicBehaviorEngine(config=config, **kwargs)

    def test_compute_dynamic_modulation_neutral(self):
        """compute_dynamic_modulation must return zeros for sl/tp/position."""
        dbe = self._make_dbe()
        mod = dbe.compute_dynamic_modulation()
        assert mod["sl_pct"] == 0.0, f"sl_pct should be 0, got {mod['sl_pct']}"
        assert mod["tp_pct"] == 0.0, f"tp_pct should be 0, got {mod['tp_pct']}"
        assert mod["position_size_pct"] == 0.0, f"pos_size should be 0, got {mod['position_size_pct']}"
        assert mod["reward_boost"] == 1.0, "reward_boost should be neutral (1.0)"

    def test_update_risk_parameters_neutral(self):
        """update_risk_parameters must return zeros for sizing params."""
        dbe = self._make_dbe()
        result = dbe.update_risk_parameters(
            market_data={"adx": 30, "rsi": 55, "ema_fast": 100, "ema_slow": 98},
            portfolio_value=20.5,
        )
        assert result["stop_loss_pct"] == 0.0
        assert result["take_profit_pct"] == 0.0
        assert result["position_size_pct"] == 0.0
        assert result["regime"] in ("bull", "bear", "sideways", "volatile")

    def test_calculate_trade_parameters_feasible(self):
        """calculate_trade_parameters must always return feasible=True with zeros."""
        dbe = self._make_dbe()
        params = dbe.calculate_trade_parameters(capital=20.5, price=100.0)
        assert params["feasible"] is True
        assert params["position_size_pct"] == 0.0
        assert params["sl_pct"] == 0.0
        assert params["tp_pct"] == 0.0

    def test_sensor_snapshot_keys(self):
        """get_sensor_snapshot must return all keys needed by FiLM modulation."""
        dbe = self._make_dbe()
        snap = dbe.get_sensor_snapshot(
            market_data={"adx_14": 30, "volatility": 0.02, "trend_strength": 25},
            portfolio_value=20.5,
        )
        expected_keys = {
            "regime_bull", "regime_bear", "regime_sideways", "regime_volatile",
            "regime_confidence", "volatility_current", "volatility_avg",
            "trend_strength", "drawdown", "tier_index_norm",
        }
        assert expected_keys.issubset(set(snap.keys())), (
            f"Missing keys: {expected_keys - set(snap.keys())}"
        )

    def test_sensor_snapshot_values_bounded(self):
        """All sensor snapshot values should be in reasonable ranges."""
        dbe = self._make_dbe()
        snap = dbe.get_sensor_snapshot(
            market_data={"adx_14": 50, "volatility": 0.05, "trend_strength": 40},
            portfolio_value=200.0,
        )
        for key, val in snap.items():
            assert isinstance(val, float), f"{key} should be float, got {type(val)}"
            assert np.isfinite(val), f"{key} should be finite, got {val}"

    def test_regime_detection_does_not_affect_sizing(self):
        """Calling detect_market_regime should NOT change any sizing attribute."""
        dbe = self._make_dbe()
        before = dbe.compute_dynamic_modulation()
        dbe.detect_market_regime({"adx": 60, "rsi": 75, "ema_fast": 110, "ema_slow": 100})
        after = dbe.compute_dynamic_modulation()
        assert before["sl_pct"] == after["sl_pct"]
        assert before["tp_pct"] == after["tp_pct"]
        assert before["position_size_pct"] == after["position_size_pct"]

    def test_capital_tier_lookup(self):
        """get_capital_tier returns correct tier for given capital."""
        from adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine

        config = {
            "capital_tiers": [
                {"name": "Micro Capital", "min_capital": 11, "max_capital": 30,
                 "exposure_range": [70, 90], "risk_per_trade_pct": 4.0},
                {"name": "Small Capital", "min_capital": 30, "max_capital": 100,
                 "exposure_range": [35, 75], "risk_per_trade_pct": 2.0},
            ]
        }
        dbe = DynamicBehaviorEngine(config=config)
        tier = dbe.get_capital_tier(20.5)
        assert tier is not None
        assert tier["name"] == "Micro Capital"

        tier2 = dbe.get_capital_tier(50.0)
        assert tier2["name"] == "Small Capital"

    def test_reset_clears_state(self):
        """After reset, sensor state should be clean."""
        dbe = self._make_dbe()
        dbe.update_state({"volatility": 0.05, "drawdown": 0.1})
        dbe.reset()
        assert dbe.state["volatility"] == 0.0
        assert dbe.state["drawdown"] == 0.0
        assert dbe.state["current_step"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
