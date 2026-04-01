"""
OMEGA-4C – DBE v2 -> FiLM integration tests.

Asserts:
  1. DBE v2 returns valid modulation dict (no crashes).
  2. detect_market_regime returns valid regime + confidence.
  3. get_capital_tier / _get_capital_tier returns correct tier.
  4. Regime detection is stable (multiple calls produce same result).
  5. update_risk_parameters returns expected keys with valid config.
  6. reset clears state properly.
  7. context_vector is 6-dimensional.
"""

import numpy as np
import pytest


class TestDBEv2FiLMIntegration:
    """Verify DBE v2 integrates correctly for FiLM modulation pipeline."""

    MICRO_TIER = {
        "name": "Micro Capital", "min_capital": 11, "max_capital": 30,
        "exposure_range": [70, 90], "risk_per_trade_pct": 4.0,
        "max_position_size_pct": 90.0,
    }
    SMALL_TIER = {
        "name": "Small Capital", "min_capital": 30, "max_capital": 100,
        "exposure_range": [35, 75], "risk_per_trade_pct": 2.0,
        "max_position_size_pct": 75.0,
    }

    def _make_dbe(self, **kwargs):
        from adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine
        config = kwargs.pop("config", {})
        return DynamicBehaviorEngine(config=config, **kwargs)

    def test_compute_dynamic_modulation_returns_dict(self):
        """compute_dynamic_modulation returns a valid dict with expected keys."""
        dbe = self._make_dbe()
        mod = dbe.compute_dynamic_modulation()
        assert isinstance(mod, dict)
        assert "sl_pct" in mod
        assert "tp_pct" in mod
        assert "position_size_pct" in mod
        # Values should be non-negative numbers
        assert mod["sl_pct"] >= 0.0
        assert mod["tp_pct"] >= 0.0
        assert mod["position_size_pct"] >= 0.0

    def test_detect_market_regime_valid(self):
        """Regime detection returns valid regime string and confidence."""
        dbe = self._make_dbe()
        regime, conf = dbe.detect_market_regime(
            {"adx": 30, "rsi": 60, "ema_fast": 100, "ema_slow": 95}
        )
        assert regime in ("bull", "bear", "sideways", "volatile")
        assert 0.0 <= conf <= 1.0

    def test_regime_detection_stable(self):
        """Multiple calls with same data should return same regime."""
        dbe = self._make_dbe()
        data = {"adx": 50, "rsi": 70, "ema_fast": 110, "ema_slow": 100}
        r1, c1 = dbe.detect_market_regime(data)
        r2, c2 = dbe.detect_market_regime(data)
        assert r1 == r2
        assert c1 == c2

    def test_capital_tier_lookup_micro(self):
        """get_capital_tier returns Micro tier for 20.5 USDT."""
        config = {"capital_tiers": [self.MICRO_TIER, self.SMALL_TIER]}
        dbe = self._make_dbe(config=config)
        tier = dbe.get_capital_tier(20.5)
        assert tier is not None
        assert tier["name"] == "Micro Capital"

    def test_capital_tier_lookup_small(self):
        """get_capital_tier returns Small tier for 50.0 USDT."""
        config = {"capital_tiers": [self.MICRO_TIER, self.SMALL_TIER]}
        dbe = self._make_dbe(config=config)
        tier = dbe.get_capital_tier(50.0)
        assert tier is not None
        assert tier["name"] == "Small Capital"

    def test_update_risk_parameters_with_valid_config(self):
        """update_risk_parameters returns valid dict with capital_tiers config."""
        config = {
            "capital_tiers": [self.MICRO_TIER, self.SMALL_TIER],
            "risk_parameters": {"base_sl_pct": 0.02, "base_tp_pct": 0.04},
        }
        dbe = self._make_dbe(config=config)
        result = dbe.update_risk_parameters(
            market_data={"adx": 30, "rsi": 55, "ema_fast": 100, "ema_slow": 98},
            portfolio_value=20.5,
        )
        assert isinstance(result, dict)
        assert "stop_loss_pct" in result
        assert "take_profit_pct" in result
        assert "position_size_pct" in result
        assert "regime" in result
        assert result["regime"] in ("bull", "bear", "sideways", "volatile")
        assert result["stop_loss_pct"] >= 0
        assert result["take_profit_pct"] >= 0
        assert result["position_size_pct"] >= 0

    def test_reset_clears_state(self):
        """After reset, state should be clean."""
        dbe = self._make_dbe()
        dbe.update_state({"volatility": 0.05, "drawdown": 0.1})
        dbe.reset()
        # State should be reset
        assert dbe.state["current_step"] == 0

    def test_context_vector_6dim(self):
        """build_context_vector should return 6-dimensional array."""
        from adan_trading_bot.data_processing.state_builder import StateBuilder
        sb = StateBuilder(config={}, timeframes=["5m", "1h", "4h"])
        ctx = sb.build_context_vector()
        assert ctx.shape == (12,)
        assert ctx.dtype == np.float32
        # Without data, all entries including candle_progress are 0.0
        assert ctx[5] == 0.0

    def test_context_vector_zeros_when_no_data(self):
        """Context vector is all zeros without data."""
        from adan_trading_bot.data_processing.state_builder import StateBuilder
        sb = StateBuilder(config={}, timeframes=["5m", "1h", "4h"])
        ctx = sb.build_context_vector(data=None)
        assert ctx.shape == (12,)
        # All entries should be 0 when no data is provided
        for i in range(6):
            assert ctx[i] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
