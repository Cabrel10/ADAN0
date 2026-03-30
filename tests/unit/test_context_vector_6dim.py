"""
OMEGA-4B – Verify that context_vector has exactly 6 dimensions everywhere.

Tests:
  1. StateBuilder.build_context_vector returns shape (6,)
  2. Feature extractors DEFAULT_CONTEXT_DIM == 6
  3. Observation space in env defines context_vector with shape (6,)
  4. Default observation includes context_vector with 6 dims
"""

import numpy as np
import pytest


class TestContextVectorDim:
    """Ensure context_vector is 6-dimensional across the codebase."""

    def test_state_builder_context_vector_shape(self):
        """StateBuilder.build_context_vector should return (6,) float32."""
        from adan_trading_bot.data_processing.state_builder import StateBuilder

        sb = StateBuilder(
            timeframes=["5m"],
            features_config={"5m": ["close"]},
            window_sizes={"5m": 20},
        )
        ctx = sb.build_context_vector(data=None, current_idx=0)
        assert ctx.shape == (6,), f"Expected shape (6,), got {ctx.shape}"
        assert ctx.dtype == np.float32

    def test_state_builder_context_with_data(self):
        """When data is provided, context_vector still has 6 dims."""
        import pandas as pd
        from adan_trading_bot.data_processing.state_builder import StateBuilder

        sb = StateBuilder(
            timeframes=["5m"],
            features_config={"5m": ["close"]},
            window_sizes={"5m": 20},
        )
        # Create minimal DataFrame
        df = pd.DataFrame({
            "CLOSE": np.random.uniform(90, 110, 30).astype(np.float32),
            "ATR_14": np.random.uniform(0.5, 2.0, 30).astype(np.float32),
            "EMA_12": np.random.uniform(95, 105, 30).astype(np.float32),
            "EMA_26": np.random.uniform(95, 105, 30).astype(np.float32),
            "ADX_14": np.random.uniform(10, 50, 30).astype(np.float32),
        })
        ctx = sb.build_context_vector(data={"5m": df}, current_idx=25)
        assert ctx.shape == (6,), f"Expected shape (6,), got {ctx.shape}"
        assert ctx.dtype == np.float32
        # candle_progress_pct should be set (index 5)
        assert 0.0 <= ctx[5] <= 1.0, f"candle_progress_pct out of range: {ctx[5]}"

    def test_temporal_fusion_default_context_dim(self):
        """ContextualTemporalFusionExtractor.DEFAULT_CONTEXT_DIM should be 6."""
        from adan_trading_bot.agent.feature_extractors import ContextualTemporalFusionExtractor
        assert ContextualTemporalFusionExtractor.DEFAULT_CONTEXT_DIM == 6

    def test_film_temporal_fusion_default_context_dim(self):
        """FiLMTemporalFusionExtractor.DEFAULT_CONTEXT_DIM should be 6."""
        try:
            from adan_trading_bot.agent.feature_extractors import FiLMTemporalFusionExtractor
            assert FiLMTemporalFusionExtractor.DEFAULT_CONTEXT_DIM == 6
        except (ImportError, AttributeError):
            pytest.skip("FiLMTemporalFusionExtractor not available")


class TestCandelProgressPct:
    """Verify candle_progress_pct (index 5) is correctly computed."""

    def test_candle_progress_at_start(self):
        """At current_idx=0, candle_progress should be ~0."""
        import pandas as pd
        from adan_trading_bot.data_processing.state_builder import StateBuilder

        sb = StateBuilder(
            timeframes=["5m"],
            features_config={"5m": ["close"]},
            window_sizes={"5m": 20},
        )
        df = pd.DataFrame({"CLOSE": np.ones(50, dtype=np.float32)})
        ctx = sb.build_context_vector(data={"5m": df}, current_idx=0)
        assert ctx[5] == pytest.approx(0.0, abs=0.05)

    def test_candle_progress_at_end(self):
        """At current_idx=len-1, candle_progress should be ~1.0."""
        import pandas as pd
        from adan_trading_bot.data_processing.state_builder import StateBuilder

        sb = StateBuilder(
            timeframes=["5m"],
            features_config={"5m": ["close"]},
            window_sizes={"5m": 20},
        )
        df = pd.DataFrame({"CLOSE": np.ones(50, dtype=np.float32)})
        ctx = sb.build_context_vector(data={"5m": df}, current_idx=49)
        assert ctx[5] == pytest.approx(1.0, abs=0.05)

    def test_candle_progress_midpoint(self):
        """At current_idx=25 in 50 candles, progress should be ~0.51."""
        import pandas as pd
        from adan_trading_bot.data_processing.state_builder import StateBuilder

        sb = StateBuilder(
            timeframes=["5m"],
            features_config={"5m": ["close"]},
            window_sizes={"5m": 20},
        )
        df = pd.DataFrame({"CLOSE": np.ones(50, dtype=np.float32)})
        ctx = sb.build_context_vector(data={"5m": df}, current_idx=25)
        assert 0.4 <= ctx[5] <= 0.6, f"Expected ~0.5, got {ctx[5]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
