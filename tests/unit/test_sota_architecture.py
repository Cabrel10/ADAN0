"""
SOTA 2025 Architecture Validation Tests
========================================
Proves mathematical correctness of:
  1. Cyclical Time Encoding (sin/cos continuity)
  2. HMM regime probabilities (sum to 1)
  3. Symlog reward transformation (sign-preserving, bounded compression)
  4. World Model forward predictor (output shape)
  5. Context vector dimension = 12
"""
import math
import pytest
import numpy as np
import pandas as pd

# --------------- Module imports ---------------
from adan_trading_bot.data_processing.state_builder import StateBuilder
from adan_trading_bot.environment.dynamic_behavior_engine import (
    DynamicBehaviorEngine,
    N_HMM_STATES,
    HMM_AVAILABLE,
)
from adan_trading_bot.environment.reward_calculator import symlog


# =====================================================================
# 1. CYCLICAL TIME ENCODING
# =====================================================================

class TestCyclicalTimeEncoding:
    """Prove that the temporal encoding is truly circular."""

    @pytest.fixture
    def state_builder(self):
        return StateBuilder(
            features_config={"5m": ["CLOSE", "ATR_14", "EMA_12", "EMA_26", "ADX_14"]},
            window_sizes={"5m": 20},
        )

    def _make_data(self, timestamp):
        """Create minimal data with a specific timestamp."""
        df = pd.DataFrame({
            "CLOSE": [100.0],
            "ATR_14": [1.0],
            "EMA_12": [100.0],
            "EMA_26": [100.0],
            "ADX_14": [25.0],
        }, index=pd.DatetimeIndex([timestamp]))
        return {"5m": df}

    def test_context_vector_is_12_dim(self, state_builder):
        """Context vector must be 12-dimensional after SOTA upgrade."""
        ctx = state_builder.build_context_vector()
        assert ctx.shape == (12,), f"Expected shape (12,), got {ctx.shape}"
        assert ctx.dtype == np.float32

    def test_sunday_23h_close_to_monday_00h(self, state_builder):
        """Sunday 23:50 must be metrically close to Monday 00:10 in the
        HOUR dimension.  The day-of-week dimension correctly captures the
        Sunday->Monday transition as a larger jump.

        With sin/cos hour encoding, the hour distance is tiny (~0.06).
        """
        ts_sunday = pd.Timestamp("2025-03-30 23:50:00")   # Sunday 23:50
        ts_monday = pd.Timestamp("2025-03-31 00:10:00")   # Monday 00:10

        ctx_sun = state_builder.build_context_vector(
            data=self._make_data(ts_sunday), current_idx=0
        )
        ctx_mon = state_builder.build_context_vector(
            data=self._make_data(ts_monday), current_idx=0
        )

        # Hour components only [6:8] (sin_hour, cos_hour)
        hour_dist = float(np.linalg.norm(ctx_sun[6:8] - ctx_mon[6:8]))
        assert hour_dist < 0.2, (
            f"Hour cyclical distance Sun 23:50 vs Mon 00:10 = {hour_dist:.4f}, "
            f"expected < 0.2 for proper circular encoding"
        )

        # Full 6-dim cyclical distance can be larger due to DOW transition
        full_dist = float(np.linalg.norm(ctx_sun[6:12] - ctx_mon[6:12]))
        assert full_dist < 1.5, (
            f"Full cyclical distance = {full_dist:.4f}, expected < 1.5"
        )

    def test_noon_vs_midnight_far_apart(self, state_builder):
        """Noon and midnight should be maximally distant in hour encoding."""
        ts_midnight = pd.Timestamp("2025-03-31 00:00:00")
        ts_noon = pd.Timestamp("2025-03-31 12:00:00")

        ctx_mid = state_builder.build_context_vector(
            data=self._make_data(ts_midnight), current_idx=0
        )
        ctx_noon = state_builder.build_context_vector(
            data=self._make_data(ts_noon), current_idx=0
        )

        # sin/cos hour: midnight -> (0, 1), noon -> (0, -1)
        # Distance in hour components [6:8] should be ~2.0
        hour_dist = float(np.linalg.norm(ctx_mid[6:8] - ctx_noon[6:8]))
        assert hour_dist > 1.5, (
            f"Hour distance noon vs midnight = {hour_dist:.4f}, expected > 1.5"
        )

    def test_sin_cos_bounded(self, state_builder):
        """All cyclical components must be in [-1, 1]."""
        ts = pd.Timestamp("2025-06-15 14:37:00")  # arbitrary time
        ctx = state_builder.build_context_vector(
            data=self._make_data(ts), current_idx=0
        )
        cycl = ctx[6:12]
        assert np.all(cycl >= -1.0) and np.all(cycl <= 1.0), (
            f"Cyclical values out of [-1,1]: {cycl}"
        )

    def test_no_data_returns_zeros(self, state_builder):
        """Without data, cyclical components should be 0."""
        ctx = state_builder.build_context_vector()
        assert np.allclose(ctx[6:12], 0.0), (
            f"Expected zeros for cyclical when no data, got {ctx[6:12]}"
        )


# =====================================================================
# 2. HMM REGIME PROBABILITIES
# =====================================================================

class TestHMMRegime:
    """Validate HMM-based regime detection."""

    @pytest.fixture
    def dbe(self):
        config = {
            "capital_tiers": [
                {"name": "Micro Capital", "min_capital": 11, "max_capital": 30,
                 "exposure_range": [70, 90], "risk_per_trade_pct": 4.0,
                 "max_position_size_pct": 90},
            ],
        }
        return DynamicBehaviorEngine(config=config, worker_id=0)

    def test_regime_probabilities_sum_to_one(self, dbe):
        """The HMM probability vector must always sum to 1."""
        # Feed some data
        for i in range(50):
            price = 100.0 + np.random.randn() * 2
            prev_price = 100.0 + np.random.randn() * 2
            probs = dbe.get_regime_probabilities({
                "close": price,
                "prev_close": prev_price,
                "volatility": abs(np.random.randn()) * 0.01,
            })
            assert probs.shape == (N_HMM_STATES,), f"Wrong shape: {probs.shape}"
            assert abs(probs.sum() - 1.0) < 1e-5, (
                f"Probabilities sum to {probs.sum():.6f}, expected ~1.0"
            )

    def test_regime_probs_non_negative(self, dbe):
        """All probabilities must be >= 0."""
        for _ in range(20):
            probs = dbe.get_regime_probabilities({
                "close": 100 + np.random.randn(),
                "prev_close": 100.0,
                "volatility": 0.01,
            })
            assert np.all(probs >= 0), f"Negative probabilities: {probs}"

    def test_detect_regime_returns_valid_label(self, dbe):
        """detect_market_regime must return a valid label from the set."""
        for _ in range(30):
            regime, conf = dbe.detect_market_regime({
                "close": 100 + np.random.randn() * 5,
                "prev_close": 100.0,
                "volatility": 0.02,
                "adx": 30,
                "ema_fast": 101,
                "ema_slow": 100,
            })
            assert regime in {"bull", "bear", "sideways"}, (
                f"Invalid regime: {regime}"
            )
            assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"

    @pytest.mark.skipif(not HMM_AVAILABLE, reason="hmmlearn not installed")
    def test_hmm_fits_after_enough_data(self, dbe):
        """After 30+ observations, HMM should be fitted."""
        for i in range(40):
            dbe.get_regime_probabilities({
                "close": 100 + i * 0.1,
                "prev_close": 100 + (i - 1) * 0.1,
                "volatility": 0.01,
            })
        assert dbe._hmm_fitted, "HMM should be fitted after 40 observations"


# =====================================================================
# 3. SYMLOG REWARD TRANSFORM
# =====================================================================

class TestSymlogTransform:
    """Validate the DreamerV3-style symlog transform."""

    def test_symlog_zero(self):
        assert symlog(0.0) == 0.0

    def test_symlog_preserves_sign(self):
        assert symlog(5.0) > 0
        assert symlog(-5.0) < 0

    def test_symlog_symmetry(self):
        """symlog(-x) == -symlog(x)"""
        for x in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            assert abs(symlog(-x) + symlog(x)) < 1e-10

    def test_symlog_compresses_extremes(self):
        """A 100x input should NOT produce 100x output."""
        ratio = symlog(1000.0) / symlog(10.0)
        assert ratio < 5.0, (
            f"Symlog compression ratio = {ratio:.2f}, expected < 5.0"
        )

    def test_symlog_vs_formula(self):
        """Verify against the mathematical definition."""
        for x in [-100, -1, -0.01, 0, 0.01, 1, 100]:
            expected = math.copysign(1, x) * math.log1p(abs(x)) if x != 0 else 0
            assert abs(symlog(x) - expected) < 1e-10, (
                f"symlog({x}) = {symlog(x)}, expected {expected}"
            )


# =====================================================================
# 4. WORLD MODEL FORWARD PREDICTOR
# =====================================================================

class TestWorldModelPredictor:
    """Validate the auxiliary forward prediction head."""

    @pytest.fixture
    def extractor(self):
        """Create a ContextualTemporalFusionExtractor with a mock obs space."""
        import gym
        from gym import spaces
        obs_space = spaces.Dict({
            "5m": spaces.Box(low=-np.inf, high=np.inf, shape=(20, 5), dtype=np.float32),
            "1h": spaces.Box(low=-np.inf, high=np.inf, shape=(10, 5), dtype=np.float32),
            "4h": spaces.Box(low=-np.inf, high=np.inf, shape=(5, 5), dtype=np.float32),
            "portfolio_state": spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32),
            "context_vector": spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32),
        })
        from adan_trading_bot.agent.feature_extractors import ContextualTemporalFusionExtractor
        return ContextualTemporalFusionExtractor(obs_space, features_dim=128)

    def test_forward_predictor_exists(self, extractor):
        """The auxiliary head must exist."""
        assert hasattr(extractor, "forward_predictor"), (
            "Missing forward_predictor head"
        )

    def test_forward_predictor_output_shape(self, extractor):
        """Forward predictor should output a scalar per batch item."""
        import torch as th
        batch = {
            "5m": th.randn(4, 20, 5),
            "1h": th.randn(4, 10, 5),
            "4h": th.randn(4, 5, 5),
            "portfolio_state": th.randn(4, 17),
            "context_vector": th.randn(4, 12),
        }
        features = extractor(batch)
        pred = extractor._last_aux_prediction
        assert pred is not None, "Auxiliary prediction is None after forward pass"
        assert pred.shape == (4, 1), f"Expected (4,1), got {pred.shape}"

    def test_compute_auxiliary_loss(self, extractor):
        """Auxiliary MSE loss should be a scalar >= 0."""
        import torch as th
        batch = {
            "5m": th.randn(4, 20, 5),
            "1h": th.randn(4, 10, 5),
            "4h": th.randn(4, 5, 5),
            "portfolio_state": th.randn(4, 17),
            "context_vector": th.randn(4, 12),
        }
        extractor(batch)
        target = th.randn(4, 1)
        loss = extractor.compute_auxiliary_loss(target)
        assert loss.dim() == 0, f"Loss should be scalar, got dim={loss.dim()}"
        assert loss.item() >= 0, f"MSE loss should be >= 0, got {loss.item()}"


# =====================================================================
# 5. CONTEXT VECTOR DIM CONSTANT
# =====================================================================

class TestContextDimConsistency:
    """Verify context dim is 12 across all modules."""

    def test_state_builder_context_dim(self):
        assert StateBuilder.CONTEXT_DIM == 12

    def test_feature_extractor_default_dim(self):
        from adan_trading_bot.agent.feature_extractors import (
            ContextualTemporalFusionExtractor,
            TemporalFusionExtractor,
        )
        assert ContextualTemporalFusionExtractor.DEFAULT_CONTEXT_DIM == 12
        assert TemporalFusionExtractor.DEFAULT_CONTEXT_DIM == 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
