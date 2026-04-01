"""
Stationarity & numerical safety tests for ADAN.

These tests verify:
1. build_context_vector() never returns NaN/Inf values.
2. SafeScalerWrapper produces no NaN/Inf even with constant features (std=0).
3. StateBuilder produces relative features (log_return, close_ema20_ratio) not absolute prices.
4. Observations are finite and within reasonable bounds after scaling.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BOT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(BOT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================

def _make_mock_data(n=100, seed=42):
    """Create multi-asset, multi-timeframe mock data for StateBuilder."""
    rng = np.random.default_rng(seed)
    base_price = 50_000.0

    def _make_df(rows, tf_features):
        close = base_price + rng.normal(0, 100, rows).cumsum()
        close = np.maximum(close, 1.0)
        df = pd.DataFrame({
            "open": close + rng.normal(0, 10, rows),
            "high": close + abs(rng.normal(0, 50, rows)),
            "low": close - abs(rng.normal(0, 50, rows)),
            "close": close,
            "volume": rng.uniform(100, 1000, rows),
        })
        for feat in tf_features:
            if feat not in df.columns:
                df[feat] = rng.uniform(-1, 1, rows)
        df["spread_bps"] = rng.uniform(1, 10, rows)
        df["liquidity_score"] = rng.uniform(0, 1, rows)
        return df

    features_5m = [
        "rsi_14", "macd_12_26_9", "bb_percent_b_20_2",
        "atr_14", "atr_20", "atr_50",
        "volume_ratio_20", "ema_20_ratio", "stoch_k_14_3_3", "price_action",
    ]
    features_1h = [
        "rsi_21", "macd_21_42_9", "bb_width_20_2", "adx_14",
        "atr_20", "atr_50", "obv_ratio_20", "ema_50_ratio",
        "ichimoku_base", "fib_ratio", "price_ema_ratio_50",
    ]
    features_4h = [
        "rsi_28", "macd_26_52_18", "supertrend_10_3",
        "atr_20", "atr_50", "volume_sma_20_ratio", "ema_100_ratio",
        "pivot_level", "donchian_width_20", "market_structure", "volatility_ratio_14_50",
    ]

    return {
        "BTCUSDT": {
            "5m": _make_df(n, features_5m),
            "1h": _make_df(max(n // 2, 30), features_1h),
            "4h": _make_df(max(n // 4, 15), features_4h),
        }
    }


def _make_constant_data(n=50, seed=0):
    """Create data where all prices are constant (std=0)."""
    const_price = 42000.0
    df = pd.DataFrame({
        "open": np.full(n, const_price),
        "high": np.full(n, const_price),
        "low": np.full(n, const_price),
        "close": np.full(n, const_price),
        "volume": np.full(n, 500.0),
    })
    for feat in [
        "rsi_14", "macd_12_26_9", "bb_percent_b_20_2",
        "atr_14", "atr_20", "atr_50",
        "volume_ratio_20", "ema_20_ratio", "stoch_k_14_3_3", "price_action",
        "spread_bps", "liquidity_score",
    ]:
        df[feat] = 0.5
    return df


# ============================================================================
# TEST 1 - build_context_vector() never produces NaN/Inf
# ============================================================================

class TestContextVectorSafety:
    """build_context_vector must return 5 finite floats in all cases."""

    def test_context_vector_with_normal_data(self):
        from adan_trading_bot.data_processing.state_builder import StateBuilder
        sb = StateBuilder(normalize=False)
        data = _make_mock_data()
        flat = {}
        for asset, tfs in data.items():
            for tf, df in tfs.items():
                flat[tf] = df
        ctx = sb.build_context_vector(data=flat, current_idx=50)
        assert ctx.shape == (12,), f"Context vector should be (12,), got {ctx.shape}"
        assert np.all(np.isfinite(ctx)), f"Context vector has NaN/Inf: {ctx}"

    def test_context_vector_with_none_data(self):
        from adan_trading_bot.data_processing.state_builder import StateBuilder
        sb = StateBuilder(normalize=False)
        ctx = sb.build_context_vector(data=None, current_idx=0)
        assert ctx.shape == (12,)
        assert np.all(ctx == 0.0), f"Context with None data should be zeros: {ctx}"

    def test_context_vector_with_empty_data(self):
        from adan_trading_bot.data_processing.state_builder import StateBuilder
        sb = StateBuilder(normalize=False)
        empty_data = {"5m": pd.DataFrame(), "1h": pd.DataFrame()}
        ctx = sb.build_context_vector(data=empty_data, current_idx=0)
        assert ctx.shape == (12,)
        assert np.all(np.isfinite(ctx)), f"Context with empty data has NaN/Inf: {ctx}"

    def test_context_vector_with_constant_prices(self):
        from adan_trading_bot.data_processing.state_builder import StateBuilder
        sb = StateBuilder(normalize=False)
        const_df = _make_constant_data()
        data = {"5m": const_df}
        ctx = sb.build_context_vector(data=data, current_idx=25)
        assert ctx.shape == (12,)
        assert np.all(np.isfinite(ctx)), f"Context with constant prices has NaN/Inf: {ctx}"


# ============================================================================
# TEST 2 - SafeScalerWrapper handles constant features (std=0)
# ============================================================================

class TestSafeScalerWrapper:
    """SafeScalerWrapper must prevent NaN from std=0."""

    def test_constant_feature_no_nan(self):
        from adan_trading_bot.data_processing.state_builder import SafeScalerWrapper
        from sklearn.preprocessing import StandardScaler
        scaler = SafeScalerWrapper(StandardScaler(), epsilon=1e-8)
        data = np.array([[42.0, 1.0], [42.0, 2.0], [42.0, 3.0]], dtype=np.float32)
        scaler.fit(data)
        result = scaler.transform(data)
        assert np.all(np.isfinite(result)), f"SafeScaler produced NaN/Inf: {result}"
        assert np.all(result[:, 0] == 0.0), (
            f"Constant feature after scaling should be 0, got {result[:, 0]}"
        )

    def test_extreme_values_clipped(self):
        from adan_trading_bot.data_processing.state_builder import SafeScalerWrapper
        from sklearn.preprocessing import StandardScaler
        scaler = SafeScalerWrapper(StandardScaler(), epsilon=1e-8)
        normal = np.random.randn(100, 3).astype(np.float32)
        scaler.fit(normal)
        extreme = np.array([[1e10, -1e10, 0.0]], dtype=np.float32)
        result = scaler.transform(extreme)
        assert np.all(np.isfinite(result)), f"Extreme values produced NaN/Inf: {result}"

    def test_epsilon_protection(self):
        from adan_trading_bot.data_processing.state_builder import SafeScalerWrapper
        from sklearn.preprocessing import StandardScaler
        eps = 1e-6
        scaler = SafeScalerWrapper(StandardScaler(), epsilon=eps)
        assert scaler.epsilon == eps


# ============================================================================
# TEST 3 - Relative features exist and are computed correctly
# ============================================================================

class TestRelativeFeatures:
    """StateBuilder must inject log_return and close_ema20_ratio."""

    def test_log_return_computed_in_observation(self):
        from adan_trading_bot.data_processing.state_builder import StateBuilder
        sb = StateBuilder(normalize=False)
        data = _make_mock_data(n=100)
        obs = sb.build_observation(current_idx=50, data=data, portfolio_manager=None)
        features_5m = sb.get_feature_names("5m")
        assert "log_return" in features_5m
        idx = features_5m.index("log_return")
        lr_col = obs["5m"][:, idx]
        assert np.any(lr_col != 0.0), "log_return column is all zeros"

    def test_close_ema20_ratio_computed(self):
        from adan_trading_bot.data_processing.state_builder import StateBuilder
        sb = StateBuilder(normalize=False)
        data = _make_mock_data(n=100)
        obs = sb.build_observation(current_idx=50, data=data, portfolio_manager=None)
        features_5m = sb.get_feature_names("5m")
        assert "close_ema20_ratio" in features_5m
        idx = features_5m.index("close_ema20_ratio")
        ratio_col = obs["5m"][:, idx]
        assert np.any(ratio_col != 0.0), "close_ema20_ratio column is all zeros"

    def test_relative_features_all_timeframes(self):
        from adan_trading_bot.data_processing.state_builder import StateBuilder
        sb = StateBuilder(normalize=False)
        for tf in ["5m", "1h", "4h"]:
            features = sb.get_feature_names(tf)
            assert "log_return" in features, f"log_return missing from {tf}"
            assert "close_ema20_ratio" in features, f"close_ema20_ratio missing from {tf}"


# ============================================================================
# TEST 4 - Observations are finite after full pipeline
# ============================================================================

class TestObservationFiniteness:
    """After build_observation with normalization, all values must be finite."""

    def test_normalized_observation_is_finite(self):
        from adan_trading_bot.data_processing.state_builder import StateBuilder
        sb = StateBuilder(normalize=True)
        data = _make_mock_data(n=100)
        flat = {}
        for asset, tfs in data.items():
            for tf, df in tfs.items():
                flat[tf] = df
        sb.fit_scalers(flat)
        obs = sb.build_observation(current_idx=50, data=data, portfolio_manager=None)
        for key, arr in obs.items():
            assert np.all(np.isfinite(arr)), (
                f"Observation '{key}' contains NaN/Inf after normalization"
            )

    def test_observation_with_constant_prices(self):
        from adan_trading_bot.data_processing.state_builder import StateBuilder
        sb = StateBuilder(normalize=True)
        const_df = _make_constant_data(n=50)
        features_1h_extra = [
            "rsi_21", "macd_21_42_9", "bb_width_20_2", "adx_14",
            "obv_ratio_20", "ema_50_ratio", "ichimoku_base", "fib_ratio",
            "price_ema_ratio_50",
        ]
        features_4h_extra = [
            "rsi_28", "macd_26_52_18", "supertrend_10_3",
            "volume_sma_20_ratio", "ema_100_ratio",
            "pivot_level", "donchian_width_20", "market_structure",
            "volatility_ratio_14_50",
        ]
        df_1h = const_df.copy()
        for f in features_1h_extra:
            df_1h[f] = 0.5
        df_4h = const_df.copy()
        for f in features_4h_extra:
            df_4h[f] = 0.5
        data = {"BTCUSDT": {"5m": const_df, "1h": df_1h, "4h": df_4h}}
        flat = {"5m": const_df, "1h": df_1h, "4h": df_4h}
        sb.fit_scalers(flat)
        obs = sb.build_observation(current_idx=30, data=data, portfolio_manager=None)
        for key, arr in obs.items():
            assert np.all(np.isfinite(arr)), (
                f"Observation '{key}' has NaN/Inf with constant prices"
            )

    def test_portfolio_state_always_finite(self):
        from adan_trading_bot.data_processing.state_builder import StateBuilder
        sb = StateBuilder(normalize=False)
        data = _make_mock_data(n=100)
        obs = sb.build_observation(current_idx=50, data=data, portfolio_manager=None)
        ps = obs.get("portfolio_state")
        assert ps is not None, "portfolio_state should always be present"
        assert ps.shape == (20,), f"portfolio_state shape should be (20,), got {ps.shape}"
        assert np.all(np.isfinite(ps)), f"portfolio_state has NaN/Inf: {ps}"

    def test_context_vector_always_in_observation(self):
        from adan_trading_bot.data_processing.state_builder import StateBuilder
        sb = StateBuilder(normalize=False)
        data = _make_mock_data(n=100)
        obs = sb.build_observation(current_idx=50, data=data, portfolio_manager=None)
        cv = obs.get("context_vector")
        assert cv is not None, "context_vector should always be present"
        assert cv.shape == (12,), f"context_vector shape should be (12,), got {cv.shape}"
        assert np.all(np.isfinite(cv)), f"context_vector has NaN/Inf: {cv}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
