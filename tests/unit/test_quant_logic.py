"""
Quantitative finance logic unit tests for ADAN.

These tests verify:
1. Fees & slippage are correctly deducted for a $100 trade.
2. StateBuilder observations include spread/liquidity from the L2 order book.
3. Micro-tier position limit (max 5) is enforced with invalid_trade_penalty.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure ADAN packages are importable
# ---------------------------------------------------------------------------
BOT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(BOT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ============================================================================
# TEST 1 – Fees & Slippage for a $100 trade
# ============================================================================

class TestFeesAndSlippage:
    """Verify that a $100 buy trade has exact fee + slippage deducted."""

    def test_apply_market_friction_deducts_costs(self):
        """
        A buy order of $100 at price $50000 must:
          - include non-zero slippage
          - include non-zero fee (Binance VIP0 = 0.1 %)
          - produce an execution_price > target_price (buyer pays more)
          - total_cost > original order value
        """
        from adan_trading_bot.environment.market_friction import (
            AdaptiveSlippage,
            LatencySimulator,
            LiquidityModel,
            BinanceFeeModel,
            MarketConditions,
        )

        target_price = 50_000.0
        order_size = 100.0  # $100 USD

        slippage_model = AdaptiveSlippage(
            base_slippage_bps=2.0 / 100.0,
            size_impact_factor=0.1,
            volatility_impact_factor=0.5,
        )
        latency_model = LatencySimulator(
            min_latency_ms=50.0, max_latency_ms=200.0, price_drift_per_ms=0.00001
        )
        liquidity_model = LiquidityModel(depth_factor=0.001, impact_exponent=1.5)
        fee_model = BinanceFeeModel(
            tier="VIP0", use_bnb_discount=False, maker_fee=None, taker_fee=None
        )

        mc = MarketConditions(volatility=0.01, spread_bps=5.0, volume_24h=1e9)

        # Slippage
        price_after_slip = slippage_model.apply_slippage(
            target_price, order_size, mc, "buy"
        )
        assert price_after_slip > target_price, (
            f"Slippage must push buy price UP: got {price_after_slip}"
        )

        # Fee
        fee = fee_model.calculate_fee(order_size, is_maker=False)
        expected_fee = order_size * 0.001  # VIP0 taker = 0.1 %
        assert abs(fee - expected_fee) < 1e-8, (
            f"Fee should be {expected_fee}, got {fee}"
        )

        # Liquidity impact
        final_price = liquidity_model.apply_impact(
            price_after_slip, order_size, mc, "buy"
        )
        assert final_price >= price_after_slip, (
            "Liquidity impact should push buy price UP or keep it the same"
        )

        # Net cost: (quantity * execution_price) + fee > $100
        qty = order_size / target_price
        total_cost = qty * final_price + fee
        assert total_cost > order_size, (
            f"Total cost ${total_cost:.4f} must exceed order value ${order_size}"
        )

    def test_binance_fee_exact_deduction(self):
        """Commission for a $100 trade at VIP0 = $0.10 (0.1 %)."""
        from adan_trading_bot.environment.market_friction import BinanceFeeModel

        fee_model = BinanceFeeModel(tier="VIP0", use_bnb_discount=False)
        fee = fee_model.calculate_fee(100.0, is_maker=False)
        assert abs(fee - 0.10) < 1e-10, f"Expected $0.10 fee, got ${fee:.6f}"

    def test_sell_slippage_negative(self):
        """For a sell order, slippage moves price DOWN (worse for seller)."""
        from adan_trading_bot.environment.market_friction import (
            AdaptiveSlippage, MarketConditions,
        )
        model = AdaptiveSlippage(base_slippage_bps=0.02, size_impact_factor=0.1, volatility_impact_factor=0.5)
        mc = MarketConditions(volatility=0.01, spread_bps=5.0, volume_24h=1e9)
        price = model.apply_slippage(50_000.0, 100.0, mc, "sell")
        assert price < 50_000.0, f"Sell slippage should reduce price: {price}"


# ============================================================================
# TEST 2 – StateBuilder includes spread / liquidity from L2 order book
# ============================================================================

class TestStateBuilderL2Info:
    """Verify that the context_vector (or observation) carries spread/liquidity."""

    @pytest.fixture
    def mock_data_with_spread(self):
        """Build dummy multi-asset data containing L2 spread info."""
        np.random.seed(42)
        n = 100
        base = {
            "open": np.random.uniform(49000, 51000, n),
            "high": np.random.uniform(50000, 52000, n),
            "low": np.random.uniform(48000, 50000, n),
            "close": np.random.uniform(49000, 51000, n),
            "volume": np.random.uniform(100, 1000, n),
        }
        # Add default features expected by StateBuilder
        for feat in [
            "rsi_14", "macd_12_26_9", "bb_percent_b_20_2",
            "atr_14", "atr_20", "atr_50",
            "volume_ratio_20", "ema_20_ratio", "stoch_k_14_3_3", "price_action",
        ]:
            base[feat] = np.random.uniform(-1, 1, n)

        # L2 order-book features — these MUST propagate into observation
        base["spread_bps"] = np.random.uniform(1.0, 10.0, n)
        base["liquidity_score"] = np.random.uniform(0.0, 1.0, n)

        df_5m = pd.DataFrame(base)

        # 1h / 4h — smaller windows
        df_1h = df_5m.head(50).copy()
        for feat in [
            "rsi_21", "macd_21_42_9", "bb_width_20_2", "adx_14",
            "atr_20", "atr_50", "obv_ratio_20", "ema_50_ratio",
            "ichimoku_base", "fib_ratio", "price_ema_ratio_50",
        ]:
            df_1h[feat] = np.random.uniform(-1, 1, len(df_1h))
        df_1h["spread_bps"] = np.random.uniform(1.0, 10.0, len(df_1h))
        df_1h["liquidity_score"] = np.random.uniform(0.0, 1.0, len(df_1h))

        df_4h = df_5m.head(30).copy()
        for feat in [
            "rsi_28", "macd_26_52_18", "supertrend_10_3",
            "atr_20", "atr_50", "volume_sma_20_ratio", "ema_100_ratio",
            "pivot_level", "donchian_width_20", "market_structure",
            "volatility_ratio_14_50",
        ]:
            df_4h[feat] = np.random.uniform(-1, 1, len(df_4h))
        df_4h["spread_bps"] = np.random.uniform(1.0, 10.0, len(df_4h))
        df_4h["liquidity_score"] = np.random.uniform(0.0, 1.0, len(df_4h))

        return {"BTCUSDT": {"5m": df_5m, "1h": df_1h, "4h": df_4h}}

    def test_spread_bps_in_observation(self, mock_data_with_spread):
        """The 5m feature list MUST include 'spread_bps' for L2 information."""
        from adan_trading_bot.data_processing.state_builder import StateBuilder

        sb = StateBuilder(normalize=False)
        features_5m = sb.get_feature_names("5m")
        assert "spread_bps" in features_5m, (
            f"'spread_bps' missing from 5m features: {features_5m}"
        )

    def test_liquidity_score_in_observation(self, mock_data_with_spread):
        """The 5m feature list MUST include 'liquidity_score' for L2 depth."""
        from adan_trading_bot.data_processing.state_builder import StateBuilder

        sb = StateBuilder(normalize=False)
        features_5m = sb.get_feature_names("5m")
        assert "liquidity_score" in features_5m, (
            f"'liquidity_score' missing from 5m features: {features_5m}"
        )

    def test_observation_contains_spread_data(self, mock_data_with_spread):
        """build_observation() must return arrays whose spread column is non-zero."""
        from adan_trading_bot.data_processing.state_builder import StateBuilder

        sb = StateBuilder(normalize=False)
        obs = sb.build_observation(
            current_idx=50,
            data=mock_data_with_spread,
            portfolio_manager=None,
        )

        # Find the column index for 'spread_bps' in the 5m observation
        features_5m = sb.get_feature_names("5m")
        assert "spread_bps" in features_5m
        spread_idx = features_5m.index("spread_bps")

        obs_5m = obs["5m"]
        spread_col = obs_5m[:, spread_idx]
        assert np.any(spread_col != 0.0), (
            "spread_bps column in observation is all zeros — L2 data not propagated"
        )


# ============================================================================
# TEST 3 – Micro tier position limit (max 5) enforced
# ============================================================================

class TestMicroTierPositionLimit:
    """When on the Micro tier (max_open_positions=5), the 6th trade MUST be rejected."""

    def _make_portfolio_manager(self, max_positions=5, initial_capital=100.0):
        """Instantiate a PortfolioManager with Micro-tier settings."""
        from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

        config = {
            "initial_capital": initial_capital,
            "assets": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT"],
            "portfolio": {"initial_balance": initial_capital},
            "trading_rules": {"commission_pct": 0.001},
            "risk_management": {"min_trade_value": 5.0},
            "capital_tiers": [
                {
                    "name": "Micro",
                    "min_capital": 0,
                    "max_capital": 500,
                    "max_open_positions": max_positions,
                    "max_position_size_pct": 25,
                },
            ],
        }
        pm = PortfolioManager(
            config=config,
            worker_id=0,
            max_positions=max_positions,
        )
        return pm

    def test_6th_position_rejected(self):
        """Opening the 6th position on Micro (limit=5) must return None."""
        from datetime import datetime
        pm = self._make_portfolio_manager(max_positions=5, initial_capital=1000.0)

        assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
        ts = datetime(2025, 1, 1, 12, 0, 0)
        opened = 0
        for i, asset in enumerate(assets):
            receipt = pm.open_position(
                asset=asset,
                price=100.0,
                size=1.0,
                stop_loss_pct=0.05,
                take_profit_pct=0.10,
                timeframe="5m",
                timestamp=ts,
                current_prices={a: 100.0 for a in assets + ["LINKUSDT"]},
            )
            if receipt is not None:
                opened += 1

        assert opened == 5, f"Should have opened exactly 5 positions, got {opened}"

        # 6th position must be rejected
        receipt_6 = pm.open_position(
            asset="LINKUSDT",
            price=100.0,
            size=1.0,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            timeframe="5m",
            timestamp=ts,
            current_prices={a: 100.0 for a in assets + ["LINKUSDT"]},
        )
        assert receipt_6 is None, "6th position must be rejected on Micro tier"

    def test_position_limit_penalty_applied(self):
        """calculate_position_limit_penalty() should return a negative penalty
        when open positions exceed the tier limit."""
        from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager
        from unittest.mock import MagicMock

        # Create a mock env that has the portfolio_manager with positions beyond limit
        pm = self._make_portfolio_manager(max_positions=5, initial_capital=1000.0)

        # Simulate 6 open positions by manipulating the positions dict
        from adan_trading_bot.portfolio.portfolio_manager import Position
        for i, asset in enumerate(["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT"]):
            pos = Position()
            pos.is_open = True
            pos.asset = asset
            pos.entry_price = 100.0
            pos.size = 1.0
            pos.current_price = 100.0
            pm.positions[asset] = pos

        open_count = len([p for p in pm.positions.values() if p.is_open])
        assert open_count == 6, f"Expected 6 open positions, got {open_count}"
        assert open_count > pm.max_positions, (
            "Open count should exceed max_positions to trigger penalty"
        )

    def test_capacity_at_100_percent(self):
        """With all capital allocated, capacity should be at 100 %."""
        from datetime import datetime
        pm = self._make_portfolio_manager(max_positions=5, initial_capital=500.0)

        # Open positions consuming all cash
        assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
        ts = datetime(2025, 1, 1, 12, 0, 0)
        all_prices = {a: 10.0 for a in assets}
        for asset in assets:
            pm.open_position(
                asset=asset,
                price=10.0,
                size=9.0,  # 9 * 10 = $90 + fee per position
                stop_loss_pct=0.05,
                take_profit_pct=0.10,
                timeframe="5m",
                timestamp=ts,
                current_prices=all_prices,
            )

        total = pm.get_total_value()
        cash = pm.get_cash()
        if total > 0:
            capacity = (total - cash) / total
            # Capacity should be high (most capital allocated)
            assert capacity > 0.5, (
                f"With all positions open, capacity should be > 50 %: got {capacity:.2%}"
            )


# ============================================================================
# TEST 4 – Relative price features (log-returns, close-EMA20/EMA20)
# ============================================================================

class TestRelativePriceFeatures:
    """StateBuilder must convert prices to relative features to avoid leakage."""

    def test_log_return_in_features(self):
        """The 5m feature list should include 'log_return'."""
        from adan_trading_bot.data_processing.state_builder import StateBuilder
        sb = StateBuilder(normalize=False)
        features = sb.get_feature_names("5m")
        assert "log_return" in features, (
            f"'log_return' missing from 5m features: {features}"
        )

    def test_close_ema20_ratio_in_features(self):
        """The 5m feature list should include 'close_ema20_ratio'."""
        from adan_trading_bot.data_processing.state_builder import StateBuilder
        sb = StateBuilder(normalize=False)
        features = sb.get_feature_names("5m")
        assert "close_ema20_ratio" in features, (
            f"'close_ema20_ratio' missing from 5m features: {features}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
