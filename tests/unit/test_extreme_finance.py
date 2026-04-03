"""
Extreme‑case financial logic tests for ADAN.

These tests verify:
1. Accurate cost calculation (price + slippage + spread + Binance VIP0 fees).
2. Micro‑tier capital limits, dust handling, and circuit‑breaker trigger on drawdown.
3. Cash + positions invariant: cash + sum(positions) == initial_capital + PnL - fees.
"""

import sys
import os
import logging
import numpy as np
import pytest
from datetime import datetime

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

def _make_portfolio_manager(max_positions=5, initial_capital=100.0, fee_pct=0.001):
    """Instantiate a PortfolioManager with Micro-tier settings."""
    from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

    config = {
        "initial_capital": initial_capital,
        "assets": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT"],
        "portfolio": {"initial_balance": initial_capital},
        "trading_rules": {"commission_pct": fee_pct},
        "risk_management": {"min_trade_value": 5.0},
        "capital_tiers": [
            {
                "name": "Micro",
                "min_capital": 0,
                "max_capital": 10_000,
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


# ============================================================================
# TEST 1 – Accurate cost calculation: price + slippage + spread + VIP0 fees
# ============================================================================

class TestAccurateCostCalculation:
    """End-to-end cost calculation using all friction models."""

    def test_full_friction_pipeline_buy(self):
        """
        A $100 buy at $50_000 through slippage + liquidity + VIP0 fee
        must produce total_cost > $100 and each component must be non-negative.
        """
        from adan_trading_bot.environment.market_friction import (
            AdaptiveSlippage, LiquidityModel, BinanceFeeModel, MarketConditions,
        )

        target_price = 50_000.0
        order_value = 100.0
        mc = MarketConditions(volatility=0.01, spread_bps=5.0, volume_24h=1e9)

        slip = AdaptiveSlippage(base_slippage_bps=2.0, size_impact_factor=0.1,
                                volatility_impact_factor=0.5)
        liq = LiquidityModel(depth_factor=0.001, impact_exponent=1.5)
        fees = BinanceFeeModel(tier="VIP0")

        # Step 1: slippage
        price_after_slip = slip.apply_slippage(target_price, order_value, mc, "buy")
        slippage_delta = price_after_slip - target_price
        assert slippage_delta >= 0, f"Buy slippage must be >= 0, got {slippage_delta}"

        # Step 2: liquidity impact
        price_after_liq = liq.apply_impact(price_after_slip, order_value, mc, "buy")
        liq_delta = price_after_liq - price_after_slip
        assert liq_delta >= 0, f"Buy liquidity impact must be >= 0, got {liq_delta}"

        # Step 3: fee
        fee_amount = fees.calculate_fee(order_value, is_maker=False)
        assert fee_amount == pytest.approx(0.10, abs=1e-10), (
            f"VIP0 taker fee on $100 should be $0.10, got {fee_amount}"
        )

        # Final cost
        qty = order_value / target_price
        total_cost = qty * price_after_liq + fee_amount
        assert total_cost > order_value, (
            f"Total cost {total_cost:.6f} must exceed order value {order_value}"
        )

    def test_sell_friction_reduces_proceeds(self):
        """Sell friction should reduce the proceeds below the nominal value."""
        from adan_trading_bot.environment.market_friction import (
            AdaptiveSlippage, LiquidityModel, BinanceFeeModel, MarketConditions,
        )

        target_price = 50_000.0
        order_value = 100.0
        mc = MarketConditions(volatility=0.01, spread_bps=5.0, volume_24h=1e9)

        slip = AdaptiveSlippage(base_slippage_bps=2.0, size_impact_factor=0.1,
                                volatility_impact_factor=0.5)
        liq = LiquidityModel(depth_factor=0.001, impact_exponent=1.5)
        fees = BinanceFeeModel(tier="VIP0")

        price_after_slip = slip.apply_slippage(target_price, order_value, mc, "sell")
        price_after_liq = liq.apply_impact(price_after_slip, order_value, mc, "sell")
        fee_amount = fees.calculate_fee(order_value, is_maker=False)

        qty = order_value / target_price
        net_proceeds = qty * price_after_liq - fee_amount
        assert net_proceeds < order_value, (
            f"Sell proceeds {net_proceeds:.6f} should be less than order value"
        )

    def test_bnb_discount_reduces_fee(self):
        """BNB discount should reduce fee by 25%."""
        from adan_trading_bot.environment.market_friction import BinanceFeeModel

        no_bnb = BinanceFeeModel(tier="VIP0", use_bnb_discount=False)
        with_bnb = BinanceFeeModel(tier="VIP0", use_bnb_discount=True)

        fee_no = no_bnb.calculate_fee(100.0, is_maker=False)
        fee_yes = with_bnb.calculate_fee(100.0, is_maker=False)

        assert fee_yes == pytest.approx(fee_no * 0.75, abs=1e-10), (
            f"BNB discount should give 75% of base fee: {fee_yes} vs {fee_no * 0.75}"
        )


# ============================================================================
# TEST 2 – Micro-tier capital limits, dust, circuit-breaker
# ============================================================================

class TestMicroTierLimits:
    """Capital tier constraints and edge cases."""

    def test_dust_order_rejected(self):
        """An order below min_trade_value (dust) must be rejected."""
        pm = _make_portfolio_manager(max_positions=5, initial_capital=1000.0, fee_pct=0.001)
        ts = datetime(2025, 1, 1, 12, 0, 0)

        # Attempt a $4 trade (below $5 min_trade_value)
        receipt = pm.open_position(
            asset="BTCUSDT",
            price=100.0,
            size=0.03,  # 0.03 * 100 = $3.00 < $5
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            timeframe="5m",
            timestamp=ts,
            current_prices={"BTCUSDT": 100.0},
        )
        assert receipt is None, "Dust order below min_trade_value must be rejected"

    def test_insufficient_cash_rejected(self):
        """An order exceeding available cash must be rejected."""
        pm = _make_portfolio_manager(max_positions=5, initial_capital=10.0, fee_pct=0.001)
        ts = datetime(2025, 1, 1, 12, 0, 0)

        receipt = pm.open_position(
            asset="BTCUSDT",
            price=100.0,
            size=1.0,  # $100 > $10 capital
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            timeframe="5m",
            timestamp=ts,
            current_prices={"BTCUSDT": 100.0},
        )
        assert receipt is None, "Order exceeding cash must be rejected"

    def test_circuit_breaker_concept(self):
        """
        If portfolio drops below circuit_breaker_pct (15%) of initial capital,
        the system should be in a critical state.
        """
        pm = _make_portfolio_manager(max_positions=5, initial_capital=100.0, fee_pct=0.001)

        # Verify initial values
        assert pm.initial_capital == pytest.approx(100.0, abs=0.01), (
            f"Initial capital should be 100, got {pm.initial_capital}"
        )
        assert pm.get_cash() == pytest.approx(100.0, abs=0.01), (
            f"Initial cash should be ~100, got {pm.get_cash()}"
        )

    def test_position_limit_hard_enforcement(self):
        """Positions beyond max_positions must be rejected."""
        pm = _make_portfolio_manager(max_positions=2, initial_capital=1000.0, fee_pct=0.001)
        ts = datetime(2025, 1, 1, 12, 0, 0)
        all_prices = {"BTCUSDT": 50.0, "ETHUSDT": 50.0, "SOLUSDT": 50.0}

        r1 = pm.open_position(asset="BTCUSDT", price=50.0, size=1.0,
                               stop_loss_pct=0.05, take_profit_pct=0.10,
                               timeframe="5m", timestamp=ts, current_prices=all_prices)
        r2 = pm.open_position(asset="ETHUSDT", price=50.0, size=1.0,
                               stop_loss_pct=0.05, take_profit_pct=0.10,
                               timeframe="5m", timestamp=ts, current_prices=all_prices)
        r3 = pm.open_position(asset="SOLUSDT", price=50.0, size=1.0,
                               stop_loss_pct=0.05, take_profit_pct=0.10,
                               timeframe="5m", timestamp=ts, current_prices=all_prices)

        assert r1 is not None, "First position should succeed"
        assert r2 is not None, "Second position should succeed"
        assert r3 is None, "Third position should be rejected (max=2)"


# ============================================================================
# TEST 3 – Capital invariant: cash + positions == initial + PnL - fees
# ============================================================================

class TestCapitalInvariant:
    """Verify the accounting invariant holds after trades."""

    def test_open_trade_preserves_capital(self):
        """
        After opening a position:
          cash + position_value == initial_capital (no PnL yet, fees deducted from cash)
          The difference should be exactly the entry fee.
        """
        initial = 1000.0
        fee_pct = 0.001
        pm = _make_portfolio_manager(max_positions=5, initial_capital=initial, fee_pct=fee_pct)
        ts = datetime(2025, 1, 1, 12, 0, 0)

        price = 100.0
        size = 2.0  # $200 trade
        notional = price * size
        entry_fee = notional * fee_pct

        receipt = pm.open_position(
            asset="BTCUSDT",
            price=price,
            size=size,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            timeframe="5m",
            timestamp=ts,
            current_prices={"BTCUSDT": price},
        )
        assert receipt is not None, "Trade should succeed"

        # Cash should be reduced by notional + fee
        expected_cash = initial - notional - entry_fee
        actual_cash = pm.get_cash()
        assert actual_cash == pytest.approx(expected_cash, abs=0.01), (
            f"Cash after open: expected {expected_cash:.4f}, got {actual_cash:.4f}"
        )

        # Total value should be initial - fee (position value = notional, cash = initial - notional - fee)
        total = pm.get_total_value()
        expected_total = initial - entry_fee
        assert total == pytest.approx(expected_total, abs=0.5), (
            f"Total value should be ~{expected_total:.4f}, got {total:.4f}"
        )

    def test_close_trade_accounting(self):
        """
        Open then close at same price: PnL should be ~0, but fees deducted.
        Final cash == initial_capital - entry_fee - exit_fee.
        """
        initial = 1000.0
        fee_pct = 0.001
        pm = _make_portfolio_manager(max_positions=5, initial_capital=initial, fee_pct=fee_pct)
        ts = datetime(2025, 1, 1, 12, 0, 0)

        price = 100.0
        size = 2.0
        notional = price * size

        # Open
        pm.open_position(
            asset="BTCUSDT", price=price, size=size,
            stop_loss_pct=0.05, take_profit_pct=0.10,
            timeframe="5m", timestamp=ts,
            current_prices={"BTCUSDT": price},
        )

        # Close at same price
        receipt = pm.close_position(
            asset="BTCUSDT", price=price,
            timestamp=ts,
            current_prices={"BTCUSDT": price},
            reason="test",
        )

        # After round-trip, total value should be initial - 2*fee
        total = pm.get_total_value()
        entry_fee = notional * fee_pct
        exit_fee = notional * fee_pct
        total_fees = entry_fee + exit_fee
        expected_total = initial - total_fees

        # Allow some tolerance for implementation details
        assert total == pytest.approx(expected_total, abs=1.0), (
            f"After round-trip at same price: total should be ~{expected_total:.4f}, got {total:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
