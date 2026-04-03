#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Market Friction Models for ADAN 2.0 Realistic Trading Environment.

Per ADAN 2.0 Spec (Requirements 2.2, 2.3, 2.4):
- AdaptiveSlippage: Size and condition-based slippage
- LatencySimulator: Network delay effects
- LiquidityModel: Order impact on execution price
"""

import logging
import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MarketConditions:
    """Current market conditions for friction calculations."""
    volatility: float = 0.01  # ATR or realized volatility
    spread_bps: float = 5.0  # Bid-ask spread in basis points
    volume_24h: float = 1e9  # 24h volume in USDT


class AdaptiveSlippage:
    """
    Calculates adaptive slippage based on order size and market conditions.
    
    Slippage increases with:
    - Larger order size relative to average volume
    - Higher market volatility
    - Wider bid-ask spreads
    """
    
    def __init__(
        self,
        base_slippage_bps: float = 2.0,  # Base slippage in basis points
        size_impact_factor: float = 0.1,  # How much size affects slippage
        volatility_impact_factor: float = 0.5  # How much volatility affects slippage
    ):
        """
        Initialize slippage calculator.
        
        Args:
            base_slippage_bps: Minimum slippage for small orders
            size_impact_factor: Multiplier for order size impact
            volatility_impact_factor: Multiplier for volatility impact
        """
        self.base_slippage_bps = base_slippage_bps
        self.size_impact_factor = size_impact_factor
        self.volatility_impact_factor = volatility_impact_factor
        
        logger.info(
            f"AdaptiveSlippage initialized (base={base_slippage_bps}bps, "
            f"size_factor={size_impact_factor}, vol_factor={volatility_impact_factor})"
        )
    
    def calculate_slippage(
        self,
        price: float,
        order_size_usd: float,
        market_conditions: MarketConditions,
        side: str = "buy"
    ) -> float:
        """
        Calculate slippage for an order.
        
        Args:
            price: Current market price
            order_size_usd: Order size in USDT
            market_conditions: Current market state
            side: "buy" or "sell"
            
        Returns:
            Slippage amount to add/subtract from price
        """
        # Base slippage (always present)
        slippage_bps = self.base_slippage_bps
        
        # Add spread component
        slippage_bps += market_conditions.spread_bps / 2.0
        
        # Add size impact (larger orders get more slippage)
        # Simple model: slippage ~ sqrt(order_size / avg_volume)
        avg_order_size = market_conditions.volume_24h / 1000  # Assume 1000 orders/day
        if avg_order_size > 0:
            size_ratio = order_size_usd / avg_order_size
            size_slippage = self.size_impact_factor * np.sqrt(max(size_ratio, 0.01))
            slippage_bps += size_slippage
        
        # Add volatility impact
        volatility_slippage = self.volatility_impact_factor * market_conditions.volatility * 10000  # Convert to bps
        slippage_bps += volatility_slippage
        
        # Convert basis points to price change
        slippage_pct = slippage_bps / 10000.0
        slippage_amount = price * slippage_pct
        
        # Apply direction: buyers pay more, sellers get less
        if side == "buy":
            return slippage_amount  # Positive = worse price
        else:
            return -slippage_amount  # Negative = worse price for seller
    
    def apply_slippage(
        self,
        price: float,
        order_size_usd: float,
        market_conditions: MarketConditions,
        side: str = "buy"
    ) -> float:
        """
        Apply slippage to a price.
        
        Args:
            price: Target price
            order_size_usd: Order size in USDT
            market_conditions: Market state
            side: "buy" or "sell"
            
        Returns:
            Execution price after slippage
        """
        slippage = self.calculate_slippage(price, order_size_usd, market_conditions, side)
        return price + slippage


class LatencySimulator:
    """
    Simulates network latency effects on execution price.
    
    Models realistic network delay (50-200ms) and potential price movement
    during that delay.
    """
    
    def __init__(
        self,
        min_latency_ms: float = 50.0,
        max_latency_ms: float = 200.0,
        price_drift_per_ms: float = 0.00001  # % price drift per ms
    ):
        """
        Initialize latency simulator.
        
        Args:
            min_latency_ms: Minimum network latency
            max_latency_ms: Maximum network latency
            price_drift_per_ms: Expected price drift per millisecond
        """
        self.min_latency_ms = min_latency_ms
        self.max_latency_ms = max_latency_ms
        self.price_drift_per_ms = price_drift_per_ms
        
        logger.info(
            f"LatencySimulator initialized ({min_latency_ms}-{max_latency_ms}ms, "
            f"drift={price_drift_per_ms}%/ms)"
        )
    
    def simulate_latency_impact(
        self,
        price: float,
        market_conditions: MarketConditions,
        rng: Optional[np.random.Generator] = None
    ) -> tuple[float, float]:
        """
        Simulate price impact due to network latency.
        
        Args:
            price: Current market price
            market_conditions: Market state
            rng: Random number generator for reproducibility
            
        Returns:
            (execution_price, latency_ms)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Random latency in range
        latency_ms = rng.uniform(self.min_latency_ms, self.max_latency_ms)
        
        # Price drift during latency (stochastic)
        # Higher volatility = more drift potential
        drift_std = market_conditions.volatility * self.price_drift_per_ms * latency_ms
        price_drift_pct = rng.normal(0, drift_std)
        
        # Apply drift
        execution_price = price * (1.0 + price_drift_pct)
        
        return execution_price, latency_ms
    
    def apply_latency(
        self,
        price: float,
        market_conditions: MarketConditions,
        rng: Optional[np.random.Generator] = None
    ) -> float:
        """
        Apply latency effect to execution price.
        
        Args:
            price: Target price
            market_conditions: Market state
            rng: RNG for reproducibility
            
        Returns:
            Execution price after latency
        """
        execution_price, _ = self.simulate_latency_impact(price, market_conditions, rng)
        return execution_price


class LiquidityModel:
    """
    Models order impact on execution price based on market depth.
    
    Large orders experience price impact as they consume liquidity
    from the order book.
    """
    
    def __init__(
        self,
        depth_factor: float = 0.001,  # Typical market depth as % of volume
        impact_exponent: float = 1.5  # Non-linear impact (>1 = superlinear)
    ):
        """
        Initialize liquidity model.
        
        Args:
            depth_factor: Market depth as fraction of daily volume
            impact_exponent: How impact scales with order size
        """
        self.depth_factor = depth_factor
        self.impact_exponent = impact_exponent
        
        logger.info(
            f"LiquidityModel initialized (depth_factor={depth_factor}, "
            f"impact_exp={impact_exponent})"
        )
    
    def calculate_impact(
        self,
        price: float,
        order_size_usd: float,
        market_conditions: MarketConditions,
        side: str = "buy"
    ) -> float:
        """
        Calculate price impact for an order.
        
        Args:
            price: Current market price
            order_size_usd: Order size in USDT
            market_conditions: Market state
            side: "buy" or "sell"
            
        Returns:
            Price impact (positive for buys, negative for sells in terms of cost)
        """
        # Estimate available liquidity
        available_liquidity = market_conditions.volume_24h * self.depth_factor
        
        if available_liquidity <= 0:
            # No liquidity data, use conservative impact
            liquidity_ratio = 0.1
        else:
            liquidity_ratio = min(order_size_usd / available_liquidity, 1.0)
        
        # Non-linear impact: small orders have minimal impact, large orders get hit hard
        impact_pct = (liquidity_ratio ** self.impact_exponent) * 0.01  # Max 1% impact
        
        # Apply direction
        if side == "buy":
            return price * impact_pct  # Positive = pay more
        else:
            return -price * impact_pct  # Negative = receive less
    
    def apply_impact(
        self,
        price: float,
        order_size_usd: float,
        market_conditions: MarketConditions,
        side: str = "buy"
    ) -> float:
        """
        Apply liquidity impact to execution price.
        
        Args:
            price: Target price
            order_size_usd: Order size
            market_conditions: Market state
            side: "buy" or "sell"
            
        Returns:
            Execution price after liquidity impact
        """
        impact = self.calculate_impact(price, order_size_usd, market_conditions, side)
        return price + impact


class BinanceFeeModel:
    """
    Realistic Binance fee structure.
    
    Fees depend on:
    - Trading tier (based on 30-day volume or BNB holdings)
    - Maker vs Taker
    - Using BNB for fee discount
    """
    
    def __init__(
        self,
        tier: str = "VIP0",  # VIP tiers: VIP0 (normal), VIP1-9
        use_bnb_discount: bool = False,
        maker_fee: Optional[float] = None,
        taker_fee: Optional[float] = None
    ):
        """
        Initialize Binance fee model.
        
        Args:
            tier: Binance VIP tier (VIP0 = regular user)
            use_bnb_discount: Whether to apply BNB 25% discount
            maker_fee: Explicit maker fee rate (overrides tier)
            taker_fee: Explicit taker fee rate (overrides tier)
        """
        self.tier = tier
        self.use_bnb_discount = use_bnb_discount
        self.explicit_maker_fee = maker_fee
        self.explicit_taker_fee = taker_fee
        
        # Binance fee structure (spot trading, as of 2024)
        self.fee_schedule = {
            "VIP0": {"maker": 0.001, "taker": 0.001},  # 0.1%
            "VIP1": {"maker": 0.0009, "taker": 0.001},
            "VIP2": {"maker": 0.0008, "taker": 0.0009},
            # ... (simplified, only showing VIP0-2)
        }
        
        logger.info(
            f"BinanceFeeModel initialized (tier={tier}, BNB_discount={use_bnb_discount})"
        )
    
    def get_fee_rate(self, is_maker: bool = False) -> float:
        """
        Get fee rate for current tier.
        
        Args:
            is_maker: True if maker order, False if taker
            
        Returns:
            Fee rate (e.g., 0.001 = 0.1%)
        """
        if is_maker and self.explicit_maker_fee is not None:
            return self.explicit_maker_fee
        if not is_maker and self.explicit_taker_fee is not None:
            return self.explicit_taker_fee
            
        fees = self.fee_schedule.get(self.tier, self.fee_schedule["VIP0"])
        base_fee = fees["maker"] if is_maker else fees["taker"]
        
        # Apply BNB discount (25% off)
        if self.use_bnb_discount:
            base_fee *= 0.75
        
        return base_fee
    
    def calculate_fee(
        self,
        order_value_usd: float,
        is_maker: bool = False
    ) -> float:
        """
        Calculate fee for an order.
        
        Args:
            order_value_usd: Order value in USDT
            is_maker: Whether it's a maker order
            
        Returns:
            Fee amount in USDT
        """
        fee_rate = self.get_fee_rate(is_maker)
        return order_value_usd * fee_rate


class StaleDataSimulator:
    """
    Simulates stale data (lag) in observations.
    
    Randomly returns previous observations instead of current ones
    to mimic network latency or websocket delays.
    """
    
    def __init__(
        self,
        prob_stale: float = 0.0,  # Probability of stale data (0.0 to 1.0)
        max_lag_steps: int = 3    # Maximum lag in steps
    ):
        """
        Initialize stale data simulator.
        
        Args:
            prob_stale: Probability of receiving stale data
            max_lag_steps: Maximum number of steps to lag behind
        """
        self.prob_stale = prob_stale
        self.max_lag_steps = max_lag_steps
        self.history = []  # Buffer of recent observations
        
        logger.info(
            f"StaleDataSimulator initialized (prob={prob_stale}, max_lag={max_lag_steps})"
        )
    
    def get_observation(
        self,
        current_obs: Any,
        rng: Optional[np.random.Generator] = None
    ) -> Any:
        """
        Get potentially stale observation.
        
        Args:
            current_obs: The actual current observation
            rng: Random number generator
            
        Returns:
            Current or stale observation
        """
        # Update history
        self.history.append(current_obs)
        if len(self.history) > self.max_lag_steps + 1:
            self.history.pop(0)
            
        if self.prob_stale <= 0 or len(self.history) < 2:
            return current_obs
            
        if rng is None:
            rng = np.random.default_rng()
            
        # Decide if stale
        if rng.random() < self.prob_stale:
            # Choose a lag amount (1 to max_lag, limited by history size)
            max_available_lag = len(self.history) - 1
            lag = rng.integers(1, max_available_lag + 1)
            return self.history[-(lag + 1)]
            
        return current_obs
    
    def reset(self):
        """Reset history."""
        self.history = []

