#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TradeFrequencyController - Enforces trading frequency constraints.

Per ADAN 2.0 Spec (Requirements 2.5, 2.6, 2.7):
- Minimum intervals between trades
- Daily trade frequency limits
- Per-asset cooldown periods
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FrequencyConfig:
    """Configuration for trade frequency constraints."""
    min_interval_steps: int = 6  # Minimum steps between ANY trades
    daily_trade_limit: int = 10  # Maximum trades per day
    asset_cooldown_steps: int = 3  # Cooldown per asset after trade
    
    # Per-timeframe force trade thresholds
    force_trade_steps_by_tf: Dict[str, int] = field(default_factory=lambda: {
        "5m": 15,
        "1h": 20,
        "4h": 50
    })


class TradeFrequencyController:
    """
    Enforces trading frequency constraints to prevent overtrading.
    
    Responsibilities:
    - Track last trade step globally and per-asset
    - Enforce minimum intervals between trades
    - Enforce daily trade limits
    - Enforce per-asset cooldown periods
    - Provide force trade logic
    """
    
    def __init__(self, config: FrequencyConfig):
        """
        Initialize the controller.
        
        Args:
            config: FrequencyConfig with constraint parameters
        """
        self.config = config
        
        # Global tracking
        self.last_trade_step: int = -999  # Last trade step (any asset)
        self.daily_trade_count: int = 0
        self.current_day: int = 0
        
        # Per-asset tracking
        self.asset_last_trade: Dict[str, int] = {}  # asset -> last trade step
        self.asset_trade_count: Dict[str, int] = {}  # asset -> trades today
        
        # Per-timeframe tracking for force trades
        self.last_trade_by_tf: Dict[str, int] = {
            "5m": -999,
            "1h": -999,
            "4h": -999
        }
        
        logger.info(f"TradeFrequencyController initialized with config: {config}")
    
    def can_open_trade(
        self, 
        asset: str, 
        current_step: int,
        check_global: bool = True,
        check_asset: bool = True,
        check_daily: bool = True
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a trade can be opened given all constraints.
        
        Args:
            asset: Asset symbol
            current_step: Current environment step
            check_global: Check global minimum interval
            check_asset: Check per-asset cooldown
            check_daily: Check daily trade limit
            
        Returns:
            (can_trade, reason) - True if allowed, False with reason if blocked
        """
        # Check daily limit (only natural trades count)
        if check_daily and self.daily_trade_count >= self.config.daily_trade_limit:
            return False, f"Daily limit reached ({self.daily_trade_count}/{self.config.daily_trade_limit})"
        
        # Check global minimum interval
        if check_global:
            steps_since_last = current_step - self.last_trade_step
            if steps_since_last < self.config.min_interval_steps:
                return False, f"Global interval not met ({steps_since_last}/{self.config.min_interval_steps})"
        
        # Check per-asset cooldown
        if check_asset and asset in self.asset_last_trade:
            steps_since_asset = current_step - self.asset_last_trade[asset]
            if steps_since_asset < self.config.asset_cooldown_steps:
                return False, f"Asset cooldown active ({steps_since_asset}/{self.config.asset_cooldown_steps})"
        
        return True, None
    
    def can_close_trade(
        self, 
        asset: str, 
        current_step: int
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a trade can be closed.
        
        Currently no constraints on closing, but included for API symmetry.
        
        Args:
            asset: Asset symbol
            current_step: Current environment step
            
        Returns:
            (can_trade, reason)
        """
        # No constraints on closing for now
        # Could add minimum hold duration here if needed
        return True, None
    
    def record_trade(
        self, 
        asset: str, 
        current_step: int,
        timeframe: Optional[str] = None,
        is_forced: bool = False
    ) -> None:
        """
        Record a trade execution for constraint tracking.
        
        Args:
            asset: Asset symbol
            current_step: Current environment step
            timeframe: Optional timeframe for the trade (5m, 1h, 4h)
            is_forced: Whether this is a forced trade (doesn't count toward daily limit)
        """
        # Update global tracking
        self.last_trade_step = current_step
        # Only count natural trades toward daily limit
        if not is_forced:
            self.daily_trade_count += 1
        
        # Update per-asset tracking (ONLY for natural trades to avoid cooldown interference)
        if not is_forced:
            self.asset_last_trade[asset] = current_step
        self.asset_trade_count[asset] = self.asset_trade_count.get(asset, 0) + 1
        
        # Update per-timeframe tracking if provided
        if timeframe and timeframe in self.last_trade_by_tf:
            self.last_trade_by_tf[timeframe] = current_step
        
        logger.debug(
            f"Trade recorded: {asset} at step {current_step} "
            f"(daily: {self.daily_trade_count}, tf: {timeframe})"
        )
    
    def should_force_trade(
        self,
        current_step: int,
        timeframe: str
    ) -> bool:
        """
        Check if force trade is required for a given timeframe.
        
        Args:
            current_step: Current environment step
            timeframe: Timeframe to check (5m, 1h, 4h)
            
        Returns:
            True if force trade should be triggered
        """
        if timeframe not in self.config.force_trade_steps_by_tf:
            return False
        
        threshold = self.config.force_trade_steps_by_tf[timeframe]
        last_trade = self.last_trade_by_tf.get(timeframe, -999)
        steps_since = current_step - last_trade
        
        return steps_since >= threshold
    
    def reset_daily(self, new_day: int) -> None:
        """
        Reset daily counters (called at start of new trading day).
        
        Args:
            new_day: New day number
        """
        if new_day != self.current_day:
            logger.info(
                f"Daily reset: Day {self.current_day} -> {new_day} "
                f"(trades: {self.daily_trade_count})"
            )
            self.current_day = new_day
            self.daily_trade_count = 0
            self.asset_trade_count.clear()
    
    def reset(self) -> None:
        """Reset all tracking (called at episode start)."""
        self.last_trade_step = -999
        self.daily_trade_count = 0
        self.current_day = 0
        self.asset_last_trade.clear()
        self.asset_trade_count.clear()
        self.last_trade_by_tf = {
            "5m": -999,
            "1h": -999,
            "4h": -999
        }
        logger.debug("TradeFrequencyController reset")
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            "daily_trade_count": self.daily_trade_count,
            "daily_limit": self.config.daily_trade_limit,
            "last_trade_step": self.last_trade_step,
            "asset_cooldowns": {
                asset: step for asset, step in self.asset_last_trade.items()
            },
            "timeframe_last_trades": self.last_trade_by_tf.copy()
        }
