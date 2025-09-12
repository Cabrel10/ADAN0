"""
Fee management module for the ADAN Trading Bot.

This module provides comprehensive fee calculation and management for different
trading scenarios, exchanges, and asset types.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FeeType(Enum):
    """Types of trading fees."""
    MAKER = "maker"
    TAKER = "taker"
    WITHDRAWAL = "withdrawal"
    DEPOSIT = "deposit"
    FUNDING = "funding"
    LIQUIDATION = "liquidation"


class FeeStructure(Enum):
    """Fee structure types."""
    PERCENTAGE = "percentage"
    FIXED = "fixed"
    TIERED = "tiered"
    VOLUME_BASED = "volume_based"


@dataclass
class FeeConfig:
    """Configuration for fee calculation."""
    fee_type: FeeType
    structure: FeeStructure
    rate: float
    minimum_fee: Optional[float] = None
    maximum_fee: Optional[float] = None
    currency: str = "USD"
    tiers: Optional[List[Dict[str, float]]] = None


@dataclass
class FeeCalculationResult:
    """Result of fee calculation."""
    total_fee: float
    fee_breakdown: Dict[str, float]
    currency: str
    effective_rate: float
    warnings: List[str] = None


class FeeManager:
    """
    Comprehensive fee management system.
    
    This class handles fee calculation for different trading scenarios,
    exchanges, and asset types with support for complex fee structures.
    """
    
    def __init__(self,
                 default_maker_fee: float = 0.001,  # 0.1%
                 default_taker_fee: float = 0.0015,  # 0.15%
                 default_withdrawal_fee: float = 0.0005,  # 0.05%
                 enable_fee_optimization: bool = True,
                 volume_discount_enabled: bool = True):
        """
        Initialize the FeeManager.
        
        Args:
            default_maker_fee: Default maker fee rate
            default_taker_fee: Default taker fee rate
            default_withdrawal_fee: Default withdrawal fee rate
            enable_fee_optimization: Whether to optimize for lower fees
            volume_discount_enabled: Whether to apply volume discounts
        """
        self.default_maker_fee = default_maker_fee
        self.default_taker_fee = default_taker_fee
        self.default_withdrawal_fee = default_withdrawal_fee
        self.enable_fee_optimization = enable_fee_optimization
        self.volume_discount_enabled = volume_discount_enabled
        
        # Fee configurations for different exchanges/scenarios
        self.fee_configs = self._initialize_default_configs()
        
        # Volume tracking for discounts
        self.volume_history = {}
        
        # Fee statistics
        self.fee_stats = {
            'total_fees_paid': 0.0,
            'fees_by_type': {fee_type: 0.0 for fee_type in FeeType},
            'fee_optimization_savings': 0.0,
            'total_trades': 0
        }
        
        logger.info("FeeManager initialized with fee optimization enabled")
    
    def _initialize_default_configs(self) -> Dict[str, FeeConfig]:
        """Initialize default fee configurations."""
        return {
            'default_maker': FeeConfig(
                fee_type=FeeType.MAKER,
                structure=FeeStructure.PERCENTAGE,
                rate=self.default_maker_fee,
                minimum_fee=0.01,
                currency="USD"
            ),
            'default_taker': FeeConfig(
                fee_type=FeeType.TAKER,
                structure=FeeStructure.PERCENTAGE,
                rate=self.default_taker_fee,
                minimum_fee=0.01,
                currency="USD"
            ),
            'binance_spot': FeeConfig(
                fee_type=FeeType.TAKER,
                structure=FeeStructure.TIERED,
                rate=0.001,  # Base rate
                minimum_fee=0.01,
                currency="USD",
                tiers=[
                    {'volume_threshold': 0, 'maker_rate': 0.001, 'taker_rate': 0.001},
                    {'volume_threshold': 50, 'maker_rate': 0.0009, 'taker_rate': 0.001},
                    {'volume_threshold': 500, 'maker_rate': 0.0008, 'taker_rate': 0.001},
                    {'volume_threshold': 1000, 'maker_rate': 0.0007, 'taker_rate': 0.0009}
                ]
            ),
            'futures_funding': FeeConfig(
                fee_type=FeeType.FUNDING,
                structure=FeeStructure.PERCENTAGE,
                rate=0.0001,  # 0.01% every 8 hours
                currency="USD"
            )
        }
    
    def calculate_trading_fee(self,
                            trade_value: float,
                            asset: str,
                            is_maker: bool = False,
                            exchange: str = "default",
                            user_volume_30d: Optional[float] = None) -> FeeCalculationResult:
        """
        Calculate trading fee for a specific trade.
        
        Args:
            trade_value: Value of the trade in USD
            asset: Asset being traded
            is_maker: Whether this is a maker order
            exchange: Exchange identifier
            user_volume_30d: User's 30-day trading volume for discounts
            
        Returns:
            FeeCalculationResult
        """
        warnings = []
        fee_breakdown = {}
        
        try:
            # Determine fee configuration
            config_key = f"{exchange}_{'maker' if is_maker else 'taker'}"
            if config_key not in self.fee_configs:
                config_key = f"default_{'maker' if is_maker else 'taker'}"
            
            config = self.fee_configs[config_key]
            
            # Calculate base fee
            if config.structure == FeeStructure.PERCENTAGE:
                base_fee = trade_value * config.rate
                effective_rate = config.rate
                
            elif config.structure == FeeStructure.FIXED:
                base_fee = config.rate
                effective_rate = config.rate / trade_value if trade_value > 0 else 0
                
            elif config.structure == FeeStructure.TIERED:
                base_fee, effective_rate = self._calculate_tiered_fee(
                    trade_value, config, user_volume_30d, is_maker
                )
                
            else:
                base_fee = trade_value * self.default_taker_fee
                effective_rate = self.default_taker_fee
                warnings.append(f"Unknown fee structure: {config.structure}")
            
            fee_breakdown['base_fee'] = base_fee
            
            # Apply volume discounts
            if self.volume_discount_enabled and user_volume_30d:
                discount = self._calculate_volume_discount(user_volume_30d)
                if discount > 0:
                    discount_amount = base_fee * discount
                    fee_breakdown['volume_discount'] = -discount_amount
                    base_fee -= discount_amount
                    warnings.append(f"Applied {discount*100:.2f}% volume discount")
            
            # Apply minimum/maximum fee constraints
            if config.minimum_fee and base_fee < config.minimum_fee:
                fee_breakdown['minimum_fee_adjustment'] = config.minimum_fee - base_fee
                base_fee = config.minimum_fee
                warnings.append(f"Applied minimum fee: {config.minimum_fee}")
            
            if config.maximum_fee and base_fee > config.maximum_fee:
                fee_breakdown['maximum_fee_adjustment'] = config.maximum_fee - base_fee
                base_fee = config.maximum_fee
                warnings.append(f"Applied maximum fee cap: {config.maximum_fee}")
            
            # Asset-specific adjustments
            asset_adjustment = self._get_asset_fee_adjustment(asset)
            if asset_adjustment != 0:
                adjustment_amount = base_fee * asset_adjustment
                fee_breakdown['asset_adjustment'] = adjustment_amount
                base_fee += adjustment_amount
                warnings.append(f"Applied {asset} fee adjustment: {asset_adjustment*100:.2f}%")
            
            total_fee = base_fee
            
            # Update statistics
            self._update_fee_stats(total_fee, config.fee_type)
            
            return FeeCalculationResult(
                total_fee=total_fee,
                fee_breakdown=fee_breakdown,
                currency=config.currency,
                effective_rate=effective_rate,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Fee calculation failed: {e}")
            # Return fallback fee calculation
            fallback_fee = trade_value * self.default_taker_fee
            return FeeCalculationResult(
                total_fee=fallback_fee,
                fee_breakdown={'fallback_fee': fallback_fee},
                currency="USD",
                effective_rate=self.default_taker_fee,
                warnings=[f"Fee calculation error, using fallback: {str(e)}"]
            )
    
    def _calculate_tiered_fee(self,
                            trade_value: float,
                            config: FeeConfig,
                            user_volume_30d: Optional[float],
                            is_maker: bool) -> tuple[float, float]:
        """Calculate fee using tiered structure."""
        if not config.tiers or not user_volume_30d:
            return trade_value * config.rate, config.rate
        
        # Find appropriate tier
        applicable_tier = config.tiers[0]
        for tier in config.tiers:
            if user_volume_30d >= tier['volume_threshold']:
                applicable_tier = tier
            else:
                break
        
        rate_key = 'maker_rate' if is_maker else 'taker_rate'
        rate = applicable_tier.get(rate_key, config.rate)
        
        return trade_value * rate, rate
    
    def _calculate_volume_discount(self, volume_30d: float) -> float:
        """Calculate volume-based discount."""
        if volume_30d < 1000:
            return 0.0
        elif volume_30d < 10000:
            return 0.05  # 5% discount
        elif volume_30d < 50000:
            return 0.10  # 10% discount
        elif volume_30d < 100000:
            return 0.15  # 15% discount
        else:
            return 0.20  # 20% discount
    
    def _get_asset_fee_adjustment(self, asset: str) -> float:
        """Get asset-specific fee adjustment."""
        # Some assets might have higher/lower fees
        asset_adjustments = {
            'BTC': 0.0,      # No adjustment for BTC
            'ETH': 0.0,      # No adjustment for ETH
            'USDT': -0.1,    # 10% discount for stablecoins
            'USDC': -0.1,    # 10% discount for stablecoins
            'BUSD': -0.1,    # 10% discount for stablecoins
        }
        
        return asset_adjustments.get(asset.upper(), 0.0)
    
    def calculate_funding_fee(self,
                            position_value: float,
                            funding_rate: float,
                            hours_held: float = 8.0) -> FeeCalculationResult:
        """
        Calculate funding fee for futures positions.
        
        Args:
            position_value: Value of the position
            funding_rate: Current funding rate
            hours_held: Hours the position was held
            
        Returns:
            FeeCalculationResult
        """
        # Funding fees are typically charged every 8 hours
        funding_periods = hours_held / 8.0
        total_funding_fee = position_value * funding_rate * funding_periods
        
        fee_breakdown = {
            'funding_fee': total_funding_fee,
            'funding_rate': funding_rate,
            'periods': funding_periods
        }
        
        self._update_fee_stats(abs(total_funding_fee), FeeType.FUNDING)
        
        return FeeCalculationResult(
            total_fee=total_funding_fee,
            fee_breakdown=fee_breakdown,
            currency="USD",
            effective_rate=funding_rate,
            warnings=[]
        )
    
    def calculate_withdrawal_fee(self,
                               amount: float,
                               asset: str,
                               network: str = "default") -> FeeCalculationResult:
        """
        Calculate withdrawal fee.
        
        Args:
            amount: Amount to withdraw
            asset: Asset being withdrawn
            network: Network/blockchain for withdrawal
            
        Returns:
            FeeCalculationResult
        """
        # Network-specific fees
        network_fees = {
            'BTC': {'bitcoin': 0.0005, 'lightning': 0.000001},
            'ETH': {'ethereum': 0.005, 'polygon': 0.001, 'bsc': 0.001},
            'USDT': {'ethereum': 10.0, 'tron': 1.0, 'bsc': 1.0},
        }
        
        asset_upper = asset.upper()
        if asset_upper in network_fees and network in network_fees[asset_upper]:
            withdrawal_fee = network_fees[asset_upper][network]
        else:
            # Default percentage-based withdrawal fee
            withdrawal_fee = amount * self.default_withdrawal_fee
        
        fee_breakdown = {
            'withdrawal_fee': withdrawal_fee,
            'network': network,
            'asset': asset
        }
        
        self._update_fee_stats(withdrawal_fee, FeeType.WITHDRAWAL)
        
        return FeeCalculationResult(
            total_fee=withdrawal_fee,
            fee_breakdown=fee_breakdown,
            currency=asset,
            effective_rate=withdrawal_fee / amount if amount > 0 else 0,
            warnings=[]
        )
    
    def optimize_order_type(self,
                          trade_value: float,
                          urgency: float = 0.5) -> Dict[str, Any]:
        """
        Recommend order type to minimize fees.
        
        Args:
            trade_value: Value of the trade
            urgency: Urgency level (0-1, where 1 is most urgent)
            
        Returns:
            Dictionary with recommendations
        """
        if not self.enable_fee_optimization:
            return {'recommended_type': 'market', 'reason': 'Fee optimization disabled'}
        
        maker_fee = self.calculate_trading_fee(trade_value, "BTC", is_maker=True)
        taker_fee = self.calculate_trading_fee(trade_value, "BTC", is_maker=False)
        
        fee_savings = taker_fee.total_fee - maker_fee.total_fee
        
        # Decision based on urgency and potential savings
        if urgency > 0.8:
            return {
                'recommended_type': 'market',
                'reason': 'High urgency, immediate execution needed',
                'potential_savings': fee_savings
            }
        elif fee_savings > trade_value * 0.0005:  # If savings > 0.05%
            return {
                'recommended_type': 'limit',
                'reason': f'Potential fee savings: ${fee_savings:.4f}',
                'potential_savings': fee_savings
            }
        else:
            return {
                'recommended_type': 'market',
                'reason': 'Fee savings too small to justify delay',
                'potential_savings': fee_savings
            }
    
    def _update_fee_stats(self, fee_amount: float, fee_type: FeeType):
        """Update fee statistics."""
        self.fee_stats['total_fees_paid'] += fee_amount
        self.fee_stats['fees_by_type'][fee_type] += fee_amount
        self.fee_stats['total_trades'] += 1
    
    def get_fee_summary(self) -> Dict[str, Any]:
        """Get comprehensive fee summary."""
        return {
            'total_fees_paid': self.fee_stats['total_fees_paid'],
            'average_fee_per_trade': (
                self.fee_stats['total_fees_paid'] / self.fee_stats['total_trades']
                if self.fee_stats['total_trades'] > 0 else 0
            ),
            'fees_by_type': dict(self.fee_stats['fees_by_type']),
            'fee_optimization_savings': self.fee_stats['fee_optimization_savings'],
            'total_trades': self.fee_stats['total_trades']
        }
    
    def add_fee_config(self, name: str, config: FeeConfig):
        """Add custom fee configuration."""
        self.fee_configs[name] = config
        logger.info(f"Added fee configuration: {name}")
    
    def update_volume_history(self, user_id: str, volume: float):
        """Update user's volume history for discount calculation."""
        if user_id not in self.volume_history:
            self.volume_history[user_id] = []
        
        import time
        self.volume_history[user_id].append({
            'volume': volume,
            'timestamp': time.time()
        })
        
        # Keep only last 30 days
        cutoff_time = time.time() - (30 * 24 * 3600)
        self.volume_history[user_id] = [
            entry for entry in self.volume_history[user_id]
            if entry['timestamp'] > cutoff_time
        ]