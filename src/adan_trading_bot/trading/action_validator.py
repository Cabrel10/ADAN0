"""
Action validation configuration module for the ADAN Trading Bot.

This module provides comprehensive action validation with configurable rules,
constraints, and validation strategies for different market conditions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from .action_translator import TradingAction, ActionType

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation rules."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationStrategy(Enum):
    """Validation strategies for different scenarios."""
    STRICT = "strict"          # Reject any violations
    LENIENT = "lenient"        # Allow warnings, reject errors
    ADAPTIVE = "adaptive"      # Adjust based on market conditions
    PERMISSIVE = "permissive"  # Allow most actions with warnings


@dataclass
class ValidationRule:
    """Represents a single validation rule."""
    name: str
    description: str
    severity: ValidationSeverity
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    validator_func: Optional[Callable] = None


@dataclass
class ValidationResult:
    """Result of action validation."""
    is_valid: bool
    severity: ValidationSeverity
    rule_name: str
    message: str
    suggested_fix: Optional[str] = None
    adjusted_value: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationSummary:
    """Summary of all validation results."""
    overall_valid: bool
    total_rules_checked: int
    passed_rules: int
    failed_rules: int
    results: List[ValidationResult]
    adjusted_action: Optional[TradingAction] = None
    execution_time_ms: float = 0.0


class ActionValidator:
    """
    Comprehensive action validator with configurable rules and strategies.
    
    This class provides flexible validation of trading actions with support
    for different validation strategies and market-specific rules.
    """
    
    def __init__(self,
                 strategy: ValidationStrategy = ValidationStrategy.LENIENT,
                 config_file: Optional[str] = None):
        """
        Initialize the ActionValidator.
        
        Args:
            strategy: Default validation strategy
            config_file: Path to validation configuration file
        """
        self.strategy = strategy
        self.rules = {}
        self.market_conditions = {}
        
        # Validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'rule_violations': {},
            'strategy_usage': {s.value: 0 for s in ValidationStrategy}
        }
        
        # Initialize default rules
        self._initialize_default_rules()
        
        # Load configuration if provided
        if config_file:
            self.load_config(config_file)
        
        logger.info(f"ActionValidator initialized with {strategy.value} strategy")
    
    def _initialize_default_rules(self):
        """Initialize default validation rules."""
        
        # Position size rules
        self.add_rule(ValidationRule(
            name="min_position_size",
            description="Minimum position size constraint",
            severity=ValidationSeverity.ERROR,
            parameters={'min_size': 0.001},
            validator_func=self._validate_min_position_size
        ))
        
        self.add_rule(ValidationRule(
            name="max_position_size",
            description="Maximum position size constraint",
            severity=ValidationSeverity.ERROR,
            parameters={'max_size': 1.0},
            validator_func=self._validate_max_position_size
        ))
        
        # Capital rules
        self.add_rule(ValidationRule(
            name="sufficient_capital",
            description="Sufficient capital for trade execution",
            severity=ValidationSeverity.ERROR,
            parameters={'safety_margin': 0.05},
            validator_func=self._validate_sufficient_capital
        ))
        
        # Risk management rules
        self.add_rule(ValidationRule(
            name="stop_loss_validation",
            description="Stop loss level validation",
            severity=ValidationSeverity.WARNING,
            parameters={'max_stop_loss_pct': 0.1},
            validator_func=self._validate_stop_loss
        ))
        
        self.add_rule(ValidationRule(
            name="take_profit_validation",
            description="Take profit level validation",
            severity=ValidationSeverity.WARNING,
            parameters={'min_take_profit_pct': 0.01},
            validator_func=self._validate_take_profit
        ))
        
        # Position management rules
        self.add_rule(ValidationRule(
            name="position_existence",
            description="Validate position exists for closing actions",
            severity=ValidationSeverity.ERROR,
            parameters={},
            validator_func=self._validate_position_existence
        ))
        
        # Market condition rules
        self.add_rule(ValidationRule(
            name="market_hours",
            description="Validate trading during market hours",
            severity=ValidationSeverity.WARNING,
            parameters={'enforce_market_hours': False},
            validator_func=self._validate_market_hours
        ))
        
        # Volatility rules
        self.add_rule(ValidationRule(
            name="volatility_adjustment",
            description="Adjust position size based on volatility",
            severity=ValidationSeverity.INFO,
            parameters={'volatility_threshold': 0.05},
            validator_func=self._validate_volatility_adjustment
        ))
        
        # Correlation rules
        self.add_rule(ValidationRule(
            name="correlation_limit",
            description="Limit correlated positions",
            severity=ValidationSeverity.WARNING,
            parameters={'max_correlation': 0.8},
            validator_func=self._validate_correlation_limit
        ))
        
        # Frequency rules
        self.add_rule(ValidationRule(
            name="trade_frequency",
            description="Limit trade frequency to prevent overtrading",
            severity=ValidationSeverity.WARNING,
            parameters={'max_trades_per_hour': 10},
            validator_func=self._validate_trade_frequency
        ))
    
    def add_rule(self, rule: ValidationRule):
        """Add a validation rule."""
        self.rules[rule.name] = rule
        if rule.name not in self.validation_stats['rule_violations']:
            self.validation_stats['rule_violations'][rule.name] = 0
        logger.debug(f"Added validation rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove a validation rule."""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.debug(f"Removed validation rule: {rule_name}")
    
    def enable_rule(self, rule_name: str):
        """Enable a validation rule."""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = True
            logger.debug(f"Enabled validation rule: {rule_name}")
    
    def disable_rule(self, rule_name: str):
        """Disable a validation rule."""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = False
            logger.debug(f"Disabled validation rule: {rule_name}")
    
    def validate_action(self,
                       action: TradingAction,
                       portfolio_state: Dict[str, Any],
                       market_data: Optional[Dict[str, Any]] = None,
                       strategy: Optional[ValidationStrategy] = None) -> ValidationSummary:
        """
        Validate a trading action against all enabled rules.
        
        Args:
            action: Trading action to validate
            portfolio_state: Current portfolio state
            market_data: Market data and conditions
            strategy: Validation strategy to use (overrides default)
            
        Returns:
            ValidationSummary with results
        """
        import time
        start_time = time.time()
        
        validation_strategy = strategy or self.strategy
        self.validation_stats['total_validations'] += 1
        self.validation_stats['strategy_usage'][validation_strategy.value] += 1
        
        results = []
        adjusted_action = None
        
        # Update market conditions
        if market_data:
            self.market_conditions.update(market_data)
        
        # Run all enabled rules
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                if rule.validator_func:
                    result = rule.validator_func(action, portfolio_state, market_data, rule.parameters)
                    if result:
                        results.append(result)
                        if not result.is_valid:
                            self.validation_stats['rule_violations'][rule_name] += 1
                        
                        # Handle adjustments
                        if result.adjusted_value and not adjusted_action:
                            adjusted_action = self._create_adjusted_action(action, result)
                            
            except Exception as e:
                logger.error(f"Error in validation rule {rule_name}: {e}")
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    rule_name=rule_name,
                    message=f"Validation rule failed: {str(e)}"
                ))
        
        # Determine overall validity based on strategy
        overall_valid = self._determine_overall_validity(results, validation_strategy)
        
        # Update statistics
        if overall_valid:
            self.validation_stats['passed_validations'] += 1
        else:
            self.validation_stats['failed_validations'] += 1
        
        execution_time = (time.time() - start_time) * 1000
        
        return ValidationSummary(
            overall_valid=overall_valid,
            total_rules_checked=len([r for r in self.rules.values() if r.enabled]),
            passed_rules=len([r for r in results if r.is_valid]),
            failed_rules=len([r for r in results if not r.is_valid]),
            results=results,
            adjusted_action=adjusted_action,
            execution_time_ms=execution_time
        )
    
    def _determine_overall_validity(self, results: List[ValidationResult], strategy: ValidationStrategy) -> bool:
        """Determine overall validity based on strategy and results."""
        
        if strategy == ValidationStrategy.PERMISSIVE:
            # Only reject on critical errors
            return not any(r.severity == ValidationSeverity.CRITICAL and not r.is_valid for r in results)
        
        elif strategy == ValidationStrategy.LENIENT:
            # Reject on errors and critical, allow warnings
            return not any(r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                          and not r.is_valid for r in results)
        
        elif strategy == ValidationStrategy.STRICT:
            # Reject on any failure
            return all(r.is_valid for r in results)
        
        elif strategy == ValidationStrategy.ADAPTIVE:
            # Adjust based on market conditions
            volatility = self.market_conditions.get('volatility', 0.02)
            if volatility > 0.05:  # High volatility - be more lenient
                return not any(r.severity == ValidationSeverity.CRITICAL and not r.is_valid for r in results)
            else:  # Normal conditions - standard validation
                return not any(r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                              and not r.is_valid for r in results)
        
        return False
    
    def _create_adjusted_action(self, original_action: TradingAction, result: ValidationResult) -> TradingAction:
        """Create an adjusted action based on validation result."""
        # This is a simplified implementation - in practice, you'd handle different adjustment types
        adjusted_action = TradingAction(
            action_type=original_action.action_type,
            asset=original_action.asset,
            size=result.adjusted_value if isinstance(result.adjusted_value, (int, float)) else original_action.size,
            price=original_action.price,
            stop_loss=original_action.stop_loss,
            take_profit=original_action.take_profit,
            confidence=original_action.confidence,
            metadata={
                **(original_action.metadata or {}),
                'adjusted_by_rule': result.rule_name,
                'adjustment_reason': result.message
            }
        )
        return adjusted_action
    
    # Validation rule implementations
    def _validate_min_position_size(self, action: TradingAction, portfolio_state: Dict[str, Any], 
                                   market_data: Optional[Dict[str, Any]], params: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate minimum position size."""
        min_size = params.get('min_size', 0.001)
        
        if action.action_type == ActionType.HOLD:
            return None
        
        if action.size < min_size:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_name="min_position_size",
                message=f"Position size {action.size:.6f} below minimum {min_size:.6f}",
                suggested_fix=f"Increase position size to at least {min_size:.6f}"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            rule_name="min_position_size",
            message="Position size meets minimum requirement"
        )
    
    def _validate_max_position_size(self, action: TradingAction, portfolio_state: Dict[str, Any], 
                                   market_data: Optional[Dict[str, Any]], params: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate maximum position size."""
        max_size = params.get('max_size', 1.0)
        
        if action.action_type == ActionType.HOLD:
            return None
        
        if action.size > max_size:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_name="max_position_size",
                message=f"Position size {action.size:.6f} exceeds maximum {max_size:.6f}",
                suggested_fix=f"Reduce position size to maximum {max_size:.6f}",
                adjusted_value=max_size
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            rule_name="max_position_size",
            message="Position size within maximum limit"
        )
    
    def _validate_sufficient_capital(self, action: TradingAction, portfolio_state: Dict[str, Any], 
                                    market_data: Optional[Dict[str, Any]], params: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate sufficient capital for trade."""
        if action.action_type not in [ActionType.BUY]:
            return None
        
        safety_margin = params.get('safety_margin', 0.05)
        available_capital = portfolio_state.get('cash', 0.0)
        required_capital = action.size * (action.price or 0)
        required_with_margin = required_capital * (1 + safety_margin)
        
        if available_capital < required_with_margin:
            max_affordable = available_capital / ((action.price or 1) * (1 + safety_margin))
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_name="sufficient_capital",
                message=f"Insufficient capital: need {required_with_margin:.2f}, have {available_capital:.2f}",
                suggested_fix=f"Reduce position size to {max_affordable:.6f}",
                adjusted_value=max_affordable
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            rule_name="sufficient_capital",
            message="Sufficient capital available"
        )
    
    def _validate_stop_loss(self, action: TradingAction, portfolio_state: Dict[str, Any], 
                           market_data: Optional[Dict[str, Any]], params: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate stop loss levels."""
        if not action.stop_loss or not action.price:
            return None
        
        max_stop_loss_pct = params.get('max_stop_loss_pct', 0.1)
        
        if action.action_type == ActionType.BUY:
            stop_loss_pct = (action.price - action.stop_loss) / action.price
        elif action.action_type == ActionType.SELL:
            stop_loss_pct = (action.stop_loss - action.price) / action.price
        else:
            return None
        
        if stop_loss_pct > max_stop_loss_pct:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                rule_name="stop_loss_validation",
                message=f"Stop loss too wide: {stop_loss_pct:.2%} > {max_stop_loss_pct:.2%}",
                suggested_fix=f"Tighten stop loss to maximum {max_stop_loss_pct:.2%}"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            rule_name="stop_loss_validation",
            message="Stop loss level appropriate"
        )
    
    def _validate_take_profit(self, action: TradingAction, portfolio_state: Dict[str, Any], 
                             market_data: Optional[Dict[str, Any]], params: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate take profit levels."""
        if not action.take_profit or not action.price:
            return None
        
        min_take_profit_pct = params.get('min_take_profit_pct', 0.01)
        
        if action.action_type == ActionType.BUY:
            take_profit_pct = (action.take_profit - action.price) / action.price
        elif action.action_type == ActionType.SELL:
            take_profit_pct = (action.price - action.take_profit) / action.price
        else:
            return None
        
        if take_profit_pct < min_take_profit_pct:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                rule_name="take_profit_validation",
                message=f"Take profit too narrow: {take_profit_pct:.2%} < {min_take_profit_pct:.2%}",
                suggested_fix=f"Widen take profit to minimum {min_take_profit_pct:.2%}"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            rule_name="take_profit_validation",
            message="Take profit level appropriate"
        )
    
    def _validate_position_existence(self, action: TradingAction, portfolio_state: Dict[str, Any], 
                                    market_data: Optional[Dict[str, Any]], params: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate position exists for closing actions."""
        if action.action_type not in [ActionType.CLOSE_LONG, ActionType.CLOSE_SHORT]:
            return None
        
        current_positions = portfolio_state.get('positions', {})
        current_position = current_positions.get(action.asset, 0.0)
        
        if action.action_type == ActionType.CLOSE_LONG and current_position <= 0:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_name="position_existence",
                message=f"Cannot close long position: no long position in {action.asset}",
                suggested_fix="Change action to BUY or select different asset"
            )
        
        if action.action_type == ActionType.CLOSE_SHORT and current_position >= 0:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_name="position_existence",
                message=f"Cannot close short position: no short position in {action.asset}",
                suggested_fix="Change action to SELL or select different asset"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            rule_name="position_existence",
            message="Position exists for closing action"
        )
    
    def _validate_market_hours(self, action: TradingAction, portfolio_state: Dict[str, Any], 
                              market_data: Optional[Dict[str, Any]], params: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate trading during market hours."""
        enforce_market_hours = params.get('enforce_market_hours', False)
        
        if not enforce_market_hours:
            return None
        
        # This is a simplified implementation - in practice, you'd check actual market hours
        import datetime
        current_hour = datetime.datetime.now().hour
        
        # Assume market hours are 9 AM to 4 PM
        if not (9 <= current_hour <= 16):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                rule_name="market_hours",
                message="Trading outside market hours",
                suggested_fix="Wait for market hours or disable this rule for 24/7 markets"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            rule_name="market_hours",
            message="Trading within market hours"
        )
    
    def _validate_volatility_adjustment(self, action: TradingAction, portfolio_state: Dict[str, Any], 
                                       market_data: Optional[Dict[str, Any]], params: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate volatility-based position adjustment."""
        if not market_data or 'volatility' not in market_data:
            return None
        
        volatility_threshold = params.get('volatility_threshold', 0.05)
        volatility = market_data['volatility']
        
        if volatility > volatility_threshold:
            # Suggest reducing position size in high volatility
            adjustment_factor = volatility_threshold / volatility
            adjusted_size = action.size * adjustment_factor
            
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                rule_name="volatility_adjustment",
                message=f"High volatility detected ({volatility:.2%}), consider reducing position size",
                suggested_fix=f"Reduce position size to {adjusted_size:.6f}",
                adjusted_value=adjusted_size,
                metadata={'volatility': volatility, 'threshold': volatility_threshold}
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            rule_name="volatility_adjustment",
            message="Volatility within acceptable range"
        )
    
    def _validate_correlation_limit(self, action: TradingAction, portfolio_state: Dict[str, Any], 
                                   market_data: Optional[Dict[str, Any]], params: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate correlation limits between positions."""
        # This is a simplified implementation - in practice, you'd calculate actual correlations
        max_correlation = params.get('max_correlation', 0.8)
        
        # Placeholder implementation
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            rule_name="correlation_limit",
            message="Correlation check passed (placeholder implementation)"
        )
    
    def _validate_trade_frequency(self, action: TradingAction, portfolio_state: Dict[str, Any], 
                                 market_data: Optional[Dict[str, Any]], params: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate trade frequency to prevent overtrading."""
        # This is a simplified implementation - in practice, you'd track actual trade history
        max_trades_per_hour = params.get('max_trades_per_hour', 10)
        
        # Placeholder implementation
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            rule_name="trade_frequency",
            message="Trade frequency check passed (placeholder implementation)"
        )
    
    def load_config(self, config_file: str):
        """Load validation configuration from file."""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Update strategy
            if 'strategy' in config:
                self.strategy = ValidationStrategy(config['strategy'])
            
            # Update rule parameters
            if 'rules' in config:
                for rule_name, rule_config in config['rules'].items():
                    if rule_name in self.rules:
                        if 'enabled' in rule_config:
                            self.rules[rule_name].enabled = rule_config['enabled']
                        if 'parameters' in rule_config:
                            self.rules[rule_name].parameters.update(rule_config['parameters'])
                        if 'severity' in rule_config:
                            self.rules[rule_name].severity = ValidationSeverity(rule_config['severity'])
            
            logger.info(f"Loaded validation configuration from {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load validation configuration: {e}")
    
    def save_config(self, config_file: str):
        """Save current validation configuration to file."""
        try:
            config = {
                'strategy': self.strategy.value,
                'rules': {}
            }
            
            for rule_name, rule in self.rules.items():
                config['rules'][rule_name] = {
                    'enabled': rule.enabled,
                    'severity': rule.severity.value,
                    'parameters': rule.parameters,
                    'description': rule.description
                }
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved validation configuration to {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save validation configuration: {e}")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        stats = self.validation_stats.copy()
        
        # Add success rate
        total = stats['total_validations']
        if total > 0:
            stats['success_rate'] = stats['passed_validations'] / total * 100
            stats['failure_rate'] = stats['failed_validations'] / total * 100
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset validation statistics."""
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'rule_violations': {rule_name: 0 for rule_name in self.rules.keys()},
            'strategy_usage': {s.value: 0 for s in ValidationStrategy}
        }
        logger.info("Validation statistics reset")
    
    def get_rule_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all validation rules."""
        return {
            rule_name: {
                'description': rule.description,
                'severity': rule.severity.value,
                'enabled': rule.enabled,
                'parameters': rule.parameters
            }
            for rule_name, rule in self.rules.items()
        }