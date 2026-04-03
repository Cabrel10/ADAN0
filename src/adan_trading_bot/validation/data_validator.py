"""
Data Validator - Validates calculated indicators against Binance reference data.

This module compares calculated RSI, ADX, ATR values against reference values from
Binance API to detect data corruption. It implements deviation thresholds (5% warning,
10% halt) and data freshness checks.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from ..indicators.calculator import IndicatorCalculator

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of indicator validation."""
    status: str  # "pass", "warning", "halt"
    rsi_deviation: float  # percentage
    adx_deviation: float  # percentage
    atr_deviation: float  # percentage
    reference_values: Dict[str, float]
    calculated_values: Dict[str, float]
    timestamp: datetime
    message: str


class DataValidator:
    """Validate calculated indicators against Binance reference data."""
    
    WARNING_THRESHOLD = 5.0  # 5% deviation triggers warning
    HALT_THRESHOLD = 10.0    # 10% deviation triggers halt
    MAX_DATA_AGE_MINUTES = 5  # Maximum age for fresh data
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize data validator.
        
        Args:
            api_key: Binance API key (optional for public data)
            api_secret: Binance API secret (optional for public data)
        """
        self.exchange = ccxt.binance({
            'apiKey': api_key or '',
            'secret': api_secret or '',
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        
        self.calculator = IndicatorCalculator()
        
        logger.info("DataValidator initialized")
    
    def get_reference_indicators(self, symbol: str = 'BTC/USDT', 
                               timeframe: str = '5m', limit: int = 100) -> Dict[str, float]:
        """
        Get reference indicator values from Binance.
        
        Args:
            symbol: Trading symbol (default BTC/USDT)
            timeframe: Timeframe for data (default 5m)
            limit: Number of candles to fetch (default 100)
            
        Returns:
            Dictionary with reference RSI, ADX, ATR values
            
        Raises:
            Exception: If unable to fetch data from Binance
        """
        try:
            logger.debug(f"Fetching reference data for {symbol} {timeframe}")
            
            # Fetch OHLCV data from Binance
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) < 30:
                raise ValueError(f"Insufficient data from Binance: {len(ohlcv) if ohlcv else 0} candles")
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate reference indicators
            reference = self.calculator.calculate_all(df)
            
            # Add metadata
            reference['data_timestamp'] = df['timestamp'].iloc[-1]
            reference['data_age_seconds'] = (datetime.utcnow() - df['timestamp'].iloc[-1].to_pydatetime()).total_seconds()
            
            logger.info(f"Reference indicators: RSI={reference['rsi']:.2f}, ADX={reference['adx']:.2f}, ATR={reference['atr']:.6f}")
            
            return reference
            
        except Exception as e:
            logger.error(f"Failed to get reference indicators: {e}")
            raise
    
    def validate_indicators(self, calculated: Dict[str, float], 
                          reference: Dict[str, float]) -> ValidationResult:
        """
        Validate calculated indicators against reference values.
        
        Args:
            calculated: Dictionary with calculated indicator values
            reference: Dictionary with reference indicator values
            
        Returns:
            ValidationResult with status and deviation details
        """
        timestamp = datetime.utcnow()
        
        # Calculate deviations
        rsi_deviation = self._calculate_deviation(calculated.get('rsi', 0), reference.get('rsi', 0))
        adx_deviation = self._calculate_deviation(calculated.get('adx', 0), reference.get('adx', 0))
        atr_deviation = self._calculate_deviation(calculated.get('atr', 0), reference.get('atr', 0))
        
        # Determine status
        max_deviation = max(abs(rsi_deviation), abs(adx_deviation), abs(atr_deviation))
        
        if max_deviation >= self.HALT_THRESHOLD:
            status = "halt"
            message = f"CRITICAL: Maximum deviation {max_deviation:.1f}% exceeds halt threshold {self.HALT_THRESHOLD}%"
        elif max_deviation >= self.WARNING_THRESHOLD:
            status = "warning"
            message = f"WARNING: Maximum deviation {max_deviation:.1f}% exceeds warning threshold {self.WARNING_THRESHOLD}%"
        else:
            status = "pass"
            message = f"PASS: Maximum deviation {max_deviation:.1f}% within acceptable range"
        
        result = ValidationResult(
            status=status,
            rsi_deviation=rsi_deviation,
            adx_deviation=adx_deviation,
            atr_deviation=atr_deviation,
            reference_values=reference.copy(),
            calculated_values=calculated.copy(),
            timestamp=timestamp,
            message=message
        )
        
        # Log result
        if status == "halt":
            logger.critical(f"HALT: {message}")
            logger.critical(f"Deviations - RSI: {rsi_deviation:.1f}%, ADX: {adx_deviation:.1f}%, ATR: {atr_deviation:.1f}%")
        elif status == "warning":
            logger.warning(f"WARNING: {message}")
            logger.warning(f"Deviations - RSI: {rsi_deviation:.1f}%, ADX: {adx_deviation:.1f}%, ATR: {atr_deviation:.1f}%")
        else:
            logger.info(f"VALIDATION PASS: {message}")
            logger.debug(f"Deviations - RSI: {rsi_deviation:.1f}%, ADX: {adx_deviation:.1f}%, ATR: {atr_deviation:.1f}%")
        
        return result
    
    def check_data_freshness(self, timestamp: datetime) -> bool:
        """
        Check if data is fresh (not older than MAX_DATA_AGE_MINUTES).
        
        Args:
            timestamp: Timestamp of the data
            
        Returns:
            True if data is fresh, False if stale
        """
        now = datetime.utcnow()
        age_minutes = (now - timestamp).total_seconds() / 60
        
        is_fresh = age_minutes <= self.MAX_DATA_AGE_MINUTES
        
        if not is_fresh:
            logger.warning(f"Stale data detected: {age_minutes:.1f} minutes old (max {self.MAX_DATA_AGE_MINUTES})")
        else:
            logger.debug(f"Data is fresh: {age_minutes:.1f} minutes old")
        
        return is_fresh
    
    def check_mock_data_usage(self) -> bool:
        """
        Check if system is using mock or test data.
        
        Returns:
            True if using real data, False if using mock/test data
        """
        try:
            # Try to fetch a small amount of real data
            test_data = self.exchange.fetch_ohlcv('BTC/USDT', '1h', limit=1)
            
            if not test_data:
                logger.error("No data returned from Binance - possible mock data usage")
                return False
            
            # Check if data looks realistic
            price = test_data[0][4]  # Close price
            if price < 1000 or price > 1000000:  # Unrealistic BTC price
                logger.error(f"Unrealistic BTC price {price} - possible mock data")
                return False
            
            logger.debug(f"Real data confirmed: BTC price {price}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify real data usage: {e}")
            return False
    
    def validate_full_pipeline(self, symbol: str = 'BTC/USDT', 
                             calculated_indicators: Optional[Dict[str, float]] = None) -> ValidationResult:
        """
        Run complete validation pipeline.
        
        Args:
            symbol: Trading symbol to validate
            calculated_indicators: Pre-calculated indicators to validate (optional)
            
        Returns:
            ValidationResult with complete validation status
        """
        logger.info(f"Starting full validation pipeline for {symbol}")
        
        try:
            # Check mock data usage
            if not self.check_mock_data_usage():
                return ValidationResult(
                    status="halt",
                    rsi_deviation=0,
                    adx_deviation=0,
                    atr_deviation=0,
                    reference_values={},
                    calculated_values=calculated_indicators or {},
                    timestamp=datetime.utcnow(),
                    message="HALT: Mock or test data detected in production"
                )
            
            # Get reference indicators
            reference = self.get_reference_indicators(symbol)
            
            # Check data freshness
            if 'data_timestamp' in reference:
                if not self.check_data_freshness(reference['data_timestamp']):
                    return ValidationResult(
                        status="warning",
                        rsi_deviation=0,
                        adx_deviation=0,
                        atr_deviation=0,
                        reference_values=reference,
                        calculated_values=calculated_indicators or {},
                        timestamp=datetime.utcnow(),
                        message="WARNING: Reference data is stale"
                    )
            
            # If no calculated indicators provided, use reference as baseline
            if calculated_indicators is None:
                logger.info("No calculated indicators provided, returning reference values")
                return ValidationResult(
                    status="pass",
                    rsi_deviation=0,
                    adx_deviation=0,
                    atr_deviation=0,
                    reference_values=reference,
                    calculated_values=reference,
                    timestamp=datetime.utcnow(),
                    message="PASS: Reference indicators validated"
                )
            
            # Validate calculated vs reference
            return self.validate_indicators(calculated_indicators, reference)
            
        except Exception as e:
            logger.error(f"Validation pipeline failed: {e}")
            return ValidationResult(
                status="halt",
                rsi_deviation=0,
                adx_deviation=0,
                atr_deviation=0,
                reference_values={},
                calculated_values=calculated_indicators or {},
                timestamp=datetime.utcnow(),
                message=f"HALT: Validation pipeline error: {e}"
            )
    
    def _calculate_deviation(self, calculated: float, reference: float) -> float:
        """
        Calculate percentage deviation between calculated and reference values.
        
        Args:
            calculated: Calculated value
            reference: Reference value
            
        Returns:
            Percentage deviation (positive if calculated > reference)
        """
        if reference == 0:
            return 0.0 if calculated == 0 else 100.0
        
        return ((calculated - reference) / reference) * 100.0
    
    def export_diagnostic_report(self, result: ValidationResult, filepath: str) -> None:
        """
        Export detailed diagnostic report to file.
        
        Args:
            result: ValidationResult to export
            filepath: Path to save the report
        """
        try:
            report = {
                'timestamp': result.timestamp.isoformat(),
                'status': result.status,
                'message': result.message,
                'deviations': {
                    'rsi': result.rsi_deviation,
                    'adx': result.adx_deviation,
                    'atr': result.atr_deviation
                },
                'calculated_values': result.calculated_values,
                'reference_values': result.reference_values,
                'thresholds': {
                    'warning': self.WARNING_THRESHOLD,
                    'halt': self.HALT_THRESHOLD
                }
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Diagnostic report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export diagnostic report: {e}")
