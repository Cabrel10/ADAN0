"""
Indicator Calculator - Correct RSI, ADX, ATR formulas using the standard pandas_ta library.

This module uses the industry-standard pandas_ta library to ensure calculations are
correct, robust, and align with common trading platforms.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class IndicatorCalculator:
    """Calculate technical indicators using the pandas_ta library."""
    
    DEFAULT_PERIOD = 14

    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = DEFAULT_PERIOD) -> float:
        """Calculate RSI using a standard Wilder-style formula."""
        if not isinstance(close, pd.Series):
            close = pd.Series(close)

        if len(close) < period + 1:
            return 50.0

        try:
            rsi_series = ta.rsi(close, length=period)
            if rsi_series is None or rsi_series.empty:
                return 50.0
            return float(rsi_series.iloc[-1])
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = DEFAULT_PERIOD) -> float:
        """Calculate ATR using pandas_ta."""
        try:
            atr_series = ta.atr(high, low, close, length=period)
            if atr_series is None or atr_series.empty:
                return 0.0
            return float(atr_series.iloc[-1])
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0.0

    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = DEFAULT_PERIOD) -> float:
        """Calculate ADX using pandas_ta."""
        try:
            if len(high) < 2 * period:
                 return 25.0

            adx_df = ta.adx(high=high, low=low, close=close, length=period)
            if adx_df is None or adx_df.empty:
                return 25.0
            
            # ADX is usually the first column (ADX_14)
            return float(adx_df.iloc[-1, 0])
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return 25.0

    @staticmethod
    def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
        """Calculate MACD Line using pandas_ta."""
        try:
            macd_df = ta.macd(close, fast=fast, slow=slow, signal=signal)
            if macd_df is None or macd_df.empty:
                return 0.0
            # MACD line is the first column (MACD_12_26_9)
            return float(macd_df.iloc[-1, 0])
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return 0.0

    @staticmethod
    def calculate_bb_percent_b(close: pd.Series, length: int = 20, std: float = 2.0) -> float:
        """Calculate Bollinger Bands %B."""
        try:
            bb_df = ta.bbands(close, length=length, std=std)
            if bb_df is None or bb_df.empty:
                return 0.5
            # %B is usually the 5th column (BBP_20_2.0)
            # Columns: BBL, BBM, BBU, BBB, BBP
            return float(bb_df.iloc[-1, 4])
        except Exception as e:
            logger.error(f"Error calculating BB %B: {e}")
            return 0.5

    @staticmethod
    def calculate_bb_width(close: pd.Series, length: int = 20, std: float = 2.0) -> float:
        """Calculate Bollinger Bands Width."""
        try:
            bb_df = ta.bbands(close, length=length, std=std)
            if bb_df is None or bb_df.empty:
                return 0.0
            # Bandwidth is usually the 4th column (BBB_20_2.0)
            return float(bb_df.iloc[-1, 3])
        except Exception as e:
            logger.error(f"Error calculating BB Width: {e}")
            return 0.0

    @staticmethod
    def calculate_stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3, smooth_k: int = 3) -> float:
        """Calculate Stochastic %K."""
        try:
            stoch_df = ta.stoch(high, low, close, k=k, d=d, smooth_k=smooth_k)
            if stoch_df is None or stoch_df.empty:
                return 50.0
            # %K is the first column (STOCHk_14_3_3)
            return float(stoch_df.iloc[-1, 0])
        except Exception as e:
            logger.error(f"Error calculating Stoch K: {e}")
            return 50.0

    @staticmethod
    def calculate_supertrend(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 10, multiplier: float = 3.0) -> float:
        """Calculate Supertrend value (trend direction or value)."""
        try:
            st_df = ta.supertrend(high, low, close, length=length, multiplier=multiplier)
            if st_df is None or st_df.empty:
                return 0.0
            # Supertrend value is first column, direction is second
            return float(st_df.iloc[-1, 0])
        except Exception as e:
            logger.error(f"Error calculating Supertrend: {e}")
            return 0.0

    @staticmethod
    def calculate_ichimoku_base(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Calculate Ichimoku Base Line (Kijun-sen)."""
        try:
            # Standard Ichimoku settings: 9, 26, 52
            ichi_df, _ = ta.ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52)
            if ichi_df is None or ichi_df.empty:
                return 0.0
            # Kijun-sen (Base Line) is usually named IKS_26
            col_name = 'IKS_26'
            if col_name in ichi_df.columns:
                return float(ichi_df[col_name].iloc[-1])
            return float(ichi_df.iloc[-1, 1]) # Fallback index
        except Exception as e:
            logger.error(f"Error calculating Ichimoku: {e}")
            return 0.0

    @staticmethod
    def calculate_donchian_width(high: pd.Series, low: pd.Series, length: int = 20) -> float:
        """Calculate Donchian Channel Width."""
        try:
            donch_df = ta.donchian(high, low, lower_length=length, upper_length=length)
            if donch_df is None or donch_df.empty:
                return 0.0
            # Width = Upper - Lower
            # Columns: DCL, DCM, DCU
            upper = donch_df.iloc[-1, 2]
            lower = donch_df.iloc[-1, 0]
            return float(upper - lower)
        except Exception as e:
            logger.error(f"Error calculating Donchian Width: {e}")
            return 0.0

    @staticmethod
    def calculate_pivot_level(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Calculate Pivot Point (Classic)."""
        try:
            # Pivot = (High + Low + Close) / 3
            # Using previous candle for current pivot level
            if len(close) < 2:
                return float(close.iloc[-1])
            
            h = float(high.iloc[-2])
            l = float(low.iloc[-2])
            c = float(close.iloc[-2])
            return (h + l + c) / 3.0
        except Exception as e:
            logger.error(f"Error calculating Pivot: {e}")
            return 0.0

    @staticmethod
    def calculate_all_features(ohlcv: pd.DataFrame, timeframe: str) -> Dict[str, float]:
        """
        Calculate ALL features required for a specific timeframe.
        
        Args:
            ohlcv: DataFrame with columns: open, high, low, close, volume
            timeframe: '5m', '1h', or '4h'
            
        Returns:
            Dictionary with all required features for the timeframe
        """
        try:
            # Basic data
            close = ohlcv['close']
            high = ohlcv['high']
            low = ohlcv['low']
            volume = ohlcv['volume']
            current_price = float(close.iloc[-1])
            
            # Common indicators
            rsi_14 = IndicatorCalculator.calculate_rsi(close, 14)
            atr_14 = IndicatorCalculator.calculate_atr(high, low, close, 14)
            atr_20 = IndicatorCalculator.calculate_atr(high, low, close, 20)
            atr_50 = IndicatorCalculator.calculate_atr(high, low, close, 50)
            
            features = {
                'open': float(ohlcv['open'].iloc[-1]),
                'high': float(high.iloc[-1]),
                'low': float(low.iloc[-1]),
                'close': current_price,
                'volume': float(volume.iloc[-1]),
            }

            if timeframe == '5m':
                # 5m Features (14 total)
                # 1-5: OHLCV (already added)
                # 6: RSI_14
                features['rsi_14'] = rsi_14
                # 7: MACD_12_26_9
                features['macd_12_26_9'] = IndicatorCalculator.calculate_macd(close, 12, 26, 9)
                # 8: BB_PERCENT_B_20_2
                features['bb_percent_b_20_2'] = IndicatorCalculator.calculate_bb_percent_b(close, 20, 2.0)
                # 9: ATR_14
                features['atr_14'] = atr_14
                # 10: ATR_20
                features['atr_20'] = atr_20
                # 11: ATR_50
                features['atr_50'] = atr_50
                # 12: VOLUME_RATIO_20 (Vol / SMA20)
                vol_sma = volume.rolling(20).mean().iloc[-1]
                features['volume_ratio_20'] = float(volume.iloc[-1] / vol_sma) if vol_sma > 0 else 1.0
                # 13: EMA_20_RATIO (Price / EMA20)
                ema_20 = ta.ema(close, length=20).iloc[-1]
                features['ema_20_ratio'] = float(current_price / ema_20) if ema_20 > 0 else 1.0
                # 14: STOCH_K_14_3_3
                features['stoch_k_14_3_3'] = IndicatorCalculator.calculate_stoch_k(high, low, close, 14, 3, 3)

            elif timeframe == '1h':
                # 1h Features (14 total)
                # 1-5: OHLCV
                # 6: RSI_21
                features['rsi_21'] = IndicatorCalculator.calculate_rsi(close, 21)
                # 7: MACD_21_42_9
                features['macd_21_42_9'] = IndicatorCalculator.calculate_macd(close, 21, 42, 9)
                # 8: BB_WIDTH_20_2
                features['bb_width_20_2'] = IndicatorCalculator.calculate_bb_width(close, 20, 2.0)
                # 9: ADX_14
                features['adx_14'] = IndicatorCalculator.calculate_adx(high, low, close, 14)
                # 10: ATR_20
                features['atr_20'] = atr_20
                # 11: ATR_50
                features['atr_50'] = atr_50
                # 12: OBV_RATIO_20 (OBV / SMA20(OBV)) - Simplified as OBV slope proxy or just OBV normalized? 
                # Checking training data: it's likely OBV / SMA(OBV)
                obv = ta.obv(close, volume)
                obv_sma = obv.rolling(20).mean().iloc[-1]
                features['obv_ratio_20'] = float(obv.iloc[-1] / obv_sma) if obv_sma != 0 else 1.0
                # 13: EMA_50_RATIO
                ema_50 = ta.ema(close, length=50).iloc[-1]
                features['ema_50_ratio'] = float(current_price / ema_50) if ema_50 > 0 else 1.0
                # 14: ICHIMOKU_BASE
                features['ichimoku_base'] = IndicatorCalculator.calculate_ichimoku_base(high, low, close)

            elif timeframe == '4h':
                # 4h Features (14 total)
                # 1-5: OHLCV
                # 6: RSI_28
                features['rsi_28'] = IndicatorCalculator.calculate_rsi(close, 28)
                # 7: MACD_26_52_18
                features['macd_26_52_18'] = IndicatorCalculator.calculate_macd(close, 26, 52, 18)
                # 8: SUPERTREND_10_3
                features['supertrend_10_3'] = IndicatorCalculator.calculate_supertrend(high, low, close, 10, 3.0)
                # 9: ATR_20
                features['atr_20'] = atr_20
                # 10: ATR_50
                features['atr_50'] = atr_50
                # 11: VOLUME_SMA_20_RATIO
                vol_sma = volume.rolling(20).mean().iloc[-1]
                features['volume_sma_20_ratio'] = float(volume.iloc[-1] / vol_sma) if vol_sma > 0 else 1.0
                # 12: EMA_100_RATIO
                ema_100 = ta.ema(close, length=100).iloc[-1]
                features['ema_100_ratio'] = float(current_price / ema_100) if ema_100 > 0 else 1.0
                # 13: PIVOT_LEVEL
                features['pivot_level'] = IndicatorCalculator.calculate_pivot_level(high, low, close)
                # 14: DONCHIAN_WIDTH_20
                features['donchian_width_20'] = IndicatorCalculator.calculate_donchian_width(high, low, 20)

            return features

        except Exception as e:
            logger.error(f"Error calculating features for {timeframe}: {e}")
            # Return basic features with zeros for indicators
            return {
                'open': float(ohlcv['open'].iloc[-1]),
                'high': float(ohlcv['high'].iloc[-1]),
                'low': float(ohlcv['low'].iloc[-1]),
                'close': float(ohlcv['close'].iloc[-1]),
                'volume': float(ohlcv['volume'].iloc[-1]),
                # Fill rest with zeros dynamically in caller or here if needed
            }

    @staticmethod
    def calculate_features_df(ohlcv: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Calculate ALL features and return the DataFrame with new columns.
        This is used to build the full observation window (history).
        """
        df = ohlcv.copy()
        
        # Ensure lowercase columns
        df.columns = [c.lower() for c in df.columns]
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        try:
            # Common Indicators
            # ATRs
            df['atr_14'] = ta.atr(high, low, close, length=14)
            df['atr_20'] = ta.atr(high, low, close, length=20)
            df['atr_50'] = ta.atr(high, low, close, length=50)
            
            if timeframe == '5m':
                # RSI 14
                df['rsi_14'] = ta.rsi(close, length=14)
                
                # MACD 12, 26, 9
                macd = ta.macd(close, fast=12, slow=26, signal=9)
                if macd is not None:
                    df['macd_12_26_9'] = macd['MACD_12_26_9']
                
                # BB %B 20, 2
                bb = ta.bbands(close, length=20, std=2.0)
                if bb is not None:
                    df['bb_percent_b_20_2'] = bb['BBP_20_2.0']
                
                # Volume Ratio 20
                vol_sma = volume.rolling(20).mean()
                df['volume_ratio_20'] = volume / vol_sma
                
                # EMA 20 Ratio
                ema_20 = ta.ema(close, length=20)
                df['ema_20_ratio'] = close / ema_20
                
                # Stoch K 14, 3, 3
                stoch = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
                if stoch is not None:
                    df['stoch_k_14_3_3'] = stoch['STOCHk_14_3_3']
                    
            elif timeframe == '1h':
                # RSI 21
                df['rsi_21'] = ta.rsi(close, length=21)
                
                # MACD 21, 42, 9
                macd = ta.macd(close, fast=21, slow=42, signal=9)
                if macd is not None:
                    df['macd_21_42_9'] = macd['MACD_21_42_9']
                
                # BB Width 20, 2
                bb = ta.bbands(close, length=20, std=2.0)
                if bb is not None:
                    df['bb_width_20_2'] = bb['BBB_20_2.0']
                
                # ADX 14
                adx = ta.adx(high, low, close, length=14)
                if adx is not None:
                    df['adx_14'] = adx['ADX_14']
                
                # OBV Ratio 20
                obv = ta.obv(close, volume)
                obv_sma = obv.rolling(20).mean()
                df['obv_ratio_20'] = obv / obv_sma
                
                # EMA 50 Ratio
                ema_50 = ta.ema(close, length=50)
                df['ema_50_ratio'] = close / ema_50
                
                # Ichimoku Base (Kijun-sen)
                ichi, _ = ta.ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52)
                if ichi is not None:
                    # Kijun-sen is usually IKS_26
                    if 'IKS_26' in ichi.columns:
                        df['ichimoku_base'] = ichi['IKS_26']
                    else:
                        df['ichimoku_base'] = ichi.iloc[:, 1] # Fallback
                        
            elif timeframe == '4h':
                # RSI 28
                df['rsi_28'] = ta.rsi(close, length=28)
                
                # MACD 26, 52, 18
                macd = ta.macd(close, fast=26, slow=52, signal=18)
                if macd is not None:
                    df['macd_26_52_18'] = macd['MACD_26_52_18']
                
                # Supertrend 10, 3
                st = ta.supertrend(high, low, close, length=10, multiplier=3.0)
                if st is not None:
                    df['supertrend_10_3'] = st.iloc[:, 0] # Value
                
                # Volume SMA 20 Ratio
                vol_sma = volume.rolling(20).mean()
                # Check if vol_sma has data
                if vol_sma is not None and not vol_sma.empty:
                    df['volume_sma_20_ratio'] = volume / vol_sma
                
                # EMA 100 Ratio
                ema_100 = ta.ema(close, length=100)
                if ema_100 is not None and not ema_100.empty:
                    df['ema_100_ratio'] = close / ema_100
                
                # Pivot Level (Classic) - using previous candle
                # Shifted high/low/close
                prev_h = high.shift(1)
                prev_l = low.shift(1)
                prev_c = close.shift(1)
                df['pivot_level'] = (prev_h + prev_l + prev_c) / 3.0
                
                # Donchian Width 20
                donch = ta.donchian(high, low, lower_length=20, upper_length=20)
                if donch is not None and not donch.empty:
                    # Width = Upper - Lower (DCU - DCL)
                    # Columns: DCL, DCM, DCU
                    df['donchian_width_20'] = donch.iloc[:, 2] - donch.iloc[:, 0]

            # Fill NaNs created by indicators (e.g. at start of series)
            df = df.fillna(0.0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating features df for {timeframe}: {e}")
            return df

    @staticmethod
    def calculate_all(ohlcv: pd.DataFrame, period: int = DEFAULT_PERIOD) -> Dict[str, float]:
        """Legacy method for backward compatibility."""
        return IndicatorCalculator.calculate_all_features(ohlcv, '5m')
