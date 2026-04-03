#!/usr/bin/env python

class FeatureEngineer:
    """
    Classe factice pour résoudre les problèmes d'importation.
    """
    def __init__(self):
        pass

# -*- coding: utf-8 -*-
"""
This module provides the FeatureEngineer class for adding technical
indicators and other features to financial market data.
"""

import logging
import traceback
import pandas as pd
import numpy as np
import pandas_ta as ta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Class for feature engineering on market data.

    This class handles the calculation of technical indicators based on a
    structured configuration file.
    """

    def __init__(self, data_config, models_dir):
        """
        Initializes the FeatureEngineer.

        Args:
            data_config (dict): Configuration for data and features.
            models_dir (str): Directory for models (not used in this version).
        """
        if not isinstance(data_config, dict):
            raise TypeError("data_config must be a dictionary.")

        self.data_config = data_config
        self.models_dir = Path(models_dir)
        self._validate_config()

    def _validate_config(self):
        """Validates the necessary configuration."""
        fe_config = self.data_config.get('feature_engineering')
        if not fe_config or 'indicators' not in fe_config:
            raise ValueError(
                "Config must contain 'feature_engineering' and 'indicators'."
            )

    def _get_indicators_for_timeframe(self, timeframe):
        """
        Constructs a final list of indicators for a specific timeframe.
        """
        fe_conf = self.data_config['feature_engineering']
        conf = fe_conf['indicators']
        common_params = conf.get('common', {})
        tf_conf = conf.get('timeframes', {}).get(timeframe, {})
        
        logger.info(f"Configuration for {timeframe}: {tf_conf}")

        indicator_strings = []
        for category, indicators in tf_conf.items():
            logger.info(f"Category {category} indicators: {indicators}")
            indicator_strings.extend(indicators)
        
        logger.info(f"All indicator strings for {timeframe}: {indicator_strings}")

        final_definitions = []
        processed_macds = set()

        for ind_str in indicator_strings:
            parts = [p.lower() for p in ind_str.split('_')]
            name = parts[0]
            params = parts[1:]

            logger.info(f"Processing indicator string: {ind_str}, name: {name}, params: {params}")
            
            if name == 'macd' and 'hist' in params:
                params.remove('hist')
                macd_key = f"macd_{'_'.join(params)}"
                if macd_key in processed_macds:
                    logger.debug(f"Skipping duplicate MACD: {macd_key}")
                    continue
                processed_macds.add(macd_key)
                logger.debug(f"Added MACD to processed set: {macd_key}")

            base_params = common_params.get(name, {}).copy()
            param_map = {}

            try:
                if name == 'rsi':
                    param_map['length'] = int(params[0])
                elif name == 'stoch':
                    # Handle stoch_k_14_3_3 -> ['k', '14', '3', '3']
                    p_vals = [p for p in params if p.isdigit()]
                    if len(p_vals) >= 3:
                        param_map.update({'k': int(p_vals[0]), 'd': int(p_vals[1]), 'smooth_k': int(p_vals[2])})
                    elif len(p_vals) >= 2:
                        param_map.update({'k': int(p_vals[0]), 'd': int(p_vals[1])})
                elif name in ['cci', 'roc', 'mfi', 'ema', 'sma', 'adx', 'atr']:
                    param_map['length'] = int(params[0])
                elif name == 'supertrend':
                    param_map.update({
                        'length': int(params[0]), 'multiplier': float(params[1])
                    })
                elif name == 'psar':
                    param_map.update({
                        'step': float(params[0]), 'max_step': float(params[1])
                    })
                elif name == 'bb':
                    # Handle bb_width_20_2 -> ['width', '20', '2']
                    p_vals = [p for p in params if p.replace('.', '', 1).isdigit()]
                    if p_vals:
                        param_map.update({
                            'length': int(p_vals[0]), 'std': float(p_vals[1])
                        })
                    if 'width' in params:
                        name = 'bb_width' # Custom kind
                    elif 'percent' in params or 'b' in params:
                        name = 'bb_percent' # Custom kind
                    else:
                        name = 'bbands'
                elif name == 'macd':
                    param_map.update({
                        'fast': int(params[0]),
                        'slow': int(params[1]),
                        'signal': int(params[2])
                    })
                elif name == 'macd':
                    param_map.update({
                        'fast': int(params[0]),
                        'slow': int(params[1]),
                        'signal': int(params[2])
                    })
                elif name == 'ichimoku':
                    if 'base' in params:
                        name = 'ichimoku_base'
                    else:
                        logger.info(f"Processing ICHIMOKU with params: {params}")
                        param_map.update({
                            'tenkan': int(params[0]),
                            'kijun': int(params[1]),
                            'senkou': int(params[2])
                        })
                        name = 'ichimoku'
                elif name == 'obv':
                    if 'ratio' in params:
                        name = 'obv_ratio' # Custom kind
                        p_vals = [p for p in params if p.isdigit()]
                        if p_vals:
                            param_map['length'] = int(p_vals[0])
                elif name == 'pivot':
                    name = 'pivot_level' # Custom kind
                elif name == 'donchian':
                    if 'width' in params:
                        name = 'donchian_width' # Custom kind
                        p_vals = [p for p in params if p.isdigit()]
                        if p_vals:
                            param_map['length'] = int(p_vals[0])
                elif name == 'fib':
                    if 'ratio' in params:
                        name = 'fib_ratio'
                elif name == 'vwap':
                    if 'ratio' in params:
                        name = 'vwap_ratio'
                elif name == 'price':
                    if 'action' in params:
                        name = 'price_action'
                    elif 'ema' in params:
                        name = 'price_ema_ratio_50' # Specific case
                elif name == 'volume':
                    if 'ratio' in params:
                        name = 'volume_ratio_20' # Specific case
                    elif 'sma' in params:
                        name = 'volume_sma_20_ratio' # Specific case

            except (IndexError, ValueError) as e:
                logger.error(f"Invalid params for {ind_str}: {e}")
                continue

            base_params.update(param_map)
            base_params['kind'] = name
            final_definitions.append(base_params)

        if not final_definitions:
            logger.warning(f"No indicators configured for {timeframe}.")

        return final_definitions

    def calculate_indicators_for_single_timeframe(self, df, timeframe):
        """
        Calculates indicators for a single timeframe.
        """
        self._validate_dataframe(df)
        df = df.copy()  # Créer une copie pour éviter les effets de bord
        indicators = self._get_indicators_for_timeframe(timeframe)
        
        logger.info(f"Indicateurs à calculer pour {timeframe}: {[ind['kind'] for ind in indicators]}")
        logger.info(f"Colonnes avant calcul: {df.columns.tolist()}")

        for indicator_params in indicators:
            params = indicator_params.copy()
            kind = params.pop('kind')
            logger.info(f"Calcul de l'indicateur: {kind} avec les paramètres: {params}")
            
            try:
                # Custom Indicators Handling inside the loop
                if kind == 'bb_width':
                    length = params.get('length', 20)
                    std = params.get('std', 2.0)
                    bb = df.ta.bbands(length=length, std=std)
                    if bb is not None:
                        # Pandas TA columns: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0 (bandwidth), BBP_20_2.0 (percent)
                        # BBB is bandwidth
                        width_col = f"BBB_{length}_{std}"
                        if width_col in bb.columns:
                             df[f"bb_width_{length}_{int(std)}"] = bb[width_col]
                        else:
                            # Fallback if column name differs
                            upper = bb.iloc[:, 2]
                            lower = bb.iloc[:, 0]
                            mid = bb.iloc[:, 1]
                            df[f"bb_width_{length}_{int(std)}"] = (upper - lower) / mid
                    continue

                elif kind == 'bb_percent':
                    length = params.get('length', 20)
                    std = params.get('std', 2.0)
                    bb = df.ta.bbands(length=length, std=std)
                    if bb is not None:
                        # BBP is percent b
                        pct_col = f"BBP_{length}_{std}"
                        if pct_col in bb.columns:
                             df[f"bb_percent_b_{length}_{int(std)}"] = bb[pct_col]
                    continue

                elif kind == 'obv_ratio':
                    length = params.get('length', 20)
                    obv = df.ta.obv()
                    if obv is not None:
                        obv_sma = obv.rolling(window=length).mean()
                        df[f"obv_ratio_{length}"] = obv / obv_sma
                    continue

                elif kind == 'pivot_level':
                    # Simple Pivot: (H+L+C)/3
                    df['pivot_level'] = (df['high'] + df['low'] + df['close']) / 3
                    continue
                
                elif kind == 'donchian_width':
                    length = params.get('length', 20)
                    low_min = df['low'].rolling(window=length).min()
                    high_max = df['high'].rolling(window=length).max()
                    df[f"donchian_width_{length}"] = (high_max - low_min) / df['close']
                    continue

                if kind not in dir(df.ta):
                    logger.warning(f"L'indicateur '{kind}' n'est pas disponible dans pandas_ta")
                    continue
                    
                indicator_func = getattr(df.ta, kind)
                
                # Gestion spéciale pour l'indicateur ICHIMOKU
                if kind == 'ichimoku':
                    logger.info(f"Traitement de l'indicateur ICHIMOKU avec les paramètres: {params}")
                    try:
                        # L'indicateur ICHIMOKU renvoie deux DataFrames
                        result_visible, result_projected = indicator_func(**params, append=False)
                        
                        logger.debug(f"Résultat visible ICHIMOKU colonnes: {result_visible.columns.tolist()}")
                        
                        # Vérifier et ajouter les colonnes visibles
                        for col in result_visible.columns:
                            if col not in df.columns:
                                df[col] = result_visible[col]
                        
                        # Ajouter les colonnes projetées si elles ne sont pas vides
                        if not result_projected.empty:
                            for col in result_projected.columns:
                                proj_col = f"{col}_proj"
                                if proj_col not in df.columns:
                                    df[proj_col] = result_projected[col]
                        
                        logger.debug(f"Colonnes après {kind}: {df.columns.tolist()}")
                        
                    except Exception as e:
                        logger.error(f"Erreur lors du traitement de l'indicateur ICHIMOKU: {str(e)}")
                        logger.error(f"Type de l'erreur: {type(e).__name__}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        continue
                else:
                    # Pour les autres indicateurs, utiliser le comportement par défaut
                    try:
                        result = indicator_func(append=False, **params)
                        if result is not None:
                            if isinstance(result, tuple):
                                # Gérer les indicateurs qui retournent des tuples
                                for res in result:
                                    if isinstance(res, pd.DataFrame):
                                        for col in res.columns:
                                            if col not in df.columns:
                                                df[col] = res[col]
                            elif isinstance(result, pd.DataFrame):
                                for col in result.columns:
                                    if col not in df.columns:
                                        df[col] = result[col]
                            elif isinstance(result, pd.Series):
                                col_name = result.name if result.name else f"{kind}"
                                df[col_name] = result
                            
                            logger.debug(f"Colonnes après {kind}: {df.columns.tolist()}")
                    except Exception as e:
                        logger.error(f"Erreur lors du calcul de {kind}: {str(e)}")
                        logger.error(f"Type de l'erreur: {type(e).__name__}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        continue
            
            except Exception as e:
                logger.error(f"Erreur inattendue avec l'indicateur {kind}: {str(e)}")
                logger.error(f"Type de l'erreur: {type(e).__name__}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        # Custom Indicators Implementation (Remaining ones)
        
        # 1. Volatility Ratio (ATR(14) / ATR(50))
        if 'volatility_ratio_14_50' not in df.columns:
            try:
                atr14 = df.ta.atr(length=14)
                atr50 = df.ta.atr(length=50)
                if atr14 is not None and atr50 is not None:
                    df['volatility_ratio_14_50'] = atr14 / atr50
            except Exception as e:
                logger.warning(f"Failed to calculate volatility_ratio_14_50: {e}")

        # 3. EMA Ratios
        for length in [20, 50, 100]:
            col_name = f'ema_{length}_ratio'
            if col_name not in df.columns:
                try:
                    ema = df.ta.ema(length=length)
                    if ema is not None:
                        df[col_name] = df['close'] / ema
                except Exception as e:
                    logger.warning(f"Failed to calculate {col_name}: {e}")
                    
        # 4. Price EMA Ratio (Duplicate of above but specific name)
        if 'price_ema_ratio_50' not in df.columns:
             if 'ema_50_ratio' in df.columns:
                 df['price_ema_ratio_50'] = df['ema_50_ratio']

        # 5. Ichimoku Base (Kijun-sen)
        if 'ichimoku_base' not in df.columns:
            try:
                # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
                period26_high = df['high'].rolling(window=26).max()
                period26_low = df['low'].rolling(window=26).min()
                df['ichimoku_base'] = (period26_high + period26_low) / 2
            except Exception as e:
                logger.warning(f"Failed to calculate ichimoku_base: {e}")

        # 6. Fib Ratio (Stochastic-like position in range)
        if 'fib_ratio' not in df.columns:
            try:
                # Using 14 period by default
                period = 14
                low_min = df['low'].rolling(window=period).min()
                high_max = df['high'].rolling(window=period).max()
                range_val = high_max - low_min
                # Avoid division by zero
                range_val = range_val.replace(0, 1e-9) 
                df['fib_ratio'] = (df['close'] - low_min) / range_val
            except Exception as e:
                logger.warning(f"Failed to calculate fib_ratio: {e}")

        # 7. Market Structure (Simple Trend)
        if 'market_structure' not in df.columns:
            try:
                # 1 if Close > SMA(20), -1 otherwise
                sma20 = df.ta.sma(length=20)
                df['market_structure'] = 0
                if sma20 is not None:
                    df.loc[df['close'] > sma20, 'market_structure'] = 1
                    df.loc[df['close'] < sma20, 'market_structure'] = -1
            except Exception as e:
                logger.warning(f"Failed to calculate market_structure: {e}")
                
        # 8. Volume SMA Ratio
        if 'volume_sma_20_ratio' not in df.columns:
            try:
                vol_sma = df['volume'].rolling(window=20).mean()
                df['volume_sma_20_ratio'] = df['volume'] / vol_sma
            except Exception as e:
                logger.warning(f"Failed to calculate volume_sma_20_ratio: {e}")

        # 9. Volume Ratio (Volume / SMA(Volume))
        if 'volume_ratio_20' not in df.columns:
            try:
                vol_sma = df['volume'].rolling(window=20).mean()
                df['volume_ratio_20'] = df['volume'] / vol_sma
            except Exception as e:
                logger.warning(f"Failed to calculate volume_ratio_20: {e}")

        # 10. V WAP Ratio (Close / VWAP) - CRITICAL FEATURE
        if 'vwap_ratio' not in df.columns:
            try:
                # VWAP requires DatetimeIndex
                if 'timestamp' in df.columns:
                    temp_df = df.copy()
                    temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
                    temp_df.set_index('timestamp', inplace=True)
                    vwap = temp_df.ta.vwap()
                    if vwap is not None and len(vwap) > 0:
                        df['vwap_ratio'] = df['close'] / vwap.values
                    else:
                        # Fallback: Simple volume-weighted average
                        cum_vol_price = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum()
                        cum_vol = df['volume'].cumsum()
                        vwap_fallback = cum_vol_price / cum_vol.replace(0, 1)
                        df['vwap_ratio'] = df['close'] / vwap_fallback
                        logger.info("Using fallback VWAP calculation")
                else:
                    # No timestamp - use simple VWAP
                    cum_vol_price = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum()
                    cum_vol = df['volume'].cumsum()
                    vwap_fallback = cum_vol_price / cum_vol.replace(0, 1)
                    df['vwap_ratio'] = df['close'] / vwap_fallback
                    logger.info("Using simple VWAP (no timestamp)")
            except Exception as e:
                logger.error(f"CRITICAL: vwap_ratio calculation failed: {e}")
                # MANDATORY FALLBACK - cannot leave this missing
                df['vwap_ratio'] = 1.0  # Neutral ratio
                logger.warning("Using neutral vwap_ratio=1.0 as last resort")


        # 11. Price Action (Close - Open) / Open - CRITICAL FEATURE
        if 'price_action' not in df.columns:
            try:
                df['price_action'] = (df['close'] - df['open']) / df['open'].replace(0, 1e-9)
            except Exception as e:
                logger.error(f"CRITICAL: price_action calculation failed: {e}")
                df['price_action'] = 0.0  # Neutral (no movement)
                logger.warning("Using neutral price_action=0.0 as fallback")


        # Normalize column names to match StateBuilder expectations
        # 1. Lowercase everything
        df.columns = [c.lower() for c in df.columns]
        
        # 2. Map specific Pandas TA names to config names
        rename_map = {}
        for col in df.columns:
            if col.startswith('atrr_'):
                rename_map[col] = col.replace('atrr_', 'atr_')
            elif col.startswith('supert_'):
                # SUPERT_10_3.0 -> supertrend_10_3
                # Remove .0 if present
                new_col = col.replace('supert_', 'supertrend_').replace('.0', '')
                rename_map[col] = new_col
            elif col.startswith('stochk_'):
                rename_map[col] = col.replace('stochk_', 'stoch_k_')
            elif col.startswith('stochd_'):
                rename_map[col] = col.replace('stochd_', 'stoch_d_')
            elif col.startswith('bb_percent_b_'):
                # Ensure consistent naming if needed
                pass 

        if rename_map:
            df.rename(columns=rename_map, inplace=True)

        # 🔧 FIX CRITIQUE: Gestion ROBUSTE des NaN AVANT tout traitement
        original_nan_count = df.isna().sum().sum()
        
        if original_nan_count > 0:
            logger.info(f"⚠️ {original_nan_count} NaN détectés - Application traitement robuste")
            
            # 1. Forward-fill puis backward-fill pour continuité temporelle
            df = df.ffill().bfill()
            
            # 2. Remplacer inf/-inf par NaN pour traitement uniforme
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # 3. Valeurs par défaut INTELLIGENTES par type d'indicateur
            indicator_defaults = {
                'rsi': 50.0,              # RSI neutre
                'macd': 0.0,              # MACD neutre
                'macd_signal': 0.0,
                'macd_hist': 0.0,
                'bb_percent': 0.5,        # Milieu des bandes de Bollinger
                'bb_width': 0.0,
                'atr': 0.02,              # Volatilité typique 2%
                'volatility': 0.02,
                'ema_ratio': 1.0,         # Prix = EMA
                'sma_ratio': 1.0,
                'price_ema_ratio': 1.0,
                'stoch': 50.0,            # Stochastique neutre
                'volume_ratio': 1.0,      # Volume = moyenne
                'obv_ratio': 0.0,
                'adx': 25.0,              # ADX neutre (tendance modérée)
                'pivot': 0.0,
                'donchian': 0.0,
                'supertrend': 0.0,
                'vwap_ratio': 1.0,
                'fib_ratio': 0.0,
                'ichimoku': 0.0,
                'market_structure': 0.0,
                'price_action': 0.0
            }
            
            # 4. Appliquer les valeurs par défaut colonne par colonne
            nan_fixes_applied = 0
            for col in df.columns:
                if df[col].isna().any():
                    na_count = df[col].isna().sum()
                    
                    # Trouver la valeur par défaut appropriée
                    default_value = 0.0
                    col_lower = col.lower()
                    
                    for pattern, value in indicator_defaults.items():
                        if pattern in col_lower:
                            default_value = value
                            break
                    
                    # Appliquer le remplacement
                    df[col] = df[col].fillna(default_value)
                    nan_fixes_applied += 1
                    
                    if na_count > 0:
                        logger.debug(f"   {col}: {na_count} NaN → {default_value}")
            
            # 5. Dernier recours: remplacer tout NaN restant par 0
            final_nan_count = df.isna().sum().sum()
            if final_nan_count > 0:
                logger.warning(f"⚠️ {final_nan_count} NaN restants après traitement - remplacement par 0")
                df = df.fillna(0.0)
            
            logger.info(f"✅ NaN handling: {original_nan_count} NaN → {nan_fixes_applied} colonnes corrigées → {final_nan_count} restants")
        
        # Supprimer les colonnes en double qui pourraient apparaître
        df = df.loc[:, ~df.columns.duplicated()]
        

        # 🔧 FIX: Filtrer pour garder UNIQUEMENT les colonnes du training
        TRAIN_COLUMNS = {
            '5m': ['open', 'high', 'low', 'close', 'volume', 'rsi_14', 
                   'macd_12_26_9',
                   'bb_percent_b_20_2', 'atr_14', 'atr_20', 'atr_50', 'volume_ratio_20', 
                   'ema_20_ratio', 'stoch_k_14_3_3', 'vwap_ratio', 'price_action'],
            '1h': ['open', 'high', 'low', 'close', 'volume', 'rsi_21',
                   'macd_21_42_9',
                   'bb_width_20_2', 'adx_14', 'atr_20', 'atr_50', 'obv_ratio_20', 'ema_50_ratio',
                   'ichimoku_base', 'fib_ratio', 'price_ema_ratio_50'],
            '4h': ['open', 'high', 'low', 'close', 'volume', 'rsi_28',
                   'macd_26_52_18',
                   'supertrend_10_3', 'atr_20', 'atr_50', 'volume_sma_20_ratio', 'ema_100_ratio',
                   'pivot_level', 'donchian_width_20', 'market_structure',
                   'volatility_ratio_14_50']
        }
        
        if timeframe in TRAIN_COLUMNS:
            expected_cols = TRAIN_COLUMNS[timeframe]
            # Garder uniquement les colonnes qui existent ET font partie du training
            cols_to_keep = [c for c in expected_cols if c in df.columns]
            cols_dropped = [c for c in df.columns if c not in expected_cols]
            
            logger.info(f"🔍 DEBUG FILTER: timeframe={timeframe}, expected={len(expected_cols)}, found={len(cols_to_keep)}, dropped={len(cols_dropped)}")
            logger.info(f"   Columns to keep: {cols_to_keep}")
            logger.info(f"   Columns to drop: {cols_dropped}")
            
            df = df[cols_to_keep]
            logger.info(f"✅ Filtered to {len(cols_to_keep)} training columns (dropped {len(cols_dropped)}: {cols_dropped[:5]}...)")
        
        logger.info(f"Colonnes finales après calcul ({len(df.columns)}): {df.columns.tolist()}")
        return df

    def _validate_dataframe(self, df):
        """
        Validates that the DataFrame contains the necessary columns.
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain: {required_columns}")
        return df
