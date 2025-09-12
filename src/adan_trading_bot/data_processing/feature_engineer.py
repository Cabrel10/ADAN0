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
                    param_map.update({'k': int(params[0]), 'd': int(params[1])})
                elif name in ['cci', 'roc', 'mfi', 'ema', 'sma']:
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
                    name = 'bbands'
                    param_map.update({
                        'length': int(params[0]), 'std': float(params[1])
                    })
                elif name == 'macd':
                    param_map.update({
                        'fast': int(params[0]),
                        'slow': int(params[1]),
                        'signal': int(params[2])
                    })
                elif name == 'ichimoku':
                    logger.info(f"Processing ICHIMOKU with params: {params}")
                    param_map.update({
                        'tenkan': int(params[0]),
                        'kijun': int(params[1]),
                        'senkou': int(params[2])
                    })
                    name = 'ichimoku'
                    logger.info(f"ICHIMOKU parameters after processing: {param_map}")
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
        
        # Supprimer les colonnes en double qui pourraient apparaître
        df = df.loc[:, ~df.columns.duplicated()]
        
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
