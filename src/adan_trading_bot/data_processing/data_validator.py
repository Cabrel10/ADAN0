#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module de validation des données pour le bot de trading ADAN.

Ce module sépare la gestion des prix bruts (nécessaires pour l'exécution)
des indicateurs techniques, implémente un système de warm-up et valide
la qualité des données avant traitement.

Principes :
1. Prix vs Indicateurs : Seuls les prix (close) déterminent l'exécution
2. Valid From : Calcul de la première date où les données sont complètes
3. No Interpolation : Jamais d'interpolation linéaire pour les prix d'exécution
4. Forward Fill : Utilisation du dernier prix valide avec limite temporelle
5. Chunk Rejection : Rejet des chunks avec données insuffisantes
"""

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Classe principale pour la validation et préparation des données de trading.

    Sépare la gestion des prix d'exécution des indicateurs techniques,
    calcule les périodes de warm-up et valide la qualité des données.
    """

    def __init__(self, config: Dict[str, Any], worker_id: int = 0):
        """
        Initialise le validateur de données.

        Args:
            config: Configuration du système
            worker_id: ID du worker pour les logs
        """
        self.config = config
        self.worker_id = worker_id

        # Colonnes critiques pour l'exécution (prix uniquement)
        self.execution_columns = {'open', 'high', 'low', 'close', 'volume'}

        # Colonnes d'indicateurs (peuvent être NaN au début)
        self.indicator_columns = {
            'rsi_14', 'rsi_21', 'rsi_28',
            'macd_12_26_9', 'macd_signal_12_26_9', 'macd_hist_12_26_9',
            'macd_21_42_9', 'macd_26_52_18',
            'bb_percent_b_20_2', 'bb_width_20_2',
            'atr_14', 'adx_14',
            'volume_ratio_20', 'obv_ratio_20', 'volume_sma_20_ratio',
            'ema_20_ratio', 'ema_50_ratio', 'ema_100_ratio',
            'stoch_k_14_3_3', 'vwap_ratio', 'price_action',
            'ichimoku_base', 'fib_ratio', 'price_ema_ratio_50',
            'supertrend_10_3', 'pivot_level', 'donchian_width_20',
            'market_structure', 'volatility_ratio_14_50'
        }

        # Paramètres de tolérance
        self.max_price_gap_minutes = config.get('data_validation', {}).get('max_price_gap_minutes', 15)
        self.max_interpolation_pct = config.get('data_validation', {}).get('max_interpolation_pct', 2.0)

        # Cache des valid_from calculés
        self._valid_from_cache: Dict[Tuple[str, str], Optional[pd.Timestamp]] = {}

        # Statistiques de validation
        self.stats = {
            'chunks_validated': 0,
            'chunks_rejected': 0,
            'price_gaps_filled': 0,
            'invalid_chunks': {},
            'valid_from_calculated': 0
        }

    def calculate_valid_from(self, data: pd.DataFrame, asset: str, timeframe: str) -> Optional[pd.Timestamp]:
        """
        Calcule la première date où toutes les colonnes nécessaires sont non-nulles.

        Args:
            data: DataFrame avec les données de l'actif/timeframe
            asset: Nom de l'actif (ex: 'BTCUSDT')
            timeframe: Timeframe (ex: '5m', '1h', '4h')

        Returns:
            Timestamp de la première date valide ou None si données insuffisantes
        """
        cache_key = (asset, timeframe)
        if cache_key in self._valid_from_cache:
            return self._valid_from_cache[cache_key]

        try:
            if data.empty:
                self._valid_from_cache[cache_key] = None
                return None

            # Vérifier les colonnes d'exécution (critiques)
            execution_cols_present = []
            for col in self.execution_columns:
                matching_cols = [c for c in data.columns if c.lower() == col.lower()]
                if matching_cols:
                    execution_cols_present.extend(matching_cols)

            if not execution_cols_present:
                logger.error(f"MISSING_EXECUTION_COLS | asset={asset} | tf={timeframe} | "
                           f"required={self.execution_columns} | available={list(data.columns)}")
                self._valid_from_cache[cache_key] = None
                return None

            # Trouver la première date où close est non-null (minimum requis)
            close_col = self._find_close_column(data)
            if close_col is None:
                logger.error(f"MISSING_CLOSE_COL | asset={asset} | tf={timeframe}")
                self._valid_from_cache[cache_key] = None
                return None

            close_series = data[close_col].dropna()
            if close_series.empty:
                logger.error(f"NO_VALID_CLOSE | asset={asset} | tf={timeframe}")
                self._valid_from_cache[cache_key] = None
                return None

            first_valid_close = close_series.index[0]

            # Pour les indicateurs, chercher quand au moins 80% sont disponibles
            indicator_cols_present = [c for c in data.columns
                                    if any(ind in c.lower() for ind in self.indicator_columns)]

            if indicator_cols_present:
                # Calculer le pourcentage d'indicateurs non-null par ligne
                indicator_data = data[indicator_cols_present]
                non_null_pct = indicator_data.notna().sum(axis=1) / len(indicator_cols_present)

                # Trouver la première date avec 80% d'indicateurs valides
                sufficient_indicators = non_null_pct >= 0.8
                if sufficient_indicators.any():
                    first_valid_indicators = sufficient_indicators[sufficient_indicators].index[0]
                    valid_from = max(first_valid_close, first_valid_indicators)
                else:
                    # Si pas assez d'indicateurs, utiliser seulement close + marge de sécurité
                    valid_from = first_valid_close
                    logger.warning(f"INSUFFICIENT_INDICATORS | asset={asset} | tf={timeframe} | "
                                 f"max_coverage={non_null_pct.max():.2%}")
            else:
                valid_from = first_valid_close

            # Ajouter une marge de sécurité basée sur le timeframe
            safety_margin = self._get_safety_margin(timeframe)

            # S'assurer que valid_from est un Timestamp avant l'addition
            if isinstance(valid_from, (int, np.integer)):
                # Si c'est un index numérique, le convertir en timestamp
                if hasattr(data.index, 'to_pydatetime'):
                    valid_from = data.index[valid_from] if valid_from < len(data.index) else data.index[0]
                else:
                    logger.warning(f"INVALID_INDEX_TYPE | asset={asset} | tf={timeframe} | "
                                 f"index_type={type(data.index)} | valid_from_type={type(valid_from)}")
                    valid_from = data.index[0] if len(data.index) > 0 else pd.Timestamp.now()

            # Vérifier que valid_from est maintenant un Timestamp
            if isinstance(valid_from, pd.Timestamp):
                valid_from = valid_from + safety_margin
            else:
                logger.error(f"TYPE_ERROR_VALID_FROM | asset={asset} | tf={timeframe} | "
                           f"valid_from_type={type(valid_from)} | Cannot add Timedelta")
                valid_from = data.index[0] if len(data.index) > 0 else pd.Timestamp.now()

            # Vérifier que valid_from n'est pas au-delà des données disponibles
            if valid_from >= data.index[-1]:
                logger.warning(f"VALID_FROM_TOO_LATE | asset={asset} | tf={timeframe} | "
                             f"valid_from={valid_from} | data_end={data.index[-1]}")
                valid_from = first_valid_close

            self._valid_from_cache[cache_key] = valid_from
            self.stats['valid_from_calculated'] += 1

            logger.info(f"VALID_FROM_CALCULATED | asset={asset} | tf={timeframe} | "
                       f"valid_from={valid_from} | first_close={first_valid_close} | "
                       f"indicators_available={len(indicator_cols_present)}")

            return valid_from

        except Exception as e:
            logger.error(f"ERROR_CALCULATING_VALID_FROM | asset={asset} | tf={timeframe} | error={str(e)}")
            self._valid_from_cache[cache_key] = None
            return None

    def validate_chunk_data(self, chunk_data: Dict[str, Dict[str, pd.DataFrame]],
                          chunk_start: pd.Timestamp, chunk_end: pd.Timestamp) -> Tuple[bool, Dict[str, Any]]:
        """
        Valide un chunk de données multi-actifs/timeframes.

        Args:
            chunk_data: Données du chunk {asset: {timeframe: DataFrame}}
            chunk_start: Début du chunk
            chunk_end: Fin du chunk

        Returns:
            Tuple (is_valid, validation_info)
        """
        self.stats['chunks_validated'] += 1

        validation_info = {
            'chunk_start': chunk_start,
            'chunk_end': chunk_end,
            'assets_validated': 0,
            'assets_rejected': 0,
            'effective_start': None,
            'rejection_reasons': [],
            'asset_details': {}
        }

        max_valid_from = chunk_start
        rejected_assets = []

        for asset, timeframe_data in chunk_data.items():
            asset_valid = True
            asset_info = {
                'timeframes': {},
                'valid_from': None,
                'status': 'valid'
            }

            for timeframe, data in timeframe_data.items():
                tf_info = self._validate_timeframe_data(data, asset, timeframe, chunk_start, chunk_end)
                asset_info['timeframes'][timeframe] = tf_info

                if not tf_info['is_valid']:
                    asset_valid = False
                    rejected_assets.append(f"{asset}_{timeframe}")
                    validation_info['rejection_reasons'].append(
                        f"{asset}_{timeframe}: {tf_info['rejection_reason']}"
                    )

                # Mettre à jour le valid_from maximum
                if tf_info['valid_from'] and tf_info['valid_from'] > max_valid_from:
                    max_valid_from = tf_info['valid_from']

            if asset_valid:
                validation_info['assets_validated'] += 1
                asset_info['status'] = 'valid'
            else:
                validation_info['assets_rejected'] += 1
                asset_info['status'] = 'rejected'

            validation_info['asset_details'][asset] = asset_info

        # Déterminer si le chunk est globalement valide
        effective_start = max(chunk_start, max_valid_from)
        validation_info['effective_start'] = effective_start

        # Le chunk est invalide si effective_start dépasse chunk_end
        is_valid = effective_start <= chunk_end

        if not is_valid:
            self.stats['chunks_rejected'] += 1
            self.stats['invalid_chunks'][f"{chunk_start}_{chunk_end}"] = validation_info
            logger.warning(f"CHUNK_REJECTED | start={chunk_start} | end={chunk_end} | "
                         f"effective_start={effective_start} | "
                         f"rejected_assets={rejected_assets}")
        else:
            logger.info(f"CHUNK_VALIDATED | start={chunk_start} | end={chunk_end} | "
                       f"effective_start={effective_start} | "
                       f"assets_ok={validation_info['assets_validated']} | "
                       f"assets_rejected={validation_info['assets_rejected']}")

        return is_valid, validation_info

    def clean_price_data(self, data: pd.DataFrame, asset: str, timeframe: str) -> pd.DataFrame:
        """
        Nettoie les données de prix selon les règles strictes.

        Args:
            data: DataFrame avec les données brutes
            asset: Nom de l'actif
            timeframe: Timeframe

        Returns:
            DataFrame nettoyé
        """
        if data.empty:
            return data

        cleaned_data = data.copy()
        close_col = self._find_close_column(cleaned_data)

        if close_col is None:
            logger.error(f"NO_CLOSE_COLUMN | asset={asset} | tf={timeframe}")
            return cleaned_data

        # Règle 1: Jamais d'interpolation linéaire pour les prix
        # Règle 2: Forward fill uniquement avec limite temporelle

        initial_nan_count = cleaned_data[close_col].isna().sum()
        if initial_nan_count > 0:
            logger.info(f"PRICE_CLEANING | asset={asset} | tf={timeframe} | "
                       f"initial_nans={initial_nan_count}/{len(cleaned_data)}")

            # Forward fill avec limite (max 3 périodes consécutives)
            filled_data = self._forward_fill_with_limit(cleaned_data[close_col], limit=3)
            cleaned_data[close_col] = filled_data

            remaining_nan_count = cleaned_data[close_col].isna().sum()
            filled_count = initial_nan_count - remaining_nan_count

            if filled_count > 0:
                self.stats['price_gaps_filled'] += filled_count
                logger.info(f"PRICE_GAPS_FILLED | asset={asset} | tf={timeframe} | "
                           f"filled={filled_count} | remaining_nans={remaining_nan_count}")

            # Si il reste des NaN, les signaler mais ne pas interpoler
            if remaining_nan_count > 0:
                pct_missing = (remaining_nan_count / len(cleaned_data)) * 100
                logger.warning(f"PRICE_GAPS_REMAINING | asset={asset} | tf={timeframe} | "
                             f"missing={remaining_nan_count}/{len(cleaned_data)} ({pct_missing:.1f}%)")

        # Nettoyer les autres colonnes de prix de manière similaire
        for price_col in ['open', 'high', 'low']:
            col_name = self._find_column_case_insensitive(cleaned_data, price_col)
            if col_name:
                initial_nans = cleaned_data[col_name].isna().sum()
                if initial_nans > 0:
                    filled_data = self._forward_fill_with_limit(cleaned_data[col_name], limit=3)
                    cleaned_data[col_name] = filled_data

        return cleaned_data

    def clean_indicator_data(self, data: pd.DataFrame, asset: str, timeframe: str) -> pd.DataFrame:
        """
        Nettoie les données d'indicateurs avec des règles plus permissives.

        Args:
            data: DataFrame avec les données brutes
            asset: Nom de l'actif
            timeframe: Timeframe

        Returns:
            DataFrame nettoyé
        """
        if data.empty:
            return data

        cleaned_data = data.copy()
        indicator_cols = [c for c in cleaned_data.columns
                         if any(ind in c.lower() for ind in self.indicator_columns)]

        for col in indicator_cols:
            initial_nans = cleaned_data[col].isna().sum()
            if initial_nans > 0:
                # Pour les indicateurs, forward fill plus permissif
                filled_data = self._forward_fill_with_limit(cleaned_data[col], limit=10)
                cleaned_data[col] = filled_data

                remaining_nans = cleaned_data[col].isna().sum()
                if remaining_nans < initial_nans:
                    logger.debug(f"INDICATOR_FILLED | asset={asset} | tf={timeframe} | "
                               f"indicator={col} | filled={initial_nans - remaining_nans}")

        return cleaned_data

    def get_validation_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de validation."""
        return {
            **self.stats,
            'cache_size': len(self._valid_from_cache),
            'validation_rate': (self.stats['chunks_validated'] - self.stats['chunks_rejected']) / max(1, self.stats['chunks_validated'])
        }

    def _validate_timeframe_data(self, data: pd.DataFrame, asset: str, timeframe: str,
                               chunk_start: pd.Timestamp, chunk_end: pd.Timestamp) -> Dict[str, Any]:
        """Valide les données d'un timeframe spécifique."""
        tf_info = {
            'is_valid': False,
            'valid_from': None,
            'rejection_reason': None,
            'data_quality': {},
            'price_gaps': 0,
            'indicator_coverage': 0.0
        }

        if data.empty:
            tf_info['rejection_reason'] = 'empty_data'
            return tf_info

        # Calculer valid_from
        valid_from = self.calculate_valid_from(data, asset, timeframe)
        tf_info['valid_from'] = valid_from

        if valid_from is None:
            tf_info['rejection_reason'] = 'no_valid_from'
            return tf_info

        # Vérifier si le chunk a suffisamment de données après valid_from
        effective_start = max(chunk_start, valid_from)
        if effective_start >= chunk_end:
            tf_info['rejection_reason'] = f'valid_from_too_late_{effective_start}_vs_{chunk_end}'
            return tf_info

        # Évaluer la qualité des prix
        close_col = self._find_close_column(data)
        if close_col:
            price_data = data[close_col][effective_start:chunk_end]
            if not price_data.empty:
                nan_count = price_data.isna().sum()
                tf_info['price_gaps'] = nan_count

                # Rejeter si trop de gaps dans les prix
                gap_pct = (nan_count / len(price_data)) * 100
                if gap_pct > self.max_interpolation_pct:
                    tf_info['rejection_reason'] = f'excessive_price_gaps_{gap_pct:.1f}%'
                    return tf_info

        # Évaluer la couverture des indicateurs
        indicator_cols = [c for c in data.columns
                         if any(ind in c.lower() for ind in self.indicator_columns)]
        if indicator_cols:
            indicator_data = data[indicator_cols][effective_start:chunk_end]
            if not indicator_data.empty:
                coverage = indicator_data.notna().mean().mean()
                tf_info['indicator_coverage'] = coverage

        tf_info['is_valid'] = True
        return tf_info

    def _find_close_column(self, data: pd.DataFrame) -> Optional[str]:
        """Trouve la colonne close (insensible à la casse)."""
        for col in data.columns:
            if col.lower() == 'close':
                return col
        return None

    def _find_column_case_insensitive(self, data: pd.DataFrame, target: str) -> Optional[str]:
        """Trouve une colonne en ignorant la casse."""
        for col in data.columns:
            if col.lower() == target.lower():
                return col
        return None

    def _forward_fill_with_limit(self, series: pd.Series, limit: int) -> pd.Series:
        """Forward fill avec limite du nombre de valeurs consécutives."""
        return series.fillna(method='ffill', limit=limit)

    def _get_safety_margin(self, timeframe: str) -> pd.Timedelta:
        """Calcule la marge de sécurité basée sur le timeframe."""
        margins = {
            '5m': pd.Timedelta(minutes=30),   # 6 périodes
            '1h': pd.Timedelta(hours=6),      # 6 périodes
            '4h': pd.Timedelta(hours=24),     # 6 périodes
            '1d': pd.Timedelta(days=7)        # 7 périodes
        }
        return margins.get(timeframe, pd.Timedelta(minutes=30))


class DataQualityMonitor:
    """
    Moniteur de qualité des données en temps réel.

    Suit les métriques de qualité et détecte les problèmes de données.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quality_history = []
        self.alert_thresholds = {
            'max_price_gap_pct': 5.0,
            'min_indicator_coverage': 0.7,
            'max_rejected_chunks_pct': 10.0
        }

    def record_quality_metrics(self, validation_info: Dict[str, Any]):
        """Enregistre les métriques de qualité d'un chunk."""
        timestamp = pd.Timestamp.now()

        metrics = {
            'timestamp': timestamp,
            'chunk_start': validation_info['chunk_start'],
            'chunk_end': validation_info['chunk_end'],
            'assets_validated': validation_info['assets_validated'],
            'assets_rejected': validation_info['assets_rejected'],
            'rejection_reasons': validation_info['rejection_reasons']
        }

        self.quality_history.append(metrics)

        # Garder seulement les 1000 dernières métriques
        if len(self.quality_history) > 1000:
            self.quality_history = self.quality_history[-1000:]

    def get_quality_report(self, lookback_hours: int = 24) -> Dict[str, Any]:
        """Génère un rapport de qualité des données."""
        cutoff = pd.Timestamp.now() - pd.Timedelta(hours=lookback_hours)
        recent_metrics = [m for m in self.quality_history if m['timestamp'] >= cutoff]

        if not recent_metrics:
            return {'status': 'no_data', 'lookback_hours': lookback_hours}

        total_chunks = len(recent_metrics)
        rejected_chunks = sum(1 for m in recent_metrics if m['assets_rejected'] > 0)

        return {
            'status': 'ok',
            'lookback_hours': lookback_hours,
            'total_chunks': total_chunks,
            'rejected_chunks': rejected_chunks,
            'rejection_rate': rejected_chunks / total_chunks if total_chunks > 0 else 0,
            'common_rejection_reasons': self._get_common_rejection_reasons(recent_metrics),
            'quality_trend': self._calculate_quality_trend(recent_metrics)
        }

    def _get_common_rejection_reasons(self, metrics: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyse les raisons de rejet les plus communes."""
        reason_counts = {}
        for m in metrics:
            for reason in m['rejection_reasons']:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        # Trier par fréquence
        return dict(sorted(reason_counts.items(), key=lambda x: x[1], reverse=True))

    def _calculate_quality_trend(self, metrics: List[Dict[str, Any]]) -> str:
        """Calcule la tendance de qualité."""
        if len(metrics) < 10:
            return 'insufficient_data'

        # Diviser en deux moitiés
        mid = len(metrics) // 2
        first_half = metrics[:mid]
        second_half = metrics[mid:]

        first_rejection_rate = sum(1 for m in first_half if m['assets_rejected'] > 0) / len(first_half)
        second_rejection_rate = sum(1 for m in second_half if m['assets_rejected'] > 0) / len(second_half)

        if second_rejection_rate < first_rejection_rate * 0.8:
            return 'improving'
        elif second_rejection_rate > first_rejection_rate * 1.2:
            return 'deteriorating'
        else:
            return 'stable'
