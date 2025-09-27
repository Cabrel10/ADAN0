#!/usr/bin/env python3
"""
Script pour régénérer tous les fichiers parquet avec les indicateurs spécifiques par timeframe
selon la configuration dans config.yaml

Utilisation:
    python regenerate_parquets_with_timeframe_indicators.py [--dry-run]
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import talib as ta
import argparse

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parquet_regeneration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ParquetRegenerator:
    """Classe pour régénérer les parquets avec les bons indicateurs par timeframe"""

    def __init__(self, config_path: str = "bot/config/config.yaml"):
        """Initialize the regenerator with configuration"""
        self.config_path = config_path
        self.config = self._load_config()
        self.features_config = self.config['data']['features_config']['timeframes']
        self.base_dir = Path("data/processed/indicators")

        logger.info(f"Configuration chargée depuis: {config_path}")
        logger.info(f"Timeframes configurés: {list(self.features_config.keys())}")

    def _load_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis le fichier YAML"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            raise

    def _extract_base_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extrait les données OHLCV + timestamp de base d'un DataFrame"""
        base_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']

        # Chercher les colonnes qui existent (case insensitive)
        available_columns = []
        df_columns_lower = [col.lower() for col in df.columns]

        for base_col in base_columns:
            for i, df_col in enumerate(df_columns_lower):
                if df_col == base_col.lower():
                    available_columns.append(df.columns[i])
                    break
            else:
                if base_col != 'timestamp':  # timestamp might not exist yet
                    logger.warning(f"Colonne manquante: {base_col}")

        if len(available_columns) < 5:  # Au minimum OHLCV
            raise ValueError(f"Données insuffisantes. Colonnes trouvées: {available_columns}")

        base_df = df[available_columns].copy()

        # Standardiser les noms de colonnes
        column_mapping = {}
        for col in base_df.columns:
            col_lower = col.lower()
            if col_lower in ['open', 'high', 'low', 'close', 'volume', 'timestamp']:
                column_mapping[col] = col_lower

        base_df = base_df.rename(columns=column_mapping)

        # Ajouter timestamp si manquant (utiliser l'index si c'est un DatetimeIndex)
        if 'timestamp' not in base_df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                base_df['timestamp'] = df.index
            else:
                # Créer un timestamp fictif basé sur l'index
                base_df['timestamp'] = pd.date_range(
                    start='2023-01-01', periods=len(base_df), freq='5min'
                )
                logger.warning("Timestamp manquant, création d'un timestamp fictif")

        return base_df

    def _calculate_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Calcule tous les indicateurs pour un timeframe donné"""
        if timeframe not in self.features_config:
            logger.error(f"Timeframe {timeframe} non configuré")
            return df

        indicators = self.features_config[timeframe].get('indicators', [])
        logger.info(f"Calcul de {len(indicators)} indicateurs pour {timeframe}: {indicators}")

        # Faire une copie pour éviter les modifications
        result_df = df.copy()

        for indicator in indicators:
            try:
                result_df = self._add_single_indicator(result_df, indicator, timeframe)
                logger.debug(f"✅ Indicateur {indicator} calculé")
            except Exception as e:
                logger.error(f"❌ Erreur lors du calcul de {indicator}: {e}")

        return result_df

    def _add_single_indicator(self, df: pd.DataFrame, indicator: str, timeframe: str) -> pd.DataFrame:
        """Ajoute un indicateur spécifique au DataFrame"""

        # RSI avec périodes différentes
        if indicator.startswith('rsi_'):
            period = int(indicator.split('_')[1])
            df[indicator] = ta.RSI(df['close'].values, timeperiod=period)

        # MACD avec paramètres différents
        elif indicator.startswith('macd_'):
            parts = indicator.split('_')
            if len(parts) >= 4:
                fast, slow, signal = int(parts[1]), int(parts[2]), int(parts[3])
                macd_line, macd_signal, macd_hist = ta.MACD(
                    df['close'].values, fastperiod=fast, slowperiod=slow, signalperiod=signal
                )
                df[f'macd_{fast}_{slow}_{signal}'] = macd_line
                df[f'macd_signal_{fast}_{slow}_{signal}'] = macd_signal
                df[f'macd_hist_{fast}_{slow}_{signal}'] = macd_hist

        # Bollinger Bands %B
        elif indicator == 'bb_percent_b_20_2':
            bb_upper, bb_middle, bb_lower = ta.BBANDS(df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
            df[indicator] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

        # Bollinger Bands Width
        elif indicator == 'bb_width_20_2':
            bb_upper, bb_middle, bb_lower = ta.BBANDS(df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
            df[indicator] = (bb_upper - bb_lower) / bb_middle

        # ATR
        elif indicator.startswith('atr_'):
            period = int(indicator.split('_')[1])
            df[indicator] = ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)

        # Volume Ratio
        elif indicator.startswith('volume_ratio_'):
            period = int(indicator.split('_')[2])
            volume_sma = ta.SMA(df['volume'].values, timeperiod=period)
            df[indicator] = df['volume'] / volume_sma

        # EMA Ratios
        elif indicator.startswith('ema_') and indicator.endswith('_ratio'):
            period = int(indicator.split('_')[1])
            ema = ta.EMA(df['close'].values, timeperiod=period)
            df[indicator] = df['close'] / ema

        # Stochastic K
        elif indicator.startswith('stoch_k_'):
            parts = indicator.split('_')
            if len(parts) >= 5:
                k_period, d_period = int(parts[2]), int(parts[4])
                slowk, slowd = ta.STOCH(
                    df['high'].values, df['low'].values, df['close'].values,
                    fastk_period=k_period, slowk_period=d_period, slowd_period=d_period
                )
                df[indicator] = slowk

        # VWAP Ratio
        elif indicator == 'vwap_ratio':
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            volume_price = typical_price * df['volume']
            vwap = volume_price.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
            df[indicator] = df['close'] / vwap

        # Price Action (Typical Price)
        elif indicator == 'price_action':
            df[indicator] = (df['high'] + df['low'] + df['close']) / 3

        # ADX
        elif indicator.startswith('adx_'):
            period = int(indicator.split('_')[1])
            df[indicator] = ta.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)

        # OBV Ratio
        elif indicator.startswith('obv_ratio_'):
            period = int(indicator.split('_')[2])
            obv = ta.OBV(df['close'].values, df['volume'].values)
            obv_sma = ta.SMA(obv, timeperiod=period)
            df[indicator] = obv / obv_sma

        # Ichimoku Base Line
        elif indicator == 'ichimoku_base':
            period = 26
            high_max = df['high'].rolling(window=period).max()
            low_min = df['low'].rolling(window=period).min()
            df[indicator] = (high_max + low_min) / 2

        # Fibonacci Ratio (approximation avec golden ratio)
        elif indicator == 'fib_ratio':
            ema_short = ta.EMA(df['close'].values, timeperiod=13)
            ema_long = ta.EMA(df['close'].values, timeperiod=21)
            df[indicator] = ema_short / ema_long

        # Price/EMA Ratio
        elif indicator.startswith('price_ema_ratio_'):
            period = int(indicator.split('_')[3])
            ema = ta.EMA(df['close'].values, timeperiod=period)
            df[indicator] = df['close'] / ema

        # SuperTrend
        elif indicator.startswith('supertrend_'):
            parts = indicator.split('_')
            if len(parts) >= 3:
                period, multiplier = int(parts[1]), float(parts[2])
                hl2 = (df['high'] + df['low']) / 2
                atr = ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
                upper_band = hl2 + (multiplier * atr)
                lower_band = hl2 - (multiplier * atr)

                supertrend = pd.Series(index=df.index, dtype=float)
                trend = pd.Series(index=df.index, dtype=int)

                for i in range(len(df)):
                    if i == 0:
                        supertrend.iloc[i] = lower_band.iloc[i]
                        trend.iloc[i] = 1
                    else:
                        if df['close'].iloc[i] > supertrend.iloc[i-1]:
                            supertrend.iloc[i] = lower_band.iloc[i]
                            trend.iloc[i] = 1
                        else:
                            supertrend.iloc[i] = upper_band.iloc[i]
                            trend.iloc[i] = -1

                df[indicator] = supertrend

        # Volume SMA Ratio
        elif indicator.startswith('volume_sma_') and indicator.endswith('_ratio'):
            period = int(indicator.split('_')[2])
            volume_sma = ta.SMA(df['volume'].values, timeperiod=period)
            df[indicator] = df['volume'] / volume_sma

        # Pivot Levels (approximation avec HLC/3)
        elif indicator == 'pivot_level':
            df[indicator] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3

        # Donchian Width
        elif indicator.startswith('donchian_width_'):
            period = int(indicator.split('_')[2])
            highest = df['high'].rolling(window=period).max()
            lowest = df['low'].rolling(window=period).min()
            df[indicator] = (highest - lowest) / df['close']

        # Market Structure (approximation avec higher highs/lower lows)
        elif indicator == 'market_structure':
            lookback = 20
            recent_high = df['high'].rolling(window=lookback).max()
            recent_low = df['low'].rolling(window=lookback).min()
            df[indicator] = (df['high'] / recent_high - df['low'] / recent_low)

        # Volatility Ratio
        elif indicator.startswith('volatility_ratio_'):
            parts = indicator.split('_')
            if len(parts) >= 4:
                atr_period, sma_period = int(parts[2]), int(parts[3])
                atr = ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=atr_period)
                atr_sma = ta.SMA(atr, timeperiod=sma_period)
                df[indicator] = atr / atr_sma

        else:
            logger.warning(f"Indicateur non implémenté: {indicator}")

        return df

    def _process_single_file(self, file_path: Path, dry_run: bool = False) -> bool:
        """Traite un seul fichier parquet"""
        try:
            logger.info(f"Traitement de: {file_path}")

            # Déterminer le timeframe depuis le nom du fichier
            timeframe = file_path.stem  # '5m', '1h', '4h'

            if timeframe not in self.features_config:
                logger.error(f"Timeframe {timeframe} non configuré, passage au suivant")
                return False

            # Lire le fichier existant
            df_original = pd.read_parquet(file_path)
            logger.info(f"Données originales: {df_original.shape} - Colonnes: {len(df_original.columns)}")

            # Extraire les données de base
            df_base = self._extract_base_data(df_original)
            logger.info(f"Données de base extraites: {df_base.shape}")

            # Calculer les nouveaux indicateurs
            df_with_indicators = self._calculate_indicators(df_base, timeframe)
            logger.info(f"Données avec indicateurs: {df_with_indicators.shape}")

            # Gérer le timestamp selon la configuration
            timestamp_config = self.config['data'].get('timestamp', {})
            if timestamp_config.get('set_as_index', False) and 'timestamp' in df_with_indicators.columns:
                df_with_indicators.set_index('timestamp', inplace=True)
                logger.info("Timestamp défini comme index")

            if not dry_run:
                # Sauvegarder le fichier mis à jour
                backup_path = file_path.with_suffix('.parquet.backup')
                if not backup_path.exists():
                    df_original.to_parquet(backup_path)
                    logger.info(f"Sauvegarde créée: {backup_path}")

                df_with_indicators.to_parquet(file_path)
                logger.info(f"✅ Fichier mis à jour: {file_path}")
            else:
                logger.info(f"Mode dry-run: {file_path} serait mis à jour")

            # Log des colonnes finales
            final_columns = list(df_with_indicators.columns)
            logger.info(f"Colonnes finales ({len(final_columns)}): {final_columns}")

            return True

        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement de {file_path}: {e}")
            return False

    def regenerate_all(self, dry_run: bool = False) -> None:
        """Régénère tous les fichiers parquet"""
        logger.info("=" * 80)
        logger.info("DÉBUT DE LA RÉGÉNÉRATION DES PARQUETS")
        logger.info("=" * 80)

        if dry_run:
            logger.info("🔍 MODE DRY-RUN ACTIVÉ - Aucune modification ne sera faite")

        # Trouver tous les fichiers parquet
        parquet_files = []
        for split in ['train', 'val', 'test']:
            split_dir = self.base_dir / split
            if split_dir.exists():
                for asset_dir in split_dir.iterdir():
                    if asset_dir.is_dir():
                        for parquet_file in asset_dir.glob('*.parquet'):
                            if not parquet_file.name.endswith('.backup'):
                                parquet_files.append(parquet_file)

        logger.info(f"📁 Fichiers parquet trouvés: {len(parquet_files)}")

        # Traiter chaque fichier
        success_count = 0
        for i, file_path in enumerate(parquet_files, 1):
            logger.info(f"\n[{i}/{len(parquet_files)}] Traitement en cours...")
            if self._process_single_file(file_path, dry_run):
                success_count += 1

        # Résumé
        logger.info("\n" + "=" * 80)
        logger.info("RÉSUMÉ DE LA RÉGÉNÉRATION")
        logger.info("=" * 80)
        logger.info(f"✅ Fichiers traités avec succès: {success_count}")
        logger.info(f"❌ Fichiers en erreur: {len(parquet_files) - success_count}")
        logger.info(f"📊 Total: {len(parquet_files)}")

        if not dry_run:
            logger.info("💾 Les fichiers originaux ont été sauvegardés avec l'extension .backup")
        logger.info("=" * 80)

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(
        description="Régénère tous les fichiers parquet avec les indicateurs spécifiques par timeframe"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Mode dry-run: affiche ce qui serait fait sans modifier les fichiers'
    )
    parser.add_argument(
        '--config',
        default='bot/config/config.yaml',
        help='Chemin vers le fichier de configuration (défaut: bot/config/config.yaml)'
    )

    args = parser.parse_args()

    try:
        regenerator = ParquetRegenerator(args.config)
        regenerator.regenerate_all(dry_run=args.dry_run)

        if args.dry_run:
            print("\n🔍 Mode dry-run terminé. Exécutez sans --dry-run pour appliquer les modifications.")
        else:
            print("\n✅ Régénération terminée avec succès !")

    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
