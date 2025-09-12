#!/usr/bin/env python3
"""
Script pour traiter les données de marché brutes et les convertir en fichiers Parquet
avec la structure de répertoires appropriée pour l'entraînement.
"""

import os
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging
import numpy as np

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketDataProcessor:
    """Classe pour traiter les données de marché et les convertir en format Parquet."""

    def __init__(self, config_path: str = None):
        """Initialise le processeur avec les chemins de configuration."""
        if config_path is None:
            # Chemin relatif au script
            base_dir = Path(__file__).parent.parent
            config_path = str(base_dir / 'config' / 'config.yaml')
        """Initialise le processeur avec les chemins de configuration."""
        self.config = self._load_config(config_path)
        self.timeframes = self.config['feature_engineering']['timeframes']
        self.assets = self.config['data']['file_structure']['assets']

        # Définition des chemins
        # Utilisation du chemin source des données brutes depuis le répertoire parent
        self.raw_data_dir = Path('/home/morningstar/Documents/trading/data/raw')
        self.processed_dir = Path(self.config['paths']['processed_data_dir'])
        self.indicators_dir = Path(self.config['paths']['indicators_data_dir'])

        # Création des répertoires si nécessaire
        self.indicators_dir.mkdir(parents=True, exist_ok=True)

        # Configuration des indicateurs techniques
        self.ta_config = self.config.get('feature_engineering', {})

    def _load_config(self, config_path: str) -> Dict:
        """Charge le fichier de configuration YAML."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            raise

    def _read_csv_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Lit un fichier CSV et retourne un DataFrame."""
        try:
            df = pd.read_csv(
                file_path,
                parse_dates=['Date'],
                index_col='Date'
            )
            return df.sort_index()
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du fichier {file_path}: {e}")
            return None

    def _calculate_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Calcule les indicateurs techniques pour un DataFrame donné en fonction de la configuration.

        Args:
            df: DataFrame contenant les données OHLCV
            timeframe: Période de temps (5m, 1h, 4h)

        Returns:
            DataFrame avec les indicateurs ajoutés
        """
        # Copie pour éviter les avertissements
        df = df.copy()

        # Conversion en valeurs numériques
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Récupération de la configuration des indicateurs pour ce timeframe
        indicators_config = self.ta_config.get('indicators', {}).get('timeframes', {}).get(timeframe, {})

        # Calcul des indicateurs de momentum
        if 'momentum' in indicators_config:
            for indicator in indicators_config['momentum']:
                if indicator == 'RSI_14':
                    # RSI (14 périodes)
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['RSI_14'] = 100 - (100 / (1 + rs))

                elif indicator == 'STOCHk_14_3_3':
                    # Stochastique %K (14,3,3)
                    low_min = df['Low'].rolling(window=14).min()
                    high_max = df['High'].rolling(window=14).max()
                    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
                    df['STOCHk_14_3_3'] = k.rolling(window=3).mean()

                elif indicator == 'STOCHd_14_3_3':
                    # Stochastique %D (moyenne mobile de %K sur 3 périodes)
                    if 'STOCHk_14_3_3' not in df.columns:
                        low_min = df['Low'].rolling(window=14).min()
                        high_max = df['High'].rolling(window=14).max()
                        k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
                        df['STOCHk_14_3_3'] = k.rolling(window=3).mean()
                    df['STOCHd_14_3_3'] = df['STOCHk_14_3_3'].rolling(window=3).mean()

                elif indicator == 'CCI_20_0.015' or indicator == 'CCI_20':
                    # CCI (20, 0.015)
                    tp = (df['High'] + df['Low'] + df['Close']) / 3
                    sma = tp.rolling(window=20).mean()
                    mad = (tp - sma).abs().rolling(window=20).mean()
                    df['CCI_20'] = (tp - sma) / (0.015 * mad)

                elif indicator == 'ROC_9':
                    # Taux de variation (9 périodes)
                    df['ROC_9'] = df['Close'].pct_change(periods=9) * 100

                elif indicator == 'MFI_14':
                    # Money Flow Index (14 périodes)
                    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
                    money_flow = typical_price * df['Volume']

                    # Flux positif/négatif
                    positive_flow = ((typical_price > typical_price.shift(1)) * money_flow).rolling(window=14).sum()
                    negative_flow = ((typical_price < typical_price.shift(1)) * money_flow).rolling(window=14).sum()

                    # Calcul du ratio et du MFI
                    mf_ratio = positive_flow / negative_flow
                    df['MFI_14'] = 100 - (100 / (1 + mf_ratio))

                elif 'MACD' in indicator:
                    # Gestion des différents types de MACD
                    if indicator == 'MACD_12_26_9':
                        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                        df['MACD'] = exp1 - exp2
                        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                    elif indicator == 'MACD_HIST_12_26_9':
                        if 'MACD' not in df.columns:
                            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                            df['MACD'] = exp1 - exp2
                            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # Indicateurs de tendance
        if 'trend' in indicators_config:
            for indicator in indicators_config['trend']:
                if indicator == 'EMA_5':
                    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
                elif indicator == 'EMA_20':
                    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
                elif indicator == 'EMA_50':
                    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
                elif indicator == 'EMA_100':
                    df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
                elif indicator == 'EMA_200':
                    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
                elif indicator == 'SMA_200':
                    df['SMA_200'] = df['Close'].rolling(window=200).mean()
                elif indicator == 'SUPERTREND_14_2.0':
                    # Calcul du SuperTrend (14, 2.0)
                    atr_period = 14
                    multiplier = 2.0

                    high = df['High']
                    low = df['Low']
                    close = df['Close']

                    # Calcul de l'ATR
                    tr1 = high - low
                    tr2 = (high - close.shift()).abs()
                    tr3 = (low - close.shift()).abs()
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr = tr.rolling(window=atr_period).mean()

                    # Bandes supérieure et inférieure
                    hl2 = (high + low) / 2
                    upper_band = hl2 + (multiplier * atr)
                    lower_band = hl2 - (multiplier * atr)

                    # Calcul du SuperTrend
                    supertrend = pd.Series(index=df.index, dtype=float)
                    direction = pd.Series(1, index=df.index)

                    for i in range(1, len(df)):
                        if close.iloc[i] > upper_band.iloc[i-1]:
                            direction.iloc[i] = 1
                        elif close.iloc[i] < lower_band.iloc[i-1]:
                            direction.iloc[i] = -1
                        else:
                            direction.iloc[i] = direction.iloc[i-1]

                        if direction.iloc[i] < 0 and upper_band.iloc[i] < supertrend.iloc[i-1] if i > 0 else False:
                            supertrend.iloc[i] = upper_band.iloc[i]
                        elif direction.iloc[i] == 1 and lower_band.iloc[i] > supertrend.iloc[i-1] if i > 0 else False:
                            supertrend.iloc[i] = lower_band.iloc[i]
                        else:
                            if direction.iloc[i] == 1:
                                supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1] if i > 0 else lower_band.iloc[i])
                            else:
                                supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1] if i > 0 else upper_band.iloc[i])

                    df['SUPERTREND_14_2.0'] = supertrend

                elif indicator == 'PSAR_0.02_0.2':
                    # Parabolic SAR (0.02, 0.2)
                    high = df['High']
                    low = df['Low']

                    # Initialisation
                    psar = df['Close'].copy()
                    trend = pd.Series(1, index=df.index)
                    af = 0.02  # Facteur d'accélération initial
                    max_af = 0.2  # Facteur d'accélération maximum

                    # Points extrêmes
                    extreme_point = high.iloc[0]

                    for i in range(2, len(df)):
                        if trend.iloc[i-1] == 1:
                            # Tendance haussière
                            psar.iloc[i] = psar.iloc[i-1] + af * (extreme_point - psar.iloc[i-1])

                            # Vérification du renversement
                            if low.iloc[i] < psar.iloc[i]:
                                trend.iloc[i] = -1
                                psar.iloc[i] = max(high.iloc[i-1], high.iloc[i-2], high.iloc[i])
                                extreme_point = low.iloc[i]
                                af = 0.02
                            else:
                                trend.iloc[i] = 1
                                if high.iloc[i] > extreme_point:
                                    extreme_point = high.iloc[i]
                                    af = min(af + 0.02, max_af)
                        else:
                            # Tendance baissière
                            psar.iloc[i] = psar.iloc[i-1] + af * (extreme_point - psar.iloc[i-1])

                            # Vérification du renversement
                            if high.iloc[i] > psar.iloc[i]:
                                trend.iloc[i] = 1
                                psar.iloc[i] = min(low.iloc[i-1], low.iloc[i-2], low.iloc[i])
                                extreme_point = high.iloc[i]
                                af = 0.02
                            else:
                                trend.iloc[i] = -1
                                if low.iloc[i] < extreme_point:
                                    extreme_point = low.iloc[i]
                                    af = min(af + 0.02, max_af)

                    df['PSAR_0.02_0.2'] = psar

                elif indicator == 'ICHIMOKU_9_26_52':
                    # Ichimoku Cloud
                    high = df['High']
                    low = df['Low']

                    # Tenkan-sen (Conversion Line)
                    period9_high = high.rolling(window=9).max()
                    period9_low = low.rolling(window=9).min()
                    df['Ichimoku_Tenkan'] = (period9_high + period9_low) / 2

                    # Kijun-sen (Base Line)
                    period26_high = high.rolling(window=26).max()
                    period26_low = low.rolling(window=26).min()
                    df['Ichimoku_Kijun'] = (period26_high + period26_low) / 2

                    # Senkou Span A (Leading Span A)
                    df['Ichimoku_Senkou_A'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(26)

                    # Senkou Span B (Leading Span B)
                    period52_high = high.rolling(window=52).max()
                    period52_low = low.rolling(window=52).min()
                    df['Ichimoku_Senkou_B'] = ((period52_high + period52_low) / 2).shift(26)

                    # Chikou Span (Lagging Span)
                    df['Ichimoku_Chikou'] = df['Close'].shift(-26)

        # Indicateurs de volatilité
        if 'volatility' in indicators_config:
            for indicator in indicators_config['volatility']:
                if indicator == 'ATR_14':
                    # Average True Range (14 périodes)
                    high = df['High']
                    low = df['Low']
                    close = df['Close']

                    tr1 = high - low
                    tr2 = (high - close.shift()).abs()
                    tr3 = (low - close.shift()).abs()

                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    df['ATR_14'] = tr.rolling(window=14).mean()

                elif indicator == 'BB_20_2.0':
                    # Bollinger Bands (20, 2.0)
                    sma = df['Close'].rolling(window=20).mean()
                    std = df['Close'].rolling(window=20).std()
                    df['BB_upper'] = sma + (std * 2)
                    df['BB_middle'] = sma
                    df['BB_lower'] = sma - (std * 2)

        # Indicateurs de volume
        if 'volume' in indicators_config:
            for indicator in indicators_config['volume']:
                if indicator == 'VWAP':
                    # Volume Weighted Average Price
                    if 'Volume' in df.columns:
                        tp = (df['High'] + df['Low'] + df['Close']) / 3
                        df['VWAP'] = (tp * df['Volume']).cumsum() / df['Volume'].cumsum()

                elif indicator == 'OBV':
                    # On-Balance Volume
                    if 'Volume' in df.columns:
                        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        # Suppression des valeurs NaN
        df = df.dropna()

        return df

    def process_asset(self, asset: str) -> None:
        """Traite un actif pour toutes les timeframes."""
        logger.info(f"Traitement de l'actif: {asset}")

        for timeframe in self.timeframes:
            try:
                # Chemin du fichier source
                file_name = f"{asset}_{timeframe}.csv"
                file_path = self.raw_data_dir / file_name

                if not file_path.exists():
                    logger.warning(f"Fichier introuvable: {file_path}")
                    continue

                # Lecture du fichier CSV
                df = self._read_csv_file(file_path)
                if df is None or df.empty:
                    logger.warning(f"Impossible de lire le fichier ou fichier vide: {file_path}")
                    continue

                # Calcul des indicateurs techniques
                df = self._calculate_indicators(df, timeframe)

                # Séparation en ensembles d'entraînement, de validation et de test (80/10/10)
                total_rows = len(df)
                train_size = int(0.8 * total_rows)
                val_size = int(0.1 * total_rows)

                train_df = df.iloc[:train_size]
                val_df = df.iloc[train_size:train_size + val_size]
                test_df = df.iloc[train_size + val_size:]

                # Enregistrement des données traitées
                self._save_parquet(train_df, asset, timeframe, 'train')
                self._save_parquet(val_df, asset, timeframe, 'val')
                self._save_parquet(test_df, asset, timeframe, 'test')

            except Exception as e:
                logger.error(f"Erreur lors du traitement de {asset} ({timeframe}): {e}")

    def _save_parquet(self, df: pd.DataFrame, asset: str, timeframe: str, split: str = 'train') -> None:
        """Enregistre le DataFrame au format Parquet dans la structure de dossiers appropriée."""
        try:
            # Création du répertoire de destination
            if split == 'train':
                save_dir = self.indicators_dir / 'train' / timeframe
            else:
                save_dir = self.indicators_dir / split / asset.lower()

            save_dir.mkdir(parents=True, exist_ok=True)

            # Nom du fichier de sortie
            if split == 'train':
                output_file = save_dir / f"{asset}.parquet"
            else:
                output_file = save_dir / f"{timeframe}.parquet"

            # Sauvegarde en Parquet
            df.to_parquet(output_file)
            logger.info(f"Fichier enregistré: {output_file}")

        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement du fichier Parquet: {e}")

    def process_all_assets(self) -> None:
        """Traite tous les actifs configurés."""
        logger.info("Début du traitement de tous les actifs")

        for asset in self.assets:
            try:
                self.process_asset(asset)
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {asset}: {e}")
                continue

        logger.info("Traitement terminé pour tous les actifs")


def main():
    """Fonction principale."""
    try:
        processor = MarketDataProcessor()
        processor.process_all_assets()
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du script: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
