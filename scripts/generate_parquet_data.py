#!/usr/bin/env python3
"""
Script pour générer les fichiers Parquet à partir des données brutes CSV.
Assure la conformité avec la structure et les indicateurs attendus.
"""
import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import ta
from tqdm import tqdm
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('parquet_generation.log')
    ]
)
logger = logging.getLogger(__name__)

class DataGenerator:
    def __init__(self, config_path: str = None):
        """Initialise le générateur avec la configuration."""
        if config_path is None:
            config_path = 'bot/config/config.yaml'
        
        self.config = self._load_config(config_path)
        self.base_dir = Path('/home/morningstar/Documents/trading')
        self.raw_data_dir = self.base_dir / 'data/raw'
        self.processed_dir = self.base_dir / 'data/processed'
        self.indicators_dir = self.processed_dir / 'indicators'
        
        # Paramètres de données
        self.assets = self.config['environment']['assets']
        self.timeframes = ['5m', '1h', '4h']  # Définition directe des timeframes
        
        # Configuration des indicateurs par timeframe
        self.features_config = {
            '5m': {
                'indicators': [
                    'RSI_14', 'ATR_14', 'ADX_14', 'EMA_5', 'EMA_20', 'EMA_50', 
                    'MACD_12_26_9', 'SUPERTREND_14_2.0', 'BB_UPPER', 'VWAP', 'OBV',
                    'ATR_PCT', 'STOCH_K_14_3_3', 'EMA_200', 'EMA_RATIO_FAST_SLOW',
                    'EMA_12', 'EMA_26'
                ]
            },
            '1h': {
                'indicators': [
                    'RSI_14', 'ATR_14', 'ADX_14', 'EMA_12', 'EMA_50', 'EMA_200', 
                    'MACD_12_26_9', 'SUPERTREND_14_2.0', 'BB_UPPER', 'VWAP', 'OBV',
                    'ATR_PCT', 'STOCH_K_14_3_3', 'EMA_5', 'EMA_20', 'EMA_RATIO_FAST_SLOW',
                    'EMA_26'
                ]
            },
            '4h': {
                'indicators': [
                    'RSI_14', 'ATR_14', 'ADX_14', 'EMA_12', 'EMA_50', 'EMA_200', 
                    'MACD_12_26_9', 'SUPERTREND_14_2.0', 'BB_UPPER', 'VWAP', 'OBV',
                    'ATR_PCT', 'STOCH_K_14_3_3', 'EMA_5', 'EMA_20', 'EMA_RATIO_FAST_SLOW',
                    'EMA_26'
                ]
            }
        }
        
        # Créer les répertoires si nécessaire
        self._create_directories()
    
    def _load_config(self, config_path: str) -> dict:
        """Charge le fichier de configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            raise
    
    def _create_directories(self):
        """Crée les répertoires nécessaires."""
        # Répertoires principaux
        self.indicators_dir.mkdir(parents=True, exist_ok=True)
        
        # Répertoires pour chaque split (train/val/test)
        for split in ['train', 'val', 'test']:
            split_dir = self.indicators_dir / split
            split_dir.mkdir(exist_ok=True)
            
            # Répertoires pour chaque actif
            for asset in self.assets:
                asset_dir = split_dir / asset.upper()
                asset_dir.mkdir(exist_ok=True)
    
    def _load_raw_data(self, asset: str, timeframe: str) -> pd.DataFrame:
        """Charge les données brutes pour un actif et une timeframe donnés."""
        filename = f"{asset.upper()}_{timeframe}.csv"
        filepath = self.raw_data_dir / filename
        
        try:
            # Charger les données brutes avec gestion des types
            df = pd.read_csv(filepath)
            
            # Vérifier et normaliser les noms de colonnes
            column_mapping = {
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            
            # Renommer les colonnes
            df = df.rename(columns=column_mapping)
            
            # Vérifier les colonnes requises
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Colonne manquante dans {filename}: {col}")
            
            # Convertir la date en datetime et la définir comme index
            df['datetime'] = pd.to_datetime(df['timestamp'])
            df.set_index('datetime', inplace=True)
            
            # Supprimer les doublons de dates
            df = df[~df.index.duplicated(keep='first')]
            
            # Vérifier les valeurs manquantes ou nulles
            if df.isnull().any().any():
                logger.warning(f"Valeurs manquantes détectées dans {filename} avant le traitement")
            
            # Vérifier les zéros dans les données OHLC
            zero_mask = (df[['open', 'high', 'low', 'close']] == 0).any(axis=1)
            if zero_mask.any():
                logger.warning(f"{zero_mask.sum()} lignes avec des zéros détectées dans {filename}, suppression...")
                df = df[~zero_mask]
                
            # Vérifier les volumes nuls
            zero_volume = (df['volume'] == 0)
            if zero_volume.any():
                logger.warning(f"{zero_volume.sum()} lignes avec volume nul détectées dans {filename}, suppression...")
                df = df[~zero_volume]
            
            # Filtrer à partir de janvier 2024
            df = df[df.index >= '2024-01-01']
            
            # Trier par date
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {filepath}: {e}")
            return None
    
    def _add_technical_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Ajoute les indicateurs techniques spécifiés dans la configuration."""
        if timeframe not in self.features_config:
            logger.warning(f"Aucune configuration d'indicateurs pour le timeframe {timeframe}")
            return df
        
        indicators = self.features_config[timeframe].get('indicators', [])
        
        # Faire une copie pour éviter les avertissements
        df = df.copy()
        
        # Ajouter chaque indicateur
        for indicator in indicators:
            try:
                if indicator == 'RSI_14':
                    df['RSI_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
                
                elif indicator == 'CCI_20':
                    df['CCI_20'] = ta.trend.CCIIndicator(
                        high=df['high'], 
                        low=df['low'], 
                        close=df['close'], 
                        window=20
                    ).cci()
                
                elif indicator == 'ROC_9':
                    df['ROC_9'] = ta.momentum.ROCIndicator(df['close'], window=9).roc()
                
                elif indicator == 'MFI_14':
                    df['MFI_14'] = ta.volume.MFIIndicator(
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        volume=df['volume'],
                        window=14
                    ).money_flow_index()
                
                elif indicator == 'EMA_5':
                    df['EMA_5'] = ta.trend.EMAIndicator(df['close'], window=5).ema_indicator()
                
                elif indicator == 'EMA_20':
                    df['EMA_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
                
                elif indicator == 'SUPERTREND_14_2.0':
                    st = ta.trend.SuperTrend(
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        window=14,
                        multiplier=2.0
                    )
                    df['SUPERTREND_14_2.0'] = st.super_trend()
                
                elif indicator == 'PSAR_0.02_0.2':
                    psar = ta.trend.PSARIndicator(
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        step=0.02,
                        max_step=0.2
                    )
                    df['PSAR_0.02_0.2'] = psar.psar()
                
                elif indicator == 'ATR_14':
                    df['ATR_14'] = ta.volatility.AverageTrueRange(
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        window=14
                    ).average_true_range()
                
                elif indicator == 'BB_UPPER':
                    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
                    df['BB_UPPER'] = bb.bollinger_hband()
                    df['BB_MIDDLE'] = bb.bollinger_mavg()
                    df['BB_LOWER'] = bb.bollinger_lband()
                
                elif indicator == 'VWAP':
                    # VWAP nécessite des données tick-by-tick, on utilise un proxy avec le prix moyen
                    df['TP'] = (df['high'] + df['low'] + df['close']) / 3
                    df['TPV'] = df['TP'] * df['volume']
                    df['VWAP'] = df['TPV'].cumsum() / df['volume'].cumsum()
                    df.drop(['TP', 'TPV'], axis=1, inplace=True)
                
                elif indicator == 'OBV':
                    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
                        close=df['close'],
                        volume=df['volume']
                    ).on_balance_volume()
                
                elif indicator == 'ATR_PCT':
                    atr = ta.volatility.AverageTrueRange(
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        window=14
                    ).average_true_range()
                    df['ATR_PCT'] = (atr / df['close']) * 100
                
                elif indicator == 'MACD_12_26_9':
                    macd = ta.trend.MACD(
                        close=df['close'],
                        window_slow=26,
                        window_fast=12,
                        window_sign=9
                    )
                    df['MACD_12_26_9'] = macd.macd()
                    df['MACD_SIGNAL_12_26_9'] = macd.macd_signal()
                    df['MACD_HIST_12_26_9'] = macd.macd_diff()
                
                elif indicator == 'ADX_14':
                    df['ADX_14'] = ta.trend.ADXIndicator(
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        window=14
                    ).adx()
                
                elif indicator == 'STOCH_K_14_3_3':
                    stoch = ta.momentum.StochasticOscillator(
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        window=14,
                        smooth_window=3
                    )
                    df['STOCH_K_14_3_3'] = stoch.stoch()
                    df['STOCH_D_14_3_3'] = stoch.stoch_signal()
                
                elif indicator == 'EMA_200':
                    df['EMA_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
                
                elif indicator == 'EMA_RATIO_FAST_SLOW':
                    ema_fast = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
                    ema_slow = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
                    df['EMA_RATIO_FAST_SLOW'] = ema_fast / ema_slow
                
                elif indicator == 'EMA_12':
                    df['EMA_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
                
                elif indicator == 'EMA_26':
                    df['EMA_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
            
            except Exception as e:
                logger.warning(f"Erreur lors de l'ajout de l'indicateur {indicator}: {e}")
        
        return df
    
    def _split_data(self, df: pd.DataFrame) -> dict:
        """Sépare les données en ensembles d'entraînement, de validation et de test."""
        # Définir les plages de dates pour chaque ensemble
        train_end = df.index.max() - timedelta(days=60)  # 2 derniers mois pour le test
        val_end = train_end - timedelta(days=30)         # 1 mois avant pour la validation
        
        train_data = df[df.index <= val_end]
        val_data = df[(df.index > val_end) & (df.index <= train_end)]
        test_data = df[df.index > train_end]
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def _clean_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Nettoie les indicateurs techniques en supprimant les lignes avec des valeurs manquantes ou nulles."""
        if timeframe not in self.features_config:
            return df
            
        # Faire une copie pour éviter les avertissements
        df_cleaned = df.copy()
        
        # Récupérer la liste des indicateurs pour cette timeframe
        indicators = self.features_config[timeframe].get('indicators', [])
        
        # Si pas d'indicateurs, retourner les données telles quelles
        if not indicators:
            return df_cleaned
            
        # Liste des colonnes d'indicateurs à vérifier
        indicator_columns = []
        
        # Mapper les indicateurs aux colonnes correspondantes
        indicator_map = {
            'RSI_14': ['RSI_14'],
            'CCI_20': ['CCI_20'],
            'ROC_9': ['ROC_9'],
            'MFI_14': ['MFI_14'],
            'EMA_5': ['EMA_5'],
            'EMA_20': ['EMA_20'],
            'SUPERTREND_14_2.0': ['SUPERTREND_14_2.0'],
            'PSAR_0.02_0.2': ['PSAR_0.02_0.2'],
            'ATR_14': ['ATR_14'],
            'BB_UPPER': ['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER'],
            'VWAP': ['VWAP'],
            'OBV': ['OBV'],
            'ATR_PCT': ['ATR_PCT'],
            'MACD_12_26_9': ['MACD_12_26_9', 'MACD_SIGNAL_12_26_9', 'MACD_HIST_12_26_9'],
            'ADX_14': ['ADX_14'],
            'STOCH_K_14_3_3': ['STOCH_K_14_3_3', 'STOCH_D_14_3_3'],
            'EMA_200': ['EMA_200'],
            'EMA_RATIO_FAST_SLOW': ['EMA_RATIO_FAST_SLOW'],
            'EMA_12': ['EMA_12'],
            'EMA_26': ['EMA_26']
        }
        
        # Construire la liste des colonnes d'indicateurs à vérifier
        for indicator in indicators:
            if indicator in indicator_map:
                indicator_columns.extend(indicator_map[indicator])
        
        # Supprimer les doublons
        indicator_columns = list(dict.fromkeys(indicator_columns))
        
        # Ne garder que les colonnes qui existent dans le DataFrame
        indicator_columns = [col for col in indicator_columns if col in df_cleaned.columns]
        
        if not indicator_columns:
            return df_cleaned
            
        # Avant le nettoyage
        initial_rows = len(df_cleaned)
        
        # Supprimer les lignes avec des valeurs manquantes dans les indicateurs
        df_cleaned = df_cleaned.dropna(subset=indicator_columns, how='any')
        
        # Supprimer les lignes avec des valeurs infinies dans les indicateurs
        for col in indicator_columns:
            df_cleaned = df_cleaned[~df_cleaned[col].isin([np.inf, -np.inf])]
        
        # Vérifier les zéros dans les indicateurs
        zero_mask = (df_cleaned[indicator_columns] == 0).any(axis=1)
        if zero_mask.any():
            logger.warning(f"{zero_mask.sum()} lignes avec des zéros dans les indicateurs détectées, suppression...")
            df_cleaned = df_cleaned[~zero_mask]
        
        # Après le nettoyage
        final_rows = len(df_cleaned)
        removed_rows = initial_rows - final_rows
        
        if removed_rows > 0:
            logger.info(f"Nettoyage des indicateurs: {removed_rows} lignes supprimées sur {initial_rows} ({removed_rows/initial_rows*100:.2f}%)")
        
        return df_cleaned
    
    def _save_parquet(self, df: pd.DataFrame, asset: str, timeframe: str, split: str):
        """Enregistre les données au format Parquet."""
        if df.empty:
            logger.warning(f"Aucune donnée à enregistrer pour {asset} {timeframe} {split}")
            return
        
        # Nettoyer les indicateurs avant l'enregistrement
        df_cleaned = self._clean_indicators(df, timeframe)
        
        # Créer le répertoire de destination
        output_dir = self.indicators_dir / split / asset.upper()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Chemin du fichier de sortie
        output_file = output_dir / f"{timeframe}.parquet"
        
        try:
            # Sélectionner uniquement les colonnes nécessaires
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            indicator_cols = [col for col in df_cleaned.columns if col not in required_cols]
            
            # Trier les colonnes pour une cohérence
            all_cols = required_cols + indicator_cols
            df_cleaned = df_cleaned[all_cols]
            
            # Convertir les noms de colonnes en majuscules
            df_cleaned.columns = [col.upper() for col in df_cleaned.columns]
            
            # Enregistrer en Parquet
            df_cleaned.to_parquet(output_file, index=True)
            logger.info(f"Données nettoyées enregistrées dans {output_file} ({len(df_cleaned)} lignes)")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de {output_file}: {e}")
            raise
    
    def process_asset(self, asset: str):
        """Traite un actif pour toutes les timeframes."""
        logger.info(f"Traitement de l'actif: {asset}")
        
        for timeframe in self.timeframes:
            logger.info(f"  - Timeframe: {timeframe}")
            
            # Charger les données brutes
            df = self._load_raw_data(asset, timeframe)
            if df is None or df.empty:
                logger.warning(f"Aucune donnée pour {asset} {timeframe}")
                continue
            
            # Ajouter les indicateurs techniques
            df = self._add_technical_indicators(df, timeframe)
            
            # Séparer en train/val/test
            split_data = self._split_data(df)
            
            # Enregistrer chaque split
            for split, data in split_data.items():
                self._save_parquet(data, asset, timeframe, split)
    
    def run(self):
        """Exécute le traitement pour tous les actifs."""
        logger.info("Début de la génération des données Parquet")
        
        for asset in tqdm(self.assets, desc="Traitement des actifs"):
            self.process_asset(asset)
        
        logger.info("Génération des données Parquet terminée")

if __name__ == "__main__":
    # Créer et exécuter le générateur de données
    generator = DataGenerator()
    generator.run()
