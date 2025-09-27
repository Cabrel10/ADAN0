import pandas as pd
import numpy as np
import talib
import os

def calculate_4h_indicators(df):
    """
    Calcule les indicateurs techniques pour le timeframe 4h
    """
    result_df = df.copy()
    
    # S'assurer que les colonnes OHLCV de base sont en majuscules
    ohlcv_mapping = {
        'open': 'OPEN',
        'high': 'HIGH', 
        'low': 'LOW',
        'close': 'CLOSE',
        'volume': 'VOLUME'
    }
    
    for old_name, new_name in ohlcv_mapping.items():
        if old_name in result_df.columns:
            result_df = result_df.rename(columns={old_name: new_name})
    
    required_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
    for col in required_cols:
        if col not in result_df.columns:
            raise ValueError(f"Colonne {col} manquante dans le dataframe")
    
    # Convertir en numpy arrays
    open_prices = result_df['OPEN'].values.astype(float)
    high_prices = result_df['HIGH'].values.astype(float)
    low_prices = result_df['LOW'].values.astype(float)
    close_prices = result_df['CLOSE'].values.astype(float)
    volume = result_df['VOLUME'].values.astype(float)

    new_df = pd.DataFrame()
    
    if 'timestamp' in result_df.columns:
        new_df['timestamp'] = result_df['timestamp']
    
    new_df['OPEN'] = result_df['OPEN']
    new_df['HIGH'] = result_df['HIGH']
    new_df['LOW'] = result_df['LOW']
    new_df['CLOSE'] = result_df['CLOSE']
    new_df['VOLUME'] = result_df['VOLUME']

    # Indicateurs techniques pour 4h
    new_df['RSI_14'] = talib.RSI(close_prices, timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    new_df['MACD_HIST'] = macd_hist
    new_df['ADX_14'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
    new_df['ATR_14'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
    new_df['OBV'] = talib.OBV(close_prices, volume)
    new_df['EMA_50'] = talib.EMA(close_prices, timeperiod=50)
    new_df['EMA_200'] = talib.EMA(close_prices, timeperiod=200)
    stoch_k, stoch_d = talib.STOCH(high_prices, low_prices, close_prices, 14, 3, 0, 3, 0)
    new_df['STOCHk_14_3_3'] = stoch_k
    new_df['STOCHd_14_3_3'] = stoch_d
    new_df['SMA_200'] = talib.SMA(close_prices, timeperiod=200)

    return new_df


def process_4h_parquet_files(base_path='bot/data/processed/backups'):
    """
    Traite tous les fichiers parquet de timeframe 4h
    """
    splits = ['train', 'test', 'val']
    assets = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT']
    
    for split in splits:
        for asset in assets:
            file_path = f'{base_path}/{split}/{asset}/4h.parquet'
            
            if os.path.exists(file_path):
                print(f'Traitement de {file_path}...')
                
                try:
                    df = pd.read_parquet(file_path)
                    print(f'  - Fichier chargé: {len(df)} lignes, {len(df.columns)} colonnes')
                    
                    new_df = calculate_4h_indicators(df)
                    print(f'  - Indicateurs 4h calculés: {len(new_df.columns)} colonnes')
                    
                    backup_path = file_path.replace('.parquet', '_backup.parquet')
                    df.to_parquet(backup_path, index=False)
                    print(f'  - Sauvegarde créée: {backup_path}')
                    
                    new_df.to_parquet(file_path, index=False)
                    print(f'  - Nouveau fichier sauvegardé: {file_path}')
                    
                except Exception as e:
                    print(f'  - Erreur lors du traitement de {file_path}: {str(e)}\n')
            else:
                print(f'Fichier non trouvé: {file_path}')


def verify_4h_files(base_path='bot/data/processed/backups'):
    """
    Vérifie que tous les fichiers 4h ont bien été modifiés avec les colonnes attendues
    """
    print("=== VÉRIFICATION DES FICHIERS 4H MODIFIÉS ===\n")
    
    splits = ['train', 'test', 'val']
    assets = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT']
    
    expected_columns = [
        'timestamp', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME',
        'RSI_14', 'MACD_HIST', 'ADX_14', 'ATR_14', 'OBV',
        'EMA_50', 'EMA_200', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'SMA_200'
    ]
    
    for split in splits:
        for asset in assets:
            file_path = f'{base_path}/{split}/{asset}/4h.parquet'
            
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                print(f'{split}/{asset} - 4h:')
                print(f'  Colonnes: {len(df.columns)} (attendu: {len(expected_columns)})')
                print(f'  Lignes: {len(df)}')
                
                missing_cols = set(expected_columns) - set(df.columns)
                extra_cols = set(df.columns) - set(expected_columns)
                
                if missing_cols:
                    print(f'  ⚠️ Colonnes manquantes: {missing_cols}')
                if extra_cols:
                    print(f'  ⚠️ Colonnes supplémentaires: {extra_cols}')
                if not missing_cols and not extra_cols:
                    print(f'  ✅ Structure correcte')
                
                print()
            else:
                print(f'{split}/{asset} - 4h: ❌ Fichier non trouvé')


if __name__ == "__main__":
    print("=== MODIFICATION DES FICHIERS PARQUET 4H ===\n")
    
    process_4h_parquet_files()
    
    print("\n=== TRAITEMENT TERMINÉ ===\n")
    
    verify_4h_files()

