

import pandas as pd
import pandas_ta as ta
import os
from pathlib import Path
import yaml

# --- Configuration ---
BASE_DIR = Path('/home/morningstar/Documents/trading/bot')
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed' / 'indicators'
CONFIG_FILE = BASE_DIR / 'config' / 'config.yaml'
DATA_CONFIG_FILE = BASE_DIR / 'config' / 'data.yaml'
SPLIT_RATIOS = {'train': 0.7, 'val': 0.15, 'test': 0.15}

# --- Fonctions ---

def load_config():
    print(f"Chargement de la configuration depuis {CONFIG_FILE}...")
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Chargement de la configuration des données depuis {DATA_CONFIG_FILE}...")
    with open(DATA_CONFIG_FILE, 'r') as f:
        data_config = yaml.safe_load(f)
        
    assets = data_config['data']['file_structure']['assets']
    timeframes = data_config['data']['file_structure']['timeframes']
    indicators_config = config['environment']['observation']['features']['indicators']
    
    return assets, timeframes, indicators_config

def calculate_indicators(df, timeframe_indicators):
    print("  -> Calcul des indicateurs (méthode manuelle et robuste)...")
    
    # On s'assure que les colonnes nécessaires sont présentes
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("Le DataFrame ne contient pas les colonnes nécessaires (open, high, low, close, volume)")

    # Calcul de chaque indicateur individuellement
    # Cela évite les erreurs de la méthode ta.strategy()
    df.ta.rsi(length=14, append=True)
    df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.ema(length=5, append=True)
    df.ta.ema(length=12, append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.ema(length=26, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.obv(append=True)
    df.ta.ema(length=200, append=True)
    df.ta.sma(length=200, append=True)

    # Forcer toutes les colonnes en minuscules pour la cohérence
    df.columns = [col.lower() for col in df.columns]

    # Renommer les colonnes pour correspondre exactement à la configuration attendue
    rename_dict = {
        'macdh_12_26_9': 'macd_hist',
        'atrr_14': 'atr_14',
        'bbl_20_2.0': 'bb_lower',
        'bbm_20_2.0': 'bb_middle',
        'bbu_20_2.0': 'bb_upper',
        'adx_14': 'adx_14',
        'ema_12': 'ema_12',
        'ema_26': 'ema_26',
        'sma_20': 'sma_20'
    }
    df.rename(columns=rename_dict, inplace=True)
    
    return df

def process_file(asset, timeframe, indicators_config):
    print(f"--- Traitement de {asset} - {timeframe} ---")
    
    csv_path = RAW_DATA_DIR / asset / f'{timeframe}.csv'
    if not csv_path.exists():
        print(f"  -> Fichier CSV non trouvé: {csv_path}. Passage au suivant.")
        return

    print(f"  -> Chargement de {csv_path}...")
    df = pd.read_csv(csv_path)
    
    df.rename(columns={
        'TIMESTAMP': 'timestamp',
        'OPEN': 'open',
        'HIGH': 'high',
        'LOW': 'low',
        'CLOSE': 'close',
        'VOLUME': 'volume'
    }, inplace=True)

    print("  -> Nettoyage des données OHLCV (ffill/bfill)...")
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    df[ohlcv_cols] = df[ohlcv_cols].fillna(method='ffill').fillna(method='bfill')
    if df['close'].isna().any():
        raise ValueError(f"NaN values remain in CLOSE column after cleaning for {asset}/{timeframe}")
    
    nan_pct = df['close'].isna().mean() * 100
    if nan_pct > 0:
        print(f"  -> ATTENTION: {nan_pct:.2f}% de NaN résiduels dans CLOSE pour {asset}/{timeframe} après nettoyage.")
    else:
        print("  -> Aucune valeur NaN dans la colonne CLOSE après nettoyage.")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    df = calculate_indicators(df, indicators_config.get(timeframe, []))
    
    # Renommer les colonnes pour correspondre exactement à la configuration
    rename_dict = {
        'MACDH_12_26_9': 'macd_hist',
        'ATRR_14': 'atr_14',
        'BBL_20_2.0': 'bb_lower',
        'BBM_20_2.0': 'bb_middle',
        'BBU_20_2.0': 'bb_upper',
        'EMA_12': 'ema_12',
        'EMA_26': 'ema_26',
        'SMA_20': 'sma_20',
        'ADX_14': 'adx_14',
        'RSI_14': 'rsi_14'
    }
    # Ne renommer que les colonnes qui existent pour éviter les erreurs
    df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns}, inplace=True)
    
    # Laisser les colonnes en minuscules, ne pas convertir en majuscules
    # df.columns = [col.upper() for col in df.columns]
    df.dropna(inplace=True)
    print(f"  -> {len(df.columns)} colonnes après ajout des indicateurs.")
    print(f"  -> {len(df)} lignes de données après suppression des NaN.")

    print("  -> Division des données en train/val/test...")
    n = len(df)
    train_end = int(n * SPLIT_RATIOS['train'])
    val_end = train_end + int(n * SPLIT_RATIOS['val'])
    
    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]
    
    print(f"     - Train: {len(df_train)} lignes")
    print(f"     - Val:   {len(df_val)} lignes")
    print(f"     - Test:  {len(df_test)} lignes")

    for split_name, df_split in [('train', df_train), ('val', df_val), ('test', df_test)]:
        if df_split.empty:
            print(f"  -> Le DataFrame pour {split_name} est vide. Sauvegarde ignorée.")
            continue
        output_dir = PROCESSED_DATA_DIR / split_name / asset
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f'{timeframe}.parquet'
        print(f"  -> Sauvegarde de {split_name} dans {output_path}...")
        df_split.to_parquet(output_path)

    print(f"--- Traitement de {asset} - {timeframe} terminé ---\n")

def main():
    print("=== DÉBUT DE LA RECONSTITUTION DES DONNÉES PARQUET ===")
    assets, timeframes, indicators_config = load_config()
    print(f"\nActifs à traiter: {assets}")
    print(f"Timeframes à traiter: {timeframes}\n")
    for asset in assets:
        for timeframe in timeframes:
            try:
                process_file(asset, timeframe, indicators_config)
            except Exception as e:
                print(f"ERREUR lors du traitement de {asset} - {timeframe}: {e}")
                continue # Passe au suivant en cas d'erreur
    print("=== RECONSTITUTION TERMINÉE ===")

if __name__ == "__main__":
    main()
