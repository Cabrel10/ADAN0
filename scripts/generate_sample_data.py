
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_sample_data():
    """Génère des données d'exemple pour les tests"""
    
    # Paramètres
    symbols = ['BTCUSDT']
    timeframes = ['5m', '1h', '4h']
    
    # Période de données
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # 30 jours de données
    
    print(f'📊 Génération de données de test...')
    print(f'   Période: {start_date.date()} → {end_date.date()}')
    print(f'   Symboles: {symbols}')
    print(f'   Timeframes: {timeframes}')
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f'
🔄 Génération {symbol} {timeframe}...')
            
            # Déterminer la fréquence selon le timeframe
            if timeframe == '5m':
                freq = '5T'
                periods = int((end_date - start_date).total_seconds() / 300)  # 5 minutes
            elif timeframe == '1h':
                freq = '1H'  
                periods = int((end_date - start_date).total_seconds() / 3600)  # 1 heure
            else:  # 4h
                freq = '4H'
                periods = int((end_date - start_date).total_seconds() / 14400)  # 4 heures
            
            # Générer les timestamps
            timestamps = pd.date_range(start=start_date, periods=periods, freq=freq)
            
            # Générer les données OHLCV réalistes
            base_price = 50000 + np.random.randn() * 5000
            
            # Prix d'ouverture (random walk avec tendance)
            price_changes = np.random.randn(periods) * 100  # Volatilité
            prices = base_price + np.cumsum(price_changes)
            
            # Générer OHLCV
            data = []
            for i in range(periods):
                open_price = prices[max(0, i-1)]
                close_price = prices[i]
                high_price = max(open_price, close_price) + abs(np.random.randn() * 50)
                low_price = min(open_price, close_price) - abs(np.random.randn() * 50)
                volume = 1000 + np.random.exponential(500)  # Volume réaliste
                
                data.append({
                    'timestamp': timestamps[i],
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            # Ajouter des indicateurs techniques
            df['rsi'] = calculate_rsi(df['close'], 14)
            df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['close'])
            df['sma_20'] = df['close'].rolling(20).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            
            # Créer le répertoire
            output_dir = f'data/processed/indicators/val/{symbol}'
            os.makedirs(output_dir, exist_ok=True)
            
            # Sauvegarder
            output_file = f'{output_dir}/{timeframe}.parquet'
            df.to_parquet(output_file)
            
            print(f'✅ {len(df)} lignes sauvegardées dans {output_file}')

def calculate_rsi(prices, period=14):
    """Calcule le RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calcule le MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calcule les bandes de Bollinger"""
    middle = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower

if __name__ == '__main__':
    generate_sample_data()
    print('
✅ Génération de données terminée!')
