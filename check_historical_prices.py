import requests
from datetime import datetime
import pandas as pd

def get_historical_price(symbol, timestamp):
    """Récupère le prix historique d'une paire de trading sur Binance."""
    url = "https://api.binance.com/api/v3/klines"

    # Convertir la date en millisecondes
    start_time = int(datetime.strptime(timestamp, "%Y-%m-%d").timestamp() * 1000)
    end_time = start_time + 24 * 60 * 60 * 1000  # 24 heures plus tard

    params = {
        'symbol': symbol,
        'interval': '1d',  # 1 jour
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if not data:
            return None

        # Les données sont au format [timestamp, open, high, low, close, volume, ...]
        return {
            'symbol': symbol,
            'date': datetime.fromtimestamp(data[0][0]/1000).strftime('%Y-%m-%d'),
            'open': float(data[0][1]),
            'high': float(data[0][2]),
            'low': float(data[0][3]),
            'close': float(data[0][4]),
            'volume': float(data[0][5])
        }

    except Exception as e:
        print(f"Erreur pour {symbol}: {str(e)}")
        return None

# Paires de trading à vérifier
symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'SOLUSDT', 'ADAUSDT']
date_to_check = '2024-01-01'

print(f"Récupération des prix pour le {date_to_check}...\n")

# Récupérer les prix pour chaque symbole
prices = []
for symbol in symbols:
    price_data = get_historical_price(symbol, date_to_check)
    if price_data:
        prices.append(price_data)

# Afficher les résultats dans un tableau
if prices:
    df = pd.DataFrame(prices)
    print(df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']].to_string(index=False))
else:
    print("Aucune donnée trouvée pour cette date.")

print("\nNote: Ces données sont fournies par l'API Binance.")
