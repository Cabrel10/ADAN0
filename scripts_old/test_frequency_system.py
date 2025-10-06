
import sys
sys.path.append('src')

from adan_trading_bot.core.frequency_manager import FrequencyManager
import pandas as pd
import time

def test_frequency_manager():
    """Test du système de gestion des fréquences"""
    
    # Configuration de test
    config = {
        'frequency_validation': {
            'enabled': True,
            'min_trades_per_timeframe': {
                '5m': [2, 5],  # Seuils réduits pour le test
                '1h': [1, 3],
                '4h': [1, 2]
            },
            'reward_bonus': {
                'frequency_multiplier': 1.5,
                'diversity_bonus': 2.0
            }
        }
    }
    
    # Initialiser le gestionnaire
    fm = FrequencyManager(config)
    
    # Test 1: Enregistrer des trades
    print('🧪 Test 1: Enregistrement de trades...')
    
    worker_id = 'test_worker'
    trades_5m = [
        ('5m', pd.Timestamp.now() - pd.Timedelta(minutes=i*10)) for i in range(3)
    ]
    trades_1h = [
        ('1h', pd.Timestamp.now() - pd.Timedelta(hours=i)) for i in range(2)
    ]
    trades_4h = [
        ('4h', pd.Timestamp.now() - pd.Timedelta(hours=i*4)) for i in range(1)
    ]
    
    all_trades = trades_5m + trades_1h + trades_4h
    
    for tf, timestamp in all_trades:
        fm.record_trade(worker_id, tf, timestamp)
        time.sleep(0.01)  # Petite pause pour différencier les timestamps
    
    # Test 2: Validation des fréquences
    print('
📊 Test 2: Validation des fréquences...')
    validation = fm.validate_frequencies(worker_id)
    
    print(f'Score de fréquence: {validation["frequency_score"]:.3f}')
    print(f'Timeframes validés: {validation["valid_timeframes"]}/3')
    print('
Détail par timeframe:')
    for tf, status in validation['validation'].items():
        print(f'   {tf}: {status}')
    
    # Test 3: Métriques pour logging
    print('
📈 Test 3: Métriques de logging...')
    metrics = fm.get_metrics(worker_id)
    print(f'Compteurs: {metrics["counts"]}')
    print(f'Score final: {metrics["frequency_score"]:.3f}')
    
    return validation['frequency_score'] > 0

if __name__ == '__main__':
    print('🚀 Test du système de fréquence...')
    success = test_frequency_manager()
    
    if success:
        print('
✅ Système de fréquence fonctionnel!')
    else:
        print('
❌ Problème détecté dans le système de fréquence')
