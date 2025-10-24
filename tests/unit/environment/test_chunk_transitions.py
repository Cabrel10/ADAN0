import pytest
import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path

from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

# Helper function to create dummy data
def create_dummy_chunk(size, start_price, start_date_str):
    """Creates a pandas DataFrame with dummy OHLCV data."""
    dates = pd.to_datetime(pd.date_range(start=start_date_str, periods=size, freq='5min'))
    data = {
        'open': np.linspace(start_price, start_price + (size/10), size, dtype=np.float32),
        'high': np.linspace(start_price + 0.1, start_price + (size/10) + 0.1, size, dtype=np.float32),
        'low': np.linspace(start_price - 0.1, start_price + (size/10) - 0.1, size, dtype=np.float32),
        'close': np.linspace(start_price, start_price + (size/10), size, dtype=np.float32),
        'volume': np.random.randint(100, 1000, size=size).astype(np.float32)
    }
    # Add some dummy indicator columns that might be expected by the StateBuilder
    data['rsi_14'] = np.random.uniform(30, 70, size=size).astype(np.float32)
    data['macd_12_26_9'] = np.random.uniform(-1, 1, size=size).astype(np.float32)
    
    df = pd.DataFrame(data)
    df['timestamp'] = dates
    return df.set_index('timestamp')

@pytest.fixture
def test_data_path(tmp_path):
    """Creates a temporary data directory structure for tests and returns the base path."""
    data_path = tmp_path / "data" / "processed" / "indicators" / "train" / "BTCUSDT" / "5m"
    os.makedirs(data_path)
    return str(tmp_path)

@pytest.fixture
def minimal_config_real_data(test_data_path):
    """Provides a minimal, valid config dictionary pointing to the test data path."""
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Point to the temporary directory as the base
    config['paths']['base_dir'] = test_data_path
    
    # Override with minimal settings for this test
    config['environment']['max_steps'] = 250
    config['environment']['max_chunks_per_episode'] = 3
    config['data']['timeframes'] = ['5m']
    config['data']['assets'] = ['BTCUSDT']
    config['workers']['w1']['timeframes'] = ['5m']
    config['workers']['w1']['assets'] = ['BTCUSDT']
    
    # Use a minimal feature set that matches the dummy data
    minimal_features = ['open', 'high', 'low', 'close', 'volume', 'rsi_14', 'macd_12_26_9']
    config['data']['features_config']['timeframes'] = {
        '5m': {
            'price': ['open', 'high', 'low', 'close'],
            'volume': ['volume'],
            'indicators': ['rsi_14', 'macd_12_26_9']
        }
    }
    config['environment']['observation']['window_sizes'] = {'5m': 20}
    if 'features' in config['environment']['observation']:
        config['environment']['observation']['features']['base'] = ['open', 'high', 'low', 'close', 'volume']
        config['environment']['observation']['features']['indicators']['5m'] = ['rsi_14', 'macd_12_26_9']

    return config

def test_chunk_transition_with_real_data(minimal_config_real_data, test_data_path):
    """
    Tests chunk transition using real dummy parquet files on disk.
    """
    # 1. Create and save dummy data files
    chunk_0 = create_dummy_chunk(100, 1000, '2023-01-01')
    chunk_1 = create_dummy_chunk(100, 1100, '2023-01-02')
    
    data_dir = Path(test_data_path) / "data" / "processed" / "indicators" / "train" / "BTCUSDT" / "5m"
    
    chunk_0.to_parquet(data_dir / "0.parquet")
    chunk_1.to_parquet(data_dir / "1.parquet")

    # 2. Instantiate the environment (no patch needed)
    env = MultiAssetChunkedEnv(

        timeframes=['5m'],
        window_size=20,
        features_config=minimal_config_real_data['data']['features_config']['timeframes'],
        max_steps=150, # Run for 1.5 chunks
        worker_config=minimal_config_real_data['workers']['w1'],
        config=minimal_config_real_data
    )

    # 3. Run the environment loop
    obs, info = env.reset()
    terminated = False
    truncated = False
    step_count = 0
    while not (terminated or truncated):
        action = env.action_space.sample()
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
        except Exception as e:
            pytest.fail(f"Environment crashed at step {step_count}. Error: {e}")
    
    print(f"Episode finished successfully after {step_count} steps.")
    assert step_count > 100 # Ensure we at least transitioned past the first chunk