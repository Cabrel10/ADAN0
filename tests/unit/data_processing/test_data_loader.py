"""Tests unitaires pour le DataLoader."""

import pytest
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

# Simuler l'import du DataLoader réel
with patch.dict('sys.modules', {'src.adan_trading_bot.data_processing.data_loader': MagicMock()}):
    from src.adan_trading_bot.data_processing.data_loader import ChunkedDataLoader

def test_chunked_data_loader_initialization():
    """Test l'initialisation du ChunkedDataLoader."""
    # Configuration de test
    config = {
        "data": {
            "processed_data": {
                "train": "data/processed/train",
                "validation": "data/processed/validation",
                "test": "data/processed/test"
            },
            "timeframes": ["5m", "1h"],
            "features_config": {
                "5m": {"features": ["open", "close", "volume"]},
                "1h": {"features": ["open", "high", "low", "close"]}
            }
        },
        "training": {
            "batch_size": 64,
            "n_steps": 1000
        }
    }
    
    worker_config = {
        "assets": ["BTCUSDT", "ETHUSDT"],
        "timeframes": ["5m", "1h"]
    }
    
    # Initialiser le loader
    with patch('pandas.read_parquet') as mock_read_parquet:
        # Configurer le mock pour retourner un DataFrame de test
        mock_read_parquet.return_value = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [2, 3, 4],
            'low': [0.5, 1.5, 2.5],
            'close': [1.5, 2.5, 3.5],
            'volume': [100, 200, 300]
        })
        
        loader = ChunkedDataLoader(
            config=config,
            worker_config=worker_config,
            mode="train",
            chunk_size=1000
        )
    
    # Vérifications
    assert loader.mode == "train"
    assert loader.chunk_size == 1000
    assert "BTCUSDT" in loader.assets_list
    assert "5m" in loader.timeframes
    assert "1h" in loader.timeframes

def test_load_chunk():
    """Test le chargement d'un chunk de données."""
    # Configuration de test
    config = {
        "data": {
            "processed_data": {"train": "data/processed/train"},
            "timeframes": ["5m"],
            "features_config": {"5m": {"features": ["open", "close"]}}
        }
    }
    
    worker_config = {
        "assets": ["BTCUSDT"],
        "timeframes": ["5m"]
    }
    
    # Données de test
    test_data = {
        'open': [1, 2, 3],
        'high': [2, 3, 4],
        'low': [0.5, 1.5, 2.5],
        'close': [1.5, 2.5, 3.5],
        'volume': [100, 200, 300]
    }
    
    # Initialiser le loader avec des mocks
    with patch('pandas.read_parquet') as mock_read_parquet, \
         patch('os.path.exists', return_value=True):
        
        mock_read_parquet.return_value = pd.DataFrame(test_data)
        
        loader = ChunkedDataLoader(
            config=config,
            worker_config=worker_config,
            mode="train",
            chunk_size=3
        )
        
        # Charger un chunk
        chunk = loader.load_chunk(0)
    
    # Vérifications
    assert "BTCUSDT" in chunk
    assert "5m" in chunk["BTCUSDT"]
    assert len(chunk["BTCUSDT"]["5m"]) == 3
    assert "open" in chunk["BTCUSDT"]["5m"]
    assert "close" in chunk["BTCUSDT"]["5m"]

def test_get_chunk_file_path():
    """Test la construction du chemin du fichier de chunk."""
    # Configuration de test
    config = {
        "data": {
            "processed_data": {"train": "/chemin/vers/train"},
            "timeframes": ["5m"]
        }
    }
    
    worker_config = {
        "assets": ["BTCUSDT"],
        "timeframes": ["5m"]
    }
    
    # Initialiser le loader
    loader = ChunkedDataLoader(
        config=config,
        worker_config=worker_config,
        mode="train",
        chunk_size=1000
    )
    
    # Tester la construction du chemin
    path = loader._get_chunk_file_path("BTCUSDT", "5m", 0)
    expected_path = Path("/chemin/vers/train/BTCUSDT/5m.parquet")
    assert str(path) == str(expected_path)
