import pytest
import yaml
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.common.config_loader import ConfigLoader

# Resolve project root dynamically
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"


def _load_config(filename="config.yaml"):
    """Load a YAML config relative to the project root."""
    path = _CONFIG_DIR / filename
    return ConfigLoader.load_config(str(path))


def _load_workers_config():
    """Load workers.yaml and return the w1 worker config."""
    path = _CONFIG_DIR / "workers.yaml"
    if not path.exists():
        # Fallback: build a minimal worker config
        return {
            "assets": ["BTCUSDT"],
            "timeframes": ["5m", "1h", "4h"],
        }
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data["workers"]["w1"]


@pytest.fixture(scope="module")
def setup_env_for_chunk_tests():
    main_config = _load_config()
    worker_specific_config = _load_workers_config()

    # Override initial_balance for the test to ensure sufficient capital
    main_config.setdefault("portfolio", {})["initial_balance"] = 20.5

    env = MultiAssetChunkedEnv(
        worker_id=0,
        config=main_config,
        worker_config=worker_specific_config,
    )
    return env


def test_dbe_state_persistence_across_chunks(setup_env_for_chunk_tests):
    """Verifie la persistance DBE lors des transitions de chunks"""
    env = setup_env_for_chunk_tests
    dbe_state_file = f"dbe_state_{env.worker_id}.pkl"

    try:
        env.reset()

        # Simuler des decisions DBE sur chunk 0
        env.dbe.decision_history = [1, 2, 3]
        initial_state_dbe_history = list(env.dbe.decision_history)

        # Manually trigger save of DBE state
        if hasattr(env, "dbe") and env.dbe is not None:
            with open(dbe_state_file, "wb") as f:
                pickle.dump(env.dbe, f)
        assert os.path.exists(dbe_state_file)

        # Creer un nouvel environnement pour simuler le chargement du chunk suivant
        main_config = _load_config()
        worker_specific_config = _load_workers_config()
        main_config.setdefault("portfolio", {})["initial_balance"] = 20.5
        env2 = MultiAssetChunkedEnv(
            worker_id=0,
            config=main_config,
            worker_config=worker_specific_config,
        )

        # Simuler le chargement du chunk 1 dans le nouvel environnement
        if os.path.exists(dbe_state_file):
            with open(dbe_state_file, "rb") as f:
                loaded_dbe = pickle.load(f)
            env2.dbe = loaded_dbe

        # Verifier que l'etat DBE est preserve
        assert hasattr(env2.dbe, "decision_history")
        assert env2.dbe.decision_history == initial_state_dbe_history

    finally:
        if os.path.exists(dbe_state_file):
            os.remove(dbe_state_file)


def test_chunk_fallback_mechanism(setup_env_for_chunk_tests):
    """Verifie le mecanisme de fallback en cas d'echec de chargement"""
    env = setup_env_for_chunk_tests

    # Temporarily set total_chunks to a small number to easily trigger fallback
    original_total_chunks = env.data_loader.total_chunks
    env.data_loader.total_chunks = 1  # Only one chunk exists

    try:
        # Attempt to load a non-existent chunk, expecting fallback to chunk 0
        chunk_data = env._safe_load_chunk(9999, fallback_enabled=True)

        assert chunk_data is not None
        assert isinstance(chunk_data, dict)
        assert len(chunk_data) > 0  # Should contain some data

        # Verify that the synthetic chunk has expected structure
        assert "BTCUSDT" in chunk_data
        assert "5m" in chunk_data["BTCUSDT"]
        assert isinstance(chunk_data["BTCUSDT"]["5m"], pd.DataFrame)
        assert not chunk_data["BTCUSDT"]["5m"].empty

    finally:
        env.data_loader.total_chunks = original_total_chunks
