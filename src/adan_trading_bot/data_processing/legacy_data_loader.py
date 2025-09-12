#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data loading and normalization utilities for the ADAN trading bot."""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from ..common.utils import get_logger

logger = get_logger()

def load_data(split: str, merged_data_dir: str) -> pd.DataFrame:
    """Loads merged data for a specific split."""
    file_path = Path(merged_data_dir) / f"{split}_merged.parquet"
    logger.info(f"Loading data from {file_path}")
    if not file_path.exists():
        logger.error(f"Data file not found: {file_path}")
        return pd.DataFrame()
    return pd.read_parquet(file_path)

def normalize_data(data: pd.DataFrame, scaler_path: str) -> tuple[pd.DataFrame, StandardScaler]:
    """Normalizes the data and saves the scaler."""
    scaler = StandardScaler()
    data_normalized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    
    # Save scaler
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
    return data_normalized, scaler