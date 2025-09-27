#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script minimaliste pour compléter SEULEMENT les fichiers parquet manquants.
Écrit les fichiers complétés dans sibling folder 'completed/' (ne remplace pas l'original).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import traceback


OUT_SUBDIR = "completed"

# Colonnes indicateurs à considérer comme "manquants" (si absentes -> fichier à compléter)
REQUIRED_INDICATORS = [
    "ATR_14",
    "ATR_PCT",
    "ADX_14",
    "MACD_12_26_9",
    "MACD_SIGNAL_12_26_9",
    "MACD_HIST_12_26_9",
    "EMA_RATIO_FAST_SLOW",
    "STOCH_K_14_3_3",
    "STOCH_D_14_3_3",
    "EMA_200",
    "EMA_12",
    "EMA_26"
]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # uppercase
    df.columns = [str(c).upper() for c in df.columns]
    # harmonise variantes courantes
    col_map = {}
    for c in df.columns:
        if c.startswith("STOCHK") or c.startswith("STOCH_K"):
            col_map[c] = "STOCH_K_14_3_3"
        if c.startswith("STOCHD") or c.startswith("STOCH_D"):
            col_map[c] = "STOCH_D_14_3_3"
        if c == "STOCHK_14_3_3" or c == "STOCH_K_14_3_3":
            col_map[c] = "STOCH_K_14_3_3"
        # d'autres mappings peuvent être ajoutés si nécessaires
    if col_map:
        df = df.rename(columns=col_map)
    return df

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df['CLOSE'].shift(1)
    tr1 = df['HIGH'] - df['LOW']
    tr2 = (df['HIGH'] - prev_close).abs()
    tr3 = (df['LOW'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.index = df.index
    return tr

def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

def macd_lines(df: pd.DataFrame, fast=12, slow=26, signal=9):
    ema_fast = df['CLOSE'].ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = df['CLOSE'].ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def stoch_series(df: pd.DataFrame, k_period=14, d_period=3, smooth_k=3):
    low_min = df['LOW'].rolling(window=k_period).min()
    high_max = df['HIGH'].rolling(window=k_period).max()
    k_line = 100 * (df['CLOSE'] - low_min) / (high_max - low_min)
    k_line = k_line.rolling(window=smooth_k).mean()
    d_line = k_line.rolling(window=d_period).mean()
    return k_line, d_line

def adx_series(df: pd.DataFrame, period=14):
    up_move = df['HIGH'].diff()
    down_move = -df['LOW'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(df)
    # Smooth with Wilder (EWMA with alpha=1/period approximates Wilder smoothing)
    atr_w = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr_w)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr_w)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx_line = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return adx_line

def compute_missing_indicators(df: pd.DataFrame, missing: list):
    # Assure colonnes prix présentes
    if not set(['OPEN','HIGH','LOW','CLOSE']).issubset(set(df.columns)):
        raise ValueError("Colonnes de prix manquantes (OPEN/HIGH/LOW/CLOSE)")

    results = {}
    if "ATR_14" in missing or "ATR_PCT" in missing:
        atr14 = atr_series(df, period=14)
        results['ATR_14'] = atr14
    if "ATR_PCT" in missing:
        # ATR_PCT = ATR_14 / CLOSE
        atr14 = results.get('ATR_14', atr_series(df, period=14))
        results['ATR_PCT'] = atr14 / df['CLOSE']
    if "MACD_12_26_9" in missing or "MACD_SIGNAL_12_26_9" in missing or "MACD_HIST_12_26_9" in missing:
        macd_line, macd_signal, macd_hist = macd_lines(df, fast=12, slow=26, signal=9)
        results['MACD_12_26_9'] = macd_line
        results['MACD_SIGNAL_12_26_9'] = macd_signal
        results['MACD_HIST_12_26_9'] = macd_hist
    if "ADX_14" in missing:
        results['ADX_14'] = adx_series(df, period=14)
    if "STOCH_K_14_3_3" in missing or "STOCH_D_14_3_3" in missing:
        k, d = stoch_series(df, k_period=14, d_period=3, smooth_k=3)
        results['STOCH_K_14_3_3'] = k
        results['STOCH_D_14_3_3'] = d
    if "EMA_200" in missing:
        results['EMA_200'] = df['CLOSE'].ewm(span=200, adjust=False, min_periods=200).mean()
    if "EMA_12" in missing:
        results['EMA_12'] = df['CLOSE'].ewm(span=12, adjust=False, min_periods=12).mean()
    if "EMA_26" in missing:
        results['EMA_26'] = df['CLOSE'].ewm(span=26, adjust=False, min_periods=26).mean()
    if "EMA_RATIO_FAST_SLOW" in missing:
        # prefer existing EMA_5/EMA_20 if present; sinon use 12/26
        if 'EMA_5' in df.columns and 'EMA_20' in df.columns:
            results['EMA_RATIO_FAST_SLOW'] = df['EMA_5'] / df['EMA_20']
        else:
            ema_fast = df['CLOSE'].ewm(span=12, adjust=False, min_periods=12).mean()
            ema_slow = df['CLOSE'].ewm(span=26, adjust=False, min_periods=26).mean()
            results['EMA_RATIO_FAST_SLOW'] = ema_fast / ema_slow
    return results

def process_file(path: Path):
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return {'file': str(path), 'status': 'read_error', 'error': str(e)}

    df = normalize_columns(df)
    present = set(df.columns)
    missing = [c for c in REQUIRED_INDICATORS if c not in present]
    if not missing:
        return {'file': str(path), 'status': 'already_complete'}

    # compute only missing indicators
    try:
        computed = compute_missing_indicators(df, missing)
    except Exception as e:
        tb = traceback.format_exc()
        return {'file': str(path), 'status': 'compute_error', 'error': str(e), 'trace': tb}

    # attach computed series (do not overwrite existing)
    for name, series in computed.items():
        if name not in df.columns:
            df[name] = series

    # ensure output dir
    dest_dir = path.parent / OUT_SUBDIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / path.name

    try:
        df.to_parquet(out_path, index=False)
    except Exception as e:
        return {'file': str(path), 'status': 'write_error', 'error': str(e)}

    nan_counts = {k: int(df[k].isna().sum()) for k in computed.keys()}
    return {'file': str(path), 'status': 'completed', 'added': list(computed.keys()), 'nan_counts': nan_counts, 'out': str(out_path)}

def main():
    for split in ['train', 'val', 'test']:
        BASE_DIR = Path(f"bot/data/processed/indicators/{split}")
        results = []
        if not BASE_DIR.exists():
            print(f"BASE_DIR introuvable: {BASE_DIR}")
            continue
        
        files = list(BASE_DIR.rglob("*.parquet"))
        if not files:
            print(f"Aucun fichier .parquet trouvé sous {BASE_DIR}")
            continue
            
        print(f"Traitement de {len(files)} fichiers dans {BASE_DIR}...")
        for p in files:
            # Ignorer le sous-dossier 'completed'
            if 'completed' in p.parts:
                continue
            res = process_file(p)
            results.append(res)
            # log minimal
            if res.get('status') == 'completed':
                print(f"[OK] complété: {p} -> {res.get('out')}, ajoutés: {res.get('added')}")
            elif res.get('status') == 'already_complete':
                print(f"[SKIP] complet: {p}")
            else:
                print(f"[ERR] {p}, {res.get('status')}, {res.get('error', '')}")

        # write report
        report_path = Path(f"parquet_completion_report_{split}.json")
        report_path.write_text(json.dumps(results, indent=2))
        print(f"Rapport -> {report_path}")

if __name__ == "__main__":
    main()
