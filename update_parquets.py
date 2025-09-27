#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour mettre à jour les fichiers Parquet avec la structure d'indicateurs standardisée.
Crée des sauvegardes avant modification et génère un rapport des changements.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import shutil
from typing import Dict, List, Optional
import traceback

# Configuration
BASE_DIR = Path("bot/data/processed/indicators")
BACKUP_DIR = Path("bot/data/processed/backups")
REPORT_FILE = "parquet_update_report.json"

# Définition des colonnes par timeframe
COLUMNS_5M = [
    "timestamp", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME",
    "RSI_14", "ATR_14", "ADX_14", "EMA_5", "EMA_20", "EMA_50",
    "MACD_HIST", "SUPERTREND_14_2.0"
]

COLUMNS_1H = [
    "timestamp", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME",
    "RSI_14", "ATR_14", "ADX_14", "EMA_12", "EMA_50", "EMA_200",
    "MACD_HIST", "SUPERTREND_14_2.0"
]

COLUMNS_4H = COLUMNS_1H  # Même structure que 1H

def get_required_columns(timeframe: str) -> List[str]:
    """Retourne les colonnes requises pour un timeframe donné."""
    if timeframe == "5m":
        return COLUMNS_5M
    elif timeframe == "1h":
        return COLUMNS_1H
    elif timeframe == "4h":
        return COLUMNS_4H
    else:
        raise ValueError(f"Timeframe non supporté: {timeframe}")

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calcule l'Average True Range (ATR)."""
    high = df['HIGH']
    low = df['LOW']
    close = df['CLOSE']
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    return atr

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calcule l'Average Directional Index (ADX)."""
    high = df['HIGH']
    low = df['LOW']
    close = df['CLOSE']
    
    # Calcul des mouvements directionnels
    up = high.diff()
    down = low.diff() * -1
    
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    # Lissage
    alpha = 1 / period
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean() / 
                     (df['HIGH'] - df['LOW']).ewm(alpha=alpha, adjust=False).mean())
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean() / 
                      (df['HIGH'] - df['LOW']).ewm(alpha=alpha, adjust=False).mean())
    
    # Calcul de l'ADX
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    
    return adx

def calculate_macd_hist(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Calcule l'histogramme MACD."""
    close = df['CLOSE']
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line

def calculate_supertrend(df: pd.DataFrame, period: int = 14, multiplier: float = 2.0) -> pd.Series:
    """Calcule le SuperTrend."""
    high = df['HIGH']
    low = df['LOW']
    close = df['CLOSE']
    
    # Calcul de l'ATR
    atr = calculate_atr(df, period)
    
    # Bandes supérieure et inférieure
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialisation
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(1, index=df.index)
    
    # Calcul du SuperTrend
    for i in range(1, len(df)):
        if close.iloc[i] > upper_band.iloc[i-1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
            
        if direction.iloc[i] == 1:
            supertrend.iloc[i] = lower_band.iloc[i]
        else:
            supertrend.iloc[i] = upper_band.iloc[i]
    
    return supertrend

def update_parquet_file(file_path: Path, report: Dict) -> bool:
    """Met à jour un fichier Parquet avec la structure d'indicateurs standardisée."""
    try:
        # Vérifier si le fichier existe
        if not file_path.exists():
            report["status"] = "error"
            report["message"] = "Fichier source introuvable"
            return False
        
        # Créer un dossier de sauvegarde s'il n'existe pas
        backup_path = BACKUP_DIR / file_path.relative_to(BASE_DIR)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le fichier original
        shutil.copy2(file_path, backup_path)
        report["backup"] = str(backup_path)
        
        # Lire le fichier Parquet
        df = pd.read_parquet(file_path)
        original_columns = set(df.columns)
        report["original_columns"] = list(original_columns)
        
        # Déterminer le timeframe à partir du nom du fichier
        timeframe = file_path.name.split('.')[0]  # Extrait '5m' de '5m.parquet'
        required_columns = get_required_columns(timeframe)
        report["required_columns"] = required_columns
        
        # Vérifier les colonnes manquantes
        missing_columns = set(required_columns) - set(original_columns)
        report["missing_columns"] = list(missing_columns)
        
        # Ajouter les indicateurs manquants
        if "ATR_14" in missing_columns:
            df["ATR_14"] = calculate_atr(df)
        
        if "ADX_14" in missing_columns:
            df["ADX_14"] = calculate_adx(df)
        
        if "MACD_HIST" in missing_columns:
            df["MACD_HIST"] = calculate_macd_hist(df)
        
        if "SUPERTREND_14_2.0" in missing_columns:
            df["SUPERTREND_14_2.0"] = calculate_supertrend(df)
            
        # Calculer les EMA manquants
        close = df['CLOSE']
        
        if "EMA_5" in missing_columns and "5m" in str(file_path):
            df["EMA_5"] = close.ewm(span=5, adjust=False).mean()
            
        if "EMA_12" in missing_columns and ("1h" in str(file_path) or "4h" in str(file_path)):
            df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
            
        if "EMA_20" in missing_columns and "5m" in str(file_path):
            df["EMA_20"] = close.ewm(span=20, adjust=False).mean()
            
        if "EMA_50" in missing_columns:
            df["EMA_50"] = close.ewm(span=50, adjust=False).mean()
            
        if "EMA_200" in missing_columns:
            df["EMA_200"] = close.ewm(span=200, adjust=False).mean()
        
        # Renommer les colonnes si nécessaire
        if "SMA_200" in df.columns and "EMA_200" not in df.columns:
            df["EMA_200"] = df["SMA_200"]
            if "SMA_200" not in required_columns:
                df = df.drop(columns=["SMA_200"])
        
        # Vérifier si toutes les colonnes requises sont présentes
        current_columns = set(df.columns)
        still_missing = set(required_columns) - current_columns
        if still_missing:
            report["status"] = "error"
            report["message"] = f"Impossible d'ajouter toutes les colonnes manquantes: {still_missing}"
            return False
        
        # Réorganiser les colonnes selon l'ordre requis
        df = df[required_columns]
        
        # Sauvegarder le fichier mis à jour
        df.to_parquet(file_path)
        
        report["status"] = "success"
        report["updated_columns"] = list(df.columns)
        return True
        
    except Exception as e:
        report["status"] = "error"
        report["message"] = str(e)
        report["traceback"] = traceback.format_exc()
        return False

def main():
    # Créer le dossier de sauvegardes
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialiser le rapport
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "processed_files": [],
        "success_count": 0,
        "error_count": 0
    }
    
    # Parcourir tous les fichiers Parquet
    for parquet_file in BASE_DIR.rglob("*.parquet"):
        # Ignorer les fichiers dans les dossiers de sauvegarde
        if "backup" in str(parquet_file) or "completed" in str(parquet_file):
            continue
            
        file_report = {
            "file": str(parquet_file),
            "status": "pending"
        }
        
        print(f"Traitement de {parquet_file}...")
        
        if update_parquet_file(parquet_file, file_report):
            report["success_count"] += 1
        else:
            report["error_count"] += 1
        
        report["processed_files"].append(file_report)
        print(f"  Statut: {file_report['status']}")
    
    # Sauvegarder le rapport
    with open(REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nTraitement terminé. Rapport sauvegardé dans {REPORT_FILE}")
    print(f"Résumé: {report['success_count']} succès, {report['error_count']} échecs")

if __name__ == "__main__":
    main()
