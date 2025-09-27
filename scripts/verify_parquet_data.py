#!/usr/bin/env python3
"""
Script pour vérifier la cohérence des données Parquet générées.
- Vérifie les valeurs nulles ou zéro dans les indicateurs
- Vérifie la synchronisation des dates entre les actifs pour chaque timeframe
- Vérifie l'emplacement des fichiers
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('parquet_verification.log')
    ]
)
logger = logging.getLogger(__name__)

class ParquetVerifier:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.indicators_dir = self.base_dir / 'data/processed/indicators'
        self.assets = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT']
        self.timeframes = ['5m', '1h', '4h']
        self.splits = ['train', 'val', 'test']
        
    def verify_all(self):
        """Exécute toutes les vérifications."""
        logger.info("Début de la vérification des données Parquet")
        
        # Vérification des fichiers manquants
        self.check_missing_files()
        
        # Vérification des valeurs nulles et zéros
        self.check_null_and_zero_values()
        
        # Vérification de la synchronisation des dates
        self.check_date_synchronization()
        
        logger.info("Vérification terminée")
    
    def check_missing_files(self):
        """Vérifie les fichiers manquants."""
        logger.info("Vérification des fichiers manquants...")
        missing_files = []
        
        for split in self.splits:
            for asset in self.assets:
                for tf in self.timeframes:
                    file_path = self.indicators_dir / split / asset / f"{tf}.parquet"
                    if not file_path.exists():
                        missing_files.append(str(file_path))
        
        if missing_files:
            logger.warning(f"{len(missing_files)} fichiers manquants détectés:")
            for f in missing_files[:10]:  # Afficher seulement les 10 premiers pour éviter la surcharge
                logger.warning(f"  - {f}")
            if len(missing_files) > 10:
                logger.warning(f"  ... et {len(missing_files) - 10} autres fichiers manquants")
        else:
            logger.info("Aucun fichier manquant détecté.")
    
    def check_null_and_zero_values(self):
        """Vérifie les valeurs nulles et zéros dans les indicateurs."""
        logger.info("Vérification des valeurs nulles et zéros...")
        
        for split in self.splits:
            for asset in self.assets:
                for tf in self.timeframes:
                    file_path = self.indicators_dir / split / asset / f"{tf}.parquet"
                    if not file_path.exists():
                        continue
                    
                    try:
                        # Charger les données
                        df = pd.read_parquet(file_path)
                        
                        # Vérifier les valeurs manquantes
                        null_cols = df.isnull().sum()
                        null_cols = null_cols[null_cols > 0]
                        
                        if not null_cols.empty:
                            logger.warning(f"Valeurs nulles dans {split}/{asset}/{tf}.parquet:")
                            for col, count in null_cols.items():
                                logger.warning(f"  - {col}: {count} valeurs nulles")
                        
                        # Vérifier les zéros dans les indicateurs (sauf pour les indicateurs qui peuvent être nuls)
                        zero_cols = {}
                        for col in df.columns:
                            if col in ['open', 'high', 'low', 'close', 'volume']:
                                continue
                            zero_count = (df[col] == 0).sum()
                            if zero_count > 0:
                                zero_cols[col] = zero_count
                        
                        if zero_cols:
                            logger.warning(f"Zéros dans les indicateurs de {split}/{asset}/{tf}.parquet:")
                            for col, count in zero_cols.items():
                                logger.warning(f"  - {col}: {count} zéros")
                        
                    except Exception as e:
                        logger.error(f"Erreur lors de la vérification de {file_path}: {e}")
    
    def check_date_synchronization(self):
        """Vérifie la synchronisation des dates entre les actifs pour chaque timeframe."""
        logger.info("Vérification de la synchronisation des dates...")
        
        for split in self.splits:
            for tf in self.timeframes:
                logger.info(f"\nVérification de la synchronisation pour {split}/{tf}:")
                
                # Récupérer les dates pour chaque actif
                dates = {}
                for asset in self.assets:
                    file_path = self.indicators_dir / split / asset / f"{tf}.parquet"
                    if not file_path.exists():
                        logger.warning(f"Fichier manquant: {file_path}")
                        continue
                    
                    try:
                        df = pd.read_parquet(file_path)
                        dates[asset] = set(df.index)
                        logger.info(f"  - {asset}: {len(dates[asset])} timestamps")
                    except Exception as e:
                        logger.error(f"Erreur lors du chargement de {file_path}: {e}")
                
                if len(dates) < 2:
                    logger.warning("Pas assez d'actifs pour vérifier la synchronisation")
                    continue
                
                # Vérifier l'intersection des dates
                common_dates = set.intersection(*dates.values())
                logger.info(f"  Dates communes: {len(common_dates)}")
                
                # Vérifier les dates manquantes pour chaque actif
                for asset, asset_dates in dates.items():
                    missing = common_dates - asset_dates
                    if missing:
                        logger.warning(f"  {asset} manque {len(missing)} dates présentes chez d'autres actifs")
                        # Afficher les 5 premières dates manquantes
                        for date in sorted(missing)[:5]:
                            logger.warning(f"    - {date}")
                        if len(missing) > 5:
                            logger.warning(f"    ... et {len(missing) - 5} dates supplémentaires")
                
                # Vérifier la plage de dates
                if common_dates:
                    min_date = min(common_dates)
                    max_date = max(common_dates)
                    logger.info(f"  Plage de dates communes: {min_date} à {max_date}")

if __name__ == "__main__":
    verifier = ParquetVerifier('/home/morningstar/Documents/trading')
    verifier.verify_all()
