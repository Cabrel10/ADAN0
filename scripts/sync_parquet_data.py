#!/usr/bin/env python3
"""
Script pour synchroniser les données Parquet entre les différents actifs.
"""
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Set, List
from collections import defaultdict

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('parquet_sync.log')
    ]
)
logger = logging.getLogger(__name__)

class ParquetSynchronizer:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.indicators_dir = self.base_dir / 'data/processed/indicators'
        self.synced_dir = self.base_dir / 'data/processed/indicators_synced'
        self.assets = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT']
        self.timeframes = ['5m', '1h', '4h']
        self.splits = ['train', 'val', 'test']
        
        # Créer le répertoire de sortie
        self.synced_dir.mkdir(exist_ok=True)
        for split in self.splits:
            (self.synced_dir / split).mkdir(exist_ok=True)
            for asset in self.assets:
                (self.synced_dir / split / asset).mkdir(parents=True, exist_ok=True)
    
    def sync_all(self):
        """Synchronise toutes les données."""
        logger.info("Début de la synchronisation des données Parquet")
        
        for split in self.splits:
            for tf in self.timeframes:
                self.sync_timeframe(split, tf)
        
        logger.info("Synchronisation terminée")
    
    def sync_timeframe(self, split: str, tf: str):
        """Synchronise les données pour une timeframe donnée."""
        logger.info(f"\nSynchronisation de {split}/{tf}:")
        
        # Charger les données pour chaque actif
        data = {}
        for asset in self.assets:
            file_path = self.indicators_dir / split / asset / f"{tf}.parquet"
            try:
                df = pd.read_parquet(file_path)
                data[asset] = df
                logger.info(f"  - {asset}: {len(df)} timestamps")
            except Exception as e:
                logger.error(f"Erreur lors du chargement de {file_path}: {e}")
                return
        
        # Trouver les timestamps communs
        common_timestamps = set.intersection(*[set(df.index) for df in data.values()])
        logger.info(f"  Timestamps communs: {len(common_timestamps)}")
        
        if not common_timestamps:
            logger.warning(f"Aucun timestamp commun trouvé pour {split}/{tf}")
            return
        
        # Trier les timestamps
        common_timestamps = sorted(common_timestamps)
        
        # Sauvegarder les données synchronisées
        for asset, df in data.items():
            # Filtrer pour ne garder que les timestamps communs
            synced_df = df.loc[common_timestamps].copy()
            
            # Vérifier les valeurs manquantes
            if synced_df.isnull().any().any():
                logger.warning(f"  {asset}: {synced_df.isnull().sum().sum()} valeurs manquantes après synchronisation")
            
            # Sauvegarder
            output_path = self.synced_dir / split / asset / f"{tf}.parquet"
            synced_df.to_parquet(output_path)
            logger.info(f"  {asset}: {len(synced_df)} timestamps sauvegardés")
        
        # Sauvegarder les timestamps communs
        timestamps_df = pd.DataFrame(index=common_timestamps)
        timestamps_path = self.synced_dir / f"common_timestamps_{split}_{tf}.csv"
        timestamps_df.to_csv(timestamps_path)
        logger.info(f"  Timestamps communs sauvegardés dans {timestamps_path}")

if __name__ == "__main__":
    synchronizer = ParquetSynchronizer('/home/morningstar/Documents/trading')
    synchronizer.sync_all()
