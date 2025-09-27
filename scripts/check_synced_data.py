#!/usr/bin/env python3
"""
Script pour vérifier la cohérence des données synchronisées.
"""
import pandas as pd
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_synced_data(base_dir: str):
    """Vérifie la cohérence des données synchronisées."""
    base_path = Path(base_dir)
    synced_dir = base_path / 'data/processed/indicators_synced'
    
    if not synced_dir.exists():
        logger.error("Le répertoire des données synchronisées n'existe pas.")
        return
    
    # Vérifier la structure des dossiers
    splits = ['train', 'val', 'test']
    timeframes = ['5m', '1h', '4h']
    assets = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT']
    
    # Vérifier les fichiers de timestamps communs
    for split in splits:
        for tf in timeframes:
            ts_file = synced_dir / f"common_timestamps_{split}_{tf}.csv"
            if not ts_file.exists():
                logger.warning(f"Fichier de timestamps manquant: {ts_file}")
                continue
            
            # Charger les timestamps communs
            try:
                common_ts = pd.read_csv(ts_file, index_col=0, parse_dates=True).index
                logger.info(f"\nVérification de {split}/{tf}:")
                logger.info(f"  - {len(common_ts)} timestamps communs")
                logger.info(f"  - Plage: {common_ts.min()} à {common_ts.max()}")
                
                # Vérifier que tous les actifs ont les mêmes timestamps
                for asset in assets:
                    file_path = synced_dir / split / asset / f"{tf}.parquet"
                    if not file_path.exists():
                        logger.warning(f"  Fichier manquant: {file_path}")
                        continue
                    
                    df = pd.read_parquet(file_path)
                    if not df.index.equals(common_ts):
                        logger.error(f"  Les timestamps de {asset} ne correspondent pas aux timestamps communs!")
                    else:
                        logger.info(f"  ✓ {asset}: {len(df)} timestamps (OK)")
                        
                    # Vérifier les valeurs manquantes
                    null_cols = df.isnull().sum()
                    null_cols = null_cols[null_cols > 0]
                    if not null_cols.empty:
                        logger.warning(f"  {asset} a des valeurs manquantes: {dict(null_cols)}")
                    
                    # Vérifier les zéros dans les indicateurs
                    for col in df.columns:
                        if col in ['open', 'high', 'low', 'close', 'volume']:
                            continue
                        zero_count = (df[col] == 0).sum()
                        if zero_count > 0:
                            logger.warning(f"  {asset}.{col}: {zero_count} zéros détectés")
            
            except Exception as e:
                logger.error(f"Erreur lors de la vérification de {ts_file}: {e}")

if __name__ == "__main__":
    check_synced_data('/home/morningstar/Documents/trading')
