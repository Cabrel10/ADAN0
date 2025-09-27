#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour corriger spécifiquement les colonnes problématiques dans les fichiers parquet.
"""

import os
import sys
import glob
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fix_specific_issues.log')
    ]
)
logger = logging.getLogger(__name__)

# Colonnes spécifiques à corriger
COLUMNS_TO_FIX = [
    'STOCH_K_14_3_3', 'STOCH_D_14_3_3',
    'MACD_HIST_12_26_9', 'MACD_12_26_9', 'MACD_SIGNAL_12_26_9',
    'EMA_12', 'EMA_26', 'EMA_200',
    'ADX_14'
]

class ParquetFixer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.processed_count = 0
        self.fixed_count = 0
        self.error_count = 0
        self.backup_dir = self.data_dir.parent / 'backup_parquet'
        
        # Créer le dossier de sauvegarde s'il n'existe pas
        self.backup_dir.mkdir(exist_ok=True)
    
    def find_parquet_files(self):
        """Trouve tous les fichiers parquet dans le répertoire de données."""
        pattern = str(self.data_dir / '**' / '*.parquet')
        return glob.glob(pattern, recursive=True)
    
    def backup_file(self, filepath):
        """Crée une sauvegarde du fichier s'il n'existe pas déjà."""
        rel_path = Path(filepath).relative_to(self.data_dir)
        backup_path = self.backup_dir / rel_path
        
        # Créer les sous-répertoires si nécessaire
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Faire une copie de sauvegarde si elle n'existe pas
        if not backup_path.exists():
            import shutil
            shutil.copy2(filepath, backup_path)
            logger.info(f"Sauvegarde créée: {backup_path}")
        
        return backup_path
    
    def fix_columns(self, df):
        """Corrige les colonnes problématiques."""
        fixed = False
        
        for col in COLUMNS_TO_FIX:
            if col in df.columns:
                # Vérifier s'il y a des valeurs problématiques
                has_nan = df[col].isna().any()
                has_inf = np.isinf(df[col]).any()
                
                if has_nan or has_inf:
                    # Faire une copie avant modification
                    original_values = df[col].copy()
                    
                    # Remplacer les infinis par NaN
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    
                    # Remplir les NaN avec interpolation linéaire
                    df[col] = df[col].interpolate(method='linear')
                    
                    # Remplir les valeurs manquantes restantes
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                    
                    # Vérifier si des corrections ont été apportées
                    if not df[col].equals(original_values):
                        fixed = True
                        logger.info(f"Colonne corrigée: {col} (NaN: {has_nan}, Inf: {has_inf})")
        
        return df, fixed
    
    def process_file(self, filepath):
        """Traite un fichier individuel."""
        try:
            # Lire le fichier
            df = pd.read_parquet(filepath)
            
            # Vérifier si le fichier contient des colonnes à corriger
            cols_to_fix = [col for col in COLUMNS_TO_FIX if col in df.columns]
            
            if not cols_to_fix:
                logger.debug(f"Aucune colonne à corriger dans {filepath}")
                return False
            
            logger.info(f"\nTraitement de {filepath}")
            logger.info(f"Colonnes à vérifier: {', '.join(cols_to_fix)}")
            
            # Créer une sauvegarde avant modification
            self.backup_file(filepath)
            
            # Corriger les colonnes
            df_fixed, was_fixed = self.fix_columns(df)
            
            if was_fixed:
                # Sauvegarder le fichier corrigé
                df_fixed.to_parquet(filepath, index=False)
                logger.info(f"Fichier corrigé avec succès: {filepath}")
                return True
            else:
                logger.info("Aucune correction nécessaire")
                return False
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Erreur lors du traitement de {filepath}: {str(e)}", exc_info=True)
            return False
    
    def run(self):
        """Exécute la correction sur tous les fichiers trouvés."""
        files = self.find_parquet_files()
        if not files:
            logger.error(f"Aucun fichier parquet trouvé dans {self.data_dir}")
            return
        
        logger.info(f"Traitement de {len(files)} fichiers parquet...")
        
        for filepath in tqdm(files, desc="Correction des fichiers"):
            self.processed_count += 1
            if self.process_file(filepath):
                self.fixed_count += 1
        
        # Résumé
        logger.info("\n" + "="*50)
        logger.info("RÉSUMÉ DE LA CORRECTION")
        logger.info("="*50)
        logger.info(f"Fichiers traités: {self.processed_count}")
        logger.info(f"Fichiers corrigés: {self.fixed_count}")
        logger.info(f"Erreurs: {self.error_count}")
        logger.info("="*50)


def main():
    """Fonction principale."""
    # Vérifier les arguments
    if len(sys.argv) != 2:
        print(f"Utilisation: {sys.argv[0]} <répertoire_des_données>")
        print("Exemple: python fix_specific_issues.py /chemin/vers/les/données")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    if not os.path.isdir(data_dir):
        print(f"Erreur: Le répertoire {data_dir} n'existe pas")
        sys.exit(1)
    
    # Exécuter le correcteur
    fixer = ParquetFixer(data_dir)
    fixer.run()


if __name__ == "__main__":
    main()
