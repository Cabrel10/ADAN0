#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour corriger les problèmes de données dans les fichiers parquet.

Ce script identifie et corrige les valeurs NaN et infinies dans les fichiers de données.
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
        logging.FileHandler('fix_parquet.log')
    ]
)
logger = logging.getLogger(__name__)

class ParquetFixer:
    """Classe pour corriger les problèmes dans les fichiers parquet."""
    
    def __init__(self, data_dir):
        """Initialise le correcteur avec le répertoire des données."""
        self.data_dir = Path(data_dir)
        self.processed_count = 0
        self.fixed_count = 0
        self.error_count = 0
        
    def find_parquet_files(self):
        """Trouve tous les fichiers parquet dans le répertoire de données."""
        pattern = str(self.data_dir / '**' / '*.parquet')
        return glob.glob(pattern, recursive=True)
    
    def fix_file(self, filepath):
        """Corrige un fichier parquet individuel."""
        try:
            # Lire le fichier
            df = pd.read_parquet(filepath)
            original_columns = df.columns.tolist()
            original_shape = df.shape
            
            # Vérifier les colonnes numériques
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                logger.warning(f"Aucune colonne numérique trouvée dans {filepath}")
                return False
                
            # Compter les valeurs problématiques avant correction
            nan_before = df[numeric_cols].isna().sum().sum()
            inf_before = np.isinf(df[numeric_cols]).sum().sum()
            
            if nan_before == 0 and inf_before == 0:
                logger.debug(f"Aucun problème détecté dans {filepath}")
                return False
                
            logger.info(f"\nTraitement de {filepath}")
            logger.info(f"Valeurs manquantes: {nan_before}, Valeurs infinies: {inf_before}")
            
            # Appliquer les corrections par type d'indicateur
            for col in numeric_cols:
                # Pour les indicateurs normalisés (RSI, CCI, etc.)
                if any(x in col.lower() for x in ['rsi', 'cci', 'mfi', 'stoch']):
                    # Remplacer les infinis par NaN d'abord
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    # Remplir avec la valeur médiane de la colonne ou 50 pour RSI
                    fill_value = 50.0 if 'rsi' in col.lower() else df[col].median()
                    df[col] = df[col].fillna(fill_value)
                # Pour les moyennes mobiles et indicateurs de tendance
                elif any(x in col.lower() for x in ['ma', 'ema', 'sma', 'macd', 'bb_', 'atr']):
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    # Interpoler les valeurs manquantes
                    df[col] = df[col].interpolate()
                    # Remplir les valeurs manquantes restantes avec la première/last valid
                    df[col] = df[col].fillna(method='bfill').fillna(method='ffill').fillna(0)
                # Pour les prix et volumes
                elif col.lower() in ['open', 'high', 'low', 'close', 'volume']:
                    # Pour les prix, on ne peut pas interpoler n'importe comment
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    # Remplir avec la dernière valeur valide
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                    # Si toujours des NaN (au début du fichier), remplir avec 0
                    df[col] = df[col].fillna(0)
                # Pour les autres indicateurs
                else:
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    df[col] = df[col].interpolate().fillna(method='bfill').fillna(method='ffill').fillna(0)
            
            # Vérifier les valeurs après correction
            nan_after = df[numeric_cols].isna().sum().sum()
            inf_after = np.isinf(df[numeric_cols]).sum().sum()
            
            # Vérifier si des colonnes ont été perdues
            if set(original_columns) != set(df.columns):
                logger.error(f"Colonnes perdues pendant le traitement de {filepath}")
                return False
                
            # Vérifier si la forme des données a changé
            if df.shape != original_shape:
                logger.error(f"La forme des données a changé pendant le traitement de {filepath}")
                return False
                
            # Sauvegarder le fichier corrigé
            if nan_after == 0 and inf_after == 0:
                # Créer une sauvegarde
                backup_path = f"{filepath}.bak"
                if not os.path.exists(backup_path):
                    os.rename(filepath, backup_path)
                # Sauvegarder le fichier corrigé
                df.to_parquet(filepath, index=False)
                self.fixed_count += 1
                logger.info(f"Fichier corrigé avec succès. Sauvegarde: {backup_path}")
                return True
            else:
                logger.warning(f"Il reste des valeurs problématiques après correction (NaN: {nan_after}, Inf: {inf_after})")
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
            self.fix_file(filepath)
        
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
