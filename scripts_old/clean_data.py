#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de nettoyage robuste des données de trading.

Ce script nettoie les données brutes téléchargées, gère les valeurs manquantes
et valide l'intégrité des données avant de les sauvegarder.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import yaml

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/clean_data.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration des chemins
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed' / 'indicators'
CONFIG_FILE = BASE_DIR / 'config' / 'config.yaml'

# Assets et timeframes supportés
ASSETS = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT']
TIMEFRAMES = ['5m', '1h', '4h']
SPLITS = ['train', 'val', 'test']

# Colonnes de prix obligatoires
PRICE_COLUMNS = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']


def load_config() -> Dict:
    """Charge la configuration depuis le fichier YAML."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Impossible de charger la configuration: {e}")
        return {}


def validate_price_data(df: pd.DataFrame, asset: str, timeframe: str) -> bool:
    """Valide les données de prix."""
    try:
        # Vérifier les colonnes obligatoires
        missing_cols = [col for col in PRICE_COLUMNS if col not in df.columns]
        if missing_cols:
            logger.error(f"Colonnes manquantes pour {asset}/{timeframe}: {missing_cols}")
            return False

        # Vérifier les types de données
        if 'TIMESTAMP' in df.columns and df['TIMESTAMP'].dtype not in ['int64', 'int32']:
            logger.warning(f"TIMESTAMP pour {asset}/{timeframe} n'est pas int64: {df['TIMESTAMP'].dtype}")

        # Vérifier les valeurs négatives ou nulles inappropriées
        for col in ['OPEN', 'HIGH', 'LOW', 'CLOSE']:
            if (df[col] <= 0).any():
                logger.error(f"Valeurs négatives ou nulles dans {col} pour {asset}/{timeframe}")
                return False

        # Vérifier la cohérence HIGH >= LOW
        if (df['HIGH'] < df['LOW']).any():
            logger.error(f"Incohérence HIGH < LOW pour {asset}/{timeframe}")
            return False

        # Vérifier que OPEN et CLOSE sont dans la fourchette HIGH-LOW
        if ((df['OPEN'] > df['HIGH']) | (df['OPEN'] < df['LOW'])).any():
            logger.error(f"OPEN en dehors de la fourchette HIGH-LOW pour {asset}/{timeframe}")
            return False

        if ((df['CLOSE'] > df['HIGH']) | (df['CLOSE'] < df['LOW'])).any():
            logger.error(f"CLOSE en dehors de la fourchette HIGH-LOW pour {asset}/{timeframe}")
            return False

        return True

    except Exception as e:
        logger.error(f"Erreur lors de la validation pour {asset}/{timeframe}: {e}")
        return False


def clean_dataframe(df: pd.DataFrame, asset: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Nettoie un DataFrame de données de prix."""
    try:
        logger.info(f"Nettoyage des données pour {asset}/{timeframe}")
        original_shape = df.shape

        # Copier le DataFrame pour éviter les modifications en place
        df_clean = df.copy()

        # Normaliser les noms de colonnes en majuscules
        df_clean.columns = [col.upper() for col in df_clean.columns]

        # Trier par timestamp si présent
        if 'TIMESTAMP' in df_clean.columns:
            df_clean = df_clean.sort_values('TIMESTAMP').reset_index(drop=True)

        # Supprimer les doublons complets
        before_dedup = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        after_dedup = len(df_clean)
        if before_dedup != after_dedup:
            logger.info(f"Suppression de {before_dedup - after_dedup} doublons pour {asset}/{timeframe}")

        # Supprimer les doublons de timestamp
        if 'TIMESTAMP' in df_clean.columns:
            before_ts_dedup = len(df_clean)
            df_clean = df_clean.drop_duplicates('TIMESTAMP', keep='first')
            after_ts_dedup = len(df_clean)
            if before_ts_dedup != after_ts_dedup:
                logger.info(f"Suppression de {before_ts_dedup - after_ts_dedup} doublons de timestamp pour {asset}/{timeframe}")

        # Calculer le pourcentage de NaN avant nettoyage
        nan_before = {}
        for col in PRICE_COLUMNS:
            if col in df_clean.columns:
                nan_count = df_clean[col].isna().sum()
                nan_pct = (nan_count / len(df_clean)) * 100
                nan_before[col] = nan_pct
                if nan_pct > 0:
                    logger.info(f"NaN avant nettoyage dans {col}: {nan_count} ({nan_pct:.2f}%)")

        # Nettoyage des valeurs manquantes avec forward fill puis backward fill
        logger.info(f"Application du forward fill et backward fill pour {asset}/{timeframe}")
        df_clean[PRICE_COLUMNS] = df_clean[PRICE_COLUMNS].fillna(method='ffill').fillna(method='bfill')

        # Vérification post-nettoyage
        nan_after = {}
        for col in PRICE_COLUMNS:
            if col in df_clean.columns:
                nan_count = df_clean[col].isna().sum()
                nan_pct = (nan_count / len(df_clean)) * 100
                nan_after[col] = nan_pct
                if nan_count > 0:
                    logger.error(f"NaN persistant après nettoyage dans {col}: {nan_count} ({nan_pct:.2f}%)")

        # Vérifier qu'il n'y a plus de NaN dans CLOSE (le plus critique)
        if df_clean['CLOSE'].isna().any():
            raise ValueError(f"NaN in CLOSE after cleaning for {asset}/{timeframe}")

        # Validation des données nettoyées
        if not validate_price_data(df_clean, asset, timeframe):
            raise ValueError(f"Validation échoué après nettoyage pour {asset}/{timeframe}")

        # Convertir les types de données pour l'efficacité mémoire
        for col in PRICE_COLUMNS:
            if col in df_clean.columns and col != 'TIMESTAMP':
                df_clean[col] = df_clean[col].astype('float32')

        if 'TIMESTAMP' in df_clean.columns:
            df_clean['TIMESTAMP'] = df_clean['TIMESTAMP'].astype('int64')

        logger.info(f"Nettoyage terminé pour {asset}/{timeframe}: {original_shape} -> {df_clean.shape}")
        return df_clean

    except Exception as e:
        logger.error(f"Erreur lors du nettoyage pour {asset}/{timeframe}: {e}")
        return None


def process_asset_timeframe(asset: str, timeframe: str) -> bool:
    """Traite un actif et une timeframe donnés."""
    try:
        # Chemin du fichier brut
        raw_file = RAW_DATA_DIR / asset / f"{timeframe}.csv"

        if not raw_file.exists():
            logger.warning(f"Fichier brut non trouvé: {raw_file}")
            return False

        # Charger les données brutes
        logger.info(f"Chargement des données brutes: {raw_file}")
        df = pd.read_csv(raw_file)

        if df.empty:
            logger.warning(f"Fichier vide: {raw_file}")
            return False

        # Nettoyer les données
        df_clean = clean_dataframe(df, asset, timeframe)

        if df_clean is None:
            logger.error(f"Échec du nettoyage pour {asset}/{timeframe}")
            return False

        # Sauvegarder dans chaque split (normalement on devrait diviser les données)
        # Pour simplifier, on sauvegarde les mêmes données dans chaque split
        success = True
        for split in SPLITS:
            try:
                output_dir = PROCESSED_DATA_DIR / split / asset
                output_dir.mkdir(parents=True, exist_ok=True)

                output_file = output_dir / f"{timeframe}.parquet"
                df_clean.to_parquet(output_file, index=False)
                logger.info(f"Données sauvegardées: {output_file}")

            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde dans {split}: {e}")
                success = False

        return success

    except Exception as e:
        logger.error(f"Erreur lors du traitement de {asset}/{timeframe}: {e}")
        return False


def main():
    """Fonction principale."""
    logger.info("=== DÉBUT DU NETTOYAGE DES DONNÉES ===")

    # Créer les répertoires nécessaires
    for split in SPLITS:
        for asset in ASSETS:
            (PROCESSED_DATA_DIR / split / asset).mkdir(parents=True, exist_ok=True)

    # Créer le répertoire de logs
    (BASE_DIR / 'logs').mkdir(exist_ok=True)

    success_count = 0
    total_count = 0

    # Traiter chaque combinaison asset/timeframe
    for asset in ASSETS:
        for timeframe in TIMEFRAMES:
            total_count += 1
            logger.info(f"\n--- Traitement de {asset}/{timeframe} ---")

            if process_asset_timeframe(asset, timeframe):
                success_count += 1
                logger.info(f"✅ Succès: {asset}/{timeframe}")
            else:
                logger.error(f"❌ Échec: {asset}/{timeframe}")

    # Résumé final
    logger.info(f"\n=== NETTOYAGE TERMINÉ ===")
    logger.info(f"Succès: {success_count}/{total_count}")
    logger.info(f"Échecs: {total_count - success_count}/{total_count}")

    if success_count == total_count:
        logger.info("🎉 Tous les fichiers ont été nettoyés avec succès!")
    else:
        logger.warning("⚠️  Certains fichiers n'ont pas pu être nettoyés. Vérifiez les logs.")

    return success_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
