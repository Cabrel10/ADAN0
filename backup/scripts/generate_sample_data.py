#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to generate sample data for trading.
"""

import logging
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

# Configuration du chemin du projet
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import local après configuration du chemin
from src.adan_trading_bot.data_processing.feature_engineer import FeatureEngineer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load YAML configuration and resolve path placeholders."""

    def resolve_paths(node, root_config):
        if isinstance(node, dict):
            for key, value in node.items():
                node[key] = resolve_paths(value, root_config)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                node[i] = resolve_paths(item, root_config)
        elif isinstance(node, str):
            placeholders = re.findall(r'\$\{(.*?)\}', node)
            for placeholder in placeholders:
                keys = placeholder.split('.')
                value = root_config
                try:
                    for key in keys:
                        value = value[key]
                    node = node.replace(f'${{{placeholder}}}', str(value))
                except (KeyError, TypeError):
                    logger.warning(f"Placeholder '{{{placeholder}}}' not found.")
        return node

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return resolve_paths(config, config)


def process_and_save_data(config):
    """Process raw data, add indicators, and save to Parquet."""
    raw_data_dir = Path(config['paths']['raw_data_dir'])
    processed_dir = Path(config['paths']['indicators_data_dir'])
    # Liste de tous les assets disponibles
    assets = ['ADA', 'BTC', 'ETH', 'SOL', 'XRP', 'ARB']
    logger.info(f"Traitement forcé de tous les assets: {assets}")

    # Récupération des timeframes depuis la config
    timeframes = config['feature_engineering']['timeframes']
    logger.info(f"Timeframes à traiter: {timeframes}")

    # Création du dossier de sortie s'il n'existe pas
    processed_dir.mkdir(parents=True, exist_ok=True)

    feature_engineer = FeatureEngineer(
        config, models_dir=config['paths']['models_dir']
    )

    for asset in assets:
        for timeframe in timeframes:
            logger.info(f"Processing {asset} for timeframe {timeframe}...")

            csv_file_name = f"{asset}USDT.csv"
            csv_file_path = raw_data_dir / timeframe / csv_file_name

            if not csv_file_path.is_file():
                logger.warning(f"File not found: {csv_file_path}, skipping.")
                continue

            try:
                df = pd.read_csv(csv_file_path)
                logger.info(f"CSV file loaded. Shape: {df.shape}")

                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

                # Calcul des indicateurs
                df_indic = feature_engineer.calculate_indicators_for_single_timeframe(
                    df=df.copy(),
                    timeframe=timeframe
                )
                logger.info(
                    f"Indicators calculated. "
                    f"Shape: {df_indic.shape}"
                )

                # Normalisation des noms de colonnes en majuscules
                df_indic.columns = [col.upper() for col in df_indic.columns]

                # Renommer les colonnes pour correspondre aux noms attendus
                column_mapping = {
                    # Format standard
                    'SUPERT_14_2.0': 'SUPERTREND_14_2.0',
                    'SUPERT_14_3.0': 'SUPERTREND_14_3.0',
                    'MACDH_12_26_9': 'MACD_HIST_12_26_9',
                    'PSARL_0.02_0.2': 'PSAR_0.02_0.2',  # Ligne supérieure du PSAR
                    'PSARS_0.02_0.2': 'PSAR_0.02_0.2',  # Ligne inférieure du PSAR
                    'STOCHK_14_3_3': 'STOCHk_14_3_3',   # Correction de la casse
                    'STOCHD_14_3_3': 'STOCHd_14_3_3',   # Correction de la casse
                    'ICHIMOKU_9_26_52_BASE': 'ICHIMOKU_9_26_52',  # Pour l'indicateur ICHIMOKU
                }

                # Supprimer les colonnes en double après renommage
                for old_name, new_name in column_mapping.items():
                    if (old_name in df_indic.columns and
                            new_name in df_indic.columns):
                        df_indic = df_indic.drop(columns=[old_name])
                    elif old_name in df_indic.columns:
                        df_indic = df_indic.rename(
                            columns={old_name: new_name}
                        )

                # Vérification des indicateurs requis par timeframe AVANT suppression
                required_indicators = {
                    '5m': ['STOCHk_14_3_3', 'STOCHd_14_3_3'],
                    '1h': ['ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ICS_26'],  # Composantes ICHIMOKU
                    '4h': ['ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ICS_26', 'SUPERT_14_3.0']  # Composantes ICHIMOKU + SUPERTREND
                }

                # Vérification des indicateurs manquants avant suppression
                missing_indicators = [
                    ind for ind in required_indicators.get(timeframe, [])
                    if ind not in df_indic.columns
                ]

                if missing_indicators:
                    logger.warning(
                        f"Missing indicators for {timeframe} before cleanup: {missing_indicators}"
                    )

                # Colonnes à supprimer (uniquement les colonnes vraiment inutiles)
                columns_to_drop = [
                    # Colonnes techniques intermédiaires
                    'SUPERTD_14_2.0', 'SUPERTL_14_2.0', 'SUPERTS_14_2.0',  # SUPERTREND intermédiaires
                    'SUPERTD_14_3.0', 'SUPERTL_14_3.0', 'SUPERTS_14_3.0',  # SUPERTREND intermédiaires
                    'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',  # Bollinger Bands intermédiaires
                    'MACDS_12_26_9',  # Ligne de signal MACD (on garde l'histogramme)
                    'ATRR_14',  # ATR ratio
                    'VWAP_D',  # VWAP journalier (on garde le VWAP par défaut)
                    # Ne plus supprimer les colonnes ICHIMOKU
                ]

                # Ne supprimer que les colonnes qui existent
                columns_to_drop = [col for col in columns_to_drop if col in df_indic.columns]
                if columns_to_drop:
                    logger.info(f"Suppression des colonnes intermédiaires: {columns_to_drop}")
                    df_indic = df_indic.drop(columns=columns_to_drop)

                # Vérification finale des indicateurs requis
                missing_indicators = [
                    ind for ind in required_indicators.get(timeframe, [])
                    if ind not in df_indic.columns
                ]

                # Vérification des indicateurs manquants
                missing_indicators = [
                    ind for ind in required_indicators.get(timeframe, [])
                    if ind not in df_indic.columns
                ]

                if missing_indicators:
                    logger.warning(
                        f"Missing indicators for {timeframe}: {missing_indicators}"
                    )

                # Stratégie de gestion des valeurs manquantes
                initial_rows = len(df_indic)
                initial_na = df_indic.isna().sum().sum()

                if initial_na > 0:
                    logger.info(f"Traitement de {initial_na} valeurs manquantes...")

                    # 1. Identifier les colonnes avec trop de valeurs manquantes (>50%)
                    na_ratio = df_indic.isna().mean()
                    columns_to_drop = na_ratio[na_ratio > 0.5].index.tolist()

                    if columns_to_drop:
                        logger.warning(
                            f"Suppression des colonnes avec plus de 50% de valeurs manquantes: {columns_to_drop}"
                        )
                        df_indic = df_indic.drop(columns=columns_to_drop)

                    # 2. Remplissage des valeurs manquantes par ordre de priorité
                    # 2.1 Remplissage avant/arrière pour les indicateurs techniques
                    df_indic = df_indic.ffill().bfill()

                    # 2.2 Interpolation linéaire pour les séries temporelles
                    numeric_cols = df_indic.select_dtypes(include=['float64', 'int64']).columns
                    for col in numeric_cols:
                        df_indic[col] = df_indic[col].interpolate(method='linear')

                    # 2.3 Remplissage par la moyenne pour les colonnes numériques restantes
                    for col in numeric_cols:
                        if df_indic[col].isna().any():
                            df_indic[col] = df_indic[col].fillna(df_indic[col].mean())

                # Vérification finale des valeurs manquantes
                remaining_na = df_indic.isna().sum().sum()
                if remaining_na > 0:
                    logger.warning(
                        f"Il reste {remaining_na} valeurs manquantes après nettoyage. "
                        "Ces valeurs seront remplies avec des zéros."
                    )
                    df_indic = df_indic.fillna(0)

                # Vérification de la taille finale du DataFrame
                final_rows = len(df_indic)
                if final_rows < 100:  # Seuil arbitraire pour détecter un problème
                    logger.error(
                        f"Trop peu de données après nettoyage pour {asset} {timeframe} "
                        f"({final_rows} lignes). Vérifiez les données sources."
                    )
                    # Au lieu de sauter, on garde les données disponibles avec un avertissement
                    logger.warning("Conservation des données disponibles malgré le faible nombre de lignes.")

                # Calcul du nombre de lignes supprimées (pour la rétrocompatibilité)
                removed = initial_rows - final_rows

                logger.info(
                    f"Traitement terminé. {initial_rows} lignes initiales, "
                    f"{final_rows} lignes conservées, {removed} lignes supprimées, "
                    f"{initial_na} valeurs manquantes traitées. "
                    f"Shape finale: {df_indic.shape}"
                )

                # Colonnes d'indicateurs (hors colonnes originales)
                indicator_cols = [
                    col for col in df_indic.columns
                    if col not in df.columns
                ]

                # Suppression des lignes avec valeurs manquantes
                df_indic.dropna(
                    subset=indicator_cols,
                    how='all',
                    inplace=True
                )
                final_shape = df_indic.shape
                logger.info(f"Final data shape after cleaning: {final_shape}")

                # Création du dossier de sortie
                output_dir = processed_dir / asset
                output_dir.mkdir(parents=True, exist_ok=True)

                # Sauvegarde des données
                output_path = output_dir / f"{timeframe}.parquet"
                df_indic.to_parquet(output_path)
                logger.info(
                    f"Saved data to {output_path}"
                )

            except Exception as e:
                logger.error(f"Error processing {csv_file_path}: {e}", exc_info=True)


def main():
    """Main function to run the data processing."""
    config_path = project_root / 'config' / 'config.yaml'
    if not config_path.exists():
        logger.error(f"Configuration file not found at: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    process_and_save_data(config)


if __name__ == "__main__":
    main()
