#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chargeur de données pour le projet ADAN.
Charge les données de trading à partir de fichiers parquet organisés par actif et timeframe.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..common.config_loader import ConfigLoader
from .data_validator import DataValidator, DataQualityMonitor
from ..utils.smart_logger import create_smart_logger

# Configuration du logger
logger = logging.getLogger(__name__)


class ChunkedDataLoader:
    """
    Chargeur de données pour le projet ADAN.
    Charge les données de trading à partir de fichiers parquet organisés par actif et timeframe.
    """

    def __init__(self, config, worker_config, worker_id=0):
        """
        Initialise le chargeur de données.

        Args:
            config: Configuration principale de l'application
            worker_config: Configuration spécifique au worker contenant:
                - timeframes: liste des timeframes à charger (ex: ["5m", "1h"])
                - data_split: jeu de données à charger (ex: 'train', 'validation', 'test')
                - assets: liste des actifs à charger (optionnel, utilise la config par défaut si non spécifié)
            worker_id: ID du worker pour les logs
        """
        self.config = ConfigLoader.resolve_env_vars(config)
        self.worker_config = ConfigLoader.resolve_env_vars(
            worker_config, root_config=self.config
        )

        # Worker ID pour éviter la duplication de logs
        self.worker_id = worker_id

        # Initialiser le SmartLogger pour ce worker
        self.smart_logger = create_smart_logger(
            worker_id, total_workers=4, logger_name="data_loader"
        )

        # Cache pour éviter les doublons de logs
        self._last_logs = {}
        import time

        self._time_module = time

        # 1) Timeframes à charger
        self.timeframes = self.worker_config.get(
            "timeframes", self.config["data"].get("timeframes", [])
        )

        # 2) Split (train / test / train_stress_test…)
        self.data_split = self.worker_config.get(
            "data_split_override", self.config["data"].get("data_split", "train")
        )

        # 3) Actifs (permet override si besoin)
        self.assets_list = [
            asset.upper()
            for asset in self.worker_config.get(
                "assets",  # Utilise directement 'assets' du worker_config
                self.config.get("environment", {}).get(
                    "assets", []
                ),  # Fallback sur la config environment
            )
        ]

        # Initialise le dictionnaire des features par timeframe
        self.features_by_timeframe = self._init_features_by_timeframe()

        # Vérifie que tous les timeframes demandés sont pris en charge
        self._validate_timeframes()

        # Initialisation du validateur de données
        self.data_validator = DataValidator(config, worker_id)
        self.quality_monitor = DataQualityMonitor(config)

        # Configuration de la taille des chunks par timeframe
        # Priorité: worker_config.chunk_sizes > config.data.chunk_sizes > défauts optimisés
        default_chunk_sizes = {"5m": 5328, "1h": 242, "4h": 111}
        cfg_chunk_sizes = (
            self.worker_config.get("chunk_sizes") or {}
        ) or self.config.get("data", {}).get("chunk_sizes", {})
        # Conserver uniquement les timeframes demandés, sinon fallback sur défaut
        self.chunk_sizes = {}
        for tf in self.timeframes:
            if isinstance(cfg_chunk_sizes, dict) and tf in cfg_chunk_sizes:
                self.chunk_sizes[tf] = int(cfg_chunk_sizes[tf])
            else:
                # défaut si dispo, sinon utiliser longueur complète (sera bornée plus tard)
                self.chunk_sizes[tf] = int(default_chunk_sizes.get(tf, 10_000_000))

        # Initialise le nombre total de chunks en fonction des données disponibles
        self.total_chunks = self._calculate_total_chunks()
        logger.info(f"Total chunks disponibles: {self.total_chunks}")

        logger.info(
            f"Initialisation du ChunkedDataLoader avec {len(self.assets_list)} actifs et {len(self.timeframes)} timeframes"
        )

        logger.debug(f"Actifs: {self.assets_list}")
        logger.debug(f"Timeframes: {self.timeframes}")
        logger.debug(f"Jeu de données: {self.data_split}")

        # Vérifie que nous avons des actifs et des timeframes
        if not self.assets_list:
            raise ValueError(
                "Aucun actif défini dans la configuration du worker ou la configuration principale."
            )
        if not self.timeframes:
            raise ValueError("Aucun timeframe défini dans la configuration du worker.")

        # Configuration du parallélisme
        self.max_workers = min(8, (os.cpu_count() or 4) * 2)
        logger.info(
            f"Chargement des données pour {len(self.assets_list)} actifs et "
            f"{len(self.timeframes)} timeframes en parallèle (max {self.max_workers} workers)"
        )

    def log_info(self, message, step=None):
        """Log un message avec le système intelligent SmartLogger."""
        self.smart_logger.smart_info(logger, message, step)

    def _validate_timeframes(self):
        """
        Vérifie que tous les timeframes demandés sont pris en charge par la configuration.

        Raises:
            ValueError: Si un timeframe demandé n'est pas pris en charge
        """
        supported_timeframes = (
            self.config.get("data", {})
            .get("features_config", {})
            .get("timeframes", {})
            .keys()
        )

        for tf in self.timeframes:
            if tf not in supported_timeframes:
                raise ValueError(
                    f"Le timeframe '{tf}' n'est pas pris en charge. "
                    f"Timeframes disponibles: {list(supported_timeframes)}"
                )

    def _init_features_by_timeframe(self) -> Dict[str, List[str]]:
        """
        Initialise le dictionnaire des features par timeframe à partir de la configuration.

        Returns:
            Dictionnaire des features par timeframe
        """
        features_by_timeframe = {}
        timeframe_features = (
            self.config.get("data", {}).get("features_config", {}).get("timeframes", {})
        )

        for timeframe in self.timeframes:
            if timeframe in timeframe_features:
                timeframe_config = timeframe_features[timeframe]

                # Combiner toutes les features de price, volume et indicators
                all_features = []

                # Ajouter les features de prix
                if "price" in timeframe_config:
                    price_features = timeframe_config["price"]
                    if isinstance(price_features, list):
                        all_features.extend(price_features)
                    else:
                        all_features.append(price_features)

                # Ajouter les features de volume
                if "volume" in timeframe_config:
                    volume_features = timeframe_config["volume"]
                    if isinstance(volume_features, list):
                        all_features.extend(volume_features)
                    else:
                        all_features.append(volume_features)

                # Ajouter les indicateurs
                if "indicators" in timeframe_config:
                    indicator_features = timeframe_config["indicators"]
                    if isinstance(indicator_features, list):
                        all_features.extend(indicator_features)
                    else:
                        all_features.append(indicator_features)

                if all_features:
                    features_by_timeframe[timeframe] = all_features

        if not features_by_timeframe:
            # Configuration par défaut si aucune features_config n'est trouvée
            logger.warning(
                f"[DATA_LOADER Worker {self.worker_id}] Aucune configuration de features trouvée, utilisation des features par défaut"
            )
            for timeframe in self.timeframes:
                features_by_timeframe[timeframe] = [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",  # OHLCV de base
                    "sma_20",
                    "ema_12",
                    "rsi_14",
                    "macd",
                    "bbands_upper",
                    "bbands_lower",  # Indicateurs de base
                ]

        return features_by_timeframe

    def _get_data_path(self, asset: str, timeframe: str) -> Path:
        """
        Construit le chemin vers le fichier de données pour un actif et un timeframe donnés.

        La structure attendue est : {split}/{asset}/{timeframe}.parquet
        Par exemple: data/processed/indicators/val/btcusdt/5m.parquet

        Args:
            asset: Symbole de l'actif (ex: 'BTCUSDT')
            timeframe: Période de temps (ex: '5m', '1h')

        Returns:
            Chemin vers le fichier de données

        Raises:
            FileNotFoundError: Si le fichier n'est pas trouvé
            KeyError: Si la configuration est manquante
        """
        try:
            # Récupérer le répertoire de base des données
            base_dir = Path(self.config["paths"]["processed_data_dir"])

            # Utiliser directement le répertoire du split spécifié dans data_dirs
            data_dirs = self.config.get("data", {}).get("data_dirs", {})

            # Déterminer le répertoire de données en fonction du split
            if self.data_split in data_dirs:
                data_dir = Path(data_dirs[self.data_split])
            elif "base" in data_dirs:
                data_dir = Path(data_dirs["base"]) / self.data_split
            else:
                data_dir = base_dir / "indicators" / self.data_split

            logger.debug(f"Recherche des données dans: {data_dir}")

            # Nettoyer le nom de l'actif (supprimer / et -) et forcer en minuscules
            clean_asset = asset.replace("/", "").replace("-", "")

            # Extraire le timeframe de base (sans les combinaisons)
            base_timeframe = timeframe.split("_")[0] if "_" in timeframe else timeframe
            base_timeframe = base_timeframe.lower()  # Forcer en minuscules

            # Liste des variantes de casse à essayer pour l'actif
            asset_variants = [
                clean_asset.upper(),  # Tout en majuscules (ex: BTCUSDT) - nouvelle structure préférée
                clean_asset.lower(),  # Tout en minuscules (ex: btcusdt) - pour compatibilité
                clean_asset,  # Cas d'origine
            ]

            # Liste des variantes de casse à essayer pour le timeframe
            timeframe_variants = [
                base_timeframe.lower(),  # minuscules (ex: 5m)
                base_timeframe.upper(),  # majuscules (ex: 5M)
                base_timeframe,  # cas d'origine
            ]

            # Essayer chaque combinaison de variantes
            for asset_variant in asset_variants:
                for tf_variant in timeframe_variants:
                    file_path = data_dir / asset_variant / f"{tf_variant}.parquet"
                    if file_path.exists():
                        logger.debug(f"Fichier trouvé: {file_path}")
                        return file_path

            # Si on arrive ici, aucun fichier n'a été trouvé
            error_msg = (
                f"Fichier de données introuvable pour {asset}/{timeframe}.\n"
                f"Dossier de recherche: {data_dir}\n"
                f"Actifs testés: {', '.join(asset_variants)}\n"
                f"Timeframes testés: {', '.join(timeframe_variants)}\n"
                f"Vérifiez que le fichier existe et que les permissions sont correctes.\n"
                f"Structure attendue: {{split}}/{{asset}}/{{timeframe}}.parquet"
            )
            raise FileNotFoundError(error_msg)

        except KeyError as e:
            logger.error(f"Configuration des chemins manquante ou incorrecte: {str(e)}")
            raise
            logger.error(f"Erreur: {str(e)}")
            if "paths" in self.config:
                logger.error(
                    f"Chemins disponibles: {list(self.config['paths'].keys())}"
                )
            raise KeyError(
                f"Erreur de configuration des chemins: {str(e)}\n"
                f"Vérifiez que la configuration contient les chemins nécessaires."
            )

    def _load_asset_timeframe(self, asset: str, timeframe: str) -> pd.DataFrame:
        """
        Charge les données d'un actif et d'un timeframe spécifique.

        Args:
            asset: Symbole de l'actif (ex: 'BTC')
            timeframe: Période de temps (ex: '5m', '1h')

        Returns:
            DataFrame contenant les données demandées

        Raises:
            FileNotFoundError: Si le fichier n'est pas trouvé
            ValueError: Si les données sont corrompues ou incomplètes
        """
        file_path = self._get_data_path(asset, timeframe)

        try:
            # Load all columns from the parquet file
            df = pd.read_parquet(file_path)

            # Vérifie que le DataFrame n'est pas vide
            if df.empty:
                raise ValueError(f"Le fichier {file_path} est vide.")

            # Exclure la colonne timestamp si elle existe
            if "timestamp" in df.columns:
                df = df.drop("timestamp", axis=1)

            # Vérifie les colonnes OHLCV de base (en tenant compte de la casse)
            required_base_columns = {"Open", "High", "Low", "Close", "Volume"}
            available_columns = set(df.columns.str.upper())
            missing_base_columns = {
                col
                for col in required_base_columns
                if col.upper() not in available_columns
            }

            if missing_base_columns:
                raise ValueError(
                    f"Colonnes OHLCV manquantes dans {file_path}: {missing_base_columns}\n"
                    f"Colonnes disponibles: {sorted(df.columns)}"
                )

            # Log du nombre réel de features utilisées
            n_indicators = len(df.columns) - 5  # Total - OHLCV
            self.log_info(
                f"[DATA_LOADER] {asset}/{timeframe}: {len(df.columns)} colonnes (OHLCV + {n_indicators} indicateurs)"
            )
            logger.debug(
                f"Colonnes utilisées pour {asset}/{timeframe}: {sorted(df.columns)}"
            )

            # Vérifier et corriger les prix de clôture manquants avec interpolation (insensible à la casse)
            close_col = next(
                (col for col in df.columns if col.lower() == "close"), None
            )

            if close_col:
                nan_count_before = df[close_col].isna().sum()
                if nan_count_before > 0:
                    self.log_info(
                        f"[DATA_LOADER] {nan_count_before} prix de clôture manquants détectés pour {asset}/{timeframe}. Correction en cours..."
                    )

                    # Étape 1: Interpolation linéaire pour les valeurs intérieures
                    df[close_col] = df[close_col].interpolate(method="linear")

                    # Étape 2: Forward fill puis backward fill pour les bords
                    df[close_col] = df[close_col].ffill().bfill()

                    nan_count_after = df[close_col].isna().sum()
                    if nan_count_after > 0:
                        logger.warning(
                            f"[DATA_LOADER] {nan_count_after} NaN restants dans {close_col} pour {asset}/{timeframe} après correction. Remplacement par la dernière valeur valide."
                        )
                        df[close_col] = (
                            df[close_col].fillna(method="ffill").fillna(method="bfill")
                        )
                        if (
                            df[close_col].isna().any()
                        ):  # Si toujours des NaN (ex: début de fichier)
                            df[close_col] = df[close_col].fillna(
                                0
                            )  # Remplacer par 0 en dernier recours

                    # Vérification des valeurs aberrantes (prix <= 0)
                    invalid_prices = (df[close_col] <= 0).sum()
                    if invalid_prices > 0:
                        logger.warning(
                            f"[DATA_LOADER] {invalid_prices} prix invalides (<=0) détectés dans {close_col}. Remplacement par NaN puis interpolation."
                        )
                        df.loc[df[close_col] <= 0, close_col] = np.nan
                        df[close_col] = (
                            df[close_col].interpolate(method="linear").ffill().bfill()
                        )

            else:
                # Ce log ne devrait plus apparaître si vos données sont correctes
                logger.error(
                    f"Colonne 'close' introuvable pour {asset}/{timeframe} dans le DataLoader !"
                )

            logger.debug(f"Données chargées pour {asset} {timeframe}: {len(df)} lignes")
            return df

        except Exception as e:
            logger.error(f"Erreur lors du chargement de {file_path}: {str(e)}")
            raise

    def _load_asset_timeframe_parallel(
        self, asset: str, tf: str, max_retries: int = 3
    ) -> Tuple[str, str, pd.DataFrame]:
        """
        Charge les données d'un actif et d'un timeframe spécifique.
        Méthode utilisée pour le chargement parallèle.

        Args:
            asset: Symbole de l'actif
            tf: Timeframe à charger
            max_retries: Nombre maximum de tentatives en cas d'échec

        Returns:
            Tuple (asset, timeframe, DataFrame)

        Raises:
            Exception: Si le chargement échoue après plusieurs tentatives
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                logger.debug(f"Tentative {attempt + 1}/{max_retries} pour {asset} {tf}")
                df = self._load_asset_timeframe(asset, tf)
                logger.debug(f"Chargement réussi de {asset} {tf}: {len(df)} lignes")
                return asset, tf, df

            except Exception as e:
                last_error = e
                wait_time = (attempt + 1) * 2  # Attente exponentielle
                logger.warning(
                    f"Échec du chargement de {asset} {tf} (tentative {attempt + 1}/{max_retries}): {str(e)}. "
                    f"Nouvelle tentative dans {wait_time}s..."
                )
                time.sleep(wait_time)

        # Si on arrive ici, toutes les tentatives ont échoué
        error_msg = f"Impossible de charger {asset} {tf} après {max_retries} tentatives: {str(last_error)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    @lru_cache(maxsize=10)
    def load_chunk(self, chunk_idx: int = 0) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Charge un chunk de données pour tous les actifs et timeframes configurés en parallèle.
        Applique la validation des données prix vs indicateurs.

        Args:
            chunk_idx: Index du chunk à charger (non utilisé, conservé pour compatibilité)

        Returns:
            Dictionnaire imbriqué {actif: {timeframe: DataFrame}}

        Raises:
            RuntimeError: Si le chargement échoue pour un ou plusieurs actifs/timeframes
        """
        start_time = time.time()
        data = {asset: {} for asset in self.assets_list}
        total_tasks = len(self.assets_list) * len(self.timeframes)
        failed_loads = []

        self.log_info(
            f"[DATA_LOADER] Début du chargement du chunk {chunk_idx} pour {len(self.assets_list)} actifs et {len(self.timeframes)} timeframes"
        )

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Création des tâches de chargement
                futures = {
                    executor.submit(self._load_asset_timeframe_parallel, asset, tf): (
                        asset,
                        tf,
                    )
                    for asset in self.assets_list
                    for tf in self.timeframes
                }

                # Traitement des résultats au fur et à mesure
                with tqdm(total=total_tasks, desc="Chargement des données") as pbar:
                    for future in as_completed(futures):
                        asset, tf = futures[future]
                        try:
                            asset, tf, df = future.result()

                            # Appliquer la taille de chunk cible par timeframe (prend les dernières lignes)
                            logger.debug(
                                f"[DEBUG] Before slicing - Asset: {asset}, Timeframe: {tf}, Original df len: {len(df)}, Target len: {self.chunk_sizes.get(tf, len(df))}"
                            )
                            target_len = int(self.chunk_sizes.get(tf, len(df)))
                            if target_len <= 0:
                                target_len = len(df)
                            if len(df) > target_len:
                                df = df.iloc[-target_len:]
                            else:
                                # Si les données sont plus courtes que la cible, on log un avertissement
                                logger.warning(
                                    f"{asset} {tf}: données ({len(df)}) plus courtes que la taille de chunk cible ({target_len})."
                                )
                            logger.debug(
                                f"[DEBUG] After slicing - Asset: {asset}, Timeframe: {tf}, Final df len: {len(df)}"
                            )
                            data[asset][tf] = df
                            pbar.update(1)
                            pbar.set_postfix_str(f"{asset} {tf} - {len(df)} lignes")
                        except Exception as e:
                            logger.error(
                                f"Échec du chargement de {asset} {tf}: {str(e)}"
                            )
                            failed_loads.append((asset, tf, str(e)))
                            # Continue processing other assets/timeframes even if one fails

            # Validation du chunk avec le nouveau système
            if data:
                chunk_start = pd.Timestamp.now() - pd.Timedelta(hours=24)  # Estimation
                chunk_end = pd.Timestamp.now()

                is_valid, validation_info = self.data_validator.validate_chunk_data(
                    data, chunk_start, chunk_end
                )

                if not is_valid:
                    logger.warning(
                        f"[DATA_LOADER] Chunk {chunk_idx} validation failed: {validation_info['rejection_reasons']}"
                    )
                    # Continue avec les données disponibles mais log l'avertissement
                else:
                    logger.info(
                        f"[DATA_LOADER] Chunk {chunk_idx} validation passed: {validation_info['assets_validated']} assets validated"
                    )

            # Vérifier si tous les chargements ont réussi
            if failed_loads:
                failed_str = ", ".join(f"{asset}/{tf}" for asset, tf, _ in failed_loads)
                error_msg = (
                    f"Échec du chargement pour {len(failed_loads)}/{total_tasks} combinaisons actif/timeframe: {failed_str}"
                    f"\nDétails des erreurs:\n"
                )
                for asset, tf, err in failed_loads:
                    error_msg += f"- {asset} {tf}: {err}\n"

                # Si tous les chargements ont échoué, on lève une exception
                if len(failed_loads) == total_tasks:
                    raise RuntimeError(f"Tous les chargements ont échoué:\n{error_msg}")

                # Sinon, on log l'erreur mais on continue
                self.smart_logger.smart_error(logger, error_msg)

            # Vérifier que toutes les données attendues ont été chargées
            for asset in self.assets_list:
                if asset not in data or not data[asset]:
                    self.smart_logger.smart_error(
                        logger,
                        f"[DATA_LOADER] Aucune donnée chargée pour l'actif {asset}",
                    )
                    continue

                for tf in self.timeframes:
                    if tf not in data[asset]:
                        self.smart_logger.smart_warning(
                            logger,
                            f"[DATA_LOADER] Données manquantes pour {asset} {tf}",
                        )

            duration = time.time() - start_time
            validation_stats = self.data_validator.get_validation_stats()
            self.log_info(
                f"[DATA_LOADER] Chunk {chunk_idx} chargé en {duration:.2f}s | "
                f"Validation: {validation_stats['validation_rate']:.2%} success rate"
            )

            return data

        except Exception as e:
            logger.error(
                f"Erreur critique lors du chargement du chunk {chunk_idx}: {str(e)}"
            )
            logger.exception("Détails de l'erreur:")
            raise

    def _calculate_total_chunks(self) -> int:
        """
        Calcule le nombre total de chunks disponibles pour les données chargées.

        Le calcul est basé sur la taille des données disponibles et la taille des chunks
        configurée pour chaque timeframe. Le nombre de chunks est déterminé par le timeframe
        ayant le moins de données par rapport à la taille de ses chunks.

        Returns:
            int: Nombre total de chunks disponibles
        """
        # Calculer d'abord le nombre de chunks en fonction des données disponibles
        min_chunks = float("inf")

        # Parcourir tous les actifs et timeframes pour trouver le plus petit nombre de chunks
        for asset in self.assets_list:
            for tf in self.timeframes:
                try:
                    # Charger les données pour cet actif et ce timeframe
                    df = self._load_asset_timeframe(asset, tf)
                    chunk_size = self.chunk_sizes.get(tf, len(df))

                    # Calculer le nombre de chunks pour ce timeframe
                    num_chunks = max(1, len(df) // chunk_size)

                    # Prendre le plus petit nombre de chunks parmi tous les timeframes
                    if num_chunks < min_chunks:
                        min_chunks = num_chunks

                except Exception as e:
                    logger.warning(
                        f"Erreur lors du calcul des chunks pour {asset} {tf}: {str(e)}"
                    )
                    continue

        # Si on n'a pas pu déterminer le nombre de chunks, on retourne 1 par défaut
        if min_chunks == float("inf"):
            logger.warning(
                "Impossible de déterminer le nombre de chunks, utilisation de la valeur par défaut (1)"
            )
            calculated_chunks = 1
        else:
            calculated_chunks = min_chunks
            logger.info(f"Nombre total de chunks calculé : {calculated_chunks}")

        # Prendre le minimum entre la config et les données disponibles
        max_chunks = self.config.get("environment", {}).get("max_chunks_per_episode")
        if max_chunks is not None:
            final_chunks = min(int(max_chunks), calculated_chunks)
            logger.info(
                f"Chunks limités par configuration : {calculated_chunks} → {final_chunks}"
            )
            return final_chunks

        logger.info(f"Nombre de chunks utilisé : {calculated_chunks}")
        return calculated_chunks

    def get_available_assets(self, split="train"):
        """
        Sélectionne les meilleurs actifs basés sur le Sharpe Momentum Ratio.

        Formule: S_i = (momentum / volatility) * (1 / sqrt(max(correlation, 0.1)))

        Args:
            split: Jeu de données à utiliser ('train', 'val', 'test')

        Returns:
            Liste des actifs triés par score décroissant (meilleurs scores en premier)
        """
        import numpy as np

        logger.info(
            f"Calcul du Sharpe Momentum Ratio pour la sélection d'actifs (split: {split})"
        )

        # Récupérer les actifs disponibles
        assets = self.worker_config.get(
            "assets", self.config["environment"].get("assets", ["btcusdt"])
        )
        scores = {}

        for asset in assets:
            try:
                # Charger les données 1h pour le calcul (timeframe optimal pour momentum)
                timeframe = "1h"
                data_path = self._get_data_path(asset.upper(), timeframe)

                if not os.path.exists(data_path):
                    logger.warning(
                        f"Fichier non trouvé pour {asset}/{timeframe}: {data_path}"
                    )
                    continue

                df = pd.read_parquet(data_path)

                if df.empty or len(df) < 21:  # Besoin d'au moins 21 périodes
                    logger.warning(
                        f"Données insuffisantes pour {asset}: {len(df)} lignes"
                    )
                    continue

                # Calcul du momentum (rendement sur 20 périodes)
                close_prices = df["close"] if "close" in df.columns else df["CLOSE"]
                momentum = (
                    close_prices.iloc[-1] - close_prices.iloc[-21]
                ) / close_prices.iloc[-21]

                # Calcul de la volatilité (écart-type des rendements)
                returns = close_prices.pct_change().dropna()
                volatility = returns.std()

                # Calcul de la corrélation prix/volume (proxy pour la liquidité du marché)
                volume = df["volume"] if "volume" in df.columns else df["VOLUME"]
                correlation = close_prices.corr(volume)

                # Éviter les divisions par zéro et les corrélations nulles
                if volatility <= 0:
                    volatility = 0.01
                correlation = max(
                    abs(correlation) if not np.isnan(correlation) else 0.1, 0.1
                )

                # Calcul du Sharpe Momentum Ratio
                sharpe_momentum = (momentum / volatility) * (1 / np.sqrt(correlation))
                scores[asset] = sharpe_momentum

                logger.debug(
                    f"{asset}: momentum={momentum:.4f}, vol={volatility:.4f}, "
                    f"corr={correlation:.4f}, score={sharpe_momentum:.4f}"
                )

            except Exception as e:
                logger.error(f"Erreur lors du calcul pour {asset}: {str(e)}")
                scores[asset] = -999  # Score très faible en cas d'erreur

        # Trier par score décroissant (meilleurs scores en premier)
        sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        logger.info(f"Scores Sharpe Momentum calculés: {dict(sorted_assets)}")

        return [asset for asset, score in sorted_assets]
