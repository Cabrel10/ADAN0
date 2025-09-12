"""Utils de mise en cache pour les données d'entraînement."""
import logging
from pathlib import Path
from typing import Callable, Dict, Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


class DataCacheManager:
    """Gère le cache des données et des scalers pour l'entraînement."""

    def __init__(self, cache_dir: str) -> None:
        """Initialise le gestionnaire de cache.

        Args:
            cache_dir: Répertoire de stockage du cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.scaler_cache: Dict[str, StandardScaler] = {}

    def get_data_cache_key(self, asset: str, timeframe: str) -> str:
        """Génère une clé de cache pour un actif et un timeframe.
        Args:
            asset: Nom de l'actif (ex: 'BTC/USDT')
            timeframe: Période (ex: '1h', '4h')
        Returns:
            Clé de cache unique pour la paire actif/timeframe
        """
        return f"{asset}_{timeframe}"

    def get_or_load_data(
        self, asset: str, timeframe: str, loader_fn: Callable[[], pd.DataFrame]
    ) -> Tuple[pd.DataFrame, StandardScaler]:
        """Récupère les données du cache ou les charge via loader_fn.
        Args:
            asset: Nom de l'actif (ex: 'BTC/USDT')
            timeframe: Période (ex: '1h', '4h')
            loader_fn: Fonction pour charger les données
        Returns:
            Tuple contenant les données et le scaler
        """
        cache_key = self.get_data_cache_key(asset, timeframe)

        if cache_key not in self.data_cache:
            logger.info(f"Chargement des données pour {asset} {timeframe}")
            data = loader_fn()
            self.data_cache[cache_key] = data

            # Mettre à jour le scaler si nécessaire
            if timeframe not in self.scaler_cache:
                logger.info(f"Création du scaler pour {timeframe}")
                numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
                scaler = StandardScaler()
                scaler.fit(data[numeric_cols])
                self.scaler_cache[timeframe] = scaler
        else:
            logger.debug(f"Données trouvées dans le cache pour {asset} {timeframe}")

        return self.data_cache[cache_key], self.scaler_cache[timeframe]

    def load_or_process_data(
        self,
        data_loader,
        asset: str,
        timeframe: str,
        force_reload: bool = False,
        **kwargs,
    ) -> Tuple[pd.DataFrame, StandardScaler]:
        """Charge ou récupère les données depuis le cache.
        Args:
            data_loader: Instance de ChunkedDataLoader
            asset: Nom de l'actif
            timeframe: Période temporelle (ex: '5m', '1h')
            force_reload: Force le rechargement des données
            **kwargs: Arguments supplémentaires pour le chargeur
        Returns:
            Tuple contenant (données, scaler)
        """
        cache_key = self.get_data_cache_key(asset, timeframe)
        if not force_reload and cache_key in self.data_cache:
            logger.debug("Récupération des données depuis le cache: %s", cache_key)
            return (self.data_cache[cache_key], self.scaler_cache[timeframe])

        logger.info("Chargement des données pour %s %s", asset, timeframe)
        data = data_loader.load_data(asset, timeframe, **kwargs)
        self.data_cache[cache_key] = data

        if timeframe not in self.scaler_cache or force_reload:
            logger.info("Création du scaler pour %s", timeframe)
            numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
            scaler = StandardScaler()
            scaler.fit(data[numeric_cols])
            self.scaler_cache[timeframe] = scaler

        return data, self.scaler_cache[timeframe]

    def save_cache(self) -> None:
        """Sauvegarde le cache sur disque."""
        cache_file = self.cache_dir / "data_cache.pkl"
        joblib.dump({"data": self.data_cache, "scalers": self.scaler_cache}, cache_file)
        logger.info(f"Cache sauvegardé dans {cache_file}")

    def load_cache(self) -> bool:
        """Charge le cache depuis le disque.

        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        cache_file = self.cache_dir / "data_cache.pkl"
        if cache_file.exists():
            try:
                cache = joblib.load(cache_file)
                self.data_cache = cache.get("data", {})
                self.scaler_cache = cache.get("scalers", {})
                logger.info(f"Cache chargé depuis {cache_file}")
                return True
            except Exception as e:
                logger.warning(f"Erreur lors du chargement du cache: {e}")
                return False
        return False
