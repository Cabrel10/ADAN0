"""
Gestionnaire de chemins portable pour ADAN Trading Bot.
Utilise des chemins relatifs au répertoire racine du projet.
"""

import os
from pathlib import Path
from typing import Dict, Any


class PathManager:
    """Gère les chemins du projet de manière portable."""

    # Répertoire racine du projet (détecté automatiquement)
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

    # Chemins de base
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    SCRIPTS_DIR = PROJECT_ROOT / "scripts"
    CONFIGS_DIR = PROJECT_ROOT / "config"
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
    RESULTS_DIR = PROJECT_ROOT / "results"

    # Chemins de données
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    INDICATORS_DIR = PROCESSED_DATA_DIR / "indicators"

    # Chemins de modèles
    RL_AGENTS_DIR = MODELS_DIR / "rl_agents"

    # Chemins de logs
    TRAINING_LOGS_DIR = LOGS_DIR / "training"
    BACKTEST_LOGS_DIR = LOGS_DIR / "backtest"

    # Chemins de rapports
    METRICS_REPORTS_DIR = REPORTS_DIR / "metrics"
    FIGURES_REPORTS_DIR = REPORTS_DIR / "figures"

    @classmethod
    def ensure_dirs_exist(cls) -> None:
        """Crée tous les répertoires nécessaires s'ils n'existent pas."""
        dirs = [
            cls.DATA_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR,
            cls.REPORTS_DIR,
            cls.SCRIPTS_DIR,
            cls.CONFIGS_DIR,
            cls.CHECKPOINTS_DIR,
            cls.RESULTS_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.INDICATORS_DIR,
            cls.RL_AGENTS_DIR,
            cls.TRAINING_LOGS_DIR,
            cls.BACKTEST_LOGS_DIR,
            cls.METRICS_REPORTS_DIR,
            cls.FIGURES_REPORTS_DIR,
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_config_path(cls, config_name: str = "config.yaml") -> Path:
        """Retourne le chemin vers un fichier de configuration."""
        return cls.CONFIGS_DIR / config_name

    @classmethod
    def get_data_path(
        cls, asset: str, timeframe: str, data_type: str = "train"
    ) -> Path:
        """Retourne le chemin vers un fichier de données."""
        if data_type == "train":
            return cls.INDICATORS_DIR / "train" / asset / f"{timeframe}.parquet"
        elif data_type == "val":
            return cls.INDICATORS_DIR / "val" / asset / f"{timeframe}.parquet"
        elif data_type == "test":
            return cls.INDICATORS_DIR / "test" / asset / f"{timeframe}.parquet"
        else:
            raise ValueError(f"Type de données inconnu: {data_type}")

    @classmethod
    def get_model_path(
        cls, algorithm: str, asset: str, timestamp: str
    ) -> Path:
        """Retourne le chemin vers un modèle entraîné."""
        return cls.RL_AGENTS_DIR / algorithm / asset / timestamp

    @classmethod
    def get_checkpoint_path(cls, checkpoint_name: str) -> Path:
        """Retourne le chemin vers un checkpoint."""
        return cls.CHECKPOINTS_DIR / checkpoint_name

    @classmethod
    def get_log_path(cls, log_name: str, log_type: str = "training") -> Path:
        """Retourne le chemin vers un fichier de log."""
        if log_type == "training":
            return cls.TRAINING_LOGS_DIR / log_name
        elif log_type == "backtest":
            return cls.BACKTEST_LOGS_DIR / log_name
        else:
            raise ValueError(f"Type de log inconnu: {log_type}")

    @classmethod
    def get_report_path(cls, report_name: str, report_type: str = "metrics") -> Path:
        """Retourne le chemin vers un rapport."""
        if report_type == "metrics":
            return cls.METRICS_REPORTS_DIR / report_name
        elif report_type == "figures":
            return cls.FIGURES_REPORTS_DIR / report_name
        else:
            raise ValueError(f"Type de rapport inconnu: {report_type}")

    @classmethod
    def to_dict(cls) -> Dict[str, str]:
        """Retourne un dictionnaire de tous les chemins."""
        return {
            "PROJECT_ROOT": str(cls.PROJECT_ROOT),
            "DATA_DIR": str(cls.DATA_DIR),
            "MODELS_DIR": str(cls.MODELS_DIR),
            "LOGS_DIR": str(cls.LOGS_DIR),
            "REPORTS_DIR": str(cls.REPORTS_DIR),
            "SCRIPTS_DIR": str(cls.SCRIPTS_DIR),
            "CONFIGS_DIR": str(cls.CONFIGS_DIR),
            "CHECKPOINTS_DIR": str(cls.CHECKPOINTS_DIR),
            "RESULTS_DIR": str(cls.RESULTS_DIR),
            "RAW_DATA_DIR": str(cls.RAW_DATA_DIR),
            "PROCESSED_DATA_DIR": str(cls.PROCESSED_DATA_DIR),
            "INDICATORS_DIR": str(cls.INDICATORS_DIR),
            "RL_AGENTS_DIR": str(cls.RL_AGENTS_DIR),
            "TRAINING_LOGS_DIR": str(cls.TRAINING_LOGS_DIR),
            "BACKTEST_LOGS_DIR": str(cls.BACKTEST_LOGS_DIR),
            "METRICS_REPORTS_DIR": str(cls.METRICS_REPORTS_DIR),
            "FIGURES_REPORTS_DIR": str(cls.FIGURES_REPORTS_DIR),
        }

    @classmethod
    def print_paths(cls) -> None:
        """Affiche tous les chemins du projet."""
        print("\n" + "=" * 80)
        print("CHEMINS DU PROJET ADAN TRADING BOT")
        print("=" * 80)
        for key, value in cls.to_dict().items():
            print(f"{key:30} : {value}")
        print("=" * 80 + "\n")
