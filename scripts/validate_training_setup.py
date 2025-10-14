#!/usr/bin/env python3
"""
Script de validation pré-entraînement ADAN
Vérifie que tous les composants sont prêts pour l'entraînement complet
"""

import os
import sys
import yaml
import importlib
import logging
from pathlib import Path
from typing import Dict, List, Any
import torch
import numpy as np

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TrainingSetupValidator:
    """Validateur de configuration pré-entraînement"""

    def __init__(self, config_path: str = "bot/config/config.yaml"):
        self.config_path = config_path
        self.config = None
        self.validation_results = []

    def log_result(self, check_name: str, status: bool, message: str = ""):
        """Log un résultat de validation"""
        symbol = "✅" if status else "❌"
        self.validation_results.append((check_name, status, message))
        logger.info(f"{symbol} {check_name}: {message}")

    def load_config(self) -> bool:
        """Charge et valide la configuration"""
        try:
            if not os.path.exists(self.config_path):
                self.log_result(
                    "Configuration File", False, f"File not found: {self.config_path}"
                )
                return False

            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)

            self.log_result("Configuration File", True, "Loaded successfully")
            return True
        except Exception as e:
            self.log_result("Configuration File", False, f"Error loading: {e}")
            return False

    def validate_hyperparameters(self) -> bool:
        """Valide les hyperparamètres optimaux d'Optuna"""
        if not self.config:
            return False

        expected_hyperparams = {
            "w1": {
                "ent_coef": 0.0692561845110885,
                "learning_rate": 0.0000148322299348171,
                "batch_size": 128,
                "n_steps": 2048,
                "gamma": 0.931597523645738,
            },
            "w2": {
                "ent_coef": 0.0780972698389227,
                "learning_rate": 0.000150456050261472,
                "batch_size": 64,
                "n_steps": 1024,
                "gamma": 0.9451100218114,
            },
            "w3": {
                "ent_coef": 0.0733901049281371,
                "learning_rate": 0.000306666604891347,
                "batch_size": 64,
                "n_steps": 1024,
                "gamma": 0.933285598753683,
            },
            "w4": {
                "ent_coef": 0.0871767614575532,
                "learning_rate": 0.000491932356084968,
                "batch_size": 64,
                "n_steps": 2048,
                "gamma": 0.969306445782012,
            },
        }

        workers = self.config.get("workers", {})
        valid_count = 0

        for worker_id, expected_params in expected_hyperparams.items():
            if worker_id in workers:
                agent_config = workers[worker_id].get("agent_config", {})

                # Vérifier les paramètres clés
                valid_params = 0
                for param, expected_value in expected_params.items():
                    if param in agent_config:
                        actual_value = agent_config[param]
                        if isinstance(expected_value, float):
                            if abs(actual_value - expected_value) < 1e-10:
                                valid_params += 1
                        else:
                            if actual_value == expected_value:
                                valid_params += 1

                if valid_params >= 3:  # Au moins 3/5 paramètres corrects
                    valid_count += 1
                    self.log_result(
                        f"Hyperparams {worker_id}", True, f"{valid_params}/5 params OK"
                    )
                else:
                    self.log_result(
                        f"Hyperparams {worker_id}",
                        False,
                        f"Only {valid_params}/5 params OK",
                    )
            else:
                self.log_result(
                    f"Hyperparams {worker_id}", False, "Worker config not found"
                )

        success = valid_count >= 3  # Au moins 3/4 workers OK
        self.log_result(
            "Overall Hyperparameters", success, f"{valid_count}/4 workers configured"
        )
        return success

    def validate_fusion_weights(self) -> bool:
        """Valide les poids de fusion optimaux"""
        if not self.config:
            return False

        expected_weights = {0: 0.25, 1: 0.27, 2: 0.30, 3: 0.18}

        fusion_config = self.config.get("model_fusion", {})
        if not fusion_config.get("enabled", False):
            self.log_result("Model Fusion", False, "Fusion not enabled")
            return False

        weights = fusion_config.get("weights", {})

        valid_weights = 0
        for worker_idx, expected_weight in expected_weights.items():
            actual_weight = weights.get(worker_idx, 0)
            if abs(actual_weight - expected_weight) < 0.01:
                valid_weights += 1

        success = valid_weights == 4
        self.log_result("Model Fusion", success, f"{valid_weights}/4 weights correct")
        return success

    def validate_paths(self) -> bool:
        """Valide les chemins nécessaires"""
        if not self.config:
            return False

        paths_config = self.config.get("paths", {})
        required_paths = [
            ("data_dir", "data"),
            ("trained_models_dir", "models/rl_agents"),
            ("logs_dir", "logs"),
        ]

        valid_paths = 0
        for path_key, default_path in required_paths:
            path = paths_config.get(path_key, default_path)

            # Créer le chemin s'il n'existe pas
            try:
                os.makedirs(path, exist_ok=True)
                self.log_result(f"Path {path_key}", True, f"OK: {path}")
                valid_paths += 1
            except Exception as e:
                self.log_result(f"Path {path_key}", False, f"Error: {e}")

        success = valid_paths == len(required_paths)
        return success

    def validate_dependencies(self) -> bool:
        """Valide les dépendances Python"""
        required_modules = [
            ("torch", "PyTorch"),
            ("stable_baselines3", "Stable Baselines3"),
            ("numpy", "NumPy"),
            ("pandas", "Pandas"),
            ("matplotlib", "Matplotlib"),
            ("seaborn", "Seaborn"),
            ("yaml", "PyYAML"),
        ]

        valid_deps = 0
        for module_name, display_name in required_modules:
            try:
                importlib.import_module(module_name)
                self.log_result(f"Dependency {display_name}", True, "Available")
                valid_deps += 1
            except ImportError:
                self.log_result(f"Dependency {display_name}", False, "Not found")

        success = valid_deps == len(required_modules)
        return success

    def validate_training_config(self) -> bool:
        """Valide la configuration d'entraînement"""
        if not self.config:
            return False

        training_config = self.config.get("training", {})

        checks = [
            ("timesteps_per_instance", lambda x: x >= 50000, "Timesteps >= 50k"),
            ("save_freq", lambda x: x > 0, "Save frequency > 0"),
        ]

        valid_checks = 0
        for key, validator, description in checks:
            value = training_config.get(key, 0)
            if validator(value):
                self.log_result(f"Training {key}", True, f"{description}: {value}")
                valid_checks += 1
            else:
                self.log_result(
                    f"Training {key}", False, f"Invalid {description}: {value}"
                )

        # Vérifier export config
        export_config = training_config.get("final_model_export", {})
        if export_config.get("enabled", False):
            formats = export_config.get("formats", [])
            if "pytorch" in formats:
                self.log_result("Export PyTorch", True, "Enabled")
                valid_checks += 1
            else:
                self.log_result("Export PyTorch", False, "Not configured")

        success = valid_checks >= 2
        return success

    def validate_data_availability(self) -> bool:
        """Valide la disponibilité des données d'entraînement"""
        if not self.config:
            return False

        data_config = self.config.get("data", {})

        # Vérifier les actifs configurés
        assets = data_config.get("assets", [])
        if not assets:
            self.log_result("Data Assets", False, "No assets configured")
            return False

        # Vérifier les timeframes
        timeframes = data_config.get("timeframes", [])
        expected_timeframes = ["5m", "1h", "4h"]

        valid_timeframes = sum(1 for tf in expected_timeframes if tf in timeframes)

        success = len(assets) >= 1 and valid_timeframes >= 2
        self.log_result(
            "Data Config",
            success,
            f"Assets: {len(assets)}, Timeframes: {valid_timeframes}/3",
        )
        return success

    def validate_gpu_availability(self) -> bool:
        """Valide la disponibilité du GPU"""
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                self.log_result(
                    "GPU Support", True, f"Available: {gpu_name} ({gpu_count} devices)"
                )
                return True
            else:
                self.log_result(
                    "GPU Support", False, "CUDA not available (will use CPU)"
                )
                return True  # CPU training is still valid
        except Exception as e:
            self.log_result("GPU Support", False, f"Error checking GPU: {e}")
            return False

    def run_full_validation(self) -> bool:
        """Exécute toutes les validations"""
        logger.info("=" * 80)
        logger.info("🔍 ADAN TRAINING SETUP VALIDATION")
        logger.info("=" * 80)

        validations = [
            ("Configuration", self.load_config),
            ("Hyperparameters", self.validate_hyperparameters),
            ("Fusion Weights", self.validate_fusion_weights),
            ("File Paths", self.validate_paths),
            ("Dependencies", self.validate_dependencies),
            ("Training Config", self.validate_training_config),
            ("Data Config", self.validate_data_availability),
            ("GPU Support", self.validate_gpu_availability),
        ]

        passed = 0
        total = len(validations)

        for name, validator in validations:
            logger.info(f"\n--- Validating {name} ---")
            if validator():
                passed += 1

        logger.info("\n" + "=" * 80)
        logger.info(f"📊 VALIDATION SUMMARY: {passed}/{total} checks passed")

        if passed == total:
            logger.info("✅ ALL VALIDATIONS PASSED - Ready for training!")
            logger.info(
                "🚀 You can now run: timeout 120s python3 bot/scripts/train_parallel_agents.py --config-path bot/config/config.yaml --checkpoint-dir bot/checkpoints --resume"
            )
        elif passed >= total - 2:
            logger.info("⚠️ MOSTLY READY - Minor issues found but training should work")
        else:
            logger.info("❌ SIGNIFICANT ISSUES FOUND - Please fix before training")

        logger.info("=" * 80)

        return passed >= total - 1  # Allow for 1 minor failure

    def generate_training_summary(self):
        """Génère un résumé de la configuration d'entraînement"""
        if not self.config:
            return

        logger.info("\n🎯 TRAINING CONFIGURATION SUMMARY:")
        logger.info("-" * 50)

        # Workers info
        workers = self.config.get("workers", {})
        logger.info(f"👥 Workers: {len(workers)} configured")

        for worker_id, worker_config in workers.items():
            name = worker_config.get("name", "Unknown")
            logger.info(f"  • {worker_id}: {name}")

        # Training params
        training = self.config.get("training", {})
        timesteps = training.get("timesteps_per_instance", 0)
        logger.info(f"📊 Timesteps per instance: {timesteps:,}")

        # Fusion weights
        fusion = self.config.get("model_fusion", {})
        if fusion.get("enabled"):
            weights = fusion.get("weights", {})
            logger.info("🔗 Fusion weights:")
            for w_id, weight in weights.items():
                logger.info(f"  • w{w_id}: {weight:.1%}")

        # Export formats
        export_config = training.get("final_model_export", {})
        if export_config.get("enabled"):
            formats = export_config.get("formats", [])
            logger.info(f"💾 Export formats: {', '.join(formats)}")


def main():
    """Point d'entrée principal"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate ADAN training setup")
    parser.add_argument(
        "--config", "-c", default="bot/config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--summary",
        "-s",
        action="store_true",
        help="Show training configuration summary",
    )

    args = parser.parse_args()

    validator = TrainingSetupValidator(args.config)

    success = validator.run_full_validation()

    if args.summary:
        validator.generate_training_summary()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
