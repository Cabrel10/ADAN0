#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de diagnostic général pour le Trading Bot ADAN
Teste tous les modules et identifie les erreurs récurrentes
"""

import sys
import os
import traceback
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings

# Ajouter le chemin du projet
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "bot" / "src"))

class TradingBotDiagnostics:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.success = []
        self.module_errors = {}

    def log_error(self, category: str, error: str, details: str = ""):
        self.errors.append({
            'category': category,
            'error': error,
            'details': details,
            'traceback': traceback.format_exc()
        })
        print(f"❌ ERROR [{category}]: {error}")
        if details:
            print(f"   Details: {details}")

    def log_warning(self, category: str, warning: str):
        self.warnings.append({
            'category': category,
            'warning': warning
        })
        print(f"⚠️  WARNING [{category}]: {warning}")

    def log_success(self, category: str, message: str):
        self.success.append({
            'category': category,
            'message': message
        })
        print(f"✅ SUCCESS [{category}]: {message}")

    def test_basic_imports(self):
        """Test des importations de base"""
        print("\n🔍 TESTING BASIC IMPORTS...")

        basic_modules = [
            'numpy', 'pandas', 'torch', 'gymnasium', 'stable_baselines3',
            'yaml', 'ccxt', 'ta', 'sklearn', 'joblib'
        ]

        for module in basic_modules:
            try:
                importlib.import_module(module)
                self.log_success("BASIC_IMPORTS", f"{module} imported successfully")
            except ImportError as e:
                self.log_error("BASIC_IMPORTS", f"Failed to import {module}", str(e))

    def test_project_structure(self):
        """Test de la structure du projet"""
        print("\n🔍 TESTING PROJECT STRUCTURE...")

        required_paths = [
            "bot/src/adan_trading_bot",
            "bot/config/config.yaml",
            "bot/data/processed/indicators",
            "bot/scripts/train_parallel_agents.py",
            "bot/models",
            "bot/logs"
        ]

        for path in required_paths:
            full_path = project_root / path
            if full_path.exists():
                self.log_success("PROJECT_STRUCTURE", f"Path exists: {path}")
            else:
                self.log_error("PROJECT_STRUCTURE", f"Missing path: {path}")

    def test_adan_modules(self):
        """Test des modules ADAN"""
        print("\n🔍 TESTING ADAN MODULES...")

        modules_to_test = [
            'adan_trading_bot',
            'adan_trading_bot.environment',
            'adan_trading_bot.portfolio',
            'adan_trading_bot.agent',
            'adan_trading_bot.data_processing',
            'adan_trading_bot.common'
        ]

        for module_name in modules_to_test:
            try:
                module = importlib.import_module(module_name)
                self.log_success("ADAN_MODULES", f"Module {module_name} imported")

                # Test des classes principales
                if hasattr(module, '__all__'):
                    for class_name in module.__all__:
                        try:
                            getattr(module, class_name)
                            self.log_success("ADAN_MODULES", f"Class {class_name} accessible")
                        except AttributeError as e:
                            self.log_error("ADAN_MODULES", f"Class {class_name} not accessible", str(e))

            except ImportError as e:
                self.log_error("ADAN_MODULES", f"Failed to import {module_name}", str(e))

    def test_portfolio_manager_methods(self):
        """Test spécifique des méthodes PortfolioManager"""
        print("\n🔍 TESTING PORTFOLIO MANAGER METHODS...")

        try:
            from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

            # Créer une instance temporaire pour inspecter les méthodes
            env_config = {'initial_balance': 1000.0, 'default_currency': 'USDT'}
            portfolio = PortfolioManager(env_config=env_config)

            # Méthodes critiques à vérifier
            critical_methods = [
                'get_available_cash',  # ERREUR RÉCURRENTE #1
                'get_available_capital',
                'calculate_position_size',
                'validate_position',
                'get_portfolio_value',
                'get_total_equity',
                'update_portfolio',
                'execute_trade'
            ]

            for method_name in critical_methods:
                if hasattr(portfolio, method_name):
                    self.log_success("PORTFOLIO_METHODS", f"Method {method_name} exists")
                else:
                    self.log_error("PORTFOLIO_METHODS", f"Method {method_name} MISSING",
                                 "This is a critical error that causes runtime failures")

        except Exception as e:
            self.log_error("PORTFOLIO_METHODS", "Failed to test PortfolioManager", str(e))

    def test_environment_methods(self):
        """Test des méthodes d'environnement"""
        print("\n🔍 TESTING ENVIRONMENT METHODS...")

        try:
            from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

            # Vérifier les méthodes critiques
            critical_methods = [
                'step', 'reset', 'render', '_execute_trades',
                '_build_observation', '_calculate_reward'
            ]

            for method_name in critical_methods:
                if hasattr(MultiAssetChunkedEnv, method_name):
                    self.log_success("ENVIRONMENT_METHODS", f"Method {method_name} exists")
                else:
                    self.log_error("ENVIRONMENT_METHODS", f"Method {method_name} missing")

        except Exception as e:
            self.log_error("ENVIRONMENT_METHODS", "Failed to test Environment", str(e))

    def test_data_loading(self):
        """Test du chargement des données"""
        print("\n🔍 TESTING DATA LOADING...")

        try:
            from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader

            # Test du chargement des données
            data_dir = project_root / "bot" / "data" / "processed" / "indicators" / "val"

            if data_dir.exists():
                self.log_success("DATA_LOADING", f"Data directory exists: {data_dir}")

                # Vérifier les fichiers parquet
                parquet_files = list(data_dir.rglob("*.parquet"))
                if parquet_files:
                    self.log_success("DATA_LOADING", f"Found {len(parquet_files)} parquet files")
                else:
                    self.log_error("DATA_LOADING", "No parquet files found")
            else:
                self.log_error("DATA_LOADING", f"Data directory missing: {data_dir}")

        except Exception as e:
            self.log_error("DATA_LOADING", "Failed to test data loading", str(e))

    def test_configuration(self):
        """Test de la configuration"""
        print("\n🔍 TESTING CONFIGURATION...")

        try:
            import yaml
            config_path = project_root / "bot" / "config" / "config.yaml"

            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                self.log_success("CONFIGURATION", "Config file loaded successfully")

                # Vérifier les sections critiques
                critical_sections = [
                    'environment', 'agent', 'data', 'portfolio', 'paths'
                ]

                for section in critical_sections:
                    if section in config:
                        self.log_success("CONFIGURATION", f"Section '{section}' found")
                    else:
                        self.log_error("CONFIGURATION", f"Section '{section}' missing")
            else:
                self.log_error("CONFIGURATION", "Config file missing")

        except Exception as e:
            self.log_error("CONFIGURATION", "Failed to test configuration", str(e))

    def test_training_dependencies(self):
        """Test des dépendances d'entraînement"""
        print("\n🔍 TESTING TRAINING DEPENDENCIES...")

        try:
            # Test stable-baselines3 et gymnasium compatibility
            import gymnasium as gym
            from stable_baselines3 import PPO
            from stable_baselines3.common.env_checker import check_env

            self.log_success("TRAINING_DEPS", "PPO and Gymnasium imported successfully")

            # Test d'un environnement simple
            env = gym.make('CartPole-v1')
            check_env(env)
            self.log_success("TRAINING_DEPS", "Environment checker passed")

        except Exception as e:
            self.log_error("TRAINING_DEPS", "Training dependencies failed", str(e))

    def test_common_errors(self):
        """Test des erreurs communes identifiées"""
        print("\n🔍 TESTING COMMON ERRORS...")

        # Test 1: Erreur get_available_cash
        try:
            from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager
            env_config = {'initial_balance': 1000.0, 'default_currency': 'USDT'}
            portfolio = PortfolioManager(env_config=env_config)

            if hasattr(portfolio, 'get_available_cash'):
                self.log_success("COMMON_ERRORS", "get_available_cash method exists")
            else:
                self.log_error("COMMON_ERRORS", "RÉCURRENT ERROR #1: get_available_cash method missing",
                             "Code calls get_available_cash() but method is named get_available_capital()")

        except Exception as e:
            self.log_error("COMMON_ERRORS", "Failed to test portfolio methods", str(e))

        # Test 2: Problèmes d'imports circulaires
        try:
            import adan_trading_bot.environment.multi_asset_chunked_env
            import adan_trading_bot.portfolio.portfolio_manager
            import adan_trading_bot.agent.ppo_agent
            self.log_success("COMMON_ERRORS", "No circular import detected")
        except Exception as e:
            self.log_error("COMMON_ERRORS", "POSSIBLE RÉCURRENT ERROR #2: Circular import detected", str(e))

    def run_diagnostics(self):
        """Exécute tous les tests de diagnostic"""
        print("🚀 STARTING TRADING BOT DIAGNOSTICS...")
        print("=" * 60)

        # Exécuter tous les tests
        self.test_basic_imports()
        self.test_project_structure()
        self.test_adan_modules()
        self.test_portfolio_manager_methods()
        self.test_environment_methods()
        self.test_data_loading()
        self.test_configuration()
        self.test_training_dependencies()
        self.test_common_errors()

        # Rapport final
        self.generate_report()

    def generate_report(self):
        """Génère le rapport de diagnostic"""
        print("\n" + "=" * 60)
        print("📊 DIAGNOSTIC REPORT")
        print("=" * 60)

        print(f"✅ SUCCESS: {len(self.success)}")
        print(f"⚠️  WARNINGS: {len(self.warnings)}")
        print(f"❌ ERRORS: {len(self.errors)}")

        if self.errors:
            print("\n🔥 CRITICAL ERRORS TO FIX:")
            error_categories = {}
            for error in self.errors:
                category = error['category']
                if category not in error_categories:
                    error_categories[category] = []
                error_categories[category].append(error)

            for category, errors in error_categories.items():
                print(f"\n📂 {category}:")
                for i, error in enumerate(errors, 1):
                    print(f"  {i}. {error['error']}")
                    if error['details']:
                        print(f"     → {error['details']}")

        # Erreurs récurrentes identifiées
        print("\n🎯 IDENTIFIED RECURRING ERRORS:")
        recurring_errors = [
            "❌ ERREUR RÉCURRENTE #1: 'PortfolioManager' object has no attribute 'get_available_cash'",
            "   → SOLUTION: Remplacer get_available_cash() par get_available_capital()",
            "   → FICHIER: bot/src/adan_trading_bot/environment/multi_asset_chunked_env.py:3608"
        ]

        for error in recurring_errors:
            print(error)

        print(f"\n📈 OVERALL STATUS: {'🔴 FAILED' if self.errors else '🟢 PASSED'}")

if __name__ == "__main__":
    diagnostics = TradingBotDiagnostics()
    diagnostics.run_diagnostics()
