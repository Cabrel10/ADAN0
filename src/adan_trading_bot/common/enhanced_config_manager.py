"""
Enhanced Configuration Manager with hot-reload capabilities.
Implements task 2.1 requirements for dynamic configuration management.
"""

import os
import json
import logging
import threading
import time
from typing import Dict, Any, Callable, List, Optional, Union
from pathlib import Path
from datetime import datetime
import yaml
import jsonschema
from jsonschema import validate, ValidationError

from .config_loader import ConfigLoader
from .config_validator import ConfigValidator
from .config_watcher import ConfigWatcher

logger = logging.getLogger(__name__)


class EnhancedConfigManager:
    """
    Enhanced configuration manager with hot-reload capabilities.

    Features:
    - Hot-reload of configuration files without restart
    - JSON schema validation
    - Configuration change callbacks
    - Thread-safe configuration access
    - Fallback to previous configuration on validation errors
    """

    def __init__(self, config_dir: str = "config", enable_hot_reload: bool = True):
        """
        Initialize the Enhanced Configuration Manager.

        Args:
            config_dir: Directory containing configuration files
            enable_hot_reload: Enable automatic hot-reload of configurations
        """
        self.config_dir = Path(config_dir)
        self.enable_hot_reload = enable_hot_reload

        # Configuration storage
        self.config_cache: Dict[str, Dict[str, Any]] = {}
        self.config_schemas: Dict[str, Dict[str, Any]] = {}
        self.config_lock = threading.RLock()

        # Validation and loading components
        self.config_loader = ConfigLoader()
        self.config_validator = ConfigValidator()
        self.config_watcher = None

        # Callbacks for configuration changes
        self.change_callbacks: Dict[str, List[Callable]] = {}

        # Configuration metadata
        self.last_loaded: Dict[str, datetime] = {}
        self.load_errors: Dict[str, str] = {}

        # Initialize
        self._initialize()

    def _initialize(self):
        """Initialize the configuration manager."""
        logger.info(f"Initializing EnhancedConfigManager with config_dir: {self.config_dir}")

        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration schemas
        self._load_schemas()

        # Load initial configurations
        self._load_all_configs()

        # Start hot-reload watcher if enabled
        if self.enable_hot_reload:
            self._start_hot_reload()

    def _load_schemas(self):
        """Load JSON schemas for configuration validation."""
        schema_dir = self.config_dir / "schemas"
        if not schema_dir.exists():
            logger.info("No schema directory found, creating default schemas")
            self._create_default_schemas()
            return

        for schema_file in schema_dir.glob("*.json"):
            config_type = schema_file.stem
            try:
                with open(schema_file, 'r') as f:
                    schema = json.load(f)
                self.config_schemas[config_type] = schema
                logger.info(f"Loaded schema for {config_type}")
            except Exception as e:
                logger.error(f"Failed to load schema {schema_file}: {e}")

    def _create_default_schemas(self):
        """Create default JSON schemas for configuration validation."""
        schema_dir = self.config_dir / "schemas"
        schema_dir.mkdir(exist_ok=True)

        # Default schemas for different configuration types
        schemas = {
            "model": {
                "type": "object",
                "properties": {
                    "architecture": {"type": "object"},
                    "diagnostics": {"type": "object"},
                    "style": {"type": "object"}
                },
                "required": ["architecture"]
            },
            "environment": {
                "type": "object",
                "properties": {
                    "initial_balance": {"type": "number", "minimum": 0},
                    "trading_fees": {"type": "number", "minimum": 0},
                    "max_steps": {"type": "integer", "minimum": 1},
                    "assets": {"type": "array", "items": {"type": "string"}},
                    "observation": {"type": "object"},
                    "memory": {"type": "object"},
                    "risk_management": {"type": "object"}
                },
                "required": ["initial_balance", "assets", "observation"]
            },
            "trading": {
                "type": "object",
                "properties": {
                    "futures_enabled": {"type": "boolean"},
                    "leverage": {"type": "number", "minimum": 1},
                    "commission_pct": {"type": "number", "minimum": 0},
                    "min_order_value_usdt": {"type": "number", "minimum": 0}
                },
                "required": ["commission_pct"]
            }
        }

        for config_type, schema in schemas.items():
            schema_file = schema_dir / f"{config_type}.json"
            with open(schema_file, 'w') as f:
                json.dump(schema, f, indent=2)
            self.config_schemas[config_type] = schema
            logger.info(f"Created default schema for {config_type}")

    def _load_all_configs(self):
        """Load all configuration files from the config directory."""
        config_files = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.yml"))

        for config_file in config_files:
            config_type = config_file.stem
            if config_type.endswith("_config"):
                config_type = config_type[:-7]  # Remove "_config" suffix

            self._load_single_config(config_type, config_file)

    def _load_single_config(self, config_type: str, config_path: Path):
        """Load a single configuration file."""
        try:
            logger.info(f"Loading configuration: {config_type} from {config_path}")

            # Load configuration using ConfigLoader (handles environment variables)
            config = self.config_loader.load_config(str(config_path))

            # Validate configuration if schema exists
            if config_type in self.config_schemas:
                self._validate_config(config_type, config)

            # Store configuration
            with self.config_lock:
                self.config_cache[config_type] = config
                self.last_loaded[config_type] = datetime.now()
                if config_type in self.load_errors:
                    del self.load_errors[config_type]

            logger.info(f"Successfully loaded configuration: {config_type}")

            # Notify callbacks
            self._notify_callbacks(config_type, config, {"action": "loaded"})

        except Exception as e:
            error_msg = f"Failed to load configuration {config_type}: {e}"
            logger.error(error_msg)
            self.load_errors[config_type] = error_msg

    def _validate_config(self, config_type: str, config: Dict[str, Any]):
        """Validate configuration against its schema."""
        if config_type not in self.config_schemas:
            logger.warning(f"No schema found for configuration type: {config_type}")
            return

        try:
            validate(instance=config, schema=self.config_schemas[config_type])
            logger.debug(f"Configuration validation passed for {config_type}")
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed for {config_type}: {e.message}")

    def _start_hot_reload(self):
        """Start the hot-reload watcher."""
        try:
            self.config_watcher = ConfigWatcher(
                config_dir=str(self.config_dir),
                enabled=True
            )

            # Register callback for configuration changes
            self.config_watcher.register_callback('*', self._handle_config_change)

            logger.info("Hot-reload watcher started")
        except Exception as e:
            logger.error(f"Failed to start hot-reload watcher: {e}")
            self.enable_hot_reload = False

    def _handle_config_change(self, config_type: str, new_config: Dict[str, Any], changes: Dict[str, Any]):
        """Handle configuration file changes from the watcher."""
        logger.info(f"Configuration change detected for {config_type}")

        try:
            # Validate new configuration
            if config_type in self.config_schemas:
                self._validate_config(config_type, new_config)

            # Store previous configuration for rollback
            with self.config_lock:
                previous_config = self.config_cache.get(config_type, {}).copy()

                # Update configuration
                self.config_cache[config_type] = new_config
                self.last_loaded[config_type] = datetime.now()

                if config_type in self.load_errors:
                    del self.load_errors[config_type]

            logger.info(f"Configuration hot-reloaded: {config_type}")

            # Notify callbacks
            self._notify_callbacks(config_type, new_config, changes)

        except Exception as e:
            logger.error(f"Failed to hot-reload configuration {config_type}: {e}")

            # Keep previous configuration on validation failure
            error_msg = f"Hot-reload failed for {config_type}: {e}"
            self.load_errors[config_type] = error_msg

    def _notify_callbacks(self, config_type: str, config: Dict[str, Any], changes: Dict[str, Any]):
        """Notify registered callbacks about configuration changes."""
        callbacks = self.change_callbacks.get(config_type, [])
        callbacks.extend(self.change_callbacks.get('*', []))  # Global callbacks

        for callback in callbacks:
            try:
                callback(config_type, config, changes)
            except Exception as e:
                logger.error(f"Error in configuration change callback: {e}")

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file and resolve all variables.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Resolved configuration dictionary
        """
        return self.config_loader.load_config(config_path)

    def get_config(self, config_type: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration by type.

        Args:
            config_type: Type of configuration to retrieve

        Returns:
            Configuration dictionary or None if not found
        """
        with self.config_lock:
            return self.config_cache.get(config_type, {}).copy()

    def get_config_value(self, config_type: str, key_path: str, default: Any = None) -> Any:
        """
        Get a specific configuration value using dot notation.

        Args:
            config_type: Type of configuration
            key_path: Dot-separated path to the value (e.g., 'agent.learning_rate')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        config = self.get_config(config_type)
        if not config:
            return default

        keys = key_path.split('.')
        current = config

        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default

    def set_config_value(self, config_type: str, key_path: str, value: Any) -> bool:
        """
        Set a specific configuration value using dot notation.

        Args:
            config_type: Type of configuration
            key_path: Dot-separated path to the value
            value: Value to set

        Returns:
            True if successful, False otherwise
        """
        with self.config_lock:
            if config_type not in self.config_cache:
                self.config_cache[config_type] = {}

            config = self.config_cache[config_type]
            keys = key_path.split('.')

            # Navigate to the parent of the target key
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Set the value
            current[keys[-1]] = value

            # Validate if schema exists
            try:
                if config_type in self.config_schemas:
                    self._validate_config(config_type, config)

                # Notify callbacks
                changes = {key_path: {"action": "modified", "new_value": value}}
                self._notify_callbacks(config_type, config, changes)

                return True
            except Exception as e:
                logger.error(f"Failed to set config value {key_path}: {e}")
                return False

    def register_change_callback(self, config_type: str, callback: Callable[[str, Dict[str, Any], Dict[str, Any]], None]):
        """
        Register a callback for configuration changes.

        Args:
            config_type: Configuration type to watch ('*' for all types)
            callback: Function to call on changes (config_type, new_config, changes)
        """
        if config_type not in self.change_callbacks:
            self.change_callbacks[config_type] = []

        self.change_callbacks[config_type].append(callback)
        logger.info(f"Registered change callback for {config_type}")

    def validate_config(self, config_type: str, config: Dict[str, Any]) -> bool:
        """
        Validate a configuration against its schema.

        Args:
            config_type: Type of configuration
            config: Configuration to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            self._validate_config(config_type, config)
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def reload_config(self, config_type: str) -> bool:
        """
        Force reload a specific configuration.

        Args:
            config_type: Type of configuration to reload

        Returns:
            True if successful, False otherwise
        """
        config_file = self.config_dir / f"{config_type}.yaml"
        if not config_file.exists():
            config_file = self.config_dir / f"{config_type}_config.yaml"

        if config_file.exists():
            self._load_single_config(config_type, config_file)
            return True
        else:
            logger.error(f"Configuration file not found for {config_type}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the configuration manager.

        Returns:
            Status information dictionary
        """
        with self.config_lock:
            return {
                "config_dir": str(self.config_dir),
                "hot_reload_enabled": self.enable_hot_reload,
                "loaded_configs": list(self.config_cache.keys()),
                "schemas_loaded": list(self.config_schemas.keys()),
                "last_loaded": {
                    config_type: timestamp.isoformat()
                    for config_type, timestamp in self.last_loaded.items()
                },
                "load_errors": self.load_errors.copy(),
                "callback_count": {
                    config_type: len(callbacks)
                    for config_type, callbacks in self.change_callbacks.items()
                }
            }

    def shutdown(self):
        """Shutdown the configuration manager."""
        logger.info("Shutting down EnhancedConfigManager")

        if self.config_watcher:
            self.config_watcher.stop()

        # Clear callbacks
        self.change_callbacks.clear()

        logger.info("EnhancedConfigManager shutdown completed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# Singleton instance for global access
_config_manager_instance = None
_config_manager_lock = threading.Lock()


def get_config_manager(config_dir: str = "config", enable_hot_reload: bool = True) -> EnhancedConfigManager:
    """
    Get the global configuration manager instance.

    Args:
        config_dir: Configuration directory
        enable_hot_reload: Enable hot-reload functionality

    Returns:
        EnhancedConfigManager instance
    """
    global _config_manager_instance

    with _config_manager_lock:
        if _config_manager_instance is None:
            _config_manager_instance = EnhancedConfigManager(
                config_dir=config_dir,
                enable_hot_reload=enable_hot_reload
            )
        return _config_manager_instance


def shutdown_config_manager():
    """Shutdown the global configuration manager."""
    global _config_manager_instance

    with _config_manager_lock:
        if _config_manager_instance is not None:
            _config_manager_instance.shutdown()
            _config_manager_instance = None
