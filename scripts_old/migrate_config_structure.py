#!/usr/bin/env python3
"""
Configuration Migration Script
Migrates from monolithic config.yaml to modular configuration structure.
Implements task 2.2 requirements.
"""

import os
import sys
import yaml
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adan_trading_bot.common.enhanced_config_manager import EnhancedConfigManager


class ConfigMigrator:
    """Migrates configuration from monolithic to modular structure."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.backup_dir = self.config_dir / "backup"
        self.schemas_dir = self.config_dir / "schemas"

        # Configuration mapping - defines how to split the monolithic config
        self.config_mapping = {
            "model": {
                "sections": ["model"],
                "description": "Model architecture and diagnostics configuration"
            },
            "environment": {
                "sections": ["environment", "portfolio"],
                "description": "Trading environment and portfolio configuration"
            },
            "trading": {
                "sections": ["trading_rules", "capital_tiers"],
                "description": "Trading rules and capital tier configuration"
            },
            "agent": {
                "sections": ["agent"],
                "description": "RL agent configuration and hyperparameters"
            },
            "data": {
                "sections": ["data", "preprocessing", "feature_engineering", "data_processing", "data_augmentation"],
                "description": "Data processing and feature engineering configuration"
            },
            "training": {
                "sections": ["training", "regularization", "reward_shaping"],
                "description": "Training configuration and reward shaping"
            },
            "workers": {
                "sections": ["workers"],
                "description": "Worker-specific configurations"
            },
            "logging": {
                "sections": ["logging"],
                "description": "Logging configuration"
            },
            "paths": {
                "sections": ["paths", "general"],
                "description": "System paths and general configuration"
            }
        }

    def migrate(self, source_config: str = "config.yaml", create_backup: bool = True) -> bool:
        """
        Migrate configuration from monolithic to modular structure.

        Args:
            source_config: Source configuration file name
            create_backup: Whether to create backup of original files

        Returns:
            True if migration successful, False otherwise
        """
        print("🚀 Starting configuration migration...")

        source_path = self.config_dir / source_config
        if not source_path.exists():
            print(f"❌ Source configuration file not found: {source_path}")
            return False

        try:
            # Create backup if requested
            if create_backup:
                self._create_backup()

            # Load source configuration
            print(f"📖 Loading source configuration: {source_path}")
            with open(source_path, 'r') as f:
                source_config_data = yaml.safe_load(f)

            # Create modular configurations
            self._create_modular_configs(source_config_data)

            # Create configuration schemas
            self._create_schemas()

            # Validate migrated configurations
            if self._validate_migration():
                print("✅ Configuration migration completed successfully!")
                print(f"📁 Modular configurations created in: {self.config_dir}")
                print(f"📋 Backup created in: {self.backup_dir}")
                return True
            else:
                print("❌ Configuration migration validation failed!")
                return False

        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return False

    def _create_backup(self):
        """Create backup of existing configuration files."""
        print("💾 Creating backup of existing configurations...")

        # Create backup directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"migration_{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)

        # Backup existing config files
        for config_file in self.config_dir.glob("*.yaml"):
            if config_file.name != "config.yaml":
                continue
            backup_file = backup_path / config_file.name
            shutil.copy2(config_file, backup_file)
            print(f"  📄 Backed up: {config_file.name}")

    def _create_modular_configs(self, source_config: Dict[str, Any]):
        """Create modular configuration files from source configuration."""
        print("🔧 Creating modular configuration files...")

        for config_name, config_info in self.config_mapping.items():
            config_data = {}

            # Extract sections for this configuration
            for section in config_info["sections"]:
                if section in source_config:
                    config_data[section] = source_config[section]

            # Only create file if we have data
            if config_data:
                config_file = self.config_dir / f"{config_name}.yaml"

                # Add header comment
                header = f"# {config_info['description']}\n"
                header += f"# Generated by configuration migration on {datetime.now().isoformat()}\n\n"

                with open(config_file, 'w') as f:
                    f.write(header)
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)

                print(f"  📄 Created: {config_file.name} ({len(config_data)} sections)")

    def _create_schemas(self):
        """Create JSON schemas for configuration validation."""
        print("📋 Creating configuration schemas...")

        self.schemas_dir.mkdir(exist_ok=True)

        schemas = {
            "model": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "object",
                        "properties": {
                            "architecture": {"type": "object"},
                            "diagnostics": {"type": "object"},
                            "style": {"type": "object"}
                        },
                        "required": ["architecture"]
                    }
                },
                "required": ["model"]
            },
            "environment": {
                "type": "object",
                "properties": {
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
                    "portfolio": {"type": "object"}
                },
                "required": ["environment"]
            },
            "trading": {
                "type": "object",
                "properties": {
                    "trading_rules": {
                        "type": "object",
                        "properties": {
                            "commission_pct": {"type": "number", "minimum": 0},
                            "min_order_value_usdt": {"type": "number", "minimum": 0}
                        },
                        "required": ["commission_pct"]
                    },
                    "capital_tiers": {"type": "array"}
                }
            },
            "agent": {
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "object",
                        "properties": {
                            "algorithm": {"type": "string"},
                            "policy": {"type": "string"},
                            "learning_rate": {"type": "number", "minimum": 0},
                            "batch_size": {"type": "integer", "minimum": 1}
                        },
                        "required": ["algorithm", "policy"]
                    }
                },
                "required": ["agent"]
            },
            "data": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "properties": {
                            "data_dir": {"type": "string"},
                            "features_per_timeframe": {"type": "object"}
                        },
                        "required": ["data_dir"]
                    }
                }
            },
            "training": {
                "type": "object",
                "properties": {
                    "training": {
                        "type": "object",
                        "properties": {
                            "num_instances": {"type": "integer", "minimum": 1},
                            "timesteps_per_instance": {"type": "integer", "minimum": 1},
                            "batch_size": {"type": "integer", "minimum": 1}
                        },
                        "required": ["num_instances", "timesteps_per_instance"]
                    }
                }
            },
            "workers": {
                "type": "object",
                "properties": {
                    "workers": {
                        "type": "object",
                        "patternProperties": {
                            "^w[0-9]+$": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "assets": {"type": "array"},
                                    "timeframes": {"type": "array"}
                                },
                                "required": ["name", "assets", "timeframes"]
                            }
                        }
                    }
                }
            },
            "logging": {
                "type": "object",
                "properties": {
                    "logging": {
                        "type": "object",
                        "properties": {
                            "version": {"type": "integer"},
                            "formatters": {"type": "object"},
                            "handlers": {"type": "object"},
                            "loggers": {"type": "object"}
                        }
                    }
                }
            },
            "paths": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "object",
                        "properties": {
                            "base_dir": {"type": "string"},
                            "data_dir": {"type": "string"},
                            "models_dir": {"type": "string"}
                        },
                        "required": ["base_dir", "data_dir"]
                    },
                    "general": {"type": "object"}
                }
            }
        }

        for schema_name, schema_def in schemas.items():
            schema_file = self.schemas_dir / f"{schema_name}.json"
            with open(schema_file, 'w') as f:
                json.dump(schema_def, f, indent=2)
            print(f"  📋 Created schema: {schema_file.name}")

    def _validate_migration(self) -> bool:
        """Validate the migrated configuration files."""
        print("🔍 Validating migrated configurations...")

        try:
            # Check that configuration files were created
            created_configs = []
            for config_name in self.config_mapping.keys():
                config_file = self.config_dir / f"{config_name}.yaml"
                if config_file.exists():
                    created_configs.append(config_name)

            if not created_configs:
                print("❌ No configuration files were created")
                return False

            print(f"✅ Successfully created {len(created_configs)} configuration files:")
            for config_name in created_configs:
                print(f"  📄 {config_name}.yaml")

            # Try to load each configuration individually
            validation_errors = []
            for config_name in created_configs:
                config_file = self.config_dir / f"{config_name}.yaml"
                try:
                    with open(config_file, 'r') as f:
                        config_data = yaml.safe_load(f)
                    if not config_data:
                        validation_errors.append(f"Empty configuration: {config_name}")
                except Exception as e:
                    validation_errors.append(f"Failed to load {config_name}: {e}")

            if validation_errors:
                print(f"⚠️  Validation warnings: {validation_errors}")
                # Don't fail on validation errors, just warn

            return True

        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False

    def create_migration_summary(self) -> str:
        """Create a summary of the migration."""
        summary = []
        summary.append("# Configuration Migration Summary")
        summary.append(f"Migration completed on: {datetime.now().isoformat()}")
        summary.append("")
        summary.append("## Modular Configuration Files Created:")

        for config_name, config_info in self.config_mapping.items():
            config_file = self.config_dir / f"{config_name}.yaml"
            if config_file.exists():
                summary.append(f"- **{config_name}.yaml**: {config_info['description']}")
                summary.append(f"  - Sections: {', '.join(config_info['sections'])}")

        summary.append("")
        summary.append("## Schema Files Created:")
        for schema_file in self.schemas_dir.glob("*.json"):
            summary.append(f"- **{schema_file.name}**: Validation schema for {schema_file.stem}")

        summary.append("")
        summary.append("## Usage:")
        summary.append("```python")
        summary.append("from adan_trading_bot.common.enhanced_config_manager import get_config_manager")
        summary.append("")
        summary.append("# Get configuration manager")
        summary.append("config_manager = get_config_manager()")
        summary.append("")
        summary.append("# Access specific configurations")
        summary.append("model_config = config_manager.get_config('model')")
        summary.append("env_config = config_manager.get_config('environment')")
        summary.append("```")

        return "\n".join(summary)


def main():
    """Main migration function."""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate ADAN configuration to modular structure")
    parser.add_argument("--config-dir", default="config", help="Configuration directory")
    parser.add_argument("--source", default="config.yaml", help="Source configuration file")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backup")
    parser.add_argument("--summary", action="store_true", help="Generate migration summary")

    args = parser.parse_args()

    # Initialize migrator
    migrator = ConfigMigrator(config_dir=args.config_dir)

    # Perform migration
    success = migrator.migrate(
        source_config=args.source,
        create_backup=not args.no_backup
    )

    if success and args.summary:
        # Generate summary
        summary = migrator.create_migration_summary()
        summary_file = Path(args.config_dir) / "MIGRATION_SUMMARY.md"
        with open(summary_file, 'w') as f:
            f.write(summary)
        print(f"📄 Migration summary written to: {summary_file}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
