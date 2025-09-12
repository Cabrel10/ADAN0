#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to validate the consolidated configuration for the ADAN trading bot.

This script checks that the main `config.yaml` is present and that the `workers`
section is well-formed with all required parameters.
"""

import sys
import yaml
from pathlib import Path


def validate_workers_config(config_path: Path) -> bool:
    """Loads and validates the worker configuration from the main config file."""
    print(f"--> Validating worker configurations in: {config_path}")

    if not config_path.exists():
        print(f"[ERROR] Configuration file not found at {config_path}", file=sys.stderr)
        return False

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"[ERROR] Failed to parse YAML file: {e}", file=sys.stderr)
        return False

    if 'workers' not in config_data:
        print("[ERROR] 'workers' section not found in the configuration.", file=sys.stderr)
        return False

    workers = config_data.get('workers', {})
    if not isinstance(workers, dict) or not workers:
        print("[ERROR] 'workers' section must be a non-empty dictionary.", file=sys.stderr)
        return False

    print(f"    Found {len(workers)} worker configurations. Validating each...")
    all_valid = True
    required_keys = ['name', 'timeframes', 'reward_config', 'dbe_config', 'agent_config']

    for worker_id, worker_config in workers.items():
        missing_keys = [key for key in required_keys if key not in worker_config]
        if missing_keys:
            print(f"  - [FAIL] Worker '{worker_id}': Missing required keys: {missing_keys}")
            all_valid = False
        else:
            print(f"  - [PASS] Worker '{worker_id}': All required keys are present.")

    return all_valid


def main() -> int:
    """Main validation function."""
    print("=====================================================")
    print("         ADAN Configuration Validator         ")
    print("=====================================================")

    # Project root is two levels up from the script's directory (bot/scripts -> bot)
    project_root = Path(__file__).parent.parent
    config_file = project_root / 'config' / 'config.yaml'

    is_valid = validate_workers_config(config_file)

    print("-----------------------------------------------------")
    if is_valid:
        print("✅ Configuration validation successful!")
        return 0
    else:
        print("❌ Configuration validation failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
