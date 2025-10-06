#!/usr/bin/env bash
set -euo pipefail

# Run only the targeted timeout-related unit tests
pytest -q tests/unit/utils/test_timeout_manager.py \
           tests/unit/training/test_trainer_timeout_integration.py \
           tests/unit/training/test_parallel_trainer_timeout.py \
           -c /dev/null
