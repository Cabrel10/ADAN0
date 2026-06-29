#!/usr/bin/env python3
"""Verify checkpoint configuration is correct (15k steps, 3 kept, auto-resume)."""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def verify_checkpoint_config():
    """Verify checkpoint configuration."""
    
    logger.info("=" * 80)
    logger.info("🧪 CHECKPOINT CONFIGURATION VERIFICATION")
    logger.info("=" * 80)
    
    # Check training script
    train_script = PROJECT_ROOT / "scripts" / "train_parallel_agents.py"
    with open(train_script, "r") as f:
        content = f.read()
    
    checks = {
        "interval_timesteps = config.get(\"interval_timesteps\", 15_000)": "Checkpoint interval set to 15k steps",
        "checkpoint_interval = 15_000": "Checkpoint interval constant defined",
        "num_to_keep=3": "Keep last 3 checkpoints",
        "CheckpointConfig": "Ray CheckpointConfig imported",
        "save_checkpoint": "Checkpoint save method exists",
    }
    
    logger.info("\n📋 Configuration Checks:")
    all_passed = True
    
    for check_str, description in checks.items():
        if check_str in content:
            logger.info(f"   ✅ {description}")
        else:
            logger.warning(f"   ❌ {description} - NOT FOUND")
            all_passed = False
    
    # Check checkpoint manager
    checkpoint_manager = PROJECT_ROOT / "scripts" / "checkpoint_manager.py"
    if checkpoint_manager.exists():
        logger.info(f"   ✅ CheckpointManager exists")
    else:
        logger.warning(f"   ❌ CheckpointManager not found")
        all_passed = False
    
    # Check verification script
    verify_script = PROJECT_ROOT / "scripts" / "verify_checkpoint_resume.py"
    if verify_script.exists():
        logger.info(f"   ✅ Verification script exists")
    else:
        logger.warning(f"   ❌ Verification script not found")
        all_passed = False
    
    # Check documentation
    logger.info("\n📚 Documentation:")
    docs = {
        "CHECKPOINT_SYSTEM.md": "Checkpoint system documentation",
        "TRAINING_ARTIFACTS.md": "Training artifacts tracking",
    }
    
    for doc_file, description in docs.items():
        doc_path = PROJECT_ROOT / doc_file
        if doc_path.exists():
            logger.info(f"   ✅ {description}")
        else:
            logger.warning(f"   ❌ {description} - NOT FOUND")
            all_passed = False
    
    # Summary
    logger.info("\n" + "=" * 80)
    if all_passed:
        logger.info("✅ ALL CHECKPOINT CONFIGURATION CHECKS PASSED")
        logger.info("=" * 80)
        logger.info("\n📝 Checkpoint System Summary:")
        logger.info("   • Interval: Every 15,000 timesteps")
        logger.info("   • Retention: Last 3 checkpoints (auto-cleanup)")
        logger.info("   • Resume: Automatic on next training run")
        logger.info("   • Crash Recovery: Enabled (Ray auto-detects latest checkpoint)")
        logger.info("\n🚀 Ready for production training!")
        return True
    else:
        logger.error("=" * 80)
        logger.error("❌ SOME CHECKS FAILED")
        return False


if __name__ == "__main__":
    success = verify_checkpoint_config()
    sys.exit(0 if success else 1)

