#!/usr/bin/env python3
"""Verify checkpoint save/resume pipeline — end-to-end test.

Tests:
  1. Save checkpoint with model + VecNormalize + metadata
  2. Load checkpoint and verify integrity
  3. Resume training from checkpoint
  4. Verify training continues correctly (no step reset)
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# Setup path
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

# Import checkpoint manager
sys.path.insert(0, str(_SCRIPT_DIR))
from checkpoint_manager import CheckpointManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = _SCRIPT_DIR.parent


def test_checkpoint_pipeline():
    """End-to-end checkpoint save/resume test."""
    
    logger.info("=" * 80)
    logger.info("🧪 CHECKPOINT PIPELINE VERIFICATION")
    logger.info("=" * 80)
    
    # Load config
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = ConfigLoader.load_config(str(config_path))
    logger.info(f"✅ Config loaded from {config_path}")
    
    # Create temp checkpoint dir
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize checkpoint manager
        mgr = CheckpointManager(str(ckpt_dir), max_checkpoints=3)
        logger.info(f"✅ CheckpointManager initialized at {ckpt_dir}")
        
        # Create minimal environment
        worker_config = config.get("workers", {}).get("w1", {})
        worker_config["worker_id"] = 0
        worker_config.setdefault("data_split_override", "train")
        
        loader = ChunkedDataLoader(config=config, worker_config=worker_config, worker_id=0)
        data = loader.load_chunk(0)
        logger.info(f"✅ Data loaded: {list(data.keys())} assets")
        
        env = MultiAssetChunkedEnv(
            data=data,
            config=config,
            worker_config=worker_config,
            worker_id=0,
            live_mode=False,
        )
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, gamma=0.99)
        logger.info(f"✅ Environment created")
        
        # Create PPO model
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=32,
            n_epochs=4,
            verbose=0,
            device="cpu",
        )
        logger.info(f"✅ PPO model created")
        
        # Train for a few steps
        logger.info("🔄 Training for 500 steps...")
        model.learn(total_timesteps=500, reset_num_timesteps=True)
        initial_steps = model.num_timesteps
        logger.info(f"✅ Training complete: {initial_steps} steps")
        
        # Save checkpoint
        temp_model_path = ckpt_dir / "temp_model.zip"
        temp_vecnorm_path = ckpt_dir / "temp_vecnorm.pkl"
        model.save(str(temp_model_path))
        vec_env.save(str(temp_vecnorm_path))
        
        worker_state = {
            "total_timesteps": initial_steps,
            "learning_rate": 3e-4,
            "ent_coef": 0.01,
            "gamma": 0.99,
        }
        
        ckpt_id, success = mgr.save_checkpoint(
            str(temp_model_path),
            str(temp_vecnorm_path),
            worker_state,
            step=initial_steps,
            metrics={"mean_reward": 0.5, "sharpe": 1.2},
        )
        
        if not success:
            logger.error("❌ Failed to save checkpoint")
            return False
        logger.info(f"✅ Checkpoint saved: {ckpt_id}")
        
        # Verify checkpoint integrity
        if not mgr.verify_checkpoint_integrity(ckpt_id):
            logger.error("❌ Checkpoint integrity check failed")
            return False
        logger.info(f"✅ Checkpoint integrity verified")
        
        # List checkpoints
        ckpts = mgr.list_checkpoints()
        logger.info(f"✅ Available checkpoints: {len(ckpts)}")
        for ckpt in ckpts:
            logger.info(f"   - {ckpt['id']} (step {ckpt['step']})")
        
        # Load checkpoint
        model_path, vecnorm_path, loaded_state, success = mgr.load_checkpoint(ckpt_id)
        if not success:
            logger.error("❌ Failed to load checkpoint")
            return False
        logger.info(f"✅ Checkpoint loaded successfully")
        logger.info(f"   Model: {model_path}")
        logger.info(f"   VecNorm: {vecnorm_path}")
        logger.info(f"   State: {loaded_state}")
        
        # Resume training
        logger.info("🔄 Resuming training from checkpoint...")
        
        # Reload environment (fresh)
        env2 = MultiAssetChunkedEnv(
            data=data,
            config=config,
            worker_config=worker_config,
            worker_id=0,
            live_mode=False,
        )
        vec_env2 = DummyVecEnv([lambda: env2])
        
        # Load VecNormalize from checkpoint
        vec_env2 = VecNormalize.load(vecnorm_path, vec_env2)
        vec_env2.training = True
        vec_env2.norm_reward = False
        
        # Load model from checkpoint
        model2 = PPO.load(model_path, env=vec_env2, device="cpu")
        
        # Verify step count is preserved
        if model2.num_timesteps != initial_steps:
            logger.error(f"❌ Step count mismatch: {model2.num_timesteps} != {initial_steps}")
            return False
        logger.info(f"✅ Step count preserved: {model2.num_timesteps}")
        
        # Continue training
        model2.learn(total_timesteps=500, reset_num_timesteps=False)
        final_steps = model2.num_timesteps
        
        # PPO collects steps in batches of n_steps (128), so final_steps may be slightly more than expected
        # We just need to verify that it's greater than initial_steps (i.e., training continued)
        min_expected_steps = initial_steps + 400  # Allow some tolerance for batch collection
        if final_steps <= initial_steps:
            logger.error(f"❌ Training did not continue: {final_steps} <= {initial_steps}")
            return False
        if final_steps < min_expected_steps:
            logger.error(f"❌ Final step count too low: {final_steps} < {min_expected_steps}")
            return False
        logger.info(f"✅ Training resumed correctly: {initial_steps} → {final_steps} (collected {final_steps - initial_steps} steps)")
        
        # Save second checkpoint
        temp_model_path2 = ckpt_dir / "temp_model2.zip"
        temp_vecnorm_path2 = ckpt_dir / "temp_vecnorm2.pkl"
        model2.save(str(temp_model_path2))
        vec_env2.save(str(temp_vecnorm_path2))
        
        ckpt_id2, success = mgr.save_checkpoint(
            str(temp_model_path2),
            str(temp_vecnorm_path2),
            worker_state,
            step=final_steps,
            metrics={"mean_reward": 0.6, "sharpe": 1.3},
        )
        
        if not success:
            logger.error("❌ Failed to save second checkpoint")
            return False
        logger.info(f"✅ Second checkpoint saved: {ckpt_id2}")
        
        # Verify cleanup (should keep only 3 checkpoints)
        ckpts = mgr.list_checkpoints()
        logger.info(f"✅ Checkpoints after cleanup: {len(ckpts)} (max={mgr.max_checkpoints})")
        
        # Get latest checkpoint
        latest = mgr.get_latest_checkpoint()
        logger.info(f"✅ Latest checkpoint: {latest}")
        
        logger.info("=" * 80)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("=" * 80)
        return True


if __name__ == "__main__":
    try:
        success = test_checkpoint_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}", exc_info=True)
        sys.exit(1)
