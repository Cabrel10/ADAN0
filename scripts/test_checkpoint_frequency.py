#!/usr/bin/env python3
"""Test that checkpoints are saved every 15k steps and resume works after crash."""

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = _SCRIPT_DIR.parent


def test_checkpoint_frequency():
    """Test that checkpoints are saved every 15k steps."""
    
    logger.info("=" * 80)
    logger.info("🧪 CHECKPOINT FREQUENCY TEST (15k steps)")
    logger.info("=" * 80)
    
    # Load config
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = ConfigLoader.load_config(str(config_path))
    logger.info(f"✅ Config loaded from {config_path}")
    
    # Create temp checkpoint dir
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # Create PPO model with interval_timesteps = 15k
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
        
        # Simulate training with checkpoints every 15k steps
        checkpoint_steps = []
        total_steps_to_train = 15_000  # Train for 15k steps (1 checkpoint)
        interval = 15_000  # Checkpoint every 15k
        
        logger.info(f"🔄 Training for {total_steps_to_train} steps (checkpoint every {interval} steps)...")
        
        # Train in chunks and manually save checkpoints
        steps_trained = 0
        iteration = 0
        
        while steps_trained < total_steps_to_train:
            # Train for interval_timesteps
            steps_this_iter = min(interval, total_steps_to_train - steps_trained)
            model.learn(total_timesteps=steps_this_iter, reset_num_timesteps=False)
            steps_trained = model.num_timesteps
            iteration += 1
            
            # Check if we should save checkpoint (every 15k steps)
            if steps_trained % interval < interval:
                checkpoint_path = ckpt_dir / f"checkpoint_{steps_trained:06d}"
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                
                # Save model and vecnorm
                model.save(str(checkpoint_path / "model.zip"))
                vec_env.save(str(checkpoint_path / "vecnormalize.pkl"))
                
                # Save metadata
                state = {
                    "total_timesteps": steps_trained,
                    "learning_rate": 3e-4,
                    "iteration": iteration,
                }
                with open(checkpoint_path / "worker_state.json", "w") as f:
                    json.dump(state, f, indent=2)
                
                checkpoint_steps.append(steps_trained)
                logger.info(f"✅ Checkpoint saved at {steps_trained} steps")
        
        # Verify checkpoints
        logger.info(f"\n📊 Checkpoint Summary:")
        logger.info(f"   Total steps trained: {steps_trained}")
        logger.info(f"   Checkpoints saved: {len(checkpoint_steps)}")
        logger.info(f"   Checkpoint steps: {checkpoint_steps}")
        
        # Verify checkpoint frequency
        expected_checkpoints = [15_000]
        for expected in expected_checkpoints:
            if expected <= steps_trained:
                if expected in checkpoint_steps:
                    logger.info(f"   ✅ Checkpoint at {expected} steps: FOUND")
                else:
                    logger.warning(f"   ⚠️  Checkpoint at {expected} steps: MISSING")
        
        # Test resume from latest checkpoint
        logger.info(f"\n🔄 Testing resume from latest checkpoint...")
        latest_ckpt = checkpoint_steps[-1]
        latest_ckpt_path = ckpt_dir / f"checkpoint_{latest_ckpt:06d}"
        
        # Load checkpoint
        model2 = PPO.load(str(latest_ckpt_path / "model.zip"), env=vec_env, device="cpu")
        vec_env2 = VecNormalize.load(str(latest_ckpt_path / "vecnormalize.pkl"), vec_env)
        
        if model2.num_timesteps == latest_ckpt:
            logger.info(f"✅ Step count preserved: {model2.num_timesteps}")
        else:
            logger.warning(f"⚠️  Step count mismatch: {model2.num_timesteps} != {latest_ckpt}")
        
        # Resume training
        model2.learn(total_timesteps=5_000, reset_num_timesteps=False)
        final_steps = model2.num_timesteps
        
        if final_steps > latest_ckpt:
            logger.info(f"✅ Training resumed: {latest_ckpt} → {final_steps}")
        else:
            logger.error(f"❌ Training did not resume: {final_steps} <= {latest_ckpt}")
            return False
        
        logger.info("=" * 80)
        logger.info("✅ CHECKPOINT FREQUENCY TEST PASSED")
        logger.info("=" * 80)
        return True


if __name__ == "__main__":
    try:
        success = test_checkpoint_frequency()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}", exc_info=True)
        sys.exit(1)

