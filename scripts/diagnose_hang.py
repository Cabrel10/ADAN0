#!/usr/bin/env python3
"""
Diagnostic script to find the infinite loop causing Ray GCS timeout.
Runs the environment in isolation (no Ray) and monitors for hangs.
"""

import sys
import os
import signal
import time
import traceback
from threading import Thread

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def timeout_handler(signum, frame):
    print("\n⏱️  TIMEOUT! The environment hung for 30 seconds.")
    print("Stack trace at timeout:")
    traceback.print_stack(frame)
    sys.exit(1)

def run_env_test():
    """Run environment test with timeout protection."""
    from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
    from adan_trading_bot.common.config_loader import load_config
    import numpy as np
    
    print("Loading config...")
    config = load_config()
    
    print("Creating environment...")
    env = MultiAssetChunkedEnv(config=config, worker_id=0)
    
    print("Resetting environment...")
    obs, info = env.reset()
    print(f"✅ Reset successful. Observation keys: {obs.keys()}")
    
    print("\nRunning 1000 steps with timeout protection...")
    for step in range(1000):
        # Set timeout for each step
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout per step
        
        try:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Cancel timeout if step completed
            signal.alarm(0)
            
            if step % 100 == 0:
                print(f"✅ Step {step}: reward={reward:.4f}, done={terminated or truncated}")
            
            if step == 915:
                print(f"\n⚠️  Reached step 915 (where hang occurs in training)")
                print(f"   Current step_in_chunk: {env.step_in_chunk}")
                print(f"   Current chunk: {env.current_chunk_idx}")
                print(f"   Portfolio value: {env.portfolio_manager.get_portfolio_value():.2f}")
            
            if terminated or truncated:
                print(f"Episode ended at step {step}")
                break
                
        except Exception as e:
            signal.alarm(0)
            print(f"\n❌ Error at step {step}: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    print("=" * 60)
    print("ADAN Environment Hang Diagnostic")
    print("=" * 60)
    print("\nThis script runs the environment in isolation to find the hang.")
    print("If it hangs, the timeout will trigger and show the stack trace.\n")
    
    try:
        run_env_test()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(0)
