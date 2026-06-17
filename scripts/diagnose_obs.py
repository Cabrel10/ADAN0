
import os
import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Add src to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from stable_baselines3 import PPO
from adan_trading_bot.trading.live_state_builder import LiveStateBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diagnose")

def diagnose():
    checkpoint_path = str(REPO_ROOT / "checkpoints" / "ppo_adan0_sandbox_500224steps.zip")
    print(f"Loading model: {checkpoint_path}")
    model = PPO.load(checkpoint_path, device="cpu")
    
    # 1. Load Parquet data for fitting
    print("\n--- Loading Parquet data for fitting ---")
    data_dict = {}
    val_dir = REPO_ROOT / "data" / "processed" / "indicators" / "val" / "BTCUSDT"
    for tf in ["5m", "1h", "4h"]:
        path = val_dir / f"{tf}.parquet"
        if path.exists():
            data_dict[tf] = pd.read_parquet(path)
            print(f"Loaded {tf}: {len(data_dict[tf])} rows")
        else:
            print(f"Missing {tf} parquet!")
            
    # 2. Init LiveStateBuilder and fit on Parquet
    lsb = LiveStateBuilder(exchange_id="binance", symbol="BTC/USDT")
    print("\nFitting internal StateBuilder on Parquet data...")
    lsb.state_builder.fit_scalers({"BTCUSDT": data_dict})
    
    # CRITICAL: Prevent refitting on live data
    lsb.state_builder.scalers_loaded_from_training = True
    print("Scalers locked.")
    
    # 3. Get live obs
    print("\n--- LiveStateBuilder (Fitted & Locked on Parquet) ---")
    portfolio_state = np.zeros(20, dtype=np.float32)
    portfolio_state[0] = 20.5
    portfolio_state[1] = 20.5
    
    obs = lsb.build_observation(portfolio_state=portfolio_state)
    
    for tf in ["5m", "1h", "4h"]:
        data = obs[tf]
        print(f"{tf} obs: shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")
    
    # 4. Predict
    obs_batch = {k: np.expand_dims(v, 0) for k, v in obs.items()}
    action, _ = model.predict(obs_batch, deterministic=True)
    print(f"\nAction: {action.flatten()}")
    
    direction = action.flatten()[0]
    print(f"Direction: {direction:.4f}")

if __name__ == "__main__":
    diagnose()
