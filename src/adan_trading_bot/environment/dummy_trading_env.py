#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dummy trading environment for VecNormalize loading in production.

This environment is used ONLY to load VecNormalize statistics in paper_trading_monitor.py.
It does NOT execute any trading logic - it's a minimal wrapper that provides the correct
observation_space and action_space structure.

CRITICAL: The observation_space MUST match MultiAssetChunkedEnv EXACTLY.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TradingEnvDummy(gym.Env):
    """
    Minimal dummy environment for VecNormalize loading.
    
    This environment provides the correct observation_space and action_space
    without any trading logic. It's used to load VecNormalize statistics
    that were saved during training.
    
    CRITICAL: observation_space must match MultiAssetChunkedEnv exactly.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, config=None):
        """
        Initialize the dummy environment.
        
        Args:
            config: Optional configuration dict (ignored, for compatibility)
        """
        super().__init__()
        
        # Define observation_space to match MultiAssetChunkedEnv
        # Based on multi_asset_chunked_env.py lines 1710-1743
        self.observation_space = spaces.Dict({
            '5m': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(20, 14),  # window_size=20, n_features=14
                dtype=np.float32,
            ),
            '1h': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(20, 14),  # window_size=20, n_features=14
                dtype=np.float32,
            ),
            '4h': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(20, 14),  # window_size=20, n_features=14
                dtype=np.float32,
            ),
            'portfolio_state': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(20,),     # DEFAULT_PORTFOLIO_STATE_SIZE = 20
                dtype=np.float32,
            ),
        })
        
        # Define action_space to match MultiAssetChunkedEnv
        # Based on multi_asset_chunked_env.py lines 1694-1701
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(25,),  # 5 assets × 5 dimensions
            dtype=np.float32,
        )
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment (dummy implementation).
        
        Returns:
            observation: Dict with zero-filled arrays matching observation_space
            info: Empty dict
        """
        super().reset(seed=seed)
        
        # Return dummy observation with correct structure
        observation = {
            '5m': np.zeros((20, 14), dtype=np.float32),
            '1h': np.zeros((20, 14), dtype=np.float32),
            '4h': np.zeros((20, 14), dtype=np.float32),
            'portfolio_state': np.zeros(20, dtype=np.float32),
        }
        
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Step function (not implemented - this is a dummy environment).
        
        Raises:
            NotImplementedError: This environment is only for VecNormalize loading.
        """
        raise NotImplementedError(
            "TradingEnvDummy is only for VecNormalize loading. "
            "Do not call step() on this environment."
        )
    
    def render(self):
        """Render function (not implemented)."""
        pass
    
    def close(self):
        """Close the environment."""
        pass
