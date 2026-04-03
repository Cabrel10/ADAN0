#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SeedManager: Centralized seed management for reproducibility.

This module ensures all random number generators (RNGs) across the codebase
are initialized with the same seed for reproducible experiments.
"""

import random
import logging
from typing import Dict, Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SeedManager:
    """
    Centralized manager for setting and retrieving RNG states.
    
    Ensures reproducibility across:
    - Python's random module
    - NumPy's random number generation
    - PyTorch's random number generation
    - Gym/Gymnasium environments
    """
    
    _seed: Optional[int] = None
    _initialized: bool = False
    
    @classmethod
    def initialize(cls, seed: int = 42) -> None:
        """
        Initialize all random number generators with the given seed.
        
        Args:
            seed: The seed value (default: 42)
        """
        cls._seed = seed
        
        # Python's built-in random
        random.seed(seed)
        logger.info(f"✅ Set Python random seed: {seed}")
        
        # NumPy
        np.random.seed(seed)
        logger.info(f"✅ Set NumPy random seed: {seed}")
        
        # PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For deterministic behavior on GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info(f"✅ Set PyTorch CUDA seed: {seed}")
        logger.info(f"✅ Set PyTorch seed: {seed}")
        
        cls._initialized = True
        logger.info(f"🎲 SeedManager initialized with seed={seed}")
    
    @classmethod
    def get_rng_states(cls) -> Dict[str, Any]:
        """
        Get current RNG states for checkpointing.
        
        Returns:
            Dictionary containing all RNG states
        """
        if not cls._initialized:
            logger.warning("SeedManager not initialized. Returning empty states.")
            return {}
        
        states = {
            'seed': cls._seed,
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
        }
        
        if torch.cuda.is_available():
            states['torch_cuda_random_state'] = torch.cuda.get_rng_state()
        
        logger.debug("Retrieved RNG states for checkpointing")
        return states
    
    @classmethod
    def set_rng_states(cls, states: Dict[str, Any]) -> None:
        """
        Restore RNG states from checkpoint.
        
        Args:
            states: Dictionary containing RNG states
        """
        if not states:
            logger.warning("Empty states provided. Skipping RNG state restoration.")
            return
        
        try:
            # Restore seed value
            if 'seed' in states:
                cls._seed = states['seed']
            
            # Restore Python random state
            if 'python_random_state' in states:
                random.setstate(states['python_random_state'])
                logger.debug("Restored Python random state")
            
            # Restore NumPy random state
            if 'numpy_random_state' in states:
                np.random.set_state(states['numpy_random_state'])
                logger.debug("Restored NumPy random state")
            
            # Restore PyTorch random state
            if 'torch_random_state' in states:
                torch.set_rng_state(states['torch_random_state'])
                logger.debug("Restored PyTorch random state")
            
            # Restore PyTorch CUDA random state
            if torch.cuda.is_available() and 'torch_cuda_random_state' in states:
                torch.cuda.set_rng_state(states['torch_cuda_random_state'])
                logger.debug("Restored PyTorch CUDA random state")
            
            cls._initialized = True
            logger.info(f"✅ RNG states restored from checkpoint (seed={cls._seed})")
            
        except Exception as e:
            logger.error(f"Failed to restore RNG states: {e}", exc_info=True)
            raise
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if SeedManager has been initialized."""
        return cls._initialized
    
    @classmethod
    def get_seed(cls) -> Optional[int]:
        """Get the current seed value."""
        return cls._seed


# Convenience function for quick initialization
def set_seed(seed: int = 42) -> None:
    """
    Quick initialization function.
    
    Args:
        seed: The seed value (default: 42)
    """
    SeedManager.initialize(seed)
