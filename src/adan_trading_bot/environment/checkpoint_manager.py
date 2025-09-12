"""
Checkpoint and Recovery System for ADAN Trading Bot.

This module provides functionality for saving and loading training checkpoints
with automatic recovery capabilities.
"""
import os
import json
import time
import logging
import tempfile
import shutil
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, TypeVar, Generic, Type
import torch
import numpy as np

from adan_trading_bot.environment.error_handling import handle_errors

T = TypeVar('T')

@dataclass
class CheckpointMetadata:
    """Metadata for a training checkpoint."""
    timestamp: float = field(default_factory=time.time)
    episode: int = 0
    total_steps: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    custom_data: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create metadata from dictionary."""
        return cls(**data)

class CheckpointManager(Generic[T]):
    """
    Manages saving and loading of model checkpoints with automatic recovery.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        checkpoint_interval: int = 1000,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the CheckpointManager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            checkpoint_interval: Save checkpoint every N steps
            logger: Logger instance (creates one if None)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max(max_checkpoints, 1)
        self.checkpoint_interval = max(checkpoint_interval, 1)
        self.logger = logger or logging.getLogger(__name__)

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track the latest checkpoint
        self.latest_checkpoint: Optional[str] = None
        self.latest_metadata: Optional[CheckpointMetadata] = None

    @handle_errors()
    def save_checkpoint(
        self,
        model: T,
        optimizer: Optional[torch.optim.Optimizer] = None,
        episode: int = 0,
        total_steps: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        custom_data: Optional[Dict[str, Any]] = None,
        is_final: bool = False
    ) -> Optional[str]:
        """
        Save a checkpoint of the current training state.

        Args:
            model: Model to save
            optimizer: Optimizer state to save
            episode: Current episode number
            total_steps: Total number of steps taken
            metrics: Dictionary of metrics to save
            custom_data: Additional custom data to save
            is_final: If True, always save regardless of interval

        Returns:
            Path to the saved checkpoint, or None if not saved
        """
        # Only save at specified intervals unless it's a final save
        if not is_final and total_steps % self.checkpoint_interval != 0:
            return None

        # Create metadata
        metadata = CheckpointMetadata(
            episode=episode,
            total_steps=total_steps,
            metrics=metrics or {},
            custom_data=custom_data or {}
        )

        # Create temporary directory for atomic save
        with tempfile.TemporaryDirectory(prefix='checkpoint_') as temp_dir:
            temp_path = Path(temp_dir)

            # Save model
            model_path = temp_path / 'model.pt'
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), model_path)

            # Save optimizer state if provided
            optimizer_path = temp_path / 'optimizer.pt'
            if optimizer is not None:
                torch.save(optimizer.state_dict(), optimizer_path)

            # Save metadata
            metadata_path = temp_path / 'metadata.json'
            def _safe_default(o: Any):
                """Best-effort JSON serializer for common non-serializable objects."""
                # numpy types
                if isinstance(o, (np.integer, )):
                    return int(o)
                if isinstance(o, (np.floating, )):
                    return float(o)
                if isinstance(o, (np.ndarray, )):
                    return o.tolist()
                # pathlib Paths
                if isinstance(o, Path):
                    return str(o)
                # datetimes
                if isinstance(o, datetime):
                    return o.isoformat()
                # types
                if isinstance(o, type):
                    return o.__name__
                # objects with __dict__
                if hasattr(o, '__dict__'):
                    try:
                        return {k: v for k, v in vars(o).items()}
                    except Exception:
                        return str(o)
                # fallback
                return str(o)

            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, default=_safe_default)

            # Create checkpoint directory with timestamp
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            checkpoint_name = f"checkpoint_{timestamp}_ep{episode:06d}_step{total_steps:010d}"
            if is_final:
                checkpoint_name += "_final"

            dest_dir = self.checkpoint_dir / checkpoint_name

            # Ensure base directory still exists (in case it was removed externally)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            # Move temporary directory to final location
            shutil.move(temp_dir, dest_dir)

            # Update latest checkpoint
            self.latest_checkpoint = str(dest_dir)
            self.latest_metadata = metadata

            # Clean up old checkpoints
            self._cleanup_old_checkpoints()

            self.logger.info(
                f"Checkpoint saved to {dest_dir} "
                f"(episode={episode}, steps={total_steps})"
            )

            return str(dest_dir)

    @handle_errors()
    def load_latest_checkpoint(
        self,
        model: Optional[T] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[str] = None
    ) -> Tuple[Optional[T], Optional[torch.optim.Optimizer], Optional[CheckpointMetadata]]:
        """
        Load the most recent checkpoint.

        Args:
            model: Model to load state into (if None, only metadata is loaded)
            optimizer: Optimizer to load state into (if None, only model state is loaded)
            map_location: Device to map the storage to (for PyTorch loading)

        Returns:
            Tuple of (model, optimizer, metadata) with loaded states
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            self.logger.warning("No checkpoints found")
            return model, optimizer, None

        latest_checkpoint = checkpoints[-1]  # Sorted by timestamp
        return self.load_checkpoint(latest_checkpoint, model, optimizer, map_location)

    @handle_errors()
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: Optional[T] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[str] = None
    ) -> Tuple[Optional[T], Optional[torch.optim.Optimizer], Optional[CheckpointMetadata]]:
        """
        Load a specific checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint directory
            model: Model to load state into (if None, only metadata is loaded)
            optimizer: Optimizer to load state into (if None, only model state is loaded)
            map_location: Device to map the storage to (for PyTorch loading)

        Returns:
            Tuple of (model, optimizer, metadata) with loaded states
        """
        checkpoint_path = Path(checkpoint_path)

        # Load metadata
        metadata_path = checkpoint_path / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = CheckpointMetadata.from_dict(json.load(f))

        # Load model state if model is provided
        if model is not None:
            model_path = checkpoint_path / 'model.pt'
            if model_path.exists():
                if hasattr(model, 'load_state_dict'):
                    state_dict = torch.load(model_path, map_location=map_location)
                    model.load_state_dict(state_dict)
                else:
                    self.logger.warning("Model does not support state_dict loading")

        # Load optimizer state if optimizer is provided
        if optimizer is not None:
            optimizer_path = checkpoint_path / 'optimizer.pt'
            if optimizer_path.exists():
                optimizer_state = torch.load(optimizer_path, map_location=map_location)
                optimizer.load_state_dict(optimizer_state)

        self.latest_checkpoint = str(checkpoint_path)
        self.latest_metadata = metadata

        self.logger.info(
            f"Loaded checkpoint from {checkpoint_path} "
            f"(episode={metadata.episode}, steps={metadata.total_steps})"
        )

        return model, optimizer, metadata

    def list_checkpoints(self) -> list[str]:
        """
        List all available checkpoints, sorted by episode and step number.

        Returns:
            List of checkpoint directory paths, sorted by episode and step number
        """
        if not self.checkpoint_dir.exists():
            return []

        def get_episode_step(checkpoint_path):
            """Extract episode and step numbers from checkpoint path."""
            import re
            match = re.search(r'ep(\d+)_step(\d+)', str(checkpoint_path))
            if match:
                return int(match.group(1)), int(match.group(2))
            return (0, 0)  # Default to (0, 0) if pattern not found

        checkpoints = [
            str(d) for d in self.checkpoint_dir.glob('checkpoint_*')
            if d.is_dir()
        ]

        # Sort by episode and step numbers
        checkpoints.sort(key=get_episode_step)

        return checkpoints

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to respect max_checkpoints limit."""
        checkpoints = self.list_checkpoints()

        while len(checkpoints) > self.max_checkpoints:
            oldest_checkpoint = checkpoints.pop(0)
            try:
                shutil.rmtree(oldest_checkpoint)
                self.logger.debug(f"Removed old checkpoint: {oldest_checkpoint}")
            except Exception as e:
                self.logger.error(f"Failed to remove checkpoint {oldest_checkpoint}: {e}")

    def get_latest_metadata(self) -> Optional[CheckpointMetadata]:
        """Get metadata from the latest checkpoint."""
        if self.latest_metadata is None and self.latest_checkpoint:
            try:
                metadata_path = Path(self.latest_checkpoint) / 'metadata.json'
                with open(metadata_path, 'r') as f:
                    self.latest_metadata = CheckpointMetadata.from_dict(json.load(f))
            except Exception as e:
                self.logger.error(f"Failed to load metadata: {e}")
        return self.latest_metadata

# Example usage:
if __name__ == "__main__":
    import torch.nn as nn

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Example model and optimizer
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    )
    optimizer = torch.optim.Adam(model.parameters())

    # Initialize checkpoint manager
    checkpoint_dir = "./checkpoints"
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=3,
        checkpoint_interval=1000,
        logger=logger
    )

    # Save a checkpoint
    metrics = {"reward": 1.23, "loss": 0.456}
    checkpoint_path = checkpoint_manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        episode=1,
        total_steps=1000,
        metrics=metrics
    )

    # Load the latest checkpoint
    loaded_model, loaded_optimizer, metadata = checkpoint_manager.load_latest_checkpoint()

    if metadata:
        print(f"Loaded checkpoint from episode {metadata.episode}, "
              f"steps: {metadata.total_steps}, "
              f"metrics: {metadata.metrics}")
