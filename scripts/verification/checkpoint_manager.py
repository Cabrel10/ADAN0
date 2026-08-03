"""Checkpoint Manager — Robust model + scaler persistence.

Handles:
  1. Atomic checkpoint saves (write to temp, then rename)
  2. Checkpoint integrity verification (checksums)
  3. Automatic cleanup of old checkpoints
  4. Resume detection and validation
  5. Scaler consistency across train/test
"""

import hashlib
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model + VecNormalize checkpoint lifecycle."""

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """
        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Max number of checkpoints to keep (older ones deleted)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.metadata_file = self.checkpoint_dir / "checkpoint_manifest.json"

    def _compute_checksum(self, filepath: str) -> str:
        """Compute SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _load_manifest(self) -> Dict:
        """Load checkpoint manifest (list of saved checkpoints + metadata)."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load manifest: {e}. Starting fresh.")
        return {"checkpoints": [], "last_saved": None}

    def _save_manifest(self, manifest: Dict):
        """Save checkpoint manifest atomically."""
        temp_file = self.metadata_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w") as f:
                json.dump(manifest, f, indent=2)
            temp_file.replace(self.metadata_file)
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def save_checkpoint(
        self,
        model_path: str,
        vecnorm_path: str,
        worker_state: Dict,
        step: int,
        metrics: Optional[Dict] = None,
    ) -> Tuple[str, bool]:
        """
        Save model + VecNormalize + metadata atomically.

        Args:
            model_path: Path to model.zip
            vecnorm_path: Path to vecnormalize.pkl
            worker_state: Dict with training state (lr, ent_coef, gamma, etc.)
            step: Current training step
            metrics: Optional dict with training metrics

        Returns:
            (checkpoint_id, success)
        """
        checkpoint_id = f"ckpt_{step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_subdir = self.checkpoint_dir / checkpoint_id
        checkpoint_subdir.mkdir(parents=True, exist_ok=True)

        try:
            # Copy files atomically (write to temp, then rename)
            temp_model = checkpoint_subdir / "model.zip.tmp"
            temp_vecnorm = checkpoint_subdir / "vecnormalize.pkl.tmp"

            shutil.copy2(model_path, temp_model)
            shutil.copy2(vecnorm_path, temp_vecnorm)

            # Rename to final names (atomic on POSIX)
            temp_model.replace(checkpoint_subdir / "model.zip")
            temp_vecnorm.replace(checkpoint_subdir / "vecnormalize.pkl")

            # Compute checksums
            model_checksum = self._compute_checksum(str(checkpoint_subdir / "model.zip"))
            vecnorm_checksum = self._compute_checksum(str(checkpoint_subdir / "vecnormalize.pkl"))

            # Save metadata
            metadata = {
                "checkpoint_id": checkpoint_id,
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "worker_state": worker_state,
                "metrics": metrics or {},
                "checksums": {
                    "model": model_checksum,
                    "vecnorm": vecnorm_checksum,
                },
                "files": {
                    "model": "model.zip",
                    "vecnorm": "vecnormalize.pkl",
                    "metadata": "checkpoint.json",
                },
            }
            with open(checkpoint_subdir / "checkpoint.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Update manifest
            manifest = self._load_manifest()
            manifest["checkpoints"].append({
                "id": checkpoint_id,
                "step": step,
                "timestamp": metadata["timestamp"],
                "metrics": metrics or {},
            })
            manifest["last_saved"] = checkpoint_id
            self._save_manifest(manifest)

            logger.info(f"✅ Checkpoint saved: {checkpoint_id} (step {step})")

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()

            return checkpoint_id, True

        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}", exc_info=True)
            # Cleanup partial checkpoint
            if checkpoint_subdir.exists():
                shutil.rmtree(checkpoint_subdir)
            return "", False

    def load_checkpoint(self, checkpoint_id: str) -> Tuple[str, str, Dict, bool]:
        """
        Load model + VecNormalize + metadata from checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to load (e.g., "ckpt_50000_20260531_120000")

        Returns:
            (model_path, vecnorm_path, worker_state, success)
        """
        checkpoint_subdir = self.checkpoint_dir / checkpoint_id
        if not checkpoint_subdir.exists():
            logger.error(f"❌ Checkpoint not found: {checkpoint_id}")
            return "", "", {}, False

        try:
            model_path = checkpoint_subdir / "model.zip"
            vecnorm_path = checkpoint_subdir / "vecnormalize.pkl"
            metadata_path = checkpoint_subdir / "checkpoint.json"

            # Verify files exist
            if not model_path.exists() or not vecnorm_path.exists():
                logger.error(f"❌ Checkpoint files missing: {checkpoint_id}")
                return "", "", {}, False

            # Load metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Verify checksums
            model_checksum = self._compute_checksum(str(model_path))
            vecnorm_checksum = self._compute_checksum(str(vecnorm_path))

            if model_checksum != metadata["checksums"]["model"]:
                logger.error(f"❌ Model checksum mismatch: {checkpoint_id}")
                return "", "", {}, False

            if vecnorm_checksum != metadata["checksums"]["vecnorm"]:
                logger.error(f"❌ VecNormalize checksum mismatch: {checkpoint_id}")
                return "", "", {}, False

            logger.info(f"✅ Checkpoint loaded: {checkpoint_id} (step {metadata['step']})")
            return str(model_path), str(vecnorm_path), metadata.get("worker_state", {}), True

        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}", exc_info=True)
            return "", "", {}, False

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get ID of most recent checkpoint."""
        manifest = self._load_manifest()
        if manifest["last_saved"]:
            return manifest["last_saved"]
        return None

    def list_checkpoints(self) -> list:
        """List all available checkpoints (newest first)."""
        manifest = self._load_manifest()
        return sorted(
            manifest["checkpoints"],
            key=lambda x: x["step"],
            reverse=True
        )

    def _cleanup_old_checkpoints(self):
        """Delete old checkpoints, keeping only max_checkpoints."""
        manifest = self._load_manifest()
        if len(manifest["checkpoints"]) <= self.max_checkpoints:
            return

        # Sort by step (descending) and keep only the newest
        sorted_ckpts = sorted(
            manifest["checkpoints"],
            key=lambda x: x["step"],
            reverse=True
        )
        to_delete = sorted_ckpts[self.max_checkpoints:]

        for ckpt in to_delete:
            ckpt_dir = self.checkpoint_dir / ckpt["id"]
            try:
                shutil.rmtree(ckpt_dir)
                logger.info(f"🗑️  Deleted old checkpoint: {ckpt['id']}")
            except Exception as e:
                logger.warning(f"Could not delete {ckpt['id']}: {e}")

        # Update manifest
        manifest["checkpoints"] = sorted_ckpts[:self.max_checkpoints]
        self._save_manifest(manifest)

    def verify_checkpoint_integrity(self, checkpoint_id: str) -> bool:
        """Verify checkpoint is not corrupted."""
        _, _, _, success = self.load_checkpoint(checkpoint_id)
        return success


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    mgr = CheckpointManager("/tmp/test_ckpts", max_checkpoints=3)
    
    # Simulate save
    print("Testing checkpoint manager...")
    print(f"Latest: {mgr.get_latest_checkpoint()}")
    print(f"All: {mgr.list_checkpoints()}")
