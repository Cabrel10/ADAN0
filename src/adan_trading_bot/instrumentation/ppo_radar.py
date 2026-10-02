"""PPO explained-variance radar (ADAN0 chantier).

SB3 callback that writes ONE CSV row per PPO update, explicitly linked to the
rollout window that produced it::

    update_id, rollout_start_step, rollout_end_step, timesteps, n_updates,
    explained_variance, approx_kl, clip_fraction, value_loss,
    policy_gradient_loss, entropy_loss, std, learning_rate, loss

Mechanism: SB3's ``PPO.train()`` records ``train/*`` scalars into the model
logger at every update and ``train/n_updates`` grows monotonically. The
callback runs on every env step; when ``n_updates`` changes, the previous
rollout window ``[rollout_start_step, rollout_end_step]`` is closed and the
metrics of the update trained on THAT window are emitted. The linkage is
explicit (window columns), never assumed from a bare ``timesteps`` column.

Telemetry must never alter training: every step is wrapped in a broad
except and always returns True.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, Optional

from stable_baselines3.common.callbacks import BaseCallback

RADAR_FIELDS = (
    "update_id",
    "rollout_start_step",
    "rollout_end_step",
    "timesteps",
    "n_updates",
    "explained_variance",
    "approx_kl",
    "clip_fraction",
    "value_loss",
    "policy_gradient_loss",
    "entropy_loss",
    "std",
    "learning_rate",
    "loss",
)

# CSV column -> SB3 logger key (recorded by PPO.train()).
_KEY_MAP = {
    "n_updates": "train/n_updates",
    "explained_variance": "train/explained_variance",
    "approx_kl": "train/approx_kl",
    "clip_fraction": "train/clip_fraction",
    "value_loss": "train/value_loss",
    "policy_gradient_loss": "train/policy_gradient_loss",
    "entropy_loss": "train/entropy_loss",
    "std": "train/std",
    "learning_rate": "train/learning_rate",
    "loss": "train/loss",
}


class PPORadarCallback(BaseCallback):
    """Emit one CSV row per PPO update with its rollout window."""

    def __init__(self, csv_path: str = "logs/ppo_radar/ppo_radar.csv", verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self._last_n_updates: int = -1
        self._rollout_start: int = 0
        self._rows_written: int = 0

    def _on_training_start(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerow(RADAR_FIELDS)
        except Exception:
            pass

    def _logger_values(self) -> Dict[str, Any]:
        try:
            return dict(getattr(self.model.logger, "name_to_value", {}) or {})
        except Exception:
            return {}

    def _on_step(self) -> bool:
        try:
            values = self._logger_values()
            raw_updates = values.get("train/n_updates")
            if raw_updates is None:
                return True
            n_updates = int(raw_updates)
            if n_updates == self._last_n_updates:
                return True
            if "train/approx_kl" not in values:
                # Logger was reset or PPO has not trained yet.
                self._last_n_updates = n_updates
                return True

            row: Dict[str, Any] = {
                "update_id": n_updates,
                "rollout_start_step": self._rollout_start,
                "rollout_end_step": int(self.num_timesteps),
                "timesteps": int(self.num_timesteps),
            }
            for column, logger_key in _KEY_MAP.items():
                row[column] = values.get(logger_key)

            with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow([row.get(col) for col in RADAR_FIELDS])
            self._rows_written += 1
            self._last_n_updates = n_updates
            self._rollout_start = int(self.num_timesteps)
        except Exception:
            pass
        return True

    def _on_training_end(self) -> None:
        # Flush the final update if training ended between steps.
        try:
            values = self._logger_values()
            n_updates = int(values.get("train/n_updates", -1))
            if n_updates > self._last_n_updates and "train/approx_kl" in values:
                row = {
                    "update_id": n_updates,
                    "rollout_start_step": self._rollout_start,
                    "rollout_end_step": int(self.num_timesteps),
                    "timesteps": int(self.num_timesteps),
                }
                for column, logger_key in _KEY_MAP.items():
                    row[column] = values.get(logger_key)
                with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerow([row.get(col) for col in RADAR_FIELDS])
                self._last_n_updates = n_updates
        except Exception:
            pass
