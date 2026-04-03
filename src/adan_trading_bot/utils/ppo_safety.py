import logging
from typing import Optional

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


def clamp_policy_log_std(model, min_log_std: float = -5.0, max_log_std: float = 2.0) -> None:
    """Clamp les paramètres log_std de la policy PPO pour éviter l'explosion numérique.

    Cette fonction est sûre à appeler même si la policy ne possède pas d'attribut log_std
    explicite : dans ce cas elle parcourt tous les paramètres et ne clamp que ceux qui
    contiennent "log_std" dans leur nom.
    """
    try:
        # Cas le plus courant dans SB3: attribute direct
        if hasattr(model, "policy") and hasattr(model.policy, "log_std"):
            with torch.no_grad():
                model.policy.log_std.data.clamp_(min=min_log_std, max=max_log_std)

        # Clamp de secours: tous les paramètres dont le nom contient "log_std"
        if hasattr(model, "policy") and hasattr(model.policy, "named_parameters"):
            with torch.no_grad():
                for name, param in model.policy.named_parameters():
                    if "log_std" in name and param.data.is_floating_point():
                        param.data.clamp_(min=min_log_std, max=max_log_std)
    except Exception as e:
        logger.warning(f"ppo_safety.clamp_policy_log_std failed: {e}")


def get_policy_std(model) -> Optional[float]:
    """Retourne un estimateur du std moyen de la policy si disponible."""
    try:
        if hasattr(model, "policy") and hasattr(model.policy, "log_std"):
            log_std = model.policy.log_std.detach().cpu().numpy()
            std = np.exp(log_std)
            return float(np.mean(std))
    except Exception:
        return None
    return None


class PpoStdSafetyCallback(BaseCallback):
    """Callback SB3 qui surveille et clampe log_std à chaque rollout.

    - Clamp systématique de log_std dans [min_log_std, max_log_std]
    - Log d'un WARNING si le std moyen dépasse std_warn_threshold
    """

    def __init__(
        self,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
        std_warn_threshold: float = 100.0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.std_warn_threshold = std_warn_threshold

    def _on_training_start(self) -> None:
        clamp_policy_log_std(self.model, self.min_log_std, self.max_log_std)
        std = get_policy_std(self.model)
        if std is not None and self.verbose > 0:
            logger.info(f"[PpoStdSafetyCallback] Initial policy std={std:.4f}")

    def _on_rollout_end(self) -> None:
        clamp_policy_log_std(self.model, self.min_log_std, self.max_log_std)
        std = get_policy_std(self.model)
        if std is not None and std > self.std_warn_threshold:
            logger.warning(
                f"[PpoStdSafetyCallback] Policy std très élevé détecté: std={std:.4f}"
            )

    def _on_step(self) -> bool:
        # Rien de plus à chaque pas; tout est fait en fin de rollout
        return True
