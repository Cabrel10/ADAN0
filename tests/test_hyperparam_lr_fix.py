"""
Test forensic V32 — le fix du HyperparameterModulator applique-t-il RÉELLEMENT
le learning rate à l'optimiseur SB3 ?

BUG V31 : `self.agent.learning_rate = x` ne touche jamais optimizer.param_groups.
SB3 fixe le LR via self.lr_schedule(progress) à chaque train(). Ce test échoue
sur l'ancien code et passe avec _apply_lr().

Lancer :
    ADAN0/.../python -m pytest tests/test_hyperparam_lr_fix.py -q
ou   python tests/test_hyperparam_lr_fix.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from adan_trading_bot.training.hyperparam_modulator import HyperparameterModulator


class _Dummy(gym.Env):
    """Env minimal pour instancier un PPO réel (optimiseur inclus)."""
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        self._t += 1
        return (self.observation_space.sample(), 0.0, self._t >= 8, False, {})


def _optimizer_lr(model):
    return model.policy.optimizer.param_groups[0]["lr"]


def test_apply_lr_reaches_optimizer():
    base_lr = 3e-4
    model = PPO("MlpPolicy", _Dummy(), learning_rate=base_lr,
                n_steps=16, batch_size=16, n_epochs=1, verbose=0)

    mod = HyperparameterModulator(model, {
        "min_learning_rate": 1e-6,
        "max_learning_rate": 1e-3,
        "defensive_lr_factor": 0.5,   # DEFENSIVE -> LR/2
    })
    # initial_params doit avoir capté le LR de base
    mod.initial_params["learning_rate"] = base_lr

    changes = mod.adjust_params({"risk_mode": "DEFENSIVE"})
    assert "learning_rate" in changes, "modulate() n'a pas signalé de changement de LR"
    target = base_lr * 0.5

    # 1) l'optimiseur doit refléter le nouveau LR IMMÉDIATEMENT
    assert abs(_optimizer_lr(model) - target) < 1e-12, (
        f"optimizer LR={_optimizer_lr(model)} != cible {target} "
        "(le fix ne pousse pas dans param_groups)")

    # 2) et surtout : après un train() réel, SB3 ré-applique lr_schedule ->
    #    doit RESTER à la cible (preuve que lr_schedule a bien été remplacé,
    #    pas seulement param_groups qui serait écrasé par l'ancien schedule).
    model.learn(total_timesteps=16)
    assert abs(_optimizer_lr(model) - target) < 1e-12, (
        f"après train(), optimizer LR={_optimizer_lr(model)} est retombé — "
        "lr_schedule n'a pas été remplacé (BUG V31 non corrigé)")

    print(f"✅ LR appliqué et persistant après train(): {_optimizer_lr(model):.2e} (cible {target:.2e})")


def test_ent_coef_applied():
    model = PPO("MlpPolicy", _Dummy(), ent_coef=0.01,
                n_steps=16, batch_size=16, n_epochs=1, verbose=0)
    mod = HyperparameterModulator(model, {
        "min_ent_coef": 1e-4, "max_ent_coef": 0.1,
        "defensive_ent_factor": 2.0,
    })
    mod.initial_params["ent_coef"] = 0.01
    changes = mod.adjust_params({"risk_mode": "DEFENSIVE"})
    assert "ent_coef" in changes
    assert abs(model.ent_coef - 0.02) < 1e-9, f"ent_coef={model.ent_coef} != 0.02"
    print(f"✅ ent_coef appliqué: {model.ent_coef}")


if __name__ == "__main__":
    test_apply_lr_reaches_optimizer()
    test_ent_coef_applied()
    print("\n🎉 Tous les tests forensic LR/ent_coef PASS")
