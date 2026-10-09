"""Tests de ActionSaturationGuard (réveil d'exploration anti-saturation).

Style auto-exécutable (sans pytest). Vérifie :
  * détection de saturation par dimension ;
  * déclenchement après ``patience`` rollouts ;
  * bump du log_std en forme gSDE (2D) ET DiagGaussian (1D) ;
  * borne max_log_std respectée ;
  * mode dry-run (intervene=False) ne touche pas le réseau.
"""
from __future__ import annotations

import traceback

import numpy as np
import torch

from adan_trading_bot.utils.action_saturation_guard import ActionSaturationGuard


class _Space:
    def __init__(self, a):
        self.shape = (a,)


class _Policy:
    def __init__(self, log_std):
        self.log_std = torch.nn.Parameter(torch.tensor(log_std, dtype=torch.float32))


class _Buffer:
    def __init__(self, actions):
        self.actions = actions


class _Logger:
    def __init__(self):
        self.records = {}

    def record(self, k, v):
        self.records[k] = v


class _Model:
    """Mock minimal compatible avec ce que le callback lit."""
    def __init__(self, A, log_std, actions):
        self.action_space = _Space(A)
        self.policy = _Policy(log_std)
        self.rollout_buffer = _Buffer(actions)
        # SB3 BaseCallback.logger est une PROPERTY read-only qui renvoie
        # self.model.logger -> le logger DOIT vivre sur le modèle, pas sur
        # le callback (écrire guard._logger est ignoré par la property).
        self.logger = _Logger()


def _attach(guard, model):
    # injecte le modèle sans passer par init_callback (évite l'init SB3).
    # guard.logger -> guard.model.logger -> model.logger (le mock _Logger).
    guard.model = model


def _make_actions(A, sat_dims, n=500, n_envs=1):
    """Actions (n, n_envs, A) : sat_dims collées à -1, autres variées."""
    rng = np.random.default_rng(0)
    acts = rng.uniform(-0.5, 0.5, size=(n, n_envs, A)).astype(np.float32)
    for j in sat_dims:
        acts[:, :, j] = -1.0
    return acts


# ── 1. détection : marque la bonne dimension ─────────────────────────────────
def test_detects_saturated_dim_after_patience():
    A = 5
    guard = ActionSaturationGuard(patience=2, intervene=True, verbose=0)
    model = _Model(A, log_std=np.zeros((64, A)), actions=_make_actions(A, [1]))
    _attach(guard, model)
    guard._on_training_start()
    before = model.policy.log_std.data[:, 1].clone()
    guard._on_rollout_end()  # streak[1] = 1, pas encore d'action
    mid = model.policy.log_std.data[:, 1].clone()
    assert torch.allclose(before, mid)  # patience pas atteinte
    guard._on_rollout_end()  # streak[1] = 2 -> bump
    after = model.policy.log_std.data[:, 1]
    assert (after > mid).all()  # log_std relevé sur la dim saturée
    assert guard._n_bumps == 1


def test_non_saturated_dims_untouched():
    A = 5
    guard = ActionSaturationGuard(patience=1, intervene=True, verbose=0)
    model = _Model(A, log_std=np.zeros((64, A)), actions=_make_actions(A, [1]))
    _attach(guard, model)
    guard._on_training_start()
    guard._on_rollout_end()  # patience=1 -> bump immédiat sur dim 1
    ls = model.policy.log_std.data
    # dim 1 relevée, dims 0/2/3/4 inchangées
    assert (ls[:, 1] > 0).all()
    for j in (0, 2, 3, 4):
        assert torch.allclose(ls[:, j], torch.zeros_like(ls[:, j]))


# ── 2. forme DiagGaussian (1D) ───────────────────────────────────────────────
def test_diag_gaussian_1d_log_std():
    A = 5
    guard = ActionSaturationGuard(patience=1, intervene=True, verbose=0)
    model = _Model(A, log_std=np.zeros(A), actions=_make_actions(A, [4]))
    _attach(guard, model)
    guard._on_training_start()
    guard._on_rollout_end()
    ls = model.policy.log_std.data
    assert ls[4] > 0  # tp relevé
    assert torch.allclose(ls[:4], torch.zeros(4))


# ── 3. borne max_log_std respectée ───────────────────────────────────────────
def test_respects_max_log_std():
    A = 5
    guard = ActionSaturationGuard(patience=1, bump_log_std=5.0, max_log_std=1.0,
                                  intervene=True, verbose=0)
    model = _Model(A, log_std=np.zeros((64, A)), actions=_make_actions(A, [1]))
    _attach(guard, model)
    guard._on_training_start()
    guard._on_rollout_end()
    assert model.policy.log_std.data[:, 1].max().item() <= 1.0 + 1e-6


# ── 4. dry-run ne touche pas le réseau ───────────────────────────────────────
def test_dry_run_no_change():
    A = 5
    guard = ActionSaturationGuard(patience=1, intervene=False, verbose=0)
    model = _Model(A, log_std=np.zeros((64, A)), actions=_make_actions(A, [1]))
    _attach(guard, model)
    guard._on_training_start()
    guard._on_rollout_end()
    assert torch.allclose(model.policy.log_std.data,
                          torch.zeros_like(model.policy.log_std.data))
    assert guard._n_bumps == 0


# ── 5. 2D actions sans dimension envs ────────────────────────────────────────
def test_handles_2d_actions():
    A = 5
    guard = ActionSaturationGuard(patience=1, intervene=True, verbose=0)
    acts2d = _make_actions(A, [1], n=300, n_envs=1).reshape(-1, A)
    model = _Model(A, log_std=np.zeros((64, A)), actions=acts2d)
    _attach(guard, model)
    guard._on_training_start()
    guard._on_rollout_end()
    assert (model.policy.log_std.data[:, 1] > 0).all()


# ── 6. logs TensorBoard renseignés ───────────────────────────────────────────
def test_records_metrics():
    A = 5
    guard = ActionSaturationGuard(patience=5, intervene=True, verbose=0)
    model = _Model(A, log_std=np.zeros((64, A)), actions=_make_actions(A, [1]))
    _attach(guard, model)
    guard._on_training_start()
    guard._on_rollout_end()
    recs = guard.logger.records
    assert "saturation/size_sat_frac" in recs
    assert recs["saturation/size_sat_frac"] >= 0.95  # size collé à -1
    assert "saturation/direction_action_std" in recs


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} tests passés")
    return passed == len(fns)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
