"""Tests de ActionDimMonitor (instrumentation par dimension — MESURE seulement).

Style auto-exécutable (sans pytest). Vérifie :
  * journalisation post-tanh mean/std/sat_frac par dimension ;
  * AUCUNE modification du réseau (log_std inchangé) ;
  * respect de log_every ;
  * CSV optionnel écrit une ligne par fenêtre journalisée ;
  * robustesse si rollout_buffer / actions absents.

NB : BaseCallback.logger est une property read-only renvoyant self.model.logger
-> le mock _Logger DOIT vivre sur le modèle (méthode validée pour le guard).
"""
from __future__ import annotations

import os
import tempfile
import traceback

import numpy as np
import torch

from adan_trading_bot.utils.action_dim_monitor import ActionDimMonitor


class _Logger:
    def __init__(self):
        self.records = {}

    def record(self, k, v):
        self.records[k] = v


class _Policy:
    def __init__(self, log_std):
        self.log_std = torch.nn.Parameter(torch.tensor(log_std, dtype=torch.float32))

    def set_training_mode(self, mode):
        pass


class _Buffer:
    def __init__(self, actions):
        self.actions = actions
        self.observations = None  # pré-tanh désactivé dans ces tests


class _Model:
    def __init__(self, log_std, actions):
        self.policy = _Policy(log_std)
        self.rollout_buffer = _Buffer(actions)
        self.num_timesteps = 1234
        self.device = "cpu"
        self.logger = _Logger()  # SB3 property logger -> model.logger


def _attach(mon, model):
    mon.model = model


def _make_actions(A, sat_dims, n=400, n_envs=2):
    rng = np.random.default_rng(0)
    acts = rng.uniform(-0.5, 0.5, size=(n, n_envs, A)).astype(np.float32)
    for j in sat_dims:
        acts[:, :, j] = -1.0
    return acts


# ── 1. journalise mean/std/sat par dimension ─────────────────────────────────
def test_records_per_dim():
    A = 5
    mon = ActionDimMonitor(log_every=1, pre_tanh_batch=0, verbose=0)
    model = _Model(np.zeros((64, A)), _make_actions(A, [1]))
    _attach(mon, model)
    mon._on_rollout_end()
    recs = model.logger.records
    assert "actiondim/size_post_mean" in recs
    assert "actiondim/size_post_std" in recs
    assert "actiondim/size_sat_frac" in recs
    # size collé à -1 -> mean≈-1, std≈0, sat_frac≈1
    assert abs(recs["actiondim/size_post_mean"] + 1.0) < 1e-3
    assert recs["actiondim/size_post_std"] < 1e-3
    assert recs["actiondim/size_sat_frac"] >= 0.99
    # direction non saturée -> sat_frac faible
    assert recs["actiondim/direction_sat_frac"] < 0.1


# ── 2. NE MODIFIE PAS le réseau ──────────────────────────────────────────────
def test_does_not_touch_network():
    A = 5
    mon = ActionDimMonitor(log_every=1, pre_tanh_batch=0, verbose=0)
    ls0 = np.random.RandomState(1).randn(64, A).astype(np.float32)
    model = _Model(ls0.copy(), _make_actions(A, [1]))
    _attach(mon, model)
    for _ in range(5):
        mon._on_rollout_end()
    assert torch.allclose(model.policy.log_std.data, torch.tensor(ls0))  # inchangé


# ── 3. log_every respecté ────────────────────────────────────────────────────
def test_log_every():
    A = 5
    mon = ActionDimMonitor(log_every=3, pre_tanh_batch=0, verbose=0)
    model = _Model(np.zeros((64, A)), _make_actions(A, [1]))
    _attach(mon, model)
    model.logger.records.clear()
    mon._on_rollout_end()  # idx=1 -> pas de log
    assert model.logger.records == {}
    mon._on_rollout_end()  # idx=2 -> pas de log
    assert model.logger.records == {}
    mon._on_rollout_end()  # idx=3 -> log
    assert "actiondim/size_post_mean" in model.logger.records


# ── 4. CSV optionnel ──────────────────────────────────────────────────────────
def test_csv_output():
    A = 5
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ad.csv")
        mon = ActionDimMonitor(log_every=1, pre_tanh_batch=0, csv_path=path,
                               verbose=0)
        model = _Model(np.zeros((64, A)), _make_actions(A, [1]))
        _attach(mon, model)
        mon._on_rollout_end()
        mon._on_rollout_end()
        assert os.path.exists(path)
        with open(path) as f:
            lines = f.read().strip().splitlines()
        assert len(lines) == 3  # header + 2 fenêtres
        assert lines[0].startswith("step,")
        assert "size_post_mean" in lines[0]


# ── 5. actions 2D (sans dim envs) ─────────────────────────────────────────────
def test_handles_2d_actions():
    A = 5
    mon = ActionDimMonitor(log_every=1, pre_tanh_batch=0, verbose=0)
    acts2d = _make_actions(A, [4], n=200, n_envs=1).reshape(-1, A)
    model = _Model(np.zeros((64, A)), acts2d)
    _attach(mon, model)
    mon._on_rollout_end()
    assert model.logger.records["actiondim/tp_sat_frac"] >= 0.99


# ── 6. robustesse buffer absent ──────────────────────────────────────────────
def test_no_buffer_no_crash():
    A = 5
    mon = ActionDimMonitor(log_every=1, pre_tanh_batch=0, verbose=0)
    model = _Model(np.zeros((64, A)), None)  # actions=None
    _attach(mon, model)
    mon._on_rollout_end()  # ne doit pas lever
    assert model.logger.records == {}


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
