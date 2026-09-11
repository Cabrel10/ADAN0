#!/usr/bin/env python3
"""
PROFILEUR FPS ADAN0 — PHASE C (mesure du vrai goulot).
Lance un mini-entrainement SB3 (sandbox) sous cProfile et mesure:
  - FPS global (steps/s)
  - temps cumule par fonction (env.step / forward / learn / logging / pandas / numpy / io)
Sortie: dump cProfile + top fonctions par temps cumule.

Usage: python profile_fps.py <n_steps>
"""
import os
import sys
import time
import cProfile
import pstats
import io as _io

# Bride threads AVANT import torch/numpy si demande (sinon laisse libre)
NTH = os.environ.get("PROFILE_NTHREADS")
if NTH:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = NTH

ROOT = "/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("ADAN_TRAINING_SILENT", "1")
os.environ.setdefault("ADAN_USE_SDE", "0")

import yaml  # noqa
import numpy as np  # noqa
import torch  # noqa
if NTH:
    torch.set_num_threads(int(NTH))
    torch.set_num_interop_threads(1)

from stable_baselines3 import PPO  # noqa
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa
from adan_trading_bot.environment.multi_asset_chunked_env import (  # noqa
    MultiAssetChunkedEnv)


def build_env_and_model():
    with open(os.path.join(ROOT, "config/config.yaml")) as f:
        config = yaml.safe_load(f)
    sandbox_cfg = config.get("sandbox", {})
    workers = config.get("workers", [])
    wcfg = None
    for w in workers:
        if isinstance(w, dict):
            wcfg = w
            break
    env = MultiAssetChunkedEnv(config=config, worker_config=wcfg)
    venv = DummyVecEnv([lambda: env])
    n_steps = int(sandbox_cfg.get("n_steps", 512))
    batch = int(sandbox_cfg.get("batch_size", 64))
    n_epochs = int(os.environ.get("ADAN_N_EPOCHS",
                                  sandbox_cfg.get("n_epochs", 10)))
    model = PPO("MultiInputPolicy", venv, n_steps=n_steps, batch_size=batch,
                n_epochs=n_epochs, verbose=0, device="cpu")
    return model, n_steps


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    print(f"[profile] threads={NTH or 'LIBRE'} target_steps={n_steps}",
          flush=True)
    model, ppo_nsteps = build_env_and_model()
    print(f"[profile] model built. PPO n_steps={ppo_nsteps}", flush=True)

    pr = cProfile.Profile()
    t0 = time.time()
    pr.enable()
    model.learn(total_timesteps=n_steps, reset_num_timesteps=True)
    pr.disable()
    elapsed = time.time() - t0

    fps = n_steps / elapsed if elapsed > 0 else 0
    print(f"\n========== RESULTAT FPS ==========", flush=True)
    print(f"steps={n_steps} elapsed={elapsed:.1f}s FPS={fps:.2f}", flush=True)
    print(f"==================================\n", flush=True)

    # dump
    ts = time.strftime("%Y%m%d_%H%M%S")
    dump = os.path.join(ROOT, "logs", "forensic",
                        f"profile_{NTH or 'libre'}_{ts}.prof")
    pr.dump_stats(dump)
    print(f"[profile] dump -> {dump}", flush=True)

    # top par temps cumule
    s = _io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(35)
    print("===== TOP CUMULATIVE =====", flush=True)
    print(s.getvalue(), flush=True)

    # top par temps propre (tottime)
    s2 = _io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats("tottime")
    ps2.print_stats(25)
    print("===== TOP TOTTIME (temps propre) =====", flush=True)
    print(s2.getvalue(), flush=True)


if __name__ == "__main__":
    main()
