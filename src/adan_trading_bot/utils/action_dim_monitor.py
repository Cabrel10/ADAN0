"""ActionDimMonitor — instrumentation par dimension d'action (MESURE SEULEMENT).

Demandé par l'utilisateur (point de prudence 3 + ordre révisé) :
  « Relancer un entraînement court instrumenté. Observer size_mean, size_std,
    tp_mean, tp_std à intervalles réguliers. Si SIZE reste gelée → guard. »

Ce callback NE MODIFIE RIEN. Il se contente de journaliser, toutes les
``log_every`` itérations de rollout, pour chaque tête d'action :
  * post-tanh : mean / std des actions réellement émises (rollout_buffer.actions) ;
  * pré-tanh  : mean / std de μ et σ de la distribution gSDE (sur un mini-batch
    d'observations du buffer) — c'est CE qui dit si SIZE se réveille (μ remonte)
    ou reste noyée (μ ≈ -7).

À NE PAS confondre avec ActionSaturationGuard (qui, lui, intervient). Ici on
respecte la consigne « finir le test du guard mais NE PAS l'intégrer » : on
mesure d'abord, on corrige seulement si la mesure prouve que c'est nécessaire.

Sorties :
  * logger SB3 (TensorBoard) : ``actiondim/{tête}_post_mean`` etc. ;
  * console (logging) : une ligne récap compacte par fenêtre ;
  * optionnel CSV : ``ACTIONDIM_CSV`` (chemin) -> 1 ligne / fenêtre.
"""
from __future__ import annotations

import csv
import logging
import os
from typing import List, Optional

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)

DEFAULT_ACTION_NAMES = ["direction", "size", "tf", "sl", "tp"]


class ActionDimMonitor(BaseCallback):
    """Journalise mean/std par tête (post-tanh) ET μ/σ pré-tanh. Aucune écriture réseau.

    Args:
        action_names: noms des dimensions (par défaut ADAN 5 têtes).
        log_every: nb de rollouts entre deux journalisations (1 = chaque rollout).
        pre_tanh_batch: nb d'obs du buffer pour estimer μ/σ pré-tanh (0 = désactivé).
        csv_path: si fourni, append d'une ligne par fenêtre.
    """

    def __init__(
        self,
        action_names: Optional[List[str]] = None,
        log_every: int = 1,
        pre_tanh_batch: int = 256,
        csv_path: Optional[str] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.action_names = action_names or DEFAULT_ACTION_NAMES
        self.log_every = max(1, int(log_every))
        self.pre_tanh_batch = int(pre_tanh_batch)
        self.csv_path = csv_path or os.environ.get("ACTIONDIM_CSV")
        self._rollout_idx = 0
        self._csv_header_written = False

    def _name(self, j: int) -> str:
        return self.action_names[j] if j < len(self.action_names) else f"a{j}"

    # ── extraction pré-tanh (μ, σ) depuis la distribution gSDE ───────────────
    def _pre_tanh_stats(self):
        """Retourne (mu_mean, mu_std, sig_mean) par dim, ou None si indisponible."""
        if self.pre_tanh_batch <= 0:
            return None
        buf = getattr(self.model, "rollout_buffer", None)
        policy = getattr(self.model, "policy", None)
        if buf is None or policy is None:
            return None
        obs = getattr(buf, "observations", None)
        if obs is None:
            return None
        try:
            # obs peut être un dict (MultiInputPolicy) ou un array.
            if isinstance(obs, dict):
                n = next(iter(obs.values())).shape[0]
                k = min(self.pre_tanh_batch, n)
                idx = np.random.default_rng(0).integers(0, n, size=k)
                sample = {key: torch.as_tensor(
                    np.asarray(val)[idx].reshape(k, *np.asarray(val).shape[1:]),
                    device=self.model.device, dtype=torch.float32)
                    for key, val in obs.items()}
                # buffers SB3 ont shape (n_steps, n_envs, ...) -> aplatir envs
                sample = {key: v.reshape(-1, *v.shape[2:]) if v.dim() > 2 else v
                          for key, v in sample.items()}
            else:
                arr = np.asarray(obs)
                arr = arr.reshape(-1, arr.shape[-1])
                k = min(self.pre_tanh_batch, arr.shape[0])
                idx = np.random.default_rng(0).integers(0, arr.shape[0], size=k)
                sample = torch.as_tensor(arr[idx], device=self.model.device,
                                         dtype=torch.float32)
            policy.set_training_mode(False)
            with torch.no_grad():
                dist = policy.get_distribution(sample)
                inner = dist.distribution  # Normal pré-squash
                mu = inner.mean.detach().cpu().numpy()
                sig = inner.scale.detach().cpu().numpy()
            mu = mu.reshape(-1, mu.shape[-1])
            sig = sig.reshape(-1, sig.shape[-1])
            return mu.mean(0), mu.std(0), sig.mean(0)
        except Exception as e:  # noqa: BLE001
            if self.verbose:
                logger.debug("[ActionDimMonitor] pré-tanh indisponible: %s", e)
            return None

    def _on_rollout_end(self) -> None:
        self._rollout_idx += 1
        if self._rollout_idx % self.log_every != 0:
            return
        buf = getattr(self.model, "rollout_buffer", None)
        if buf is None or getattr(buf, "actions", None) is None:
            return
        acts = np.asarray(buf.actions)
        if acts.ndim == 3:
            acts = acts.reshape(-1, acts.shape[-1])
        elif acts.ndim != 2:
            return
        A = acts.shape[1]
        post_mean = acts.mean(0)
        post_std = acts.std(0)
        sat_frac = np.mean(np.abs(acts) >= 0.98, axis=0)

        pre = self._pre_tanh_stats()
        step = int(getattr(self.model, "num_timesteps", 0))

        # journalisation TensorBoard
        if self.logger is not None:
            for j in range(A):
                nm = self._name(j)
                self.logger.record(f"actiondim/{nm}_post_mean", float(post_mean[j]))
                self.logger.record(f"actiondim/{nm}_post_std", float(post_std[j]))
                self.logger.record(f"actiondim/{nm}_sat_frac", float(sat_frac[j]))
                if pre is not None:
                    self.logger.record(f"actiondim/{nm}_mu_mean", float(pre[0][j]))
                    self.logger.record(f"actiondim/{nm}_sigma_mean", float(pre[2][j]))

        # console récap compacte
        parts = []
        for j in range(A):
            nm = self._name(j)
            seg = f"{nm}:μ̂={post_mean[j]:+.2f}/σ̂={post_std[j]:.3f}"
            if pre is not None:
                seg += f"|μ={pre[0][j]:+.2f}/σ={pre[2][j]:.2f}"
            parts.append(seg)
        logger.info("[ActionDim] step=%d  %s", step, "  ".join(parts))

        # CSV optionnel
        if self.csv_path:
            self._append_csv(step, A, post_mean, post_std, sat_frac, pre)

    def _append_csv(self, step, A, post_mean, post_std, sat_frac, pre):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)),
                        exist_ok=True)
            new = not os.path.exists(self.csv_path)
            with open(self.csv_path, "a", newline="") as f:
                w = csv.writer(f)
                if new and not self._csv_header_written:
                    hdr = ["step"]
                    for j in range(A):
                        nm = self._name(j)
                        hdr += [f"{nm}_post_mean", f"{nm}_post_std",
                                f"{nm}_sat_frac", f"{nm}_mu_mean", f"{nm}_sigma_mean"]
                    w.writerow(hdr)
                    self._csv_header_written = True
                row = [step]
                for j in range(A):
                    mu = pre[0][j] if pre is not None else ""
                    sg = pre[2][j] if pre is not None else ""
                    row += [float(post_mean[j]), float(post_std[j]),
                            float(sat_frac[j]), mu, sg]
                w.writerow(row)
        except Exception as e:  # noqa: BLE001
            if self.verbose:
                logger.debug("[ActionDimMonitor] CSV échoué: %s", e)

    def _on_step(self) -> bool:
        return True
