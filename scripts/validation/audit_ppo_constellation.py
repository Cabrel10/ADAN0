#!/usr/bin/env python3
"""Audit the full PPO constellation on a real smoke checkpoint.

What this answers
-----------------
Everything measured until now used a *random* policy. This script loads the
checkpoint produced by a real smoke run and measures the trained policy's own
"energy" plus the whole learning chain, on BTCUSDT_BINANCE/train with the HMM
fix and the terminated/truncated fix active and the fee gate ON.

Policy energy (SB3 DiagGaussianDistribution over Box(-1,1,5)):
    action_mean  p01/p50/p99   per dimension
    action_std   p01/p50/p99   (exp(log_std))
    log_std      per dimension
    entropy      mean/p05/p95
    fraction |a0| < deadband
    fraction saturated at +/-1

Chain integrity:
    raw action -> requested -> feasible -> routed -> executed -> reward
    correlation between the policy's a0 and the CLEAN HMM posterior
    context_vector[3] (does the brain use the timing signal at all?)

Update metrics are read from the smoke's own SB3 logger output, not recomputed:
    approx_kl, clip_fraction, explained_variance, value_loss,
    policy_gradient_loss, entropy_loss, std, n_updates

Output: logs/validation/ppo_constellation_<ts>.json
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("ADAN_TRAINING_SILENT", "1")
os.environ.setdefault("ADAN_RICH_STEP_EVERY", "999999")
os.environ.setdefault("ADAN_FREE_SLTP", "1")

ASSET = os.environ.get("DIAG_ASSET", "BTCUSDT_BINANCE")
SPLIT = os.environ.get("DIAG_SPLIT", "train")
STEPS = int(os.environ.get("DIAG_STEPS", "300"))
SEED = int(os.environ.get("DIAG_SEED", "330500"))
CKPT = os.environ.get("DIAG_CKPT", "")
SMOKE_LOG = os.environ.get("DIAG_SMOKE_LOG", "/tmp/smoke_btc.log")

_SLTP = {
    "BTCUSDT_BINANCE": {"ADAN_TP_LO": "0.0135", "ADAN_TP_HI": "0.0222",
                        "ADAN_SL_HI": "0.0235"},
    "DOGEUSDT_BINANCE": {"ADAN_TP_LO": "0.003", "ADAN_TP_HI": "0.090",
                         "ADAN_SL_HI": "0.060"},
}
for _k, _v in _SLTP.get(ASSET, {}).items():
    os.environ.setdefault(_k, _v)


def pct(a, q):
    return round(float(np.percentile(a, q)), 6) if len(a) else None


def build_env():
    from adan_trading_bot.common.config_loader import ConfigLoader
    from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
    from adan_trading_bot.environment.multi_asset_chunked_env import (
        MultiAssetChunkedEnv,
    )

    cfg = ConfigLoader.load_config(str(REPO_ROOT / "config" / "config.yaml"))
    cfg.setdefault("environment", {})["rich_display_interval"] = 999999
    wc = copy.deepcopy(cfg.get("workers", {}).get("w1", {}))
    wc.update({
        "worker_id": 0, "data_split": SPLIT, "data_split_override": SPLIT,
        "timeframes": ["5m", "1h", "4h"], "assets": [ASSET],
    })
    cfg.setdefault("data", {})["assets"] = [ASSET]
    cfg.setdefault("environment", {})["assets"] = [ASSET]
    data = ChunkedDataLoader(config=cfg, worker_config=wc,
                             worker_id=0).load_chunk(0)
    env = MultiAssetChunkedEnv(data=data, config=cfg, worker_config=wc,
                               worker_id=0, live_mode=False)
    return env, cfg


def parse_smoke_log(path: str) -> dict:
    """Read the SB3 table rows the smoke printed. Never recompute them."""
    keys = ("approx_kl", "clip_fraction", "explained_variance", "value_loss",
            "policy_gradient_loss", "entropy_loss", "std", "n_updates",
            "loss", "total_timesteps", "fps", "iterations")
    out: dict[str, list] = {k: [] for k in keys}
    p = Path(path)
    if not p.exists():
        return {"error": f"smoke log not found: {path}"}
    txt = p.read_text(errors="ignore").replace("\x00", "")
    for line in txt.splitlines():
        m = re.match(r"\|\s+(\w+)\s+\|\s+([-\d.e+]+)\s+\|", line.strip())
        if m and m.group(1) in out:
            try:
                out[m.group(1)].append(float(m.group(2)))
            except ValueError:
                pass
    summary = {}
    for k, v in out.items():
        if v:
            summary[k] = {"n": len(v), "first": v[0], "last": v[-1],
                          "min": min(v), "max": max(v),
                          "mean": round(float(np.mean(v)), 8)}
    return summary


def main() -> None:
    report: dict = {
        "audit": "ppo_constellation",
        "asset": ASSET, "split": SPLIT, "steps": STEPS, "seed": SEED,
        "checkpoint": CKPT,
        "fee_gate_active": os.environ.get("ADAN_DISABLE_EV_FEE_GATE")
        in (None, "", "0"),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ---------- 1. UPDATE METRICS, read from the smoke's own logger --------
    report["ppo_update_metrics_from_smoke_log"] = parse_smoke_log(SMOKE_LOG)

    # ---------- 2. POLICY ENERGY + CHAIN, on the real env -----------------
    if not CKPT or not Path(CKPT).exists():
        report["policy_energy"] = {
            "error": "no checkpoint available; policy energy not measurable"}
        report["verdict"] = "INCOMPLETE"
    else:
        import torch
        from stable_baselines3 import PPO

        env, cfg = build_env()
        obs, _ = env.reset(seed=SEED)
        model = PPO.load(CKPT, device="cpu")
        pol = model.policy

        thr = float((cfg.get("environment", {})
                     .get("action_thresholds", {}) or {}).get("5m", 0.05))

        means, stds, ents, logps, a0s, p_hmms = [], [], [], [], [], []
        requested, executed = Counter(), Counter()
        kinds = Counter()
        rewards, comps_seen = [], Counter()
        sat, deadband = 0, 0
        boundaries = 0
        exceptions = 0

        # log_std is a policy parameter, not per-step: read it once.
        try:
            ls = pol.log_std.detach().cpu().numpy().ravel()
            report["log_std_per_dim"] = [round(float(x), 6) for x in ls]
            report["std_per_dim"] = [round(float(np.exp(x)), 6) for x in ls]
        except Exception as exc:
            report["log_std_per_dim"] = f"unavailable: {exc}"

        for _ in range(STEPS):
            try:
                obs_t, _ = pol.obs_to_tensor(obs)
                with torch.no_grad():
                    dist = pol.get_distribution(obs_t)
                    mu = dist.distribution.mean.cpu().numpy().ravel()
                    sd = dist.distribution.stddev.cpu().numpy().ravel()
                    ent = float(dist.entropy().cpu().numpy().ravel()[0])
                    act_t = dist.get_actions(deterministic=False)
                    lp = float(pol.get_distribution(obs_t)
                               .log_prob(act_t).cpu().numpy().ravel()[0])
                action = act_t.cpu().numpy().ravel().astype(np.float32)
            except Exception:
                exceptions += 1
                break

            means.append(mu)
            stds.append(sd)
            ents.append(ent)
            logps.append(lp)
            a0 = float(np.clip(action[0], -1.0, 1.0))
            a0s.append(a0)
            if abs(a0) >= 0.999:
                sat += 1
            if abs(a0) <= thr:
                deadband += 1

            pm = getattr(env, "portfolio_manager", None)
            positions = getattr(pm, "positions", {}) or {}
            pos = positions.get(ASSET)
            in_pos = bool(pos is not None and getattr(pos, "is_open", False))
            want = "HOLD"
            if in_pos and a0 < -thr:
                want = "SELL"
            elif (not in_pos) and a0 > thr:
                want = "BUY"
            requested[want] += 1

            before = int((getattr(env, "action_pipeline_counts", {})
                          or {}).get("trade_executed", 0))
            try:
                obs, rew, term, trunc, info = env.step(action)
            except Exception:
                exceptions += 1
                break
            after = int((getattr(env, "action_pipeline_counts", {})
                         or {}).get("trade_executed", 0))
            executed[want if after > before else "HOLD"] += 1
            rewards.append(float(rew))
            kinds[str(info.get("termination_kind", "none"))] += 1
            for k, v in (getattr(env, "_last_reward_components", None)
                         or {}).items():
                try:
                    if abs(float(v)) > 0.0:
                        comps_seen[k] += 1
                except Exception:
                    pass

            # CLEAN HMM posterior as the policy sees it
            try:
                cv = info.get("context_vector")
                if cv is not None and len(cv) > 5:
                    p_hmms.append(float(cv[3]))
            except Exception:
                pass

            if term or trunc:
                boundaries += 1
                obs, _ = env.reset()

        M = np.asarray(means) if means else np.zeros((0, 5))
        S = np.asarray(stds) if stds else np.zeros((0, 5))
        n = len(a0s)

        report["policy_energy"] = {
            "n_steps_measured": n,
            "exceptions": exceptions,
            "action_mean_per_dim": {
                f"dim{i}": {"p01": pct(M[:, i], 1), "p50": pct(M[:, i], 50),
                            "p99": pct(M[:, i], 99)}
                for i in range(M.shape[1])} if n else {},
            "action_std_per_dim": {
                f"dim{i}": {"p01": pct(S[:, i], 1), "p50": pct(S[:, i], 50),
                            "p99": pct(S[:, i], 99)}
                for i in range(S.shape[1])} if n else {},
            "entropy": {"mean": round(float(np.mean(ents)), 6) if ents else None,
                        "p05": pct(ents, 5), "p95": pct(ents, 95)},
            "log_prob": {"mean": round(float(np.mean(logps)), 6) if logps else None,
                         "p05": pct(logps, 5), "p95": pct(logps, 95)},
            "a0_distribution": {"p01": pct(a0s, 1), "p50": pct(a0s, 50),
                                "p99": pct(a0s, 99),
                                "mean": round(float(np.mean(a0s)), 6) if a0s else None},
            "frac_saturated_abs_ge_0.999": round(sat / max(1, n), 6),
            "frac_inside_deadband": round(deadband / max(1, n), 6),
            "deadband_threshold": thr,
        }

        # correlation policy intent <-> clean timing signal
        corr = None
        if len(p_hmms) == len(a0s) and len(a0s) > 10:
            try:
                if np.std(p_hmms) > 1e-12 and np.std(a0s) > 1e-12:
                    corr = round(float(np.corrcoef(a0s, p_hmms)[0, 1]), 6)
            except Exception:
                corr = None
        report["timing_signal_usage"] = {
            "p_hmm_observed_n": len(p_hmms),
            "p_hmm_distinct": len(set(round(x, 6) for x in p_hmms)),
            "p_hmm_mean": round(float(np.mean(p_hmms)), 6) if p_hmms else None,
            "corr_a0_vs_p_hmm": corr,
            "note": ("a non-null correlation only shows the brain's intent "
                     "co-moves with the clean regime posterior; it does not "
                     "prove the distinction was learned"),
        }

        report["chain"] = {
            "requested": dict(requested),
            "executed": dict(executed),
            "termination_kinds": dict(kinds),
            "boundaries_hit": boundaries,
            "window_boundaries_are_truncated": kinds.get("terminal", 0) == 0
            or "terminal_is_economic_death_only",
            "reward": {"mean": round(float(np.mean(rewards)), 6) if rewards else None,
                       "std": round(float(np.std(rewards)), 6) if rewards else None,
                       "min": round(float(min(rewards)), 6) if rewards else None,
                       "max": round(float(max(rewards)), 6) if rewards else None,
                       "nonzero_share": round(
                           sum(1 for r in rewards if abs(r) > 1e-12)
                           / max(1, len(rewards)), 6)},
            "reward_components_nonzero_counts": dict(
                sorted(comps_seen.items(), key=lambda kv: -kv[1])[:20]),
            "action_pipeline_counts": dict(
                getattr(env, "action_pipeline_counts", {}) or {}),
            "rejection_reasons": dict(
                getattr(env, "rejection_reasons", {}) or {}),
        }

        checks = {
            "no_exception": exceptions == 0,
            "policy_explores": bool(np.mean([float(np.mean(x)) for x in S]) > 1e-3)
            if len(S) else False,
            "actions_reach_pipeline": (report["chain"]["action_pipeline_counts"]
                                       .get("policy", 0) > 0),
            "trades_executed": (report["chain"]["action_pipeline_counts"]
                                .get("trade_executed", 0) > 0,),
            "reward_varies": (report["chain"]["reward"]["std"] or 0.0) > 1e-6,
            "fee_gate_active": report["fee_gate_active"],
        }
        checks["trades_executed"] = bool(checks["trades_executed"][0])
        report["health_checks"] = checks
        report["verdict"] = "PASS" if all(checks.values()) else "FAIL"

    out = REPO_ROOT / "logs" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"ppo_constellation_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"\n[WROTE] {path}")


if __name__ == "__main__":
    main()
