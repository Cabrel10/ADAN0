#!/usr/bin/env python3
"""Deterministic ENVIRONMENT PROBE for ADAN0 — NO PPO, NO POLICY.

Objectif (session V32 preflight, RÈGLE utilisateur):
  Prouver par test déterministe (actions FIXÉES, pas échantillonnées par un
  réseau) comment `a0 -> routing -> action exécutée -> pénalité -> reward ->
  reward final` se comporte, AVANT toute modif du reward / RAL / V32.

Ce script:
  1. Instancie MultiAssetChunkedEnv avec la config ACTUELLE (V31 corrigée:
     sell_while_flat=0.0, buy_while_open=0.0, seuils par TF).
  2. Injecte des séquences d'actions DÉTERMINISTES (pas de modèle):
       - HOLD constant (a0=0.0)
       - BUY-intent constant (a0=+0.5)
       - SELL-intent constant (a0=-0.5)
       - alternance BUY/SELL
       - marche aléatoire uniforme dans [-1,1] (proxy politique NON entraînée)
  3. Journalise par step: a0, discrete route, action exécutée, routing_reject,
     TOUTE la décomposition du reward (info["reward_components"]),
     raw_reward, final_reward.
  4. Rejoue sur DEUX chunks (haussier / baissier) pour distinguer un biais
     STRUCTUREL du reward d'un biais dû à la distribution du dataset.
  5. Sonde les pénalités comme FONCTIONS (err=1,2,4,8 -> P1,P2,P4,P8, ratios)
     pour révéler la vraie fonction effective APRÈS symlog/clamp.

Sortie JSONL: logs/probe/probe_env_<ts>.jsonl (une ligne par step + résumés).
AUCUNE modification du code d'entraînement/reward. Read-only sur le comportement.

Usage:
    ADAN_REWARD_TELEM=0 python scripts/diagnostics/probe_env_deterministic.py \
        --steps 60 --seed 42
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adan_trading_bot.common.config_loader import ConfigLoader  # noqa: E402
from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader  # noqa: E402
from adan_trading_bot.environment.multi_asset_chunked_env import (  # noqa: E402
    MultiAssetChunkedEnv,
)
from adan_trading_bot.environment.action_routing import (  # noqa: E402
    route_action_by_state,
    BUY,
    SELL,
    HOLD,
)

# ── Seuils "FAVORABLE" définis AVANT de lire les résultats (anti-biais) ──────
# (leçon L3=0.831 pris pour argent comptant : on fixe les critères d'abord)
THRESHOLDS = {
    "random_mean_reward_max_abs": 0.02,   # bruit pur -> reward moyen ~ 0
    "action_reward_spread_min":   1e-4,   # BUY/SELL/HOLD doivent différer un peu
    "sell_flat_must_not_profit":  0.0,    # SELL en FLAT ne doit jamais être rentable
    "penalty_reaches_action":     0.10,   # une pénalité "sentie" > 0.1 en magnitude
}

DIR_LABEL = {BUY: "BUY", SELL: "SELL", HOLD: "HOLD"}


def build_env(split: str, seed: int):
    """Réplique EXACTE du pattern action_pipeline_harness (env-build prouvé)."""
    cfg = ConfigLoader.load_config(str(REPO_ROOT / "config" / "config.yaml"))
    cfg.setdefault("environment", {})["rich_display_interval"] = 999999
    wc = copy.deepcopy(cfg.get("workers", {}).get("w1", {}))
    wc.update({
        "worker_id": 0,
        "data_split": split,
        "data_split_override": split,
        "timeframes": ["5m", "1h", "4h"],
        "assets": ["BTCUSDT"],
    })
    data = ChunkedDataLoader(config=cfg, worker_config=wc, worker_id=0).load_chunk(0)
    env = MultiAssetChunkedEnv(
        data=data, config=cfg, worker_config=wc, worker_id=0, live_mode=False,
    )
    env.reset(seed=seed)
    # seuil d'action effectif (TF dominant 5m=0.05 par config)
    thr = float(cfg.get("environment", {})
                .get("action_thresholds", {})
                .get("5m", 0.05)) if isinstance(cfg.get("environment"), dict) else 0.05
    # fallback: lire au niveau racine si présent
    try:
        thr = float(cfg["action_thresholds"]["5m"])
    except (KeyError, TypeError):
        pass
    return env, cfg, thr


def position_open(env) -> bool:
    pm = getattr(env, "portfolio_manager", None)
    if pm is None:
        return False
    for p in getattr(pm, "positions", {}).values():
        if p is not None and bool(getattr(p, "is_open", False)):
            return True
    return False


def controlled_action(a0: float, sl_raw: float = 1.0, tp_raw: float = 1.0) -> np.ndarray:
    """[direction, size, timeframe(5m=-1), SL, TP] — mêmes conventions harness.

    IMPORTANT (prouvé par la sonde v1): sl_raw/tp_raw=0 => sl_pct=0 =>
    p_min_required=0.99 (env l.9513 "no SL = reject") => l'EV fee_gate bloque
    TOUS les BUY à p_hmm=0.5. On passe donc sl_raw=tp_raw=1.0 (défaut harness)
    pour que le BUY puisse s'exécuter et atteindre l'état LONG.
    """
    return np.asarray([a0, 0.0, -1.0, sl_raw, tp_raw], dtype=np.float32)


def scripted_a0(seq: str, step: int, rng: np.random.Generator) -> float:
    if seq == "hold":
        return 0.0
    if seq == "buy":
        return 0.5
    if seq == "sell":
        return -0.5
    if seq == "alt":
        return 0.5 if (step % 2 == 0) else -0.5
    if seq == "random":
        return float(rng.uniform(-1.0, 1.0))
    raise ValueError(seq)


def run_sequence(env, thr, seq: str, steps: int, seed: int, out,
                 sl_raw: float = 1.0, tp_raw: float = 1.0) -> dict:
    """Exécute une séquence déterministe et journalise chaque step."""
    rng = np.random.default_rng(seed)
    agg = {
        "seq": seq, "steps": 0,
        "sum_final": 0.0, "sum_raw": 0.0,
        "route_counts": Counter(), "reject_counts": Counter(),
        "by_route_reward": defaultdict(float), "by_route_n": Counter(),
        "comp_sums": defaultdict(float),
    }
    for i in range(steps):
        in_pos = position_open(env)
        a0 = scripted_a0(seq, i, rng)
        # route THÉORIQUE (miroir de route_action_by_state, pour comparer au réel)
        route_theory = route_action_by_state(
            a0, in_position=in_pos, slot_available=True, threshold=thr,
        )
        action = controlled_action(a0, sl_raw=sl_raw, tp_raw=tp_raw)
        _obs, reward, term, trunc, info = env.step(action)
        rc = info.get("reward_components", {}) or {}
        executed = bool(getattr(env, "_last_trade_executed", False))
        reject_reason = None
        # reconstruire la raison de rejet la plus probable
        if route_theory == HOLD and abs(a0) <= thr:
            reject_reason = "deadband"
        elif route_theory in (BUY,) and not executed:
            reject_reason = "buy_not_executed"
        elif route_theory in (SELL,) and not executed:
            reject_reason = "sell_not_executed"

        line = {
            "type": "step", "seq": seq, "i": i,
            "state": "LONG" if in_pos else "FLAT",
            "a0": round(a0, 4), "thr": thr,
            "route_theory": DIR_LABEL[route_theory],
            "executed": executed,
            "reject_reason": reject_reason,
            "final_reward": round(float(reward), 6),
            "raw": round(float(rc.get("raw", 0.0)), 6),
            "behavior_penalty": round(float(rc.get("behavior_penalty", 0.0)), 6),
            "behavior_invalid_penalty": round(float(rc.get("behavior_invalid_penalty", 0.0)), 6),
            "drawdown_penalty": round(float(rc.get("drawdown_penalty", 0.0)), 6),
            "action_anchor_penalty": round(float(rc.get("action_anchor_penalty", 0.0)), 6),
            "symmetry_penalty": round(float(rc.get("symmetry_penalty", 0.0)), 6),
            "action_entropy_penalty": round(float(rc.get("action_entropy_penalty", 0.0)), 6),
            "saturation_penalty": round(float(rc.get("saturation_penalty", 0.0)), 6),
            "pnl_reward": round(float(rc.get("pnl_reward", 0.0)), 6),
            "pnl": round(float(rc.get("pnl", 0.0)), 6),
        }
        out.write(json.dumps(line) + "\n")

        agg["steps"] += 1
        agg["sum_final"] += float(reward)
        agg["sum_raw"] += float(rc.get("raw", 0.0))
        agg["route_counts"][DIR_LABEL[route_theory]] += 1
        if reject_reason:
            agg["reject_counts"][reject_reason] += 1
        agg["by_route_reward"][DIR_LABEL[route_theory]] += float(reward)
        agg["by_route_n"][DIR_LABEL[route_theory]] += 1
        for k in ("behavior_penalty", "drawdown_penalty", "action_anchor_penalty",
                  "symmetry_penalty", "action_entropy_penalty", "saturation_penalty",
                  "pnl_reward"):
            agg["comp_sums"][k] += float(rc.get(k, 0.0))

        if term or trunc:
            env.reset(seed=seed + 1000 + i)
    # moyennes
    n = max(1, agg["steps"])
    agg["mean_final"] = agg["sum_final"] / n
    agg["mean_raw"] = agg["sum_raw"] / n
    agg["mean_by_route"] = {
        k: (agg["by_route_reward"][k] / agg["by_route_n"][k])
        for k in agg["by_route_n"] if agg["by_route_n"][k] > 0
    }
    # sérialisation propre
    agg["route_counts"] = dict(agg["route_counts"])
    agg["reject_counts"] = dict(agg["reject_counts"])
    agg["by_route_reward"] = dict(agg["by_route_reward"])
    agg["by_route_n"] = dict(agg["by_route_n"])
    agg["comp_sums"] = {k: round(v, 6) for k, v in agg["comp_sums"].items()}
    return agg


def probe_penalty_as_function(out) -> dict:
    """Sonde les pénalités comme FONCTIONS: err=1,2,4,8 -> P1,P2,P4,P8.

    Teste le symlog EFFECTIF (final=sign*log1p(|raw|)) et les formes
    quadratiques annoncées, pour révéler ce qui est réellement appliqué.
    """
    def symlog(x):
        return math.copysign(math.log1p(abs(x)), x)

    errs = [1.0, 2.0, 4.0, 8.0]
    results = {"errs": errs}

    # (a) symlog pur sur une pénalité linéaire (raw = -k*err)
    lin = [symlog(-e) for e in errs]
    # (b) symlog sur une pénalité quadratique (raw = -err^2)
    quad = [symlog(-(e * e)) for e in errs]
    # (c) action_anchor_penalty effective: -min(cap=0.02, lambda*(excess^2))
    #     excess = |a0|-deadzone(0.30); on prend a0 = err normalisé pour la forme
    cap, lam, dz = 0.02, 0.05, 0.30
    def anchor(a0):
        excess = abs(a0) - dz
        if excess <= 0:
            return 0.0
        return -min(cap, lam * (excess * excess))
    anch = [anchor(min(1.0, 0.3 + e * 0.1)) for e in errs]  # a0 croissant

    def ratios(vals):
        r = []
        for j in range(1, len(vals)):
            prev = vals[j - 1]
            r.append(round(vals[j] / prev, 4) if prev not in (0.0,) else None)
        return r

    results["symlog_linear"] = {"P": [round(v, 6) for v in lin], "ratios_P2P1_P4P2_P8P4": ratios(lin)}
    results["symlog_quadratic"] = {"P": [round(v, 6) for v in quad], "ratios": ratios(quad)}
    results["action_anchor_effective"] = {"P": [round(v, 6) for v in anch], "ratios": ratios(anch)}
    out.write(json.dumps({"type": "penalty_function_probe", **results}) + "\n")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", nargs="+", default=["train"],
                    help="data splits/chunks à rejouer (proxy haussier/baissier)")
    ap.add_argument("--disable-ev-gate", action="store_true",
                    help="ADAN_DISABLE_EV_FEE_GATE=1 (mode advisory: BUY non bloqué)")
    ap.add_argument("--sl-raw", type=float, default=1.0,
                    help="valeur brute SL injectée (|x|>=0.9 => saturation_penalty)")
    ap.add_argument("--tp-raw", type=float, default=1.0,
                    help="valeur brute TP injectée (|x|>=0.9 => saturation_penalty)")
    args = ap.parse_args()

    if args.disable_ev_gate:
        os.environ["ADAN_DISABLE_EV_FEE_GATE"] = "1"
        print("[probe] EV fee gate DISABLED (advisory) — BUY peut s'exécuter", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "logs" / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"probe_env_{ts}.jsonl"

    sequences = ["hold", "buy", "sell", "alt", "random"]
    summary = {"ts": ts, "steps": args.steps, "seed": args.seed,
               "sl_raw": args.sl_raw, "tp_raw": args.tp_raw,
               "thresholds": THRESHOLDS, "runs": []}

    with open(out_path, "w") as out:
        # 1) sonde des pénalités comme fonctions (indépendant du marché)
        pen = probe_penalty_as_function(out)
        summary["penalty_function_probe"] = pen

        # 2) séquences déterministes sur chaque split (chunk)
        for split in args.splits:
            print(f"[probe] build env split={split} ...", flush=True)
            env, cfg, thr = build_env(split, args.seed)
            print(f"[probe] thr(5m)={thr}", flush=True)
            for seq in sequences:
                env.reset(seed=args.seed)
                agg = run_sequence(env, thr, seq, args.steps, args.seed, out,
                                   sl_raw=args.sl_raw, tp_raw=args.tp_raw)
                agg["split"] = split
                summary["runs"].append(agg)
                print(f"[probe] split={split} seq={seq:6s} "
                      f"mean_final={agg['mean_final']:+.5f} "
                      f"routes={agg['route_counts']} "
                      f"rejects={agg['reject_counts']}", flush=True)

        out.write(json.dumps({"type": "summary", **summary}) + "\n")

    # verdict summary séparé
    verdict_path = out_dir / f"probe_verdict_{ts}.json"
    with open(verdict_path, "w") as vf:
        json.dump(summary, vf, indent=2, default=str)
    print(f"[probe] JSONL: {out_path}")
    print(f"[probe] VERDICT: {verdict_path}")


if __name__ == "__main__":
    main()
