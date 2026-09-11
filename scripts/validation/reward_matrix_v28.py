"""Synthetic exhaustive reward matrix — post V28 fix validation.

Validates the user's 9-row specification (state x action x situation) after
the additive invalid-intent penalty wiring (reward_shaping.behavior_penalties):

  | Position | Action | Situation       | Expected signal                    |
  |----------|--------|-----------------|------------------------------------|
  | FLAT     | BUY    | ouverture       | faible cout / ~neutre (NO invalid) |
  | FLAT     | SELL   | invalide        | forte penalite (-0.28 additive)    |
  | FLAT     | HOLD   | attente         | faible (time_decay only)           |
  | OPEN     | BUY    | invalide        | forte penalite (-0.28 additive)    |
  | OPEN     | SELL   | perte           | negatif economique                 |
  | OPEN     | SELL   | profit partiel  | positif, < TP                      |
  | OPEN     | SELL   | TP              | maximal                            |
  | OPEN     | HOLD   | trade gagnant   | latent positif                     |
  | OPEN     | HOLD   | trade perdant   | latent negatif                     |

Success criterion (user): the economically best action gets the best signal,
regardless of its BUY/SELL/HOLD label. We assert:
  1. FLAT/SELL invalid  -> behavior_invalid_penalty == sell_while_flat (< 0)
  2. OPEN/BUY invalid   -> behavior_invalid_penalty == buy_while_open (< 0)
  3. FLAT/BUY valid     -> behavior_invalid_penalty == 0 (no entry penalty)
  4. Invalid intents are strictly WORSE than the same-state HOLD baseline.

Forcing: we override only a[0] (direction), keep policy-like size/sl/tp.
To reach OPEN state deterministically we first force a valid BUY (flat),
confirm the position opened, then run the OPEN-state scenario. Market path
is the REAL val chunk (same data the 320k autopsy used) — not mocked.

Usage:
  python scripts/validation/reward_matrix_v28.py [--split val]
"""
from __future__ import annotations

import copy
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("ADAN_TRAINING_SILENT", "1")
os.environ.setdefault("ADAN_RICH_STEP_EVERY", "999999")
logging.disable(logging.WARNING)

OUT_DIR = REPO_ROOT / "logs" / "validation" / "reward_matrix_v28"
SELL_THRESHOLD = 0.05
BUY_THRESHOLD = 0.05


def build_env(cfg, wc, worker_idx=0):
    from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
    from adan_trading_bot.environment.multi_asset_chunked_env import (
        MultiAssetChunkedEnv,
    )
    data = ChunkedDataLoader(
        config=cfg, worker_config=wc, worker_id=worker_idx
    ).load_chunk(0)
    return MultiAssetChunkedEnv(
        data=data, config=cfg, worker_config=wc, worker_id=worker_idx,
        live_mode=False,
    )


def _fresh_env(cfg, wc):
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = build_env(cfg, wc)
    vec = DummyVecEnv([lambda: env])
    obs = vec.reset()
    return env, vec, obs


def _action(direction: float, size: float = 0.6, sl: float = 0.3,
            tp: float = 0.5) -> np.ndarray:
    """5-dim action for BTCUSDT-only env: [dir, size, tf, sl, tp]."""
    return np.array([direction, size, 0.0, sl, tp], dtype=np.float32)


def _step(vec, a):
    obs, r, dones, infos = vec.step(a.reshape(1, -1))
    info = infos[0] if isinstance(infos, (list, tuple)) and infos else (infos or {})
    return obs, float(np.asarray(r).flat[0]), bool(np.asarray(dones).flat[0]), info


def _rc(env) -> dict:
    return dict(getattr(env, "_last_reward_components", {}) or {})


def _open_pos_count(env) -> int:
    try:
        return sum(
            1 for p in env.portfolio_manager.positions.values()
            if getattr(p, "is_open", False)
        )
    except Exception:
        return -1


def _behavior_pen(env) -> float:
    return float(getattr(env, "_behavior_penalty_step", 0.0))


def run_flat_scenarios(cfg, wc) -> dict:
    """FLAT state: BUY (valid), SELL (invalid), HOLD (baseline)."""
    out = {}

    # FLAT + SELL (invalid) — expect additive penalty sell_while_flat
    env, vec, obs = _fresh_env(cfg, wc)
    pens, raws = [], []
    for _ in range(30):
        obs, r, done, info = _step(vec, _action(-1.0))
        pens.append(_behavior_pen(env))
        raws.append(r)
        if done:
            obs = vec.reset()
    out["FLAT_SELL"] = {
        "mean_behavior_invalid_penalty": float(np.mean(pens)),
        "penalized_values": sorted({round(float(x), 6) for x in pens if x < 0}),
        "n_penalized": int(sum(1 for x in pens if x < 0)),
        "n": len(pens),
        "mean_raw_reward": float(np.mean(raws)),
    }

    # FLAT + HOLD (baseline) — expect NO invalid penalty
    env, vec, obs = _fresh_env(cfg, wc)
    pens, raws = [], []
    for _ in range(30):
        obs, r, done, info = _step(vec, _action(0.0))
        pens.append(_behavior_pen(env))
        raws.append(r)
        if done:
            obs = vec.reset()
    out["FLAT_HOLD"] = {
        "mean_behavior_invalid_penalty": float(np.mean(pens)),
        "n_penalized": int(sum(1 for x in pens if x < 0)),
        "n": len(pens),
        "mean_raw_reward": float(np.mean(raws)),
    }

    # FLAT + BUY (valid opening) — expect NO invalid penalty, position opens
    # (persevere up to 500 steps: the EV fee gate passes ~2% of steps on val)
    env, vec, obs = _fresh_env(cfg, wc)
    pens, raws, opened = [], [], 0
    for _ in range(500):
        obs, r, done, info = _step(vec, _action(1.0, size=0.9, sl=0.3, tp=1.0))
        pens.append(_behavior_pen(env))
        raws.append(r)
        opened = max(opened, _open_pos_count(env))
        if opened > 0:
            break
        if done:
            obs = vec.reset()
    out["FLAT_BUY"] = {
        "mean_behavior_invalid_penalty": float(np.mean(pens)),
        "n_penalized": int(sum(1 for x in pens if x < 0)),
        "n": len(pens),
        "mean_raw_reward": float(np.mean(raws)),
        "position_opened": int(opened),
    }
    return out


def _force_open_position(cfg, wc, max_tries: int = 500):
    """Open a position by forcing valid BUY from FLAT. Returns (env,vec,obs)
    with an open position, or None.

    NOTE: the EV fee gate passes only ~2% of steps on the val window (arm C
    of the counterfactual battery: 59 trades / 3000 steps), so we must
    persevere. The forced BUY uses a wide TP (tp_raw=1.0) to clear the
    TP >= 3x fees gate as often as possible.
    """
    env, vec, obs = _fresh_env(cfg, wc)
    for _ in range(max_tries):
        obs, r, done, info = _step(vec, _action(1.0, size=0.9, sl=0.3, tp=1.0))
        if done:
            obs = vec.reset()
        if _open_pos_count(env) > 0:
            return env, vec, obs
    return None


def _close_receipts(env) -> list:
    """Best-effort harvest of close receipts (reason + pnl) from the env."""
    recs = []
    for attr in ("trade_receipts", "_trade_receipts", "closed_trades",
                 "_closed_trades", "trade_log", "_trade_log"):
        v = getattr(env, attr, None)
        if isinstance(v, (list, tuple)) and v:
            for r in v:
                if isinstance(r, dict) and ("pnl" in r or "realized_pnl" in r):
                    recs.append(r)
    # fallback: portfolio manager trade history
    try:
        pm = env.portfolio_manager
        for attr in ("trade_history", "closed_positions", "trades"):
            v = getattr(pm, attr, None)
            if isinstance(v, (list, tuple)) and v:
                for r in v:
                    d = r if isinstance(r, dict) else getattr(r, "__dict__", {})
                    if isinstance(d, dict) and (
                            "pnl" in d or "realized_pnl" in d or "net_pnl" in d):
                        recs.append(d)
    except Exception:
        pass
    return recs


def run_open_scenarios(cfg, wc) -> dict:
    """OPEN state: BUY (invalid), SELL (valid close), HOLD (baseline).

    Economic sub-rows (SELL perte / profit / TP, HOLD gagnant / perdant) are
    classified from observed close receipts. V28 does NOT modify the economic
    layer, so these rows are informational: we verify ordering when observed,
    and mark 'not_observed' otherwise (never a hard FAIL).
    """
    out = {}

    # OPEN + BUY (invalid reinforcement) — expect additive buy_while_open.
    # Hold a single position and spam BUY for 30 steps.
    got = _force_open_position(cfg, wc)
    if got:
        env, vec, obs = got
        pens, raws = [], []
        for _ in range(30):
            if _open_pos_count(env) == 0:
                break
            obs, r, done, info = _step(vec, _action(1.0, size=0.9, sl=0.3, tp=1.0))
            pens.append(_behavior_pen(env))
            raws.append(r)
            if done:
                obs = vec.reset()
                break
        out["OPEN_BUY"] = {
            "mean_behavior_invalid_penalty": float(np.mean(pens)) if pens else None,
            "penalized_values": sorted({round(float(x), 6) for x in pens if x < 0}),
            "n_penalized": int(sum(1 for x in pens if x < 0)),
            "n": len(pens),
            "mean_raw_reward": float(np.mean(raws)) if raws else None,
        }
    else:
        out["OPEN_BUY"] = {"error": "could_not_open_position"}

    # OPEN + HOLD (baseline) — expect NO invalid penalty; capture latent signal
    got = _force_open_position(cfg, wc)
    if got:
        env, vec, obs = got
        pens, raws, latents = [], [], []
        for _ in range(30):
            if _open_pos_count(env) == 0:
                break
            obs, r, done, info = _step(vec, _action(0.0))
            pens.append(_behavior_pen(env))
            raws.append(r)
            latents.append(float(_rc(env).get("latent_pnl", 0.0)))
            if done:
                obs = vec.reset()
                break
        out["OPEN_HOLD"] = {
            "mean_behavior_invalid_penalty": float(np.mean(pens)) if pens else None,
            "n_penalized": int(sum(1 for x in pens if x < 0)),
            "n": len(pens),
            "mean_raw_reward": float(np.mean(raws)) if raws else None,
            "mean_latent_pnl": float(np.mean(latents)) if latents else None,
        }
    else:
        out["OPEN_HOLD"] = {"error": "could_not_open_position"}

    # OPEN + SELL (valid close) — expect NO invalid penalty (economic only).
    # Bucket observed closes by realized pnl sign and close reason.
    got = _force_open_position(cfg, wc)
    if got:
        env, vec, obs = got
        pens, raws = [], []
        close_events = []
        prev_closed = len(getattr(env, "_step_closed_receipts", []))
        all_recs_seen = 0
        steps_budget = 800
        for _ in range(steps_budget):
            # detect a close THIS step via the step receipts list length
            cur_recs = list(getattr(env, "_step_closed_receipts", []))
            if len(cur_recs) > 0 and len(getattr(env, "_all_episode_receipts", [])) > all_recs_seen:
                all_recs_seen = len(getattr(env, "_all_episode_receipts", []))
                last = cur_recs[-1]
                pnl = last.get("pnl", last.get("realized_pnl", last.get("net_pnl")))
                reason = last.get("reason", last.get("close_reason", last.get("exit_reason")))
                close_events.append({
                    "reward_at_close": raws[-1] if raws else None,
                    "pnl": (float(pnl) if isinstance(pnl, (int, float)) else None),
                    "reason": reason,
                })
            if _open_pos_count(env) == 0:
                # re-open perseverantly to keep sampling closes
                obs, r, done, info = _step(vec, _action(1.0, size=0.9, sl=0.3, tp=1.0))
                if done:
                    obs = vec.reset()
                continue
            obs, r, done, info = _step(vec, _action(-1.0))
            pens.append(_behavior_pen(env))
            raws.append(r)
            if done:
                obs = vec.reset()
        wins = [e for e in close_events if isinstance(e.get("pnl"), float) and e["pnl"] > 0]
        losses = [e for e in close_events if isinstance(e.get("pnl"), float) and e["pnl"] <= 0]
        out["OPEN_SELL"] = {
            "mean_behavior_invalid_penalty": float(np.mean(pens)) if pens else None,
            "n_penalized": int(sum(1 for x in pens if x < 0)),
            "n": len(pens),
            "mean_raw_reward": float(np.mean(raws)) if raws else None,
            "closes_observed": len(close_events),
            "close_events": close_events[:20],
            "win_close_mean_reward": float(np.mean(
                [e["reward_at_close"] for e in wins if e["reward_at_close"] is not None]))
                if wins else None,
            "loss_close_mean_reward": float(np.mean(
                [e["reward_at_close"] for e in losses if e["reward_at_close"] is not None]))
                if losses else None,
        }
    else:
        out["OPEN_SELL"] = {"error": "could_not_open_position"}
    return out


def grade(matrix: dict, bp: dict) -> dict:
    """Assert the success criterion."""
    sell_pen = float(bp.get("sell_while_flat", -0.28))
    buy_pen = float(bp.get("buy_while_open", -0.28))
    checks = {}

    fs = matrix.get("FLAT_SELL", {})
    fh = matrix.get("FLAT_HOLD", {})
    fb = matrix.get("FLAT_BUY", {})
    ob = matrix.get("OPEN_BUY", {})
    oh = matrix.get("OPEN_HOLD", {})
    osc = matrix.get("OPEN_SELL", {})

    # 1. FLAT/SELL invalid carries the configured additive penalty
    fs_vals = fs.get("penalized_values", [])
    checks["flat_sell_penalized"] = (
        fs.get("n_penalized", 0) > 0
        and len(fs_vals) == 1
        and abs(fs_vals[0] - sell_pen) < 1e-6
    )
    # 2. OPEN/BUY invalid carries the configured additive penalty.
    #    Compare the PER-STEP penalized value (not the mean): a step where the
    #    position closed via SL mid-scenario then BUY-flat is VALID (no pen)
    #    must not dilute the check.
    ob_vals = ob.get("penalized_values", [])
    checks["open_buy_penalized"] = (
        ob.get("n_penalized", 0) > 0
        and len(ob_vals) == 1
        and abs(ob_vals[0] - buy_pen) < 1e-6
    )
    # 3. Valid intents are NOT hit by the invalid penalty
    checks["flat_buy_no_invalid_pen"] = fb.get("n_penalized", -1) == 0
    checks["flat_hold_no_invalid_pen"] = fh.get("n_penalized", -1) == 0
    checks["open_sell_no_invalid_pen"] = osc.get("n_penalized", -1) == 0
    checks["open_hold_no_invalid_pen"] = oh.get("n_penalized", -1) == 0
    # 4. Invalid is strictly worse than same-state baseline HOLD
    checks["flat_sell_worse_than_hold"] = (
        fs.get("mean_raw_reward", 0.0) < fh.get("mean_raw_reward", 0.0)
    )
    checks["open_buy_worse_than_hold"] = (
        ob.get("mean_raw_reward") is not None
        and oh.get("mean_raw_reward") is not None
        and ob["mean_raw_reward"] < oh["mean_raw_reward"]
    )
    # 5. A valid BUY could actually open (routing not broken)
    checks["flat_buy_opened"] = fb.get("position_opened", 0) > 0

    # Economic-layer ordering (informational — V28 does not modify it):
    # a winning close must out-signal a losing close when both observed.
    info_checks = {}
    wr = osc.get("win_close_mean_reward")
    lr = osc.get("loss_close_mean_reward")
    if wr is not None and lr is not None:
        info_checks["win_close_beats_loss_close"] = wr > lr
    else:
        info_checks["win_close_beats_loss_close"] = "not_observed"

    passed = sum(1 for v in checks.values() if v)
    return {
        "checks": checks,
        "economic_layer_informational": info_checks,
        "passed": passed,
        "total": len(checks),
        "verdict": "PASS" if passed == len(checks) else "FAIL",
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="val")
    args = ap.parse_args()

    from adan_trading_bot.common.config_loader import ConfigLoader
    cfg = ConfigLoader.load_config(str(REPO_ROOT / "config" / "config.yaml"))
    cfg.setdefault("environment", {})["rich_display_interval"] = 999999
    wc = copy.deepcopy(cfg.get("workers", {}).get("w1", {}))
    wc.update({
        "worker_id": 0, "data_split_override": args.split,
        "timeframes": ["5m", "1h", "4h"], "assets": ["BTCUSDT"],
    })

    bp = (cfg.get("reward_shaping", {}) or {}).get("behavior_penalties", {}) or {}
    print(f"[matrix] behavior_penalties: {bp}", file=sys.stderr)

    matrix = {}
    matrix.update(run_flat_scenarios(cfg, wc))
    matrix.update(run_open_scenarios(cfg, wc))

    report = {
        "split": args.split,
        "behavior_penalties_config": bp,
        "matrix": matrix,
        "grading": grade(matrix, bp),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "reward_matrix.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["grading"]["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
