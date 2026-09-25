"""Action-vs-state probe — is the policy choosing legal actions?

For each step we read:
  - the model's REQUESTED discrete action (env._last_discrete_action_requested:
        0=HOLD, 1=BUY, 2=SELL), captured AFTER step() each tick
  - whether the env currently holds a position (is_open)

We build a confusion matrix action x state to test the hypothesis:
  "the policy outputs SELL while flat and BUY while full" (wrong-for-state).
If most steps fall on the 'illegal' diagonal, the policy never learned the
state->legal-action mapping and wastes gradient on impossible moves.

Usage:
  PYTHONPATH=src python3 scripts/backtest/action_state_probe.py \
      --ckpt checkpoints/ppo_adan0_sandbox_checkpoint_430000_steps.zip \
      --split test --steps 3000 --out logs/validation/forensic/probe_430k.json
"""
from __future__ import annotations
import argparse, copy, json, logging, os, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("ADAN_TRAINING_SILENT", "1")
os.environ.setdefault("ADAN_RICH_STEP_EVERY", "999999")
logging.disable(logging.WARNING)


def _is_open(pm) -> bool:
    try:
        positions = getattr(pm, "positions", None)
        if isinstance(positions, dict):
            for p in positions.values():
                if p is None:
                    continue
                if bool(getattr(p, "is_open", False)):
                    return True
                if float(getattr(p, "size", 0) or 0) > 0:
                    return True
            return False
    except Exception:
        pass
    # fallback: exposure
    try:
        return float(getattr(pm, "get_total_exposure", lambda: 0.0)()) > 1e-9
    except Exception:
        return False


def run_probe(ckpt: str, steps: int, split: str) -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from adan_trading_bot.common.config_loader import ConfigLoader
    from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
    from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

    cfg = ConfigLoader.load_config(str(REPO_ROOT / "config" / "config.yaml"))
    cfg.setdefault("environment", {})["rich_display_interval"] = 999999
    wc = copy.deepcopy(cfg.get("workers", {}).get("w1", {}))
    wc.update({"worker_id": 0, "data_split": split, "data_split_override": split,
               "timeframes": ["5m", "1h", "4h"], "assets": ["BTCUSDT"]})
    data = ChunkedDataLoader(config=cfg, worker_config=wc, worker_id=0).load_chunk(0)
    env = MultiAssetChunkedEnv(data=data, config=cfg, worker_config=wc,
                               worker_id=0, live_mode=False)
    vec = DummyVecEnv([lambda: env])
    model = PPO.load(ckpt, device="cpu"); model.set_env(vec)
    obs = vec.reset()
    u = vec.envs[0]; pm = u.portfolio_manager

    # action(req) x state(open?) confusion
    names = {0: "HOLD", 1: "BUY", 2: "SELL"}
    conf = {a: {"flat": 0, "open": 0} for a in (0, 1, 2)}
    cont = []  # raw action[0]
    n_eps = 0
    for s in range(steps):
        open_before = _is_open(pm)
        action, _ = model.predict(obs, deterministic=True)
        cont.append(float(np.ravel(action)[0]))
        obs, _r, dones, _i = vec.step(action)
        req = int(getattr(u, "_last_discrete_action_requested", 0) or 0)
        conf[req]["open" if open_before else "flat"] += 1
        if bool(np.ravel(dones)[0]):
            n_eps += 1; obs = vec.reset(); u = vec.envs[0]; pm = u.portfolio_manager

    total = sum(conf[a][k] for a in conf for k in conf[a])
    # legal moves: HOLD always legal; BUY legal when flat; SELL legal when open
    legal = conf[0]["flat"] + conf[0]["open"] + conf[1]["flat"] + conf[2]["open"]
    illegal = conf[1]["open"] + conf[2]["flat"]
    rej = dict(getattr(u, "rejection_reasons", {}) or {})
    carr = np.array(cont)

    # Hypothesis A test: is action0 oscillating in a narrow band near the
    # decision threshold (so discretization flips it artificially)?
    thr = 0.01  # action_threshold used by the env
    near_thr = float((np.abs(np.abs(carr) - thr) <= 0.05).mean())
    # histogram of action0 in 10 bins over [-1,1]
    hist, edges = np.histogram(carr, bins=10, range=(-1.0, 1.0))
    histo = {f"[{edges[i]:+.1f},{edges[i+1]:+.1f})": int(hist[i])
             for i in range(len(hist))}

    # When FLAT, what does the agent prefer? HOLD vs SELL(illegal) vs BUY(legal)
    flat_total = conf[0]["flat"] + conf[1]["flat"] + conf[2]["flat"]
    open_total = conf[0]["open"] + conf[1]["open"] + conf[2]["open"]
    flat_breakdown = {
        "HOLD_pct": round(conf[0]["flat"] / max(1, flat_total), 3),
        "BUY_legal_pct": round(conf[1]["flat"] / max(1, flat_total), 3),
        "SELL_illegal_pct": round(conf[2]["flat"] / max(1, flat_total), 3),
    }
    open_breakdown = {
        "HOLD_pct": round(conf[0]["open"] / max(1, open_total), 3),
        "BUY_illegal_pct": round(conf[1]["open"] / max(1, open_total), 3),
        "SELL_legal_pct": round(conf[2]["open"] / max(1, open_total), 3),
    }
    return {
        "checkpoint": os.path.basename(ckpt), "split": split, "steps": steps,
        "episodes": n_eps,
        "confusion": {names[a]: conf[a] for a in conf},
        "steps_flat": flat_total, "steps_open": open_total,
        "flat_action_breakdown": flat_breakdown,
        "open_action_breakdown": open_breakdown,
        "legal_steps": legal, "illegal_steps": illegal,
        "illegal_ratio": round(illegal / max(1, total), 4),
        "action0_mean": round(float(carr.mean()), 4),
        "action0_std": round(float(carr.std()), 4),
        "action0_pct_sell_zone": round(float((carr < -thr).mean()), 4),
        "action0_pct_buy_zone": round(float((carr > thr).mean()), 4),
        "action0_pct_hold_zone": round(float((np.abs(carr) <= thr).mean()), 4),
        "action0_pct_near_threshold_band": round(near_thr, 4),
        "action0_histogram": histo,
        "rejection_reasons": rej,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    print(f"[probe] {a.ckpt} steps={a.steps}", file=sys.stderr)
    r = run_probe(a.ckpt, a.steps, a.split)
    out = Path(a.out) if a.out else (REPO_ROOT / "logs/validation/forensic" /
                                     f"probe_{Path(a.ckpt).stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
