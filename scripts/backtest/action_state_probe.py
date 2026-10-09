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


def semantic_policy_intent(raw_a0: float, in_position: bool, threshold: float) -> str:
    """Name the policy intent without confusing V12 routing with execution."""
    if in_position:
        return "SELL" if raw_a0 < -threshold else "HOLD_POSITION"
    if raw_a0 > threshold:
        return "BUY"
    if abs(raw_a0) <= threshold:
        return "WAIT"
    return "NO_ENTRY_RAW"


def classify_transition(
    *,
    intent: str,
    routed: int,
    executed: int,
    gate_reasons: list[str],
    close_reason: str = "",
) -> str:
    """Classify patience, forced waiting, holding, and autonomous trades."""
    close = close_reason.upper()
    if executed == 2:
        for marker, category in (
            ("MAX_DURATION", "MAX_DURATION_CLOSE"),
            ("MAXDURATION", "MAX_DURATION_CLOSE"),
            ("TP", "TP_CLOSE"),
            ("TAKE", "TP_CLOSE"),
            ("SL", "SL_CLOSE"),
            ("STOP", "SL_CLOSE"),
        ):
            if marker in close:
                return category
        return "AGENT_CLOSE" if routed == 2 else "MARKET_CLOSE"
    if executed == 1:
        return "BUY_EXECUTED"
    if routed == 1:
        return "BLOCKED_BUY" if gate_reasons else "BUY_NOT_EXECUTED"
    if routed == 2:
        return "BLOCKED_SELL" if gate_reasons else "SELL_NOT_EXECUTED"
    if intent == "HOLD_POSITION":
        return "VALID_HOLD"
    if intent == "WAIT":
        return "PATIENT_WAIT"
    if intent == "NO_ENTRY_RAW":
        return "NO_ENTRY_RAW"
    return "ROUTED_HOLD"


def run_probe(
    ckpt: str,
    steps: int,
    split: str,
    capture_trace: bool = False,
    deterministic: bool = True,
    seed: int = 0,
) -> dict:
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
    model.set_random_seed(seed)
    obs = vec.reset()
    u = vec.envs[0]; pm = u.portfolio_manager

    # action(req) x state(open?) confusion
    names = {0: "HOLD", 1: "BUY", 2: "SELL"}
    conf = {a: {"flat": 0, "open": 0} for a in (0, 1, 2)}
    cont = []  # continuous policy action[0]
    trace = []
    category_counts: dict[str, int] = {}
    intent_counts: dict[str, int] = {}
    route_execution_counts: dict[str, int] = {}
    n_eps = 0
    thr = float(
        cfg.get("trading_rules", {}).get("frequency", {}).get("action_threshold", 0.01)
    )
    for s in range(steps):
        open_before = _is_open(pm)
        action, _ = model.predict(obs, deterministic=deterministic)
        raw_a0 = float(np.ravel(action)[0])
        cont.append(raw_a0)
        intent = semantic_policy_intent(raw_a0, open_before, thr)
        rejection_before = dict(getattr(u, "rejection_reasons", {}) or {})
        portfolio_before = np.asarray(obs["portfolio_state"]).reshape(-1)
        cooldown_remaining = (
            float(portfolio_before[26]) if portfolio_before.size > 26 else 0.0
        )
        time_in_position = (
            float(portfolio_before[12]) if portfolio_before.size > 12 else 0.0
        )
        last_sell = getattr(u, "_last_sell_step_by_asset", {}).get("BTCUSDT")
        time_since_sell = None if last_sell is None else int(u.current_step - last_sell)

        obs, rewards, dones, _i = vec.step(action)
        routed = int(getattr(u, "_last_route_action", 0) or 0)
        executed = int(getattr(u, "_last_discrete_action", 0) or 0)
        rejection_after = dict(getattr(u, "rejection_reasons", {}) or {})
        gate_reasons = sorted(
            key for key, value in rejection_after.items()
            if int(value) > int(rejection_before.get(key, 0))
        )
        closed = list(getattr(u, "_step_closed_receipts", []) or [])
        close_reason = ""
        if closed and isinstance(closed[0], dict):
            close_reason = str(closed[0].get("reason", closed[0].get("close_reason", "")))
        category = classify_transition(
            intent=intent,
            routed=routed,
            executed=executed,
            gate_reasons=gate_reasons,
            close_reason=close_reason,
        )
        req = int(getattr(u, "_last_discrete_action_requested", routed) or 0)
        conf[req]["open" if open_before else "flat"] += 1
        category_counts[category] = category_counts.get(category, 0) + 1
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        route_exec_key = f"{names.get(routed, routed)}->{names.get(executed, executed)}"
        route_execution_counts[route_exec_key] = route_execution_counts.get(route_exec_key, 0) + 1
        if capture_trace:
            trace.append({
                "step": int(s),
                "env_step": int(getattr(u, "current_step", s)),
                "position_state": "LONG" if open_before else "FLAT",
                "raw_a0": raw_a0,
                "policy_intent": intent,
                "routed_action": names.get(routed, str(routed)),
                "executed_action": names.get(executed, str(executed)),
                "category": category,
                "cooldown_remaining": cooldown_remaining,
                "time_in_position_normalized": time_in_position,
                "time_since_sell": time_since_sell,
                "gate_reasons": gate_reasons,
                "close_reason": close_reason,
                "reward": float(np.ravel(rewards)[0]),
                "realized_pnl": float(getattr(u, "_step_realized_pnl", 0.0) or 0.0),
            })
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
        "deterministic": deterministic, "seed": seed,
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
        "semantic_intents": intent_counts,
        "transition_categories": category_counts,
        "route_execution": route_execution_counts,
        "trace": trace if capture_trace else None,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--out", default=None)
    p.add_argument(
        "--trace-out",
        default=None,
        help="Optional JSONL path for per-step intent/route/gate/execution telemetry.",
    )
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample the trained PPO distribution instead of using its mean action.",
    )
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    print(f"[probe] {a.ckpt} steps={a.steps}", file=sys.stderr)
    r = run_probe(
        a.ckpt,
        a.steps,
        a.split,
        capture_trace=bool(a.trace_out),
        deterministic=not a.stochastic,
        seed=a.seed,
    )
    out = Path(a.out) if a.out else (REPO_ROOT / "logs/validation/forensic" /
                                     f"probe_{Path(a.ckpt).stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    trace = r.pop("trace", None)
    out.write_text(json.dumps(r, indent=2))
    if a.trace_out and trace is not None:
        trace_out = Path(a.trace_out)
        trace_out.parent.mkdir(parents=True, exist_ok=True)
        trace_out.write_text("".join(json.dumps(row) + "\n" for row in trace))
    print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
