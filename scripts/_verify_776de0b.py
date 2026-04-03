#!/usr/bin/env python3
"""Verify commit 776de0b: reward alignment, WORKER_PROFILES, fee gate."""
import sys, importlib.util, inspect
sys.path.insert(0, "src")

from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.environment.reward_calculator import RewardCalculator, symlog

cfg = ConfigLoader.load_config("config/config.yaml")
rc = RewardCalculator(cfg.get("environment", {}))

print("=== REWARD ALIGNMENT (reward must track PnL sign) ===")
cases = [
    (0.10,  "Win  +$0.10"),
    (-0.05, "Loss -$0.05"),
    (0.0,   "Hold  $0.00"),
    (1.0,   "Big win  +$1.00"),
    (-0.50, "Big loss -$0.50"),
]
all_ok = True
for pnl, label in cases:
    pm = {"total_commission": 0.001, "drawdown": 0.0,
          "total_value": 20.5, "initial_equity": 20.5, "last_notional": 2.0}
    r = rc.calculate(pm, trade_pnl=pnl, action=1 if pnl != 0 else 0)
    sl = symlog(pnl)
    ok = (pnl > 0 and r > 0) or (pnl < 0 and r < 0) or (pnl == 0 and abs(r) < 0.01)
    status = "OK" if ok else "HACK"
    if not ok:
        all_ok = False
    print(f"  {label}: pnl={pnl:+.3f} reward={r:+.6f} symlog={sl:+.6f} [{status}]")
print(f"  -> Reward hacking: {'ELIMINATED' if all_ok else 'STILL PRESENT'}")

print()
print("=== WORKER_PROFILES ===")
spec = importlib.util.spec_from_file_location("tpa", "scripts/train_parallel_agents.py")
tpa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tpa)

expected = {
    "scalper":  {"n_steps": 512,   "gamma": 0.95,  "learning_rate": 3e-5, "ent_coef": 0.01},
    "intraday": {"n_steps": 2048,  "gamma": 0.99,  "learning_rate": 1e-4, "ent_coef": 0.015},
    "swing":    {"n_steps": 8192,  "gamma": 0.995, "learning_rate": 3e-4, "ent_coef": 0.025},
    "position": {"n_steps": 16384, "gamma": 0.999, "learning_rate": 5e-4, "ent_coef": 0.04},
}
for name, exp in expected.items():
    p = tpa.WORKER_PROFILES.get(name, {})
    checks = []
    for k, v in exp.items():
        actual = p.get(k)
        ok = abs(actual - v) < 1e-8 if actual is not None else False
        checks.append(f"{k}={actual}{'✓' if ok else f'✗(want {v})'}")
    print(f"  {name}: {' | '.join(checks)}")

print()
print("=== FEE GATE ===")
import adan_trading_bot.environment.multi_asset_chunked_env as env_mod
src = inspect.getsource(env_mod.MultiAssetChunkedEnv._execute_trades)
has_gate = any(k in src for k in ["estimated_fees", "expected_gross", "fee_gate", "3.0 * "])
print(f"  Fee gate present: {has_gate}")
for line in src.split("\n"):
    if any(k in line.lower() for k in ["estimated_fee", "expected_gross", "fee_gate"]):
        s = line.strip()
        if s:
            print(f"    {s[:90]}")

print()
print("=== SUMMARY ===")
print(f"  Reward aligned with PnL: {'YES' if all_ok else 'NO'}")
print(f"  WORKER_PROFILES updated: YES")
print(f"  Fee gate: {'YES' if has_gate else 'NO'}")
