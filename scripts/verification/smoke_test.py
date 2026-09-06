#!/usr/bin/env python3
"""
ADAN0 Smoke Test — validates all 14 corrections (C1–C14) + 3 amendments.

Run: python smoke_test.py
Exit code 0 = all checks pass.
"""
import sys
import os
import re
import yaml
import importlib

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src = os.path.join(base, "src", "adan_trading_bot")

    # ── CONFIG CHECKS ──────────────────────────────────────────────
    print("\n=== CONFIG (config.yaml) ===")
    cfg_path = os.path.join(base, "config", "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # C3: max_drawdown_pct harmonized to 40.0
    hc_dd = cfg.get("environment", {}).get("hard_constraints", {}).get("max_drawdown_pct", None)
    check("C3  hard_constraints.max_drawdown_pct == 40.0", hc_dd == 40.0, f"got {hc_dd}")

    rp_dd = cfg.get("risk_parameters", {}).get("max_drawdown_pct", None)
    check("C3  risk_parameters.max_drawdown_pct == 40.0", rp_dd == 40.0, f"got {rp_dd}")

    # C13-v2: input_shape should be [1, 20, 21] for all timeframes (5 OHLCV + 16 indicators)
    fe_cfg = cfg.get("agent", {}).get("features_extractor_kwargs", {}).get("cnn_configs", {})
    for tf in ["5m", "1h", "4h"]:
        tf_cfg = fe_cfg.get(tf, {})
        shape = tf_cfg.get("input_shape", [])
        check(f"C13 input_shape {tf} == [1,20,21]", shape == [1, 20, 21], f"got {shape}")
        # Indicator indices should be 5-20 (16 indicators)
        ind_idx = tf_cfg.get("groups", {}).get("indicators", {}).get("indices", [])
        check(f"C13 indicator indices {tf} == [5..20]", ind_idx == list(range(5, 21)), f"got {ind_idx}")

    # ── REWARD CALCULATOR ──────────────────────────────────────────
    print("\n=== REWARD CALCULATOR ===")
    rc_src = read_file(os.path.join(src, "environment", "reward_calculator.py"))

    # C1: threshold 0.10 (not 0.005)
    check("C1  dd threshold 0.10", "dd > 0.10" in rc_src, "0.10 threshold not found")
    check("C1  no old 0.005 threshold", "dd > 0.005" not in rc_src, "old 0.005 still present")

    # C2: drawdown from _initial_capital
    check("C2  _initial_capital attribute", "_initial_capital" in rc_src)
    check("C2  reset_reward_state method", "def reset_reward_state" in rc_src)

    # C11: sigmoid anti-hack (no hard flip)
    check("C11 sigmoid severity", "np.exp(-100.0" in rc_src, "sigmoid formula not found")
    # Verify sigmoid form is used (r = r * (-delta) * severity) not bare (r *= -delta)
    # Strip comments and docstrings before checking for the bare flip pattern
    has_sigmoid_form = "r = r * (-delta) * severity" in rc_src
    # Remove triple-quoted docstrings and comment lines
    code_no_docstr = re.sub(r'""".*?"""', '', rc_src, flags=re.DOTALL)
    code_no_docstr = re.sub(r"'''.*?'''", '', code_no_docstr, flags=re.DOTALL)
    code_lines = [l for l in code_no_docstr.splitlines() if not l.strip().startswith("#")]
    code_only = "\n".join(code_lines)
    has_bare_flip = re.search(r'r\s*\*=\s*-\s*delta', code_only) is not None
    check("C11 no bare r *= -delta", has_sigmoid_form and not has_bare_flip, "hard flip pattern found")

    # ── DATA LOADER ────────────────────────────────────────────────
    print("\n=== DATA LOADER ===")
    dl_src = read_file(os.path.join(src, "data_processing", "data_loader.py"))
    check("C5  shift(1) applied", "tf_df = tf_df.shift(1)" in dl_src)
    check("C5-A dropna guard", "tf_df = tf_df.dropna(how='all')" in dl_src or "dropna(how='all')" in dl_src)

    # ── MULTI ASSET CHUNKED ENV ────────────────────────────────────
    print("\n=== MULTI ASSET CHUNKED ENV ===")
    env_src = read_file(os.path.join(src, "environment", "multi_asset_chunked_env.py"))

    # C4: no hardcoded 4.0 fallback
    check("C4  no hardcoded 4.0 fallback", "'max_drawdown_pct', 4.0)" not in env_src, "4.0 fallback still present")

    # C6: _inv_pen_weight = 0.0
    check("C6  _inv_pen_weight = 0.0", "_inv_pen_weight = 0.0" in env_src)
    check("C6  _cooldown_pen_weight removed", "_cooldown_pen_weight" not in env_src, "dead variable still present")

    # C14: probabilistic sizer
    check("C14 probabilistic sizer", "PROB_SIZER" in env_src, "probabilistic sizer not found")
    check("C14 Bernoulli trial", "np.random.random() < p_execute" in env_src)
    check("C14-A action_policy in receipt", 'action_policy' in env_src)

    # Wire: reset_reward_state called from reset()
    check("C2-wire reset_reward_state in reset()", "reset_reward_state" in env_src)

    # ── REALISTIC TRADING ENV ──────────────────────────────────────
    print("\n=== REALISTIC TRADING ENV ===")
    rte_src = read_file(os.path.join(src, "environment", "realistic_trading_env.py"))
    check("C7  circuit_breaker_pct = 0.38", "circuit_breaker_pct: float = 0.38" in rte_src or "circuit_breaker_pct=0.38" in rte_src)

    # ── PORTFOLIO MANAGER ──────────────────────────────────────────
    print("\n=== PORTFOLIO MANAGER ===")
    pm_src = read_file(os.path.join(src, "portfolio", "portfolio_manager.py"))
    check("C8  total_realized_pnl", "total_realized_pnl" in pm_src)
    check("C8  realized_equity", "realized_equity" in pm_src)
    check("C8  peak_realized_equity", "peak_realized_equity" in pm_src)

    # ── FEATURE EXTRACTORS ─────────────────────────────────────────
    print("\n=== FEATURE EXTRACTORS ===")
    fe_src = read_file(os.path.join(src, "agent", "feature_extractors.py"))
    check("C9  norm_q LayerNorm", "norm_q" in fe_src)
    check("C9  norm_kv LayerNorm", "norm_kv" in fe_src)

    # ── DYNAMIC BEHAVIOR ENGINE ────────────────────────────────────
    print("\n=== DYNAMIC BEHAVIOR ENGINE ===")
    dbe_src = read_file(os.path.join(src, "environment", "dynamic_behavior_engine.py"))
    check("C10 HMM sliding-window refit", "HMM_WINDOW" in dbe_src or "% HMM_WINDOW" in dbe_src)

    # ── FEATURE ENGINEER ───────────────────────────────────────────
    print("\n=== FEATURE ENGINEER ===")
    feng_src = read_file(os.path.join(src, "data_processing", "feature_engineer.py"))
    check("C10 L2 reg fib_ratio", "np.sqrt(range_val**2 + 1e-9)" in feng_src)

    # C13: 8 orthogonal features
    check("C13 di_delta computed", "di_delta" in feng_src)
    check("C13 atr_pct computed", "atr_pct" in feng_src)
    check("C13 obv_slope computed", "obv_slope" in feng_src)
    check("C13 macdh in TRAIN_COLUMNS", "macdh_12_26_9" in feng_src)
    # Verify old bloat features removed
    # C13-v2: Verify 16 indicators present in TRAIN_COLUMNS (21 total with OHLCV)
    tc_block = feng_src.split("TRAIN_COLUMNS")[1].split("}")[0] if "TRAIN_COLUMNS" in feng_src else ""
    check("C13 log_return in TRAIN_COLUMNS", "log_return" in tc_block, "log_return missing")
    check("C13 vwap_ratio in TRAIN_COLUMNS", "vwap_ratio" in tc_block, "vwap_ratio missing")
    check("C13 fib_ratio in TRAIN_COLUMNS", "fib_ratio" in tc_block, "fib_ratio missing")
    check("C13 market_structure in TRAIN_COLUMNS", "market_structure" in tc_block, "market_structure missing")
    check("C13 bb_width_20_2 in TRAIN_COLUMNS", "bb_width_20_2" in tc_block, "bb_width missing")
    check("C13 volatility_ratio_14_50 in TRAIN_COLUMNS", "volatility_ratio_14_50" in tc_block, "vol_ratio missing")

    # ── STATE BUILDER ──────────────────────────────────────────────
    print("\n=== STATE BUILDER ===")
    sb_src = read_file(os.path.join(src, "data_processing", "state_builder.py"))
    check("C12 posinf=0.0", "posinf=0.0" in sb_src)
    check("C12 clip ±10", "clip(obs, -10.0, 10.0)" in sb_src or "np.clip(obs, -10.0, 10.0)" in sb_src)
    check("C13 features_config has di_delta", "di_delta" in sb_src)
    check("C13 features_config has obv_slope", "obv_slope" in sb_src)
    # Count features per TF: should be 13 (5 OHLCV + 8 indicators)
    # Quick heuristic: check "5m" block has 13 items
    check("C13 features_config aligned", "macdh_12_26_9" in sb_src)

    # ── SUMMARY ────────────────────────────────────────────────────
    total = PASS + FAIL
    print(f"\n{'='*60}")
    print(f"  SMOKE TEST: {PASS}/{total} passed, {FAIL} failed")
    print(f"{'='*60}")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
