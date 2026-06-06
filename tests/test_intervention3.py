#!/usr/bin/env python3
"""Intervention 3 verification: context_vector dim=17, oracle_probs sum~1."""
import sys, os
import numpy as np
import pandas as pd

# Resolve project root relative to this file (tests/ -> project root)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

# Direct import to avoid gym/torch chain
import importlib.util

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load StateBuilder directly
sb_mod = _load_module(
    "state_builder",
    os.path.join(_PROJECT_ROOT, "src/adan_trading_bot/data_processing/state_builder.py")
)
StateBuilder = sb_mod.StateBuilder

passed = 0
failed = 0

# --- Test 1: CONTEXT_DIM = 17 ---
try:
    assert StateBuilder.CONTEXT_DIM == 17, f"Expected 17, got {StateBuilder.CONTEXT_DIM}"
    print("PASS T1: CONTEXT_DIM = 17")
    passed += 1
except Exception as e:
    print(f"FAIL T1: {e}")
    failed += 1

# --- Test 2: build_context_vector returns shape (17,) ---
try:
    # StateBuilder expects features_config dict (TF -> feature list), not full YAML
    sb = StateBuilder()  # Uses default features_config with all 3 TFs
    ctx = sb.build_context_vector()
    assert ctx.shape == (17,), f"Expected (17,), got {ctx.shape}"
    print(f"PASS T2: context shape = {ctx.shape}")
    passed += 1
except Exception as e:
    print(f"FAIL T2: {e}")
    failed += 1

# --- Test 3: context[-3:] sums to ~1.0 with default oracle_probs ---
try:
    oracle_part = ctx[14:17]
    s = float(oracle_part.sum())
    assert abs(s - 1.0) < 0.01, f"oracle_probs sum = {s}, expected ~1.0"
    print(f"PASS T3: oracle_probs sum = {s:.4f} (uniform default)")
    passed += 1
except Exception as e:
    print(f"FAIL T3: {e}")
    failed += 1

# --- Test 4: context[-3:] sums to ~1.0 with custom oracle_probs ---
try:
    custom_probs = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    ctx2 = sb.build_context_vector(oracle_probs=custom_probs)
    oracle_part2 = ctx2[14:17]
    assert np.allclose(oracle_part2, custom_probs, atol=0.01), \
        f"Expected {custom_probs}, got {oracle_part2}"
    s2 = float(oracle_part2.sum())
    assert abs(s2 - 1.0) < 0.01, f"sum = {s2}"
    print(f"PASS T4: custom oracle_probs = {oracle_part2}, sum = {s2:.4f}")
    passed += 1
except Exception as e:
    print(f"FAIL T4: {e}")
    failed += 1

# --- Test 5: build_observation passes oracle_probs through ---
try:
    # Build minimal data
    dates = pd.date_range("2025-01-01", periods=100, freq="5min")
    df5m = pd.DataFrame({
        "open": np.random.uniform(95000, 105000, 100),
        "high": np.random.uniform(95000, 105000, 100),
        "low": np.random.uniform(95000, 105000, 100),
        "close": np.random.uniform(95000, 105000, 100),
        "volume": np.random.uniform(1, 100, 100),
    }, index=dates)
    # Add required indicator columns
    for col in ["ema_20_ratio", "macdh_12_26_9", "rsi_14", "adx_14", "di_delta",
                "atr_pct", "bb_percent_b_20_2", "obv_slope", "volume_ratio_20",
                "volatility_ratio_14_50", "fib_ratio", "price_action", "vwap_ratio",
                "market_structure", "bb_width_20_2", "log_return"]:
        df5m[col] = np.random.randn(100) * 0.1

    data = {"BTCUSDT": {"5m": df5m}}
    op = np.array([0.1, 0.6, 0.3], dtype=np.float32)
    obs = sb.build_observation(
        current_idx=50, data=data,
        hmm_probs=np.array([0.4, 0.4, 0.2]),
        oracle_probs=op,
    )
    cv = obs["context_vector"]
    assert cv.shape == (17,), f"build_observation context shape = {cv.shape}"
    assert np.allclose(cv[14:17], op, atol=0.02), f"oracle not propagated: {cv[14:17]}"
    print(f"PASS T5: build_observation oracle propagation OK: {cv[14:17]}")
    passed += 1
except Exception as e:
    print(f"FAIL T5: {e}")
    failed += 1

# --- Test 6: ExogenousRegimeOracle loads and returns shape (3,) ---
try:
    orc_mod = _load_module(
        "exogenous_regime_oracle",
        os.path.join(_PROJECT_ROOT, "src/adan_trading_bot/environment/exogenous_regime_oracle.py")
    )
    oracle = orc_mod.ExogenousRegimeOracle()
    probs = oracle.predict_proba_safe(np.zeros(5))
    assert probs.shape == (3,), f"Oracle shape = {probs.shape}"
    assert abs(probs.sum() - 1.0) < 0.01, f"Oracle sum = {probs.sum()}"
    print(f"PASS T6: Oracle unfitted returns uniform {probs}, sum={probs.sum():.4f}")
    passed += 1

    # Try loading the trained model
    oracle_pkl = os.path.join(_PROJECT_ROOT, "models/exog_oracle.pkl")
    if os.path.isfile(oracle_pkl):
        oracle.load(oracle_pkl)
        probs2 = oracle.predict_proba_safe(np.random.randn(5))
        assert probs2.shape == (3,) and abs(probs2.sum() - 1.0) < 0.01
        print(f"  -> Loaded trained oracle: {probs2}, sum={probs2.sum():.4f}")
except Exception as e:
    print(f"FAIL T6: {e}")
    failed += 1

print(f"\n{'='*60}")
print(f"Intervention 3: {passed}/{passed+failed} passed")
if failed > 0:
    sys.exit(1)
else:
    print("ALL INTERVENTION 3 TESTS PASSED")
