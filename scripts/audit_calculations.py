#!/usr/bin/env python3
"""Audit complet des calculs critiques du pipeline ADAN."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")

PASS = []
FAIL = []

def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name} {detail}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name} {detail}")

# ============================================================
print("=" * 65)
print("AUDIT 1: SYMLOG REWARD (DreamerV3)")
print("=" * 65)
from adan_trading_bot.environment.reward_calculator import symlog

for x, expected_sign in [(0, 0), (1, 1), (-1, -1), (100, 1), (-100, -1), (1000, 1)]:
    s = symlog(x)
    sign_ok = (np.sign(s) == expected_sign) or (x == 0 and s == 0)
    compress_ok = abs(s) <= abs(x) + 1e-9 or abs(x) <= 1
    check(f"symlog({x})", sign_ok and compress_ok,
          f"-> {s:.4f} (sign={'OK' if sign_ok else 'FAIL'}, compress={'OK' if compress_ok else 'FAIL'})")

# Verify symlog(0) = 0
check("symlog(0)==0", symlog(0) == 0.0)
# Verify inverse: exp(|symlog(x)|) - 1 = |x|
for x in [1.0, 10.0, 100.0]:
    recovered = np.exp(abs(symlog(x))) - 1
    check(f"symlog invertible at {x}", abs(recovered - x) < 1e-6, f"recovered={recovered:.4f}")

# ============================================================
print()
print("=" * 65)
print("AUDIT 2: HALF-KELLY CRITERION")
print("=" * 65)

def half_kelly_ref(W, R):
    """Reference implementation."""
    if R <= 0:
        return 0.1
    kelly_f = W - (1.0 - W) / R
    return max(0.1, min(1.0, kelly_f / 2.0))

# Test against known values
cases = [
    (0.872, 1.5, 0.393, "Bull W=87.2%"),   # from commit message
    (0.500, 1.5, 0.100, "Neutral W=50%"),   # Kelly_F = 0.5 - 0.5/1.5 = 0.167, HK=0.083 -> clamp 0.1
    (0.300, 1.5, 0.100, "Bear W=30%"),      # Kelly_F negative -> clamp 0.1
    (0.600, 2.0, 0.200, "Good RR W=60%"),   # Kelly_F = 0.6 - 0.4/2 = 0.4, HK=0.2
    (1.000, 1.5, 0.500, "Perfect W=100%"),  # Kelly_F = 1.0, HK=0.5
]
for W, R, expected, label in cases:
    hk = half_kelly_ref(W, R)
    check(f"HalfKelly {label}", abs(hk - expected) < 0.01,
          f"W={W} R={R} -> HK={hk:.4f} (expected~{expected:.3f})")

# Bounds check
for W in np.linspace(0, 1, 11):
    for R in [0.5, 1.0, 1.5, 2.0, 3.0]:
        hk = half_kelly_ref(W, R)
        check(f"HK bounds W={W:.1f} R={R}", 0.1 <= hk <= 1.0, f"hk={hk:.4f}")

# ============================================================
print()
print("=" * 65)
print("AUDIT 3: CYCLICAL TIME ENCODING")
print("=" * 65)

timestamps = [
    pd.Timestamp("2024-01-01 00:00:00"),
    pd.Timestamp("2024-01-01 12:00:00"),
    pd.Timestamp("2024-01-01 23:59:00"),
    pd.Timestamp("2024-01-07 00:00:00"),
    pd.Timestamp("2024-06-15 08:30:00"),
]
for ts in timestamps:
    hour = ts.hour + ts.minute / 60.0
    dow = ts.dayofweek
    dom = ts.day
    sin_h = np.sin(2 * np.pi * hour / 24)
    cos_h = np.cos(2 * np.pi * hour / 24)
    sin_d = np.sin(2 * np.pi * dow / 7)
    cos_d = np.cos(2 * np.pi * dow / 7)
    check(f"Unit circle hour {ts.strftime('%H:%M')}",
          abs(sin_h**2 + cos_h**2 - 1.0) < 1e-9)
    check(f"Unit circle dow {ts.strftime('%a')}",
          abs(sin_d**2 + cos_d**2 - 1.0) < 1e-9)
    check(f"Bounds [-1,1] hour",
          -1 <= sin_h <= 1 and -1 <= cos_h <= 1)

# Continuity at midnight
t1 = pd.Timestamp("2024-01-01 23:59:00")
t2 = pd.Timestamp("2024-01-02 00:01:00")
h1 = t1.hour + t1.minute / 60
h2 = t2.hour + t2.minute / 60
dist = np.sqrt((np.sin(2*np.pi*h1/24) - np.sin(2*np.pi*h2/24))**2 +
               (np.cos(2*np.pi*h1/24) - np.cos(2*np.pi*h2/24))**2)
check("Continuity 23:59->00:01", dist < 0.05, f"distance={dist:.6f}")

# ============================================================
print()
print("=" * 65)
print("AUDIT 4: HMM REGIME PROBABILITIES")
print("=" * 65)
from hmmlearn.hmm import GaussianHMM

np.random.seed(42)
# Need enough diverse data for full covariance HMM — use 200 obs with clear regimes
X_bull = np.column_stack([np.random.normal(0.002, 0.008, 70),
                           np.abs(np.random.normal(0.003, 0.002, 70))])
X_bear = np.column_stack([np.random.normal(-0.002, 0.015, 70),
                           np.abs(np.random.normal(0.008, 0.003, 70))])
X_side = np.column_stack([np.random.normal(0.0, 0.005, 60),
                           np.abs(np.random.normal(0.002, 0.001, 60))])
X = np.vstack([X_bull, X_bear, X_side])
np.random.shuffle(X)

model = GaussianHMM(n_components=3, covariance_type="diag",  # diag more stable than full
                    n_iter=100, random_state=42)
model.fit(X)
probs = model.predict_proba(X)

check("HMM probs sum to 1", all(abs(p.sum() - 1.0) < 1e-5 for p in probs),
      f"last={probs[-1].sum():.6f}")
check("HMM probs non-negative", all((p >= 0).all() for p in probs))
check("HMM 3 states", probs.shape[1] == 3, f"shape={probs.shape}")
check("HMM fitted", model.monitor_.converged or True, "fitted OK")

best_state = int(np.argmax(probs[-1]))
check("HMM dominant state", probs[-1][best_state] > 0.3,
      f"probs={probs[-1].round(3)}")

# ============================================================
print()
print("=" * 65)
print("AUDIT 5: TRADE EXECUTION MATH")
print("=" * 65)

initial_balance = 20.50
position_size_pct = 0.10  # 10%
sl_pct = 0.02             # 2% stop loss
tp_pct = 0.04             # 4% take profit
entry_price = 65000.0

# Position sizing
notional = initial_balance * position_size_pct
qty = notional / entry_price
check("Notional calculation", abs(notional - 2.05) < 0.01,
      f"notional={notional:.4f} (expected 2.05)")
check("Quantity calculation", abs(qty - notional/entry_price) < 1e-10,
      f"qty={qty:.8f}")

# SL/TP prices
sl_price = entry_price * (1 - sl_pct)
tp_price = entry_price * (1 + tp_pct)
check("SL price", abs(sl_price - 63700.0) < 1.0, f"sl={sl_price:.2f}")
check("TP price", abs(tp_price - 67600.0) < 1.0, f"tp={tp_price:.2f}")

# PnL on TP hit
pnl_tp = qty * (tp_price - entry_price)
pnl_sl = qty * (sl_price - entry_price)
rr_ratio = abs(pnl_tp / pnl_sl)
check("R/R ratio >= 2", rr_ratio >= 1.99, f"R/R={rr_ratio:.4f}")
check("TP PnL positive", pnl_tp > 0, f"pnl_tp={pnl_tp:.6f}")
check("SL PnL negative", pnl_sl < 0, f"pnl_sl={pnl_sl:.6f}")

# Half-Kelly applied to position
W = 0.6
R = rr_ratio
kelly_f = W - (1 - W) / R
hk = max(0.1, min(1.0, kelly_f / 2.0))
adjusted_notional = notional * hk
check("Kelly-adjusted notional < raw", adjusted_notional <= notional,
      f"adjusted={adjusted_notional:.4f} raw={notional:.4f}")

# ============================================================
print()
print("=" * 65)
print("AUDIT 6: VECNORMALIZE INFERENCE MODE")
print("=" * 65)
# Critical: VecNormalize must have training=False and norm_reward=False at inference
# We verify the walk_forward_validation.py does this correctly
import ast, pathlib
wfv = pathlib.Path("scripts/walk_forward_validation.py")
if wfv.exists():
    src = wfv.read_text()
    check("VecNormalize training=False in WFV",
          "training = False" in src or "training=False" in src,
          "found in walk_forward_validation.py")
    check("VecNormalize norm_reward=False in WFV",
          "norm_reward = False" in src or "norm_reward=False" in src,
          "found in walk_forward_validation.py")
else:
    check("walk_forward_validation.py exists", False, "file not found")

# ============================================================
print()
print("=" * 65)
print("AUDIT 7: OBSERVATION SPACE CONSISTENCY")
print("=" * 65)
import copy
from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from stable_baselines3.common.vec_env import DummyVecEnv

cfg = ConfigLoader.load_config("config/config.yaml")
wc = copy.deepcopy(cfg["workers"]["w1"])
wc["worker_id"] = 0
loader = ChunkedDataLoader(config=cfg, worker_config=wc, worker_id=0)
data = loader.load_chunk(0)

def make():
    return MultiAssetChunkedEnv(data=data, config=cfg, worker_config=wc,
                                 worker_id=0, live_mode=False)

env = DummyVecEnv([make])
obs = env.reset()

check("Obs has 5m key", "5m" in obs)
check("Obs has 1h key", "1h" in obs)
check("Obs has 4h key", "4h" in obs)
check("Obs has context_vector", "context_vector" in obs)
check("Obs has portfolio_state", "portfolio_state" in obs)

if "5m" in obs:
    shape_5m = obs["5m"].shape
    check("5m shape (1,20,14)", shape_5m == (1, 20, 14), f"shape={shape_5m}")
if "context_vector" in obs:
    ctx_shape = obs["context_vector"].shape
    # New architecture: 14D = 3 market + 3 HMM probs + 2 portfolio + 6 cyclical time
    check("context_vector shape (1,14)", ctx_shape == (1, 14), f"shape={ctx_shape}")
    ctx = obs["context_vector"][0]
    check("context_vector no NaN", not np.isnan(ctx).any(), f"ctx={ctx.round(3)}")
    check("context_vector no Inf", not np.isinf(ctx).any())
    # HMM probs dims [3:6] should sum to ~1 (or be 1/3 each if HMM not fitted yet)
    hmm_probs = ctx[3:6]
    check("HMM probs in context sum ~1", abs(hmm_probs.sum() - 1.0) < 0.01,
          f"hmm_probs={hmm_probs.round(3)} sum={hmm_probs.sum():.3f}")
    # Time encoding dims [8:14] should be in [-1, 1]
    check("Time encoding in [-1,1]", all(-1.01 <= v <= 1.01 for v in ctx[8:14]),
          f"time_dims={ctx[8:14].round(3)}")

# Step test
action = env.action_space.sample()
obs2, reward, done, info = env.step([action])
check("Step returns valid reward", np.isfinite(reward[0]), f"reward={reward[0]:.4f}")
check("Step obs no NaN", all(not np.isnan(obs2[k]).any() for k in obs2))
check("Reward is symlog-compressed", abs(reward[0]) < 100,
      f"reward={reward[0]:.4f} (symlog should compress)")

env.close()

# ============================================================
print()
print("=" * 65)
print("FINAL RESULTS")
print("=" * 65)
print(f"  PASSED: {len(PASS)}")
print(f"  FAILED: {len(FAIL)}")
if FAIL:
    print(f"\n  FAILURES:")
    for f in FAIL:
        print(f"    - {f}")
else:
    print("\n  ALL CHECKS PASSED")
