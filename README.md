# ADAN0 -- Autonomous Digital Asset Navigator

PPO-based reinforcement learning bot for BTC/USDT trading.

---

## Current Status (V2 — Future-Guided Arena, 2026-06-24)

**PHASE**: Execution. The project moved from technical exploration to a focused
diagnostic-driven execution phase. The central problem identified and being
fixed: the agent was **exploiting the reward instead of trading**, which caused
the policy to learn a degenerate, frozen action distribution.

### The decisive diagnostic (what V2 is built on)

A pre-tanh audit of the 500K checkpoint (`scripts/audit_pre_tanh.py`) proved the
`size` head was **frozen by its mean μ, not by its std σ**:

| Head signal | 500K (old reward) | Meaning |
|-------------|-------------------|---------|
| `μ(size)` (pre-tanh) | **-7.20** | `tanh(-7.2) ≈ -1.0000` → no gradient (`tanh'≈0`) |
| `σ(size)` (pre-tanh) | **3.24** (largest of all heads) | exploration *wanted* to fire, but μ pinned the output |
| `μ(size)` of a **fresh** model | **≈ 0.0** | the network is **not born broken** |

**Key revelation**: a fresh model starts with `μ(size) ≈ 0` and explores freely.
The `μ = -7.2` was *learned* over 500K steps of the old reward. Therefore the
problem is the **learning signal (reward / credit assignment), not the
architecture** (CNN, attention, gSDE and the observation pipeline all work).

### Cas A vs Cas B decision framework

The V2 instrumented run is designed to cleanly decide between two hypotheses:

- **Cas A** — `μ(size)` becomes/stays healthy (centered, `|μ| < ~3`) → PPO
  relearns with the new reward → **no guard needed** → redo C4/H4 with real
  MFE/MAE, recalibrate A5, activate the reward bridge.
- **Cas B** — `μ(size)` stays ≈ -7 despite high σ → the reward / credit-assignment
  is still wrong → integrate `ActionSaturationGuard` (already written & tested,
  kept in reserve).

### Live V2 diagnostic run (50K steps, sandbox/CPU)

### ⚠️ CRITICAL FINDING — the first 50K V2 run was architecturally invalid

A first 50K instrumented run (sandbox mode) showed a healthy `μ(size)` and real
TP/SL, which *looked* like a "Cas A" win. An **execution audit**
(`scripts/audit_execution.py`) then proved that run did **not** use the real
architecture:

- The sandbox training path built `policy_kwargs` **without**
  `features_extractor_class`, so SB3 silently fell back to its default
  `CombinedExtractor` — a **0-parameter flatten** of the Dict observation.
- Therefore the **CNN, cross-attention, FiLM context and the auxiliary
  forward-predictor never ran**. The "V2 run" trained a **bare MLP**, not the
  `ContextualTemporalFusionExtractor`.
- This explained the suspicious speed (11.7 vs 4.4 steps/s) and made any
  μ(size) / Cas A conclusion **invalid** — it wasn't even the same model.

**This is exactly the failure mode to guard against**: a long run that trains
"a different system than the one you think you are training".

**Fix** (in `scripts/train_parallel_agents.py::sandbox_train`): sandbox now wires
the same `ContextualTemporalFusionExtractor` as heavy mode. Proof via
`scripts/audit_execution.py` on a fresh checkpoint:

| Module | #params | weights changed | gradient flows |
|--------|---------|-----------------|----------------|
| CNN (cnn_layers) | 266,130 | 100% | ✅ 99.3 |
| ATTENTION (cross_attention) | 398,208 | 100% | ✅ 32.2 |
| MEMORY/CONTEXT (FiLM) | 105,984 | 100% | ✅ 5.2 |
| FUSION | 744,192 | 100% | ✅ 28.4 |
| AUX (forward_predictor) | 99,075 | 99.6% | (aux-loss wired in PPO update) |
| policy/value heads + log_std | — | 100% | ✅ |

Checkpoint size jumped 2.2 MB → 7.4 MB (the missing 2.5M+ params are now real).

### ⚠️ CRITICAL FINDING #2 — gSDE variance explosion (the REAL root cause)

Once the real architecture ran, gSDE σ **exploded** (3.4 → 13 → 41 → 110 in a
few rollouts) and `μ(size)` collapsed to -7/-12. A measurement script
(`scripts/diag_gsde_latent.py`) found the exact cause:

```
gSDE variance = (latent_sde²) @ (get_std(log_std)²)   [SB3 source]
  => for ~uniform std:  σ_eff ≈ ||features||₂ · exp(log_std_init)
```

- MEASURED `||features||₂ ≈ 11.4` (real extractor, features_dim=256).
- So the historical `log_std_init=-0.5` gives **σ_eff ≈ 6.9 AT INIT** → chaotic
  actions → PPO drives log_std up → σ diverges.
- **The "frozen size μ=-7" was the network's DEFENSE against this chaos**
  (saturating tanh to stop the noise). It was a *symptom*, not the disease.

**Theories tested and DISPROVED with evidence:**
- *"observations not normalized"* → FALSE. `StateBuilder` already normalizes +
  clips obs to [-10,10] (measured: 5m∈[0,1], 1h/4h∈[-4.4,8.4]); heavy/500K_FIXED
  used `VecNormalize(norm_obs=False)` too. Re-adding obs normalization would
  *double-normalize* and destroy the signal.
- *"add LayerNorm on the features"* → makes it WORSE (||·||₂ → √256 = 16).

**Fix (measured, SB3-documented, no architecture change):**

| Option | Setting | Result (σ over 3k steps) | Trade-off |
|--------|---------|--------------------------|-----------|
| **A — bounded gSDE** | `log_std_init=-2.0` + `use_expln=True` | 0.93 → 1.7 → 1.9 → **4.3** (peaks, no divergence) | keeps state-dependent exploration; σ still creeps slowly |
| **B — DiagGaussian (recommended for 500K)** | `ADAN_USE_SDE=0`, `log_std_init=-1.0` | **0.366 → 0.367 → 0.369** (flat) | σ mathematically decoupled from features → cannot diverge; loses state-dependence |

Both keep `μ(size)` centered (no more collapse to -7) and produce real TP/SL
(TP/SL ≈ 99% of closes, AGENT_CLOSE ≈ 1%, MAX_DURATION = 0).

> **Recommended for the 500K production run: Option B** (`ADAN_USE_SDE=0`) — it is
> provably stable for the full duration. Option A is fine for shorter
> state-dependent-exploration experiments but must be watched.

> **Status: NO scientific Cas A/B verdict yet.** Two blocking bugs were found and
> fixed (wrong extractor, gSDE divergence). The architecture now provably runs
> end to end AND trains stably. The Cas A / Cas B question must be answered on a
> real 500K run with the fixes. The earlier "Cas A confirmed" claim is withdrawn.

### ⚠️ CRITICAL FINDING #4 — TP/SL incoherent with the market + the Future Arena was a DEAD pipeline

A 500K run launched on the VPS produced a **broken, "cowardly" model** — not impatient,
not lazy, but mathematically doomed by the reward geometry. The trade stats are brutal:

| Metric | Value | Meaning |
|--------|-------|---------|
| SL_HIT | **258 trades (84%)**, avg **−$0.92** | the stop is hit almost systematically |
| TP_HIT | 42 trades (14%), avg +$1.40 | the TP is rarely reached |
| AGENT_CLOSE | 20 trades (6%), avg +$0.06 | manual micro-scalping to escape |
| Hold mean | **5.5 steps (~45 min)** | the model never holds |
| **Expectancy** | **(0.84×−0.92)+(0.14×+1.40)+(0.06×+0.06) = −$0.58 / trade** | guaranteed bleed |

**Root cause A — TP/SL bands are ~10–50× the achievable move.** The env clamps TP/SL
with a hardcoded `_PROFILE_BOUNDS` table calibrated for a **fake 0.80% round-trip fee**
(4× the real 0.10% Binance fee), forcing intraday **TP into the 8–12% range**. We measured
the **real BTC future wicks** on the actual dataset:

| Horizon | MFE p50 | MFE p75 | MFE p90 |
|---------|---------|---------|---------|
| 5m × 15 bars (1h15) | 0.22% | 0.45% | 0.84% |
| 5m × 36 bars (3h) | 0.36% | 0.73% | **1.44%** |

A TP of 8–12% over minutes/hours is **practically never touched** → the SL fires first →
the model panics with AGENT_CLOSE. Exactly the user's diagnosis: *"cibler 4 à 5 % en
quelques minutes est pratiquement impossible, normal que le model panique."*

**Root cause B — the Future Arena (the heart of the project) was never connected.** The
`future_arena/` package (1703 lines) implements the **"Arène Guidée par le Futur"
(Privileged Oracle-Critic)** philosophy exactly:

- During training the whole dataset is in RAM, so the env reads the **future** of the
  current chunk (1 chunk ≈ 1 day) and computes, per trade:
  - **MFE** (Maximum Favorable Excursion = future upper wick),
  - **MAE** (Maximum Adverse Excursion = future lower wick).
- It then **scores the TP/SL the agent chose** against those values:
  *future max wick = +3% and the agent puts TP at +2.5% → big bonus (realistic, capturable);
  TP at +10% → penalty (utopian, never touched).*
- Zones **🟢/🟡/🔴** are derived per chunk from the ex-post RR = MFE/MAE
  (🟢 RR ≥ 1.5 **and** MFE ≥ 0.6%; 🔴 RR ≤ 0.8; 🟡 in between).
- This is **reward shaping only** (anti-oracle guard §10.10: future data NEVER enters the
  actor's observation).

But a `grep` proved **zero** `future_arena` imports/calls in the training env, and the
config had **no `reward_shaping.future_reward` block** → the entire pipeline was dead
code. The model was therefore a pure mirror of `_PROFILE_BOUNDS`, with no signal teaching
it to choose a *capturable* TP/SL.

**Fix (this commit):**
1. **Market-aware `_PROFILE_BOUNDS`** — bands re-derived from the real wick distribution and
   widened **downward** so the agent *can* pick realistic small TPs (scalper TP from ~0.5%,
   intraday in the user-requested **3–6%** envelope) instead of being forced ≥4–8%. Fee gate
   uses the real `commission_pct` (0.4% spot) not the fake 0.80%.
2. **RewardBridge connected** — the env now instantiates `RewardBridge.from_config(self.config)`,
   computes MFE/MAE **ex-post per closed trade** from the in-RAM chunk
   (`compute_mfe_mae(chunk_df, entry_idx, LOW, horizon)`), and **adds the capped
   `future_contrib`** (≤ 0.60, never dominates PnL) to `raw_reward`. EQS / sl_quality /
   tp_quality now appear in the reward breakdown.
3. **New config block** `reward_shaping.future_reward: {enabled, mode: future_guided,
   round_trip_fees, max_future_contrib}` activates the bridge.

> Philosophy (user): *"le but n'est pas que le model soit le miroir des données mais qu'il
> arrive à mieux maîtriser son SL et son TP."* The bridge teaches **mastery of TP/SL within
> a plausible band**, it does not hardcode the answer.

**Next milestones (on the corrected, stabilized architecture)**:
- Run the real 500K on the VPS, single profile, `ADAN_USE_SDE=0` (the VPS already
  produced 500K_FIXED; with one profile the load is lower → no need for Kali yet).
- Monitor `σ` stays < ~3 and `μ(size)` stays in [-2, +2] via the ActionDim CSV.
- 100K-200K: size distribution should spread (0.1 / 0.2 / 0.35 / 0.6 / 0.8).
- Re-run `scripts/audit_execution.py` on the 100K checkpoint to confirm all
  modules keep training.

---

## Architecture

```
4 Workers (PBT - Population-Based Training)
  W1 Scalper:   5m  | gamma=0.95  | n_steps=512
  W2 Intraday:  1h  | gamma=0.99  | n_steps=2048
  W3 Swing:     4h  | gamma=0.995 | n_steps=8192
  W4 Position:  4h  | gamma=0.999 | n_steps=16384

Observation Space: Dict{5m:(20,21), 1h:(20,21), 4h:(20,21), context_vector:(17,), portfolio_state:(20,)}
Action Space: Box(-1,1,(5,)) [direction, size_pct, tf_pref, sl_pct, tp_pct]
Feature Extractor: ContextualTemporalFusionExtractor (CNN + Channel/Temporal Attention)
Exploration: gSDE (State-Dependent Exploration)
  - heavy/default: log_std_init=-0.5
  - V2 diagnostic:  log_std_init=0.0  (std0≈1.0) — wider exploration to break the freeze
```

The 5 action heads are `[direction, size, tf_pref, sl, tp]`, each a squashed
Gaussian. Because `squash_output=False` in gSDE, the *pre-tanh* μ and σ of every
head can be read directly via `policy.get_distribution(obs).distribution.mean`
(μ) and `.scale` (σ) — this is exactly what the audit and the monitor exploit.

---

## V2 adjustments and their impact

Every V2 change is incremental, builds on existing code, and is measured. None
of them touch the network architecture (the diagnostic proved that is sound).

| Adjustment | Env var / flag | What it does | Why / impact |
|-----------|----------------|--------------|--------------|
| **Wider exploration** | `ADAN_LOG_STD_INIT=0.0` | Starts gSDE at std0≈1.0 instead of ≈0.6 | Gives the `size` head room to escape the learned `μ=-7` basin instead of collapsing again |
| **Higher entropy** | `ADAN_ENT_COEF=0.02` | Raises PPO entropy bonus | Keeps the policy from prematurely committing to a saturated action; sustains σ |
| **Per-head monitor** | `ADAN_ACTIONDIM=1` | Activates `ActionDimMonitor` (measure-only callback) | Logs pre-tanh μ/σ + post-tanh mean/std + saturation fraction per head to TensorBoard, console and CSV every rollout. **Never modifies the network.** |
| **Monitor CSV** | `ADAN_ACTIONDIM_CSV=path.csv` | Writes the per-rollout metrics to CSV | Enables `analyze_actiondim.py` to emit the Cas A / Cas B verdict offline |
| **Anti-OOM launcher** | `scripts/run_adan_v2.sh` | Auto-detects host (Kali / VPS / sandbox / GPU); forces sandbox mode when free RAM < 12 GB | Ray needs ≥12 GB; below that the object store crashes. The launcher prevents OOM by dropping to single-process PPO. |
| **Saturation guard (reserve)** | — | `ActionSaturationGuard` callback bumps `log_std` after N saturated rollouts | **Not integrated yet.** Only used if the run shows Cas B (frozen μ despite high σ). 7/7 unit tests pass. |

### Diagnostic vs production reward

The reward is being redesigned to be **future-guided** (it credits actions using
what actually happens to price afterwards — real MFE/MAE — rather than a shaped
proxy). The current run uses the bridged reward to verify the freeze breaks
before the full future-guided reward and the C4/H4 zones are recalibrated.

---

## Quick Start

### GPU Training (Colab/Kaggle)

Open `notebooks/ADAN_Full_Training_H100.ipynb` in Colab with H100/A100 GPU.

The notebook handles the full pipeline:
1. Clone repo + install deps
2. Download 6 years BTC data via CCXT (Bitget)
3. Compute 21 technical indicators per TF
4. Create train/test/val splits (70/20/10)
5. Train 4 workers with Ray Tune PBT (500K+ steps)
6. Deterministic OOS backtest on test split
7. Export model package for paper trading

### Local CI/Sandbox

```bash
# Install
pip install -e .

# Run 5000-step smoke test (CPU only, ~5 min)
PYTHONPATH=src python scripts/train_parallel_agents.py --mode sandbox --steps 5000

# Run OOS backtest
PYTHONPATH=src python scripts/deterministic_backtest.py --steps 500 --split test
```

### Heavy Training (local GPU)

```bash
PYTHONPATH=src python scripts/train_parallel_agents.py \
  --mode heavy \
  --steps 500000 \
  --num-cpus 8 \
  --num-samples 4 \
  --profiles scalper,intraday,swing,position \
  --checkpoint-dir checkpoints/heavy
```

---

## Reward Formula (S16 Final - Tier Hysteresis)

```
reward = symlog(raw) + tier_bonus
raw = pnl_net_scaled - trade_cost - drawdown_penalty + time_decay + survival_bonus

Where:
  pnl_net_scaled = (realized_pnl - commission*1.5) * 100 / initial_capital
  time_decay = -0.001 (on non-trade steps only)
  trade_cost = commission_pct * 100 / initial_capital
  drawdown_penalty = 50.0 * drawdown_pct^2 (quadratic)
  survival_bonus = +0.001/step (prevents 'impossible game' strategies)
  symlog(x) = sign(x) * ln(|x| + 1)

TIER BONUSES (10x multiplier):
  Micro tier promotion: +5.0
  Small tier promotion: +10.0
  Medium tier promotion: +20.0
  Large tier promotion: +40.0

STAGNATION PENALTIES (soft - ÷4 from baseline):
  Tier 1 stagnation: -0.005/step
  Tier 2 stagnation: -0.010/step
  Tier 3 stagnation: -0.015/step
  Tier 4 stagnation: -0.020/step

ZERO shaping. Pure capital progression + realized PnL.
```

---

## Project Structure

```
ADAN0/
  config/config.yaml             # Central configuration (workers, rewards, capital tiers)
  scripts/
    train_parallel_agents.py     # Main training (sandbox + heavy mode)
    deterministic_backtest.py    # OOS evaluation
    download_ccxt_data.py        # Data download via CCXT
    compute_features_real.py     # 21 indicators per TF
    create_train_test_val_splits.py  # 70/20/10 chronological splits
  src/adan_trading_bot/
    environment/
      multi_asset_chunked_env.py # Main RL environment (Gym)
      reward_calculator.py       # Reward computation
      dynamic_behavior_engine.py # DBE / regime adaptation
      exogenous_regime_oracle.py # HMM oracle
    data_processing/
      data_loader.py             # ChunkedDataLoader (parquet)
      state_builder.py           # Observation construction
      feature_engineer.py        # Technical indicators
    agent/
      feature_extractors.py      # CNN + Attention (ContextualTemporalFusionExtractor)
    portfolio/
      portfolio_manager.py       # Position tracking, PnL
  notebooks/
    ADAN_Full_Training_H100.ipynb  # Full pipeline for Colab/Kaggle
  .github/workflows/
    adan0_relay.yml              # CI relay training + OOS backtest
```

---

## Deployment Target

- **Exchanges**: Bitget, Binance (via CCXT)
- **Mode**: Paper trading first, then real with micro capital ($20.50)
- **Capital tiers**: Micro ($0-50), Small ($50-500), Medium ($500-5K)
- **Risk**: Pareto risk detector, dynamic SL/TP from DBE

## Documentation

All detailed analysis, diagnostics, and session reports are in the `docs/` folder:

- `docs/ANALYSIS_COMPLETE_BUG_AUDIT.md` — Complete data integrity verification
- `docs/METRICS_DEEP_DIVE.md` — Performance metric deep dives
- `docs/POLAR_REWARD_*.md` — Polar reward system evolution
- `docs/SESSION_*_*.md` — Training session reports and diagnostics
- `docs/TIER_BASED_REWARD_IMPLEMENTATION.md` — Tier hysteresis implementation details
- `docs/TRAINING_*.md` — Training progress snapshots and readiness checklists

See `docs/COMPREHENSIVE_AUDIT_GUIDE.md` for a complete guide to all documentation.

1. **No reward shaping** — Only realized PnL from closed trades is a valid reward
2. **No unrealized PnL in reward** — Violates Ng 1999 (depends on past actions, not state)
3. **Honest reporting** — If the model loses money, report it honestly
4. **explained_variance > 0** before production — If critic can't predict value, policy is random
5. **2 weeks paper trading** minimum before real money
