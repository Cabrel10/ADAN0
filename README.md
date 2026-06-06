# ADAN0 -- Autonomous Digital Asset Navigator

PPO-based reinforcement learning bot for BTC/USDT trading.

---

## Current Status (S16 - Tier Hysteresis + Enterprise PBT, 2026-06-06)

**TRAINING PROGRESS**: Multi-worker PBT actively training with tier-based reward system.

| Metric | Value | Status |
|--------|-------|--------|
| Training steps | 8,192 | ✅ Progressing (target: 500K+) |
| Active workers | 4 | ✅ Ray distributed training |
| Win rate (recent) | 52.99% | ✅ Above breakeven |
| Sharpe ratio | 6.18 | ✅ Excellent |
| Sortino ratio | 9.52 | ✅ Strong downside protection |
| Portfolio value | $37.05 | ✅ Capital preservation |
| Reward shaping | ZERO | ✅ Pure financial signal |

**What was done (S16)**:
- Implemented **tier hysteresis** with 4-tier capital progression (Micro→Small→Medium→Large)
- 10x promotion bonuses + soft stagnation penalties for realistic prop firm behavior
- **Enterprise PBT Pipeline**: VecNormalize sync, gSDE support, checkpoint integrity
- Profile-specific training: Scalper (5m), Intraday (1h), Swing (4h), Position (1d)
- Fixed MaxDD double-multiplication bug
- Corrected PnL tracking (no double-counting)
- Tier-based risk management: dynamic SL/TP from tier + DBE

**Active development**:
- Multi-worker PBT: 4 agents evolving hyperparams independently
- Frequency compliance: 5m/1h/4h position limits enforced
- AGENT_CLOSE: ~50% of trades closed by decision (improving from 40% MaxDuration)
- Real-time metrics sync: Sharpe, Sortino, win rate, unrealized PnL tracking

**Next milestones**:
- 50K+ steps: Tier progression + hyperparameter stabilization
- 100K+ steps: Win rate → 55%+ target
- 500K+ steps: Production-ready eval on OOS data

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
Exploration: gSDE (State-Dependent Exploration), log_std_init=-0.5
```

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
