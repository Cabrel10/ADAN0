# ADAN0 -- Autonomous Digital Asset Navigator

PPO-based reinforcement learning bot for BTC/USDT trading.

---

## Current Status (S15 Hard Reset, 2026-06-03)

**HONEST ASSESSMENT**: The model is NOT production-ready.

| Metric | Value | Required |
|--------|-------|----------|
| Training steps | 5,120 | 500,000+ |
| Explained variance | -6.54 | > 0.0 |
| OOS trades (test) | 13 | 50+ |
| OOS return | +33.82% | Statistically significant |
| Max drawdown | 18.15% | < 20% |
| Reward shaping | ZERO | ZERO |

**What was done (S15)**:
- Purged ALL reward shaping (capacity_reward, frequency_reward, PBRS)
- Reward = `symlog(realized_pnl_scaled - costs - drawdown + time_decay)`
- Restored CI hyperparams: n_steps=512, batch=64, n_epochs=10
- Fixed workflow YAML: OOS backtest uses `--split test`
- Fixed Colab H100 resource detection

**What needs to be done**:
- Train 500K+ steps on GPU (Colab H100 or Kaggle)
- Achieve `explained_variance > 0` (critic converges)
- Verify performance on bearish OOS data
- Paper trade for 2 weeks before any real capital

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

## Reward Formula (S15 Final)

```
reward = symlog(raw)
raw = pnl_net_scaled - trade_cost - drawdown_penalty + time_decay + inv_penalty

Where:
  pnl_net_scaled = (realized_pnl - commission*1.5) * 100 / initial_capital
  time_decay = -0.001 (on non-trade steps only)
  trade_cost = commission_pct * 100 / initial_capital
  drawdown_penalty = config-driven (default 80.0 * drawdown_pct)
  inv_penalty = penalty for invalid trade attempts
  symlog(x) = sign(x) * ln(|x| + 1)

ZERO shaping. ZERO bonuses. Pure realized PnL from closed trades.
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

---

## Critical Rules

1. **No reward shaping** — Only realized PnL from closed trades is a valid reward
2. **No unrealized PnL in reward** — Violates Ng 1999 (depends on past actions, not state)
3. **Honest reporting** — If the model loses money, report it honestly
4. **explained_variance > 0** before production — If critic can't predict value, policy is random
5. **2 weeks paper trading** minimum before real money
