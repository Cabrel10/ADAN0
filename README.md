# ADAN -- Autonomous Digital Asset Navigator

Multi-agent reinforcement learning system for cryptocurrency trading.
Cross-Attention Hierarchical architecture with FiLM conditioning, Time2Vec
temporal embeddings, and Hidden Markov Model regime detection.

Trained with PPO (Proximal Policy Optimization) via Stable-Baselines3.
Supports any crypto pair. Capital Tier system dynamically scales risk
from micro ($11) to enterprise ($2000+) accounts.


## Quick Start

```bash
git clone https://github.com/Cabrel10/ADAN0.git
cd ADAN0
pip install -r requirements.txt
pip install -e .

# Interactive launcher (recommended):
./run_adan.sh

# Or run each step manually:
python scripts/generate_colab_dataset.py --symbols BTCUSDT ETHUSDT --candles 5000
python scripts/train_simple_ppo.py --steps 30000
python scripts/backtest_engine.py --model models/rl_agents/ppo_adan_simple.zip
```

No API keys needed for dataset generation (synthetic mode is the default).
To use real Binance data, copy `.env.example` to `.env` and fill in your keys.


## Repository Structure

```
ADAN0/
  config/              Configuration files
    config.yaml        Master config (capital tiers, agent, data, workers)
    workers.yaml       Worker timeframe and duration settings
    trading.yaml       Trading rules and risk parameters
  scripts/             Executable entry points
    generate_colab_dataset.py   Dataset generator (synthetic or live)
    train_simple_ppo.py         Single-GPU PPO training
    train_parallel_agents.py    Ray Tune PBT multi-worker training
    backtest_engine.py          Backtest with ensemble support
    paper_trading_monitor.py    Live paper trading on Binance
  src/adan_trading_bot/         Core library
    agent/             Feature extractors (FiLM, Time2Vec, HMM)
    environment/       MultiAssetChunkedEnv, StateBuilder, reward
    data_processing/   ChunkedDataLoader, DataValidator
    portfolio/         Portfolio manager
    risk_management/   Dynamic Boundary Enforcement
    common/            ConfigLoader, logging, utilities
  tests/               Test suite (pytest)
  run_adan.sh          Interactive launcher
  .env.example         Environment variable template
  requirements.txt     Python dependencies
```


## Architecture

The observation pipeline builds a multi-scale input for each step:

```
5m OHLCV  --> [Conv1D + ResBlock] --\
1h OHLCV  --> [Conv1D + ResBlock] ---+--> Cross-Attention --> MLP --> Box(25,)
4h OHLCV  --> [Conv1D + ResBlock] --/                          |
portfolio --> [Linear] ----------------/                  Symlog reward

context_vector (12 dims) --> FiLM modulation on every layer
  [Volatility, Trend, ADX, Regime, Drawdown, Candle_Progress,
   sin_hour, cos_hour, sin_weekday, cos_weekday, sin_dom, cos_dom]
```

The model outputs a continuous 25-dimensional target-weight vector:
- action[0] > +0.33 : BUY signal
- action[0] < -0.33 : SELL signal
- action[0] < -0.1 while long : DYNAMIC EXIT (immediate market close)


## Data Generation

The dataset generator supports two modes:

**Synthetic (default, no API needed):**
```bash
python scripts/generate_colab_dataset.py \
    --symbols BTCUSDT ETHUSDT XRPUSDT SOLUSDT \
    --candles 10000 \
    --split train
```

Generates realistic price data using geometric Brownian motion with
volatility clustering. Asset profiles (base price, daily vol, trend)
are built in for BTC, ETH, XRP, SOL, BNB, DOGE, ADA, AVAX, LINK, DOT.
Unknown symbols get a generic profile.

**Live (requires Binance API keys in .env):**
```bash
python scripts/generate_colab_dataset.py \
    --live --no-testnet \
    --symbols BTC/USDT ETH/USDT \
    --candles 50000
```

Both modes produce the same output format:
`data/processed/indicators/{split}/{SYMBOL}/{5m,1h,4h}.parquet`

All timeframes are aligned to the 5-minute Master Clock via forward-fill.
This eliminates time-travel data leakage between timeframes.


## Training

**Simple (single machine):**
```bash
python scripts/train_simple_ppo.py --steps 50000
# Output: models/rl_agents/ppo_adan_simple.zip + vecnormalize.pkl
```

**Production (Ray Tune PBT, multi-worker):**
```bash
python scripts/train_parallel_agents.py \
    --config config/config.yaml \
    --profiles scalper intraday swing position \
    --steps 2000000 \
    --num-cpus 8
```


## Capital Tiers

Risk parameters are controlled exclusively by Capital Tiers.
Workers define only timeframe and trade duration, never risk.

| Tier       | Balance        | Exposure   | Risk/Trade |
|------------|---------------|------------|------------|
| Micro      | $11 - $30     | 70 - 90%   | 4.0%       |
| Small      | $30 - $100    | 35 - 75%   | 2.0%       |
| Medium     | $100 - $500   | 45 - 60%   | 2.25%      |
| High       | $500 - $2000  | 20 - 35%   | 2.75%      |
| Enterprise | $2000+        | 5 - 15%    | 3.0%       |


## Worker Profiles

| Profile   | Worker | Timeframe | n_steps | batch_size |
|-----------|--------|-----------|---------|------------|
| Scalper   | w1     | 5m        | 512     | 64         |
| Intraday  | w2     | 1h        | 512     | 64         |
| Swing     | w3     | 4h        | 512     | 64         |
| Position  | w4     | 4h        | 1024    | 128        |


## Inference

The inference pipeline (backtest and paper trading) loads the
VecNormalize statistics from training to ensure observation distributions
match exactly:

```python
vec_env = VecNormalize.load("models/rl_agents/vecnormalize.pkl", env)
vec_env.training = False
vec_env.norm_reward = False
```

This was a critical bug fix -- without it, the model receives unnormalized
observations and produces random actions.


## Configuration

All configuration is in `config/config.yaml`. Key sections:

- `capital_tiers`: Risk authority (exposure ranges, risk per trade)
- `agent`: PPO hyperparameters (lr, gamma, clip_range, etc.)
- `data`: Asset list, timeframes, feature configuration
- `workers`: Worker specialization (timeframe, max steps, max trades)

Workers never override risk parameters. Capital Tiers are the sole authority.


## Testing

```bash
pytest tests/ -v
```
