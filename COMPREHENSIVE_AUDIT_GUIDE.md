# 🔍 COMPREHENSIVE CHUNK AUDIT GUIDE

## Overview

Complete audit suite for validating agent training results on entire chunks. Tests include:

1. **Trade Validation** - Verify PnL calculations on individual trades
2. **Position Validation** - Check position accounting and consistency
3. **PnL Accounting** - Validate realized vs unrealized split
4. **Value Function** - Measure value network effectiveness (R²)
5. **Lookahead Bias** - Detect if agent uses future information
6. **Generalization** - Test performance across different market contexts

---

## Quick Start

### One-Line Execution

```bash
chmod +x scripts/run_full_audit.sh
./scripts/run_full_audit.sh --checkpoint ./checkpoints/episode_1 --chunk 1
```

### Expected Output

```
audit_results/
├── chunk1_audit.json              # Comprehensive audit results
├── generalization_test.json       # Walk-forward test results
└── AUDIT_REPORT_1.md             # Combined report
```

---

## Detailed Usage

### Script 1: Comprehensive Chunk Audit

**Purpose:** Validates all data within a single chunk

```bash
python3 scripts/audit_chunk_comprehensive.py \
    --checkpoint ./checkpoints/episode_1 \
    --chunk 1 \
    --output audit_results/chunk1_audit.json
```

#### Output JSON Structure

```json
{
  "metadata": {
    "trades_loaded": 1250,
    "steps_loaded": 25000
  },
  "trade_validation": {
    "total_trades": 1250,
    "valid_trades": 1247,
    "invalid_trades": 3,
    "pnl_discrepancies": 3,
    "total_realized_pnl_sample": 1970.33,
    "status": "✅ VALID"
  },
  "position_validation": {
    "portfolio_value": 71.98,
    "cash": 47.80,
    "open_positions_value": 24.18,
    "realized_equity": 1990.83,
    "implied_gap": 1918.85,
    "consistency_error": 0.0012,
    "status": "✅ CONSISTENT"
  },
  "pnl_validation": {
    "initial_capital": 20.50,
    "final_portfolio": 71.98,
    "total_pnl": 51.48,
    "realized_pnl_reported": 1970.33,
    "realized_pnl_from_trades": 1968.42,
    "unrealized_pnl": -1918.85,
    "equation_error": 0.00056,
    "status": "✅ VALID"
  },
  "value_function_validation": {
    "samples": 987,
    "correlation": -0.0234,
    "r_squared": 0.0789,
    "mean_value_pred": 45.23,
    "mean_actual_return": 2.15,
    "quality": "❌ POOR",
    "status": "❌ POOR"
  },
  "lookahead_bias_check": {
    "anomalies_found": 0,
    "status": "✅ CLEAN"
  },
  "summary": {
    "timestamp": "2026-06-06T10:30:00",
    "overall_status": "⚠️ ISSUES",
    "critical_issues": 1
  }
}
```

#### Interpretation Guide

| Field | Good Range | Bad Range |
|-------|-----------|-----------|
| `valid_trades` | > 98% | < 90% |
| `consistency_error` | < 0.01 | > 1.0 |
| `equation_error` | < 0.001 | > 0.1 |
| `r_squared` | > 0.5 | < 0.1 |
| `anomalies_found` | 0 | > 10 |

### Script 2: Generalization Test

**Purpose:** Test if agent generalizes across different market contexts

```bash
python3 scripts/test_generalization.py \
    --model ./checkpoints/episode_1 \
    --mode walk-forward \
    --output audit_results/generalization_test.json
```

#### Test Scenarios

**Scenario 1: Bullish → Bearish (HARD)**
- Train on Chunk 2 (bullish BTC)
- Test on Chunk 1 (bearish BTC)
- Expected: Some performance drop, but strategy should still work

**Scenario 2: Bearish → Bullish (EASY)**
- Train on Chunk 1 (bearish BTC)
- Test on Chunk 2 (bullish BTC)
- Expected: Slight improvement if agent is market-neutral

#### Output JSON Structure

```json
{
  "tests": {
    "train_chunk2_test_chunk1": {
      "train_metrics": {
        "sharpe": 5.9369,
        "win_rate": 0.498,
        "avg_return_per_trade": 0.062,
        "market_context": "Bullish"
      },
      "test_metrics": {
        "sharpe": 1.2345,
        "win_rate": 0.450,
        "avg_return_per_trade": 0.008,
        "market_context": "Bearish"
      },
      "comparison": {
        "sharpe_degradation_pct": 79.2,
        "quality": "❌ VERY_POOR"
      }
    }
  }
}
```

#### Interpretation

- **Degradation < 10%:** ✅ Good generalization
- **Degradation 10-30%:** ⚠️ Acceptable
- **Degradation 30-60%:** ⚠️ Significant overfitting
- **Degradation > 60%:** ❌ Agent is trend-follower

---

## Validation Criteria

### Trade Validation (Phase 1)

**Passes if:**
- ✅ `invalid_trades` < 2% of total
- ✅ `pnl_discrepancies` < 5
- ✅ `status` = "✅ VALID"

**Fails if:**
- ❌ `invalid_trades` > 5%
- ❌ `status` = "❌ ISSUES_FOUND"

**Action if Fails:**
```
→ Extract first 50 trades manually
→ Recalculate PnL: (exit - entry) × size × direction
→ Compare with reported pnl
→ Check portfolio_manager.close_position() logic
```

---

### Position Validation (Phase 2)

**Passes if:**
- ✅ `consistency_error` < 0.01
- ✅ `status` = "✅ CONSISTENT"

**Interpretation of Gap:**
```
Gap = Realized Equity - Portfolio Value
    = Sum of closed trades - Current portfolio value
    = Unrealized losses on open positions

Valid if:
  - Gap is negative (open positions down)
  - OR Gap is positive but < realized_pnl
  - Gap should be stable in final steps
```

**Fails if:**
- ❌ `consistency_error` > 1.0
- ❌ Gap oscillates wildly
- ❌ Gap is positive and large (suspicious)

**Action if Fails:**
```
→ Extract positions dict from final step
→ For each position: entry_price, current_price, size
→ Calculate unrealized = (current - entry) × size
→ Sum should match reported gap
```

---

### PnL Accounting (Phase 3)

**Passes if:**
- ✅ `equation_error` < 0.001
- ✅ `status` = "✅ VALID"

**Equation:** `realized_equity + unrealized_pnl = portfolio_value`

**Fails if:**
- ❌ `equation_error` > 0.1
- ❌ Realized PnL from trade log ≠ reported realized

**Action if Fails:**
```
→ Check if realized_pnl is cumulative or per-step
→ Check if positions are properly closed at episode end
→ Verify: total_realized_pnl reset to 0 at reset()
```

---

### Value Function (Phase 4)

**Passes if:**
- ✅ `r_squared` > 0.3
- ✅ `quality` = "✅ GOOD" or "⚠️ ACCEPTABLE"

**Fails if:**
- ❌ `r_squared` < 0.1
- ❌ `quality` = "❌ POOR"

**Interpretation:**
```
R² = Fraction of return variance explained by value function

High R² (> 0.5):  Value network is learning meaningful state features
Medium R² (0.1-0.5): Value network has some signal but noisy
Low R² (< 0.1):  Value network is useless, agent using heuristic
```

**Action if Fails:**
```
→ Retrain value network with:
  - Better hyperparameters
  - More training episodes
  - Different architecture (attention, residual blocks)
  
OR
  
→ Accept that agent is using simple heuristic
→ Validate that heuristic is profitable but not generalizable
```

---

### Lookahead Bias (Phase 5)

**Passes if:**
- ✅ `anomalies_found` = 0
- ✅ `status` = "✅ CLEAN"

**Fails if:**
- ❌ `anomalies_found` > 5
- ❌ `status` = "⚠️ POSSIBLE_BIAS"

**CRITICAL:** If lookahead bias detected, results are **INVALIDATED**

**Action if Fails:**
```
❌ STOP ALL TESTING

→ Review observation construction
  - Check if observation includes current_price only
  - Verify no future prices in state
  
→ Review reward calculation
  - Check if reward uses next_step prices
  - Verify all prices are from current step only
  
→ Retrain from scratch with fixed observation logic
```

---

## Generalization Test Interpretation

### Output Example

```json
{
  "tests": {
    "train_chunk2_test_chunk1": {
      "comparison": {
        "train_sharpe": 5.9369,
        "test_sharpe": 1.2345,
        "sharpe_degradation_pct": 79.2,
        "quality": "❌ VERY_POOR (Severe overfitting)",
        "train_market": "Bullish",
        "test_market": "Bearish"
      }
    },
    "train_chunk1_test_chunk2": {
      "comparison": {
        "train_sharpe": 0.8234,
        "test_sharpe": 5.9369,
        "sharpe_degradation_pct": -620.5,
        "quality": "✅ EXCELLENT (Improves on easier task)",
        "train_market": "Bearish",
        "test_market": "Bullish"
      }
    }
  }
}
```

### Verdict Matrix

| Train → Test | Degradation | Verdict | Action |
|---|---|---|---|
| Bullish → Bearish | > 70% | ❌ Not generalizable | Reject |
| Bullish → Bearish | 30-70% | ⚠️ Trend-dependent | Revisit strategy |
| Bullish → Bearish | < 30% | ✅ Generalizable | Accept |
| Bearish → Bullish | Improves | ✅ Neutral or better | Good sign |
| Bearish → Bullish | Degrades | ⚠️ Investigate | Possible issue |

---

## Complete Audit Workflow

### Step 1: Run Comprehensive Audit

```bash
python3 scripts/audit_chunk_comprehensive.py \
    --checkpoint ./checkpoints/episode_1 \
    --chunk 1 \
    --output chunk1_audit.json

# Review chunk1_audit.json
# Check all "status" fields
```

### Step 2: Identify Issues

```
If any status = "❌ FAILED":
  → Stop and fix that component first
  
If any status = "⚠️ WARNING":
  → Investigate but can continue
  
If all status = "✅ PASS":
  → Proceed to generalization test
```

### Step 3: Run Generalization Test

```bash
python3 scripts/test_generalization.py \
    --model ./checkpoints/episode_1 \
    --mode walk-forward \
    --output generalization_test.json

# Review generalization_test.json
# Check "quality" fields
```

### Step 4: Make Decision

```
If sharpe_degradation < 30%:
  → Agent is generalizable
  → Proceed to paper trading
  
If sharpe_degradation 30-60%:
  → Agent is context-dependent
  → Retrain with data augmentation or market-neutral rewards
  
If sharpe_degradation > 60%:
  → Agent is severely overfitted
  → ❌ DO NOT DEPLOY
  → Redesign strategy from scratch
```

---

## Common Issues & Solutions

### Issue: Trade validation fails (invalid_trades > 5%)

**Symptoms:**
- `status` = "❌ ISSUES_FOUND"
- Large PnL discrepancies

**Diagnosis:**
```bash
# Extract first 50 trades and check manually
head -50 trade_log.csv | awk '{
  pnl = ($exit - $entry) * $size * $direction
  print "Trade: " $id ", Expected: " pnl ", Got: " $pnl
}'
```

**Solution:**
1. Check `portfolio_manager.close_position()` logic
2. Verify size calculation: `size = notional / price`
3. Check direction encoding: `1=long, -1=short`

---

### Issue: Value function R² < 0.1

**Symptoms:**
- `r_squared` near 0
- Agent still profitable but value network useless

**Diagnosis:**
This is expected if agent uses simple heuristic (e.g., always buy in uptrend)

**Solution:**
1. Accept and document (value network not needed for heuristic)
2. OR retrain with curriculum learning
3. OR redesign value network architecture

---

### Issue: Generalization test shows 80%+ degradation

**Symptoms:**
- Sharpe 5.9 in bullish → Sharpe 1.2 in bearish
- Agent works only in trending markets

**Diagnosis:**
Agent is a trend-follower, not a market-neutral trader

**Solution:**
1. Add bearish training data (market-neutral rewards)
2. Implement portfolio hedge (long/short positions)
3. Use ensemble of bull + bear strategies

---

### Issue: Lookahead bias detected

**Symptoms:**
- `anomalies_found` > 0
- Agent knowing future prices

**CRITICAL ACTION:**
```bash
❌ STOP EVERYTHING

1. Review observation construction in env
2. Check if next_price used in current observation
3. Verify reward doesn't use future prices
4. Retrain from scratch
5. Run audit again
```

---

## Performance Benchmarks

### "Good" Results

```json
{
  "trade_validation": {
    "invalid_trades": 0,
    "pnl_discrepancies": 0,
    "status": "✅ VALID"
  },
  "position_validation": {
    "consistency_error": 0.0001,
    "status": "✅ CONSISTENT"
  },
  "pnl_validation": {
    "equation_error": 0.00001,
    "status": "✅ VALID"
  },
  "value_function_validation": {
    "r_squared": 0.35,
    "quality": "⚠️ ACCEPTABLE"
  },
  "lookahead_bias_check": {
    "anomalies_found": 0,
    "status": "✅ CLEAN"
  }
}
```

Plus:
```json
{
  "train_chunk2_test_chunk1": {
    "sharpe_degradation_pct": 25.0,
    "quality": "⚠️ ACCEPTABLE"
  }
}
```

### Red Flag Results

```json
{
  "trade_validation": {
    "invalid_trades": 12,
    "status": "❌ ISSUES_FOUND"
  }
}
```

OR

```json
{
  "value_function_validation": {
    "r_squared": 0.035,
    "quality": "❌ POOR"
  }
}
```

OR

```json
{
  "lookahead_bias_check": {
    "anomalies_found": 47,
    "status": "⚠️ POSSIBLE_BIAS"
  }
}
```

OR

```json
{
  "train_chunk2_test_chunk1": {
    "sharpe_degradation_pct": 85.0,
    "quality": "❌ VERY_POOR"
  }
}
```

---

## Next Steps After Audit

### If All Tests Pass ✅

1. Deploy to **paper trading** for 1 week
2. Monitor real-time performance
3. If paper trading succeeds, deploy to **live trading** with small capital

### If Tests Partially Fail ⚠️

1. Fix identified issues
2. Rerun just that phase of audit
3. Proceed to next phase

### If Tests Fail ❌

1. **Value function poor?** → Retrain network
2. **Trades invalid?** → Debug PnL logic
3. **Position inconsistent?** → Fix position tracking
4. **Generalization bad?** → Redesign strategy
5. **Lookahead bias?** → Start from scratch

---

## Appendix: File Locations

```
Repository Structure:
├── scripts/
│   ├── audit_chunk_comprehensive.py   # Main audit script
│   ├── test_generalization.py         # Walk-forward test
│   └── run_full_audit.sh              # Orchestration
├── audit_results/                     # Output directory
│   ├── chunk1_audit.json
│   ├── chunk2_audit.json
│   ├── generalization_test.json
│   └── AUDIT_REPORT_*.md
└── checkpoints/                       # Model checkpoints
    ├── episode_1/
    ├── episode_2/
    └── ...
```

---

## Questions?

Refer to:
- `ANALYSIS_COMPLETE_BUG_AUDIT.md` - Detailed bug analysis
- `METRICS_DEEP_DIVE.md` - Metrics interpretation
- `DATA_SNAPSHOT_AND_CHARTS.md` - Visualizations & data
- `AUDIT_CONCLUSIONS_AND_NEXT_STEPS.md` - Recommendations
