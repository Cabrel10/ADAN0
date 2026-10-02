# Session 11b Summary: Tier-Based Reward Asymmetry Adjustment

**Date**: 2026-06-05  
**Status**: ✅ Code Complete, 🚀 Training Restarted (PID 1026570)

---

## What Was Fixed

### Problem
Previous tier-based reward system had **extreme asymmetry** that made training feel "impossible":
- Stagnation penalty: -0.0146/step (too harsh)
- Promotion bonus: +0.5 (too weak relative to daily penalties)
- Agent was losing -7.3 reward just for existing 500 steps

### Solution: 10× Promotion + 4× Softer Stagnation + 5× PnL

#### Configuration Changes (`config/config.yaml`)

| Component | Before | After | Factor |
|-----------|--------|-------|--------|
| **Promotion Bonus (Micro)** | +0.5 | +5.0 | ×10 |
| **Promotion Bonus (Small)** | +1.0 | +10.0 | ×10 |
| **Promotion Bonus (Medium)** | +2.0 | +20.0 | ×10 |
| **Promotion Bonus (High)** | +4.0 | +40.0 | ×10 |
| **Stagnation Rate (Micro)** | -0.002 | -0.0005 | ÷4 |
| **Stagnation Rate (Small)** | -0.001 | -0.00025 | ÷4 |
| **Stagnation Rate (Medium)** | -0.0005 | -0.000125 | ÷4 |
| **Stagnation Rate (High)** | -0.0002 | -0.00005 | ÷4 |

#### Reward Function Changes (`multi_asset_chunked_env.py`)

```python
# Before: pnl_reward = pnl_pct * 0.1
# After:
pnl_base_reward = pnl_pct * 0.5  # ×5 stronger

# New survival bonus (prevents suicide):
survival_bonus = 0.001  # +0.001 per step just for existing
```

### Math: New Asymmetry (Balanced)

**Scenario**: Agent stuck in Micro tier, no profitable trades

```
Over 500 steps:
- Stagnation penalty: -0.0005 × ln(1 + 500) ≈ -0.0035/step × 500 = -1.75 total
- Inaction penalty: -0.01/step × 500 = -5.0 total
- Survival bonus: +0.001/step × 500 = +0.5 total
- Net pressure: -1.75 - 5.0 + 0.5 = -6.25 (tough but not "impossible")

When agent achieves promotion (Micro → Small):
- Promotion bonus: +5.0 (clears 3 days of stagnation penalty!)
- Message to agent: "Your effort was worth it"
```

**Compare to before**:
- Old: -7.3 stagnation + -5.0 inaction = -12.3 (agent thinks "I can't win")
- New: -1.75 stagnation + -5.0 inaction + 0.5 survival = -6.25 (agent thinks "I need to try harder")

---

## Training Progress

### Session 11a (Failed)
- Duration: ~9 minutes (~2050 steps)
- Crash: Ray GCS timeout (infrastructure, not code)
- Agent Status: Trading actively, capital $15.24

### Session 11b (Current)
- **PID**: 1026570
- **Start**: 2026-06-05 17:56:34 UTC
- **Mode**: `--light` (2 workers, resume from checkpoint)
- **Config**: RAY_GCS_RPC_CLIENT_TIMEOUT_S=2400 (2× longer timeout)
- **Status**: ✅ Running

---

## Key Insights

### 1. The Asymmetry Was Intentional (Prop Firm Model)
- Real prop firms: "Small profits OK, big losses = fired"
- Our system: same concept

### 2. But Too Extreme Was Counterproductive
- Agent learned: "Every day in Micro = -1.75 points"
- Compounded: -7.3 total per 500 steps made promotion feel impossible
- **Fix**: Divide stagnation by 4, multiply promotion by 10

### 3. Survival Bonus Prevents "Suicide" Strategies
- Without it: agent might intentionally crash account to avoid long-term penalty
- With +0.001/step: agent is incentivized to stay alive (small positive)

### 4. Ray Does NOT Need Internet
- GCS crash was local resource issue (memory/CPU)
- **NOT** GitHub/SSH/network related
- Restarted with 2× timeout + freed RAM = should be stable

---

## Files Changed

1. **`config/config.yaml`** (lines 1253–1295)
   - 10× promotion bonuses
   - 4× softer stagnation rates
   
2. **`src/adan_trading_bot/environment/multi_asset_chunked_env.py`** (lines 5945–6072)
   - 5× PnL reward amplification
   - +0.001 survival bonus
   - Updated TIER_REWARD logging
   
3. **`TRAINING_CRASH_AUDIT_REPORT.md`** (new)
   - Root cause analysis
   - Solutions & recommendations

---

## Next Monitoring Targets

### Healthy Indicators (Expected)
- ✅ Training continues beyond 10 minutes
- ✅ Agent attempts trades (BUY/SELL actions)
- ✅ Portfolio fluctuates ($15–25 range in Micro)
- ✅ TIER_REWARD logs appear every 50 steps

### Early Warning Signs (Would Indicate Problems)
- ❌ New crashes (GCS timeout again → memory/CPU issue)
- ❌ NaN rewards (logic error in new code)
- ❌ Agent stuck in one action (policy broken)

---

## Commits Needed

- ✅ Code changes committed to `genspark_ai_developer` branch (commit 590c12b)
- ⏳ GitHub push pending (SSH auth needed)
- ⏳ Merge to main pending (after successful training validation)

---

**Status**: Ready to monitor training and validate improvements.  
**Recommendation**: Run for at least 1 hour (20k+ steps) before final assessment.
