# 🔬 REWARD/PENALTY BALANCE ANALYSIS — Session 12 Final

## QUESTION: Est-ce que les rewards compensent les penalties? Sommes-nous équilibrés?

### 📊 VERDICT: **NON. System est BROKEN. Rewards << Penalties.**

---

## 1️⃣ REWARD COMPONENTS (Positive Incentives)

### A. PnL Base Reward
```python
pnl_base_reward = pnl_pct * 0.5
```
- **Scale**: 0.5× (upgraded from 0.1× in Session 11b)
- **Typical episode**: Capital $20.50 → $18.35 = **-9.25% loss**
- **Episode cumulative**: -9.25% × 0.5 = **-0.046 total reward from PnL**
- **Per step avg**: -0.046 / 2000 steps ≈ **-0.000023/step**

❌ **Problem**: With 0.80% fees and tight SL, losses are NORMAL. This baseline signal is **massively negative**.

---

### B. Promotion Bonus
```python
promotion_bonus = tier_info.get("promotion_bonus", 0.5)
```
- **Micro → Small**: +5.0 (one-time)
- **Small → Medium**: +10.0 (one-time)
- **Medium → High**: +20.0 (one-time)
- **High → Enterprise**: +40.0 (one-time)

**Reality Check**: 
- Starting capital: $20.50 (Micro tier: $11-$30)
- To reach Small tier: need $30+ capital
- With -25% losses already at step 2045, agent is **going DOWN in tiers, not UP**
- Current state: $15.30 = still Micro Capital = **$0 promotion bonus**

❌ **Critical**: Promotion bonus is irrelevant if agent is hemorrhaging capital.

---

### C. Survival Bonus
```python
survival_bonus = 0.001  # +0.001/step just for existing
```
- **Per step**: +0.001
- **Episode total (2000 steps)**: +0.001 × 2000 = **+2.0**

✅ **This is the ONLY positive signal right now.**

---

### D. Patience Bonus (NEW)
```python
if steps_since_last_trade > 100:
    patience_bonus_val = 0.005 * log(steps_since_last_trade - 100)
```
- **Activation**: Only after 100 steps without trade
- **At 200 steps waiting**: +0.005 × log(101) ≈ +0.025
- **Max realistic (500 steps waiting)**: +0.005 × log(401) ≈ +0.033

🟡 **Problem**: Agent is actively trading (loses money), so this never activates.

---

## 2️⃣ PENALTY COMPONENTS (Negative Incentives)

### A. Drawdown Penalty (QUADRATIC — HARSH)
```python
drawdown_penalty = -50.0 * (abs_dd ** 2) * dd_factor
```

**Real values observed**:
- At step 2045: Portfolio $15.30, Initial $20.50
- Drawdown: -25.4%
- Calculation: `-50 × (0.254)² × 2.0` = `-50 × 0.0645 × 2.0` = **-6.45 PER STEP**

❌ **CATASTROPHIC**: A single -25% drawdown adds **-6.45 reward penalty every single step**.

**Cumulative at step 2045**: -6.45 × 2045 ≈ **-13,191 total penalty**

---

### B. Stagnation Penalty (Logarithmic)
```python
if steps_in_tier > max_steps_in_tier:
    stagnation_penalty = stagnation_rate * log1p(excess_steps)
```

**For Micro tier** (max 500 steps):
- At step 2045: excess = 2045 - 500 = 1545 steps
- Penalty: `-0.0005 × log(1546)` ≈ `-0.0005 × 7.34` = **-0.00367/step**
- Episode total: **-0.00367 × 2045 ≈ -7.5**

🟡 **Moderate penalty but ADDITIVE to drawdown.**

---

### C. Demotion Penalty
```python
demotion_penalty = -float(previous_tier_info.get("promotion_bonus", 0.5))
```

- If tier drops: **matches the promotion bonus** (e.g., -5.0 for Micro drop)
- Currently: Agent is stuck in Micro, so no demotion yet
- But when capital hits $11 (tier floor), demotion to... nothing = reset

🔴 **Existential risk**: Below Micro tier = game over.

---

### D. Time Decay Baseline
```python
time_decay: -0.01
```

- **Per step**: -0.01
- **Episode total (2000 steps)**: -0.01 × 2000 = **-20.0**

❌ **This is ALWAYS active, every step.**

---

### E. Invalid Trade Penalty
```python
invalid_trade_penalty_weight: 0.005
```

- Each invalid action (trading when gate closed): -0.005
- At step 2045: 114 invalid attempts → `-0.005 × 114` ≈ **-0.57**

🟡 **Minor compared to others.**

---

## 3️⃣ BALANCE SHEET: What's Actually Happening?

### Episode Snapshot (Current run, step ~2045)

| Component | Value | Notes |
|-----------|-------|-------|
| **REWARDS** |  |  |
| PnL Base (-9.25%) | -0.046 | Getting worse |
| Promotion Bonus | 0.0 | Not achieved |
| Survival Bonus | +2.0 | Only positive signal |
| Patience Bonus | 0.0 | No holding >100 steps |
| **TOTAL REWARDS** | **+2.0** | Anemic |
| **PENALTIES** |  |  |
| Time Decay (-0.01/step) | -20.0 | **Relentless** |
| Drawdown (-25.4%) | -13,191 | **CATASTROPHIC** |
| Stagnation | -7.5 | Additive |
| Invalid Trades | -0.57 | Minor |
| **TOTAL PENALTIES** | **-13,219** | Overwhelms everything |
| **NET (Raw)** | **-13,217** | 🔴 Massively negative |
| **After Symlog** | ~-3.5 to -4.0 | Still crushing |

---

## 4️⃣ ROOT CAUSE: Why Agent is Dying

### Problem 1: Drawdown Penalty is EXPONENTIAL OVERKILL
- **Formula**: `-50 × (drawdown)² × factor`
- **At -10%**: `-50 × 0.01 × 2.0 = -1.0` ✓ (reasonable)
- **At -20%**: `-50 × 0.04 × 2.0 = -4.0` ❌ (harsh but surviv able)
- **At -25%**: `-50 × 0.0625 × 2.0 = -6.25` ❌❌ (death penalty, every step)

**The agent gets -6.25 reward EVERY STEP for being -25% down.**

→ **After 100 steps in drawdown: cumulative = -625 penalty**

→ **Agent knows: recovery is impossible, surrender is optimal**

---

### Problem 2: Time Decay Prevents Recovery
- **-0.01/step** means the agent LOSES reward just for existing
- With 0.80% fees and 2% SL, every trade is fight-or-die
- Recovery from -25% requires ~50+ winning trades at 0.5% avg profit/trade
- But time decay costs -0.01 × 50 = -0.5 just to attempt recovery
- **Math**: Win +0.05 on trade, pay -0.01 time decay = +0.04 net → **Too slow**

---

### Problem 3: Survival Bonus is TOO SMALL
- **Survival bonus**: +0.001/step
- **Time decay**: -0.01/step
- **Net baseline**: -0.009/step (always negative, always dragging)
- **Agent learns**: "Death is better than this torture" → stop trading → bleed out

---

### Problem 4: Rewards Only Come from Promotion/Winning
- **Promotion**: Unreachable (agent is going DOWN)
- **Winning trades**: Impossible at 2% SL + 0.80% fees + market noise
- **Only reward source left**: Survival bonus (+0.001/step) — **pathetic**

---

## 5️⃣ THE IMBALANCE: Numbers Don't Lie

### If agent could make a winning trade:
- **Best-case**: +0.5% realized PnL → `+0.005 × 0.5 = +0.0025` reward
- **Cost to survive during trade**:
  - Time decay: -0.01
  - Holding time (assume 10 steps): -0.10
  - Total survival cost: **-0.10**
- **Net for winning trade**: +0.0025 - 0.10 = **-0.0975** ❌

**Even a winning trade is PUNISHED.**

---

### If agent goes on 10-trade winning streak (+0.5% avg):
- **Total realized PnL**: +5%
- **Reward**: 0.05 × 0.5 = **+0.025**
- **Time decay (100 steps)**: -1.0
- **Net for winning streak**: +0.025 - 1.0 = **-0.975** ❌

**The agent is penalized for trading at all, even when winning.**

---

## 6️⃣ DIAGNOSIS: System is STRUCTURALLY BROKEN

### The Core Problem
```
Rewards favor:
  ✓ Promotion (unreachable)
  ✓ Patience (but prevents profit)
  ✓ Survival (too weak to counter penalties)

Penalties crush:
  ✗ Time decay (-0.01/step, relentless)
  ✗ Drawdown (quadratic, exponential growth)
  ✗ Stagnation (logarithmic, always positive)

Result: Agent is incentivized to:
  1. NOT TRADE (avoid SL risk)
  2. NOT HOLD (time decay punishes waiting)
  3. SURRENDER (death is easier than -0.01/step torture)
```

---

## 7️⃣ WHAT'S NEEDED: REBALANCING

### Option A: Reduce Time Decay
```yaml
# Current: -0.01/step
# Proposed: -0.001/step (10× reduction)
time_decay: -0.001
```
- **Rationale**: Agent needs room to breathe during recovery
- **Cost**: Slightly weaker signal to explore
- **Benefit**: Winning trades become net positive

### Option B: Increase Survival Bonus
```yaml
# Current: +0.001/step
# Proposed: +0.01/step (10× increase)
survival_bonus: 0.01
```
- **Rationale**: Counterbalance time decay
- **Net baseline**: -0.01 + 0.01 = 0.0 (neutral, no bleed)
- **Benefit**: Agent not punished for existing

### Option C: Reduce Drawdown Penalty Factor
```yaml
# Current: Micro = 2.0 factor, formula = -50 × dd² × 2.0
# Proposed: Micro = 0.5 factor (4× reduction)
drawdown_penalty_factor: 0.5  # was 2.0
```
- **At -25%**: -50 × 0.0625 × 0.5 = -1.56 (vs -6.25) — 4× lighter
- **Rationale**: Drawdown is already visible in PnL reward; quadratic is extra punishment
- **Benefit**: Recovery becomes theoretically possible

### Option D: Cap Symlog Compression
```python
# Current: symlog compresses all rewards
# Issue: -13,000 raw → -3.5 symlog (still crushing)
# Proposed: Separate symlog for rewards vs penalties
final_reward = sign(raw) * log1p(abs(raw)) * 0.5  # Reduce magnitude
```

---

## 8️⃣ IMMEDIATE ACTIONS REQUIRED

**If we want agent to survive this session**:

1. **Kill the current training** (PID 1326036)
2. **Adjust config**:
   - `time_decay: -0.001` (was -0.01)
   - `survival_bonus: 0.01` (was +0.001)
   - `drawdown_penalty_factor: 0.5` for Micro (was 2.0)
3. **Test theory** with smaller episode on paper:
   - Run 100 steps manually to see if math works
4. **Relaunch training**

---

## SUMMARY TABLE: Current vs. Proposed

| Metric | Current | Proposed | Impact |
|--------|---------|----------|--------|
| Time Decay | -0.01/step | -0.001/step | Reduce torture |
| Survival Bonus | +0.001/step | +0.01/step | Counterbalance |
| Drawdown Factor (Micro) | 2.0× | 0.5× | Make recovery possible |
| Net Baseline | -0.009/step | 0.0/step | Neutral, no bleed |
| Winning Trade (+0.5%) | -0.0975 | +0.0025 | Positive incentive |

---

## FINAL WORD

**Current system rewards caution so harshly that the agent chooses surrender.**

The agent is literally learning: *"It's better to do nothing than to try."*

This is not a feature. This is a **bug in reward design**.
