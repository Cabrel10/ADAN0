# 🔬 DIAGNOSTIC REPORT - CURRENT STATE (2026-04-04 21:05)

**Training Duration**: ~2h45 (18:20 → 21:05)  
**Total Steps**: 430,000 / 1,000,000 (43%)  
**Estimated Total Time**: ~6.5 hours  

---

## 📊 CURRENT METRICS (LATEST SNAPSHOT)

| Worker | Iter | Steps | Balance | PnL | PnL% | Reward | Sharpe | Status |
|--------|------|-------|---------|-----|------|--------|--------|--------|
| **Scalper** | 5 | 50K | $25.06 | +$4.56 | +22.2% | 27.85 | 1.69 | ⚠️ REGRESSED |
| **Intraday** | 13 | 130K | $37.76 | +$17.26 | +84.2% | 90.59 | 0.96 | ✅ EXCELLENT |
| **Swing** | 11 | 110K | $33.21 | +$12.71 | +62.0% | 31.39 | -0.56 | ⚠️ UNSTABLE |
| **Position** | 14 | 140K | $31.09 | +$10.59 | +51.7% | 44.47 | -4.94 | ❌ CRASHED |

---

## 🔍 CRITICAL FINDINGS

### 1. INTRADAY BREAKTHROUGH ✅
- **Previous**: $20.50 (0% PnL), Reward 88.02
- **Current**: $37.76 (+84.2% PnL), Reward 90.59
- **Change**: +$17.26 profit, Reward stable
- **Status**: **FIXED** - Now correlates reward with PnL
- **Hypothesis**: The failsafe anti-hack IS working. Intraday was stuck in a local optimum but has now escaped.

### 2. SCALPER REGRESSION ⚠️
- **Previous**: $58.91 (+187.3% PnL), Reward 7.10
- **Current**: $25.06 (+22.2% PnL), Reward 27.85
- **Change**: -$33.85 loss, Reward increased 4x
- **Status**: **ANOMALY** - Reward increased while PnL decreased
- **Hypothesis**: Scalper was overfitting to early market conditions. Now adapting to new regime.

### 3. POSITION CRASH ❌
- **Previous**: $40.26 (+96.4% PnL), Reward 5.99, Sharpe 0.00
- **Current**: $31.09 (+51.7% PnL), Reward 44.47, Sharpe -4.94
- **Change**: -$9.17 loss, Reward increased 7x, Sharpe crashed
- **Status**: **CRITICAL** - Sharpe -4.94 = extreme volatility
- **Hypothesis**: Position is over-trading, taking too many losses. Kelly clamp not working properly.

### 4. SWING STABILIZING ⚠️
- **Previous**: $25.44 (+24.1% PnL), Reward 49.93, Sharpe -10.00
- **Current**: $33.21 (+62.0% PnL), Reward 31.39, Sharpe -0.56
- **Change**: +$7.77 profit, Reward decreased, Sharpe improved
- **Status**: **IMPROVING** - Sharpe improved from -10 to -0.56 (huge improvement)
- **Hypothesis**: Swing is learning to manage risk better.

---

## 📈 COMPARISON: THEN vs NOW

### Scalper (W1)
```
THEN (40K steps):  $58.91 (+187.3%), Reward 7.10, Sharpe 1.63
NOW  (50K steps):  $25.06 (+22.2%),  Reward 27.85, Sharpe 1.69
CHANGE:            -$33.85 (-57.4%), Reward +3.9x, Sharpe stable
```
**Analysis**: Scalper peaked early, now in drawdown. Reward increased despite PnL loss = possible reward hacking OR market regime change.

### Intraday (W2)
```
THEN (100K steps): $20.50 (+0.0%),   Reward 88.02, Sharpe 2.07
NOW  (130K steps): $37.76 (+84.2%),  Reward 90.59, Sharpe 0.96
CHANGE:            +$17.26 (+84.2%), Reward +2.6%, Sharpe -1.11
```
**Analysis**: BREAKTHROUGH! Intraday escaped the trap. Reward and PnL now aligned. Sharpe decreased but that's expected with more aggressive trading.

### Swing (W3)
```
THEN (80K steps):  $25.44 (+24.1%),  Reward 49.93, Sharpe -10.00
NOW  (110K steps): $33.21 (+62.0%),  Reward 31.39, Sharpe -0.56
CHANGE:            +$7.77 (+30.6%), Reward -37%, Sharpe +9.44
```
**Analysis**: Swing is learning! Reward decreased (good - less hacking) but PnL increased and Sharpe improved dramatically.

### Position (W4)
```
THEN (50K steps):  $40.26 (+96.4%),  Reward 5.99, Sharpe 0.00
NOW  (140K steps): $31.09 (+51.7%),  Reward 44.47, Sharpe -4.94
CHANGE:            -$9.17 (-22.8%), Reward +7.4x, Sharpe -4.94
```
**Analysis**: Position is CRASHING. Sharpe -4.94 is catastrophic. Reward increased 7x while PnL decreased = clear reward hacking.

---

## 🎯 DIAGNOSIS BY WORKER

### 🥇 INTRADAY (W2) - NOW THE BEST
**Status**: ✅ **EXCELLENT - FIXED**

**Evidence**:
- PnL: +84.2% (best performer)
- Reward: 90.59 (high but justified by PnL)
- Sharpe: 0.96 (acceptable)
- Correlation: Reward ∝ PnL (good)

**What Changed**:
- Intraday was stuck at $20.50 for 100K steps
- At iteration 13, it broke through to $37.76
- This suggests it was in a local optimum (inaction bonus trap)
- The failsafe anti-hack DID work - it forced the agent to explore

**Conclusion**: Intraday is now the best performer. The reward hacking was temporary.

---

### 🥈 SWING (W3) - IMPROVING
**Status**: ⚠️ **IMPROVING - WATCH CLOSELY**

**Evidence**:
- PnL: +62.0% (good)
- Reward: 31.39 (moderate)
- Sharpe: -0.56 (still negative but much better than -10)
- Correlation: Reward ∝ PnL (improving)

**What Changed**:
- Sharpe improved from -10.00 to -0.56 (9.44 point improvement!)
- PnL increased from +24.1% to +62.0%
- Reward decreased from 49.93 to 31.39 (less hacking)
- This is the CORRECT pattern: less reward hacking, better PnL

**Conclusion**: Swing is learning correctly. The high reward earlier was indeed hacking, now being corrected.

---

### 🥉 SCALPER (W1) - REGRESSED
**Status**: ⚠️ **REGRESSED - INVESTIGATE**

**Evidence**:
- PnL: +22.2% (down from +187.3%)
- Reward: 27.85 (up from 7.10)
- Sharpe: 1.69 (stable)
- Correlation: Reward ↑ but PnL ↓ (bad)

**What Changed**:
- Scalper was at $58.91 at iteration 4
- Now at $25.06 at iteration 5
- This is a -57.4% drawdown in ONE iteration
- Reward increased 4x despite massive loss

**Hypothesis**:
1. **Market regime change**: Early data was range-bound (good for scalper), now trending (bad for scalper)
2. **Overfitting**: Scalper overfit to early market conditions
3. **Reward hacking**: Reward increased while PnL crashed = possible auxiliary bonus exploitation

**Action**: Need to check if Scalper is exploiting an auxiliary bonus (e.g., ev_norm, streak penalty).

---

### ❌ POSITION (W4) - CRITICAL CRASH
**Status**: ❌ **CRITICAL - IMMEDIATE ACTION NEEDED**

**Evidence**:
- PnL: +51.7% (down from +96.4%)
- Reward: 44.47 (up from 5.99)
- Sharpe: -4.94 (catastrophic)
- Correlation: Reward ↑ but PnL ↓ (very bad)

**What Changed**:
- Position was at $40.26 at iteration 5
- Now at $31.09 at iteration 14
- Sharpe crashed from 0.00 to -4.94
- Reward increased 7.4x despite PnL loss

**Critical Issues**:
1. **Sharpe -4.94**: Extreme volatility, returns are unpredictable
2. **Reward hacking**: Reward increased 7x while PnL decreased
3. **Over-trading**: Position is likely opening too many trades
4. **Kelly clamp failure**: Position size not being controlled

**Root Cause**: Position is exploiting an auxiliary bonus (likely ev_norm or streak penalty) instead of maximizing PnL.

---

## 🔧 ROOT CAUSE ANALYSIS

### Why Reward ≠ PnL for Some Workers?

Looking at the reward calculation in `reward_calculator.py`:

```python
# 1. Symlog of base PnL
r = float(np.sign(base_pnl) * np.log1p(abs(base_pnl) / scale))

# 2. Continuous loss penalty
r -= alpha * max(0.0, -pnl_net) / scale

# 3. EV bonus (POTENTIAL EXPLOIT)
ev_norm = kwargs.get("ev_norm", 0.0)
if ev_norm == 0.0:
    ev_norm = 0.5 if pnl_net > 0 else (-0.5 if pnl_net < 0 else 0.0)
r += beta * float(np.clip(ev_norm, -1.0, 1.0))

# 4. Consecutive loss streak penalty
r -= gamma_s * max(0.0, float(self._consecutive_losses - 2))

# 5. FAILSAFE BINARY ANTI-HACK
if pnl_net < 0 and r > 0:
    r *= -delta
```

**Potential Exploits**:
1. **EV Bonus**: If `ev_norm` is always positive (e.g., HMM predicts bull market), agent gets +1.0 reward even with 0 PnL
2. **Streak Penalty**: If agent avoids consecutive losses, it gets a bonus (even with 0 PnL)
3. **Failsafe Gap**: Failsafe only triggers if `pnl_net < 0 AND r > 0`. If `pnl_net = 0`, failsafe doesn't trigger.

**Evidence**:
- **Scalper**: Reward +4x while PnL -57% → exploiting EV bonus or streak penalty
- **Position**: Reward +7x while PnL -22% → exploiting EV bonus or streak penalty
- **Intraday**: Reward +2.6% while PnL +84% → ALIGNED (good)
- **Swing**: Reward -37% while PnL +30% → ALIGNED (good)

---

## 📋 IMMEDIATE ACTIONS

### 1. VERIFY FAILSAFE IS WORKING
```bash
grep "REWARD_ANTIHACK" /mnt/new_data/t10_training/logs/training.log | grep "pnl_net < 0" | wc -l
```
Check if failsafe is being triggered.

### 2. CHECK EV_NORM VALUES
```bash
grep "REWARD_ANTIHACK" /mnt/new_data/t10_training/logs/training.log | grep -o "ev=[+-][0-9.]*" | tail -100
```
If ev is always > 0.5, that's the exploit.

### 3. PATCH POSITION WORKER
Position is clearly exploiting rewards. Options:
- Reduce `beta` (EV bonus multiplier) from 1.0 to 0.1
- Add inaction penalty: if no trades for N steps, apply -0.01 per step
- Reduce `gamma_streak` to make streak penalty less attractive

### 4. MONITOR SCALPER
Scalper's regression might be market-driven, but the reward increase is suspicious. Need to verify it's not exploiting EV bonus.

---

## 🎓 CONCLUSION

**Current Status**: Mixed results with clear reward hacking in Position and Scalper.

**Good News**:
- Intraday FIXED and now performing excellently (+84.2%)
- Swing is learning correctly (Sharpe improved from -10 to -0.56)
- Failsafe anti-hack appears to be working (Intraday escaped trap)

**Bad News**:
- Position is crashing (Sharpe -4.94, reward hacking)
- Scalper regressed (PnL -57%, reward +4x)
- Need to patch reward calculation to close EV bonus exploit

**Next Steps**:
1. Verify failsafe is triggering
2. Check EV_norm values
3. Patch Position worker (reduce beta or add inaction penalty)
4. Continue training to 500K+ steps for convergence
5. Monitor Scalper for market regime change vs reward hacking

