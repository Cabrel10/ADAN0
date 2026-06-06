# Session 12: Immediate Action Items

**Status**: Ready to Launch Training Test  
**Time Estimate**: 30-40 minutes for validation run

---

## What's Done ✅

1. **Drawdown Penalty Fixed** (Quadratic formula, not sigmoid)
   - File: `src/adan_trading_bot/environment/multi_asset_chunked_env.py` (Lines 5990–6005)
   - Agent now learns: "Lose >5% = punished severely"

2. **SL/TP Widened** (1–2.5% instead of 0.3–0.8%)
   - Two locations updated consistently
   - Files: `src/adan_trading_bot/environment/multi_asset_chunked_env.py` (Lines 1142–1150 and 6975–6990)
   - Expected: Fewer false stops, more profitable trades

3. **Code Validated** ✅ Compiles cleanly

4. **User Queries Addressed** ✅ All three verified

---

## Launch Training Test

**Step 1**: Prepare environment
```bash
# Free RAM
killall chrome 2>/dev/null || true
sync && echo 3 > /proc/sys/vm/drop_caches

# Set Ray timeout
export RAY_GCS_RPC_CLIENT_TIMEOUT_S=2400

# Start training
bash scripts/launch_training.sh --light --resume
```

**Step 2**: Monitor (open new terminal)
```bash
# Watch logs in real-time
tail -f logs/central/adan_*.log | grep -E "DRAWDOWN_PENALTY|AGENT_CLOSE|TIER_REWARD|CRASHED"
```

**Step 3**: Check at 10-minute mark
```bash
# Verify:
# 1. Process still running?
ps aux | grep python | grep -v grep | wc -l

# 2. Portfolio value (should be >$15)?
tail -50 logs/central/adan_*.log | grep "Capital="

# 3. Drawdown penalty working?
grep DRAWDOWN_PENALTY logs/central/adan_*.log | tail -5
```

---

## Expected Behavior (Training Test)

### Healthy Signs (Want to See)
- ✅ Training continues beyond 20 minutes
- ✅ Logs show `[DRAWDOWN_PENALTY] DD=-X.XX | penalty=-Y.YYYY`
- ✅ Agent takes `[AGENT_CLOSE]` actions
- ✅ Portfolio stays in $15–$25 range
- ✅ Tier progression attempts visible

### Warning Signs (Stop & Investigate)
- ❌ New crash at 2–3 minutes (OOM? GCS timeout?)
- ❌ NaN in reward logs
- ❌ Agent always taking same action (policy frozen)
- ❌ Portfolio dropping rapidly (< $10)

---

## If Successful: Commit & Push

```bash
# Stage changes
git add src/adan_trading_bot/environment/multi_asset_chunked_env.py

# Commit
git commit -m "Session 12: Fix drawdown penalty quadratic + widen SL/TP bounds

- Replace sigmoid tanh() with quadratic -50*(dd%)² for harsh loss penalties
- Widen scalper SL from 0.3-0.8% to 1-2.5% (reduces noise-induced stops)
- Both locations (update_market_price + _execute_trades) now consistent
- Expected: Fix -34% capital collapse from S11b, restore profitability"

# Push to GitHub
git push -u origin genspark_ai_developer
```

Then create PR on GitHub main branch.

---

## If Unsuccessful: Debugging Steps

### Scenario 1: Same Crash at ~2200 steps
**Action**: Check Ray logs
```bash
grep ERROR logs/ray_results/*/logs/gcs_server.out | tail -20
```

**Likely Cause**: Still memory issue, not code issue
**Next Step**: Reduce workers to 1, or increase machine RAM

### Scenario 2: Drawdown Penalty Not Firing
**Action**: Check logs
```bash
grep DRAWDOWN_PENALTY logs/central/adan_*.log | head -10
```

**Likely Cause**: Portfolio drawdown < 1% (not triggered)
**Check**: Is agent actually losing money or just holding?

### Scenario 3: Agent Paralyzed (Same Action Every Step)
**Action**: Check policy output
```bash
grep "ACTION_DIFF" logs/central/adan_*.log | tail -10
```

**Likely Cause**: Neural network gradient exploded
**Next Step**: Reduce learning rate in config

---

## Expected Results After 30 min

**Metric** | **Session 11b (Failed)** | **Session 12 (Target)**
----------|-------------------------|---------------------
Duration | 2.7 min | >30 min
Capital Start | $20.50 | $20.50
Capital @ 10min | $14.34 | $18–$22 ✨
Capital @ 30min | *Crashed* | $22–$28 (promotion?)
Drawdown Penalty | -0.468 (soft) | -0.125 to -2.0 (harsh) ✨
SL Width | 0.5% (noisy) | 1% (stable) ✨

*✨ = Expected improvement*

---

## Summary

You've implemented two critical fixes:

1. **Punishment Scales Now**: Losing 20% is 4× worse than losing 10%
2. **Stops Are Rational**: SL respects market noise, doesn't get stopped-out for no reason

Combined, these should let the agent actually learn a profitable strategy. If training runs for 30+ minutes without collapse, the fixes worked.

**Launch test now!** ➡️ `bash scripts/launch_training.sh --light --resume`

