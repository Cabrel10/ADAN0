# 📌 MONITORING COMMANDS — ADAN0 TRAINING

Quick reference for monitoring and analyzing the training.

---

## 📊 LIVE MONITORING (Real-time)

### Check Process Status
```bash
cat /tmp/prod_pid.txt && ps -o pid,stat,%cpu,%mem,etime -p $(cat /tmp/prod_pid.txt)
```

**Expected output**:
```
108238
    PID STAT %CPU %MEM     ELAPSED
 108238 RNl   250 33.2    10:16:29
```

### Watch Progress (every 30 seconds)
```bash
while true; do
  tail -1 logs/training/fa_500k_prod_20260627_234932.log | grep -oE "\[STEP [0-9]+" || echo "..."
  sleep 30
done
```

### Latest Metrics (PPO stats)
```bash
tail -500 logs/training/fa_500k_prod_20260627_234932.log | grep -E "total_timesteps|value_loss|entropy_loss|approx_kl" | tail -10
```

---

## 📈 ANALYSIS AFTER TRAINING COMPLETES

### 1. Extract Final Metrics
```bash
cd ~/webapp/MORNINGSTAR/ADAN0
LOG="logs/training/fa_500k_prod_20260627_234932.log"

echo "=== FINAL SB3 METRICS ===" 
tail -100 "$LOG" | grep -E "total_timesteps|value_loss|entropy_loss|approx_kl|explained_variance"

echo ""
echo "=== FINAL TRADING METRICS ===" 
tail -100 "$LOG" | grep "METRICS_SYNC" | tail -5
```

### 2. Extract Trade Distribution
```bash
grep "TRADE_AUDIT_CLOSE" "$LOG" | tail -500 > /tmp/trades_dump.txt

# Count actions
echo "=== ACTION DISTRIBUTION ===" 
grep -o "Action=[A-Z]*" /tmp/trades_dump.txt | sort | uniq -c

# Count PnL signs
echo "=== PnL DISTRIBUTION ===" 
grep -o "pnl=[^,]*" /tmp/trades_dump.txt | sort | uniq -c
```

### 3. Win Rate Verification
```bash
# Extract closed trade PnLs
grep "TRADE_AUDIT_CLOSE" "$LOG" | grep "Status=CLOSED" | \
  awk '{for(i=1;i<=NF;i++) if($i ~ /^pnl/) print $i}' | \
  awk -F= '{print $2}' | \
  awk '{if ($1 > 0) wins++; else losses++} END {print "Wins:", wins, "Losses:", losses, "WR:", wins/(wins+losses)*100"%"}'
```

### 4. % HOLD vs Trading
```bash
# Fraction of steps where action was HOLD
grep "TRADE_AUDIT_CLOSE\|ACTION_DIFF" "$LOG" | tail -2000 | \
  grep "Requested=HOLD" | wc -l > /tmp/hold_count.txt

grep "TRADE_AUDIT_CLOSE\|ACTION_DIFF" "$LOG" | tail -2000 | \
  wc -l > /tmp/total_count.txt

python3 << 'EOF'
hold = int(open('/tmp/hold_count.txt').read())
total = int(open('/tmp/total_count.txt').read())
print(f"HOLD %: {hold/total*100:.1f}%")
print(f"Trading %: {(total-hold)/total*100:.1f}%")
EOF
```

---

## 🔧 QUICK DIAGNOSTICS

### Check for Crashes/Errors
```bash
grep -i "error\|exception\|fatal" "$LOG" | tail -20
```

### Check Memory Usage Over Time
```bash
# Rough estimate from process state
ps -o rss= -p $(cat /tmp/prod_pid.txt) | awk '{print $1/1024 " MB"}'
```

### Check Checkpoint Savings
```bash
ls -lht checkpoints/ppo_adan0_sandbox_checkpoint_*.zip | head -5
```

---

## 📋 POST-TRAINING ANALYSIS SCRIPT

Save this as `analyze_training.py`:

```python
#!/usr/bin/env python3
import re
from pathlib import Path
from collections import defaultdict

log_file = Path("logs/training/fa_500k_prod_20260627_234932.log")
lines = log_file.read_text().split('\n')

# Extract final epoch metrics
final_metrics = {}
for line in lines[-200:]:
    if 'total_timesteps' in line:
        m = re.search(r'\|\s+total_timesteps\s+\|\s+(\d+)', line)
        if m: final_metrics['total_timesteps'] = int(m.group(1))
    
    if 'value_loss' in line:
        m = re.search(r'\|\s+value_loss\s+\|\s+([\d.]+)', line)
        if m: final_metrics['value_loss'] = float(m.group(1))
    
    if 'entropy_loss' in line:
        m = re.search(r'\|\s+entropy_loss\s+\|\s+(-[\d.]+)', line)
        if m: final_metrics['entropy_loss'] = float(m.group(1))
    
    if 'explained_variance' in line:
        m = re.search(r'\|\s+explained_variance\s+\|\s+([\d.]+)', line)
        if m: final_metrics['explained_variance'] = float(m.group(1))

# Extract final trading metrics
final_trading = {}
for line in lines[-100:]:
    if '[METRICS_SYNC]' in line and 'Sharpe=' in line:
        m = re.search(r'Sharpe=([\d.]+), Sortino=([\d.]+), WinRate=([\d.]+)%', line)
        if m:
            final_trading['sharpe'] = float(m.group(1))
            final_trading['sortino'] = float(m.group(2))
            final_trading['win_rate'] = float(m.group(3))

print("=" * 60)
print("TRAINING FINAL REPORT")
print("=" * 60)

print("\nPPO Metrics:")
for k, v in final_metrics.items():
    print(f"  {k}: {v}")

print("\nTrading Metrics:")
for k, v in final_trading.items():
    print(f"  {k}: {v}")

print("\nDiagnosis:")
if final_trading.get('win_rate', 0) < 25:
    print("  ❌ CRITICAL: Win Rate < 25% suggests reward hacking")
    print("     → Investigate TRADE_AUDIT_CLOSE")
else:
    print("  ✅ Win Rate acceptable (> 25%)")

if final_metrics.get('entropy_loss', -1) < -7.5:
    print("  ⚠️  Policy converged very rapidly")
    print("     → Check if strategy is degenerate")

print("\n" + "=" * 60)
```

Run it:
```bash
python3 analyze_training.py
```

---

## 🎯 KEY QUESTIONS TO ANSWER

Before moving forward:

1. **What % of steps have action == HOLD?**
   - Answer: _____%
   - GOOD: 45-55% | BAD: > 75%

2. **What is the win rate of CLOSED trades?**
   - Answer: _____%
   - GOOD: > 35% | OKAY: 25-35% | BAD: < 25%

3. **What is average PnL per closed trade?**
   - Answer: $_____ 
   - Sign: [Positive / Negative]

4. **How long does agent hold positions on average?**
   - Answer: _____ minutes/hours/days
   - GOOD: 1-24h | BAD: > 100h (too passive)

5. **How many trades per day?**
   - Answer: _____
   - GOOD: 5-50 | BAD: < 2 (inert) or > 200 (noise)

6. **What is final entropy_loss value?**
   - Answer: _____
   - GOOD: > -8.0 | BAD: < -8.5 (over-converged)

---

## 📌 DECISION LOGIC

```
IF win_rate > 35% AND hold_pct < 60% AND pnl_avg > 0:
    ✅ ACCEPT → Deploy to paper
    
ELSE IF win_rate > 25% AND hold_pct < 70%:
    🟡 ITERATE → Apply Phase 2 fixes, re-run
    
ELSE IF win_rate <= 25% OR hold_pct > 75%:
    🔴 REDESIGN → Major reward function overhaul
    
ELSE:
    ❓ INVESTIGATE → Need manual review
```

---

## 🚀 SAMPLE OUTPUT

Expected output from analysis:

```
============================================================
TRAINING FINAL REPORT
============================================================

PPO Metrics:
  total_timesteps: 500000
  value_loss: 0.0075
  entropy_loss: -7.35
  explained_variance: 0.31

Trading Metrics:
  sharpe: 3.8
  sortino: 6.5
  win_rate: 21.2

Diagnosis:
  ❌ CRITICAL: Win Rate < 25% suggests reward hacking
     → Investigate TRADE_AUDIT_CLOSE
  ⚠️  Policy converged very rapidly
     → Check if strategy is degenerate

============================================================
```

If this is the output → **MAJOR REDESIGN NEEDED** (see ACTION_PLAN_POST_TRAINING.md)

