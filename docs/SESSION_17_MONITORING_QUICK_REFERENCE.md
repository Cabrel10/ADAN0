# Session 17: Monitoring Quick Reference Card

**Use this during training to quickly verify everything is working**

---

## 🚀 LAUNCH

```bash
bash scripts/run_adan_pro.sh
```

---

## ✅ WHAT YOU SHOULD SEE

### Startup (First 1-2 minutes)

```
✅ System cleanup complete
✅ Filesystem optimization complete
✅ Directories verified
✅ Environment configured
🎯 FRESH START mode  (or 🔄 RESUME MODE if resuming)
✅ Conda environment activated
```

### Ray Initialization (Next 1-2 minutes)

```
💾 Object Store: 4.0GB (25% of 16.0GB RAM) + SSD Spilling
📁 Spill Dir: /mnt/new_data/ray_spill
🛡️  Memory Threshold: 88%
⏱️  GCS Reconnect: 600s
📊 CPUs: 8, Samples: 2, Envs/worker: 2
```

### Training Progress (Continuous)

```
[After 2500 steps]   ✅ Checkpoint saved at 2500 steps
[After 5000 steps]   ✅ Checkpoint saved at 5000 steps
[After 7500 steps]   ✅ Checkpoint saved at 7500 steps
[etc...]
```

---

## 🚨 RED FLAGS - Things That Should NOT Happen

### ❌ Object Store Wrong Size
```
❌ Object Store: 2.0GB instead of 4.0GB
Fix: Object store fix not applied - check main() around line 1314
```

### ❌ GCS Timeout Wrong
```
❌ GCS Reconnect: 30s instead of 600s
Fix: Timeout fix not applied - check environment variables
```

### ❌ Crash Before 2500 Steps
```
❌ Training crashed at ~200 steps
Fix: Object store too small - check line 1314 object store fix
```

### ❌ No Checkpoints Saved
```
❌ No checkpoint directories created
Fix: Accumulator not initialized - check setup() method
```

### ❌ Auto-Detection Didn't Work
```
❌ Manual --resume flag needed
Fix: STEP 5 in bash script not working - check run_adan_pro.sh
```

### ❌ Memory Errors
```
❌ "Memory: 92% (exceeds threshold 88%)"
Fix: Workers being killed - this is intentional safety feature
```

---

## 📊 WHAT TO MONITOR

### Every 30 Seconds (Real-time Terminal)
- [ ] Is training still running? (New log lines appearing?)
- [ ] Any error messages? (Red text, exceptions?)
- [ ] Memory stable? (Not spiking unpredictably?)

### Every 5 Minutes
- [ ] Checkpoint saved? (Should see message every 2500 steps)
- [ ] Training progressing? (Step counter increasing?)
- [ ] GCS errors? (Should see none with 600s timeout)

### Every 30 Minutes
- [ ] Multiple checkpoints saved? (At least 6+ by step 15,000)
- [ ] Consistent checkpoint interval? (Every 2500 steps)
- [ ] No OOM errors? (Should handle up to 88% memory)

### Log File (Optional - For Detailed Analysis)
```bash
tail -f /mnt/new_data/adan_logs/training/production_run.log
```

---

## 📁 VERIFY CHECKPOINTS

### Check Checkpoint Directory
```bash
ls -la /mnt/new_data/adan_logs/checkpoints/adan_pbt_training/
```

**Expected output (after 15,000 steps)**:
```
total 400
drwxr-xr-x  6 user user   4096 Jun  8 12:45 .
drwxr-xr-x 10 user user   4096 Jun  8 12:00 ..
drwxr-xr-x  3 user user   4096 Jun  8 12:05 checkpoint_00002500
drwxr-xr-x  3 user user   4096 Jun  8 12:10 checkpoint_00005000
drwxr-xr-x  3 user user   4096 Jun  8 12:15 checkpoint_00007500
drwxr-xr-x  3 user user   4096 Jun  8 12:20 checkpoint_00010000
drwxr-xr-x  3 user user   4096 Jun  8 12:25 checkpoint_00012500
drwxr-xr-x  3 user user   4096 Jun  8 12:30 checkpoint_00015000
```

### Check Latest Checkpoint Size
```bash
du -sh /mnt/new_data/adan_logs/checkpoints/adan_pbt_training/checkpoint_*/
```

**Expected**: Consistent sizes (50-200MB per checkpoint)

---

## 📈 VERIFY MEMORY USAGE

### Check Memory During Training
```bash
watch -n 1 'free -h'
```

**Expected**:
- RAM used: ~12-14GB (out of 16GB)
- Swap used: ~2-4GB
- Available: ~1-2GB

### Check Ray Memory
```bash
ps aux | grep ray
```

**Expected**: Multiple Ray processes using reasonable amounts

---

## ⏱️ TIMELINE EXPECTATIONS

| Time | Event | What You'll See |
|------|-------|-----------------|
| 0:00 | Start | System cleanup messages |
| 0:30 | Ray init | "Object Store: 4.0GB..." message |
| 1:00 | Training starts | First training logs appear |
| 2:30 | Checkpoint 1 | "✅ Checkpoint saved at 2500 steps" |
| 5:00 | Checkpoint 2 | "✅ Checkpoint saved at 5000 steps" |
| 7:30 | Checkpoint 3 | "✅ Checkpoint saved at 7500 steps" |
| 10:00 | Checkpoint 4 | "✅ Checkpoint saved at 10000 steps" |
| 30:00 | Checkpoint 12 | ~12 checkpoints saved, training progressing |
| 60:00 | Checkpoint 24 | Many checkpoints, keeps most recent 10 |

---

## 🔄 RESUME MODE VERIFICATION

### When Restarting (After Checkpoint Exists)

**Expected output**:
```
📋 STEP 5: Checkpoint Detection...
   ✅ Found 6 checkpoint(s)
   📌 Latest: checkpoint_00015000
   🔄 RESUME MODE enabled

📌 Resume Mode: ✅ YES (Resuming from checkpoint)
```

**Then training continues from step 15,000+**

---

## 🛑 WHEN TO STOP TRAINING

### Let it run for:
- [ ] At least 2500 steps (verify 1st checkpoint)
- [ ] At least 5000 steps (verify 2nd checkpoint)
- [ ] At least 10,000 steps (verify stability)

### Can stop by:
```bash
# Graceful stop
Ctrl+C

# Force stop (if needed)
pkill -9 python
```

---

## 📝 THINGS TO DOCUMENT

After 1 hour, check:
- [ ] How many checkpoints created? (Should be ~24 at 2500-step intervals)
- [ ] Any errors in logs?
- [ ] Memory usage pattern (stable or drifting?)
- [ ] Training metrics improving?

---

## 🆘 EMERGENCY COMMANDS

### If training freezes (no output for >2 minutes)
```bash
# Check if Ray is responding
python -c "import ray; ray.init(); print('Ray OK'); ray.shutdown()"
```

### If memory explodes
```bash
# Clean cache and restart
echo 3 | sudo tee /proc/sys/vm/drop_caches
bash scripts/run_adan_pro.sh
```

### If checkpoints not saving
```bash
# Check checkpoint directory
ls -la /mnt/new_data/adan_logs/checkpoints/adan_pbt_training/

# Check Python logs for errors
tail -100 /mnt/new_data/adan_logs/training/production_run.log | grep -i error
```

### If auto-resume not working
```bash
# Check if checkpoints exist
find /mnt/new_data/adan_logs/checkpoints -name "checkpoint_*" -type d

# Check bash STEP 5 logic
bash -x scripts/run_adan_pro.sh 2>&1 | grep -A5 "STEP 5"
```

---

## ✅ SUCCESS INDICATORS

After 1 hour of training:
- ✅ Object store shows 4GB
- ✅ At least 24 checkpoint saves (every 2500 steps)
- ✅ No OOM errors
- ✅ No GCS timeout errors
- ✅ Training progressing smoothly
- ✅ Memory under 88% threshold

---

## 📞 DEBUG INFO TO COLLECT IF ISSUES

If something goes wrong, collect:

1. **Terminal output** (first 50 lines)
2. **Log file excerpt** (last 100 lines)
   ```bash
   tail -100 /mnt/new_data/adan_logs/training/production_run.log
   ```
3. **Checkpoint status**
   ```bash
   ls -la /mnt/new_data/adan_logs/checkpoints/adan_pbt_training/
   ```
4. **Memory usage**
   ```bash
   free -h && ps aux | grep -E "(python|ray)" | head -5
   ```
5. **Ray status**
   ```bash
   ray status 2>/dev/null || echo "Ray not running"
   ```

---

## Quick Copy-Paste Monitoring Commands

```bash
# Watch memory in real-time
watch -n 1 'free -h; echo "---"; ls /mnt/new_data/adan_logs/checkpoints/adan_pbt_training/ | wc -l'

# Check latest logs
tail -50 /mnt/new_data/adan_logs/training/production_run.log

# Count checkpoints
find /mnt/new_data/adan_logs/checkpoints/adan_pbt_training -name "checkpoint_*" | wc -l

# Check if process running
pgrep -f "train_parallel" && echo "Training running" || echo "Training stopped"

# Get training step count
grep "Checkpoint saved at" /mnt/new_data/adan_logs/training/production_run.log | tail -1
```

---

**Last Updated**: June 8, 2026  
**Status**: Ready for Production Monitoring
