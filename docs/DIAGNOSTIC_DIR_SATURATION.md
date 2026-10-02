# ADAN0 — Diagnostic Dir=+1.000 Saturation

**Date:** 2026-06-16
**Status:** CONFIRMED BROKEN — ALL checkpoints have saturated weights

## Executive Summary

ALL 11 checkpoints (50k to 500k steps) produce saturated tanh outputs 
(±1.000) regardless of input. The model is fundamentally broken and must 
be retrained.

## Evidence

### Test 1: Random observations N(0,1)

Every checkpoint was tested with 20 random observations sampled from N(0,1):

| Checkpoint | Saturated/20 | Status |
|-----------|-------------|--------|
| 50k steps | 9/20 | WEAK |
| 100k steps | 6/20 | WEAK (best) |
| 150k steps | 7/20 | WEAK |
| 200k steps | 10/20 | WEAK |
| 250k steps | 16/20 | DEAD |
| 300k steps | 14/20 | DEAD |
| 350k steps | 14/20 | DEAD |
| 400k steps | 17/20 | DEAD |
| 450k steps | 17/20 | DEAD |
| 500k steps (latest) | 16/20 | DEAD |

### Test 2: Live observations (properly normalized + clipped)

Checkpoint 100k tested with:
- Parquet-fitted scalers (locked to training distribution)
- Observations clipped to [-5, 5]
- Result: Dir=+1.000 for ALL 10 inferences

### Test 3: Scaled-down random observations (* 0.1)

Checkpoint 100k with very small random inputs:
- 0/20 saturated, std=0.32
- Shows the model HAS learned something, but the live input distribution
  doesn't match what it expects.

## Root Causes

### 1. Scaler Distribution Mismatch

Training data contains raw prices (60000-82000) that go through StateBuilder
scalers. But:
- 5m uses MinMax scaler → [0, 1]
- 1h uses Standard scaler → N(0, 1)  
- 4h uses Robust scaler → varies

Live data goes through DIFFERENT scalers fitted on different data.
The scalers are NEVER saved with the checkpoint.

### 2. Progressive Saturation During Training

The saturation worsens with more training steps (6/20 at 100k → 17/20 at 
450k). This indicates the policy gradient is pushing weights toward extremes,
likely because:
- Reward is sparse (realized PnL only, after S15 Hard Reset)
- Observations are poorly conditioned
- The network learns to "always be long" as a safe default

### 3. Missing VecNormalize

No `_vecnorm.pkl` file exists for any checkpoint. If training used 
VecNormalize (even partially), its absence at inference causes total 
observation distribution shift.

## Required Fixes for Retraining

1. **Save scalers WITH checkpoint**: After training, pickle the 
   StateBuilder scalers and save alongside the .zip
2. **Add saturation monitoring**: During training, log the std of 
   action outputs. If std < 0.1 for 1000+ steps, something is wrong.
3. **Use VecNormalize consistently**: Either always use it (train + 
   deploy) or never use it.
4. **Reduce observation dimensionality**: 21 features × 20 bars × 3 TFs 
   = 1260 dimensions is very large for a 128-64 network.

## Immediate Safety Fixes Applied

1. `run_bot.py`: Hard clip observations to [-5, 5]
2. `run_bot.py`: Changed default max_position_pct from 20% to 5%
3. `run_bot.py`: Added DEBUG_OBS and DEBUG_ACTION logging
4. `run_bot.py`: Changed inference to deterministic=False for gSDE exploration
5. `run_bot.py`: Added saturation alarm after 10 consecutive saturated ticks
6. `execution_engine.py`: Hard cap position size at 5% (was 20%)
7. `execution_engine.py`: Added mechanical cap at 10% regardless of config
8. `execution_engine.py`: Added SIZING log for every trade

## Conclusion

**The model cannot be used for trading.** Even in paper mode, it will 
open one position and hold it forever with maximum sizing. The 72h test 
is meaningless in this state.

Next step: Retrain with proper scaler persistence and saturation monitoring.
