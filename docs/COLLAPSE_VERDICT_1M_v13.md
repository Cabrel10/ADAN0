# VERDICT — 1M V13 run + git archaeology (6 June vs today)

**Date:** 2026-07-04 (T+3h reprise)
**Run:** train_v13_1M (PID 157053), --steps 1000000, profiles scalper+intraday,
holding_cost=0.006, smart_flat=0.05, breaker OFF.

## 1. `--steps` bug: FIXED & PROVEN
- Proof run `--steps 3000` → JSON `"steps": 3000` (was 10000).
- This 1M run reached **88064 real timesteps** (buggy run stopped at 10240).
- Checkpoints saved by step (25k/50k/75k). Bug is closed.

## 2. V13 COLLAPSED — holding_cost + smart_flat INSUFFICIENT

Trajectory (diag_v13_1M, 177 points, saved as diag_v13_1M_COLLAPSED_88k.csv):

| step | pct_buy | a0_mean |
|---|---|---|
| 500 | 0.544 | +0.04 |
| 5500 | 0.716 | +0.22 |
| 7000 | 0.822 | +0.32 |
| **11000** | **0.922** | +0.54 (crosses 0.90) |
| **14000** | **0.976** | +0.70 (full collapse) |
| 18500 | 0.998 | +1.02 |
| 88000 | 1.000 | **+4.34** |

**Two pathologies:**
1. BUY-runaway persists (pct_buy → 1.0 by @18k).
2. `a0_mean` **diverges unbounded to +4.34** — the continuous policy mean runs
   away toward +∞ (not just tanh-saturation). Entropy even *rises* (0.419→0.441)
   because the Gaussian mean drifts far outside [-1,1] while std stays ~0.38.

smart_flat + holding_cost=0.006 **delayed nothing meaningfully** and did not prevent
collapse. The variance-asymmetry fix, at this magnitude, is too weak.

## 3. GIT ARCHAEOLOGY — what changed 6 June → today (user's lead) — CORRECTED

Refined by reading the actual application site (not just config). The 6-June commit
7405039 applied `time_decay` in **`reward_calculator.py` L.316-321**, NOT in the env:

```python
# reward_calculator.py @ 7405039
_env_td = os.environ.get("ADAN_TIME_DECAY")
if _env_td is not None:
    time_decay = float(_env_td)
else:
    time_decay = float(self.config.get("time_decay", -1e-3))
r += time_decay          # SYMMETRIC, every step, unconditional
```

**Two corrections to the earlier note:**
1. The working baseline was **`time_decay = -1e-3` (-0.001)**, NOT -0.01. `config.yaml`
   declares -0.01 but the env never read it — so -0.01 was never validated. The
   validated magnitude is **-0.001**.
2. `capacity_weight` / `position_bonus` were **already REVERTED** on 6-June. The code
   comment states: *"REVERT des hacks reward S12 … le position_bonus et unrealized PnL
   delta ont EMPIRE les resultats CI: Run#8 ev=-5.11 → Run#9 ev=-9.29 (REGRESSION).
   Seule recompense valide = realized_pnl des trades fermes."*

**=> My earlier concern ("time_decay + capacity_weight rewards holding, wrong direction")
is LIFTED.** On 6-June there was NO capacity_weight; the reward was just
`realized_PnL + time_decay(-0.001)`. A symmetric per-step cost with no offsetting
position bonus IS the clean anti-drift lever that worked.

The current V4 reward stack (pnl_base, latent_pnl, future_contrib, closure_bonus,
symmetry_penalty, saturation_penalty) has **NO always-on per-step term**. `holding_cost`
(my v13 add) only fires in-position — it is not the same lever.

## 3bis. §2 latent_pnl measurement on CURRENT V13 config — VERDICT

Measured `|contribution|` share on the two most recent V13 reward-component CSVs
(`reward_components_v13_holdcost.csv` n=18, `reward_components_v13_nofuture.csv` n=20):

| term | holdcost share | nofuture share |
|---|---|---|
| pnl_base | 46.4% | 92.7% |
| future_contrib | 33.2% | 0.0% |
| closure_bonus | 19.5% | 0.0% |
| symmetry_penalty | 0.9% | 6.7% |
| **latent_pnl** | **0.0%** (3/18) | **0.6%** (4/20) |

**VERDICT:** `latent_pnl` remains **negligible** on the current V13 config
(holding_cost/smart_flat did NOT raise its relative weight). The "latent-PnL purge"
hypothesis is **NOT supported** — `latent_pnl_shaping` is left UNTOUCHED (per §2).

## 4. DECISION (one variable at a time)

Restore `time_decay` at its **validated magnitude -0.001** as the SINGLE isolated
variable. Hook is env-var driven (`ADAN_TIME_DECAY`), default OFF, symmetric per-step,
matching the 6-June `reward_calculator.py` semantics. Test protocol (§3):
- `ADAN_TIME_DECAY=-0.001` ONLY; `ADAN_SMART_FLAT=0`, `ADAN_HOLDING_COST=0`.
- Single profile **intraday** (avoid 2-worker instability per user directive).
- Short run first, `ADAN_DIAG_EVERY=250`, reward telemetry ON.
- Verify pct_buy does NOT cross 0.90 and a0_mean does NOT diverge, before any long run.

This is NOT a new hypothesis — it restores a term the working 6-June version had, at the
magnitude that version actually used.

## 5. §1 PROOF — time_decay hook is ACTIVE (residual test)

Run `launch_timedecay_isolated.sh` with `ADAN_TIME_DECAY=-0.001`, smart_flat=0,
holding_cost=0. On `reward_components_td_iso.csv` (n=51), residual =
`raw_reward - Σ(known components)`:

```
residual mean = -1.00002e-03,  std = 1.4e-07  (min -1.001e-03, max -1.000e-03)
```

Residual is a **rock-constant -0.001** in BOTH `flat` and `long` states → the symmetric
per-step time_decay is applied exactly as the 6-June `reward_calculator.py` did, and no
other shaping term leaks in. Hook activation CONFIRMED.

## 6. CONFOUNDER FOUND — `ADAN_LOG_STD_INIT` was never isolated

First isolated run (`td_iso` std=-1.0) showed pct_buy 0.46→0.74 over 5000 steps
(slope a0_mean +5.6e-05), **STEEPER** than the V13-holdcost reference (+2.8e-05).
Counter-intuitive — investigation revealed a **confounding variable**:

| run | ADAN_LOG_STD_INIT | a0_std | policy_entropy |
|---|---|---|---|
| v13_holdcost / nofuture (baselines) | default **-2.0** (std0≈0.135) | ~0.13 | ~-0.58 |
| td_iso (my launcher) | forced **-1.0** (std0≈0.37) | ~0.36 | ~+0.42 |

The `td_iso` launcher inherited `ADAN_LOG_STD_INIT=-1.0` from the v5 template — a
**2.7× wider initial action std**. A wider std explores action-space faster, so a0_mean
drifts faster — this is a std effect, NOT a time_decay effect. The comparison was
**two variables off** (time_decay AND log_std_init), violating the one-variable rule.

**Fix:** relaunched `td_iso` at the code default `ADAN_LOG_STD_INIT=-2.0` (std≈0.135),
10k steps, to compare against v13_holdcost/nofuture at ONE variable of difference
(time_decay only). The std=-1.0 run is preserved as `*_stdneg1_5k.csv`.

## 7. VERDICT (Cas C) — time_decay SYMMETRIC is the WRONG lever; holding_cost wins

Clean one-variable comparison (all: `ADAN_LOG_STD_INIT=-2.0` default, profile intraday,
window [2000,10000] steps, OLS slope on `a0_pct_buy`):

| lever (isolated) | pct_buy slope | pct_buy @10k | interpretation |
|---|---|---|---|
| **holding_cost=0.006** | **+1.8e-05** | **0.65** | asymmetric (position-only) |
| time_decay=-0.001 (6-June) | +6.0e-05 | 0.90 | **symmetric** → weak vs runaway |

**`holding_cost=0.006` is 3.3× more effective than `time_decay=-0.001` alone.**

**Mechanistic reason (not intuition):** `time_decay` is SYMMETRIC — it penalises `flat`
and `long` steps identically (residual test §5 proved -0.001 in BOTH states). It exerts
**zero differential pressure** against *staying in position*, which is the core of the
BUY-runaway. `holding_cost` is ASYMMETRIC (fires only when a position is open) → it
directly attacks the mechanism.

This REFINES the 6-June archaeology: the 6-June symmetric time_decay worked in a reward
world of *realized-PnL-only* (no future_contrib / closure_bonus). Today's variance
asymmetry needs an ASYMMETRIC lever. **Conclusion: pursue `holding_cost` (recalibrated),
NOT symmetric time_decay.** time_decay hook stays in the code (opt-in, default OFF) as a
proven-inert tool, but is NOT the fix.

**Next (measured, isolated):** bracket `holding_cost` ∈ {0.006, 0.012, 0.02} at 15k
steps, std=-2.0, intraday, everything else OFF, to find the magnitude that flattens the
pct_buy slope toward ~0 without over-correcting into pct_sell runaway.

## 8. RUN LONG hc=0.012 — trajectoire @10k (EVERY=2000, std=-2.0, intraday, breaker OFF)

| step | a0_mean | pct_buy | pct_sell | entropy |
|---|---|---|---|---|
| 2000 | +0.002 | 0.477 | 0.459 | -0.581 |
| 4000 | +0.012 | 0.496 | 0.434 | -0.581 |
| 6000 | +0.024 | 0.537 | 0.405 | -0.581 |
| 8000 | +0.035 | 0.576 | 0.368 | -0.580 |
| 10000 | +0.045 | 0.599 | 0.342 | -0.580 |

Pente OLS [2000,10000]: pct_buy **+1.62e-05/step** (CI95 [1.26,1.98]e-05),
a0_mean +5.5e-06. explained_variance mesurée ~0.44 (value function apprend).

**Verdict intermédiaire:** 0.012 penche encore côté BUY (pente ≈ celle de 0.006 =
+1.8e-05). Δ=pct_buy-pct_sell=0.257 @10k (>0.10). **0.012 sous-corrige légèrement** —
l'équilibre vrai est entre 0.012 (BUY) et 0.020 (SELL fort), estimé **~0.015-0.016**.
Run laissé actif pour capturer l'horizon complet (plateau vs collapse lent) comme
demandé. NB: pente 9× plus lente que le V13 original (std=-1.0, +6e-05 sur a0_mean).

## 9. RUN LONG hc=0.012 — VERDICT FINAL @110k : COLLAPSE (0.012 sous-corrige)

Full horizon reached (110k, run was alive). Answers to the 5 questions:
- Q1 passed 70k? **YES** (110k).
- Q2 pct_buy slope? **RISING then saturated**: 0.477(2k)→0.599(10k)→0.922(20k)→1.0(34k+).
- Q3 delta pct_buy-pct_sell? **RISING to 1.0** (saturated from 34k).
- Q4 explained_variance? **POSITIVE ~0.65-0.75** (critic stays useful even in collapse).
- Q5 collapse? **YES.** pct_buy=1.0/pct_sell=0.0 from ~34k; a0_mean diverges unbounded
  to +1.33 @110k; portfolio 20.5→14.04 (**-31%**).

Onset: pct_buy≥0.90 @20k, =1.0 @34k. So hc=0.012 only **DELAYED** collapse
(vs ~11k for V13-std-1.0) but did not prevent it. **A positive pct_buy slope, however
small (+1.6e-05), leads to full collapse at long horizon.** 0.012 under-corrects.

**Decision (user: relaunch >=500k after correction):** the equilibrium must produce a
pct_buy slope ~0 or slightly negative, durably. Bracket bounds: 0.012=BUY collapse,
0.020=SELL over-correction (at 15k). Relaunch at **hc=0.016** (nearer the correcting
side), 500k, same isolation (std=-2.0, intraday, time_decay/smart_flat OFF), breaker
OFF, diag EVERY=2000.
