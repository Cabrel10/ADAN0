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
