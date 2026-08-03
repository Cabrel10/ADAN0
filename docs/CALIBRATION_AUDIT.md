# CALIBRATION AUDIT — Reward Constants (ADAN0 v12/v13)

**Date:** 2026-07-04
**Branch:** feat/diagnostic-v4
**Purpose:** Systematic audit of every reward constant in the project, classifying each
as **DERIVED BY MEASURE** (its value comes from an explicit numeric comparison to the
magnitude of the phenomenon it must counter, measured on real project data) vs
**CHOSEN BY ANALOGY / INTUITION** (its value comes from a relative comparison to other
constants in the same file — "between X and Y" — or from a default never re-checked
empirically).

> **Why this doc exists.** The `holding_cost=0.001` cycle burned a full run testing a
> magnitude that was mathematically incapable of influencing the gradient, because it
> was sized against the *wrong comparator* (`closure_bonus=0.5`, a rare close-event)
> instead of against the components that compete for the same gradient *at the same
> step in position*. This is a **systemic pattern**, not an isolated bug: several
> constants were interpolated by analogy. This audit surfaces all of them so future
> ablations test something real instead of noise.

---

## The distinguishing criterion

| Category | Definition |
|---|---|
| ✅ **DERIVED BY MEASURE** | Value justified by a numeric comparison to the magnitude of the phenomenon it counters, computed on real data (e.g. the C1 warmup simulation of brake/fuel crossover). |
| ❌ **CHOSEN BY ANALOGY** | Value justified by position relative to other constants ("between hysteresis 0.4 and sell_no_position 1.2") without ever measuring the underlying signal. |
| ⚠️ **TO VERIFY** | Principle may be sound but magnitude was never checked against the signal it targets. |

---

## §2 GROUND-TRUTH MEASUREMENT (the missing reference)

Measured on `logs/training/reward_components_v12.csv` — **74 genuinely distinct
reward evaluations** (0 exact duplicates; the "74 rows / 7 steps" is a *labeling*
artifact, see §dup below, NOT data corruption), **n=52 rows in `position_state=long`**
(the context a holding cost competes in):

| Component (per-step, in position) | frac active | mean-abs when active | max magnitude |
|---|---|---|---|
| `symmetry_penalty`   | 31%  | **0.00404** | 0.01285 |
| `latent_pnl`         | 25%  | 0.00022     | 0.00069 |
| `pnl_base`           | 2% (close only) | 0.36585 | — (rare) |
| future_contrib / closure_bonus / drawdown / entropy / saturation | 0% in-position | — | never active mid-position |

- **Sum of mean-abs of per-step competing components (excl. rare pnl_base/closure): 0.00426**
- **std of their per-step sum: 0.00256**

**This 0.00426 / 0.00404 is the correct reference magnitude for any per-step
in-position reward term** — NOT `closure_bonus=0.5`.

---

## AUDIT TABLE

| Constant | Value | Origin of the choice | Verdict |
|---|---|---|---|
| `sterile_warmup_steps_self_caused` | 15000 | C1 calculation done seriously (numeric simulation of the brake/fuel crossover) | ✅ **DERIVED** — reference example |
| `min_notional` severity (Cas A) | 0.05 | "near-zero because non-controllable" — principle correct, magnitude never checked vs the signal it neutralizes | ⚠️ **TO VERIFY** (low priority: it's *meant* to be near-zero, and it neutralizes a non-fault) |
| `anti_spam_hold` severity | 0.8 | "between hysteresis (0.4) and sell_no_position (1.2)" — analogy of ranking position, not a measure | ❌ **ANALOGY — re-derive** |
| `min_notional_self_caused` severity | 0.55 | "moderate, between 0.4 and 0.8" — same interpolation-by-analogy defect | ❌ **ANALOGY** (branch neutralized in v12 → low priority) |
| `sell_no_position` severity | 1.2 | "stays the worst" — ordinal, not magnitude-measured | ⚠️ **TO VERIFY** (ordinal intent is defensible) |
| `hysteresis` severity | 0.4 | anchor of the analogy chain above | ⚠️ **TO VERIFY** |
| **`ADAN_HOLDING_COST`** | **0.001** | Compared only to `closure_bonus` (0.5, a rare close-event). **Measured §2: 4× smaller than `symmetry_penalty` (0.00404), same order as the NOISE (std 0.00256) of the per-step sum → drowned before it can compete.** | ❌ **MIS-CALIBRATED, CONFIRMED BY MEASURE** → re-derive to h ∈ [0.004, 0.012], test as bracket |
| `ADAN_ENT_COEF` | 0.03–0.04 | carried over from earliest runs; not re-evaluated after v12 action-routing architecture change | ⚠️ **TO VERIFY** (re-evaluate post-routing) |
| `latent_pnl_shaping.lambda_gain` | 0.10 | reasonable default, never empirically re-checked | ⚠️ **TO VERIFY** (measured mean-abs 0.00022 → currently a very small signal in practice) |
| `latent_pnl_shaping.lambda_loss` | 0.15 | asymmetry (loss>gain) is intentional & defensible; the *ratio* 1.5 is by principle, the *absolute* magnitude is not measured | ⚠️ **TO VERIFY** |
| `latent_pnl_shaping.every_n_steps` | 3 | cadence default, never re-checked | ⚠️ **TO VERIFY** |
| `latent_pnl_shaping.cap` | 0.30 | plafond by analogy to max_future_contrib (0.60) | ⚠️ **TO VERIFY** |
| `saturation_penalty.lambda` | 0.10 | default; never active in-position sample (0/52) → effectively untested | ⚠️ **TO VERIFY** |
| `saturation_penalty.cap` | 0.20 | default | ⚠️ **TO VERIFY** |
| `action_entropy.lambda_switch` | 0.03 | default | ⚠️ **TO VERIFY** |
| `sterile_action_penalty_cap` | 0.30 | DIAGNOSTIC-V5: raised from 0.10 with reasoning ("reaches ~15× base on persistent collapse") — reasoning present but not a measurement of the competing signal | ⚠️ **TO VERIFY** (has rationale, not a measure) |
| `sterile_action_geom_ratio` | 1.6 | "friction croissante par palier" — principle | ⚠️ **TO VERIFY** |
| `max_future_contrib` | 0.60 | "le PnL reste roi" — bounded by design principle (PnL must dominate) | ✅ **DERIVED-by-principle** (explicit dominance ordering) |
| **FEES** `commission`=0.0025 / `round_trip_fees`=0.005 | 0.5% | **REAL market fees — BINDING, DO NOT TOUCH** | ✅ **GROUND TRUTH (immutable)** |

---

## §dup — reward_components_v12.csv "duplicate bug" — RESOLVED (not a data bug)

- File: 74 rows, only 7 unique `step` values, **only worker 0 present**.
- **0 exact-duplicate rows** (subset = all components + a0 + state + action).
- → The 74 rows are **74 genuinely distinct per-step reward evaluations**. The
  repetition is a **step-labeling artifact**: the logger writes one row per env step
  inside the diagnostic window but stamps them all with the same coarse (diag-cadence)
  `step` value.
- **Consequence:** the CSV is **trustworthy for magnitude measurement** (§2 stands).
  Fix to schedule (low priority): stamp each row with the true global step, not the
  diag-window step, so the file is also usable for per-step *slope* analysis.

---

## §5 — 1h/4h DATA COVERAGE (reserve on ALL "signal" conclusions)

Measured on `data/processed/indicators/train/BTCUSDT/`:

| TF | rows | start | end | % of 5m training window covered |
|---|---|---|---|---|
| 5m | 18544 | 2025-06-29 | **2026-05-12** | 100% (this IS the training window) |
| 1h | 5483 | 2022-07-14 | **2025-08-15** | **14.6%** — missing last ~270 days (**~9 months**) |
| 4h | 1685 | 2022-10-14 | 2026-02-08 | 70.6% — missing last ~92 days (~3 months) |

**VERDICT:** The degenerate coverage the user recalled from V10 **persists in
V12/V13**. For ~85% of the run the **1h channel is frozen (ffill) or stale**; the
multi-TF fusion effectively operates on 5m + a mostly-frozen 1h + a partially-frozen
4h. This must be stated as a **reserve in any conclusion about "the signal the model
exploits"** and independently motivates the C3b temporal-shuffle test. It could
partly explain difficulty learning stable behaviour **independently of the reward**.

---

## ACTIONABLE OUTPUTS OF THIS AUDIT

1. `ADAN_HOLDING_COST` re-derived → **bracket 0.003 / 0.006 / 0.012 / 0.024** (centered
   on measured 0.004–0.0043, spanning "just matching" → "dominating").
2. Diagnostic cadence for short/validation runs → `ADAN_DIAG_EVERY=500` (≥20 pts on 10k).
3. Statistical discipline: <30 points → trajectory comparison + simple linear
   regression with honest 95% CI on the slope. **NO PCA/LDA/SVD** (artifacts at this scale).
4. reward_components logger → stamp true global step (deferred).
5. 1h/4h coverage → documented reserve; regeneration is a separate data task.
