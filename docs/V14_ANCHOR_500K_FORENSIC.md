# V14 ACTION ANCHOR — 500k Run Forensic Verdict

**Date:** 2026-07-07
**Run:** `train_v14_500k_20260706_224303.log` / `diag_v14_500k.csv` (500 points, full 500k steps)
**Config:** `ADAN_ANCHOR_LAMBDA=0.05 ADAN_ANCHOR_DEADZONE=0.30 ADAN_ANCHOR_CAP=0.02`,
`HOLDING_COST=0.001 ENT_COEF=0.04 N_EPOCHS=10 USE_SDE=0`, fees unchanged (0.5%).

---

## VERDICT: COLLAPSE CONFIRMED — the anchor DELAYED but did NOT prevent it.

The symmetric no-op action anchor slowed the directional drift by roughly **1.5–2×**
in the early window, but the model still collapsed to **100% BUY** and the raw a0 head
still ran away to **a0_mean = +6.23** by 500k — essentially identical terminal state to
the historical SELFIX/MANIFESTO collapses.

### Collapse onset (data-driven, this run)
| threshold | crossed at |
|-----------|-----------|
| pct_buy ≥ 0.60 | @4,000 |
| pct_buy ≥ 0.70 | @6,000 |
| pct_buy ≥ 0.80 | @11,000 |
| pct_buy ≥ 0.85 (WARN) | @15,000 |
| pct_buy ≥ 0.90 | @17,000 |
| pct_buy ≥ 0.97 (CRIT) | @23,000 |
| pct_buy ≥ 0.99 | @27,000 |
| pct_buy = 1.00 | @35,000 |
| \|a0_mean\| ≥ 1.0 | @55,000 |
| \|a0_mean\| ≥ 1.8 | @77,000 |
| \|a0_mean\| ≥ 5.0 | @233,000 |

### V14 anchor vs MANIFESTO (historical collapse) — same milestones
| step | V14 pct_buy / a0_mean | MANIFESTO pct_buy / a0_mean |
|------|-----------------------|------------------------------|
| 5,000 | 0.650 / +0.064 | 0.601 / +0.042 |
| 10,000 | 0.776 / +0.120 | 0.833 / +0.143 |
| 15,000 | 0.887 / +0.172 | 0.971 / +0.256 |
| 20,000 | 0.951 / +0.222 | 1.000 / +0.443 |
| 30,000 | 0.997 / +0.388 | 1.000 / +0.746 |
| 50,000 | 1.000 / +0.893 | 1.000 / +1.134 |

**Interpretation:** the anchor bought ~1.5–2× more time (pct_buy hit 1.0 at ~35k
instead of ~20k; a0_mean grew ~30% slower) but the attractor is unchanged. The anchor
is a friction term, not a fix.

### Why the anchor failed (measured root, confirmed the user's prediction)
The a0 mass during the drift lives **below the dead-zone**. At step 4,000 the histogram
(`0|0|0|33|281|561|122|3|0|0`, bins of width 0.2 over [-1,1]) shows ~68% of mass in
`[0, 0.4)` — i.e. |a0| mostly under the 0.30 dead-zone. The anchor therefore **does not
fire on the very actions that drive the drift**. The user flagged exactly this: "le
deadzone à 0.30 pourrait être trop large (la dérive commence à |a0| ~ 0.15)".

A smaller dead-zone (0.10) would fire earlier, but this run also shows the anchor cap
(0.02) is dominated once the policy commits: by 50k the reward gradient from the
collapsed policy overwhelms a 0.02 restoring force. Lowering dead-zone alone would delay
collapse further but is very unlikely to change the terminal state — same class of
band-aid.

### Plateau check
Rolling 10-window pct_buy std < 0.02 **and** pct_buy < 0.85: **0 windows.**
There was never a real plateau. The apparent 6k→7k flat (0.704→0.703) was noise inside
a monotone climb.

### Entropy
-0.5810 (start) → -0.4013 (end): the policy became MORE deterministic as it collapsed,
consistent with a confident one-sided attractor, not exploration.

---

## DECISION (per user stop-conditions)

The user's instruction: *"t'arreter avant d'avoir relancer un train si les resultats
sont catastrophiques"*. The result is catastrophic (identical terminal collapse).

Therefore:
- **DO NOT** launch another training run (a deadzone=0.10 retry is the same class of
  band-aid — the forensic shows the cap is overwhelmed once committed; it would only
  move the onset, not the destination).
- **DO NOT** paper-trade the 500k checkpoint — it is a 100%-BUY collapsed model
  (a0_mean=+6.23), guaranteed to be a losing/degenerate trader.
- **STOP** and hand back to the user with this forensic.

## What the evidence now says the real fix is
Three independent 500k runs (MANIFESTO, SELFIX, V14-ANCHOR) with three different reward
tweaks all converge to the SAME 100%-BUY / a0-runaway attractor. This is not a
coefficient problem — it is an **objective-function** problem, as established in prior
sessions:

- The optimisation target (ΔPnL-based reward, with a0-independent no-op gradient) has a
  degenerate optimum at "always buy". Friction terms (holding cost, anchor) delay but do
  not remove that optimum.
- The consensus redesign (reward = ΔUtility: `U_t = log(W_t) − λ_D·D_t − λ_V·σ_t −
  λ_C·C_t`, judged at the trade-CYCLE level via a Behavior Layer) changes the optimum
  itself and is the only remaining path with a mechanistic reason to work.

This redesign is a **design change to confirm with the user before implementing** — not
started in this session.
