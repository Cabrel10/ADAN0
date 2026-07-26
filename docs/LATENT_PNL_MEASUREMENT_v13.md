# latent_pnl Contribution — Measurement (v13)

**Date:** 2026-07-04
**Question (analyst hypothesis):** does `latent_pnl_contrib` reward *holding* with a
"constant flux of small virtual gains", making LONG > FLAT in expectation and driving
the BUY-runaway?

**Method:** measure `latent_pnl` on the real V12 reward components
(`logs/training/reward_components_v12.csv`), restricted to `position_state=long`
(n=52 genuinely-distinct per-step evaluations — see CALIBRATION_AUDIT.md §dup).

## Result

| metric | value |
|---|---|
| frac steps latent_pnl == 0 (in position) | **75.0%** (fires every 3 steps by design) |
| frac positive | 3.8% (n=2, mean +0.000158) |
| frac negative | 21.2% (n=11, mean −0.000231) |
| **NET sum over all in-position steps** | **−0.00222** (NEGATIVE) |
| per-step mean | −0.000043 |

## Verdict

**The analyst hypothesis is NOT supported by the measured data.** On this config,
`latent_pnl` nets **negative** while holding — it does *not* create a positive flux
that rewards staying LONG. Two reasons:

1. The asymmetry `lambda_loss=0.15 > lambda_gain=0.10` deliberately penalizes latent
   drawdown more than it rewards latent profit.
2. In the sampled window, in-position steps spent more time in latent loss than gain
   (11 negative vs 2 positive nonzero applications).

This is consistent with the user's note that "cette option a déjà été testée." The
mechanism is theoretically plausible but the numbers rule it out **for this config**.

## Implication

The remaining reward asymmetry driving the slow BUY drift is **NOT** latent_pnl. It is
consistent with the **variance-asymmetry** driver (HOLD-flat reward=0 zero-variance vs
BUY-flat reward≠0 nonzero-variance), which the `ADAN_SMART_FLAT` term targets. Whether
smart_flat is *strong enough* is the open question the 1M run answers.

> Caveat: n=52 is a small sample from ONE early window. If a future run logs
> `reward_components` continuously (fix the coarse step-stamp), re-measure on a larger,
> later window before drawing a final conclusion.
