# V31-500k Ray/PBT — Capture pré-arrêt (2026-08-18 10:27)

## Verdict corrections anti-collapse (ancre 0.05 + DiagGaussian): CONFIRMÉ STABLE
- a0_mean_raw borné ~[-0.036, +0.021] (vs explosion vers -8.6/-14.8 avant)
- a0_std_raw stable ~0.37 (vs explosion vers 108/85 avant)
- anchor_loss NON-NUL: 4.4e-05 -> 1.4e-04 (ancre ACTIVE dans la loss)
- approx_kl 0.0026-0.0075 (<<< target_kl 0.03, loin du seuil 1.5x=0.045)
- SatGuard: 0 événement (vs 1460 avant)
- entropy_loss ~-2.13 (stable, vs -20.3/-20.6 avant)

## Goulet d'étranglement: CONFIRMÉ
- 2 workers à 287-330% CPU chacun (~600% cumulé) sur ~6 coeurs -> sursouscription
- 53 threads/worker -> contention BLAS/OpenMP pendant backward
- Cycle PPO: paires de tables (w0+w1, gap 0-45s) séparées par ~460-497s (~8 min)
- Débit réel: 1.06 steps/sec/worker = 63.6 steps/min
- ETA 500k steps/worker: 131h = 5.5 JOURS (inacceptable)
- result.json VIDES (0 ligne) + 0 checkpoint disque: PBT n'a rapporté aucune itération complète en 1h50

## PBT: AUCUNE MUTATION OBSERVÉE (probable, à confirmer)
- perturbation_interval=2, time_attr=training_iteration
- result.json vides -> training_iteration jamais incrémenté -> PBT jamais déclenché
- sl/tp initiaux: w0 tp=0.1249 sl=0.0275 | w1 tp=0.0533 sl=0.0288

## Rentabilité (PRÉLIMINAIRE, non-conclusif à ~7000 steps)
- w0 scalper: WR~15.75%, Sharpe 0.377, 146 trades, PnL négatif
- w1 intraday: WR~19.42%, Sharpe 1.148, 139 trades, PnL négatif
- Clôtures: MaxDuration 219, SL 45, TP 17, AGENT_CLOSE 10
- NE PAS déclarer rentable: trop tôt, PnL<0, WR<<60%
