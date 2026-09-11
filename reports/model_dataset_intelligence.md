# ADAN — Bulletin d’intelligence dataset/modèle

Généré: `2026-08-05T20:41:48.562526+00:00`

Horizon principal: **36 barres 5m**.

## Verdict Arena

**GREEN — ARENA_GATE_PASSED**

- PASS `exactly_16_canonical_features`
- PASS `all_120_nonlinear_pairs`
- PASS `trade_join_complete`
- PASS `trade_sample_sufficient`
- PASS `strict_json_telemetry`
- PASS `policy_attribution_available`
- PASS `collector_teacher_sample_sufficient`
- PASS `collector_teacher_strict_json`
- PASS `market_signal_above_random`
- PASS `trade_target_evaluable`

## Portée et intégrité

- Les labels futurs sont utilisés uniquement pour l’analyse ex-post, jamais comme observation acteur.
- Aucun rapprochement ambigu des trades par prix n’est effectué.
- Les raisons de fermeture ci-dessous sont des simulations contrefactuelles OHLC; un double hit dans une barre reste `AMBIGUOUS_SAME_BAR`.
- SHAP n’est pas requis: information mutuelle, Extra Trees, gradient boosting et permutation held-out sont fournis.

## Qualité prédictive hors échantillon

| Cible | Extra Trees | Gradient Boosting | Baseline |
|---|---:|---:|---:|
| future_return (R²) | -0.458 | -0.855 | MAE constante 0.00405 |
| mfe_atr (R²) | 0.004 | -0.117 | MAE constante 3.011 |
| mae_atr (R²) | -0.988 | -2.673 | MAE constante 2.547 |
| time_to_mfe (R²) | -0.039 | -0.379 | MAE constante 10.54 |

| Cible classification | Extra Trees AUC | Boosting AUC |
|---|---:|---:|
| direction_up | 0.534 | 0.549 |
| good_long | 0.525 | 0.553 |

## Features dominantes par permutation held-out

- **future_return**: adx_14, vwap_ratio, di_delta, price_action, log_return
- **mfe_atr**: atr_pct, vwap_ratio, volatility_ratio_14_50, log_return, obv_slope
- **mae_atr**: adx_14, volatility_ratio_14_50, rsi, di_delta, fib_ratio
- **time_to_mfe**: adx_14, vwap_ratio, macdh, fib_ratio, bb_width_20_2

## Professeur Arena — 8 445 échantillons historiques

Échantillons valides: **8445**; états présents uniques: **4844**; split groupé sans fuite d'états répétés.
Le collector historique contient **9 indicateurs variables**, pas les 16 features canoniques; `regime` et les trois bits timeframe sont constants. Le MFE brut n'a pas été persisté: `tp_atr` est un proxy censuré par le plancher de frais.
- **tp_above_collector_floor**: AUC Extra Trees 0.797, AUC boosting 0.780; top permutation: atr_pct, volatility_ratio, adx, di_delta, rsi.
- **profitable_trade**: AUC Extra Trees 0.817, AUC boosting 0.781; top permutation: atr_pct, volatility_ratio, adx, rsi, ema_ratio.
- **tp_atr**: R² Extra Trees 0.427, R² boosting 0.202; top permutation: atr_pct, adx, volatility_ratio, rsi, macdh.
- **sl_atr**: R² Extra Trees 0.312, R² boosting 0.054; top permutation: atr_pct, adx, volatility_ratio, di_delta, rsi.
- **duration**: R² Extra Trees 0.194, R² boosting 0.110; top permutation: atr_pct, volatility_ratio, adx, di_delta, macdh.
- **Arena vs réseau (tête TP)**: Arena=atr_pct, adx_14, volatility_ratio_14_50, rsi, macdh; PPO=rsi, fib_ratio, di_delta, market_structure, adx_14; overlap=adx_14, rsi.

## Arena conditionnée par les trades v24

Jointure exacte: 118/118 trades.
PnL net des trades joints: -7.378631.
- **mfe_gt_tp_min**: AUC Extra Trees 0.515, AUC Gradient Boosting 0.431.
- **chosen_tp_attainable**: non évaluable (both temporal partitions must contain two classes).
- **profitable_trade**: AUC Extra Trees 0.433, AUC Gradient Boosting 0.498.

## Bons trades vs mauvais trades (long contrefactuel)

Définition: long entry: good if horizon return > +20 bps; bad if < -20 bps. Bons=668, mauvais=687, neutres=892.

Plus grands écarts standardisés: vwap_ratio (-0.57σ), atr_pct (-0.30σ), bb_width_20_2 (-0.22σ), adx_14 (+0.16σ), macdh (+0.09σ), di_delta (-0.04σ), volume_ratio_20 (+0.03σ), market_structure (+0.03σ)

## Régimes

K sélectionné par silhouette: **3**.
- Régime 0 (**forte_volatilite**): 0.8%, retour moyen -0.016%, MFE/ATR médian 1.13, MAE/ATR médian 1.30, TP conseillé 1.13 ATR, SL conseillé 2.63 ATR.
- Régime 1 (**tendance**): 49.6%, retour moyen -0.089%, MFE/ATR médian 2.62, MAE/ATR médian 2.86, TP conseillé 2.62 ATR, SL conseillé 5.40 ATR.
- Régime 2 (**range**): 49.5%, retour moyen -0.029%, MFE/ATR médian 2.83, MAE/ATR médian 2.89, TP conseillé 2.83 ATR, SL conseillé 5.77 ATR.

## Sorties réseau

Checkpoint: `checkpoints/v24_smoke_ray/adan_pbt_training/ADAN_PBT_Worker_6c9e7_00000_0_ent_coef=0.0163,gamma=0.9583,learning_rate=0.0001,sl_pct=0.0169,tp_pct=0.0678,worker_config=worker_i_2026-08-02_11-34-57/checkpoint_000000/model.zip` (ContextualTemporalFusionExtractor).
- **direction**: fib_ratio (0.002592), market_structure (0.00208), rsi (0.001218), bb_percent_b_20_2 (0.001181), volatility_ratio_14_50 (0.001132)
- **size**: fib_ratio (0.003551), market_structure (0.002524), bb_percent_b_20_2 (0.001063), di_delta (0.001058), rsi (0.0009783)
- **sl**: market_structure (0.001715), fib_ratio (0.001446), di_delta (0.001226), bb_percent_b_20_2 (0.0009672), adx_14 (0.0009001)
- **tp**: rsi (0.002575), fib_ratio (0.002445), di_delta (0.002381), market_structure (0.001656), adx_14 (0.001086)

## Décisions

- **SUPPORTED_FOR_ABLATION — Represent SL/TP in ATR multiples**: ATR units normalize volatility-scale drift and enable comparable barrier geometry, but the weak held-out risk predictability does not justify an untested production replacement.
- **SUPPORTED — Add current SL/TP distances and remaining time to the observation while a position is open**: These are Markov state variables required to evaluate risk geometry; they are not future leakage.
- **NOT_YET_SUPPORTED — Train a separate Risk Head on ex-post Future Arena labels**: Held-out MFE/MAE predictability is weak (best R²=0.004); improve labels/data before architectural separation.
- **REJECTED_PENDING_ABLATION — Remove SL/TP from PPO action immediately**: Attribution and market predictability alone do not prove that a supervised Risk Head outperforms joint PPO control; run matched-seed ablations first.
- **REJECTED — Change PPO/reward from this bulletin alone**: Direction predictability is reported objectively (best held-out AUC=0.549); no intuitive reward edit is justified without an ablation.

## Limites bloquantes de télémétrie trade

- Trade MFE/MAE uses exact entry timestamp and a fixed future horizon; it does not invent intrabar ordering.
- The immutable feature snapshot is the decision-close row t while execution is open[t+1], preventing actor leakage.
- Policy attribution uses real validation observations but does not claim SHAP-equivalent causal credit.
- A 2048-step smoke policy is diagnostic evidence, not a converged production policy.

Le JSON contient l’ensemble des distributions, importances, ablations, règles, dépendances partielles, interactions, clusters et grilles TP/SL/ATR.
