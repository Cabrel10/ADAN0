# ADAN — Bulletin d’intelligence dataset/modèle

Généré: `2026-08-01T08:11:42.312539+00:00`  
Horizon principal: **36 barres 5m**.

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

## Bons trades vs mauvais trades (long contrefactuel)

Définition: long entry: good if horizon return > +20 bps; bad if < -20 bps. Bons=668, mauvais=687, neutres=892.

Plus grands écarts standardisés: vwap_ratio (-0.57σ), atr_pct (-0.30σ), bb_width_20_2 (-0.22σ), adx_14 (+0.16σ), macdh (+0.09σ), di_delta (-0.04σ), volume_ratio_20 (+0.03σ), market_structure (+0.03σ)

## Régimes

K sélectionné par silhouette: **3**.
- Régime 0: 0.8%, retour moyen -0.016%, MFE/ATR médian 1.13, MAE/ATR médian 1.30.
- Régime 1: 49.6%, retour moyen -0.089%, MFE/ATR médian 2.62, MAE/ATR médian 2.86.
- Régime 2: 49.5%, retour moyen -0.029%, MFE/ATR médian 2.83, MAE/ATR médian 2.89.

## Sorties réseau

Checkpoint: `checkpoints/v21_smoke_gsde/adan_pbt_training/ADAN_PBT_Worker_517b3_00003_3_ent_coef=0.0260,gamma=0.9922,learning_rate=0.0007,sl_pct=0.0582,tp_pct=0.0351,worker_idx=3_2026-07-31_07-00-53/checkpoint_000000/model.zip` (ContextualTemporalFusionExtractor).
Têtes saturées après clipping sur ≥95% des observations: **direction, size, sl, tp**. Les rangs ci-dessous utilisent donc les moyennes pré-clipping.
- **direction**: market_structure (0.01423), adx_14 (0.0122), volatility_ratio_14_50 (0.01191), fib_ratio (0.007946), bb_percent_b_20_2 (0.006933)
- **size**: market_structure (0.05613), adx_14 (0.05573), volatility_ratio_14_50 (0.05183), bb_percent_b_20_2 (0.03403), fib_ratio (0.03369)
- **sl**: market_structure (0.1241), adx_14 (0.1192), volatility_ratio_14_50 (0.1101), fib_ratio (0.07262), bb_percent_b_20_2 (0.06664)
- **tp**: adx_14 (0.04617), market_structure (0.04604), volatility_ratio_14_50 (0.04288), fib_ratio (0.02793), bb_percent_b_20_2 (0.02631)

## Décisions

- **SUPPORTED_FOR_ABLATION — Represent SL/TP in ATR multiples**: ATR units normalize volatility-scale drift and enable comparable barrier geometry, but the weak held-out risk predictability does not justify an untested production replacement.
- **SUPPORTED — Add current SL/TP distances and remaining time to the observation while a position is open**: These are Markov state variables required to evaluate risk geometry; they are not future leakage.
- **NOT_YET_SUPPORTED — Train a separate Risk Head on ex-post Future Arena labels**: Held-out MFE/MAE predictability is weak (best R²=0.004); improve labels/data before architectural separation.
- **REJECTED_PENDING_ABLATION — Remove SL/TP from PPO action immediately**: Attribution and market predictability alone do not prove that a supervised Risk Head outperforms joint PPO control; run matched-seed ablations first.
- **REJECTED — Change PPO/reward from this bulletin alone**: Direction predictability is reported objectively (best held-out AUC=0.549); no intuitive reward edit is justified without an ablation.

## Limites bloquantes de télémétrie trade

- Current OPEN/CLOSE JSONL lacks an exact market timestamp/row id and the 16-feature entry snapshot.
- Actual trade MFE/MAE, TP_raw, TP/ATR and close-reason attribution therefore cannot be reconstructed without ambiguity.
- The next smoke must emit entry_market_timestamp, entry_row_id, ATR_raw and the immutable entry feature vector.
- Policy attribution uses real validation observations but does not claim SHAP-equivalent causal credit.

Le JSON contient l’ensemble des distributions, importances, interactions, clusters et grilles TP/SL/ATR.
