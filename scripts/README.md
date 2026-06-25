# Scripts ADAN0 — organisation par catégorie

Les **points d'entrée** (launchers `.sh` + l'entraîneur principal) restent à la
racine de `scripts/`. Les outils sont rangés par catégorie.

## Racine (points d'entrée)
| Fichier | Rôle |
|---|---|
| `train_parallel_agents.py` | Entraîneur principal (modes `heavy` et `sandbox`) |
| `run_adan_v2.sh` | Launcher V2 (defaults sûrs : DiagGaussian, ENT_COEF=0) |
| `run_adan_pro.sh` / `run_adan_sandbox.sh` | Launchers GPU / VPS |
| `run_full_audit.sh` | Pipeline d'audit complet |
| `live.sh` | Supervision du bot live |

## `training/`
Scripts d'entraînement annexes : `train_oracle.py`, `run_bot.py`.

## `diagnostics/`
Diagnostic & audit modèle : `audit_execution.py`, `audit_pre_tanh.py`,
`audit_tp_head.py`, `diag_gsde_latent.py`, `diagnose_obs.py`,
`diagnose_saturation.py`, `analyze_actiondim.py`, `inspect_action_heads.py`,
`inspect_policy.py`.

## `backtest/`
`backtest_fixed_capital.py`, `deterministic_backtest.py`,
`offline_reward_replay.py`.

## `data/`
Données & features : `download_ccxt_data.py`, `compute_features_real.py`,
`create_train_test_val_splits.py`, `extract_training_scalers.py`,
`verify_parquet.py`.

## `monitoring/`
`live_monitor.py`, `paper_trading_monitor.py`.

## `verification/`
`verify_checkpoint_config.py`, `verify_checkpoint_resume.py`,
`checkpoint_manager.py`, `smoke_test.py`, `test_pnl_flow.py`.
