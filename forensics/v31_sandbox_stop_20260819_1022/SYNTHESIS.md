# V31 Sandbox — Synthèse d'arrêt (2026-08-19 10:22 CEST)

## Directive
Arrêt absolu ordonné par l'utilisateur : *« arrete v31 maintenant meme ci les instruction demande de lui laisser en vie »*.
Assez de diagnostics collectés. **AUCUNE correction hyperparamètre appliquée** (interdiction explicite : pas de λ↑, pas de hard clamp, pas de changement reward/EV gate, pas de V32).

## État final au moment de l'arrêt
- **total_timesteps PPO : 442 880** (dernière ligne log)
- **Dernier ANCHOR_DEBUG** : upd=1652, a0_mean=**-8.4426**, a0_std=0.3995, anchor=2.6649
  - `adv_BUY=nan adv_HOLD=nan adv_SELL=±0.0000`
  - `nB=0 nH=0 nS=64..2048`
- Processus arrêtés proprement par SIGTERM : sandbox PID 1131558, monitor 1138382, wrappers 1131547/1138379.
- Ressources post-arrêt : RAM 9.0 Gi dispo, aucun processus `train_parallel_agents` résiduel.

## Verdict classifié

| # | Finding | Statut |
|---|---------|--------|
| 1 | Collapse d'action SELL pur : disparition totale de BUY/HOLD du buffer PPO (nB=0, nH=0, adv=nan) sur les dernières centaines d'updates | **CONFIRMÉ** |
| 2 | Dérive monotone de μ (a0_mean_raw) 0 → -8.44 sur ~1652 updates ; σ stable 0.378-0.399 | **CONFIRMÉ** |
| 3 | Mécanisme : saturation tanh(-8.2)≈-1 → gradient policy mort → état absorbant ; seule l'ancre L2 (λ=0.05, λμ²≈2.66) agit encore, équilibre loin de 0 | **CONFIRMÉ** |
| 4 | reward_SELL ≈ -0.0352 (mauvais) mais persistance SELL → boucle d'auto-renforcement par disparition des alternatives, pas par récompense | **CONFIRMÉ** |
| 5 | routing_reject = 414 598 / 434 310 (95.5%) ; trade_executed = 0.32% — conséquence du collapse, pas sa cause | **CONFIRMÉ** |
| 6 | EV_GATE responsable du collapse | **RÉFUTÉ** (94 occurrences / 430k+ steps ; W=0.010 = clamp floor `max(0.01, min(0.99, bull_prob))` l.8740) |
| 7 | V31 (sell_while_flat 0.28→0.0) a retiré un garde-fou anti-biais SELL — contributeur probable, pas cause structurelle | **PROBABLE** |
| 8 | 693 trades réalisés (JSONL dédup.), WR=19.48%, PnL=-36.01$, PF=0.2208 ; closes 258 SL_HIT / 106 TP_HIT / 74 AGENT_CLOSE | **CONFIRMÉ** |
| 9 | Point exact de disparition de la diversité (policy vs sampling vs mapping vs routing) | **NON RÉSOLU** — greps pénalités en cours |

## Artefacts figés
- `a0_drift_series.txt` — 865 points (upd, a0_mean) série complète de la dérive
- `final_log_tail.txt` — 2000 dernières lignes du log sandbox
- `trade_audit_close_dedup.txt` — 438 closes dédupliqués (le log duplique chaque ligne)
- `pipeline_counters_final.txt` — compteurs pipeline ACTION_DIFF finaux
- `checkpoint_inventory.txt` — inventaire checkpoints/
- `ev_gate_w_distribution.txt` — distribution W (78/94 au plancher 0.010)
- `../forensic_7h_20260818_1814.txt` — capture forensique 7 sections (629 lignes)

## Références clés (lecture seule)
- Log source : `logs/training/v31_sandbox_500k_20260818_1035.log` (~132 MB)
- Rewards : `logs/rewards/worker_0_rewards_20260818_103519.jsonl` (615 MB, 179 571 records)
- Ancre L2 : `src/adan_trading_bot/agent/feature_extractors.py` (`anchor_loss = λ·μ²`, λ=0.05)
- Pénalités comportementales : `multi_asset_chunked_env.py` ~l.8366-8440 (EMA 0.97 + escalade géométrique 1.6^streak)
- Config : `config/config.yaml` ~l.1400-1460 (sell_while_flat: 0.0, buy_while_open: 0.0 — neutralisés en V31)

## Interdictions actives
- ❌ Pas de relance d'entraînement
- ❌ Pas de anchor_lambda 0.1/0.2, hard clamp μ, changement reward, changement EV gate, V32
- ✅ Analyse read-only autorisée (greps architecture pénalités)
