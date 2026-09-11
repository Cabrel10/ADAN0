# RADAR D'APPRENTISSAGE ADAN V31 — artefacts figes (read-only)

Run: 2026-08-18 10:35:22 -> 2026-08-18 12:40:36 (2.09h) | trades dedup=219 | updates PPO=865 | collapse durable a upd=368

## Scores (0-100)

- **L1_consequences** : 15.8
- **L2_erreurs** : 40.9
- **L3_environnement** : 83.1
- **L4_coherence** : 2.1
- **L5_performance** : 33.3

## Verdicts classes

- [CONFIRME] **L4.collapse_SELL_absorbing** — dernier bloc: share_SELL=1.0, a0_mean=-8.065, advBUY_nan=100.0% ; bascule durable upd=368
- [PROBABLE] **L4.spam_sterile_routing** — sell->hold sterile events=95, inv_penalty_total=-3.2
- [CONFIRME] **L2.repetition_apres_rejet** — P(SELL|SELL sterile t-1)=0.126 vs baseline P(SELL)=0.011 -> PAS d apprentissage d evitement
- [PROBABLE] **L2.apprend_a_eviter_SL** — taux SL H1=0.608 -> H2=0.415 (amelioration)
- [PROBABLE] **L3.adaptation_vol_frequence** — corr(vol, freq_trades)=0.831 ; prix H1=66750.4 -> H2=66440.9 (-0.46%)
- [CONFIRME] **L5.performance_trend** — WR H1=23.5% -> H2=21.8% ; Sharpe(global,indicatif)=-8.81

## Performance par fenetre (deciles temporels)

| win | trades | WR% | PnL | PF | avg_hold | TP | SL | AGENT |
|-----|--------|-----|-----|----|--------|----|----|-------|
| 0 | 36 | 22.2 | -2.723 | 0.29 | 7.4 | 8 | 21 | 7 |
| 1 | 17 | 29.4 | -1.061 | 0.34 | 11.9 | 5 | 12 | 0 |
| 2 | 35 | 17.1 | -3.211 | 0.19 | 9.5 | 6 | 22 | 7 |
| 3 | 18 | 22.2 | -1.0 | 0.31 | 10.3 | 4 | 7 | 7 |
| 4 | 34 | 26.5 | -2.612 | 0.29 | 9.3 | 9 | 25 | 0 |
| 5 | 14 | 28.6 | -0.57 | 0.45 | 9.1 | 4 | 3 | 7 |
| 6 | 31 | 35.5 | -1.714 | 0.45 | 7.8 | 11 | 20 | 0 |
| 7 | 9 | 22.2 | -0.391 | 0.26 | 5.4 | 1 | 1 | 7 |
| 8 | 22 | 22.7 | -1.839 | 0.31 | 11.4 | 5 | 17 | 0 |
| 9 | 3 | 0.0 | -0.244 | 0.0 | 3.0 | 0 | 1 | 2 |

## Diversite politique par bloc d'updates

| blk | upd | a0_mean | %BUY | %HOLD | %SELL | %advBUY_nan |
|-----|-----|---------|------|-------|-------|--------------|
| 0 | 4-344 | -0.251 | 0.249 | 0.166 | 0.585 | 7.0 |
| 1 | 348-586 | -3.263 | 0.0 | 0.0 | 1.0 | 100.0 |
| 2 | 587-756 | -6.046 | 0.0 | 0.0 | 1.0 | 100.0 |
| 3 | 757-902 | -7.406 | 0.0 | 0.0 | 1.0 | 100.0 |
| 4 | 906-1057 | -8.681 | 0.0 | 0.0 | 1.0 | 100.0 |
| 5 | 1058-1177 | -8.979 | 0.0 | 0.0 | 1.0 | 100.0 |
| 6 | 1178-1281 | -8.647 | 0.0 | 0.0 | 1.0 | 100.0 |
| 7 | 1282-1380 | -8.332 | 0.0 | 0.0 | 1.0 | 100.0 |
| 8 | 1384-1513 | -8.053 | 0.0 | 0.0 | 1.0 | 100.0 |
| 9 | 1517-1644 | -8.065 | 0.0 | 0.0 | 1.0 | 100.0 |

## Apprentissage d'erreurs

- P(SELL | SELL sterile a t-1) = 0.12631578947368421 vs baseline P(SELL) = 0.010769753996145562
- Evenements SELL->HOLD steriles: 95
- inv_penalty total = -3.2 (events non nuls: 320)
- Taux SL par fenetre: [0.583, 0.706, 0.629, 0.389, 0.735, 0.214, 0.645, 0.111, 0.773, 0.333]
- Ajustement hold: {'hold_after_SL_H1': 10.1, 'hold_after_SL_H2': 9.4, 'hold_after_TP_H1': 10.2, 'hold_after_TP_H2': 9.1, 'hold_after_AGENT_H1': 6.5, 'hold_after_AGENT_H2': 5.2}

## Adaptation environnement

- price_mean_H1: 66750.4
- price_mean_H2: 66440.9
- price_drift_pct: -0.46
- trade2trade_vol_H1: 4.031
- trade2trade_vol_H2: 5.234
- trades_per_win: [36, 17, 35, 18, 34, 14, 31, 9, 22, 3]
- corr_vol_tradefreq: 0.831