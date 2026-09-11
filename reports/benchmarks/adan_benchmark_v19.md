# ADAN Permanent Benchmark

Generated: `2026-07-26T16:49:23.909263+00:00`
Asset/timeframe: `BTCUSDT` / `5m`
Horizon: `36` bars

## Methodological contract

- **Causal**: market descriptors and PPO telemetry available at decision/training time.
- **Ex-post**: MFE/MAE labels used only to grade decisions after the fact.
- **Physical ceiling**: same-entry and full-opportunity results are non-causal bounds.
- Opportunity classes are empirical calibration quantiles, not asset-specific constants.
- Configured fees and slippage are read without reduction or negotiation.

## Global score

| Domain | Score /20 |
|---|---:|
| Entry | 10.266 |
| Exit | 7.349 |
| Risk | 8.298 |
| Ppo | 14.553 |
| **Global** | **10.116** |

## A — Dataset report

```json
{
  "methodology": "per-split market identity; model-independent",
  "splits": {
    "test": {
      "methodology": "causal market description plus explicitly labelled ex-post excursions",
      "rows": 2283,
      "start": "2026-07-12 11:10:00",
      "end": "2026-07-20 09:20:00",
      "missing_cells": 0,
      "duplicate_timestamps": 0,
      "bar_returns": {
        "n": 2282,
        "mean": 2.6074510372404474e-06,
        "std": 0.0009744576053699012,
        "min": -0.005137003401678997,
        "p01": -0.0026539325378509048,
        "p05": -0.001457901469836892,
        "p25": -0.0004618469813880144,
        "median": -1.3604301130343277e-05,
        "p75": 0.00043750225303479384,
        "p95": 0.001547409401097199,
        "p99": 0.0027198350884105028,
        "max": 0.00825012655541646
      },
      "atr_pct": {
        "n": 2283,
        "mean": 0.0011656413667661407,
        "std": 0.0005690935180025641,
        "min": 0.00011006416725432151,
        "p01": 0.00021583753980356418,
        "p05": 0.00038093336089027347,
        "p25": 0.0007646225234594786,
        "median": 0.0010635950605205361,
        "p75": 0.0014766799686881254,
        "p95": 0.0023056426641409364,
        "p99": 0.002593238405855125,
        "max": 0.0029196712302069585
      },
      "atr_to_round_trip_cost_median": 0.21271901210410724,
      "path_efficiency": 0.003239866986519172,
      "mfe": {
        "n": 2282,
        "mean": 0.004241603130539306,
        "std": 0.004326736393142422,
        "min": 0.0,
        "p01": 0.0,
        "p05": 0.00021983663189226632,
        "p25": 0.0012265235110838469,
        "median": 0.0029698934592541234,
        "p75": 0.005648576725255164,
        "p95": 0.013629509592491968,
        "p99": 0.021452020201658446,
        "max": 0.025908072445143038
      },
      "mae": {
        "n": 2282,
        "mean": 0.004156209562365898,
        "std": 0.0039020506108947764,
        "min": 0.0,
        "p01": 0.0,
        "p05": 0.00018541733087075633,
        "p25": 0.0011991398470690715,
        "median": 0.002994205138508086,
        "p75": 0.005814313637791067,
        "p95": 0.012519460653149492,
        "p99": 0.01621210577094876,
        "max": 0.022288533573391077
      },
      "time_to_mfe_steps": {
        "n": 2282,
        "mean": 16.690184049079754,
        "std": 11.956427615128993,
        "min": 1.0,
        "p01": 1.0,
        "p05": 1.0,
        "p25": 5.0,
        "median": 15.0,
        "p75": 28.0,
        "p95": 36.0,
        "p99": 36.0,
        "max": 36.0
      },
      "economically_viable_rate": 0.2957931638913234
    },
    "train": {
      "methodology": "causal market description plus explicitly labelled ex-post excursions",
      "rows": 7991,
      "start": "2026-05-01 07:55:00",
      "end": "2026-07-12 11:05:00",
      "missing_cells": 0,
      "duplicate_timestamps": 0,
      "bar_returns": {
        "n": 7990,
        "mean": -2.1129604242873202e-05,
        "std": 0.0021650436084042347,
        "min": -0.09258721260450753,
        "p01": -0.0036124188343908724,
        "p05": -0.0019199249192809686,
        "p25": -0.000580823907712108,
        "median": -7.4341065635086245e-06,
        "p75": 0.0005680221913122452,
        "p95": 0.0019284314429078963,
        "p99": 0.00402290622727734,
        "max": 0.03658005163702116
      },
      "atr_pct": {
        "n": 7991,
        "mean": 0.001666123999148236,
        "std": 0.000933326647827022,
        "min": 0.00033679779662107814,
        "p01": 0.00048641424387322636,
        "p05": 0.0007419841979589632,
        "p25": 0.0010855431699149161,
        "median": 0.0014285165118659623,
        "p75": 0.0019562389378763654,
        "p95": 0.0034597945287802995,
        "p99": 0.005625342106704746,
        "max": 0.009384453092674643
      },
      "atr_to_round_trip_cost_median": 0.28570330237319247,
      "path_efficiency": 0.02342125578585054,
      "mfe": {
        "n": 7990,
        "mean": 0.005958813906468973,
        "std": 0.0069297154343175285,
        "min": 0.0,
        "p01": 0.0,
        "p05": 0.0002562672558976649,
        "p25": 0.0017072364426022064,
        "median": 0.003820188543065473,
        "p75": 0.007376603450475937,
        "p95": 0.01920549419864794,
        "p99": 0.03586346778636122,
        "max": 0.06484923516516825
      },
      "mae": {
        "n": 7990,
        "mean": 0.007041742949344387,
        "std": 0.010970714080868874,
        "min": 0.0,
        "p01": 0.0,
        "p05": 0.0002762560543835255,
        "p25": 0.001856299745629725,
        "median": 0.004362393338912191,
        "p75": 0.008118733738388312,
        "p95": 0.019945769576166214,
        "p99": 0.05412642495923389,
        "max": 0.11789417559064673
      },
      "time_to_mfe_steps": {
        "n": 7990,
        "mean": 17.15244055068836,
        "std": 12.187902696581325,
        "min": 1.0,
        "p01": 1.0,
        "p05": 1.0,
        "p25": 5.0,
        "median": 16.0,
        "p75": 29.0,
        "p95": 36.0,
        "p99": 36.0,
        "max": 36.0
      },
      "economically_viable_rate": 0.39224030037546936
    },
    "val": {
      "methodology": "causal market description plus explicitly labelled ex-post excursions",
      "rows": 1143,
      "start": "2026-07-20 09:25:00",
      "end": "2026-07-24 08:35:00",
      "missing_cells": 0,
      "duplicate_timestamps": 0,
      "bar_returns": {
        "n": 1142,
        "mean": 1.668622327694747e-05,
        "std": 0.0009265625932971194,
        "min": -0.0058461181923522565,
        "p01": -0.0021644357890207564,
        "p05": -0.0014037752536386892,
        "p25": -0.0004768915429655962,
        "median": -7.546029645588703e-08,
        "p75": 0.0005000651669066492,
        "p95": 0.0015302907032145553,
        "p99": 0.0024296784903754575,
        "max": 0.007074544862089915
      },
      "atr_pct": {
        "n": 1143,
        "mean": 0.0011819319442723118,
        "std": 0.0003801109194412871,
        "min": 0.0006138189829726183,
        "p01": 0.0006362609871976583,
        "p05": 0.0007409372148005632,
        "p25": 0.0009166513137680325,
        "median": 0.001096401358569508,
        "p75": 0.0013421020736226956,
        "p95": 0.0018982939336477643,
        "p99": 0.002454562939689355,
        "max": 0.002743920156571144
      },
      "atr_to_round_trip_cost_median": 0.21928027171390158,
      "path_efficiency": 0.025084715082495785,
      "mfe": {
        "n": 1142,
        "mean": 0.004314746729047798,
        "std": 0.0036922461966048363,
        "min": 0.0,
        "p01": 0.0,
        "p05": 0.00016434281974621507,
        "p25": 0.0015328016989125683,
        "median": 0.0035120903144218295,
        "p75": 0.006123646802117766,
        "p95": 0.011428404219151072,
        "p99": 0.018745428229695157,
        "max": 0.02191294084838105
      },
      "mae": {
        "n": 1142,
        "mean": 0.0038232267980479904,
        "std": 0.003005221624827121,
        "min": 0.0,
        "p01": 0.0,
        "p05": 0.00024032015836648174,
        "p25": 0.0013728476921817812,
        "median": 0.0031097115164382062,
        "p75": 0.005665582832121182,
        "p95": 0.009988318694737361,
        "p99": 0.012495098709570007,
        "max": 0.01424386767874754
      },
      "time_to_mfe_steps": {
        "n": 1142,
        "mean": 17.88091068301226,
        "std": 12.176669570644222,
        "min": 1.0,
        "p01": 1.0,
        "p05": 1.0,
        "p25": 6.0,
        "median": 18.0,
        "p75": 29.0,
        "p95": 36.0,
        "p99": 36.0,
        "max": 36.0
      },
      "economically_viable_rate": 0.3607705779334501
    }
  }
}
```

## B — Opportunity report

```json
{
  "methodology": "adaptive fee-aware ex-post opportunity frontier",
  "calibration_split": "train",
  "splits": {
    "test": {
      "methodology": "ex-post non-causal physical opportunity map",
      "class_basis": "calibration-split empirical quintiles of (MFE-cost)/(MAE+cost)",
      "adaptive_thresholds": [
        -0.3095400695394244,
        -0.17799438115025512,
        -0.009218277039958051,
        0.45829232710730217
      ],
      "class_counts": {
        "toxic": 780,
        "weak": 489,
        "neutral": 327,
        "good": 394,
        "excellent": 292
      },
      "class_rates": {
        "toxic": 0.3418054338299737,
        "weak": 0.21428571428571427,
        "neutral": 0.14329535495179668,
        "good": 0.1726555652936021,
        "excellent": 0.12795793163891322
      },
      "fee_aware_quality": {
        "n": 2282,
        "mean": -0.0337698319905655,
        "std": 0.6343631117155637,
        "min": -0.7886878756013431,
        "p01": -0.7600672383635761,
        "p05": -0.6505264216721831,
        "p25": -0.3681079059625051,
        "median": -0.21489268578940168,
        "p75": 0.0781825493124191,
        "p95": 1.26589450747157,
        "p99": 2.737768380115073,
        "max": 4.114776910167053
      },
      "viable_count": 675,
      "viable_rate": 0.2957931638913234
    },
    "train": {
      "methodology": "ex-post non-causal physical opportunity map",
      "class_basis": "calibration-split empirical quintiles of (MFE-cost)/(MAE+cost)",
      "adaptive_thresholds": [
        -0.3095400695394244,
        -0.17799438115025512,
        -0.009218277039958051,
        0.45829232710730217
      ],
      "class_counts": {
        "toxic": 1598,
        "weak": 1598,
        "neutral": 1598,
        "good": 1598,
        "excellent": 1598
      },
      "class_rates": {
        "toxic": 0.2,
        "weak": 0.2,
        "neutral": 0.2,
        "good": 0.2,
        "excellent": 0.2
      },
      "fee_aware_quality": {
        "n": 7990,
        "mean": 0.19901034413377824,
        "std": 0.891813045993302,
        "min": -0.9553683040881548,
        "p01": -0.5973276881928179,
        "p05": -0.4429764308611113,
        "p25": -0.2777835181824644,
        "median": -0.09991604043659565,
        "p75": 0.2773058685662565,
        "p95": 2.078331604477027,
        "p99": 3.7942670532343303,
        "max": 8.411205294949452
      },
      "viable_count": 3134,
      "viable_rate": 0.39224030037546936
    },
    "val": {
      "methodology": "ex-post non-causal physical opportunity map",
      "class_basis": "calibration-split empirical quintiles of (MFE-cost)/(MAE+cost)",
      "adaptive_thresholds": [
        -0.3095400695394244,
        -0.17799438115025512,
        -0.009218277039958051,
        0.45829232710730217
      ],
      "class_counts": {
        "toxic": 343,
        "weak": 228,
        "neutral": 149,
        "good": 270,
        "excellent": 152
      },
      "class_rates": {
        "toxic": 0.30035026269702275,
        "weak": 0.19964973730297722,
        "neutral": 0.13047285464098074,
        "good": 0.23642732049036777,
        "excellent": 0.1330998248686515
      },
      "fee_aware_quality": {
        "n": 1142,
        "mean": -0.014621153326880253,
        "std": 0.48860376177672626,
        "min": -0.9855342059522351,
        "p01": -0.5646889621496857,
        "p05": -0.4792376975142153,
        "p25": -0.33525997792159,
        "median": -0.17799853784384703,
        "p75": 0.16428160595523558,
        "p95": 1.0086352632328588,
        "p99": 1.7494447383972493,
        "max": 2.9524870081662957
      },
      "viable_count": 412,
      "viable_rate": 0.3607705779334501
    }
  }
}
```

## C — Entry report

```json
{
  "methodology": "entry decisions scored against ex-post labels; no PnL used",
  "event_precision": 0.40568383658969803,
  "unique_opportunity_recall": 0.6171027440970006,
  "f1": 0.4895422241993395,
  "false_positive_events": 5019,
  "false_negative_unique_rows": 1200,
  "good_opportunity_hit_rate": 0.6171027440970006,
  "bad_opportunity_avoidance": 0.40074135090609553,
  "market_viable_base_rate": 0.39224030037546936,
  "precision_uplift_points_vs_market": 0.013443536214228669,
  "entered_events": 8445,
  "entered_unique_rows": 4844
}
```

## D — Exit report

```json
{
  "methodology": "actual outcomes on identical entries versus ex-post exit ceiling",
  "trades": 8445,
  "win_rate": 0.16589698046181173,
  "profit_factor": 0.14997214216006938,
  "expectancy_pnl_units": -0.05451664387645642,
  "total_pnl_units": -460.39305753667446,
  "oracle_same_entry_win_rate": 0.40568383658969803,
  "profitable_opportunity_conversion": 0.4071803852889667,
  "exit_regret_rate": 0.5928196147110333,
  "hold_steps": {
    "n": 8445,
    "mean": 16.612670219064537,
    "std": 5.210774058252072,
    "min": 0.0,
    "p01": 2.0,
    "p05": 5.0,
    "p25": 16.0,
    "median": 16.0,
    "p75": 21.0,
    "p95": 21.0,
    "p99": 21.0,
    "max": 21.0
  },
  "time_to_mfe_steps_same_entries": {
    "n": 8445,
    "mean": 17.019064535227944,
    "std": 12.225365174677318,
    "min": 1.0,
    "p01": 1.0,
    "p05": 1.0,
    "p25": 5.0,
    "median": 16.0,
    "p75": 29.0,
    "p95": 36.0,
    "p99": 36.0,
    "max": 36.0
  },
  "exit_delay_vs_mfe_steps": {
    "n": 8445,
    "mean": -0.4063943161634103,
    "std": 12.798864175418396,
    "min": -35.0,
    "p01": -30.0,
    "p05": -20.0,
    "p25": -12.0,
    "median": 1.0,
    "p75": 11.0,
    "p95": 19.0,
    "p99": 20.0,
    "max": 20.0
  },
  "close_reasons": {
    "AGENT_CLOSE": 450,
    "CHUNK_END_FORCE_CLOSE": 1,
    "MAX_DURATION": 2418,
    "MaxDuration": 3610,
    "stop_loss": 1795,
    "take_profit": 171
  },
  "max_duration_rate": 0.7137951450562463,
  "mfe_capture_ratio": null,
  "mfe_capture_ratio_unavailable_reason": "Arena V19 stores PnL in currency units but omits entry notional/exit price; a dimensionally valid PnL-return/MFE ratio cannot be reconstructed."
}
```

## E — Risk report

```json
{
  "methodology": "actual sequential Arena PnL, reset at persisted episode boundaries; no annualization without a stable clock",
  "initial_capital": 20.5,
  "episodes": 65,
  "total_pnl_all_episodes": -460.39305753667446,
  "mean_episode_ending_equity": 13.41702988405116,
  "median_episode_ending_equity": 13.282448898951102,
  "minimum_episode_equity": 12.285138561547452,
  "ending_equity": 13.41702988405116,
  "max_drawdown": 0.4098563968011663,
  "var_95_pnl_units": -0.14792182485384572,
  "expected_shortfall_95_pnl_units": -0.1628232772052181,
  "sharpe_per_trade": -0.792476802002526,
  "sortino_per_trade": -1.2456482034867402,
  "empirical_kelly_fraction": -0.9402883288388002,
  "position_sizing_audit": null,
  "position_sizing_unavailable_reason": "Arena V19 does not persist entry notional or risk-at-entry."
}
```

## F — PPO report

```json
{
  "approx_kl": {
    "n": 976,
    "mean": 0.08137228747172132,
    "std": 0.06318730428781022,
    "min": 0.0073266346,
    "p01": 0.014888624500000001,
    "p05": 0.022234745,
    "p25": 0.04303960475,
    "median": 0.062038979999999994,
    "p75": 0.0969309755,
    "p95": 0.2229676625,
    "p99": 0.3282902975,
    "max": 0.41470918,
    "last": 0.16886857
  },
  "clip_fraction": {
    "n": 976,
    "mean": 0.5249436475409837,
    "std": 0.12145303644662525,
    "min": 0.238,
    "p01": 0.30325,
    "p05": 0.34824999999999995,
    "p25": 0.433,
    "median": 0.5125,
    "p75": 0.609,
    "p95": 0.75,
    "p99": 0.812,
    "max": 0.859,
    "last": 0.625
  },
  "entropy_loss": {
    "n": 976,
    "mean": -7.811946721311475,
    "std": 1.267058642572931,
    "min": -10.4,
    "p01": -9.93,
    "p05": -9.5025,
    "p25": -8.79,
    "median": -8.17,
    "p75": -6.7325,
    "p95": -5.6775,
    "p99": -5.565,
    "max": -5.46,
    "last": -7.73
  },
  "ep_rew_mean": {
    "n": 962,
    "mean": -485.92827442827445,
    "std": 144.6583545944964,
    "min": -609.0,
    "p01": -609.0,
    "p05": -606.0,
    "p25": -584.0,
    "median": -561.0,
    "p75": -411.0,
    "p95": -220.0,
    "p99": -220.0,
    "max": -220.0,
    "last": -580.0
  },
  "explained_variance": {
    "n": 976,
    "mean": 0.02572616905737705,
    "std": 0.35753805541112565,
    "min": -2.36,
    "p01": -1.66,
    "p05": -0.5925,
    "p25": -0.054325,
    "median": 0.07930000000000001,
    "p75": 0.218,
    "p95": 0.42725,
    "p99": 0.5605,
    "max": 0.668,
    "last": 0.327
  },
  "policy_gradient_loss": {
    "n": 975,
    "mean": 0.04865297846153846,
    "std": 0.04433894586582849,
    "min": -0.0288,
    "p01": -0.0052824,
    "p05": 0.003416000000000001,
    "p25": 0.0189,
    "median": 0.0369,
    "p75": 0.06659999999999999,
    "p95": 0.126,
    "p99": 0.21277999999999997,
    "max": 0.388,
    "last": 0.075
  },
  "std": {
    "n": 976,
    "mean": 0.1363002049180328,
    "std": 0.001157229573540814,
    "min": 0.135,
    "p01": 0.135,
    "p05": 0.135,
    "p25": 0.135,
    "median": 0.136,
    "p75": 0.137,
    "p95": 0.138,
    "p99": 0.138,
    "max": 0.138,
    "last": 0.138
  },
  "value_loss": {
    "n": 976,
    "mean": 0.27795338114754103,
    "std": 0.24421614785638918,
    "min": 0.0154,
    "p01": 0.0242,
    "p05": 0.049100000000000005,
    "p25": 0.124,
    "median": 0.213,
    "p75": 0.353,
    "p95": 0.714,
    "p99": 1.2425,
    "max": 2.67,
    "last": 0.346
  },
  "a0_mean": {
    "n": 977,
    "mean": -0.3753859774820881,
    "std": 0.30492297350751185,
    "min": -1.3938,
    "p01": -1.18582,
    "p05": -0.9622200000000001,
    "p25": -0.5884,
    "median": -0.3151,
    "p75": -0.1557,
    "p95": -0.0012,
    "p99": 0.0011440000000000055,
    "max": 0.0034,
    "last": -0.5431
  },
  "a0_std": {
    "n": 977,
    "mean": 1.25417164790174,
    "std": 0.3192677416437428,
    "min": 0.7279,
    "p01": 0.742216,
    "p05": 0.7581,
    "p25": 0.9575,
    "median": 1.3167,
    "p75": 1.466,
    "p95": 1.7542799999999998,
    "p99": 1.922092,
    "max": 2.1274,
    "last": 1.2469
  },
  "critic_negative_update_rate": 0.32991803278688525,
  "methodology": "causal training telemetry emitted by PPO",
  "action_diagnostics": {
    "available": true,
    "rows": 250,
    "timesteps": {
      "n": 250,
      "mean": 251000.0,
      "std": 144336.41259224922,
      "min": 2000.0,
      "p01": 6980.0,
      "p05": 26900.000000000004,
      "p25": 126500.0,
      "median": 251000.0,
      "p75": 375500.0,
      "p95": 475099.99999999994,
      "p99": 495020.0,
      "max": 500000.0
    },
    "a0_mean": {
      "n": 250,
      "mean": -0.3970612,
      "std": 0.3265858240257223,
      "min": -1.3069,
      "p01": -1.197949,
      "p05": -1.06534,
      "p25": -0.6173,
      "median": -0.33435,
      "p75": -0.13515,
      "p95": 0.009564999999999995,
      "p99": 0.05345199999999995,
      "max": 0.0826
    },
    "a0_std": {
      "n": 250,
      "mean": 1.2937296,
      "std": 0.3391162737525877,
      "min": 0.7044,
      "p01": 0.731791,
      "p05": 0.78517,
      "p25": 0.934125,
      "median": 1.3565,
      "p75": 1.5391750000000002,
      "p95": 1.8260299999999998,
      "p99": 1.981909,
      "max": 2.1399
    },
    "a0_pct_buy": {
      "n": 250,
      "mean": 0.391956,
      "std": 0.07068472298877601,
      "min": 0.235,
      "p01": 0.257215,
      "p05": 0.274075,
      "p25": 0.33575,
      "median": 0.39825,
      "p75": 0.444375,
      "p95": 0.49682499999999996,
      "p99": 0.5270299999999999,
      "max": 0.536
    },
    "a0_pct_sell": {
      "n": 250,
      "mean": 0.6013499999999999,
      "std": 0.07251601547244581,
      "min": 0.453,
      "p01": 0.46096000000000004,
      "p05": 0.491675,
      "p25": 0.54725,
      "median": 0.597,
      "p75": 0.659,
      "p95": 0.7196499999999999,
      "p99": 0.7390399999999999,
      "max": 0.7605
    },
    "a0_pct_hold_band": {
      "n": 250,
      "mean": 0.006694000000000001,
      "std": 0.0031986503403779536,
      "min": 0.0,
      "p01": 0.001,
      "p05": 0.002,
      "p25": 0.0045,
      "median": 0.0065,
      "p75": 0.0085,
      "p95": 0.012,
      "p99": 0.015,
      "max": 0.019
    },
    "req_HOLD_pct": {
      "n": 250,
      "mean": 0.630222,
      "std": 0.03944154809334948,
      "min": 0.552,
      "p01": 0.557745,
      "p05": 0.573725,
      "p25": 0.598875,
      "median": 0.6275,
      "p75": 0.6583749999999999,
      "p95": 0.699425,
      "p99": 0.722795,
      "max": 0.739
    },
    "req_BUY_pct": {
      "n": 250,
      "mean": 0.241968,
      "std": 0.04383908046480902,
      "min": 0.1545,
      "p01": 0.16599,
      "p05": 0.17345,
      "p25": 0.207,
      "median": 0.2455,
      "p75": 0.27737500000000004,
      "p95": 0.3096,
      "p99": 0.33003,
      "max": 0.343
    },
    "req_SELL_pct": {
      "n": 250,
      "mean": 0.12781,
      "std": 0.025213942571521812,
      "min": 0.0685,
      "p01": 0.07299,
      "p05": 0.086675,
      "p25": 0.110625,
      "median": 0.12725,
      "p75": 0.143,
      "p95": 0.17154999999999998,
      "p99": 0.19027499999999997,
      "max": 0.21
    },
    "steps_flat_pct": {
      "n": 250,
      "mean": 0.719012,
      "std": 0.05105489061784385,
      "min": 0.5955,
      "p01": 0.60896,
      "p05": 0.635225,
      "p25": 0.685125,
      "median": 0.71575,
      "p75": 0.754375,
      "p95": 0.8027,
      "p99": 0.8423149999999999,
      "max": 0.855
    },
    "steps_open_pct": {
      "n": 250,
      "mean": 0.28098800000000007,
      "std": 0.05105489061784385,
      "min": 0.145,
      "p01": 0.157685,
      "p05": 0.1973,
      "p25": 0.24562499999999998,
      "median": 0.28425,
      "p75": 0.314875,
      "p95": 0.36477499999999996,
      "p99": 0.39104,
      "max": 0.4045
    },
    "illegal_ratio": {
      "n": 250,
      "mean": 0.349136,
      "std": 0.0383513689977294,
      "min": 0.2475,
      "p01": 0.260235,
      "p05": 0.2837,
      "p25": 0.3205,
      "median": 0.351,
      "p75": 0.3785,
      "p95": 0.40677499999999994,
      "p99": 0.41901999999999995,
      "max": 0.424
    },
    "policy_entropy": {
      "n": 250,
      "mean": -0.574668,
      "std": 0.006046289440640429,
      "min": -0.5821,
      "p01": -0.5817019999999999,
      "p05": -0.5814,
      "p25": -0.5807,
      "median": -0.57675,
      "p75": -0.56765,
      "p95": -0.5644,
      "p99": -0.5635,
      "max": -0.5634
    }
  }
}
```

## G — Score details

```json
{
  "methodology": "scale-free transparent component mean; A/B describe difficulty and are not credited to the model",
  "components_0_to_1": {
    "entry": {
      "precision_skill_vs_market": 0.5221198217363441,
      "opportunity_recall": 0.6171027440970006,
      "bad_opportunity_avoidance": 0.40074135090609553
    },
    "exit": {
      "profitable_opportunity_conversion": 0.4071803852889667,
      "win_rate_relative_to_same_entry_oracle": 0.4089316987740806,
      "non_maxduration_rate": 0.2862048549437537
    },
    "risk": {
      "capital_survival": 0.654489262636642,
      "drawdown_control": 0.5901436031988336,
      "positive_expectancy": 0.0
    },
    "ppo": {
      "explained_variance": 0.5128630845286886,
      "nonnegative_critic_updates": 0.6700819672131147,
      "finite_policy_direction": 1.0
    }
  },
  "domain_scores_0_to_20": {
    "entry": 10.26642611159627,
    "exit": 7.348779593378674,
    "risk": 8.29755243890317,
    "ppo": 14.552967011612024
  },
  "global_score_0_to_20": 10.116431288872535
}
```
