# prod_scalers/ — FROZEN PKL ARE BANNED

Frozen `prod_scalers/*.pkl` scalers were the ROOT CAUSE of repeated paper-trading
saturation (constant `dir=+1.0` actions). A frozen scaler had been fitted on a
2.2% high-price tail (1h close mean=116k vs train mean=52k), so live BTC
normalized to ~-13σ and every raw-price feature clipped to -10. The PPO then
saw out-of-distribution observations and emitted constant saturated actions.

## The rule
`LiveStateBuilder.fit_on_parquet()` ALWAYS refits scalers INLINE on the
`data/processed/indicators/train/<SYMBOL>` Parquet data — the exact split and
inline-fit path used by the training/backtest environment
(`MultiAssetChunkedEnv`). This guarantees Training == Backtest == Live.

NEVER reintroduce frozen-pkl loading for the live bot. The old pkl are kept in
`_BANNED_DO_NOT_LOAD/` only for forensic reference.
