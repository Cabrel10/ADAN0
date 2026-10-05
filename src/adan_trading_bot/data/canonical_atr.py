"""Single canonical completed-hour ATR source shared by snapshot and labeler.

Retains GATE1 arithmetic SMA14/TR/lag1. Initial TR after a missing previous
close is H-L (GATE1 bootstrap convention); its provenance explicitly records
that bootstrap. Four-hour ATR is deliberately NOT implemented or certified.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

ATR_DEFINITION = "systemone-1h-sma14-tr-lag1-v1"


@dataclass(frozen=True)
class ATRHour:
    start: pd.Timestamp
    high: float
    low: float
    close: float
    previous_close: float
    true_range: float
    bootstrap: bool


@dataclass(frozen=True)
class ATR1hObservation:
    value: float
    fraction: float
    available: bool
    available_at: pd.Timestamp | None
    source_hours: tuple[ATRHour, ...]
    unit: str = "quote_price"
    fraction_unit: str = "fraction_ATR_div_close_t"
    lag_containers: int = 1
    definition: str = ATR_DEFINITION
    reason: str = ""


class CanonicalATR1h:
    def __init__(self, values, ids, candles):
        self.values, self.ids, self.candles = values, ids, candles

    def observation(self, i, close):
        hour_id = int(self.ids[i])
        value = float(self.values[i])
        valid = bool(np.isfinite(value) and np.isfinite(close) and close > 0)
        hours = ()
        if valid:
            records = self.candles.loc[hour_id - 14:hour_id - 1]
            hours = tuple(ATRHour(pd.Timestamp(int(k) * 3600 * 10**9),
                                 float(r.h), float(r.l), float(r.c), float(r.prev_c),
                                 float(r.tr), bool(r.bootstrap)) for k, r in records.iterrows())
            if len(hours) != 14:
                raise ValueError("Canonical ATR provenance must have 14 source hours")
        return ATR1hObservation(value, value / close if valid else float("nan"), valid,
                                pd.Timestamp(hour_id * 3600 * 10**9) if valid else None,
                                hours, reason="" if valid else "warmup_or_incomplete_invalid_hour_or_price")


def canonical_atr_1h(b):
    """Aggregate UTC/open-timestamp 5m bars; never use a running container.

    Count alone is insufficient: every hour must have 12 unique grid-aligned,
    finite, coherent OHLCV bars. Missing/invalid hours reset SMA14 warmup.
    """
    ts = b.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    if b.n == 0 or b.ts.hasnans or (np.diff(ts) <= 0).any():
        raise ValueError("ATR source requires nonempty unique ordered timestamps")
    ids = ts // (3600 * 10**9)
    finite = np.isfinite(np.column_stack([b.o, b.h, b.l, b.c, b.v])).all(axis=1)
    valid = (finite & (b.o > 0) & (b.l > 0) & (b.c > 0) & (b.v >= 0)
             & (b.h >= np.maximum(b.o, b.c)) & (b.l <= np.minimum(b.o, b.c))
             & (ts % (300 * 10**9) == 0))
    bars = pd.DataFrame({"id": ids, "h": b.h, "l": b.l, "c": b.c, "valid": valid})
    candles = bars.groupby("id").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"),
                                      count=("c", "count"), valid=("valid", "all"))
    candles = candles.reindex(np.arange(ids.min(), ids.max() + 1))
    complete = candles['count'].eq(12) & candles['valid'].eq(True)
    candles.loc[~complete, ['h', 'l', 'c']] = np.nan
    candles['prev_c'] = candles.c.shift(1)
    candles['bootstrap'] = complete & candles.prev_c.isna()
    tr = pd.concat([candles.h - candles.l, (candles.h - candles.prev_c).abs(),
                    (candles.l - candles.prev_c).abs()], axis=1).max(axis=1)
    tr[~complete] = np.nan
    candles['tr'] = tr
    atr = tr.rolling(14, min_periods=14).mean().shift(1)
    return CanonicalATR1h(atr.reindex(ids).to_numpy(), ids, candles)


def compute_true_atr_1h(b):
    """Public labeler-compatible API; single runtime implementation."""
    return canonical_atr_1h(b).values
