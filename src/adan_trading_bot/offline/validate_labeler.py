"""GATE 1: compare the actual compute_labels output to scalar raw-OHLC truth.

Contract: decision at close of bar i (timestamps are bar OPEN times), entry
at open[i+1], scan i+1..i+horizon inclusive. MFE/MAE are nonnegative price
fractions until exit, INCLUDING the entire exit bar; intrabar order is unknown.
Simultaneous TP/SL means SL. Times are 1-based bars from entry, 0 if censored.
Y_WIN currently means TP-first, not independently realized portfolio profit.
Expansion uses next 12 bars and 1.8 times ATR14 of completed, contiguous hours.
The oracle never uses production direction, containers, ATR, or label arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from adan_trading_bot.data.nested_state_builder import NestedStateBuilder
from adan_trading_bot.offline import labeler_mfe_mae as production

COLS = ["open", "high", "low", "close", "volume"]
HORIZON = 288
ATOL = 1e-12
FIELDS = ["mfe", "mae", "y_tp_first", "y_sl_first", "y_win",
          "y_sweep_confirme", "y_expansion_imminente", "y_direction",
          "time_to_tp", "time_to_sl", "net_return", "atr_1h_new", "plan_valid"]


def reference_atr(df, i):
    """Raw bars only; no rolling, builder containers or production ATR helper."""
    current_hour = df.index[i].floor("h")
    hourly = []
    # Need 14 TRs, plus previous close if that preceding hour exists.
    for offset in range(15, 0, -1):
        start = current_hour - pd.Timedelta(hours=offset)
        positions = [j for j in range(max(0, i - 192), i + 1)
                     if start <= df.index[j] < start + pd.Timedelta(hours=1)]
        expected = pd.date_range(start, periods=12, freq="5min")
        if len(positions) != 12 or not df.index[positions].equals(expected):
            hourly.append(None)
        else:
            bars = df.iloc[positions]
            hourly.append((float(bars.high.max()), float(bars.low.min()),
                           float(bars.close.iloc[-1])))
    if any(x is None for x in hourly[-14:]):
        return float("nan")
    ranges = []
    for j in range(1, 15):
        hi, lo, _ = hourly[j]
        prev = hourly[j - 1]
        ranges.append(hi - lo if prev is None else
                      max(hi - lo, abs(hi - prev[2]), abs(lo - prev[2])))
    return sum(ranges) / 14


def naive_reference(df, i, atr_cache=None):
    """Independent scalar oracle, including sweep-derived baseline direction."""
    o, h, l, c, _ = map(float, df.iloc[i][COLS])
    high_sweep = low_sweep = False
    if i >= 10:
        previous = df.iloc[i - 10:i]
        high_sweep = (h > float(previous.high.max()) and
                      c < float(previous.high.max()) and
                      h - max(o, c) >= 0.40 * max(h - l, 1e-12))
        low_sweep = (l < float(previous.low.min()) and
                     c > float(previous.low.min()) and
                     min(o, c) - l >= 0.40 * max(h - l, 1e-12))
    direction = 1 if high_sweep and not low_sweep else (0 if low_sweep and not high_sweep else 2)
    hour = df.index[i].floor("h")
    if atr_cache is None:
        atr = reference_atr(df, i)
    else:
        if hour not in atr_cache:
            atr_cache[hour] = reference_atr(df, i)
        atr = atr_cache[hour]
    future = df.iloc[i + 1:i + 13]
    expansion = int(len(future) == 12 and np.isfinite(atr) and
                    float(future.high.max() - future.low.min()) >= 1.8 * atr)
    valid = i + HORIZON < len(df)
    result = dict(mfe=0.0, mae=0.0, y_tp_first=0, y_sl_first=0, y_win=0,
                  y_sweep_confirme=int(df.index[i].minute in (50, 55) and
                                       (high_sweep or low_sweep)),
                  y_expansion_imminente=expansion, y_direction=direction,
                  time_to_tp=0, time_to_sl=0, net_return=0.0,
                  atr_1h_new=atr, plan_valid=int(valid))
    if not valid:
        return result
    entry = float(df.open.iloc[i + 1])
    risk = entry * 0.012
    short = direction == 1
    stop = entry + risk if short else entry - risk
    target = entry - 3.5 * risk if short else entry + 3.5 * risk
    for j in range(i + 1, i + HORIZON + 1):
        high, low = float(df.high.iloc[j]), float(df.low.iloc[j])
        favorable = (entry - low if short else high - entry) / entry
        adverse = (high - entry if short else entry - low) / entry
        result["mfe"] = max(result["mfe"], favorable)
        result["mae"] = max(result["mae"], adverse)
        stop_hit = high >= stop if short else low <= stop
        target_hit = low <= target if short else high >= target
        if stop_hit:
            result.update(y_sl_first=1, time_to_sl=j - i,
                          net_return=-1.0 - 0.0008 / 0.012)
            break
        if target_hit:
            result.update(y_tp_first=1, y_win=1, time_to_tp=j - i,
                          net_return=3.5 - 0.0008 / 0.012)
            break
    return result


def compare(df, indices, tag):
    builder = NestedStateBuilder(df)
    actual = production.compute_labels(builder, production.build_features(builder))
    errors = []
    atr_cache = {}
    maxima = {field: 0.0 for field in FIELDS}
    counts = {field: 0 for field in FIELDS}
    for i in indices:
        expected = naive_reference(df, int(i), atr_cache)
        for field in FIELDS:
            if field not in actual:
                errors.append(dict(sample=tag, index=int(i), field=field, reason="missing production output"))
                counts[field] += 1
                continue
            # Convert scalars BEFORE equality: NumPy float32 == Python float
            # can round the reference and falsely report exact equality.
            got, want = float(actual[field][i]), float(expected[field])
            equal = (np.isnan(got) and np.isnan(want)) or got == want
            if field in ("mfe", "mae", "net_return", "atr_1h_new"):
                equal = equal or np.isclose(got, want, atol=ATOL, rtol=0)
            if np.isfinite(got) and np.isfinite(want):
                maxima[field] = max(maxima[field], abs(float(got) - float(want)))
            if not equal:
                counts[field] += 1
                if len(errors) < 30:
                    errors.append(dict(sample=tag, index=int(i), decision_open=str(df.index[i]),
                                       entry_timestamp=str(df.index[i + 1]) if i + 1 < len(df) else None,
                                       field=field, production=float(got), reference=float(want)))
    return dict(sample=tag, count=len(indices), divergences=counts,
                max_absolute_error=maxima, examples=errors,
                start=str(df.index[0]), end=str(df.index[-1]))


def fixture(direction, event, event_bar=1):
    df = pd.DataFrame({"open": 100.0, "high": 100.2, "low": 99.8,
                       "close": 100.0, "volume": 1.0},
                      index=pd.date_range("2020-01-01", periods=600, freq="5min"))
    i = 202  # 16:50, valid sweep phase and 14 complete hours of history
    df.iloc[i, df.columns.get_loc("high" if direction == 1 else "low")] = 102.0 if direction == 1 else 98.0
    if event != "none":
        entry, risk = 100.0, 100.0 * 0.012
        stop = entry + risk if direction == 1 else entry - risk
        target = entry - 3.5 * risk if direction == 1 else entry + 3.5 * risk
        if event in ("sl", "both"):
            df.iloc[i + event_bar, df.columns.get_loc("high" if direction == 1 else "low")] = stop
        if event in ("tp", "both"):
            df.iloc[i + event_bar, df.columns.get_loc("low" if direction == 1 else "high")] = target
    return df, i


def synthetic_tests():
    reports = []
    for direction in (0, 1):
        for event in ("tp", "sl", "both", "none"):
            for event_bar in (1, HORIZON):
                df, i = fixture(direction, event, event_bar)
                truth = naive_reference(df, i)
                assert truth["y_direction"] == direction
                assert truth["y_tp_first"] == int(event == "tp")
                assert truth["y_sl_first"] == int(event in ("sl", "both"))
                assert truth["time_to_tp"] == (event_bar if event == "tp" else 0)
                assert truth["time_to_sl"] == (event_bar if event in ("sl", "both") else 0)
                reports.append(compare(df, [i, len(df) - HORIZON - 1,
                                             len(df) - HORIZON, len(df) - 1],
                                       f"direction={direction}/{event}/bar={event_bar}"))
    # Independent ATR causality check: future edits cannot change historical ATR.
    df, i = fixture(0, "none")
    original = production.compute_true_atr_1h(NestedStateBuilder(df))
    changed = df.copy()
    changed.iloc[i + 1:, changed.columns.get_loc("high")] *= 5
    revised = production.compute_true_atr_1h(NestedStateBuilder(changed))
    if not np.allclose(original[:i + 1], revised[:i + 1], atol=ATOL, rtol=0, equal_nan=True):
        raise AssertionError("ATR reads future bars")
    reports.append(compare(df, [0, 11, 12, 167, 168, i], "ATR-warmup"))
    return reports


def malformed_tests():
    """Production must reject malformed input instead of manufacturing labels."""
    df, _ = fixture(0, "none")
    variants = {"nan": df.copy(), "gap": df.drop(df.index[250]),
                "duplicate": pd.concat([df.iloc[:250], df.iloc[249:]]),
                "ohlc": df.copy(), "zero-price": df.copy(), "negative-volume": df.copy()}
    variants["nan"].iloc[250, 1] = np.nan
    variants["ohlc"].iloc[250, 1] = 90
    variants["zero-price"].iloc[250, :4] = 0
    variants["negative-volume"].iloc[250, 4] = -1
    for name, bad in variants.items():
        b = NestedStateBuilder(bad)
        try:
            production.compute_labels(b, production.build_features(b))
        except ValueError:
            continue
        raise AssertionError(f"Malformed {name} input silently accepted")
    return list(variants)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3000, help="exact number of TRAIN decision timestamps")
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--data", default=production.PARQUET)
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--check-malformed", action="store_true")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    if args.n <= 0:
        ap.error("--n must be positive")
    assert (production.MAX_HOLD, production.SL_PCT, production.TP_R,
            production.LOOKBACK_SWEEP, production.WICK_MIN) == (288, 0.012, 3.5, 10, 0.40)
    reports = synthetic_tests()
    metadata = dict(seed=args.seed, python=platform.python_version(), numpy=np.__version__,
                    pandas=pd.__version__, atol=ATOL, rtol=0,
                    production_sha256=hashlib.sha256(Path(production.__file__).read_bytes()).hexdigest())
    if args.check_malformed:
        metadata["malformed_rejections"] = malformed_tests()
    if not args.synthetic_only:
        source = pd.read_parquet(args.data, columns=COLS)
        train = source[(source.index >= "2017-01-01") & (source.index < "2022-01-01")]
        rng = np.random.default_rng(args.seed)
        # Twelve independent continuous windows spread over TRAIN; never concatenate
        # distant observations into a fake timeline. Oracle runs on sampled decisions.
        cuts = np.linspace(0, len(train) - 5000, 12, dtype=int)
        remaining = args.n
        all_timestamps = []
        for block, start in enumerate(cuts):
            df = train.iloc[start:start + 5000]
            ns = df.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
            if (np.diff(ns) != 300_000_000_000).any():
                # Deterministically advance within this stratum until continuous.
                for start in range(start, min(start + 20000, len(train) - 5000), 500):
                    df = train.iloc[start:start + 5000]
                    ns = df.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
                    if (np.diff(ns) == 300_000_000_000).all():
                        break
                else:
                    raise RuntimeError("No continuous TRAIN window in stratum")
            candidates = np.arange(192, len(df) - HORIZON)
            size = remaining // (12 - block)
            if size > len(candidates):
                raise ValueError("--n exceeds distinct timestamps available in windows")
            indices = np.sort(rng.choice(candidates, size=size, replace=False))
            remaining -= size
            all_timestamps.extend(str(df.index[j + 1]) for j in indices)
            reports.append(compare(df, indices, f"TRAIN-block-{block}"))
        assert len(all_timestamps) == args.n
        assert len(set(all_timestamps)) == args.n, "Duplicate sampled entries"
        metadata.update(dataset=args.data, source_train_rows=len(train), n_real=args.n,
                        train_start=str(train.index[0]), train_end=str(train.index[-1]),
                        entry_timestamp_sha256=hashlib.sha256("\n".join(all_timestamps).encode()).hexdigest())
    failures = sum(sum(r["divergences"].values()) for r in reports)
    report = dict(metadata=metadata, failures=failures, comparisons=reports)
    print(json.dumps(report, indent=2, allow_nan=True))
    if args.report:
        workspace = Path("/home/ubuntu/webapp").resolve()
        target = args.report.resolve()
        if not target.is_relative_to(workspace) or not target.parent.is_dir():
            raise ValueError("Report must have an existing parent inside workspace")
        target.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    if failures:
        raise SystemExit(1)
    print("GATE 1 numerical comparison PASSED (not validation of economics or all labels).")


if __name__ == "__main__":
    main()
