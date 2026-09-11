#!/usr/bin/env python3
"""Pretty-print a single fixed-capital backtest JSON as one summary line."""
import json
import sys


def main() -> int:
    out, label = sys.argv[1], sys.argv[2]
    try:
        d = json.load(open(out))
    except Exception as e:  # noqa: BLE001
        print(f"RESULT {label}: ERR {e}")
        return 0
    print(
        "RESULT %s: n_trades=%s WR=%.3f PF=%.3f exp=%.4f%% ret=%.3f%% sharpe=%.3f | %s"
        % (
            label,
            d.get("n_trades"),
            d.get("win_rate", 0) or 0,
            d.get("profit_factor", 0) or 0,
            d.get("avg_pnl_pct_per_trade", 0) or 0,
            d.get("total_return_pct", 0) or 0,
            d.get("sharpe_like", 0) or 0,
            d.get("verdict"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
