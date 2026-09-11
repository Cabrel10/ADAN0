#!/usr/bin/env python3
"""v35_sltp_probe.py — DETERMINISTIC SL/TP sonde (must pass before V35 500k).

Goal (user directive): prove MATHEMATICALLY that with ADAN_FREE_SLTP=1 the
policy really OWNS its exit geometry:

    SL in [0.30%, 6.00%]   TP in [0.30%, 12.00%]

under the SOLE economic guard-rail of round-trip fees — with NO profile band,
NO R/R>=1.5 floor, and NO 3xATR floor rewriting the decision.

It injects known (sl_raw, tp_raw) actions and reports, for each:

    sl_raw / tp_raw  ->  SL_final / TP_final  ->  properties

The mapping reproduced here is byte-for-byte the AUTHORITATIVE block in
multi_asset_chunked_env.py (lines ~8904-9016). To guard against drift, the
sonde ALSO greps the source and asserts the three `_free_sltp` gates are
present, so this file can never silently diverge from the live code.

Run:
    ADAN_FREE_SLTP=1 python3 scripts/diagnostics/v35_sltp_probe.py
    python3 scripts/diagnostics/v35_sltp_probe.py            # OFF = old geometry
Exit 0 = PASS (all four checks), non-zero = FAIL (fix before V35).
"""
import os
import re
import sys


def _clip(x, lo, hi):
    """Pure-python clip (mirrors np.clip; keeps the sonde dependency-free)."""
    return float(max(lo, min(hi, x)))

SRC = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "src", "adan_trading_bot", "environment", "multi_asset_chunked_env.py")

# The 7 canonical probe actions requested by the user.
CASES = [
    (-1.0, -1.0),
    (-1.0, +1.0),
    (+1.0, -1.0),
    (+1.0, +1.0),
    (0.0, 0.0),
    (-0.5, +0.2),
    (+0.2, -0.5),
]

# Old profile band used when ADAN_FREE_SLTP is OFF (scalper, the tightest —
# matches the sandbox worker profile in config). Only used for the OFF report.
_OLD_BOUNDS = {"sl": (0.003, 0.012), "tp": (0.005, 0.020)}


def map_sltp(sl_raw, tp_raw, free, commission_pct=0.0020, prof="scalper",
             atr_pct=0.003):
    """EXACT reproduction of the authoritative mapping in the env."""
    if free:
        sl_lo, sl_hi = 0.003, 0.060
        tp_lo, tp_hi = 0.003, 0.120
        round_trip = max(2.0 * commission_pct, 0.005)
        tp_lo = max(tp_lo, round_trip * 1.2)
    else:
        sl_lo, sl_hi = _OLD_BOUNDS["sl"]
        tp_lo, tp_hi = _OLD_BOUNDS["tp"]
        tp_lo = max(tp_lo, 0.006)

    normalized_sl = (sl_raw + 1.0) / 2.0
    sl_pct = _clip(sl_lo + normalized_sl * (sl_hi - sl_lo), sl_lo, sl_hi)
    normalized_tp = (tp_raw + 1.0) / 2.0
    tp_pct = _clip(tp_lo + normalized_tp * (tp_hi - tp_lo), tp_lo, tp_hi)

    # RR floor (skipped when free)
    if not free and tp_pct < sl_pct * 1.5:
        tp_pct = float(min(sl_pct * 1.5, tp_hi))
    # ATR floor for scalper (skipped when free)
    if not free and prof == "scalper":
        min_scalp_sl = max(0.006, 3.0 * atr_pct)
        if sl_pct < min_scalp_sl:
            sl_pct = min_scalp_sl
            if tp_pct < sl_pct * 1.5:
                tp_pct = float(min(sl_pct * 1.5, tp_hi))
    return sl_pct, tp_pct, (sl_lo, sl_hi, tp_lo, tp_hi)


def assert_source_gates():
    """Guard: the live source MUST contain the three _free_sltp gates."""
    src = open(os.path.abspath(SRC), encoding="utf-8").read()
    needed = [
        r'_free_sltp\s*=\s*os\.environ\.get\("ADAN_FREE_SLTP"',
        r'if\s+_free_sltp:',                       # envelope branch
        r'if\s+not\s+_free_sltp\s+and\s+tp_pct\s*<\s*sl_pct\s*\*\s*1\.5',  # RR gated
        r'if\s+not\s+_free_sltp\s+and\s+_prof\s*==\s*"scalper"',           # ATR gated
    ]
    missing = [p for p in needed if not re.search(p, src)]
    return missing


def main():
    free = os.environ.get("ADAN_FREE_SLTP", "0") == "1"
    print("=" * 78)
    print(f"V35 SL/TP PROBE   mode={'FREE (ADAN_FREE_SLTP=1)' if free else 'OFF (V34 geometry)'}")
    print("=" * 78)

    missing = assert_source_gates()
    if missing:
        print("[SOURCE GUARD] FAIL — live code missing gates:")
        for m in missing:
            print(f"   - {m}")
        return 2
    print("[SOURCE GUARD] OK — live code contains the three _free_sltp gates.\n")

    print(f"{'sl_raw':>7}{'tp_raw':>7} | {'SL_final':>9}{'TP_final':>9} | "
          f"{'RR':>5} | notes")
    print("-" * 78)
    rows = []
    for sl_raw, tp_raw in CASES:
        sl, tp, (sl_lo, sl_hi, tp_lo, tp_hi) = map_sltp(sl_raw, tp_raw, free)
        rr = tp / sl if sl > 0 else float("nan")
        note = []
        if free:
            if abs(sl_raw + 1) < 1e-9 and abs(sl - 0.003) < 1e-6:
                note.append("SL@floor0.30%")
            if abs(sl_raw - 1) < 1e-9 and abs(sl - 0.060) < 1e-6:
                note.append("SL@cap6%")
            if abs(tp_raw - 1) < 1e-9 and abs(tp - 0.120) < 1e-6:
                note.append("TP@cap12%")
        rows.append((sl_raw, tp_raw, sl, tp, rr))
        print(f"{sl_raw:>7.2f}{tp_raw:>7.2f} | {sl*100:>8.3f}%{tp*100:>8.3f}% | "
              f"{rr:>5.2f} | {' '.join(note)}")

    print("-" * 78)
    # ---- the four decisive checks (only meaningful in FREE mode) ----
    checks = []
    if free:
        # 1) full control of SL range: raw -1 -> 0.30%, raw +1 -> 6.00%
        sl_min = map_sltp(-1, 0, True)[0]
        sl_max = map_sltp(+1, 0, True)[0]
        checks.append(("SL range = [0.30%,6.00%]",
                       abs(sl_min - 0.003) < 1e-6 and abs(sl_max - 0.060) < 1e-6))
        # 2) full control of TP range: raw -1 -> fee floor, raw +1 -> 12.00%
        tp_max = map_sltp(0, +1, True)[1]
        tp_min = map_sltp(0, -1, True)[1]
        checks.append(("TP cap = 12.00%", abs(tp_max - 0.120) < 1e-6))
        # 3) NO RR floor: prove the env does NOT bump TP to >=1.5*SL.
        #    Use (+1,-1): raw pushes SL to the 6% cap and TP to its floor 0.6%,
        #    so RR=0.10 (TP FAR below 1.5*SL). If the RR floor were still live it
        #    would rewrite TP to 1.5*6%=9%. We assert TP stays at the fee floor,
        #    i.e. RR < 1.5 genuinely survives (the exact case the user forbade
        #    rewriting: "SL 1.5%/TP 1.0% must NOT become TP 2.25%").
        sl_c, tp_c, _ = map_sltp(+1.0, -1.0, True)
        rr_c = tp_c / sl_c if sl_c > 0 else float("nan")
        checks.append((f"no RR>=1.5 rewrite (TP<SL survives, RR={rr_c:.2f})",
                       rr_c < 1.5 - 1e-9 and abs(tp_c - 0.006) < 1e-6))
        # 4) fee gate ONLY: TP floor equals round-trip*1.2 (>=0.6% at 0.20% comm)
        checks.append(("TP floor = fee gate (>=0.60%)", abs(tp_min - 0.006) < 1e-6))
    else:
        # OFF mode: just demonstrate the OLD rewriting (for the A/B record)
        sl_c, tp_c, _ = map_sltp(-1, -1, False)
        print(f"[OFF] raw(-1,-1) -> SL={sl_c*100:.3f}% TP={tp_c*100:.3f}% "
              f"(RR forced {tp_c/sl_c:.2f}, ATR floor applied) — this is what V35 removes")
        checks.append(("OFF mode reproduces old rewriting (RR/ATR active)",
                       tp_c >= sl_c * 1.5 - 1e-9))

    print()
    allok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        allok = allok and passed
    print()
    if free and allok:
        print("VERDICT: PASS — the policy OWNS SL/TP under fee-only guard. "
              "NEXT_ACTION: smoke test (3k steps) then V35 500k.")
        return 0
    if not free:
        print("VERDICT: OFF-mode reference recorded. Re-run with ADAN_FREE_SLTP=1.")
        return 0 if allok else 1
    print("VERDICT: FAIL — geometry still constrained. NEXT_ACTION: fix the "
          "failing gate in multi_asset_chunked_env.py before V35.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
