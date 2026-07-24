#!/usr/bin/env python3
"""
DIAGNOSTIC-V4 hotfix: NameError '_base' in [STERILE_SELL] log.

The V4 PART 4 refactor moved the local _base/_r/_cap definitions into the
_sterile_penalty_for_tier() helper, but the [STERILE_SELL] warning log
(line ~8000) still referenced those now-undefined locals, raising
`NameError: name '_base' is not defined` every step.

This patch rewrites that log f-string to re-read the three components from
config locally (deterministic, read-only) so the log works without the
removed locals. Numeric reward behavior is unchanged.

Idempotent: refuses to run twice (checks for the already-fixed marker).
"""
import sys
from pathlib import Path

FILE = Path(__file__).resolve().parents[2] / \
    "src/adan_trading_bot/environment/multi_asset_chunked_env.py"

OLD = (
    '                _sterile_pen = _sterile_penalty_for_tier()\n'
    '                self._step_invalid_penalty += -_sterile_pen\n'
    '                if self.current_step % 50 == 0:\n'
    '                    self.logger.warning(\n'
    '                        f"[STERILE_SELL] {asset} | SELL sans position | "\n'
    '                        f"tier={_tname}(k={_k}) | pen=-{_sterile_pen:.5f} "\n'
    '                        f"(base={_base:.4f} r={_r} cap={_cap})"\n'
    '                    )\n'
)

NEW = (
    '                _sterile_pen = _sterile_penalty_for_tier()\n'
    '                self._step_invalid_penalty += -_sterile_pen\n'
    '                if self.current_step % 50 == 0:\n'
    '                    # DIAGNOSTIC-V4: log components re-read from config\n'
    '                    # because _base/_r/_cap now live in the helper.\n'
    '                    _rs_log = self.config.get("reward_shaping", {})\n'
    '                    _base_log = float(\n'
    '                        _rs_log.get("invalid_trade_penalty_weight", 0.005))\n'
    '                    _r_log = float(\n'
    '                        _rs_log.get("sterile_action_geom_ratio", 1.6))\n'
    '                    _cap_log = float(\n'
    '                        _rs_log.get("sterile_action_penalty_cap", 0.05))\n'
    '                    self.logger.warning(\n'
    '                        f"[STERILE_SELL] {asset} | SELL sans position | "\n'
    '                        f"tier={_tname}(k={_k}) | pen=-{_sterile_pen:.5f} "\n'
    '                        f"(base={_base_log:.4f} r={_r_log} cap={_cap_log})"\n'
    '                    )\n'
)


def main():
    src = FILE.read_text()

    if "_base_log" in src and OLD not in src:
        print("ALREADY PATCHED (found _base_log, no old block). No-op.")
        return 0

    n = src.count(OLD)
    if n != 1:
        print(f"ERROR: expected exactly 1 occurrence of OLD block, found {n}.")
        # Help debugging: check for the orphan reference
        if "f\"(base={_base:.4f} r={_r} cap={_cap})\"" in src:
            print("  -> orphan '_base' log line IS present but block differs.")
        return 1

    src = src.replace(OLD, NEW, 1)
    FILE.write_text(src)
    print("PATCHED OK: [STERILE_SELL] log no longer references _base/_r/_cap.")
    # sanity: ensure no remaining orphan
    if "(base={_base:.4f} r={_r} cap={_cap})" in FILE.read_text():
        print("WARNING: orphan reference still present after patch!")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
