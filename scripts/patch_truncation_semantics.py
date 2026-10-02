#!/usr/bin/env python3
"""ADAN0 patcher: correct Gymnasium terminated/truncated semantics.

Root cause fixed here
---------------------
`multi_asset_chunked_env.step()` returned `terminated = done` for ALL episode
endings.  Two of those endings are pure *time-limit truncations*, not MDP
terminals:

  * `self.current_step >= self.max_steps`            (L~3941)
  * `self.current_chunk_idx >= chunks_limit`         (L~4063)

With `terminated=True`, SB3 does **not** bootstrap the value function at the
boundary: the target return is computed as if all future reward were exactly
zero.  Those boundaries are far more frequent than real economic deaths, so the
critic is trained against systematically wrong targets from the very first
update — consistent with `explained_variance < 0` on step-one for both BTC and
DOGE, and with corrupted advantages pushing the policy into a degenerate mode.

After this patch a single flag `self._termination_kind` carries the semantics:
  * "terminal"  -> DRAWDOWN_KILL / BANKRUPT / explosion / load failure
                   => terminated=True, truncated=False  (no bootstrap: correct,
                      the economic episode really ended)
  * "truncated" -> max_steps / max_chunks_per_episode
                   => terminated=False, truncated=True  (bootstrap: correct,
                      the process continues, only our window stops)

Idempotent: re-running detects the markers and makes no change.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ENV = Path(__file__).resolve().parents[1] / "src/adan_trading_bot/environment/multi_asset_chunked_env.py"

MARK = "ADAN0_TRUNCATION_SEMANTICS"


def fail(msg: str) -> None:
    print(f"[PATCH][FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    src = ENV.read_text(encoding="utf-8")
    if MARK in src:
        print("[PATCH][SKIP] already applied (marker present)")
        return
    orig = src

    # ------------------------------------------------------------------ 1
    # Initialise the semantics flag where `done = False` is initialised in
    # the termination-check block.
    anchor = '            done = False\n            termination_reason = ""\n'
    if anchor not in src:
        fail("anchor 'done = False / termination_reason' not found")
    src = src.replace(
        anchor,
        anchor
        + f'            # {MARK}: default semantics for this step. Overridden to\n'
        + '            # "truncated" by pure time-limit boundaries below.\n'
        + '            self._termination_kind = "terminal"\n',
        1,
    )

    # ------------------------------------------------------------------ 2
    # max_steps reached == time-limit truncation.
    a2 = (
        "            if self.current_step >= self.max_steps:\n"
        "                done = True\n"
        "                termination_reason = (\n"
    )
    if a2 not in src:
        fail("anchor max_steps not found")
    src = src.replace(
        a2,
        "            if self.current_step >= self.max_steps:\n"
        "                done = True\n"
        f'                # {MARK}: time limit, NOT an MDP terminal. SB3 must\n'
        "                # bootstrap the value function at this boundary.\n"
        '                self._termination_kind = "truncated"\n'
        "                termination_reason = (\n",
        1,
    )

    # ------------------------------------------------------------------ 3
    # max_chunks_per_episode reached == data-window truncation.
    a3 = (
        "                if self.current_chunk_idx >= chunks_limit:\n"
        "                    done = True\n"
        "                    self.done = True\n"
    )
    if a3 not in src:
        fail("anchor chunks_limit not found")
    src = src.replace(
        a3,
        "                if self.current_chunk_idx >= chunks_limit:\n"
        "                    done = True\n"
        "                    self.done = True\n"
        f'                    # {MARK}: end of the data window, NOT an economic\n'
        "                    # death. Bootstrap required.\n"
        '                    self._termination_kind = "truncated"\n',
        1,
    )

    # ------------------------------------------------------------------ 4
    # Chunk load failure keeps "terminal" semantics (real dead end) — make it
    # explicit so a previous "truncated" in the same step cannot leak through.
    a4 = (
        "                        done = True\n"
        "                        self.done = True\n"
        "                        termination_reason = (\n"
        '                            f"Failed to load chunk'
    )
    if a4 not in src:
        fail("anchor chunk-load-failure not found")
    src = src.replace(
        a4,
        "                        done = True\n"
        "                        self.done = True\n"
        f'                        # {MARK}: unrecoverable, genuine terminal.\n'
        '                        self._termination_kind = "terminal"\n'
        "                        termination_reason = (\n"
        '                            f"Failed to load chunk',
        1,
    )

    # ------------------------------------------------------------------ 5
    # The return-flag computation itself.
    a5 = (
        "            # Use local 'done' to signal termination for this step\n"
        "            terminated = done\n"
        "            truncated = False\n"
        "\n"
        '            max_steps = getattr(self, "_max_episode_steps", float("inf"))\n'
        "            if self.current_step >= max_steps:\n"
        "                truncated = True\n"
        "                self.done = True\n"
    )
    if a5 not in src:
        fail("anchor terminated/truncated computation not found")
    src = src.replace(
        a5,
        "            # Use local 'done' to signal termination for this step.\n"
        f"            # {MARK}: split Gymnasium semantics. `_termination_kind`\n"
        '            # is "truncated" for pure time/data-window limits and\n'
        '            # "terminal" for economic deaths (DRAWDOWN_KILL, BANKRUPT,\n'
        "            # explosion, unrecoverable load error). SB3 bootstraps the\n"
        "            # value function iff truncated=True and terminated=False,\n"
        "            # which is exactly what a window boundary requires.\n"
        '            _kind = getattr(self, "_termination_kind", "terminal")\n'
        "            terminated = bool(done) and _kind != \"truncated\"\n"
        "            truncated = bool(done) and _kind == \"truncated\"\n"
        "\n"
        '            max_steps = getattr(self, "_max_episode_steps", float("inf"))\n'
        "            if self.current_step >= max_steps:\n"
        "                # Hard external time limit: truncation, never a terminal.\n"
        "                truncated = True\n"
        "                terminated = False\n"
        "                self.done = True\n"
        '            info_termination_kind = "truncated" if truncated else ("terminal" if terminated else "none")\n',
        1,
    )

    # ------------------------------------------------------------------ 6
    # Expose the semantics in `info` for the causal record / audits.
    a6 = (
        "            info = self._get_info()\n"
        "\n"
        '            if hasattr(self, "_last_reward_components"):\n'
        '                info.update({"reward_components": self._last_reward_components})\n'
    )
    if a6 not in src:
        fail("anchor info/reward_components not found")
    src = src.replace(
        a6,
        "            info = self._get_info()\n"
        f'            # {MARK}: make the boundary semantics auditable downstream.\n'
        '            info["termination_kind"] = info_termination_kind\n'
        '            info["terminated"] = bool(terminated)\n'
        '            info["truncated"] = bool(truncated)\n'
        "\n"
        '            if hasattr(self, "_last_reward_components"):\n'
        '                info.update({"reward_components": self._last_reward_components})\n',
        1,
    )

    if src == orig:
        fail("no modification produced")

    ENV.write_text(src, encoding="utf-8")
    n = len(re.findall(MARK, src))
    print(f"[PATCH][OK] {ENV} patched, {n} markers inserted")


if __name__ == "__main__":
    main()
