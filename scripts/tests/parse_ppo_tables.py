#!/usr/bin/env python3
"""
ÉTAPE 1 / ÉTAPE 5 tool.
Parse SB3 PPO logger tables from a training log into a clean time series keyed on
the REAL total_timesteps (not env "Starting step" chunk counter).

Extracts per update: total_timesteps, learning_rate, approx_kl, clip_fraction,
explained_variance, std, entropy_loss, policy_gradient_loss, value_loss, loss.

Usage: python3 parse_ppo_tables.py <logfile> [--csv out.csv]
"""
import re
import sys
import argparse

# An SB3 table block is delimited by dashed lines; rows look like:  |    key    |  value  |
ROW_RE = re.compile(r"\|\s*([A-Za-z0-9_/ ]+?)\s*\|\s*([-+0-9.eE]+)\s*\|")

# Metric key -> canonical name (SB3 uses these nested keys)
KEYMAP = {
    "total_timesteps": "total_timesteps",
    "learning_rate": "learning_rate",
    "approx_kl": "approx_kl",
    "clip_fraction": "clip_fraction",
    "clip_range": "clip_range",
    "explained_variance": "explained_variance",
    "std": "std",
    "entropy_loss": "entropy_loss",
    "policy_gradient_loss": "policy_gradient_loss",
    "value_loss": "value_loss",
    "loss": "loss",
    "n_updates": "n_updates",
}


def parse(path):
    """Split the log into table blocks and pull metrics out of each."""
    updates = []
    cur = {}
    in_table = False
    with open(path, "r", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            # A table starts/ends on a line that is all dashes (len>=20)
            if set(stripped) == {"-"} and len(stripped) >= 20:
                if in_table and cur:
                    updates.append(cur)
                    cur = {}
                in_table = not in_table if not cur else True
                # Toggle: enter on first dash-line, flush+reset on closing dash-line
                if not stripped:
                    continue
                # Simpler: treat every all-dash line as a boundary
                in_table = True
                continue
            m = ROW_RE.search(line)
            if m:
                key = m.group(1).strip()
                val = m.group(2).strip()
                if key in KEYMAP:
                    try:
                        cur[KEYMAP[key]] = float(val)
                    except ValueError:
                        pass
    if cur:
        updates.append(cur)
    # keep only blocks that actually have a PPO update (total_timesteps + at least one train metric)
    clean = [u for u in updates if "total_timesteps" in u and ("clip_fraction" in u or "approx_kl" in u)]
    # dedup consecutive identical total_timesteps (SB3 prints one table per rollout)
    dedup = []
    seen = set()
    for u in sorted(clean, key=lambda x: x["total_timesteps"]):
        ts = u["total_timesteps"]
        if ts in seen:
            # merge missing fields
            dedup[-1].update({k: v for k, v in u.items() if k not in dedup[-1]})
        else:
            seen.add(ts)
            dedup.append(u)
    return dedup


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--max-ts", type=float, default=None, help="only rows with total_timesteps <= this")
    args = ap.parse_args()

    ups = parse(args.logfile)
    if args.max_ts is not None:
        ups = [u for u in ups if u["total_timesteps"] <= args.max_ts]

    cols = ["total_timesteps", "learning_rate", "approx_kl", "clip_fraction",
            "explained_variance", "std", "entropy_loss", "value_loss"]
    print(f"# parsed {len(ups)} PPO updates from {args.logfile}")
    print("ts        lr         kl        clip     ev        std      ent      vloss")
    for u in ups:
        def g(k):
            return u.get(k, float("nan"))
        print(f"{g('total_timesteps'):<9.0f} {g('learning_rate'):<10.3e} "
              f"{g('approx_kl'):<9.4f} {g('clip_fraction'):<8.3f} "
              f"{g('explained_variance'):<9.3f} {g('std'):<8.3f} "
              f"{g('entropy_loss'):<8.2f} {g('value_loss'):<8.3f}")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write(",".join(cols) + "\n")
            for u in ups:
                f.write(",".join(str(u.get(c, "")) for c in cols) + "\n")
        print(f"# wrote {args.csv}")

    # correlations + transition detection
    have = [u for u in ups if "clip_fraction" in u and "learning_rate" in u]
    if len(have) >= 3:
        clip = [u["clip_fraction"] for u in have]
        lr = [u["learning_rate"] for u in have]
        ts = [u["total_timesteps"] for u in have]
        kl = [u.get("approx_kl", float("nan")) for u in have]
        print("\n# --- CORRELATIONS ---")
        print(f"r(clip, LR)        = {pearson(clip, lr):.4f}")
        print(f"r(clip, total_ts)  = {pearson(clip, ts):.4f}")
        print(f"r(approx_kl, LR)   = {pearson([k for k in kl if k==k], [l for k,l in zip(kl,lr) if k==k]):.4f}")

        # first clip >= 0.30
        for u in have:
            if u["clip_fraction"] >= 0.30:
                print(f"\n# first clip_fraction >= 0.30 at ts={u['total_timesteps']:.0f} "
                      f"LR={u['learning_rate']:.3e} kl={u.get('approx_kl', float('nan')):.4f} "
                      f"std={u.get('std', float('nan')):.3f} ent={u.get('entropy_loss', float('nan')):.2f}")
                break
        peak = max(have, key=lambda x: x["clip_fraction"])
        print(f"# peak clip_fraction={peak['clip_fraction']:.3f} at ts={peak['total_timesteps']:.0f} "
              f"LR={peak['learning_rate']:.3e}")


if __name__ == "__main__":
    main()
