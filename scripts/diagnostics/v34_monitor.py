#!/usr/bin/env python3
"""V34 surveillance — critic-fix A/B vs V33.

Tracks the decisive chain across the critical zone (upd ~200-450 where V33
collapsed): explained_variance (EV), a0_mean (mu), nB/nH/nS (action balance),
approx_kl, clip_fraction. Prints a compact rolling table and a verdict.

The ONE change in V34 vs V33 is ADAN_NORM_REWARD=1 (reward/return
normalization -> unit-variance critic value target). Everything else
(anchor lambda=0.20, MTM off, PPO hp) is identical, so any EV / balance
improvement is attributable to that single variable.

Usage:
    python3 scripts/diagnostics/v34_monitor.py <train_log> [--once]
"""
import argparse
import re
import sys
import time

EV_RE = re.compile(r"explained_variance\s*\|\s*([-0-9.]+)")
ANCHOR_RE = re.compile(
    r"ANCHOR_DEBUG\]\s*upd=(\d+).*?a0_mean=([-0-9.]+).*?"
    r"nB=(\d+)\s+nS=(\d+)\s+nH=(\d+)"
)
KL_RE = re.compile(r"approx_kl\s*\|\s*([-0-9.eE]+)")
CLIP_RE = re.compile(r"clip_fraction\s*\|\s*([-0-9.eE]+)")


def scan(path):
    evs, anchors, kls, clips = [], [], [], []
    try:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                m = EV_RE.search(line)
                if m:
                    evs.append(float(m.group(1)))
                    continue
                m = ANCHOR_RE.search(line)
                if m:
                    anchors.append((int(m.group(1)), float(m.group(2)),
                                    int(m.group(3)), int(m.group(4)),
                                    int(m.group(5))))
                    continue
                m = KL_RE.search(line)
                if m:
                    kls.append(float(m.group(1)))
                    continue
                m = CLIP_RE.search(line)
                if m:
                    clips.append(float(m.group(1)))
    except FileNotFoundError:
        return None
    return evs, anchors, kls, clips


def verdict(evs, anchors):
    """Classify the run state: confirme / probable / infirme / non resolu."""
    if not evs:
        return "non resolu", "no EV points yet"
    recent_ev = evs[-10:]
    mean_ev = sum(recent_ev) / len(recent_ev)
    pos_frac = sum(1 for e in recent_ev if e > 0) / len(recent_ev)
    # collapse check from action balance
    collapsed = False
    if anchors:
        upd, mu, nB, nS, nH = anchors[-1]
        tot = max(1, nB + nS + nH)
        if nB == 0 and nS / tot > 0.9:
            collapsed = True
    if collapsed:
        return "infirme", (f"COLLAPSE: last upd nB=0, SELL-absorbed, "
                           f"mean_ev={mean_ev:.3f} — critic fix did NOT hold")
    if mean_ev > 0.3:
        return "confirme", (f"critic LEARNING: mean_ev={mean_ev:.3f} "
                            f"(>0.3 target), pos_frac={pos_frac:.0%}")
    if mean_ev > -0.1 and pos_frac >= 0.4:
        return "probable", (f"critic improving: mean_ev={mean_ev:.3f} "
                            f"(V33 was -0.8..-2.0), pos_frac={pos_frac:.0%}")
    return "non resolu", (f"mean_ev={mean_ev:.3f}, pos_frac={pos_frac:.0%} "
                          f"— still noisy, keep surveilling")


def report(path):
    res = scan(path)
    if res is None:
        print(f"[V34_MON] log not found: {path}")
        return
    evs, anchors, kls, clips = res
    print("=" * 70)
    print(f"[V34_MON] {path}")
    print(f"  EV points: {len(evs)} | anchor points: {len(anchors)}")
    if evs:
        print(f"  EV last10: {[round(e, 3) for e in evs[-10:]]}")
        print(f"  EV min/mean/max: {min(evs):.3f} / "
              f"{sum(evs)/len(evs):.3f} / {max(evs):.3f}")
    if anchors:
        print("  upd    mu       nB     nS     nH")
        for upd, mu, nB, nS, nH in anchors[-8:]:
            print(f"  {upd:<6} {mu:+.4f}  {nB:<6} {nS:<6} {nH:<6}")
    if kls:
        print(f"  approx_kl last: {kls[-3:]}")
    if clips:
        print(f"  clip_fraction last: {clips[-3:]}")
    status, reason = verdict(evs, anchors)
    print(f"  STATUS: {status.upper()} — {reason}")
    # NEXT_ACTION per autonomy loop
    if status == "confirme":
        na = "continue to 500k; then eval_v34 + backtest vs baselines"
    elif status == "probable":
        na = "keep surveilling through upd 450 (V33 collapse zone)"
    elif status == "infirme":
        na = ("norm_reward alone insufficient -> next critic lever: "
              "vf_coef up or n_epochs down (V16 reco #3)")
    else:
        na = "gather more updates; re-evaluate at next checkpoint"
    print(f"  NEXT_ACTION: {na}")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=180)
    args = ap.parse_args()
    if args.once:
        report(args.log)
        return
    while True:
        report(args.log)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
