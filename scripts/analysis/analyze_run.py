#!/usr/bin/env python3
"""analyze_run.py — Analyse quantitative + ML d'un run PPO ADAN0.

Parse le log d'entraînement (fa_500k.log) et produit :
  A. Évolution RL : ep_rew_mean, policy_loss, value_loss, entropy, approx_kl,
     explained_variance, clip_fraction — par update PPO (chronologique).
  B. Évolution trading : TP_HIT / SL_HIT / AGENT_CLOSE counts + ratios cumulés
     par fenêtre (early / mid / late) pour voir si le comportement évolue.
  C. Comportement : FA_WATCHDOG (future_share) + ACTION_DIST (saturation,
     tp_pct_mean) au fil du temps.
  D. Verdict heuristique : évolue / stagne / collapse / cherche exploitation.

Lecture seule. N'instancie aucun modèle. Streaming ligne par ligne (log ~1 Go).
"""
from __future__ import annotations

import re
import sys
import json
from collections import defaultdict

LOG = sys.argv[1] if len(sys.argv) > 1 else "logs/training/fa_500k.log"

# ── Regex ────────────────────────────────────────────────────────────────
RE_METRIC = re.compile(r"\|\s+([a-z_]+)\s+\|\s+(-?[0-9.eE+]+)\s+\|")
RE_FA = re.compile(
    r"FA_WATCHDOG (\w+)\] Worker \d+ \| future_share=([0-9.]+)%.*?"
    r"mean_abs_future=([0-9.]+).*?mean_abs_pnl=([0-9.]+).*?n=(\d+)"
)
RE_ACT = re.compile(
    r"ACTION_DIST (\w+)\] W\d+ \| tp_raw_mean=([-+0-9.]+) sl_raw_mean=([-+0-9.]+) \| "
    r"tp_sat=(\d+)%.*?sl_sat=(\d+)% \| tp_pct_mean=([0-9.]+)%.*?n=(\d+)"
)
RE_CLOSE = re.compile(r"\b(TP_HIT|SL_HIT|AGENT_CLOSE)\b")

# RL metric names we track (printed by SB3 logger)
RL_KEYS = {
    "ep_rew_mean", "approx_kl", "clip_fraction", "entropy_loss",
    "policy_gradient_loss", "value_loss", "explained_variance",
}

# ── Accumulators ───────────────────────────────────────────────────────────
rl_series = defaultdict(list)      # key -> [values in chronological order]
fa_samples = []                    # (n, future_share, mean_abs_future, mean_abs_pnl)
act_samples = []                   # (n, tp_raw, sl_raw, tp_sat, sl_sat, tp_pct)
close_counts = defaultdict(int)
# trade evolution: bucket closes into 3 thirds by line index
close_seq = []                     # list of "TP"/"SL"/"AG" in order

n_lines = 0
with open(LOG, "r", errors="ignore") as f:
    for line in f:
        n_lines += 1
        if "|" in line:
            m = RE_METRIC.search(line)
            if m and m.group(1) in RL_KEYS:
                try:
                    rl_series[m.group(1)].append(float(m.group(2)))
                except ValueError:
                    pass
        if "FA_WATCHDOG" in line:
            m = RE_FA.search(line)
            if m:
                fa_samples.append((int(m.group(5)), float(m.group(2)),
                                   float(m.group(3)), float(m.group(4))))
        if "ACTION_DIST" in line:
            m = RE_ACT.search(line)
            if m:
                act_samples.append((int(m.group(7)), float(m.group(2)),
                                    float(m.group(3)), int(m.group(4)),
                                    int(m.group(5)), float(m.group(6))))
        if "_HIT" in line or "AGENT_CLOSE" in line:
            for tag in RE_CLOSE.findall(line):
                close_counts[tag] += 1
                close_seq.append(tag[:2])  # TP/SL/AG


def stats(xs):
    if not xs:
        return None
    n = len(xs)
    return {
        "n": n, "first": round(xs[0], 4), "last": round(xs[-1], 4),
        "min": round(min(xs), 4), "max": round(max(xs), 4),
        "mean": round(sum(xs) / n, 4),
    }


def trend(xs, frac=0.2):
    """Compare mean of last frac vs first frac -> direction."""
    if len(xs) < 10:
        return "n/a"
    k = max(1, int(len(xs) * frac))
    early = sum(xs[:k]) / k
    late = sum(xs[-k:]) / k
    d = late - early
    rng = (max(xs) - min(xs)) or 1e-9
    rel = d / rng
    if abs(rel) < 0.1:
        return f"STABLE ({early:.3f}->{late:.3f})"
    return f"{'UP' if d > 0 else 'DOWN'} ({early:.3f}->{late:.3f}, {rel:+.0%})"


def thirds_ratio(seq):
    """TP/SL ratio per third of the run (early/mid/late)."""
    if not seq:
        return []
    t = len(seq) // 3 or 1
    out = []
    for i, (a, b) in enumerate([(0, t), (t, 2 * t), (2 * t, len(seq))]):
        chunk = seq[a:b]
        tp = chunk.count("TP")
        sl = chunk.count("SL")
        ag = chunk.count("AG")
        tot = len(chunk) or 1
        out.append({
            "third": ["early", "mid", "late"][i],
            "TP%": round(100 * tp / tot, 1),
            "SL%": round(100 * sl / tot, 1),
            "AG%": round(100 * ag / tot, 1),
            "TP/SL": round(tp / sl, 2) if sl else None,
            "n": tot,
        })
    return out


report = {
    "log": LOG,
    "lines_parsed": n_lines,
    "A_RL_evolution": {k: {"stats": stats(v), "trend": trend(v)}
                       for k, v in sorted(rl_series.items())},
    "B_trading": {
        "totals": dict(close_counts),
        "TP/SL_global": round(close_counts["TP_HIT"] / close_counts["SL_HIT"], 2)
        if close_counts.get("SL_HIT") else None,
        "evolution_thirds": thirds_ratio(close_seq),
    },
    "C_behavior": {
        "FA_future_share_pct": stats([s[1] for s in fa_samples]),
        "FA_mean_abs_future": stats([s[2] for s in fa_samples]),
        "FA_mean_abs_pnl": stats([s[3] for s in fa_samples]),
        "ACT_tp_raw_mean": stats([s[1] for s in act_samples]),
        "ACT_sl_raw_mean": stats([s[2] for s in act_samples]),
        "ACT_tp_sat_pct": stats([s[3] for s in act_samples]),
        "ACT_sl_sat_pct": stats([s[4] for s in act_samples]),
        "ACT_tp_pct_mean": stats([s[5] for s in act_samples]),
    },
}

print(json.dumps(report, indent=2))
