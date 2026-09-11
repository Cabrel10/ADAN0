#!/usr/bin/env python3
"""checkpoint_evolution.py — Évolution PPO + comportement segmentée par tranche
de timesteps (proxy des checkpoints 50k/100k/150k/200k...).

Parse fa_500k.log et découpe les métriques PPO + ACTION_DIST + FA_WATCHDOG +
trades par fenêtre de `total_timesteps`. Permet de répondre : à quel checkpoint
le modèle progresse / stagne / régresse.

Le SB3 logger imprime des blocs `time/total_timesteps`. On rattache chaque update
au total_timesteps le plus récent, puis on agrège par bin.

Lecture seule, streaming.
"""
from __future__ import annotations
import re, sys, json
from collections import defaultdict

LOG = sys.argv[1] if len(sys.argv) > 1 else "logs/training/fa_500k.log"
BIN = int(sys.argv[2]) if len(sys.argv) > 2 else 50000

RE_TS = re.compile(r"total_timesteps\s+\|\s+(\d+)")
RE_METRIC = re.compile(r"\|\s+([a-z_/]+)\s+\|\s+(-?[0-9.eE+]+)\s+\|")
RE_ACT = re.compile(
    r"ACTION_DIST \w+\] W\d+ \| tp_raw_mean=([-+0-9.]+) sl_raw_mean=([-+0-9.]+) \| "
    r"tp_sat=(\d+)%.*?sl_sat=(\d+)% \| tp_pct_mean=([0-9.]+)%"
)
RE_FA = re.compile(r"FA_WATCHDOG \w+\].*?future_share=([0-9.]+)%")
RE_CLOSE = re.compile(r"\b(TP_HIT|SL_HIT|AGENT_CLOSE)\b")

TRACK = {"ep_rew_mean", "approx_kl", "clip_fraction", "entropy_loss",
         "policy_gradient_loss", "value_loss", "explained_variance"}

bins = defaultdict(lambda: defaultdict(list))   # bin -> metric -> [vals]
cur_ts = 0

def binid(ts):
    return (ts // BIN) * BIN

with open(LOG, "r", errors="ignore") as f:
    for line in f:
        mt = RE_TS.search(line)
        if mt:
            cur_ts = int(mt.group(1))
            continue
        b = binid(cur_ts)
        if "|" in line:
            mm = RE_METRIC.search(line)
            if mm:
                key = mm.group(1).split("/")[-1]
                if key in TRACK:
                    try:
                        bins[b][key].append(float(mm.group(2)))
                    except ValueError:
                        pass
        if "ACTION_DIST" in line:
            ma = RE_ACT.search(line)
            if ma:
                bins[b]["tp_raw"].append(float(ma.group(1)))
                bins[b]["sl_raw"].append(float(ma.group(2)))
                bins[b]["tp_sat"].append(float(ma.group(3)))
                bins[b]["sl_sat"].append(float(ma.group(4)))
                bins[b]["tp_pct"].append(float(ma.group(5)))
        if "FA_WATCHDOG" in line:
            mf = RE_FA.search(line)
            if mf:
                bins[b]["fa_share"].append(float(mf.group(1)))
        if "_HIT" in line or "AGENT_CLOSE" in line:
            for tag in RE_CLOSE.findall(line):
                bins[b][tag].append(1)

def avg(xs):
    return round(sum(xs) / len(xs), 4) if xs else None

rows = []
for b in sorted(bins):
    d = bins[b]
    tp = len(d.get("TP_HIT", []))
    sl = len(d.get("SL_HIT", []))
    rows.append({
        "ckpt_~steps": b + BIN,
        "ep_rew_mean": avg(d.get("ep_rew_mean", [])),
        "entropy": avg(d.get("entropy_loss", [])),
        "value_loss": avg(d.get("value_loss", [])),
        "explained_var": avg(d.get("explained_variance", [])),
        "approx_kl": avg(d.get("approx_kl", [])),
        "clip_frac": avg(d.get("clip_fraction", [])),
        "tp_pct_mean": avg(d.get("tp_pct", [])),
        "tp_sat%": avg(d.get("tp_sat", [])),
        "sl_sat%": avg(d.get("sl_sat", [])),
        "fa_share%": avg(d.get("fa_share", [])),
        "TP_HIT": tp, "SL_HIT": sl,
        "TP/SL": round(tp / sl, 2) if sl else None,
    })

# Verdict heuristique par transition
def verdict(rows):
    out = []
    for i in range(1, len(rows)):
        a, b = rows[i-1], rows[i]
        msgs = []
        if a["ep_rew_mean"] and b["ep_rew_mean"]:
            dr = b["ep_rew_mean"] - a["ep_rew_mean"]
            msgs.append(f"reward {'↑' if dr>0 else '↓'}{abs(dr):.0f}")
        if a["TP/SL"] and b["TP/SL"]:
            msgs.append(f"TP/SL {a['TP/SL']}→{b['TP/SL']}")
        if b["explained_var"] is not None:
            msgs.append(f"expl_var={b['explained_var']}")
        # plateau detection
        if (a["ep_rew_mean"] and b["ep_rew_mean"] and
                abs(b["ep_rew_mean"] - a["ep_rew_mean"]) / (abs(a["ep_rew_mean"]) + 1e-9) < 0.05):
            tag = "PLATEAU"
        elif b["ep_rew_mean"] and a["ep_rew_mean"] and b["ep_rew_mean"] > a["ep_rew_mean"]:
            tag = "PROGRESSE"
        else:
            tag = "REGRESSE?"
        out.append({"transition": f"{a['ckpt_~steps']}→{b['ckpt_~steps']}",
                    "verdict": tag, "detail": "; ".join(msgs)})
    return out

print(json.dumps({"bin_size": BIN, "checkpoints": rows,
                  "transitions": verdict(rows)}, indent=2))
