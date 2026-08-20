#!/usr/bin/env python3
"""eval_v33.py — Évaluateur GO/NO-GO d'un run de training ADAN (V33+).

Applique la table de décision utilisateur sur un log d'entraînement (ou un radar
JSONL). Réutilisable pour V33, V34, ... — passe le log en argument.

Usage:
    python scripts/diagnostics/eval_v33.py logs/v32/v33_train_<ts>.log
    python scripts/diagnostics/eval_v33.py logs/v32/v33_train_<ts>.log --radar logs/v32/radar_v33_<ts>.jsonl

Critères (GO si TOUS verts) :
    PV finale         > 20.50           (sinon NO-GO)
    Diversité         nB/nH/nS > 0 persistants
    EV critic         médian dernier tiers > 0.3  (warmup EV<0 toléré au début)
    μ final           dans [-0.5, +0.5] (|μ|>1.0 = collapse)
    σ final           > 0.1             (<0.1 = gel)
    Early-stops       < 30%             (>50% = NO-GO)
    Collapse          aucun nB=0/nS=0 persistant, aucun adv=NaN persistant

Sortie: résumé lisible + code retour 0 (GO) / 1 (NO-GO) / 2 (indéterminé/en cours).
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

# ---- Regex sur le format réel du log SB3 (tables pipe) + ANCHOR_DEBUG ----
RE_EV = re.compile(r"\|\s*explained_variance\s*\|\s*([-\d.eEnan]+)\s*\|")
RE_ANCHOR = re.compile(
    r"a0_mean=([-\d.eEnan]+)\s+a0_std=([-\d.eEnan]+).*?"
    r"nB=(\d+)\s+nS=(\d+)\s+nH=(\d+)"
)
RE_ADVNAN = re.compile(r"adv_BUY=nan")
RE_PORT = re.compile(r"Portfolio value:\s*([\d.]+)")
RE_NUPD = re.compile(r"\|\s*n_updates\s*\|\s*(\d+)\s*\|")
RE_TIMESTEPS = re.compile(r"\|\s*total_timesteps\s*\|\s*(\d+)\s*\|")
# SB3 logs "Early stopping at step X due to reaching max kl" when target_kl trips
RE_EARLYSTOP = re.compile(r"[Ee]arly stopping")

INITIAL_PV = 20.50


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def parse_log(path: Path):
    data = {
        "ev": [], "mu": [], "std": [],
        "nB": [], "nS": [], "nH": [],
        "port": [], "adv_nan": 0, "early_stops": 0,
        "n_updates": 0, "total_timesteps": 0,
    }
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = RE_EV.search(line)
            if m:
                data["ev"].append(_f(m.group(1)))
            m = RE_ANCHOR.search(line)
            if m:
                data["mu"].append(_f(m.group(1)))
                data["std"].append(_f(m.group(2)))
                data["nB"].append(int(m.group(3)))
                data["nS"].append(int(m.group(4)))
                data["nH"].append(int(m.group(5)))
            if RE_ADVNAN.search(line):
                data["adv_nan"] += 1
            if RE_EARLYSTOP.search(line):
                data["early_stops"] += 1
            m = RE_PORT.search(line)
            if m:
                data["port"].append(_f(m.group(1)))
            m = RE_NUPD.search(line)
            if m:
                data["n_updates"] = int(m.group(1))
            m = RE_TIMESTEPS.search(line)
            if m:
                data["total_timesteps"] = int(m.group(1))
    return data


def last_third(seq):
    if not seq:
        return []
    n = max(1, len(seq) // 3)
    return seq[-n:]


def median_or_nan(seq):
    seq = [x for x in seq if x == x]  # drop NaN
    return statistics.median(seq) if seq else float("nan")


def evaluate(data, target_steps=500000):
    checks = []
    verdict_go = True
    indeterminate = False

    # --- Progress ---
    ts = data["total_timesteps"]
    finished = ts >= target_steps * 0.98
    if not finished:
        indeterminate = True

    # --- PV finale ---
    pv = data["port"][-1] if data["port"] else float("nan")
    pv_ok = pv == pv and pv > INITIAL_PV
    checks.append(("PV finale > 20.50", pv, "GO" if pv_ok else "NO-GO"))
    verdict_go &= pv_ok

    # --- Diversité (dernier tiers: aucune action à 0) ---
    lB, lS, lH = last_third(data["nB"]), last_third(data["nS"]), last_third(data["nH"])
    div_ok = bool(lB) and min(lB) > 0 and min(lS) > 0 and min(lH) > 0
    div_str = (f"minB={min(lB) if lB else '-'} "
               f"minS={min(lS) if lS else '-'} "
               f"minH={min(lH) if lH else '-'}")
    checks.append(("Diversité nB/nH/nS > 0 (dernier tiers)", div_str,
                   "GO" if div_ok else "NO-GO"))
    verdict_go &= div_ok

    # --- EV médian dernier tiers > 0.3 ---
    ev_med = median_or_nan(last_third(data["ev"]))
    ev_ok = ev_med == ev_med and ev_med > 0.3
    checks.append(("EV médian (dernier tiers) > 0.3", round(ev_med, 3),
                   "GO" if ev_ok else "NO-GO"))
    verdict_go &= ev_ok

    # --- μ final dans [-0.5, +0.5] ---
    mu = data["mu"][-1] if data["mu"] else float("nan")
    mu_ok = mu == mu and abs(mu) <= 0.5
    mu_collapse = mu == mu and abs(mu) > 1.0
    checks.append(("μ final ∈ [-0.5,+0.5]", round(mu, 4) if mu == mu else "nan",
                   "GO" if mu_ok else ("COLLAPSE" if mu_collapse else "NO-GO")))
    verdict_go &= mu_ok

    # --- σ final > 0.1 ---
    sd = data["std"][-1] if data["std"] else float("nan")
    sd_ok = sd == sd and sd > 0.1
    checks.append(("σ final > 0.1", round(sd, 4) if sd == sd else "nan",
                   "GO" if sd_ok else "NO-GO(gel)"))
    verdict_go &= sd_ok

    # --- Early-stops < 30% des updates ---
    nupd = max(1, data["n_updates"])
    es_frac = data["early_stops"] / nupd
    es_ok = es_frac < 0.30
    checks.append(("Early-stops < 30%", f"{es_frac:.1%} ({data['early_stops']}/{nupd})",
                   "GO" if es_ok else "NO-GO"))
    verdict_go &= es_ok

    # --- Collapse: nB=0 persistant OU adv_nan persistant ---
    nB0_tail = lB and max(lB) == 0
    collapse = bool(nB0_tail) or data["adv_nan"] > 5
    checks.append(("Aucun collapse (nB=0 / adv=NaN)",
                   f"nB0_tail={bool(nB0_tail)} adv_nan={data['adv_nan']}",
                   "GO" if not collapse else "COLLAPSE"))
    verdict_go &= not collapse

    return {
        "finished": finished,
        "total_timesteps": ts,
        "verdict": ("GO" if verdict_go else "NO-GO"),
        "indeterminate": indeterminate and verdict_go,
        "checks": checks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="train log path")
    ap.add_argument("--target-steps", type=int, default=500000)
    args = ap.parse_args()
    path = Path(args.log)
    if not path.exists():
        # allow glob
        matches = sorted(Path(".").glob(args.log))
        if matches:
            path = matches[-1]
        else:
            print(f"log introuvable: {args.log}")
            sys.exit(2)

    data = parse_log(path)
    res = evaluate(data, args.target_steps)

    print("=" * 60)
    print(f"EVAL {path.name}")
    print(f"  total_timesteps={res['total_timesteps']} "
          f"(finished={res['finished']})  n_updates={data['n_updates']}")
    print("=" * 60)
    for name, val, status in res["checks"]:
        mark = {"GO": "✓", "NO-GO": "✗", "COLLAPSE": "💥"}.get(
            status, "✗") if not status.startswith("NO-GO") else "✗"
        print(f"  [{mark}] {name:42s} = {str(val):22s} -> {status}")
    print("-" * 60)
    if res["indeterminate"]:
        print("  VERDICT: EN COURS (métriques saines mais 500k pas atteints)")
        sys.exit(2)
    print(f"  VERDICT: {res['verdict']}")
    sys.exit(0 if res["verdict"] == "GO" else 1)


if __name__ == "__main__":
    main()
