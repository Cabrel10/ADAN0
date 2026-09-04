#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUDIT COMPARATIF ADAN — BTC vs DOGE
Version corrigée - regex robustes
"""

from __future__ import annotations

import re
import math
import statistics
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("/home/ubuntu/webapp/MORNINGSTAR/ADAN0")

RUNS = {
    "BTC": ROOT / "logs/v37_500k/btc_500k.log",
    "DOGE": ROOT / "logs/doge_500k/doge_500k.log",
}

# ============================================================
# UTILITAIRES
# ============================================================

def pct(a: float, b: float) -> float:
    return 100.0 * a / b if b else 0.0


def safe_float(x: str):
    try:
        return float(x)
    except Exception:
        return None


def section(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def subsection(title):
    print()
    print("-" * 90)
    print(title)
    print("-" * 90)


# ============================================================
# EXTRACTION SB3
# ============================================================

def extract_sb3_timesteps(text: str):
    vals = []
    patterns = [
        r"total_timesteps\s*\|\s*(\d+)",
        r"total_timesteps\s*[:=]\s*(\d+)",
        r"Total timesteps\s*[:=]\s*(\d+)",
    ]
    for pattern in patterns:
        vals.extend(int(x) for x in re.findall(pattern, text))
    return vals


# ============================================================
# EXTRACTION PPO
# ============================================================

def extract_ppo(text: str):
    keys = [
        "train/approx_kl",
        "train/clip_fraction",
        "train/entropy_loss",
        "train/explained_variance",
        "train/value_loss",
        "train/learning_rate",
        "train/policy_gradient_loss",
        "train/loss",
        "train/n_updates",
    ]
    result = {}
    for key in keys:
        escaped = re.escape(key)
        patterns = [
            rf"\|\s*{escaped}\s*\|\s*([-+0-9.eE]+)",
            rf"{escaped}\s*[:=]\s*([-+0-9.eE]+)",
            rf"'{escaped}'\s*:\s*([-+0-9.eE]+)",
            rf'"{escaped}"\s*:\s*([-+0-9.eE]+)',
        ]
        values = []
        for pattern in patterns:
            for value in re.findall(pattern, text):
                f = safe_float(value)
                if f is not None:
                    values.append(f)
        if values:
            result[key] = values
    return result


# ============================================================
# ACTION_DIFF — CORRIGÉE
# ============================================================

# On suppose une ligne comme : (ACTION_DIFF) Step 123 Requested=BUY Executed=SELL
ACTION_DIFF_RE = re.compile(
    r"\(ACTION_DIFF\)\s*Step\s+(?P<step>\d+)\s+Requested=(?P<requested>[A-Za-z_]+)\s+Executed=(?P<executed>[A-Za-z_]+)"
)

def extract_action_diff(text: str):
    events = []
    for m in ACTION_DIFF_RE.finditer(text):
        events.append({
            "step": int(m.group("step")),
            "requested": m.group("requested").upper(),
            "executed": m.group("executed").upper(),
        })
    return events


# ============================================================
# TARGET_WEIGHT — CORRIGÉE
# ============================================================

# On suppose une ligne comme : (TARGET_WEIGHT) Step 123 Asset=BTC Action=BUY Raw=0.25
TARGET_RE = re.compile(
    r"\(TARGET_WEIGHT\)\s*Step\s+(?P<step>\d+)\s+Asset=(?P<asset>[A-Za-z0-9_]+)\s+Action=(?P<action>[A-Za-z_]+)\s+Raw=(?P<raw>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)

def extract_target_weight(text: str):
    events = []
    for m in TARGET_RE.finditer(text):
        events.append({
            "step": int(m.group("step")),
            "asset": m.group("asset"),
            "action": m.group("action").upper(),
            "raw": float(m.group("raw")),
        })
    return events


# ============================================================
# PIPELINE
# ============================================================

def extract_pipeline(text: str):
    matches = re.findall(r"pipeline=\{([^{}]*)\}", text)
    snapshots = []
    for raw in matches:
        snap = {}
        for key, value in re.findall(r"'([^']+)'\s*:\s*(-?\d+)", raw):
            snap[key] = int(value)
        if snap:
            snapshots.append(snap)
    return snapshots


# ============================================================
# BUDGET
# ============================================================

def extract_budget(text: str):
    vals = []
    patterns = [
        r"budget=([+-]?(?:\d+(?:\.\d*)?|\.\d+))/",
        r"budget=([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
    ]
    for pattern in patterns:
        vals.extend(float(x) for x in re.findall(pattern, text))
    return vals


# ============================================================
# CAPITAL
# ============================================================

def extract_capital(text: str):
    patterns = [
        r"Portfolio value:\s*\$?([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
        r"Portfolio value\s*[:=]\s*\$?([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
    ]
    vals = []
    for pattern in patterns:
        vals.extend(float(x) for x in re.findall(pattern, text))
    return vals


# ============================================================
# PNL
# ============================================================

def extract_pnl(text: str):
    patterns = [
        r"PnL:\s*\$([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
        r"PnL\s*[:=]\s*\$?([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
    ]
    vals = []
    for pattern in patterns:
        vals.extend(float(x) for x in re.findall(pattern, text))
    return vals


# ============================================================
# POSITIONS
# ============================================================

def extract_close_reasons(text: str):
    patterns = [
        r"\(POSITION FERM[ÉE]E\)[^\n]*Raison:\s*([^\s|]+)",
        r"\(POSITION FERM[ÉE]E\)[^\n]*Reason:\s*([^\s|]+)",
    ]
    reasons = []
    for pattern in patterns:
        reasons.extend(re.findall(pattern, text))
    return Counter(reasons)


def count_literal(text: str, token: str):
    return text.count(token)


# ============================================================
# RAW DISTRIBUTION
# ============================================================

def classify_raw(x):
    if x > 0.10:
        return "BUY"
    if x < -0.10:
        return "SELL"
    return "HOLD"


def raw_statistics(target_events):
    if not target_events:
        return None
    values = [e["raw"] for e in target_events]
    counts = Counter(classify_raw(x) for x in values)
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) >= 2 else 0.0,
        "buy": counts["BUY"],
        "sell": counts["SELL"],
        "hold": counts["HOLD"],
    }


def raw_by_step_window(target_events, window=10000):
    result = defaultdict(list)
    for e in target_events:
        w = (e["step"] // window) * window
        result[w].append(e["raw"])
    return result


# ============================================================
# COHÉRENCE DES COMPTEURS
# ============================================================

def compare_counters(sb3_ts, action_events, target_events):
    result = {}
    if sb3_ts and action_events:
        result["action_diff_per_sb3"] = len(action_events) / sb3_ts
    if sb3_ts and target_events:
        result["target_per_sb3"] = len(target_events) / sb3_ts
    if action_events and target_events:
        result["target_per_action"] = len(target_events) / len(action_events)
    return result


# ============================================================
# ANALYSE ACTION_DIFF
# ============================================================

def analyze_action_diff(events):
    if not events:
        return {}
    pairs = Counter((e["requested"], e["executed"]) for e in events)
    requested = Counter(e["requested"] for e in events)
    executed = Counter(e["executed"] for e in events)
    exact = sum(count for (req, exe), count in pairs.items() if req == exe)
    return {
        "pairs": pairs,
        "requested": requested,
        "executed": executed,
        "exact_match": exact,
        "exact_match_pct": pct(exact, len(events)),
    }


# ============================================================
# ANALYSE PNL
# ============================================================

def analyze_pnl(pnls):
    if not pnls:
        return {}
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    zeros = [x for x in pnls if x == 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0)
    return {
        "n": len(pnls),
        "wins": len(wins),
        "losses": len(losses),
        "zeros": len(zeros),
        "gross_win": gross_win,
        "gross_loss": gross_loss,
        "net": sum(pnls),
        "pf": pf,
        "mean": statistics.fmean(pnls),
        "min": min(pnls),
        "max": max(pnls),
    }


# ============================================================
# CHARGEMENT
# ============================================================

def load_run(name, path):
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(errors="ignore")

    sb3_ts_values = extract_sb3_timesteps(text)
    action_events = extract_action_diff(text)
    target_events = extract_target_weight(text)
    pipeline = extract_pipeline(text)
    budgets = extract_budget(text)
    capital = extract_capital(text)
    pnls = extract_pnl(text)
    ppo = extract_ppo(text)

    return {
        "name": name,
        "path": path,
        "lines": text.count("\n") + 1,
        "sb3_ts_values": sb3_ts_values,
        "sb3_ts_max": max(sb3_ts_values) if sb3_ts_values else None,
        "action_events": action_events,
        "target_events": target_events,
        "pipeline": pipeline,
        "budgets": budgets,
        "capital": capital,
        "pnls": pnls,
        "ppo": ppo,
        "action_analysis": analyze_action_diff(action_events),
        "raw_stats": raw_statistics(target_events),
        "raw_windows": raw_by_step_window(target_events),
        "pnl_analysis": analyze_pnl(pnls),
        "close_reasons": extract_close_reasons(text),
        "trades_opened": count_literal(text, "[POSITION OUVERTE]"),
        "drawdown_kill": count_literal(text, "DRAWDOWN_KILL"),
        "counter_ratios": compare_counters(
            max(sb3_ts_values) if sb3_ts_values else None,
            action_events,
            target_events
        ),
    }


# ============================================================
# AFFICHAGE (inchangé, mais adapté si des champs manquent)
# ============================================================

def print_run(d):
    section(f"{d['name']} — DONNÉES BRUTES")
    print(f"Fichier                  : {d['path']}")
    print(f"Lignes                   : {d['lines']}")
    print()
    print("SB3 timesteps trouvés    :", d["sb3_ts_values"])
    print("SB3 max                  :", d["sb3_ts_max"])
    print()
    print("ACTION_DIFF événements   :", len(d["action_events"]))
    print("TARGET_WEIGHT événements :", len(d["target_events"]))
    print()
    print("ATTENTION :")
    print("  ACTION_DIFF != automatiquement nombre de décisions PPO.")
    print("  TARGET_WEIGHT != automatiquement nombre de décisions PPO.")
    print("  Les compteurs pipeline ne sont pas convertis en %")
    print("  tant que leur unité temporelle n'est pas démontrée.")

    subsection("Ratios d'observation — PAS des taux de rejet")
    for k, v in d["counter_ratios"].items():
        print(f"{k:30s}: {v:.8f}")

    subsection("ACTION_DIFF — Requested vs Executed")
    aa = d["action_analysis"]
    if not aa:
        print("Aucun ACTION_DIFF exploitable.")
    else:
        print(f"Événements               : {len(d['action_events'])}")
        print(f"Exact match               : {aa['exact_match']} ({aa['exact_match_pct']:.2f}%)")
        print()
        print("Matrice complète :")
        for (req, exe), n in aa["pairs"].most_common():
            print(f"  {req:8s} -> {exe:8s} : {n:7d} ({pct(n, len(d['action_events'])):.2f}%)")

    subsection("TARGET_WEIGHT / Raw")
    rs = d["raw_stats"]
    if not rs:
        print("Aucun TARGET_WEIGHT exploitable.")
    else:
        print(f"n                        : {rs['n']}")
        print(f"min                      : {rs['min']:.6f}")
        print(f"max                      : {rs['max']:.6f}")
        print(f"mean                     : {rs['mean']:.6f}")
        print(f"median                   : {rs['median']:.6f}")
        print(f"stdev                    : {rs['stdev']:.6f}")
        print()
        print(f"BUY                      : {rs['buy']} ({pct(rs['buy'], rs['n']):.2f}%)")
        print(f"SELL                     : {rs['sell']} ({pct(rs['sell'], rs['n']):.2f}%)")
        print(f"HOLD                     : {rs['hold']} ({pct(rs['hold'], rs['n']):.2f}%)")

    subsection("Raw par fenêtre de 10 000 — descriptif uniquement")
    if not d["raw_windows"]:
        print("Aucune donnée.")
    else:
        for w in sorted(d["raw_windows"]):
            vals = d["raw_windows"][w]
            c = Counter(classify_raw(x) for x in vals)
            print(
                f"{w:8d} | n={len(vals):6d} | "
                f"BUY={pct(c['BUY'], len(vals)):6.2f}% | "
                f"SELL={pct(c['SELL'], len(vals)):6.2f}% | "
                f"HOLD={pct(c['HOLD'], len(vals)):6.2f}% | "
                f"mean={statistics.fmean(vals):+.5f}"
            )

    subsection("PIPELINE — snapshots")
    if not d["pipeline"]:
        print("Aucun snapshot pipeline trouvé.")
    else:
        print(f"Nombre de snapshots      : {len(d['pipeline'])}")
        first = d["pipeline"][0]
        last = d["pipeline"][-1]
        print()
        print("Premier snapshot :")
        for k, v in first.items():
            print(f"  {k:25s} {v}")
        print()
        print("Dernier snapshot :")
        for k, v in last.items():
            print(f"  {k:25s} {v}")
        print()
        print("Évolution premier -> dernier :")
        keys = sorted(set(first) | set(last))
        for k in keys:
            a = first.get(k, 0)
            b = last.get(k, 0)
            print(f"  {k:25s} {a:10d} -> {b:10d} (delta={b-a:+d})")

    subsection("PPO — valeurs effectivement trouvées")
    if not d["ppo"]:
        print("Aucune métrique PPO reconnue par les formats de recherche.")
        print("Cela ne signifie PAS qu'elles sont absentes du training.")
    else:
        for key, values in d["ppo"].items():
            print(f"{key:32s} n={len(values):4d} first={values[0]:.8g} last={values[-1]:.8g}")

    subsection("BUDGET")
    if not d["budgets"]:
        print("Aucune valeur budget détectée.")
    else:
        zeros = sum(1 for x in d["budgets"] if abs(x) < 1e-12)
        print(f"n                        : {len(d['budgets'])}")
        print(f"min                      : {min(d['budgets']):.6f}")
        print(f"max                      : {max(d['budgets']):.6f}")
        print(f"mean                     : {statistics.fmean(d['budgets']):.6f}")
        print(f"zeros                    : {zeros} ({pct(zeros, len(d['budgets'])):.2f}%)")

    subsection("CAPITAL")
    if not d["capital"]:
        print("Aucune valeur de capital détectée.")
    else:
        cap = d["capital"]
        print(f"n                        : {len(cap)}")
        print(f"first                    : {cap[0]:.6f}")
        print(f"last                     : {cap[-1]:.6f}")
        print(f"min                      : {min(cap):.6f}")
        print(f"max / peak               : {max(cap):.6f}")

    subsection("POSITIONS / FERMETURES")
    print(f"Positions ouvertes       : {d['trades_opened']}")
    print(f"DRAWDOWN_KILL            : {d['drawdown_kill']}")
    if d["close_reasons"]:
        print()
        print("Raisons de fermeture :")
        for reason, count in d["close_reasons"].most_common():
            print(f"  {reason:30s} {count}")

    subsection("PnL — parser corrigé")
    pa = d["pnl_analysis"]
    if not pa:
        print("Aucun PnL reconnu.")
    else:
        print(f"n                        : {pa['n']}")
        print(f"wins                     : {pa['wins']}")
        print(f"losses                   : {pa['losses']}")
        print(f"zeros                    : {pa['zeros']}")
        print(f"gross win                : {pa['gross_win']:.8f}")
        print(f"gross loss               : {pa['gross_loss']:.8f}")
        print(f"net                      : {pa['net']:.8f}")
        if math.isinf(pa["pf"]):
            print("profit factor            : INF")
        else:
            print(f"profit factor            : {pa['pf']:.8f}")
        print(f"mean PnL                 : {pa['mean']:.8f}")
        print(f"min PnL                  : {pa['min']:.8f}")
        print(f"max PnL                  : {pa['max']:.8f}")


# ============================================================
# COMPARAISON
# ============================================================

def compare_runs(btc, doge):
    section("COMPARAISON BTC vs DOGE — UNIQUEMENT DONNÉES COMPARABLES")
    rows = [
        ("SB3 timesteps", btc["sb3_ts_max"], doge["sb3_ts_max"]),
        ("ACTION_DIFF événements", len(btc["action_events"]), len(doge["action_events"])),
        ("TARGET_WEIGHT événements", len(btc["target_events"]), len(doge["target_events"])),
        ("positions ouvertes", btc["trades_opened"], doge["trades_opened"]),
        ("DRAWDOWN_KILL", btc["drawdown_kill"], doge["drawdown_kill"]),
    ]
    for name, a, b in rows:
        print(f"{name:35s} | BTC={str(a):>12s} | DOGE={str(b):>12s}")

    subsection("Capital")
    for name, d in [("BTC", btc), ("DOGE", doge)]:
        if not d["capital"]:
            print(f"{name}: N/A")
            continue
        cap = d["capital"]
        print(f"{name:5s} first={cap[0]:.4f} last={cap[-1]:.4f} min={min(cap):.4f} peak={max(cap):.4f} delta_peak={max(cap)-cap[0]:+.4f}")

    subsection("Raw action — comparaison descriptive")
    for name, d in [("BTC", btc), ("DOGE", doge)]:
        rs = d["raw_stats"]
        if not rs:
            print(f"{name}: N/A")
            continue
        print(f"{name:5s} n={rs['n']:6d} BUY={pct(rs['buy'],rs['n']):6.2f}% SELL={pct(rs['sell'],rs['n']):6.2f}% HOLD={pct(rs['hold'],rs['n']):6.2f}% mean={rs['mean']:+.5f}")

    subsection("Requested -> Executed")
    for name, d in [("BTC", btc), ("DOGE", doge)]:
        aa = d["action_analysis"]
        if not aa:
            print(f"{name}: N/A")
            continue
        print(f"{name:5s} exact_match={aa['exact_match_pct']:.2f}%")
        for (req, exe), n in aa["pairs"].most_common():
            print(f"       {req:8s} -> {exe:8s} {n:7d} ({pct(n,len(d['action_events'])):.2f}%)")

    subsection("PnL")
    for name, d in [("BTC", btc), ("DOGE", doge)]:
        pa = d["pnl_analysis"]
        if not pa:
            print(f"{name}: N/A")
            continue
        print(f"{name:5s} n={pa['n']} wins={pa['wins']} losses={pa['losses']} zeros={pa['zeros']} net={pa['net']:+.6f} PF={pa['pf']}")

    subsection("Pipeline — NE PAS interpréter comme taux")
    print("Les valeurs ci-dessous sont affichées comme compteurs bruts.")
    print("Aucun pourcentage routing_reject/policy n'est calculé.")
    for name, d in [("BTC", btc), ("DOGE", doge)]:
        if not d["pipeline"]:
            print(f"{name}: aucun snapshot")
            continue
        p = d["pipeline"][-1]
        print()
        print(name)
        for key in ["policy", "deadband_reject", "routing_reject", "budget_insufficient",
                    "close_gap_active", "daily_close_quota", "below_break_even",
                    "portfolio_reject", "trade_executed"]:
            if key in p:
                print(f"  {key:25s} {p[key]:10d}")


# ============================================================
# DIAGNOSTIC
# ============================================================

def diagnostic(btc, doge):
    section("DIAGNOSTIC AUTOMATIQUE — NIVEAU DE CERTITUDE")
    conclusions = []

    for name, d in [("BTC", btc), ("DOGE", doge)]:
        if d["capital"]:
            peak = max(d["capital"])
            if peak < 21.0:
                conclusions.append(f"{name}: peak capital observé < 21 USDT.")
            else:
                conclusions.append(f"{name}: peak capital observé >= 21 USDT.")

    if btc["raw_stats"] and doge["raw_stats"]:
        btc_sell = pct(btc["raw_stats"]["sell"], btc["raw_stats"]["n"])
        doge_sell = pct(doge["raw_stats"]["sell"], doge["raw_stats"]["n"])
        if btc_sell > 80 and doge_sell > 80:
            conclusions.append("BTC et DOGE montrent tous deux une forte proportion de Raw SELL dans les TARGET_WEIGHT capturés.")
            conclusions.append("Cela constitue un signal commun dans le log, mais NE prouve PAS à lui seul que la policy PPO a choisi SELL à chaque timestep.")

    for name, d in [("BTC", btc), ("DOGE", doge)]:
        aa = d["action_analysis"]
        if aa and aa["exact_match_pct"] > 90:
            conclusions.append(f"{name}: ACTION_DIFF exact-match > 90%, mais cette métrique est dominée par les événements présents dans ACTION_DIFF et ne doit pas être interprétée comme fidélité globale de la policy.")

    conclusions.append("Les compteurs pipeline ne sont pas convertis en pourcentages car leur unité temporelle doit être démontrée avant division par SB3 total_timesteps.")

    if not btc["ppo"] or not doge["ppo"]:
        conclusions.append("Les métriques PPO ne sont pas encore suffisamment extraites des logs pour permettre un diagnostic PPO.")

    for name, d in [("BTC", btc), ("DOGE", doge)]:
        pa = d["pnl_analysis"]
        if pa and pa["wins"] == 0 and pa["losses"] > 0:
            conclusions.append(f"{name}: le parser trouve 0 gain positif parmi {pa['n']} PnL. À vérifier contre le format réel des lignes de fermeture avant d'utiliser ce résultat comme preuve économique.")

    for c in conclusions:
        print("•", c)


# ============================================================
# MAIN
# ============================================================

def main():
    section("ADAN — AUDIT COMPARATIF BTC / DOGE")
    print("Racine :", ROOT)
    print()
    print("Runs analysés :")
    for name, path in RUNS.items():
        print(f"  {name:5s} -> {path}")

    data = {}
    for name, path in RUNS.items():
        try:
            data[name] = load_run(name, path)
        except Exception as e:
            print()
            print(f"[ERREUR] {name}: {e}")

    if "BTC" not in data or "DOGE" not in data:
        raise SystemExit("Impossible d'analyser BTC et DOGE simultanément.")

    btc = data["BTC"]
    doge = data["DOGE"]

    print_run(btc)
    print_run(doge)
    compare_runs(btc, doge)
    diagnostic(btc, doge)

    section("FIN DE L'AUDIT")
    print("Aucune modification du code ADAN n'a été effectuée.")
    print()
    print("IMPORTANT : ce rapport est descriptif. Il ne modifie ni reward, ni PPO, ni routing.")


if __name__ == "__main__":
    main()
