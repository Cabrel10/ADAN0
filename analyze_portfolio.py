#!/usr/bin/env python3
"""Analyse du portefeuille ADAN V29 sandbox depuis la trace action_pipeline JSONL."""
import json
import sys
from collections import Counter, defaultdict
import statistics

PATH = "logs/action_pipeline/v29_500k_sandbox_w0.jsonl"

def main():
    # Accumulateurs
    opened_notional = []          # tailles acceptées (open)
    rejected_size = defaultdict(list)  # tailles demandées par type de rejet
    close_reasons = Counter()     # TP / SL / agent_close / MaxDuration
    close_pnl = defaultdict(list)
    illegal_routing = Counter()   # sell_while_flat / buy_while_long
    deadband = 0
    barriers = Counter()          # fee_gate, hysteresis, budget, cooldown, pm, risk
    policy_total = 0
    capitals = []

    with open(PATH, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            stage = d.get("stage")
            reason = d.get("reason", "")
            lifec = d.get("lifecycle_event")

            if stage == "policy":
                policy_total += 1
            elif stage == "routing_reject":
                illegal_routing[reason] += 1
            elif stage == "deadband_reject":
                deadband += 1
            elif stage in ("barrier_reject", "budget_reject", "portfolio_reject", "risk_gate", "pm_rejected"):
                barriers[stage + "/" + reason] += 1
                # proxy de taille demandée : size_raw si présent, sinon |action_in|
                raw = d.get("size_raw")
                if raw is None:
                    raw = d.get("action_in")
                if raw is not None:
                    try:
                        rejected_size[stage + "/" + reason].append(abs(float(raw)))
                    except (TypeError, ValueError):
                        pass
            elif stage == "trade_executed":
                if lifec == "open":
                    n = d.get("notional_usd") or d.get("notional")
                    if n is None and d.get("size_raw") is not None:
                        n = d.get("size_raw")
                    if n is not None:
                        try:
                            opened_notional.append(abs(float(n)))
                        except (TypeError, ValueError):
                            pass
                elif lifec == "close":
                    close_reasons[reason] += 1
                    pnl = d.get("pnl_net") or d.get("pnl")
                    if pnl is not None:
                        try:
                            close_pnl[reason].append(float(pnl))
                        except (TypeError, ValueError):
                            pass

            cap = d.get("capital_after") or d.get("capital")
            if cap is not None:
                try:
                    capitals.append(float(cap))
                except (TypeError, ValueError):
                    pass

    def stats(xs):
        if not xs:
            return "n/a"
        return f"n={len(xs)} moy={statistics.mean(xs):.2f} med={statistics.median(xs):.2f} min={min(xs):.2f} max={max(xs):.2f}"

    print("=" * 60)
    print("=== ADAN V29 SANDBOX — METRIQUES PORTEFEUILLE ===")
    print("=" * 60)
    print(f"\nDecisions policy total: {policy_total}")
    print(f"\n--- TAILLE POSITIONS ACCEPTEES (open) ---")
    print(f"  {stats(opened_notional)}")
    print(f"\n--- TAILLE DEMANDEE par type de rejet (proxy size_raw) ---")
    for stage, xs in sorted(rejected_size.items()):
        print(f"  {stage}: {stats(xs)}")

    print(f"\n--- ISSUES DES TRADES (close) ---")
    total_close = sum(close_reasons.values())
    for r, c in close_reasons.most_common():
        pnls = close_pnl.get(r, [])
        pnl_str = f"pnl_moy={statistics.mean(pnls):+.4f}" if pnls else "pnl n/a"
        pct = (c / total_close * 100) if total_close else 0
        print(f"  {r:25s}: {c:4d} ({pct:5.1f}%)  {pnl_str}")

    print(f"\n--- ACTIONS ILLEGALES / ROUTING NEUTRALISE ---")
    for r, c in illegal_routing.most_common():
        print(f"  {r:25s}: {c:4d}")
    print(f"  deadband (signal faible)  : {deadband:4d}")

    print(f"\n--- BARRIERES ECONOMIQUES/OPERATIONNELLES ---")
    for k, c in barriers.most_common():
        print(f"  {k:40s}: {c:4d}")

    print(f"\n--- CAPITAL ---")
    if capitals:
        print(f"  dernier capital: ${capitals[-1]:.2f}  (min ${min(capitals):.2f}, max ${max(capitals):.2f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
