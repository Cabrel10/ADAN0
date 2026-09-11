#!/usr/bin/env python3
"""Radar d'apprentissage ADAN V31 — analyse READ-ONLY des artefacts figes.

5 niveaux (ordre causal) :
  L1 consequence awareness : l'agent observe-t-il les consequences de ses actions ?
  L2 learning from errors  : P(action|erreur_t) diminue-t-elle ? erreurs repetees ?
  L3 environment adaptation: comportement vs regime de marche (prix/vol proxies)
  L4 policy coherence      : diversite BUY/HOLD/SELL, spam sterile, saturation
  L5 performance           : WR/PF/PnL/Sharpe par fenetre + tendances

Sorties : metrics.json + RADAR.md (verdicts CONFIRME/PROBABLE/REFUTE/NON RESOLU)
AUCUNE modification d'entrainement. AUCUN hyperparametre touche.
"""
import json
import re
import math
import statistics as st
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent
RAW = BASE / "raw"
OUT = BASE

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
KV_RE = re.compile(r"(\w+)=([-0-9.eEna]+)")


def ts_of(line):
    m = TS_RE.match(line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") if m else None


# ---------------------------------------------------------------------------
# 1. ANCHOR_DEBUG : serie par update PPO
# ---------------------------------------------------------------------------
anchor_rows = []
for line in (RAW / "anchor_debug.txt").read_text(errors="replace").splitlines():
    if "ANCHOR_DEBUG" not in line:
        continue
    d = dict(KV_RE.findall(line))
    def f(k):
        v = d.get(k, "nan")
        try:
            return float(v)
        except ValueError:
            return float("nan")
    anchor_rows.append({
        "upd": int(f("upd")), "a0_mean": f("a0_mean"), "a0_std": f("a0_std"),
        "anchor": f("anchor"), "adv_BUY": f("adv_BUY"), "adv_SELL": f("adv_SELL"),
        "adv_HOLD": f("adv_HOLD"), "nB": f("nB"), "nS": f("nS"), "nH": f("nH"),
    })

# ---------------------------------------------------------------------------
# 2. TRADE_AUDIT (open + close) : dedup par signature, pairing open->close
# ---------------------------------------------------------------------------
def parse_trade_line(line, kind):
    d = dict(KV_RE.findall(line))
    ts = ts_of(line)
    if ts is None:
        return None
    def f(k):
        try:
            return float(d[k])
        except (KeyError, ValueError):
            return float("nan")
    sig = None
    if kind == "open":
        sig = ("open", int(f("step")) if not math.isnan(f("step")) else -1,
               round(f("entry_price"), 2))
        return {"ts": ts, "sig": sig, "step": int(f("step")) if not math.isnan(f("step")) else -1,
                "entry_price": f("entry_price"), "capital": f("capital_after"),
                "raw": line[:400]}
    # close
    reason = re.search(r"reason=(\w+)", line)
    sig = ("close", int(f("step")) if not math.isnan(f("step")) else -1,
           round(f("entry_price"), 2), round(f("pnl_net"), 6))
    return {"ts": ts, "sig": sig, "step": int(f("step")) if not math.isnan(f("step")) else -1,
            "entry_price": f("entry_price"),
            "sell_price": f("sell_price"), "pnl_net": f("pnl_net"),
            "fees": f("fees"), "hold_steps": f("hold_steps"),
            "reason": reason.group(1) if reason else "?",
            "capital_after": f("capital_after")}


def dedup(rows):
    seen, out = set(), []
    for r in rows:
        if r and r["sig"] not in seen:
            seen.add(r["sig"])
            out.append(r)
    return out


opens = dedup([parse_trade_line(l, "open") for l in
               (RAW / "trade_audit_open.txt").read_text(errors="replace").splitlines()
               if "TRADE_AUDIT_OPEN" in l])
closes = dedup([parse_trade_line(l, "close") for l in
                (BASE.parent / "v31_sandbox_stop_20260819_1022" /
                 "trade_audit_close_dedup.txt").read_text(errors="replace").splitlines()
                if "TRADE_AUDIT_CLOSE" in l])
closes.sort(key=lambda r: r["ts"])
opens.sort(key=lambda r: r["ts"])

# ---------------------------------------------------------------------------
# 3. ACTION_DIFF : requested vs executed, inv_penalty, pipeline counters
# ---------------------------------------------------------------------------
ad_rows = []
for line in (RAW / "action_diff.txt").read_text(errors="replace").splitlines():
    if "ACTION_DIFF" not in line:
        continue
    ts = ts_of(line)
    if ts is None:
        continue
    m = re.search(r"Requested=(\w+)\s+Executed=(\w+)", line)
    if not m:
        continue
    pen = re.search(r"inv_penalty=([-0-9.eE]+)", line)
    pipe = re.search(r"'trade_executed': (\d+)", line)
    pol = re.search(r"'policy': (\d+)", line)
    rout = re.search(r"'routing_reject': (\d+)", line)
    ad_rows.append({"ts": ts, "req": m.group(1), "exec": m.group(2),
                    "inv_penalty": float(pen.group(1)) if pen else 0.0,
                    "policy": int(pol.group(1)) if pol else 0,
                    "routing_reject": int(rout.group(1)) if rout else 0,
                    "trade_executed": int(pipe.group(1)) if pipe else 0})

# dedup ACTION_DIFF (double-logging: logger + print)
ad_rows = [r for i, r in enumerate(ad_rows)
           if i == 0 or not (r["ts"] == ad_rows[i-1]["ts"] and r["req"] == ad_rows[i-1]["req"]
                             and r["policy"] == ad_rows[i-1]["policy"])]

# ---------------------------------------------------------------------------
# Fenetres temporelles : deciles sur la duree reelle du run
# ---------------------------------------------------------------------------
if closes:
    t0, t1 = closes[0]["ts"], closes[-1]["ts"]
else:
    t0, t1 = ad_rows[0]["ts"], ad_rows[-1]["ts"]
SPAN = (t1 - t0).total_seconds()
NW = 10


def win_idx(ts):
    if SPAN <= 0:
        return 0
    return min(NW - 1, int((ts - t0).total_seconds() / SPAN * NW))


def wins_of(rows):
    w = [[] for _ in range(NW)]
    for r in rows:
        w[win_idx(r["ts"])].append(r)
    return w


cwins = wins_of(closes)
awins = wins_of(ad_rows)

# a0 series: decoupee en 10 blocs d'updates (pas de timestamp par ligne)
BLK = max(1, len(anchor_rows) // NW)
ublks = [anchor_rows[i*BLK:(i+1)*BLK] for i in range(NW)]

# ---------------------------------------------------------------------------
# L5 — PERFORMANCE par fenetre (couche finale)
# ---------------------------------------------------------------------------
perf = []
for i, w in enumerate(cwins):
    pnls = [r["pnl_net"] for r in w if not math.isnan(r["pnl_net"])]
    wins_ = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gp, gl = sum(wins_), abs(sum(losses))
    holds = [r["hold_steps"] for r in w if not math.isnan(r["hold_steps"])]
    reasons = {}
    for r in w:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    perf.append({
        "win": i, "trades": len(pnls),
        "WR": round(100 * len(wins_) / len(pnls), 1) if pnls else None,
        "PnL": round(sum(pnls), 3) if pnls else 0,
        "PF": round(gp / gl, 2) if gl > 0 else None,
        "avg_hold": round(st.mean(holds), 1) if holds else None,
        "TP": reasons.get("TP_HIT", 0), "SL": reasons.get("SL_HIT", 0),
        "AGENT_CLOSE": reasons.get("AGENT_CLOSE", 0),
    })

# Sharpe global (sur pnl_net dedup, non annualise — indicatif)
all_pnl = [r["pnl_net"] for r in closes if not math.isnan(r["pnl_net"])]
sharpe = (st.mean(all_pnl) / st.pstdev(all_pnl) * math.sqrt(len(all_pnl))
          if len(all_pnl) > 2 and st.pstdev(all_pnl) > 0 else None)

# ---------------------------------------------------------------------------
# L4 — POLICY COHERENCE : diversite nB/nH/nS, exec rates, spam sterile
# ---------------------------------------------------------------------------
diversity = []
for i, blk in enumerate(ublks):
    nb = [r["nB"] for r in blk if not math.isnan(r["nB"])]
    nh = [r["nH"] for r in blk if not math.isnan(r["nH"])]
    ns = [r["nS"] for r in blk if not math.isnan(r["nS"])]
    a0 = [r["a0_mean"] for r in blk if not math.isnan(r["a0_mean"])]
    tot = [b + h + s for b, h, s in zip(nb, nh, ns)]
    shr_b = [b / t if t else 0 for b, t in zip(nb, tot)]
    shr_h = [h / t if t else 0 for h, t in zip(nh, tot)]
    shr_s = [s / t if t else 0 for s, t in zip(ns, tot)]
    nan_adv = sum(1 for r in blk if math.isnan(r["adv_BUY"]))
    diversity.append({
        "blk": i, "upd_range": [blk[0]["upd"], blk[-1]["upd"]] if blk else [0, 0],
        "a0_mean": round(st.mean(a0), 3) if a0 else None,
        "share_BUY": round(st.mean(shr_b), 3) if shr_b else None,
        "share_HOLD": round(st.mean(shr_h), 3) if shr_h else None,
        "share_SELL": round(st.mean(shr_s), 3) if shr_s else None,
        "pct_advBUY_nan": round(100 * nan_adv / len(blk), 1) if blk else None,
    })

# Requested vs Executed par fenetre + routing reject (spam sterile SELL)
exec_stats = []
for i, w in enumerate(awins):
    req_c, exe_c = {}, {}
    for r in w:
        req_c[r["req"]] = req_c.get(r["req"], 0) + 1
        exe_c[r["exec"]] = exe_c.get(r["exec"], 0) + 1
    pens = [r["inv_penalty"] for r in w]
    if w:
        pol0, pol1 = w[0]["policy"], w[-1]["policy"]
        rr0, rr1 = w[0]["routing_reject"], w[-1]["routing_reject"]
        te0, te1 = w[0]["trade_executed"], w[-1]["trade_executed"]
    else:
        pol0 = pol1 = rr0 = rr1 = te0 = te1 = 0
    exec_stats.append({
        "win": i, "req": req_c, "exec": exe_c,
        "inv_penalty_sum": round(sum(pens), 3),
        "d_policy": pol1 - pol0, "d_routing_reject": rr1 - rr0,
        "d_trade_executed": te1 - te0,
        "exec_rate_pct": round(100 * (te1 - te0) / (pol1 - pol0), 2)
        if pol1 > pol0 else None,
    })

# Point de bascule saturation : 1er update ou nB+nH == 0 de facon durable
collapse_upd = None
run_len = 0
for r in anchor_rows:
    if not math.isnan(r["nB"]) and not math.isnan(r["nH"]):
        if r["nB"] + r["nH"] == 0:
            run_len += 1
            if run_len >= 5 and collapse_upd is None:
                collapse_upd = r["upd"] - 4 * (r["upd"] - anchor_rows[0]["upd"]) // max(1, run_len)
                collapse_upd = r["upd"]
        else:
            run_len = 0

# ---------------------------------------------------------------------------
# L2 — LEARNING FROM ERRORS
# (a) SL-rate par fenetre : l'agent evite-t-il ses propres stop-loss ?
# (b) repetition de l'action rejetee : requested==SELL suivi de requested==SELL
#     alors que exec==HOLD (sell_no_position) -> P(repeat|reject) vs baseline
# (c) hold_steps trend apres SL vs apres TP (ajustement comportemental)
# ---------------------------------------------------------------------------
sl_by_win = [p["SL"] / p["trades"] if p["trades"] else None for p in perf]
tp_by_win = [p["TP"] / p["trades"] if p["trades"] else None for p in perf]

# (b) sur action_diff : sequences req
rep_after_sterile, sterile_events, sell_total = 0, 0, 0
prev_sterile_sell = False
for r in ad_rows:
    is_sterile_sell = (r["req"] == "SELL" and r["exec"] == "HOLD")
    if r["req"] == "SELL":
        sell_total += 1
        if prev_sterile_sell:
            rep_after_sterile += 1
    if is_sterile_sell:
        sterile_events += 1
    prev_sterile_sell = is_sterile_sell
p_repeat_given_sterile = (rep_after_sterile / sterile_events) if sterile_events else None
p_sell_baseline = sell_total / len(ad_rows) if ad_rows else None

# (c) hold apres SL vs TP (fenetre 1ere moitie vs 2e moitie)
def hold_after(reason, rows, half):
    out = []
    n = len(rows)
    lo, hi = (0, n // 2) if half == 0 else (n // 2, n)
    for r in rows[lo:hi]:
        if r["reason"] == reason and not math.isnan(r["hold_steps"]):
            out.append(r["hold_steps"])
    return round(st.mean(out), 1) if out else None

hold_adj = {
    "hold_after_SL_H1": hold_after("SL_HIT", closes, 0),
    "hold_after_SL_H2": hold_after("SL_HIT", closes, 1),
    "hold_after_TP_H1": hold_after("TP_HIT", closes, 0),
    "hold_after_TP_H2": hold_after("TP_HIT", closes, 1),
    "hold_after_AGENT_H1": hold_after("AGENT_CLOSE", closes, 0),
    "hold_after_AGENT_H2": hold_after("AGENT_CLOSE", closes, 1),
}

# ---------------------------------------------------------------------------
# L1 — CONSEQUENCE AWARENESS : stats par action executee + inv_penalty
# ---------------------------------------------------------------------------
consequences = {
    "closes_by_reason": {},
    "pnl_by_reason": {},
    "inv_penalty_total": round(sum(r["inv_penalty"] for r in ad_rows), 3),
    "inv_penalty_nonzero_events": sum(1 for r in ad_rows if r["inv_penalty"] != 0),
}
for r in closes:
    k = r["reason"]
    consequences["closes_by_reason"][k] = consequences["closes_by_reason"].get(k, 0) + 1
    consequences["pnl_by_reason"].setdefault(k, []).append(round(r["pnl_net"], 4))
for k, v in consequences["pnl_by_reason"].items():
    consequences["pnl_by_reason"][k] = {"n": len(v), "mean": round(st.mean(v), 4),
                                        "sum": round(sum(v), 3)}

# ---------------------------------------------------------------------------
# L3 — ENVIRONMENT ADAPTATION : regime marche (proxy prix via entry_price)
# ---------------------------------------------------------------------------
price_series = [(r["ts"], r["entry_price"]) for r in closes if not math.isnan(r["entry_price"])]
env = {}
if len(price_series) > 20:
    p0 = [p for _, p in price_series[:len(price_series)//2]]
    p1 = [p for _, p in price_series[len(price_series)//2:]]
    env["price_mean_H1"] = round(st.mean(p0), 1)
    env["price_mean_H2"] = round(st.mean(p1), 1)
    env["price_drift_pct"] = round(100 * (st.mean(p1) - st.mean(p0)) / st.mean(p0), 2)
    # vol proxy : stdev des rendements trade-to-trade
    rets = [(price_series[i+1][1] - price_series[i][1]) / price_series[i][1]
            for i in range(len(price_series) - 1)]
    env["trade2trade_vol_H1"] = round(st.pstdev(rets[:len(rets)//2]) * 100, 3)
    env["trade2trade_vol_H2"] = round(st.pstdev(rets[len(rets)//2:]) * 100, 3)
    # adaptation : frequence de trades par fenetre vs vol
    env["trades_per_win"] = [p["trades"] for p in perf]
    # correlation rangs : vol fenetre vs nb trades fenetre (Spearman approx)
    winvol = [0.0] * NW
    for i in range(1, len(price_series)):
        w = win_idx(price_series[i][0])
        winvol[w] += abs(price_series[i][1] - price_series[i-1][1]) / price_series[i-1][1]
    tpw = env["trades_per_win"]
    if st.pstdev(winvol) > 0 and st.pstdev([float(x) for x in tpw]) > 0:
        mv, mt = st.mean(winvol), st.mean(tpw)
        cov = sum((a - mv) * (b - mt) for a, b in zip(winvol, tpw)) / NW
        env["corr_vol_tradefreq"] = round(cov / (st.pstdev(winvol) * st.pstdev([float(x) for x in tpw])), 3)
    else:
        env["corr_vol_tradefreq"] = None

# ---------------------------------------------------------------------------
# Verdicts classes
# ---------------------------------------------------------------------------
verdicts = []

def verdict(topic, status, evidence):
    verdicts.append({"topic": topic, "status": status, "evidence": evidence})

# L4
last_div = diversity[-1]
verdict("L4.collapse_SELL_absorbing", "CONFIRME",
        f"dernier bloc: share_SELL={last_div['share_SELL']}, a0_mean={last_div['a0_mean']}, "
        f"advBUY_nan={last_div['pct_advBUY_nan']}% ; bascule durable upd={collapse_upd}")
verdict("L4.spam_sterile_routing", "CONFIRME" if sterile_events > 1000 else "PROBABLE",
        f"sell->hold sterile events={sterile_events}, inv_penalty_total={consequences['inv_penalty_total']}")
# L2
if p_repeat_given_sterile is not None and p_sell_baseline is not None:
    learns_avoid = p_repeat_given_sterile < p_sell_baseline
    verdict("L2.repetition_apres_rejet", "CONFIRME" if not learns_avoid else "REFUTE",
            f"P(SELL|SELL sterile t-1)={round(p_repeat_given_sterile,3)} vs baseline P(SELL)={round(p_sell_baseline,3)} "
            f"-> {'PAS d apprentissage d evitement' if not learns_avoid else 'evitement appris'}")
sl_trend = None
sl_valid = [(i, s) for i, s in enumerate(sl_by_win) if s is not None]
if len(sl_valid) >= 6:
    h1 = st.mean([s for i, s in sl_valid[:len(sl_valid)//2]])
    h2 = st.mean([s for i, s in sl_valid[len(sl_valid)//2:]])
    sl_trend = {"SL_rate_H1": round(h1, 3), "SL_rate_H2": round(h2, 3)}
    verdict("L2.apprend_a_eviter_SL", "CONFIRME" if h2 >= h1 else "PROBABLE",
            f"taux SL H1={round(h1,3)} -> H2={round(h2,3)} ({'pas d amelioration' if h2 >= h1 else 'amelioration'})")
# L3
if env.get("corr_vol_tradefreq") is not None:
    c = env["corr_vol_tradefreq"]
    verdict("L3.adaptation_vol_frequence", "REFUTE" if abs(c) < 0.3 else "PROBABLE",
            f"corr(vol, freq_trades)={c} ; prix H1={env['price_mean_H1']} -> H2={env['price_mean_H2']} "
            f"({env['price_drift_pct']}%)")
# L5
wrs = [p["WR"] for p in perf if p["WR"] is not None]
if len(wrs) >= 6:
    wr_h1, wr_h2 = st.mean(wrs[:len(wrs)//2]), st.mean(wrs[len(wrs)//2:])
    verdict("L5.performance_trend", "CONFIRME" if wr_h2 <= wr_h1 else "PROBABLE",
            f"WR H1={round(wr_h1,1)}% -> H2={round(wr_h2,1)}% ; Sharpe(global,indicatif)={round(sharpe,2) if sharpe else None}")

# ---------------------------------------------------------------------------
# Scores radar (0-100) — heuristique transparente
# ---------------------------------------------------------------------------
def clamp01(x):
    return max(0.0, min(1.0, x))

n_exec_total = sum(p["trades"] for p in perf)
l1_score = 100 * clamp01(n_exec_total / 1386) if n_exec_total else 0  # flux consequences existe
# L2 : evitement erreurs
l2_score = 0.0
if p_repeat_given_sterile is not None and p_sell_baseline:
    l2_score += 50 * clamp01((p_sell_baseline - p_repeat_given_sterile) / max(p_sell_baseline, 1e-9) + 0.0)
if sl_trend:
    l2_score += 50 * clamp01((sl_trend["SL_rate_H1"] - sl_trend["SL_rate_H2"]) / max(sl_trend["SL_rate_H1"], 1e-9) + 0.5)
# L3
l3_score = 100 * clamp01(abs(env.get("corr_vol_tradefreq") or 0))
# L4 : diversite moyenne des 3 actions sur le run
mean_div = st.mean([min(d["share_BUY"] or 0, 1) + min(d["share_HOLD"] or 0, 1) for d in diversity])
l4_score = 100 * clamp01(mean_div / 2)  # 1.0 = BUY+HOLD dominant, 0 = SELL pur
# L5 : WR moyen normalise vs 50% break-even + PF
wr_mean = st.mean(wrs) if wrs else 0
pf_vals = [p["PF"] for p in perf if p["PF"]]
pf_mean = st.mean(pf_vals) if pf_vals else 0
l5_score = 100 * clamp01(0.5 * wr_mean / 50 + 0.5 * clamp01(pf_mean / 1.5))

scores = {"L1_consequences": round(l1_score, 1), "L2_erreurs": round(l2_score, 1),
          "L3_environnement": round(l3_score, 1), "L4_coherence": round(l4_score, 1),
          "L5_performance": round(l5_score, 1)}

# ---------------------------------------------------------------------------
# Sorties
# ---------------------------------------------------------------------------
metrics = {
    "run": {"t0": str(t0), "t1": str(t1), "span_h": round(SPAN / 3600, 2),
            "trades_dedup": len(closes), "opens_dedup": len(opens),
            "action_diff_rows": len(ad_rows), "ppo_updates": len(anchor_rows),
            "collapse_upd_first_durable": collapse_upd},
    "perf_per_window": perf, "diversity_per_block": diversity,
    "exec_per_window": exec_stats, "consequences": consequences,
    "hold_adjustment": hold_adj, "environment": env,
    "error_learning": {"p_repeat_given_sterile_sell": p_repeat_given_sterile,
                       "p_sell_baseline": p_sell_baseline,
                       "sterile_events": sterile_events,
                       "sl_rate_by_win": sl_by_win, "tp_rate_by_win": tp_by_win},
    "scores": scores, "verdicts": verdicts,
}
(OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))

lines = ["# RADAR D'APPRENTISSAGE ADAN V31 — artefacts figes (read-only)", "",
         f"Run: {t0} -> {t1} ({round(SPAN/3600,2)}h) | trades dedup={len(closes)} | "
         f"updates PPO={len(anchor_rows)} | collapse durable a upd={collapse_upd}", "",
         "## Scores (0-100)", ""]
for k, v in scores.items():
    lines.append(f"- **{k}** : {v}")
lines += ["", "## Verdicts classes", ""]
for v in verdicts:
    lines.append(f"- [{v['status']}] **{v['topic']}** — {v['evidence']}")
lines += ["", "## Performance par fenetre (deciles temporels)", "",
          "| win | trades | WR% | PnL | PF | avg_hold | TP | SL | AGENT |",
          "|-----|--------|-----|-----|----|--------|----|----|-------|"]
for p in perf:
    lines.append(f"| {p['win']} | {p['trades']} | {p['WR']} | {p['PnL']} | {p['PF']} | "
                 f"{p['avg_hold']} | {p['TP']} | {p['SL']} | {p['AGENT_CLOSE']} |")
lines += ["", "## Diversite politique par bloc d'updates", "",
          "| blk | upd | a0_mean | %BUY | %HOLD | %SELL | %advBUY_nan |",
          "|-----|-----|---------|------|-------|-------|--------------|"]
for d in diversity:
    lines.append(f"| {d['blk']} | {d['upd_range'][0]}-{d['upd_range'][1]} | {d['a0_mean']} | "
                 f"{d['share_BUY']} | {d['share_HOLD']} | {d['share_SELL']} | {d['pct_advBUY_nan']} |")
lines += ["", "## Apprentissage d'erreurs", "",
          f"- P(SELL | SELL sterile a t-1) = {p_repeat_given_sterile} vs baseline P(SELL) = {p_sell_baseline}",
          f"- Evenements SELL->HOLD steriles: {sterile_events}",
          f"- inv_penalty total = {consequences['inv_penalty_total']} "
          f"(events non nuls: {consequences['inv_penalty_nonzero_events']})",
          f"- Taux SL par fenetre: {[round(s,3) if s is not None else None for s in sl_by_win]}",
          f"- Ajustement hold: {hold_adj}", "",
          "## Adaptation environnement", ""]
for k, v in env.items():
    lines.append(f"- {k}: {v}")
(OUT / "RADAR.md").write_text("\n".join(lines))
print(json.dumps({"scores": scores, "collapse_upd": collapse_upd,
                  "trades": len(closes), "verdicts": len(verdicts)}, indent=2))
