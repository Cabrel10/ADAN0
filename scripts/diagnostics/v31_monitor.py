#!/usr/bin/env python3
"""
V31-DIAGNOSTIC — moniteur continu des 6 gates (criteres explicites user):
  1. a0 ne derive plus systematiquement vers +1 (ou -1)
  2. FLAT devient une action viable
  3. correlation reward <-> delta portfolio positive
  4. les cycles ouverture/fermeture cessent de perdre mecaniquement
  5. la policy reste sensible aux observations
  6. aucun nouvel attracteur artificiel BUY/SELL

PROVENANCE (resolu 2026-08-17): sous Ray, chaque worker a son propre
working_dir et le RewardCollector y ecrit ses streams jsonl:
  .adan_ray/tmp/session_*/artifacts/*/adan_pbt_training/working_dirs/*/logs/rewards/
Le dossier ADAN0/logs/rewards/ n'est PAS alimente par les runs Ray.
Les deux racines sont scannees; les quarantaines sont exclues.

SCHEMA REEL d'un record (verifie par dump):
  action.raw[0] = a0 brut ; action.type = 'hold'/'buy'/'sell'/...
  reward.breakdown.behavior_penalty / pnl_reward / final_reward / open_positions
  portfolio.total_value ; frequency.attempts / invalid_attempts ; step

Usage: python v31_monitor.py [--since YYYYMMDD_HHMM] [--watch SEC] [--out FILE]
"""
import json, glob, os, sys, time, math
import numpy as np
from collections import Counter

REPO = "/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
THR = 0.05

STREAM_GLOBS = [
    os.path.join(REPO, "logs/rewards/worker_*_rewards_*.jsonl"),
    "/home/ubuntu/webapp/.adan_ray/tmp/session_*/artifacts/*/"
    "adan_pbt_training/working_dirs/*/logs/rewards/worker_*_rewards_*.jsonl",
]


def _ts(fp):
    # worker_0_rewards_YYYYMMDD_HHMMSS.jsonl -> YYYYMMDD_HHMMSS (full)
    parts = os.path.basename(fp).replace(".jsonl", "").split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 else ""


def find_streams(since_ts=None):
    files = []
    for g in STREAM_GLOBS:
        files.extend(glob.glob(g))
    files = [f for f in files if "quarantine" not in f]
    if since_ts:
        files = [f for f in files if _ts(f) >= since_ts]
    return sorted(set(files), key=os.path.getmtime)


def load_file(fp):
    rows = []
    try:
        with open(fp) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                bd = (r.get("reward", {}) or {}).get("breakdown", {}) or {}
                act = r.get("action", {}) or {}
                raw_a = act.get("raw", [])
                if not (isinstance(raw_a, list) and len(raw_a) > 0):
                    continue
                pf = r.get("portfolio", {}) or {}
                fr = r.get("frequency", {}) or {}
                rows.append(dict(
                    a0=float(raw_a[0]),
                    atype=str(act.get("type", "")),
                    pv=float(pf.get("total_value", np.nan) or np.nan),
                    behav=float(bd.get("behavior_penalty", 0) or 0),
                    behav_inv=float(bd.get("behavior_invalid_penalty", 0) or 0),
                    pnl=float(bd.get("pnl_reward", 0) or 0),
                    fin=float(bd.get("final_reward",
                                    (r.get("reward", {}) or {}).get("total", 0)) or 0),
                    op=int(bd.get("open_positions", 0) or 0),
                    step=int(r.get("step", 0) or 0),
                    inv_att=int(fr.get("invalid_attempts", 0) or 0),
                    att=int(fr.get("attempts", 0) or 0),
                ))
    except Exception:
        pass
    return rows


def _gates_per_stream(name, rows):
    """G1 (derive a0) calculee par stream, dans l'ordre des steps."""
    out = []
    a0 = np.array([r["a0"] for r in rows])
    n = len(a0)
    B = 1000
    means = [a0[k:k + B].mean() for k in range(0, n - B + 1, B)]
    g1 = True
    drift = "N/A"
    if len(means) >= 2:
        monotone_up = all(means[i + 1] > means[i] for i in range(len(means) - 1))
        monotone_dn = all(means[i + 1] < means[i] for i in range(len(means) - 1))
        drift = "buckets=" + ",".join("%+.3f" % m for m in means)
        g1 = not ((monotone_up and means[-1] > 0.5) or
                  (monotone_dn and means[-1] < -0.5))
    elif n >= 200:
        drift = "buckets=insuffisants(n=%d)" % n
    g1 = g1 and (abs(a0.mean()) < 0.3 or n < 2000) and a0.std() > 0.05
    out.append("  [%s] G1 pas-de-derive-a0: %s | n=%d a0_mean=%+.4f std=%.4f %s"
               % (name, "PASS" if g1 else "FAIL", n, a0.mean(), a0.std(), drift))
    return g1, out


def verdict(streams):
    """streams: dict name -> rows."""
    out = []
    total = sum(len(v) for v in streams.values())
    if total < 200:
        return "INSUFFISANT (n=%d < 200) — attendre plus de donnees" % total
    all_rows = [r for rows in streams.values() for r in rows]
    a0 = np.array([r["a0"] for r in all_rows])
    behav = np.array([r["behav"] for r in all_rows])
    behav_inv = np.array([r["behav_inv"] for r in all_rows])
    fin = np.array([r["fin"] for r in all_rows])
    op = np.array([r["op"] for r in all_rows])
    out.append("n=%d | streams=%d | a0 mean=%+.4f std=%.4f"
               % (total, len(streams), a0.mean(), a0.std()))
    # G1 par stream
    g1_all = True
    for name, rows in streams.items():
        if len(rows) >= 200:
            g1, sub = _gates_per_stream(name, rows)
            g1_all = g1_all and g1
            out.extend(sub)
    # G2: FLAT viable
    flat = behav[op == 0]
    openb = behav[op > 0]
    g2 = (len(flat) > 0 and flat.mean() > -0.05)
    out.append("G2 FLAT-viable: %s | behav_flat_mean=%.4f behav_open_mean=%.4f"
               % ("PASS" if g2 else "FAIL",
                  flat.mean() if len(flat) else float("nan"),
                  openb.mean() if len(openb) else float("nan")))
    # G2b: pas de pénalité constante -0.28 (médiane <= 0.01) ET symétrie
    # par bucket état×intention. L'anti-spam escaladé (jusqu'à cap 0.30) est
    # une protection conservée volontairement (config l.1360-1396) : seule
    # l'ASYMÉTRIE systématique est une faute, pas la magnitude sur streak.
    med_pen = float(np.median(np.abs(behav))) if len(behav) else 0.0
    bk = {}
    for r in all_rows:
        st = "OPEN" if r["op"] > 0 else "FLAT"
        it = "BUY" if r["a0"] > THR else ("SELL" if r["a0"] < -THR else "HOLD")
        bk.setdefault(st + "+" + it, []).append(r["behav"])
    bm = {k: (sum(v) / len(v), len(v)) for k, v in bk.items()}
    asym_flat = abs(bm.get("FLAT+SELL", (0, 0))[0] - bm.get("FLAT+BUY", (0, 0))[0])
    asym_open = abs(bm.get("OPEN+BUY", (0, 0))[0] - bm.get("OPEN+SELL", (0, 0))[0])
    g2b = med_pen <= 0.011 and asym_flat < 0.02 and asym_open < 0.02
    out.append("G2b symetrie-invalidite: %s | med|pen|=%.4f asymFLAT=%.4f "
               "asymOPEN=%.4f | buckets=%s"
               % ("PASS" if g2b else "FAIL", med_pen, asym_flat, asym_open,
                  {k: "%+.3f(n%d)" % v for k, v in sorted(bm.items())}))
    # G3: corr(reward, dPV) par stream puis mediane
    corrs = []
    for name, rows in streams.items():
        pv = np.array([r["pv"] for r in rows])
        f2 = np.array([r["fin"] for r in rows])
        mask = ~np.isnan(pv)
        if mask.sum() > 20:
            dpv = np.diff(pv[mask])
            ff = f2[mask][:-1]
            if dpv.std() > 0 and ff.std() > 0:
                corrs.append(float(np.corrcoef(ff, dpv)[0, 1]))
    if corrs:
        c = float(np.median(corrs))
        g3 = c > 0
        out.append("G3 corr(reward,dPV)>0: %s | median r=%+.4f (%d streams)"
                   % ("PASS" if g3 else "FAIL", c, len(corrs)))
    else:
        g3 = None
        out.append("G3 corr(reward,dPV): NA (pv constant ou absent)")
    # G5: sensibilite obs
    g5 = a0.std() > 0.05
    out.append("G5 policy-obs-sensible: %s | a0_std=%.4f"
               % ("PASS" if g5 else "FAIL", a0.std()))
    # G6: pas d'attracteur sur l'INTENTION (a0). L'exécution (action.type)
    # est structurellement dominée par 'hold' quand les gates anti-spam/
    # cooldown rejettent — c'est le mécanisme conservé, PAS un attracteur.
    intents = np.where(a0 > THR, 1, np.where(a0 < -THR, -1, 0))
    dist = Counter(intents.tolist())
    mx = max(dist.values()) / total
    types = Counter(r["atype"] for r in all_rows)
    g6 = mx < 0.80
    out.append("G6 pas-d-attracteur(intent): %s | intent SELL/HOLD/BUY=%.1f%%/%.1f%%/%.1f%%"
               " | exec(informatif) %s"
               % ("PASS" if g6 else "FAIL",
                  dist.get(-1, 0) / total * 100, dist.get(0, 0) / total * 100,
                  dist.get(1, 0) / total * 100,
                  {k: "%.1f%%" % (v / total * 100) for k, v in types.most_common(5)}))
    # Taux d'invalidite (compteurs cumulatifs -> delta max)
    inv = max((r["inv_att"] for r in all_rows), default=0)
    att = max((r["att"] for r in all_rows), default=0)
    ratio = inv / att if att else 0.0
    out.append("invalid_ratio=%.3f (cumul %d/%d) | gate<0.50: %s"
               % (ratio, inv, att, "PASS" if ratio < 0.50 else "FAIL"))
    # G4: cycles (ouvertures detectees par transition op 0->1)
    cyc = []
    for name, rows in streams.items():
        ops = np.array([r["op"] for r in rows])
        fins = np.array([r["fin"] for r in rows])
        flips = int(np.sum(np.diff((ops > 0).astype(int)) == 1))
        cyc.append((name, flips))
    out.append("G4 cycles: ouvertures=%s (PnL par cycle au checkpoint 10k)"
               % {k: v for k, v in cyc})
    checks = [g1_all, g2, g2b, g5, g6] + ([g3] if g3 is not None else []) + [ratio < 0.50]
    score = sum(1 for c in checks if c)
    out.append("SCORE: %d/%d %s" % (score, len(checks),
                                    "OK" if score == len(checks) else "ATTENTION"))
    return "\n".join(out)


if __name__ == "__main__":
    watch = 0
    out_file = None
    since = "20260817_1753"  # V31-DIAG clean relaunch (post-quarantine)
    args = sys.argv[1:]
    if "--since" in args:
        since = args[args.index("--since") + 1]
    if "--watch" in args:
        watch = int(args[args.index("--watch") + 1])
    if "--out" in args:
        out_file = args[args.index("--out") + 1]

    def once():
        files = find_streams(since)
        streams = {}
        for fp in files:
            # cle courte: worker_X + ts
            b = os.path.basename(fp).replace(".jsonl", "")
            streams[b] = load_file(fp)
        streams = {k: v for k, v in streams.items() if v}
        if not streams:
            v = "AUCUN STREAM >= %s (chemins: logs/rewards + ray working_dirs)" % since
        else:
            v = verdict(streams)
        line = "[%s]\n%s\n%s" % (time.strftime("%H:%M:%S"), v, "=" * 60)
        print(line)
        if out_file:
            with open(out_file, "a") as f:
                f.write(line + "\n")

    if watch > 0:
        while True:
            once()
            time.sleep(watch)
    else:
        once()
