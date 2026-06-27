#!/usr/bin/env python3
"""Watchdog NON-destructif pour le run 500k v5 (post FIX A+B + anti-deadlock).

Cahier des charges utilisateur (2026-06-27):
  - Surveillance toutes les N minutes: step max, fps, RAM, Requested/Executed,
    STERILE_SELL, BUY->HOLD, distribution action[0], + metriques PPO (approx_kl,
    clip_fraction, entropy_loss, value_loss, explained_variance).
  - Critères d'arrêt AUTOMATIQUE (KILL):
      * OHLC_INCOHER > 0                         (bug cross-TF revenu)
      * Exception/Traceback > 0                  (crash)
      * step figé > FREEZE_KILL_MIN minutes      (deadlock)
      * fps < 3 pendant >= 2 sweeps consécutifs  (deadlock lent)
      * action[0] recollé à une extrémité (|med|>0.95) >= 2 sweeps  (collapse revenu)
      * RAM disponible < 300 Mo                  (OOM imminent)
  - PAS de rotation destructive: on n'écrit que des snapshots/corrélations à part,
    on ne touche JAMAIS au log brut.

Sortie:
  - logs/surveillance/v5_timeline.csv  (1 ligne / sweep, pour corrélation)
  - logs/surveillance/v5_report.md     (rapport humain append-only)
  - kill du process si critère d'arrêt atteint (avec raison loggée).
"""
import csv
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone

ROOT = "/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
LOG = sys.argv[1] if len(sys.argv) > 1 else None
if not LOG:
    print("usage: watchdog_500k_v5.py <raw_log_path> [interval_sec]")
    sys.exit(2)
INTERVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 300  # 5 min
CSV_PATH = os.path.join(ROOT, "logs/surveillance/v5_timeline.csv")
REPORT = os.path.join(ROOT, "logs/surveillance/v5_report.md")
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

# Seuils d'arrêt
FREEZE_KILL_MIN = 20         # step figé > 20 min -> kill
FPS_MIN = 3.0                # fps < 3 sur 2 sweeps -> kill
ACT_EXTREME = 0.95           # |median(action[0])| > 0.95 sur 2 sweeps -> kill
RAM_MIN_MB = 300             # RAM dispo < 300 Mo -> kill

_step_re = re.compile(r"\[STEP (\d+)")
_re_act = re.compile(r"action: \[\s*([-0-9.]+)")
_re_kv = {
    "approx_kl": re.compile(r"approx_kl\s*\|\s*([-0-9.e]+)"),
    "clip_fraction": re.compile(r"clip_fraction\s*\|\s*([-0-9.e]+)"),
    "entropy_loss": re.compile(r"entropy_loss\s*\|\s*([-0-9.e]+)"),
    "value_loss": re.compile(r"value_loss\s*\|\s*([-0-9.e]+)"),
    "explained_variance": re.compile(r"explained_variance\s*\|\s*([-0-9.e]+)"),
    "fps": re.compile(r"fps\s*\|\s*([0-9.]+)"),
}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


def train_pids():
    out = sh("pgrep -f 'train_parallel_agents.py'")
    return [p for p in out.split() if p.strip()]


def tail_text(path, nlines=20000):
    return sh(f"tail -n {nlines} '{path}'")


def max_step(path):
    out = sh(f"grep -oE '\\[STEP [0-9]+' '{path}' | grep -oE '[0-9]+' | sort -n | tail -1")
    try:
        return int(out)
    except Exception:
        return -1


def count(path, pat, ci=True):
    flag = "-ic" if ci else "-c"
    out = sh(f"grep {flag} '{pat}' '{path}'")
    try:
        return int(out.splitlines()[0]) if out else 0
    except Exception:
        return 0


def action0_stats(text):
    vals = [float(m) for m in _re_act.findall(text)]
    if not vals:
        return (0, 0.0, 0.0, 0.0)
    vals.sort()
    n = len(vals)
    return (n, vals[0], vals[n // 2], vals[-1])


def last_metric(text, key):
    ms = _re_kv[key].findall(text)
    try:
        return float(ms[-1]) if ms else None
    except Exception:
        return None


def req_exec(text):
    d = {}
    for m in re.findall(r"Requested=([A-Z]+) Executed=([A-Z]+)", text):
        d[f"{m[0]}->{m[1]}"] = d.get(f"{m[0]}->{m[1]}", 0) + 1
    return d


def ram_avail_mb():
    out = sh("free -m | awk '/Mem:/{print $7}'")
    try:
        return int(out)
    except Exception:
        return 99999


def proc_diag(pids):
    """Retourne (cpu_pct, state, rss_mb) du process d'entrainement principal.
    Permet de distinguer (table utilisateur):
      CPU 0% + pas de log -> deadlock ; CPU 100% + pas d'avancement -> boucle ;
      RAM qui monte -> fuite ; state D -> I/O bloque ; threads + CPU bas -> OpenMP.
    """
    if not pids:
        return (0.0, "?", 0)
    pid = pids[0]
    out = sh(f"ps -p {pid} -o %cpu=,stat=,rss=")
    try:
        parts = out.split()
        cpu = float(parts[0]); state = parts[1]; rss = int(parts[2]) // 1024
        return (cpu, state, rss)
    except Exception:
        return (0.0, "?", 0)


def kill_run(reason):
    log(f"!!! KILL: {reason}")
    for p in train_pids():
        sh(f"kill -9 {p}")
    with open(REPORT, "a") as f:
        f.write(f"\n## {now()} — ARRÊT AUTOMATIQUE\n- Raison: **{reason}**\n")


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open(REPORT, "a") as f:
        f.write(line + "\n")


# CSV header
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow([
            "ts", "max_step", "fps", "ram_mb", "cpu_pct", "proc_state",
            "rss_mb", "act_n", "act_min", "act_med",
            "act_max", "approx_kl", "clip_fraction", "entropy_loss", "value_loss",
            "explained_variance", "ohlc_incoher", "exceptions", "sterile_sell",
            "buy_hold", "sell_hold", "buy_buy", "sell_sell", "open", "close",
        ])

log(f"watchdog v5 démarré (log={os.path.basename(LOG)}, interval={INTERVAL}s)")
prev_step = -1
prev_step_time = time.time()
low_fps_streak = 0
extreme_streak = 0

while True:
    if not train_pids():
        log("processus d'entraînement terminé — watchdog s'arrête.")
        break
    if not os.path.exists(LOG):
        time.sleep(5)
        continue

    text = tail_text(LOG)
    ms = max_step(LOG)
    fps = last_metric(text, "fps")
    ram = ram_avail_mb()
    pids_now = train_pids()
    cpu_pct, pstate, rss_mb = proc_diag(pids_now)
    a_n, a_min, a_med, a_max = action0_stats(text)
    kl = last_metric(text, "approx_kl")
    clip = last_metric(text, "clip_fraction")
    ent = last_metric(text, "entropy_loss")
    vloss = last_metric(text, "value_loss")
    ev = last_metric(text, "explained_variance")
    ohlc = count(LOG, "OHLC_INCOHER")
    exc = count(LOG, "Traceback") + count(LOG, "Exception") + count(LOG, "Error:")
    sterile = count(LOG, "STERILE_SELL")
    re_d = req_exec(text)
    opn = count(LOG, "TRADE_AUDIT_OPEN")
    cls = count(LOG, "TRADE_AUDIT_CLOSE")

    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            now(), ms, fps, ram, cpu_pct, pstate, rss_mb,
            a_n, a_min, a_med, a_max, kl, clip, ent,
            vloss, ev, ohlc, exc, sterile,
            re_d.get("BUY->HOLD", 0), re_d.get("SELL->HOLD", 0),
            re_d.get("BUY->BUY", 0), re_d.get("SELL->SELL", 0), opn, cls,
        ])

    log(f"step={ms} fps={fps} cpu={cpu_pct}% state={pstate} rss={rss_mb}MB "
        f"ram={ram}MB act0[min={a_min:.2f} med={a_med:.2f} "
        f"max={a_max:.2f}] kl={kl} clip={clip} ent={ent} ev={ev} | "
        f"OHLC={ohlc} exc={exc} sterile={sterile} | "
        f"BUY>H={re_d.get('BUY->HOLD',0)} SELL>H={re_d.get('SELL->HOLD',0)} "
        f"OPEN={opn} CLOSE={cls}")

    # Diagnostic auto du type de gel (table utilisateur) si step stagne
    if ms == prev_step and ms >= 0:
        if cpu_pct < 5:
            log(f"  ↳ DIAG: CPU≈{cpu_pct}% + step fige -> DEADLOCK probable (OpenMP/lock)")
        elif cpu_pct > 80:
            log(f"  ↳ DIAG: CPU≈{cpu_pct}% + step fige -> BOUCLE INFINIE probable")
        if pstate.startswith("D"):
            log(f"  ↳ DIAG: state={pstate} -> I/O BLOQUE (disque/FD)")

    # ── CRITÈRES D'ARRÊT ──
    if ohlc > 0:
        kill_run(f"OHLC_INCOHER={ohlc} (bug cross-TF revenu)"); break
    if exc > 0:
        kill_run(f"exceptions={exc} (crash)"); break
    if ram < RAM_MIN_MB:
        kill_run(f"RAM dispo={ram}MB < {RAM_MIN_MB}MB (OOM imminent)"); break

    # step figé
    if ms == prev_step and ms >= 0:
        frozen_min = (time.time() - prev_step_time) / 60.0
        if frozen_min > FREEZE_KILL_MIN:
            kill_run(f"step figé à {ms} depuis {frozen_min:.1f} min (deadlock)"); break
    else:
        prev_step = ms
        prev_step_time = time.time()

    # fps bas répété
    if fps is not None and fps < FPS_MIN:
        low_fps_streak += 1
        if low_fps_streak >= 2:
            kill_run(f"fps={fps} < {FPS_MIN} sur {low_fps_streak} sweeps (deadlock lent)"); break
    else:
        low_fps_streak = 0

    # action[0] recollé à une extrémité (collapse) — seulement après warmup (step>5000)
    if ms > 5000 and a_n > 50 and abs(a_med) > ACT_EXTREME:
        extreme_streak += 1
        if extreme_streak >= 2:
            kill_run(f"action[0] médiane={a_med:.2f} (|.|>{ACT_EXTREME}) sur "
                     f"{extreme_streak} sweeps (collapse revenu)"); break
    else:
        extreme_streak = 0

    time.sleep(INTERVAL)

log("watchdog v5 terminé.")
