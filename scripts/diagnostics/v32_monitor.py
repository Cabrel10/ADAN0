#!/usr/bin/env python3
"""V32 surveillance monitor — radar léger sur le log d'entraînement.

Parcourt le log train, extrait périodiquement :
- step courant, vitesse (steps/min)
- ANCHOR_DEBUG nB/nH/nS (diversité action brute pré-routing)
- mu/std / log_std si loggés
- approx_kl / explained_variance / entropy_loss (santé PPO)
- EPISODE_REJECTIONS (gates), TRADE_AUDIT (exécutions)
- portfolio value (dérive capital)

Détecte AVANT l'état absorbant (règle utilisateur) :
- une action nB/nH/nS devient rare (<5% du buffer) ou disparaît (0)
- explained_variance NaN
- portfolio en chute libre

Écrit un radar compact JSONL: logs/v32/radar_<ts>.jsonl
N'interrompt PAS l'entraînement — surveillance passive.
"""
import json
import re
import sys
import time
from collections import deque
from pathlib import Path

LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else None
INTERVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 120  # secondes
if not LOG or not LOG.exists():
    print("usage: v32_monitor.py <train_log> [interval_s]")
    sys.exit(1)

OUT = LOG.parent / (LOG.stem.replace("v32_train", "radar") + ".jsonl")

RE_STEP = re.compile(r"Starting step (\d+)")
RE_ANCHOR = re.compile(r"nB=(\d+)\s+nS=(\d+)\s+nH=(\d+)")
RE_KL = re.compile(r"approx_kl[\"']?\s*[:=]\s*([-\d.eE]+)")
RE_EV = re.compile(r"explained_variance[\"']?\s*[:=]\s*([-\d.eEnan]+)")
RE_ENT = re.compile(r"entropy_loss[\"']?\s*[:=]\s*([-\d.eE]+)")
RE_MU = re.compile(r"mu[_ ]?mean[\"']?\s*[:=]\s*([-\d.eE]+)")
RE_STD = re.compile(r"std[_ ]?mean[\"']?\s*[:=]\s*([-\d.eE]+)")
RE_PORT = re.compile(r"Portfolio value:\s*([\d.]+)")
RE_REJ = re.compile(r"\[EPISODE_REJECTIONS\].*Reasons:\s*(\{[^}]+\})")


def tail_scan(path, last_pos):
    with open(path, "r", errors="replace") as f:
        f.seek(last_pos)
        chunk = f.read()
        return chunk, f.tell()


def alerts_from(radar):
    a = []
    tot = radar.get("nB", 0) + radar.get("nH", 0) + radar.get("nS", 0)
    if tot > 0:
        for k in ("nB", "nH", "nS"):
            frac = radar.get(k, 0) / tot
            if radar.get(k, 0) == 0:
                a.append(f"{k}=0 (action brute disparue)")
            elif frac < 0.05:
                a.append(f"{k} rare ({frac:.1%})")
    ev = radar.get("explained_variance")
    if ev is not None and (ev != ev):  # NaN
        a.append("explained_variance=NaN")
    return a


def main():
    last_pos = 0
    port_hist = deque(maxlen=50)
    prev_step = 0
    prev_t = time.time()
    print(f"[monitor] {LOG} -> {OUT} every {INTERVAL}s", flush=True)
    with open(OUT, "a") as out:
        while True:
            time.sleep(INTERVAL)
            try:
                chunk, last_pos = tail_scan(LOG, last_pos)
            except FileNotFoundError:
                print("[monitor] log disparu, arret", flush=True)
                return
            if not chunk:
                # process peut être fini
                continue
            steps = RE_STEP.findall(chunk)
            step = int(steps[-1]) if steps else prev_step
            anchors = RE_ANCHOR.findall(chunk)
            ports = [float(x) for x in RE_PORT.findall(chunk)]
            for p in ports:
                port_hist.append(p)
            now = time.time()
            spm = (step - prev_step) / max((now - prev_t) / 60.0, 1e-6)
            radar = {
                "ts": time.strftime("%H:%M:%S"),
                "step": step,
                "steps_per_min": round(spm, 1),
            }
            if anchors:
                nB, nS, nH = map(int, anchors[-1])
                radar.update(nB=nB, nS=nS, nH=nH)
            for rx, key in ((RE_KL, "approx_kl"), (RE_EV, "explained_variance"),
                            (RE_ENT, "entropy_loss"), (RE_MU, "mu_mean"),
                            (RE_STD, "std_mean")):
                m = rx.findall(chunk)
                if m:
                    try:
                        radar[key] = float(m[-1])
                    except ValueError:
                        radar[key] = m[-1]
            rej = RE_REJ.findall(chunk)
            if rej:
                radar["last_rejections"] = rej[-1][:200]
            if port_hist:
                radar["portfolio"] = port_hist[-1]
                radar["portfolio_min"] = min(port_hist)
            radar["alerts"] = alerts_from(radar)
            out.write(json.dumps(radar) + "\n")
            out.flush()
            flag = " ⚠ " + ";".join(radar["alerts"]) if radar["alerts"] else ""
            print(f"[radar] step={step} spm={spm:.0f} "
                  f"nB/nH/nS={radar.get('nB','-')}/{radar.get('nH','-')}/{radar.get('nS','-')} "
                  f"port={radar.get('portfolio','-')}{flag}", flush=True)
            prev_step, prev_t = step, now


if __name__ == "__main__":
    main()
