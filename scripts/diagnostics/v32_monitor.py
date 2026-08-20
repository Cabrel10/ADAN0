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

# Normalize any "<vN>_train_<ts>" stem into "radar_<vN>_<ts>" (handles v32/v33/v34...)
_stem = re.sub(r"^(v\d+)_train_", r"radar_\1_", LOG.stem)
if _stem == LOG.stem:  # fallback for stems without a version prefix
    _stem = LOG.stem.replace("_train", "").replace("v32", "radar")
    if "radar" not in _stem:
        _stem = "radar_" + _stem
OUT = LOG.parent / (_stem + ".jsonl")

RE_STEP = re.compile(r"Starting step (\d+)")
RE_ANCHOR = re.compile(r"nB=(\d+)\s+nS=(\d+)\s+nH=(\d+)")
# SB3 logs metrics as pipe-delimited tables:  |    approx_kl    | 0.012   |
RE_KL = re.compile(r"\|\s*approx_kl\s*\|\s*([-\d.eEnan]+)\s*\|")
RE_EV = re.compile(r"\|\s*explained_variance\s*\|\s*([-\d.eEnan]+)\s*\|")
RE_ENT = re.compile(r"\|\s*entropy_loss\s*\|\s*([-\d.eEnan]+)\s*\|")
RE_CLIP = re.compile(r"\|\s*clip_fraction\s*\|\s*([-\d.eEnan]+)\s*\|")
# mu/std come from the [ANCHOR_DEBUG] line: a0_mean=... a0_std=...
RE_MU = re.compile(r"a0_mean=([-\d.eEnan]+)")
RE_STD = re.compile(r"a0_std=([-\d.eEnan]+)")
RE_ANCHORVAL = re.compile(r"ANCHOR_DEBUG.*?anchor=([-\d.eEnan]+)")
# tanh_mu_mean SB3 custom metric (policy mean after tanh)
RE_TANHMU = re.compile(r"\|\s*tanh_mu_mean\s*\|\s*([-\d.eEnan]+)\s*\|")
RE_PORT = re.compile(r"Portfolio value:\s*([\d.]+)")
RE_REJ = re.compile(r"\[EPISODE_REJECTIONS\].*Reasons:\s*(\{[^}]+\})")


def tail_scan(path, last_pos):
    with open(path, "r", errors="replace") as f:
        f.seek(last_pos)
        chunk = f.read()
        return chunk, f.tell()


# Persistence counters (stateful) — a single warmup EV<0 must NOT trigger SIGSTOP.
_STATE = {"ev_neg": 0, "kl_hi": 0, "nB0": 0, "nS0": 0}
# SIGSTOP thresholds (user intervention table): EV<0 >5 updates, KL>0.15 persistent,
# nB/nS=0 >3 updates. Radar samples every INTERVAL; counters count consecutive samples.
EV_NEG_LIMIT = 5
KL_HI_LIMIT = 3
N0_LIMIT = 3


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
        # persistence for nB / nS = 0 (absorption)
        _STATE["nB0"] = _STATE["nB0"] + 1 if radar.get("nB", 0) == 0 else 0
        _STATE["nS0"] = _STATE["nS0"] + 1 if radar.get("nS", 0) == 0 else 0
        if _STATE["nB0"] >= N0_LIMIT:
            a.append(f"SIGSTOP: nB=0 x{_STATE['nB0']} (absorption)")
        if _STATE["nS0"] >= N0_LIMIT:
            a.append(f"SIGSTOP: nS=0 x{_STATE['nS0']} (absorption)")
    ev = radar.get("explained_variance")
    if ev is not None:
        if ev != ev:  # NaN
            a.append("explained_variance=NaN")
            _STATE["ev_neg"] += 1
        elif ev < 0:
            _STATE["ev_neg"] += 1
            if _STATE["ev_neg"] >= EV_NEG_LIMIT:
                a.append(f"SIGSTOP: EV<0 x{_STATE['ev_neg']} critic aveugle ({ev:.3f})")
            else:
                a.append(f"EV<0 warmup ({ev:.3f}) [{_STATE['ev_neg']}/{EV_NEG_LIMIT}]")
        else:
            _STATE["ev_neg"] = 0
    mu = radar.get("mu_mean")
    if mu is not None and mu == mu:
        am = abs(mu)
        if am > 1.0:
            a.append(f"SIGSTOP: |mu|>1.0 DERIVE ({mu:+.3f})")
        elif am > 0.8:
            a.append(f"|mu|>0.8 derive serieuse ({mu:+.3f})")
        elif am > 0.5:
            a.append(f"|mu|>0.5 pre-alerte ({mu:+.3f})")
    sd = radar.get("std_mean")
    if sd is not None and sd == sd and sd < 0.1:
        a.append(f"SIGSTOP: sigma<0.1 GEL ({sd:.3f})")
    kl = radar.get("approx_kl")
    if kl is not None and kl == kl:
        _STATE["kl_hi"] = _STATE["kl_hi"] + 1 if kl > 0.15 else 0
        if _STATE["kl_hi"] >= KL_HI_LIMIT:
            a.append(f"SIGSTOP: KL>0.15 x{_STATE['kl_hi']} ({kl:.3f})")
        elif kl > 0.15:
            a.append(f"KL>0.15 ({kl:.3f}) [{_STATE['kl_hi']}/{KL_HI_LIMIT}]")
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
                            (RE_ENT, "entropy_loss"), (RE_CLIP, "clip_fraction"),
                            (RE_MU, "mu_mean"), (RE_STD, "std_mean"),
                            (RE_ANCHORVAL, "anchor"), (RE_TANHMU, "tanh_mu_mean")):
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
