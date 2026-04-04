#!/usr/bin/env python3
"""
ADAN0 Live Monitor — tableau temps réel par worker
Refresh toutes les 5 secondes, lit directement training.log + result.json
"""
import re, os, sys, time, json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
LOG_FILE    = Path("/mnt/new_data/t10_training/logs/training.log")
RAY_RESULTS = Path("/mnt/new_data/t10_training/ray_results/adan_pbt_training")
REFRESH_SEC = 5
INITIAL_BAL = 20.50
READ_TAIL   = 20000  # lignes lues depuis la fin du log

# ── Couleurs ANSI ─────────────────────────────────────────────────────────────
R  = "\033[91m"   # rouge
G  = "\033[92m"   # vert
Y  = "\033[93m"   # jaune
B  = "\033[94m"   # bleu
M  = "\033[95m"   # magenta
C  = "\033[96m"   # cyan
W  = "\033[97m"   # blanc
DIM= "\033[2m"
RST= "\033[0m"
BOLD="\033[1m"
CLR= "\033[2J\033[H"  # clear screen

# ── Regex ─────────────────────────────────────────────────────────────────────
RE_DBE   = re.compile(
    r"pid=(\d+).*\[DBE_V2_FINAL\]\s+(W\d+\s+\w+)\s*\|"
    r".*?'name':\s*'([^']+)'.*?"
    r"Regime=(\w+).*?"
    r"SL=([\d.]+)%.*?TP=([\d.]+)%"
)
RE_OPEN  = re.compile(
    r"pid=(\d+).*\[TRADE_OPEN\]\s+(\w+)\s+size=([\d.]+)\s+notional=([\d.]+)"
    r"\s+SL=([\d.]+)%\s+TP=([\d.]+)%\s+tier=([^\[]+)"
)
RE_CLOSE = re.compile(
    r"pid=(\d+).*\[POSITION FERM[EÉ]+\]\s+(\w+):.*?PnL:\s*\$([+\-][\d.]+)"
)
RE_CASH  = re.compile(r"pid=(\d+).*cash_balance\s*\|\s*([\d.]+)")
RE_KELLY = re.compile(r"pid=(\d+).*\[KELLY_CLAMPED\].*notional=\$([\d.]+)")
RE_GATE  = re.compile(r"pid=(\d+).*\[RISK_GATE\]")
RE_WORKER_ID = re.compile(r"\[Worker\s+(\d+)\]")

# ── Helpers ───────────────────────────────────────────────────────────────────
WORKER_NAMES = {
    "W1": "Scalper",
    "W2": "Intraday",
    "W3": "Swing",
    "W4": "Position",
}

def tail_lines(path: Path, n: int) -> list[str]:
    """Lit les n dernières lignes d'un fichier volumineux efficacement."""
    if not path.exists():
        return []
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            buf  = min(n * 200, size)
            f.seek(max(0, size - buf))
            raw  = f.read()
        return raw.decode("utf-8", errors="ignore").splitlines()[-n:]
    except Exception:
        return []

def read_result_json(worker_key: str) -> dict:
    """Lit le dernier résultat Ray Tune pour un worker (ex: 'd585c_00001')."""
    dirs = list(RAY_RESULTS.glob(f"ADAN_PBT_Worker_{worker_key}*"))
    if not dirs:
        return {}
    rfile = dirs[0] / "result.json"
    if not rfile.exists():
        return {}
    try:
        with open(rfile) as f:
            lines = f.readlines()
        for line in reversed(lines):
            try:
                return json.loads(line)
            except Exception:
                continue
    except Exception:
        pass
    return {}

def color_pnl(val: float) -> str:
    s = f"{val:+.2f}"
    return (G if val > 0 else R if val < 0 else DIM) + s + RST

def color_pct(val: float) -> str:
    s = f"{val:+.1f}%"
    return (G if val > 0 else R if val < 0 else DIM) + s + RST

def color_sharpe(val: float) -> str:
    s = f"{val:.2f}"
    if val >= 1.5:  return G + BOLD + s + RST
    if val >= 0.5:  return G + s + RST
    if val >= 0.0:  return Y + s + RST
    return R + s + RST

def bar(val: float, lo: float, hi: float, width: int = 8) -> str:
    """Mini barre de progression."""
    pct = max(0.0, min(1.0, (val - lo) / max(hi - lo, 1e-9)))
    filled = int(pct * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"

# ── Parsing du log ────────────────────────────────────────────────────────────
def parse_log(lines: list[str]) -> dict:
    """
    Retourne un dict par PID avec les infos les plus récentes.
    Structure: {pid: {worker, tier, regime, sl, tp, open_pos, last_pnls, cash, kelly_notional, risk_gates}}
    """
    state: dict = defaultdict(lambda: {
        "worker": "?", "tier": "?", "regime": "?",
        "sl": 0.0, "tp": 0.0,
        "open_pos": 0,
        "last_pnls": [],   # 3 derniers PnL fermés
        "cash": None,
        "kelly_notional": None,
        "risk_gates": 0,
        "last_open": None,  # dernier trade ouvert (asset, notional, sl, tp)
    })

    open_count: dict = defaultdict(int)   # pid -> nb positions actuellement ouvertes

    for line in lines:
        # DBE → tier, regime, SL, TP
        m = RE_DBE.search(line)
        if m:
            pid, wname, tier, regime, sl, tp = m.groups()
            s = state[pid]
            s["worker"]  = wname.strip()
            s["tier"]    = tier
            s["regime"]  = regime
            s["sl"]      = float(sl)
            s["tp"]      = float(tp)
            continue

        # TRADE_OPEN
        m = RE_OPEN.search(line)
        if m:
            pid, asset, size, notional, sl, tp, tier = m.groups()
            open_count[pid] += 1
            state[pid]["last_open"] = {
                "asset": asset, "notional": float(notional),
                "sl": float(sl), "tp": float(tp),
            }
            continue

        # POSITION FERMÉE → PnL
        m = RE_CLOSE.search(line)
        if m:
            pid, asset, pnl = m.groups()
            pnl_f = float(pnl)
            lst = state[pid]["last_pnls"]
            lst.append(pnl_f)
            if len(lst) > 3:
                lst.pop(0)
            # fermeture = -1 position ouverte
            if open_count[pid] > 0:
                open_count[pid] -= 1
            continue

        # cash_balance
        m = RE_CASH.search(line)
        if m:
            pid, cash = m.groups()
            state[pid]["cash"] = float(cash)
            continue

        # KELLY_CLAMPED
        m = RE_KELLY.search(line)
        if m:
            pid, notional = m.groups()
            state[pid]["kelly_notional"] = float(notional)
            continue

        # RISK_GATE
        m = RE_GATE.search(line)
        if m:
            pid = m.group(1)
            state[pid]["risk_gates"] += 1

    # Injecter le compte de positions ouvertes
    for pid, cnt in open_count.items():
        state[pid]["open_pos"] = cnt

    return dict(state)

# ── Résolution PID → worker ───────────────────────────────────────────────────
# On mappe les PIDs aux workers via le champ "worker" parsé depuis DBE
def resolve_workers(state: dict) -> dict:
    """
    Regroupe par nom de worker (W1 Scalper, etc.) en gardant le PID le plus actif.
    """
    by_worker: dict = {}
    for pid, data in state.items():
        wname = data.get("worker", "?")
        if wname == "?":
            continue
        # Garder le PID qui a le plus de données (last_pnls ou cash)
        existing = by_worker.get(wname)
        if existing is None:
            by_worker[wname] = {**data, "pid": pid}
        else:
            # Préférer celui avec cash connu
            if data.get("cash") is not None and existing.get("cash") is None:
                by_worker[wname] = {**data, "pid": pid}
    return by_worker

# ── Résultats Ray Tune ────────────────────────────────────────────────────────
RAY_WORKER_KEYS = {
    "W1 Scalper":   "00000",
    "W2 Intraday":  "00001",
    "W3 Swing":     "00002",
    "W4 Position":  "00003",
}

def get_ray_metrics() -> dict:
    """Cherche le bon préfixe de run (ex: d585c) et lit les 4 result.json."""
    # Trouver le run le plus récent
    all_dirs = sorted(RAY_RESULTS.glob("ADAN_PBT_Worker_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not all_dirs:
        return {}
    # Extraire le préfixe (5 chars après ADAN_PBT_Worker_)
    prefix_match = re.search(r"ADAN_PBT_Worker_([a-z0-9]{5})_", all_dirs[0].name)
    if not prefix_match:
        return {}
    prefix = prefix_match.group(1)

    result = {}
    for wname, suffix in RAY_WORKER_KEYS.items():
        key = f"{prefix}_{suffix}"
        data = read_result_json(key)
        if data:
            result[wname] = {
                "iteration": data.get("training_iteration", 0),
                "steps":     data.get("training_iteration", 0) * 10000,
                "balance":   data.get("mean_balance", INITIAL_BAL),
                "sharpe":    data.get("mean_sharpe", 0.0),
                "reward":    data.get("mean_reward", 0.0),
            }
    return result

# ── Affichage ─────────────────────────────────────────────────────────────────
WORKER_ORDER = ["W1 Scalper", "W2 Intraday", "W3 Swing", "W4 Position"]

def render(log_state: dict, ray_metrics: dict):
    now = datetime.now().strftime("%H:%M:%S")
    lines_out = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines_out.append(f"{BOLD}{C}{'═'*90}{RST}")
    lines_out.append(
        f"{BOLD}{C}  ADAN0 PBT LIVE MONITOR{RST}  {DIM}{now}{RST}"
        f"   {DIM}refresh {REFRESH_SEC}s  |  log: {LOG_FILE.name}{RST}"
    )
    lines_out.append(f"{BOLD}{C}{'═'*90}{RST}")

    # ── Tableau principal ──────────────────────────────────────────────────────
    hdr = (
        f"  {'WORKER':<14} {'PALIER':<14} {'SOLDE':>8} {'PnL%':>7} "
        f"{'SHARPE':>7} {'REWARD':>8} {'ITER':>5} {'STEPS':>9}"
    )
    lines_out.append(BOLD + W + hdr + RST)
    lines_out.append(f"  {'─'*88}")

    for wname in WORKER_ORDER:
        ray  = ray_metrics.get(wname, {})
        log  = log_state.get(wname, {})

        balance  = ray.get("balance", INITIAL_BAL)
        pnl_pct  = (balance - INITIAL_BAL) / INITIAL_BAL * 100
        sharpe   = ray.get("sharpe", 0.0)
        reward   = ray.get("reward", 0.0)
        iteration= ray.get("iteration", 0)
        steps    = ray.get("steps", 0)
        tier     = log.get("tier", "?")
        regime   = log.get("regime", "?")

        # Couleur du nom selon performance
        if pnl_pct > 20:   wcolor = G + BOLD
        elif pnl_pct > 0:  wcolor = G
        elif pnl_pct > -10:wcolor = Y
        else:               wcolor = R

        row = (
            f"  {wcolor}{wname:<14}{RST}"
            f" {DIM}{tier:<14}{RST}"
            f" {W}${balance:>7.2f}{RST}"
            f" {color_pct(pnl_pct):>14}"
            f" {color_sharpe(sharpe):>14}"
            f" {(G if reward > 0 else R)}{reward:>+8.2f}{RST}"
            f" {DIM}{iteration:>5}{RST}"
            f" {DIM}{steps:>9,}{RST}"
        )
        lines_out.append(row)

    lines_out.append(f"  {'─'*88}")

    # ── Détails par worker ─────────────────────────────────────────────────────
    lines_out.append("")
    lines_out.append(f"{BOLD}{W}  DÉTAILS TEMPS RÉEL (depuis logs){RST}")
    lines_out.append(f"  {'─'*88}")

    detail_hdr = (
        f"  {'WORKER':<14} {'RÉGIME':<10} {'SL':>6} {'TP':>6} "
        f"{'POS.OUVERTES':>13} {'KELLY $':>9} {'RISK_GATE':>10}  "
        f"{'3 DERNIERS PnL'}"
    )
    lines_out.append(BOLD + DIM + detail_hdr + RST)
    lines_out.append(f"  {'─'*88}")

    for wname in WORKER_ORDER:
        log = log_state.get(wname, {})

        regime   = log.get("regime", "—")
        sl       = log.get("sl", 0.0)
        tp       = log.get("tp", 0.0)
        open_pos = log.get("open_pos", 0)
        kelly    = log.get("kelly_notional")
        gates    = log.get("risk_gates", 0)
        pnls     = log.get("last_pnls", [])

        # Couleur SL/TP selon conformité (Micro Capital: SL 2%, TP 4%)
        sl_ok = 1.5 <= sl <= 5.5
        tp_ok = 3.0 <= tp <= 10.0
        sl_str = (G if sl_ok else R) + f"{sl:.2f}%" + RST
        tp_str = (G if tp_ok else R) + f"{tp:.2f}%" + RST

        # Positions ouvertes (max 1 en Micro Capital)
        pos_str = (G if open_pos <= 1 else R + BOLD) + f"{open_pos}" + RST

        # Kelly notional
        kelly_str = f"${kelly:.2f}" if kelly else "  —  "

        # RISK_GATE (0 = parfait)
        gate_str = (G + "0" if gates == 0 else R + str(gates)) + RST

        # 3 derniers PnL
        pnl_parts = []
        for p in pnls[-3:]:
            pnl_parts.append(color_pnl(p))
        pnl_str = "  ".join(pnl_parts) if pnl_parts else f"{DIM}aucun encore{RST}"

        row = (
            f"  {BOLD}{wname:<14}{RST}"
            f" {DIM}{regime:<10}{RST}"
            f" {sl_str:>13}"
            f" {tp_str:>13}"
            f" {pos_str:>20}"
            f" {DIM}{kelly_str:>9}{RST}"
            f" {gate_str:>17}"
            f"  {pnl_str}"
        )
        lines_out.append(row)

    # ── Dernière position ouverte par worker ───────────────────────────────────
    lines_out.append("")
    lines_out.append(f"{BOLD}{W}  DERNIÈRE POSITION OUVERTE{RST}")
    lines_out.append(f"  {'─'*88}")

    for wname in WORKER_ORDER:
        log  = log_state.get(wname, {})
        last = log.get("last_open")
        if last:
            row = (
                f"  {BOLD}{wname:<14}{RST}"
                f"  {C}{last['asset']}{RST}"
                f"  notional={W}${last['notional']:.2f}{RST}"
                f"  SL={R}{last['sl']:.2f}%{RST}"
                f"  TP={G}{last['tp']:.2f}%{RST}"
            )
        else:
            row = f"  {DIM}{wname:<14}  aucune position ouverte récente{RST}"
        lines_out.append(row)

    # ── Footer ────────────────────────────────────────────────────────────────
    lines_out.append("")
    lines_out.append(f"{DIM}  Légende: {G}■{RST}{DIM} positif  {R}■{RST}{DIM} négatif  "
                     f"{Y}■{RST}{DIM} attention  "
                     f"SL/TP en {G}vert{RST}{DIM} = dans les bornes  "
                     f"RISK_GATE={G}0{RST}{DIM} = aucun dépassement{RST}")
    lines_out.append(f"{BOLD}{C}{'═'*90}{RST}")
    lines_out.append(f"{DIM}  Ctrl+C pour quitter{RST}")

    return "\n".join(lines_out)

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print(f"{C}Démarrage du monitor ADAN0... (log: {LOG_FILE}){RST}")
    if not LOG_FILE.exists():
        print(f"{R}⚠  Log introuvable: {LOG_FILE}{RST}")
        print(f"{Y}   Lance l'entraînement d'abord, puis relance ce script.{RST}")

    try:
        while True:
            lines   = tail_lines(LOG_FILE, READ_TAIL)
            raw_state = parse_log(lines)
            log_state = resolve_workers(raw_state)
            ray_metrics = get_ray_metrics()

            output = render(log_state, ray_metrics)
            # Clear + affichage atomique
            sys.stdout.write(CLR + output + "\n")
            sys.stdout.flush()

            time.sleep(REFRESH_SEC)

    except KeyboardInterrupt:
        print(f"\n{Y}Monitor arrêté.{RST}")

if __name__ == "__main__":
    main()
