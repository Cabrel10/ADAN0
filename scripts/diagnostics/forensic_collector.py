#!/usr/bin/env python3
"""
Collecteur forensique du freeze 12417 — version ptrace-safe (Session 19+).

Contexte env: ptrace_scope=1 + pas de root => strace/py-spy en ATTACH refuses.
Contournement: py-spy peut tracer un process qu'il a LUI-MEME lance (record).
Donc on lance le training SOUS `py-spy record` (traceur autorise) qui produit
un flamegraph: au moment du freeze, la fonction bloquante domine le profil.

En parallele, ce collecteur surveille SANS ptrace:
  - /proc/PID/wchan      (fonction kernel d'attente: futex_wait / fsync / io_schedule)
  - /proc/PID/stat       (state R/S/D, CPU ticks, RSS)
  - /proc/PID/io         (rchar/wchar/syscw: debit I/O)
  - iostat -xz           (saturation disque globale)
  - free / vmstat        (memoire)
Et detecte le freeze (step fige + log fige) pour horodater la batterie.

Ce script NE LANCE PAS le training (fait par le launcher). Il s'attache en
LECTURE /proc au PID donne + collecte wchan/state/io. py-spy record tourne
separement (lance par le launcher comme parent).

Usage:
    python forensic_collector.py <LOGFILE> <TRAIN_PID> [freeze_s=90] [poll_s=10]
"""
import os
import sys
import time
import subprocess
import datetime

ROOT = "/home/ubuntu/webapp/MORNINGSTAR/ADAN0"


def now():
    return datetime.datetime.now().strftime("%H:%M:%S")


def run(cmd, timeout=30):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT {timeout}s] {cmd}\n"
    except Exception as e:
        return f"[ERR] {cmd}: {e}\n"


def last_step(logfile):
    out = run(f"tail -n 4000 '{logfile}' | grep -oE '\\[STEP [0-9]+' | "
              f"grep -oE '[0-9]+' | tail -1", timeout=15).strip()
    return int(out) if out.isdigit() else -1


def proc_alive(pid):
    return os.path.exists(f"/proc/{pid}")


def proc_stat(pid):
    """(state, utime+stime ticks, rss_kb)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            raw = f.read()
        after = raw[raw.rindex(")") + 2:].split()
        state = after[0]
        utime = int(after[11]); stime = int(after[12])
        rss_pages = int(after[21])
        rss_kb = rss_pages * (os.sysconf("SC_PAGE_SIZE") // 1024)
        return state, utime + stime, rss_kb
    except Exception:
        return "?", -1, -1


def read_wchan(pid):
    try:
        with open(f"/proc/{pid}/wchan") as f:
            return f.read().strip() or "(running)"
    except Exception:
        return "?"


def all_threads_wchan(pid):
    lines = []
    try:
        for t in sorted(os.listdir(f"/proc/{pid}/task")):
            try:
                with open(f"/proc/{pid}/task/{t}/stat") as f:
                    st = f.read().split()
                # state = champ apres comm; reparse robuste
                with open(f"/proc/{pid}/task/{t}/stat") as f:
                    raw = f.read()
                state = raw[raw.rindex(")") + 2:].split()[0]
            except Exception:
                state = "?"
            try:
                with open(f"/proc/{pid}/task/{t}/wchan") as f:
                    wc = f.read().strip() or "(running)"
            except Exception:
                wc = "?"
            lines.append(f"tid={t} state={state} wchan={wc}")
    except Exception as e:
        lines.append(f"[err {e}]")
    return "\n".join(lines)


def collect_battery(pid, outdir):
    os.makedirs(outdir, exist_ok=True)

    def save(name, content):
        with open(os.path.join(outdir, name), "w") as f:
            f.write(content)

    print(f"[{now()}] === BATTERIE FORENSIQUE pid={pid} -> {outdir} ===",
          flush=True)

    save("01_ps.txt", run(
        "ps -eo pid,ppid,etime,%cpu,%mem,stat,nlwp,cmd "
        "| grep -E 'train_parallel|py-spy|PID' | grep -v grep", 20))
    save("02_top_threads.txt", run(f"top -H -b -n 1 -p {pid} 2>&1 | head -70", 25))
    save("03_proc_status.txt", run(f"cat /proc/{pid}/status 2>&1", 10))
    save("04_proc_io.txt", run(f"cat /proc/{pid}/io 2>&1", 10))
    # PREUVE CLE #1: wchan de tous les threads (futex/fsync/io_schedule)
    save("05_threads_wchan.txt", all_threads_wchan(pid))
    # PREUVE CLE #2: iostat saturation
    save("06_iostat.txt", run("iostat -xz 1 4 2>&1", 14))
    save("07_free_vmstat.txt", run("free -m; echo '---vmstat---'; vmstat 1 3", 10))
    # checkpoints
    save("08_checkpoints.txt", run(
        "echo '== du =='; du -sh checkpoints 2>&1; "
        "echo '== ls tail =='; ls -lht checkpoints 2>&1 | head -15; "
        "echo '== count zip =='; find checkpoints -name '*.zip' 2>/dev/null | wc -l; "
        "echo '== disk =='; df -h .; echo '== inodes =='; df -i .", 20))
    # tentative py-spy dump (echouera si record tient deja le ptrace, mais on essaie)
    save("09_pyspy_attach_try.txt", run(
        f"timeout 20 /home/ubuntu/webapp/MORNINGSTAR/miniconda3/envs/"
        f"trading_env/bin/py-spy dump --pid {pid} 2>&1", 25))
    print(f"[{now()}] === BATTERIE OK -> {outdir} ===", flush=True)


def main():
    if len(sys.argv) < 3:
        print("usage: forensic_collector.py <LOGFILE> <PID> [freeze_s=90] [poll_s=10]")
        sys.exit(1)
    logfile, pid = sys.argv[1], sys.argv[2]
    freeze_s = int(sys.argv[3]) if len(sys.argv) > 3 else 90
    poll_s = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(ROOT, "logs", "forensic", ts)
    os.makedirs(base, exist_ok=True)
    timeline = os.path.join(base, "timeline.csv")
    with open(timeline, "w") as f:
        f.write("time,step,state,cpu_ticks_delta,rss_mb,wchan,log_age_s,step_frozen_s\n")

    print(f"[{now()}] forensic start pid={pid} freeze_thr={freeze_s}s out={base}",
          flush=True)

    prev_step = prev_ticks = -1
    prev_logsize = -1
    step_unchanged_since = None
    log_unchanged_since = None
    captured = False

    while True:
        if not proc_alive(pid):
            print(f"[{now()}] process {pid} DISPARU. Stop.", flush=True)
            with open(timeline, "a") as f:
                f.write(f"{now()},,DEAD,,,,,process_gone\n")
            break

        step = last_step(logfile)
        state, ticks, rss_kb = proc_stat(pid)
        wchan = read_wchan(pid)
        try:
            logsize = os.path.getsize(logfile)
            log_age = time.time() - os.path.getmtime(logfile)
        except Exception:
            logsize, log_age = -1, -1

        ticks_delta = (ticks - prev_ticks) if prev_ticks >= 0 else 0
        rss_mb = rss_kb // 1024 if rss_kb > 0 else -1

        if step == prev_step and step > 0:
            if step_unchanged_since is None:
                step_unchanged_since = time.time()
        else:
            step_unchanged_since = None
        if logsize == prev_logsize:
            if log_unchanged_since is None:
                log_unchanged_since = time.time()
        else:
            log_unchanged_since = None

        step_frozen = (time.time() - step_unchanged_since) if step_unchanged_since else 0
        log_frozen = (time.time() - log_unchanged_since) if log_unchanged_since else 0

        with open(timeline, "a") as f:
            f.write(f"{now()},{step},{state},{ticks_delta},{rss_mb},{wchan},"
                    f"{log_age:.0f},{step_frozen:.0f}\n")
        print(f"[{now()}] step={step} state={state} dCPU={ticks_delta} "
              f"rss={rss_mb}MB wchan={wchan} step_frozen={step_frozen:.0f}s "
              f"log_frozen={log_frozen:.0f}s", flush=True)

        if step_frozen >= freeze_s and log_frozen >= freeze_s and not captured:
            print(f"[{now()}] !!! FREEZE (step={step} fige {step_frozen:.0f}s) "
                  f"wchan={wchan} -> COLLECTE !!!", flush=True)
            collect_battery(pid, os.path.join(base, f"freeze_pid{pid}_t0"))
            captured = True
            time.sleep(45)
            collect_battery(pid, os.path.join(base, f"freeze_pid{pid}_t45"))
            print(f"[{now()}] captures faites; monitoring continue.", flush=True)

        prev_step, prev_ticks, prev_logsize = step, ticks, logsize
        time.sleep(poll_s)

    print(f"[{now()}] forensic END. Preuves: {base}", flush=True)


if __name__ == "__main__":
    main()
