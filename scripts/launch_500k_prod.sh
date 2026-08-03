#!/usr/bin/env bash
# Launch the authorized V25 500k run in single-worker sandbox mode.
# All runtime artifacts stay under ADAN0. The launcher refuses concurrent runs
# and never kills unrelated processes.
set -euo pipefail

ROOT="/home/ubuntu/webapp/MORNINGSTAR/ADAN0"
PY="$ROOT/../miniconda3/envs/trading_env/bin/python"
RUNTIME_DIR="$ROOT/logs/training/500k_runtime"
CURRENT_PID="$RUNTIME_DIR/current.pid"
CURRENT_MANIFEST="$RUNTIME_DIR/current_manifest.json"
STEPS="${STEPS:-500000}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: scripts/launch_500k_prod.sh [--dry-run]

Environment:
  STEPS=N   Override 500000 only for launcher validation/smoke tests.
EOF
}

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; usage >&2; exit 2 ;;
    esac
done

cd "$ROOT"
[[ "$PWD" == "$ROOT" ]] || { echo "Refusing to run outside $ROOT" >&2; exit 1; }
[[ -x "$PY" ]] || { echo "Python environment unavailable: $PY" >&2; exit 1; }
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "STEPS must be a positive integer" >&2; exit 2; }

# Gate authorization is machine-checked, not inferred from a completed process.
"$PY" - <<'PY'
import json
from pathlib import Path

root = Path.cwd()
arena = json.loads((root / "reports/arena_supervised_bulletin_v24.json").read_text())["arena_verdict"]
lifecycle = json.loads((root / "reports/v25_finance_smoke_lifecycle.json").read_text())
finance = json.loads((root / "reports/v25_finance_smoke_finance.json").read_text())
checks = {
    "Arena": arena.get("status") == "GREEN" and arena.get("authorization") == "ARENA_GATE_PASSED",
    "Lifecycle": lifecycle.get("ok") is True and not lifecycle.get("violations") and not lifecycle.get("unclosed_positions"),
    "Finance": finance.get("status") == "GREEN" and finance.get("authorization") == "FINANCE_GATE_PASSED",
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit("500k NOT AUTHORIZED; failed gates: " + ", ".join(failed))
print("Preflight gates: Arena=GREEN Lifecycle=GREEN Finance=GREEN")
PY

# Refuse an already-running managed job; never use broad pkill patterns.
if [[ -f "$CURRENT_PID" ]]; then
    old_pid="$(cat "$CURRENT_PID" 2>/dev/null || true)"
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
        old_cmd="$(tr '\0' ' ' < "/proc/$old_pid/cmdline" 2>/dev/null || true)"
        echo "Refusing concurrent 500k run: pid=$old_pid command=$old_cmd" >&2
        exit 3
    fi
fi

RUN_ID="v25_$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$ROOT/logs/training/adan_500k_${RUN_ID}.log"
TRACE="$ROOT/logs/action_pipeline/adan_500k_${RUN_ID}_w{worker_id}.jsonl"
ACTIONDIM="$ROOT/logs/training/actiondim_500k_${RUN_ID}.csv"
DIAG="$ROOT/logs/training/diag_500k_${RUN_ID}.csv"
CHECKPOINT="$ROOT/checkpoints/ppo_adan0_FA_500k_${RUN_ID}.zip"
MANIFEST="$RUNTIME_DIR/${RUN_ID}.json"

mkdir -p "$RUNTIME_DIR" "$ROOT/logs/training" "$ROOT/logs/action_pipeline" "$ROOT/checkpoints"

COMMAND=(
    nice -n 10 taskset -c 1-3 "$PY" scripts/train_parallel_agents.py
    --mode sandbox
    --steps "$STEPS"
    --profiles scalper
    --checkpoint-out "$CHECKPOINT"
)

if (( DRY_RUN )); then
    cat <<EOF
DRY RUN — no process launched
mode=sandbox (single worker; no Ray/PBT mutation)
steps=$STEPS
ADAN_USE_SDE=0
ADAN_LOG_STD_INIT=-1.0
ADAN_ENT_COEF=0.0
log=$LOG
trace=$TRACE
actiondim=$ACTIONDIM
diagnostic=$DIAG
checkpoint=$CHECKPOINT
command=${COMMAND[*]}
EOF
    exit 0
fi

export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3 ADAN_NUM_THREADS=3
export OMP_DYNAMIC=FALSE KMP_BLOCKTIME=0
export ADAN_USE_SDE=0 ADAN_LOG_STD_INIT=-1.0 ADAN_ENT_COEF=0.0
export ADAN_TRAINING_SILENT=1 ADAN_N_EPOCHS=10 ADAN_CKPT_FREQ=10000
export ADAN_PIPELINE_TRACE_PATH="$TRACE"
export ADAN_ACTIONDIM=1 ADAN_ACTIONDIM_CSV="$ACTIONDIM" ADAN_ACTIONDIM_EVERY=1
export ADAN_DIAG_COLLAPSE=1 ADAN_DIAG_CSV="$DIAG" ADAN_DIAG_EVERY=10000
export ADAN_COLLAPSE_BREAKER=0

nohup "${COMMAND[@]}" >"$LOG" 2>&1 &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$CURRENT_PID"

RUN_ID="$RUN_ID" TRAIN_PID="$TRAIN_PID" STEPS="$STEPS" LOG="$LOG" TRACE="$TRACE" \
ACTIONDIM="$ACTIONDIM" DIAG="$DIAG" CHECKPOINT="$CHECKPOINT" MANIFEST="$MANIFEST" \
"$PY" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

manifest = {
    "run_id": os.environ["RUN_ID"],
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "pid": int(os.environ["TRAIN_PID"]),
    "status": "STARTING",
    "mode": "sandbox",
    "profile": "scalper",
    "steps": int(os.environ["STEPS"]),
    "exploration": {"use_sde": False, "log_std_init": -1.0, "ent_coef": 0.0},
    "paths": {
        "log": os.environ["LOG"],
        "pipeline_trace_pattern": os.environ["TRACE"],
        "actiondim_csv": os.environ["ACTIONDIM"],
        "diagnostic_csv": os.environ["DIAG"],
        "checkpoint": os.environ["CHECKPOINT"],
    },
}
path = Path(os.environ["MANIFEST"])
path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
Path("logs/training/500k_runtime/current_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
PY

sleep 5
if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "500k process exited during startup; inspect $LOG" >&2
    tail -80 "$LOG" >&2 || true
    exit 4
fi

printf '500k launched: run_id=%s pid=%s\n' "$RUN_ID" "$TRAIN_PID"
printf 'log=%s\nmanifest=%s\ncheckpoint=%s\n' "$LOG" "$MANIFEST" "$CHECKPOINT"
printf 'monitor: scripts/live.sh\n'
