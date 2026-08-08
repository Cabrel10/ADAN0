#!/usr/bin/env bash
# Launch the authorized 500k run with the evidence-backed V16 correction:
# DiagGaussian + mark-to-market reward + loss-level L2 action anchor.
# All runtime artifacts stay under ADAN0; unrelated processes are never killed.
set -euo pipefail

ROOT="${ROOT:-/home/ubuntu/webapp/MORNINGSTAR/ADAN0}"
PY="${PY:-$ROOT/../miniconda3/envs/trading_env/bin/python}"
RUNTIME_DIR="${RUNTIME_DIR:-$ROOT/logs/training/500k_runtime}"
CURRENT_PID="${CURRENT_PID:-$RUNTIME_DIR/current.pid}"
# V26 current_manifest.json is a protected historical artifact. V27 writes its
# own live pointer and never overwrites the V26 manifest.
CURRENT_MANIFEST="${CURRENT_MANIFEST:-$RUNTIME_DIR/current_v27_manifest.json}"
STEPS="${STEPS:-500000}"
DRY_RUN=0
MONITOR_PID=""

usage() {
    cat <<'EOF'
Usage: scripts/launch_500k_prod.sh [--dry-run]

Environment:
  STEPS=N   Override 500000 only for launcher validation.
EOF
}

# Internal detached lifecycle monitor. It finalizes both manifests after the
# managed training process exits, including breaker stops that return code 0.
if [[ "${1:-}" == "--monitor" ]]; then
    MONITOR_PID="${2:-}"
    [[ "$MONITOR_PID" =~ ^[0-9]+$ ]] || exit 2
    while kill -0 "$MONITOR_PID" 2>/dev/null; do sleep 2; done
    cd "$ROOT"
    "$PY" - <<'PYFINAL'
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path(os.environ["MANIFEST"])
log_path = Path(os.environ["LOG"])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
requested = int(manifest["requested_timesteps"])

cumulative = None
matches = re.findall(r'"cumulative_steps"\s*:\s*(\d+)', log)
if matches:
    cumulative = int(matches[-1])
else:
    matches = re.findall(r"total_timesteps\s*\|\s*([0-9.eE+-]+)", log)
    if matches:
        try:
            cumulative = int(float(matches[-1]))
        except ValueError:
            pass

stop_reason = None
if "[CRITIC-BREAKER] Training STOPPED" in log:
    status = "STOPPED_BY_CRITIC_BREAKER"
    match = re.findall(r"\[CRITIC-BREAKER\].*?—\s*(.*?)\.\s*Inspect", log)
    stop_reason = match[-1] if match else "critic breaker"
elif "[COLLAPSE-BREAKER] Training STOPPED" in log:
    status = "STOPPED_BY_COLLAPSE_BREAKER"
    stop_reason = "directional collapse breaker"
elif cumulative is not None and cumulative >= requested:
    status = "COMPLETED"
elif "Traceback (most recent call last)" in log or "CUDA out of memory" in log:
    status = "FAILED"
    stop_reason = "runtime exception; inspect log"
else:
    status = "EXITED_EARLY"
    stop_reason = "process exited before requested timesteps; inspect log"

manifest.update({
    "status": status,
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "reached_timesteps": cumulative,
    "last_completed_rollout_timesteps": (
        cumulative - (cumulative % 512) if cumulative is not None else None
    ),
    "stop_reason": stop_reason,
})
serialized = json.dumps(manifest, indent=2) + "\n"
manifest_path.write_text(serialized, encoding="utf-8")
Path(os.environ["CURRENT_MANIFEST"]).write_text(serialized, encoding="utf-8")

pid_file = Path(os.environ["CURRENT_PID"])
try:
    if pid_file.read_text().strip() == os.environ["TRAIN_PID"]:
        pid_file.unlink()
except FileNotFoundError:
    pass
PYFINAL
    exit 0
fi

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

"$PY" - <<'PYGATE'
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
PYGATE

if [[ -f "$CURRENT_PID" ]]; then
    old_pid="$(cat "$CURRENT_PID" 2>/dev/null || true)"
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
        old_cmd="$(tr '\0' ' ' < "/proc/$old_pid/cmdline" 2>/dev/null || true)"
        echo "Refusing concurrent 500k run: pid=$old_pid command=$old_cmd" >&2
        exit 3
    fi
fi

RUN_ID="v27_hmm_semantic_$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$ROOT/logs/training/adan_500k_${RUN_ID}.log"
TRACE="$ROOT/logs/action_pipeline/adan_500k_${RUN_ID}_w{worker_id}.jsonl"
ACTIONDIM="$ROOT/logs/training/actiondim_500k_${RUN_ID}.csv"
DIAG="$ROOT/logs/training/diag_500k_${RUN_ID}.csv"
REWARD_TELEM="$ROOT/logs/training/reward_components_500k_${RUN_ID}.csv"
CHECKPOINT="$ROOT/checkpoints/ppo_adan0_FA_500k_${RUN_ID}.zip"
CKPT_PREFIX="ppo_adan0_${RUN_ID}_checkpoint"
MANIFEST="$RUNTIME_DIR/${RUN_ID}.json"

mkdir -p "$RUNTIME_DIR" "$ROOT/logs/training" "$ROOT/logs/action_pipeline" "$ROOT/checkpoints"

COMMAND=(
    nice -n 10 taskset -c 1-3 "$PY" scripts/train_parallel_agents.py
    --mode sandbox
    --steps "$STEPS"
    --profiles scalper
    --config config/config.yaml
    --checkpoint-out "$CHECKPOINT"
)

if (( DRY_RUN )); then
    cat <<EOF
DRY RUN — no process launched
mode=sandbox (single worker; no Ray/PBT mutation)
steps=$STEPS
algorithm=WorldModelPPO
ADAN_USE_SDE=0
ADAN_LOG_STD_INIT=-1.0
ADAN_ENT_COEF=0.05
ADAN_N_EPOCHS=4
ADAN_MTM_REWARD=1
ADAN_L2_ANCHOR_LAMBDA=0.05
ADAN_AUX_LOSS_COEF=0.0
ADAN_CRITIC_BREAKER=1
ADAN_COLLAPSE_BREAKER=1
ADAN_DIAG_EVERY=512
ADAN_REWARD_TELEM=1
ADAN_SAVE_SCALERS=0
ADAN_ARENA_COLLECT=1
ADAN_DISABLE_EV_FEE_GATE=0
log=$LOG
trace=$TRACE
actiondim=$ACTIONDIM
diagnostic=$DIAG
reward_telemetry=$REWARD_TELEM
checkpoint=$CHECKPOINT
checkpoint_prefix=$CKPT_PREFIX
command=${COMMAND[*]}
EOF
    exit 0
fi

export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONPATH="src:."
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3 ADAN_NUM_THREADS=3
export OMP_DYNAMIC=FALSE KMP_BLOCKTIME=0
export ADAN_USE_SDE=0 ADAN_LOG_STD_INIT=-1.0 ADAN_ENT_COEF=0.05
export ADAN_N_EPOCHS=4 ADAN_CKPT_FREQ=10000 ADAN_CKPT_PREFIX="$CKPT_PREFIX"
export ADAN_MTM_REWARD=1 ADAN_L2_ANCHOR_LAMBDA=0.05 ADAN_AUX_LOSS_COEF=0.0
export ADAN_CRITIC_BREAKER=1 ADAN_CRITIC_EV_MIN=-0.2 ADAN_CRITIC_EV_WINDOWS=10
export ADAN_CRITIC_VALUE_LOSS_MAX=1000000 ADAN_COLLAPSE_BREAKER=1
export ADAN_TRAINING_SILENT=1 ADAN_PIPELINE_TRACE_PATH="$TRACE"
export ADAN_ACTIONDIM=1 ADAN_ACTIONDIM_CSV="$ACTIONDIM" ADAN_ACTIONDIM_EVERY=1
export ADAN_DIAG_COLLAPSE=1 ADAN_DIAG_CSV="$DIAG" ADAN_DIAG_EVERY=512
export ADAN_REWARD_TELEM=1 ADAN_REWARD_TELEM_EVERY=100 ADAN_REWARD_TELEM_CSV="$REWARD_TELEM"
export ADAN_SAVE_SCALERS=0 ADAN_ARENA_COLLECT=1 ADAN_DISABLE_EV_FEE_GATE=0

nohup "${COMMAND[@]}" >"$LOG" 2>&1 &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$CURRENT_PID"

RUN_ID="$RUN_ID" TRAIN_PID="$TRAIN_PID" STEPS="$STEPS" LOG="$LOG" TRACE="$TRACE" \
ACTIONDIM="$ACTIONDIM" DIAG="$DIAG" REWARD_TELEM="$REWARD_TELEM" \
CHECKPOINT="$CHECKPOINT" CKPT_PREFIX="$CKPT_PREFIX" MANIFEST="$MANIFEST" \
"$PY" - <<'PYMANIFEST'
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
    "requested_timesteps": int(os.environ["STEPS"]),
    "correction_basis": "V27 semantic HMM posteriors plus validated V16 MTM anchor baseline",
    "exploration": {"use_sde": False, "distribution": "DiagGaussian", "log_std_init": -1.0, "ent_coef": 0.05},
    "ppo": {"class": "WorldModelPPO", "n_steps": 512, "n_epochs": 4, "checkpoint_freq": 10000},
    "reward": {"mtm_enabled": True, "l2_anchor_lambda": 0.05, "aux_loss_coef": 0.0},
    "critic_breaker": {"enabled": True, "ev_min": -0.2, "windows": 10, "value_loss_max": 1000000},
    "collapse_breaker": {"enabled": True},
    "diagnostics": {"every": 512, "reward_telemetry_every": 100, "arena_collect": True},
    "hmm": {"states": 3, "min_obs": 60, "window": 500, "semantic_order": ["bull", "sideways", "bear"]},
    "scalers": {"persisted": True, "refit_allowed": False, "save_at_shutdown": False},
    "paths": {
        "log": os.environ["LOG"],
        "pipeline_trace_pattern": os.environ["TRACE"],
        "actiondim_csv": os.environ["ACTIONDIM"],
        "diagnostic_csv": os.environ["DIAG"],
        "reward_telemetry_csv": os.environ["REWARD_TELEM"],
        "checkpoint": os.environ["CHECKPOINT"],
        "checkpoint_prefix": os.environ["CKPT_PREFIX"],
    },
}
serialized = json.dumps(manifest, indent=2) + "\n"
Path(os.environ["MANIFEST"]).write_text(serialized, encoding="utf-8")
Path(os.environ["CURRENT_MANIFEST"]).write_text(serialized, encoding="utf-8")
PYMANIFEST

sleep 5
if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "500k process exited during startup; inspect $LOG" >&2
    tail -80 "$LOG" >&2 || true
    exit 4
fi

# The detached monitor inherits only explicit paths and finalizes lifecycle state.
nohup env MANIFEST="$MANIFEST" LOG="$LOG" CURRENT_MANIFEST="$CURRENT_MANIFEST" \
CURRENT_PID="$CURRENT_PID" TRAIN_PID="$TRAIN_PID" \
"$0" --monitor "$TRAIN_PID" >/dev/null 2>&1 &
MONITOR_PID=$!

MANIFEST="$MANIFEST" CURRENT_MANIFEST="$CURRENT_MANIFEST" MONITOR_PID="$MONITOR_PID" \
"$PY" - <<'PYRUNNING'
import json
import os
from pathlib import Path
path = Path(os.environ["MANIFEST"])
manifest = json.loads(path.read_text(encoding="utf-8"))
manifest["status"] = "RUNNING"
manifest["monitor_pid"] = int(os.environ["MONITOR_PID"])
serialized = json.dumps(manifest, indent=2) + "\n"
path.write_text(serialized, encoding="utf-8")
Path(os.environ["CURRENT_MANIFEST"]).write_text(serialized, encoding="utf-8")
PYRUNNING

printf '500k launched: run_id=%s pid=%s monitor_pid=%s\n' "$RUN_ID" "$TRAIN_PID" "$MONITOR_PID"
printf 'log=%s\nmanifest=%s\ncheckpoint=%s\n' "$LOG" "$MANIFEST" "$CHECKPOINT"
printf 'monitor: scripts/live.sh\n'
