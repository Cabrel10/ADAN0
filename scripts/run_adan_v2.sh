#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# ADAN V2 — LANCEUR DIAGNOSTIQUE PORTABLE (run instrumenté "réveil de SIZE")
# ═══════════════════════════════════════════════════════════════════════════════
# Objet : entraînement COURT et INSTRUMENTÉ (50k-100k steps) pour observer si la
#         moyenne pré-tanh μ(size)=-7.2 remonte vers le centre.
#         -> Cas A : μ remonte (ex. -7.2 -> -2.5) = PPO réapprend, PAS de guard.
#         -> Cas B : μ stagne (~-7) malgré σ≈3 = problème reward/credit-assignment.
#
# Diagnostic basé sur diagnostics/audit_pre_tanh.py :
#   size  μ=-7.20 σ=3.24 -> figée par la MOYENNE (pas l'exploration)
#   tp    μ=+4.13 σ=1.11 -> saturée plafond (Cas B' A7 v2)
#   direction/tf/sl : saines
#
# CE QUI EST ACTIVÉ : ActionDimMonitor (MESURE SEULE, ne modifie rien).
# CE QUI N'EST PAS ACTIVÉ : ActionSaturationGuard (reste en réserve, cf. consigne).
#
# Paramètres V2 (overrides via env, lus par train_parallel_agents.py) :
#   ADAN_USE_SDE        (def 0)     -> 0=DiagGaussian (σ DÉCOUPLÉE des features,
#                                      mathématiquement stable — RECOMMANDÉ 500k) ;
#                                      1=gSDE (state-dependent, dérive possible).
#   ADAN_LOG_STD_INIT   (def -1.0)  -> DiagGaussian: std0≈exp(-1.0)≈0.37.
#                                      ⚠ NE PLUS mettre 0.0/-0.5 : avec gSDE +
#                                      ||features||₂≈11.4 => σ_eff≈6.9 -> DIVERGENCE
#                                      (cf. README CRITICAL FINDING #2).
#   ADAN_USE_EXPLN      (def 1)     -> gSDE seul : borne la croissance de variance.
#   ADAN_ENT_COEF       (def 0.0)   -> 0.02 poussait aussi l'explosion ; 0.0 sûr.
#   ADAN_ACTIONDIM=1                -> active l'instrumentation par tête
#
# MACHINES SUPPORTÉES (auto-détection, override avec --machine) :
#   kali   : poste Kali 16 Go RAM + swap          -> Ray PBT, 2 workers, DummyVec
#   vps    : VPS 8 Go RAM + 4 CPU EPYC            -> Sandbox (PAS de Ray), 1 worker
#   gpu    : machine avec GPU (nvidia-smi présent) -> Ray PBT, 2 workers, GPU
#   mac    : Mac (Darwin) puissant                -> Ray PBT, 2 workers, DummyVec
#   sandbox: fallback (ce conteneur, <8 Go libres) -> Sandbox, 1 worker
#
# OOM : Ray est vicieux côté mémoire. Sur <12 Go on FORCE le mode Sandbox
#       (sans Ray) car 2 trials Ray + object store + spill explosent la RAM.
#       Réf. Ray : object_store_memory, _memory, RAY_memory_usage_threshold.
#
# Usage :
#   scripts/run_adan_v2.sh                       # auto-détection
#   scripts/run_adan_v2.sh --machine kali        # forcer une machine
#   scripts/run_adan_v2.sh --steps 100000        # nb de steps
#   scripts/run_adan_v2.sh --dry-run             # afficher la commande sans lancer
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── 0. Localisation projet (portable, pas de chemin absolu codé en dur) ──────────
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ── Defaults V2 ──────────────────────────────────────────────────────────────────
MACHINE=""                 # auto-détection si vide
STEPS="${ADAN_STEPS:-80000}"          # 50k-100k recommandé
STEPS_PER_ITER="${ADAN_STEPS_PER_ITER:-5000}"
USE_SDE="${ADAN_USE_SDE:-0}"          # 0=DiagGaussian (stable), 1=gSDE
USE_EXPLN="${ADAN_USE_EXPLN:-1}"      # gSDE seul : borne la variance
LOG_STD_INIT="${ADAN_LOG_STD_INIT:--1.0}"   # std0≈0.37 (SAFE) — NE PAS remettre 0.0
ENT_COEF="${ADAN_ENT_COEF:-0.0}"     # 0.0 sûr (0.02 amplifiait l'explosion)
PROFILES="${ADAN_PROFILES:-intraday swing}"   # 2 profils par défaut (diagnostic)
DRY_RUN=0

# ── 1. Args ──────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --machine)        MACHINE="$2"; shift 2;;
    --use-sde)        USE_SDE="$2"; shift 2;;
    --steps)          STEPS="$2"; shift 2;;
    --steps-per-iter) STEPS_PER_ITER="$2"; shift 2;;
    --log-std-init)   LOG_STD_INIT="$2"; shift 2;;
    --ent-coef)       ENT_COEF="$2"; shift 2;;
    --profiles)       PROFILES="$2"; shift 2;;
    --dry-run)        DRY_RUN=1; shift;;
    -h|--help)        sed -n '2,55p' "$0"; exit 0;;
    *) echo "Argument inconnu: $1"; exit 2;;
  esac
done

# ── 2. Détection ressources ──────────────────────────────────────────────────────
OS="$(uname -s)"
if [[ "$OS" == "Darwin" ]]; then
  TOTAL_MB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 ))
  FREE_MB=$TOTAL_MB   # approximation (macOS ne donne pas 'available' simplement)
  NCPU=$(sysctl -n hw.ncpu)
else
  TOTAL_MB=$(free -m | awk '/^Mem:/{print $2}')
  FREE_MB=$(free -m | awk '/^Mem:/{print $7}')   # available
  NCPU=$(nproc)
fi
HAS_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  HAS_GPU=1
fi

# ── 3. Auto-détection machine si non forcée ──────────────────────────────────────
if [[ -z "$MACHINE" ]]; then
  if [[ "$OS" == "Darwin" ]]; then
    MACHINE="mac"
  elif [[ "$HAS_GPU" == "1" ]]; then
    MACHINE="gpu"
  elif [[ -d /mnt/new_data ]] || [[ "$(hostname)" == *kali* ]]; then
    MACHINE="kali"
  elif [[ "$TOTAL_MB" -le 9000 ]]; then
    MACHINE="vps"
  else
    MACHINE="sandbox"
  fi
fi

# ── 4. Garde-fou OOM : Ray interdit sous ~12 Go disponibles ──────────────────────
USE_RAY=1
case "$MACHINE" in
  vps|sandbox) USE_RAY=0;;          # 8 Go -> jamais de Ray (OOM garanti à 2 trials)
  kali|gpu|mac) USE_RAY=1;;
esac
# Surcharge de sécurité : même sur kali, si <12 Go DISPONIBLES, on bascule sandbox.
if [[ "$USE_RAY" == "1" && "$FREE_MB" -lt 12000 ]]; then
  echo "⚠  RAM disponible=${FREE_MB}Mo < 12000Mo -> Ray DÉSACTIVÉ (anti-OOM)."
  echo "   Bascule en mode Sandbox (1 worker, sans object store Ray)."
  USE_RAY=0
fi

NUM_SAMPLES=2   # diagnostic : 2 profils max (PAS 4 — ce n'est pas un sweep HP)
ENVS_PER_WORKER=2

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  ADAN V2 — RUN DIAGNOSTIQUE INSTRUMENTÉ"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  Machine        : $MACHINE   (OS=$OS, GPU=$HAS_GPU)"
echo "  RAM totale/dispo: ${TOTAL_MB}Mo / ${FREE_MB}Mo   CPU=$NCPU"
echo "  Mode exécution : $([[ "$USE_RAY" == "1" ]] && echo "Ray PBT ($NUM_SAMPLES workers, DummyVec)" || echo "Sandbox (1 worker, SANS Ray)")"
echo "  Steps          : $STEPS (par itér: $STEPS_PER_ITER)"
echo "  use_sde        : $USE_SDE ($([[ "$USE_SDE" == "0" ]] && echo "DiagGaussian — STABLE" || echo "gSDE use_expln=$USE_EXPLN"))"
echo "  log_std_init   : $LOG_STD_INIT (std0≈$(awk "BEGIN{printf \"%.2f\", exp($LOG_STD_INIT)}"))"
echo "  ent_coef       : $ENT_COEF"
echo "  Profils        : $PROFILES"
echo "  Instrumentation: ActionDimMonitor=ON (MESURE)   Guard=OFF (réserve)"
echo "═══════════════════════════════════════════════════════════════════════════════"

# ── 5. Cleanup léger (pas de sudo : portable) ────────────────────────────────────
pkill -9 -f "python.*train_parallel" 2>/dev/null || true
if [[ "$USE_RAY" == "1" ]]; then
  pkill -9 -f "ray::" 2>/dev/null || true
  rm -rf /tmp/ray_adan_v2/* 2>/dev/null || true
  mkdir -p /tmp/ray_adan_v2/spill /tmp/ray_adan_v2/tmp
fi
mkdir -p "$PROJECT_DIR/logs/training" "$PROJECT_DIR/checkpoints"

# ── 6. Environnement (instrumentation + overrides V2) ────────────────────────────
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export ADAN_ACTIONDIM=1
export ADAN_ACTIONDIM_EVERY="${ADAN_ACTIONDIM_EVERY:-1}"
export ADAN_USE_SDE="$USE_SDE"
export ADAN_USE_EXPLN="$USE_EXPLN"
export ADAN_LOG_STD_INIT="$LOG_STD_INIT"
export ADAN_ENT_COEF="$ENT_COEF"
export OMP_NUM_THREADS="$([[ $NCPU -ge 4 ]] && echo 4 || echo $NCPU)"
if [[ "$USE_RAY" == "1" ]]; then
  export RAY_NODE_IP_ADDRESS="127.0.0.1"
  export RAY_memory_monitor_refresh_ms=0
  export RAY_memory_usage_threshold=0.85
  export RAY_gcs_rpc_server_reconnect_timeout_s=600
  export RAY_TMPDIR=/tmp/ray_adan_v2/tmp
fi

# ── 7. Sélection interpréteur python (conda env si dispo) ────────────────────────
PYBIN="python"
for cand in \
  "$PROJECT_DIR/../miniconda3/envs/trading_env/bin/python" \
  "$HOME/miniconda3/envs/trading_env/bin/python" \
  "/home/morningstar/miniconda3/envs/trading_env/bin/python"; do
  if [[ -x "$cand" ]]; then PYBIN="$cand"; break; fi
done
echo "  Python         : $PYBIN"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$PROJECT_DIR/logs/training/v2_${MACHINE}_${TIMESTAMP}.log"
export ADAN_ACTIONDIM_CSV="$PROJECT_DIR/logs/training/actiondim_v2_${MACHINE}_${TIMESTAMP}.csv"
echo "  Log            : $LOG_FILE"
echo "  ActionDim CSV  : $ADAN_ACTIONDIM_CSV"
echo ""

# ── 8. Construction de la commande ───────────────────────────────────────────────
if [[ "$USE_RAY" == "1" ]]; then
  CMD=("$PYBIN" scripts/train_parallel_agents.py
       --mode heavy
       --num-cpus "$NCPU"
       --num-samples "$NUM_SAMPLES"
       --envs-per-worker "$ENVS_PER_WORKER"
       --no-subproc
       --profiles $PROFILES
       --steps "$STEPS"
       --steps-per-iter "$STEPS_PER_ITER"
       --checkpoint-dir "$PROJECT_DIR/checkpoints")
else
  # Mode Sandbox : 1 entraînement SB3, pas de Ray (protège la RAM du VPS).
  CMD=("$PYBIN" scripts/train_parallel_agents.py
       --mode sandbox
       --steps "$STEPS")
fi

echo "Commande : ${CMD[*]}"
echo ""
if [[ "$DRY_RUN" == "1" ]]; then
  echo "(--dry-run : commande NON exécutée)"; exit 0
fi

# ── 9. Lancement (foreground + tee) ──────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  DÉMARRAGE — surveiller la colonne size:μ dans les lignes [ActionDim]"
echo "═══════════════════════════════════════════════════════════════════════════════"
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  TERMINÉ (exit=$EXIT_CODE)"
echo "  Log  : $LOG_FILE"
echo "  CSV  : $ADAN_ACTIONDIM_CSV"
echo "  Analyse : python scripts/diagnostics/analyze_actiondim.py \"$ADAN_ACTIONDIM_CSV\""
echo "═══════════════════════════════════════════════════════════════════════════════"
exit $EXIT_CODE
