#!/bin/bash
# ============================================================================
# ADAN Training v11 - 1M Steps avec corrections critiques
# ============================================================================
# Corrections appliquées:
# 1. Chunk size 5m: 25k → 50k (évite overfitting)
# 2. Logique chunks: MIN → MAX (diversité des données)
# 3. Features: Retiré prix absolus, ajouté log_return + close_ema20_ratio
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "============================================================================"
echo "🚀 ADAN TRAINING v11 - 1M STEPS"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  Steps:        1,000,000"
echo "  CPUs:         4"
echo "  Samples:      2"
echo "  Mode:         heavy (PBT)"
echo "  Checkpoint:   /mnt/new_data/adan_logs/checkpoints"
echo "  Log:          /mnt/new_data/adan_logs/training/train_v11.log"
echo ""
echo "Corrections appliquées:"
echo "  ✓ Chunk size 5m: 50k (évite overfitting)"
echo "  ✓ Logique chunks: MAX (diversité)"
echo "  ✓ Features: Stationnaires (log_return, close_ema20_ratio)"
echo ""
echo "============================================================================"
echo ""

# Créer les répertoires s'ils n'existent pas
mkdir -p /mnt/new_data/adan_logs/checkpoints
mkdir -p /mnt/new_data/adan_logs/training
mkdir -p /mnt/new_data/adan_logs/metrics

# Lancer l'entraînement
nohup python scripts/train_parallel_agents.py \
  --mode heavy \
  --steps 1000000 \
  --num-cpus 4 \
  --num-samples 2 \
  --no-subproc \
  --checkpoint-dir /mnt/new_data/adan_logs/checkpoints \
  > /mnt/new_data/adan_logs/training/train_v11.log 2>&1 &

TRAIN_PID=$!

echo "✓ Entraînement lancé (PID: $TRAIN_PID)"
echo ""
echo "Pour monitorer:"
echo "  tail -f /mnt/new_data/adan_logs/training/train_v11.log"
echo ""
echo "Pour arrêter:"
echo "  kill $TRAIN_PID"
echo ""
