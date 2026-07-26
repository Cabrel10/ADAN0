#!/usr/bin/env bash
# Isolation CPU REVERSIBLE pour entrainement ADAN0 sur VPS partage (2026-06-27).
# Le VPS heberge 2 stacks Docker (gaintime crash-loop + whatsapp) qui saturent
# le CPU. On les confine sur le coeur 0 et on reserve les coeurs 1-3 a ADAN0.
# Gain mesure: FPS ADAN0 ~2.9 -> ~8.5 (x2.9).
#
# Usage:
#   bash isolate_cpu.sh confine    # confine les stacks tierces sur coeur 0
#   bash isolate_cpu.sh release     # rend tous les coeurs aux conteneurs (annule)
set -u

THIRD_PARTY=(
  gaintime-celery_worker-1 gaintime-celery_beat-1 gaintime-web-1
  gaintime-bot-1 gaintime-nginx-1 gaintime-db-1 gaintime-redis-1
  whatsapp-evolution-api-1 whatsapp-n8n-1 whatsapp-redis-1 whatsapp-db-1
)

ACTION="${1:-confine}"

if [ "$ACTION" = "confine" ]; then
  echo "=== Confinement stacks tierces -> coeur 0 (ADAN0 garde 1-3) ==="
  for c in "${THIRD_PARTY[@]}"; do
    if docker update --cpuset-cpus=0 "$c" >/dev/null 2>&1; then
      echo "  [OK] $c -> coeur 0"
    else
      echo "  [skip] $c (absent)"
    fi
  done
  echo "Lancer ADAN0 avec: taskset -c 1-3 <python> ... (cf launch_500k_prod.sh)"
elif [ "$ACTION" = "release" ]; then
  echo "=== Liberation: tous les coeurs (0-3) rendus aux conteneurs ==="
  for c in "${THIRD_PARTY[@]}"; do
    docker update --cpuset-cpus=0-3 "$c" >/dev/null 2>&1 && echo "  [OK] $c -> 0-3"
  done
else
  echo "usage: isolate_cpu.sh {confine|release}"
  exit 1
fi
