#!/usr/bin/env bash
set -euo pipefail

REPORT_FILE="report.log"
DATE_TS="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

# Run targeted tests and capture output
OUT=$(pytest -q tests/unit/utils/test_timeout_manager.py \
              tests/unit/training/test_trainer_timeout_integration.py \
              tests/unit/training/test_parallel_trainer_timeout.py \
              -c /dev/null 2>&1 || true)

PASSED=$(echo "$OUT" | grep -Eo "^[0-9]+ passed" | awk '{print $1}' || echo 0)
FAILED=$(echo "$OUT" | grep -Eo "[0-9]+ failed" | awk '{print $1}' || echo 0)

{
  echo "[$DATE_TS] Rapport d'exécution des tests ciblés";
  echo "-------------------------------------------";
  echo "$OUT";
  echo;
  if [ "$FAILED" != "0" ]; then
    echo "Statut: ECHEC";
  else
    echo "Statut: SUCCES";
  fi
} > "$REPORT_FILE"

# Print summary to stdout
if [ "$FAILED" != "0" ]; then
  echo "Tests ciblés échoués ($FAILED échec[s]). Voir $REPORT_FILE" >&2
  exit 1
else
  echo "Tests ciblés réussis ($PASSED passés). Rapport: $REPORT_FILE"
fi
