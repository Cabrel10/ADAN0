#!/usr/bin/env bash
# Start the ADAN Mission Control backend (FastAPI + uvicorn).
# Serves the built React frontend from web/frontend/dist and all /api + /ws routes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # -> web/
BACKEND="${HERE}/backend"
PORT="${ADAN_PORT:-8770}"

cd "$BACKEND"
if [ ! -x ".venv/bin/python" ]; then
  echo "[backend] creating venv from conda base python ..."
  /home/ubuntu/webapp/MORNINGSTAR/miniconda3/bin/python -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi

echo "[backend] starting uvicorn on 0.0.0.0:${PORT}"
PYTHONUNBUFFERED=1 exec .venv/bin/python -m uvicorn app.main:app \
  --host 0.0.0.0 --port "${PORT}" --workers 1
