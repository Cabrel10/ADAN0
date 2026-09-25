#!/bin/bash
cd /home/ubuntu/webapp/MORNINGSTAR/ADAN0/web/backend
exec ./.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8770
