"""ADAN0 Terminal — FastAPI backend (MVP).

Read-only Mission Control over the real training artifacts:
  - logs/training/train_v4_500k.log
  - logs/training/diagnostic_collapse_v4.csv
  - checkpoints/*.zip
  - config/config.yaml  (fees surfaced, never modified)

Serves the built frontend (web/frontend/dist) when present.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import settings
from .services import (
    checkpoint_service,
    config_service,
    log_service,
    system_service,
    telemetry_service,
)

app = FastAPI(title="ADAN0 Terminal", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "adan0-terminal"}


@app.get("/api/training/status")
def training_status() -> dict:
    proc = system_service.training_process()
    prog = log_service.parse_progress()
    last_ts = prog.get("last_timestep")
    elapsed = proc.get("elapsed_sec") or 0
    steps_per_min = None
    if last_ts and elapsed:
        steps_per_min = round(last_ts / (elapsed / 60.0), 1)
    return {
        "process": proc,
        "last_timestep": last_ts,
        "target_steps": settings.TARGET_TOTAL_STEPS,
        "progress_pct": round(100.0 * (last_ts or 0) / settings.TARGET_TOTAL_STEPS, 2),
        "steps_per_min": steps_per_min,
        "has_errors": prog.get("has_errors"),
        "error_count": prog.get("error_count"),
        "last_error": prog.get("last_error"),
    }


@app.get("/api/training/telemetry")
def telemetry(since: int = 0) -> dict:
    rows = telemetry_service.read_telemetry(since=since)
    return {"rows": rows, "count": len(rows)}


@app.get("/api/training/collapse")
def collapse() -> dict:
    return telemetry_service.collapse_verdict()


@app.get("/api/training/log")
def log(tail: int = 200) -> dict:
    return {"lines": log_service.tail_lines(tail)}


@app.get("/api/checkpoints")
def checkpoints() -> dict:
    cks = checkpoint_service.list_checkpoints()
    return {"checkpoints": cks, "count": len(cks)}


@app.get("/api/config")
def config() -> dict:
    return config_service.safe_config()


@app.get("/api/system")
def system() -> dict:
    return system_service.system_stats()


@app.get("/api/runs")
def runs() -> dict:
    """Detected runs = present log + checkpoints, summarized as one active run."""
    prog = log_service.parse_progress()
    proc = system_service.training_process()
    cks = checkpoint_service.list_checkpoints()
    return {
        "runs": [
            {
                "id": "v4_500k",
                "name": "DIAGNOSTIC-V4 500k (scalper, BTC/USDT)",
                "status": "training" if proc.get("running") else "stopped",
                "last_timestep": prog.get("last_timestep"),
                "checkpoints": len(cks),
            }
        ]
    }


@app.websocket("/ws/training")
async def ws_training(ws: WebSocket) -> None:
    await ws.accept()
    last_sent_ts = 0
    try:
        while True:
            status = training_status()
            new_rows = telemetry_service.read_telemetry(since=last_sent_ts)
            if new_rows:
                last_sent_ts = new_rows[-1]["timesteps"]
            payload = {
                "type": "tick",
                "status": status,
                "telemetry": new_rows,
                "collapse": telemetry_service.collapse_verdict(),
                "system": system_service.system_stats(),
            }
            await ws.send_text(json.dumps(payload, default=str))
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        return
    except Exception:
        return


# --- Serve built frontend if present ---
_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(_DIST / "assets")), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(_DIST / "index.html"))

    @app.get("/{full_path:path}")
    def spa(full_path: str) -> FileResponse:
        target = _DIST / full_path
        if target.exists() and target.is_file():
            return FileResponse(str(target))
        return FileResponse(str(_DIST / "index.html"))
