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

from pydantic import BaseModel

from . import settings
from .services import (
    analytics_service,
    checkpoint_service,
    config_service,
    control_service,
    log_service,
    metrics_engine,
    system_service,
    telemetry_service,
)

OPENAPI_TAGS = [
    {"name": "Health", "description": "Liveness / readiness du backend."},
    {"name": "Training", "description": "Suivi temps réel du run 500k V4 : "
     "statut process, télémétrie collapse, verdict, log."},
    {"name": "Models", "description": "Checkpoints SB3 (checkpoints/*.zip)."},
    {"name": "Config", "description": "Extrait read-only de config.yaml — "
     "expose les frais (verrouillés) sans jamais les modifier."},
    {"name": "System", "description": "CPU / RAM / swap du VPS via psutil."},
    {"name": "Runs", "description": "Runs détectés (log + checkpoints)."},
    {"name": "Analytics", "description": "Trades, equity, matrices de métriques "
     "(Sharpe/Sortino/Calmar/PF/VaR/CVaR), registre de modèles, bougies OHLCV."},
    {"name": "Control", "description": "Pilotage de l'entraînement : "
     "Hyperparameter Lab, choix du worker, launch/stop. FRAIS VERROUILLÉS."},
    {"name": "Metrics", "description": "Metrics Engine fiable : métriques "
     "RECALCULÉES depuis les données brutes, avec provenance "
     "(value/computed_at/source/window) + Validator (dashboard vs recompute)."},
    {"name": "Agent", "description": "Endpoint unique machine-readable : un agent "
     "IA peut lire TOUT l'état fiable du système avec une seule URL."},
]

DESCRIPTION = """
**ADAN0 Terminal — Mission Control API** (lecture seule).

Observe les artefacts RÉELS produits par `scripts/train_parallel_agents.py` :
- `logs/training/train_v4_500k.log`
- `logs/training/diagnostic_collapse_v4.csv`
- `checkpoints/*.zip`
- `config/config.yaml` (frais exposés, **jamais modifiés** : commission 0.0025,
  round_trip_fees 0.005).

WebSocket temps réel : `ws://<host>/ws/training` (push toutes les 3 s).
"""

app = FastAPI(
    title="ADAN0 Terminal API",
    version="0.1.0",
    description=DESCRIPTION,
    openapi_tags=OPENAPI_TAGS,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["Health"], summary="Statut du service")
def health() -> dict:
    return {"status": "ok", "service": "adan0-terminal"}


@app.get("/api/training/status", tags=["Training"],
         summary="Statut du process d'entraînement + progression")
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


@app.get("/api/training/telemetry", tags=["Training"],
         summary="Lignes de télémétrie collapse (CSV) après le step `since`")
def telemetry(since: int = 0) -> dict:
    rows = telemetry_service.read_telemetry(since=since)
    return {"rows": rows, "count": len(rows)}


@app.get("/api/training/collapse", tags=["Training"],
         summary="Verdict de collapse basé sur les FAITS (a0_std, HOLD%, illegal)")
def collapse() -> dict:
    return telemetry_service.collapse_verdict()


@app.get("/api/training/log", tags=["Training"],
         summary="N dernières lignes du log d'entraînement")
def log(tail: int = 200) -> dict:
    return {"lines": log_service.tail_lines(tail)}


@app.get("/api/checkpoints", tags=["Models"],
         summary="Liste des checkpoints SB3 (.zip)")
def checkpoints() -> dict:
    cks = checkpoint_service.list_checkpoints()
    return {"checkpoints": cks, "count": len(cks)}


@app.get("/api/config", tags=["Config"],
         summary="Config sûre (frais verrouillés + reward shaping + sandbox)")
def config() -> dict:
    return config_service.safe_config()


@app.get("/api/system", tags=["System"],
         summary="Stats système VPS (CPU/RAM/swap)")
def system() -> dict:
    return system_service.system_stats()


@app.get("/api/runs", tags=["Runs"],
         summary="Runs détectés (log + checkpoints)")
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


# ------------------------------- Analytics -------------------------------- #
@app.get("/api/analytics/registry", tags=["Analytics"],
         summary="Registre de modèles : backtests par checkpoint")
def analytics_registry() -> dict:
    return {"models": analytics_service.model_registry(),
            "named": analytics_service.named_backtests()}


@app.get("/api/analytics/named", tags=["Analytics"],
         summary="Backtests nommés (logs/validation/backtest_*.json)")
def analytics_named() -> dict:
    return {"backtests": analytics_service.named_backtests()}


@app.get("/api/analytics/equity", tags=["Analytics"],
         summary="Courbe d'equity reconstruite depuis les trades")
def analytics_equity() -> dict:
    data = analytics_service.load_trades(limit=5000)
    eq = analytics_service.equity_curve_from_trades(data["trades"])
    return {"points": eq, "file": data["file"]}


@app.get("/api/analytics/trades", tags=["Analytics"],
         summary="Trades trade-par-trade + courbe d'equity")
def analytics_trades(limit: int = 2000) -> dict:
    data = analytics_service.load_trades(limit=limit)
    eq = analytics_service.equity_curve_from_trades(data["trades"])
    return {**data, "equity": eq}


@app.get("/api/analytics/metrics", tags=["Analytics"],
         summary="Matrices de métriques (Sharpe/Sortino/Calmar/PF/VaR/CVaR…)")
def analytics_metrics(limit: int = 5000) -> dict:
    data = analytics_service.load_trades(limit=limit)
    return {
        "file": data["file"],
        "metrics": analytics_service.compute_metrics(data["trades"]),
        "confusion": analytics_service.confusion_buy_sell(data["trades"]),
    }


@app.get("/api/analytics/candles", tags=["Analytics"],
         summary="Bougies OHLCV par timeframe (5m/1h/4h) pour TradingView")
def analytics_candles(timeframe: str = "5m", limit: int = 500) -> dict:
    return analytics_service.candles(timeframe=timeframe, limit=limit)


# ----------------------------- Metrics Engine ----------------------------- #
@app.get("/api/metrics/performance", tags=["Metrics"],
         summary="Bloc performance+risque RECALCULÉ depuis le CSV brut (avec provenance)")
def metrics_performance() -> dict:
    return metrics_engine.cached(
        "perf", 5.0, lambda: metrics_engine.performance_block())


@app.get("/api/metrics/rl", tags=["Metrics"],
         summary="Métriques RL (entropy/illegal/HOLD/std/histo) avec provenance")
def metrics_rl() -> dict:
    return metrics_engine.cached("rl", 3.0, metrics_engine.rl_block)


@app.get("/api/metrics/validate", tags=["Metrics"],
         summary="Validator : dashboard vs recompute indépendant (match/mismatch)")
def metrics_validate() -> dict:
    return metrics_engine.cached("validate", 5.0, lambda: metrics_engine.validate())


@app.get("/api/metrics/markers", tags=["Metrics"],
         summary="Validation des marqueurs de trade (prix dans la plage marché)")
def metrics_markers(timeframe: str = "5m") -> dict:
    return metrics_engine.validate_markers(timeframe=timeframe)


@app.get("/api/metrics/equity_drawdown", tags=["Metrics"],
         summary="Courbes equity + drawdown reconstruites depuis les trades")
def metrics_equity_drawdown() -> dict:
    return metrics_engine.cached(
        "eqdd", 5.0, lambda: metrics_engine.equity_and_drawdown())


# -------------------------------- Agent ----------------------------------- #
@app.get("/api/agent/snapshot", tags=["Agent"],
         summary="État COMPLET et fiable du système en une seule URL (pour agent IA)")
def agent_snapshot() -> dict:
    """Single machine-readable snapshot. Everything an AI agent needs to reason
    about ADAN's state — training progress, collapse verdict, reliable metrics
    (with provenance), validation result, fees (locked), system — in one call."""
    def build() -> dict:
        status = training_status()
        collapse = telemetry_service.collapse_verdict()
        perf = metrics_engine.performance_block()
        rl = metrics_engine.rl_block()
        val = metrics_engine.validate()
        cfg = config_service.safe_config()
        return {
            "service": "adan0-mission-control",
            "generated_at": metrics_engine._now_iso(),
            "schema": "agent.snapshot.v1",
            "training": status,
            "collapse": collapse,
            "metrics_reliable": {
                "performance": perf,
                "rl": rl,
            },
            "validation": val,
            "fees_locked": {
                "commission": cfg.get("fees", {}).get("commission"),
                "round_trip_fees": cfg.get("fees", {}).get("round_trip_fees"),
                "editable": False,
            },
            "system": system_service.system_stats(),
            "checkpoints_count": len(checkpoint_service.list_checkpoints()),
            "notes": [
                "All metrics are recomputed from raw CSV with provenance.",
                "validation.all_match must be true to trust the numbers.",
                "Fees are immutable at 0.5% (commission 0.0025 / round_trip 0.005).",
            ],
        }
    return metrics_engine.cached("agent_snapshot", 4.0, build)


# -------------------------------- Control --------------------------------- #
class LaunchRequest(BaseModel):
    steps: int = 50000
    worker: str = "scalper"
    hyperparams: dict = {}
    diag: bool = True


@app.get("/api/control/status", tags=["Control"],
         summary="État du process d'entraînement (pilotage)")
def control_status() -> dict:
    return {"status": control_service.status(),
            "workers": control_service.WORKERS,
            "editable_hyperparams": list(control_service.HYPERPARAM_ENV.keys()),
            "forbidden": sorted(control_service.FORBIDDEN_KEYS),
            "fees_locked": True}


@app.post("/api/control/launch", tags=["Control"],
          summary="Lancer un run (worker + hyperparams sûrs). FRAIS VERROUILLÉS.")
def control_launch(req: LaunchRequest) -> dict:
    return control_service.launch(
        steps=req.steps, worker=req.worker,
        hyperparams=req.hyperparams, diag=req.diag)


@app.post("/api/control/stop", tags=["Control"],
          summary="Arrêter le run d'entraînement actif")
def control_stop() -> dict:
    return control_service.stop()


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

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        # Never let the SPA fallback swallow API/WS/doc routes.
        if full_path.startswith(("api/", "ws/", "docs", "redoc", "openapi")):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        target = _DIST / full_path
        if target.exists() and target.is_file():
            return FileResponse(str(target))
        return FileResponse(str(_DIST / "index.html"))
