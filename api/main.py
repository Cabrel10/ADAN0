
import logging
import os
import sys
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
import uvicorn
import uuid
import subprocess
import json

# Add the project root to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ADAN Backend API",
    description="API for managing ADAN trading bot operations (training, backtesting, etc.)",
    version="0.1.0",
)

# In-memory task storage (for demonstration purposes)
tasks = {}

class TrainRequest(BaseModel):
    instance_id: int
    total_timesteps: int
    config_override: Dict[str, Any] = {}

class BacktestRequest(BaseModel):
    start_date: str
    end_date: str
    config_override: Dict[str, Any] = {}

@app.get("/")
async def read_root():
    return {"message": "ADAN Backend API is running!"}

@app.post("/train")
async def train_model(request: TrainRequest):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "PENDING", "progress": 0.0, "results": None}
    logger.info(f"Training task {task_id} received for instance {request.instance_id}")

    # In a real application, you would offload this to a background task queue
    # For now, we'll just simulate success
    tasks[task_id].update({"status": "COMPLETED", "progress": 1.0, "results": {"message": "Training simulated successfully"}})
    logger.info(f"Training task {task_id} simulated completion.")

    return {"message": "Training started successfully", "task_id": task_id}

@app.post("/backtest")
async def run_backtest(request: BacktestRequest):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "PENDING", "progress": 0.0, "results": None}
    logger.info(f"Backtest task {task_id} received for dates {request.start_date} to {request.end_date}")

    # Simulate backtest completion
    tasks[task_id].update({"status": "COMPLETED", "progress": 1.0, "results": {"message": "Backtest simulated successfully"}})
    logger.info(f"Backtest task {task_id} simulated completion.")

    return {"message": "Backtest started successfully", "task_id": task_id}

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.get("/metrics")
async def get_metrics(run_id: str = None):
    # Simulate fetching metrics
    metrics = {"total_pnl": 1000.0, "sharpe_ratio": 1.5, "max_drawdown": 0.05}
    if run_id:
        metrics["run_id"] = run_id
    return {"metrics": metrics}

@app.patch("/config")
async def update_config(path: str, value: Any):
    # This is a simplified example. Real implementation needs robust config management.
    logger.info(f"Config update request: path={path}, value={value}")
    # Simulate config update
    return {"message": "Configuration updated successfully"}

@app.get("/config")
async def get_config():
    # Simulate fetching config
    current_config = {"environment": {"initial_capital": 10000.0}, "trading_rules": {"stop_loss": 0.02}}
    return {"config": current_config}

# MCP Router
mcp_router = APIRouter(prefix="/mcp")

@mcp_router.post("/")
@mcp_router.post("")
async def mcp_root_post():
    return {"jsonrpc": "2.0", "result": True, "id": 1}

@mcp_router.get("/tools")
async def get_mcp_tools():
    return [
        {
            "name": "adan_ping",
            "description": "Pings the ADAN backend to check if it's alive.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "adan_train",
            "description": "Launches a training process for the ADAN bot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instance_id": {"type": "integer"},
                    "total_timesteps": {"type": "integer"}
                },
                "required": ["instance_id", "total_timesteps"]
            }
        }
    ]

@mcp_router.post("/call")
async def call_mcp_tool(tool_call: Dict[str, Any]):
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args", {})

    logger.info(f"MCP tool call received: {tool_name} with args {tool_args}")

    if tool_name == "adan_ping":
        return {"result": "pong"}
    elif tool_name == "adan_train":
        # Launch the training script as a subprocess
        script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'train_parallel_agents.py')
        
        command = [
            sys.executable, # Use the same python interpreter
            script_path,
            "--instance_id", str(tool_args.get("instance_id")),
            "--total_timesteps", str(tool_args.get("total_timesteps"))
        ]
        
        if tool_args.get("config_override"):
            command.extend(["--config_override", json.dumps(tool_args["config_override"])])

        # Run in background and don't wait
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return {"result": {"message": "Training process initiated in background."}}
    else:
        raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found.")

app.include_router(mcp_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
