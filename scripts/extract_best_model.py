#!/usr/bin/env python3
"""
ADAN Best Model Extractor — Production Version
=================================================
Scans Ray Tune results directories (``ray_results/``, ``/tmp/ray/``),
identifies the trial with the highest ``mean_reward`` or ``mean_sharpe``,
copies its ``model.zip`` and ``vecnormalize.pkl`` to
``models/rl_agents/production/``, and prints a performance summary table.

Also supports extracting from simple PPO training (non-Ray) by checking
``models/rl_agents/ppo_adan_simple.zip``.

Usage:
    python scripts/extract_best_model.py
    python scripts/extract_best_model.py --metric mean_sharpe
    python scripts/extract_best_model.py --ray-dir /custom/ray_results
"""

import argparse
import csv
import glob
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default search directories for Ray results
DEFAULT_RAY_DIRS = [
    PROJECT_ROOT / "logs" / "ray_results",
    PROJECT_ROOT / "logs" / "ray_results" / "adan_pbt_training",
    Path("/tmp/ray"),
    Path("/tmp/ray/session_latest"),
]

PRODUCTION_DIR = PROJECT_ROOT / "models" / "rl_agents" / "production"
SIMPLE_MODEL_DIR = PROJECT_ROOT / "models" / "rl_agents"


def scan_ray_results(ray_dirs: list, metric: str = "mean_reward") -> list:
    """Scan Ray Tune result directories for trial data.

    Looks for ``progress.csv`` files in trial subdirectories and extracts
    the best row per trial according to *metric*.

    Returns:
        List of dicts with keys: reward, sharpe, balance, path, worker,
        iteration, metric_value.
    """
    trials = []

    for base_dir in ray_dirs:
        base = Path(base_dir)
        if not base.exists():
            continue

        # Find progress.csv files (Ray Tune convention)
        for csv_path in base.rglob("progress.csv"):
            trial_dir = csv_path.parent
            try:
                with open(csv_path) as f:
                    rows = list(csv.DictReader(f))
                if not rows:
                    continue

                # Find best row by chosen metric
                best_row = None
                best_val = -1e18
                for row in rows:
                    val = float(row.get(metric, 0) or 0)
                    if val > best_val:
                        best_val = val
                        best_row = row

                if best_row is None:
                    continue

                trials.append({
                    "reward": float(best_row.get("mean_reward", 0) or 0),
                    "sharpe": float(best_row.get("mean_sharpe", 0) or 0),
                    "balance": float(best_row.get("mean_balance", 0) or 0),
                    "realized_pnl": float(best_row.get("realized_pnl", 0) or 0),
                    "timesteps": int(float(best_row.get("timesteps_total", 0) or 0)),
                    "iteration": int(float(best_row.get("training_iteration", 0) or 0)),
                    "lr": float(best_row.get("learning_rate", 0) or 0),
                    "ent_coef": float(best_row.get("ent_coef", 0) or 0),
                    "gamma": float(best_row.get("gamma", 0) or 0),
                    "path": trial_dir,
                    "worker": trial_dir.name,
                    "metric_value": best_val,
                    "source": "ray_tune",
                })
            except Exception as e:
                print(f"  [SKIP] {csv_path}: {e}")

    # Also scan for result.json (newer Ray format)
    for base_dir in ray_dirs:
        base = Path(base_dir)
        if not base.exists():
            continue
        for json_path in base.rglob("result.json"):
            trial_dir = json_path.parent
            if any(t["path"] == trial_dir for t in trials):
                continue  # Already scanned via CSV
            try:
                with open(json_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        val = float(data.get(metric, 0) or 0)
                        trials.append({
                            "reward": float(data.get("mean_reward", 0) or 0),
                            "sharpe": float(data.get("mean_sharpe", 0) or 0),
                            "balance": float(data.get("mean_balance", 0) or 0),
                            "realized_pnl": float(data.get("realized_pnl", 0) or 0),
                            "timesteps": int(data.get("timesteps_total", 0) or 0),
                            "iteration": int(data.get("training_iteration", 0) or 0),
                            "lr": float(data.get("learning_rate", 0) or 0),
                            "ent_coef": float(data.get("ent_coef", 0) or 0),
                            "gamma": float(data.get("gamma", 0) or 0),
                            "path": trial_dir,
                            "worker": trial_dir.name,
                            "metric_value": val,
                            "source": "ray_tune_json",
                        })
            except Exception:
                pass

    return trials


def find_simple_model() -> dict:
    """Check for a simple PPO model (non-Ray) in models/rl_agents/."""
    model_path = SIMPLE_MODEL_DIR / "ppo_adan_simple.zip"
    vecnorm_path = SIMPLE_MODEL_DIR / "vecnormalize.pkl"

    if not model_path.exists():
        return None

    return {
        "reward": 0.0,
        "sharpe": 0.0,
        "balance": 0.0,
        "realized_pnl": 0.0,
        "timesteps": 0,
        "iteration": 0,
        "lr": 0.0,
        "ent_coef": 0.0,
        "gamma": 0.0,
        "path": SIMPLE_MODEL_DIR,
        "worker": "simple_ppo",
        "metric_value": 0.0,
        "source": "simple_ppo",
        "model_file": str(model_path),
        "vecnorm_file": str(vecnorm_path) if vecnorm_path.exists() else None,
    }


def find_checkpoint_files(trial_dir: Path) -> dict:
    """Find model.zip and vecnormalize.pkl in a trial directory.

    Checks:
      1. Latest checkpoint_* subdirectory
      2. Direct files in trial_dir
      3. Any nested directory with model.zip
    """
    files = {"model": None, "vecnorm": None}

    # Try checkpoint subdirectories (sorted ascending; take last)
    checkpoints = sorted(trial_dir.glob("checkpoint_*"))
    if checkpoints:
        ckpt = checkpoints[-1]
        m = ckpt / "model.zip"
        v = ckpt / "vecnormalize.pkl"
        if m.exists():
            files["model"] = str(m)
        if v.exists():
            files["vecnorm"] = str(v)
        if files["model"]:
            return files

    # Try direct files
    for name, key in [("model.zip", "model"), ("vecnormalize.pkl", "vecnorm")]:
        p = trial_dir / name
        if p.exists():
            files[key] = str(p)

    if files["model"]:
        return files

    # Recursive search
    for m in trial_dir.rglob("model.zip"):
        files["model"] = str(m)
        v = m.parent / "vecnormalize.pkl"
        if v.exists():
            files["vecnorm"] = str(v)
        break

    return files


def copy_to_production(trial: dict) -> bool:
    """Copy best model files to production directory."""
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)

    if trial.get("source") == "simple_ppo":
        model_src = trial.get("model_file")
        vecnorm_src = trial.get("vecnorm_file")
    else:
        ckpt_files = find_checkpoint_files(Path(trial["path"]))
        model_src = ckpt_files["model"]
        vecnorm_src = ckpt_files["vecnorm"]

    if not model_src or not Path(model_src).exists():
        print(f"  [ERROR] No model.zip found in {trial['path']}")
        return False

    # Copy model
    dst_model = PRODUCTION_DIR / "model.zip"
    shutil.copy2(model_src, dst_model)
    print(f"  Copied model.zip -> {dst_model}")

    # Copy vecnormalize
    if vecnorm_src and Path(vecnorm_src).exists():
        dst_vec = PRODUCTION_DIR / "vecnormalize.pkl"
        shutil.copy2(vecnorm_src, dst_vec)
        print(f"  Copied vecnormalize.pkl -> {dst_vec}")

    # Copy worker_state.json if available
    ws_src = Path(trial["path"]) / "worker_state.json"
    if not ws_src.exists():
        # Check in checkpoint dirs
        for ckpt in sorted(Path(trial["path"]).glob("checkpoint_*")):
            ws = ckpt / "worker_state.json"
            if ws.exists():
                ws_src = ws
                break
    if ws_src.exists():
        shutil.copy2(ws_src, PRODUCTION_DIR / "worker_state.json")

    # Write metadata
    meta = {
        "source_trial": trial["worker"],
        "source_path": str(trial["path"]),
        "source_type": trial["source"],
        "metric_used": "mean_reward",
        "mean_reward": trial["reward"],
        "mean_sharpe": trial["sharpe"],
        "mean_balance": trial["balance"],
        "realized_pnl": trial["realized_pnl"],
        "timesteps_total": trial["timesteps"],
        "training_iteration": trial["iteration"],
        "learning_rate": trial["lr"],
        "ent_coef": trial["ent_coef"],
        "gamma": trial["gamma"],
        "extracted_at": datetime.now().isoformat(),
        "usage": {
            "load_model": "PPO.load('models/rl_agents/production/model.zip')",
            "load_vecnorm": (
                "env = VecNormalize.load('models/rl_agents/production/vecnormalize.pkl', env)\n"
                "env.training = False\n"
                "env.norm_reward = False"
            ),
        },
    }
    with open(PRODUCTION_DIR / "extraction_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return True


def print_performance_table(trials: list, best_idx: int, metric: str):
    """Print a formatted performance comparison table."""
    print("\n" + "=" * 100)
    print("  ADAN MODEL EXTRACTION - PERFORMANCE TABLE")
    print("=" * 100)
    print(
        f"  {'#':<4} {'Trial':<30} {'Source':<12} "
        f"{'Reward':>10} {'Sharpe':>10} {'Balance':>10} "
        f"{'Steps':>10} {'LR':>10} {'Sel':>4}"
    )
    print("-" * 100)

    for i, t in enumerate(trials):
        sel = " <--" if i == best_idx else ""
        print(
            f"  {i+1:<4} {t['worker'][:28]:<30} {t['source']:<12} "
            f"{t['reward']:>10.4f} {t['sharpe']:>10.4f} {t['balance']:>10.2f} "
            f"{t['timesteps']:>10,} {t['lr']:>10.2e} {sel}"
        )

    print("-" * 100)
    best = trials[best_idx]
    print(f"  SELECTED: {best['worker']} (by {metric}={best['metric_value']:.6f})")
    print(f"  OUTPUT:   {PRODUCTION_DIR}/model.zip")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="ADAN Best Model Extractor")
    parser.add_argument(
        "--metric", default="mean_reward",
        choices=["mean_reward", "mean_sharpe", "mean_balance", "realized_pnl"],
        help="Metric to select best trial (default: mean_reward)",
    )
    parser.add_argument(
        "--ray-dir", type=str, default=None,
        help="Additional Ray results directory to scan",
    )
    parser.add_argument(
        "--include-simple", action="store_true", default=True,
        help="Include simple PPO model as candidate (default: True)",
    )
    args = parser.parse_args()

    print("\n[ADAN] Scanning for trained models...")

    # Build search dirs
    search_dirs = list(DEFAULT_RAY_DIRS)
    if args.ray_dir:
        search_dirs.insert(0, Path(args.ray_dir))

    # Scan Ray results
    trials = scan_ray_results(search_dirs, metric=args.metric)
    print(f"  Found {len(trials)} Ray Tune trial(s)")

    # Include simple PPO model
    if args.include_simple:
        simple = find_simple_model()
        if simple:
            trials.append(simple)
            print(f"  Found simple PPO model at {simple['path']}")

    if not trials:
        print("\n[ERROR] No trained models found.")
        print("  Searched in:")
        for d in search_dirs:
            print(f"    - {d}")
        print(f"    - {SIMPLE_MODEL_DIR}")
        print("\n  Train a model first:")
        print("    python scripts/train_simple_ppo.py --steps 30000")
        print("    python scripts/train_parallel_agents.py --steps 1000000")
        sys.exit(1)

    # Sort by metric and select best
    trials.sort(key=lambda t: t["metric_value"], reverse=True)
    best_idx = 0

    # Print table
    print_performance_table(trials, best_idx, args.metric)

    # Copy to production
    print(f"\n[ADAN] Extracting best model to {PRODUCTION_DIR}...")
    success = copy_to_production(trials[best_idx])

    if success:
        print("\n[SUCCESS] Best model extracted to production directory.")
        print(f"  Load with:")
        print(f"    from stable_baselines3 import PPO")
        print(f"    model = PPO.load('{PRODUCTION_DIR}/model.zip')")
    else:
        print("\n[ERROR] Failed to extract model.")
        sys.exit(1)


if __name__ == "__main__":
    main()
