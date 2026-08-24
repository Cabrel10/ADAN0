#!/usr/bin/env python3
"""
Per-asset launcher for the frozen 5y experiment — one INDEPENDENT brain per asset.

It derives a config from the frozen config.yaml that points ONLY at the target
Binance dataset (BTCUSDT_binance or DOGEUSDT_binance), leaving reward / architecture /
action space / DBE / Future Arena / SL-TP mapping / PPO untouched, then runs one
sandbox training segment via train_parallel_agents.sandbox_train.

Invariant env vars are set here (not hot-changed mid-run):
  ADAN_FREE_SLTP=1            (frozen SL/TP contract — network owns geometry)
  ADAN_FORCE_FIT_SCALERS=1    (scaler TRAIN-only, per-asset fresh fit)

Usage:
  PYTHONPATH=src launch_asset_run.py --asset BTCUSDT_binance --steps 2000 \
      --checkpoint-out checkpoints/adan_BTC/seg_00002000.zip [--resume-from PREV.zip]

Emits a JSON result line (the sandbox_train dict) to stdout and to --result-json.
"""
import os, sys, json, argparse, copy, tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def derive_config(base_path: Path, asset: str, steps: int) -> Path:
    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    # Point every asset-selecting key at the single target Binance asset.
    cfg.setdefault("data", {})["assets"] = [asset]
    cfg["data"]["data_split"] = "train"
    cfg.setdefault("environment", {})["assets"] = [asset]

    # Per-worker asset override (sandbox uses w1 by default).
    for wk, wcfg in (cfg.get("workers", {}) or {}).items():
        if isinstance(wcfg, dict):
            wcfg["assets"] = [asset]
            wcfg["data_split_override"] = "train"

    # sandbox segment size (sandbox_train reads sandbox.max_training_steps when --steps None)
    cfg.setdefault("sandbox", {})["max_training_steps"] = int(steps)

    out = Path(tempfile.gettempdir()) / f"config_{asset}.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, choices=["BTCUSDT_BINANCE", "DOGEUSDT_BINANCE"])
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--checkpoint-out", required=True)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--result-json", default=None)
    ap.add_argument("--base-config", default=str(ROOT / "config" / "config.yaml"))
    args = ap.parse_args()

    # ── frozen invariant env (set BEFORE importing the env/trainer) ──
    os.environ.setdefault("ADAN_FREE_SLTP", "1")
    os.environ.setdefault("ADAN_FORCE_FIT_SCALERS", "1")

    derived = derive_config(Path(args.base_config), args.asset, args.steps)
    print(f"[LAUNCH] asset={args.asset} steps={args.steps} "
          f"free_sltp={os.environ['ADAN_FREE_SLTP']} "
          f"force_fit_scalers={os.environ['ADAN_FORCE_FIT_SCALERS']} "
          f"resume={args.resume_from} out={args.checkpoint_out}", flush=True)

    from train_parallel_agents import sandbox_train

    Path(args.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
    result = sandbox_train(
        steps=args.steps,
        config_path=str(derived),
        resume_ckpt=args.resume_from,
        checkpoint_out=args.checkpoint_out,
    )
    payload = {"asset": args.asset, "steps": args.steps,
               "checkpoint_out": args.checkpoint_out,
               "resume_from": args.resume_from, "result": result}
    if args.result_json:
        Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.result_json).write_text(json.dumps(payload, indent=2, default=str))
    print("RESULT_JSON_BEGIN")
    print(json.dumps(payload, default=str))
    print("RESULT_JSON_END")


if __name__ == "__main__":
    main()
