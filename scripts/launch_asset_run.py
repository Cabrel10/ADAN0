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

    # ── V30 CHECKPOINT SEPARATION (2026-08-26, autonomous audit) ──────────
    # ROOT CAUSE of "checkpoints mélangés BTC/DOGE": both runs wrote the same
    # name_prefix (ppo_adan0_sandbox_checkpoint) into checkpoints/, so the two
    # independent per-asset brains physically overwrote each other. Derive a
    # DISTINCT, asset-scoped prefix so BTC and DOGE weights never collide.
    #   BTCUSDT_BINANCE  -> ppo_adan0_BTCUSDT
    #   DOGEUSDT_BINANCE -> ppo_adan0_DOGEUSDT
    _asset_tag = str(args.asset).split("_")[0].upper()  # BTCUSDT / DOGEUSDT
    os.environ["ADAN_CKPT_PREFIX"] = f"ppo_adan0_{_asset_tag}"
    # V30 exploration fix is now enforced by config (use_sde:false, log_std:-1.0);
    # do NOT re-inject the buggy ADAN_USE_SDE here. Leave unset so config wins.

    # ── V30 DATA-DRIVEN TP CEILING (per-asset, from empirical ATR) ─────────
    # Measured on the real 5y parquet (scripts audit 2026-08-26):
    #   BTC  1h ATR%: med 0.78  p95 2.22  p99 3.68   | 5m med 0.20
    #   DOGE 1h ATR%: med 1.17  p95 3.46  p99 7.10   | 5m med 0.30
    # The old flat 12% TP was unreachable for BTC (TP head starved). We size
    # each ceiling to the asset's realistic multi-bar favorable excursion:
    #   BTC : tp_hi 6%  (~2.7x 1h p95 ATR)  — reachable swing, no fantasy target
    #   DOGE: tp_hi 9%  (DOGE ~1.6x more volatile) — still below the flat 12%
    # SL ceiling left at the physical 6% (both assets' p99 SL noise fits under).
    _TP_HI = {"BTCUSDT": "0.060", "DOGEUSDT": "0.090"}
    os.environ.setdefault("ADAN_TP_HI", _TP_HI.get(_asset_tag, "0.060"))
    os.environ.setdefault("ADAN_TP_LO", "0.003")
    os.environ.setdefault("ADAN_SL_HI", "0.060")
    # Radar telemetry ON so EV/KL/log_std/entropy are tracked during the run.
    os.environ.setdefault("ADAN_DIAG_COLLAPSE", "1")

    derived = derive_config(Path(args.base_config), args.asset, args.steps)
    print(f"[LAUNCH] asset={args.asset} steps={args.steps} "
          f"ckpt_prefix={os.environ['ADAN_CKPT_PREFIX']} "
          f"tp_hi={os.environ['ADAN_TP_HI']} sl_hi={os.environ['ADAN_SL_HI']} "
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
