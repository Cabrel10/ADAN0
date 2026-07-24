"""Read-only, safe extract of config.yaml (fees + reward shaping + sandbox).

NEVER writes config.yaml. The fees are surfaced specifically so the UI can
prove they remain intact: commission 0.0025, round_trip_fees 0.005.
"""
from __future__ import annotations

from typing import Any

import yaml

from .. import settings

_cache: dict[str, Any] | None = None


def _load() -> dict[str, Any]:
    global _cache
    if _cache is not None:
        return _cache
    try:
        with settings.CONFIG_PATH.open("r") as f:
            _cache = yaml.safe_load(f) or {}
    except Exception:
        _cache = {}
    return _cache


def _find_commission(cfg: dict[str, Any]) -> Any:
    # Search a few likely locations without deep traversal assumptions.
    for path in (("environment", "commission"), ("trading", "commission"),
                 ("commission",)):
        node: Any = cfg
        ok = True
        for k in path:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                ok = False
                break
        if ok and isinstance(node, (int, float)):
            return node
    # fallback: recursive shallow scan
    def scan(d, key, depth=0):
        if depth > 4 or not isinstance(d, dict):
            return None
        if key in d and isinstance(d[key], (int, float)):
            return d[key]
        for v in d.values():
            r = scan(v, key, depth + 1)
            if r is not None:
                return r
        return None
    return scan(cfg, "commission")


def safe_config() -> dict[str, Any]:
    cfg = _load()
    rs = cfg.get("reward_shaping", {}) if isinstance(cfg, dict) else {}
    sandbox = cfg.get("sandbox", {}) if isinstance(cfg, dict) else {}

    def scan(d, key, depth=0):
        if depth > 5 or not isinstance(d, dict):
            return None
        if key in d:
            return d[key]
        for v in d.values():
            r = scan(v, key, depth + 1)
            if r is not None:
                return r
        return None

    return {
        "fees": {
            "commission": _find_commission(cfg),
            "round_trip_fees": scan(cfg, "round_trip_fees"),
        },
        "reward_shaping": {
            "invalid_trade_penalty_weight": rs.get("invalid_trade_penalty_weight"),
            "sterile_action_geom_ratio": rs.get("sterile_action_geom_ratio"),
            "sterile_action_penalty_cap": rs.get("sterile_action_penalty_cap"),
        },
        "sandbox": {
            "ent_coef": sandbox.get("ent_coef"),
        },
        "profile": "scalper",
        "asset": "BTC/USDT",
        "leverage": 1,
    }
