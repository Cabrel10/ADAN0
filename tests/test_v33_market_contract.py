import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parents[1]


def _load_launcher():
    spec = importlib.util.spec_from_file_location("launch_asset_run", ROOT / "scripts" / "launch_asset_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_btc_h40_domain_is_future_arena_quantile_derived():
    source = (ROOT / "scripts" / "launch_asset_run.py").read_text()
    assert '"BTCUSDT": {"tp_lo": "0.006", "tp_hi": "0.0222", "sl_hi": "0.0235"}' in source
    assert "MFE p50/p75/p90 = 0.5813% / 1.2018% / 2.2131%" in source
    assert "|MAE| p50/p75/p90 = 0.5932% / 1.2400% / 2.3436%" in source
    assert '"BTCUSDT": "0.060"' not in source
