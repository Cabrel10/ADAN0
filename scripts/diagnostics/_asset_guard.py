#!/usr/bin/env python3
"""_asset_guard.py — garde-fou PARTAGE d'identite du dataset (univers launcher).

Pourquoi ce module existe
-------------------------
Le gate canonique scripts/validation/financial_stability_check.py porte deja
LAUNCHER_ASSETS + une RuntimeError (ADAN0_GATE_ASSET_SOURCE_FIX), mais ce
garde-fou vivait dans UN SEUL fichier. Les probes diag_mtf_* (rounds 1-4)
ont ete ecrits sans le reutiliser : R3 a melange BTCUSDT/BTCUSDT_BINANCE/
DOGEUSDT/DOGEUSDT_BINANCE comme quatre univers equivalents, et R4 a appris
ses poids de confluence sur BTCUSDT = 7 991 barres (~28 jours) au lieu de
BTCUSDT_BINANCE = 662 643 barres (~6,3 ans). Verdicts R3/R4 = SUSPECTS.

Regle desormais structurelle (pas de la memoire de session) :
  - UN SEUL endroit definit l'univers du launcher : ici.
  - Tout script de diagnostic importe get_launcher_assets() et ne possede
    AUCUNE liste d'assets locale.
  - Chaque chargement de parquet passe par assert_dataset_identity() qui
    JOURNALISE asset + chemin absolu + nombre de lignes : un rapport ambigu
    devient impossible a produire.
  - Les petits controles (BTCUSDT / DOGEUSDT) ne peuvent etre charges QUE
    via assert_control_asset() explicite — jamais melanges aux stats launcher.

Reference launcher : scripts/launch_asset_run.py L57.
"""
from pathlib import Path

# Univers reellement trade/charge par le launcher ADAN.
LAUNCHER_ASSETS = ("BTCUSDT_BINANCE", "DOGEUSDT_BINANCE")
DEFAULT_ASSET = "BTCUSDT_BINANCE"

# Controles courts (jamais dans les statistiques officielles).
CONTROL_ASSETS = ("BTCUSDT", "DOGEUSDT")

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "processed" / "indicators"


def get_launcher_assets():
    """Univers officiel du launcher, la seule source de verite."""
    return LAUNCHER_ASSETS


def assert_launcher_asset(asset: str) -> str:
    """Echoue immediatement si l'asset n'appartient pas a l'univers launcher."""
    if asset not in LAUNCHER_ASSETS:
        raise RuntimeError(
            f"FORBIDDEN DIAGNOSTIC ASSET: {asset!r}. L'univers du launcher est "
            f"{sorted(LAUNCHER_ASSETS)}. Un probe doit mesurer l'univers que le "
            f"run charge reellement (cf. ADAN0_GATE_ASSET_SOURCE_FIX)."
        )
    return asset


def assert_control_asset(asset: str) -> str:
    """Marque explicitement un chargement de controle court (hors stats officielles)."""
    if asset not in CONTROL_ASSETS:
        raise RuntimeError(f"{asset!r} n'est pas un controle connu {sorted(CONTROL_ASSETS)}")
    print(f"[ASSET-GUARD] CONTROLE EXPLICITE (hors stats officielles): asset={asset}")
    return asset


def assert_dataset_identity(asset: str, tf: str, split: str, n_rows: int) -> None:
    """Journalise asset + chemin absolu + rows au moment du chargement.

    Appele juste apres chaque read_parquet : tout rapport peut alors prouver
    QUEL fichier a alimente chaque etape du pipeline.
    """
    assert_launcher_asset(asset)
    path = (DATA_ROOT / split / asset / f"{tf}.parquet").resolve()
    print(f"[DATASET] asset={asset} tf={tf} split={split} path={path} rows={n_rows}")
