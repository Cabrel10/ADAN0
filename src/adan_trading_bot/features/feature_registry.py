"""
feature_registry.py — Registre canonique des 1 026 variables d'ADAN-System-One
=============================================================================

Ce module définit et expose le registre exhaustif et structuré des 1 026 variables
du système ADAN-System-One. Aucune variable n'est implicite ou anonyme.

Chaque variable est caractérisée par :
  - name : identifiant unique
  - source : provenance / module / capteur
  - famille : groupe fonctionnel
  - timeframe : 5m, 1h, 4h, trade, 1d, global
  - type : float, int, bool, categorical
  - unite : price, pct, ratio, volume, usdt, steps, count, boolean, index
  - fenetre : lookback utilisé (barres ou static)
  - transformation : calcul déterministe ou raw
  - dependances_deterministes : liste des variables parentes dont elle dérive
  - frequence_maj : cadence de mise à jour (per_5m_bar, per_trade, static, etc.)
  - disponibilite_t : statut de disponibilité causale au timestamp t
  - role_potentiel : perception, contexte, plan, risque, portefeuille

Référence : ORDRE 3, dev.md, config/feature_registry.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Union


VALID_ROLES = {"perception", "contexte", "plan", "risque", "portefeuille"}
VALID_TIMEFRAMES = {"5m", "1h", "4h", "trade", "1d", "global"}


@dataclass(frozen=True)
class FeatureEntry:
    """Entrée immuable pour une variable du système."""
    name: str
    source: str
    famille: str
    timeframe: str
    type: str
    unite: str
    fenetre: Union[int, str]
    transformation: str
    dependances_deterministes: List[str]
    frequence_maj: str
    disponibilite_t: str
    role_potentiel: str

    def __post_init__(self):
        if self.role_potentiel not in VALID_ROLES:
            raise ValueError(f"Rôle invalide '{self.role_potentiel}' pour {self.name}. Doit être dans {VALID_ROLES}")


class FeatureRegistry:
    """Gestionnaire et indexeur des 1 026 variables."""

    def __init__(self, entries: List[FeatureEntry]):
        self._entries = entries
        self._by_name: Dict[str, FeatureEntry] = {}
        self._by_role: Dict[str, List[FeatureEntry]] = {r: [] for r in VALID_ROLES}
        self._by_family: Dict[str, List[FeatureEntry]] = {}
        self._by_timeframe: Dict[str, List[FeatureEntry]] = {}

        for e in entries:
            if e.name in self._by_name:
                raise ValueError(f"Nom de variable en double : {e.name}")
            self._by_name[e.name] = e
            self._by_role[e.role_potentiel].append(e)
            self._by_family.setdefault(e.famille, []).append(e)
            self._by_timeframe.setdefault(e.timeframe, []).append(e)

    def __len__(self) -> int:
        return len(self._entries)

    def get_variable(self, name: str) -> Optional[FeatureEntry]:
        return self._by_name.get(name)

    def __getitem__(self, name: str) -> FeatureEntry:
        if name not in self._by_name:
            raise KeyError(f"Variable '{name}' inconnue du registre (1026 variables)")
        return self._by_name[name]

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def get_by_role(self, role: str) -> List[FeatureEntry]:
        if role not in self._by_role:
            raise ValueError(f"Rôle inconnu '{role}'. Rôles valides : {list(VALID_ROLES)}")
        return list(self._by_role[role])

    def get_by_family(self, family: str) -> List[FeatureEntry]:
        return list(self._by_family.get(family, []))

    def get_by_timeframe(self, timeframe: str) -> List[FeatureEntry]:
        return list(self._by_timeframe.get(timeframe, []))

    def all_variables(self) -> List[FeatureEntry]:
        return list(self._entries)

    def all_names(self) -> List[str]:
        return list(self._by_name.keys())

    def get_deterministic_dependencies(self, name: str) -> List[str]:
        var = self[name]
        return list(var.dependances_deterministes)

    def summary(self) -> Dict[str, Union[int, Dict[str, int]]]:
        return {
            "total_variables": len(self._entries),
            "by_role": {r: len(items) for r, items in self._by_role.items()},
            "by_family": {f: len(items) for f, items in self._by_family.items()},
            "by_timeframe": {t: len(items) for t, items in self._by_timeframe.items()}
        }

    @classmethod
    def load_from_json(cls, json_path: Optional[str] = None) -> "FeatureRegistry":
        if json_path is None:
            # Emplacement par défaut : config/feature_registry.json
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            json_path = os.path.join(repo_root, "config", "feature_registry.json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Fichier de registre introuvable : {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        entries = [
            FeatureEntry(
                name=d["name"],
                source=d["source"],
                famille=d["famille"],
                timeframe=d["timeframe"],
                type=d["type"],
                unite=d["unite"],
                fenetre=d["fenetre"],
                transformation=d["transformation"],
                dependances_deterministes=d.get("dependances_deterministes", []),
                frequence_maj=d["frequence_maj"],
                disponibilite_t=d["disponibilite_t"],
                role_potentiel=d["role_potentiel"]
            )
            for d in raw_data
        ]
        return cls(entries)


# Singleton d'accès global
_GLOBAL_REGISTRY: Optional[FeatureRegistry] = None


def get_feature_registry() -> FeatureRegistry:
    """Retourne l'instance singleton du registre des 1 026 variables."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = FeatureRegistry.load_from_json()
    return _GLOBAL_REGISTRY


def snapshot_values(registry, snapshot):
    """Resolve only values actually supplied by the existing causal snapshot.

    This is an audit adapter, NOT a second registry or a 1026-value filler.
    Unsupported declarations remain absent; missing values are never zeros.
    """
    import re
    import numpy as np

    bar_fields = {"open": 0, "high": 1, "low": 2, "close": 3,
                  "volume": 4, "wick_up": 5, "wick_down": 6, "body": 7}
    result = {}
    for entry in registry.all_variables():
        name = entry.name
        if name.startswith("bar_5m.") and name[7:] in bar_fields:
            result[name] = float(snapshot.bar_5m[bar_fields[name[7:]]])
            continue
        match = re.fullmatch(r"seq_5m\.lag_(\d+)\.(\w+)", name)
        if match and match[2] in bar_fields:
            lag = int(match[1])
            if 0 < lag < len(snapshot.seq_5m):
                result[name] = float(snapshot.seq_5m[-1 - lag, bar_fields[match[2]]])
            continue
        for prefix, suffix in (("c1h.", "1h"), ("c4h.", "4h")):
            if not name.startswith(prefix):
                continue
            field = name[len(prefix):]
            run = getattr(snapshot, "running_" + suffix)
            prev = getattr(snapshot, "prev_" + suffix)
            direct = {"open": run[0], "running_high": run[1],
                      "running_low": run[2], "running_vol": run[3],
                      "pos_in_" + suffix: getattr(snapshot, "pos_in_" + suffix),
                      "phase_" + suffix: getattr(snapshot, "phase_" + suffix),
                      "bar_index": round(getattr(snapshot, "phase_" + suffix) * (12 if suffix == "1h" else 48)),
                      "sweep_high": getattr(snapshot, "sweep_high_" + suffix),
                      "sweep_low": getattr(snapshot, "sweep_low_" + suffix)}
            direct.update({"prev_" + key: prev[j] for j, key in
                           enumerate(("open", "high", "low", "close", "volume"))})
            if field in direct:
                result[name] = float(direct[field])
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("Snapshot contains nonfinite audited variables")
    return result


def audit_existing_registry(registry_path="config/feature_registry.json",
                            data_path="data/processed/BTCUSDT_binance/BTCUSDT_5m_featured.parquet",
                            config_path="config/config.yaml"):
    """Evidence-based inventory. Declaration alone NEVER proves availability.

    Static config existence is reported separately from computed market values.
    Current config is NOT a point-in-time historical config archive. No values
    (especially credentials) are exported in this audit.
    """
    import hashlib
    import platform
    from pathlib import Path
    from collections import Counter
    import numpy as np
    import pandas as pd
    import yaml
    from adan_trading_bot.data.nested_state_builder import NestedStateBuilder

    path = Path(registry_path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = json.loads(path.read_text())
    registry = FeatureRegistry.load_from_json(str(path))
    frame = pd.read_parquet(data_path)
    columns = list(frame.columns)
    train = frame.loc[(frame.index >= "2017-01-01") & (frame.index < "2022-01-01")]
    for start in range(0, len(train) - 600, 600):
        sample = train.iloc[start:start + 600]
        ts_ns = sample.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        if (np.diff(ts_ns) == 300_000_000_000).all():
            break
    else:
        raise ValueError("No continuous TRAIN audit window")
    i = 300
    builder = NestedStateBuilder(sample)
    snapshot = builder.snapshot(i)
    if not snapshot.integrity_ok:
        raise ValueError("Invalid snapshot in audit")
    values = snapshot_values(registry, snapshot)
    edited = sample.copy()
    edited.iloc[i + 1:, edited.columns.get_loc("high")] *= 5
    edited.iloc[i + 1:, edited.columns.get_loc("volume")] *= 3
    other = snapshot_values(registry, NestedStateBuilder(edited).snapshot(i))
    if values != other:
        raise AssertionError("Audited snapshot values change when future is mutated")

    config = yaml.safe_load(Path(config_path).read_text())
    names = set(registry.all_names())
    rows = []
    for entry in raw:
        name = entry["name"]
        computed = name in values
        config_present = False
        if name.startswith("config."):
            obj = config
            try:
                for key in name[7:].split("."):
                    obj = obj[key]
                config_present = True
            except (KeyError, TypeError):
                pass
        sensitive = any(token in name.lower() for token in ("api_key", "api_secret", "password", "secret", "token"))
        hazards = []
        if entry["source"].startswith("labeler."):
            hazards.append("labeler_source_requires_asof_trace; reint label uses next close")
        if "atr_" in name and entry["source"] == "indicators.atr":
            hazards.append("declared_running_dependencies_do_not_match_completed_hour_ATR")
        if name in ("risk.max_portfolio_exposure_pct", "risk.max_trade_allocation_pct", "risk.min_order_value_usdt"):
            hazards.append("obsolete_micro_capital_contract")
        if sensitive:
            hazards.append("sensitive_operational_setting; never feed model or export value")
        if name.startswith("config.reward_shaping."):
            hazards.append("legacy_PPO_reward_setting_not_market_observation")
        deps = entry["dependances_deterministes"]
        rows.append({"name": name, "source": entry["source"], "family": entry["famille"],
                     "subfamily": entry.get("sous_famille"), "timeframe": entry["timeframe"],
                     "type": entry["type"], "unit": entry["unite"], "window": entry["fenetre"],
                     "transformation": entry["transformation"], "lineage": entry.get("lineage"),
                     "dependencies": deps, "update_frequency": entry["frequence_maj"],
                     "role": entry["role_potentiel"], "declared": True,
                     "computed_in_existing_snapshot": computed,
                     "available_at_t_verified": computed,
                     "future_safe_verified": computed,
                     "verification_scope": "continuous_TRAIN_window_future_mutation" if computed else "unverified",
                     "static_config_path_exists_now": config_present,
                     "historical_config_asof_verified": False if name.startswith("config.") else None,
                     "dataset_column_present": name in columns,
                     "derived_declared": bool(deps),
                     "missing_dependency_names": [d for d in deps if d not in names],
                     "missing_metadata": [key for key in ("sous_famille", "lineage", "future_safe") if key not in entry],
                     "future_dependency_status": "unknown" if hazards else ("not_observed_in_test_scope" if computed else "unverified"),
                     "hazards": hazards,
                     "evidence": "NestedStateBuilder.snapshot; unchanged under future mutation" if computed else
                                 ("YAML path exists, no runtime/asof proof" if config_present else "declaration_only")})
    summary = {"declared": len(rows), "computed_market_snapshot_verified": len(values),
               "available_at_t_verified": sum(x["available_at_t_verified"] for x in rows),
               "future_safe_verified_in_test_scope": sum(x["future_safe_verified"] for x in rows),
               "derived_declared": sum(x["derived_declared"] for x in rows),
               "static_config_declared": sum(x["name"].startswith("config.") for x in rows),
               "static_config_paths_present_now": sum(x["static_config_path_exists_now"] for x in rows),
               "future_dependent_confirmed": 0,
               "future_dependency_unverified": sum(not x["future_safe_verified"] for x in rows),
               "label_source_hazards": sum(x["source"].startswith("labeler.") for x in rows),
               "entries_missing_metadata": sum(bool(x["missing_metadata"]) for x in rows),
               "unresolved_dependency_references": sum(len(x["missing_dependency_names"]) for x in rows),
               "by_family": dict(Counter(x["family"] for x in rows))}
    assert len(rows) == 1026 and len(names) == 1026
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    return {"kind": "AUDIT_EVIDENCE_NOT_A_SECOND_REGISTRY", "registry_sha256": before,
            "summary": summary, "dataset": data_path, "config": config_path,
            "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
            "sample": {"rows": len(sample), "start": str(sample.index[0]),
                       "end": str(sample.index[-1]), "decision": str(sample.index[i]),
                       "seed": None, "selection": "first_continuous_600_bar_TRAIN_window"},
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "pyyaml": yaml.__version__},
            "interpretation": "Counts are verified lower bounds in the stated test scope; unknown is not safe. Static paths are not calculated market variables. No registry rows deleted or rewritten.",
            "entries": rows}


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Audit the existing registry, never recreate it")
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    target = arguments.report.resolve()
    if not target.is_relative_to(Path("/home/ubuntu/webapp")) or not target.parent.is_dir():
        raise ValueError("Report parent must exist within workspace")
    result = audit_existing_registry()
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["summary"], indent=2))
