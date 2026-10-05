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
from dataclasses import dataclass, asdict, fields
from typing import Dict, List, Optional, Set, Union


VALID_ROLES = {"perception", "contexte", "plan", "risque", "portefeuille"}
VALID_TIMEFRAMES = {"5m", "1h", "4h", "trade", "1d", "global"}
VALID_CATEGORIES = {"MARKET_FEATURE", "CONTEXT_FEATURE", "PORTFOLIO_FEATURE",
                    "PLAN_FEATURE", "RISK_FEATURE", "DERIVED_FEATURE", "LABEL_ONLY",
                    "CONFIG_ONLY", "UNRESOLVED"}
FUTURE_SAFE_STATUSES = {"VERIFIED", "UNKNOWN", "UNSAFE"}


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
    category: str = "UNRESOLVED"
    status: str = "UNRESOLVED"
    future_safe: str = "UNKNOWN"
    available_at_t: bool = False
    reason: str = "Availability has not been demonstrated"
    missing_source: Optional[bool] = None
    missing_runtime_mapping: bool = True
    lineage: Optional[dict] = None
    verification: Optional[dict] = None
    atr_definition: Optional[dict] = None
    sous_famille: Optional[str] = None

    def __post_init__(self):
        if self.category not in VALID_CATEGORIES or self.future_safe not in FUTURE_SAFE_STATUSES:
            raise ValueError(f"Invalid classification or future-safe status for {self.name}")
        if self.status not in {"RESOLVED", "UNRESOLVED"}:
            raise ValueError(f"Invalid resolution status for {self.name}")
        if self.future_safe == "VERIFIED" and not self.verification:
            raise ValueError(f"VERIFIED requires mutation-test evidence: {self.name}")
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
                role_potentiel=d["role_potentiel"],
                **{key: d[key] for key in ("category", "status", "future_safe", "available_at_t",
                                          "reason", "missing_source", "missing_runtime_mapping", "lineage",
                                          "verification", "atr_definition", "sous_famille") if key in d}
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
        if name in ("c1h.atr_1h", "c1h.atr_1h_pct"):
            observation = getattr(snapshot, "atr_1h", None)
            if observation is not None and observation.available:
                result[name] = float(observation.value if name == "c1h.atr_1h" else observation.fraction)
            continue
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
    # Modify ALL future raw OHLCV fields, preserving future OHLC coherence.
    for column in ("open", "high", "low", "close"):
        edited.iloc[i + 1:, edited.columns.get_loc(column)] *= 3
    edited.iloc[i + 1:, edited.columns.get_loc("volume")] *= 5
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
                     "category": entry.get("category", "UNRESOLVED"),
                     "resolution_status": entry.get("status", "UNRESOLVED"),
                     "future_safe": entry.get("future_safe", "UNKNOWN"),
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
               "by_family": dict(Counter(x["family"] for x in rows)),
               "classification": {c: sum(x["category"] == c for x in rows) for c in sorted(VALID_CATEGORIES)},
               "future_safe_status_counts": {s: sum(x["future_safe"] == s for x in rows) for s in ("VERIFIED", "UNKNOWN", "UNSAFE")}}
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


def reconcile_atr_definitions(registry_path="config/feature_registry.json"):
    """Record explicit semantic decisions without renaming or resolving variables.

    The 1h baseline producer already passed GATE 1. The snapshot does NOT
    currently carry ATR; a geometry float argument is not provenance proof.
    Unknown mappings remain UNKNOWN. Running range/phase is never called ATR.
    """
    from pathlib import Path
    path = Path(registry_path).resolve()
    if not path.is_relative_to(Path("/home/ubuntu/webapp")):
        raise ValueError("Registry outside workspace")
    entries = json.loads(path.read_text())
    decisions = []
    for entry in entries:
        name = entry["name"]
        if "atr" not in name.lower():
            continue
        base = {"name": name, "declared_source": entry["source"],
                "timeframe": entry["timeframe"], "availability": "UNRESOLVED",
                "causal_contract": "no future fill; warmup unavailable; no implicit zero",
                "name_changed": False, "declared_dependencies": entry["dependances_deterministes"]}
        if name.startswith("config.") or name.startswith("risk."):
            base.update(formula=entry["transformation"], kind="SETTING_NOT_AN_ATR_OBSERVATION",
                        runtime_source=entry["source"], consistency="not a volatility measurement")
        elif name in ("atr_14", "atr_5m_pct"):
            base.update(kind="LEGACY_5M_ATR", definition_version="legacy-5m-rma14",
                        formula="Wilder/RMA14 of TR_5m; divided by close_t" if name.endswith("pct") else
                                "Wilder/RMA14(TR_5m), TR=max(H-L,abs(H-prevC),abs(L-prevC))",
                        runtime_source="data_processing.feature_engineer: pandas_ta ATR default; exact backend/seed unverified",
                        smoothing="Wilder/RMA, NOT arithmetic SMA",
                        consistency="labeler.atr14 uses SMA14 at 5m; distinct legacy definition, not silently aliased",
                        causal_contract="5m bars through t only; warmup and backend init must be validated")
        elif name.startswith(("c1h.", "c4h.")):
            hours = 1 if name.startswith("c1h.") else 4
            ratio = "range_to_atr" in name
            fraction = name.endswith("_pct")
            formula = f"SMA14(TR of COMPLETE {hours}h containers), TR=max(H-L,abs(H-prevC),abs(L-prevC)); lag 1 container"
            if fraction:
                formula += "; divide by close_5m(t), fraction not percent-points"
            if ratio:
                formula = f"running_range_{hours}h(t) / canonical_atr_{hours}h(t)"
            parents = ([{"name": f"c{hours}h.running_range", "time_offset_bars": 0},
                        {"name": f"c{hours}h.atr_{hours}h", "time_offset_bars": 0}] if ratio else
                       [{"name": f"c{hours}h.prev_high", "time_offset_bars": "last_14_complete_containers"},
                        {"name": f"c{hours}h.prev_low", "time_offset_bars": "last_14_complete_containers"},
                        {"name": f"c{hours}h.prev_close", "time_offset_bars": "last_15_complete_containers"}])
            if fraction:
                parents = [{"name": f"c{hours}h.atr_{hours}h", "time_offset_bars": 0},
                           {"name": "bar_5m.close", "time_offset_bars": 0}]
            base.update(kind="CANONICAL_COMPLETED_CONTAINER_ATR", formula=formula,
                        definition_version=f"systemone-{hours}h-sma14-tr-lag1-v1", smoothing="arithmetic SMA14",
                        runtime_source="offline.labeler_mfe_mae.compute_true_atr_1h" if hours == 1 else None,
                        availability=f"previous 14 full consecutive {hours}h containers; UTC bar-open timestamps; current container excluded even at its final 5m close",
                        causal_contract="complete containers only; lag one; gaps reset warmup; no bfill; fraction normalization uses causal current close",
                        snapshot_consistency="NOT_WIRED: existing LivingStateSnapshot contains no ATR field",
                        geometry_consistency="NOT_PROVEN: scalar atr_1h_pct lacks provenance; legacy _atr_pct_from_snapshot now raises instead of returning a range/phase proxy",
                        labeler_consistency="1h base implemented and GATE1 validated; normalized value is ATR / close_t" if hours == 1 else "4h canonical implementation unresolved",
                        corrected_parents=parents,
                        consistency="explicit definition decision; old running-H/L parent claims retained as historical declaration, not verified constraints")
            entry["lineage"] = dict(entry["lineage"], parents=parents,
                                    definition_status="DECLARED_CANONICAL_NOT_RUNTIME_RESOLVED")
        else:
            base.update(kind="UNRESOLVED_ATR_RELATED", formula=entry["transformation"], runtime_source=None,
                        consistency="needs explicit source trace")
        entry["atr_definition"] = base
        decisions.append(base)
    assert len(entries) == 1026
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    return decisions


def classify_existing_registry(registry_path="config/feature_registry.json"):
    """Enrich entries IN PLACE after rerunning mutation evidence; never recreate.

    Category and resolution are different: LABEL_ONLY/CONFIG_ONLY may remain
    UNRESOLVED for runtime mapping. All original declarations are preserved.
    """
    import re
    import hashlib
    import inspect
    from pathlib import Path
    from collections import Counter
    from adan_trading_bot.data import nested_state_builder as producer

    path = Path(registry_path).resolve()
    if not path.is_relative_to(Path("/home/ubuntu/webapp")):
        raise ValueError("Registry must remain within workspace")
    raw = json.loads(path.read_text())
    names_before = [x["name"] for x in raw]
    audit = audit_existing_registry(str(path))
    audited = {row["name"]: row for row in audit["entries"]}
    for entry in raw:
        row = audited[entry["name"]]
        verified = row["computed_in_existing_snapshot"]
        name = entry["name"]
        if name.startswith("config."):
            category = "CONFIG_ONLY"
        elif entry["source"].startswith("labeler."):
            category = "LABEL_ONLY"
        elif not verified:
            category = "UNRESOLVED"
        elif entry["famille"] == "portfolio":
            category = "PORTFOLIO_FEATURE"
        elif entry["famille"] == "plan":
            category = "PLAN_FEATURE"
        elif entry["famille"] == "risk_limits":
            category = "RISK_FEATURE"
        elif entry["transformation"] in ("k_div_12", "m_div_48", "k_raw", "m_raw") or name.endswith("bar_index") or ".phase_" in name:
            category = "CONTEXT_FEATURE"
        elif entry["transformation"] == "raw":
            category = "MARKET_FEATURE"
        else:
            category = "DERIVED_FEATURE"
        lag = re.search(r"seq_5m\.lag_(\d+)\.", name)
        offset = -int(lag[1]) if lag else 0
        parents = [{"name": p, "time_offset_bars": offset}
                   for p in entry["dependances_deterministes"]]
        # Previous completed OHLCV was missing explicit parents in old metadata.
        if ".prev_" in name and not parents:
            field = name.split(".prev_", 1)[1]
            if field in ("open", "high", "low", "close", "volume"):
                parents = [{"name": "bar_5m." + field, "time_offset_bars": "previous_complete_container"}]
        if category == "LABEL_ONLY":
            reason = "Ex-post label source; prohibited in STATE even if a future mapping is later resolved"
        elif category == "CONFIG_ONLY":
            reason = "Static operational setting, not market data; no mutation/asof runtime proof"
        elif verified:
            reason = "Existing snapshot resolver passed raw TRAIN future-mutation test"
        else:
            reason = "No demonstrated value-at-t producer/runtime mapping; declared causal_t is insufficient"
        evidence = None
        if verified:
            evidence = {"test": "snapshot_future_mutation", "passed": True,
                        "feature": name, "sample": audit["sample"],
                        "mutations": ["ALL_OHLC_after_t_times_3", "volume_after_t_times_5"],
                        "producer_sha256": hashlib.sha256(Path(producer.__file__).read_bytes()).hexdigest(),
                        "adapter_sha256": hashlib.sha256(inspect.getsource(snapshot_values).encode()).hexdigest()}
            if name in ("c1h.atr_1h", "c1h.atr_1h_pct"):
                from adan_trading_bot.data import canonical_atr
                evidence["canonical_source_sha256"] = hashlib.sha256(Path(canonical_atr.__file__).read_bytes()).hexdigest()
                evidence["bridge_tests"] = "tests/test_canonical_atr_bridge.py: source identity, value-by-value equality, future and elapsed-current-hour mutations, warmup/gaps"

        atr_parents = (entry.get("atr_definition") or {}).get("corrected_parents")
        if atr_parents is not None:
            parents = atr_parents
        entry.update(category=category, status="RESOLVED" if verified else "UNRESOLVED",
                     future_safe="VERIFIED" if verified else ("UNSAFE" if entry.get("future_safe") == "UNSAFE" else "UNKNOWN"),
                     available_at_t=verified,
                     reason=reason,
                     missing_source=False if verified or row["static_config_path_exists_now"] else True,
                     missing_runtime_mapping=not verified,
                     sous_famille=entry.get("sous_famille") or entry["source"].split(".")[0],
                     lineage={"kind": "STATIC_CONFIG" if category == "CONFIG_ONLY" else
                              "EX_POST_ONLY" if category == "LABEL_ONLY" else
                              "RUNTIME_VERIFIED" if verified else "DECLARED_UNVERIFIED",
                              "source": entry["source"], "operation": entry["transformation"],
                              "parents": parents,
                              "runtime_mapping": "snapshot_values:" + name if verified else None},
                     verification=evidence)
        # Keep causal_t only as a historic declaration, never as the contract.
        entry.setdefault("declared_disponibilite_t", entry["disponibilite_t"])
        entry["disponibilite_t"] = "VERIFIED_AT_T" if verified else "UNRESOLVED"
    assert [x["name"] for x in raw] == names_before and len(raw) == 1026
    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
    counts = Counter(x["category"] for x in raw)
    safe = Counter(x["future_safe"] for x in raw)
    return {"classification": {k: counts[k] for k in sorted(VALID_CATEGORIES)},
            "future_safe": {k: safe[k] for k in ("VERIFIED", "UNKNOWN", "UNSAFE")},
            "resolution": dict(Counter(x["status"] for x in raw)),
            "registry_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


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
