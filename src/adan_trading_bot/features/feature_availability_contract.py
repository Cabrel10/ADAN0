"""Mandatory STATE(t) boundary: a declaration is not an available feature.

The full canonical registry is retained. This contract resolves a named subset
without filling unknowns with zero and without flattening it into a model input.
Gates for labels, plans, geometry and GPU training remain separate and blocked.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from adan_trading_bot.data import nested_state_builder as producer
from adan_trading_bot.features.feature_registry import snapshot_values


class FeatureAvailabilityError(ValueError):
    pass


class FeatureAvailabilityContract:
    FORBIDDEN_CATEGORIES = {"LABEL_ONLY", "CONFIG_ONLY", "UNRESOLVED"}
    OUTCOME_NAMES = {"next_close", "future_high", "future_low", "tp_first", "sl_first",
                     "mfe", "mae", "time_to_tp", "time_to_sl", "net_return"}
    training_authorized = False

    def __init__(self, registry):
        self.registry = registry
        self.producer_hash = hashlib.sha256(Path(producer.__file__).read_bytes()).hexdigest()
        self.adapter_hash = hashlib.sha256(inspect.getsource(snapshot_values).encode()).hexdigest()

    def require(self, name):
        if name not in self.registry:
            raise FeatureAvailabilityError(f"Not declared: {name}")
        entry = self.registry[name]
        leaf = name.rsplit(".", 1)[-1].lower()
        if (entry.category in self.FORBIDDEN_CATEGORIES or entry.source.startswith("labeler.")
                or name.startswith(("config.", "y_")) or leaf in self.OUTCOME_NAMES):
            raise FeatureAvailabilityError(f"Prohibited STATE entry: {name} ({entry.category})")
        if (entry.status != "RESOLVED" or entry.future_safe != "VERIFIED"
                or not entry.available_at_t or entry.missing_runtime_mapping):
            raise FeatureAvailabilityError(f"No verified availability-at-t: {name}")
        proof = entry.verification or {}
        if (proof.get("passed") is not True or proof.get("feature") != name
                or proof.get("test") != "snapshot_future_mutation"
                or not proof.get("mutations") or not proof.get("sample")):
            raise FeatureAvailabilityError(f"Missing per-feature mutation evidence: {name}")
        if (proof.get("producer_sha256") != self.producer_hash
                or proof.get("adapter_sha256") != self.adapter_hash):
            raise FeatureAvailabilityError(f"Stale source hash; rerun future mutation audit: {name}")
        if not entry.lineage or entry.lineage.get("kind") != "RUNTIME_VERIFIED":
            raise FeatureAvailabilityError(f"Missing verified lineage: {name}")
        return entry

    def eligible_names(self):
        names = []
        for name in self.registry.all_names():
            try:
                self.require(name)
            except FeatureAvailabilityError:
                continue
            names.append(name)
        return names

    def materialize(self, snapshot, names=None):
        if not snapshot.integrity_ok:
            raise FeatureAvailabilityError("Integrity veto: no STATE from invalid snapshot")
        selected = self.eligible_names() if names is None else list(names)
        if len(selected) != len(set(selected)):
            raise FeatureAvailabilityError("Duplicate requested names")
        for name in selected:
            self.require(name)
        values = snapshot_values(self.registry, snapshot)
        missing = set(selected) - values.keys()
        if missing:
            raise FeatureAvailabilityError(f"Runtime resolver missing: {sorted(missing)}")
        return {name: values[name] for name in selected}

    def require_training(self):
        raise FeatureAvailabilityError("GPU/training BLOCKED: labels, plans and geometry gates not locked")
