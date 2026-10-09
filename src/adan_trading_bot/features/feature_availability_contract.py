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
from adan_trading_bot.data import canonical_atr
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
        self.canonical_hash = hashlib.sha256(Path(canonical_atr.__file__).read_bytes()).hexdigest()

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
        if name in ("c1h.atr_1h", "c1h.atr_1h_pct") and proof.get("canonical_source_sha256") != self.canonical_hash:
            raise FeatureAvailabilityError("Stale canonical ATR source proof")
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
        if any(name in ("c1h.atr_1h", "c1h.atr_1h_pct") for name in selected):
            self.validate_snapshot_atr(snapshot)
        values = snapshot_values(self.registry, snapshot)
        missing = set(selected) - values.keys()
        if missing:
            raise FeatureAvailabilityError(f"Runtime resolver missing: {sorted(missing)}")
        return {name: values[name] for name in selected}

    def validate_snapshot_atr(self, snapshot):
        import math
        import pandas as pd
        observation = getattr(snapshot, "atr_1h", None)
        if observation is None or not observation.available:
            raise FeatureAvailabilityError("ATR unavailable: warmup/gap/invalid source; no zero or NaN model input")
        if (observation.definition != canonical_atr.ATR_DEFINITION or observation.lag_containers != 1
                or observation.unit != "quote_price" or observation.fraction_unit != "fraction_ATR_div_close_t"
                or len(observation.source_hours) != 14 or not math.isfinite(observation.value)
                or not math.isfinite(observation.fraction) or observation.value < 0
                or not math.isfinite(snapshot.price) or snapshot.price <= 0):
            raise FeatureAvailabilityError("Invalid ATR definition, units or provenance")
        timestamp = pd.Timestamp(snapshot.timestamp)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC").tz_localize(None)
        available_at = timestamp.floor("h")
        if observation.available_at != available_at:
            raise FeatureAvailabilityError("ATR availability clock mismatch")
        for j, hour in enumerate(observation.source_hours):
            if hour.start != available_at - pd.Timedelta(hours=14 - j):
                raise FeatureAvailabilityError("ATR source hours incomplete, nonconsecutive or current-hour contaminated")
            tr = hour.high - hour.low if hour.bootstrap else max(hour.high - hour.low,
                         abs(hour.high - hour.previous_close), abs(hour.low - hour.previous_close))
            if hour.bootstrap and j != 0:
                raise FeatureAvailabilityError("ATR bootstrap inside consecutive source window")
            if not math.isfinite(tr) or not math.isclose(tr, hour.true_range, rel_tol=1e-12, abs_tol=1e-9):
                raise FeatureAvailabilityError("ATR true-range provenance mismatch")
            if j and hour.previous_close != observation.source_hours[j - 1].close:
                raise FeatureAvailabilityError("ATR previous-close lineage mismatch")
        mean = sum(x.true_range for x in observation.source_hours) / 14
        if not math.isclose(observation.value, mean, rel_tol=1e-12, abs_tol=1e-9):
            raise FeatureAvailabilityError("ATR SMA14 mismatch")
        if not math.isclose(observation.fraction, observation.value / snapshot.price, rel_tol=1e-12, abs_tol=1e-12):
            raise FeatureAvailabilityError("ATR fraction units mismatch (possible factor 100)")
        return observation

    def require_training(self):
        raise FeatureAvailabilityError("GPU/training BLOCKED: labels, plans and geometry gates not locked")
