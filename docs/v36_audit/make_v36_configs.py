#!/usr/bin/env python3
"""Génère les 3 configs d'ablation V36 par override ciblé du config de base.
Ne touche JAMAIS config/config.yaml (référence). Écrit config/config_v36{a,b,c}.yaml.
Isolation causale : chaque bras change UNE famille de paramètres.
"""
import yaml, copy, os

BASE = "/home/ubuntu/webapp/MORNINGSTAR/ADAN0/config/config.yaml"
OUTDIR = "/home/ubuntu/webapp/MORNINGSTAR/ADAN0/config"

with open(BASE) as f:
    base = yaml.safe_load(f)

def setk(cfg, path, val):
    """set nested key 'a.b.c' = val, creating dicts as needed."""
    d = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        if p not in d or not isinstance(d[p], dict):
            d[p] = {}
        d = d[p]
    old = d.get(parts[-1], "<absent>")
    d[parts[-1]] = val
    print(f"    {path}: {old} -> {val}")

# ---- V36-A : Finance pure ---------------------------------------------------
print("[V36-A] Finance pure (signal financier seul + drawdown borné)")
a = copy.deepcopy(base)
setk(a, "reward_shaping.future_reward.enabled", False)
setk(a, "trading_rules.symmetry_enforcement.enabled", False)
setk(a, "trading_rules.close_intention_penalty.enabled", False)
with open(f"{OUTDIR}/config_v36a.yaml", "w") as f:
    yaml.safe_dump(a, f, sort_keys=False, allow_unicode=True)

# ---- V36-B : Finance + Future Arena fortement borné -------------------------
print("[V36-B] Finance + Future Arena borné (max_future_contrib 0.60->0.15)")
b = copy.deepcopy(base)
setk(b, "reward_shaping.future_reward.enabled", True)
setk(b, "reward_shaping.future_reward.max_future_contrib", 0.15)
setk(b, "trading_rules.symmetry_enforcement.enabled", False)
setk(b, "trading_rules.close_intention_penalty.enabled", False)
with open(f"{OUTDIR}/config_v36b.yaml", "w") as f:
    yaml.safe_dump(b, f, sort_keys=False, allow_unicode=True)

# ---- V36-C : Symmetry réconcilié avec free_sltp -----------------------------
print("[V36-C] Symmetry léger, réconcilié avec SL/TP libres")
c = copy.deepcopy(base)
setk(c, "reward_shaping.future_reward.enabled", False)
setk(c, "trading_rules.symmetry_enforcement.enabled", True)
setk(c, "trading_rules.symmetry_enforcement.rr_tolerance", 1.5)
setk(c, "trading_rules.symmetry_enforcement.max_step_penalty", 0.03)
setk(c, "trading_rules.close_intention_penalty.enabled", False)
with open(f"{OUTDIR}/config_v36c.yaml", "w") as f:
    yaml.safe_dump(c, f, sort_keys=False, allow_unicode=True)

print("\nOK. 3 configs écrites. config/config.yaml INTACT.")
