#!/usr/bin/env python3
"""Audit comportemental complet du run BTC v37 + ACP.

Questions:
  1. Gestion de l'energie (decision_budget) par l'agent : trajectoire, usage.
  2. EV (explained_variance) du value function au fil du run.
  3. Taux de rejet par cause.
  4. ACP sur les features comportementales par step -> profil :
     trader debutant (structure, directionnalite, gestion du risque)
     vs spammeur idiot (bruit, churn sterile, aucune structure).

Sources:
  logs/v37_500k/btc_500k.log        (budget, ACTION_DIFF, TARGET_WEIGHT, SB3)
  logs/rewards/worker_0_rewards_20260903_*.jsonl  (reward breakdown par step)
"""
import json
import re
import sys
import ast
import glob
from collections import Counter

import numpy as np

LOG = "logs/v37_500k/btc_500k.log"

# ---------------------------------------------------------------- log parsing
BUDGET_RE = re.compile(r"\[ACTION_DIFF\] Step (\d+) .*?budget=([\d.]+)/([\d.]+)")
TW_RE = re.compile(r"\[TARGET_WEIGHT\] Step (\d+) \| \S+ \| Action=(\w+) \| "
                   r"Raw=(-?[\d.eE+-]+) \| Thr=([\d.]+)")
PIPE_RE = re.compile(r"pipeline=(\{.*\})\s*$")
EV_RE = re.compile(r"explained_variance\s*\|\s*(-?[\d.]+)")
KL_RE = re.compile(r"approx_kl\s*\|\s*(-?[\d.]+)")
CLIP_RE = re.compile(r"clip_fraction\s*\|\s*(-?[\d.]+)")
ENT_RE = re.compile(r"entropy_loss\s*\|\s*(-?[\d.]+)")

budgets = []          # (step, budget, budget_max)
raw_actions = []      # (step, raw_a0, thr, routed)
ev_series, kl_series, clip_series, ent_series = [], [], [], []
pipe_last = {}

with open(LOG, encoding="utf-8", errors="replace") as f:
    for line in f:
        m = BUDGET_RE.search(line)
        if m:
            budgets.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
            pm = PIPE_RE.search(line)
            if pm:
                try:
                    d = ast.literal_eval(pm.group(1))
                    if "policy" in d:
                        pipe_last = d
                except Exception:
                    pass
            continue
        m = TW_RE.search(line)
        if m:
            raw_actions.append((int(m.group(1)), float(m.group(3)),
                                float(m.group(4)), m.group(2)))
            continue
        for rx, acc in ((EV_RE, ev_series), (KL_RE, kl_series),
                        (CLIP_RE, clip_series), (ENT_RE, ent_series)):
            m = rx.search(line)
            if m:
                acc.append(float(m.group(1)))

# ------------------------------------------------------- reward jsonl parsing
FEATURES = ["pnl", "drawdown_penalty", "behavior_penalty", "inaction_penalty",
            "patience_bonus", "closure_bonus", "promotion_bonus",
            "demotion_penalty", "future_contrib", "saturation_penalty",
            "final_reward"]

rows = []  # (step, feature vector, action_total)
for path in sorted(glob.glob("logs/rewards/worker_0_rewards_20260903_*.jsonl")):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rw = rec.get("reward")
            if not isinstance(rw, dict):
                continue
            bd = rw.get("breakdown") or {}
            if not isinstance(bd, dict):
                continue
            row = [float(bd.get(k, 0.0) or 0.0) for k in FEATURES[:-1]]
            row.append(float(rw.get("total", 0.0) or 0.0))
            rows.append((int(rec.get("step", 0)), row))

# ------------------------------------------------------------- report helpers
def pct(x, total):
    return 100.0 * x / max(1, total)

print("=" * 78)
print("AUDIT COMPLET — RUN BTC v37 (arrete au critere 135k steps / peak 20.51)")
print("=" * 78)

# --- 1. ENERGIE ---
print("\n[1] GESTION DE L'ENERGIE (decision_budget)")
if budgets:
    b = np.array([x[1] for x in budgets])
    print(f"  observations        : {len(b)} (echantillon 1/50 steps)")
    print(f"  moyenne             : {b.mean():.3f} / {budgets[0][2]:.2f}")
    print(f"  mediane             : {np.median(b):.3f}")
    print(f"  min / max           : {b.min():.3f} / {b.max():.3f}")
    print(f"  % temps a 0 (epuise): {pct((b < 0.01).sum(), len(b)):.1f}%")
    print(f"  % temps > 0.9       : {pct((b > 0.9).sum(), len(b)):.1f}%")
    print(f"  % bande utile .1-.9 : {pct(((b >= 0.1) & (b <= 0.9)).sum(), len(b)):.1f}%")
    # transitions epuisement
    exhausted = int(((b[1:] < 0.01) & (b[:-1] >= 0.01)).sum())
    print(f"  episodes d'epuisement (descente sous 0.01) : {exhausted}")
    verdict_e = ("l'agent VIT dans la bande utile (gestion active)"
                 if 0.05 < ((b >= 0.1) & (b <= 0.9)).mean() < 0.95
                 else "budget sature/epuise en permanence (pas de gestion)")
    print(f"  -> {verdict_e}")

# --- 2. EV / SB3 ---
print("\n[2] APPRENTISSAGE PPO (EV / KL / clip / entropy)")
def _stat(name, s):
    if not s:
        print(f"  {name:20s}: n/a"); return
    a = np.array(s)
    print(f"  {name:20s}: n={len(a):4d} first={a[0]:+.3f} last={a[-1]:+.3f} "
          f"mean={a.mean():+.3f} min={a.min():+.3f} max={a.max():+.3f}")
_stat("explained_variance", ev_series)
_stat("approx_kl", kl_series)
_stat("clip_fraction", clip_series)
_stat("entropy_loss", ent_series)
if ev_series:
    pos = sum(1 for x in ev_series if x > 0)
    print(f"  EV > 0 sur {pos}/{len(ev_series)} updates ({pct(pos, len(ev_series)):.0f}%)")
    print("  -> EV negative quasi permanente = value function JAMAIS calibree"
          if pos < 0.1 * len(ev_series) else "  -> EV parfois positive")

# --- 3. REJETS ---
print("\n[3] REJETS (compteurs finaux cumules)")
if pipe_last:
    pol = pipe_last.get("policy", 0)
    for k, v in pipe_last.items():
        if k != "policy":
            print(f"  {k:24s} {v:7d}  ({pct(v, pol):5.1f}% des {pol} decisions)")

# --- 4. RAW ACTION distribution ---
print("\n[4] DISTRIBUTION DES INTENTIONS BRUTES (echantillon TARGET_WEIGHT)")
if raw_actions:
    cats = Counter()
    raws = []
    for _, a0, thr, routed in raw_actions:
        raws.append(a0)
        if a0 > thr:
            cats["intent_BUY"] += 1
        elif a0 < -thr:
            cats["intent_SELL"] += 1
        else:
            cats["neutral"] += 1
    r = np.array(raws)
    n = len(raw_actions)
    for k in ("intent_BUY", "intent_SELL", "neutral"):
        print(f"  {k:14s} {cats[k]:5d} ({pct(cats[k], n):5.1f}%)")
    print(f"  raw a0 : mean={r.mean():+.3f} std={r.std():.3f} "
          f"min={r.min():+.3f} max={r.max():+.3f}")
    # autocorrelation lag-1 : spammeur = serie chaotique/constante ; trader = structure
    if n > 10:
        ac1 = np.corrcoef(r[:-1], r[1:])[0, 1]
        print(f"  autocorrelation lag-1 de a0 : {ac1:+.3f} "
              f"({'serie lisse/persistante' if ac1 > 0.5 else 'serie alternee' if ac1 < -0.3 else 'faible memoire'})")

# --- 5. ACP sur reward breakdown ---
print("\n[5] ACP — PROFIL COMPORTEMENTAL (reward breakdown par step)")
if len(rows) > 50:
    X = np.array([r[1] for r in rows], dtype=float)
    X = X[~np.isnan(X).any(axis=1)]
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (X - mu) / sd
    cov = np.cov(Z.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    total_var = vals.sum()
    print(f"  steps exploitables : {len(Z)}")
    for i in range(min(4, len(vals))):
        print(f"  PC{i+1}: {100*vals[i]/total_var:5.1f}% variance expliquee")
    print("\n  Loadings PC1 (ce qui structure le plus les recompenses) :")
    for j, k in enumerate(FEATURES):
        print(f"    {k:22s} {vecs[j,0]:+.3f}")
    print("\n  Loadings PC2 :")
    for j, k in enumerate(FEATURES):
        print(f"    {k:22s} {vecs[j,1]:+.3f}")

    # --- verdict quantifie ---
    print("\n[6] VERDICT — TRADER DEBUTANT ou SPAMMEUR ?")
    score, notes = 0, []
    # a) recompenses liees au PnL dominent-elles les penalites steriles ?
    pnl_var = np.var(X[:, FEATURES.index("pnl")])
    beh_var = np.var(X[:, FEATURES.index("behavior_penalty")])
    fr = X[:, FEATURES.index("final_reward")]
    neg = (fr < -0.01).mean()
    pos = (fr > 0.01).mean()
    print(f"  var(pnl_reward)={pnl_var:.4f} vs var(behavior_penalty)={beh_var:.4f}")
    print(f"  steps reward>+0.01 : {pct(pos,1):.1f}% | reward<-0.01 : {pct(neg,1):.1f}%")
    if pos > 0.02:
        score += 2; notes.append("des steps gagnants existent (+2)")
    if pnl_var > beh_var:
        score += 2; notes.append("la variance vient du PnL, pas des penalites (+2)")
    else:
        notes.append("la variance vient des penalites, pas du PnL (+0)")
    # b) intentions
    if raw_actions:
        sell_flat = pct(cats["intent_SELL"], n)
        if sell_flat > 80:
            score -= 2; notes.append(f"intentions SELL {sell_flat:.0f}% alors que FLAT la plupart du temps (-2)")
        if cats["intent_BUY"] > 0:
            score += 1; notes.append("des intents BUY existent (+1)")
    # c) EV
    if ev_series and sum(1 for x in ev_series if x > 0) == 0:
        score -= 2; notes.append("EV jamais positive : aucune valeur apprise (-2)")
    # d) gestion energie
    if budgets and 0.05 < ((b >= 0.1) & (b <= 0.9)).mean():
        score += 1; notes.append("budget utilise dans la bande utile (+1)")
    print(f"\n  SCORE COMPORTEMENTAL : {score:+d}")
    for t in notes:
        print(f"   - {t}")
    if score >= 3:
        print("\n  => TRADER DEBUTANT : structure detectable, erreurs d'apprentissage")
    elif score >= 0:
        print("\n  => PROFIL MIXTE / APPRENTI NON CONVERGE : ni trader ni pur spammeur")
    else:
        print("\n  => SPAMMEUR : pas de structure exploitable, churn sterile domine")
else:
    print("  pas assez de steps reward pour l'ACP")
