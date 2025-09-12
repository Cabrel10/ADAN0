import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Paramètres de simulation
N = 500  # Nombre de pas de temps
start_capital = 10000

np.random.seed(42)

timestamps = [datetime.now() - timedelta(minutes=5*(N-i)) for i in range(N)]
capital = [start_capital]
drawdown = [0.0]
sl_pct = [0.02]
tp_pct = [0.04]
risk_mode = []
reward_boost = []
penalty_inaction = []

mode_choices = ['NORMAL', 'DEFENSIVE', 'AGGRESSIVE']

for i in range(1, N):
    # Simuler capital
    change = np.random.normal(0, 10)
    capital.append(capital[-1] + change)
    # Simuler drawdown
    peak = max(capital)
    dd = max(0, (peak - capital[-1]) / peak * 100)
    drawdown.append(dd)
    # SL/TP
    sl = 0.02 + 0.01 * (dd > 5) + 0.01 * (dd > 10)
    tp = 0.04 + 0.01 * (dd > 10)
    sl_pct.append(sl)
    tp_pct.append(tp)
    # Mode de risque
    if dd > 10:
        mode = 'DEFENSIVE'
    elif dd > 5:
        mode = np.random.choice(['DEFENSIVE', 'NORMAL'])
    else:
        mode = np.random.choice(['NORMAL', 'AGGRESSIVE'])
    risk_mode.append(mode)
    # Reward boost
    boost = 1.0 + 0.5 * (mode == 'AGGRESSIVE')
    reward_boost.append(boost)
    # Penalty inaction
    penalty = -0.05 * (mode == 'DEFENSIVE')
    penalty_inaction.append(penalty)

# Correction des longueurs
risk_mode = ['NORMAL'] + risk_mode
reward_boost = [1.0] + reward_boost
penalty_inaction = [0.0] + penalty_inaction

# Générer les entrées JSONL
records = []
for i in range(N):
    records.append({
        "timestamp": timestamps[i].isoformat(),
        "capital": float(capital[i]),
        "drawdown": float(drawdown[i]),
        "sl_pct": float(sl_pct[i]),
        "tp_pct": float(tp_pct[i]),
        "risk_mode": risk_mode[i],
        "reward_boost": reward_boost[i],
        "penalty_inaction": penalty_inaction[i]
    })

# Sauvegarde
out_path = Path("logs/endurance_metrics.jsonl")
with open(out_path, "w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")

print(f"Fichier de test généré: {out_path} ({N} lignes)")
