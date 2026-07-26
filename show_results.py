import json
import glob
import os

# Liste des fichiers JSON
files = glob.glob('logs/validation/confidence_scan/bt_*.json')

# Filtrer ceux dont le nom après "bt_" est numérique (ignore "test10k")
valid_files = []
for f in files:
    basename = os.path.basename(f)
    ckpt_str = basename.replace('bt_', '').replace('.json', '')
    if ckpt_str.isdigit():
        valid_files.append((int(ckpt_str), f))
valid_files.sort()  # tri par checkpoint

# En-tête
print(f"{'Checkpoint':<10} | {'Trades':<6} | {'WR':<6} | {'PF':<5} | {'Expectancy':<10} | {'Return':<7} | {'Best Trd':<9} | {'Worst Trd':<9}")
print('-' * 83)

for ckpt, fpath in valid_files:
    with open(fpath) as f:
        data = json.load(f)
    
    # Métriques de base
    trades = data.get('env_total_trades', 0)
    win_rate = data.get('win_rate', 0)  # entre 0 et 1
    wr = f"{win_rate*100:.1f}%"
    pf = f"{data.get('profit_factor', 0):.2f}"
    pnl = data.get('env_total_realized_pnl', 0.0)
    exp = f"{pnl/trades:.3f}%" if trades > 0 else "0.000%"
    ret = f"{data.get('total_return_pct', 0):.2f}%"
    
    # Best / Worst : essayer de les lire directement, sinon les calculer depuis les trades
    best = data.get('best_trade_pct')
    worst = data.get('worst_trade_pct')
    
    if best is None or worst is None:
        # Si la liste des trades est disponible, on calcule
        trades_list = data.get('trades', [])
        if trades_list:
            pcts = [t.get('pnl_pct', 0) for t in trades_list if 'pnl_pct' in t]
            if pcts:
                best = max(pcts)
                worst = min(pcts)
            else:
                best = worst = 0.0
        else:
            best = worst = 0.0
    
    best_str = f"{best:+.3f}%" if best is not None else "N/A"
    worst_str = f"{worst:+.3f}%" if worst is not None else "N/A"
    
    print(f"{ckpt:<10} | {trades:<6} | {wr:<6} | {pf:<5} | {exp:<10} | {ret:<7} | {best_str:<9} | {worst_str:<9}")
