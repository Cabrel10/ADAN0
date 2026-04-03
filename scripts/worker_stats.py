#!/usr/bin/env python3
"""Statistiques détaillées par worker : win rate, R/R, Sharpe, Kelly, raisonnement trades."""
import sys, copy
import numpy as np
sys.path.insert(0, "src")

from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from stable_baselines3.common.vec_env import DummyVecEnv

cfg = ConfigLoader.load_config("config/config.yaml")

PROFILES = {
    0: "Scalper  (5m  — haute freq)",
    1: "Intraday (1h  — moyen terme)",
    2: "Swing    (4h  — tendance)",
    3: "Position (4h  — long terme)",
}

SEP = "=" * 62

for wid in range(4):
    wkey = f"w{wid+1}"
    wc = copy.deepcopy(cfg["workers"][wkey])
    wc["worker_id"] = wid

    loader = ChunkedDataLoader(config=cfg, worker_config=wc, worker_id=wid)
    data = loader.load_chunk(0)

    def make(w=wc, d=data, i=wid):
        return MultiAssetChunkedEnv(
            data=d, config=cfg, worker_config=w, worker_id=i, live_mode=False
        )

    env = DummyVecEnv([make])
    obs = env.reset()

    rewards, balances, kelly_mods = [], [20.5], []
    wins, losses = 0, 0
    win_pnls, loss_pnls = [], []
    trade_log = []   # last 5 trades for reasoning display

    for step in range(800):
        action = env.action_space.sample()
        obs, reward, done, info = env.step([action])
        r = float(reward[0])
        rewards.append(r)

        inf = info[0] if info else {}

        # Balance
        pm = inf.get("portfolio", {})
        bal = (pm.get("total_value") or pm.get("portfolio_value")
               or pm.get("total_capital") or balances[-1])
        try:
            balances.append(float(bal))
        except Exception:
            balances.append(balances[-1])

        # Kelly
        km = inf.get("kelly_modifier")
        if km is not None:
            try:
                kelly_mods.append(float(km))
            except Exception:
                pass

        # Trade PnL
        trade_info = inf.get("trade") or inf.get("last_trade")
        if trade_info:
            pnl = float(trade_info.get("pnl", 0) or 0)
            side = trade_info.get("side", "?")
            tf   = trade_info.get("timeframe", "?")
            reason = trade_info.get("reason", "signal")
            if pnl > 0:
                wins += 1
                win_pnls.append(pnl)
            elif pnl < 0:
                losses += 1
                loss_pnls.append(abs(pnl))
            if len(trade_log) < 5:
                trade_log.append({
                    "step": step, "side": side, "tf": tf,
                    "pnl": pnl, "reason": reason,
                    "balance": balances[-1],
                })

        if done[0]:
            obs = env.reset()

    env.close()

    # ── Stats ──────────────────────────────────────────────────
    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    avg_win  = np.mean(win_pnls)  if win_pnls  else 0.0
    avg_loss = np.mean(loss_pnls) if loss_pnls else 0.0
    rr       = avg_win / avg_loss if avg_loss > 0 else 0.0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    r_arr  = np.array(rewards)
    sharpe = (r_arr.mean() / (r_arr.std() + 1e-8)) * np.sqrt(252 * 288)

    bal_arr = np.array(balances)
    peak    = np.maximum.accumulate(bal_arr)
    dd      = (peak - bal_arr) / (peak + 1e-8)
    max_dd  = float(dd.max() * 100)
    final_bal = balances[-1]
    pnl_net   = final_bal - 20.5
    pnl_pct   = pnl_net / 20.5 * 100

    avg_kelly = float(np.mean(kelly_mods)) if kelly_mods else None

    # HMM context
    ctx = obs["context_vector"][0]
    hmm = ctx[3:6]
    regime_labels = ["bull", "sideways", "bear"]
    dominant_idx  = int(np.argmax(hmm))
    dominant      = regime_labels[dominant_idx]

    # ── Display ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  WORKER {wid} — {PROFILES[wid]}")
    print(SEP)
    print(f"  Steps simulés      : 800  |  Trades : {total_trades}")
    print(f"  Wins / Losses      : {wins} / {losses}")
    print()
    print(f"  Win Rate           : {win_rate*100:6.1f}%")
    print(f"  Avg Win            : ${avg_win:8.5f}")
    print(f"  Avg Loss           : ${avg_loss:8.5f}")
    print(f"  Risk / Reward      : {rr:6.2f}x  (target >= 2.0)")
    print(f"  Expectancy/trade   : ${expectancy:+.5f}")
    print(f"  Sharpe (annualisé) : {sharpe:7.3f}  (>1.0 = bon)")
    print(f"  Max Drawdown       : {max_dd:6.2f}%  (limite 4%)")
    print(f"  Balance finale     : ${final_bal:.3f}  (départ $20.50)")
    print(f"  P&L net            : ${pnl_net:+.3f}  ({pnl_pct:+.1f}%)")
    if avg_kelly is not None:
        print(f"  Half-Kelly moyen   : {avg_kelly:.3f}  (1.0=plein, 0.1=min)")

    print()
    print(f"  Régime HMM actuel  :")
    for lbl, p in zip(regime_labels, hmm):
        bar = "█" * int(p * 30)
        marker = " <-- DOMINANT" if lbl == dominant else ""
        print(f"    {lbl:8s} {p:.3f}  {bar}{marker}")

    print()
    print(f"  Encodage temporel  :")
    print(f"    sin_hour={ctx[8]:+.3f}  cos_hour={ctx[9]:+.3f}")
    print(f"    sin_dow ={ctx[10]:+.3f}  cos_dow ={ctx[11]:+.3f}")

    print()
    if trade_log:
        print(f"  Derniers trades (raisonnement) :")
        for t in trade_log:
            outcome = "WIN " if t["pnl"] > 0 else ("LOSS" if t["pnl"] < 0 else "FLAT")
            print(f"    step={t['step']:4d} | {t['side']:5s} | TF={t['tf']:3s} | "
                  f"PnL=${t['pnl']:+.5f} | {outcome} | bal=${t['balance']:.3f} | {t['reason']}")
    else:
        print("  Aucun trade enregistré dans les 800 steps (actions aléatoires)")
        print("  -> Normal : le modèle non entraîné explore l'espace d'action")

print(f"\n{SEP}")
print("  NOTE: Stats sur actions ALÉATOIRES (modèle non chargé)")
print("  -> Reflète la mécanique de l'env, pas la politique apprise")
print("  -> Relancer avec PPO.load() pour les stats du modèle entraîné")
print(SEP)
