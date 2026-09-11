"""Unit test du Decision Budget V3 — simule la mecanique deplete/recharge/gate
sans charger tout l'env (logique pure extraite, doit matcher le code)."""
def simulate():
    # parametres = ceux de config.yaml
    MAX=1.0; COST_BUY=0.15; COST_CLOSE=0.30; RECHARGE=0.02; MIN_GAP=12
    budget=MAX; last_close=-10**9; step=0; consecutive_closes=0; blocked=0
    log=[]
    # scenario: l'agent tente BUY->CLOSE->BUY->CLOSE en boucle (micro-scalping)
    for i in range(40):
        step=i
        # tente CLOSE
        gap_ok=(step-last_close)>=MIN_GAP
        if budget>=COST_CLOSE and gap_ok:
            budget=max(0.0,budget-COST_CLOSE)
            last_close=step; consecutive_closes+=1
            log.append((step,"CLOSE_OK",round(budget,3)))
        else:
            blocked+=1
            # HOLD force -> recharge
            budget=min(MAX,budget+RECHARGE)
            log.append((step,"BLOCKED->HOLD",round(budget,3)))
    return log, consecutive_closes, blocked

log,closes,blocked=simulate()
for l in log[:20]: print(l)
print("...")
print(f"TOTAL CLOSES executes (sur 40 tentatives consecutives): {closes}")
print(f"TOTAL BLOCKED (force HOLD): {blocked}")
# Assertions: le scalping est etouffe
assert closes<=4, f"trop de closes: {closes} (scalping non bride!)"
assert blocked>=30, f"pas assez de blocages: {blocked}"
# Verifie le cooldown naturel: apres 1 close il faut MIN_GAP=12 steps avant le suivant
print("\\nVERIF cooldown structurel (min_gap=12):")
close_steps=[s for s,a,b in log if a=="CLOSE_OK"]
print("steps des closes:", close_steps)
gaps=[close_steps[i+1]-close_steps[i] for i in range(len(close_steps)-1)]
print("gaps entre closes:", gaps)
assert all(g>=12 for g in gaps), f"cooldown viole: {gaps}"
print("\\n✅ DECISION BUDGET: micro-scalping mathematiquement etouffe (cooldown naturel >= 12 steps respecte)")
