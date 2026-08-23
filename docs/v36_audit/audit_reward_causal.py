#!/usr/bin/env python3
"""
AUDIT CAUSAL DU REWARD V35 — sans relancer aucun entraînement.

Objectif (dérivé des deux analyses de conversation) :
  1. Quantifier la magnitude RÉELLE de chaque terme du reward (pas juste FA vs PnL).
  2. Sur les steps de CLÔTURE (future_contrib != 0), tester si future_contrib
     domine pnl_base et s'il est DÉCORRÉLÉ / ANTI-corrélé du PnL réel.
     -> "hindsight non apparié" : reward de clôture piloté par l'oracle MFE/MAE
        plutôt que par le résultat financier.
  3. Décomposer le reward par action (BUY/SELL/HOLD) et par signe de PnL réel.

Streaming ligne-à-ligne : ne charge JAMAIS 1.87 GB en mémoire.
"""
import sys, json, math
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/ubuntu/webapp/MORNINGSTAR/ADAN0/logs/rewards/worker_0_rewards_20260822_162820.jsonl"
MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0 = tout

# Termes RÉELLEMENT additionnés dans raw_reward (L.7400) et qui atteignent PPO.
# VÉRIFIÉ par réconciliation: raw == somme exacte de ces 12 termes (diff=0.0000).
# capacity_reward / pos_limit_penalty / duration_penalty / frequency_daily
# sont PUREMENT TÉLÉMÉTRIQUES (dict rc[]) et N'ATTEIGNENT PAS PPO -> exclus.
TERMS = [
    "pnl_reward", "behavior_penalty", "action_anchor_penalty", "closure_bonus",
    "drawdown_penalty", "symmetry_penalty", "action_entropy_penalty",
    "future_contrib", "latent_pnl", "saturation_penalty",
    "promotion_bonus", "demotion_penalty",
]
# Termes télémétriques suivis SÉPARÉMENT (ne vont pas à PPO) pour vérif.
TELEM_ONLY = ["capacity_reward", "pos_limit_penalty", "duration_penalty"]

class Stat:
    __slots__=("n","s","s2","amin","amax","abs_s")
    def __init__(self): self.n=0; self.s=0.0; self.s2=0.0; self.amin=1e18; self.amax=-1e18; self.abs_s=0.0
    def add(self,x):
        self.n+=1; self.s+=x; self.s2+=x*x; self.abs_s+=abs(x)
        if x<self.amin:self.amin=x
        if x>self.amax:self.amax=x
    def mean(self): return self.s/self.n if self.n else 0.0
    def std(self):
        if self.n<2:return 0.0
        v=self.s2/self.n-(self.s/self.n)**2
        return math.sqrt(v) if v>0 else 0.0
    def mabs(self): return self.abs_s/self.n if self.n else 0.0

term_stats = {t:Stat() for t in TERMS}

# Corrélation streaming (Welford-like) entre 2 séries sur steps de clôture.
class Corr:
    __slots__=("n","sx","sy","sxy","sx2","sy2")
    def __init__(self): self.n=0;self.sx=0.0;self.sy=0.0;self.sxy=0.0;self.sx2=0.0;self.sy2=0.0
    def add(self,x,y):
        self.n+=1;self.sx+=x;self.sy+=y;self.sxy+=x*y;self.sx2+=x*x;self.sy2+=y*y
    def r(self):
        if self.n<2:return float('nan')
        cov=self.sxy-self.sx*self.sy/self.n
        vx=self.sx2-self.sx*self.sx/self.n
        vy=self.sy2-self.sy*self.sy/self.n
        d=math.sqrt(vx*vy)
        return cov/d if d>1e-12 else float('nan')

corr_fc_pnl   = Corr()   # future_contrib vs realized_pnl (clôtures)
corr_fc_pnlr  = Corr()   # future_contrib vs pnl_reward term (clôtures)
corr_tot_pnl  = Corr()   # reward.total vs realized_pnl (clôtures)
corr_totpre_pnl = Corr() # raw_before_symlog vs realized_pnl (clôtures)

# Reward moyen par action & signe de PnL sur clôtures.
by_pnl_sign = {"win":Stat(),"loss":Stat(),"flat":Stat()}      # reward.total sur clôture
fc_by_pnl_sign = {"win":Stat(),"loss":Stat(),"flat":Stat()}   # future_contrib sur clôture
# Cas critique : trade PERDANT mais future_contrib POSITIF (l'oracle récompense une perte)
loss_but_fc_pos = 0
win_but_fc_neg  = 0
n_closed = 0
n_total  = 0

with open(PATH, "r") as f:
    for line in f:
        n_total += 1
        if MAX and n_total>MAX: break
        try: d=json.loads(line)
        except: continue
        bd = d.get("reward",{}).get("breakdown",{})
        for t in TERMS:
            v=bd.get(t,0.0)
            if isinstance(v,(int,float)): term_stats[t].add(float(v))
        realized = d.get("pnl",{}).get("realized",0.0) or 0.0
        fc = bd.get("future_contrib",0.0) or 0.0
        if fc != 0.0:  # step de clôture
            n_closed += 1
            tot = d.get("reward",{}).get("total",0.0) or 0.0
            pre = d.get("reward",{}).get("raw_before_symlog",0.0) or 0.0
            pnlr = bd.get("pnl_reward",0.0) or 0.0
            corr_fc_pnl.add(fc, realized)
            corr_fc_pnlr.add(fc, pnlr)
            corr_tot_pnl.add(tot, realized)
            corr_totpre_pnl.add(pre, realized)
            sign = "win" if realized>0 else ("loss" if realized<0 else "flat")
            by_pnl_sign[sign].add(tot)
            fc_by_pnl_sign[sign].add(fc)
            if realized<0 and fc>0: loss_but_fc_pos += 1
            if realized>0 and fc<0: win_but_fc_neg  += 1

print("="*72)
print(f"AUDIT CAUSAL REWARD V35  |  lignes lues={n_total:,}  clôtures={n_closed:,}")
print("="*72)
print("\n[1] MAGNITUDE MOYENNE ABSOLUE DE CHAQUE TERME (tri décroissant)")
print(f"{'terme':<24}{'mean_abs':>12}{'mean':>12}{'std':>10}{'min':>10}{'max':>10}")
for t,st in sorted(term_stats.items(), key=lambda kv:-kv[1].mabs()):
    if st.n==0: continue
    print(f"{t:<24}{st.mabs():>12.5f}{st.mean():>12.5f}{st.std():>10.4f}{st.amin:>10.3f}{st.amax:>10.3f}")

tot_mabs = sum(st.mabs() for st in term_stats.values())
print(f"\n  Somme des mean_abs = {tot_mabs:.5f}")
print("  Part de chaque terme dans l'amplitude totale du raw_reward :")
for t,st in sorted(term_stats.items(), key=lambda kv:-kv[1].mabs()):
    if st.n==0 or tot_mabs<=0: continue
    share=st.mabs()/tot_mabs
    if share>0.005:
        print(f"    {t:<24}{share:>7.1%}")

print("\n[2] CORRÉLATIONS SUR STEPS DE CLÔTURE (le test décisif)")
print(f"  corr(future_contrib , realized_pnl)   = {corr_fc_pnl.r():+.4f}")
print(f"  corr(future_contrib , pnl_reward term)= {corr_fc_pnlr.r():+.4f}")
print(f"  corr(reward.total   , realized_pnl)   = {corr_tot_pnl.r():+.4f}")
print(f"  corr(raw_pre_symlog , realized_pnl)   = {corr_totpre_pnl.r():+.4f}")

print("\n[3] REWARD TOTAL MOYEN PAR SIGNE DU PnL RÉEL (clôtures)")
for k in ("win","loss","flat"):
    s=by_pnl_sign[k]; fcs=fc_by_pnl_sign[k]
    print(f"  {k:<5} n={s.n:>6}  reward.total_mean={s.mean():+.5f}  future_contrib_mean={fcs.mean():+.5f}")

print("\n[4] INCOHÉRENCES ORACLE (hindsight non apparié)")
print(f"  trades PERDANTS mais future_contrib > 0 : {loss_but_fc_pos:,}")
print(f"  trades GAGNANTS mais future_contrib < 0 : {win_but_fc_neg:,}")
if n_closed:
    print(f"  => {100*loss_but_fc_pos/n_closed:.1f}% des clôtures = perte récompensée par l'oracle")
print("="*72)
