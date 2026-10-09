"""GATE 7: plan-conditioned System One judgment head.

Z = GroupedRelationalPerception(state(t), plan(t), portfolio(t)).
Query-Slot questions are now about the EXPLICIT candidate plan:
  CHOICE  outcome   : {TP_FIRST, SL_FIRST, TIMEOUT}  -> P(TP before SL) directly
  NOUL    y_win     : P(NET_RETURN > 0 after costs)  (≠ TP_FIRST)
  SCORE   net_return: Huber regression of NET_RETURN in R (clipped to [-3, tp_r])
The legacy LONG/SHORT direction classifier and defective legacy labels are
absent by design. Temperatures are fitted on VAL only, bounded [0.05, 10].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

OUTCOMES = ('TP_FIRST', 'SL_FIRST', 'TIMEOUT')
RETURN_CLIP = (-3.0, 6.0)


class PlanJudgmentCore(nn.Module):
    def __init__(self, dim_z=128, dim_slot=32, hidden=128):
        super().__init__()
        from adan_trading_bot.policy.market_contract import load_market_contract
        self.market_contract_sha256 = load_market_contract().sha256()
        self.slot_outcome = nn.Parameter(torch.randn(dim_slot) * 0.02)
        self.slot_win = nn.Parameter(torch.randn(dim_slot) * 0.02)
        self.slot_return = nn.Parameter(torch.randn(dim_slot) * 0.02)
        self.adapter = nn.Linear(dim_z + dim_slot, hidden)
        self.outcome = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, len(OUTCOMES)))
        self.win = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.ret = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.register_buffer('temp_outcome', torch.ones(()))
        self.register_buffer('temp_win', torch.ones(()))
        self.register_buffer('calibrated', torch.zeros((), dtype=torch.bool))

    def _fuse(self, z, slot):
        return self.adapter(torch.cat([z, slot.expand(z.shape[0], -1)], dim=-1))

    def forward(self, z):
        return {'outcome': self.outcome(self._fuse(z, self.slot_outcome)),
                'win': self.win(self._fuse(z, self.slot_win)).squeeze(-1),
                'net_return': self.ret(self._fuse(z, self.slot_return)).squeeze(-1)}

    def calibrated_probabilities(self, logits):
        p_outcome = F.softmax(logits['outcome'] / self.temp_outcome, dim=-1)
        p_win = torch.sigmoid(logits['win'] / self.temp_win)
        return {'p_tp_first': p_outcome[:, 0], 'p_sl_first': p_outcome[:, 1], 'p_timeout': p_outcome[:, 2],
                'p_win': p_win, 'expected_net_return_r': logits['net_return']}


def plan_losses(logits, targets, outcome_weight=1.0, win_weight=1.0, return_weight=0.25):
    """Proper scoring rules: CE + multiclass Brier, BCE + Brier, Huber on R."""
    p = F.softmax(logits['outcome'], dim=-1)
    onehot = F.one_hot(targets['outcome'], len(OUTCOMES)).float()
    ce = F.cross_entropy(logits['outcome'], targets['outcome'])
    brier_outcome = ((p - onehot) ** 2).sum(-1).mean()
    pw = torch.sigmoid(logits['win'])
    bce = F.binary_cross_entropy_with_logits(logits['win'], targets['win'])
    brier_win = ((pw - targets['win']) ** 2).mean()
    huber = F.smooth_l1_loss(logits['net_return'], targets['net_return'].clamp(*RETURN_CLIP))
    total = outcome_weight * (ce + brier_outcome) + win_weight * (bce + brier_win) + return_weight * huber
    return total, {'ce_outcome': ce.item(), 'brier_outcome': brier_outcome.item(), 'bce_win': bce.item(),
                   'brier_win': brier_win.item(), 'huber_return': huber.item()}
