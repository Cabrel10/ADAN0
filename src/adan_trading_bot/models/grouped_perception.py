"""GATE 5 grouped relational perception — replaces any flat registry tensor.

Contract:
  * Inputs are ONLY names admitted by FeatureAvailabilityContract (currently 283).
    Every admitted name must be assigned to exactly one semantic group; unknown,
    unassigned or missing names raise. No zero-filling, no all-1026 tensor.
  * Groups:  bar_5m (8 scalars) | seq_5m (35 lags x 7 channels, temporal CNN)
             | c1h container (16) | c4h container (14).  Plan and portfolio are
             separate conditioning inputs (FiLM), never mixed into market groups.
  * Price-level scalars are made scale-free causally per state: prices divided
    by the state's own close_t (log-ratio), volumes by the state's own running
    1h volume; this uses only information available at t.
  * Within-group sparse message passing uses ONLY verified graph edges whose
    endpoints both lie in that group (VERIFIED_DETERMINISTIC/TEMPORAL). Predictive
    or unverified edges are never hard structure.
  * Cross-group attention over group tokens, then FiLM by plan+portfolio -> Z.
No training is performed or authorized by this module.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

SEQ_FIELDS = ('open', 'high', 'low', 'close', 'volume', 'wick_up', 'wick_down')
PRICE_LIKE = {'open', 'high', 'low', 'close', 'running_high', 'running_low',
              'prev_open', 'prev_high', 'prev_low', 'prev_close', 'atr_1h'}
SPREAD_LIKE = {'wick_up', 'wick_down', 'body'}
VOLUME_LIKE = {'volume', 'running_vol', 'prev_volume'}
PLAN_FIELDS = ('direction_long', 'sl_pct', 'tp_r', 'horizon_frac', 'sl_rel_min', 'sl_rel_max')
PORTFOLIO_FIELDS = ('allocation_fraction', 'risk_budget_frac', 'notional_frac', 'positions_open')


class PerceptionContractError(ValueError):
    pass


@dataclass(frozen=True)
class GroupLayout:
    bar: tuple
    seq: tuple          # (lag, field) -> name, lags ascending 1..L
    seq_lags: int
    c1h: tuple
    c4h: tuple

    @property
    def names(self):
        return self.bar + tuple(n for n in self.seq) + self.c1h + self.c4h


def build_layout(admitted_names, availability_contract):
    """Assign every admitted name to exactly one group; fail on anything else.

    Each name is re-checked against the availability contract: a group prefix
    (e.g. c4h.atr_4h, UNKNOWN) never grants admission by itself.
    """
    from adan_trading_bot.features.feature_availability_contract import FeatureAvailabilityError
    names = list(admitted_names)
    if len(names) != len(set(names)):
        raise PerceptionContractError('Duplicate admitted names')
    for n in names:
        try:
            availability_contract.require(n)
        except FeatureAvailabilityError as error:
            raise PerceptionContractError(f'Not admitted by availability contract: {n}') from error
    bar, c1h, c4h, seq = [], [], [], {}
    for n in names:
        m = re.fullmatch(r'seq_5m\.lag_(\d+)\.(\w+)', n)
        if m and m[2] in SEQ_FIELDS:
            seq[(int(m[1]), m[2])] = n
        elif n.startswith('bar_5m.'):
            bar.append(n)
        elif n.startswith('c1h.'):
            c1h.append(n)
        elif n.startswith('c4h.'):
            c4h.append(n)
        else:
            raise PerceptionContractError(f'Admitted name without semantic group: {n}')
    lags = sorted({k[0] for k in seq})
    if not lags or lags != list(range(1, lags[-1] + 1)):
        raise PerceptionContractError('Sequence lags must be complete 1..L')
    missing = [(l, f) for l in lags for f in SEQ_FIELDS if (l, f) not in seq]
    if missing:
        raise PerceptionContractError(f'Incomplete sequence channels: {missing[:3]}')
    if 'bar_5m.close' not in bar or 'c1h.running_vol' not in c1h:
        raise PerceptionContractError('Normalization anchors close_t / running_vol_1h must be admitted')
    ordered_seq = tuple(seq[(l, f)] for l in lags for f in SEQ_FIELDS)
    layout = GroupLayout(tuple(bar), ordered_seq, len(lags), tuple(c1h), tuple(c4h))
    if sorted(layout.names) != sorted(names):
        raise PerceptionContractError('Group partition is not exact')
    return layout


def _normalize(name, value, close, vol_ref):
    leaf = name.rsplit('.', 1)[-1]
    if leaf in PRICE_LIKE and leaf != 'atr_1h':
        if value <= 0:
            raise PerceptionContractError(f'Nonpositive price {name}')
        return float(np.log(value / close))
    if leaf == 'atr_1h' or leaf in SPREAD_LIKE:
        return float(value / close)
    if leaf in VOLUME_LIKE:
        return float(np.log1p(value / vol_ref))
    return float(value)   # already scale-free: phase, position, sweep flags, bar_index, atr_1h_pct


def encode_state(layout, values):
    """Named dict -> grouped tensors. Missing or nonfinite -> explicit error."""
    missing = [n for n in layout.names if n not in values]
    if missing:
        raise PerceptionContractError(f'Missing admitted values: {missing[:3]}')
    extra = set(values) - set(layout.names)
    if extra:
        raise PerceptionContractError(f'Unexpected unadmitted inputs: {sorted(extra)[:3]}')
    raw = np.array([values[n] for n in layout.names], dtype=float)
    if not np.isfinite(raw).all():
        raise PerceptionContractError('Nonfinite admitted value')
    close = float(values['bar_5m.close'])
    vol_ref = max(float(values['c1h.running_vol']), 1e-12)
    norm = lambda group: np.array([_normalize(n, values[n], close, vol_ref) for n in group], dtype=np.float32)
    seq = norm(layout.seq).reshape(layout.seq_lags, len(SEQ_FIELDS))[::-1].copy()  # oldest -> newest
    return {'bar': norm(layout.bar), 'seq': seq, 'c1h': norm(layout.c1h), 'c4h': norm(layout.c4h)}


def encode_plan(plan_row, portfolio_context, market_contract=None):
    """Explicit plan + portfolio conditioning (no outcome columns accepted)."""
    forbidden = {'Y_WIN', 'Y_TP_FIRST', 'Y_SL_FIRST', 'MFE', 'MAE', 'NET_RETURN', 'TIMEOUT',
                 'TIME_TO_TP', 'TIME_TO_SL', 'exit_price'}
    if forbidden & set(plan_row):
        raise PerceptionContractError('Outcome columns must never enter the plan encoder')
    from adan_trading_bot.policy.market_contract import load_market_contract, MarketContractError
    market = market_contract or load_market_contract()
    try:
        market.require_direction(plan_row['direction'])
    except MarketContractError as error:
        raise PerceptionContractError(str(error)) from error
    sl, lo, hi = float(plan_row['sl_pct']), float(plan_row['sl_min_bound']), float(plan_row['sl_max_bound'])
    if not 0 < lo <= sl <= hi or sl < market.min_sl_for_costs - 1e-12:
        raise PerceptionContractError('Plan outside its own ATR bounds')
    plan = np.array([1.0 if plan_row['direction'] == 'LONG' else 0.0, sl, float(plan_row['tp_r']),
                     float(plan_row['horizon']) / 288.0, sl / lo - 1.0, hi / sl - 1.0], dtype=np.float32)
    equity = float(portfolio_context['equity_usd'])
    port = np.array([float(portfolio_context['allocation_fraction']),
                     float(portfolio_context['risk_budget_usd']) / equity,
                     float(portfolio_context['notional_usd']) / equity,
                     float(portfolio_context['positions_open'])], dtype=np.float32)
    if not (np.isfinite(plan).all() and np.isfinite(port).all()):
        raise PerceptionContractError('Nonfinite plan/portfolio conditioning')
    return plan, port


def group_edge_index(graph, availability_contract, group_names):
    """Verified edges restricted to one group, re-indexed locally."""
    local = {n: i for i, n in enumerate(group_names)}
    pairs = [(local[e.source], local[e.target]) for e in graph.constraint_edges(availability_contract)
             if e.source in local and e.target in local]
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


class GroupScalarEncoder(nn.Module):
    """Per-variable embedding + identity embedding + sparse verified message passing."""

    def __init__(self, n_vars, edge_index, dim):
        super().__init__()
        self.value = nn.Linear(1, dim)
        self.identity = nn.Embedding(n_vars, dim)
        self.self_lin = nn.Linear(dim, dim)
        self.neigh_lin = nn.Linear(dim, dim, bias=False)
        self.register_buffer('edge_index', edge_index)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.GELU())

    def forward(self, x):                       # x: (B, n_vars)
        h = self.value(x.unsqueeze(-1)) + self.identity.weight.unsqueeze(0)
        msg = torch.zeros_like(h)
        if self.edge_index.shape[1]:
            src, dst = self.edge_index
            msg.index_add_(1, dst, self.neigh_lin(h[:, src]))
        h = self.out(self.self_lin(h) + msg)
        return h.mean(dim=1)                    # (B, dim) group token


class TemporalEncoder(nn.Module):
    """Causal 1D CNN over the 5m sequence (oldest -> newest)."""

    def __init__(self, channels, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, dim, 3, padding=2), nn.GELU(),
            nn.Conv1d(dim, dim, 3, padding=2, dilation=2), nn.GELU())
        self.trim = (2, 4)

    def forward(self, seq):                     # (B, L, C)
        x = seq.transpose(1, 2)
        x = self.conv[1](self.conv[0](x)[..., :-self.trim[0]])
        x = self.conv[3](self.conv[2](x)[..., :-self.trim[1]])
        return x[..., -1]                       # last causal step


class GroupedRelationalPerception(nn.Module):
    def __init__(self, layout, graph, availability_contract, dim=64, z_dim=128, heads=4):
        super().__init__()
        self.layout = layout
        self.bar = GroupScalarEncoder(len(layout.bar), group_edge_index(graph, availability_contract, layout.bar), dim)
        self.c1h = GroupScalarEncoder(len(layout.c1h), group_edge_index(graph, availability_contract, layout.c1h), dim)
        self.c4h = GroupScalarEncoder(len(layout.c4h), group_edge_index(graph, availability_contract, layout.c4h), dim)
        self.seq = TemporalEncoder(len(SEQ_FIELDS), dim)
        self.group_id = nn.Embedding(4, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.film = nn.Sequential(nn.Linear(len(PLAN_FIELDS) + len(PORTFOLIO_FIELDS), 2 * dim), nn.GELU(),
                                  nn.Linear(2 * dim, 2 * dim))
        self.head = nn.Sequential(nn.Linear(dim, z_dim), nn.LayerNorm(z_dim), nn.GELU())

    def forward(self, bar, seq, c1h, c4h, plan, portfolio):
        tokens = torch.stack([self.bar(bar), self.seq(seq), self.c1h(c1h), self.c4h(c4h)], dim=1)
        tokens = tokens + self.group_id.weight.unsqueeze(0)
        attended, _ = self.attn(tokens, tokens, tokens)
        pooled = self.norm(tokens + attended).mean(dim=1)
        gamma, beta = self.film(torch.cat([plan, portfolio], dim=-1)).chunk(2, dim=-1)
        return self.head((1 + gamma) * pooled + beta)
