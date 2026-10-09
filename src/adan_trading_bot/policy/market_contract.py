"""Single central market/action-space and cost contract (SPOT by default).

Derived ONLY from config/config.yaml trading_rules:
  futures_enabled == False and leverage == 1  ->  market = SPOT STRICT
    entry directions = ('LONG',)            : BUY base asset with quote cash
    actions          = BUY / SELL_EXIT / HOLD : SELL only closes an open long
    SHORT, borrowing, margin                 : INVALID
  A LONG/SHORT contract would require an explicit futures/margin configuration
  that exists in the portfolio/execution path; it is NOT inferred here.
Costs: commission_pct and slippage_pct are PER SIDE (action_routing per_side);
round-trip cost = 2 * (commission + slippage). The live account fee tier is
NOT queried (no API credentials are used); configured values are authoritative
and recorded with the config SHA256.
Every consumer (candidate factory, plan outcomes, dataset, oracle, risk,
trainer, tests) must call `load_market_contract()` and `require_direction`.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

FEES_R_MAX = 0.30


class MarketContractError(ValueError):
    pass


@dataclass(frozen=True)
class MarketContract:
    market: str
    entry_directions: tuple
    actions: tuple
    commission_per_side: float
    slippage_per_side: float
    leverage: float
    config_sha256: str
    fees_r_max: float = FEES_R_MAX
    fee_source: str = 'config/config.yaml:trading_rules (live account tier NOT queried)'
    fee_verified: bool = False
    fee_evidence_sha256: str = ''
    commission_exit_per_side: float | None = None

    def __post_init__(self):
        if (self.market, self.entry_directions, self.actions, self.leverage) != (
                'SPOT', ('LONG',), ('BUY', 'SELL_EXIT', 'HOLD'), 1.0):
            raise MarketContractError('Only validated SPOT strict execution is implemented')
        rates = (self.commission_per_side, self.slippage_per_side,
                 self.commission_exit_per_side if self.commission_exit_per_side is not None else self.commission_per_side)
        if not all(math.isfinite(x) and 0 <= x < 0.05 for x in rates):
            raise MarketContractError('Invalid spot costs')
        if not math.isfinite(self.fees_r_max) or self.fees_r_max <= 0:
            raise MarketContractError('Invalid FEES_R_MAX')

    @property
    def cost_rt(self):
        exit_fee = self.commission_per_side if self.commission_exit_per_side is None else self.commission_exit_per_side
        return self.commission_per_side + exit_fee + 2.0 * self.slippage_per_side

    def require_verified_fees(self):
        if not self.fee_verified or not self.fee_evidence_sha256:
            raise MarketContractError('NO-GO: account spot fee tier has not been verified; config costs are diagnostic only')

    def action_space_sha256(self):
        payload = {'market': self.market, 'entry_directions': self.entry_directions,
                   'actions': self.actions, 'leverage': self.leverage}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def validate_dataset(self, plans, manifest=None):
        # Scan ALL rows, not just the sampled oracle/trainer subset. Fail on SHORT first.
        if plans.empty or not set(plans.direction).issubset(self.entry_directions):
            raise MarketContractError('Inadmissible direction in SPOT dataset (SHORT is forbidden)')
        if not all(math.isfinite(float(x)) and math.isclose(float(x), self.cost_rt, rel_tol=0, abs_tol=1e-15)
                   for x in plans.fees_rt):
            raise MarketContractError('Dataset cost differs from central market contract')
        if manifest is not None:
            if manifest.get('market_contract_sha256') != self.sha256():
                raise MarketContractError('Dataset market/cost contract hash mismatch')
            if manifest.get('action_space_contract_sha256') != self.action_space_sha256():
                raise MarketContractError('Dataset action-space contract hash mismatch')

    @property
    def min_sl_for_costs(self):
        return self.cost_rt / self.fees_r_max

    def require_direction(self, direction):
        if direction not in self.entry_directions:
            raise MarketContractError(f'{direction} is not an admissible entry under {self.market} contract')
        return direction

    def sha256(self):
        payload = json.dumps({**asdict(self), 'cost_rt': self.cost_rt}, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


def load_market_contract(config_path='config/config.yaml'):
    import yaml
    raw = Path(config_path).read_bytes()
    rules = yaml.safe_load(raw)['trading_rules']
    futures = rules.get('futures_enabled')
    leverage = float(rules.get('leverage', 1))
    commission = float(rules['commission_pct'])
    slippage = float(rules.get('slippage_pct', 0.0))
    if not isinstance(futures, bool):
        raise MarketContractError('trading_rules.futures_enabled must be an explicit boolean')
    if not all(math.isfinite(x) and x >= 0 for x in (commission, slippage)) or commission >= 0.05:
        raise MarketContractError('Invalid per-side cost configuration')
    if futures:
        raise MarketContractError('Futures/short contract requested but no validated margin/short execution path exists')
    if leverage != 1.0:
        raise MarketContractError('Spot strict requires leverage == 1')
    return MarketContract('SPOT', ('LONG',), ('BUY', 'SELL_EXIT', 'HOLD'), commission, slippage, leverage,
                          hashlib.sha256(raw).hexdigest())
