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
queried only by the explicit read-only CLI. Without a fresh account evidence
snapshot, configured costs are diagnostic ONLY and production/500K is blocked.
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


def load_market_contract(config_path='config/config.yaml', fee_evidence_path='config/spot_account_fees.json'):
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
    kwargs = {}
    evidence_path = Path(fee_evidence_path)
    if evidence_path.exists():
        from datetime import datetime, timezone
        evidence_raw = evidence_path.read_bytes()
        evidence = json.loads(evidence_raw)
        if (evidence.get('source'), evidence.get('market'), evidence.get('symbol'), evidence.get('discount_applied')) != (
                'binance:/api/v3/account/commission', 'SPOT', 'BTCUSDT', False):
            raise MarketContractError('Unsupported account fee evidence; never infer discounts')
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(evidence['retrieved_at'])).total_seconds()
        if not 0 <= age <= 86400:
            raise MarketContractError('Account fee evidence expired or future-dated (24h freshness limit)')
        commission = float(evidence['buy_taker_per_side'])
        kwargs = {'commission_exit_per_side': float(evidence['sell_taker_per_side']),
                  'fee_verified': True, 'fee_source': evidence['source'] + ':undiscounted_taker',
                  'fee_evidence_sha256': hashlib.sha256(evidence_raw).hexdigest()}
    return MarketContract('SPOT', ('LONG',), ('BUY', 'SELL_EXIT', 'HOLD'), commission, slippage, leverage,
                          hashlib.sha256(raw).hexdigest(), **kwargs)


def fetch_account_fee_evidence(output):
    """Signed READ-ONLY account commission request; never place orders or print secrets.

    Conservative market-entry/stop/exit assumption: undiscounted taker + buyer/seller,
    including tax and special commission. BNB discount is NOT assumed, even if enabled.
    This is not a claim of exact future realized fill costs.
    """
    import os
    import hmac
    import time
    import urllib.parse
    import urllib.request
    from datetime import datetime, timezone
    target = Path(output).resolve()
    if not target.is_relative_to(Path('/home/ubuntu/webapp')) or not target.parent.is_dir() or target.exists():
        raise MarketContractError('Evidence output must be new and inside workspace with an existing parent')
    key, secret = os.getenv('ADAN_API_KEY'), os.getenv('ADAN_API_SECRET')
    if not key or not secret:
        raise MarketContractError('NO-GO: ADAN_API_KEY/ADAN_API_SECRET absent; account fee tier unavailable')
    query = urllib.parse.urlencode({'symbol': 'BTCUSDT', 'timestamp': int(time.time()*1000), 'recvWindow': 5000})
    signature = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    request = urllib.request.Request('https://api.binance.com/api/v3/account/commission?' + query + '&signature=' + signature,
                                     headers={'X-MBX-APIKEY': key})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data = json.load(response)
    except Exception as error:
        # Do not expose signed URL, credentials or response content in tracebacks.
        raise MarketContractError('Account commission read failed (' + type(error).__name__ + ')') from None
    if data.get('symbol') != 'BTCUSDT':
        raise MarketContractError('Unexpected symbol in account fee response')
    components = {name: {k: float(data[name][k]) for k in ('maker', 'taker', 'buyer', 'seller')}
                  for name in ('standardCommission', 'taxCommission', 'specialCommission')}
    if not all(math.isfinite(v) and v >= 0 for group in components.values() for v in group.values()):
        raise MarketContractError('Invalid account commission response')
    evidence = {'source': 'binance:/api/v3/account/commission', 'market': 'SPOT', 'symbol': 'BTCUSDT',
                'retrieved_at': datetime.now(timezone.utc).isoformat(), 'discount_applied': False,
                'components': components,
                'buy_taker_per_side': sum(g['taker'] + g['buyer'] for g in components.values()),
                'sell_taker_per_side': sum(g['taker'] + g['seller'] for g in components.values())}
    target.write_text(json.dumps(evidence, indent=2) + '\n')
    print('Read-only account fee evidence saved; no orders submitted.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch-account-fees', type=Path, required=True)
    args = parser.parse_args()
    try:
        fetch_account_fee_evidence(args.fetch_account_fees)
    except MarketContractError as error:
        raise SystemExit(str(error)) from None
