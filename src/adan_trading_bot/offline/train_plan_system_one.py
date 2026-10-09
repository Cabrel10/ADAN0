"""GATE 8 trainer: GroupedRelationalPerception + PlanJudgmentCore on State×Plan.

Uses ONLY the versioned relational datasets (data/plan_dataset/{train,val}_v1)
with manifests; TEST is never loaded. Features stay grouped (never flat registry).
Per-group standardization is fitted on TRAIN states only and stored in the
checkpoint. Validation is TIME-ordered VAL. Checkpoints every --ckpt-every steps,
best-VAL checkpoint, full resume (model, optimizer, scheduler, RNG, step, config,
manifests hashes, git commit). Baselines reported: TRAIN-prior constant predictor.

Device policy: CPU runs are capped at --max-cpu-steps (default 10_000).
500K requires CUDA (and passing 1K/5K/10K GPU benchmarks recorded beforehand).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from adan_trading_bot.features.feature_registry import get_feature_registry
from adan_trading_bot.features.feature_availability_contract import FeatureAvailabilityContract
from adan_trading_bot.features.relation_graph import RelationGraph
from adan_trading_bot.models.grouped_perception import build_layout, encode_state, GroupedRelationalPerception, SEQ_FIELDS
from adan_trading_bot.models.plan_judgment_core import PlanJudgmentCore, plan_losses, OUTCOMES
from adan_trading_bot.offline.labeler_mfe_mae import PARQUET
from adan_trading_bot.policy.market_contract import load_market_contract

GROUPS = ('bar', 'seq', 'c1h', 'c4h')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def load_split(directory, layout):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['split'] not in ('train', 'val') or manifest.get('test_touched'):
        raise ValueError('Only TRAIN/VAL manifests are allowed')
    for name in ('states', 'plans'):
        if sha(directory / f'{name}.parquet') != manifest[f'{name}_sha256']:
            raise ValueError(f'{name}.parquet hash mismatch vs manifest')
    states = pd.read_parquet(directory / 'states.parquet')
    plans = pd.read_parquet(directory / 'plans.parquet')
    market = load_market_contract()
    market.validate_dataset(plans, manifest)
    if not states.state_id.is_unique or not plans.plan_id.is_unique:
        raise ValueError('Duplicate state/plan ID; never silently deduplicate')
    if not set(plans.state_id).issubset(set(states.state_id)):
        raise ValueError('Orphan plan')
    outcomes = {'Y_WIN','Y_TP_FIRST','Y_SL_FIRST','TIMEOUT','NET_RETURN','MFE','MAE','TIME_TO_TP','TIME_TO_SL'}
    if outcomes & set(states.columns):
        raise ValueError('Outcome leaked into state inputs')
    if not ((plans.Y_TP_FIRST + plans.Y_SL_FIRST + plans.TIMEOUT) == 1).all():
        raise ValueError('Invalid outcome partition')
    if not (plans.Y_WIN == (plans.NET_RETURN > 0).astype(int)).all():
        raise ValueError('Y_WIN is not NET_RETURN > 0')
    if not np.isfinite(states[list(layout.names)].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite state')
    if manifest['registry_sha256'] != sha('config/feature_registry.json'):
        raise ValueError('Registry hash mismatch')
    if sorted(manifest['feature_names']) != sorted(layout.names):
        raise ValueError('Dataset feature names differ from contract-admitted layout')
    encoded = {g: [] for g in GROUPS}
    for _, row in states.iterrows():
        e = encode_state(layout, {n: float(row[n]) for n in layout.names})
        for g in GROUPS: encoded[g].append(e[g])
    tensors = {g: torch.tensor(np.stack(encoded[g])) for g in GROUPS}
    index = {s: i for i, s in enumerate(states.state_id)}
    ctx = [json.loads(c) for c in plans.portfolio_context_json]
    plan = np.stack([(plans.direction == 'LONG').astype(float), plans.sl_pct, plans.tp_r, plans.horizon / 288.0,
                     plans.sl_pct / plans.sl_min_bound - 1, plans.sl_max_bound / plans.sl_pct - 1], 1).astype(np.float32)
    port = np.array([[c['allocation_fraction'], c['risk_budget_usd'] / c['equity_usd'],
                      c['notional_usd'] / c['equity_usd'], c['positions_open']] for c in ctx], dtype=np.float32)
    outcome = np.where(plans.Y_TP_FIRST == 1, 0, np.where(plans.Y_SL_FIRST == 1, 1, 2))
    data = {'state_idx': torch.tensor(plans.state_id.map(index).to_numpy(), dtype=torch.long),
            'plan': torch.tensor(plan), 'portfolio': torch.tensor(port),
            'outcome': torch.tensor(outcome, dtype=torch.long),
            'win': torch.tensor(plans.Y_WIN.to_numpy(dtype=np.float32)),
            'net_return': torch.tensor(plans.NET_RETURN.to_numpy(dtype=np.float32)),
            'decision_ts': pd.to_datetime(plans.decision_timestamp).to_numpy()}
    for k in ('plan', 'portfolio', 'net_return'):
        if not torch.isfinite(data[k]).all(): raise ValueError(f'Nonfinite {k}')
    return tensors, data, manifest


class Standardizer:
    def __init__(self, tensors=None, state=None):
        if state is not None:
            self.mean = {g: torch.tensor(v[0]) for g, v in state.items()}
            self.std = {g: torch.tensor(v[1]) for g, v in state.items()}
            return
        self.mean = {g: tensors[g].reshape(-1, tensors[g].shape[-1]).mean(0) for g in GROUPS}
        self.std = {g: tensors[g].reshape(-1, tensors[g].shape[-1]).std(0).clamp_min(1e-6) for g in GROUPS}

    def __call__(self, tensors):
        return {g: (tensors[g] - self.mean[g]) / self.std[g] for g in GROUPS}

    def state(self):
        return {g: (self.mean[g].tolist(), self.std[g].tolist()) for g in GROUPS}


def batch(tensors, data, idx, device):
    s = data['state_idx'][idx]
    return ({g: tensors[g][s].to(device) for g in GROUPS}, data['plan'][idx].to(device), data['portfolio'][idx].to(device),
            {'outcome': data['outcome'][idx].to(device), 'win': data['win'][idx].to(device),
             'net_return': data['net_return'][idx].to(device)})


@torch.no_grad()
def evaluate(perception, core, tensors, data, device, temps=None, max_rows=50000):
    perception.eval(); core.eval()
    n = len(data['outcome']); rows = torch.arange(n) if n <= max_rows else torch.linspace(0, n - 1, max_rows).long()
    outs, targets = [], []
    for chunk in rows.split(4096):
        g, p, q, y = batch(tensors, data, chunk, device)
        logits = core(perception(g['bar'], g['seq'], g['c1h'], g['c4h'], p, q))
        outs.append({k: v.float().cpu() for k, v in logits.items()}); targets.append({k: v.cpu() for k, v in y.items()})
    logits = {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
    y = {k: torch.cat([t[k] for t in targets]) for k in targets[0]}
    t_out, t_win = (1.0, 1.0) if temps is None else temps
    p_out = F.softmax(logits['outcome'] / t_out, -1); p_win = torch.sigmoid(logits['win'] / t_win)
    onehot = F.one_hot(y['outcome'], 3).float()
    metrics = {'nll_outcome': float(F.cross_entropy(logits['outcome'] / t_out, y['outcome'])),
               'brier_outcome': float(((p_out - onehot) ** 2).sum(-1).mean()),
               'nll_win': float(F.binary_cross_entropy_with_logits(logits['win'] / t_win, y['win'])),
               'brier_win': float(((p_win - y['win']) ** 2).mean()),
               'ece_win': ece(p_win.numpy(), y['win'].numpy()), 'ece_tp_first': ece(p_out[:, 0].numpy(), (y['outcome'] == 0).float().numpy()),
               'mae_net_return': float((logits['net_return'] - y['net_return']).abs().mean()), 'rows': len(rows),
               'logit_abs_max': float(max(logits['outcome'].abs().max(), logits['win'].abs().max()))}
    perception.train(); core.train()
    return metrics, logits, y


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b if b < 1 else p <= b)
        if m.any(): total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def prior_baseline(train, val):
    prior = torch.bincount(train['outcome'], minlength=3).float(); prior /= prior.sum()
    pw = float(train['win'].mean())
    onehot = F.one_hot(val['outcome'], 3).float()
    return {'nll_outcome': float(-(onehot * prior.log()).sum(-1).mean()),
            'brier_outcome': float(((prior - onehot) ** 2).sum(-1).mean()),
            'nll_win': float(F.binary_cross_entropy(torch.full_like(val['win'], pw), val['win'])),
            'brier_win': float(((pw - val['win']) ** 2).mean()), 'train_prior_outcome': prior.tolist(), 'train_prior_win': pw}


def fit_temperature(logits, target, kind, iters=300):
    log_t = torch.zeros((), requires_grad=True); opt = torch.optim.Adam([log_t], lr=0.05)
    for _ in range(iters):
        t = log_t.exp().clamp(0.05, 10.0); opt.zero_grad()
        loss = F.cross_entropy(logits / t, target) if kind == 'ce' else F.binary_cross_entropy_with_logits(logits / t, target)
        loss.backward(); opt.step()
    return float(log_t.exp().clamp(0.05, 10.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', default='data/plan_dataset/train_v1'); ap.add_argument('--val', default='data/plan_dataset/val_v1')
    ap.add_argument('--steps', type=int, required=True); ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--eval-every', type=int, default=1000); ap.add_argument('--ckpt-every', type=int, default=50000)
    ap.add_argument('--out', default='models/plan_system_one'); ap.add_argument('--resume', action='store_true')
    ap.add_argument('--seed', type=int, default=1729); ap.add_argument('--max-cpu-steps', type=int, default=10000)
    ap.add_argument('--require-cuda', action='store_true'); ap.add_argument('--amp', action='store_true')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.require_cuda and device.type != 'cuda': raise SystemExit('CUDA required but unavailable')
    if device.type == 'cpu' and args.steps > args.max_cpu_steps:
        raise SystemExit(f'CPU training capped at {args.max_cpu_steps} steps; use GPU for {args.steps}')
    if args.steps > 10000:
        bench = Path(args.out) / 'gpu_benchmarks.json'
        if not bench.exists() or not all(json.loads(bench.read_text()).get(str(k), {}).get('passed') for k in (1000, 5000, 10000)):
            raise SystemExit('Runs >10K require passing 1K/5K/10K GPU benchmarks recorded in gpu_benchmarks.json')
    market = load_market_contract()
    if args.steps > 10000:
        market.require_verified_fees()
        raise SystemExit('500K blocked: LONG-only TRAIN/VAL EV, dataset oracle, mandatory baselines and example-count gates are not yet signed off')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    registry = get_feature_registry(); contract = FeatureAvailabilityContract(registry)
    layout = build_layout(contract.eligible_names(), contract)
    raw = pd.read_parquet(PARQUET, columns=['open', 'high', 'low', 'close', 'volume'])
    graph_window = raw[(raw.index >= '2017-08-18') & (raw.index < '2017-08-21')].iloc[:600]
    graph = RelationGraph(registry).qualify_on_raw_train(graph_window, contract)
    tr_t, tr, tr_m = load_split(args.train, layout); va_t, va, va_m = load_split(args.val, layout)
    if tr_m['split'] != 'train' or va_m['split'] != 'val': raise ValueError('Split manifest mismatch')
    if pd.Timestamp(tr_m['end']) >= pd.Timestamp(va_m['start']): raise ValueError('TRAIN must precede VAL')
    norm = Standardizer(tr_t); tr_t, va_t = norm(tr_t), norm(va_t)

    perception = GroupedRelationalPerception(layout, graph, contract).to(device)
    core = PlanJudgmentCore().to(device)
    params = list(perception.parameters()) + list(core.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    warm = max(1, min(2000, args.steps // 20))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == 'cuda')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    config = {**vars(args), 'device': str(device), 'git_commit': commit, 'train_manifest': tr_m['plans_sha256'],
              'val_manifest': va_m['plans_sha256'], 'registry_sha256': tr_m['registry_sha256'], 'torch': torch.__version__,
              'market': market.market, 'market_contract_sha256': market.sha256(),
              'action_space_contract_sha256': market.action_space_sha256(), 'fee_verified': market.fee_verified,
              'params': sum(p.numel() for p in params), 'train_plans': len(tr['outcome']), 'val_plans': len(va['outcome'])}
    step, best, history = 0, float('inf'), []
    last = out / 'last.pt'
    if args.resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        if ck['config']['train_manifest'] != config['train_manifest'] or ck['config']['val_manifest'] != config['val_manifest']:
            raise SystemExit('Resume refused: dataset manifests changed')
        perception.load_state_dict(ck['perception']); core.load_state_dict(ck['core'])
        opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['scheduler'])
        step, best, history = ck['step'], ck['best'], ck['history']
        torch.set_rng_state(ck['rng_cpu']); np.random.set_state(ck['rng_np']); random.setstate(ck['rng_py'])
    baseline = prior_baseline(tr, va)
    print(json.dumps({'config': config, 'val_prior_baseline': baseline}, indent=2, default=str), flush=True)

    def save(path, extra=None):
        torch.save({'perception': perception.state_dict(), 'core': core.state_dict(), 'optimizer': opt.state_dict(),
                    'scheduler': sched.state_dict(), 'step': step, 'best': best, 'history': history, 'config': config,
                    'standardizer': norm.state(), 'layout_names': list(layout.names), 'rng_cpu': torch.get_rng_state(),
                    'rng_np': np.random.get_state(), 'rng_py': random.getstate(), **(extra or {})}, path)

    generator = torch.Generator().manual_seed(args.seed + step)
    n = len(tr['outcome']); t0 = time.time(); t_last = t0
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    while step < args.steps:
        idx = torch.randint(0, n, (args.batch,), generator=generator)
        g, p, q, y = batch(tr_t, tr, idx, device)
        with torch.autocast(device.type, enabled=args.amp and device.type == 'cuda'):
            logits = core(perception(g['bar'], g['seq'], g['c1h'], g['c4h'], p, q))
        loss, parts = plan_losses({k: v.float() for k, v in logits.items()}, y)
        if not torch.isfinite(loss): raise SystemExit(f'Nonfinite loss at step {step}')
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        grad = float(torch.nn.utils.clip_grad_norm_(params, 1.0)); scaler.step(opt); scaler.update(); sched.step(); step += 1
        if step % args.eval_every == 0 or step == args.steps:
            metrics, _, _ = evaluate(perception, core, va_t, va, device)
            now = time.time()
            record = {'step': step, 'train_loss': float(loss), **{f'train_{k}': v for k, v in parts.items()}, 'grad_norm': grad,
                      'lr': sched.get_last_lr()[0], 'steps_per_s': args.eval_every / max(now - t_last, 1e-9),
                      'gpu_mem_gb': torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else None,
                      **{f'val_{k}': v for k, v in metrics.items()}}
            t_last = now; history.append(record); print(json.dumps(record), flush=True)
            score = metrics['nll_outcome'] + metrics['nll_win']
            if score < best: best = score; save(out / 'best.pt')
        if step % args.ckpt_every == 0: save(out / f'step_{step}.pt')
        if step % args.eval_every == 0: save(last)
    save(last)
    best_ck = torch.load(out / 'best.pt', map_location=device, weights_only=False)
    perception.load_state_dict(best_ck['perception']); core.load_state_dict(best_ck['core'])
    _, logits, y = evaluate(perception, core, va_t, va, device)
    t_out = fit_temperature(logits['outcome'], y['outcome'], 'ce'); t_win = fit_temperature(logits['win'], y['win'], 'bce')
    calibrated, _, _ = evaluate(perception, core, va_t, va, device, temps=(t_out, t_win))
    core.temp_outcome.fill_(t_out); core.temp_win.fill_(t_win); core.calibrated.fill_(True)
    save(out / 'best_calibrated.pt', {'temperatures': {'outcome': t_out, 'win': t_win}})
    beats = {k: calibrated[k] < baseline[k] for k in ('nll_outcome', 'brier_outcome', 'nll_win', 'brier_win')}
    report = {'config': config, 'steps': step, 'wall_s': time.time() - t0, 'best_val_score': best, 'val_prior_baseline': baseline,
              'val_calibrated': calibrated, 'temperatures': {'outcome': t_out, 'win': t_win}, 'beats_prior_baseline': beats,
              'history': history, 'test_evaluated': False,
              'note': 'VAL used for model selection and temperature only. TEST untouched. Temperature != proof of calibration: see ECE.'}
    (out / f'report_{step}.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({'final_val_calibrated': calibrated, 'baseline': baseline, 'beats': beats}, indent=2))


if __name__ == '__main__':
    main()
