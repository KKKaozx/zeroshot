"""Isolated fixed-update/checkpoint primitives for the future OXE runner.

The caller supplies the model state explicitly (so a frozen backbone need not be
saved). Checkpoints are taken only after optimizer updates, without gradient
accumulation or asynchronous data workers. This module does not admit a dataset,
select an architecture, or change the existing epoch-based training entry point.
Only load trusted local checkpoints: RNG state uses Python/NumPy serialization.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import tempfile

import numpy as np
import torch


FORMAT = 'fixed-update-training-state-v1'


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(v) for v in value)
    return copy.deepcopy(value)


class ResumableBatchPlan:
    """Own the current pass, cursor and sampling RNG, independent of model RNG.

    Each pass contains N draws. Shuffle uses a permutation; weighted sampling
    draws N indices with replacement. The last batch of each pass is retained,
    including a short tail. Prefetch is intentionally excluded: reserve exactly
    one batch and commit it only after its optimizer update succeeds.
    """
    def __init__(self, indices, *, batch_size, total_updates, seed,
                 dataset_identity, mode='shuffle', weights=None):
        self.indices = list(indices)
        if (not self.indices or any(type(i) is not int or i < 0 for i in self.indices)
                or len(set(self.indices)) != len(self.indices)):
            raise ValueError('Unique nonnegative training indices required')
        if not isinstance(dataset_identity, str) or not dataset_identity:
            raise ValueError('Frozen dataset/window identity required')
        if type(seed) is not int or seed < 0 or seed >= 2**63:
            raise ValueError('Seed must be an integer in [0, 2**63)')
        self.batch_size = _positive_int(batch_size, 'batch_size')
        self.total_updates = _positive_int(total_updates, 'total_updates')
        if mode not in ('shuffle', 'weighted_replacement'):
            raise ValueError('Unknown sampling mode')
        if mode == 'weighted_replacement':
            self.weights = torch.as_tensor(weights, dtype=torch.float64).clone()
            if (self.weights.shape != (len(self.indices),)
                    or not torch.isfinite(self.weights).all() or not (self.weights > 0).all()
                    or not torch.isfinite(self.weights.sum())):
                raise ValueError('Finite positive weight for every eligible index required')
        else:
            if weights is not None:
                raise ValueError('Shuffle must not silently ignore sampling weights')
            self.weights = None
        self.contract = {
            'dataset_identity': dataset_identity, 'indices_sha256': _digest(self.indices),
            'size': len(self.indices), 'batch_size': batch_size, 'total_updates': total_updates,
            'seed': seed, 'mode': mode, 'drop_last': False,
            'weights_sha256': _digest(self.weights.tolist()) if self.weights is not None else None,
        }
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.order = torch.empty(0, dtype=torch.int64)
        self.cursor, self.pass_index, self.completed_updates = 0, -1, 0
        self._pending_end = None

    def begin_batch(self):
        if self._pending_end is not None:
            raise RuntimeError('Commit or abort the reserved batch first')
        if self.completed_updates >= self.total_updates:
            raise StopIteration('Fixed total update budget exhausted')
        if self.cursor == len(self.order):
            n = len(self.indices)
            self.order = (torch.randperm(n, generator=self.generator)
                          if self.weights is None else
                          torch.multinomial(self.weights, n, replacement=True,
                                            generator=self.generator))
            self.cursor = 0
            self.pass_index += 1
        end = min(self.cursor + self.batch_size, len(self.order))
        self._pending_end = end
        return [self.indices[i] for i in self.order[self.cursor:end].tolist()]

    def commit_batch(self):
        if self._pending_end is None:
            raise RuntimeError('No reserved optimizer batch')
        self.cursor = self._pending_end
        self._pending_end = None
        self.completed_updates += 1

    def abort_batch(self):
        """Release reservation; the same indices will be returned on retry.

        The caller is responsible for rolling back any partial optimizer update.
        This is not a rollback of model parameters or model RNG.
        """
        if self._pending_end is None:
            raise RuntimeError('No reserved batch to abort')
        self._pending_end = None

    def state_dict(self):
        if self._pending_end is not None:
            raise RuntimeError('Checkpoint only at a committed optimizer boundary')
        return _snapshot({'contract': self.contract, 'order': self.order,
                          'generator_state': self.generator.get_state(),
                          'cursor': self.cursor, 'pass_index': self.pass_index,
                          'completed_updates': self.completed_updates})

    def validate_state(self, state):
        if self._pending_end is not None:
            raise RuntimeError('Cannot restore over a reserved batch')
        if state['contract'] != self.contract:
            raise ValueError('Dataset, batch, sampling or total-budget contract differs')
        n = len(self.indices)
        cursor, pass_index, done = (state[k] for k in ('cursor', 'pass_index', 'completed_updates'))
        order, rng = state['order'], state['generator_state']
        if (any(type(x) is not int for x in (cursor, pass_index, done))
                or not 0 <= done <= self.total_updates or not 0 <= cursor <= n):
            raise ValueError('Invalid update counter or sampling cursor')
        if (not isinstance(order, torch.Tensor) or order.ndim != 1
                or order.dtype != torch.int64 or order.device.type != 'cpu'):
            raise ValueError('CPU int64 sample order required')
        if pass_index == -1:
            if order.numel() or cursor or done:
                raise ValueError('Invalid unstarted sampling state')
        else:
            if (pass_index < 0 or order.shape != (n,) or (order < 0).any()
                    or (order >= n).any() or (cursor != n and cursor % self.batch_size)):
                raise ValueError('Invalid sample order or batch boundary')
            if self.weights is None and len(torch.unique(order)) != n:
                raise ValueError('Shuffle order must contain every position once')
            expected = pass_index * math.ceil(n/self.batch_size) + math.ceil(cursor/self.batch_size)
            if expected != done:
                raise ValueError('Sample cursor and optimizer-update counter disagree')
        test_generator = torch.Generator(device='cpu')
        test_generator.set_state(rng)  # validate against an isolated generator

    def load_state_dict(self, state):
        self.validate_state(state)
        self.order = state['order'].clone()
        self.generator.set_state(state['generator_state'].clone())
        self.cursor, self.pass_index, self.completed_updates = (
            state[k] for k in ('cursor', 'pass_index', 'completed_updates'))


class UpdateSchedule:
    """Explicit immutable LR horizon; completed_updates is the sole clock.

    apply() sets rates for the next update. Warmup uses (completed+1)/warmup;
    cosine then uses (completed-warmup)/(total-warmup), reaching zero after
    the last completed update. Segment boundaries never alter this horizon.
    """
    def __init__(self, *, total_updates, base_lrs, kind, warmup_updates=0):
        total_updates = _positive_int(total_updates, 'total_updates')
        if (type(warmup_updates) is not int or not 0 <= warmup_updates < total_updates
                or kind not in ('constant', 'cosine')):
            raise ValueError('Invalid schedule or warmup horizon')
        rates = [float(x) for x in base_lrs]
        if not rates or any(not math.isfinite(x) or x <= 0 for x in rates):
            raise ValueError('Explicit finite positive base LR per optimizer group required')
        self.contract = {'total_updates': total_updates, 'base_lrs': rates,
                         'kind': kind, 'warmup_updates': warmup_updates}

    def rates(self, completed_updates):
        c = self.contract
        if type(completed_updates) is not int or not 0 <= completed_updates <= c['total_updates']:
            raise ValueError('Update count outside the fixed schedule')
        if completed_updates < c['warmup_updates']:
            scale = (completed_updates + 1) / c['warmup_updates']
        elif c['kind'] == 'constant':
            scale = 1.
        else:
            phase = (completed_updates-c['warmup_updates'])/(c['total_updates']-c['warmup_updates'])
            scale = .5*(1.+math.cos(math.pi*phase))
        return [lr*scale for lr in c['base_lrs']]

    def apply(self, optimizer, completed_updates):
        rates = self.rates(completed_updates)
        if len(optimizer.param_groups) != len(rates):
            raise ValueError('Optimizer group count differs from declared base rates')
        for group, rate in zip(optimizer.param_groups, rates):
            group['lr'] = rate


def _environment(include_cuda):
    if not include_cuda and torch.cuda.is_initialized():
        raise ValueError('Do not omit RNG for initialized CUDA devices')
    if include_cuda and not torch.cuda.is_available():
        raise ValueError('CUDA RNG requested but unavailable')
    return {
        'python': list(sys.version_info[:3]), 'platform': platform.system(),
        'numpy': np.__version__, 'torch': str(torch.__version__),
        'cuda_version': torch.version.cuda,
        'cuda_devices': torch.cuda.device_count() if include_cuda else 0,
        'deterministic_debug_mode': torch.get_deterministic_debug_mode(),
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
    }


def capture_rng(*, include_cuda):
    environment = _environment(include_cuda)
    return _snapshot({'environment': environment, 'include_cuda': include_cuda,
                      'python': random.getstate(), 'numpy': np.random.get_state(),
                      'torch_cpu': torch.get_rng_state(),
                      'torch_cuda': torch.cuda.get_rng_state_all() if include_cuda else []})


def _validate_rng(state):
    if state['environment'] != _environment(state['include_cuda']):
        raise ValueError('RNG environment/backend settings differ; exact replay is not qualified')
    if len(state['torch_cuda']) != state['environment']['cuda_devices']:
        raise ValueError('Saved CUDA device count differs')
    random.Random().setstate(state['python'])
    np.random.RandomState().set_state(state['numpy'])
    torch.Generator(device='cpu').set_state(state['torch_cpu'])
    if any(not isinstance(s, torch.Tensor) or s.dtype != torch.uint8
           or s.ndim != 1 or s.device.type != 'cpu' for s in state['torch_cuda']):
        raise ValueError('Invalid CUDA RNG state tensor')


def restore_rng(state):
    _validate_rng(state)
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if state['include_cuda']:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def save_checkpoint(path, *, model_state, optimizer, plan, schedule, contract,
                    include_cuda, extra=None):
    """Atomically save after commit + LR apply; pass trainable model state only.

    contract must bind the model/backbone/data/augmentation/optimizer identities
    in the eventual main runner. This helper cannot infer those identities.
    """
    if not isinstance(contract, dict) or not contract:
        raise ValueError('Explicit nonempty training contract required')
    plan_state = plan.state_dict()
    if schedule.contract['total_updates'] != plan.total_updates:
        raise ValueError('Sampling and LR total budgets differ')
    expected_lrs = schedule.rates(plan.completed_updates)
    if [g['lr'] for g in optimizer.param_groups] != expected_lrs:
        raise ValueError('Apply the next-update LR before taking a checkpoint')
    payload = _snapshot({'format': FORMAT, 'contract': contract, 'contract_sha256': _digest(contract),
                         'model': model_state, 'optimizer': optimizer.state_dict(),
                         'plan': plan_state, 'schedule': schedule.contract,
                         'rng': capture_rng(include_cuda=include_cuda), 'extra': extra})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def restore_checkpoint(path, *, model_state_loader, optimizer, plan, schedule, contract):
    """Restore a trusted local checkpoint; restore RNG last, after initialization.

    Contract/budget/cursor checks run before calling the model loader. Any caller
    loader failure is fatal; this function does not roll back partial loads.
    """
    payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    if (payload.get('format') != FORMAT or payload.get('contract') != contract
            or payload.get('contract_sha256') != _digest(contract)):
        raise ValueError('Checkpoint training contract differs')
    plan.validate_state(payload['plan'])
    if payload['schedule'] != schedule.contract:
        raise ValueError('LR schedule/base rates/total horizon differ')
    if schedule.contract['total_updates'] != plan.total_updates:
        raise ValueError('Sampling and LR budgets differ')
    expected_lrs = schedule.rates(payload['plan']['completed_updates'])
    if [g['lr'] for g in payload['optimizer']['param_groups']] != expected_lrs:
        raise ValueError('Saved optimizer LR disagrees with the update count')
    state = payload['rng']
    _validate_rng(state)
    model_state_loader(payload['model'])
    optimizer.load_state_dict(payload['optimizer'])
    plan.load_state_dict(payload['plan'])
    restore_rng(state)
    return payload['extra']
