"""Rebuild disposable local inference snapshots from committed checkpoints.

No endpoint fitting, readout selection, optimizer update, or Drive checkpoint
write occurs here. The caller first verifies the frozen endpoint catalog.
"""
import os
from pathlib import Path

import torch


def restore(ctx, rows, required, weight_sha, endpoint_stores=None):
    folder = Path(ctx['local'])/'frozen_weights'
    folder.mkdir(parents=True, exist_ok=True)
    restored = []
    for row in rows:
        path = folder/(row['weights_sha']+'.pt')
        if path.is_file():
            weights = torch.load(path, map_location='cpu', weights_only=True)
            if weight_sha(weights) != row['weights_sha']:
                raise ValueError('Local frozen weights differ from frozen endpoint: '+row['key'])
            continue
        state = None
        for store in (endpoint_stores or [ctx['store']]):
            state = store.load(row['key'])
            if state is not None:
                break
        if state is None:
            raise FileNotFoundError('Missing frozen endpoint in verified primary and reuse sources: '+row['key'])
        if state['updates'] != row['additional_updates']:
            raise ValueError('Committed checkpoint is not the frozen endpoint: '+row['key'])
        if weight_sha(state['weights']) != row['weights_sha']:
            raise ValueError('Committed weights differ from frozen endpoint: '+row['key'])
        pending = path.with_name(path.name+'.pending-'+str(os.getpid()))
        try:
            with pending.open('wb') as f:
                torch.save(state['weights'], f)
                f.flush()
                os.fsync(f.fileno())
            check = torch.load(pending, map_location='cpu', weights_only=True)
            if weight_sha(check) != row['weights_sha']:
                raise ValueError('Restored frozen-weight read-back failed: '+row['key'])
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)
        restored.append(row['key'])
        print('RESTORED FROZEN ANALYSIS WEIGHTS', row['key'], flush=True)
        del state, check
    return restored
