"""Reconstruction-only continuations; preserve released classes and losses."""
import copy, gc, hashlib, time
import numpy as np
import torch
from coco_baselines import training
from coco_baselines.storage import identity
from coco_adaptation import core


def fit(ctx, key, method, initial, updates, seed, max_updates=None):
    cfg, base, s = ctx['cfg'], ctx['base'], ctx['store']
    device = ctx['device']
    fingerprint = identity(dict(method=method, initial=training.weight_sha(initial),
        updates=updates, seed=seed, lr=cfg['lr'], snr=cfg['snr'], clip=cfg['grad_clip'],
        images=cfg['images_per_update'], train_ids=ctx['packs']['train']['ids'],
        selection_ids=ctx['packs']['select']['ids'], base=base))
    torch.manual_seed(seed)
    if device.startswith('cuda'): torch.cuda.manual_seed_all(seed)
    m = core.new_model(method, base, initial, device)
    opt = torch.optim.Adam(m.parameters(), lr=cfg['lr'])
    rng = np.random.default_rng(seed)
    st = s.load(key)

    def measure():
        p = ctx['packs']['select']
        return training.measure(m, p['x'].reshape(-1, m.d), device, cfg['batch'])

    def eligible(v):
        return method != 'group_lasso' or abs(v['active_groups']-base['active_groups']) <= base['sparsity_tolerance']*base['active_groups']

    if st is None:
        v = measure()
        if not np.isfinite([v['mse'],v['ve'],v['active_groups']]).all():
            raise FloatingPointError('Nonfinite initial development measurement: '+key)
        st = dict(updates=0, target=updates, fingerprint=fingerprint, method=method,
            weights=training.weights(m), history=[dict(update=0, selection=v, sparsity_eligible=eligible(v))],
            best_mse=training.weights(m), best_mse_score=v['mse'], best_mse_update=0,
            best_sparse=training.weights(m) if eligible(v) else None,
            best_sparse_score=v['mse'] if eligible(v) else None, best_sparse_update=0 if eligible(v) else None,
            seconds=0., sampled_batches_sha=hashlib.sha256().hexdigest())
    else:
        if st['fingerprint'] != fingerprint: raise ValueError('Continuation identity changed: '+key)
        m.load_state_dict(st['weights']); opt.load_state_dict(st['optimizer'])
        rng.bit_generator.state = st['sample_rng']
        torch.set_rng_state(st['torch_rng'])
        if device.startswith('cuda'): torch.cuda.set_rng_state_all(st['cuda_rng'])

    def save():
        st.update(weights=training.weights(m), optimizer=copy.deepcopy(opt.state_dict()),
            sample_rng=copy.deepcopy(rng.bit_generator.state), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None)
        s.save(key, st, drive=True)

    limit = updates if max_updates is None else min(updates, max_updates)
    print(key, 'resume', st['updates'], '/', updates, flush=True)
    save()  # A complete update-zero state exists even if the first step fails.
    committed = True
    try:
        while st['updates'] < limit:
            committed = False
            idx = rng.integers(len(ctx['packs']['train']['ids']), size=cfg['images_per_update'])
            st['sampled_batches_sha'] = hashlib.sha256(st['sampled_batches_sha'].encode()+idx.tobytes()).hexdigest()
            x = torch.as_tensor(np.array(ctx['packs']['train']['x'][idx], copy=True), device=device)
            xx = x.reshape(-1, m.d); m.train(); started = time.monotonic()
            noisy = xx + cfg['snr']*xx.std()*torch.randn_like(xx) if cfg['snr'] else xx
            loss, info = m.loss(noisy, target=xx)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite native objective: '+key)
            opt.zero_grad(); loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(m.parameters(), cfg['grad_clip']))
            if not np.isfinite(norm): raise FloatingPointError('Nonfinite gradient: '+key)
            opt.step(); m.normalize_decoder()
            if device.startswith('cuda'): torch.cuda.synchronize()
            st['seconds'] += time.monotonic()-started; st['updates'] += 1
            committed = True
            u = st['updates']
            if u % cfg['evaluate_every'] == 0 or u == updates:
                v = measure(); ok = eligible(v)
                if not np.isfinite([v['mse'],v['ve'],v['active_groups']]).all():
                    raise FloatingPointError('Nonfinite development measurement: '+key)
                if v['mse'] < st['best_mse_score']:
                    st.update(best_mse=training.weights(m), best_mse_score=v['mse'], best_mse_update=u)
                if ok and (st['best_sparse_score'] is None or v['mse'] < st['best_sparse_score']):
                    st.update(best_sparse=training.weights(m), best_sparse_score=v['mse'], best_sparse_update=u)
                st['history'].append(dict(update=u, selection=v, sparsity_eligible=ok,
                    native_loss=float(loss.detach()), native_info={k:float(v) for k,v in info.items()},
                    gradient_norm=norm, seconds_per_update=st['seconds']/u))
                save(); s.write('histories/'+key+'.json',st['history'])
                s.event('continuation_checkpoint',key=key,update=u,total=updates,
                    mse=v['mse'],VE=v['ve'],active_groups=v['active_groups'],sparsity_eligible=ok)
            elif u % cfg['local_save_every'] == 0:
                save(); s.event('continuation_progress',key=key,update=u,total=updates,
                    loss=float(loss.detach()),seconds_per_update=st['seconds']/u)
        save()
    except BaseException:
        # Never persist a partially completed optimizer step or mutated dual
        # buffers. On a mid-step interrupt, replay from the last valid slot.
        if committed: save()
        raise
    finally:
        del m,opt
        gc.collect()
        if device.startswith('cuda'): torch.cuda.empty_cache()
    return st
