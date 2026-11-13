"""Resumable wrapper around unchanged Goodfire classes and released loss calls."""
import copy
import hashlib
import time

import numpy as np
import torch
from bsf import VanillaBSF, GrassmannianBSF, GroupLassoBSF

CLASSES = dict(vanilla=VanillaBSF, grassmannian=GrassmannianBSF, group_lasso=GroupLassoBSF)


def make_model(method, cfg, d=768):
    kw = dict(d=d, n_groups=cfg['groups'], group_size=cfg['block_rank'])
    if method == 'group_lasso': kw['target_l0'] = cfg['active_groups']
    else: kw['l0'] = cfg['active_groups']
    return CLASSES[method](**kw)


def weights(model): return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def weight_sha(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        a = v.contiguous().numpy(); h.update(k.encode()); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


@torch.no_grad()
def measure(model, x, device, batch=2048):
    model.eval(); sse = 0.; count = 0; active = 0.; alive = np.zeros(model.n_groups, bool)
    # Use one global held-out mean, not a separate mean for each mini-batch.
    mean = np.mean(x, axis=0, dtype=np.float64)
    total = float(np.sum((x.astype(np.float64)-mean)**2))
    for lo in range(0, len(x), batch):
        xb = torch.as_tensor(x[lo:lo+batch], dtype=torch.float32, device=device)
        pred, z = model(xb); sse += float((pred-xb).double().square().sum())
        on = z.norm(dim=-1) > 1e-6; active += float(on.sum()); alive |= on.any(0).cpu().numpy(); count += len(xb)
    return dict(mse=sse/(count*x.shape[1]), ve=1-sse/max(total, 1e-12), active_groups=active/count, dead_groups=int((~alive).sum()), patches=count)


def fit(store, key, method, x, selection, cfg, seed, lr, device, max_updates=None):
    """No masks or labels enter this function. Full floor batches like bsf.train."""
    torch.manual_seed(seed); model = make_model(method, cfg, x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    nb = len(x)//cfg['batch']; target = nb*cfg['epochs']
    if nb < 1 or target < 4: raise ValueError('Need at least one full batch and four training updates')
    schedule = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=target, pct_start=cfg['warmup'], anneal_strategy='cos', div_factor=25, final_div_factor=1e4)
    state = store.load(key)
    if state is None:
        state = dict(updates=0, weights=weights(model), best=None, fallback=None, best_score=None, fallback_score=None,
                     best_update=None, history=[], seconds=0., permutation=None, opt=opt.state_dict(), scheduler=schedule.state_dict(),
                     rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None,
                     method=method, seed=seed, lr=lr, target=target, parameters=sum(p.numel() for p in model.parameters()), class_name=type(model).__module__+'.'+type(model).__name__)
    else:
        if state['target'] != target: raise ValueError('Training horizon changed; use a new run identity')
        model.load_state_dict(state['weights']); opt.load_state_dict(state['opt']); schedule.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['rng'])
        if device.startswith('cuda') and state['cuda_rng'] is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
    tx = torch.as_tensor(x, dtype=torch.float32, device=device)
    stop = target if max_updates is None else min(target, max_updates)
    print(key, 'resume', state['updates'], '/', target, flush=True)
    if state['updates'] >= stop: return state

    def checkpoint(drive=True):
        state.update(weights=weights(model), opt=copy.deepcopy(opt.state_dict()), scheduler=copy.deepcopy(schedule.state_dict()),
                     rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None)
        store.save(key, state, drive)

    try:
        while state['updates'] < stop:
            u = state['updates']; epoch, bi = divmod(u, nb)
            if bi == 0: state['permutation'] = torch.randperm(len(tx))
            idx = state['permutation'][bi*cfg['batch']:(bi+1)*cfg['batch']]
            xb = tx[idx.to(device)]; model.train(); started = time.monotonic()
            noisy = xb if cfg['snr'] <= 0 else xb + cfg['snr']*xb.std()*torch.randn_like(xb)
            loss, info = model.loss(noisy, target=xb)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            opt.zero_grad(); loss.backward(); opt.step(); schedule.step(); model.normalize_decoder()
            if device.startswith('cuda'): torch.cuda.synchronize()
            state['seconds'] += time.monotonic()-started; state['updates'] += 1
            due = state['updates'] % cfg['save_updates'] == 0 or state['updates'] == target
            if due:
                val = measure(model, selection, device, cfg['batch'])
                eligible = method != 'group_lasso' or abs(val['active_groups']-cfg['active_groups']) <= cfg['sparsity_tolerance']*cfg['active_groups']
                row = dict(update=state['updates'], epoch=state['updates']/nb, training_loss=float(loss.detach()), lr=float(opt.param_groups[0]['lr']),
                           selection=val, sparsity_eligible=eligible, loss_info={k: float(v) for k, v in info.items()}, seconds_per_update=state['seconds']/state['updates'])
                state['history'].append(row)
                if state['fallback_score'] is None or val['mse'] < state['fallback_score']:
                    state.update(fallback=weights(model), fallback_score=val['mse'])
                if eligible and (state['best_score'] is None or val['mse'] < state['best_score']):
                    state.update(best=weights(model), best_score=val['mse'], best_update=state['updates'])
                checkpoint()
                store.write(f'histories/{key}.json', state['history'])
                store.event('training_checkpoint', key=key, updates=state['updates'], total=target, epoch=row['epoch'], selection_mse=val['mse'], selection_ve=val['ve'], active_groups=val['active_groups'], seconds_per_update=row['seconds_per_update'])
            elif state['updates'] % cfg['local_save_updates'] == 0:
                checkpoint(drive=False)
                store.event('training_heartbeat', key=key, updates=state['updates'], total=target, compute_seconds=state['seconds'])
        checkpoint()
    except BaseException as exc:
        checkpoint(drive=False)
        try:
            checkpoint(); store.event('training_interrupted', key=key, updates=state['updates'], error_type=type(exc).__name__, error=str(exc))
        except BaseException: pass
        raise
    return state


def preflight(x, cfg, device):
    result = []; batch = min(cfg['batch'], len(x))
    for method in cfg['methods']:
        torch.manual_seed(123); m = make_model(method, cfg, x.shape[1]).to(device); opt = torch.optim.Adam(m.parameters(), lr=cfg['lrs'][0])
        xb = torch.as_tensor(x[:batch], device=device); times = []
        for i in range(8):
            t = time.monotonic(); m.train(); loss, _ = m.loss(xb, target=xb)
            opt.zero_grad(); loss.backward(); opt.step(); m.normalize_decoder()
            if device.startswith('cuda'): torch.cuda.synchronize()
            if i >= 3: times.append(time.monotonic()-t)
        m.eval()
        with torch.no_grad():
            pred, z = m(xb); contributions = torch.einsum('ngr,grd->nd', z, m.atoms())
            if not torch.allclose(pred, contributions, atol=1e-4, rtol=1e-4): raise ValueError('Decoder contribution accounting failed')
        seconds = float(np.median(times)); updates = (len(x)//cfg['batch'])*cfg['epochs']
        result.append(dict(method=method, seconds_per_update=seconds, fits=len(cfg['seeds'])*len(cfg['lrs']), updates_per_fit=updates, gradient_minutes=seconds*updates*len(cfg['seeds'])*len(cfg['lrs'])/60, parameters=sum(p.numel() for p in m.parameters())))
        del m, opt
    return dict(measurements=result, rough_gradient_minutes=sum(r['gradient_minutes'] for r in result),
                excludes='Image/annotation downloads, DINO extraction, validation, checkpoint IO and final annotation audit; this is a compute estimate, not a runtime guarantee.')

