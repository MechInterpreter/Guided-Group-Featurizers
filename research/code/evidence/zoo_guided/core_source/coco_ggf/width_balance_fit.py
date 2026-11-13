"""Resumable five-arm continuation and bounded GradNorm-inspired weighting.

The reconstruction coefficient stays one. Two object-loss coefficients adapt
using shared encoder gradients only; this is not unmodified canonical GradNorm.
"""
import copy
import gc
import hashlib
import time
import numpy as np
import torch
from coco_baselines import training
from coco_baselines.storage import identity
from coco_adaptation import core


def state_identity(value):
    """Hash every controller field, including Adam state and reference samples."""
    if torch.is_tensor(value):
        return dict(tensor_sha=training.weight_sha({'value':value.detach().cpu()}))
    if isinstance(value,np.ndarray):
        return dict(array_sha=hashlib.sha256(value.tobytes()).hexdigest(),shape=list(value.shape),dtype=str(value.dtype))
    if isinstance(value,dict):
        return {str(k):state_identity(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [state_identity(v) for v in value]
    if isinstance(value,np.generic):
        return value.item()
    return value


class Balancer:
    def __init__(self, recipe, device):
        self.recipe = recipe
        self.log_weights = torch.nn.Parameter(torch.tensor([np.log(.05), np.log(.01)], dtype=torch.float32, device=device))
        self.optimizer = torch.optim.Adam([self.log_weights], lr=recipe['weight_lr'])
        self.reference_sum = np.zeros(3, np.float64)
        self.reference_count = 0
        self.reference = None
        self.adaptations = 0

    def weights(self):
        bounds = self.recipe['bounds']
        w = self.log_weights.exp()
        clamped = torch.stack([w[0].clamp(*bounds['capture']),w[1].clamp(*bounds['detection'])])
        # Adam projects log parameters after each step. Preserve their gradient
        # at the boundary so exp/log roundoff cannot trap a weight at its floor.
        bounded = w + (clamped-w).detach()
        return torch.stack([w.new_tensor(1.),bounded[0],bounded[1]])

    def observe(self, losses):
        values = np.asarray([float(v.detach()) for v in losses])
        if not np.isfinite(values).all() or (values <= 1e-12).any():
            return False
        if self.reference is None:
            self.reference_sum += values
            self.reference_count += 1
            if self.reference_count == self.recipe['reference_batches']:
                self.reference = self.reference_sum / self.reference_count
        return self.reference is not None

    def adapt(self, losses, raw_norms):
        if not torch.isfinite(raw_norms).all():
            raise FloatingPointError('Nonfinite balancing norms')
        if self.reference is None or (raw_norms <= 1e-12).any():
            return dict(updated=False, reason='missing reference or zero shared gradient')
        ratio = torch.stack([v.detach() for v in losses]) / torch.as_tensor(self.reference, dtype=raw_norms.dtype, device=raw_norms.device)
        rates = ratio / ratio.mean().clamp_min(1e-12)
        strengths = self.weights() * raw_norms.detach()
        target = strengths.detach().mean() * rates.pow(self.recipe['alpha'])
        # Reconstruction is the fixed anchor; only the two object weights learn.
        objective = (strengths[1:] - target[1:]).abs().sum()
        self.optimizer.zero_grad()
        objective.backward()
        if not torch.isfinite(self.log_weights.grad).all():
            raise FloatingPointError('Nonfinite balancing gradient')
        self.optimizer.step()
        with torch.no_grad():
            for j, key in enumerate(['capture', 'detection']):
                self.log_weights[j].clamp_(np.log(self.recipe['bounds'][key][0]), np.log(self.recipe['bounds'][key][1]))
        self.adaptations += 1
        return dict(updated=True, objective=float(objective.detach()), target_norms=target.cpu().tolist(), relative_rates=rates.cpu().tolist())

    def state_dict(self):
        return dict(log_weights=self.log_weights.detach().cpu().clone(), optimizer=copy.deepcopy(self.optimizer.state_dict()), reference_sum=self.reference_sum.copy(), reference_count=self.reference_count, reference=None if self.reference is None else self.reference.copy(), adaptations=self.adaptations)

    def load_state_dict(self, state):
        self.log_weights.data.copy_(state['log_weights'].to(self.log_weights.device))
        self.optimizer.load_state_dict(state['optimizer'])
        for key in ['reference_sum', 'reference_count', 'reference', 'adaptations']:
            setattr(self, key, copy.deepcopy(state[key]))


def encoder_diagnostics(model, losses):
    parameters = [model.W_enc, model.b_enc]
    vectors = []
    for loss in losses:
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        vectors.append(torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p, g in zip(parameters, grads)]))
    norms = torch.stack([v.norm() for v in vectors])
    cosines = {}
    names = ['reconstruction', 'capture', 'detection']
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        denominator = float(norms[i] * norms[j])
        cosines[names[i] + '_vs_' + names[j]] = None if denominator <= 1e-20 else float(vectors[i] @ vectors[j]) / denominator
    if not torch.isfinite(norms).all():
        raise FloatingPointError('Nonfinite shared encoder gradient')
    return norms, cosines


def fit(ctx, key, initial, targets, updates, seed, balanced=False, supervised=True,
        reference_mse=None, initial_head=None, initial_balance=None, max_updates=None):
    cfg, base, store, device = ctx['cfg'], ctx['base'], ctx['store'], ctx['device']
    recipe = ctx['recipe']['balance']
    if reference_mse is None or not np.isfinite(reference_mse) or reference_mse<=0:
        raise ValueError('A positive finite common reconstruction reference is required')
    fingerprint = identity(dict(initial=training.weight_sha(initial), targets=targets, updates=updates, seed=seed,
        balanced=balanced, supervised=supervised, reference_mse=reference_mse,
        initial_head_sha=None if initial_head is None else training.weight_sha({'head':initial_head}),
        initial_balance=state_identity(initial_balance), base=base, cfg=cfg, balance=recipe,
        train_ids=ctx['packs']['train']['ids'], select_ids=ctx['packs']['select']['ids']))
    if balanced and not supervised:
        raise ValueError('Loss balancing requires the two supervised tasks')
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    model = core.new_model('vanilla', base, initial, device)
    head = torch.nn.Parameter(torch.zeros(len(targets), 2, device=device))
    if initial_head is not None:
        if initial_head.shape != head.shape or not torch.isfinite(initial_head).all():
            raise ValueError('Invalid continuation head')
        head.data.copy_(initial_head.to(device))
    opt = torch.optim.Adam(list(model.parameters()) + ([head] if supervised else []), lr=cfg['lr'])
    controller = Balancer(recipe, device) if balanced else None
    if initial_balance is not None:
        if controller is None:
            raise ValueError('Unexpected balancing state')
        controller.load_state_dict(initial_balance)
    rng = np.random.default_rng(seed)
    state = store.load(key)

    def measure():
        if supervised:
            result = core.validation(model, ctx['packs']['select'], targets, cfg, device)
            if not np.isfinite([result['macro_capture'],result['macro_ap'],result['reconstruction']['mse'],result['reconstruction']['ve']]).all():
                raise FloatingPointError('Nonfinite development result')
            return result
        r = training.measure(model, ctx['packs']['select']['x'].reshape(-1,model.d), device, cfg['batch'])
        if not np.isfinite([r['mse'],r['ve']]).all():
            raise FloatingPointError('Nonfinite reconstruction result')
        return dict(reconstruction=r, macro_capture=None, macro_ap=None, score=None)

    if state is None:
        v = measure()
        state = dict(updates=0, target=updates, fingerprint=fingerprint, weights=training.weights(model),
            head=head.detach().cpu().clone(), history=[dict(update=0,validation=v,eligible=v['reconstruction']['mse']<=reference_mse*1.02)],
            seconds=0., sampled_batches_sha=hashlib.sha256().hexdigest(), sampled_noise_sha=hashlib.sha256().hexdigest(),
            best_mse=training.weights(model), best_mse_score=v['reconstruction']['mse'], best_mse_update=0,
            best_eligible=None, best_eligible_update=None, best_eligible_score=None)
        if supervised and state['history'][0]['eligible']:
            state.update(best_eligible=training.weights(model),best_eligible_update=0,best_eligible_score=v['score'])
    else:
        if state['fingerprint'] != fingerprint:
            raise ValueError('Continuation identity changed: '+key)
        model.load_state_dict(state['weights'])
        head.data.copy_(state['head'].to(device))
        opt.load_state_dict(state['optimizer'])
        rng.bit_generator.state = state['sample_rng']
        torch.set_rng_state(state['torch_rng'])
        if device.startswith('cuda'):
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        if controller is not None:
            controller.load_state_dict(state['balance'])

    def save():
        state.update(weights=training.weights(model), head=head.detach().cpu().clone(),
            optimizer=copy.deepcopy(opt.state_dict()), sample_rng=copy.deepcopy(rng.bit_generator.state),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None,
            balance=None if controller is None else controller.state_dict())
        store.save(key,state,drive=True)

    limit = updates if max_updates is None else min(updates,max_updates)
    print(key,'resume',state['updates'],'/',updates,flush=True)
    save()
    committed = True
    try:
        while state['updates'] < limit:
            committed = False
            indices = rng.integers(len(ctx['packs']['train']['ids']),size=cfg['images_per_update'])
            state['sampled_batches_sha'] = hashlib.sha256(state['sampled_batches_sha'].encode()+indices.tobytes()).hexdigest()
            x = torch.as_tensor(np.array(ctx['packs']['train']['x'][indices],copy=True),device=device)
            xx=x.reshape(-1,model.d)
            model.train()
            started=time.monotonic()
            noise=torch.randn_like(xx) if cfg['snr'] else torch.zeros_like(xx)
            # Hash RNG state instead of copying a large noise tensor back to CPU.
            noise_state=torch.cuda.get_rng_state().cpu().numpy().tobytes() if device.startswith('cuda') else torch.get_rng_state().numpy().tobytes()
            state['sampled_noise_sha']=hashlib.sha256(state['sampled_noise_sha'].encode()+noise_state).hexdigest()
            noisy=xx+cfg['snr']*xx.std()*noise if cfg['snr'] else xx
            rec,_=model.loss(noisy,target=xx)
            cap=det=rec*0
            if supervised:
                cap,det,_=core.supervised_loss(model,x,[ctx['packs']['train']['masks'][i] for i in indices],targets,head,cfg)
            losses=[rec,cap,det]
            if not all(torch.isfinite(v) for v in losses):
                raise FloatingPointError('Nonfinite task loss')
            u=state['updates']+1
            weights_used=controller.weights().detach() if controller is not None else rec.new_tensor([1.,.05,.01])
            diagnostic=None
            if supervised and ((balanced and u%recipe['every']==0) or u%cfg['evaluate_every']==0):
                norms,cosines=encoder_diagnostics(model,losses)
                diagnostic=dict(raw_encoder_gradient_norms=norms.cpu().tolist(),weighted_encoder_gradient_norms=(norms*weights_used).cpu().tolist(),encoder_gradient_cosines=cosines)
            ready=controller.observe(losses) if controller is not None else False
            if controller is not None and ready and u%recipe['every']==0:
                diagnostic['adaptation']=controller.adapt(losses,norms)
            objective=rec if not supervised else rec+weights_used[1]*cap+weights_used[2]*det
            opt.zero_grad()
            objective.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(list(model.parameters())+([head] if supervised else []),cfg['grad_clip']))
            if not np.isfinite(norm):
                raise FloatingPointError('Nonfinite model gradient')
            opt.step()
            model.normalize_decoder()
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            state['updates']=u
            state['seconds']+=time.monotonic()-started
            committed=True
            if u%cfg['evaluate_every']==0 or u==updates:
                v=measure()
                eligible=v['reconstruction']['mse']<=reference_mse*1.02
                if v['reconstruction']['mse']<state['best_mse_score']:
                    state.update(best_mse=training.weights(model),best_mse_score=v['reconstruction']['mse'],best_mse_update=u)
                if supervised and eligible and (state['best_eligible_score'] is None or v['score']>state['best_eligible_score']):
                    state.update(best_eligible=training.weights(model),best_eligible_score=v['score'],best_eligible_update=u)
                state['history'].append(dict(update=u,validation=v,eligible=eligible,losses=[float(v.detach()) for v in losses],weights_used=weights_used.cpu().tolist(),weights_next=None if controller is None else controller.weights().detach().cpu().tolist(),diagnostics=diagnostic,gradient_norm=norm,seconds_per_update=state['seconds']/u))
                save()
                store.write('histories/'+key+'.json',state['history'])
                store.event('continuation_checkpoint',key=key,update=u,total=updates,capture=v['macro_capture'],AP=v['macro_ap'],VE=v['reconstruction']['ve'],eligible=eligible,weights=weights_used.cpu().tolist(),seconds_per_update=state['seconds']/u)
            elif u%cfg['local_save_every']==0:
                save()
                store.event('continuation_progress',key=key,update=u,total=updates,seconds_per_update=state['seconds']/u)
        save()
    except BaseException:
        # A partial optimizer/balancer update must replay from the last valid slot.
        if committed:
            save()
        raise
    finally:
        del model,head,opt,controller
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    return state
