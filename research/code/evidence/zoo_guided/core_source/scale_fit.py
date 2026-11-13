"""Fixed-endpoint paired fits with exact checkpointed RNG/Adam/controller state."""
import copy,gc,hashlib,time
import numpy as np
import torch
from coco_baselines import training
from coco_baselines.storage import identity
from coco_adaptation import core
from coco_ggf.capacity_fit import Balancer,state_identity
from scale_models import model,encoder_parameters
from scale_metrics import evaluate


def fit(ctx,key,method,groups,active,seed,initial,target,limit,targets=None,cap=.4,
        learning_rate=None,fixed=False,coefficient=.001,stream=0):
    rule=ctx['rule'];device=ctx['device'];s=ctx['store'];d=ctx['packs']['train']['x'].shape[-1]
    supervised=targets is not None;lr=rule['lr'] if learning_rate is None else learning_rate
    recipe=copy.deepcopy(ctx['balance_recipe']);recipe['bounds']['capture'][1]=cap;recipe['bounds']['detection'][1]=cap
    ident=identity(dict(method=method,groups=groups,k=active,width=rule['width'],seed=seed,initial=training.weight_sha(initial),
        target=target,targets=targets,cap=cap,lr=lr,fixed=fixed,coefficient=coefficient,stream=stream,
        train=ctx['packs']['train']['identity'],select=ctx['packs']['select']['identity'],rule=rule,balance=recipe))
    st=s.load(key)
    if st is not None:
        if st['fingerprint']!=ident:raise ValueError('Resume identity changed: '+key)
        if st['updates']>=min(target,limit):
            print('Reused completed fit/prefix:',key,st['updates'],'/',target,flush=True)
            return st
    torch.manual_seed(seed)
    m=model(method,groups,rule['width'],active,d,coefficient).to(device);m.load_state_dict(initial)
    head=torch.nn.Parameter(torch.zeros(len(targets),2,device=device)) if supervised else None
    opt=torch.optim.Adam(list(m.parameters())+([head] if supervised else []),lr=lr)
    controller=Balancer(recipe,device) if supervised and not fixed else None
    # Construction must not advance the common image/noise schedule.
    phase_seed=126000+seed+100000*stream
    rng=np.random.default_rng(phase_seed);noise=torch.Generator(device=device).manual_seed(phase_seed)
    if st is None:
        st=dict(updates=0,target=target,fingerprint=ident,method=method,groups=groups,active=active,
            seconds=0.,history=[],sampled_batches_sha=hashlib.sha256().hexdigest(),sampled_noise_sha=hashlib.sha256().hexdigest())
    else:
        m.load_state_dict(st['weights']);opt.load_state_dict(st['optimizer']);rng.bit_generator.state=st['sample_rng'];noise.set_state(st['noise_rng'])
        if supervised:head.data.copy_(st['head'].to(device))
        if controller is not None:controller.load_state_dict(st['balance'])
    ctx['active_key']=key
    def save(force=False):
        st.update(weights=training.weights(m),optimizer=copy.deepcopy(opt.state_dict()),sample_rng=copy.deepcopy(rng.bit_generator.state),
            noise_rng=noise.get_state(),head=None if head is None else head.detach().cpu().clone(),
            balance=None if controller is None else controller.state_dict())
        save_fn=getattr(s,'_width_drive_recovery_original_save',s.save) if force else s.save
        save_fn(key,st,drive=True)
    def score():
        if supervised:return evaluate(m,ctx['packs']['select'],targets,ctx['cfg'],device)
        return dict(reconstruction=training.measure(m,ctx['packs']['select']['x'].reshape(-1,d),device,512))
    save(force=True);committed=True
    print(key,'resume',st['updates'],'/',target,'stage limit',limit,flush=True)
    try:
        while st['updates']<min(target,limit):
            committed=False
            ids=rng.integers(len(ctx['packs']['train']['ids']),size=rule['images_per_update'])
            x=torch.as_tensor(np.array(ctx['packs']['train']['x'][ids],copy=True),device=device);xx=x.reshape(-1,d)
            perturb=torch.randn(xx.shape,device=device,generator=noise)
            next_batch=hashlib.sha256(st['sampled_batches_sha'].encode()+ids.tobytes()).hexdigest()
            next_noise=hashlib.sha256(st['sampled_noise_sha'].encode()+perturb.detach().cpu().numpy().tobytes()).hexdigest()
            m.train();started=time.monotonic();rec,info=m.loss(xx+rule['snr']*xx.std()*perturb,target=xx)
            cap_loss=det_loss=rec*0;weights=[1.,0.,0.]
            u=st['updates']+1
            if supervised:
                cap_loss,det_loss,_=core.supervised_loss(m,x,[ctx['packs']['train']['masks'][i] for i in ids],targets,head,ctx['cfg'])
                terms=[rec,cap_loss,det_loss]
                if controller is not None:
                    controller.observe(terms)
                    weights=controller.weights().detach().cpu().tolist()
                    if u%recipe['every']==0 and controller.reference is not None:
                        norms=[]
                        for term in terms:
                            gradients=torch.autograd.grad(term,encoder_parameters(m),retain_graph=True,allow_unused=True)
                            norms.append(torch.sqrt(sum((g.detach().square().sum() for g in gradients if g is not None),rec.detach()*0)))
                        controller.adapt(terms,torch.stack(norms))
                else:weights=rule['fixed_weights']
            loss=rec+weights[1]*cap_loss+weights[2]*det_loss
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective '+key)
            opt.zero_grad(set_to_none=True);loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(list(m.parameters())+([head] if supervised else []),rule['clip']))
            if not np.isfinite(norm):raise FloatingPointError('Nonfinite gradient '+key)
            opt.step();m.normalize_decoder()
            if device.startswith('cuda'):torch.cuda.synchronize()
            st.update(updates=u,seconds=st['seconds']+time.monotonic()-started,sampled_batches_sha=next_batch,sampled_noise_sha=next_noise)
            committed=True
            if u%rule['evaluate_every']==0 or u==limit or u==target:
                val=score();st['history'].append(dict(update=u,validation=val,weights=weights,native_info=info,gradient_norm=norm))
                save(force=u==limit or u==target);s.write('histories/'+key+'.json',st['history'])
                s.event('study_checkpoint',key=key,update=u,total=target,limit=limit,capture=val.get('macro_capture'),AP=val.get('macro_ap'),
                    VE=val['reconstruction']['ve'],active_groups=val['reconstruction']['active_groups'],weights=weights,seconds_per_update=st['seconds']/u)
            elif u%rule['local_save_every']==0:
                save();s.event('study_progress',key=key,update=u,total=target,limit=limit,seconds_per_update=st['seconds']/u)
        save(force=True)
    except BaseException:
        if committed:save(force=True)
        # A partial Adam/controller update is never saved. Outer cleanup flushes
        # the newest valid LOCAL generation, which contains both optimizers.
        raise
    finally:
        del m,head,opt,controller;gc.collect()
        if device.startswith('cuda'):torch.cuda.empty_cache()
    ctx['active_key']=None
    return st
