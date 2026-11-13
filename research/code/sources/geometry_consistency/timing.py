"""Disposable full-width update measurements; no saved production weights changed."""
import copy,gc,json,time
import numpy as np
import torch
from coco_baselines import training
from coco_coordinated import run as memory
from coco_adaptation import core
from coco_ggf.capacity_fit import Balancer
from scale_models import model,encoder_parameters
from experiment import G,K,B,SEEDS,PRIMARY,EXTRA,WARM,GUIDED,CAP

def benchmark(ctx):
    s=ctx['store'];hardware=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__)
    old=s.load('timing_preflight')
    if old is not None and old['hardware']==hardware:
        print('REUSED HARDWARE TIMING',json.dumps(old),flush=True);return old
    x=torch.as_tensor(np.array(ctx['packs']['train']['x'][:8],copy=True),device=ctx['device']);d=x.shape[-1];xx=x.reshape(-1,d)
    masks=ctx['packs']['train']['masks'][:8]
    source_catalog=json.loads((ctx['sources']['scale'].results/'factorial_progress.json').read_text())
    targets=next(r['targets'] for r in source_catalog['models'] if r['seed']==21 and r['kind']=='agf_g' and r['groups']==1024 and r['active_groups']==16)
    measurements={};torch.manual_seed(126999)
    # Deliberately disposable optimizer/controller states, not any production checkpoint key.
    for method,supervised in [('released_vanilla',False),('released_vanilla',True),*[(v,False) for v in EXTRA]]:
        tag='guided' if supervised else method;m=model(method,G,B,K,d,.001).to(ctx['device'])
        head=torch.nn.Parameter(torch.zeros(75,2,device=ctx['device'])) if supervised else None
        opt=torch.optim.Adam(list(m.parameters())+([head] if supervised else []),lr=.0003)
        recipe=copy.deepcopy(ctx['balance_recipe']);recipe['bounds']['capture'][1]=CAP;recipe['bounds']['detection'][1]=CAP
        controller=Balancer(recipe,ctx['device']) if supervised else None
        # Include a full 20-step QR projection cadence for paper Grassmannian.
        times=[];count=31 if supervised else 23
        for u in range(1,count+1):
            torch.cuda.synchronize();begun=time.monotonic();m.train()
            rec,_=m.loss(xx+.1*xx.std()*torch.randn_like(xx),target=xx);terms=[rec];weights=[1.,0.,0.]
            if supervised:
                cap,det,_=core.supervised_loss(m,x,masks,targets,head,ctx['cfg']);terms+=[cap,det]
                controller.observe(terms);weights=controller.weights().detach().cpu().tolist()
                if u%recipe['every']==0 and controller.reference is not None:
                    norms=[]
                    for term in terms:
                        grads=torch.autograd.grad(term,encoder_parameters(m),retain_graph=True,allow_unused=True)
                        norms.append(torch.sqrt(sum((g.detach().square().sum() for g in grads if g is not None),rec.detach()*0)))
                    controller.adapt(terms,torch.stack(norms))
            loss=sum(w*l for w,l in zip(weights,terms));opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(list(m.parameters())+([head] if supervised else []),5.);opt.step();m.normalize_decoder()
            torch.cuda.synchronize();elapsed=time.monotonic()-begun
            if u>3:times.append(elapsed)
        measurements[tag]=dict(mean_seconds_per_update=float(np.mean(times)),measured_updates=len(times))
        print('MEASURED UPDATE COST',tag,measurements[tag],flush=True)
        del m,head,opt,controller;gc.collect();torch.cuda.empty_cache()
    # Check source/current states, so no completed training is counted as new.
    work=[];seen=set()
    for kind in PRIMARY+EXTRA:
        for seed in SEEDS:
            method='released_vanilla' if kind in PRIMARY else kind
            base=f'g1024_k16_s{seed}_'+('mask' if kind=='agf_g' else method)
            warmkey=base+'_warm'
            key=base+'_lr3000_p10000_'+kind+('_c040' if kind in ['agf_g','supervised_vanilla'] else '')
            for stage,fitkey,target in [('warm',warmkey,WARM),('continuation',key,GUIDED)]:
                if fitkey in seen:continue
                seen.add(fitkey);state=s.load(fitkey);saved=0 if state is None else state['updates']
                if saved>target:raise ValueError('Checkpoint exceeds fixed timing budget')
                tag='guided' if stage=='continuation' and kind in ['agf_g','supervised_vanilla'] else method
                seconds=(target-saved)*measurements[tag]['mean_seconds_per_update']
                work.append(dict(kind=kind,seed=seed,stage=stage,remaining_updates=target-saved,gradient_seconds=seconds,primary=kind in PRIMARY))
                del state;gc.collect()
    out=dict(hardware=hardware,measurements=measurements,work=work,
        primary_optimizer_only_hours=sum(r['gradient_seconds'] for r in work if r['primary'])/3600,
        all_optimizer_only_hours=sum(r['gradient_seconds'] for r in work)/3600,
        note='Lower bound from measured optimizer work. Mapping, validation, Drive, VOC extraction and bootstrap ADD time. Disposable preflight is not full-run throughput; no automatic stopping limit.')
    s.save('timing_preflight',out);s.write('timing_preflight.json',out)
    print('MEASURED OPTIMIZER-ONLY HOURS (additional overhead remains)',json.dumps(out),flush=True);return out
