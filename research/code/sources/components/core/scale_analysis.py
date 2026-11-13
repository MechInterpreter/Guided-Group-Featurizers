"""TRAIN rank diagnostics, held-out Gaussian coding proxy and measured costs.

The coding proxy is NOT a realized compressor or Goodfire's Grassmannian MDL
estimator. It charges the full stored model, canonical contribution spectra,
code means/bases, support and residual at predeclared distortion levels.
"""
import gc,math,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import gammaln
from coco_baselines import training
from coco_baselines.storage import identity
from coco_coordinated import run as memory
from rank_core import Moments,load_moments,save_moments,block_statistics,summarize,atomic_json,sha
from rank_worker import csv_write
from scale_models import model
from scale_evaluate import verify_freeze


def ranks(m,x,folder,signature,device,hard_topk=True,batch=256):
    moments,generation=load_moments(folder,signature,m.n_groups,m.group_size)
    if moments.patches>len(x) or (moments.patches%batch and moments.patches!=len(x)):raise ValueError('Rank offset changed')
    m.eval();steps=0
    try:
        with torch.inference_mode():
            for lo in range(moments.patches,len(x),batch):
                xb=torch.as_tensor(np.array(x[lo:lo+batch],copy=True),device=device);z=m.encode(xb);on=z.ne(0).any(-1)
                if not torch.isfinite(z).all():raise ValueError('Nonfinite rank codes')
                if hard_topk and int(on.sum(-1).max())>m.l0:raise ValueError('Rank Top-k violation')
                groups=on.nonzero()[:,1].cpu().numpy();values=z[on].cpu().numpy()
                before=(moments.n.copy(),moments.mean.copy(),moments.m2.copy(),moments.patches)
                try:moments.add_selected(groups,values,len(xb))
                except BaseException:
                    moments.n,moments.mean,moments.m2,moments.patches=before;raise
                steps+=1
                if steps%32==0 or moments.patches==len(x):
                    generation+=1;save_moments(folder,signature,moments,generation)
                    print('TRAIN rank:',moments.patches,'/',len(x),flush=True)
    except BaseException:
        generation+=1;save_moments(folder,signature,moments,generation);raise
    return moments


def gaussian_proxy(contribution_spectra,firing,residual_spectra,distortion,support_bits,model_bits,side_bits,N):
    if distortion<=0 or N<1:raise ValueError('Positive distortion and cohort size required')
    code=float(np.sum(np.asarray(firing)[:,None]*.5*np.log2(1+np.maximum(contribution_spectra,0)/distortion)))
    residual=float(np.sum(.5*np.log2(1+np.maximum(residual_spectra,0)/distortion)))
    model=float(model_bits/N);side=float(side_bits/N)
    return dict(support_bits_per_patch=float(support_bits),code_bits_per_patch=code,residual_bits_per_patch=residual,
                model_bits_per_patch=model,side_information_bits_per_patch=side,total_bits_per_patch=support_bits+code+residual+model+side)


def coding_proxy(ctx,m,row,folder):
    s=ctx['store'];key='coding_'+row['key'];saved=s.load(key)
    if saved is not None:
        if saved['weights_sha']!=row['weights_sha']:raise ValueError('Coding model identity differs')
        return saved
    p=ctx['packs']['confirm'];x=p['x'].reshape(-1,m.d);signature=identity(dict(weights=row['weights_sha'],cohort=p['identity'],scheme='gaussian-full-state-v1'))
    # Resumable whole combined accumulation: codes and centered residuals.
    keyacc='coding_acc_'+row['key'];acc=s.load(keyacc)
    cm=Moments(m.n_groups,m.group_size);rm=Moments(1,m.d);support_sum=0.;offset=0
    if acc is not None:
        if acc['signature']!=signature:raise ValueError('Coding cohort identity differs')
        cm=acc['code'];rm=acc['residual'];support_sum=acc['support_sum'];offset=cm.patches
    m.eval();steps=0
    def save():s.save(keyacc,dict(signature=signature,code=cm,residual=rm,support_sum=support_sum))
    try:
        with torch.inference_mode():
            for lo in range(offset,len(x),256):
                xb=torch.as_tensor(np.array(x[lo:lo+256],copy=True),device=ctx['device']);prediction,z=m(xb)
                if not torch.isfinite(z).all() or not torch.isfinite(prediction).all():raise ValueError('Nonfinite coding values')
                on=z.ne(0).any(-1);K=on.sum(-1).cpu().numpy();groups=on.nonzero()[:,1].cpu().numpy();values=z[on].cpu().numpy()
                residual=(xb-prediction).cpu().numpy();G=m.n_groups
                before=(cm.n.copy(),cm.mean.copy(),cm.m2.copy(),cm.patches,rm.n.copy(),rm.mean.copy(),rm.m2.copy(),rm.patches,support_sum)
                try:
                    cm.add_selected(groups,values,len(xb));rm.add_selected(np.zeros(len(xb),int),residual,len(xb))
                    support_sum+=float(((gammaln(G+1)-gammaln(K+1)-gammaln(G-K+1))/np.log(2)).sum())
                except BaseException:
                    cm.n,cm.mean,cm.m2,cm.patches,rm.n,rm.mean,rm.m2,rm.patches,support_sum=before;raise
                steps+=1
                if steps%32==0 or cm.patches==len(x):save();print('Coding calibration:',cm.patches,'/',len(x),flush=True)
    except BaseException:save();raise
    decoder=m.atoms().cpu().double().numpy();_,arrays=block_statistics(cm,decoder)
    ev=np.linalg.eigvalsh(rm.m2[0]/max(rm.n[0]-1,1));ev=np.maximum(ev,0)
    # Canonical decoded-contribution spectra remove arbitrary within-block
    # scale/rotation effects from the conditional Gaussian code term.
    contribution=arrays['contribution_eigenvalues']
    weights=training.weights(m);state_bits=sum(t.numel()*t.element_size()*8 for t in weights.values())+row['head_parameters']*32
    G=m.n_groups;b=m.group_size;d=m.d
    side_bits=32*(G*b+G*b*b+d+d*d)
    # Includes per-patch K, conditional means/bases and residual mean/basis.
    # Full encoder/decoder/buffer costs are charged equally, not chart DoFs.
    variance=ctx['coding_reference_variance'];N=len(ctx['packs']['train']['x'])*196
    records=[]
    for fraction in ctx['rule']['mdl_distortions']:
        r=gaussian_proxy(contribution,cm.n/cm.patches,ev,fraction*variance,support_sum/len(x)+math.ceil(math.log2(G+1)),state_bits,side_bits,N)
        r.update(distortion_fraction=fraction,reference_variance=variance,amortization_patches=N,calibration_patches=len(x),kind=row['kind'],seed=row['seed']);records.append(r)
    out=dict(weights_sha=row['weights_sha'],signature=signature,records=records,
        scope='Two-part Gaussian coding-cost proxy, not a realized bitstream, universal MDL optimum, or unmodified Goodfire estimator.',
        coding='Enumerative support, per-patch K, active decoded-contribution covariance Gaussian rates; centered residual Gaussian rates; 32-bit means/bases and full stored state.',
        calibration='Same historical confirmation cohort for every model; no final holdout values enter coding or selection.')
    s.save(key,out);s.write('coding_proxy/'+key+'.json',out);return out


@torch.no_grad()
def deployment(m,x,device):
    m.eval();xb=torch.as_tensor(np.array(x[:196],copy=True),device=device)
    if device.startswith('cuda'):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    for _ in range(5):m(xb)
    times=[]
    for _ in range(30):
        if device.startswith('cuda'):torch.cuda.synchronize()
        start=time.perf_counter();m(xb)
        if device.startswith('cuda'):torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
    return dict(parameters=sum(p.numel() for p in m.parameters()),median_ms_per_196_patch_image=1000*float(np.median(times)),
                p95_ms=1000*float(np.quantile(times,.95)),peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30 if device.startswith('cuda') else None,
                excludes='Backbone, masks, input loading, readout fitting and training; synchronized BSF forward only')


def run(ctx):
    frozen=verify_freeze(ctx);s=ctx['store'];old=s.load('analysis_complete')
    if old is not None:return old
    x=ctx['packs']['train']['x'].reshape(-1,ctx['packs']['train']['x'].shape[-1]);rows=[];blocks=[];coding=[];costs=[]
    total=0.;count=0
    for lo in range(0,len(x),2048):
        part=np.asarray(x[lo:lo+2048],np.float64);total+=float(np.square(part).sum());count+=part.size
    ctx['coding_reference_variance']=total/count
    folder=s.results/'effective_rank';folder.mkdir(exist_ok=True)
    for row in frozen['models']:
        dest=folder/'models'/row['key'];dest.mkdir(parents=True,exist_ok=True);signature=identity(dict(weights=row['weights_sha'],train=ctx['packs']['train']['identity'],rank=row['rank'],groups=row['groups']))
        completion=dest/'complete.json';state=memory.required(s,row['key'])
        m=model(row['method'],row['groups'],row['rank'],row['active_groups'],x.shape[-1],row['coefficient']).to(ctx['device']);m.load_state_dict(state['weights']);del state
        if completion.exists():
            done=json.loads(completion.read_text())
            if done['signature']!=signature or not done['complete'] or any(sha(dest/n)!=h for n,h in done['hashes'].items()):raise ValueError('Rank completion changed')
            rr=json.loads((dest/'summary.json').read_text());bb=json.loads((dest/'blocks.json').read_text())
        else:
            mom=ranks(m,x,folder/'accumulators'/row['key'],signature,ctx['device'],not row['method'].endswith('group_lasso'))
            bb,arrays=block_statistics(mom,m.atoms().cpu().double().numpy());meta=dict(arm=row['kind'],seed=row['seed'],rank=row['rank'],active_groups=row['active_groups'],
                groups=row['groups'],weights_sha=row['weights_sha'],semantic_initialization=row['masked'])
            for r in bb:r.update(meta)
            rr=[dict(r,**meta) for r in summarize(bb)]
            atomic_json(dest/'blocks.json',bb);atomic_json(dest/'summary.json',rr)
            np.savez_compressed(dest/'spectra.npz',**arrays)
            atomic_json(completion,dict(complete=True,signature=signature,patches=len(x),hashes={p.name:sha(p) for p in dest.iterdir() if p.name!='complete.json'}))
        rows.extend(rr);blocks.extend(bb);coding.extend(coding_proxy(ctx,m,row,folder)['records'])
        key='cost_'+row['key'];cost=s.load(key)
        if cost is None:
            cost=dict(kind=row['kind'],seed=row['seed'],weights_sha=row['weights_sha'],**deployment(m,x,ctx['device']));s.save(key,cost)
        costs.append(cost);csv_write(folder/'model_summaries.csv',rows);csv_write(s.results/'coding_cost_proxy.csv',coding);csv_write(s.results/'deployment_costs.csv',costs)
        del m;gc.collect();torch.cuda.empty_cache()
    csv_write(folder/'block_statistics.csv',blocks)
    means=[]
    for arm in sorted({r['arm'] for r in rows}):
        for family in ['all','initialized_slots','reserve_slots']:
            for subset in ['all_estimable','at_least_32_active_patches']:
                rr=[r for r in rows if r['arm']==arm and r['slot_family']==family and r['sample_subset']==subset]
                if len(rr)!=7 or len({r['seed'] for r in rr})!=7:raise ValueError('Missing seven-seed rank cohort')
                out=dict(arm=arm,slot_family=family,sample_subset=subset,seeds=7,allocated_width=rr[0]['rank'],groups=rr[0]['groups'])
                for key in ['code_effective_rank_mean','contribution_effective_rank_mean','code_utilization_mean',
                            'code_participation_ratio_mean','code_covariance_stable_rank_mean','code_matrix_stable_rank_mean','valid_groups','inactive_groups','median_active_patches']:
                    vv=[r[key] for r in rr if r[key] is not None]
                    out[key]=float(np.mean(vv)) if vv else None;out[key+'_seed_sd']=float(np.std(vv,ddof=1)) if len(vv)>1 else None
                means.append(out)
    csv_write(folder/'seven_seed_means.csv',means)
    plots(s.results/'figures',means,coding,costs)
    out=dict(complete=True,models=len(frozen['models']),train_patches=len(x),gradient_updates=0,new_test_inference=False,
             rank_scope='Conditional active-code variation; first 75 slots are semantically initialized only in mask-based arms.',
             mdl_scope='Explicit Gaussian coding-cost proxy; see coding_proxy reports. Not a measured compressor.')
    s.save('analysis_complete',out);s.write('analysis_complete.json',out);return out


import json


def plots(folder,rank_means,coding,costs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rr=[r for r in rank_means if r['slot_family']=='initialized_slots' and r['sample_subset']=='at_least_32_active_patches']
    fig,axes=plt.subplots(1,2,figsize=(13,6),sharey=True,layout='constrained')
    for ax,key,title in zip(axes,['code_effective_rank_mean','code_utilization_mean'],['Coefficient effective rank','Utilization']):
        for i,r in enumerate(rr):
            value=r[key]
            if value is not None:ax.barh(i,value,xerr=r[key+'_seed_sd'],color='#087f83' if r['arm'].startswith('agf') else '#879cad',height=.65)
        ax.set_yticks(range(len(rr)),[r['arm'].replace('_',' ') for r in rr]);ax.set_title(title);ax.spines[['top','right']].set_visible(False)
    axes[0].invert_yaxis();fig.suptitle('First 75 slots; semantic initialization only in masked arms / TRAIN / error bars = seed SD')
    fig.savefig(folder/'effective_rank.png',dpi=200);fig.savefig(folder/'effective_rank.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
    for kind in sorted({r['kind'] for r in coding}):
        fractions=sorted({r['distortion_fraction'] for r in coding if r['kind']==kind})
        values=[np.mean([r['total_bits_per_patch'] for r in coding if r['kind']==kind and r['distortion_fraction']==f]) for f in fractions]
        ax.plot(fractions,values,marker='o',label=kind.replace('_',' '))
    ax.set_xlabel('Predeclared distortion fraction');ax.set_ylabel('Estimated bits per patch');ax.set_title('Gaussian coding-cost proxy; not a measured compressor')
    ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.15);fig.savefig(folder/'coding_cost_proxy.png',dpi=200);fig.savefig(folder/'coding_cost_proxy.pdf');plt.close(fig)
