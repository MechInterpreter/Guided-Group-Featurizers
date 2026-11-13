"""Bounded, capacity-matched SLiCS-inspired COCO adaptation.

Paper-based NNLS + sequential rank-one SVD dictionary fitting. Explicit changes:
patch-mask support; minibatches; finite NNLS;
label-blind dense NNLS then Top-16 prune/refit at inference. Not exact SLiCS.
"""
import copy, gc, json, time, traceback, zipfile
from pathlib import Path
import numpy as np
import torch
from coco_baselines import metrics, training
from coco_baselines.storage import identity, serial
from coco_adaptation import core, inputs
from coco_coordinated import run as memory
from . import loss_weight_sweep as lineage, study


def settings():
    return dict(version=1, study='slics_capacity_matched_coco_pilot', seeds=[21],
        warm_updates=1000, guided_updates=1000, images_per_update=4,
        evaluate_every=250, save_every=50, rank=3, active_groups=16,
        nnls_iterations=300, nnls_tolerance=1e-4, lr=.0003,
        lambda_capture=.05, lambda_detection=.01, snr=.1,
        include_image_label_control=True, include_capacity_sweep=False,
        block_sizes=[2,3,4], active_group_grid=[8,16,32],
        scope='Development comparator pilot; historical confirmation images are an audit, not a new test.')


@torch.no_grad()
def nnls(x, D, allowed=None, iterations=300, tolerance=1e-4):
    """FISTA with nonnegative projection; relative projected-gradient KKT audit."""
    if allowed is None: allowed=torch.ones((len(x),len(D)),dtype=torch.bool,device=x.device)
    if allowed.shape!=(len(x),len(D)): raise ValueError('NNLS support shape')
    gram=D@D.T; cross=x@D.T
    L=torch.linalg.eigvalsh(gram).amax().clamp_min(1e-8)
    z=torch.zeros_like(cross); y=z.clone(); t=1.
    used=0
    for it in range(iterations):
        new=((y-(y@gram-cross)/L).clamp_min(0))*allowed
        nt=(1+(1+4*t*t)**.5)/2
        y=new+(t-1)/nt*(new-z); z=new; t=nt; used=it+1
        if used%25==0 or used==iterations:
            grad=z@gram-cross
            pg=(z-(z-grad).clamp_min(0))*allowed
            error=float(pg.abs().amax()/cross.abs().amax().clamp_min(1e-8))
            if error<=tolerance: break
    if not torch.isfinite(z).all(): raise FloatingPointError('NNLS nonfinite')
    return z, dict(iterations=used,relative_kkt_error=error,converged=error<=tolerance,
                   inactive_code_max=float(z[~allowed].abs().amax()) if bool((~allowed).any()) else 0.)


@torch.no_grad()
def atom_sweep(x,D,z,allowed):
    """Leading singular vector via the smaller exact symmetric Gram problem."""
    residual=x-z@D; accepted=rejected=dead=0
    before=float(residual.square().sum())
    for j in range(len(D)):
        rows=allowed[:,j]
        if not bool(rows.any()): dead+=1; continue
        old=z[rows,j].clone(); oldatom=D[j].clone()
        E=residual[rows]+old[:,None]*oldatom
        if float(E.square().sum())<1e-10: dead+=1; continue
        if len(E)<=E.shape[1]:
            _,vectors=torch.linalg.eigh(E@E.T)
            coef=vectors[:,-1]
            atom=E.T@coef; atom=atom/atom.norm().clamp_min(1e-8)
        else:
            _,vectors=torch.linalg.eigh(E.T@E); atom=vectors[:,-1]
        beta=E@atom
        if float(beta.clamp_min(0).square().sum())<float((-beta).clamp_min(0).square().sum()):
            atom=-atom; beta=-beta
        beta=beta.clamp_min(0); newres=E-beta[:,None]*atom
        olderror=float(residual[rows].square().sum()); newerror=float(newres.square().sum())
        # Conservative adaptation: never increase this minibatch's residual.
        if newerror>olderror+1e-5*max(1.,olderror): rejected+=1; continue
        D[j]=atom; z[rows,j]=beta; residual[rows]=newres; accepted+=1
    return D,dict(atom_updates=accepted,rejected_nonmonotone_atom_updates=rejected,unused_atoms=dead,
                  residual_sse_before=before,residual_sse_after=float(residual.square().sum()))


class SlicsModel(torch.nn.Module):
    def __init__(self,D,cfg,dense=False):
        super().__init__(); self.register_buffer('D',D.clone())
        self.n_groups=len(D)//cfg['rank']; self.group_size=cfg['rank']; self.d=D.shape[1]
        self.cfg=cfg; self.dense=dense; self.solver_diagnostics=[]
        self.inference_seconds=0.; self.inference_patches=0
    def atoms(self): return self.D.reshape(self.n_groups,self.group_size,self.d)
    def decoder_atoms(self): return self.D
    @torch.no_grad()
    def encode(self,x):
        started=time.monotonic()
        z,info=nnls(x,self.D,iterations=self.cfg['nnls_iterations'],tolerance=self.cfg['nnls_tolerance'])
        if not self.dense:
            heat=z.reshape(-1,self.n_groups,self.group_size).norm(dim=-1)
            ids=heat.topk(min(self.cfg['active_groups'],self.n_groups),dim=-1).indices
            on=torch.zeros_like(heat,dtype=torch.bool).scatter_(1,ids,True)
            allowed=on.repeat_interleave(self.group_size,dim=1)
            z,info=nnls(x,self.D,allowed,self.cfg['nnls_iterations'],self.cfg['nnls_tolerance'])
        self.solver_diagnostics.append(info)
        if x.device.type=='cuda': torch.cuda.synchronize()
        self.inference_seconds+=time.monotonic()-started; self.inference_patches+=len(x)
        return z.reshape(-1,self.n_groups,self.group_size)
    def forward(self,x):
        z=self.encode(x); return z.flatten(1)@self.D,z


def prepare(source_output,output,local,source_sha,versions,cfg,device,require_drive=True):
    defaults=settings()
    if set(cfg)!=set(defaults) or cfg['rank']!=3 or cfg['active_groups']!=16: raise ValueError('Unsupported protocol')
    if not cfg['seeds'] or any(s not in (21,22,23) for s in cfg['seeds']) or len(set(cfg['seeds']))!=len(cfg['seeds']): raise ValueError('Unique seeds 21--23 required')
    for k in ('warm_updates','guided_updates','images_per_update','evaluate_every','save_every','nnls_iterations'):
        if type(cfg[k]) is not int or cfg[k]<=0: raise ValueError('Positive count: '+k)
    # Reuse verified reader, but never reuse the old 1024-group model as a matched arm.
    verification=lineage.settings(); verification['seeds']=cfg['seeds']; verification['include_scratch']=False
    ctx=lineage.prepare(source_output,Path(output)/'input_verification',local,source_sha,versions,
                        verification,device,require_drive)
    ctx['cfg']=dict(ctx['cfg'],groups=75,updates=cfg['guided_updates'],images_per_update=cfg['images_per_update'],
        lr=cfg['lr'],lambda_capture=cfg['lambda_capture'],lambda_detection=cfg['lambda_detection'],
        evaluate_every=cfg['evaluate_every'],local_save_every=cfg['save_every'],
        diagnostic_grad_every=cfg['evaluate_every'],snr=cfg['snr'],device=device,preview_images=0)
    ctx['base']=dict(ctx['base'],groups=75,block_rank=3,active_groups=16)
    ctx['store']=inputs.JournalStore(output,dict(config=cfg,source_sha=source_sha,versions=versions,device=device,
        source_run=ctx['source'].run_id,activation_run=ctx['original'].run_id),Path(local)/'comparison_rescue',require_drive)
    ctx['comparison']=cfg; ctx['device']=device
    ctx['store'].write('protocol_settings.json',dict(config=cfg,dictionary_groups=75,directions=225,
        interpretation='Capacity-matched task adaptation of SLiCS, not an exact published replication. The 1024-group GGF result is context only.',
        changes=['75 category groups; no background group added','COCO patch-mask or image-presence support',
                 'Minibatch finite NNLS, exact leading-SVD atom updates with nonincrease safeguard',
                 'Label-blind dense NNLS then Top-16 prune/refit at inference; dense readout also reported'],
        compute='Same presented images/patch counts per arm; SLiCS excludes crowd patches and logs unsupported patches; SGD updates and alternating NNLS/SVD sweeps are not equal compute. Report measured train and inference times.',
        labels='Masks constrain training support only; no mask/category is passed to encode or reconstruction at evaluation.',
        selection='All arms report development capture/AP/VE frontier. Weights selected by capture + 0.2 AP within 2% MSE of the common native start, or explicitly ineligible.',
        audit='Historical 400 confirmation images, group mappings chosen on historical 400 mapping images after weights freeze. Not a fresh test.'))
    return ctx


def support(masks,cohort,cfg,kind,device):
    f=np.stack([m['fraction'] for m in masks]).astype(np.float32)/255
    crowd=np.stack([m['crowd'] for m in masks])
    if kind=='image':
        pos=np.stack([[any(o['category']==c for o in m['instances']) for c in cohort] for m in masks])
        pos=np.repeat(pos[:,None,:],196,axis=1)
    else: pos=f[:,:,cohort]>=.5
    # Original SLiCS support has no extra unlabeled/background group.
    allowed=pos
    known=crowd.reshape(-1)<=.1
    return torch.as_tensor(np.repeat(allowed.reshape(-1,75),cfg['rank'],axis=1),device=device),known


def fixed_category_targets(ctx,m):
    p=ctx['packs']['train']; cfg=ctx['cfg']
    a=metrics.audit_arrays(m,p['x'],p['masks'],p['ids'],cfg,ctx['device'])
    t=core.choose_targets(a,cfg,ctx['provenance']['categories']); del a; gc.collect()
    if len(t)!=75: raise ValueError('Training mapping missed a frozen category')
    return t


def dev(ctx,m,targets):
    p=ctx['packs']['select']
    result=core.validation(m,p,targets,ctx['cfg'],ctx['device'])
    if not np.isfinite([result['macro_capture'],result['macro_ap'],result['reconstruction']['mse']]).all(): raise FloatingPointError('Invalid full-cohort score')
    return result


def sl_targets(ctx,m):
    # Category ownership is training-defined, never selected on development data.
    return [dict(category_index=c,category=ctx['provenance']['categories'][c]['name'],
        contrast_group=g,detector_group=g,heat_mean=0.,heat_scale=1.)
        for g,c in enumerate(ctx['cfg']['category_indices'])]


def batch(ctx,seed,u):
    # Stateless draws make interrupted/resumed arms see exactly the same images.
    n=len(ctx['packs']['train']['ids']); size=ctx['comparison']['images_per_update']
    warm=ctx['comparison']['warm_updates']
    if u<warm:
        ids=np.random.default_rng(np.random.SeedSequence([10500,seed,u])).integers(n,size=size)
    else:
        # Exactly the guided phase's seeded image stream, without mutable RNG.
        ids=np.random.default_rng(71000+seed).integers(n,size=(u-warm+1,size))[-1]
    p=ctx['packs']['train']; x=torch.as_tensor(np.array(p['x'][ids],copy=True),device=ctx['device'])
    return x,[p['masks'][i] for i in ids],identity(ids)


def init_slics(ctx,seed,kind):
    # Training-only per-label truncated SVD initialization; no test access.
    rng=np.random.default_rng(seed+900); p=ctx['packs']['train']; atoms=[]; report=[]
    for g,c in enumerate(ctx['cfg']['category_indices']+[None]):
        rows=[]
        for i in rng.permutation(len(p['ids'])):
            mask=p['masks'][i]
            if c is None: take=mask['crowd']<=.1
            elif kind=='image': take=np.ones(196,bool) if any(o['category']==c for o in mask['instances']) else np.zeros(196,bool)
            else: take=(mask['fraction'][:,c].astype(float)/255>=.5)&(mask['crowd']<=.1)
            rows.extend(np.asarray(p['x'][i][take],np.float32))
            if len(rows)>=512: break
        if len(rows)<3: raise ValueError('Training-only SVD lacks category '+str(c))
        E=torch.as_tensor(np.array(rows[:512]),device=ctx['device'])
        _,vectors=torch.linalg.eigh(E.T@E); D=vectors[:,-3:].T.flip(0)
        beta=E@D.T; flip=beta.clamp_min(0).square().sum(0)<(-beta).clamp_min(0).square().sum(0)
        D[flip]*=-1; atoms.append(D); report.append(dict(category=c,patches=len(E)))
    ctx['store'].write(f'initialization_s{seed}_{kind}.json',report)
    return torch.cat(atoms)


def fit_slics(ctx,seed,kind,initial,initial_mse,max_updates=None):
    s=ctx['store']; cfg=ctx['comparison']; key=f'slics_{kind}_s{seed}'; total=cfg['warm_updates']+cfg['guided_updates']
    st=s.load(key)
    if st is None:
        m=SlicsModel(initial,cfg).to(ctx['device']); targets=sl_targets(ctx,m)
        v=dev(ctx,m,targets)
        st=dict(updates=0,D=initial.cpu(),best=None,best_score=None,best_update=None,history=[],seconds=0.,
                targets=targets,initial_mse=initial_mse,total=total,batch_hash=identity([]))
    else:
        m=SlicsModel(st['D'].to(ctx['device']),cfg); targets=st['targets']
        if st['total']!=total: raise ValueError('SLiCS horizon changed')
    stop=total if max_updates is None else min(total,max_updates)
    def save(drive=True): st['D']=m.D.detach().cpu().clone(); s.save(key,st,drive)
    try:
        while st['updates']<stop:
            u=st['updates']; x,masks,bh=batch(ctx,seed,u); allowed,known=support(masks,ctx['cfg']['category_indices'],cfg,kind,ctx['device'])
            xx=x.reshape(-1,m.d)[torch.as_tensor(known,device=x.device)]; allowed=allowed[torch.as_tensor(known,device=x.device)]
            if not len(xx): raise ValueError('Entire training batch is crowd')
            started=time.monotonic()
            z,solve=nnls(xx,m.D,allowed,cfg['nnls_iterations'],cfg['nnls_tolerance'])
            D,update=atom_sweep(xx,m.D,z,allowed); m.D.copy_(D)
            study.sync(ctx['device']); st['seconds']+=time.monotonic()-started; st['updates']+=1
            st['batch_hash']=identity([st['batch_hash'],bh]); u=st['updates']
            if u%cfg['evaluate_every']==0 or u==total:
                v=dev(ctx,m,targets); eligible=v['reconstruction']['mse']<=initial_mse*1.02
                if eligible and (st['best_score'] is None or v['score']>st['best_score']):
                    st.update(best=m.D.detach().cpu().clone(),best_score=v['score'],best_update=u)
                rec=dict(update=u,validation=v,eligible=bool(eligible),solver=solve,atom_sweep=update,unsupported_training_patch_fraction=float((~allowed.any(1)).float().mean()),crowd_patches_excluded=int((~known).sum()),
                    seconds=st['seconds'],batch_hash=st['batch_hash'],rejection_reasons=[] if eligible else ['common_native_reconstruction_guard'])
                st['history'].append(rec); s.write(f'histories/{key}.json',st['history']); save()
                s.event('slics_checkpoint',key=key,update=u,total=total,capture=v['macro_capture'],AP=v['macro_ap'],VE=v['reconstruction']['ve'],solver=solve)
            elif u%cfg['save_every']==0: save(); s.event('slics_training',key=key,update=u,total=total,solver=solve)
        save()
    except BaseException: save(False); raise
    return st


def fit_native(ctx,seed):
    s=ctx['store']; cfg=ctx['comparison']; key=ctx.get('tag','')+f'compact_native_s{seed}'; st=s.load(key)
    torch.manual_seed(seed); m=training.make_model('vanilla',ctx['base']).to(ctx['device'])
    opt=torch.optim.Adam(m.parameters(),lr=.0003)
    if st is None: st=dict(updates=0,weights=training.weights(m),best=None,best_mse=None,best_update=0,history=[],seconds=0.)
    else:
        m.load_state_dict(st['weights']); opt.load_state_dict(st['optimizer']); torch.set_rng_state(st['rng'])
        if ctx['device']=='cuda': torch.cuda.set_rng_state_all(st['cuda_rng'])
    def save():
        st.update(weights=training.weights(m),optimizer=opt.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if ctx['device']=='cuda' else None)
        s.save(key,st)
    try:
        while st['updates']<cfg['warm_updates']:
            x,masks,bh=batch(ctx,seed,st['updates']); xx=x.reshape(-1,m.d); started=time.monotonic()
            noisy=xx+cfg['snr']*xx.std()*torch.randn_like(xx); loss,_=m.loss(noisy,target=xx)
            if not torch.isfinite(loss): raise FloatingPointError('Native loss')
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.); opt.step(); m.normalize_decoder()
            study.sync(ctx['device']); st['seconds']+=time.monotonic()-started; st['updates']+=1; u=st['updates']
            if u%cfg['evaluate_every']==0 or u==cfg['warm_updates']:
                v=training.measure(m,ctx['packs']['select']['x'].reshape(-1,m.d),ctx['device'],512)
                st['history'].append(dict(update=u,selection=v))
                if st['best_mse'] is None or v['mse']<st['best_mse']: st.update(best=training.weights(m),best_mse=v['mse'],best_update=u)
                s.event('native_checkpoint',seed=seed,update=u,total=cfg['warm_updates'],selection=v)
            if u%cfg['save_every']==0 or u==cfg['warm_updates']: save()
    except BaseException: save(); raise
    m.load_state_dict(st['best']); return m,st


def train(ctx):
    s=ctx['store']; cfg=ctx['comparison']; old=s.load('comparison_frozen')
    if old is not None: return old
    models=[]
    for seed in cfg['seeds']:
        m,native=fit_native(ctx,seed); targets=s.load(f'compact_targets_s{seed}')
        if targets is None: targets=s.save(f'compact_targets_s{seed}',fixed_category_targets(ctx,m))
        init=training.weights(m); initial_mse=dev(ctx,m,targets)['reconstruction']['mse']; del m
        for arm,supervised in [('continued',False),('ggf',True)]:
            key=f'compact_{arm}_s{seed}'
            # Same guided batches/noise/update budget for these paired SGD arms.
            st=study.fit(ctx,key,init,targets,supervised,71000+seed)
            models.append(dict(key=key,kind='native',seed=seed,arm=arm,field='best',updates=native['updates']+st['updates'],
                dictionary_atoms=225,groups=75,rank=3,parameters=sum(v.numel() for v in st['best'].values()),
                training_seconds=native['seconds']+st['seconds'],eligible=True))
        for kind in (['mask','image'] if cfg['include_image_label_control'] else ['mask']):
            initial=s.load(f'slics_initial_{kind}_s{seed}')
            if initial is None: initial=s.save(f'slics_initial_{kind}_s{seed}',init_slics(ctx,seed,kind).cpu())
            st=fit_slics(ctx,seed,kind,initial.to(ctx['device']),initial_mse)
            models.append(dict(key=f'slics_{kind}_s{seed}',kind='slics',seed=seed,arm='SLiCS-inspired '+kind,
                field='best' if st['best'] is not None else 'D',updates=st['updates'],dictionary_atoms=225,
                groups=75,rank=3,parameters=225*768,training_seconds=st['seconds'],eligible=st['best'] is not None))
        del native,init; gc.collect()
    if cfg['include_capacity_sweep']:
        for rank in cfg['block_sizes']:
            for active in cfg['active_group_grid']:
                if (rank,active)==(3,16): continue
                if type(rank) is not int or rank<=0 or type(active) is not int or not 1<=active<=75: raise ValueError('Invalid capacity grid')
                branch=dict(ctx,tag=f'b{rank}_k{active}_',base=dict(ctx['base'],block_rank=rank,active_groups=active),
                    cfg=dict(ctx['cfg'],block_rank=rank,active_groups=active),comparison=dict(cfg,rank=rank,active_groups=active))
                for seed in cfg['seeds']:
                    m,native=fit_native(branch,seed); target_key=branch['tag']+f'compact_targets_s{seed}'
                    targets=s.load(target_key)
                    if targets is None: targets=s.save(target_key,fixed_category_targets(branch,m))
                    init=training.weights(m); del m
                    for arm,supervised in [('continued',False),('ggf',True)]:
                        key=branch['tag']+f'compact_{arm}_s{seed}'
                        st=study.fit(branch,key,init,targets,supervised,71000+seed)
                        models.append(dict(key=key,kind='native',seed=seed,arm=arm,field='best',
                            updates=native['updates']+st['updates'],dictionary_atoms=75*rank,groups=75,rank=rank,
                            active_group_budget=active,parameters=sum(v.numel() for v in st['best'].values()),
                            training_seconds=native['seconds']+st['seconds'],eligible=True,
                            role='GGF/native capacity sweep; SLiCS is not compared at this setting'))
                    del native,init;gc.collect()
    for row in models:row.setdefault('active_group_budget',16)
    out=dict(complete=True,models=models,no_test_access=True,protocol_adaptation=True)
    s.save('comparison_frozen',out); s.write('models_frozen.json',out); return out


def preflight(ctx):
    cfg=ctx['comparison']; s=ctx['store']; old=s.load('preflight')
    if old is not None: return old
    x,masks,_=batch(ctx,cfg['seeds'][0],0); xx=x.flatten(0,1)
    D=torch.randn(225,768,device=ctx['device']); D=D/D.norm(dim=-1,keepdim=True)
    allowed,known=support(masks,ctx['cfg']['category_indices'],cfg,'mask',ctx['device'])
    use=torch.as_tensor(known,device=x.device); start=time.monotonic()
    z,info=nnls(xx[use],D,allowed[use],cfg['nnls_iterations'],cfg['nnls_tolerance'])
    D,atom=atom_sweep(xx[use],D,z,allowed[use]); study.sync(ctx['device']); seconds=time.monotonic()-start
    out=dict(numerical_ok=bool(torch.isfinite(D).all() and torch.isfinite(z).all()),
        measured_slics_mask_seconds_per_update=seconds,solver=info,atom_sweep=atom,
        rough_mask_training_hours=seconds*(cfg['warm_updates']+cfg['guided_updates'])*len(cfg['seeds'])/3600,
        excludes='Image-support SLiCS is usually slower; mappings, validation, SGD, inference, Drive and initialization excluded. Not an ETA.',
        parameters=dict(dictionary_each=225*768,native_encoder_decoder=2*225*768+225,ggf_extra_heads=150),
        same_dictionary_capacity=True,same_total_parameter_count=False,compute_matched=False)
    s.save('preflight',out); s.write('preflight.json',out); return out


def evaluate(ctx):
    s=ctx['store']; old=s.load('comparison_summary')
    if old is not None: return old
    frozen=s.load('comparison_frozen')
    if frozen is None or not frozen['complete']: raise RuntimeError('Freeze every arm before audit')
    # Freeze model weights before opening either mapping or confirmation cache.
    norm=memory.required(ctx['original'],'normalization')
    packs={k:memory.normalized_pack(ctx['original'],k,ctx['provenance'],norm,ctx['local'],ctx['cfg']['extract_chunk']) for k in ('mapping','confirm')}
    rows=[]
    for row in frozen['models']:
        for dense in ([False,True] if row['kind']=='slics' else [False]):
            key=row['key']+('_dense' if dense else '_top16'); previous=s.load('audit_'+key)
            if previous is not None: rows.append(previous); continue
            state=s.load(row['key'])
            base=dict(ctx['base'],block_rank=row['rank'],active_groups=row['active_group_budget'])
            m=SlicsModel(state[row['field']].to(ctx['device']),ctx['comparison'],dense) if row['kind']=='slics' else core.new_model('vanilla',base,state[row['field']],ctx['device'])
            maps=s.load('mapping_'+key)
            if maps is None: maps=s.save('mapping_'+key,fixed_mapping(ctx,m,packs['mapping']))
            started=time.monotonic(); v=core.validation(m,packs['confirm'],maps,ctx['cfg'],ctx['device'])
            audit_seconds=time.monotonic()-started
            result=dict(**row,readout='dense NNLS' if dense else 'at most 16 groups',audit=v,
                solver_diagnostics=getattr(m,'solver_diagnostics',[]),mapping=maps,
                audit_seconds=audit_seconds,nnls_encode_seconds=getattr(m,'inference_seconds',None),
                nnls_encode_patches=getattr(m,'inference_patches',None),
                audit_images=len(packs['confirm']['ids']),audit_scope='Previously inspected historical confirmation; not a fresh test.')
            s.save('audit_'+key,result); s.write('audits/'+key+'.json',result); rows.append(result)
            s.event('audit_complete',key=key,capture=v['macro_capture'],AP=v['macro_ap'],VE=v['reconstruction']['ve'])
            del m; gc.collect()
    out=dict(complete=True,models=rows,publication_claim='Bounded adapted-method pilot only; no exact replication, full-capacity or equal-compute claim.')
    s.save('comparison_summary',out); s.write('summary.json',out)
    return out


def fixed_mapping(ctx,m,pack):
    a=metrics.audit_arrays(m,pack['x'],pack['masks'],pack['ids'],ctx['cfg'],ctx['device'])
    targets=core.choose_targets(a,ctx['cfg'],ctx['provenance']['categories']); del a
    if len(targets)!=75: raise ValueError('Mapping/audit cohort cannot be silently reduced')
    return targets


def export(ctx):
    s=ctx['store']; out=s.output/f'slics_comparison_{s.run_id[:12]}_results.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for p in s.results.rglob('*'):
            if p.is_file(): z.write(p,'results/'+p.relative_to(s.results).as_posix())
    return out
